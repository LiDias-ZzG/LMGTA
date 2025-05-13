import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from cdlib import algorithms
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--runs', type=str, default=1)
parser.add_argument('--dataset', type=str,default='BlogCatalog')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed' 'ACM_tiny' 'citation' 'DBLP''dblpv7_both' 'citationv1_both' 'acmv9_both'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  #max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['cora','citeseer','pubmed','citation']:
        args.lr = 1e-3
    elif args.dataset in ['DBLP','ACM','Amazon','Flickr','BlogCatalog']:
        args.lr = 5e-4

if args.num_epoch is None:
    if args.dataset in ['cora','citeseer','pubmed']:
        args.num_epoch = 100
    elif args.dataset in ['ACM', 'DBLP','citation','Amazon','Flickr','BlogCatalog']:
        args.num_epoch = 400
candidate = 0
if args.dataset == 'BlogCatalog':
    candidate = 0.8
if args.dataset == 'Flickr':
    candidate = 0.1

for run in range(args.runs):
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    print(args.alpha,args.beta)
    print('Dataset: ', args.dataset)
    seed = args.seed + run
    print('seed: ', seed)

    # Set random seed
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and preprocess data
    adj, features,  idx_train, idx_val, \
    idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
    degree = np.sum(adj, axis=0)
    degree_avg = np.mean(degree)
    features, _ = preprocess_features(features)
    dgl_graph = adj_to_dgl_graph(adj)
    torch.save(ano_label,'ano_label_Flickr.pt')
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]

    adj_org = adj.todense()
    adj, adj_raw = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    adj_org = torch.FloatTensor(adj_org[np.newaxis])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, args.alpha , args.beta)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        adj_org = adj_org.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
    added_adj_zero_col[:, -1, :] = 1.
    added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))

    if torch.cuda.is_available():
        added_adj_zero_row = added_adj_zero_row.cuda()
        added_adj_zero_col = added_adj_zero_col.cuda()
        added_feat_zero_row = added_feat_zero_row.cuda()
    time_start = time.time()
    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):
            loss_full_batch = torch.zeros((nb_nodes, 1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):
                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

                ba = []
                bf = []
                ba_sgn0 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    cur_adj_sgn = adj_org[:, subgraphs[i], :][:, :, subgraphs[i]]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    ba_sgn0.append(cur_adj_sgn)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                ba_sgn0 = torch.cat(ba_sgn0)
                bf_sgn0 = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                ba_sgn1, bf_sgn1 = to_sgn(ba_sgn0, bf_sgn0)
                ba_sgn2, bf_sgn2 = to_sgn(ba_sgn1, bf_sgn1)
                ba_sgn1 = normalize_adj_torch(ba_sgn1)
                ba_sgn2 = normalize_adj_torch(ba_sgn2)
                ba_sgn1 = ba_sgn1.cuda()
                bf_sgn1 = bf_sgn1.cuda()
                ba_sgn2 = ba_sgn2.cuda()
                bf_sgn2 = bf_sgn2.cuda()
                logits, _ = model(bf, ba, bf_sgn1, ba_sgn1, bf_sgn1, ba_sgn1)
                loss_all = b_xent(logits, lbl)

                loss = torch.mean(loss_all)

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)

    # Test model
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pkl'))

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    nodes_embed = torch.zeros([nb_nodes, args.embedding_dim], dtype=torch.float).cuda()

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                ba = []
                bf = []
                ba_sgn0 = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    cur_adj_sgn = adj_org[:, subgraphs[i], :][:, :, subgraphs[i]]
                    ba.append(cur_adj)
                    bf.append(cur_feat)
                    ba_sgn0.append(cur_adj_sgn)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                ba_sgn0 = torch.cat(ba_sgn0)
                bf_sgn0 = torch.cat((bf[:, :-1, :], added_feat_zero_row), dim=1)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
                ba_sgn1, bf_sgn1 = to_sgn(ba_sgn0, bf_sgn0)
                ba_sgn2, bf_sgn2 = to_sgn(ba_sgn1, bf_sgn1)
                ba_sgn1 = normalize_adj_torch(ba_sgn1)
                ba_sgn2 = normalize_adj_torch(ba_sgn2)
                ba_sgn1 = ba_sgn1.cuda()
                bf_sgn1 = bf_sgn1.cuda()
                ba_sgn2 = ba_sgn2.cuda()
                bf_sgn2 = bf_sgn2.cuda()

                with torch.no_grad():
                    logits, batch_embed = model(bf, ba, bf_sgn1, ba_sgn1, bf_sgn1, ba_sgn1)
                    logits = torch.squeeze(logits)
                    logits = torch.sigmoid(logits)

                    if round == args.auc_test_rounds - 1:
                        nodes_embed[idx] = batch_embed

                ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                # ano_score_p = - logits[:cur_batch_size].cpu().numpy()
                # ano_score_n = logits[cur_batch_size:].cpu().numpy()

                multi_round_ano_score[round, idx] = ano_score
                # multi_round_ano_score_p[round, idx] = ano_score_p
                # multi_round_ano_score_n[round, idx] = ano_score_n

            pbar_test.update(1)

    attr_ano_score_final = np.mean(multi_round_ano_score, axis=0)
    attr_scaler = MinMaxScaler()
    attr_ano_score_final = attr_scaler.fit_transform(attr_ano_score_final.reshape(-1, 1)).reshape(-1)

    features_norm = F.normalize(nodes_embed, p=2, dim=1)
    features_similarity = torch.matmul(features_norm, features_norm.transpose(0, 1)).squeeze(0).cpu()

    net = nx.from_numpy_array(adj_raw)
    net.remove_edges_from(nx.selfloop_edges(net))
    adj_raw = nx.to_numpy_array(net)
    multi_round_stru_ano_score = []
    list_temp = list(algorithms.scan(net, 0.3, 5).communities)
    degree = np.squeeze(np.array(degree))
    for set in list_temp:
            core_temp_size = len(set)
            similar_temp = 0
            similar_num = 0
            scores_temp = np.zeros(nb_nodes)
            for idx in set:
                    for idy in set:
                        if idx != idy:
                            similar_temp += features_similarity[idx][idy]
                            similar_num += 1
            scores_temp[list(set)] = core_temp_size * 1 / (similar_temp / similar_num)
            multi_round_stru_ano_score.append(scores_temp)

    multi_round_stru_ano_score = np.array(multi_round_stru_ano_score)
    multi_round_stru_ano_score = np.mean(multi_round_stru_ano_score, axis=0)
    stru_scaler = MinMaxScaler()
    stru_ano_score_final = stru_scaler.fit_transform(multi_round_stru_ano_score.reshape(-1, 1)).reshape(-1)
    # ano_score_final_p = np.mean(multi_round_ano_score_p, axis=0)
    # ano_score_final_n = np.mean(multi_round_ano_score_n, axis=0)
    alpha_list = list([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    for i in alpha_list:
        ano_score_final = i * attr_ano_score_final  + (1-i) * stru_ano_score_final
        auc = roc_auc_score(ano_label, ano_score_final)
        torch.save(ano_score_final,"ADGAD_{}_{}.pt".format(i,auc))
        print('i:',i,'AUC:{:.4f}'.format(auc))
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)



