import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from numpy.lib import pad



def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj_raw = adj
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), adj_raw

def normalize_adj_torch(adj):
    """Symmetrically normalize adjacency matrix."""
    identity_matrix1 = torch.eye(adj.shape[1])
    adj = adj.to(torch.float32)
    degree = torch.sum(adj, dim=2)
    d_inv_sqrt = 1. / torch.sqrt(degree)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    normalized_adj = adj @ d_mat_inv_sqrt @ torch.transpose(d_mat_inv_sqrt, dim0=-2, dim1=-1)
    return normalized_adj + identity_matrix1

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat,  idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    nx_graph = nx.from_numpy_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def load_data_py(dataset):
    data = torch.load("./pygod_benchmark/{}.pt".format(dataset))
    adj = sp.coo_matrix(
        (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(data.num_nodes, data.num_nodes))
    features = data.x.numpy()
    features = sp.lil_matrix(features)
    label = (data.y.int()).numpy()
    adj = sp.csr_matrix(adj)
    return adj, features, label

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv


def to_sgn(adjacency_matrix, node_attributes, subgraph_size=4):
    batch_size, nb, feature_size = node_attributes.shape
    adjacency_matrix = adjacency_matrix.cpu().numpy()
    node_attributes = node_attributes.cpu().numpy()
    max_size = 0
    ba_sgn = []
    bf_sgn = []
    for i in range(batch_size):
        G = nx.Graph(adjacency_matrix[i, :, :])
        # 给原始图添加节点属性
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            ba_sgn.append(np.zeros((1,1)))
            bf_sgn.append(np.zeros((1, feature_size)))  # 适当的属性矩阵形状
            # 处理空图的情况
            continue
        nx.set_node_attributes(G, {j: {'attribute': attr} for j, attr in enumerate(node_attributes[i, :, :])})
        # 创建线图
        graph_to_line = nx.line_graph(G)
        graph_line = nx.convert_node_labels_to_integers(graph_to_line, first_label=0, ordering='default')
        # 创建一个映射，将线图节点映射回原始图中的两个节点
        line_to_original_nodes = {node: nodes for node, nodes in enumerate(graph_to_line.nodes(data='original_nodes'))}
        # 遍历线图的节点
        for node in graph_line.nodes():
            # 获取线图节点对应的原图中的两个节点
            original_nodes = line_to_original_nodes[node]
            # 获取原图节点的属性
            data1, data2 = G.nodes[original_nodes[0][0]]['attribute'], \
                           G.nodes[original_nodes[0][1]]['attribute']
            # 使用 NumPy 计算属性的平均值
            combined_data = np.mean([data1, data2], axis=0)
            # 更新线图节点的属性
            graph_line.nodes[node]['attribute'] = combined_data.tolist()  # 转回列表形式

        if len(graph_line) > subgraph_size:
            selected_nodes = np.random.choice(graph_line.nodes, size=subgraph_size, replace=False)
            graph_line = graph_line.subgraph(selected_nodes)

        adjacency_matrix_sgn = nx.adjacency_matrix(graph_line).toarray()
        ba_sgn.append(adjacency_matrix_sgn)
        attribute_matrix_sgn = np.array([node_data['attribute'] for node, node_data in graph_line.nodes(data=True)])
        bf_sgn.append(attribute_matrix_sgn)
        # 更新最大大小
        max_size = max(max_size, adjacency_matrix_sgn.shape[0])
    # 填充邻接矩阵和属性矩阵
    for i in range(batch_size):
        pad_rows = max_size - ba_sgn[i].shape[0]
        ba_sgn[i] = pad(ba_sgn[i], ((0, pad_rows), (0, pad_rows)), mode='constant', constant_values=0)
        bf_sgn[i] = pad(bf_sgn[i], ((0, pad_rows), (0, 0)), mode='constant', constant_values=0)

    return torch.tensor(np.array(ba_sgn),dtype=torch.float32), torch.tensor(np.array(bf_sgn),dtype=torch.float32)