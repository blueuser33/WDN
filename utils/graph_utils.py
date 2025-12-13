import numpy as np
import torch
import scipy.sparse as sp


def get_adj_from_epyt(epyt_helper):
    """
    从 EpytHelper 中提取邻接矩阵 (NxN)
    """
    raw_data = epyt_helper.get_raw_data()
    edge_index = raw_data['edge_index_directed']  # [2, E]
    num_nodes = raw_data['pressures'].shape[1]

    # 构建邻接矩阵 (无向图或有向图视具体模型需求，通常管网视为无向通达)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src, dst] = 1
        adj[dst, src] = 1  # 对称化

    return adj


def get_normalized_adj(W_nodes):
    """
    计算 GCN 需要的 D^-0.5 * (A+I) * D^-0.5
    """
    W_nodes = W_nodes + np.eye(W_nodes.shape[0])  # Add self-loops
    D = np.array(np.sum(W_nodes, axis=1)).flatten()
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return D_mat_inv_sqrt.dot(W_nodes).dot(D_mat_inv_sqrt)


def generate_cheb_poly(L, K):
    """
    生成 K 阶切比雪夫多项式，用于 ChebConv
    """
    support = []
    L = np.array(L)
    n_node = L.shape[0]

    # 将拉普拉斯矩阵的特征值缩放到 [-1, 1]
    w, v = np.linalg.eig(L)
    max_eig_val = np.max(w)
    if max_eig_val > 0:
        L = (2 / max_eig_val) * L - np.eye(n_node)

    # 递归计算: T0=I, T1=L, Tk = 2*L*Tk-1 - Tk-2
    support.append(np.eye(n_node))
    support.append(L)
    for k in range(2, K):
        support.append(2 * L.dot(support[-1]) - support[-2])

    return np.stack(support, axis=0)  # [K, N, N]