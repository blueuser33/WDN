import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class WaterDataset(Dataset):
    """统一的数据预处理模块"""

    def __init__(self, raw_data):
        self.normalized_data=self._normalize_data(raw_data)
        self._build_list(self.normalized_data)

    def _build_list(self,normalized_data):
        node_static = torch.from_numpy(
            np.stack([normalized_data['node_elevations'], normalized_data['node_demands']], axis=1)).float()
        node_type = torch.from_numpy(normalized_data['node_type']).long()
        reservoir_index = torch.from_numpy(normalized_data['reservoir_indices']).long()

        edge_static_attr_directed = np.stack(
            [normalized_data['link_diameters'], normalized_data['link_lengths'], normalized_data['link_roughnesses']], axis=1)

        edge_index_directed = torch.from_numpy(normalized_data['edge_index_directed']).long()
        row, col = edge_index_directed
        edge_index = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
        edge_static_attr = torch.from_numpy(
            np.concatenate([edge_static_attr_directed, edge_static_attr_directed], axis=0)).float()

        num_nodes = len(normalized_data['node_elevations'])
        graph_topology = Data(edge_index=edge_index, num_nodes=num_nodes)
        structural_data = preprocess_for_graph_transformer(graph_topology)

        self.raw_data_list = []
        for t in range(normalized_data['pressures'].shape[0]):
            data = Data()
            data.node_static = node_static
            data.node_type = node_type
            data.reservoir_index = reservoir_index
            data.pressure = torch.from_numpy(raw_data['pressures'][t]).float().view(-1, 1)
            data.edge_index = edge_index
            data.edge_static_attr = edge_static_attr
            flow_directed = torch.from_numpy(raw_data['flows'][t]).float().view(-1, 1)
            data.flow = torch.cat([flow_directed, -flow_directed], dim=0)
            data.degree_encoding = structural_data['degree_encoding']
            data.spd_matrix = structural_data['spd_matrix']
            data.edge_map = structural_data['edge_map']
            data.num_nodes = num_nodes
            self.raw_data_list.append(data)
    def _normalize_data(self,raw_data):
        self.pressure_scaler = StandardScaler()
        self.flow_scaler = StandardScaler()
        self.demand_scaler = StandardScaler()

        # 节点静态特征归一化器
        self.elevation_scaler = StandardScaler()
        self.node_demand_scaler = StandardScaler()
        # node_type 是分类变量，不归一化
        # reservoir_indices 是索引，不归一化

        # 边静态特征归一化器
        self.diameter_scaler = StandardScaler()
        self.length_scaler = StandardScaler()
        self.roughness_scaler = StandardScaler()
        self.is_fitted = False
        self.fit(raw_data)
        return self.transform(raw_data)

    def fit(self, raw_data):
        """拟合所有归一化器"""
        print("Fitting normalizers for water network data...")

        # 动态特征拟合
        self.pressure_scaler.fit(raw_data['pressures'].reshape(-1, 1))
        self.flow_scaler.fit(raw_data['flows'].reshape(-1, 1))
        self.demand_scaler.fit(raw_data['demand_real'].reshape(-1, 1))

        # 节点静态特征拟合
        self.elevation_scaler.fit(raw_data['node_elevations'].reshape(-1, 1))
        self.node_demand_scaler.fit(raw_data['node_demands'].reshape(-1, 1))

        # 边静态特征拟合
        self.diameter_scaler.fit(raw_data['link_diameters'].reshape(-1, 1))
        self.length_scaler.fit(raw_data['link_lengths'].reshape(-1, 1))
        self.roughness_scaler.fit(raw_data['link_roughnesses'].reshape(-1, 1))


    def transform(self, raw_data):
        """转换数据"""

        normalized_data = {}

        # 动态特征转换
        T, N = raw_data['pressures'].shape
        E = raw_data['link_diameters'].shape[0]

        normalized_data['pressures'] = self.pressure_scaler.transform(
            raw_data['pressures'].reshape(-1, 1)
        ).reshape(T, N)

        normalized_data['flows'] = self.flow_scaler.transform(
            raw_data['flows'].reshape(-1, 1)
        ).reshape(T, E)

        normalized_data['demand_real'] = self.demand_scaler.transform(
            raw_data['demand_real'].reshape(-1, 1)
        ).reshape(T, N)

        # 节点静态特征转换
        normalized_data['node_elevations'] = self.elevation_scaler.transform(
            raw_data['node_elevations'].reshape(-1, 1)
        ).flatten()

        normalized_data['node_demands'] = self.node_demand_scaler.transform(
            raw_data['node_demands'].reshape(-1, 1)
        ).flatten()

        # 分类和索引变量直接复制
        normalized_data['node_type'] = raw_data['node_type'].copy()
        normalized_data['reservoir_indices'] = raw_data['reservoir_indices'].copy()
        normalized_data['edge_index_directed'] = raw_data['edge_index_directed'].copy()

        # 边静态特征转换
        normalized_data['link_diameters'] = self.diameter_scaler.transform(
            raw_data['link_diameters'].reshape(-1, 1)
        ).flatten()

        normalized_data['link_lengths'] = self.length_scaler.transform(
            raw_data['link_lengths'].reshape(-1, 1)
        ).flatten()

        normalized_data['link_roughnesses'] = self.roughness_scaler.transform(
            raw_data['link_roughnesses'].reshape(-1, 1)
        ).flatten()

        return normalized_data

    def inverse_transform_pressure(self, normalized_pressure):
        """反转换压力数据"""
        return self.pressure_scaler.inverse_transform(normalized_pressure)

    def inverse_transform_flow(self, normalized_flow):
        """反转换流量数据"""
        return self.flow_scaler.inverse_transform(normalized_flow)

    def inverse_transform_demand(self, normalized_demand):
        """反转换需求数据"""
        return self.demand_scaler.inverse_transform(normalized_demand)
    def _create_gso(self):
        """创建图移位算子(Graph Shift Operator)"""
        # 根据边索引创建邻接矩阵
        adj_matrix = np.zeros((self.node_count, self.node_count))
        for i in range(self.edge_index.shape[1]):
            u, v = self.edge_index[0, i], self.edge_index[1, i]
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1  # 无向图

        # 归一化邻接矩阵
        degree = np.sum(adj_matrix, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5, where=degree != 0)
        degree_inv_sqrt[degree == 0] = 0
        laplacian = np.eye(self.node_count) - degree_inv_sqrt[:, np.newaxis] * adj_matrix * degree_inv_sqrt[np.newaxis,
                                                                                            :]

        return torch.tensor(laplacian, dtype=torch.float32)

    def get_model_specific_data(self, model_type, n_his=12, n_pred=3, batch_size=32):
        """获取特定模型所需的数据"""
        if model_type == 'STGCN':
            dataset = STGCNDataset(
                self.raw_data['pressures_norm'],
                n_his, n_pred
            )
        elif model_type == 'GWN':
            dataset = GWNDataset(
                self.raw_data['pressures_norm'],
                n_his, n_pred
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 分割数据集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)

        train_set = torch.utils.data.Subset(dataset, range(0, train_size))
        val_set = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        test_set = torch.utils.data.Subset(dataset, range(train_size + val_size, total_size))

        # 创建数据加载器
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # 返回数据加载器、图结构信息和节点数量
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'gso': self.gso,
            'num_nodes': self.node_count,
            'pressure_mean': self.pressure_mean,
            'pressure_std': self.pressure_std
        }


