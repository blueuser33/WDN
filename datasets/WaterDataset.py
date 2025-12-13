import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datasets.epanet_data import EpytHelper
from datasets.utils import preprocess_for_graph_transformer


class WaterDataset(Dataset):
    """统一的数据预处理模块"""

    def __init__(self, raw_data,n_his=12, n_pred=3):
        super().__init__()
        self.normalized_data=self._normalize_data(raw_data)
        self._build_list(self.normalized_data , n_his ,n_pred)

    def _build_list(self,normalized_data, n_his ,n_pred):
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

        pressures = normalized_data['pressures']  # [T, N]
        flows = normalized_data['flows']  # [T, E]
        demands = normalized_data['demand_real']  # [T, N]

        T = pressures.shape[0]  # 总时间步数

        # 构建时序样本
        self.sequence_data_list = []

        for start_idx in range(0, T - n_his - n_pred + 1):
            end_idx = start_idx + n_his
            pred_start = end_idx
            pred_end = pred_start + n_pred

            data = Data()

            # 静态特征 (不变)
            data.node_static = node_static
            data.node_type = node_type
            data.reservoir_index = reservoir_index
            data.edge_index = edge_index
            data.edge_static_attr = edge_static_attr
            data.degree_encoding = structural_data['degree_encoding']
            data.spd_matrix = structural_data['spd_matrix']
            data.edge_map = structural_data['edge_map']
            data.num_nodes = num_nodes

            # 动态输入特征 [n_his, ...]
            data.x_pressure = torch.from_numpy(pressures[start_idx:end_idx]).float()  # [n_his, N]
            data.x_flow = torch.from_numpy(flows[start_idx:end_idx]).float()  # [n_his, E]
            data.x_demand = torch.from_numpy(demands[start_idx:end_idx]).float()  # [n_his, N]

            # 目标输出特征 [n_pred, ...]
            data.y_pressure = torch.from_numpy(pressures[pred_start:pred_end]).float()  # [n_pred, N]
            data.y_flow = torch.from_numpy(flows[pred_start:pred_end]).float()  # [n_pred, E]

            # 时间索引信息
            data.start_idx = start_idx
            data.end_idx = end_idx
            data.pred_start = pred_start
            data.pred_end = pred_end

            self.sequence_data_list.append(data)
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
        self.wfit(raw_data)
        return self.wtransform(raw_data)

    def wfit(self, raw_data):
        """拟合所有归一化器"""

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


    def wtransform(self, raw_data):
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

    def __len__(self):
        return len(self.sequence_data_list)

    def __getitem__(self, idx):
        return self.sequence_data_list[idx]
    def gen_train_loader(self, train_ratio=0.9, batch_size=1, shuffle=True):
        total_len = len(self)
        train_end = int(total_len * train_ratio)
        train_dataset = Subset(self, range(0, train_end))
        val_dataset   = Subset(self, range(train_end, total_len))
        test_dataset  = Subset(self, range(train_end, total_len))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


# raw_data=EpytHelper('/data/zsm/case01/data/d-town.inp').get_raw_data()
# waterDataset = WaterDataset(raw_data)
# train_loader, val_loader, test_loader=waterDataset.gen_train_loader()
# for i, batch in enumerate(train_loader):
#     print(f"Batch {i}:")
#     print(f"  - 批量大小: {batch.num_graphs}")
#     print(f"  - 节点数量: {batch.num_nodes}")
#     print(f"  - 输入压力形状: {batch.x_pressure.shape}")  # [batch_size, n_his, N]
#     print(f"  - 输入流量形状: {batch.x_flow.shape}")      # [batch_size, n_his, E]
#     print(f"  - 目标压力形状: {batch.y_pressure.shape}")  # [batch_size, n_pred, N]
#     print(f"  - 边索引形状: {batch.edge_index.shape}")    # [2, num_edges * batch_size]