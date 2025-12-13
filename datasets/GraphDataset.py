import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from epanet_data import EpytHelper
import pandas as pd
class WDNDataset(Dataset):
    def __init__(self, raw_data,
                 lookback=12,  # 输入过去多少个时间步 (例如1小时: 12 * 5min)
                 horizon=1,  # 预测未来多少个时间步
                 mode='train',  # 'train', 'val', 'test'
                 split_ratio=[0.7, 0.1, 0.2],
                 scaler=None):  # 传入已有的scaler以保证训练/测试一致性
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon
        self.mode = mode

        # 提取动态数据 (Shape: [Time, Nodes/Edges])
        # 注意: 转换为 float32 以适应 PyTorch
        self.pressures = raw_data['pressures'].astype(np.float32)  # [T, N]
        self.flows = raw_data['flows'].astype(np.float32)  # [T, E]
        self.demands = raw_data['demand_real'].astype(np.float32)  # [T, N]

        # 提取静态数据
        self.node_elevations = raw_data['node_elevations'].astype(np.float32)  # [N]
        self.node_type = raw_data['node_type'].astype(np.float32)  # [N]

        self.link_lengths = raw_data['link_lengths'].astype(np.float32)  # [E]
        self.link_diameters = raw_data['link_diameters'].astype(np.float32)  # [E]
        self.link_roughnesses = raw_data['link_roughnesses'].astype(np.float32)  # [E]

        # 拓扑结构
        self.edge_index = torch.tensor(raw_data['edge_index_directed'], dtype=torch.long)

        # 2. 数据划分 (按时间维度切分)
        total_time = self.pressures.shape[0]
        train_end = int(total_time * split_ratio[0])
        val_end = int(total_time * (split_ratio[0] + split_ratio[1]))

        if mode == 'train':
            self.time_indices = range(0, train_end - lookback - horizon)
        elif mode == 'val':
            self.time_indices = range(train_end, val_end - lookback - horizon)
        elif mode == 'test':
            self.time_indices = range(val_end, total_time - lookback - horizon)

        # 3. 数据归一化处理
        # 如果是训练集，我们需要fit scaler；如果是验证/测试集，我们使用传入的scaler
        if scaler is None and mode == 'train':
            self.scaler = {
                'pressure': StandardScaler(),
                'flow': StandardScaler(),
                'demand': StandardScaler(),
                'elevation': MinMaxScaler(),
                'length': MinMaxScaler(),
                'diameter': MinMaxScaler(),
                'roughness': MinMaxScaler()
            }
            # Fit transforms
            self.pressures = self.scaler['pressure'].fit_transform(self.pressures)
            self.flows = self.scaler['flow'].fit_transform(self.flows)
            self.demands = self.scaler['demand'].fit_transform(self.demands)

            # Static features (reshape needed for sklearn)
            self.node_elevations = self.scaler['elevation'].fit_transform(self.node_elevations.reshape(-1, 1)).flatten()
            self.link_lengths = self.scaler['length'].fit_transform(self.link_lengths.reshape(-1, 1)).flatten()
            self.link_diameters = self.scaler['diameter'].fit_transform(self.link_diameters.reshape(-1, 1)).flatten()
            self.link_roughnesses = self.scaler['roughness'].fit_transform(
                self.link_roughnesses.reshape(-1, 1)).flatten()

        elif scaler is not None:
            self.scaler = scaler
            # Transform only
            self.pressures = self.scaler['pressure'].transform(self.pressures)
            self.flows = self.scaler['flow'].transform(self.flows)
            self.demands = self.scaler['demand'].transform(self.demands)

            self.node_elevations = self.scaler['elevation'].transform(self.node_elevations.reshape(-1, 1)).flatten()
            self.link_lengths = self.scaler['length'].transform(self.link_lengths.reshape(-1, 1)).flatten()
            self.link_diameters = self.scaler['diameter'].transform(self.link_diameters.reshape(-1, 1)).flatten()
            self.link_roughnesses = self.scaler['roughness'].transform(self.link_roughnesses.reshape(-1, 1)).flatten()
        else:
            raise ValueError("Scaler must be provided for validation/test sets")

        # 将数据转为 Tensor
        self.pressures = torch.tensor(self.pressures)
        self.flows = torch.tensor(self.flows)
        self.demands = torch.tensor(self.demands)
        self.node_elevations = torch.tensor(self.node_elevations)
        self.node_type = torch.tensor(self.node_type)
        self.link_lengths = torch.tensor(self.link_lengths)
        self.link_diameters = torch.tensor(self.link_diameters)
        self.link_roughnesses = torch.tensor(self.link_roughnesses)

    def len(self):
        return len(self.time_indices)

    def get(self, idx):
        # 实际的时间索引
        t = self.time_indices[idx]

        # --- 构建节点特征 Node Features (x) ---
        # 形状: [Num_Nodes, Num_Features]
        # 特征工程: 过去 lookback 时间步的 [Pressure, Demand] + Static [Elevation, Type]
        # 动态特征被展平或作为序列通道

        # 这里我们采用一种简单的拼接方式：
        # x_dynamic: [N, lookback * 2] (Pressure + Demand)
        p_window = self.pressures[t: t + self.lookback, :].T  # [N, L]
        d_window = self.demands[t: t + self.lookback, :].T  # [N, L]

        x_static = torch.stack([self.node_elevations, self.node_type], dim=1)  # [N, 2]

        # 拼接: [N, L + L + 2]
        x = torch.cat([p_window, d_window, x_static], dim=1)

        # --- 构建边特征 Edge Attributes (edge_attr) ---
        # 形状: [Num_Edges, Num_Features]
        # 特征: 过去 lookback 时间步的 [Flow] + Static [Length, Diameter, Roughness]

        f_window = self.flows[t: t + self.lookback, :].T  # [E, L]
        edge_static = torch.stack([self.link_lengths, self.link_diameters, self.link_roughnesses], dim=1)  # [E, 3]

        edge_attr = torch.cat([f_window, edge_static], dim=1)

        # --- 构建标签 Targets (y) ---
        # 预测未来的 Pressure (节点) 和 Flow (管道)
        # y_pressure: [N, Horizon]
        # y_flow: [E, Horizon]

        y_p = self.pressures[t + self.lookback: t + self.lookback + self.horizon, :].T
        y_f = self.flows[t + self.lookback: t + self.lookback + self.horizon, :].T

        # PyG 的 Data 对象通常只能持有一个 y，或者我们可以自定义属性
        # 这里我们将 y 设为压力，y_flow 设为额外属性
        data = Data(x=x,
                    edge_index=self.edge_index,
                    edge_attr=edge_attr,
                    y=y_p,
                    y_flow=y_f)

        return data

# INP_PATH = '/data/zsm/case01/data/d-town.inp'  # 修改为你的实际路径
# BATCH_SIZE = 4  # 设置一个小 Batch 方便观察
# LOOKBACK = 12  # 过去12个时间步
# HORIZON = 1  # 预测未来1步
#
# print(f"--- 1. 初始化仿真器: {INP_PATH} ---")
# sim = EpytHelper(INP_PATH, hrs=168)
#
# print("\n--- 2. 构建数据集 ---")
# train_dataset = WDNDataset(
#     raw_data=sim.get_raw_data(),
#     lookback=LOOKBACK,
#     horizon=HORIZON,
#     mode='train',
#     split_ratio=[0.8, 0.1, 0.1]
# )
#
# loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# # ==============================
# # 2. 获取并检查一个 Batch
# # ==============================
# batch = next(iter(loader))
#
# print("\n" + "=" * 40)
# print("       BATCH 数据结构检查       ")
# print("=" * 40)
#
# # 获取网络基本信息
# num_nodes = 407
# num_edges = 459
#
# print(f"PyG Batch 对象: \n{batch}")
# print(f"\n[拓扑信息]")
# print(f"  - 原始图节点数 (N): {num_nodes}")
# print(f"  - 原始图边数   (E): {num_edges}")
# print(f"  - Batch Size   (B): {BATCH_SIZE}")
# print(f"  - Batch 中总节点数 (B*N): {batch.num_nodes} (应等于 {BATCH_SIZE * num_nodes})")
# print(f"  - Batch 中总边数   (B*E): {batch.num_edges} (应等于 {BATCH_SIZE * num_edges})")
#
# print(f"\n[特征维度 (Inputs)]")
# # x: [Batch_Nodes, Features]
# # Features = 2 * lookback (Pressure+Demand) + 2 (Elevation+Type)
# print(f"  - x (节点特征): {batch.x.shape} -> [Batch*N, 2*L + 2]")
# print(f"  - edge_attr (边特征): {batch.edge_attr.shape} -> [Batch*E, L + 3]")
# print(f"  - edge_index (连接关系): {batch.edge_index.shape}")
#
# print(f"\n[标签维度 (Targets)]")
# print(f"  - y (未来压力): {batch.y.shape} -> [Batch*N, Horizon]")
# print(f"  - y_flow (未来流量): {batch.y_flow.shape} -> [Batch*E, Horizon]")
#
# # ==============================
# # 3. 数值统计与分布
# # ==============================
# print("\n" + "=" * 40)
# print("       数值统计 (归一化后)       ")
# print("=" * 40)
#
# # 转换 x 为 Pandas DataFrame 方便查看前几行
# feature_names = [f'P_t-{i}' for i in range(LOOKBACK)] + \
#                 [f'D_t-{i}' for i in range(LOOKBACK)] + \
#                 ['Elevation', 'Type']
#
# df_x = pd.DataFrame(batch.x.numpy(), columns=feature_names)
#
# print("\n1. 节点特征 (前5行数据样本):")
# print(df_x.iloc[:5, [0, 1, LOOKBACK, LOOKBACK + 1, -2, -1]])  # 打印部分列：压力, 需求, 静态特征
#
# print("\n2. 数据范围统计 (检查是否 Normalize 成功):")
# stats = df_x.describe().loc[['mean', 'std', 'min', 'max']]
# print(stats.iloc[:, [0, LOOKBACK, -2]])  # 只看 Pressure, Demand, Elevation 的统计
#
# # 简单的断言检查
# assert not torch.isnan(batch.x).any(), "错误: 输入特征 x 中包含 NaN !"
# assert not torch.isnan(batch.y).any(), "错误: 标签 y 中包含 NaN !"
# print("\n>> 数据完整性检查通过 (无 NaN)")
#
# # ==============================
# # 4. 反归一化验证 (物理意义检查)
# # ==============================
# print("\n" + "=" * 40)
# print("       反归一化验证 (物理单位)       ")
# print("=" * 40)
#
# scaler_p = train_dataset.scaler['pressure']
# scaler_f = train_dataset.scaler['flow']
#
# # --- 修正代码开始 ---
#
# # 1. 提取前 N 个节点的数据 (对应第1个图)
# # batch.y shape: [B*N, 1] -> 取出 [N, 1]
# y_norm = batch.y[:num_nodes].detach().cpu().numpy()
#
# # 2. Reshape 以匹配 Scaler 的特征数
# # Scaler 期望输入: [Samples, Features(Nodes)] -> [1, 407]
# # 我们将 [407, 1] 转置为 [1, 407]
# y_real_row = scaler_p.inverse_transform(y_norm.reshape(1, -1))
#
# # 3. 转回列向量方便打印
# y_real = y_real_row.reshape(-1, 1)
#
# # 对流量做同样的处理
# # batch.y_flow shape: [B*E, 1] -> 取出 [E, 1]
# yf_norm = batch.y_flow[:num_edges].detach().cpu().numpy()
#
# # Reshape: [E, 1] -> [1, E] -> inverse -> [E, 1]
# yf_real = scaler_f.inverse_transform(yf_norm.reshape(1, -1)).reshape(-1, 1)
#
# # --- 修正代码结束 ---
#
# print(f"示例: 节点 0 的预测目标 (未来第1个时刻)")
# print(f"  - 归一化压力值: {y_norm[0, 0]:.4f}")
# print(f"  - 真实压力值  : {y_real[0, 0]:.4f} m (水头)")
#
# print(f"示例: 管道 0 的预测目标")
# print(f"  - 归一化流量值: {yf_norm[0, 0]:.4f}")
# print(f"  - 真实流量值  : {yf_real[0, 0]:.6f} m³/s")