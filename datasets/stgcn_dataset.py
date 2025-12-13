import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler


class STGCNDataset(Dataset):
    def __init__(self, epyt_helper,
                 lookback=12,
                 horizon=3,
                 mode='train',
                 split_ratio=[0.9, 0.1],
                 scaler=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.mode = mode

        # 1. 获取原始数据
        raw_data = epyt_helper.get_raw_data()

        # [Time, Nodes]
        self.pressures = raw_data['pressures'].astype(np.float32)
        self.demands = raw_data['demand_real'].astype(np.float32)

        total_time, self.num_nodes = self.pressures.shape

        # 2. 数据划分
        train_end = int(total_time * split_ratio[0])
        val_end = int(total_time * (split_ratio[0] + split_ratio[1]))

        if mode == 'train':
            self.indices = range(0, train_end - lookback - horizon + 1)
        elif mode == 'val':
            self.indices = range(train_end, val_end - lookback - horizon + 1)
        elif mode == 'test':
            self.indices = range(train_end, val_end - lookback - horizon + 1)

        # 3. 归一化 (StandardScaler)
        if mode == 'train':
            self.scaler = StandardScaler()
            self.pressures_norm = self.scaler.fit_transform(self.pressures)

            # Demand 可以用另一个 scaler，或者简单的 MinMax
            self.demand_scaler = StandardScaler()
            self.demands_norm = self.demand_scaler.fit_transform(self.demands)
        else:
            if scaler is None:
                raise ValueError("Scaler required for val/test")
            self.scaler = scaler['pressure']
            self.demand_scaler = scaler['demand']

            self.pressures_norm = self.scaler.transform(self.pressures)
            self.demands_norm = self.demand_scaler.transform(self.demands)

    def get_scalers(self):
        return {'pressure': self.scaler, 'demand': self.demand_scaler}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # --- Input X ---
        # 目标形状: [Channels, Time, Nodes]
        # C=0: Pressure, C=1: Demand

        p_seq = self.pressures_norm[t: t + self.lookback, :]  # [L, N]
        d_seq = self.demands_norm[t: t + self.lookback, :]  # [L, N]

        # 转置为 [Time, Nodes] -> 堆叠 -> [2, Time, Nodes]
        x_data = np.stack([p_seq, d_seq], axis=0)

        # --- Label Y ---
        # 目标形状: [Horizon, Nodes] (预测未来几步的压力)
        # 这里只预测压力
        y_data = self.pressures_norm[t + self.lookback: t + self.lookback + self.horizon, :]

        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)