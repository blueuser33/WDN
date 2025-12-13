import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler


class STAEformerDataset(Dataset):
    def __init__(self, epyt_helper,
                 lookback=12,
                 horizon=3,
                 mode='train',
                 split_ratio=0.9,
                 scaler=None):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.mode = mode

        # 1. 获取原始数据
        raw_data = epyt_helper.get_raw_data()
        self.pressures = raw_data['pressures'].astype(np.float32)  # [Total_Time, Nodes]
        self.num_nodes = self.pressures.shape[1]
        total_time = self.pressures.shape[0]

        # 2. 生成时间特征 (TOD, DOW)
        steps_per_day = 96

        # 生成全局时间索引
        time_indices = np.arange(total_time)

        # 计算 TOD (0~1)
        self.tod = ((48+time_indices) % steps_per_day) / steps_per_day
        self.tod = self.tod.astype(np.float32)

        # 计算 DOW (0~6)
        # 假设仿真从周一(0)开始
        # self.dow = (time_indices // steps_per_day) % 7
        # self.dow = self.dow.astype(np.float32)

        # 3. 数据划分
        train_end = int(total_time * split_ratio)
        val_end = total_time

        if mode == 'train':
            self.indices = range(0, train_end - lookback - horizon + 1)
            # Fit Scaler
            if scaler is None:
                self.scaler = StandardScaler()
                self.pressures_norm = self.scaler.fit_transform(self.pressures)
            else:
                self.scaler = scaler
                self.pressures_norm = self.scaler.transform(self.pressures)
        elif mode == 'val':
            self.indices = range(train_end, val_end - lookback - horizon + 1)
            self.scaler = scaler
            self.pressures_norm = self.scaler.transform(self.pressures)
        else:  # test
            self.indices = range(train_end, val_end - lookback - horizon + 1)
            self.scaler = scaler
            self.pressures_norm = self.scaler.transform(self.pressures)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # --- 输入 X (Lookback) ---
        # 1. 压力数据 [L, N]
        p_x = self.pressures_norm[t: t + self.lookback, :]

        # 2. TOD [L, N] (扩展维度以匹配节点数)
        tod_x = self.tod[t: t + self.lookback]
        tod_x = np.tile(tod_x[:, np.newaxis], (1, self.num_nodes))

        # 3. DOW [L, N]
        # dow_x = self.dow[t: t + self.lookback]
        # dow_x = np.tile(dow_x[:, np.newaxis], (1, self.num_nodes))

        # 堆叠特征: [L, N, 3] -> (Pressure, TOD, DOW)
        # 注意: STAEformer 要求的 tod/dow 在特定的通道索引 (1和2)
        x_tensor = np.stack([p_x, tod_x], axis=-1)

        # --- 标签 Y (Horizon) ---
        # 预测未来 Horizon 步的压力
        y_tensor = self.pressures_norm[t + self.lookback: t + self.lookback + self.horizon, :]

        return torch.tensor(x_tensor, dtype=torch.float32), torch.tensor(y_tensor, dtype=torch.float32)