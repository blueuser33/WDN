# file: generate_epanet_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.epanet_data import EpytHelper
class EpanetTimeSeriesDataset(Dataset):
    """将 EPANET 仿真输出包装成 GraphWaveNet 可训练的数据集"""
    def __init__(self, raw_data, n_his=12, n_pred=3):
        self.pressures = raw_data["pressures"]  # shape [T, N]
        self.num_nodes = self.pressures.shape[1]
        self.n_his = n_his
        self.n_pred = n_pred

        # 归一化（按节点标准化）
        self.mean = self.pressures.mean(axis=0, keepdims=True)
        self.std = self.pressures.std(axis=0, keepdims=True) + 1e-6
        self.pressures_norm = (self.pressures - self.mean) / self.std

    def __len__(self):
        return self.pressures.shape[0] - self.n_his - self.n_pred

    def __getitem__(self, idx):
        x = self.pressures_norm[idx:idx+self.n_his].T  # [N, n_his]
        y = self.pressures_norm[idx+self.n_his:idx+self.n_his+self.n_pred].T  # [N, n_pred]
        return torch.tensor(x).unsqueeze(0), torch.tensor(y).unsqueeze(0)  # [1,N,T]

def build_loaders(raw_data, n_his=12, n_pred=3, batch_size=32, split=(0.7,0.2,0.1)):
    dataset = EpanetTimeSeriesDataset(raw_data, n_his, n_pred)
    total = len(dataset)
    n_train = int(total*split[0])
    n_val = int(total*split[1])
    n_test = total - n_train - n_val

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train,n_val,n_test])
    return (
        DataLoader(train_set,batch_size=batch_size,shuffle=True),
        DataLoader(val_set,batch_size=batch_size,shuffle=False),
        DataLoader(test_set,batch_size=batch_size,shuffle=False)
    )
