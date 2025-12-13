import os

import numpy as np
import scipy.sparse as sp
import torch
import argparse

from datasets.epanet_data import EpytHelper
from model.GWN import gwnet
from datasets.WaterDataset import  WaterDataset
from trainer import ModelTrainer
from tqdm import tqdm

# 定义GWN专用训练器，处理输出维度问题
class GWNTrainer(ModelTrainer):
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(x)

            # GWN特定处理：调整输出维度以匹配目标
            output = output.permute(0,3,1,2)  # [B,C,N,1] -> [B,1,N,C]

            # 计算损失
            loss = self.criterion(output, y)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                output = self.model(x)

                # GWN特定处理：调整输出维度以匹配目标
                output = output.permute(0,3,1,2) # [B,C,N,1] -> [B,1,N,C]

                loss = self.criterion(output, y)
                total_loss += loss.item()

        return total_loss / len(val_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='/data/zsm/case01/data/d-town.inp')
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_save_path', type=str, default='./models/gwn_best.pth')

    # GWN 特定参数
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_bool', type=bool, default=True)
    parser.add_argument('--addaptadj', type=bool, default=True)  # 是否学习自适应图
    parser.add_argument('--aptonly', type=bool, default=False)  # 是否只使用自适应图(忽略物理图)
    parser.add_argument('--randomadj', type=bool, default=True)
    parser.add_argument('--nhid', type=int, default=32)

    return parser.parse_args()


class GWNDataLoaderWrapper:
    """
    输入包装器：将 PyG [T, N] 转换为 GWN 所需的 [Batch, C, N, T]
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            batch = batch.to(self.device)

            x = batch.x_pressure
            x = x.unsqueeze(0).unsqueeze(0)
            x=x.permute(0,1,3,2).contiguous()
            y = batch.y_pressure
            y = y.unsqueeze(0).unsqueeze(0)

            yield x, y

    def __len__(self):
        return len(self.dataloader)


def calculate_normalized_adj(edge_index, num_nodes):
    """
    计算行归一化的邻接矩阵 D^-1 A (Asymmetric Normalization)
    用于 Graph WaveNet 的 diffusion convolution
    """
    row, col = edge_index
    data = np.ones(len(row))
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)

    # 构建有向/无向图的邻接矩阵策略
    # 此处构建最大连通性：A = A + A.T (视为双向连通) + I
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(num_nodes)

    # Row normalization
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    normalized_adj = d_mat_inv.dot(adj).astype(np.float32)

    # 返回稠密矩阵列表 (GWN supports 参数是一个 list)
    return [torch.from_numpy(normalized_adj.todense())]


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"File {args.data_file} not found.")

    print("Loading raw data...")
    ep_helper = EpytHelper(args.data_file)
    raw_data = ep_helper.get_raw_data()

    # 2. 初始化 Dataset
    print("Initializing Dataset...")
    dataset = WaterDataset(
        raw_data=raw_data,
        n_his=args.n_his,
        n_pred=args.n_pred
    )

    # 3. 计算物理图结构 (Supports)
    print("Calculating Adjacency Matrix...")
    sample_data = dataset[0]
    edge_index = sample_data.edge_index.numpy()
    num_nodes = sample_data.num_nodes

    # 计算归一化邻接矩阵 (并转为 Tensor 放到 Device)
    # GWN 接受 supports 为列表，例如 [adj] 或 [adj_forward, adj_backward]
    adj_mx = calculate_normalized_adj(edge_index, num_nodes)
    supports = [adj.to(device) for adj in adj_mx]

    # 4. 包装 DataLoaders
    # 先获取 PyG Loader
    train_loader_pyg, val_loader_pyg, test_loader_pyg = dataset.gen_train_loader(
        batch_size=args.batch_size,
        shuffle=True
    )

    # 再包装成 GWN 格式 [B, 1, N, T]
    train_loader = GWNDataLoaderWrapper(train_loader_pyg, device)
    val_loader = GWNDataLoaderWrapper(val_loader_pyg, device)
    test_loader = GWNDataLoaderWrapper(test_loader_pyg, device)

    # 5. 创建模型
    print("Building Graph WaveNet...")
    model = gwnet(
        device=device,
        num_nodes=num_nodes,
        in_dim=1,  # 输入特征数 (压力)
        out_dim=args.n_pred,  # 输出时间步长 (GWN最后通常映射到预测长度)
        dropout=args.dropout,
        supports=supports,  # 物理图
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,  # 开启自适应图学习
        aptinit=None,  # 自适应图随机初始化
        layers=2,  # 默认层数
        blocks=4,  # 默认块数
        kernel_size=2  # TCN 卷积核大小
    ).to(device)


    # 6. 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    # 使用通用 Trainer
    trainer = GWNTrainer(model, device, optimizer)

    # 注入 scaler 用于测试
    if hasattr(trainer, 'scaler'):
        trainer.scaler = dataset.pressure_scaler

    print("Start Training...")
    trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        model_save_path=args.model_save_path
    )

    print("Testing...")
    trainer.test(test_loader, args.model_save_path)


if __name__ == '__main__':
    main()