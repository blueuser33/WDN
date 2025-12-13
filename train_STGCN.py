import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model.STGCN import STGCNChebGraphConv
from datasets.epanet_data import EpytHelper  # 你的 EpytHelper
from datasets.stgcn_dataset import STGCNDataset
from utils.graph_utils import get_adj_from_epyt, get_normalized_adj, generate_cheb_poly


def get_args():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--inp_path', type=str, default='/data/zsm/case01/data/d-town.inp')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--figure_dir', type=str, default='./figures')

    # STGCN 参数
    parser.add_argument('--n_his', type=int, default=12, help='Lookback window')
    parser.add_argument('--n_pred', type=int, default=3, help='Horizon window')
    parser.add_argument('--Kt', type=int, default=3, help='Temporal kernel size')
    parser.add_argument('--Ks', type=int, default=3, help='Spatial kernel size (Cheb order)')
    parser.add_argument('--act_func', type=str, default='glu')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv')
    parser.add_argument('--enable_bias', action='store_true', default=True)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--gso_type', type=str, default='cheb', help='cheb or sym_norm')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)

    return parser.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    device = torch.device(args.device)

    print("1. Loading Data...")
    sim = EpytHelper(args.inp_path, hrs=168)
    raw_data = sim.get_raw_data()
    n_vertex = raw_data['pressures'].shape[1]

    # 构建数据集
    train_ds = STGCNDataset(sim, args.n_his, args.n_pred, 'train')
    scalers = train_ds.get_scalers()
    val_ds = STGCNDataset(sim, args.n_his, args.n_pred, 'val', scaler=scalers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"   Nodes: {n_vertex}, Train Samples: {len(train_ds)}")

    # 2. 构建图结构 (GSO)
    print("2. Processing Graph Structure...")
    adj = get_adj_from_epyt(sim)  # [N, N]

    if args.gso_type == 'cheb':
        # 切比雪夫多项式
        L = get_normalized_adj(adj)
        gso = generate_cheb_poly(L, 2)[1]
        gso = torch.from_numpy(gso).float().to(device)
    else:
        # 普通归一化邻接矩阵
        norm_adj = get_normalized_adj(adj)
        gso = torch.from_numpy(norm_adj).float().to(device)

    # 这里的 gso 会传入模型
    setattr(args, 'gso', gso)

    # 3. 定义模型 Blocks
    # Input Channel = 2 (Pressure, Demand)
    # 结构: [In, Hidden, Out]
    # Block 1: 2 -> 64 -> 64
    # Block 2: 64 -> 64 -> 128
    # 最后一个列表通常定义 Output Block 之前的全连接维度
    blocks = [
        [2],                 # blocks[0][-1] = 2 (Input: Pressure + Demand)
        [64, 32, 64],        # blocks[1]: ST-Conv1 (通道: 2 -> 64)
        [64, 32, 64],        # blocks[2]: ST-Conv2 (通道: 64 -> 64), 也是 blocks[-3]
        [128,64],               # blocks[3]: Output FC Hidden (128), 也是 blocks[-2]
        [args.n_pred]        # blocks[4]: Output Horizon (3), 也是 blocks[-1]
    ]
    print("3. Initializing STGCN...")
    model = STGCNChebGraphConv(args, blocks, n_vertex).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 4. 训练循环
    print("4. Start Training...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            # x shape: [B, 2, T, N]
            # y shape: [B, T_out, N]
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            output = model(x)

            if output.dim() == 4:
                preds = output.squeeze(2)
            else:
                preds = output

            # 如果模型输出的时间步和 Label 不一致，取 min
            min_len = min(preds.shape[1], y.shape[1])
            loss = criterion(preds[:, :min_len, :], y[:, :min_len, :])

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                if output.dim() == 4:
                    preds = output.squeeze(2)
                else:
                    preds = output

                min_len = min(preds.shape[1], y.shape[1])
                val_loss += criterion(preds[:, :min_len, :], y[:, :min_len, :]).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step()

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        print(f"Epoch {epoch + 1:02d} | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'StgcnDtown.pth'))

    # 画图
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.legend()
    plt.savefig(os.path.join(args.figure_dir, 'StgcnDtown_loss.png'))
    print("Done.")


if __name__ == "__main__":
    main()