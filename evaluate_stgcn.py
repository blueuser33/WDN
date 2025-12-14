import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 引入项目模块 ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.STGCN import STGCNChebGraphConv
from datasets.epanet_data import EpytHelper
from datasets.stgcn_dataset import STGCNDataset
from utils.graph_utils import get_adj_from_epyt, get_normalized_adj, generate_cheb_poly

INP_PATH = '/data/zsm/case01/data/d-town.inp'
MODEL_PATH = '/data/zsm/case01/models/StgcnDtown.pth'
HRS = 168
LOOKBACK = 12
HORIZON = 3
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    def __init__(self):
        self.n_his = LOOKBACK
        self.n_pred = HORIZON
        self.Kt = 3
        self.Ks = 3
        self.act_func = 'glu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.0
        self.gso_type = 'cheb'


def evaluate_and_plot():
    print(f"Using device: {DEVICE}")
    args = Config()

    # 1. 初始化数据环境
    print("1. Loading Data...")
    sim = EpytHelper(INP_PATH, hrs=HRS)
    raw_data = sim.get_raw_data()
    num_nodes = raw_data['pressures'].shape[1]
    print(f"Total Nodes: {num_nodes}")

    # 2. 准备 Scaler
    print("Preparing Scaler (from Train set)...")
    train_ds = STGCNDataset(sim, LOOKBACK, HORIZON, 'train')
    scalers = train_ds.get_scalers()
    p_scaler = scalers['pressure']

    # 3. 准备测试集
    print("Preparing Test Set...")
    test_ds = STGCNDataset(sim, LOOKBACK, HORIZON, 'test', scaler=scalers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 4. 重建图结构
    print("Rebuilding Graph Structure...")
    adj = get_adj_from_epyt(sim)
    norm_adj = get_normalized_adj(adj)
    if args.graph_conv_type == 'cheb_graph_conv':
        cheb_polys = generate_cheb_poly(norm_adj, 2)
        gso_numpy = cheb_polys[1]
        gso = torch.from_numpy(gso_numpy).float().to(DEVICE)
    else:
        gso = torch.from_numpy(norm_adj).float().to(DEVICE)
    setattr(args, 'gso', gso)

    # 5. 初始化模型 & 加载权重
    blocks = [
        [2],
        [64, 32, 64],
        [64, 32, 64],
        [128, 64],
        [HORIZON]
    ]

    print("Initializing Model...")
    model = STGCNChebGraphConv(args, blocks, num_nodes).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model.eval()

    # 6. 推理循环
    print("Running Inference...")
    all_preds_norm = []
    all_targets_norm = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)  # [B, 2, T, N]
            y = y.to(DEVICE)  # [B, H, N]

            output = model(x)  # [B, H, 1, N]
            if output.dim() == 4:
                pred = output.squeeze(2)  # [B, H, N]
            else:
                pred = output

            all_preds_norm.append(pred.cpu().numpy())
            all_targets_norm.append(y.cpu().numpy())

    # 拼接: [Total_Samples, Horizon, Nodes]
    all_preds_norm = np.concatenate(all_preds_norm, axis=0)
    all_targets_norm = np.concatenate(all_targets_norm, axis=0)

    # 7. 多步反归一化与 MAE 计算
    # -----------------------------------------------------------
    print("Evaluating Multi-step Metrics (Real Scale)...")

    total_samples = all_preds_norm.shape[0]
    horizon_steps = all_preds_norm.shape[1]

    # 存储每一步的指标
    step_maes = []
    step_rmses = []

    # 用于绘图的数据 (取所有样本)
    # Shape: [Total_Samples, Horizon, Nodes]
    all_preds_real = np.zeros_like(all_preds_norm)
    all_targets_real = np.zeros_like(all_targets_norm)

    # 逐步处理 (Step-wise Processing)
    for h in range(horizon_steps):
        # 取出第 h 步的所有预测: [Total_Samples, Nodes]
        pred_step_norm = all_preds_norm[:, h, :]
        target_step_norm = all_targets_norm[:, h, :]

        # 反归一化 (Scaler 期望输入 [N_samples, N_nodes])
        pred_step_real = p_scaler.inverse_transform(pred_step_norm)
        target_step_real = p_scaler.inverse_transform(target_step_norm)

        # 存入大数组以便绘图或其他用途
        all_preds_real[:, h, :] = pred_step_real
        all_targets_real[:, h, :] = target_step_real

        # 计算该步的指标
        mae = mean_absolute_error(target_step_real, pred_step_real)
        rmse = np.sqrt(mean_squared_error(target_step_real, pred_step_real))

        step_maes.append(mae)
        step_rmses.append(rmse)

        print(f"Step {h + 1} ({(h + 1) * 5} min) -> MAE: {mae:.4f} m | RMSE: {rmse:.4f} m")

    # 计算整体平均指标
    overall_mae = np.mean(step_maes)
    overall_rmse = np.mean(step_rmses)

    print("=" * 40)
    print(f"Overall Performance (Avg over {horizon_steps} steps):")
    print(f"MAE : {overall_mae:.4f} m")
    print(f"RMSE: {overall_rmse:.4f} m")
    print("=" * 40)

    # 8. 可视化绘图 (默认画第1步，也可改为其他)
    # -----------------------------------------------------------
    # 为了绘图函数通用，我们传第0步 (t+1) 的数据进去
    plot_results(all_preds_real[:, 0, :], all_targets_real[:, 0, :], num_nodes)


def plot_results(preds, targets, num_nodes):
    """
    preds, targets: shape [Time, Nodes] (已经反归一化)
    """
    np.random.seed(42)
    # 随机选点或指定点
    plot_nodes = np.random.choice(range(num_nodes), size=3, replace=False)
    # plot_nodes = [10, 50, 100]

    total_time = preds.shape[0]
    plot_len = min(288, total_time)  # 画一天

    plt.figure(figsize=(15, 10))

    for i, node_id in enumerate(plot_nodes):
        plt.subplot(3, 1, i + 1)
        plt.plot(targets[:plot_len, node_id], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
        plt.plot(preds[:plot_len, node_id], label='Prediction (t+1)', color='crimson', linestyle='--', linewidth=1.5)
        plt.title(f"Node {node_id} Pressure Prediction")
        plt.ylabel("Pressure (m)")
        if i == 2:
            plt.xlabel("Time Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = './figures/StgcnDtown_Eval.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()