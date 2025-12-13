import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 引入项目模块 ---
# 请确保路径正确，如果不正确请修改 sys.path 或 import 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.STGCN import STGCNChebGraphConv
from datasets.epanet_data import EpytHelper
from datasets.stgcn_dataset import STGCNDataset
from utils.graph_utils import get_adj_from_epyt, get_normalized_adj, generate_cheb_poly

# ================= 配置参数 (需与训练时一致) =================
INP_PATH = '/data/zsm/case01/data/d-town.inp'
MODEL_PATH = '/data/zsm/case01/models/StgcnDtown.pth'  # 修改为你保存的模型路径
HRS = 168
LOOKBACK = 12
HORIZON = 3
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 这里的参数必须和 train_STGCN.py 里的 args 一模一样
class Config:
    def __init__(self):
        self.n_his = LOOKBACK
        self.n_pred = HORIZON
        self.Kt = 3
        self.Ks = 3
        self.act_func = 'glu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.0  # 测试时 Dropout 设为 0
        self.gso_type = 'cheb'  # 假设训练时用的是 cheb


def evaluate_and_plot():
    print(f"Using device: {DEVICE}")
    args = Config()

    # 1. 初始化数据环境
    # -----------------------------------------------------------
    print("1. Loading Data...")
    sim = EpytHelper(INP_PATH, hrs=HRS)
    raw_data = sim.get_raw_data()
    num_nodes = raw_data['pressures'].shape[1]
    print(f"Total Nodes: {num_nodes}")

    # 2. 准备 Scaler (从训练集获取)
    # -----------------------------------------------------------
    print("Preparing Scaler (from Train set)...")
    # 创建训练集对象只为了 fit scaler
    train_ds = STGCNDataset(sim, LOOKBACK, HORIZON, 'train')
    # 获取 scalers 字典 {'pressure': p_scaler, 'demand': d_scaler}
    scalers = train_ds.get_scalers()
    p_scaler = scalers['pressure']  # 我们只关心压力的反归一化

    # 3. 准备测试集
    # -----------------------------------------------------------
    print("Preparing Test Set...")
    test_ds = STGCNDataset(sim, LOOKBACK, HORIZON, 'test', scaler=scalers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 4. 重建图结构 (GSO) - 必须与训练完全一致
    # -----------------------------------------------------------
    print("Rebuilding Graph Structure...")
    adj = get_adj_from_epyt(sim)
    norm_adj = get_normalized_adj(adj)

    if args.graph_conv_type == 'cheb_graph_conv':
        # 这里假设使用 train_STGCN.py 最后的修正版 (直接取切比雪夫多项式的第1项作为 L_tilde)
        cheb_polys = generate_cheb_poly(norm_adj, 2)
        gso_numpy = cheb_polys[1]  # 取 L_tilde, 形状 [N, N]
        gso = torch.from_numpy(gso_numpy).float().to(DEVICE)
    else:
        gso = torch.from_numpy(norm_adj).float().to(DEVICE)

    setattr(args, 'gso', gso)

    # 5. 初始化模型 & 加载权重
    # -----------------------------------------------------------
    # 重构 Blocks (必须与训练代码中的 blocks 一致)
    # 结构: [Input], [In, Mid, Out], [In, Mid, Out], [FC_In, FC_Out], [Horizon]
    blocks = [
        [2],  # Input: Pressure + Demand
        [64, 32, 64],  # ST-Conv1
        [64, 32, 64],  # ST-Conv2
        [128, 64],  # OutputBlock FC Hidden
        [HORIZON]  # Output Horizon
    ]

    print("Initializing Model...")
    model = STGCNChebGraphConv(args, blocks, num_nodes).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        # map_location 确保在 CPU 机器上也能加载 GPU 模型
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model.eval()

    # 6. 推理循环
    # -----------------------------------------------------------
    preds_list = []
    targets_list = []

    print("Running Inference...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)  # [B, 2, T, N]
            y = y.to(DEVICE)  # [B, Horizon, N]

            # Forward
            # Output Shape: [Batch, Horizon, 1, Nodes]
            output = model(x)

            # 挤压维度: [B, H, 1, N] -> [B, H, N]
            if output.dim() == 4:
                pred = output.squeeze(2)
            else:
                pred = output

            # 我们取 Horizon 的第 0 步 (t+1) 进行评估
            pred_t1 = pred[:, 0, :]  # [B, N]
            target_t1 = y[:, 0, :]  # [B, N]

            preds_list.append(pred_t1.cpu().numpy())
            targets_list.append(target_t1.cpu().numpy())

    # 拼接所有 Batch
    # Shape: [Total_Test_Samples, Nodes]
    all_preds_norm = np.concatenate(preds_list, axis=0)
    all_targets_norm = np.concatenate(targets_list, axis=0)

    # 计算归一化后的指标
    mse_norm = mean_squared_error(all_targets_norm, all_preds_norm)
    print(f"Normalized MSE: {mse_norm:.6f}")

    # 7. 反归一化 (还原为真实物理单位: 米)
    # -----------------------------------------------------------
    print("Inverse Transforming...")
    # Scaler 期望输入 [Samples, Nodes]
    all_preds_real = p_scaler.inverse_transform(all_preds_norm)
    all_targets_real = p_scaler.inverse_transform(all_targets_norm)

    # 计算真实指标
    mae_real = mean_absolute_error(all_targets_real, all_preds_real)
    rmse_real = np.sqrt(mean_squared_error(all_targets_real, all_preds_real))

    print("-" * 30)
    print(f"Test Results (Horizon=1 / 15 mins):")
    print(f"MAE : {mae_real:.4f} m")
    print(f"RMSE: {rmse_real:.4f} m")
    print("-" * 30)

    # 8. 可视化绘图
    # -----------------------------------------------------------
    plot_results(all_preds_real, all_targets_real, num_nodes)


def plot_results(preds, targets, num_nodes):
    """
    preds, targets: shape [Time, Nodes]
    """
    # 随机选择 3 个节点
    np.random.seed(2)
    plot_nodes = np.random.choice(range(num_nodes), size=3, replace=False)

    # 或者手动指定，比如查看压力波动大的节点
    # plot_nodes = [0, 10, 20]

    total_time = preds.shape[0]
    # 只画前 288 个点 (约 24 小时)，画太多看不清细节
    plot_len = min(288, total_time)

    plt.figure(figsize=(15, 10))

    for i, node_id in enumerate(plot_nodes):
        plt.subplot(3, 1, i + 1)

        # 真实值
        plt.plot(targets[:plot_len, node_id], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
        # 预测值
        plt.plot(preds[:plot_len, node_id], label='Prediction (t+1)', color='blue', linestyle='--', linewidth=1.5)

        plt.title(f"Node {node_id} Pressure Prediction")
        plt.ylabel("Pressure (m)")
        if i == 2:
            plt.xlabel("Time Steps (5 min intervals)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = './stgcn_prediction_result.png'
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()