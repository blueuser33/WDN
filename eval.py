import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datasets.epanet_data import EpytHelper
from model.STAEformer import STAEformer
from datasets.stae_dataset import STAEformerDataset

# ================= 配置参数 =================
INP_PATH = '/data/zsm/case01/data/d-town.inp'
MODEL_PATH = './models/STAEFormer_20251206_210559.pth'  # 训练保存的模型路径
HRS = 168
LOOKBACK = 12
HORIZON = 3
BATCH_SIZE = 32  # 测试时显存压力小，可以大一点
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_and_plot():
    print(f"Using device: {DEVICE}")

    # 1. 初始化数据环境
    # -----------------------------------------------------------
    print("Loading Data...")
    sim = EpytHelper(INP_PATH, hrs=HRS)
    raw_data = sim.get_raw_data()
    num_nodes = raw_data['pressures'].shape[1]
    print(f"Total Nodes: {num_nodes}")

    # 关键：我们需要先加载训练集以获取 fit 好的 Scaler
    print("Preparing Scaler (from Train set)...")
    train_ds = STAEformerDataset(sim, LOOKBACK, HORIZON, 'train')
    scaler = train_ds.scaler

    # 加载测试集 (传入训练好的 scaler)
    print("Preparing Test Set...")
    test_ds = STAEformerDataset(sim, LOOKBACK, HORIZON, 'test', scaler=scaler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 2. 加载模型
    # -----------------------------------------------------------
    print("Loading Model...")
    model = STAEformer(
        num_nodes=num_nodes,
        in_steps=LOOKBACK,
        out_steps=HORIZON,
        steps_per_day=96,
        input_dim=1,  # 必须与训练一致
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=0,
        spatial_embedding_dim=24,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.0,  # 测试模式不需要 Dropout
        use_mixed_proj=True
    ).to(DEVICE)

    # 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model.eval()

    # 3. 推理循环
    # -----------------------------------------------------------
    preds_list = []
    targets_list = []

    print("Running Inference...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward
            # pred shape: [Batch, Horizon, Nodes, 1]
            pred = model(x)

            # 我们这里只提取 "未来第1个时刻" (t+1) 进行连续画图对比
            # 如果你想看长时预测能力，可以取 pred[:, -1, :, :] (未来第12个时刻)

            # 取出 Horizon 中的第 0 步 (即预测的下一步)
            # shape: [Batch, Nodes]
            pred_t1 = pred[:, 0, :, 0]
            target_t1 = y[:, 0, :]

            preds_list.append(pred_t1.cpu().numpy())
            targets_list.append(target_t1.cpu().numpy())

    # 拼接所有 Batch
    # Shape: [Total_Test_Samples, Nodes]
    all_preds_norm = np.concatenate(preds_list, axis=0)
    all_targets_norm = np.concatenate(targets_list, axis=0)

    mse_norm = mean_squared_error(all_targets_norm, all_preds_norm)
    print("\n" + "="*40)
    print(f"Normalized MSE (Model Loss Level): {mse_norm:.6f}")
    print("="*40)

    # 4. 反归一化 (还原为真实物理单位: 米)
    # -----------------------------------------------------------
    print("Inverse Transforming...")
    # Scaler 期望输入 [Samples, Nodes]
    all_preds_real = scaler.inverse_transform(all_preds_norm)
    all_targets_real = scaler.inverse_transform(all_targets_norm)
    mse_real = mean_squared_error(all_targets_real, all_preds_real)
    rmse_real = np.sqrt(mse_real)
    mae_real = mean_absolute_error(all_targets_real, all_preds_real)
    print("\n" + "=" * 40)
    print("Real-World Physical Metrics (Unit: Meter)")
    print("-" * 40)
    print(f"MSE  : {mse_real:.6f} m^2")
    print(f"RMSE : {rmse_real:.6f} m")
    print(f"MAE  : {mae_real:.6f} m")
    print("=" * 40 + "\n")
    # 5. 可视化绘图
    # -----------------------------------------------------------
    plot_results(all_preds_real, all_targets_real, num_nodes)


def plot_results(preds, targets, num_nodes):
    """
    preds, targets: shape [Time, Nodes]
    """
    # 随机选择 3 个节点，或者你可以指定特定节点 ID
    np.random.seed(1)
    plot_nodes = np.random.choice(range(num_nodes), size=3, replace=False)

    total_time = preds.shape[0]
    plot_len = total_time

    plt.figure(figsize=(15, 10))

    for i, node_id in enumerate(plot_nodes):
        plt.subplot(3, 1, i + 1)

        # 真实值
        plt.plot(targets[:plot_len, node_id], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
        # 预测值
        plt.plot(preds[:plot_len, node_id], label='Prediction (t+1)', color='red', linestyle='--', linewidth=1.5)

        plt.title(f"Node {node_id} Pressure Prediction")
        plt.ylabel("Pressure (m)")
        if i == 2:
            plt.xlabel("Time Steps (5 min intervals)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./figures/prediction_result.png')  # 保存图片
    print("Plot saved to 'prediction_result.png'")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()