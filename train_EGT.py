import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from datasets.epanet_data import EpytHelper
from datasets.GraphDataset import WDNDataset
from model.EGT import WaterPredictor
from physics_loss import PhysicsLoss


def get_args():
    parser = argparse.ArgumentParser(description="EGT Training with Physics Constraints")

    # --- 数据路径 ---
    parser.add_argument('--data_path', type=str, default='/data/zsm/case01/data/d-town.inp', help='Path to .inp file')
    parser.add_argument('--save_dir', type=str, default='/data/zsm/case01/models/', help='Directory to save models')

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    # --- 模型架构参数 ---
    parser.add_argument('--lookback', type=int, default=12, help='Input history length')
    parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon steps')
    parser.add_argument('--d_model', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of EGT layers')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    # --- 物理约束参数 ---
    parser.add_argument('--lambda_mass', type=float, default=1000, help='Weight for Mass Conservation Loss')
    parser.add_argument('--lambda_head', type=float, default=0.005, help='Weight for Head Loss')

    # --- 设备 ---
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def train_epoch(model, loader, optimizer, criterion, device, args, physics_criterion=None):
    model.train()
    total_loss = 0
    total_loss_p = 0
    total_loss_f = 0
    total_phy_mass = 0
    total_phy_head = 0

    loop = tqdm(loader, desc="Train", leave=False)

    for batch in loop:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward
        # Output Shape: [Batch*N, Horizon] / [Batch*E, Horizon]
        pred_p, pred_f = model(batch)

        # 1. 数据驱动损失 (MSE) - 计算所有步的平均 MSE
        loss_p = criterion(pred_p, batch.y)
        loss_f = criterion(pred_f, batch.y_flow)

        # 基础损失 (压力 + 0.5 * 流量)
        loss_data = loss_p + 0.5 * loss_f

        # 2. 物理约束损失 (Physics Loss) - 【改进：对 Horizon 的每一步都进行约束】
        loss_mass_accum = torch.tensor(0.0, device=device)
        loss_head_accum = torch.tensor(0.0, device=device)

        if physics_criterion is not None:
            curr_batch_size = batch.num_graphs
            num_nodes = batch.num_nodes // curr_batch_size
            num_edges = batch.edge_attr.shape[0] // curr_batch_size

            # 循环计算未来每一步的物理损失
            for h in range(args.horizon):
                # 提取第 h 步的数据并 reshape 为 [Batch, N]
                p_step_h = pred_p[:, h].view(curr_batch_size, num_nodes)
                f_step_h = pred_f[:, h].view(curr_batch_size, num_edges)

                # 确保 Dataset 提供了 y_demand (在 WDNDataset.get 中需要返回 y_demand)
                # batch.y_demand shape: [Batch*N, Horizon]
                if hasattr(batch, 'y_demand'):
                    d_step_h = batch.y_demand[:, h].view(curr_batch_size, num_nodes)
                    l_m, l_h = physics_criterion(p_step_h, f_step_h, d_step_h)

                    loss_mass_accum += l_m
                    loss_head_accum += l_h

            # 取平均 (Loss per step)
            loss_mass_accum /= args.horizon
            loss_head_accum /= args.horizon

        # 总损失
        loss = loss_data + args.lambda_mass * loss_mass_accum + args.lambda_head * loss_head_accum

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_p += loss_p.item()
        total_loss_f += loss_f.item()
        total_phy_mass += loss_mass_accum.item()
        total_phy_head += loss_head_accum.item()

        loop.set_postfix(L=loss.item(), P=loss_p.item(), M=loss_mass_accum.item())

    avg_len = len(loader)
    return (total_loss / avg_len,
            total_loss_p / avg_len,
            total_loss_f / avg_len)


def validate(model, loader, criterion, device, scaler_dict, num_nodes, num_edges, args):
    """
    公平对比验证函数：
    1. 计算每一预测步 (Step 1, 2, 3...) 的真实物理 MAE。
    2. 计算所有步的平均 MAE。
    """
    model.eval()

    # 损失累加器
    total_loss = 0
    total_loss_p = 0
    total_loss_f = 0

    # 物理误差累加器 (Step-wise)
    # step_maes_p[0] 代表 step 1, step_maes_p[1] 代表 step 2...
    step_maes_p = np.zeros(args.horizon)
    step_maes_f = np.zeros(args.horizon)

    scaler_p = scaler_dict['pressure']
    scaler_f = scaler_dict['flow']

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_p, pred_f = model(batch)

            # 1. 计算 Normalized Loss
            loss_p = criterion(pred_p, batch.y)
            loss_f = criterion(pred_f, batch.y_flow)
            loss = loss_p + 0.5 * loss_f

            total_loss += loss.item()
            total_loss_p += loss_p.item()
            total_loss_f += loss_f.item()

            # 2. 反归一化并计算分步真实误差
            curr_batch_size = batch.num_graphs

            # --- Reshape Logic ---
            # Input: [B*N, H] -> Reshape to [B, N, H] -> Permute to [B, H, N]
            # 这样第1维就是时间步，方便循环
            pred_p_3d = pred_p.view(curr_batch_size, num_nodes, args.horizon).permute(0, 2, 1).cpu().numpy()
            true_p_3d = batch.y.view(curr_batch_size, num_nodes, args.horizon).permute(0, 2, 1).cpu().numpy()

            pred_f_3d = pred_f.view(curr_batch_size, num_edges, args.horizon).permute(0, 2, 1).cpu().numpy()
            true_f_3d = batch.y_flow.view(curr_batch_size, num_edges, args.horizon).permute(0, 2, 1).cpu().numpy()

            # --- Step-wise Inverse Transform Loop ---
            for h in range(args.horizon):
                # 取出第 h 步: [B, N]
                pred_p_step = pred_p_3d[:, h, :]
                true_p_step = true_p_3d[:, h, :]

                # 反归一化 (Scaler 期望输入 [Samples, Nodes])
                pred_p_real = scaler_p.inverse_transform(pred_p_step)
                true_p_real = scaler_p.inverse_transform(true_p_step)

                # 计算该步 MAE 并累加
                step_maes_p[h] += np.mean(np.abs(pred_p_real - true_p_real))

                # 流量同理
                pred_f_step = pred_f_3d[:, h, :]
                true_f_step = true_f_3d[:, h, :]
                pred_f_real = scaler_f.inverse_transform(pred_f_step)
                true_f_real = scaler_f.inverse_transform(true_f_step)
                step_maes_f[h] += np.mean(np.abs(pred_f_real - true_f_real))

    # 计算平均值
    avg_len = len(loader)

    avg_loss = total_loss / avg_len
    avg_loss_p = total_loss_p / avg_len
    avg_loss_f = total_loss_f / avg_len

    # 得到每个时间步的平均 MAE
    avg_step_maes_p = step_maes_p / avg_len
    avg_step_maes_f = step_maes_f / avg_len

    # 计算所有步的整体平均 MAE
    overall_mae_p = np.mean(avg_step_maes_p)
    overall_mae_f = np.mean(avg_step_maes_f)

    return (avg_loss, avg_loss_p, avg_loss_f,
            overall_mae_p, overall_mae_f,
            avg_step_maes_p, avg_step_maes_f)


def main():
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"========== Configuration ==========")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===================================")

    print(f"Loading data from {args.data_path}...")
    sim = EpytHelper(args.data_path)
    raw_data = sim.get_raw_data()

    # 获取拓扑信息
    num_nodes = len(raw_data['node_elevations'])
    num_edges = len(raw_data['link_lengths'])
    print(f"Graph Info: Nodes={num_nodes}, Edges={num_edges}")

    edge_static_raw = np.stack([
        raw_data['link_lengths'],
        raw_data['link_diameters'],
        raw_data['link_roughnesses']
    ], axis=1)  # [E, 3]
    edge_static_tensor = torch.tensor(edge_static_raw, dtype=torch.float32)

    node_elev_raw = torch.tensor(raw_data['node_elevations'], dtype=torch.float32)
    edge_index = torch.tensor(raw_data['edge_index_directed'], dtype=torch.long)

    # 1. 构建数据集
    train_dataset = WDNDataset(raw_data, lookback=args.lookback, horizon=args.horizon, mode='train')
    val_dataset = WDNDataset(raw_data, lookback=args.lookback, horizon=args.horizon, mode='val',
                             scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    physics_loss_fn = PhysicsLoss(
        scaler_dict=train_dataset.scaler,
        edge_index=edge_index,
        node_elevation=node_elev_raw,
        edge_static_attr=edge_static_tensor,
        device=args.device,
    )

    # 2. 初始化模型
    model = WaterPredictor(
        lookback=args.lookback,
        horizon=args.horizon,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    # 3. 训练循环
    print("Start Training...")
    best_val_mae = float('inf')  # 改用 MAE 作为保存标准

    for epoch in range(args.epochs):
        # Training
        t_loss, t_loss_p, t_loss_f = train_epoch(
            model, train_loader, optimizer, criterion,
            args.device, args, physics_criterion=physics_loss_fn
        )

        # Validation
        (v_loss, v_loss_p, v_loss_f,
         v_mae_p, v_mae_f,
         v_step_maes_p, v_step_maes_f) = validate(
            model, val_loader, criterion, args.device,
            train_dataset.scaler, num_nodes, num_edges, args
        )

        # 格式化分步误差字符串
        steps_str_p = ", ".join([f"{val:.3f}" for val in v_step_maes_p])
        scheduler.step(v_mae_p)
        # 打印详细信息
        print(f"Epoch {epoch + 1:03d} | "
              f"T_Loss: {t_loss:.4f} | "
              f"V_Loss_P: {v_loss_p:.4f} | "
              f"MAE_P (Avg): {v_mae_p:.4f} m | "
              f"MAE_P (Steps): [{steps_str_p}]")
        if v_mae_p < best_val_mae:
            best_val_mae = v_mae_p
            save_name = f'EGT_Dtown_H{args.horizon}.pth'
            torch.save(model.state_dict(), os.path.join(args.save_dir, save_name))
            print(f"--> Best Model Saved (MAE: {best_val_mae:.4f})")

    print("Training Done.")


if __name__ == '__main__':
    main()