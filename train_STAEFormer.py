import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import time
from datetime import datetime

from datasets.epanet_data import EpytHelper
from model.STAEformer import STAEformer
from datasets.stae_dataset import STAEformerDataset


def get_args():
    parser = argparse.ArgumentParser(description="STAEformer Training Script for WDN")

    # --- 数据相关参数 ---
    parser.add_argument('--inp_path', type=str, default='/data/zsm/case01/data/d-town.inp',
                        help='Path to EPANET input file')
    parser.add_argument('--hrs', type=int, default=168, help='Simulation hours')
    parser.add_argument('--lookback', type=int, default=12, help='Input sequence length (history)')
    parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon (future steps)')
    parser.add_argument('--steps_per_day', type=int, default=96,
                        help='Time steps per day (e.g., 288 for 5min intervals)')

    # --- 训练超参数 ---
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--factor', type=float, default=0.5, help='Factor for learning rate decay')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Gradient clipping value')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')

    # --- 模型架构参数 ---
    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension (physical)')
    parser.add_argument('--output_dim', type=int, default=1, help='Output feature dimension')
    parser.add_argument('--input_emb_dim', type=int, default=24, help='Input embedding dimension')
    parser.add_argument('--tod_emb_dim', type=int, default=24, help='Time of Day embedding dimension')
    parser.add_argument('--dow_emb_dim', type=int, default=0, help='Day of Week embedding dimension')
    parser.add_argument('--spatial_emb_dim', type=int, default=24, help='Spatial (Node) embedding dimension')
    parser.add_argument('--adaptive_emb_dim', type=int, default=80, help='Adaptive embedding dimension')
    parser.add_argument('--ff_dim', type=int, default=256, help='Feed forward hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use_mixed_proj', action='store_true', default=True,
                        help='Use mixed projection in output layer')

    # --- 路径与日志 ---
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save model checkpoints')
    parser.add_argument('--figure_dir', type=str, default='./figures', help='Directory to save figures')
    parser.add_argument('--model_name', type=str, default='STAEFormer', help='Base name for saved model')

    return parser.parse_args()


def main():
    # 1. 获取参数
    args = get_args()

    print(f"========== Training Configuration ==========")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("============================================")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        sim = EpytHelper(args.inp_path, hrs=args.hrs)
        raw_data = sim.get_raw_data()
        # 假设 pressures 形状为 [Time, Nodes]
        num_nodes = raw_data['pressures'].shape[1]
        print(f"Simulation loaded. Total Nodes detected: {num_nodes}")
    except Exception as e:
        print(f"Error: Failed to load simulation: {e}")
        return

    print("Building Datasets...")
    # Train
    train_ds = STAEformerDataset(sim, args.lookback, args.horizon, 'train')
    scaler = train_ds.scaler  # 获取 Scaler 用于验证集
    # Val
    val_ds = STAEformerDataset(sim, args.lookback, args.horizon, 'val', scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # 检查数据形状
    x_sample, y_sample = next(iter(train_loader))
    print(f"Data Loaded successfully.")
    print(f"Sample Input Shape: {x_sample.shape} (Batch, Lookback, Nodes, Feats)")
    print(f"Sample Target Shape: {y_sample.shape} (Batch, Horizon, Nodes)")

    # 3. 初始化模型
    model = STAEformer(
        num_nodes=num_nodes,
        in_steps=args.lookback,
        out_steps=args.horizon,
        steps_per_day=args.steps_per_day,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        input_embedding_dim=args.input_emb_dim,
        tod_embedding_dim=args.tod_emb_dim,
        dow_embedding_dim=args.dow_emb_dim,
        spatial_embedding_dim=args.spatial_emb_dim,
        adaptive_embedding_dim=args.adaptive_emb_dim,
        feed_forward_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_mixed_proj=args.use_mixed_proj
    ).to(device)

    # 4. 优化器与损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.patience, factor=args.factor
    )

    # 5. 训练循环
    history = {'train_loss': [], 'val_loss': []}

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Start Training...")
    start_time = time.time()

    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss_accum = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device)  # [B, T, N, 3]
                y = y.to(device)  # [B, T, N]


                optimizer.zero_grad()

                # Forward
                pred = model(x)  # [B, T_out, N, 1]
                pred = pred.squeeze(-1)  # [B, T_out, N]

                loss = criterion(pred, y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

                optimizer.step()

                train_loss_accum += loss.item()

            avg_train_loss = train_loss_accum / len(train_loader)

            # Validation
            model.eval()
            val_loss_accum = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x).squeeze(-1)
                    loss = criterion(pred, y)
                    val_loss_accum += loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)

            # 学习率调整
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            # 使用 flush=True 确保输出立刻刷新，不被缓存
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"Train MSE: {avg_train_loss:.6f} | "
                f"Val MSE: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.6f}",
                flush=True
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time / 60:.2f} minutes.")

    # 6. 保存模型与可视化
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(args.save_dir, f'{args.model_name}_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    # 简单绘图
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Training Loss (Horizon={args.horizon})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    # 保存训练曲线图到 logs 目录
    plot_path = os.path.join(args.figure_dir, f'loss_curve.png')
    plt.savefig(plot_path)
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()