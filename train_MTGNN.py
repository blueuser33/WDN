import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets.WaterDataset import EpanetUnifiedPreprocessor, MTGNNDataset  # 修改这里
from model.net import gtnet
from trainer import ModelTrainer

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

parser = argparse.ArgumentParser(description='MTGNN 模型训练参数')
parser.add_argument('--inp_file_path', type=str, default='/data/zsm/case01/data/d-town.inp', help='EPANET输入文件路径')
parser.add_argument('--save_dir', type=str, default='model_saved/MTGNN', help='模型保存目录')
parser.add_argument('--n_his', type=int, default=12, help='历史时间步长')
parser.add_argument('--n_pred', type=int, default=3, help='预测时间步长')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
parser.add_argument('--dropout', type=float, default=0.3, help='丢弃率')
parser.add_argument('--gcn_depth', type=int, default=2, help='图卷积深度')
parser.add_argument('--conv_channels', type=int, default=16, help='卷积通道数')
parser.add_argument('--residual_channels', type=int, default=16, help='残差通道数')
parser.add_argument('--skip_channels', type=int, default=32, help='跳跃连接通道数')
parser.add_argument('--end_channels', type=int, default=64, help='末端通道数')
parser.add_argument('--layers', type=int, default=5, help='层数')
parser.add_argument('--subgraph_size', type=int, default=20, help='子图大小')
parser.add_argument('--node_dim', type=int, default=40, help='节点维度')
parser.add_argument('--dilation_exponential', type=int, default=2, help='膨胀指数')
parser.add_argument('--gcn_true', type=bool, default=True, help='是否添加图卷积层')
parser.add_argument('--buildA_true', type=bool, default=True, help='是否构建自适应邻接矩阵')
parser.add_argument('--propalpha', type=float, default=0.05, help='传播参数alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh参数alpha')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

args = parser.parse_args()

# 创建模型保存目录
os.makedirs(args.save_dir, exist_ok=True)


class MTGNNModelTrainer(ModelTrainer):
    """MTGNN专用训练器，继承自通用训练器"""

    def train_one_epoch(self, train_loader):
        """重写训练方法以适应MTGNN的输入输出格式"""
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()

            # 前向传播 - MTGNN的输出格式是 [batch, out_dim, num_nodes, seq_out_len]
            output = self.model(x).permute(0, 3, 2, 1).contiguous()

            # 计算损失 - 不需要维度调整，因为MTGNNDataset已经处理好了格式
            loss = self.criterion(output, y)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        """重写评估方法以适应MTGNN的输入输出格式"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device).float(), y.to(self.device).float()
                output = self.model(x).permute(0, 3, 2, 1).contiguous()
                loss = self.criterion(output, y)
                total_loss += loss.item()

        return total_loss / len(val_loader)


def main():
    """主函数"""
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载和预处理数据
    print("加载和预处理数据...")
    preprocessor = EpanetUnifiedPreprocessor(
        inp_file_path=args.inp_file_path,
        hrs=168  # 使用默认值168小时
    )

    # 扩展EpanetUnifiedPreprocessor以支持MTGNN数据集
    # 方法1: 直接创建数据集和数据加载器
    pressure_data = preprocessor.raw_data['pressures_norm']
    node_count = preprocessor.node_count

    # 创建MTGNN数据集
    dataset = MTGNNDataset(pressure_data, args.n_his, args.n_pred)

    # 分割数据集
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    print("创建MTGNN模型...")
    model = gtnet(
        gcn_true=args.gcn_true,
        buildA_true=args.buildA_true,
        gcn_depth=args.gcn_depth,
        num_nodes=node_count,
        device=device,
        dropout=args.dropout,
        subgraph_size=args.subgraph_size,
        node_dim=args.node_dim,
        dilation_exponential=args.dilation_exponential,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        seq_length=args.n_his,
        in_dim=1,
        out_dim=args.n_pred,
        layers=args.layers,
        propalpha=args.propalpha,
        tanhalpha=args.tanhalpha,
        layer_norm_affline=False
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建MTGNN专用训练器
    trainer = MTGNNModelTrainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"感受野大小: {model.receptive_field}")

    # 训练模型
    print("开始训练...")
    model_save_path = os.path.join(args.save_dir, 'best_model.pth')

    # 使用ModelTrainer的train方法进行训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        model_save_path=model_save_path
    )

    # 在测试集上评估
    test_loss = trainer.test(
        test_loader=test_loader,
        model_path=model_save_path
    )

    print(f"训练完成！模型已保存到 {args.save_dir}")


if __name__ == '__main__':
    main()