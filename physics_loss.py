import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class PhysicsLoss(nn.Module):
    def __init__(self,
                 scaler_dict,
                 edge_index,
                 node_elevation,
                 edge_static_attr,  # [Length, Diameter, Roughness]
                 device):
        super().__init__()
        self.device = device
        self.edge_index = edge_index.to(device)
        self.node_elevation = node_elevation.to(device)

        # 1. 预计算反归一化参数 (保存为Tensor以保留梯度)
        # Pressure (Node)
        self.p_mean = torch.tensor(scaler_dict['pressure'].mean_, dtype=torch.float32, device=device)
        self.p_scale = torch.tensor(scaler_dict['pressure'].scale_, dtype=torch.float32, device=device)

        # Flow (Edge)
        self.f_mean = torch.tensor(scaler_dict['flow'].mean_, dtype=torch.float32, device=device)
        self.f_scale = torch.tensor(scaler_dict['flow'].scale_, dtype=torch.float32, device=device)

        # Demand (Node) - 用于质量守恒
        self.d_mean = torch.tensor(scaler_dict['demand'].mean_, dtype=torch.float32, device=device)
        self.d_scale = torch.tensor(scaler_dict['demand'].scale_, dtype=torch.float32, device=device)

        # 2. 计算哈森-威廉斯系数 R (Resistance)
        # 公式: h_f = 10.67 * L * Q^1.852 * C^-1.852 * D^-4.87 (SI单位: Q在m3/s, D在m)
        # 注意：这里需要根据你的数据单位调整常数。
        # 假设: Length(m), Diameter(m), Roughness(无量纲), Flow(m3/s)
        # 如果数据已经被归一化，我们需要先反归一化这些静态属性再计算R，或者直接传入原始值
        # 这里假设传入的 edge_static_attr 是原始物理值 (未归一化)

        L = edge_static_attr[:, 0]  # Length
        D = edge_static_attr[:, 1]  # Diameter
        C = edge_static_attr[:, 2]  # Roughness

        # 防止除零和数值不稳定，加一个极小值
        epsilon = 1e-6

        # 标准 Hazen-Williams R 计算 (SI单位)
        # 如果你的流量单位是 LPS (升/秒)，系数需要调整，但结构不变
        # R = 10.67 * L * C^(-1.852) * D^(-4.87)
        self.R = 10.67 * L * (C + epsilon).pow(-1.852) * (D + epsilon).pow(-4.87)
        self.R = self.R.to(device)

    def denormalize(self, tensor, mean, scale):
        """可微分的反归一化"""
        # tensor: [B, N/E]
        # mean/scale: [N/E] (因为StandardScaler是per-feature的)
        # 需要 reshape mean/scale 为 [1, N/E] 以进行广播
        return tensor * scale.unsqueeze(0) + mean.unsqueeze(0)

    def forward(self, pred_p_norm, pred_f_norm, true_d_norm):
        """
        pred_p_norm: 归一化的压力预测 [Batch, Nodes] (取单步)
        pred_f_norm: 归一化的流量预测 [Batch, Edges] (取单步)
        true_d_norm: 归一化的需水量真值 [Batch, Nodes]
        """

        # --- 1. 数据反归一化 (恢复物理单位) ---
        P = self.denormalize(pred_p_norm, self.p_mean, self.p_scale)  # [B, N]
        Q = self.denormalize(pred_f_norm, self.f_mean, self.f_scale)  # [B, E]
        d = self.denormalize(true_d_norm, self.d_mean, self.d_scale)  # [B, N]

        # 计算水头 H = Pressure + Elevation
        # Elevation 需要广播 [N] -> [1, N]
        H = P + self.node_elevation.unsqueeze(0)

        # --- 约束 1: 质量守恒 (Mass Conservation) ---
        # Sum(Q_in) - Sum(Q_out) = d
        src, dst = self.edge_index  # [E]

        # 对于 Batch 中的每个样本计算 (并行)
        # Q 的形状是 [B, E]
        # 我们需要对每个图单独聚合，但 edge_index 是静态的。
        # 技巧：直接利用 scatter_add 的 broadcasting 特性不行，需要循环或展开
        # 为了高效，我们处理维度的变换：

        # Q_out: 从节点流出 (source 是该节点)
        # 我们将 Q 按 edge_index[0] (source) 聚合到节点上
        # output shape: [B, N]
        q_out = torch.zeros_like(d)
        q_in = torch.zeros_like(d)

        # scatter_add 期望 input 和 index 维度匹配
        # 我们需要在维度 1 上进行 scatter
        # PyG 的 scatter 通常处理 [Total_Nodes]，这里我们手动处理 Batch

        # 方法：转置 Q 为 [E, B]，然后 scatter 到 [N, B]，再转置回 [B, N]
        Q_T = Q.t()  # [E, B]

        # 流出：聚合所有以 j 为起点的边的流量
        q_out_T = scatter(Q_T, src, dim=0, dim_size=d.shape[1], reduce='add') # [N, B]
        # 流入：聚合所有以 i 为终点的边的流量
        q_in_T = scatter(Q_T, dst, dim=0, dim_size=d.shape[1], reduce='add')  # [N, B]

        q_net = q_in_T.t() - q_out_T.t()  # [B, N] (In - Out)

        # 物理残差：|In - Out - demand|
        loss_mass = torch.mean((q_net - d) ** 2)

        # --- 约束 2: 哈森-威廉斯压降 (Head Loss) ---
        # H_i - H_j = R * Q * |Q|^0.852
        # 注意图片公式是 Q * |Q|^1.852，这实际上是 Q^2.852 (如果Q是带符号的)
        # 标准物理公式通常是 h_f = R * Q^1.852
        # 我们按照最通用的带有方向的写法: h_loss = R * Q * |Q|^(1.852 - 1) = R * Q * |Q|^0.852

        # 获取每条边起点和终点的水头
        # H: [B, N]
        h_src = H[:, src]  # [B, E]
        h_dst = H[:, dst]  # [B, E]

        # 实际压降 (模型预测的)
        delta_h_pred = h_src - h_dst  # [B, E]

        # 理论压降 (根据流量计算的)
        # Q: [B, E]
        # R: [E] -> [1, E]
        # 公式: R * Q * |Q|^0.852
        friction_slope = self.R.unsqueeze(0) * Q * torch.abs(Q).pow(0.852)

        loss_head = torch.mean((delta_h_pred - friction_slope) ** 2)

        return loss_mass, loss_head