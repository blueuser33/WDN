import torch
import torch.nn as nn
from torch_geometric.utils import softmax as sparse_softmax
from torch_geometric.nn import MessagePassing


# ==========================================
# 1. 基础组件 (保持不变)
# ==========================================

class TemporalEncoder(nn.Module):
    """处理时间序列：将 [N, T, F] 压缩为 [N, D]"""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq):
        # x_seq: [Nodes, Seq_Len, Input_Feats]
        _, h_n = self.gru(x_seq)
        return self.norm(h_n[-1])


class EGT_Attention(MessagePassing):
    """稀疏边增强注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.edge_bias_proj = nn.Linear(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, h_edge, edge_index):
        return self.propagate(edge_index, q=q, k=k, v=v, h_edge=h_edge)

    def message(self, q_i, k_j, v_j, h_edge, index, ptr, size_i):
        attn_score = (q_i * k_j).sum(dim=-1) * self.scale
        edge_bias = self.edge_bias_proj(h_edge)
        attn_score = attn_score + edge_bias
        attn_probs = sparse_softmax(attn_score, index, ptr, size_i)
        attn_probs = self.dropout(attn_probs)
        return v_j * attn_probs.unsqueeze(-1)


class Sparse_EGT_Layer(nn.Module):
    """图变换层"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = EGT_Attention(d_model, num_heads, dropout)

        self.norm1_node = nn.LayerNorm(d_model)
        self.norm2_node = nn.LayerNorm(d_model)
        self.ffn_node = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )

        self.norm_edge = nn.LayerNorm(d_model)
        self.edge_updater = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node, h_edge, edge_index):
        res = h_node
        q = self.q_proj(h_node).view(-1, self.num_heads, self.d_model // self.num_heads)
        k = self.k_proj(h_node).view(-1, self.num_heads, self.d_model // self.num_heads)
        v = self.v_proj(h_node).view(-1, self.num_heads, self.d_model // self.num_heads)

        out_node = self.attn(q, k, v, h_edge, edge_index).view(-1, self.d_model)
        out_node = self.out_proj(out_node)
        h_node = self.norm1_node(res + self.dropout(out_node))
        h_node = self.norm2_node(h_node + self.dropout(self.ffn_node(h_node)))

        src, dst = edge_index
        edge_input = torch.cat([h_node[src], h_node[dst], h_edge], dim=-1)
        h_edge_new = self.edge_updater(edge_input)
        h_edge = self.norm_edge(h_edge + self.dropout(h_edge_new))

        return h_node, h_edge


# ==========================================
# 2. 适配 Dataset 的特征融合层 (关键修改)
# ==========================================

class DualLiftingLayer(nn.Module):
    def __init__(self, lookback, d_model, dropout=0.1):
        super().__init__()
        self.lookback = lookback

        # --- Node Stream ---
        # 动态: Pressure(1) + Demand(1) = 2 inputs per step
        self.node_temporal_encoder = TemporalEncoder(2, d_model, dropout=dropout)

        # 静态: Elevation(1) + Type(Embed)
        self.node_elev_encoder = nn.Linear(1, d_model // 2)
        self.node_type_emb = nn.Embedding(10, d_model // 2)  # 假设 type < 10

        self.node_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        # --- Edge Stream ---
        # 动态: Flow(1)
        self.edge_temporal_encoder = TemporalEncoder(1, d_model, dropout=dropout)

        # 静态: Length(1) + Diameter(1) + Roughness(1) = 3
        self.edge_static_encoder = nn.Linear(3, d_model)

        self.edge_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

    def forward(self, x, edge_attr):
        """
        x: [N, 2*L + 2] -> P_win, D_win, Elev, Type
        edge_attr: [E, L + 3] -> F_win, Len, Diam, Rough
        """
        L = self.lookback

        # --- 1. Node Unpacking ---
        # 动态部分: 前 L 列是 Pressure, 中间 L 列是 Demand
        p_win = x[:, :L]  # [N, L]
        d_win = x[:, L:2 * L]  # [N, L]
        # 堆叠成时序特征 [N, L, 2]
        x_dyn_seq = torch.stack([p_win, d_win], dim=2)

        # 静态部分: 倒数第2个是 Elev, 倒数第1个是 Type
        elev = x[:, -2].unsqueeze(1)  # [N, 1]
        n_type = x[:, -1].long()  # [N] -> 转为 Long 用于 Embedding

        # --- 2. Node Encoding ---
        h_n_dyn = self.node_temporal_encoder(x_dyn_seq)  # [N, D]
        h_n_elev = self.node_elev_encoder(elev)  # [N, D/2]
        h_n_type = self.node_type_emb(n_type)  # [N, D/2]
        h_n_stat = torch.cat([h_n_elev, h_n_type], dim=-1)  # [N, D]

        h_node = self.node_fusion(torch.cat([h_n_stat, h_n_dyn], dim=-1))

        # --- 3. Edge Unpacking ---
        # 动态: 前 L 列是 Flow
        f_win = edge_attr[:, :L]  # [E, L]
        x_edge_seq = f_win.unsqueeze(-1)  # [E, L, 1]

        # 静态: 后 3 列
        edge_stat = edge_attr[:, -3:]  # [E, 3]

        # --- 4. Edge Encoding ---
        h_e_dyn = self.edge_temporal_encoder(x_edge_seq)
        h_e_stat = self.edge_static_encoder(edge_stat)

        h_edge = self.edge_fusion(torch.cat([h_e_stat, h_e_dyn], dim=-1))

        return h_node, h_edge


# ==========================================
# 3. 主模型
# ==========================================

class WaterPredictor(nn.Module):
    def __init__(self, lookback=12, horizon=1, d_model=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.lifting = DualLiftingLayer(lookback, d_model, dropout)

        self.layers = nn.ModuleList([
            Sparse_EGT_Layer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

        # 输出层预测未来 Horizon 步
        self.node_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon)
        )
        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon)
        )

    def forward(self, batch):
        # 直接使用 Batch 中的展平特征
        h_node, h_edge = self.lifting(batch.x, batch.edge_attr)

        for layer in self.layers:
            h_node, h_edge = layer(h_node, h_edge, batch.edge_index)

        pred_pressure = self.node_predictor(h_node)  # [Batch*N, Horizon]
        pred_flow = self.edge_predictor(h_edge)  # [Batch*E, Horizon]

        return pred_pressure, pred_flow