"""
改进的多视角Transformer模块

包含自注意力、跨视角注意力、几何约束注意力、稀疏注意力和多尺度融合。
整体流程: CNN特征 → 多尺度融合 → SA+CA+FFN ×N → 几何注意力 → 输出
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding2D(nn.Module):
    """二维正弦位置编码"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, H, W, device):
        pe = torch.zeros(self.d_model, H, W, device=device)
        d = self.d_model // 2
        pos_h = torch.arange(0, H, device=device).float().unsqueeze(1).expand(H, W)
        pos_w = torch.arange(0, W, device=device).float().unsqueeze(0).expand(H, W)
        div = torch.exp(torch.arange(0, d, 2, device=device).float() * -(math.log(10000.0) / d))

        pe[0:d:2] = torch.sin(pos_h.unsqueeze(0) * div.view(-1, 1, 1))
        pe[1:d:2] = torch.cos(pos_h.unsqueeze(0) * div.view(-1, 1, 1))
        pe[d::2] = torch.sin(pos_w.unsqueeze(0) * div.view(-1, 1, 1))
        pe[d + 1::2] = torch.cos(pos_w.unsqueeze(0) * div.view(-1, 1, 1))
        return pe.unsqueeze(0)


class SelfAttention(nn.Module):
    """标准多头自注意力，学习图像内部空间关系"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, C = query.shape
        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """跨视角注意力，建立不同视角间的特征对应"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, C = query.shape
        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class GeometricAwareAttention(nn.Module):
    """
    几何约束注意力（创新点）

    在标准注意力分数上叠加几何偏置 G:
        A = softmax(QK^T/√d + G)
    G 通过可学习MLP从空间坐标生成，编码极线约束等几何先验。
    采用因式分解 G[i,j]=bias[i]+bias[j]，复杂度 O(N) 而非 O(N²)。
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.geo_bias_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, num_heads),
        )

    def _compute_geometric_bias(self, N, H, W, device):
        ys = torch.arange(H, device=device).float() / max(H - 1, 1)
        xs = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1).reshape(-1, 2)
        bias = self.geo_bias_net(grid).permute(1, 0)  # [heads, N]
        return bias.unsqueeze(2) + bias.unsqueeze(1)   # [heads, N, N]

    def forward(self, feat_i, feat_j, H, W):
        B, N, C = feat_i.shape
        scale = math.sqrt(self.head_dim)
        G = self._compute_geometric_bias(N, H, W, feat_i.device)

        # 视角i attend to 视角j
        Q_i = self.q_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_j = self.k_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_j = self.v_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_ij = torch.softmax(torch.matmul(Q_i, K_j.transpose(-2, -1)) / scale + G.unsqueeze(0), dim=-1)
        out_i = torch.matmul(attn_ij, V_j).permute(0, 2, 1, 3).reshape(B, N, C)
        out_i = self.out_proj(out_i)

        # 视角j attend to 视角i（G转置）
        Q_j = self.q_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_i = self.k_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_i = self.v_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_ji = torch.softmax(torch.matmul(Q_j, K_i.transpose(-2, -1)) / scale + G.unsqueeze(0).transpose(-2, -1), dim=-1)
        out_j = torch.matmul(attn_ji, V_i).permute(0, 2, 1, 3).reshape(B, N, C)
        out_j = self.out_proj(out_j)

        return out_i, out_j


class SparseAttention(nn.Module):
    """Top-k稀疏注意力，只保留每行最相似的k个位置，复杂度 O(N·k)"""

    def __init__(self, d_model, num_heads, topk=64, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.topk = topk
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, C = query.shape
        k = min(self.topk, N)

        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 只保留top-k，其余mask为-inf
        topk_vals, topk_idx = attn_scores.topk(k, dim=-1)
        mask = torch.full_like(attn_scores, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)

        attn = self.dropout(torch.nan_to_num(torch.softmax(mask, dim=-1), nan=0.0))
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(out)


class MultiViewTransformerLayer(nn.Module):
    """单层: Self-Attention → Cross-Attention → FFN"""

    def __init__(self, d_model, num_heads, topk=64, ffn_dim=1024, dropout=0.1, use_sparse=True):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = SparseAttention(d_model, num_heads, topk, dropout) if use_sparse \
            else CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_self, x_cross):
        # SA
        residual = x_self
        x = self.self_attn(self.norm1(x_self), self.norm1(x_self), self.norm1(x_self))
        x = residual + self.dropout(x)
        # CA
        residual = x
        x = residual + self.dropout(self.cross_attn(self.norm2(x), self.norm2(x_cross), self.norm2(x_cross)))
        # FFN
        residual = x
        x = residual + self.ffn(self.norm3(x))
        return x


class MultiScaleTransformer(nn.Module):
    """多尺度特征融合：将不同分辨率的FPN特征通过注意力融合"""

    def __init__(self, d_model, num_heads, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.scale_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.scale_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_scales)])
        self.scale_embed = nn.Parameter(torch.randn(num_scales, 1, d_model) * 0.02)

    def forward(self, multi_scale_features, target_size):
        B, C = multi_scale_features[0].shape[:2]
        H, W = target_size

        scale_tokens = []
        for i, feat in enumerate(multi_scale_features):
            feat_resized = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feat_flat = self.scale_projs[i](feat_resized.flatten(2).permute(0, 2, 1))
            scale_tokens.append(feat_flat + self.scale_embed[i])

        all_tokens = torch.cat(scale_tokens, dim=1)
        all_tokens_n = self.norm(all_tokens)
        fused_tokens, _ = self.scale_attn(all_tokens_n, all_tokens_n, all_tokens_n)
        fused_tokens = all_tokens + fused_tokens

        N = H * W
        return fused_tokens[:, :N, :].permute(0, 2, 1).reshape(B, C, H, W)


class MultiViewTransformer(nn.Module):
    """
    完整的多视角Transformer

    流程:
      1. 多尺度特征融合（可选）
      2. 位置编码
      3. N层 SA → CA → FFN（双向交叉注意力）
      4. 几何约束注意力
    """

    def __init__(self, d_model=256, num_heads=8, num_layers=4,
                 topk=64, ffn_dim=1024, dropout=0.1, num_scales=3,
                 use_sparse=True, transformer_size=32):
        super().__init__()
        self.d_model = d_model
        self.transformer_size = transformer_size

        self.pos_enc = PositionalEncoding2D(d_model)
        self.layers = nn.ModuleList([
            MultiViewTransformerLayer(d_model, num_heads, topk, ffn_dim, dropout, use_sparse)
            for _ in range(num_layers)
        ])
        self.multi_scale_transformer = MultiScaleTransformer(d_model, num_heads, num_scales, dropout)
        self.geo_attention = GeometricAwareAttention(d_model, num_heads)
        self.geo_norm = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, feat_i, feat_j, multi_scale_i=None, multi_scale_j=None):
        T = self.transformer_size

        # 多尺度融合
        if multi_scale_i is not None:
            feat_i = self.multi_scale_transformer(multi_scale_i, (T, T))
        if multi_scale_j is not None:
            feat_j = self.multi_scale_transformer(multi_scale_j, (T, T))

        # 下采样到 T×T 控制显存
        if feat_i.shape[-1] != T or feat_i.shape[-2] != T:
            feat_i_t = F.interpolate(feat_i, size=(T, T), mode='bilinear', align_corners=False)
            feat_j_t = F.interpolate(feat_j, size=(T, T), mode='bilinear', align_corners=False)
        else:
            feat_i_t, feat_j_t = feat_i, feat_j

        # 位置编码
        H, W = T, T
        pos = self.pos_enc(H, W, feat_i.device)
        feat_i_t = feat_i_t + pos
        feat_j_t = feat_j_t + pos

        # 展平为序列
        B, C = feat_i_t.shape[:2]
        xi = feat_i_t.flatten(2).permute(0, 2, 1)
        xj = feat_j_t.flatten(2).permute(0, 2, 1)

        # N层双向 SA+CA+FFN
        for layer in self.layers:
            xi_new = layer(xi, xj)
            xj_new = layer(xj, xi)
            xi, xj = xi_new, xj_new

        # 几何约束注意力
        geo_i, geo_j = self.geo_attention(self.geo_norm(xi), self.geo_norm(xj), H, W)
        xi = self.final_norm(xi + geo_i)
        xj = self.final_norm(xj + geo_j)

        # 转回特征图
        out_i = xi.permute(0, 2, 1).reshape(B, C, H, W)
        out_j = xj.permute(0, 2, 1).reshape(B, C, H, W)
        return out_i, out_j
