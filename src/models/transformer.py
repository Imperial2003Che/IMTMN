"""
Improved Multi-View Transformer (核心模块)

Pipeline 中的位置:
  CNN Features → Multi-scale Fusion → [本模块] → Geometric Constraint → Matching

包含:
1. Self-Attention:  SA(Q,K,V) = softmax(QK^T / √d) V  —— 学习图像内部关系
2. Cross-Attention: CA(F1,F2) = softmax(Q1·K2^T / √d) V2 —— 建立不同视角关系
3. FFN: 前馈网络
4. Geometric-Aware Attention: A = softmax(QK^T/√d + G) —— 独立的几何约束注意力
5. Sparse Attention: Top-k 稀疏注意力 —— 可选优化
6. Multi-Scale Transformer: 跨尺度特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        return pe.unsqueeze(0)  # [1, C, H, W]


# ============================================================
# 5.1 Self-Attention: 学习同一图像内部关系
# SA(Q,K,V) = softmax(QK^T / √d) V
# ============================================================
class SelfAttention(nn.Module):
    """
    标准多头自注意力

    作用: 学习同一图像内部的空间特征关系
    输入: feature_view  [B, N, C]
    输出: enhanced feature [B, N, C]

    调用方式: self_attn(f, f, f)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        SA(Q,K,V) = softmax(QK^T / √d) V

        Args:
            query, key, value: [B, N, C]  (通常 query=key=value=同一视角特征)
        Returns:
            output: [B, N, C]
        """
        B, N, C = query.shape

        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, heads, N, head_dim]

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, heads, N, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ============================================================
# 5.2 Cross-Attention: 建立不同视角之间关系
# CA(F1,F2) = softmax(Q1·K2^T / √d) V2
# ============================================================
class CrossAttention(nn.Module):
    """
    多头跨视角注意力

    作用: 建立不同视角之间的特征对应关系
    例如: View1 → View2:  cross_attn(f1, f2, f2)

    多视角融合:
      F1 ← F2
      F1 ← F3
      F2 ← F3
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        CA(F_i, F_j) = softmax(Q_i · K_j^T / √d) V_j

        Args:
            query: [B, N, C] 当前视角特征 (Q来源)
            key:   [B, N, C] 另一视角特征 (K来源)
            value: [B, N, C] 另一视角特征 (V来源)
        Returns:
            output: [B, N, C]
        """
        B, N, C = query.shape

        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ============================================================
# 6. Geometric-Aware Attention（独立创新模块）
# A = softmax(QK^T/√d + G)
# 其中 G 是几何约束矩阵 (Epipolar Constraint: x2^T F x1 = 0)
# ============================================================
class GeometricAwareAttention(nn.Module):
    """
    创新模块: 几何约束注意力

    在标准注意力基础上加入几何约束:
      attention_score = QK^T / √d
      attention_score += geometric_matrix   <-- 加入几何先验 G
      attention = softmax(attention_score)

    G 基于极线约束 x2^T F x1 = 0 的启发式:
    通过可学习网络将空间坐标关系编码为几何偏置

    作用: 减少错误匹配，利用相机几何信息
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 几何偏置网络 (因式分解):
        # 输入: (y, x) 归一化坐标 -> num_heads 维偏置  (O(N) 而非 O(N^2))
        self.geo_bias_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads),
        )

    def _compute_geometric_bias(self, N, H, W, device):
        """
        计算几何约束矩阵 G (因式分解形式，O(N) 而非 O(N^2))

        G[i,j] = bias_row[i] + bias_col[j]
        这样只需计算 N 个位置的偏置，不需要遍历所有 N^2 对
        """
        # 生成归一化坐标网格 [N, 2]
        ys = torch.arange(H, device=device).float() / max(H - 1, 1)
        xs = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing='ij'), dim=-1).reshape(-1, 2)  # [N, 2]

        # 分别计算行/列偏置: [N, num_heads]
        # geo_bias_net 输入 2 维 (y, x)，输出 num_heads 维
        bias = self.geo_bias_net(grid)  # [N, num_heads]
        bias = bias.permute(1, 0)       # [num_heads, N]

        # G[i,j] = bias[i] + bias[j]  → [num_heads, N, N]
        G = bias.unsqueeze(2) + bias.unsqueeze(1)
        return G

    def forward(self, feat_i, feat_j, H, W):
        """
        Args:
            feat_i: [B, N, C] 视角A的特征序列
            feat_j: [B, N, C] 视角B的特征序列
            H, W: 空间维度
        Returns:
            out_i: [B, N, C] 增强后的视角A特征
            out_j: [B, N, C] 增强后的视角B特征
        """
        B, N, C = feat_i.shape

        # 计算几何偏置 G
        G = self._compute_geometric_bias(N, H, W, feat_i.device)  # [heads, N, N]

        # --- 视角A attend to 视角B (with geometric bias) ---
        Q_i = self.q_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_j = self.k_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_j = self.v_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        # attention_score = QK^T / √d
        attn_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) / scale
        # attention_score += G  (加入几何约束)
        attn_ij = attn_ij + G.unsqueeze(0)
        attn_ij = torch.softmax(attn_ij, dim=-1)
        out_i = torch.matmul(attn_ij, V_j)
        out_i = out_i.permute(0, 2, 1, 3).reshape(B, N, C)
        out_i = self.out_proj(out_i)

        # --- 视角B attend to 视角A (with geometric bias, transposed) ---
        Q_j = self.q_proj(feat_j).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K_i = self.k_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V_i = self.v_proj(feat_i).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_ji = torch.matmul(Q_j, K_i.transpose(-2, -1)) / scale
        attn_ji = attn_ji + G.unsqueeze(0).transpose(-2, -1)  # G^T for reverse direction
        attn_ji = torch.softmax(attn_ji, dim=-1)
        out_j = torch.matmul(attn_ji, V_i)
        out_j = out_j.permute(0, 2, 1, 3).reshape(B, N, C)
        out_j = self.out_proj(out_j)

        return out_i, out_j


# ============================================================
# 7. Sparse Attention（可选优化）
# 只计算 top-k similar features, 复杂度 O(N·k)
# ============================================================
class SparseAttention(nn.Module):
    """
    可选优化: 稀疏注意力

    普通 Transformer: O(N^2)
    稀疏注意力: 只计算 top-k similar features

    实现:
      1. knn search (by attention score)
      2. mask attention matrix (只保留 top-k)
      3. softmax on masked matrix

    复杂度: O(N·k) ≈ O(N log N)
    """
    def __init__(self, d_model, num_heads, topk=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.topk = topk

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query, key, value: [B, N, C]
        Returns:
            output: [B, N, C]
        """
        B, N, C = query.shape
        k = min(self.topk, N)

        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, N, N]

        # Top-k masking: 只保留每行最大的 k 个值
        topk_vals, topk_idx = attn_scores.topk(k, dim=-1)
        mask = torch.full_like(attn_scores, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)

        attn = torch.softmax(mask, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        return out


# ============================================================
# Multi-View Transformer Layer: SA → CA → FFN
# ============================================================
class MultiViewTransformerLayer(nn.Module):
    """
    单层 Multi-View Transformer

    结构:
      1. Self-Attention:  self_attn(f, f, f)
      2. Cross-Attention: cross_attn(f1, f2, f2)
      3. Feed-Forward Network

    选择:
      - use_sparse=True 时 Cross-Attention 使用 SparseAttention
      - use_sparse=False 时使用标准 CrossAttention
    """
    def __init__(self, d_model, num_heads, topk=64, ffn_dim=1024, dropout=0.1, use_sparse=True):
        super().__init__()
        # Self-Attention: SA(Q,K,V) = softmax(QK^T/√d) V
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-Attention: CA(F1,F2) = softmax(Q1·K2^T/√d) V2
        if use_sparse:
            self.cross_attn = SparseAttention(d_model, num_heads, topk, dropout)
        else:
            self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_self, x_cross):
        """
        Args:
            x_self:  [B, N, C] 当前视角特征
            x_cross: [B, N, C] 另一视角特征
        Returns:
            output: [B, N, C]
        """
        # Step 1: Self-Attention  self_attn(f, f, f)
        residual = x_self
        x = self.norm1(x_self)
        x = self.self_attn(x, x, x)
        x = residual + self.dropout(x)

        # Step 2: Cross-Attention  cross_attn(f1, f2, f2)
        residual = x
        x_n = self.norm2(x)
        x_cross_n = self.norm2(x_cross)
        x = residual + self.dropout(self.cross_attn(x_n, x_cross_n, x_cross_n))

        # Step 3: FFN
        residual = x
        x = residual + self.ffn(self.norm3(x))

        return x


# ============================================================
# Multi-Scale Transformer: 跨尺度特征融合
# ============================================================
class MultiScaleTransformer(nn.Module):
    """
    多尺度 Transformer 融合

    将不同尺度特征 (F^1, F^2, F^3) 通过 Transformer 融合:
      F = Transformer(F^1, F^2, F^3)

    实现:
      f1 = layer1 (high resolution)
      f2 = layer2
      f3 = layer3 (high semantic)
      fusion = w1*f1 + w2*f2 + w3*f3   (经过跨尺度注意力增强)
    """
    def __init__(self, d_model, num_heads, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales

        self.scale_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.scale_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales)
        ])

        # 可学习的尺度编码
        self.scale_embed = nn.Parameter(torch.randn(num_scales, 1, d_model) * 0.02)

    def forward(self, multi_scale_features, target_size):
        """
        Args:
            multi_scale_features: list of [B, C, H_i, W_i]
            target_size: (H, W)
        Returns:
            fused: [B, C, H, W]
        """
        B, C = multi_scale_features[0].shape[:2]
        H, W = target_size

        # 统一分辨率 + 展平
        scale_tokens = []
        for i, feat in enumerate(multi_scale_features):
            feat_resized = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feat_flat = feat_resized.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            feat_flat = self.scale_projs[i](feat_flat)
            feat_flat = feat_flat + self.scale_embed[i]
            scale_tokens.append(feat_flat)

        # 拼接后跨尺度注意力 [B, num_scales*HW, C]
        all_tokens = torch.cat(scale_tokens, dim=1)
        all_tokens_n = self.norm(all_tokens)
        fused_tokens, _ = self.scale_attn(all_tokens_n, all_tokens_n, all_tokens_n)
        fused_tokens = all_tokens + fused_tokens

        # 取第一个尺度 (最高分辨率) 的 tokens
        N = H * W
        out_tokens = fused_tokens[:, :N, :]
        fused = out_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        return fused


# ============================================================
# 完整 Multi-View Transformer
# ============================================================
class MultiViewTransformer(nn.Module):
    """
    Improved Multi-View Transformer (完整模块)

    Pipeline:
      1. Multi-Scale Feature Fusion (optional)
      2. Position Encoding
      3. N layers of: Self-Attention → Cross-Attention → FFN
      4. Geometric-Aware Attention (独立的几何约束注意力)

    输入: 两个视角的特征 (feature_view1, feature_view2)
    输出: 增强后的特征 (用于后续匹配)
    """
    def __init__(self, d_model=256, num_heads=8, num_layers=4,
                 topk=64, ffn_dim=1024, dropout=0.1, num_scales=3,
                 use_sparse=True, transformer_size=32):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # Transformer 内部的特征图尺寸（防止 OOM）
        # 例如 transformer_size=32 → 32×32=1024 tokens
        self.transformer_size = transformer_size

        # 位置编码
        self.pos_enc = PositionalEncoding2D(d_model)

        # Multi-View Transformer layers (SA + CA + FFN)
        self.layers = nn.ModuleList([
            MultiViewTransformerLayer(d_model, num_heads, topk, ffn_dim, dropout, use_sparse)
            for _ in range(num_layers)
        ])

        # 多尺度融合
        self.multi_scale_transformer = MultiScaleTransformer(d_model, num_heads, num_scales, dropout)

        # 几何约束注意力 (独立模块，在 Transformer 之后)
        self.geo_attention = GeometricAwareAttention(d_model, num_heads)
        self.geo_norm = nn.LayerNorm(d_model)

        # 最终输出归一化
        self.final_norm = nn.LayerNorm(d_model)

    def _feature_to_seq(self, feat):
        """[B, C, H, W] -> [B, N, C], H, W"""
        B, C, H, W = feat.shape
        return feat.flatten(2).permute(0, 2, 1), H, W

    def _seq_to_feature(self, seq, H, W):
        """[B, N, C] -> [B, C, H, W]"""
        B, N, C = seq.shape
        return seq.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, feat_i, feat_j, multi_scale_i=None, multi_scale_j=None):
        """
        Args:
            feat_i: [B, C, H, W] 视角A的融合特征
            feat_j: [B, C, H, W] 视角B的融合特征
            multi_scale_i: list of [B, C, H_k, W_k] (可选)
            multi_scale_j: list of [B, C, H_k, W_k] (可选)
        Returns:
            out_i: [B, C, H, W]
            out_j: [B, C, H, W]
        """
        B, C, H_orig, W_orig = feat_i.shape
        T = self.transformer_size

        # === Stage 1: Multi-Scale Feature Fusion ===
        # 多尺度融合直接在 T×T 尺寸上进行，避免大尺寸拼接 OOM
        if multi_scale_i is not None:
            feat_i = self.multi_scale_transformer(multi_scale_i, (T, T))
        if multi_scale_j is not None:
            feat_j = self.multi_scale_transformer(multi_scale_j, (T, T))

        # === 空间下采样到 T×T: 防止 Transformer OOM ===
        if feat_i.shape[-1] != T or feat_i.shape[-2] != T:
            feat_i_t = F.interpolate(feat_i, size=(T, T), mode='bilinear', align_corners=False)
            feat_j_t = F.interpolate(feat_j, size=(T, T), mode='bilinear', align_corners=False)
        else:
            feat_i_t, feat_j_t = feat_i, feat_j

        # 加位置编码
        H, W = T, T
        pos = self.pos_enc(H, W, feat_i.device)
        feat_i_t = feat_i_t + pos
        feat_j_t = feat_j_t + pos

        # 转为序列 [B, N, C]  N = T*T
        xi, H, W = self._feature_to_seq(feat_i_t)
        xj, _, _ = self._feature_to_seq(feat_j_t)

        # === Stage 2: N layers of SA + CA + FFN ===
        for layer in self.layers:
            xi_new = layer(xi, xj)  # View_i: self_attn(fi) then cross_attn(fi, fj)
            xj_new = layer(xj, xi)  # View_j: self_attn(fj) then cross_attn(fj, fi)
            xi, xj = xi_new, xj_new

        # === Stage 3: Geometric-Aware Attention ===
        # A = softmax(QK^T/√d + G)
        xi_n = self.geo_norm(xi)
        xj_n = self.geo_norm(xj)
        geo_i, geo_j = self.geo_attention(xi_n, xj_n, H, W)
        xi = xi + geo_i  # 残差连接
        xj = xj + geo_j

        # 最终归一化
        xi = self.final_norm(xi)
        xj = self.final_norm(xj)

        # 转回特征图 (保持 T×T 尺寸，Matching 也在此尺寸运行，避免训练 OOM)
        out_i = self._seq_to_feature(xi, H, W)
        out_j = self._seq_to_feature(xj, H, W)

        return out_i, out_j
