"""
IMTMN: Improved Multi-View Transformer Matching Network

完整 Pipeline:
  ┌──────────────────────────────────────────────────────┐
  │  Step 1: CNN Feature Extraction (ResNet50 + FPN)     │
  │          → 多尺度特征 + 融合特征                      │
  ├──────────────────────────────────────────────────────┤
  │  Step 2: Flatten + Positional Encoding               │
  │          [B,C,H,W] → [B,N,C]                        │
  ├──────────────────────────────────────────────────────┤
  │  Step 3: Multi-View Transformer                      │
  │          SA(self_attn) → CA(cross_attn) → FFN        │
  │          ×N layers, 双向: fi↔fj                      │
  ├──────────────────────────────────────────────────────┤
  │  Step 4: Geometric-Aware Attention (创新模块)         │
  │          A = softmax(QK^T/√d + G)                    │
  │          独立于 Transformer，加入几何先验              │
  ├──────────────────────────────────────────────────────┤
  │  Step 5: Geometric Constraint Module                 │
  │          预测基础矩阵 F (x2^T F x1 = 0)              │
  ├──────────────────────────────────────────────────────┤
  │  Step 6: Feature Matching Layer                      │
  │          6a. Similarity Matrix:  S = F1 · F2^T       │
  │          6b. Optimal Transport:  M = Sinkhorn(S)     │
  │          6c. Top-K Matches                           │
  └──────────────────────────────────────────────────────┘
"""
import torch
import torch.nn as nn

from src.models.backbone import ResNetFPNBackbone
from src.models.transformer import MultiViewTransformer
from src.models.matcher import FeatureMatchingLayer


class GeometricConstraintModule(nn.Module):
    """
    几何约束模块: 从匹配特征预测基础矩阵 F

    极线约束: x2^T F x1 = 0
    用于 loss_geo 计算，约束匹配的几何一致性
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_i, feat_j):
        """
        Args:
            feat_i: [B, C, H, W]
            feat_j: [B, C, H, W]
        Returns:
            F_mat: [B, 3, 3] 预测的基础矩阵
        """
        B = feat_i.shape[0]
        gi = self.pool(feat_i).squeeze(-1).squeeze(-1)  # [B, C]
        gj = self.pool(feat_j).squeeze(-1).squeeze(-1)
        combined = torch.cat([gi, gj], dim=1)  # [B, 2C]
        F_mat = self.predictor(combined).reshape(B, 3, 3)
        return F_mat


class IMTMN(nn.Module):
    """
    Improved Multi-View Transformer Matching Network

    Args:
        d_model: 特征维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        topk_attn: 稀疏注意力 Top-k
        topk_match: 匹配输出 Top-k
        sinkhorn_iters: Sinkhorn 迭代次数
        use_sparse: 是否使用稀疏注意力 (可选优化)
        pretrained_backbone: 是否使用预训练 ResNet
    """
    def __init__(self, d_model=256, num_heads=8, num_layers=4,
                 topk_attn=64, topk_match=200, sinkhorn_iters=50,
                 ffn_dim=1024, dropout=0.1, use_sparse=True,
                 transformer_size=32, pretrained_backbone=True):
        super().__init__()

        # Step 1: CNN Feature Extraction (ResNet50 + FPN)
        self.backbone = ResNetFPNBackbone(out_channels=d_model, pretrained=pretrained_backbone)

        # Step 2-4: Multi-View Transformer (含位置编码、SA+CA+FFN、GeometricAttention)
        self.mv_transformer = MultiViewTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            topk=topk_attn,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_scales=3,
            use_sparse=use_sparse,
            transformer_size=transformer_size,
        )

        # Step 5: Geometric Constraint Module (预测基础矩阵 F)
        self.geom_module = GeometricConstraintModule(d_model=d_model)

        # Step 6: Feature Matching Layer (S = F1·F2^T → Sinkhorn → matches)
        self.matching_layer = FeatureMatchingLayer(
            d_model=d_model,
            sinkhorn_iters=sinkhorn_iters,
            topk=topk_match,
        )

    def extract_features(self, x):
        """
        Step 1: CNN 多尺度特征提取
        Args:
            x: [B, 3, H, W]
        Returns:
            fusion: [B, C, H', W'] 融合特征
            multi_scale: list of [B, C, H_k, W_k]
        """
        return self.backbone(x)

    def forward(self, img_i, img_j):
        """
        完整前向传播 Pipeline

        Args:
            img_i: [B, 3, H, W] 视角A图像
            img_j: [B, 3, H, W] 视角B图像
        Returns:
            dict:
                match_matrix: [B, N, N]  匹配概率矩阵
                matches:      [B, K, 4]  Top-K 匹配坐标 (y1,x1,y2,x2)
                match_scores: [B, K]     匹配分数
                sim_matrix:   [B, N, N]  原始相似度矩阵 S = F1·F2^T
                F_matrix:     [B, 3, 3]  预测的基础矩阵
                feat_i:       [B, C, H', W']  增强后特征A
                feat_j:       [B, C, H', W']  增强后特征B
        """
        # ==== Step 1: CNN Feature Extraction ====
        fusion_i, ms_i = self.extract_features(img_i)
        fusion_j, ms_j = self.extract_features(img_j)

        # ==== Step 2-4: Multi-View Transformer ====
        # 内部包含:
        #   2. Multi-Scale Fusion + Positional Encoding
        #   3. SA → CA → FFN  ×N layers (双向交叉注意力)
        #   4. Geometric-Aware Attention: A = softmax(QK^T/√d + G)
        feat_i, feat_j = self.mv_transformer(fusion_i, fusion_j, ms_i, ms_j)

        # ==== Step 5: Geometric Constraint Module ====
        F_matrix = self.geom_module(feat_i, feat_j)

        # ==== Step 6: Feature Matching Layer ====
        # 6a. Similarity Matrix: S = F1 · F2^T
        # 6b. Optimal Transport: M = Sinkhorn(S)
        # 6c. Top-K Matches
        match_matrix, matches, match_scores, sim_matrix = self.matching_layer(feat_i, feat_j)

        return {
            'match_matrix': match_matrix,    # [B, N, N] 匹配概率
            'matches': matches,              # [B, K, 4] 匹配坐标
            'match_scores': match_scores,    # [B, K]    匹配分数
            'sim_matrix': sim_matrix,        # [B, N, N] 相似度矩阵
            'F_matrix': F_matrix,            # [B, 3, 3] 基础矩阵
            'feat_i': feat_i,                # [B, C, H', W']
            'feat_j': feat_j,                # [B, C, H', W']
        }

    def extract_global_feature(self, x):
        """
        提取单视角全局特征 (用于检索评估)
        Args:
            x: [B, 3, H, W]
        Returns:
            global_feat: [B, C]
        """
        fusion, _ = self.extract_features(x)
        return fusion.mean(dim=[2, 3])  # Global Average Pooling
