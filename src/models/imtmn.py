"""
IMTMN: Improved Multi-View Transformer Matching Network

完整流程:
  1. ResNet50+FPN 多尺度特征提取
  2. 多视角Transformer (SA+CA+FFN + 几何约束注意力)
  3. 几何约束模块 (预测基础矩阵F)
  4. 特征匹配 (相似度矩阵 + Sinkhorn最优传输)
"""
import torch
import torch.nn as nn

from src.models.backbone import ResNetFPNBackbone
from src.models.transformer import MultiViewTransformer
from src.models.matcher import FeatureMatchingLayer


class GeometricConstraintModule(nn.Module):
    """从双视角特征预测基础矩阵F，用于极线约束损失"""

    def __init__(self, d_model=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 9),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_i, feat_j):
        B = feat_i.shape[0]
        gi = self.pool(feat_i).squeeze(-1).squeeze(-1)
        gj = self.pool(feat_j).squeeze(-1).squeeze(-1)
        return self.predictor(torch.cat([gi, gj], dim=1)).reshape(B, 3, 3)


class IMTMN(nn.Module):
    """
    完整的IMTMN模型

    参数:
        d_model: 特征维度
        num_heads: 注意力头数
        num_layers: Transformer层数
        topk_attn: 稀疏注意力的k值
        topk_match: 输出匹配点数
        sinkhorn_iters: Sinkhorn迭代次数
    """

    def __init__(self, d_model=256, num_heads=8, num_layers=4,
                 topk_attn=64, topk_match=200, sinkhorn_iters=50,
                 ffn_dim=1024, dropout=0.1, use_sparse=True,
                 transformer_size=32, pretrained_backbone=True):
        super().__init__()
        self.backbone = ResNetFPNBackbone(out_channels=d_model, pretrained=pretrained_backbone)
        self.mv_transformer = MultiViewTransformer(
            d_model=d_model, num_heads=num_heads, num_layers=num_layers,
            topk=topk_attn, ffn_dim=ffn_dim, dropout=dropout,
            num_scales=3, use_sparse=use_sparse, transformer_size=transformer_size,
        )
        self.geom_module = GeometricConstraintModule(d_model=d_model)
        self.matching_layer = FeatureMatchingLayer(
            d_model=d_model, sinkhorn_iters=sinkhorn_iters, topk=topk_match,
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, img_i, img_j):
        # 特征提取
        fusion_i, ms_i = self.extract_features(img_i)
        fusion_j, ms_j = self.extract_features(img_j)

        # 多视角Transformer增强
        feat_i, feat_j = self.mv_transformer(fusion_i, fusion_j, ms_i, ms_j)

        # 几何约束
        F_matrix = self.geom_module(feat_i, feat_j)

        # 特征匹配
        match_matrix, matches, match_scores, sim_matrix = self.matching_layer(feat_i, feat_j)

        return {
            'match_matrix': match_matrix,
            'matches': matches,
            'match_scores': match_scores,
            'sim_matrix': sim_matrix,
            'F_matrix': F_matrix,
            'feat_i': feat_i,
            'feat_j': feat_j,
        }

    def extract_global_feature(self, x):
        """提取全局特征向量，用于检索评估"""
        fusion, _ = self.extract_features(x)
        return fusion.mean(dim=[2, 3])
