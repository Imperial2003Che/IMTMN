"""
特征匹配层：相似度矩阵 + Sinkhorn最优传输

流程: 特征展平 → 相似度计算 S=F1·F2^T → Sinkhorn归一化 → Top-K匹配
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityMatrix(nn.Module):
    """计算两组特征的相似度矩阵 S = proj(F1) · proj(F2)^T"""

    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )

    def forward(self, f1, f2):
        f1 = F.normalize(self.proj(f1), dim=-1)
        f2 = F.normalize(self.proj(f2), dim=-1)
        return torch.matmul(f1, f2.transpose(-1, -2))


class OptimalTransportMatcher(nn.Module):
    """
    基于Sinkhorn算法的最优传输匹配

    添加dustbin行列处理无对应点，通过迭代归一化得到双随机匹配矩阵。
    """

    def __init__(self, sinkhorn_iters=50, dustbin_score=1.5, topk=200):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.topk = topk
        self.dustbin = nn.Parameter(torch.tensor(dustbin_score))

    def sinkhorn(self, scores, num_iters):
        """Log-space Sinkhorn归一化"""
        B, M, N = scores.shape
        log_mu = torch.full((B, M), fill_value=-math.log(M), device=scores.device)
        log_nu = torch.full((B, N), fill_value=-math.log(N), device=scores.device)
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(num_iters):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)

        return torch.exp(scores + u.unsqueeze(2) + v.unsqueeze(1))

    def forward(self, S, H, W):
        B, N, _ = S.shape

        # 添加dustbin
        dustbin_row = self.dustbin.expand(B, N, 1)
        dustbin_col = self.dustbin.expand(B, 1, N + 1)
        S_aug = torch.cat([torch.cat([S, dustbin_row], dim=2), dustbin_col], dim=1)

        M = self.sinkhorn(S_aug, self.sinkhorn_iters)
        match_matrix = M[:, :N, :N]

        # Top-K匹配
        k = min(self.topk, N)
        flat_scores = match_matrix.reshape(B, -1)
        topk_scores, topk_indices = flat_scores.topk(k, dim=1)

        idx_i = topk_indices // N
        idx_j = topk_indices % N
        y1, x1 = (idx_i // W).float(), (idx_i % W).float()
        y2, x2 = (idx_j // W).float(), (idx_j % W).float()
        matches = torch.stack([y1, x1, y2, x2], dim=-1)

        return match_matrix, matches, topk_scores


class FeatureMatchingLayer(nn.Module):
    """特征匹配统一入口：相似度计算 → Sinkhorn → 匹配结果"""

    def __init__(self, d_model=256, sinkhorn_iters=50, dustbin_score=1.5, topk=200):
        super().__init__()
        self.sim_matrix = SimilarityMatrix(d_model)
        self.ot_matcher = OptimalTransportMatcher(sinkhorn_iters, dustbin_score, topk)

    def forward(self, feat_i, feat_j):
        B, C, H, W = feat_i.shape
        f1 = feat_i.flatten(2).permute(0, 2, 1)
        f2 = feat_j.flatten(2).permute(0, 2, 1)
        S = self.sim_matrix(f1, f2)
        match_matrix, matches, scores = self.ot_matcher(S, H, W)
        return match_matrix, matches, scores, S
