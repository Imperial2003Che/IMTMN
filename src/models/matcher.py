"""
Feature Matching Layer + Optimal Transport

Pipeline 中的位置:
  Transformer → Geometric Constraint → [本模块] → Matching Points

两步:
  Step 1: Similarity Matrix  S = F1 · F2^T
  Step 2: Optimal Transport   M = Sinkhorn(S)  → Top-K matches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimilarityMatrix(nn.Module):
    """
    Step 1: 显式计算相似度矩阵

    S = F1 · F2^T   (内积相似度)

    输入: F1 [B, N, C],  F2 [B, N, C]
    输出: S  [B, N, N]   相似矩阵
    """
    def __init__(self, d_model=256):
        super().__init__()
        # 特征投影: 使特征更适合匹配
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, f1, f2):
        """
        Args:
            f1: [B, N, C]  视角A特征
            f2: [B, N, C]  视角B特征
        Returns:
            S: [B, N, N]   相似度矩阵
        """
        f1 = F.normalize(self.proj(f1), dim=-1)
        f2 = F.normalize(self.proj(f2), dim=-1)
        S = torch.matmul(f1, f2.transpose(-1, -2))  # S = F1 · F2^T
        return S


class OptimalTransportMatcher(nn.Module):
    """
    Step 2: Optimal Transport 匹配

    输入: S [B, N, N]   相似度矩阵
    过程:
      1. 添加 dustbin (垃圾桶) 行/列处理不匹配点
      2. Sinkhorn 迭代归一化 → 双随机匹配矩阵 M
    输出: M [B, N, N],  matches [B, K, 4],  scores [B, K]
    """
    def __init__(self, sinkhorn_iters=50, dustbin_score=1.5, topk=200):
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.topk = topk

        # 可学习 dustbin 参数
        self.dustbin = nn.Parameter(torch.tensor(dustbin_score))

    def sinkhorn(self, scores, num_iters):
        """
        Sinkhorn 归一化算法 (log-space)

        Args:
            scores: [B, M+1, N+1] 带 dustbin 的相似矩阵
            num_iters: 迭代次数
        Returns:
            P: [B, M+1, N+1] 双随机匹配矩阵
        """
        B, M, N = scores.shape

        log_mu = torch.full((B, M), fill_value=-math.log(M), device=scores.device)
        log_nu = torch.full((B, N), fill_value=-math.log(N), device=scores.device)

        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(num_iters):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)

        P = torch.exp(scores + u.unsqueeze(2) + v.unsqueeze(1))
        return P

    def forward(self, S, H, W):
        """
        Args:
            S: [B, N, N]  相似度矩阵 (来自 SimilarityMatrix)
            H, W: 空间维度
        Returns:
            match_matrix: [B, N, N]  匹配概率矩阵
            matches:      [B, K, 4]  Top-K 匹配点 (y1, x1, y2, x2)
            scores:       [B, K]     匹配分数
        """
        B, N, _ = S.shape

        # 添加 dustbin 行和列
        dustbin_row = self.dustbin.expand(B, N, 1)
        dustbin_col = self.dustbin.expand(B, 1, N + 1)
        S_aug = torch.cat([S, dustbin_row], dim=2)       # [B, N, N+1]
        S_aug = torch.cat([S_aug, dustbin_col], dim=1)    # [B, N+1, N+1]

        # Sinkhorn 归一化 → 匹配矩阵
        M = self.sinkhorn(S_aug, self.sinkhorn_iters)  # [B, N+1, N+1]

        # 去掉 dustbin，得到有效匹配矩阵
        match_matrix = M[:, :N, :N]  # [B, N, N]

        # 提取 Top-K 匹配
        k = min(self.topk, N)
        flat_scores = match_matrix.reshape(B, -1)
        topk_scores, topk_indices = flat_scores.topk(k, dim=1)

        # 转换为坐标 (y1, x1, y2, x2)
        idx_i = topk_indices // N
        idx_j = topk_indices % N
        y1 = (idx_i // W).float()
        x1 = (idx_i % W).float()
        y2 = (idx_j // W).float()
        x2 = (idx_j % W).float()

        matches = torch.stack([y1, x1, y2, x2], dim=-1)  # [B, K, 4]

        return match_matrix, matches, topk_scores


class FeatureMatchingLayer(nn.Module):
    """
    Feature Matching Layer (统一入口)

    Pipeline:
      feat_i [B,C,H,W], feat_j [B,C,H,W]
        → flatten → SimilarityMatrix (S = F1·F2^T)
        → OptimalTransport (Sinkhorn)
        → match_matrix, matches, scores
    """
    def __init__(self, d_model=256, sinkhorn_iters=50, dustbin_score=1.5, topk=200):
        super().__init__()
        self.sim_matrix = SimilarityMatrix(d_model)
        self.ot_matcher = OptimalTransportMatcher(sinkhorn_iters, dustbin_score, topk)

    def forward(self, feat_i, feat_j):
        """
        Args:
            feat_i: [B, C, H, W]  视角A特征
            feat_j: [B, C, H, W]  视角B特征
        Returns:
            match_matrix: [B, N, N]
            matches: [B, K, 4]   (y1, x1, y2, x2)
            scores: [B, K]
            S: [B, N, N]  原始相似度矩阵
        """
        B, C, H, W = feat_i.shape

        # 展平
        f1 = feat_i.flatten(2).permute(0, 2, 1)  # [B, N, C]
        f2 = feat_j.flatten(2).permute(0, 2, 1)

        # Step 1: Similarity Matrix  S = F1 · F2^T
        S = self.sim_matrix(f1, f2)

        # Step 2: Optimal Transport (Sinkhorn)
        match_matrix, matches, scores = self.ot_matcher(S, H, W)

        return match_matrix, matches, scores, S
