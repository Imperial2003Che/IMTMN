"""
IMTMN Loss Functions

L = L_match + λ * L_geo

1. L_match: 匹配损失 - negative log-likelihood of ground-truth matches
2. L_geo: 几何一致性损失 - 极线约束 ||x2^T F x1||
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_match(match_matrix, match_scores):
    """
    匹配损失: Optimal Transport 匹配矩阵的负对数似然

    对于自监督场景 (无 ground truth 对应关系):
    - 鼓励匹配矩阵趋向于置换矩阵 (每行/每列只有一个高置信匹配)
    - 最大化 Top-K 匹配分数

    L_match = -sum(log(p_gt))

    Args:
        match_matrix: [B, N, N] 匹配概率矩阵
        match_scores: [B, K] Top-K 匹配分数
    Returns:
        scalar loss
    """
    if match_scores is None or match_scores.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    # 1. 负对数似然: 最大化匹配分数
    # 避免 log(0) 的情况
    scores_clamped = torch.clamp(match_scores, min=1e-8)
    nll_loss = -torch.log(scores_clamped).mean()

    # 2. 双随机约束: 鼓励匹配矩阵行列和接近1 (隐含在 Sinkhorn 中)
    # 这里额外加一个行列熵正则化，鼓励稀疏匹配
    row_entropy = -(match_matrix * torch.log(match_matrix + 1e-8)).sum(dim=-1).mean()
    col_entropy = -(match_matrix * torch.log(match_matrix + 1e-8)).sum(dim=-2).mean()
    entropy_reg = (row_entropy + col_entropy) * 0.01

    return nll_loss + entropy_reg


def loss_geo(F_matrix, matches):
    """
    几何一致性损失: 极线约束

    L_geo = ||x2^T F x1||

    对于预测的匹配点对和基础矩阵 F，计算极线约束残差

    Args:
        F_matrix: [B, 3, 3] 预测的基础矩阵
        matches: [B, K, 4] 匹配坐标 (y1, x1, y2, x2)
    Returns:
        scalar loss
    """
    if F_matrix is None or F_matrix.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    B, K, _ = matches.shape

    # 构建齐次坐标 [B, K, 3]
    y1, x1, y2, x2 = matches[..., 0], matches[..., 1], matches[..., 2], matches[..., 3]
    ones = torch.ones_like(x1)

    pts1 = torch.stack([x1, y1, ones], dim=-1)  # [B, K, 3]
    pts2 = torch.stack([x2, y2, ones], dim=-1)  # [B, K, 3]

    # 极线约束: x2^T F x1 = 0
    # F @ pts1^T -> [B, 3, K]
    Fx1 = torch.bmm(F_matrix, pts1.transpose(1, 2))  # [B, 3, K]
    # x2^T @ (F @ x1) -> 逐元素点积
    epipolar_error = (pts2.transpose(1, 2) * Fx1).sum(dim=1)  # [B, K]

    # 取绝对值平均
    geo_loss = epipolar_error.abs().mean()

    # 基础矩阵正则化: det(F) = 0 约束 (秩2约束)
    det_F = torch.det(F_matrix)  # [B]
    rank_reg = det_F.abs().mean() * 0.1

    return geo_loss + rank_reg


def loss_retrieval(feat_i, feat_j, margin=0.5, temperature=0.07):
    """
    对比学习检索损失: InfoNCE

    用于学习跨视角全局特征的对齐

    Args:
        feat_i: [B, C, H, W] 视角A特征
        feat_j: [B, C, H, W] 视角B特征
        margin: 三元组损失的 margin
        temperature: InfoNCE 温度参数
    Returns:
        scalar loss
    """
    # 全局平均池化 -> [B, C]
    gi = feat_i.mean(dim=[2, 3])
    gj = feat_j.mean(dim=[2, 3])

    # L2 归一化
    gi = F.normalize(gi, dim=1)
    gj = F.normalize(gj, dim=1)

    # InfoNCE: 对角线为正样本，其余为负样本
    sim = torch.matmul(gi, gj.t()) / temperature  # [B, B]
    labels = torch.arange(sim.shape[0], device=sim.device)

    loss_ij = F.cross_entropy(sim, labels)
    loss_ji = F.cross_entropy(sim.t(), labels)

    return (loss_ij + loss_ji) / 2.0


def total_loss(L_match, L_geo, L_retrieval=None,
               w_match=1.0, w_geo=0.5, w_retrieval=0.5):
    """
    总损失函数

    L = w_match * L_match + w_geo * L_geo + w_retrieval * L_retrieval

    Args:
        L_match: 匹配损失
        L_geo: 几何一致性损失
        L_retrieval: 检索对比损失 (可选)
        w_match, w_geo, w_retrieval: 损失权重
    Returns:
        scalar total loss
    """
    L = w_match * L_match + w_geo * L_geo

    if L_retrieval is not None:
        L = L + w_retrieval * L_retrieval

    return L
