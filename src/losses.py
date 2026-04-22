"""
IMTMN 损失函数

总损失: L = w_match * L_match + w_geo * L_geo + w_retrieval * L_retrieval
"""
import torch
import torch.nn.functional as F


def loss_match(match_matrix, match_scores):
    """匹配损失：最大化Top-K匹配分数 + 熵正则化鼓励稀疏匹配"""
    if match_scores is None or match_scores.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    nll_loss = -torch.log(torch.clamp(match_scores, min=1e-8)).mean()

    row_entropy = -(match_matrix * torch.log(match_matrix + 1e-8)).sum(dim=-1).mean()
    col_entropy = -(match_matrix * torch.log(match_matrix + 1e-8)).sum(dim=-2).mean()
    entropy_reg = (row_entropy + col_entropy) * 0.01

    return nll_loss + entropy_reg


def loss_geo(F_matrix, matches):
    """几何一致性损失：极线约束 ||x2^T F x1||"""
    if F_matrix is None or F_matrix.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    B, K, _ = matches.shape
    y1, x1, y2, x2 = matches[..., 0], matches[..., 1], matches[..., 2], matches[..., 3]
    ones = torch.ones_like(x1)

    pts1 = torch.stack([x1, y1, ones], dim=-1)
    pts2 = torch.stack([x2, y2, ones], dim=-1)

    Fx1 = torch.bmm(F_matrix, pts1.transpose(1, 2))
    epipolar_error = (pts2.transpose(1, 2) * Fx1).sum(dim=1)
    geo_loss = epipolar_error.abs().mean()

    # 尺度稳定正则（避免F矩阵发散）
    f_norm = torch.linalg.norm(F_matrix.reshape(B, -1), dim=1)
    scale_reg = ((f_norm - 1.0) ** 2).mean() * 0.01

    return geo_loss + scale_reg


def loss_retrieval(feat_i, feat_j, margin=0.5, temperature=0.07):
    """InfoNCE对比学习损失，用于跨视角全局特征对齐"""
    gi = F.normalize(feat_i.mean(dim=[2, 3]), dim=1)
    gj = F.normalize(feat_j.mean(dim=[2, 3]), dim=1)

    sim = torch.matmul(gi, gj.t()) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0


def total_loss(L_match, L_geo, L_retrieval=None, w_match=1.0, w_geo=0.5, w_retrieval=0.5):
    """加权总损失"""
    L = w_match * L_match + w_geo * L_geo
    if L_retrieval is not None:
        L = L + w_retrieval * L_retrieval
    return L
