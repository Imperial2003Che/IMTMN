import torch
import torch.nn as nn

class GeomHead(nn.Module):
    """
    几何头：从匹配点对预测本质矩阵 F 或本质矩阵 E
    输入：matches（列表或张量化的点对信息），输出：F参数向量（9 维）
    """
    def __init__(self, in_dim=4, out_dim=9):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, match_features):
        """
        match_features: Tensor [N, in_dim]
        返回: F矩阵参数 [N, 9] 或聚合后的单一 F
        """
        theta = self.mlp(match_features)  # [N, 9]
        # 这里简单地将每对点的 theta 变换成 3x3 矩阵参数；你也可以做聚合再输出一组矩阵
        # 及后续通过构造 F = theta.view(-1,3,3)
        F = theta.view(-1, 3, 3)
        return F