import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    基于 Linear 的 Patch Embedding（不使用 Conv2d）
    将图像分割为不重叠的 patch，然后通过线性投影映射到 embed_dim
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        gh = H // P
        gw = W // P
        # [B, C, gh, P, gw, P] -> [B, gh, gw, C, P, P] -> [B, N, C*P*P]
        x = x.reshape(B, C, gh, P, gw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, gh * gw, C * P * P)
        x = self.proj(x)  # [B, N, embed_dim]
        return x


class TransformerBranch(nn.Module):
    """
    单个视角的 Transformer 编码器分支
    PatchEmbedding → Positional Encoding → TransformerEncoder → reshape 回空间网格
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3,
                 embed_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.grid_size = img_size // patch_size

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)       # [B, N, C]
        x = x + self.pos_embed
        x = self.dropout(x)
        x = x.permute(1, 0, 2)        # [N, B, C]
        x = self.transformer(x)       # [N, B, C]
        x = x.permute(1, 0, 2)        # [B, N, C]
        x = self.norm(x)
        x = x.permute(0, 2, 1)        # [B, C, N]
        x = x.reshape(B, -1, self.grid_size, self.grid_size)  # [B, C, H', W']
        return x


class TransformerBackbone(nn.Module):
    """
    三分支独立 Transformer 编码器：UAV、Satellite、Ground 三路分支
    输出特征 Map [B, C, H', W']，其中 H'=W'=img_size/patch_size
    """
    def __init__(self, img_size=256, patch_size=16, out_channels=128,
                 num_heads=4, num_layers=4, dropout=0.1, frozen=False):
        super().__init__()
        self.branch_uav = TransformerBranch(
            img_size, patch_size, 3, out_channels, num_heads, num_layers, dropout)
        self.branch_sat = TransformerBranch(
            img_size, patch_size, 3, out_channels, num_heads, num_layers, dropout)
        self.branch_ground = TransformerBranch(
            img_size, patch_size, 3, out_channels, num_heads, num_layers, dropout)

        self.frozen = frozen
        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x_uav, x_sat, x_ground):
        fuav = self.branch_uav(x_uav)
        fsat = self.branch_sat(x_sat)
        fground = self.branch_ground(x_ground)
        return fuav, fsat, fground
