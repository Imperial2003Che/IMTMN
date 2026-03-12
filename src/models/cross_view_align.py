import torch
import torch.nn as nn

class CrossViewAlign(nn.Module):
    """
    跨视角对齐模块：简单实现为通道维度拼接后经过轻量 Transformer/注意力
    输入: fuav, fsat, fground [B, C, H, W]
    输出: f_common [B, C', H, W]
    """
    def __init__(self, in_channels=128, out_channels=128, num_heads=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(in_channels * 3, out_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fuav, fsat, fground):
        # 3x channel concatenation
        x = torch.cat([fuav, fsat, fground], dim=1)  # [B, 3C, H, W]
        B, C3, H, W = x.shape
        # [B, 3C, H, W] -> [B, HW, 3C] -> Linear -> [B, HW, C]
        x = x.view(B, C3, H * W).permute(0, 2, 1)  # [B, HW, 3C]
        x = self.proj(x)  # [B, HW, C]
        C = x.shape[-1]
        # [B, HW, C] -> [HW, B, C] for transformer
        x_flat = x.permute(1, 0, 2)  # [HW, B, C]
        x_enc = self.transformer(x_flat)  # [HW, B, C]
        x_enc = x_enc.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return x_enc