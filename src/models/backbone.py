"""
Multi-Scale Feature Extraction Backbone: ResNet50 + FPN
提取多尺度特征 F^1, F^2, F^3，然后通过加权融合得到 F^fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPN(nn.Module):
    """
    Feature Pyramid Network
    将 ResNet 多层特征融合为统一通道数的多尺度特征
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, features):
        """
        features: list of [B, C_i, H_i, W_i] from ResNet layers
        Returns: list of [B, out_channels, H_i, W_i]
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


class ResNetFPNBackbone(nn.Module):
    """
    ResNet50 + FPN 多尺度特征提取
    输出三个尺度的特征图和加权融合后的特征
    """
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # 提取 ResNet 的各层
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 channels, stride 4
        self.layer2 = resnet.layer2  # 512 channels, stride 8
        self.layer3 = resnet.layer3  # 1024 channels, stride 16

        # FPN
        self.fpn = FPN(in_channels_list=[256, 512, 1024], out_channels=out_channels)

        # 可学习的融合权重 w_k
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            F_fusion: [B, C, H/4, W/4] 融合后的特征
            multi_scale: list of 3 tensors [B, C, H_i, W_i]
        """
        x = self.layer0(x)
        c1 = self.layer1(x)   # [B, 256, H/4, W/4]
        c2 = self.layer2(c1)  # [B, 512, H/8, W/8]
        c3 = self.layer3(c2)  # [B, 1024, H/16, W/16]

        multi_scale = self.fpn([c1, c2, c3])  # 3 x [B, out_channels, H_i, W_i]

        # 加权融合到最高分辨率
        target_size = multi_scale[0].shape[2:]
        weights = torch.softmax(self.fusion_weights, dim=0)

        F_fusion = torch.zeros_like(multi_scale[0])
        for k, feat in enumerate(multi_scale):
            feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            F_fusion = F_fusion + weights[k] * feat_resized

        return F_fusion, multi_scale
