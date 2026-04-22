"""ResNet50 + FPN 多尺度特征提取"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FPN(nn.Module):
    """Feature Pyramid Network，将多层特征统一到相同通道数"""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False)
        return [conv(lat) for conv, lat in zip(self.output_convs, laterals)]


class ResNetFPNBackbone(nn.Module):
    """ResNet50 + FPN，输出三个尺度的特征图和加权融合特征"""

    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256ch, stride 4
        self.layer2 = resnet.layer2  # 512ch, stride 8
        self.layer3 = resnet.layer3  # 1024ch, stride 16
        self.fpn = FPN([256, 512, 1024], out_channels)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        multi_scale = self.fpn([c1, c2, c3])

        # 可学习权重加权融合
        target_size = multi_scale[0].shape[2:]
        weights = torch.softmax(self.fusion_weights, dim=0)
        F_fusion = sum(
            weights[k] * F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            for k, feat in enumerate(multi_scale)
        )
        return F_fusion, multi_scale
