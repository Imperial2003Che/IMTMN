import torch
import torch.nn as nn

class MatchHead(nn.Module):
    """
    稀疏匹配头：在 F_common 上预测热力图和偏移，用 Top-K 选点作为匹配对
    输入: f_common [B, C, H, W]
    输出: matches: Tensor [B*topk, 5] (x_src, y_src, x_dst, y_dst, score)
    """
    def __init__(self, in_channels=128, topk=10):
        super().__init__()
        self.heat = nn.Linear(in_channels, 1)
        self.offset = nn.Linear(in_channels, 2)
        self.topk = topk

    def forward(self, f_common):
        """
        Args:
            f_common: [B, C, H, W]
        Returns:
            matches: [B*topk, 5] tensor with (x_src, y_src, x_dst, y_dst, score)
        """
        B, C, H, W = f_common.shape
        
        # [B, C, H, W] -> [B, HW, C]
        x = f_common.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        
        # 生成热力图 [B, HW, 1] -> [B, H, W]
        heat = self.heat(x).squeeze(-1).view(B, H, W)  # [B, H, W]
        
        # 生成偏移 [B, HW, 2] -> [B, 2, H, W]
        offset = self.offset(x).permute(0, 2, 1).view(B, 2, H, W)  # [B, 2, H, W]
        
        # 展平热力图 [B, H*W]
        heat_flat = heat.view(B, -1)
        scores, idx = heat_flat.topk(min(self.topk, H * W), dim=1)  # [B, topk]
        
        # 从索引计算 (x, y) 坐标
        y_coords = idx // W
        x_coords = idx % W
        
        # 从 offset map 获取目标坐标偏移
        offset_flat = offset.view(B, 2, -1)  # [B, 2, H*W]
        
        # 构建匹配对列表
        matches_list = []
        for b in range(B):
            for k in range(scores.shape[1]):
                x_src = x_coords[b, k].float()
                y_src = y_coords[b, k].float()
                
                # 从偏移获取目标坐标
                offset_idx = idx[b, k]
                dx = offset_flat[b, 0, offset_idx]
                dy = offset_flat[b, 1, offset_idx]
                
                x_dst = x_src + dx
                y_dst = y_src + dy
                
                score = scores[b, k]
                
                # [x_src, y_src, x_dst, y_dst, score]
                # 使用 detach() 避免 requires_grad 警告
                match = torch.stack([
                    x_src.detach(),
                    y_src.detach(),
                    x_dst.detach(),
                    y_dst.detach(),
                    score
                ])
                matches_list.append(match)
        
        # 返回张量 [B*topk, 5]
        if matches_list:
            matches = torch.stack(matches_list)
        else:
            matches = torch.zeros((0, 5), dtype=torch.float32, device=f_common.device)
        
        return matches
