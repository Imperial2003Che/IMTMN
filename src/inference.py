"""
IMTMN Inference + Visualization

推理 Pipeline:
  1. 输入两张图像 (不同视角)
  2. CNN Feature Extraction (ResNet50 + FPN)
  3. Multi-View Transformer (SA → CA → FFN + GeometricAttention)
  4. Similarity Matrix: S = F1 · F2^T
  5. Optimal Transport (Sinkhorn) → 匹配矩阵
  6. Top-K Matches → 关键点对应
  7. OpenCV drawMatches 可视化

Usage:
    python -m src.inference \
        --img1 path/to/image1.jpg \
        --img2 path/to/image2.jpg \
        --checkpoint checkpoints/best_model.pth \
        --output outputs/match_result.png \
        --topk 50
"""
import argparse
import os
import cv2
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models.imtmn import IMTMN
from src.visualization_utils import draw_matches_cv2, similarity_heatmap


def load_model(config, checkpoint_path, device):
    """加载训练好的 IMTMN 模型"""
    model_cfg = config['model']
    model = IMTMN(
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_layers=model_cfg['num_layers'],
        topk_attn=model_cfg['topk_attn'],
        topk_match=model_cfg['topk_match'],
        sinkhorn_iters=model_cfg['sinkhorn_iters'],
        ffn_dim=model_cfg['ffn_dim'],
        dropout=model_cfg['dropout'],
        use_sparse=model_cfg.get('use_sparse', True),
        transformer_size=model_cfg.get('transformer_size', 32),
        pretrained_backbone=False,  # 从 checkpoint 加载权重
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def preprocess_image(img_path, img_size=256):
    """
    预处理图像

    Args:
        img_path: 图像路径
        img_size: 目标尺寸
    Returns:
        tensor: [1, 3, H, W] 归一化后的图像张量
        img_cv: OpenCV BGR 格式原图 (用于可视化)
    """
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_pil = Image.open(img_path).convert('RGB')
    tensor = transform(img_pil).unsqueeze(0)  # [1, 3, H, W]

    # 为可视化准备统一尺寸的 OpenCV 图像
    img_cv_resized = cv2.resize(img_cv, (img_size, img_size))

    return tensor, img_cv_resized


@torch.no_grad()
def run_inference(model, img1_tensor, img2_tensor, device, topk=50):
    """
    运行推理

    Pipeline:
      img1, img2
        → CNN Feature Extraction
        → Multi-View Transformer (SA → CA → FFN)
        → Geometric-Aware Attention
        → Similarity Matrix S = F1·F2^T
        → Sinkhorn → Top-K matches

    Args:
        model: IMTMN 模型
        img1_tensor: [1, 3, H, W]
        img2_tensor: [1, 3, H, W]
        device: 设备
        topk: 可视化的匹配点数
    Returns:
        matches: [K, 4] numpy 匹配坐标 (y1, x1, y2, x2)
        scores: [K] numpy 匹配分数
        sim_matrix: [N, N] numpy 相似度矩阵
    """
    img1 = img1_tensor.to(device)
    img2 = img2_tensor.to(device)

    output = model(img1, img2)

    matches = output['matches'][0].cpu().numpy()     # [K, 4]
    scores = output['match_scores'][0].cpu().numpy()  # [K]
    sim_matrix = output['sim_matrix'][0].cpu().numpy() # [N, N]

    # 只取前 topk 个
    k = min(topk, len(scores))
    matches = matches[:k]
    scores = scores[:k]

    return matches, scores, sim_matrix


def main():
    parser = argparse.ArgumentParser(description='IMTMN Inference & Visualization')
    parser.add_argument('--img1', type=str, required=True, help='视角A图像路径')
    parser.add_argument('--img2', type=str, required=True, help='视角B图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件')
    parser.add_argument('--output', type=str, default='outputs/match_result.png', help='输出路径')
    parser.add_argument('--topk', type=int, default=50, help='可视化的匹配点数')
    parser.add_argument('--img_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    # 加载模型
    model = load_model(config, args.checkpoint, device)

    # 预处理图像
    img1_tensor, img1_cv = preprocess_image(args.img1, args.img_size)
    img2_tensor, img2_cv = preprocess_image(args.img2, args.img_size)

    # 推理
    matches, scores, sim_matrix = run_inference(model, img1_tensor, img2_tensor, device, args.topk)

    # 计算特征图尺寸 (ResNet50 + FPN, stride=4)
    feat_h = args.img_size // 4
    feat_w = args.img_size // 4

    print(f"Found {len(matches)} matches")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Feature map size: {feat_h}x{feat_w}")

    # 可视化匹配结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    match_img = draw_matches_cv2(
        img1_cv, img2_cv, matches, scores,
        feat_h, feat_w, args.img_size
    )
    cv2.imwrite(args.output, match_img)
    print(f"Match visualization saved to {args.output}")

    # 可视化相似度矩阵
    sim_output = args.output.replace('.png', '_similarity.png').replace('.jpg', '_similarity.png')
    sim_heatmap = similarity_heatmap(sim_matrix)
    cv2.imwrite(sim_output, sim_heatmap)
    print(f"Similarity heatmap saved to {sim_output}")


if __name__ == '__main__':
    main()
