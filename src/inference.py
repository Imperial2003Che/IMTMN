"""IMTMN 推理脚本：输入图像对，输出匹配可视化"""
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
    cfg = config['model']
    model = IMTMN(
        d_model=cfg['d_model'], num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        topk_attn=cfg['topk_attn'], topk_match=cfg['topk_match'],
        sinkhorn_iters=cfg['sinkhorn_iters'], ffn_dim=cfg['ffn_dim'], dropout=cfg['dropout'],
        use_sparse=cfg.get('use_sparse', True), transformer_size=cfg.get('transformer_size', 32),
        pretrained_backbone=False,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def preprocess_image(img_path, img_size=256):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.open(img_path).convert('RGB')
    tensor = transform(img_pil).unsqueeze(0)
    img_cv_resized = cv2.resize(img_cv, (img_size, img_size))
    return tensor, img_cv_resized


@torch.no_grad()
def run_inference(model, img1_tensor, img2_tensor, device, topk=50):
    output = model(img1_tensor.to(device), img2_tensor.to(device))
    matches = output['matches'][0].cpu().numpy()
    scores = output['match_scores'][0].cpu().numpy()
    sim_matrix = output['sim_matrix'][0].cpu().numpy()
    k = min(topk, len(scores))
    return matches[:k], scores[:k], sim_matrix


def main():
    parser = argparse.ArgumentParser(description='IMTMN 推理与可视化')
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output', type=str, default='outputs/match_result.png')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device)

    img1_tensor, img1_cv = preprocess_image(args.img1, args.img_size)
    img2_tensor, img2_cv = preprocess_image(args.img2, args.img_size)
    matches, scores, sim_matrix = run_inference(model, img1_tensor, img2_tensor, device, args.topk)

    feat_h = feat_w = config['model'].get('transformer_size', args.img_size // 4)
    print(f"Found {len(matches)} matches, score range: [{scores.min():.4f}, {scores.max():.4f}]")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, draw_matches_cv2(img1_cv, img2_cv, matches, scores, feat_h, feat_w, args.img_size))
    print(f"Match visualization -> {args.output}")

    sim_output = args.output.replace('.png', '_similarity.png').replace('.jpg', '_similarity.png')
    cv2.imwrite(sim_output, similarity_heatmap(sim_matrix))
    print(f"Similarity heatmap -> {sim_output}")


if __name__ == '__main__':
    main()
