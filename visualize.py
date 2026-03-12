"""
IMTMN 可视化工具

功能:
  1. 训练曲线可视化 (Loss, LR)
  2. 评估指标可视化 (Recall@K, AP, Median Rank)
  3. 匹配结果可视化 (drawMatches, similarity heatmap)
  4. 综合概览图

用法：
    python3 visualize.py                          # 自动找最新的日志和结果
    python3 visualize.py --history logs/history_xxx.json --eval outputs/eval_results_xxx.json
    python3 visualize.py --match_img1 img1.jpg --match_img2 img2.jpg --checkpoint checkpoints/best_model.pth
"""
import os
import json
import glob
import argparse
import matplotlib
matplotlib.use('Agg')  # 无 GUI 模式
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import yaml

from src.inference import load_model, preprocess_image, run_inference
from src.visualization_utils import draw_matches_cv2, similarity_heatmap


def find_latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def plot_training_curves(history, save_dir):
    """绘制训练曲线：loss、lr"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # --- Loss 曲线 ---
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    if history.get('val_loss'):
        val_epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(val_epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Loss 对数坐标 ---
    ax = axes[1]
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    if history.get('val_loss'):
        val_epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(val_epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log)', fontsize=12)
    ax.set_title('Loss (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- 学习率 ---
    ax = axes[2]
    if history.get('lr'):
        lr_epochs = range(1, len(history['lr']) + 1)
        ax.plot(lr_epochs, history['lr'], 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[✓] 训练曲线已保存: {path}')


def plot_eval_results(eval_results, save_dir):
    """绘制评估结果：Recall@K 柱状图"""
    retrieval_tasks = {k: v for k, v in eval_results.items() if k != 'matching'}

    if not retrieval_tasks:
        print('[!] 没有检索指标数据，跳过')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Recall@K 分组柱状图 ---
    ax = axes[0]
    tasks = list(retrieval_tasks.keys())
    recall_keys = [k for k in list(retrieval_tasks.values())[0].keys() if k.startswith('Recall')]

    x = np.arange(len(tasks))
    width = 0.2
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, rk in enumerate(recall_keys):
        values = [retrieval_tasks[t][rk] for t in tasks]
        bars = ax.bar(x + i * width, values, width, label=rk, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.replace('_to_', ' → ') for t in tasks], fontsize=10)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall@K by Retrieval Direction', fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # --- AP 和 Median Rank ---
    ax = axes[1]
    ap_values = [retrieval_tasks[t].get('AP', 0) for t in tasks]
    median_ranks = [retrieval_tasks[t].get('Median_Rank', 0) for t in tasks]

    x = np.arange(len(tasks))
    bars1 = ax.bar(x - 0.2, ap_values, 0.35, label='AP (%)', color='#9C27B0')
    for bar, val in zip(bars1, ap_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.2, median_ranks, 0.35, label='Median Rank', color='#FF5722', alpha=0.7)
    for bar, val in zip(bars2, median_ranks):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_to_', ' → ') for t in tasks], fontsize=10)
    ax.set_ylabel('AP (%)', fontsize=12, color='#9C27B0')
    ax2.set_ylabel('Median Rank', fontsize=12, color='#FF5722')
    ax.set_title('AP & Median Rank', fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'eval_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[✓] 评估结果图已保存: {path}')


def plot_summary(history, eval_results, save_dir):
    """生成综合概览图"""
    fig = plt.figure(figsize=(16, 10))

    # 上半部分：训练曲线
    ax1 = fig.add_subplot(2, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train')
    if history.get('val_loss'):
        val_epochs = range(1, len(history['val_loss']) + 1)
        ax1.plot(val_epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val')
    ax1.set_title('Loss Curve', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train')
    if history.get('val_loss'):
        val_epochs = range(1, len(history['val_loss']) + 1)
        ax2.plot(val_epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val')
    ax2.set_yscale('log')
    ax2.set_title('Loss (Log Scale)', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 下半部分：评估指标
    retrieval_tasks = {k: v for k, v in eval_results.items() if k != 'matching'}
    if retrieval_tasks:
        ax3 = fig.add_subplot(2, 2, 3)
        tasks = list(retrieval_tasks.keys())
        recall_keys = [k for k in list(retrieval_tasks.values())[0].keys() if k.startswith('Recall')]
        x = np.arange(len(tasks))
        width = 0.2
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        for i, rk in enumerate(recall_keys):
            values = [retrieval_tasks[t][rk] for t in tasks]
            ax3.bar(x + i * width, values, width, label=rk, color=colors[i % len(colors)])
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([t.replace('_to_', '→') for t in tasks], fontsize=9)
        ax3.set_title('Recall@K', fontsize=13)
        ax3.set_ylabel('%')
        ax3.set_ylim(0, 110)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

    # 右下：文字总结
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    summary_lines = ['Model Summary', '─' * 30]
    summary_lines.append(f'Total Epochs: {len(history["train_loss"])}')
    summary_lines.append(f'Final Train Loss: {history["train_loss"][-1]:.6f}')
    if history.get('val_loss'):
        summary_lines.append(f'Final Val Loss: {history["val_loss"][-1]:.6f}')
    summary_lines.append('')
    for task, metrics in retrieval_tasks.items():
        summary_lines.append(f'[{task.replace("_to_", " → ")}]')
        for k, v in metrics.items():
            summary_lines.append(f'  {k}: {v:.2f}')
    if 'matching' in eval_results:
        summary_lines.append('')
        summary_lines.append(f'Avg Loss: {eval_results["matching"]["avg_loss"]:.6f}')
        summary_lines.append(f'Avg Matches/Batch: {eval_results["matching"]["avg_matches_per_batch"]:.1f}')

    ax4.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Training & Evaluation Overview', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[✓] 综合概览图已保存: {path}')


def visualize_matching(config_path, checkpoint, img1_path, img2_path, output_dir,
                       topk=50, img_size=256, device='cpu'):
    """可视化跨视角匹配结果 (drawMatches + similarity heatmap)"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dev = torch.device(device)
    model = load_model(config, checkpoint, dev)

    img1_tensor, img1_cv = preprocess_image(img1_path, img_size)
    img2_tensor, img2_cv = preprocess_image(img2_path, img_size)

    matches, scores, sim_matrix = run_inference(model, img1_tensor, img2_tensor, dev, topk)

    feat_h = img_size // 4
    feat_w = img_size // 4

    match_vis = draw_matches_cv2(img1_cv, img2_cv, matches, scores, feat_h, feat_w, img_size)
    heatmap = similarity_heatmap(sim_matrix)

    os.makedirs(output_dir, exist_ok=True)
    match_path = os.path.join(output_dir, 'matching_drawmatches.png')
    heatmap_path = os.path.join(output_dir, 'matching_similarity.png')

    cv2.imwrite(match_path, match_vis)
    cv2.imwrite(heatmap_path, heatmap)

    print(f'[✓] 匹配可视化已保存: {match_path}')
    print(f'[✓] 相似度热力图已保存: {heatmap_path}')


def main():
    parser = argparse.ArgumentParser(description='可视化训练和评估结果')
    parser.add_argument('--history', type=str, default=None, help='训练历史 JSON 文件路径')
    parser.add_argument('--eval', type=str, default=None, help='评估结果 JSON 文件路径')
    parser.add_argument('--output', type=str, default='outputs', help='图表保存目录')

    # 匹配可视化参数（可选）
    parser.add_argument('--match_img1', type=str, default=None, help='匹配可视化图像1路径')
    parser.add_argument('--match_img2', type=str, default=None, help='匹配可视化图像2路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='IMTMN 模型权重路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--topk', type=int, default=50, help='匹配可视化 top-k')
    parser.add_argument('--img_size', type=int, default=256, help='推理图像尺寸')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    args = parser.parse_args()

    # 自动查找最新文件
    history_file = args.history or find_latest_file('logs/history_*.json')
    eval_file = args.eval or find_latest_file('outputs/eval_results_*.json')

    os.makedirs(args.output, exist_ok=True)

    history = None
    eval_results = None

    if history_file and os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f'加载训练历史: {history_file}')
        plot_training_curves(history, args.output)
    else:
        print('[!] 未找到训练历史文件')

    if eval_file and os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
        print(f'加载评估结果: {eval_file}')
        plot_eval_results(eval_results, args.output)
    else:
        print('[!] 未找到评估结果文件')

    if history and eval_results:
        plot_summary(history, eval_results, args.output)

    # 匹配可视化（当参数完整时执行）
    if args.match_img1 and args.match_img2 and args.checkpoint:
        visualize_matching(
            config_path=args.config,
            checkpoint=args.checkpoint,
            img1_path=args.match_img1,
            img2_path=args.match_img2,
            output_dir=args.output,
            topk=args.topk,
            img_size=args.img_size,
            device=args.device,
        )
    elif args.match_img1 or args.match_img2 or args.checkpoint:
        print('[!] 若要启用匹配可视化，请同时提供 --match_img1 --match_img2 --checkpoint')

    print('\n完成！图表已保存到:', args.output)


if __name__ == '__main__':
    main()
