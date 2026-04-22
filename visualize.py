"""可视化工具：训练曲线、评估指标、匹配结果"""
import os
import json
import glob
import argparse

import matplotlib
matplotlib.use('Agg')
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train')
    if history.get('val_loss'):
        axes[0].plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', lw=1.5, label='Val')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train')
    if history.get('val_loss'):
        axes[1].plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', lw=1.5, label='Val')
    axes[1].set(xlabel='Epoch', ylabel='Loss (log)', title='Loss (Log Scale)')
    axes[1].set_yscale('log'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if history.get('lr'):
        axes[2].plot(range(1, len(history['lr']) + 1), history['lr'], 'g-', lw=1.5)
    axes[2].set(xlabel='Epoch', ylabel='LR', title='Learning Rate')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_eval_results(eval_results, save_dir):
    tasks = {k: v for k, v in eval_results.items() if k != 'matching'}
    if not tasks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    task_names = list(tasks.keys())
    recall_keys = [k for k in list(tasks.values())[0].keys() if k.startswith('Recall')]
    x = np.arange(len(task_names))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, rk in enumerate(recall_keys):
        vals = [tasks[t][rk] for t in task_names]
        bars = axes[0].bar(x + i * 0.2, vals, 0.2, label=rk, color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{v:.1f}', ha='center', fontsize=9)
    axes[0].set_xticks(x + 0.2)
    axes[0].set_xticklabels([t.replace('_to_', ' -> ') for t in task_names], fontsize=10)
    axes[0].set(ylabel='Recall (%)', title='Recall@K', ylim=(0, 110))
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

    ap_vals = [tasks[t].get('AP', 0) for t in task_names]
    axes[1].bar(x - 0.2, ap_vals, 0.35, label='AP (%)', color='#9C27B0')
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.2, [tasks[t].get('Median_Rank', 0) for t in task_names], 0.35, label='Median Rank', color='#FF5722', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([t.replace('_to_', ' -> ') for t in task_names])
    axes[1].set(ylabel='AP (%)', title='AP & Median Rank', ylim=(0, 110))
    axes[1].legend(loc='upper left'); ax2.legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(save_dir, 'eval_results.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_summary(history, eval_results, save_dir):
    fig = plt.figure(figsize=(16, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train')
    if history.get('val_loss'):
        ax1.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', lw=1.5, label='Val')
    ax1.set(title='Loss', xlabel='Epoch', ylabel='Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train')
    if history.get('val_loss'):
        ax2.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'r-', lw=1.5, label='Val')
    ax2.set_yscale('log'); ax2.set(title='Loss (Log)', xlabel='Epoch'); ax2.legend(); ax2.grid(True, alpha=0.3)

    tasks = {k: v for k, v in eval_results.items() if k != 'matching'}
    if tasks:
        ax3 = fig.add_subplot(2, 2, 3)
        names = list(tasks.keys())
        recall_keys = [k for k in list(tasks.values())[0].keys() if k.startswith('Recall')]
        x = np.arange(len(names))
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        for i, rk in enumerate(recall_keys):
            ax3.bar(x + i * 0.2, [tasks[t][rk] for t in names], 0.2, label=rk, color=colors[i % len(colors)])
        ax3.set_xticks(x + 0.2); ax3.set_xticklabels([t.replace('_to_', '->') for t in names], fontsize=9)
        ax3.set(title='Recall@K', ylabel='%', ylim=(0, 110)); ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4); ax4.axis('off')
    lines = [f'Total Epochs: {len(history["train_loss"])}',
             f'Final Train Loss: {history["train_loss"][-1]:.6f}']
    if history.get('val_loss'):
        lines.append(f'Final Val Loss: {history["val_loss"][-1]:.6f}')
    for task, m in tasks.items():
        lines.append(f'\n[{task.replace("_to_", " -> ")}]')
        for k, v in m.items():
            lines.append(f'  {k}: {v:.2f}')
    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes, fontsize=10,
             va='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Training & Evaluation Overview', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def visualize_matching(config_path, checkpoint, img1_path, img2_path, output_dir,
                       topk=50, img_size=256, device='cpu'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    dev = torch.device(device)
    model = load_model(config, checkpoint, dev)
    img1_tensor, img1_cv = preprocess_image(img1_path, img_size)
    img2_tensor, img2_cv = preprocess_image(img2_path, img_size)
    matches, scores, sim_matrix = run_inference(model, img1_tensor, img2_tensor, dev, topk)

    feat_h = feat_w = img_size // 4
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'matching.png'),
                draw_matches_cv2(img1_cv, img2_cv, matches, scores, feat_h, feat_w, img_size))
    cv2.imwrite(os.path.join(output_dir, 'similarity.png'), similarity_heatmap(sim_matrix))
    print(f'Saved to {output_dir}/')


def main():
    parser = argparse.ArgumentParser(description='可视化训练和评估结果')
    parser.add_argument('--history', type=str, default=None)
    parser.add_argument('--eval', type=str, default=None)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--match_img1', type=str, default=None)
    parser.add_argument('--match_img2', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    history_file = args.history or find_latest_file('logs/history_*.json')
    eval_file = args.eval or find_latest_file('outputs/eval_results_*.json')
    os.makedirs(args.output, exist_ok=True)

    history, eval_results = None, None

    if history_file and os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
        plot_training_curves(history, args.output)

    if eval_file and os.path.exists(eval_file):
        with open(eval_file) as f:
            eval_results = json.load(f)
        plot_eval_results(eval_results, args.output)

    if history and eval_results:
        plot_summary(history, eval_results, args.output)

    if args.match_img1 and args.match_img2 and args.checkpoint:
        visualize_matching(args.config, args.checkpoint, args.match_img1, args.match_img2,
                           args.output, args.topk, args.img_size, args.device)

    print(f'\nDone! -> {args.output}/')


if __name__ == '__main__':
    main()
