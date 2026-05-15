"""
IMTMN 主入口
基于改进Transformer的多视角图像匹配方法研究
"""
import os
import sys
import yaml
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.datasets import CombinedDataset, build_retrieval_benchmarks
from data.transforms import get_train_transforms, get_eval_transforms
from src.train import train_epoch, custom_collate_fn
from src.eval import evaluate


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    root_dir = config['data']['root_dir']
    datasets = config['data']['datasets']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    dataset_settings = config['data'].get('dataset_settings', {})
    seed = config['misc']['seed']
    eval_batch_size = config['evaluation'].get('batch_size', batch_size)

    train_transform = get_train_transforms(size=image_size)
    eval_transform = get_eval_transforms(size=image_size)

    train_dataset = CombinedDataset(
        root_dir, datasets, split='train',
        transform=train_transform, dataset_settings=dataset_settings, seed=seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn,
    )

    val_dataset = CombinedDataset(
        root_dir, datasets, split='val',
        transform=eval_transform, dataset_settings=dataset_settings, seed=seed,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn,
    )

    test_dataset = CombinedDataset(
        root_dir, datasets, split='test',
        transform=eval_transform, dataset_settings=dataset_settings, seed=seed,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, collate_fn=custom_collate_fn,
    )

    include_reverse = config['evaluation'].get('include_reverse', False)
    benchmark_kwargs = dict(
        root_dir=root_dir, datasets_list=datasets, transform=eval_transform,
        dataset_settings=dataset_settings, seed=seed, include_reverse=include_reverse,
    )
    val_benchmarks = build_retrieval_benchmarks(split='val', **benchmark_kwargs)
    test_benchmarks = build_retrieval_benchmarks(split='test', **benchmark_kwargs)

    return train_loader, val_loader, test_loader, val_benchmarks, test_benchmarks


def setup_directories(config):
    for d in [config['checkpoint']['save_dir'], config['output']['output_dir'], config['logging']['log_dir']]:
        os.makedirs(d, exist_ok=True)


def resolve_device(config):
    requested = str(config['misc'].get('device', 'auto')).lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if requested.startswith('cuda') and torch.cuda.is_available():
        return torch.device(requested)
    if requested == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        print('Warning: MPS不可用，回退到CPU')
        return torch.device('cpu')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='IMTMN 跨视角匹配训练与评估')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'], default='train')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_directories(config)

    # 固定随机种子
    seed = config['misc']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(config)

    if args.mode == 'train':
        train_loader, val_loader, _, val_benchmarks, _ = create_dataloaders(config)
        train_epoch(config, train_loader, val_loader, device, val_benchmarks=val_benchmarks)
    elif args.mode == 'eval':
        _, val_loader, _, val_benchmarks, _ = create_dataloaders(config)
        evaluate(config, val_loader, val_benchmarks, device, checkpoint=args.checkpoint, split_name='val')
    elif args.mode == 'test':
        _, _, test_loader, _, test_benchmarks = create_dataloaders(config)
        evaluate(config, test_loader, test_benchmarks, device, checkpoint=args.checkpoint, split_name='test')


if __name__ == '__main__':
    main()
