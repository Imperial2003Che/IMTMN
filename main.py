"""
IMTMN: Improved Multi-View Transformer Matching Network
主入口文件

Research on Multi-View Image Matching Method Based on Improved Transformer
"""
import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sues_dataset import CrossViewDataset, CombinedDataset
from data.transforms import get_common_transforms
from src.train import train_epoch, custom_collate_fn
from src.eval import evaluate

def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_dataloaders(config):
    root_dir = config['data']['root_dir']
    datasets = config['data']['datasets']
    image_size = config['data']['image_size']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    transform = get_common_transforms(size=image_size)
    
    train_dataset = CombinedDataset(root_dir, datasets, split='train', transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_dataset = CombinedDataset(root_dir, datasets, split='val', transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_dataset = CombinedDataset(root_dir, datasets, split='test', transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

def setup_directories(config):
    dirs = [
        config['checkpoint']['save_dir'],
        config['output']['output_dir'],
        config['logging']['log_dir']
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Cross-View Alignment and Geometry Estimation')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'], default='train', help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载的检查点路径')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_directories(config)
    
    import random
    import numpy as np
    seed = config['misc']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(config['misc']['device'] if torch.cuda.is_available() else 'cpu')
    
    if args.mode == 'train':
        train_loader, val_loader, _ = create_dataloaders(config)
        train_epoch(config, train_loader, val_loader, device)
    
    elif args.mode == 'eval':
        _, val_loader, _ = create_dataloaders(config)
        evaluate(config, val_loader, device, checkpoint=args.checkpoint)
    
    elif args.mode == 'test':
        _, _, test_loader = create_dataloaders(config)
        evaluate(config, test_loader, device, checkpoint=args.checkpoint)

if __name__ == '__main__':
    main()
