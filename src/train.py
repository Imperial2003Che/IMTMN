"""
IMTMN Training Script

训练 Improved Multi-View Transformer Matching Network
支持多视角图像对的特征匹配训练
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import json
from datetime import datetime
from tqdm import tqdm
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.models.imtmn import IMTMN
from src.losses import loss_match, loss_geo, loss_retrieval, total_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_collate_fn(batch):
    """
    自定义 collate 函数，处理缺失的模态和混合类型
    某些样本可能没有 'ground' 模态（如 SUES-200）
    """
    if not batch:
        return {}
    
    # 获取所有可用的键
    available_keys = set()
    for item in batch:
        if isinstance(item, dict):
            available_keys.update(item.keys())
    
    result = {}
    
    for key in available_keys:
        items_with_key = []
        
        for item in batch:
            if not isinstance(item, dict):
                continue
                
            if key in item:
                val = item[key]
                # 只收集 tensor 类型的数据
                if isinstance(val, torch.Tensor):
                    items_with_key.append(val)
            elif key == 'ground':
                # 对于缺失的 ground，创建一个零张量
                # 使用同一 batch 中第一个有 ground 的样本的形状作为参考
                ref_shape = None
                for ref_item in batch:
                    if isinstance(ref_item, dict) and 'ground' in ref_item:
                        ref_shape = ref_item['ground'].shape
                        break
                
                if ref_shape is None:
                    # 如果没有参考，假设与 uav 或 sat 相同
                    if 'uav' in item and isinstance(item['uav'], torch.Tensor):
                        ref_shape = item['uav'].shape
                    elif 'sat' in item and isinstance(item['sat'], torch.Tensor):
                        ref_shape = item['sat'].shape
                    else:
                        ref_shape = (3, 256, 256)  # 默认形状
                
                items_with_key.append(torch.zeros(ref_shape))
        
        # 只有当 items_with_key 中都是 tensor 时才进行 stack
        if items_with_key and all(isinstance(x, torch.Tensor) for x in items_with_key):
            try:
                result[key] = torch.stack(items_with_key)
            except RuntimeError as e:
                logger.warning(f"Could not stack tensors for key '{key}': {e}")
    
    return result

def create_model(config, device):
    """创建 IMTMN 模型"""
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
        pretrained_backbone=model_cfg['pretrained_backbone'],
    ).to(device)

    return model

def setup_logging(config):
    """设置文件日志和 TensorBoard"""
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # 文件日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # TensorBoard
    tb_writer = None
    if config['logging'].get('tensorboard', False) and HAS_TENSORBOARD:
        tb_dir = os.path.join(log_dir, f'tb_{timestamp}')
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f'TensorBoard 日志目录: {tb_dir}')
    
    # 训练记录（JSON）
    history_file = os.path.join(log_dir, f'history_{timestamp}.json')
    
    logger.info(f'日志文件: {log_file}')
    return tb_writer, history_file

def train_epoch(config, train_loader, val_loader, device):
    """IMTMN 完整训练流程"""

    model = create_model(config, device)

    # 日志
    tb_writer, history_file = setup_logging(config)
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=1e-6)
    elif config['training']['scheduler'] == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config['training']['scheduler_params']['step_size'],
            gamma=config['training']['scheduler_params']['gamma']
        )
    else:
        scheduler = None

    use_amp = config['training']['use_amp']
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    epochs = config['training']['epochs']
    save_interval = config['checkpoint']['save_interval']
    save_dir = config['checkpoint']['save_dir']
    log_interval = config['logging']['log_interval']
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'IMTMN Model Parameters: {total_params:,} total, {trainable_params:,} trainable')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, batch in enumerate(pbar):
            if not batch or 'uav' not in batch or 'sat' not in batch:
                logger.warning(f"Skipping batch {batch_idx}: missing required keys")
                continue

            uav = batch['uav'].to(device)
            sat = batch['sat'].to(device)

            autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(autocast_device, enabled=use_amp):
                # 前向传播: UAV ↔ SAT
                output = model(uav, sat)

                # 计算损失
                L_m = loss_match(output['match_matrix'], output['match_scores'])
                L_g = loss_geo(output['F_matrix'], output['matches'])
                L_r = loss_retrieval(output['feat_i'], output['feat_j'],
                                     temperature=config['loss'].get('temperature', 0.07))
                L = total_loss(L_m, L_g, L_r,
                              w_match=config['loss']['match_weight'],
                              w_geo=config['loss']['geo_weight'],
                              w_retrieval=config['loss']['retrieval_weight'])

                # 如果有 ground 模态，额外计算 UAV↔Ground 和 SAT↔Ground
                has_ground = 'ground' in batch and batch['ground'].abs().sum() > 0
                if has_ground:
                    ground = batch['ground'].to(device)
                    out_ug = model(uav, ground)
                    out_sg = model(sat, ground)

                    for extra_out in [out_ug, out_sg]:
                        L_m2 = loss_match(extra_out['match_matrix'], extra_out['match_scores'])
                        L_g2 = loss_geo(extra_out['F_matrix'], extra_out['matches'])
                        L_r2 = loss_retrieval(extra_out['feat_i'], extra_out['feat_j'],
                                              temperature=config['loss'].get('temperature', 0.07))
                        L = L + total_loss(L_m2, L_g2, L_r2,
                                          w_match=config['loss']['match_weight'],
                                          w_geo=config['loss']['geo_weight'],
                                          w_retrieval=config['loss']['retrieval_weight'])
                    L = L / 3.0  # 平均多个视角对

            optimizer.zero_grad()
            scaler.scale(L).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += L.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

        if scheduler:
            scheduler.step()

        # Epoch 总结
        current_lr = optimizer.param_groups[0]['lr']
        if num_batches > 0:
            train_loss = epoch_loss / num_batches
            logger.info(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f} - LR: {current_lr:.6f}')
            history['train_loss'].append(train_loss)
            history['lr'].append(current_lr)
            if tb_writer:
                tb_writer.add_scalar('Loss/train', train_loss, epoch + 1)
                tb_writer.add_scalar('LR', current_lr, epoch + 1)

        # 验证
        if (epoch + 1) % config['evaluation']['val_interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    if not batch or 'uav' not in batch or 'sat' not in batch:
                        continue

                    uav = batch['uav'].to(device)
                    sat = batch['sat'].to(device)

                    output = model(uav, sat)
                    L_m = loss_match(output['match_matrix'], output['match_scores'])
                    L_g = loss_geo(output['F_matrix'], output['matches'])
                    L_r = loss_retrieval(output['feat_i'], output['feat_j'],
                                         temperature=config['loss'].get('temperature', 0.07))
                    L = total_loss(L_m, L_g, L_r,
                                  w_match=config['loss']['match_weight'],
                                  w_geo=config['loss']['geo_weight'],
                                  w_retrieval=config['loss']['retrieval_weight'])

                    val_loss += L.item()
                    val_batches += 1

            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                logger.info(f'Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}')
                history['val_loss'].append(avg_val_loss)
                if tb_writer:
                    tb_writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)

                if config['checkpoint']['save_best'] and avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                    }, checkpoint_path)
                    logger.info(f'Saved best model to {checkpoint_path}')
            else:
                logger.warning(f'Epoch {epoch+1} - No valid validation batches')

        # 定期保存
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')

    # 保存训练历史
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    if tb_writer:
        tb_writer.close()

    logger.info('Training completed.')
    logger.info(f'训练历史已保存到 {history_file}')
    
    if tb_writer:
        tb_writer.close()
    
    logger.info('Training completed!')
