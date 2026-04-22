"""IMTMN 训练逻辑"""
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, SequentialLR
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.losses import loss_geo, loss_match, loss_retrieval, total_loss
from src.models.imtmn import IMTMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """自定义collate：保留tensor与元信息，ground缺失时补零"""
    if not batch:
        return {}

    available_keys = set()
    for item in batch:
        if isinstance(item, dict):
            available_keys.update(item.keys())

    result = {}
    for key in sorted(available_keys):
        values = [item.get(key) if isinstance(item, dict) else None for item in batch]
        non_none = [v for v in values if v is not None]
        if not non_none:
            continue

        if all(isinstance(v, torch.Tensor) for v in non_none):
            if key == 'ground':
                ref = non_none[0]
                filled = [v if isinstance(v, torch.Tensor) else torch.zeros_like(ref) for v in values]
            else:
                if len(non_none) != len(values):
                    continue
                filled = values
            try:
                result[key] = torch.stack(filled)
            except RuntimeError as exc:
                logger.warning("Could not stack '%s': %s", key, exc)
        else:
            result[key] = values
    return result


def create_model(config, device, pretrained_backbone=True):
    cfg = config['model']
    return IMTMN(
        d_model=cfg['d_model'], num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        topk_attn=cfg['topk_attn'], topk_match=cfg['topk_match'],
        sinkhorn_iters=cfg['sinkhorn_iters'], ffn_dim=cfg['ffn_dim'], dropout=cfg['dropout'],
        use_sparse=cfg.get('use_sparse', True), transformer_size=cfg.get('transformer_size', 32),
        pretrained_backbone=pretrained_backbone,
    ).to(device)


def setup_logging(config):
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    tb_writer = None
    if config['logging'].get('tensorboard', False) and HAS_TENSORBOARD:
        tb_dir = os.path.join(log_dir, f'tb_{timestamp}')
        tb_writer = SummaryWriter(tb_dir)

    history_file = os.path.join(log_dir, f'history_{timestamp}.json')
    return tb_writer, history_file


def _compute_pair_losses(output, config):
    ml = loss_match(output['match_matrix'], output['match_scores'])
    gl = loss_geo(output['F_matrix'], output['matches'])
    rl = loss_retrieval(output['feat_i'], output['feat_j'],
                        temperature=config['loss'].get('temperature', 0.07))
    total = total_loss(ml, gl, rl,
                       w_match=config['loss']['match_weight'],
                       w_geo=config['loss']['geo_weight'],
                       w_retrieval=config['loss']['retrieval_weight'])
    stats = {
        'match_loss': ml.detach().item(),
        'geo_loss': gl.detach().item(),
        'retrieval_loss': rl.detach().item(),
        'avg_match_score': output['match_scores'].mean().detach().item(),
        'avg_matches_per_pair': float(output['matches'].shape[1]),
    }
    return total, stats


def _compute_batch_objective(model, batch, device, config):
    uav = batch['uav'].to(device)
    sat = batch['sat'].to(device)

    pair_losses = []
    aggregate = defaultdict(float)
    outputs = [model(uav, sat)]

    has_ground = (
        'ground' in batch
        and isinstance(batch['ground'], torch.Tensor)
        and float(batch['ground'].abs().sum().detach().cpu().item()) > 0.0
    )
    if has_ground:
        ground = batch['ground'].to(device)
        outputs.append(model(uav, ground))
        outputs.append(model(sat, ground))

    for output in outputs:
        pair_total, pair_stats = _compute_pair_losses(output, config)
        pair_losses.append(pair_total)
        for k, v in pair_stats.items():
            aggregate[k] += v

    total = torch.stack(pair_losses).mean()
    metrics = {k: v / len(outputs) for k, v in aggregate.items()}
    metrics['loss'] = total.detach().item()
    metrics['view_pairs_per_batch'] = float(len(outputs))
    return total, metrics


def _average_metrics(metrics_sum, num_batches):
    if num_batches == 0:
        return {}
    return {k: v / num_batches for k, v in metrics_sum.items()}


def _evaluate_pair_loader(model, data_loader, device, config):
    metrics_sum = defaultdict(float)
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if not batch or 'uav' not in batch or 'sat' not in batch:
                continue
            _, batch_metrics = _compute_batch_objective(model, batch, device, config)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            num_batches += 1
    return _average_metrics(metrics_sum, num_batches)


def _log_epoch(prefix, epoch, epochs, metrics, lr=None, elapsed=None):
    if not metrics:
        logger.warning('Epoch %d/%d - %s: no valid batches', epoch, epochs, prefix)
        return
    msg = (f"Epoch {epoch}/{epochs} - {prefix} Loss: {metrics['loss']:.4f} "
           f"(match={metrics['match_loss']:.4f}, geo={metrics['geo_loss']:.4f}, "
           f"retrieval={metrics['retrieval_loss']:.4f}, score={metrics['avg_match_score']:.4f})")
    if lr is not None:
        msg += f' - LR: {lr:.6f}'
    if elapsed is not None:
        msg += f' - Time: {elapsed:.1f}s'
    logger.info(msg)


def _write_tb(tb_writer, prefix, metrics, step):
    if tb_writer is None:
        return
    for k, v in metrics.items():
        tb_writer.add_scalar(f'{prefix}/{k}', v, step)


def _append_history(history, prefix, metrics):
    for k, v in metrics.items():
        history.setdefault(f'{prefix}_{k}', []).append(v)


def train_epoch(config, train_loader, val_loader, device, val_benchmarks=None):
    from src.eval import evaluate_model

    model = create_model(config, device, pretrained_backbone=config['model'].get('pretrained_backbone', True))
    tb_writer, history_file = setup_logging(config)
    history = {'lr': []}

    # backbone用较小学习率微调
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['training']['lr'] * 0.1},
        {'params': other_params, 'lr': config['training']['lr']},
    ], weight_decay=config['training']['weight_decay'])

    # 学习率调度：warmup + cosine
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    if config['training']['scheduler'] == 'cosine':
        if warmup_epochs > 0:
            warmup_sched = LambdaLR(optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs)
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=config['training']['epochs'] - warmup_epochs,
                eta_min=config['training'].get('eta_min', 1e-6))
            scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=config['training']['epochs'],
                eta_min=config['training'].get('eta_min', 1e-6))
    elif config['training']['scheduler'] == 'step':
        scheduler = StepLR(optimizer,
                           step_size=config['training']['scheduler_params']['step_size'],
                           gamma=config['training']['scheduler_params']['gamma'])
    else:
        scheduler = None

    amp_enabled = bool(config['training'].get('use_amp', False) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    epochs = config['training']['epochs']
    save_interval = config['checkpoint']['save_interval']
    save_dir = config['checkpoint']['save_dir']
    log_interval = config['logging']['log_interval']
    grad_accum_steps = config['training'].get('grad_accum_steps', 1)
    os.makedirs(save_dir, exist_ok=True)

    best_recall = float('-inf')
    best_val_loss = float('inf')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Device: %s | AMP: %s', device, amp_enabled)
    logger.info('Parameters: %s total, %s trainable', f'{total_params:,}', f'{trainable_params:,}')
    logger.info('Model: d_model=%d, heads=%d, layers=%d, transformer_size=%d',
                config['model']['d_model'], config['model']['num_heads'],
                config['model']['num_layers'], config['model'].get('transformer_size', 32))
    logger.info('Training: epochs=%d, lr=%.6f, warmup=%d, batch=%d, grad_accum=%d',
                epochs, config['training']['lr'], warmup_epochs,
                config['data']['batch_size'], grad_accum_steps)
    logger.info('Train: %d pairs | Val: %d pairs', len(train_loader.dataset), len(val_loader.dataset))

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        metrics_sum = defaultdict(float)
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            if not batch or 'uav' not in batch or 'sat' not in batch:
                continue

            if amp_enabled:
                with torch.amp.autocast('cuda', enabled=True):
                    loss_val, batch_metrics = _compute_batch_objective(model, batch, device, config)
                (scaler.scale(loss_val / grad_accum_steps)).backward()
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_val, batch_metrics = _compute_batch_objective(model, batch, device, config)
                (loss_val / grad_accum_steps).backward()
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0 and num_batches > 0:
                pbar.set_postfix({
                    'loss': f'{metrics_sum["loss"] / num_batches:.4f}',
                    'score': f'{metrics_sum["avg_match_score"] / num_batches:.4f}',
                })

        if scheduler:
            scheduler.step()

        train_metrics = _average_metrics(metrics_sum, num_batches)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.perf_counter() - epoch_start
        history['lr'].append(current_lr)

        if train_metrics:
            _log_epoch('Train', epoch + 1, epochs, train_metrics, lr=current_lr, elapsed=epoch_time)
            _append_history(history, 'train', train_metrics)
            _write_tb(tb_writer, 'train', train_metrics, epoch + 1)
            if tb_writer:
                tb_writer.add_scalar('train/lr', current_lr, epoch + 1)

        retrieval_summary = {}
        val_metrics = {}

        if (epoch + 1) % config['evaluation']['val_interval'] == 0:
            val_metrics = _evaluate_pair_loader(model, val_loader, device, config)
            if val_metrics:
                _log_epoch('Val', epoch + 1, epochs, val_metrics)
                _append_history(history, 'val', val_metrics)
                _write_tb(tb_writer, 'val', val_metrics, epoch + 1)

            if config['evaluation'].get('run_retrieval_on_val', True) and val_benchmarks:
                eval_results = evaluate_model(
                    model=model, pair_loader=None, benchmarks=val_benchmarks,
                    device=device, config=config, save_results=False,
                    split_name=f'val_epoch_{epoch + 1}')
                retrieval_summary = eval_results.get('summary', {})
                if retrieval_summary:
                    logger.info('Epoch %d/%d - Val Retrieval: R@1=%.2f, mAP=%.2f, MRR=%.2f',
                                epoch + 1, epochs,
                                retrieval_summary.get('macro_recall_at_1', 0.0),
                                retrieval_summary.get('macro_map', 0.0),
                                retrieval_summary.get('macro_mrr', 0.0))
                    _append_history(history, 'val_retrieval', retrieval_summary)
                    _write_tb(tb_writer, 'val_retrieval', retrieval_summary, epoch + 1)

            # 保存最优模型
            monitor_recall = retrieval_summary.get('macro_recall_at_1')
            improved = False
            ckpt = {
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recall_at_1': best_recall, 'best_val_loss': best_val_loss,
                'config': config,
            }
            if monitor_recall is not None and monitor_recall > best_recall:
                best_recall = monitor_recall
                improved = True
                ckpt['best_recall_at_1'] = best_recall
            elif val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                improved = True
                ckpt['best_val_loss'] = best_val_loss

            if config['checkpoint']['save_best'] and improved:
                path = os.path.join(save_dir, 'best_model.pth')
                torch.save(ckpt, path)
                logger.info('Saved best model -> %s', path)

        # 定期保存checkpoint
        if (epoch + 1) % save_interval == 0:
            path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'config': config}, path)
            logger.info('Saved checkpoint -> %s', path)

    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    if tb_writer:
        tb_writer.close()
    logger.info('Training completed! History saved to %s', history_file)
