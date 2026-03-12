"""
IMTMN Evaluation Script

评估指标:
1. Matching Precision / Recall
2. Retrieval: Recall@1/5/10, AP, Median Rank
3. Pose Estimation: AUC@5°/10°/20° (需 ground truth pose)
"""
import torch
import torch.nn.functional as F
import os
import json
from datetime import datetime
from tqdm import tqdm
import logging
import numpy as np

from src.models.imtmn import IMTMN
from src.losses import loss_match, loss_geo, loss_retrieval, total_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(model, dataloader, device):
    """
    提取所有样本的全局特征向量，用于检索评估
    Returns: 各模态的全局特征 [N, C]
    """
    all_uav, all_sat, all_ground = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            if not batch or 'uav' not in batch or 'sat' not in batch:
                continue

            uav = batch['uav'].to(device)
            sat = batch['sat'].to(device)

            # 提取全局特征
            uav_feat = model.extract_global_feature(uav)  # [B, C]
            sat_feat = model.extract_global_feature(sat)

            all_uav.append(uav_feat.cpu())
            all_sat.append(sat_feat.cpu())

            ground = batch.get('ground')
            if ground is not None and ground.abs().sum() > 0:
                ground = ground.to(device)
                gnd_feat = model.extract_global_feature(ground)
                all_ground.append(gnd_feat.cpu())
            else:
                all_ground.append(torch.zeros_like(uav_feat.cpu()))

    uav_feats = torch.cat(all_uav, dim=0)
    sat_feats = torch.cat(all_sat, dim=0)
    ground_feats = torch.cat(all_ground, dim=0)
    return uav_feats, sat_feats, ground_feats


def compute_recall_at_k(query_feats, gallery_feats, ks=(1, 5, 10)):
    """
    计算检索指标：Recall@K, AP, Median Rank
    """
    query_feats = F.normalize(query_feats, dim=1)
    gallery_feats = F.normalize(gallery_feats, dim=1)

    sim_matrix = query_feats @ gallery_feats.t()
    n = sim_matrix.shape[0]
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    gt = torch.arange(n).unsqueeze(1)

    results = {}
    for k in ks:
        topk = sorted_indices[:, :k]
        correct = (topk == gt).any(dim=1).float()
        results[f'Recall@{k}'] = correct.mean().item() * 100

    ranks = (sorted_indices == gt).nonzero(as_tuple=False)[:, 1].float() + 1
    results['AP'] = (1.0 / ranks).mean().item() * 100
    results['Median_Rank'] = ranks.median().item()

    return results


def compute_matching_metrics(model, dataloader, device, config):
    """
    计算匹配质量指标
    """
    total_matches = 0
    total_loss_val = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Matching evaluation'):
            if not batch or 'uav' not in batch or 'sat' not in batch:
                continue

            uav = batch['uav'].to(device)
            sat = batch['sat'].to(device)

            output = model(uav, sat)

            L_m = loss_match(output['match_matrix'], output['match_scores'])
            L_g = loss_geo(output['F_matrix'], output['matches'])
            L_r = loss_retrieval(output['feat_i'], output['feat_j'])
            L = total_loss(L_m, L_g, L_r,
                           w_match=config['loss']['match_weight'],
                           w_geo=config['loss']['geo_weight'],
                           w_retrieval=config['loss']['retrieval_weight'])

            total_matches += output['matches'].shape[1]  # K matches per batch
            total_loss_val += L.item()
            batch_count += 1

    results = {}
    if batch_count > 0:
        results = {
            'total_matches': total_matches,
            'avg_matches_per_batch': total_matches / batch_count,
            'avg_loss': total_loss_val / batch_count,
            'avg_match_loss': total_loss_val / batch_count,
        }
    return results


def evaluate(config, data_loader, device, checkpoint=None):
    """
    评估 IMTMN 模型性能
    """
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
        pretrained_backbone=False,  # 评估时从 checkpoint 加载
    ).to(device)

    # 加载检查点
    ckpt_loaded = False
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        logger.info(f'Loaded checkpoint from {checkpoint}')
        ckpt_loaded = True
    else:
        save_dir = config['checkpoint']['save_dir']
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            logger.info(f'Loaded best model from {best_model_path}')
            ckpt_loaded = True
        else:
            checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch')]
            if checkpoint_files:
                latest_ckpt = sorted(checkpoint_files)[-1]
                latest_path = os.path.join(save_dir, latest_ckpt)
                ckpt = torch.load(latest_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model'])
                logger.info(f'Loaded checkpoint from {latest_path}')
                ckpt_loaded = True

    if not ckpt_loaded:
        logger.warning('No checkpoint loaded, evaluating with random weights')

    model.eval()

    # ==================== 1. 检索评估 ====================
    logger.info('Step 1: Extracting features for retrieval evaluation...')
    uav_feats, sat_feats, ground_feats = extract_features(model, data_loader, device)

    all_results = {}
    ks = (1, 5, 10)

    logger.info('Computing UAV -> Satellite retrieval metrics...')
    uav2sat = compute_recall_at_k(uav_feats, sat_feats, ks=ks)
    all_results['UAV_to_SAT'] = uav2sat
    for k, v in uav2sat.items():
        logger.info(f'  UAV->SAT {k}: {v:.2f}')

    logger.info('Computing Satellite -> UAV retrieval metrics...')
    sat2uav = compute_recall_at_k(sat_feats, uav_feats, ks=ks)
    all_results['SAT_to_UAV'] = sat2uav
    for k, v in sat2uav.items():
        logger.info(f'  SAT->UAV {k}: {v:.2f}')

    if ground_feats.abs().sum() > 0:
        logger.info('Computing Ground -> Satellite retrieval metrics...')
        gnd2sat = compute_recall_at_k(ground_feats, sat_feats, ks=ks)
        all_results['GND_to_SAT'] = gnd2sat
        for k, v in gnd2sat.items():
            logger.info(f'  GND->SAT {k}: {v:.2f}')

    # ==================== 2. 匹配质量评估 ====================
    logger.info('Step 2: Computing matching statistics...')
    match_results = compute_matching_metrics(model, data_loader, device, config)
    if match_results:
        all_results['matching'] = match_results
        logger.info(f'  Avg Loss: {match_results["avg_loss"]:.4f}')
        logger.info(f'  Avg Matches/Batch: {match_results["avg_matches_per_batch"]:.1f}')

    # ==================== 3. 保存结果 ====================
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'eval_results_{timestamp}.json')

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f'\nResults saved to {result_file}')

    logger.info('\n' + '=' * 60)
    logger.info('IMTMN Evaluation Summary')
    logger.info('=' * 60)
    for task, metrics in all_results.items():
        if task == 'matching':
            continue
        logger.info(f'  [{task}]')
        for k, v in metrics.items():
            logger.info(f'    {k}: {v:.2f}')
    logger.info('=' * 60)

    return all_results
