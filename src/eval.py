"""IMTMN 评估脚本：检索指标(Recall@K/mAP/MRR) + 匹配统计"""
import json
import logging
import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import loss_geo, loss_match, loss_retrieval, total_loss
from src.models.imtmn import IMTMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_model(config, device):
    cfg = config['model']
    return IMTMN(
        d_model=cfg['d_model'], num_heads=cfg['num_heads'], num_layers=cfg['num_layers'],
        topk_attn=cfg['topk_attn'], topk_match=cfg['topk_match'],
        sinkhorn_iters=cfg['sinkhorn_iters'], ffn_dim=cfg['ffn_dim'], dropout=cfg['dropout'],
        use_sparse=cfg.get('use_sparse', True), transformer_size=cfg.get('transformer_size', 32),
        pretrained_backbone=False,
    ).to(device)


def _compute_pair_losses(output, config):
    ml = loss_match(output['match_matrix'], output['match_scores'])
    gl = loss_geo(output['F_matrix'], output['matches'])
    rl = loss_retrieval(output['feat_i'], output['feat_j'],
                        temperature=config['loss'].get('temperature', 0.07))
    t = total_loss(ml, gl, rl,
                   w_match=config['loss']['match_weight'],
                   w_geo=config['loss']['geo_weight'],
                   w_retrieval=config['loss']['retrieval_weight'])
    return t, {
        'loss': t.detach().item(),
        'match_loss': ml.detach().item(),
        'geo_loss': gl.detach().item(),
        'retrieval_loss': rl.detach().item(),
        'avg_match_score': output['match_scores'].mean().detach().item(),
        'avg_matches_per_pair': float(output['matches'].shape[1]),
    }


def _compute_matching_metrics(model, pair_loader, device, config):
    if pair_loader is None:
        return {}
    metrics_sum = defaultdict(float)
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(pair_loader, desc='Matching evaluation'):
            if not batch or 'uav' not in batch or 'sat' not in batch:
                continue
            output = model(batch['uav'].to(device), batch['sat'].to(device))
            _, metrics = _compute_pair_losses(output, config)
            for k, v in metrics.items():
                metrics_sum[k] += v
            num_batches += 1
    if num_batches == 0:
        return {}
    return {k: v / num_batches for k, v in metrics_sum.items()}


def _extract_dataset_features(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    all_features, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting features', leave=False):
            features = model.extract_global_feature(batch['image'].to(device))
            all_features.append(features.detach().cpu())
            all_labels.extend(batch['label'])
    if not all_features:
        return torch.empty(0, model.backbone.fpn.output_convs[0].out_channels), []
    return torch.cat(all_features, dim=0), all_labels


def _encode_labels(query_labels, gallery_labels):
    label_to_idx = {}
    next_idx = 0

    def _convert(labels):
        nonlocal next_idx
        encoded = []
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = next_idx
                next_idx += 1
            encoded.append(label_to_idx[label])
        return torch.tensor(encoded, dtype=torch.long)

    return _convert(query_labels), _convert(gallery_labels)


def compute_retrieval_metrics(query_feats, query_labels, gallery_feats, gallery_labels, ks=(1, 5, 10)):
    if query_feats.numel() == 0 or gallery_feats.numel() == 0:
        return {}

    query_feats = F.normalize(query_feats.float(), dim=1)
    gallery_feats = F.normalize(gallery_feats.float(), dim=1)
    query_ids, gallery_ids = _encode_labels(query_labels, gallery_labels)

    sim_matrix = query_feats @ gallery_feats.t()
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)

    recalls = {k: [] for k in ks}
    aps, rrs, best_ranks, pos_counts = [], [], [], []

    for qi in range(sorted_indices.shape[0]):
        ranked_labels = gallery_ids[sorted_indices[qi]]
        positives = (ranked_labels == query_ids[qi]).nonzero(as_tuple=False).flatten()
        if positives.numel() == 0:
            continue

        pos_ranks = positives.float() + 1.0
        best_rank = pos_ranks[0].item()
        best_ranks.append(best_rank)
        pos_counts.append(float(pos_ranks.numel()))
        rrs.append(1.0 / best_rank)
        aps.append((torch.arange(1, pos_ranks.numel() + 1, dtype=torch.float32) / pos_ranks).mean().item())
        for k in ks:
            recalls[k].append(1.0 if best_rank <= k else 0.0)

    if not best_ranks:
        return {}

    n = len(best_ranks)
    results = {
        'Recall@1': sum(recalls.get(1, [0.0])) / n * 100.0,
        'mAP': sum(aps) / n * 100.0,
        'MRR': sum(rrs) / n * 100.0,
        'Median_Rank': float(torch.tensor(best_ranks).median().item()),
        'Mean_Positive_Rank': sum(best_ranks) / n,
        'Mean_Positives_Per_Query': sum(pos_counts) / n,
        'Num_Queries': len(query_labels),
        'Num_Gallery': len(gallery_labels),
        'Num_Classes': len(set(gallery_labels)),
    }
    for k in ks:
        if k != 1:
            results[f'Recall@{k}'] = sum(recalls[k]) / n * 100.0
    return results


def _evaluate_benchmark(model, benchmark, device, config):
    batch_size = config['evaluation'].get('batch_size', config['data']['batch_size'])
    num_workers = config['data'].get('num_workers', 0)
    ks = tuple(config['evaluation'].get('retrieval_ks', [1, 5, 10]))

    q_feats, q_labels = _extract_dataset_features(model, benchmark['query_dataset'], device, batch_size, num_workers)
    g_feats, g_labels = _extract_dataset_features(model, benchmark['gallery_dataset'], device, batch_size, num_workers)
    return compute_retrieval_metrics(q_feats, q_labels, g_feats, g_labels, ks=ks)


def _build_summary(retrieval_results):
    if not retrieval_results:
        return {}

    def _mean(name):
        vals = [m[name] for m in retrieval_results.values() if name in m]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        'macro_recall_at_1': _mean('Recall@1'),
        'macro_recall_at_5': _mean('Recall@5'),
        'macro_recall_at_10': _mean('Recall@10'),
        'macro_map': _mean('mAP'),
        'macro_mrr': _mean('MRR'),
        'macro_median_rank': _mean('Median_Rank'),
        'num_benchmarks': len(retrieval_results),
    }


def evaluate_model(model, pair_loader, benchmarks, device, config, save_results=True, split_name='test'):
    model.eval()
    all_results = {'split': split_name, 'retrieval': {}, 'matching': {}, 'summary': {}}

    if benchmarks:
        logger.info('Retrieval evaluation...')
        for bm in benchmarks:
            logger.info('  %s', bm['name'])
            metrics = _evaluate_benchmark(model, bm, device, config)
            all_results['retrieval'][bm['name']] = metrics
            if metrics:
                logger.info('  R@1=%.2f, R@5=%.2f, mAP=%.2f, MRR=%.2f, MedRank=%.1f',
                            metrics.get('Recall@1', 0), metrics.get('Recall@5', 0),
                            metrics.get('mAP', 0), metrics.get('MRR', 0), metrics.get('Median_Rank', 0))

    if pair_loader is not None:
        logger.info('Matching evaluation...')
        all_results['matching'] = _compute_matching_metrics(model, pair_loader, device, config)

    all_results['summary'] = _build_summary(all_results['retrieval'])

    if save_results:
        output_dir = config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f'eval_results_{split_name}_{timestamp}.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info('Results saved to %s', result_file)

    if all_results['summary']:
        logger.info('Summary: R@1=%.2f, mAP=%.2f, MRR=%.2f',
                    all_results['summary'].get('macro_recall_at_1', 0),
                    all_results['summary'].get('macro_map', 0),
                    all_results['summary'].get('macro_mrr', 0))
    return all_results


def evaluate(config, pair_loader, benchmarks, device, checkpoint=None, split_name='test'):
    model = _create_model(config, device)

    ckpt_loaded = False
    if checkpoint and os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'])
        logger.info('Loaded checkpoint: %s', checkpoint)
        ckpt_loaded = True
    else:
        save_dir = config['checkpoint']['save_dir']
        best_path = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
            logger.info('Loaded best model: %s', best_path)
            ckpt_loaded = True
        else:
            ckpt_files = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch')]
            if ckpt_files:
                latest = os.path.join(save_dir, sorted(ckpt_files)[-1])
                model.load_state_dict(torch.load(latest, map_location=device, weights_only=False)['model'])
                logger.info('Loaded checkpoint: %s', latest)
                ckpt_loaded = True

    if not ckpt_loaded:
        logger.warning('No checkpoint loaded, evaluating with random weights')

    return evaluate_model(model=model, pair_loader=pair_loader, benchmarks=benchmarks,
                          device=device, config=config, save_results=True, split_name=split_name)
