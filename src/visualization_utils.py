"""匹配可视化工具"""
import cv2
import numpy as np


def draw_matches_cv2(img1_cv, img2_cv, matches, scores, feat_h, feat_w,
                     img_size=256, circle_radius=3):
    """用OpenCV绘制匹配点对"""
    scale_h = img_size / feat_h
    scale_w = img_size / feat_w

    kp1, kp2, cv_matches = [], [], []
    for i, (m, s) in enumerate(zip(matches, scores)):
        y1, x1, y2, x2 = m
        kp1.append(cv2.KeyPoint(float(x1 * scale_w + scale_w / 2), float(y1 * scale_h + scale_h / 2), circle_radius))
        kp2.append(cv2.KeyPoint(float(x2 * scale_w + scale_w / 2), float(y2 * scale_h + scale_h / 2), circle_radius))
        cv_matches.append(cv2.DMatch(i, i, float(1.0 - s)))

    return cv2.drawMatches(
        img1_cv, kp1, img2_cv, kp2, cv_matches, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def similarity_heatmap(sim_matrix, output_size=512):
    """相似度矩阵热力图"""
    sim_min, sim_max = sim_matrix.min(), sim_matrix.max()
    if sim_max - sim_min > 1e-8:
        sim_norm = ((sim_matrix - sim_min) / (sim_max - sim_min) * 255).astype(np.uint8)
    else:
        sim_norm = np.zeros_like(sim_matrix, dtype=np.uint8)
    size = max(sim_norm.shape[0], output_size)
    return cv2.applyColorMap(cv2.resize(sim_norm, (size, size), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_JET)
