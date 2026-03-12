"""
IMTMN 匹配可视化工具

提供:
  - OpenCV drawMatches 匹配可视化
  - Similarity Matrix 热力图可视化
"""
import cv2
import numpy as np


def draw_matches_cv2(img1_cv, img2_cv, matches, scores, feat_h, feat_w,
                     img_size=256, circle_radius=3):
    """
    OpenCV drawMatches 可视化匹配点

    Args:
        img1_cv: BGR image1
        img2_cv: BGR image2
        matches: [K, 4] (y1, x1, y2, x2)
        scores: [K]
        feat_h, feat_w: 特征图尺寸
        img_size: 输入图像尺寸
    Returns:
        result_img: BGR 可视化图像
    """
    scale_h = img_size / feat_h
    scale_w = img_size / feat_w

    keypoints1 = []
    keypoints2 = []
    cv_matches = []

    for i, (m, s) in enumerate(zip(matches, scores)):
        y1, x1, y2, x2 = m

        px1 = x1 * scale_w + scale_w / 2
        py1 = y1 * scale_h + scale_h / 2
        px2 = x2 * scale_w + scale_w / 2
        py2 = y2 * scale_h + scale_h / 2

        keypoints1.append(cv2.KeyPoint(float(px1), float(py1), circle_radius))
        keypoints2.append(cv2.KeyPoint(float(px2), float(py2), circle_radius))
        cv_matches.append(cv2.DMatch(i, i, float(1.0 - s)))

    result_img = cv2.drawMatches(
        img1_cv, keypoints1,
        img2_cv, keypoints2,
        cv_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return result_img


def similarity_heatmap(sim_matrix):
    """
    相似度矩阵热力图

    Args:
        sim_matrix: [N, N]
    Returns:
        heatmap: BGR 热力图
    """
    sim_min, sim_max = sim_matrix.min(), sim_matrix.max()
    if sim_max - sim_min > 1e-8:
        sim_norm = ((sim_matrix - sim_min) / (sim_max - sim_min) * 255).astype(np.uint8)
    else:
        sim_norm = np.zeros_like(sim_matrix, dtype=np.uint8)

    size = min(512, sim_norm.shape[0])
    sim_resized = cv2.resize(sim_norm, (size, size), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(sim_resized, cv2.COLORMAP_JET)
    return heatmap
