# IMTMN: Improved Multi-View Transformer Matching Network

跨视角图像地理定位系统，专为 UAV/卫星/地面多视角匹配任务设计。

## 核心创新

### 1. 几何约束注意力 (Geometric-Aware Attention)

在标准注意力基础上引入几何偏置，利用极线约束的空间先验减少误匹配：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + G\right)$$

其中 $G$ 由位置坐标经可学习 MLP 生成，编码空间几何关系。

### 2. 稀疏注意力 (Sparse Cross-Attention)

只保留 Top-$k$ 最相似特征，将复杂度从 $O(N^2)$ 降至 $O(N \cdot k)$，同时天然过滤噪声匹配。

### 3. Optimal Transport 匹配 (Sinkhorn Algorithm)

两步匹配框架，彻底解决一对多歧义问题：

```
S = F1 · F2^T      ← 相似度矩阵
M = Sinkhorn(S)    ← 全局最优一对一分配 + dustbin 无对应点处理
```

## 整体 Pipeline

```
输入图像对 (UAV / Satellite / Ground)
    ↓
CNN Feature Extraction (ResNet50 + FPN)    ← backbone.py
    ↓
Multi-Scale Feature Fusion
    ↓
Self-Attention (SA)                        ← transformer.py
    ↓
Cross-Attention [Sparse] (CA)              ← 创新2
    ↓
Feed-Forward Network (FFN)
    ↓
Geometric-Aware Attention                  ← 创新1
    ↓
Similarity Matrix  S = F1 · F2^T           ← matcher.py
    ↓
Optimal Transport  M = Sinkhorn(S)         ← 创新3
    ↓
Top-K Matching Points
```

## 项目结构

```
project/
├── config.yaml                 # 训练配置
├── config_lite.yaml            # 轻量级配置（CPU 可用）
├── main.py                     # 训练入口
├── visualize.py                # 可视化工具
│
├── src/
│   ├── models/
│   │   ├── backbone.py         # ResNet50 + FPN 特征提取
│   │   ├── transformer.py      # 核心创新：SA + CA + Geometric Attention
│   │   ├── matcher.py          # Similarity Matrix + Optimal Transport
│   │   └── imtmn.py            # 完整模型
│   ├── datasets/               # 数据集加载器
│   ├── losses.py               # 三种损失函数
│   ├── train.py                # 训练逻辑
│   ├── eval.py                 # 评估逻辑
│   └── inference.py            # 推理脚本
│
└── datasets/                   # 数据集目录（需自行下载）
    ├── University-1652/
    └── SUES-200/
```

## 环境安装

```bash
pip install torch torchvision
pip install opencv-python pillow matplotlib pyyaml tqdm
```

## 快速开始

### 训练（CPU 轻量级）

```bash
python3 main.py --config config_lite.yaml
```

### 训练（GPU 标准）

```bash
python3 main.py
```

### 推理 + 可视化

```bash
python3 -m src.inference \
  --img1 path/to/image1.jpg \
  --img2 path/to/image2.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output outputs/match_result.png
```

### 查看训练结果

```bash
python3 visualize.py
open outputs/training_curves.png
open outputs/summary.png
```

## 损失函数

$$\mathcal{L} = w_1 \cdot \mathcal{L}_{match} + w_2 \cdot \mathcal{L}_{geo} + w_3 \cdot \mathcal{L}_{retrieval}$$

| 损失 | 含义 |
|------|------|
| $\mathcal{L}_{match}$ | 匹配矩阵的负对数似然 |
| $\mathcal{L}_{geo}$ | 极线约束 $\|x_2^T F x_1\|$ |
| $\mathcal{L}_{retrieval}$ | 跨视角 InfoNCE 对比学习 |

## 支持数据集

| 数据集 | 视角 | 说明 |
|--------|------|------|
| [University-1652](https://github.com/layumi/University1652-Baseline) | UAV / Satellite / Ground | 1652 栋大学建筑 |
| [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) | UAV / Satellite | 200 个场景，多高度 |

## 主要配置参数

```yaml
model:
  d_model: 256          # 特征维度
  num_heads: 8          # 注意力头数
  num_layers: 4         # Transformer 层数
  topk_attn: 64         # 稀疏注意力 Top-k
  transformer_size: 32  # 内部分辨率（防 OOM）
  topk_match: 200       # 输出匹配点数
  sinkhorn_iters: 50    # Sinkhorn 迭代次数
```

## 评估指标

- **Recall@1 / @5 / @10**：前 K 个检索结果命中率
- **AP**：平均精度
- **Median Rank**：正确目标的中位排名（越小越好）
