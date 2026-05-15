# IMTMN：基于改进Transformer的多视角图像匹配网络

本项目是毕业设计"基于改进Transformer的多视角图像匹配方法研究"的代码实现，主要解决无人机（UAV）、卫星和地面视角之间的跨视角地理定位问题。

## 研究背景

跨视角图像匹配是一个具有挑战性的任务——同一地点在不同视角下的外观差异非常大，传统的特征匹配方法很难处理这种巨大的视角变化。本文提出了IMTMN（Improved Multi-View Transformer Matching Network），通过引入几何约束注意力机制和最优传输匹配策略来提升跨视角匹配的准确性。

## 方法概述

整体网络结构如下：

```
输入图像对 (UAV / Satellite / Ground)
    │
    ▼
ResNet50 + FPN 多尺度特征提取          ← backbone.py
    │
    ▼
多尺度特征融合
    │
    ▼
Self-Attention → Cross-Attention → FFN  ← transformer.py
    │  (× N 层，双向交叉注意力)
    ▼
几何约束注意力 (Geometric-Aware Attention)
    │
    ▼
相似度矩阵  S = F₁ · F₂ᵀ              ← matcher.py
    │
    ▼
Sinkhorn 最优传输 → Top-K 匹配点
```

### 三个主要创新点

**1. 几何约束注意力机制**

在标准注意力的基础上引入了几何偏置项G，利用空间坐标关系编码极线约束等几何先验：

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + G\right)$$

其中G通过一个轻量MLP从归一化坐标生成，采用因式分解形式 $G_{ij} = b_i + b_j$，将复杂度从 $O(N^2)$ 降到 $O(N)$。这样可以在不显著增加计算量的前提下，让注意力机制感知到空间几何关系，减少明显不合理的匹配。

**2. 稀疏跨视角注意力**

标准的Cross-Attention需要计算所有位置对之间的注意力，复杂度为 $O(N^2)$。我们只保留每个位置注意力分数最高的Top-k个位置，其余置为负无穷后再做softmax，这样既降低了计算量到 $O(N \cdot k)$，又天然地过滤掉了噪声匹配。

**3. 基于最优传输的特征匹配**

借鉴SuperGlue的思路，使用Sinkhorn算法求解最优传输问题，将相似度矩阵转化为双随机匹配矩阵，实现全局最优的一对一匹配。同时引入dustbin机制处理无对应点的情况：

$$S = F_1 \cdot F_2^T \quad \rightarrow \quad M = \text{Sinkhorn}(S)$$

## 损失函数

总损失由三部分加权组成：

$$\mathcal{L} = w_1 \cdot \mathcal{L}_{match} + w_2 \cdot \mathcal{L}_{geo} + w_3 \cdot \mathcal{L}_{retrieval}$$

| 损失项 | 说明 |
|--------|------|
| $\mathcal{L}_{match}$ | 匹配矩阵的负对数似然 + 熵正则化，鼓励稀疏的一对一匹配 |
| $\mathcal{L}_{geo}$ | 极线约束残差 $\|x_2^T F x_1\|$，约束匹配的几何一致性 |
| $\mathcal{L}_{retrieval}$ | InfoNCE对比学习损失，拉近同一地点不同视角的全局特征 |

## 项目结构

```
project/
├── main.py                     # 训练/评估入口
├── config.yaml                 # 完整模型训练配置
├── requirements.txt            # Python依赖
│
├── data/
│   ├── datasets.py             # 数据集加载与划分
│   └── transforms.py           # 数据增强
│
├── src/
│   ├── train.py                # 训练逻辑（warmup、梯度累积、差异化学习率）
│   ├── eval.py                 # 评估（Recall@K、mAP、MRR等）
│   ├── losses.py               # 三种损失函数
│   ├── inference.py            # 单对图像推理+可视化
│   ├── visualization_utils.py  # 匹配绘制和热力图工具
│   └── models/
│       ├── imtmn.py            # 完整模型定义
│       ├── backbone.py         # ResNet50 + FPN
│       ├── transformer.py      # 多视角Transformer（含几何注意力）
│       └── matcher.py          # 相似度矩阵 + Sinkhorn匹配
│
├── checkpoints/
│   └── .gitkeep                # 本地模型权重目录，权重文件不提交到GitHub
│
├── outputs/                    # 最终展示结果
│   ├── demos/                  # 训练曲线、匹配示例、总览图
│   │   ├── matching_examples/
│   │   └── similarity_examples/
│   ├── figures/                # 论文展示图，只保留PNG最终版
│   │   ├── advanced/
│   │   ├── introduction/
│   │   ├── scientific/
│   │   └── theory/
│   └── metrics/                # 评估指标JSON
│
└── datasets/                   # 数据集（需自行下载放置）
    └── .gitkeep
```

## 使用的数据集

| 数据集 | 视角 | 规模 | 来源 |
|--------|------|------|------|
| [University-1652](https://github.com/layumi/University1652-Baseline) | UAV / 卫星 / 地面 | 1652栋建筑，72所大学 | Zheng et al., ACM MM 2020 |
| [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) | UAV / 卫星 | 200个场景，多飞行高度 | Zhu et al., 2023 |

数据集和模型权重体积较大，GitHub仓库只保留目录占位。运行训练、评估或推理前，请将数据集放入 `datasets/`，将模型权重放入 `checkpoints/`。

## 环境配置

```bash
pip install -r requirements.txt
```

## 训练

```bash
# 完整模型训练（需要GPU）
python main.py

# 指定配置文件
python main.py --config config.yaml

```

主要训练参数（config.yaml）：

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 64 | 特征维度 |
| num_heads | 4 | 注意力头数 |
| num_layers | 1 | Transformer层数 |
| transformer_size | 8 | 内部特征图尺寸（8×8=64 tokens） |
| sinkhorn_iters | 5 | Sinkhorn迭代次数 |
| epochs | 5 | 训练轮次 |
| lr | 1e-4 | 学习率 |
| warmup_epochs | 0 | 学习率预热 |
| batch_size | 1 | 批大小 |
| grad_accum_steps | 4 | 梯度累积步数 |

## 评估

```bash
# 在验证集上评估
python main.py --mode eval

# 在测试集上评估（指定checkpoint）
python main.py --mode test --checkpoint checkpoints/best_model.pth
```

评估指标：
- **Recall@1 / @5 / @10**：检索命中率
- **mAP**：平均精度
- **MRR**：平均倒数排名
- **Median Rank**：正确结果的中位排名

## 推理与可视化

```bash
# 对一对图像进行匹配推理
python -m src.inference \
  --img1 path/to/uav.jpg \
  --img2 path/to/satellite.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output outputs/match_result.png

# 最终展示图
# outputs/demos/
# outputs/figures/
```

## 参考文献

- Zheng Z, Wei Y, Yang Y. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization[C]. ACM MM, 2020.
- Sarlin P E, DeTone D, Malisiewicz T, et al. SuperGlue: Learning Feature Matching with Graph Neural Networks[C]. CVPR, 2020.
- Cuturi M. Sinkhorn Distances: Lightspeed Computation of Optimal Transport[C]. NeurIPS, 2013.
- Lin T Y, Dollár P, Girshick R, et al. Feature Pyramid Networks for Object Detection[C]. CVPR, 2017.
