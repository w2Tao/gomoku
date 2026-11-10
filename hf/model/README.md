---
license: other
library_name: pytorch
pipeline_tag: other
tags:
  - gomoku
  - five-in-a-row
  - board-game
  - value-network
  - policy-network
  - 15x15
---

# gomoku-baseline

一个 15×15 无禁手五子棋（freestyle，五连及以上获胜）的策略网络。

**185,724 参数**，单次前向直接给出每个格子的分数，取合法格中最高者落子——**没有搜索、没有 VCF、没有双三检测**。仍然在 200 局配对中胜过了本仓库冻结的规则型启发式 **65.5%**。

## 模型

```
ValuePolicyNet(channels=48, blocks=4, head_channels=24)
  stem   Conv3x3(3 -> 48) + BN + ReLU
  body   4 个残差块（每块两个 3x3 卷积 + BN）
  value  Conv1x1(48 -> 24) + BN + ReLU + Flatten + Linear(24*225 -> 3)
  policy Conv1x1(48 -> 1) + flatten + 可学习的 225 维位置偏置
```

感受野 19 ≥ 15，所以每个输出单元都能看到整个棋盘。输入是三层平面：我方棋子、对方棋子、对手最后一手。

## 训练

| 阶段 | 数据 | 步数 |
|---|---|---|
| 1. 老师蒸馏 | 冻结启发式自博弈，15 万局面（着法 + 胜负） | 8,000 |
| 2. 微调 | Gomocup 2026 引擎棋谱，50 万局面（着法 + 胜负） | 8,000 |

batch 256，8 倍二面体增强，value + policy 双头联合训练。

### 为什么两段式

单独用启发式数据（蒸馏）训出来只能达到老师水平——实测对老师 47.0%。单独用引擎棋谱从零训能到 57.5%。**先蒸馏再微调**能到 65.5%。弱老师提供的开局直觉是有价值的先验。

## 实测

| 对手 | 得分 | 局数 |
|---|---|---|
| 冻结启发式（VCF 深度 7 + 双三） | **65.5%** | 200（配对，换色，同开局） |
| 只蒸馏的版本 | 71.5% | 200 |
| 只用棋谱的版本 | 61.5% | 200 |

## 用法

```python
import torch
from model import ValuePolicyNet  # 本仓库 code/model.py

ck = torch.load("model.pt", map_location="cpu", weights_only=False)
net = ValuePolicyNet(**ck["config"]).eval()
# 注意 forward 返回 (value_logits, policy_logits)
_, policy = net(x)              # x: (N, 3, 15, 15) float32
move = policy[0].masked_fill(occupied, -inf).argmax().item()
```

格子按行主序编号，`idx = r * 15 + c`，`0 <= idx < 225`。

## 局限

- **没有搜索**：会漏掉 VCF / 双三。加上价值头做 1 层搜索还没测。
- **开局多样性差**：训练数据每个联赛只有 12 个固定开局，前几手会被这 12 个主导。
- **标签是引擎的策略化胜负**，不是博弈论最优值。
- 训练数据来自 Gomocup 2026 的公开棋谱，该来源**没有明确许可证**，见数据集卡片。
