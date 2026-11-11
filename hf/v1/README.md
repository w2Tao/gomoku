---
license: other
library_name: pytorch
pipeline_tag: other
tags:
  - gomoku
  - five-in-a-row
  - board-game
  - policy-network
  - distillation
  - 15x15
---

# gomoku-v1

15×15 无禁手五子棋（freestyle，五连及以上获胜）的**当前最强模型**，也是本项目
`ValuePolicyNet` 架构的**最终形态**。训练代号 `d3`；后续升级会更换架构。

**185,724 参数**，单次前向直接给出每个格子的分数，取合法格中最高者落子——没有搜索、
没有 VCF、没有双三检测。它由 [`gomoku-baseline`](https://huggingface.co/w2Tao/gomoku-baseline)
微调而来，训练标签来自 **RAPFI25**（2026 年 Gomocup 冠军）搜索输出的**主变**。

## 训练

| 项 | 值 |
|---|---|
| 暖启动 | `gomoku-baseline`（同一 `ValuePolicyNet` 主干） |
| 标签来源 | RAPFI25 主变；每局随机 6 手开局以打宽局面分布 |
| 标签量 | 49,849 个策略标签（`dist3.npz`） |
| 训练 | 4,000 步，lr 2e-4，batch 256，8 倍二面体增强 |
| 损失 | 只用 policy loss；`value_weight` 全 0，价值头没有训练信号 |

主变交替出现双方着法，**奇数位正是「我方该走哪」**——那才是 `TURN` 之外新增的信息。
同一个老师在想 300 ms 和 5,000 ms 时只有约 1/5 的应手一致，所以这个模型学到的是
「RAPFI25 在 300 ms 下怎么下」，不是「RAPFI25 怎么下」。

## 实测

| 对手 | `baseline` | `v1` | 局数 |
|---|---|---|---|
| 冻结启发式（内部标尺） | 63.0% | **75.7%** | 300（配对） |
| EMBRYO26（2026 年 Gomocup 第 6） | 12.8% | **30.7%** | 每臂 150 |
| TITO14（2014 年） | 0.0% | 0.0% | 24 |

重建前同一配方在 EMBRYO26 上是 13.0% → 31.2%（每臂 150，z = 3.67）；在修复/去重后的
数据集上重建并复测是 12.8% → 30.7%，结论不变。完整数据见
[`report.md` §7.13–§7.16](https://github.com/w2Tao/gomoku/blob/main/report.md)。

**诚实的水平**：对 2026 年的第一梯队还很远，对 2014 年的 TITO14 一局未胜。内部标尺
（对冻结启发式）在本项目里多次给出与真实引擎相反的排序，不能单独作为强度判据。

## 用法

```python
import torch
from model import ValuePolicyNet  # 本仓库 code/model.py

ck = torch.load("model.pt", map_location="cpu", weights_only=False)
net = ValuePolicyNet(**ck["config"]).eval()
# forward 返回 (value_logits, policy_logits)
_, policy = net(x)              # x: (N, 3, 15, 15) float32
move = policy[0].masked_fill(occupied, -inf).argmax().item()
```

格子按行主序编号，`idx = r * 15 + c`，`0 <= idx < 225`。`manifest.json` 里有权重
SHA-256、训练配方和实测来源。

## 文件

- `model.pt` —— 冻结权重（`state_dict` + `config`）。
- `manifest.json` —— 架构、参数、训练配方、SHA-256、推理方式、已知限制。

## 局限

- **没有搜索**：会漏掉 VCF / 双三。
- **老师受时限削弱**：训练标签只反映 RAPFI25 在 300 ms 下的判断。
- **标签是引擎主变，不是博弈论最优值**。
- 基座训练数据来自 Gomocup 2026 的公开棋谱（该来源**没有明确许可证**），蒸馏标签来自
  RAPFI25 对局；使用前请自行确认合规性。
