# 五子棋 15×15

一个 15×15 无禁手五子棋项目。仓库里并列存在两个独立的对弈者：

1. **规则型启发式** —— 连五、堵五、深度 7 的 VCF（连续冲四）搜索、双三攻防，再加一个静态威胁评分。**它是目前最强的 agent，并且已经冻结**（配置固化在 `play.FROZEN_HEURISTIC`，请用 `frozen_heuristic()` 构造；不要再调它）。
2. **学习的价值网络** —— 小型残差 CNN，用 solver 精确标签加蒙特卡洛 rollout 标签训练，通过 1 层搜索使用。

诚实的现状：**启发式以明显优势压过所有学习模型**（棋盘更换前在 9×9 上实测约 3–5:1）。学习管道是完整可用的、全程有度量，但还没有产出一个能和规则型对手竞争的下法。

## 快速开始

```
python app.py          # Windows 上也可以直接双击 app.bat
```

在 `http://127.0.0.1:8765` 打开本地对弈界面（端口被占用时自动换一个空闲端口）。选对手、选执子、直接下。除了 Python 3 和 PyTorch 没有别的依赖。

## 目录

```
gomoku/
  app.py / app.bat      启动器：起服务、开浏览器
  code/                 所有脚本
    paths.py            数据/快照/前端路径的唯一来源
    engine.py           棋盘尺寸、位棋盘、五连判定
    solvers.py          窗口工具、3 层强制胜检测
    vcf.py              VCF 搜索（限定深度的连续冲四）
    labeler.py          对称的 +1/-1/未知 标注
    model.py            ValueNet、ValuePolicyNet
    dataset.py          磁盘布局（N, WORDS）uint64、平面、二面体变换
    play.py             各类 agent：搜索 / 启发式 / 贪心 / 随机 / 策略
    baselines.py        冻结基线的注册表
    server.py           对弈界面的 HTTP + JSON 接口
    web/index.html      前端（无需构建）
    build_dataset.py    随机对局 → VCF 标签
    build_mc_dataset.py 随机对局 → rollout 标签
    build_policy_dataset.py  把某个 agent 蒸馏成着法目标
    build_rollout_dataset.py 并行生成 rollout
    train.py            价值网络训练
    train_policy.py     价值 + 策略训练
    freeze_baseline.py  把 checkpoint 冻结成不可变基线
    paired_eval.py、baseline_duel.py、heuristic_duel.py、depth_check.py、
    eval_heuristic.py、eval_value_by_ply.py、analyze_rollouts.py 等
  ckpt/                 模型快照（已 git 忽略）
    <baseline-id>/      model.pt + manifest.json，界面自动发现
    runs/<tag>/         训练过程的 checkpoint
  data/                 数据集（已 git 忽略）
    selfplay/           后续阶段模型对弈产生的数据
```

## 棋盘约定

* 15×15，无禁手：**五连及以上**获胜（长连也算，不是禁手）。
* 黑先。
* 格子按行主序编号，`idx = r * 15 + c`，`0 <= idx < 225`。
* 每个局面都以**规范形式**存储：我方棋子、对方棋子、对手最后一手。颜色根本不出现在表示里，所以整条管道**没有任何翻号的地方**。
* 棋盘以 `(N, 4)` uint64 打包（225 位装不进两个 64 位字）。

## 管道

```bash
# 1. 在能证明强制胜的地方生成精确标签（15×15 上约 15 局面/秒）
python code/build_dataset.py --target 5000 --out batch1.npz

# 2. 给 solver 盲区生成 rollout 标签（慢：每局面数秒）
python code/build_rollout_dataset.py --from-mc1 500 --rollouts 64 --workers 12

# 3. 训练价值网络
python code/train.py --data data/batch1.npz --steps 6000 --tag v1

# 4. 冻结成基线，然后在界面里和它下
python code/freeze_baseline.py --src v1 --id baseline0
```

`--out` 只写文件名时会自动落到 `data/`。

## 实测数据（15×15）

| 项目 | 数值 |
|---|---|
| 五连窗口总数 | 572 |
| 均匀随机对局长度 | 平均 108 手，中位 109 手 |
| 随机对局速度 | 约 8,300 局/秒 |
| 随机对局先手优势 | 黑 51.6%、白 48.4%、和棋 0% |
| 中局每局面候选着法 | 约 150 个 |
| VCF 搜索，深度 7 | 3.8 ms/次 |
| 启发式每步 | 4.1 ms |
| VCF 可判定局面（深度 7） | 62.9%（+1 占 50.8%，−1 占 12.1%，未知 37.1%） |
| 存在任何强制着法的局面 | 71.5%（即 VCF 的天花板） |
| VCF 覆盖：深度 3 → 5 → 7 → 11 | 49.2% → 50.5% → 50.8% → 50.9% |
| 随机对局结果标签的信息量 | 0.011 bit（上限 1.0） |
| solver 打标速度 | 约 15 局面/秒 |
| ValueNet（48 通道 × 4 块） | 185,451 参数，感受野 19 ≥ 15 |

两个最该记住的数字：**VCF 超过深度 5 之后几乎不再增长**（5→7 只多 0.3 个百分点，成本翻倍）；**随机 rollout 标签只带 0.011 bit**——比 9×9 时代低三倍，因为棋盘越大，乱下的结果越接近抛硬币。

作为参照，同样的测量在旧的 9×9 棋盘上普遍快 5–11 倍，这也是 15×15 上生成数据是件大工程的原因。

## 历史结果（9×9 棋盘，换盘之前）

留作标定用——**这些数字都还没有在 15×15 上重测过**。

| agent | 对贪心+ |
|---|---|
| 启发式（VCF 深度 7 + 窗口评分） | 83% |
| 学习出的策略头 | 23–39% |
| 价值网络 + 1 层搜索（baseline1） | 26% |
| 价值网络 + 1 层搜索（baseline0） | 19% |
| 价值网络 + 2 层搜索（修复版） | 30.6% |

解读：强度基本上全部来自启发式；学习出来的价值单调地在变好，但起点极低；而更深搜索暂时不划算，因为价值误差仍然大于候选着法之间的信号。

## 文档

* [`manual.md`](manual.md) —— 详细原理、实测数据，以及产生这些数据的 bug。
* [`AGENT.md`](AGENT.md) —— 本仓库的协作约定，包括怎么验证改动、以及哪些测量不可信。
