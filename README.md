# 五子棋 15×15

一个 15×15 无禁手五子棋项目。仓库里并列存在两个独立的对弈者：

1. **规则型启发式** —— 连五、堵五、深度 7 的 VCF（连续冲四）搜索、双三攻防，再加一个静态威胁评分。配置固化在 `play.FROZEN_HEURISTIC`，用 `frozen_heuristic()` 构造。**已冻结，不要再调**：它对自己的打法已经完备。
2. **学习的策略网络**（`baseline` / `v1`）—— 185,724 参数，单次前向直接给出每个格子的分数，取最高者落子。**没有搜索、没有 VCF、没有双三检测。**

## 现状

`v1` 是仓库里最强的 agent，也是界面的默认模型。它是当前 `ValuePolicyNet` 架构的最终形态
（训练代号 `d3`；后续升级会更换架构）。它由 `baseline` 微调而来，训练标签来自
RAPFI25（2026 年 Gomocup 冠军）的**主变**——引擎在自己的搜索输出里给出整条变化线，
奇数位正是「我方该走哪」。详见 [`report.md`](report.md) §7.10–7.16。

| 对手 | `baseline` | `v1`（默认） | 局数 |
|---|---|---|---|
| 冻结启发式（内部标尺） | 63.0% | 75.7% | 300 |
| EMBRYO26（2026 年 Gomocup 第 6） | **12.8%** | **30.7%** | **每臂 150** |
| TITO14（2014 年） | 0.0% | 0.0% | 24 |

EMBRYO26 一栏是**每臂 150 局的正式对比**：`v1` 比 `baseline` 高约 **18 个点**，这是本项目
唯一一个在外部标尺上显著的改进。此前二十几局的读数是 17.9% 对 32.1%——扩到 150 局后是
13.0% 对 31.2%（z = 3.67），小样本那次把两边都估高了；在修复后的数据集上重建并复测是
12.8% 对 30.7%，结论不变。

主变蒸馏有效的原因不是「教得更多」或「教得更深」，而是**换掉了谁在教**：

| 改动 | 结果 |
|---|---|
| 数据量 ×4 | 无提升 |
| 容量 ×4 | 无提升 |
| 训练步数 ×3 | +0.4σ |
| 老师搜索深度 ×3.3（300 ms → 1 s） | −0.6σ |
| **标签来源：混合棋谱 → 冠军主变** | **+18 个点，z = 3.67** |

其余试过都没有突破 +2σ：价值头 1 层搜索、2 层 negamax、战术+策略混合、自博弈迭代、
逐着法胜率头（Q head）、以引擎为对手的 RL、开局/局内双模型分工、用引擎估值建开局表。
数据和结论见 [`report.md`](report.md) §7。

**诚实的水平：离 2026 年的第一梯队还很远，对 2014 年的引擎一局未胜。** 另外 `v1` 学到的
是「RAPFI25 在 300 ms 下怎么下」而不是「RAPFI25 怎么下」——它的老师本身是受时限削弱的。

**瓶颈在学生和它的训练配方上，不在监督的数量或深度。** 继续往上需要换一个量级的东西。

## v2 实验（未替换默认）

为了验证「更大基座 + 策略先验搜索」，训练了 `v2s1`（96×6，1,037,844 参数，Gomocup
32k 步）和 `v2base`（在 167k 去重 RAPFI 主变上蒸馏 12k 步）。内部 policy 82.3% 对
v1 的 76.3%，但外部 EMBRYO26 150 局是 32.6% 对 31.9%（z = 0.12，不显著）；策略先验
搜索在 30 个开局上是 45–58%，而纯策略头是 86.7%——仍然被价值精度挡住。默认保持 `v1`。
完整数据、训练命令和失败原因见 [`report.md`](report.md) §9。

## v3 实验（未替换默认）

把 RAPFI 的 `Eval` 存成软价值标签，新增 `ValuePolicyEvalNet` 的标量头（+15,505 参数）。
采集到 6,123 个 eval 标签，但它们全部来自 RAPFI 的胜局，无法校准到胜率；标量头能
高相关地复现 Eval（MAE 0.076、corr +0.967），但把它放进搜索反而更差：1 层 6.7%、
2 层 18.3%，纯策略头 86.7%。默认仍是 `v1`，`v3eval` 不发布。详见
[`report.md`](report.md) §10。

## 快速开始

模型权重和数据集体积较大，不在 git 里，发布在 Hugging Face 上。**克隆之后先跑一次下载脚本**，它会取回四个文件（基线、最终版 v1、两个数据集）并逐个校验 SHA-256（共约 5.3 MB，可重复执行，已存在的会跳过）：

```
python code/fetch_data.py
```

然后启动对弈界面：

```
python app.py          # Windows 上也可以直接双击 app.bat
```

在 `http://127.0.0.1:8765` 打开本地对弈界面（端口被占用时自动换一个空闲端口）。选对手、选执子、直接下。

依赖只有 Python 3、PyTorch 和 NumPy。`fetch_data.py` 用的是标准库 HTTP，不需要登录 Hugging Face，也不需要 `huggingface_hub`。

> 只想看代码、不打算跑的话可以跳过第 1 步；但要跑测试、训练或对弈，这些文件必须有。
> `python code/fetch_data.py --check` 只校验不下载。

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
    value_target.py     价值目标的可执行规范（透视、和棋、样本量）
    model.py            ValueNet、ValuePolicyNet
    dataset.py          磁盘布局（N, WORDS）uint64、平面、二面体变换
    play.py             各类 agent：搜索 / 启发式 / 贪心 / 随机 / 策略
    baselines.py        冻结基线的注册表
    server.py           对弈界面的 HTTP + JSON 接口
    web/index.html      前端（无需构建）
    build_dataset.py    随机对局 → VCF 精确标签
    build_gomocup_dataset.py  Gomocup 棋谱 → 策略 + 胜负标签（当前主力数据）
    build_teacher_dataset.py  冻结启发式自博弈 → 策略 + 胜负标签
    build_distill_dataset.py  引擎主变 + Eval → 策略 / 软价值标签（v1/v3 数据）
    build_distill_all.py      合并 dist2/3/6 → 去重蒸馏标签（v2 基座数据）
    train.py            价值网络训练
    train_policy.py     价值 + 策略训练
    freeze_baseline.py  把 checkpoint 冻结成不可变基线
    push_hf.py          发布到 Hugging Face
    fetch_data.py       从 Hugging Face 取回权重与数据集（克隆后先跑这个）
    fetch_engines.py    下载 Gomocup 引擎，用作外部标尺
    duel_engine.py      我方模型 vs Gomocup 引擎（pbrain 协议）
    baseline_duel.py    基线 vs 外部对手，配对赛
    eval_three.py       多个模型互相打，逐对给分
    search_eval.py      策略先验搜索 vs 纯策略，配对评测
    policy_eval_relation.py  检查 RAPFI 实际走子是否就是 child eval 最优
    value_eval.py       价值头的三分类准确率 / MAE
    scalar_eval.py      Eval 软价值头的标量 MAE / BCE
    eval_calibrate.py   用实际胜负标定 RAPFI Eval 的温度 T
    bench_solver.py     VCF / 威胁检测的基准
    restore_check.py    验证「GitHub + Hub 能否还原项目」
    tests.py            棋盘 / 位棋盘 / 五连 / 平面
    test_solvers.py     窗口工具与强制胜检测
    test_value_target.py 价值目标的规范
    test_dataset.py     磁盘布局与二面体变换（缺 data/_smoke.npz 时跳过）
  ckpt/                 模型快照
    <baseline-id>/      model.pt + manifest.json，界面自动发现（manifest 进版本库）
    v1/                 当前 ValuePolicyNet 架构的最终形态，界面默认模型
    v2s1/ v2base/       v2 实验基座（权重本地训练，未发布；见 report.md §9）
    runs/<tag>/         训练过程的 checkpoint（已 git 忽略）
  data/                 数据集（全部 git 忽略，用 fetch_data.py 取回或 build_*.py 重建）
    gomocup15.npz       34.5 万局面（去重后）
    dist_all.npz        合并 dist2/3/6 并去重后的策略标签（v2 基座）
    dist_eval.npz       RAPFI Eval + PV 标签（v3 的标量价值头）
    其余                solver 标签、老师数据、蒸馏标签等
    gomocup/            340 MB 原始棋谱
```

## 棋盘约定

* 15×15，无禁手：**五连及以上**获胜（长连也算，不是禁手）。
* 黑先。
* 格子按行主序编号，`idx = r * 15 + c`，`0 <= idx < 225`。
* 每个局面都以**规范形式**存储：我方棋子、对方棋子、对手最后一手。颜色根本不出现在表示里，所以整条管道**没有任何翻号的地方**。
* 棋盘以 `(N, 4)` uint64 打包（225 位装不进两个 64 位字）。

## 外部标尺

我们对自己的所有「更强」都是相对本仓库的冻结启发式说的，那个坐标没有外部意义。Gomocup 的参赛引擎来自别的作者，强得多——它们是唯一能回答「我们到底什么水平」的东西。

```bash
# 下载一个引擎到 data/engines/（约 6 MB）
python code/fetch_engines.py EMBRYO26
python code/fetch_engines.py --list

# 打一场：默认 500 ms/步，颜色交替
python code/duel_engine.py EMBRYO26 v1 20 --time-ms 300
```

引擎通过 pbrain 协议（stdin/stdout 文本）驱动。**坐标是 0-based（0–14），不是 1-based** ——这一点搞错会让每颗子偏一格，两边棋盘从第一手起就分叉，表现为引擎走了"已被占据"的格子。另外三个坑：

* `INFO timeout_turn` **必须**等 `START` 返回 `OK` 之后再发，否则被丢弃，引擎按自己的默认时限思考（EMBRYO 会每步想 20 秒）
* `BEGIN` 的行为因引擎而异：TITO14/RAPFI25 收到 `BEGIN` 会**不管轮次立刻走一步**，所以执白时不能发；EMBRYO26 相反，没有 `BEGIN` 就沉默。注册表里用 `begin_white` 标注
* 超过设定时间没有任何输出就判定卡死、杀进程；每局重启进程，否则上一局的残留输出会串进下一局。慢引擎（EMBRYO）需要把阈值放宽到 25 秒

实测稳定性：TITO14 24/24 局完成、EMBRYO26 12/12 局完成，均无中止。

## 管道

```bash
# 1. 导入 Gomocup 棋谱（需要先从 gomocup.org 下载 results.zip 并解压到 data/gomocup/）
python code/build_gomocup_dataset.py --out gomocup15.npz

# 2. 生成老师数据（冻结启发式自博弈，12 进程，约 2 分钟）
python code/build_teacher_dataset.py --positions 150000 --workers 12 --out teacher15.npz

# 3. 训练策略网络（value + policy 双头）
python code/train_policy.py --data data/teacher15.npz --steps 8000 --tag t1_teacher
python code/train_policy.py --data data/gomocup15.npz --steps 8000 \
       --warm ckpt/runs/t1_teacher/best.pt --tag b2_fixed

# 4. 冻结成基线，界面里就能选它
#    当前 baseline 由 b2_fixed 冻结；t2_finetune 是 report.md §7.1 的历史 run
python code/freeze_baseline.py --src b2_fixed --id baseline --vs-heuristic 0.630

# 5. 三组模型对战
python code/eval_three.py 100
```

`--out` 只写文件名时会自动落到 `data/`。

### 蒸馏 `v1`（当前最强；训练代号 `d3`）

`v1` 是当前 `ValuePolicyNet` 架构的最终形态；后续升级会更换架构。上面前四步只到这个仓库
自己的能力上限。`v1` 的额外一步是从冠军引擎的主变里再学一遍：

```bash
# 6. 采集：让 RAPFI25 和我方模型对局，把主变每一手都当标签
python code/fetch_engines.py RAPFI25
python code/run_shards.py 6 2040 300 dist3 6     # 6 分片 x 340 局，约 10 分钟
python code/merge_npz.py dist3.npz data/dist3_s0.npz ... data/dist3_s5.npz

# 7. 微调（只用蒸馏数据，4,000 步；重建版 run 目录是 d3v2）
python code/train_policy.py --data data/dist3.npz --steps 4000 --lr 2e-4 \
       --warm ckpt/baseline/model.pt --tag d3v2
python code/freeze_baseline.py --src d3v2 --id v1 \
       --name "v1 · final ValuePolicyNet (RAPFI25 PV distillation)" \
       --vs-heuristic 0.757 --teacher "RAPFI25 principal variations" --opening 6
```

主变里**奇数位**是老师替我方想的着法——那才是新增的信息，`TURN` 本来就只回它自己那一步。
`opening` 参数控制采集时的随机开局宽度（`run_shards.py` 的第 5 个参数），从 2 加到 6 会
让老师走到更宽的局面里，标签更少但外部更强。

### 判定强度

内部标尺（对冻结启发式）**不可信**——它在本项目里三次给出与真实引擎相反的排序。判据请用
外部引擎，并且**样本要够**：24 局的置信区间宽达 0–24%。

```bash
# 两臂对比，每臂 150 局，6 分片并行，约 6 分钟
python code/duel_many.py EMBRYO26 baseline,v1 150 6 300
```

它直接给出 Wilson 区间和两臂的 z 检验，避免"点估计看起来涨了"这种误判。

### 实验脚本

下面这些是一次性的，用来产出 [`report.md`](report.md) §7 的结论，日常使用不需要它们：

| 脚本 | 用来回答什么 |
|---|---|
| `search_ct.py` | 价值头驱动搜索（1 层 / 2 层）是否优于纯策略头 |
| `search_eval.py` | 新的更大基座上，策略先验搜索 vs 纯策略 |
| `policy_eval_relation.py` | RAPFI 的实际着法是否等于 1 层 child eval 最优 |
| `value_eval.py` | 价值头的三分类准确率 / MAE，以及按手数的变化 |
| `scalar_eval.py` | Eval 软价值头的标量精度 / 对实际胜负的校准 |
| `eval_calibrate.py` | RAPFI `Eval` 到胜率/`tanh` 的温度标定 |
| `q_eval.py` | 逐着法胜率头（Q head）能否排序着法 |
| `build_selfplay_dataset.py` | 自博弈迭代有没有用 |
| `build_engine_rl_dataset.py` | 以引擎为对手做 RL 有没有梯度 |
| `handoff_eval.py` | 开局 / 局内双模型分工 |
| `opening_stats.py` | 棋谱里的开局为什么这么集中（赛制，不是演化） |
| `opening_eval.py` | 能否用引擎的 `Eval` 给开局排序 |
| `pv_stability.py` | 蒸馏标签会不会随老师的搜索深度改变 |
| `ply_eval.py` | 按手数拆分的准确率（由此发现棋谱的开局重复问题） |
| `slice_by_ply.py` | 按手数切数据集 |
| `pbrain.py` | pbrain 协议驱动（被 `duel_engine.py` 等导入） |


## 实测数据（15×15）

完整数值见 [`report.md`](report.md)。速查：

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

两个最该记住的数字：**VCF 超过深度 5 之后几乎不再增长**（5→7 只多 0.3 个百分点，成本翻倍）；**均匀随机 rollout 的结果标签只带 0.011 bit**——棋盘越大，乱下的结果越接近抛硬币，所以这条路已被彻底放弃。

## 已发布

模型和数据集已发布到 Hugging Face，可以直接下载使用：

| 仓库 | 内容 |
|---|---|
| [`gomoku-baseline`](https://huggingface.co/w2Tao/gomoku-baseline) | 冻结基线的完整权重（value + policy 双头，185,724 参数）与身份卡 |
| [`gomoku-v1`](https://huggingface.co/w2Tao/gomoku-v1) | 当前架构的最终形态 `v1`（185,724 参数，RAPFI25 主变蒸馏）与身份卡 |
| [`gomoku-15x15`](https://huggingface.co/datasets/w2Tao/gomoku-15x15) | `gomocup15.npz`（34.5 万局面，已去重）、`teacher15.npz`（15 万局面） |

三个仓库的说明卡片分别在 `hf/model/README.md`、`hf/v1/README.md` 和 `hf/dataset/README.md`。

## 文档

* [`manual.md`](manual.md) —— 原理、方法学、失败原因和 bug 目录，**不含具体数字**。
* [`report.md`](report.md) —— 所有测量数值（速度、覆盖率、消融、训练对比），每个数字都带出处。
