# 五子棋 15×15

一个 15×15 无禁手五子棋项目。仓库里并列存在两个独立的对弈者：

1. **规则型启发式** —— 连五、堵五、深度 7 的 VCF（连续冲四）搜索、双三攻防，再加一个静态威胁评分。配置固化在 `play.FROZEN_HEURISTIC`，用 `frozen_heuristic()` 构造。**已冻结，不要再调**：它对自己的打法已经完备。
2. **学习的策略网络**（`baseline`）—— 185,724 参数，单次前向直接给出每个格子的分数，取最高者落子。**没有搜索、没有 VCF、没有双三检测。**

## 现状

`d2` 是仓库里最强的 agent，也是界面的默认模型。它由 `baseline` 微调而来，训练标签来自
RAPFI25（2026 年 Gomocup 冠军）的**主变**——引擎在自己的搜索输出里给出整条变化线，
奇数位正是「我方该走哪」。详见 [`report.md`](report.md) §7.10–7.12。

| 对手 | `baseline` | `d2`（默认） | 局数 |
|---|---|---|---|
| 冻结启发式（内部标尺） | 63.0% | **77.7%** | 300 |
| EMBRYO26（2026 年 Gomocup 第 6） | 17.9% | **29.6%** | 27–28 |
| TITO14（2014 年） | 0.0% | 0.0% | 24 |

主变蒸馏是本项目唯一一个超过 +2σ 的改进（+14.7%，+4.7σ）。但要注意两件事：
**放大 3.9 倍数据只多换来 1.4 个点**（收益急剧递减），而且**外部引擎上仍不显著**——
EMBRYO26 的点估计从 17.9% 升到 29.6%，但二十几局的置信区间还重叠着。

此外九个方向试过都没有突破 +2σ：更多数据、4 倍容量、3 倍训练步数、价值头 1 层搜索、
2 层 negamax、战术+策略混合、自博弈迭代、逐着法胜率头（Q head）、以引擎为对手的 RL。
数据和结论见 [`report.md`](report.md) §7。

**诚实的水平：离 2026 年的第一梯队还很远，对 2014 年的引擎一局未胜。**

这个项目在「监督模仿引擎棋谱」这条路上已经到达稳定点。要继续必须换一个量级的东西，不是调这套配方的参数。

## 快速开始

模型权重和数据集体积较大，不在 git 里，发布在 Hugging Face 上。**克隆之后先跑一次下载脚本**，它会取回三个文件并逐个校验 SHA-256（共约 4.7 MB，可重复执行，已存在的会跳过）：

```
python code/fetch_data.py
```

然后启动对弈界面：

```
python app.py          # Windows 上也可以直接双击 app.bat
```

在 `http://127.0.0.1:8765` 打开本地对弈界面（端口被占用时自动换一个空闲端口）。选对手、选执子、直接下。

依赖只有 Python 3、PyTorch 和 NumPy。`fetch_data.py` 用的是标准库 HTTP，不需要登录 Hugging Face，也不需要 `huggingface_hub`。

> 只想看代码、不打算跑的话可以跳过第 1 步；但要跑测试、训练或对弈，三个文件必须有。
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
    train.py            价值网络训练
    train_policy.py     价值 + 策略训练
    freeze_baseline.py  把 checkpoint 冻结成不可变基线
    push_hf.py          发布到 Hugging Face
    fetch_data.py       从 Hugging Face 取回权重与数据集（克隆后先跑这个）
    fetch_engines.py    下载 Gomocup 引擎，用作外部标尺
    duel_engine.py      我方模型 vs Gomocup 引擎（pbrain 协议）
    baseline_duel.py    基线 vs 外部对手，配对赛
    eval_three.py       多个模型互相打，逐对给分
    bench_solver.py     VCF / 威胁检测的基准
    restore_check.py    验证「GitHub + Hub 能否还原项目」
  ckpt/                 模型快照
    <baseline-id>/      model.pt + manifest.json，界面自动发现（manifest 进版本库）
    runs/<tag>/         训练过程的 checkpoint（已 git 忽略）
  data/                 数据集
    gomocup15.npz       50 万局面，已纳入版本库
    其余                solver 标签、老师数据等，可重新生成（已 git 忽略）
    gomocup/            340 MB 原始棋谱（已 git 忽略）
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
python code/duel_engine.py EMBRYO26 baseline 20 --time-ms 300
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
       --warm ckpt/runs/t1_teacher/best.pt --tag t2_finetune

# 4. 冻结成基线，界面里就能选它
python code/freeze_baseline.py --src t2_finetune --id baseline --vs-greedy 0.655

# 5. 三组模型对战
python code/eval_three.py 100
```

`--out` 只写文件名时会自动落到 `data/`。

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
| [`gomoku-15x15`](https://huggingface.co/datasets/w2Tao/gomoku-15x15) | `gomocup15.npz`（50 万局面）、`teacher15.npz`（15 万局面） |

两个仓库的说明卡片分别在 `hf/model/README.md` 和 `hf/dataset/README.md`。

## 文档

* [`manual.md`](manual.md) —— 原理、方法学、失败原因和 bug 目录，**不含具体数字**。
* [`report.md`](report.md) —— 所有测量数值（速度、覆盖率、消融、训练对比），每个数字都带出处。
