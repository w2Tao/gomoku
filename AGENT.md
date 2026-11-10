# AGENT.md

在本仓库中改动代码的协作约定，适用于人，也适用于 AI。尤其是**多个 agent 同时工作**的时候。

---

## 1. 不变量

这些不是风格偏好。破坏其中任何一条，结果都会看起来正常，但其实是错的。

1. **棋盘尺寸只定义在 `code/engine.py`。** 其他文件一律不许写 `15`、`225` 或 `15 * 15`。一切从 `engine.W`、`engine.H`、`engine.CELLS`、`engine.WINDOWS` 推导。
2. **永远用规范形式。** 局面是 `(own, opp, last)`，以落子方为视角。不存在因颜色而翻的符号。**如果你发现自己在因为颜色写 `-v`，说明你已经离开了规范形式。**
3. **`last` 是对手的子**，永远不是自己的。
4. **磁盘上的棋盘是 `(N, WORDS)` uint64**，经 `dataset.pack_boards` / `dataset.unpack_board` 收发。`WORDS = ceil(CELLS/64)`。不许另立第二套布局。
5. **rollout 标签的列序是 `[负, 和, 胜]`。** 反了会训出一个和目标安静地负相关的模型。
6. **数据集按对局划分**，绝不按局面。
7. **策略 logits 在做 softmax 或 CE 之前，把已占格掩成 `-inf`。**
8. **改 `engine.W`/`H` 之前，必须重新生成所有数据集和快照。** 旧产物是静默失效的，不会报错。
9. **启发式已冻结，不要再改。** 配置在 `play.FROZEN_HEURISTIC`，构造用 `frozen_heuristic()`。它对自己的打法已经完备：1,056 个自对弈中局局面里找不到答错的局面，每次加规则都打平或更差。要提升棋力，去做学习路线或接外部棋谱。

## 2. 写入范围

同一个文件同时只能有一个写者。一个可行的划分：

| 范围 | 文件 | 典型改动 |
|---|---|---|
| 规则核心 | `engine.py`、`solvers.py`、`vcf.py`、`labeler.py` | 棋盘尺寸、检测器、搜索 |
| 数据 | `dataset.py`、`build_*.py`、`paths.py` | 格式、标签生成 |
| 模型 | `model.py`、`train*.py` | 结构、损失 |
| Agent | `play.py`、`baselines.py` | 搜索、各类 agent |
| 评估 | `baseline_duel.py`、`eval_three.py`、`bench_solver.py`、`restore_check.py` | 测量 |
| 界面 | `server.py`、`web/index.html`、`app.py` | 对弈界面 |
| 文档 | `README.md`、`manual.md`、`report.md`、`AGENT.md` | 这四份 |

文档分工：`manual.md` 讲**为什么**（原理、方法学、失败原因），**不放具体数字**；
`report.md` 放**所有数值**，每个数字都要能回答「在什么棋盘上、对什么对手/数据、用什么
方法测的」。新增测量写进 `report.md`，不要在 `manual.md` 里堆表格。

动 `engine.py` 是**全仓库事件**：先通告，改完之后把所有测试和所有数据构建都重跑一遍。

## 3. 验证

**没跑过就不算做完。** 任何改动的最低要求：

```bash
python code/tests.py            # 规则：2 万随机棋盘、两个检测器对拍
python code/test_solvers.py     # 强制胜检测 vs 暴力搜索
python code/test_value_target.py
python code/test_dataset.py     # 没有 data/_smoke.npz 会跳过
```

任何声称"更强"或"更准"的改动，还要真的打一局：

```bash
python code/baseline_duel.py 200 baseline      # 基线 vs 冻结启发式，配对换色
python code/eval_three.py 200                  # 多个模型互相打（需 ckpt/runs/）
```

**上表里的命令都必须真的能跑通。** 曾经有一版文档推荐了三个依赖已删除数据格式的
脚本（`depth_check.py`、`eval_value_by_ply.py`、`paired_eval.py`），它们全部运行即崩——
文档里写命令之前，先跑一遍。

一次 30 秒的冒烟构建很值：

```bash
python code/build_dataset.py --target 400 --out _smoke.npz --max-seconds 900
python code/train.py --data data/_smoke.npz --steps 300 --tag smoke
```

## 4. 命令分级：安全 / 昂贵 / 破坏性

**安全** —— 测试、`--help`、阅读、`bench_solver.py`、`push_hf.py --dry-run`。

**昂贵** —— 任何生成标签或训练的：
`build_dataset.py`（约 15 局面/秒）、`build_teacher_dataset.py`（12 进程约 2 分钟）、`train_policy.py`（数分钟）。这类一律放后台，**并且第一分钟内报速率**，不要盲等。

**对外发布** —— `code/push_hf.py` 会把权重和数据集传到公开的 Hugging Face 仓库，
不可撤销（只能覆盖）。跑之前先 `--dry-run` 看清单，并确认自己没有在未授权的情况下
转发第三方数据：`gomocup15.npz` 派生自 Gomocup 棋谱，该来源没有声明许可证。

**破坏性** —— `ckpt/` 或 `data/` 下的任何东西。删掉一个数据集或一份备份的 checkpoint 是不可逆的，并且会让所有引用过它的数字失效。**删之前先说清楚删什么、为什么。**

## 5. 约定

* 模块 docstring 解释**为什么**，不是"是什么"。看起来随意的常量，旁边写上产生它的那次测量。
* 热路径里不要有魔法数字——从 `engine` 或具名常量推导。
* 路径优先用 `paths.data(...)`、`paths.run_ckpt(...)`、`paths.baseline_dir(...)`，不要拼字符串。`--out` 只给文件名时会自动落进 `data/`。
* 各构建脚本的 CLI 保持一致（`--out`、`--seed`、`--workers`），这样生成一份新数据集只要一条命令。
* 不要擅自加依赖。界面只用标准库 + torch。

## 6. 已经咬过人的坑

报任何测量之前，先读 `manual.md` 第 9、10 节。摘要：

1. **确定性 agent 会让"N 局"变成 1 局。** 永远随机化开局、并做配对比较。
2. **小样本的偏差永远是偏乐观的。** 40–150 局三次给出了大效应，400–800 局时缩水。3σ 以下不要下结论。
3. **看清指标在量什么。** 用训练目标去评估同一个模型是自我实现的；有信息量的是增益出现的**位置**。
4. **裁判至少要和被评估的东西一样强。** 用 3 层检查去给 7 层标签打分，把所有 agent 低估了约 9 个点。
5. **用 `min`/`max` 聚合噪声估计会选出误差。** 在约 45 个样本上取极值，报出的是尾部，不是信号。
6. **检查相关系数的符号。** 负相关意味着列序或视角错了，不是"训练失败"。
7. **没有验收标准就不要声称修好了。** 搜索深度的验收是：候选着法里不同 Q 值 ≥ 20，**并且**配对胜率超过基线。

## 7. 协作

* 一个写范围一个 agent。动手前把范围登记到共享任务上。
* 顺序有意义：规则 → 数据格式 → 模型 → agent → 评估。改 `engine.py` 会阻塞下游一切，直到数据集重建完成。
* 当某个结果推翻了早先的结论，**在同一次改动里更新 `manual.md` 的第 9 或第 10 节**。bug 目录是这个仓库最有价值的资产。
* **失败就报失败。** 这个仓库里有几条最有用的记录都是没成功的测量：2 层 `min` 的塌缩、策略蒸馏的退步、模式评估的退步。**只有写下来它们才有价值。**
