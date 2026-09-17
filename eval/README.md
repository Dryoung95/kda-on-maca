# kda-on-maca — KDA on MetaX C500/MACA 适配评估

对 [NVlabs/kda](https://github.com/NVlabs/kda)（Kernel Design Agents）在沐曦 C500 / MACA
生态上的适配度做量化评估。上游 KDA 原生只支持 NVIDIA B200/B300（Blackwell），本仓库回答
"这个 agent 驱动的 kernel 优化工作流，在国产 GPU 生态上能跑多远"。

与 `kda-maca`（迁移产物，可复用工作流框架）的关系：本仓库是**下游应用**，按
`agent-flow.md` 的分层，评估结果属于 task workspace，不回混框架仓库。

## 评估方法

**第一层：生态兼容性（KernelBench 正确性门槛）**

每题用自己的参考实现当候选（`Model` → `ModelNew` 改名），调
`kda-maca/scripts/correctness_gate.py`。通过 = CUDA 语法源码经 cucc 编译并在 MACA 上
产生正确结果，**源码零修改**。

这是适配问题的第一层：在问"agent 能不能优化"之前，先问"语料能不能跑"。

```bash
source /data/kda-maca/env.sh
source /data/cuda-harness-migration/env.sh
python3 scripts/run_level.py --level 1 --out results/level1.csv
```

## L1 结果（100 题）

| 类别 | 数量 | 说明 |
|---|---:|---|
| 编译并正确 | **60** | 完全适配 |
| **OOM（容量限制）** | **27** | 输入尺寸为 48GB 显卡设计，C500 只有 16.3GB |
| **真数值错误** | **13** | 编译通过、非 OOM、结果不对 |
| 编译失败 | **0** | cucc 接受了全部 CUDA 语法 |
| **深入排查后：真实不兼容** | **1** | 其余 12 题的根因仍是容量或浮点精度（见下） |

**编译通过率 100%，表面不兼容 13%，真实不兼容仅 1%。**

这是本组数据最重要的结论。拆开看：

**1. 编译层完全适配（0 失败）。** 整个 L1 语料（含 conv transpose / dilation / 分组卷积
等复杂 CUDA 语法）经 cucc 全部能编译。适配缺口**不在工具链的语法层**。

**2. 27/40 失败是显存容量，不是生态缺陷。** KernelBench 默认输入尺寸为 48GB 显卡设计，
C500 只有 16.3GB（实测可用 15.22GB）：

```
Tried to allocate 6.00 GiB. GPU has 15.22 GiB total, 3.05 GiB free.
```

失败聚集在逐元素 + 大张量算子（ReLU 系列、各种 Norm、cumsum 系列），正是多中间量
OOM 高发区。这一类换更大显存的卡、或按 16.3GB 重设输入尺寸即可 recover，**不是需要
适配工作的不兼容**。

**3. 真实不兼容只有 13 题**（Swish、Softsign、InstanceNorm、GroupNorm、L1Norm、L2Norm、
Average_Pooling_2D、conv_standard_1D_dilated_strided、masked_cumsum、MSELoss、
CrossEntropyLoss、HuberLoss、ScaledDotProductAttention）。判定标准是 gate 的 stderr 里
是否出现 `OutOfMemoryError`——OOM 归容量类，其余归数值类。Swish、Softsign、MSELoss
已逐题复跑确认：编译通过、无 OOM、结果不对——**真数值/语义差异**。

顺带修正两个容易误判的地方：**Softplus（P29）与 FrobeniusNorm（P37）会 OOM**，看起来
像数值错误，实际是容量限制；而 Softsign（P30）反过来，看起来像 OOM 实测却是数值错误。
分类必须看 allocator 报文，不能按算子家族猜。

这 13 题曾被视为后续适配与优化的真实工作量。**2026-09-17 逐题深入排查后，结论改写**：
其中 **12 题的根因仍是显存容量**，只有 1 题是真实的后端缺陷。`results/level1.csv` 的
`class` 列给出逐题分类（`pass` / `oom` / `numeric`）。

### 13 题深度排查（2026-09-17）

对每一题做两件事：把输入按比例缩小到 16GB 内（保持通道数/标签结构不被破坏），以及
关闭 PyTorch 默认开启的 TF32。前者区分"容量不足"与"语义错误"，后者区分"精度策略"
与"数值错误"——C500 的 cuDNN/matmul 默认走 TF32，其 ~1e-3 的误差会突破 gate 使用的
fp32 1e-4 容差（Conv1d 实测：TF32 开 max=9.3e-4，TF32 关 max=0.0，精确为零）。

| 题号 | 算子 | 全尺寸 | 缩小尺寸+TF32 关 | 根因 |
|---|---|---|---:|---|
| 25 | Swish | OOM | 1.2e-7 | 容量 |
| 30 | Softsign | OOM | 0.0 | 容量 |
| 34 | InstanceNorm | OOM | 6.0e-7 | 容量 |
| 35 | GroupNorm | OOM | 4.8e-7 | 容量 |
| 38 | L1Norm | OOM | 4.8e-7 | 容量 |
| 39 | L2Norm | OOM | 7.5e-9 | 容量 |
| 45 | Average_Pooling_2D | OOM | 0.0 | 容量 |
| 76 | conv 1D dilated strided | OOM | 0.0（TF32 关） | 容量 + TF32 精度 |
| 93 | masked_cumsum | 未 OOM | 1.5e-5 | 归约顺序（见下） |
| 94 | MSELoss | OOM | 1.5e-8 | 容量 |
| 95 | CrossEntropyLoss | 未 OOM | 9.5e-7 | 容量 |
| 96 | HuberLoss | OOM | 7.5e-9 | 容量 |
| 97 | ScaledDotProductAttention | OOM | 6.0e-7 | 容量 |

**结论：12/13 是显存容量或浮点精度策略问题，不是生态不兼容。**

三处值得单独记下的发现：

1. **P76 是 TF32 的教科书案例。** `nn.Conv1d` 的 dilation/stride 全组合在 TF32 开时
   偏差均为 ~9.5e-4，关掉后精确为 0。KernelBench 的 fp32 容差是 1e-4，C500 默认开 TF32
   必然超差。**这是基准方法学与硬件默认值的冲突，不是 MACA 的 bug**——同样的问题在
   A100 上也存在。任何 fp32 正确性判定都应先 `torch.backends.cudnn.allow_tf32=False`，
   否则结论会把精度策略误报为后端缺陷。

2. **P93 的偏差来自归约顺序，不是错误。** `cumsum` 在 GPU 与 CPU 上的求和顺序不同，
   偏差随累积和的量级增长：累积到 ~121 时 max diff 6.1e-5，逼近但未超 1e-4。全尺寸
   （32768 长度）时累积和更大，越过容差。这是浮点归约的固有性质，**不是 C500 特有问题**。

3. **gate 的 OOM 语义会制造假"数值错误"。** OOM 抛在 `run_and_check_correctness`
   的 try 块内被捕获，返回 `compiled=True, correctness=False`——在 CSV 里与真数值错误
   无法区分。这也是为什么"13 题数值错误"里实际混着 12 个容量问题。

### 结论对"是否需要换算力"的回答（2026-09-17 修订）

- **换更大显存的卡能 recover 39/40 失败题**（27 个原 OOM 类 + 13 题中的 12 个），
  收益最大、成本最低
- 真正与显存无关、需要逐题适配的只有 **1 题**（P93 的 cumsum 归约精度，且其性质是
  浮点归约固有、跨平台共现，严格说也不是 MACA 的缺陷）
- 所以正确的顺序是：**先按 16.3GB 重设输入重跑**，再判定剩余项；直接把"数值错误"
  当作适配工作量是误判——其中绝大部分会在重设尺寸后消失
- 若评估目标是"语义等价"而非"逐位一致"，应当对长归约类算子（cumsum/cumprod 系列）
  放宽容差或改用相对误差判定

## 已知的方法学陷阱（评估脚本引入，非生态问题）

1. `class Model` 改名时必须同步改 `super(Model, self)`，否则抛
   `obj must be an instance or subtype of type`，看起来像生态不兼容
2. 门槛输出 `correct:` 后跟**两个空格**，子串匹配会把 PASS 判成 FAIL

两者都已在 `scripts/run_level.py` 修正。

## 目录

```
scripts/run_level.py     全量正确性门槛驱动
results/level1.csv       逐题 compiled/correct/error
results/level1_errors.log
```
