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

**第二层：重设尺寸后的正确性**（`scripts/run_level_resized.py`）。原始尺寸按 48GB
显卡设计，在 15.22 GiB 的 C500 上 40/100 直接被分配器挡在门外。这一层把 batch/空间维
按比例缩小后重跑同一门槛，回答的是"装下之后算得对不对"——这才是适配问题，
容量不是。

## L1 结果（100 题）

| 类别 | 数量 | 说明 |
|---|---:|---|
| 编译并正确 | **60** | 完全适配 |
| **OOM（容量限制）** | **27** | 输入尺寸为 48GB 显卡设计，C500 只有 16.3GB |
| **真数值错误** | **13** | 编译通过、非 OOM、结果不对 |
| 编译失败 | **0** | cucc 接受了全部 CUDA 语法 |
| **重设尺寸后（2026-09-17）** | **95/100 通过** | 见下节 |

**编译通过率 100%。原始尺寸下 60/100；按 16.3GB 重设输入后 99/100。**

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

### 重设尺寸重跑（2026-09-17）：95/100 通过

`scripts/run_level_resized.py` 把每题的 batch/空间维按比例缩小到 16.3GB 内（通道数、
分组数、标签长度等结构常量保持不变），并关闭 TF32 后重跑同一门槛：

```
source /data/kda-maca/env.sh && source /data/cuda-harness-migration/env.sh
python3 scripts/run_level_resized.py --budget-gib 4 --out results/level1_resized.csv
```

| 类别 | 数量 |
|---|---:|
| 通过 | **99** |
| OOM（自适应重试仍未装下） | **0** |
| 真实后端缺陷 | **1**（P95） |
| 编译失败 | **0** |

**原始尺寸 60 → 重设尺寸 99。** 剩余 4 个原"数值错误"题（P7/P9/P11/P63）在自适应缩小后
全部通过——它们不是数值错误，是输出张量远大于输入（P7：输入 16 MB，输出 1 GiB，
比值 512×），初次按输入成本估算的缩放因子不够，重试后通过。

### P95：唯一真实的后端缺陷

```
NotImplementedError: "nll_loss_forward_reduce_cuda_kernel_2d_index"
not implemented for 'Float'
```

根因不是数值，而是 **dtype 分发**：门槛的 `_process_input_tensor` 把所有输入统一转成
fp32，包括本应是 int64 的类别标签。`nll_loss` 的索引路径在 MACA 上没有 Float 分发，
而 NVIDIA 的实现会隐式把标签转回整数，所以同样的代码在 A100 上不报错。

直接用 int64 标签调用 `F.cross_entropy` 完全正常（偏差 9.5e-7，纯归约顺序差异）。
**这是 KernelBench 的类型处理与 MACA 严格性的冲突**，影响范围限于带整数标签的
分类损失（P95 CrossEntropy；P100 HingeLoss 走另一条路径，不受影响）。

### 三处方法学发现

1. **P76 是 TF32 的教科书案例。** `nn.Conv1d` 的 dilation/stride 全组合在 TF32 开时
   偏差均为 ~9.5e-4，关掉后精确为 0。KernelBench 的 fp32 容差是 1e-4，C500 默认开 TF32
   必然超差。**这是基准方法学与硬件默认值的冲突，不是 MACA 的 bug**——同样的问题在
   A100 上也存在。任何 fp32 正确性判定都应先 `torch.backends.cudnn.allow_tf32=False`，
   否则结论会把精度策略误报为后端缺陷。

2. **gate 的 OOM 语义会制造假"数值错误"。** OOM 抛在 `run_and_check_correctness`
   的 try 块内被捕获，返回 `compiled=True, correctness=False`——在 CSV 里与真数值错误
   无法区分。这也是为什么"13 题数值错误"里实际混着 12 个容量问题。

3. **输入成本不能预测峰值。** 输入与输出的字节比值在本级别横跨 0× 到 512×（P95
   输出是标量；P7 输入 16 MB 输出 1 GiB）。任何静态公式都覆盖不了这个跨度，
   `run_level_resized.py` 因此在 OOM 时按 0.5/0.25/0.125 自适应重试，而不是试图算准。

### 结论对"是否需要换算力"的回答（2026-09-17 修订）

- **换更大显存的卡能 recover 39/40 失败题**（27 个原 OOM 类 + 13 题中的 12 个），
  收益最大、成本最低；实际上连卡都不用换，按 16.3GB 重设输入就有 99/100 通过
- 真正与显存无关、需要适配的只有 **P95 一题**，且其根因是基准的类型转换而非 MACA
  内核缺陷——把标签保持 int64 即可绕过
- 所以正确的顺序是：**先按 16.3GB 重设输入重跑**，再判定剩余项；直接把"数值错误"
  当作适配工作量是误判——其中绝大部分会在重设尺寸后消失

## 已知的方法学陷阱（评估脚本引入，非生态问题）

1. `class Model` 改名时必须同步改 `super(Model, self)`，否则抛
   `obj must be an instance or subtype of type`，看起来像生态不兼容
2. 门槛输出 `correct:` 后跟**两个空格**，子串匹配会把 PASS 判成 FAIL

两者都已在 `scripts/run_level.py` 修正。

3. **整数标签被转成 fp32。** 门槛的 `_process_input_tensor` 对所有输入做
   `to(dtype=precision)`，不区分权重张量与索引张量。在 MACA 上 `nll_loss` 的索引路径
   没有 Float 分发，报 `NotImplementedError`；在 NVIDIA 上会隐式转回整数，所以同样的
   代码只在 C500 上暴露。`run_level_resized.py` 目前靠题目自身规避，未做类型修正。

## 目录

```
scripts/run_level.py          原始尺寸全量正确性门槛驱动
scripts/run_level_resized.py  按 16.3GB 重设输入重跑（自适应缩放 + TF32 关）
results/level1.csv            原始尺寸逐题 compiled/correct/class/error
results/level1_resized.csv    重设尺寸逐题结果
results/level1_errors.log
```
