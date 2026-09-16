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

**编译通过率 100%，真实不兼容率仅 13%。**

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

**3. 真实不兼容只有 13 题**（Swish、Softplus、InstanceNorm、GroupNorm、FrobeniusNorm、
L1Norm、L2Norm、Average_Pooling_2D、conv_standard_1D_dilated_strided、masked_cumsum、
MSELoss、CrossEntropyLoss、HuberLoss、ScaledDotProductAttention）。已抽查确认 Swish 与
MSELoss 非 OOM、编译通过、结果不对——**真数值/语义差异**。

这 13 题才是后续适配与优化的真实工作量。

### 结论对"是否需要换算力"的回答

- **换更大显存的卡**能直接 recover 27 题（OOM 类），收益最大、成本最低
- 剩余 13 题与显存无关，**换卡解决不了**，需要在 C500/MACA 上逐题排查数值差异
- 所以顺序应当是：先按 16.3GB 重设输入重跑（区分"容量"与"真不兼容"），再针对 13 题
  做适配；而不是直接换卡了事

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
