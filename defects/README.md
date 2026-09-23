# MACA 3.3.0 工具链缺陷报告（MetaX C500 / xcore1000）

**日期**：2026-09-22（2026-09-23 复核更正）
**环境**：MetaX C500（104 SM，16.3 GB，warp size 64），MACA SDK 3.3.0.15，驱动 3.3.0.4，
mxcc 1.0.0 + cu-bridge (cucc)，PyTorch 2.8.0+metax3.3.0.2
**用途**：以下五个缺陷经隔离实验逐条确证，**只能由沐曦官方在 SDK 中修复**——
下游用户无法绕过（见每项的"为什么下游无法绕过"）。每个缺陷的复现程序在
`defects/repro/`，全部已在本机用 `cucc` 真编译真运行通过，并经一次
**独立复核**逐条重跑（见文末"复核记录"）。

**上报前必读**：早期版本有四条错误结论，已在复核中修正——**D2 是假阳性
（已删除）**、D3 根因不是 lane 映射、D1 的"行数×8"统计不成立、
D4 的 `-fno-inline` 规避无效。请以本版为准。

**总影响**：mctlass device 层 GEMM 在所有尺寸下输出**静默错误结果**
（不崩溃、不报错、返回 `Status::kSuccess`）；wmma 的 `load_matrix_sync`
无法用于正确计算，迫使用户退回标量填充路径，手写 tensor-core GEMM
的 258 个配置无一超过 eager。

---

## 摘要表

| # | 缺陷 | 组件 | 严重度 | 静默？ | 复现 |
|---|---|---|---|---|---|
| D1 | mctlass device GEMM epilogue 每个 16 行 tile 漏写 8 行 | mctlass | **高（数据损坏）** | **是** | `mt3sweep.cu` |
| D3 | `wmma::load_matrix_sync` K 轴列侧数据错位 | wmma | **高（数据损坏）** | **是** | `diag_load.py` |
| D4 | mxcc `-O2`+ 优化级别导致 mctlass host 端 segfault | mxcc | 高（崩溃） | 否（崩溃） | `mt3bisect.cu` |
| D5 | wmma fp32/TF32 fragment 未定义 | wmma | 中（功能缺失） | 是（编译期） | `wmma_fp32_probe.cu` |
| D6 | warp shuffle 把 64-lane warp 当作两个 32-lane 子组 | runtime | 中（归约结果翻倍） | **是** | `warp_probe.cu` |

其中 **D1、D3、D6 是最危险的**：程序返回成功、结果完全错误。

> **已删除的 D2（2026-09-22 复核）**：原报告称 `store_matrix_sync` 忽略
> layout tag。**该结论为假阳性**：原复现 `diag_store2.py` 把整数常量 **16 和 17**
> 强转成 `layout_t`，而 SDK 的枚举是
> `enum layout_t { mem_row_major, mem_col_major };`（即 **0 和 1**）。
> 16/17 是未定义行为，两个越界值落到同一分支，看起来正像"tag 被忽略"。
> 用合法枚举值（0/1）重测：**row_major 与 col_major 的输出互为精确转置
> （256/256），tag 完全生效**（`/tmp/t/diag_store3.py`、`diag_fmap2.py`）。
> store_matrix_sync **没有缺陷**。

---

## D1 — mctlass device GEMM epilogue 每个 16 行 tile 漏写 8 行

**组件**：`mctlass`（MACA 的 CUTLASS 对应物），`gemm/device/gemm.h` 的 SIMT epilogue

### 现象

单位矩阵测试 `C = A @ I`（`C` 先清零）。**坏行恰好是 `r % 16 ∈ [8,15]`**——
每个 16 行 tile 的后半 8 行没被写（`/tmp/t/mt3rows.cu`，RowMajor 主算子，
pristine 头文件，`-O0 -fno-inline`）：

```
sq64   m=64 n=64 k=64   bad rows: 8..15 24..31 40..47 56..63
   bad rows by r%16: [0 0 0 0 0 0 0 0 4 4 4 4 4 4 4 4]
sq128  m=128 n=128 k=128  bad rows: 8..15 24..31 ... 120..127
   bad rows by r%16: [0 0 0 0 0 0 0 0 8 8 8 8 8 8 8 8]
```

坏行占比恒为 **0.500**。逐元素统计（`defects/repro/mctlass/mt3sweep.cu`）：

```
square 128     m=128 n=128 k=128 mismatched= 584 (of 16384) firstBad=(r=8 c=8)
square  64     m= 64 n= 64 k= 64 mismatched=  64 (of  4096) firstBad=(r=8 c=8)
square 256     m=256 n=256 k=256 mismatched=2080 (of 65536) firstBad=(r=8 c=8)
nonsquare A    m=128 n= 64 k= 96 mismatched= 254 (of  8192) firstBad=(r=8 c=8)
nonsquare B    m= 64 n=128 k= 96 mismatched= 288 (of  8192) firstBad=(r=8 c=8)
tall           m=200 n= 64 k= 64 mismatched= 272 (of 12800) firstBad=(r=8 c=8)
```

> **一处数字更正（2026-09-22 复核）**：本报告早期版本称"错误总数恒为行数×8"。
> 实测元素错误占比在 0.125–1.016 之间，**不恒定**。但结构性结论成立：
> **每个 16 行 tile 漏写后半 8 行**（`r%16 ∈ [8,15]`），与形状无关。

未写入的行保留 `C` 的初始值。**若 `C` 未清零，读到的是旧数据——看起来像
"未初始化的垃圾"，实际是 epilogue 根本没写那些行**（此前一处误判正是因此产生）。
RowMajor 与 ColumnMajor 特化**都**受损。

程序返回 `Status::kSuccess`（`init=0 run=0`），没有任何错误信号。

### 根因（已定位到代码行）

mctlass 的 epilogue 线程映射按 **32 线程 warp** 硬编码，而 C500 的 warp 是 64：

- `mctlass/gemm/warp/mma.h` 把 `WarpSize::value` 改成了 64（正确）
- 但 `epilogue/threadblock/default_thread_map_simt.h:73`、
  `epilogue/threadblock/output_tile_thread_map.h:202,227` 的 `RowArrangement` 里
  仍是硬编码的 `kWarpSize = 32`

行 0–7 正确、行 8 起错误，正好对应 32 线程映射在 64 线程 warp 下的
**前半 warp 正确、后半 warp 越界**。我们尝试过把这三处改成引用
`WarpSize<arch::OpClassSimt>::value`，`kThreads` 匹配了（512=512），但行轴迭代计数
仍不一致（`Iterations::kRow = 1` vs `Count::kRow = 4`），且任何进一步修改都触发
`memory violation(0x4)`——说明 `FragmentIteratorSimt` 与 warp 级
`TileIteratorSimt` 的 smem 偏移也按 32 线程 warp 假设。**盲改风险高，需官方统一修复。**

### 为什么下游无法绕过

epilogue 是 mctlass 内部的输出阶段，用户只能选 layout 特化、不能替换
epilogue 的线程映射。绕过它意味着不用 mctlass device 层。

### 已验证的正确替代路径

**mcblas**（`libmcblas`）正确且已调优：fp32 = 4747 µs、bf16 = 812 µs、
fp16 = 960 µs（n=4096 方阵乘法）。**不要把 mctlass device GEMM 当作可行路径，
除非先修好 D1。**

---

## D2 — `wmma::store_matrix_sync` 忽略 layout tag

---

## D3 — `wmma::load_matrix_sync` 在 K 轴列侧数据错位

**组件**：`wmma`（`__clang_maca_mma_functions.h`）

### 现象

共享内存 tile 填单位阵（sA）与行主序 0..255（sB），`acc` 初始化为 1000，
跑 `acc = A @ B`（`defects/repro/diag_load.py`，B 为单位阵，故 acc 应等于 B）：

```
row 0: [1000 1001 1002 ... 1015]      ← 完全正确
row 1: [1016 1017 ... 1031]           ← 完全正确
row 2: [1032 1033 ... 1047]           ← 完全正确
row 3: [1048 1049 ... 1063]           ← 完全正确
row 4: [994 995 1001 999 998 997 ...] ← 垃圾！
row 5: [999 1002 1002 1001 1007 ...]  ← 垃圾
```

纯 load→store 往返（不涉及 mma）进一步定位：**每行的前 4 列正确，其后全垃圾**
（`/tmp/t/wmma_rt2.cu`，`src[r][c] = 256r+c`），且垃圾幅度随 K 索引增长。

### 根因（2026-09-22 复核更正）

**原报告把根因归为"fragment 布局按 warp 32 展开，lane ≥ 32 拿错元素"——该归因不成立。**

读头文件源码（`__clang_maca_mma_functions.h:386–403`），load 的地址映射是
`row = lane & 0xf; col = (lane>>4) << 2`。用 C++ 静态计算，64 个 lane 覆盖
64/64 个 (row, col) 锚点，**零重复零越界**。

决定性实验（`/tmp/t/wmma_iso.cu`、`/tmp/t/wmma_half.cu`）：给每个 lane 的
A-fragment 填 `1000+lane`、B 填 1，跑 `mma_sync`——输出行 r 的值精确等于 `1000+r`，
即 lane l 的数据落在行 l；两半测试（low=1/high=2 与 low=2/high=1）两种模式
都给出 **24 = 8×1+8×2**，说明**前后各 32 个 lane 都正确贡献了**，
`mma_sync` 与 lane 映射本身没有 warp-32 假设。

**真实错位在 K 轴列侧**：每行前 4 列正确、其后全垃圾，垃圾随 K 索引增长。
这是 16x16x16 fragment 在 K 维度上的数据错位，不是行侧的 lane 映射问题。

### 补充观察

- 加 padding（ldm = 24 而非 16）**不能修复**（`diag_load_store.py`：两个 ldm 都失败）
- b fragment 的 `row_major`/`col_major` 标签行为互换（col_major 实际按行读）

### 为什么下游无法绕过

fragment 只能通过 `load_matrix_sync` 或逐元素标量填充得到。手写标量填充确实可行
（我们的 `kernels_wmma6.py` 用它达到全尺寸 rel = 0.00000），但**性能只有
0.233×（6835 µs vs eager 1592 µs）**——标量填充的指令吞吐把性能钉死。
要打败 eager 必须有正确的向量化 load。这是 258 个手写 wmma 配置全部失败的根因。

### 什么没坏（供官方参考）

- `__builtin_mxc_mma_16x16x16f16` intrinsic 本身**正确**（`/tmp/t/wmma_iso.cu`）
- fragment 的存储布局正确
- `fill_fragment` / `mma_sync` 正确
- `store_matrix_sync` 正确（见已删除的 D2）

---

## D4 — mxcc `-O2` 及以上优化级别导致 host 端 segfault

**组件**：mxcc（经 cu-bridge 调用）

> **2026-09-22 复核并更正**：本节原标题为"inlining 导致"，二分表称
> "`-O2 -fno-inline` 正常、只有 `-fno-inline` 能修复"。**该结论在当前 SDK
> （MACA 3.3.0.15 / mxcc 1.0.0）上不可复现**。重新隔离测试后真实规律见下，
> 与 inlining 无关。更正后的实验细节见本文末"复核记录"。

### 现象

`mctlass::gemm::device::Gemm` 对象在 host 端构造时，`-O2` 及以上优化级别
segfault（`defects/repro/mctlass/mt3bisect.cu`，3 次重编译重跑，结果稳定）：

```
$ cucc mt3bisect.cu -O2 ... -o t_O2
$ ./t_O2
Segmentation fault (core dumped)      ← exit code 139
```

二分定位（**复核后**，`mt3bisect.cu` 与一个栈上构造的最小程序均一致）：

| 编译选项 | 结果 |
|---|---|
| `-O0` / `-O1` / 不带任何 `-O` | 正常 |
| `-O2` | **segfault** |
| `-O3` / `-Os` | **segfault** |
| `-O2 -fno-inline` | **segfault（不修复）** |
| `-O2 -fno-inline-functions` | segfault |
| `-O2 -fno-elide-constructors` | segfault |
| `-O2 -fno-omit-frame-pointer` | segfault |
| `-O2 -fno-vectorize` | segfault |
| `-O2 -fno-rtti` | segfault |
| `-O2 -g` | segfault |

**触发条件是优化级别 ≥ 2，不是 inlining。** `-fno-inline` 及其同族标志
全部无效。纯 `mctlass/mctlass.h` + `gemm/device/gemm.h` 的最小程序即可复现，
与 PyTorch 无关。

### 一处与上述相反、尚未解释的观察

在 torch `load_inline` 构建路径上（`optloop/mctlass_base.py`），ninja 传给
cucc 的命令行**不含任何 `-O` 标志**，但去掉 `-fno-inline` 会**稳定 core dump**，
加上则**稳定正常**（两次独立隔离构建，各 2/2 复现）。也就是说同一个 flag 在
"直接 cucc 编译"与"经 torch 扩展编译"两条路径上效果**相反**。具体机制未定位，
但这本身是应当上报的缺陷证据。

### 为什么下游无法绕过

这不是静默错误（会崩溃）。当前可用的规避只有 **`-O0` 或 `-O1`**，代价是放弃
全部优化。`optloop/mctlass_base.py` 中的 `-fno-inline` **必须保留**——在该路径上
它是有效的，移除会导致 core dump。

**仍需官方修复**：`-O1` 以下对性能 kernel 代价过大。
---

## D5 — wmma fp32 / TF32 fragment 未定义

**组件**：`wmma`（`__clang_maca_mma_functions.h`）

### 现象

fp32 输入的 wmma fragment 无法编译（`defects/repro/wmma_fp32_probe.cu`）：

```
wmma_fp32_probe.cu:7:68: error: implicit instantiation of undefined template
  'mxmaca::wmma::fragment<mxmaca::wmma::matrix_a, 16, 16, 16, float, mxmaca::wmma::row_major>'
wmma_fp32_probe.cu:8:68: error: implicit instantiation of undefined template
  'mxmaca::wmma::fragment<mxmaca::wmma::matrix_b, 16, 16, 16, float, mxmaca::wmma::col_major>'
```

MACA 只实例化了 `__half` / `maca_bfloat16` 输入 + `float` accumulator 的
16x16x16（及 32x8x16 / 8x32x16）。**fp32 与 TF32 fragment 路径不存在。**

### 影响

C500 上不存在 TF32 tensor-core 路径。fp32 GEMM 只能走 SIMT（标量）或
先转 fp16/bf16 再用 tensor core。这决定了第二阶段性能调优的上限：

| 路径 | n=4096 用时 | vs eager fp32 (1592 µs) | 备注 |
|---|---|---|---|
| eager fp32（实为 TF32） | 1592 µs | 1.00× | 要打败的目标 |
| mcblas fp16-in / fp32-acc | **951 µs** | **1.693×** | 唯一过 1e-4 精度且 >1× |
| mcblas bf16 | 812 µs | 1.96× | err 3.8e-3，超 1e-4 容差 |
| mcblas fp32 直入 | 4743 µs | 0.34× | 打不过 TF32 eager |
| 手写 wmma（正确但标量填充） | 6835 µs | 0.233× | 受 D3 限制 |

---

## D6 — warp shuffle 把 64-lane warp 当作两个 32-lane 子组

**组件**：MACA runtime 的 warp shuffle

### 现象

`__shfl_down_sync(mask, v, 32)`：lane 0 拿到 lane 32 的值（方向对），
lane 32 拿到**自己的值**。单看这一点，"复制而非交换"的解释是自然的——
但**它恰好符合 CUDA 语义**（源 lane 越界时返回自身值），所以单凭
`shfl_down` 不足以证明缺陷。

**决定性证据在 `__shfl_sync`**（`/tmp/t/shfl_decisive.cu`）：

```
lane  0: v=1000  d32=1032  u32=1000  idx32=1032
lane  1: v=1001  d32=1033  u32=1001  idx32=1032
lane 31: v=1031  d32=1063  u32=1031  idx32=1032
lane 32: v=1032  d32=1032  u32=1000  idx32=1032
lane 33: v=1033  d32=1033  u32=1001  idx32=1032
lane 63: v=1063  d32=1063  u32=1031  idx32=1032
```

`__shfl_sync(mask, v, 32)` 在**所有 64 个 lane** 上都返回 **1032**（lane 32 的值）。
CUDA 语义下它应当做 lane→lane 的置换：lane 0 得 lane 32 的值（1032），
lane 32 得 lane 0 的值（1000）。**MACA 把 64-lane warp 当成两个 32-lane 子组，
在子组内按 (lane % 32) 广播**，跨子组的索引被折叠。

### 后果：6 轮归约结果翻倍

```
5-round shfl_down: 2016   6-round shfl_down: 4032   6-round shfl_sync: 2016
（正确答案 2016 = 0+1+...+63）
```

4032 = 2016 × 2：high half 的值被加两次。**5 轮归约在 C500 上反而是正确的**，
因为它从不跨越 offset 32；`shfl_sync` 的显式归约也给出 2016。
这与直觉相反，也与 CUDA 语义不一致。

### 影响范围

`skills/c500-kernel-wiki/wiki/hardware/warp64-implications.md` 原本把
`offset = warpSize/2`（6 轮）标为 RIGHT、5 轮标为 WRONG，本次实测后已修正。

任何从 NVIDIA 代码搬来的"标准"warp 归约，只要轮数按 `warpSize` 推导，
在 C500 上都会静默翻倍。若后续 SDK 修复成真正的置换语义，5 轮写法又会变成
丢一半数据的错误写法——**该写法危险，依赖 SDK 版本**。

### 规避

用 5 轮，或用 `__shfl_sync` 到目标 lane 的显式归约（实测给出正确的 2016），
或显式两阶段：两个 32-lane 子组内各自归约，再做一次跨子组的显式交换。

---

## 环境与构建命令

```bash
source /data/kda-maca/env.sh          # MACA_PATH、cucc 到 PATH、$MACA_CUCC_FLAGS

# D1：mctlass epilogue（.cu，pristine 头文件）
cd defects/repro/mctlass
cucc mt3sweep.cu -O0 -fno-inline -std=c++17 \
     -I/opt/maca-3.3.0/tools/cu-bridge/include \
     -I/opt/maca-3.3.0/include $MACA_CUCC_FLAGS -o mt3sweep
./mt3sweep                        # 元素级 mismatch 统计
cucc mt3rows.cu -O0 -fno-inline -std=c++17 ... -o mt3rows && ./mt3rows
                                 # 坏行 r%16 直方图（后半 8 行全坏）

# D6：warp shuffle 子组语义
cucc ../warp/shfl_decisive.cu -O2 -std=c++17 \
     -I/opt/maca-3.3.0/tools/cu-bridge/include $MACA_CUCC_FLAGS -o shfl_decisive
./shfl_decisive                   # idx32 全 64 lane 返回 1032；2016 vs 4032

# D4：-O2 segfault（exit 139）
cucc mt3bisect.cu -O2 ... -o t_O2 && ./t_O2          # segfault
cucc mt3bisect.cu -O2 -fno-inline ... -o t_ok && ./t_ok   # 仍 segfault

# wmma 复现（PyTorch load_inline，需 KernelBench 环境）
source /data/cuda-harness-migration/env.sh
cd defects/repro/wmma
python3 diag_load_store.py       # D3（纯 load→store 往返，4/256 正确）
python3 diag_store3.py           # D2 反证：合法枚举值下两 tag 互为转置
python3 diag_fmap2.py            # D2 反证：(lane,elem) 逐项 256/256 转置
```

## 复现文件清单

| 缺陷 | 文件 | 说明 |
|---|---|---|
| D1 | `defects/repro/mctlass/mt3sweep.cu` | 元素级统计，6 种尺寸 |
| D1 | `defects/repro/mctlass/mt3rows.cu` | 坏行 `r%16` 直方图（复核新增） |
| D3 | `defects/repro/wmma/diag_load.py` | 身份矩阵测试 |
| D3 | `defects/repro/wmma/diag_load_store.py` | 纯 load→store 往返 |
| D3 | `defects/repro/wmma/wmma_rt2.cu` | 每行前 4 列正确、其后垃圾（复核新增） |
| D3 根因排除 | `defects/repro/wmma/wmma_iso.cu`、`wmma_half.cu` | lane 映射与 mma 本身正常（复核新增） |
| D2 反证 | `defects/repro/wmma/diag_store3.py`、`diag_fmap2.py` | 合法枚举值下 tag 生效（复核新增） |
| D4 | `defects/repro/mctlass/mt3bisect.cu` | `-O2` vs `-O0` 二分 |
| D5 | `defects/repro/wmma/wmma_fp32_probe.cu` | 预期编译失败 |
| D6 | `defects/repro/warp/shfl_decisive.cu` | 决定性证据：`shfl_sync` 全 lane 同值（复核新增） |
| D6 | `defects/repro/warp/warp_red2.cu`、`warp_probe.cu` | 归约 2016 vs 4032 |
| mcblas 对照 | `defects/repro/mctlass/mcblas_ok.cu` | N=256 mismatched=0（复核新增） |

`-I/opt/maca-3.3.0/tools/cu-bridge/include` 不可省略，否则
`fatal error: '__macro_mxcc.h' file not found`。

## 修改过的 SDK 头文件（当前状态）

**已全部回退，SDK 处于 pristine 状态**（2026-09-22 核实：两个头文件与
`/tmp/orig_*.h` 备份 MD5 完全一致）。上述 D1 的实验性修改曾做过，
但已恢复，D1 的复现数据全部是在 pristine 头文件下重新编译测得的。

历史修改记录（仅供官方参考）：
- `epilogue/threadblock/default_thread_map_simt.h:73`
- `epilogue/threadblock/output_tile_thread_map.h:202, 227`
（把 `kWarpSize = 32` 改为 `WarpSize<arch::OpClassSimt>::value`。
修正后 `kThreads` 匹配但行轴迭代计数仍不一致，进一步修改触发
memory violation，故回退。）

原始备份：`/tmp/orig_default_thread_map_simt.h`、
`/tmp/orig_output_tile_thread_map.h`、`/tmp/orig_predicated_tile_iterator_params.h`。

---

## 2026-09-22 独立复核记录

全部 6 项缺陷在当前环境（MACA 3.3.0.15 / mxcc 1.0.0 / C500）上重新编译并运行。

**逐项复现结果：**

| # | 复现状态 | 备注 |
|---|---|---|
| D1 mctlass epilogue 漏写 8 行 | ✅ 复现，**统计已更正** | 坏行恰为 `r%16 ∈ [8,15]`，占比恒 0.500；元素错误占比不恒定（0.125–1.016），早期"行数×8"的说法已删除。返回 `Status::kSuccess`，无错误信号。 |
| D2 store 忽略 layout tag | ❌ **假阳性，已删除** | 原复现把 16/17 强转成 `layout_t`（合法值只有 0/1）。用真枚举重测：两个 tag 输出互为精确转置（256/256），tag 完全生效。 |
| D3 load 数据错位 | ✅ 复现，**根因已更正** | 现象属实（前 4 列正确、其后垃圾）。但"按 warp 32 展开 / lane≥32 拿错"不成立：lane 映射零重复零越界，两半 lane 都正确贡献。真因是 **K 轴列侧错位**。 |
| D4 `-O2` host segfault | ✅ 复现，**规避手段已更正** | `-O2` 崩溃属实，但 `-fno-inline` **不修复**（报告早期版本声称有效）。真实触发条件是优化级别 ≥ 2。 |
| D5 fp32/TF32 fragment 未定义 | ✅ 逐字复现 | 编译错误属实。准确表述是**功能缺失**：`accumulator` 的 fp32 特化存在，缺的是 `matrix_a`/`matrix_b` 的 fp32 特化（无 TF32/tensor-core fp32 路径）。 |
| D6 warp shuffle 子组语义 | ✅ 复现，**归因已更正** | 2016 vs 4032 属实。但"复制而非交换"恰好符合 CUDA 语义；决定性证据是 `__shfl_sync(_, v, 32)` 在全部 64 lane 返回同一值，归因应为"64-lane warp 被当作两个 32-lane 子组"。 |

**已核实的旁证：**
- SDK 三处头文件确认处于 pristine 状态（与 `/tmp/orig_*.h` 的 MD5 完全一致）。
- `mcblas fp16-in/fp32-acc = 951µs = 1.693×` 与 `submit-mcblas-result.json`
  （ref 1.61ms / new 0.951ms）对上；`bf16 = 812µs`、`fp32 = 4743µs` 与
  `mcblas-results.jsonl` 对上。`mcblasSgemm` N=256 **mismatched=0**——
  "用 mcblas 规避 D1/D3"的建议可靠。
- 手写 wmma 最佳 `mm_wmma3_128_256_16_4_4 = 2333µs = 0.682×`，
  `longloop2.log` 记录 258/258 correct 且无一超过 eager。

**"250 题只有 1 道 >1×" 这句表述需要澄清（上报时务必改写）：**
`batch-results.jsonl` 是**纯正确性**记录（250 行，无 speedup 字段，
182 passed / 29 failed / 39 skipped）。"只有 1 道 >1×"并非 250 题的全量性能
评测结果，而是：L1 P1 是唯一做过**性能提交**并通过的题目（1.693×），
其余题目**尚未做性能提交**。写成"评测得出"会误导官方。

**本次复核修正的四点：**
1. **D2 删除**：假阳性，根因是复现程序强转了非法枚举值。
2. **D3 改根因**：不是 lane 映射，是 K 轴列侧错位；`mma_sync` 与 lane 映射本身正常。
3. **D1 改统计**：结构性结论（每 tile 漏后半 8 行）成立，但"错误总数恒为行数×8"错误。
4. **D4 改规避**：`-fno-inline` 在纯 cucc 路径上不修复 segfault；真实边界是优化级别 ≥ 2。
   注意 `optloop/mctlass_base.py` 里的 `-fno-inline` **仍必须保留**——
   在 torch `load_inline` 路径上它确实有效，两条路径行为相反，本身值得上报。
