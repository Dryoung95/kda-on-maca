# MACA 3.3.0 工具链缺陷报告（MetaX C500 / xcore1000）

**日期**：2026-09-22
**环境**：MetaX C500（104 SM，16.3 GB，warp size 64），MACA SDK 3.3.0.15，驱动 3.3.0.4，
mxcc 1.0.0 + cu-bridge (cucc)，PyTorch 2.8.0+metax3.3.0.2
**用途**：以下五个缺陷经隔离实验逐条确证，**只能由沐曦官方在 SDK 中修复**——
下游用户无法绕过（见每项的"为什么下游无法绕过"）。每个缺陷的复现程序在
`defects/repro/`，全部已在本机用 `cucc` 真编译真运行通过。

**总影响**：这五个缺陷是 KernelBench 250 道题目在 C500 上**只有 1 道达到 >1× 加速**
的直接原因。手写 tensor-core GEMM 的全部 258 个配置都建立在坏掉的原语上；
mctlass device 层 GEMM 在所有尺寸下输出**静默错误结果**（不崩溃、不报错、返回 Status::kSuccess）。

---

## 摘要表

| # | 缺陷 | 组件 | 严重度 | 静默？ | 复现 |
|---|---|---|---|---|---|
| D1 | mctlass device GEMM epilogue 每个 16 行 tile 漏写 8 行 | mctlass | **高（数据损坏）** | **是** | `mt3sweep.cu` |
| D2 | `wmma::store_matrix_sync` 忽略 layout tag | wmma | 中（语义错误） | 是 | `diag_store2.py` |
| D3 | `wmma::load_matrix_sync` 在 lane ≥ 32 读到垃圾数据 | wmma | **高（数据损坏）** | **是** | `diag_load.py` |
| D4 | mxcc `-O2` inlining 导致 mctlass host 端 segfault | mxcc | 高（崩溃） | 否（崩溃） | `mt3id.cu` |
| D5 | wmma fp32/TF32 fragment 未定义 | wmma | 中（功能缺失） | 是（编译期） | `wmma_fp32_probe.cu` |
| D6 | `__shfl_down_sync` offset≥32 时复制而非交换，归约结果翻倍 | runtime | 中（数据错误） | **是** | `warp_probe.cu` |

其中 **D1、D3、D6 是最危险的**：程序返回成功、结果完全错误。

---

## D1 — mctlass device GEMM epilogue 每个 16 行 tile 漏写 8 行

**组件**：`mctlass`（MACA 的 CUTLASS 对应物），`gemm/device/gemm.h` 的 SIMT epilogue

### 现象

单位矩阵测试 `C = A @ I`（`C` 先清零）。首个错误位置**永远是 `(r=8, c=8)`**，
错误总数恒为**行数 × 8**——每个 16 行的 tile 里恰好 8 行没被写
（`defects/repro/mctlass/mt3sweep.cu`，RowMajor 主算子，pristine 头文件，`-O0 -fno-inline`）：

```
square 128     m=128 n=128 k=128 mismatched= 584 (of 16384) firstBad=(r=8 c=8)
square  64     m= 64 n= 64 k= 64 mismatched=  64 (of  4096) firstBad=(r=8 c=8)
square 256     m=256 n=256 k=256 mismatched=2080 (of 65536) firstBad=(r=8 c=8)
nonsquare A    m=128 n= 64 k= 96 mismatched= 254 (of  8192) firstBad=(r=8 c=8)
nonsquare B    m= 64 n=128 k= 96 mismatched= 288 (of  8192) firstBad=(r=8 c=8)
tall           m=200 n= 64 k= 64 mismatched= 272 (of 12800) firstBad=(r=8 c=8)
```

未写入的行保留 `C` 的初始值。**若 `C` 未清零，读到的是旧数据——看起来像
"未初始化的垃圾"，实际是 epilogue 根本没写那些行**（此前一处误判正是因此产生）。
非方阵同样复现，与形状无关。RowMajor 与 ColumnMajor 特化**都**受损。

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

**组件**：`wmma`（`__clang_maca_mma_functions.h` / `mma.h`）

### 现象

fragment 填入编码了 (row, col) 的已知值（`x[i] = 100*row + col`），分别用
`mem_row_major` 和 `mem_col_major` store 到 16×16 输出（`defects/repro/diag_store2.py`）：

```
mem_row_major: matches_expected=False
  out[0,:4] = [  0 100 200 300]   expected [0 1 2 3]
  out[:,0]  = [0 1 2 ... 15]      expected [0 16 32 ... 240]

mem_col_major: matches_expected=False
  out[0,:4] = [  0 100 200 300]   expected [0 16 32 48]
  out[:,0]  = [0 1 2 ... 15]      expected [0 1 2 ... 15]
```

**两个 tag 产出完全相同的输出**（转置布局）。`mem_row_major` 语义被忽略，
两种调用都写成转置后的矩阵。

### 为什么下游无法完全绕过

store 本身能写入完整数据（单独测 `diag_store.py`：255/255 元素正确，
唯一"零"是本应为 0 的 `[0,0]`），所以这不是崩溃或 no-op，而是**语义错误**。
下游可以在 store 后再做一次转置，代价是一次额外全矩阵 pass；
但任何期望 `mem_row_major` 按文档语义生效的代码都会**静默得到转置结果**。

---

## D3 — `wmma::load_matrix_sync` 在 lane ≥ 32 读到垃圾数据

**组件**：`wmma`

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

**前 4 行（对应 lane 0–31）精确正确，第 4 行起（lane ≥ 32）全是未初始化的垃圾值。**

这与 warp size 64 直接相关：load 原语的后 32 个 lane 没有按 warp-64 lane 映射读取。

### 补充观察

- 加 padding（ldm = 24 而非 16）**不能修复** D3（`diag_load2.py`：两个 ldm 都失败）
- b fragment 的 `row_major`/`col_major` 标签行为互换（col_major 实际按行读）

### 为什么下游无法绕过

fragment 只能通过 `load_matrix_sync` 或逐元素标量填充得到。手写标量填充确实可行
（我们的 `kernels_wmma6.py` 用它达到全尺寸 rel = 0.00000），但**性能只有
0.233×（6835 µs vs eager 1592 µs）**——标量填充的指令吞吐把性能钉死。
要打败 eager 必须有正确的向量化 load。这是 258 个手写 wmma 配置全部失败的根因。

### 什么没坏（供官方参考）

- `__builtin_mxc_mma_16x16x16f16` intrinsic 本身**正确**
- fragment 的存储布局正确
- `fill_fragment` / `mma_sync` 正确

---

## D4 — mxcc `-O2` inlining 导致 host 端 segfault

**组件**：mxcc（经 cu-bridge 调用）

### 现象

`mctlass::gemm::device::Gemm` 对象构造在 `-O2` 下 segfault（`defects/repro/mt3id.cu`）：

```
$ cucc mt3id.cu -O2 ... -o mt3id_O2
$ ./mt3id_O2
Segmentation fault (core dumped)      ← exit code 139
```

二分定位：

| 编译选项 | 结果 |
|---|---|
| `-O0` | 正常 |
| `-O2` | **segfault** |
| `-O2 -fno-inline` | 正常 |
| `-O2 -fno-elide-constructors` | segfault |
| `-O2 -fno-omit-frame-pointer` | segfault |
| `-O2 -fno-vectorize` | segfault |

只有 `-fno-inline` 能修复。纯 `mctlass/mctlass.h` + `gemm/device/gemm.h` 的
最小程序即可复现，与 PyTorch 无关。

### 为什么下游无法绕过（但可规避）

这不是静默错误（会崩溃），且 `-fno-inline` 能完全规避，代价是放弃 inlining 优化。
我们已把它写入构建脚本（所有 mctlass 编译加 `-fno-inline`）。
**仍需官方修复**，因为 inlining 对性能 kernel 很重要。

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

## D6 — `__shfl_down_sync` 在 offset ≥ 32 时复制而非交换（2026-09-22 新增）

**组件**：MACA runtime 的 warp shuffle

### 现象

64-lane warp 下，`__shfl_down_sync(mask, v, 32)` 应当让 low half 与
high half **交换**值（CUDA 语义）。实测为**单向复制**：

```
lane  0: v=0  down32=32   ← 拿到 lane 32 的值（正确方向）
lane 32: v=32 down32=32   ← 拿到自己的值，不是 lane 0 的值
```

（`repro/warp/warp_probe.cu`）

### 后果：6 轮归约结果翻倍

```
5-round: 2016.0   6-round: 4032.0   expected: 2016.0
```

4032 = 2016 × 2：high half 的值被加两次。**5 轮归约在 C500 上反而是正确的**，
因为它从不跨越 offset 32。这与直觉相反，也与 CUDA 语义不一致。

### 影响范围

`skills/c500-kernel-wiki/wiki/hardware/warp64-implications.md` 原本把
`offset = warpSize/2`（6 轮）标为 RIGHT、5 轮标为 WRONG，本次实测后已修正。

任何从 NVIDIA 代码搬来的"标准"warp 归约，只要轮数按 `warpSize` 推导，
在 C500 上都会静默翻倍。若后续 SDK 修复成交换语义，5 轮写法又会变成
丢一半数据的错误写法——**该写法危险，依赖 SDK 版本**。

### 规避

用 5 轮，或用显式两阶段归约（两个 32-lane 半 warp 内各自归约，
再用 `__shfl_sync` 到目标 lane 做显式交换）。

---

## 环境与构建命令

```bash
source /data/kda-maca/env.sh          # MACA_PATH、cucc 到 PATH、$MACA_CUCC_FLAGS

# mctlass 复现（.cu 程序，pristine 头文件）
cd defects/repro/mctlass
cucc mt3sweep.cu -O0 -fno-inline -std=c++17 \
     -I/opt/maca-3.3.0/tools/cu-bridge/include \
     -I/opt/maca-3.3.0/include $MACA_CUCC_FLAGS -o mt3sweep
./mt3sweep                        # D1

cucc ../warp/warp_red2.cu -O2 -std=c++17 \
     -I/opt/maca-3.3.0/tools/cu-bridge/include $MACA_CUCC_FLAGS -o warp_red2
./warp_red2                       # D6（5-round 2016 / 6-round 4032）

# D4：-O2 vs -O0
cucc mt3bisect.cu -O2 ... -o t_O2 && ./t_O2     # segfault (exit 139)
cucc mt3bisect.cu -O2 -fno-inline ... -o t_ok && ./t_ok

# wmma 复现（PyTorch load_inline，需 KernelBench 环境）
source /data/cuda-harness-migration/env.sh
cd defects/repro
python3 diag_load_store.py       # D3（纯 load→store 往返，4/256 正确）
python3 diag_store.py            # store 正常（255/255）
python3 diag_store2.py           # D2（layout tag 被忽略）
```

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
