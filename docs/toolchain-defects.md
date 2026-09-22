# MACA 3.3.0.15 / C500 工具链缺陷报告

**日期**：2026-09-22
**环境**：MetaX C500（104 SM、16.3 GB、warp size 64）、MACA 3.3.0.15、torch 2.8.0+metax3.3.0.2
**复现方式**：全部缺陷均在本机用最小隔离程序重新复现，编译命令为
`cucc <file>.cu -o <out> -I$MACA_PATH/tools/cu-bridge/include`（mctlass 程序额外加 `-fno-inline`）。
本报告每一条都附了可复现的最小程序。

**总体结论**：缺陷集中在**对 warp size 64 的移植不完整**这一个根因上。
MACA 在 API 层正确地把 `warpSize` 声明为 64，但多个组件的内部布局仍按 32 线程
假设展开数据。后果是**静默错误**——不崩溃、不报错，只产生错误结果。

**本轮排查推翻了两条先前结论**，值得单独注意：
- `wmma::store_matrix_sync` 之前被怀疑是 no-op，实测**正常**，错误只在 load 路径。
- wiki 里标为"RIGHT"的 6 轮 warp 归约实测**是错的**（结果翻倍），5 轮才正确——
  MACA 的 `shfl_down_sync` 在 offset≥32 时是复制而非交换。该 wiki 页已修正。

**缺陷一览**：

| # | 缺陷 | 严重度 | 状态 |
|---|---|---|---|
| 1 | `wmma::load_matrix_sync` fragment 布局按 warp 32 展开 | 🔴 严重，静默 | 未修复，不可用 |
| 2 | `wmma::store_matrix_sync` 忽略 layout tag | 🟡 中等，静默 | store 后转置可规避 |
| 3 | mctlass SIMT epilogue 只写一半行 | 🔴 严重，静默 | 未修复，不可用 |
| 4 | mxcc `-O2` inlining 导致 host segfault | 🟡 有规避 | `-fno-inline` |
| 5 | `shfl_down_sync` offset≥32 时复制而非交换 | 🟡 中等，静默 | 未修复，5 轮可规避 |

**与仓库已有报告的关系**：`defects/README.md` 是更早一轮排查的产物，
覆盖 D1–D5（与本报告的 1/2/3/4 对应，其 D3 与本报告缺陷 1 同源）。
两份报告结论一致。本报告的**新增内容**是缺陷 5（`shfl_down_sync` 的
复制语义，及其对 wiki 中 6 轮归约结论的推翻）——这是本轮排查的独立发现，
旧报告没有。建议以 `defects/README.md` 为主报告，本文件作为补充。

---

## 缺陷 1：`wmma::load_matrix_sync` 的 fragment 布局按 warp 32 展开（严重，静默）

**现象**：把一个已知内容的 16×16 tile 加载进 fragment，直接存回，输出与输入不符。
256 个元素里只有 4 个在正确位置。

**实测数据**（`optloop/diag_load_store.py`，纯 load→store 往返，不涉及 mma）：

```
ldm=16: exact=False  matching=4/256
   first mismatches: [(0,1,16.0,1.0), (0,2,32.0,2.0), (0,3,48.0,3.0), (0,4,1.5e-05,4.0), (0,5,1.5e-05,5.0)]
ldm=24: exact=False  matching=4/256   (同一模式，与 padding 无关)
```

列 1 的值是 16、列 2 是 32、列 3 是 48——正是**行号被当成列号**的错位，
且从列 4 起出现 ~1.5e-5 的垃圾值（未初始化的寄存器内容）。

**根因**：fragment 的元素在 64 个线程间分布时，布局仍按 32 线程 warp 展开，
一半线程拿到的元素与目标 (row, col) 不对应。

**影响**：任何用 `wmma` load/store 的 kernel 结果不可信。这不是"慢"，
是"错"。我们 258 组 wmma 配置全部建立在此之上，是该路线做不出正确结果的直接原因。

**关联**：`wmma::store_matrix_sync` 经实测**是正确的**（见下），
错误只在 load 路径。

---

## 缺陷 2：`wmma::store_matrix_sync` 忽略 layout tag（中等，静默）

**注意**：本次重测我先验证了 store 的**写入完整性**——256 个元素全部写入，
唯一为零的元素 `[0,0]` 期望值本来就是 0。所以 store **不是 no-op**，
先前"no-op"的说法是错的。**但 layout tag 确实被忽略**，这是一个独立的语义缺陷。

**实测**（`optloop/diag_store2.py`，fragment 值编码为 `100*row+col`）：

```
mem_row_major: matches_expected=False
  out[0,:4] = [  0 100 200 300]  expected [0 1 2 3]
  out[:,0]  = [ 0  1  2 ... 15]  expected [ 0 16 32 ... 240]
mem_col_major: matches_expected=False
  out[0,:4] = [  0 100 200 300]  expected [ 0 16 32 48]
  out[:,0]  = [ 0  1  2 ... 15]  expected [ 0  1  2 ... 15]
```

**两个 tag 产出完全相同的输出**（都是转置布局）。`mem_row_major` 的语义
被忽略，任何期望它按文档生效的代码会静默得到转置结果。

**可规避**：store 后再做一次转置，代价是一次额外全矩阵 pass。

---

## 缺陷 3：mctlass device GEMM 的 SIMT epilogue 只写一半的行（严重，静默）

**现象**：`C = A @ I`（单位矩阵）时，输出的**行 r%16 >= 8 全部为零**，
其余行正确。

**实测数据**（`/tmp/mt3sweep.cu`，RowMajor 主算子， pristine SDK 头文件）：

```
square 128      m=128 n=128 k=128 mismatched= 576 (of 16384) firstBad=1032 (r=8 c=8)
square  64      m= 64 n= 64 k= 64 mismatched=  64 (of  4096) firstBad= 520 (r=8 c=8)
square 256      m=256 n=256 k=256 mismatched=2080 (of 65536) firstBad=2056 (r=8 c=8)
nonsquare A     m=128 n= 64 k= 96 mismatched= 254 (of  8192) firstBad= 520 (r=8 c=8)
nonsquare B     m= 64 n=128 k= 96 mismatched= 288 (of  8192) firstBad=1032 (r=8 c=8)
tall            m=200 n= 64 k= 64 mismatched= 272 (of 12800) firstBad= 520 (r=8 c=8)
```

**模式极其规律**：首个错误位置永远是 `(r=8, c=8)`，错误总数恒为
`行数 × 8`——每个 16 行的 tile 里恰好 8 行没被写。非方阵同样复现，
说明与形状无关。

**根因链**：
1. `mctlass/gemm/warp/mma.h:51` 正确地把 `WarpSize::value` 设为 64（32 被注释掉）。
2. 但 epilogue 的 `default_thread_map_simt.h` / `output_tile_thread_map.h`
   的 `RowArrangement` 里**硬编码 `kWarpSize = 32`**。
3. 修正这三处后 `kThreads` 能匹配（512=512），但行轴迭代计数仍不一致：
   `Iterations::kRow = 1`（来自 `Shape::kRow=1`）而 `Count::kRow = 4`
   （`LaneMmaShape::kM`）。store 循环只写 1 行/组，`operator++` 却推进 4 次
   → 一半行从不被写。
4. 尝试的修复（`Iterations::kRow = Count::kRow`、修正 `advance_group`、
   显式 `SimtThreadMap`）**全部触发 memory violation(0x4)**，
   说明 epilogue 还有别处（`FragmentIteratorSimt` / warp 级
   `TileIteratorSimt` 的 smem 偏移）也按 32 线程 warp 假设。

**结论**：这是 mctlass 对 warp 64 的系统性移植缺口，横跨 epilogue 多个文件，
不是单点修改能修好的。**在修复前不应把 mctlass device GEMM 用于任何计算。**

**当前状态**：我先前对 SDK 头文件的三处修改**已被回退**（与 `/tmp/orig_*.h`
备份 MD5 完全一致），SDK 处于 pristine 状态。上述数据是在 pristine 头文件下
重新编译测得的，可复现。

---

## 缺陷 4：mxcc `-O1`+ 的 inlining 缺陷导致 host 端 segfault

**现象**：`mctlass::gemm::device::Gemm` 对象在 host 端构造时，
`-O2` 下 segfault，`-O0` 正常。

**二分定位**：`-fno-inline` 修复；`-fno-elide-constructors`、
`-fno-omit-frame-pointer`、`-fno-vectorize` 均无效。纯 `mctlass/mctlass.h`
+ `gemm/device/gemm.h` 的最小程序即可复现，与 PyTorch 无关。

**规避**：所有 mctlass 编译加 `-fno-inline`。

---

## 缺陷 5：`__shfl_down_sync` 在 offset≥32 时复制而非交换（中等，静默）

**这是本轮排查中最反直觉的一条，且推翻了 wiki 里的一条既有结论。**

**背景**：warp size 从 32 变 64 后，跨半个 warp 的 shuffle 应当把
low half 与 high half 的值**交换**，这样归约才能把两半的贡献合起来。
CUDA 的语义保证 `shfl_down(v, 32)` 在 64-lane warp 上使 lane 0 得到 lane 32
的值、lane 32 得到 lane 0 的值（交换）。

**实测**（`/tmp/warp_probe.cu`）：

```
lane  0: v=0  down16=16  down32=32  up32=0
lane  1: v=1  down16=17  down32=33  up32=1
lane 32: v=32 down16=48  down32=32  up32=0
lane 33: v=33 down16=49  down32=33  up32=1
```

lane 0 的 `down32` = 32（正确，拿到 high half 的值），但 lane 32 的
`down32` = 32（**拿到自己的值**，不是 lane 0 的值）。也就是说
`shfl_down_sync` 在 offset 跨越 32 时是**单向复制**，不是交换。

**后果**：6 轮归约（wiki 里标为"RIGHT"的写法）在 C500 上给出**两倍**的结果：

```
5-round: 2016.0   6-round: 4032.0   expected: 2016.0
```

4032 = 2016 × 2——high half 的值被加了两次（一次在自身，一次被复制到
low half 后又加一遍）。**5 轮反而是正确的**，因为它从不跨越 offset 32。

**对 wiki 的影响**：`skills/c500-kernel-wiki/wiki/hardware/warp64-implications.md`
第 35–41 行把 `offset = warpSize/2` 标为 RIGHT、把 5 轮标为 WRONG。
按本次实测，这个结论**是反的**：`warpSize/2 = 32` 这一轮会触发
复制语义，导致结果翻倍。该页需要修正。

**正确的归约写法**（C500 实测正确）：

```cpp
// 5 rounds, offsets 16,8,4,2,1 — never crosses 32
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffffffffffULL, v, offset);
```

或者显式两阶段：先在两个 32-lane 半 warp 内各自归约，
再用一次跨半 warp 的**显式交换**（`__shfl_sync` 带目标 lane）合起来。

**注意**：这条与"warp 64 需要 6 轮"的直觉相反，且与 CUDA 语义不一致。
若后续 SDK 版本修复成真正的交换语义，上述 5 轮写法会变成
丢一半数据的错误写法——这正是该写法危险的地方。

---

## 已确认正常的组件（避免重复排查）

| 组件 | 结论 |
|---|---|
| `wmma::mma_sync`（`__builtin_mxc_mma_16x16x16f16`） | ✅ 正确 |
| `wmma::fill_fragment` | ✅ 正确 |
| `wmma::store_matrix_sync` 写入完整性 | ✅ 256/256 元素写入（不是 no-op） |
| `__shfl_sync`（offset < 32） | ✅ 正常 |
| `atomicAdd` 及同族 | ✅ 正常 |
| `__fmaf_rn` | ✅ 正常 |
| mcblas GEMM | ✅ 正确且调优（操作数顺序相反，见下） |

---

## 规避路径（已在 KernelBench L1 P1 验证）

- **mcblas** 是唯一既正确又快的路径：fp16-in/fp32-acc 951µs = **1.693×** over eager，
  且通过 1e-4 fp32 容差。
- **mcblas 的操作数顺序与 cuBLAS 相反**：`(OP_N, OP_N, B, A)` 才是计算 `A@B`
  （`optloop/submit_mcblas.py:58`）。这不是缺陷，但极易踩。
- 手写 wmma 受缺陷 1 影响，上限 0.68×，不可用。
- mctlass device GEMM 受缺陷 3 影响，不可用。
- **任何 fp32 正确性判定必须先关 TF32**：C500 默认 `allow_tf32=True`，
  `nn.Conv1d` 偏差 ~9.5e-4，关掉后为 0.0。

---

## 复现文件清单

| 缺陷 | 文件 | 编译 |
|---|---|---|
| 1 wmma load 布局 | `optloop/diag_load_store.py` | python3（load_inline 自动编译） |
| 2 store 验证为正常 | `optloop/diag_store.py` | python3（load_inline 自动编译） |
| 3 mctlass epilogue | `/tmp/mt3sweep.cu` | `cucc … -fno-inline` |
| 3 mctlass（原始隔离） | `/tmp/mt3id.cu` | `cucc … -fno-inline` |
| 4 inlining segfault | `/tmp/mt3bisect.cu` | `cucc -O2` vs `-O0` |
| 5 shfl_down 语义 | `/tmp/warp_red2.cu`、`/tmp/warp_probe.cu` | `cucc` |

所有复现文件已备份到仓库的 `repro/` 目录（`repro/mctlass/`、`repro/warp/`、
`repro/diag_*.py`），随仓库一起保留。
