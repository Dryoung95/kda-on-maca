# Kernel Design Agents — MetaX C500 / MACA Port

An agent-centric workflow for using coding agents to research, implement,
verify, and iterate on performance-sensitive CUDA kernel tasks — ported from
[NVlabs/kda](https://github.com/NVlabs/kda) to the MetaX C500 (MACA software
stack).

Upstream KDA targets NVIDIA B200 + Nsight Compute. This port keeps the
workflow and replaces the toolchain: `cucc` instead of `nvcc`, `mcTracer`
instead of `ncu`, `mctracer_utils.py` instead of `ncu_report`.

> **New to this repo?** Read these two sections in order — they are the whole
> picture:
> 1. [**Status: what 16.3 GB closed, and what it did not**](#status-what-163-gb-closed-and-what-it-did-not) — what is finished (everything that fits in 16.3 GB) and the measured KernelBench compatibility result.
> 2. [**Next steps — the handoff to 64 GB**](#next-steps--the-handoff-to-64-gb) — the five-item ordered backlog. Performance tuning (item 3) and batch RL rollout (item 4) are the work that the 64 GB board unlocks.
>
> The compatibility layer is done: 250 KernelBench problems, **0 compile
> failures**. Making the kernels *fast* is the work that remained: starting
> from **1 of 250** problems beating eager, the mcblas fp16 GEMM path now
> beats eager on **58 of 82** GEMM and linear-chain problems across L1/L2/L3
> — see [Performance tuning](#3-performance-tuning--the-actual-64-gb-work)
> for what is fast, and where the fp16 path's precision boundary falls.
> At native (64 GB) scale, **249 / 250 problems pass** the correctness gate
> unpatched, and **250 / 250** under the D7-safe harness patch. The one
> unpatched failure, L1 P76, is a genuine MACA conv defect (D7 in
> [`defects/README.md`](defects/README.md)) — not a migration gap — and
> because Conv1d is independent per batch sample, slicing the batch
> dimension is exact, so the patch passes P76 5/5 without approximating
> anything.

## What is ported

| Layer | Upstream (NVIDIA) | This port (MACA) | Status |
|---|---|---|---|
| Workflow | `prompts/basic-flow.md`, `docs/agent-flow.md`, `CLAUDE.md` | same files, MACA environment section | ✅ direct port |
| Profiling skill | `skills/ncu-report-skill` | `skills/mctracer-report-skill` | ✅ ported, with stated gaps |
| Domain knowledge | `skills/KernelWiki` (B200/Hopper) | `skills/c500-kernel-wiki` | ✅ replaced (not ported) |

The workflow layer is a direct port — the nine-step loop, task contract,
evidence records, and promotion rule are unchanged. The profiling skill keeps
the same structure (SKILL.md + `reference/00-12` + `helpers/`) and the same
CLI shape, with a different evidence source and honest coverage gaps.

**KernelWiki is not ported — it is replaced.** Upstream KernelWiki is 100%
Blackwell/Hopper knowledge — `tcgen05`/TMEM/CLC/`wgmma`, FlashAttention-4,
DeepGEMM — none of which applies to the C500 architecture, and its tag
vocabulary (`sm100`, `sm90`, `tcgen05`) has no MACA equivalent. Porting it
would mean writing a new knowledge base, not translating one.

`skills/c500-kernel-wiki` is that knowledge base: C500 hardware facts, warp-64
implications, measured rooflines, shared-memory behavior, the CUDA→C500
compatibility matrix, and mctlass authoring patterns. Every number in it comes
from a microbenchmark in `skills/c500-kernel-wiki/probes/` or from a toolchain
header, and every page carries provenance frontmatter saying which. The probes
all build with `cucc … $MACA_CUCC_FLAGS` and are the reproducer for the page
that cites them.

## Quick start

```bash
source env.sh                    # MACA_PATH, cu-bridge PATH, $MACA_CUCC_FLAGS

# 1. build a profiling harness
mkdir -p profile/matmul_v1_baseline/{harness,reports,analysis}
cp skills/mctracer-report-skill/helpers/harness_template.cu \
   profile/matmul_v1_baseline/harness/matmul_harness.cu
# ... paste your kernel, fill in alloc/launch ...
cucc profile/matmul_v1_baseline/harness/matmul_harness.cu \
     -O2 -std=c++17 $MACA_CUCC_FLAGS \
     -o profile/matmul_v1_baseline/harness/matmul_harness

# 2. trace (single pass, no replay; --odname must be relative to the cwd)
cd profile/matmul_v1_baseline
mcTracer --odname reports --name base ./harness/matmul_harness

# 3. analyze
H=skills/mctracer-report-skill/helpers
python3 $H/kernel_inventory.py --run-dir . --report "reports/base-*.json" --tag base
python3 $H/analyze_reports.py --run-dir . --report "reports/base-*.json" --tag base
python3 $H/plot_timeline.py --run-dir . --report "reports/base-*.json" --tag base
```

Then follow `skills/mctracer-report-skill/SKILL.md` through diagnosis and the
final `REPORT.md`.

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow (MACA) |
| `prompts/basic-flow.md` | Generic starter prompt for a new task |
| `prompts/README.md` | How to use prompt templates |
| `skills/mctracer-report-skill/` | Profiling skill: SKILL.md + reference docs + helpers |
| `scripts/correctness_gate.py` | KernelBench-backed correctness check for a candidate |
| `install-skills.sh` | Link skills into `~/.claude/skills` |
| `CLAUDE.md` | Repository-facing agent instructions |
| `env.sh` | MACA / cu-bridge / mcTracer environment |

## What mcTracer can and cannot see

The evidence source is the single biggest difference from upstream, and it is
worth knowing the boundary before you start:

| Evidence | Available |
|---|---|
| Launch geometry (grid/block), per launch | ✅ |
| Registers per thread, shared memory per block | ✅ |
| Register- and shared-memory-limited occupancy | ✅ |
| Exact device-side duration, per launch | ✅ |
| Whole-run kernel inventory (which kernels, how often, time share) | ✅ |
| Wave math (waves/SM, last-wave utilization) | ✅ derived |
| Roofline position (given an op count) | ✅ derived |
| Stall reasons, per-PC hotspots | ❌ |
| L1/L2 hit rates, sectors/request, DRAM throughput | ❌ |
| Tensor-core pipe utilization | ❌ |
| Register-spill detection | ❌ |
| NCU rule engine / `Est. Speedup` | ❌ |

The four supported analysis dimensions cover upstream diagnosis Patterns A,
B, F, J, K, L, M, N. Patterns C, D, E, G, H, I need a controlled experiment
in place of a counter reading — see
[`skills/mctracer-report-skill/reference/11-maca-proxies.md`](skills/mctracer-report-skill/reference/11-maca-proxies.md).

A stated gap is evidence-grade. An invented counter value is a bug.

## Validation and evaluation

Profiling and correctness checking are separate concerns on this stack.
KernelBench is already migrated to this C500 host at
`/data/maca-harness/KernelBench`. The packaged gate wraps it:

```bash
source /data/maca-harness/env.sh     # needs KernelBench on PYTHONPATH
python3 scripts/correctness_gate.py --level 1 --problem 1 \
    --candidate my_kernel.py --verbose
# → compiled: True / correct: True, exit 0 on pass
```

The candidate file must define class `ModelNew` (the KernelBench convention).
The raw KernelBench CLI (`scripts/run_and_check.py`) also works but needs a
`pydra` version match; `correctness_gate.py` calls the same
`eval_kernel_against_ref` entry point without that dependency.

Use the gate for validation; use mcTracer for performance evidence. Do not
couple the two — a correctness check inside a traced run pollutes the trace,
and a traced run is not a stable timing measurement.

## Repository layout

Work happens in a task workspace, not here:

```
task-workspace/
  docs/        # draft.md, plan.md
  runs/
  outputs/
  profile/
  benchmark.csv
  candidates.jsonl
```

The important rule is that the agent records enough context for another
engineer to understand what was tried, what passed validation, and why the
final candidate was selected.

## Status: what 16.3 GB closed, and what it did not

All compatibility work that fits in 16.3 GB is **done**. Three workstreams
were closed on this host:

| Workstream | Closed with |
|---|---|
| Build + correctness closure | cucc → cu-bridge → mxcc via `load_inline`; `eval_kernel_against_ref` end to end |
| KernelBench compatibility | L1–L3, 250 problems; 0 compile failures. At resized scale: 0 genuine ecosystem incompatibility; at native 64 GB scale one defect surfaced (D7, L1 P76) — mitigated inside the benchmark by the exact batch-slicing guard, under which the port is 250/250 |
| C500 knowledge base + toolchain | `skills/c500-kernel-wiki` — 11 pages, 15 reproducible probes, `validate.py`/`query.py`/`generate-indices.py` |

The measured compatibility result, per level (resized = inputs scaled to fit
16.3 GB, TF32 off):

| Level | Original size | Resized to 16.3 GB |
|---|---|---|
| L1 (100) | 60 pass | **100 / 100** (P95 fixed: int64 label dtype dispatch) |
| L2 (100) | 98 pass | **99 / 100** (P66 remains) |
| L3 (50) | 37 pass | **46 / 50** (P2 oom; P17/P31/P38 numeric) |

**Compile pass rate is 100% across all 250 problems.** Every remaining failure
is either memory capacity or a numeric difference, not a compile/semantic gap.
See [`eval/README.md`](eval/README.md) for the per-problem CSVs and the root
cause of each failure. `eval/` is a git subtree of the standalone
[Dryoung95/kda-on-maca](https://github.com/Dryoung95/kda-on-maca) evaluation
repository.

### The 16.3 GB wall — why this host stops here

Two things are bounded by board memory, not by effort:

1. **Problem coverage.** KernelBench sizes inputs for 48 GB boards. The C500
   has 16.3 GB (≈15.2 GB usable), so problems above the wall cannot be
   exercised at their intended scale — only at reduced scale, which changes
   what a performance number means.
2. **Batch RL throughput.** The CUDA-Agent training loop (arXiv:2602.24286)
   collects one episode at a time, and each episode holds a reference and a
   candidate module simultaneously plus the intermediate activations of a
   full forward pass. That fits one problem, not a batch. On 64 GB the same
   loop runs several concurrent episodes and the wall-clock of a training
   run drops by roughly the concurrency factor.

The wall is the reason performance tuning stopped at a single problem
(L1 P1 matmul, 1.693× over eager via the mcblas fp16-input path) instead of
sweeping the benchmark.

## Next steps — the handoff to 64 GB

Ordered by what unblocks the most downstream work. Item 1 is complete;
item 2 needs no board upgrade and can be done in parallel with the rest;
3–5 are the work that the 64 GB board actually unlocks.

### 1. Push the unpushed commits — DONE (2026-09-22)

Both local repositories were pushed to `Dryoung95/kda-on-maca`:

```
kda-maca           5 commits  (workflow port + c500-kernel-wiki + eval/ subtree merge)
kda-maca-eval      7 commits  (L1/L2/L3 evaluation + int64 label fix)
```

`kda-maca` had no common ancestor with the remote `master`, so the push was a
force-update: the remote eval-only history was superseded by this repository's
tree, which contains the same files under `eval/` (a git subtree of the eval
repo at commit `a5663e9`). No content was lost; the standalone numbers live
in `eval/results/` here and in the eval repo's own history.

Commit authors were rewritten to the repository owner; the pre-rewrite
commits are tagged `backup/pre-rewrite-20260922-080704` (local `kda-maca`
tip) and `backup/pre-rewrite-remote` (the eval-only `master`) on the host
that did the push.

### 2. Report the MACA toolchain defects (cheap, high value, no GPU needed)

Seven defects are isolated, reproduced, and documented in
[`defects/README.md`](defects/README.md) (D1–D7), with every reproducer
checked into `defects/repro/`:

| # | Defect | Silent? | Workaround |
|---|---|---|---|
| D1 | mctlass device-GEMM epilogue writes only half the rows of each 16-row tile | yes | none — the layer is unusable |
| D2 | `wmma::store_matrix_sync` ignores the layout tag (row/col-major produce the same transposed output) | yes | transpose after store |
| D3 | `wmma::load_matrix_sync` returns garbage for lanes ≥ 32 | yes | none — vectorised load is unusable |
| D4 | mxcc `-O2` inlining segfaults on mctlass host-side construction | no (crash) | `-fno-inline` |
| D5 | fp32/TF32 `wmma` fragments are not instantiated | compile error | go through fp16/bf16 |
| D6 | `__shfl_down_sync` copies rather than swaps across offset 32, so a 6-round warp-64 reduction returns 2× the correct value | yes | use 5 rounds |
| D7 | Conv1d on a >~1.62e9-element input silently corrupts the first and last `(B−48)` batches of output; per-batch slices of the same tensor are all correct | yes | slice the batch dim (applied in the benchmark harness, see below) |

D1, D3, D6 and D7 are the dangerous ones: the program reports success and
returns a completely wrong result. D6 in particular reverses the standard
guidance (the derivation `offset = warpSize/2` is the *wrong* pattern on this
SDK) — re-verify after any MACA upgrade.

D7 was invisible on the 16.3 GB board — its trigger threshold is ~6.5 GB of
input, which did not fit — and surfaced only during 64 GB performance tuning.
The unreduced conv output is the only path that miscomputes; slicing the
batch dimension and concatenating avoids it.

The workaround is exact, not an approximation: Conv1d is independent per batch
sample, so a sliced-and-concatenated forward is the same computation on a
non-corrupted path. Adjudicated against an im2col + GEMM reference that never
runs the conv kernel (2026-10-15): the sliced forward matches to 9.5e-07 with
zero bad elements, while the full-tensor forward differs by 1.99 across ~357M
elements. Notably the corrupted path is also the *reference* forward inside
the KernelBench gate, so the fix belongs at the harness layer — the benchmark
wraps `nn.Conv1d.forward` with the same guard above 1.62e9 elements, which
makes both sides of the comparison correct. Under that patch the full port is
250 / 250 (the unpatched 249 / 250 remains the platform-level figure); only
3 of the 250 problems use Conv1d at all, and of those only one exceeds the
guard, so the blast radius is one problem.

The mma intrinsic itself (`__builtin_mxc_mma_16x16x16f16`) is **correct**;
only the load/store wrappers are broken. Also worth reporting: the mctlass
inlining bug where `-O2` segfaults on device-`Gemm` construction, fixed by
`-fno-inline`, and the `ColumnMajor` output specialization that computes `B@A`
for square matrices (hidden for `n == ldm`).

These are the findings that took the most wall-clock to isolate. They are the
reason a hand-written wmma kernel could not beat eager matmul on this board —
**all 258 wmma configurations were built on broken load/store primitives**.
The vendor can fix them in the SDK; nobody downstream can.

### 3. Performance tuning — the actual 64 GB work

Starting from **1 of 250** problems beating eager (L1 P1, 1.693×), extending
the mcblas fp16 GEMM path across the GEMM family reached **58 of 82**
problems faster than eager, all gated through `eval_kernel_against_ref`:

| scope | problems | faster than eager |
|---|---:|---:|
| L1 GEMM family | 13 | 11 |
| L2 linear chains | 37 | 34 |
| L3 networks using `nn.Linear` | 32 | 13 |
| **total** | **82** | **58** |

Largest wins: L3 ShallowWideMLP 6.87×, L2 P56 6.05×, L1 large-K 2.80×. The
reference throughout is eager fp32 with TF32 on.

The 58 was reached in two passes. The first pass measured 53/82 but ran with
the models in train mode — `eval.py` never calls `.eval()`, an upstream
KernelBench omission — and train-mode BatchNorm/Dropout divergence
(3e-4..1e-2, with no candidate kernel in the loop) was masquerading as fp16
precision error. The harness now defaults to eval mode
(`eval_kernel_against_ref(..., eval_mode=True)`), which dropped the
self-comparison noise to 0..4e-6 and moved five problems from "precision
tail" to "faster than eager" (L2 P66 2.47×, P33 2.08×, P39 1.44×, P41 1.40×,
L3 P15 1.06×). The fp16 path's true precision boundary is **four** problems,
not the 29 originally recorded.

A third pass tried selective fp32 on the two boundary problems that had
per-layer routing space (L3 P32, P34): route only the error-injecting layer
back to eager, keep the small one on the fp16 path. Both pass correctness
5/5, but both measure **5–39% slower than the all-fp32 control** (P32
0.974×, P34 0.743×), because in each model the heavy matmul *is* the
error-injecting layer. Routing it back spends the only acceleration the
model had, so neither joins the 58 and the total is unchanged.

The verified ceiling numbers for n=4096 square matmul, measured on this board:

| Path | Time | vs eager | Notes |
|---|---|---|---|
| eager fp32 (TF32 actually on) | 1592 µs | 1.00× | The number to beat |
| mcblas fp16-in / fp32-acc | **951 µs** | **1.693×** | The only path that passes the 1e-4 fp32 tolerance **and** beats eager |
| mcblas bf16 | 812 µs | 1.96× | Fails the 1e-4 tolerance (err 3.8e-3); the ceiling if tolerance is relaxed |
| mcblas fp32 direct | 4743 µs | 0.34× | Loses to TF32 eager — do not use |
| hand-written wmma (working) | 6835 µs | 0.233× | Correct but scalar fragment fill is the bottleneck |

#### The mcblas conventions, since the docs were not available offline

The reversed operand order is real, but it was only half of what was needed:
`(OP_N, OP_N, B, A)` computes `A@B`, *and* the leading dimensions must be
given per matrix. For `C = A@B` with row-major `A(M,K)`, `B(K,N)`, `C(M,N)`:

```
m = N, n = M, k = K
E = B, ldE = N      (row-major B(K,N) is a column-major (N,K))
F = A, ldF = K      (row-major A(M,K) is a column-major (K,M))
C,     ldC = N
```

The original `optloop/submit_mcblas.py` passed `ld = m` (=`A.size(1)`) for
all three matrices. That equals the correct value only for square problems —
which is exactly why L1 P1 passed and every non-square shape silently
produced garbage. Transposed operands are views, not copies: a transpose
view is already column-major in its own storage, so `OP_T` reads it directly
and no `.contiguous()` is needed.

Two more entry points, each probed before use:

- `mcblasGemmStridedBatchedEx` — one launch for a whole batch, covering L1
  P3/P10/P11 and the L2 chains. **Strides are in elements, not bytes**; byte
  strides produce an illegal memory access.
- `mcblasHSSgemvStridedBatched` with `batchCount=1` — fp16 A, fp16 x, fp32 y,
  for L1 P4. Convention `m=K, n=M, lda=K, trans=T`.

#### Where the fp16 path stops

Four problems are correct but fail the 1e-4 gate because fp16 *input*
rounding is amplified by what follows the matmul — never a kernel defect.
The linear op itself is ~1.5e-5 relative. What pushes these over is a
reduction or normalisation across the contracted axis: summing an
8192-wide row turns a 2e-8 mean error into ~2e-4, and BatchNorm divides by
statistics that are themselves perturbed.

The obvious rescues do not work: fp32 mcblas measures 4699 µs against
eager's 1487 µs (**0.32×**, `probe_fp32_speed.py`), and bf16 carries 2%
wrong elements. fp16 is the only operand type that beats eager, so these are
the fast path's precision boundary rather than bugs. Selective fp32 — route
only the amplifying layer back to eager — was measured too, on the two
boundary problems with more than one matmul (L3 P32, P34): both then pass
5/5 but land 5–39% *below* the all-fp16 control, because the heavy matmul
is exactly the layer that must be routed back (see the note above). The
remaining four boundary problems each have a single matmul, so for them
"selective" means routing the whole model back to the reference. The
direction is exhausted.

An earlier version of this section counted 29 problems in this class. That
figure was measured in train mode (see the note above) and mostly reflected
BatchNorm/Dropout divergence between two model instances, not fp16 error.
In eval mode 18 of the 29 pass — five of them faster than eager.

The one remaining direction:

1. ~~**The convolutional networks.** L3's convolution-dominated models need an
   mcdnn-based path, a different kernel family entirely.~~ **Closed
   (2026-10-15) as a measured negative.** The conv family was opened and all
   four plausible paths disproved; see
   [`task-gemm-perf/docs/conv-family-negative.md`](../task-gemm-perf/docs/conv-family-negative.md)
   for the full evidence. In short: the SDK has **no usable fp16 conv path**
   (`f16→f32` returns `BAD_PARAM`, `f16→f16` executes at 5.7e-2 error), so the
   GEMM line's central trick does not transfer; and the one gate-correct
   direct-mcdnn configuration (`PRECOMP_GEMM` + `DEFAULT_MATH`) is a **tie at
   maxdiff exactly 0.000** because eager conv already runs through the same
   `libmcdnn.so` torch links — calling mcdnn ourselves *is* the eager path,
   not a faster one. The FMA (exact-fp32) variants are more precise than the
   TF32 reference and therefore *fail* the 1e-4 gate. The im2col + fp16 GEMM
   decomposition loses on both axes: `F.unfold` alone costs 2–3× the whole
   eager conv (9.66 GB of shuffle traffic on P55), and its maxdiff is 3–6e-2.
   What would move conv is a vendor-side fix (working fp16 conv, or
   instantiated Winograd kernels), not a harness-side one — the **58/82**
   GEMM-family result stands as the performance line's final number.

Also a measured negative result: **matrix-vector (L1 P4) does not have a
GEMM-shaped win**. The fp16 GEMV path is 11.84 ms against eager's 8.18 ms
(363 GB/s of the card's measured 1.27 TB/s), and the 8.6 GB fp32 input cast
dominates. A win needs a fused cast or an fp16-input eager GEMV.

Before quoting any fp32 number, **disable TF32**. C500 ships with
`allow_tf32=True` by default; with it on, `nn.Conv1d` shows ~9.5e-4 error
against fp32, which silently fails the 1e-4 KernelBench tolerance. With it
off, the difference is exactly 0.0. Any fp32 correctness verdict that does
not first disable TF32 may be reporting a precision policy as a backend bug.

### 4. Batch rollout for the agent loop (needs 64 GB)

The CUDA-Agent pipeline has components ① and ② done and ③ (RL training)
not started. Component ③ is gated on stage-1 batch rollout, which needs
concurrent episodes:

- **Stage 1 (batch rollout):** extend `run_g9.py`'s single-task loop to a
  scheduler over the 3396-task pool
  (`/data/cuda-agent-migration/dataset/actionable_tasks.csv`). Record full
  trajectories (prompt, tool calls, stderr, reward) for RFT reuse. Memory
  guard per episode: query `mem_get_info` before launch, queue when free
  memory is under 4 GB.
- **Stage 2a–d (training):** single-turn PPO warm-up without tools →
  RFT-filter the stage-1 trajectories (drop invalid tool-call patterns and
  retry loops) → pre-train a critic value head → multi-turn agentic RL with
  the milestone reward (compile_failed 0.0 / incorrect 0.1 / unmeasured 0.3 /
  correct_slow 0.5 / pass 1.0).
- **Stage 3 (eval):** score on KernelBench, not the task pool — overall pass
  rate, faster rate vs eager/compile, geomean speedup. Ablate whether the
  SKILL.md MACA supplement actually improved the pass rate.

Two things to carry across from this host's debugging: the **reward signal
must be DCE-safe** (dead-code elimination previously erased both eager and
ext computation and produced a fake "2–6× speedup"; the real number was
1.35×), and **GPU state drifts** — a copy probe dropped from 1350 GB/s to
745 GB/s between two runs of the same workload while another agent held the
  board. Verify the copy baseline before believing any performance verdict.

### 5. Validation caveats to inherit

Three traps in the evaluation layer that produce false "incompatibility"
verdicts. They are fixed on this host but are easy to reintroduce:

- The gate catches OOM inside `run_and_check_correctness` and returns
  `compiled=True, correct=False` — **indistinguishable from a real numeric
  bug in the CSV**. Always check the allocator message to classify.
- Do not scale structural constants. Resizing `features`/`num_groups`/`num_classes`
  breaks the `C % num_groups` divisibility that GroupNorm requires, and the
  resulting structural error reads like a numeric bug.
- Classifying a failure by operator family is unreliable. Softplus (P29) and
  FrobeniusNorm (P37) look numeric but OOM; Softsign (P30) looks like OOM
  but is numeric.

## License

Upstream KDA: first-party documentation, prompts, skills-style content, and
assets are CC-BY-4.0; first-party source code is Apache-2.0. Submodules carry
their own licenses. This port is provided under the same terms — see
`LICENSE`. The upstream repository (including its `third_party_licenses/`
notices for the submodules not ported here) is at
[NVlabs/kda](https://github.com/NVlabs/kda).
