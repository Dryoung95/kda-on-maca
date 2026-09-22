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
> 2. [**Next steps — the handoff to 64 GB**](#next-steps-the-handoff-to-64-gb) — the five-item ordered backlog. Performance tuning (item 3) and batch RL rollout (item 4) are the work that the 64 GB board unlocks.
>
> The compatibility layer is done: 250 KernelBench problems, **0 compile
> failures, 0 genuine ecosystem incompatibility**. What is not done is making
> kernels *fast* — currently **1 of 250** problems beats eager PyTorch. That
> is the next phase.

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
`/data/cuda-harness-migration/KernelBench`. The packaged gate wraps it:

```bash
source /data/cuda-harness-migration/env.sh     # needs KernelBench on PYTHONPATH
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
| KernelBench compatibility | L1–L3, 250 problems; 0 compile failures, 0 genuine ecosystem incompatibility |
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

Three defects are isolated and reproduced but not yet reported upstream to
the vendor. The reproduction cases currently live in `/tmp` on this host and
**will be lost when this environment is torn down**. Back them up and file
them:

| Defect | Impact | Reproduction |
|---|---|---|
| `wmma store_matrix_sync` is a **no-op** | The destination buffer is not modified at all; hand-filling the fragment then storing leaves every element unchanged | `optloop/diag_store*.py` in `/data/cuda-harness-migration/` |
| `wmma load_matrix_sync` scrambles data | Hand-filled fragments + `mma_sync` + hand-written store reproduces `A@B` to 1e-6; replacing only the fill with `load_matrix_sync` breaks it | `optloop/diag_load*.py` |
| mctlass device-GEMM SIMT epilogue writes only rows `r%16<8` | `C = A@I` leaves rows `r%16>=8` as zeros — silently wrong results, not a crash | `/tmp/mt3*.cu` (mctlass `Gemm` device layer, `-fno-inline` required) |

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

Only **1 of 250 problems** currently beats eager (L1 P1, 1.693×). Extending
this is the main deliverable of the next phase. The verified ceiling numbers
for n=4096 square matmul, measured on this board:

| Path | Time | vs eager | Notes |
|---|---|---|---|
| eager fp32 (TF32 actually on) | 1592 µs | 1.00× | The number to beat |
| mcblas fp16-in / fp32-acc | **951 µs** | **1.693×** | The only path that passes the 1e-4 fp32 tolerance **and** beats eager |
| mcblas bf16 | 812 µs | 1.96× | Fails the 1e-4 tolerance (err 3.8e-3); the ceiling if tolerance is relaxed |
| mcblas fp32 direct | 4743 µs | 0.34× | Loses to TF32 eager — do not use |
| hand-written wmma (working) | 6835 µs | 0.233× | Correct but scalar fragment fill is the bottleneck |

Plan: extend `optloop/submit_mcblas.py` from L1 P1 to the other GEMM-shaped
problems, then sweep block size / K-tile / stream count. Note the mcblas
operand order is **reversed** relative to cuBLAS — `(OP_N, OP_N, B, A)` is
what computes `A@B`.

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
