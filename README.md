# Kernel Design Agents — MetaX C500 / MACA Port

An agent-centric workflow for using coding agents to research, implement,
verify, and iterate on performance-sensitive CUDA kernel tasks — ported from
[NVlabs/kda](https://github.com/NVlabs/kda) to the MetaX C500 (MACA software
stack).

Upstream KDA targets NVIDIA B200 + Nsight Compute. This port keeps the
workflow and replaces the toolchain: `cucc` instead of `nvcc`, `mcTracer`
instead of `ncu`, `mctracer_utils.py` instead of `ncu_report`.

## What is ported

| Layer | Upstream (NVIDIA) | This port (MACA) | Status |
|---|---|---|---|
| Workflow | `prompts/basic-flow.md`, `docs/agent-flow.md`, `CLAUDE.md` | same files, MACA environment section | ✅ direct port |
| Profiling skill | `skills/ncu-report-skill` | `skills/mctracer-report-skill` | ✅ ported, with stated gaps |
| Domain knowledge | `skills/KernelWiki` (B200/Hopper) | not ported | ❌ see below |

The workflow layer is a direct port — the nine-step loop, task contract,
evidence records, and promotion rule are unchanged. The profiling skill keeps
the same structure (SKILL.md + `reference/00-12` + `helpers/`) and the same
CLI shape, with a different evidence source and honest coverage gaps.

**KernelWiki is not ported.** It is 100% Blackwell/Hopper knowledge —
`tcgen05`/TMEM/CLC/`wgmma`, FlashAttention-4, DeepGEMM — none of which applies
to the C500 architecture, and its tag vocabulary (`sm100`, `sm90`, `tcgen05`)
has no MACA equivalent. Porting it would mean writing a new knowledge base,
not translating one. Add C500-specific kernel knowledge as a separate skill
when there is a corpus to draw from.

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

## License

Upstream KDA: first-party documentation, prompts, skills-style content, and
assets are CC-BY-4.0; first-party source code is Apache-2.0. Submodules carry
their own licenses. This port is provided under the same terms — see
`LICENSE`. The upstream repository (including its `third_party_licenses/`
notices for the submodules not ported here) is at
[NVlabs/kda](https://github.com/NVlabs/kda).
