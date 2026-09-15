---
name: mctracer-report-skill
description: Profile CUDA kernels with MACA mcTracer on MetaX C500 (xcore1000). Use when the user asks to profile a kernel, analyze its performance, diagnose bottlenecks, read an mcTracer report, or write an optimization plan — including variants in Chinese ("profile 一下", "为什么慢", "mcTracer 报告").
---

# Skill: CUDA Kernel Profiling (MetaX C500 / MACA)

Port of the upstream KDA `ncu-report-skill` (NVIDIA B200 / Nsight Compute) to
the MetaX MACA toolchain. Same workflow, same evidence discipline, a different
profiler.

**When to use:** user asks to profile a CUDA kernel, analyze its performance,
find its bottlenecks, or write an optimization plan based on profiling data.
Triggers include: "profile X", "为什么这个 kernel 慢", "mcTracer 报告说…",
"下一步怎么优化", "帮我看一下这份报告".

**Target hardware (this repo):** MetaX C500 (xcore1000, 104 SMs, 16.3 GB,
warp size 64). Most of the workflow is generic; C500-specific notes are
explicitly marked.

---

## What changed vs upstream

| Upstream (NVIDIA) | This port (MACA) |
|---|---|
| `ncu --set full` / `--set source` | `mcTracer --odname <run>` (single trace) |
| `.ncu-rep` binary reports | Chrome trace-event JSON |
| `ncu_report` Python module | `mctracer_utils.py` (pure stdlib JSON) |
| Replay-based hardware counters | Not available — see below |
| NCU rule engine (`Est. Speedup`) | Manual pattern matching (reference/06) |
| PM sampling timeline | Per-launch timeline (`kernel_timeline`) |

**Hard limitation.** `mcTracer` is an API-level tracer: it records every
kernel launch with its grid/block, register count, shared-memory usage,
occupancy and exact device-side duration. It does **not** sample hardware
performance counters. The following upstream analysis dimensions are
therefore unavailable on MACA and must not be faked:

- stall-reason breakdown (long/short scoreboard, barrier, …)
- per-PC / per-source-line stall hotspots
- L1/L2 cache hit rates, sectors-per-request
- DRAM throughput vs peak
- tensor-core pipe utilization
- register-spill detection via `local_ld` counts

When a diagnosis needs one of those, say so explicitly ("MACA 不可观测：
cache 命中率") and fall back to the proxies documented in
[`reference/11-maca-proxies.md`](reference/11-maca-proxies.md). A stated
gap is evidence-grade; an invented number is a bug.

---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

Most under-performing kernels are under-performing for exactly one reason
that the trace can tell you in 10 seconds — usually the launch geometry.
Don't invent hypotheses before you have the trace. Don't start coding a fix
before you've matched the observed pattern to a known diagnosis. Don't write
a wall of suggestions — rank them by evidence and expected impact.

---

## Quickstart (what to do when someone says "profile this kernel")

0. **Create a new run directory first** under `profile/<run_name>/` — **one
   directory per run**, never reuse an existing one. Each run contains its
   own `harness/`, `reports/`, `analysis/`, and `REPORT.md`. This rule is
   mandatory. See [`reference/00-directory-layout.md`](reference/00-directory-layout.md).

1. **Decide what you're profiling.** What inputs? Which dispatch path? What
   question do you want answered? If the kernel takes variable-sized inputs,
   pick specific representative shapes from the user's workload — don't
   profile with arbitrary inputs.

2. **Build a standalone harness** unless profiling through an existing
   binary. Harnesses compile in seconds, run the kernel in isolation, and
   give a clean trace with exactly one interesting kernel. Compile into
   `profile/<run_name>/harness/`. See [`reference/02-harness-guide.md`](reference/02-harness-guide.md)
   and the template in [`helpers/harness_template.cu`](helpers/harness_template.cu).

3. **Run the trace.** One pass, no replay overhead (unlike ncu's 45+ passes):

   ```bash
   cd profile/<run_name>
   mcTracer --odname reports --name <tag> ./harness/harness [args]
   ```

   **`--odname` must be a relative path** — an absolute path fails with
   `Output file open error!`. So `cd` into the run directory first, or invoke
   mcTracer with a cwd-relative output dir.

   Output: `reports/<tag>-<pid>.json`. See [`reference/03-collection.md`](reference/03-collection.md).

4. **Parse with `mctracer_utils`** — not by eye-balling the JSON. Write
   analysis outputs to `profile/<run_name>/analysis/`. Use the helpers in
   [`helpers/`](helpers/). See [`reference/04-python-api.md`](reference/04-python-api.md).

5. **Work through the analysis dimensions.** See
   [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md).
   Four are fully supported on MACA (launch geometry, timeline, kernel
   inventory, roofline); the counter-dependent ones are gapped.

6. **Match patterns to the diagnosis playbook.** See
   [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md).
   Patterns A, B, F, J, K, L, M, N are diagnosable from trace data alone;
   the rest need proxies or must be flagged unobservable.

7. **Write the report** at `profile/<run_name>/REPORT.md` with
   evidence-backed recommendations, ranked by expected impact. See
   [`reference/07-report-template.md`](reference/07-report-template.md).

---

## File index

### Reference docs (read these when you need details)

| File | Purpose |
|---|---|
| [`reference/00-directory-layout.md`](reference/00-directory-layout.md) | **Read first.** Directory / naming conventions — one run = one subdirectory, no cross-contamination |
| [`reference/01-workflow.md`](reference/01-workflow.md) | End-to-end checklist from "user request" to "final report" |
| [`reference/02-harness-guide.md`](reference/02-harness-guide.md) | When and how to build a standalone harness |
| [`reference/03-collection.md`](reference/03-collection.md) | `mcTracer` command recipes and the JSON schema |
| [`reference/04-python-api.md`](reference/04-python-api.md) | `mctracer_utils` API patterns with copy-pasteable code |
| [`reference/05-analysis-dimensions.md`](reference/05-analysis-dimensions.md) | Analysis dimensions: occupancy, timeline, inventory, roofline (+ gapped ones) |
| [`reference/06-diagnosis-playbook.md`](reference/06-diagnosis-playbook.md) | Pattern → diagnosis → fix, with MACA observability annotations |
| [`reference/07-report-template.md`](reference/07-report-template.md) | How to structure the final report |
| [`reference/10-maca-mapping.md`](reference/10-maca-mapping.md) | ncu metric → mcTracer field mapping table |
| [`reference/11-maca-proxies.md`](reference/11-maca-proxies.md) | Workarounds for counter-dependent diagnoses on MACA |
| [`reference/12-common-issues.md`](reference/12-common-issues.md) | mcTracer permissions, empty traces, demangled names, cucc include paths |

### Helpers (reusable code)

| File | Purpose |
|---|---|
| [`helpers/harness_template.cu`](helpers/harness_template.cu) | Standalone harness template — paste your kernel, fill in input allocation, done |
| [`helpers/mctracer_utils.py`](helpers/mctracer_utils.py) | Shared helpers: `load_report`, `safe`, `launch_geometry`, `C500_KEY_METRICS`, … |
| [`helpers/analyze_reports.py`](helpers/analyze_reports.py) | Extract key metrics + side-by-side comparisons from one or more traces |
| [`helpers/plot_timeline.py`](helpers/plot_timeline.py) | ASCII per-launch timeline plotter (queue depth, overlap, tail) |
| [`helpers/kernel_inventory.py`](helpers/kernel_inventory.py) | Full kernel inventory of a run: names, shapes, counts, durations |

---

## Critical lessons (don't skip)

1. **`dur` is the number that matters.** The event `args` also carry
   `queue_ts` / `submit_ts` / `complete_ts` — those bracket the *host-side
   queue lifetime*, not device execution. Duration is on the event itself
   (`dur`, nanoseconds) and matches `complete_ts - queue_ts` only loosely.
   Report `dur`.

2. **The mangled and demangled names differ.** `args.name` is mangled
   (`_Z12saxpy_kernelifPKfPf`), the event `name` is pretty
   (`saxpy_kernel(int, float, float const*, float*)`). Filter on the pretty
   name; keep the mangled one for symbol lookup.

3. **Occupancy is reported, not derived.** `mtreg_occupancy(%)` and
   `shared_memeory_occupancy(%)` (sic — the typo is in the MACA field) come
   from the runtime. Use them; cross-check with `launch_geometry()` when
   they look wrong.

4. **One trace, no replay.** Unlike ncu's 45-50 replay passes, mcTracer
   records everything in one run. That means no metric-collection failures
   to debug — but also no per-counter replay depth.

5. **Always compile with the cu-bridge include path.** `cucc` needs
   `-I$MACA_PATH/tools/cu-bridge/include` or `__macro_mxcc.h` is not found.
   See [`reference/12-common-issues.md`](reference/12-common-issues.md).

6. **Don't delegate understanding.** Run the traces yourself, open the JSON,
   cite specific field values. Never write "the trace shows it's
   memory-bound" — instead name the two or three values that back your
   conclusion (e.g. "`launch__waves_per_multiprocessor = 0.23` with a
   4096-block grid on 104 SMs, so ~77% of SMs are idle for the whole 86 µs
   kernel — this is a small-grid problem, not a memory problem"). Fill in
   the actual numbers from your trace. Specificity is the deliverable.
