# Profiling Workflow — End-to-End

The complete checklist from "user asks to profile" to "final report". Every
step has a short rationale and a pointer to the detailed doc. Ported from the
upstream ncu-report-skill workflow; step numbers line up.

---

## Phase 0 — Create a new run directory

**Always start here.** See [`00-directory-layout.md`](00-directory-layout.md)
for the full convention.

```bash
PROFILE_RUN_DIR=profile/<descriptive_run_name>        # e.g. <kernel>_v1_baseline
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}
```

- Pick a new, descriptive name for this run. Never reuse an existing directory.
- A new kernel version is a **new** run (`<kernel>_v2_optimized/`, not
  overwriting `<kernel>_v1_baseline/`).
- The same version against a different workload is also a new run — or, at
  minimum, each workload's trace gets a distinct tag.

Every artifact produced in subsequent phases is written **only** under
`$PROFILE_RUN_DIR`. Never into a sibling run's directory.

---

## Phase 0.5 — Frame the problem (before any tools)

Before typing any commands, answer these in your head (or in a short note to
the user):

1. **What kernel(s) am I profiling?** Get the exact kernel name. mcTracer
   records the demangled name on the event (`saxpy_kernel(int, float, …)`)
   and the mangled symbol in `args.name` (`_Z12saxpy_kernelifPKfPf`). Filter
   on the demangled one.
2. **Which workload / input shape?** If the kernel takes variable-sized
   inputs, pick a **specific** real workload — don't invent shapes. If the
   user has multiple representative shapes, profile the hottest one first.
3. **Which dispatch path?** Many production kernels branch on input shape to
   pick different grid configs or template instantiations. Profile each
   *active* dispatch path separately — treating them as one kernel will
   average out the real patterns.
4. **What question am I answering?** "Why is this slow?" is too vague. Better:
   "At shape X, is the grid big enough to fill 104 SMs, or is this a launch-
   geometry problem?"
5. **What is the baseline?** If there's a reference implementation (torch,
   mcBLAS, a previous version), profile it too for comparison.

If any of 1-4 are unclear, **ask the user** before profiling. Profiling the
wrong thing wastes an hour.

---

## Phase 1 — Environment check

```bash
# 1. MACA SDK + cu-bridge toolchain
source /path/to/kda-maca/env.sh
cucc --version          # cu-bridge CUDA-syntax front end
mxcc --version          # MACA device compiler

# 2. GPU is visible
mx-smi                  # confirm device and driver version
# or: python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. mcTracer is available
$MCTRACER --help        # prints usage; version is in the startup banner

# 4. Python parsing has no external deps — stdlib JSON only
python3 -c "import json; print('OK')"
```

Permissions: mcTracer does not use NVIDIA GPU performance counters, so there
is no `ERR_NVGPUCTRPERM` gate. If tracing fails to attach, see
[`12-common-issues.md`](12-common-issues.md).

---

## Phase 2 — Build a profile target

**Option A (preferred): standalone harness.** Build a small C++ driver that
launches your kernel directly. See [`02-harness-guide.md`](02-harness-guide.md).
The right choice when:

- The kernel lives inside a JIT/template build system (PyTorch inline, Triton,
  CUTLASS JIT) where you want a clean single-kernel trace.
- You want fast iteration — the harness compiles in seconds, vs minutes for
  rebuilding the whole framework.
- You want precise control over inputs (specific workload tensors).

**Option B: profile through existing binary.** Skip the harness if:

- You want to see the kernel in context (which kernels run before/after it,
  how much host-side work happens between launches). This is where mcTracer
  is *stronger* than ncu: one trace captures the whole run, not one kernel.
- The framework build already works and you just need the launch geometry.

Either way, the harness must compile with `$MACA_CUCC_FLAGS` or `cucc` cannot
find `__macro_mxcc.h`.

---

## Phase 3 — Collect the trace

One pass, one output file (no ncu replay passes):

```bash
cd "$PROFILE_RUN_DIR"
$MCTRACER --odname reports --name <tag> ./harness/your_harness [args]
```

`--odname` must be a path **relative to the cwd** — an absolute path fails
with `Output file open error!`. Output lands at
`$PROFILE_RUN_DIR/reports/<tag>-<pid>.json`. Details in
[`03-collection.md`](03-collection.md).

Wall time: the traced run takes roughly 1-2× the untraced run (mcTracer adds
per-API-call bookkeeping, not kernel replay). A harness that runs in 0.1 s
traces in well under a second — much cheaper than ncu's 45-50 replay passes.

---

## Phase 4 — Extract structured data

Do not eyeball the JSON. Parse it in Python so you can compare, aggregate, and
archive. See [`04-python-api.md`](04-python-api.md).

Minimum analysis artifacts to produce:

| Artifact | Tool | What it tells you |
|---|---|---|
| `kernel_inventory_<tag>.txt` | `kernel_inventory.py` | What actually ran: per-kernel counts, time share, grid/block |
| `metrics_key_<tag>.txt` | `analyze_reports.py` | ~12 curated metrics (launch geometry, occupancy, duration) |
| `metrics_all_<tag>_<i>.json` | `analyze_reports.py` | Every field of launch i, archived for later |
| `compare_<a>_vs_<b>.txt` | `analyze_reports.py` | Side-by-side between workloads / versions |
| `pm_timeline_plots_<tag>.txt` | `plot_timeline.py` | ASCII timelines — overlap, queue depth, tail |

Save everything under `$PROFILE_RUN_DIR/analysis/`. The user will want to
re-inspect these; if two runs mix artifacts, you've already failed.

---

## Phase 5 — Diagnose

Work through the analysis dimensions — see
[`05-analysis-dimensions.md`](05-analysis-dimensions.md):

1. **SM occupancy & launch geometry** — are enough blocks launched to fill the
   104 SMs? Is occupancy register- / shared-mem- / block-limited?
2. **Per-launch timeline** — do kernels overlap or serialize? Is there a tail?
3. **Kernel inventory** — where does the total time actually go?
4. **Roofline** — with a known op count, is the kernel compute- or memory-bound?

Counter-dependent dimensions from upstream (stall reasons, cache hit rates,
tensor-core pipe activity) are **not available on MACA** — see
[`11-maca-proxies.md`](11-maca-proxies.md) for what to use instead.

For each dimension you *can* run, write down the observed signal *and the
specific value that produced it*. "Kernel is memory bound" is useless;
something like "`launch__waves_per_multiprocessor = 0.23` means only 24% of
SMs see work, so this is a small-grid problem" is diagnosis. Fill in the
numbers from your own trace.

Then consult [`06-diagnosis-playbook.md`](06-diagnosis-playbook.md), which
maps observed patterns to likely causes and concrete fixes, with each pattern
annotated for MACA observability.

---

## Phase 6 — Write the report

Structure in [`07-report-template.md`](07-report-template.md). Key elements:

1. **Setup section**: exactly how you profiled (harness path, workloads,
   mcTracer commands). Required for reproducibility.
2. **Headline numbers**: duration, grid, block, occupancy, waves/SM. A table
   on the first page.
3. **Per-dimension analysis** with evidence (field values).
4. **Optimization directions** ranked by expected impact. Without ncu's
   `Est. Speedup` engine, the ranking comes from the wave-math and roofline
   estimates in this skill — show the arithmetic.
5. **Confidence & caveats** — including which upstream dimensions you could
   not measure.

Keep the report short enough that a busy reader can see the top 3 findings in
30 seconds. Put deep detail in the artifacts, not the prose.

---

## Anti-patterns to avoid

- ❌ **"I ran mcTracer and it says the kernel takes 86 µs"** — without naming
  the workload, the grid, and the occupancy, this is not actionable. Always
  give metric + value + what it means.
- ❌ **Inventing counter-style numbers.** mcTracer cannot report cache hit
  rates or stall reasons. Saying "long_scoreboard stalls dominate" without a
  source is a fabrication, not a diagnosis. State the gap instead.
- ❌ **Profiling with synthetic shapes that don't match real workloads.** A
  uniform batch is a very different problem from a batch with skewed per-
  element work (the latter exposes tail effects the former hides).
- ❌ **Dumping the raw JSON into the report.** Extract the numbers, cite the
  field, add your reading.
- ❌ **Proposing optimizations without evidence.** "Maybe we should use shared
  memory" is not a profiling result. A real proposal cites the launch
  geometry, the occupancy limit that binds, and the mechanism of the fix.
- ❌ **Missing the #1 finding because you got distracted by a smaller one.**
  Rank findings by impact. Small-grid problems and tail effects usually dwarf
  micro-optimizations; fix the big one first.
