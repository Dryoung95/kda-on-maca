# Final Report Template

The report is the deliverable. Everything else (trace JSON, Python artifacts)
is evidence. Structure matters: a busy reader should see the top findings in
30 seconds and be able to drill into details if they want.

Save as `$PROFILE_RUN_DIR/REPORT.md`. Ported from the upstream template; the
headline table and caveats sections are adapted to what MACA can actually
measure.

---

## Template

```markdown
# `<kernel_name>` Profiling Report

**Kernel:** `<exact kernel name>`
**Target GPU:** MetaX C500 (104 SM, 16.3 GB, warp size 64)
**Toolchain:** MACA SDK 3.3.0.15, cu-bridge (cucc), mcTracer 3.3.0.15
**Compile flags:** `cucc -O2 -std=c++17 -DUSE_MACA -I<cu-bridge>/include`
**Profile date:** YYYY-MM-DD
**Run directory:** `profile/<run_name>/`

---

## 0. Profiling setup

> How exactly did we get these numbers? Required for reproducibility.

- Harness: `profile/<run_name>/harness/*.cu` — what it is (standalone driver /
  the original binary). Why.
- Workloads: which shapes / tensors were used.
- Dispatch paths covered: each `(template params, grid, block)` combination.
- Substitutions: any upstream dimension replaced by a MACA proxy, and which
  experiment stands in for it.

Minimal runnable command listing:

    # Compile
    source <kda-maca>/env.sh
    cucc harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o harness

    # Trace (single pass; --odname is relative to the cwd)
    cd profile/<run_name>
    mcTracer --odname reports --name <tag> ./harness/harness [args]

    # Analyze
    python3 helpers/kernel_inventory.py --run-dir . \
        --report reports/<tag>-*.json --tag <tag>
    python3 helpers/analyze_reports.py --run-dir . \
        --report reports/<tag>-*.json --tag <tag>
    python3 helpers/plot_timeline.py --run-dir . \
        --report reports/<tag>-*.json --tag <tag>

### Artifacts

    profile/<run_name>/
    ├── REPORT.md                       ← this file
    ├── harness/...                     ← standalone harness
    ├── reports/<tag>-<pid>.json        ← raw mcTracer traces
    └── analysis/                       ← scripts + extracted metrics

---

## 1. Headline numbers

> A single table that tells the whole story at a glance.

| Metric | `<tag1>` | `<tag2>` | Source |
|---|---:|---:|---|
| **Duration** | X µs | Y µs | `dur` |
| Grid size (blocks) | X | Y | `launch_geometry` |
| Block size (threads) | X | Y | `launch_geometry` |
| Waves / SM | X | Y | `launch_geometry` |
| Last-wave utilization | X% | Y% | `launch_geometry` |
| Registers / thread | X | Y | `mem.registers_per_thread` |
| Shared mem / block | X B | Y B | `mem.static_shared + dynamic_shared` |
| Theoretical occupancy | X% | Y% | derived |
| Reg-limited occupancy (runtime) | X% | Y% | `mtreg_occupancy(%)` |
| In-flight launch depth | X | Y | `plot_timeline.py` |
| Achieved FLOPs (% peak) | X% | Y% | `roofline()` + supplied op count |

Rows that are n/a must be omitted, not filled with guesses.

---

## 2. Per-dimension analysis with evidence

For each of the four supported dimensions, the observed signal and the
specific value that produced it:

- **Dim 1 — Launch geometry:** grid vs 104 SMs, waves/SM, which occupancy
  limit binds. Cite `metrics_key_<tag>.txt`.
- **Dim 2 — Timeline:** in-flight shape, tail, serialization. Cite
  `pm_timeline_plots_<tag>.txt`.
- **Dim 3 — Inventory:** time share by kernel. Cite
  `kernel_inventory_<tag>.txt`.
- **Dim 4 — Roofline:** compute- vs memory-bound, with the op count used and
  where it came from.

Then the gapped dimensions, stated plainly:

> **Dim 5 (stall reasons) and Dim 6 (memory access / cache efficiency) are
> not measurable on MACA** — mcTracer is an API-level tracer without hardware
> performance counters. The binding constraint for this kernel was
> established by <experiment> instead. See
> `reference/11-maca-proxies.md`.

---

## 3. Optimization directions, ranked

Without ncu's `Est. Speedup` engine, the ranking must be backed by either
wave math or a controlled experiment. Cite which:

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <field values>
  MACA observability: ✅ / 🔶 / ❌
  Estimated impact: <backed by experiment E# or wave math>
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>
```

At most 3-5 priorities. More dilutes the signal.

---

## 4. Experiments

The evidence for any 🔶 or ❌-observability finding. For each: what you
varied, what you measured, what you concluded.

| ID | Varied | Measured | Result | Conclusion |
|---|---|---|---|---|
| E1 | block size 128 → 256 | duration | 86 µs → 61 µs | occupancy-limited |
| E2 | tile factor 16 → 32 | duration | 61 µs → 59 µs | not register-limited |

---

## 5. Confidence & caveats

- Which findings are direct trace measurements (high confidence).
- Which are inferred via proxy (medium), and the experiment that backs them.
- Which upstream dimensions could not be measured at all.
- Any device-parameter assumptions (`peak_f32_flops` is an estimate —
  relative comparisons are sound, absolute percentages are approximate).
- Whether the harness matches the production dispatch path.
```

---

## What makes a MACA report good

The upstream bar was "cite the metric value that backs each claim." The MACA
bar is the same, plus one extra: **state the observability tag** on every
finding. A reader needs to know whether a number came from the trace, from a
proxy experiment, or from static reasoning — because only the first is
reproducible by re-running mcTracer.

Three failure modes to avoid:

- **Inventing counter-style numbers.** "Cache hit rate 40%" cannot exist on
  this platform. Write the gap instead.
- **Reporting queue time as kernel time.** `queue_ts → complete_ts` includes
  queue wait and can be ~10× `dur`. Always cite `dur`.
- **Presenting derived occupancy as measured.** `launch_geometry()` computes
  limits from device constants; if the runtime-reported
  `mtreg_occupancy(%)` disagrees, the runtime wins and the discrepancy goes
  in the report.
