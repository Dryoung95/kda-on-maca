# Profile Directory Layout & Naming

**Read this first, before any collection.** Bad directory layout is the single
most common cause of mixing results from different runs, overwriting prior
profiles, or losing track of which trace belongs to which kernel version. The
rules below are non-negotiable for work in this repo.

Ported from the upstream ncu-report-skill; identical conventions, different
file extensions (`.json` traces instead of `.ncu-rep`).

---

## Top-level rule

**All profiling artifacts live under a single `profile/` directory at the repo
root.** Never scatter trace files across random locations. Never put profile
artifacts under `solution/`, `src/`, `scripts/`, or other source directories.

```
<repo_root>/
├── profile/                        ← everything profiling-related lives here
│   ├── <run_1>/
│   ├── <run_2>/
│   └── ...
├── solution/                       ← untouched by profiling
├── src/
└── ...
```

---

## One run = one subdirectory

Every time you profile a kernel — whether it's a new kernel, a new version of
the same kernel, or the same kernel on a different workload — **create a new
subdirectory under `profile/`**. Never write into an existing run's directory.

Rationale:

- Traces of different implementations of the same kernel must not overwrite
  each other. If you profile `<kernel>_v1` today and `<kernel>_v2` tomorrow,
  both traces need to coexist for A/B comparison.
- The harness itself is part of the profile: it encodes which kernel code was
  compiled, with which flags, against which workload. Keeping the harness
  source in the run dir pins the provenance.
- Analysis artifacts (`metrics_*.json`, `compare_*.txt`, ASCII plots) are tied
  to a specific set of traces; they must not be mixed.

---

## Run directory naming

Use descriptive, short, kebab-case names. Include **what** was profiled and
**when/why**, not how.

Good:
```
profile/<kernel>_v1_baseline/
profile/<kernel>_v2_optimized/
profile/<kernel>_v2_optimized_vs_v1/      # for comparison run
profile/matmul_tiled_64_baseline/
```

Bad:
```
profile/test/                   # too vague
profile/run1/                   # meaningless
profile/20260915/               # dates with no context
profile/final/                  # there's never a "final"
```

---

## Standard run layout

```
profile/<run_name>/
├── REPORT.md                       ← human-readable final report (Markdown)
├── harness/
│   ├── <kernel>_harness.cu         ← the exact source that was compiled
│   ├── <kernel>_harness            ← compiled binary
│   └── build_command.sh            ← optional: shell script that compiled it
├── reports/
│   └── <tag>-<pid>.json            ← mcTracer trace (one per process)
└── analysis/
    ├── analyze_reports.py          ← the script that produced the extractions
    ├── kernel_inventory_<tag>.txt  ← whole-run kernel inventory
    ├── metrics_all_<tag>_<i>.json  ← full field archive of launch i
    ├── metrics_key_<tag>.{txt,json}← curated key metrics
    ├── compare_<a>_vs_<b>.txt      ← side-by-side
    └── pm_timeline_plots_<tag>.txt ← ASCII timelines
```

Notes:

- mcTracer names its output `<prefix>-<pid>.json` — the pid suffix is
  unavoidable, so scripts accept a glob (`reports/<tag>-*.json`) and take the
  unique match. If you trace a multi-process workload you will get several;
  in that case pass the exact file instead of the glob.
- `<tag>` is the per-workload / per-dispatch-path label, e.g. `shape_a`,
  `shape_b`. Pick tags that are short and name the representative workload.
- Keep `analysis/analyze_reports.py` as a per-run copy (pointing at the
  run-local `reports/`), not a symlink into the repo. This way the run is
  self-contained and archivable.

---

## Comparing two runs

For A/B comparisons, create a comparison run that *references* both underlying
runs:

```
profile/<kernel>_v2_vs_v1/
├── REPORT.md                       ← describes both runs + the comparison
└── analysis/
    ├── compare.py                  ← loads traces from the two runs below
    └── compare_key_metrics.txt     ← side-by-side on key metrics
    (No trace files — they live in the referenced runs)
```

In `compare.py`, hardcode the paths to both referenced runs:
```python
V1_DIR = Path("/abs/path/to/profile/<kernel>_v1_baseline")
V2_DIR = Path("/abs/path/to/profile/<kernel>_v2_optimized")
```

The comparison run does not re-profile; it only produces comparison artifacts
and prose.

---

## Environment variable convention (optional but recommended)

Scripts pick up the run directory from a single env var, so they're easy to
redirect to different runs:

```bash
export PROFILE_RUN_DIR=/abs/path/to/profile/<kernel>_v1_baseline
mkdir -p "$PROFILE_RUN_DIR"/{harness,reports,analysis}

# build harness
cucc harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o "$PROFILE_RUN_DIR/harness/kernel_harness"

# trace (--odname must be relative to the cwd, so cd first)
cd "$PROFILE_RUN_DIR"
mcTracer --odname reports --name <tag> ./harness/kernel_harness [args]

# parse (--run-dir is fine as an absolute path)
python3 analyze_reports.py --run-dir "$PROFILE_RUN_DIR" \
    --report "$PROFILE_RUN_DIR/reports/<tag>-*.json" --tag <tag>
```

All helper scripts accept `--run-dir` and write analysis output under it.

---

## Checklist before starting a profile run

1. `mkdir -p profile/<new_run_name>/{harness,reports,analysis}` — make the
   three subdirs up front.
2. Copy or write the harness source into `profile/<new_run_name>/harness/`.
3. Compile into the same dir (with `$MACA_CUCC_FLAGS`).
4. Trace with `mcTracer --odname profile/<new_run_name>/reports --name <tag>`.
5. Put analysis scripts under `profile/<new_run_name>/analysis/`.
6. Write `REPORT.md` at `profile/<new_run_name>/REPORT.md`.
7. Before starting a *new* run, go back to step 1 with a new name — never
   write into the existing one.
