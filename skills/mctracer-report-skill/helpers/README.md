# Helpers

Reusable code for profiling harnesses and report analysis. See `../SKILL.md`
for context. These are the MACA ports of the upstream ncu-report-skill
helpers — same CLI shape, different backend.

## C++ / CUDA

| File | Purpose |
|---|---|
| `harness_template.cu` | Starting point for a profiling harness. Copy into your run dir, fill in the `TODO(you)` sections. |

### Typical harness setup

```bash
source /path/to/kda-on-maca/env.sh
cd profile/<run_name>/harness/
cp /path/to/skills/mctracer-report-skill/helpers/harness_template.cu my_kernel_harness.cu
# edit my_kernel_harness.cu to include your kernel + fill in main()

cucc my_kernel_harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o my_kernel_harness
```

Note: `$MACA_CUCC_FLAGS` (`-DUSE_MACA -I<cu-bridge>/include`) is required —
without it `cucc` fails on `__macro_mxcc.h`.

## Python

| File | Purpose |
|---|---|
| `mctracer_utils.py` | Shared helpers: `load_report`, `safe`, `launch_geometry`, `kernel_timeline`, `C500_KEY_METRICS`, … |
| `analyze_reports.py` | Extract key metrics + side-by-side comparisons from one or more traces |
| `plot_timeline.py` | ASCII per-launch timeline plots (queue depth, overlap, tail) |
| `kernel_inventory.py` | Whole-run kernel inventory: names, launch counts, time share, grid/block |

### Typical Python workflow

```bash
export PROFILE_RUN_DIR=profile/<run_name>
HELPERS=/path/to/skills/mctracer-report-skill/helpers

# Full inventory of everything that ran (no upstream equivalent; cheap and useful)
python3 $HELPERS/kernel_inventory.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/<tag>-*.json --tag <tag>

# Extract key metrics for each report
python3 $HELPERS/analyze_reports.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/<tag1>-*.json --tag <tag1> \
    --report $PROFILE_RUN_DIR/reports/<tag2>-*.json --tag <tag2>

# ASCII timeline plots
python3 $HELPERS/plot_timeline.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/<tag1>-*.json --tag <tag1>
```

Note: the glob (`reports/<tag>-*.json`) must be expanded by the **shell**, so
leave `--report` unquoted — the scripts take literal paths and do not glob
internally. The pattern is needed because mcTracer appends the traced
process's pid to the filename. `--run-dir` may be absolute; mcTracer's
`--odname` may not — see `reference/03-collection.md`.

`analyze_reports.py` and `plot_timeline.py` take repeated `--report/--tag`
pairs for side-by-side comparison; `kernel_inventory.py` takes a single
`--report`/`--tag`.

All three scripts take `--run-dir` and write under `<run-dir>/analysis/`.
`mctracer_utils.py` is pure Python stdlib — no `ncu_report`, no `PYTHONPATH`
setup, no CUDA install to locate.
