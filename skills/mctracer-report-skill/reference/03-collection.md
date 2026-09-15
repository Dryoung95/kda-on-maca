# Trace Collection Commands

The exact `mcTracer` commands to run, in what order, and what each option
does. Replaces the upstream ncu collection recipes (Recipe 1-3 collapsed into
one trace — mcTracer is single-pass).

---

## Prerequisites recap

- Harness compiled with `$MACA_CUCC_FLAGS` (see
  [`02-harness-guide.md`](02-harness-guide.md)).
- `mcTracer` available on `PATH` or via `$MCTRACER` (set by `env.sh`).
- Writable output directory — `--odname` is created if missing.

Quick sanity check:

```bash
$MCTRACER --odname /tmp/mctracer_smoke --name smoke ./your_harness [args]
ls /tmp/mctracer_smoke/
# expect: smoke-<pid>.json
```

If you see no JSON, see [`12-common-issues.md`](12-common-issues.md).

---

## Recipe 1: The one trace you need

```bash
cd "$PROFILE_RUN_DIR"
$MCTRACER --odname reports --name <tag> ./harness/harness [args]
```

| Option | Meaning |
|---|---|
| `--odname <dir>` | Output directory; created if missing. **Must be relative to the cwd** (see below). |
| `--name <prefix>` | Filename prefix. Becomes `<prefix>-<pid>.json`. |
| `--debug` | Verbose logging — use when the trace comes out empty. |
| `--mctx` | Enable mctx (context) tracing — adds per-context boundaries. Rarely needed. |
| `--attach <pid>` | Attach to an already-running process instead of launching one. |

**The relative-path requirement.** Passing an absolute path to `--odname`
fails with `FATAL: Output file open error!` and produces no trace. This is a
mcTracer limitation, not a permissions problem. Either `cd` into the run
directory first (as above) or invoke it with a cwd-relative directory.

That's the whole collection step. One pass, no replay, no per-metric
collection failures to debug. Contrast with ncu, which needed 45-50 replay
passes for `--set full` and a second `--set source` pass for per-PC stalls.

The trade-off: you get one trace containing **everything** — every kernel
launch, every `mcMalloc`, every sync — with no hardware counters. Filtering
to the kernel you care about happens at parse time (Phase 4), not collection
time.

---

## Recipe 2: Filter to the interesting kernel (parse time)

There is no `-k "regex:..."` at collection time. Instead, filter when parsing:

```python
from mctracer_utils import load_report, action_name
import re

_, actions = load_report("reports/<tag>-*.json".replace("-*", "-12345"))
rx = re.compile(r"saxpy_kernel")
for a in actions:
    if rx.search(action_name(a) or ""):
        ...
```

Or use the CLI's built-in filter:

```bash
python3 kernel_inventory.py --run-dir $PROFILE_RUN_DIR \
    --report $PROFILE_RUN_DIR/reports/<tag>-*.json --tag <tag>
```

To enumerate the kernel names a binary will export before running it:

```bash
mxcc -S my_kernel_harness.cu $MACA_CUCC_FLAGS -o - | grep __global__
# or inspect the compiled binary's symbol table
nm -C my_kernel_harness | grep ' T '
```

---

## Recipe 3: Baseline + candidate (two runs, A/B)

The most common real use — measure a baseline, apply an optimization,
measure the candidate:

```bash
# baseline
cd profile/<kernel>_v1_baseline
$MCTRACER --odname reports --name base ./harness/harness [args]

# candidate — NEW run directory, never overwrite the baseline
cd ../<kernel>_v2_optimized
$MCTRACER --odname reports --name opt ./harness/harness [args]

# compare (--run-dir is absolute-safe; only mcTracer's --odname is not)
python3 analyze_reports.py --run-dir profile/<kernel>_v2_optimized \
    --report profile/<kernel>_v1_baseline/reports/base-*.json --tag base \
    --report profile/<kernel>_v2_optimized/reports/opt-*.json --tag opt
```

Note the comparison runs in the *candidate's* run dir — or use the dedicated
`profile/<kernel>_v2_vs_v1/` comparison directory from
[`00-directory-layout.md`](00-directory-layout.md).

---

## Trace JSON schema (what you get)

Chrome trace-event format. The event that matters:

```json
{
  "ph": "X",                        // complete event: has ts + dur
  "cat": "0",                       // GPU device category
  "ts": 1789465586316884736,        // device start, ns
  "dur": 18944,                     // device-side duration, ns  ← the number
  "name": "saxpy_kernel(int, float, float const*, float*)",  // demangled
  "pid": 2,                         // device id
  "tid": 1024,                      // hardware queue
  "args": {
    "name": "_Z12saxpy_kernelifPKfPf",   // mangled symbol
    "grid": {"x": 4096, "y": 1, "z": 1},
    "block": {"x": 256, "y": 1, "z": 1},
    "mem": {
      "registers_per_thread": 6,
      "static_shared": 0,
      "dynamic_shared": 0,
      "private_per_thread": 0
    },
    "mtreg_occupancy(%)": 1,        // register-limited occupancy (sic typos
    "shared_memeory_occupancy(%)": 0,   //  are in the MACA field names)
    "queue_ts": 1789465586316841406,    // host queue start (NOT exec start)
    "submit_ts": 1789465586316850626,
    "complete_ts": 1789465586316928060, // queue lifecycle end (NOT exec end)
    "device_id": 0,
    "dispatch_id": 0,
    "is_dynamic_parallel": false,
    "is_recompiled": false,
    "launch_type": 0,
    "max_block_size": 512,
    "hw_queue_id": [538976288]
  }
}
```

Non-kernel events include `mcMalloc` / `mcMemcpy` / `mcDeviceSynchronize`
(host-side API calls, `ph: X` without `grid`), metadata process-name events,
and flow events linking host API calls to device execution (`ph: f`).

Full field inventory: see [`10-maca-mapping.md`](10-maca-mapping.md).

---

## Timing semantics — read this before reporting a duration

The trace carries four timestamps. They are NOT interchangeable:

| Field | Meaning | Use |
|---|---|---|
| `dur` | Device-side kernel execution time | **Report this.** Matches `mc_gpu_kernel_sum.py`. |
| `ts` | Device-side start | Timeline plots |
| `queue_ts` → `submit_ts` | Host-side time spent enqueueing | Launch-overhead analysis |
| `queue_ts` → `complete_ts` | Full queue lifecycle (host enqueue + device exec) | Roughly correlates with `dur` but includes queue wait |

The `queue_ts`/`complete_ts` bracket can be off by ~10× from `dur` for short
kernels (e.g. 86 µs vs 432 µs seen in practice on a PyTorch workload) — the
difference is queue depth, not execution time. Always cite `dur` when you say
"this kernel takes N µs".

---

## How long does tracing take?

Roughly 1-2× the untraced runtime. mcTracer adds per-API-call bookkeeping but
does not replay kernels. A harness that runs in 0.1 s traces in well under a
second.

Compare: ncu `--set full` took 30-60 s for the same workload because of
45-50 replay passes. On MACA, collection is never the bottleneck — analysis
is.

---

## Multi-process workloads

`--odname` is per-process. If your harness spawns children ( DataLoader
workers, multi-process data loading), each writes its own `<prefix>-<pid>.json`
under the same `--odname` dir. Pass the specific file to the analysis scripts
rather than the glob when this happens, since the glob becomes ambiguous.
