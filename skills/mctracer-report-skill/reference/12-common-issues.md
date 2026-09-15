# Common Issues & Gotchas

Collected solutions for the recurring frustrations of profiling with the MACA
toolchain. Replaces the upstream ncu permissions/error reference.

---

## `cucc` fails: `fatal error: '__macro_mxcc.h' file not found`

The single most common build failure. `cucc` needs the cu-bridge macro
headers, which are not on its default include path.

**Fix:** pass the include path explicitly.

```bash
source /path/to/kda-maca/env.sh
cucc harness.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o harness
# expands to: -DUSE_MACA -I/opt/maca-3.3.0/tools/cu-bridge/include
```

If you call `mxcc` directly, the flag is the same. The `env.sh` script also
puts the directory on `CPATH`, so plain `cucc harness.cu` works too once it
is sourced.

---

## `cucc: command not found`

`cucc` lives under the cu-bridge directory, not `bin/` at the SDK root:

```
/opt/maca-3.3.0/tools/cu-bridge/bin/cucc
```

**Fix:** `source /path/to/kda-maca/env.sh` (it prepends the cu-bridge `bin`
to `PATH`), or call it by full path.

---

## mcTracer produces no JSON

**Symptom:** the traced program runs and exits, but the output directory is
empty or was never created.

**Checks:**

1. **`--odname` was an absolute path.** This is the usual cause: mcTracer
   logs `FATAL: Output file open error!` and writes nothing. Use a path
   relative to the cwd — `cd` into the run directory and pass `reports`:
   ```bash
   cd profile/<run_name>
   $MCTRACER --odname reports --name <tag> ./harness/harness
   ```
2. **You passed `--odname <dir>`, not a file.** mcTracer creates the
   directory. If you pass an existing *file* path, output fails silently.
3. **The target command actually ran.** mcTracer wraps the target with
   `execvpe`; if the binary path is wrong you see
   `execvpe: No such file or directory` and get no trace.
4. **Run with `--debug`** for verbose logging:
   ```bash
   $MCTRACER --debug --odname dbg_out --name dbg ./harness
   ```
5. **GPU init failed.** If the target cannot see the GPU, no kernels launch
   and the trace contains no kernel events. On this host the historical root
   cause was the user not being in the `video` group — check
   `getent group video`.
6. **`--name` sets only the prefix.** The output file is
   `<dir>/<prefix>-<pid>.json`; look for the pid suffix.

---

## Trace has no kernel events

**Symptom:** JSON exists but `load_report()` returns zero actions.

**Checks:**

1. **The kernel never launched.** Run the harness *without* mcTracer and
   confirm it reaches the launch (a `printf` after `cudaDeviceSynchronize()`
   is enough).
2. **You are looking at a host-side event stream.** Kernel events are
   `ph: "X"` with a `grid` key in `args`. `mcMalloc`, `mcMemcpy`,
   `mcDeviceSynchronize` are also `X` events but have no `grid` —
   `load_report` filters them out by design.
3. **The kernel is in a library you didn't trace.** mcTracer traces the
   process, so library kernels do appear; if you expected your own kernel
   and see only `mcblas__...`, the harness took the library path.
4. **First launch only paid warm-up.** The first kernel launch in a process
   includes module load and context warmup; the harness template launches
   once for warm-up before the measured launch.

---

## Kernel name mismatch when filtering

**Symptom:** your regex finds nothing, though the kernel clearly ran.

**Cause:** mcTracer records two names per kernel event, and they are easy to
confuse:

- `event["name"]` — the **demangled** form: `saxpy_kernel(int, float, float
  const*, float*)`. **Filter on this.**
- `event["args"]["name"]` — the **mangled** symbol: `_Z12saxpy_kernelifPKfPf`.

**Fix:** inspect first, filter second:

```python
from mctracer_utils import load_report, action_name
_, actions = load_report("reports/tag-12345.json")
for a in actions:
    print(action_name(a))
```

Templates show up as `my_kernel<(int)8, (int)256>(...)` in the demangled
form — escape regex metacharacters if you match on those.

To list kernel symbols before running:

```bash
nm -C ./harness | grep ' T '
```

---

## Occupancy fields look wrong

**Symptom:** `mtreg_occupancy(%)` = 1 for a kernel with 6 registers and
256 threads, and you expected something higher.

**This is correct, not a bug.** The runtime reports the occupancy *limit
caused by that resource*, and a tiny register count with 256 threads can
still mean only a few blocks per SM depending on the arch's register
allocation granularity. Cross-check with the derived values from
`launch_geometry()`, and when the two disagree, treat the runtime number as
authoritative and record the discrepancy.

---

## Field names contain typos

The trace really does contain `shared_memeory_occupancy(%)` (sic) and
`mtreg_occupancy(%)`. These are MACA runtime field names, not transcription
errors in this skill — cite them by their literal key, including the typo.

---

## mcTracer startup is slow / RPC connection delays

Startup takes ~5 s to establish the tracer RPC connection before the target
runs. That is constant overhead, not per-kernel cost — it does not affect
`dur`, which is measured device-side.

For short harnesses this overhead dominates wall time. Don't loop the traced
command; trace once and analyze the JSON.

---

## Tracing a PyTorch process

mcTracer works on Python targets, with caveats:

```bash
$MCTRACER --odname reports --name pt python3 my_bench.py
```

- The trace will contain many framework kernels (`mcGetDeviceProperties`,
  `mcInit`, generator kernels). Use `kernel_inventory.py` to find the one you
  care about — that is exactly the filtering problem it exists to solve.
- `torch.profiler` is an alternative that gives op-level attribution; see
  Proxy 6 in `11-maca-proxies.md`.
- `ProfilerActivity` cannot be iterated as a list on this build — enumerate
  it directly (`[ProfilerActivity.CPU, ProfilerActivity.CUDA]`) instead of
  iterating `ProfilerActivity`.

---

## Permission / group access

mcTracer does not use NVIDIA performance counters, so there is no
`ERR_NVGPUCTRPERM` gate and no `NVreg_RestrictProfilingToAdminUsers` setting
to adjust. Device access is the only permission that matters:

```bash
getent group video       # your user should be a member
mx-smi                   # or python3 -c "import torch; torch.cuda.is_available()"
```

If GPU init fails with a permission error, group membership is the first
thing to check — it was the root cause of a previous GPU-init failure on this
host.
