# C500 Microarchitecture Probes

Small standalone kernels that measure one hardware property each. They are the
evidence source for the [hardware pages](../hardware/) in this wiki — on this
platform the toolchain ships no performance-counter profiler, so measurement is
the only way to establish an architecture fact.

## Build and run

```bash
source /data/kda-maca/env.sh
cucc probe_blocksize.cu -O2 -std=c++17 $MACA_CUCC_FLAGS -o probe_blocksize
./probe_blocksize
```

Each probe prints its own results to stdout. Reproduce a wiki number before
quoting it, especially after an SDK upgrade — these are properties of one
toolchain version, not eternal truths.

Every probe here is checked by `scripts/validate.py`, which compiles each one
cited by a wiki page and fails if it does not build. `scripts/generate-indices.py`
then emits `queries/by-probe.md`, the reverse index from each probe to the pages
citing it — the re-verification worklist after an SDK upgrade.

## Probes

| File | Measures | Feeds |
|---|---|---|
| `probe_blocksize.cu` | elementwise throughput vs block size (32→1024) | [warp64-implications](../wiki/hardware/warp64-implications.md) |
| `probe_smem_load.cu` | shared-memory read cost vs stride (barrier-free) | [shared-memory-banks](../wiki/hardware/shared-memory-banks.md) |
| `probe_smem_store.cu` | shared-memory write cost vs stride (barrier-free) | same |
| `probe_smem_mixed.cu` | load+store+barrier cost vs stride | same |
| `probe_sync.cu` | `__syncthreads` cost vs count per iteration | same |

## Method

Every probe follows the same discipline so results are comparable:

1. **Warm up** 3–5 launches before timing, so module load and context warmup
   land outside the measurement window.
2. **Repeat** 10–50 kernel launches inside one `cudaEvent` window and divide —
   single-launch timing is dominated by launch overhead on short kernels.
3. **Report the ratio to stride-1 or to the minimum**, not the absolute time, so
   results transfer between boards with different clocks.
4. **Sink the output** into a global write guarded by a never-true condition, so
   the compiler cannot dead-code-eliminate the work being measured.
5. **Isolate one operation per probe.** The load/store/mixed split exists
   precisely because the mixed probe's ~3.5× would be misread as a uniform bank
   conflict if the isolated probes did not show flat *reads* alongside 6.4×
   *writes*. The asymmetry is the finding.
6. **Keep addresses iteration-dependent.** A stride-only address can be hoisted
   out of the loop by the compiler, which makes the stride-1 baseline bimodal
   (observed 32.6 µs hoisted vs 81.7 µs not, same kernel). Every probe here adds
   `+ it` to its index so the access cannot be promoted out of the loop.

## What these are not

These measure hardware behavior, not kernel quality. A probe result is a fact
about the silicon; whether it matters to your kernel depends on your access
pattern. Use them to choose which optimization hypothesis to test, then test
that hypothesis on your actual kernel with the
[mctracer-report-skill](../../mctracer-report-skill/SKILL.md).
