# Agent Instructions (MetaX C500 / MACA)

This repository is the MACA port of the generic Kernel Design Agents workflow
reference. It should stay small and task-agnostic — same scope rules as
upstream, with the MACA toolchain substituted.

## Repository Rules

- Use English for repository-facing files, comments, documentation, prompts,
  and commit messages. (User-facing conversation may be Chinese.)
- Keep task-specific prompts, datasets, validators, generated implementations,
  benchmark logs, and candidate artifacts out of this repository.
- Treat benchmark competitions as downstream applications of KDA rather than
  the scope of this repository.
- Put generated outputs in `runs/`, `outputs/`, or `profile/`; these paths are
  ignored by git.
- Prefer documenting reusable workflow mechanics over documenting one task's
  private harness or acceptance thresholds.

## Environment (read before running anything)

```
GPU      MetaX C500 — 104 SMs, 16.3 GB, warp size 64 (not 32)
SDK      MACA 3.3.0.15 at /opt/maca-3.3.0
Compile  cucc (cu-bridge CUDA-syntax front end) → mxcc; no nvcc
Profile  mcTracer (API tracer, Chrome JSON); no ncu / no ncu_report
Torch    2.8.0+metax3.3.0.2 (CUDA API mapped onto MACA)
```

`source env.sh` before any build, profile, or run. It sets `MACA_PATH`,
`CUDA_HOME` (to the cu-bridge root), `PATH`, library paths, and exports
`$MACA_CUCC_FLAGS`. Without it, `cucc` is not on `PATH` and builds fail on
`__macro_mxcc.h`.

## Expected Agent Workflow

For a new task:

1. Create or enter a separate implementation workspace.
2. Define the task objective, constraints, validation command, and promotion
   criteria.
3. Use `prompts/basic-flow.md` as the starter prompt.
4. Read local task code and documentation before proposing implementation
   changes.
5. Write the initial plan draft to `docs/draft.md` inside the task workspace.
6. Convert the draft into an executable plan.
7. Implement in small iterations, validating each meaningful candidate.
8. Record candidate relationships, evaluation results, and profiling evidence
   when applicable.
9. Keep this repository focused on the reusable flow.

## Toolchain Notes (the failure modes worth knowing)

- **Compile gate:** `cucc x.cu -o x` fails with
  `fatal error: '__macro_mxcc.h' file not found` unless the cu-bridge include
  path is present. Use `$MACA_CUCC_FLAGS` (from `env.sh`). This is the most
  common build failure.
- **mcTracer `--odname` must be relative to the cwd.** An absolute path
  fails with `Output file open error!` and no trace. `cd` into the run
  directory and pass a bare directory name.
- **KernelBench candidate naming:** the correctness gate requires class
  `ModelNew`, not `Model`. The raw KernelBench CLI additionally needs a
  matching `pydra` version — `scripts/correctness_gate.py` calls
  `eval_kernel_against_ref` directly and avoids that dependency.
- **Kernel duration:** in an mcTracer trace the device-side kernel duration
  is the event's `dur` field (ns). `complete_ts - queue_ts` is the queue
  lifecycle and can be ~10× larger for short kernels. Cite `dur`.
- **Trace field typos are real:** `shared_memeory_occupancy(%)` and
  `mtreg_occupancy(%)` are the actual MACA runtime field names. Cite them
  literally.
- **Two kernel names:** event `name` is demangled
  (`saxpy_kernel(int, …)`), `args.name` is mangled (`_Z12saxpy…`). Filter on
  the demangled one.
- **Memory:** 16.3 GB total. Workloads sized for 48 GB boards OOM here; that
  is a capacity limit, not a backend bug.
- **No counters:** stall reasons, cache hit rates, DRAM throughput, and
  tensor-core utilization are not measurable on this platform. Substitute
  controlled experiments and label the substitution. See
  `skills/mctracer-report-skill/reference/11-maca-proxies.md`.

## Optional Skills

Use external skills only when they are relevant to the active task:

- `humanize` for plan generation and implementation loops.
- A domain knowledge skill for background research (upstream ships
  KernelWiki, NVIDIA-flavored; it is not ported here — see `README.md`).
- `mctracer-report-skill` for profiling and performance evidence on C500.
