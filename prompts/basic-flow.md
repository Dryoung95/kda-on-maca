# Kernel Design Agents Basic Flow Prompt (MetaX C500 / MACA)

You are working in a task implementation workspace on a MetaX C500 GPU
(MACA toolchain). Your job is to produce the best correct implementation for
the task described below.

This is the MACA port of the upstream KDA `prompts/basic-flow.md`. The
workflow is unchanged; the toolchain references are.

## Task Contract

- Task name: `<fill in>`
- Objective: `<fill in the user-facing goal>`
- Correctness requirements: `<fill in required behavior, tolerances, or invariants>`
- Performance or quality target: `<fill in measurable target if any>`
- Allowed implementation approaches: `<fill in languages, libraries, APIs, or constraints>`
- Validation command: `<fill in the command that proves correctness>`
- Evaluation command: `<fill in the command that measures the target, if different>`
- Promotion criteria: `<fill in what must be true before a candidate is accepted>`

## Environment

- **GPU:** MetaX C500 — 104 SMs, 16.3 GB, warp size 64 (note: 64, not 32).
- **Toolchain:** MACA SDK 3.3.0.15, cu-bridge (`cucc`, CUDA-syntax front end
  to `mxcc`). There is no `nvcc`; CUDA-syntax source compiles through
  `cucc`.
- **Profiling:** `mcTracer` (API-level tracer, Chrome-trace JSON output).
  There is no `nsight-compute` / `ncu` and no `ncu_report` module — use the
  `mctracer-report-skill` instead.
- **Framework:** PyTorch 2.8.0+metax (CUDA API surface mapped onto MACA).
- **Compile gate:** every `cucc` invocation needs `-DUSE_MACA
  -I/opt/maca-3.3.0/tools/cu-bridge/include` (or `source env.sh` and use
  `$MACA_CUCC_FLAGS`) or the build fails on `__macro_mxcc.h`.
- **Memory limit:** 16.3 GB total. Size workloads accordingly — large
  workloads that fit on 48 GB boards OOM here. Treat OOM as a workload-sizing
  issue, not a backend defect.

Always `source /path/to/kda-maca/env.sh` before building, profiling, or
running anything.

## Workflow

1. Read the repository structure, existing implementation, tests, and task
   documentation.
2. Identify the baseline behavior and the validation path.
3. Research only the references needed for this task.
4. Write an implementation-plan draft to `docs/draft.md`.
5. Turn the draft into an executable plan before editing code.
6. Implement one candidate at a time.
7. Run validation after each meaningful candidate.
8. Record candidate results, parent relationships, and evidence in the
   workspace.
9. Keep the final change scoped to the task contract.

## Plan Draft Requirements

The draft in `docs/draft.md` should include:

- The current baseline and how it is validated.
- The main risks and unknowns.
- Candidate implementation directions ranked by expected value and risk.
- The first concrete implementation steps.
- The exact validation and evaluation commands to run.
- The evidence required to promote, revise, or reject a candidate.

Do not start implementation until the draft exists.

## Evidence Requirements on MACA

Two things differ from the NVIDIA-flavored workflow and both affect evidence:

1. **Correctness is separate from profiling.** Validate with the correctness
   gate, which calls the migrated KernelBench `eval_kernel_against_ref`:
   ```bash
   source /data/cuda-harness-migration/env.sh
   python3 /path/to/kda-maca/scripts/correctness_gate.py \
       --level <L> --problem <P> --candidate <your ModelNew .py>
   ```
   Never couple a correctness check to a profiling run — it pollutes the
   trace and is not a stable timing measurement.
2. **Profiling evidence has stated gaps.** `mcTracer` gives launch geometry,
   register counts, occupancies and exact device-side durations — but no
   stall reasons, cache hit rates, or pipe utilization. When a diagnosis
   needs one of those, run a controlled experiment (vary the knob, re-time)
   and cite the experiment, not an invented metric. See
   `skills/mctracer-report-skill/reference/11-maca-proxies.md`.

Promote a candidate only when it satisfies the task contract and has
evidence that it improves or preserves the target metric. If you reject a
candidate, record the reason instead of silently discarding it.
