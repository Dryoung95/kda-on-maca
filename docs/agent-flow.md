# Agent Flow (MetaX C500 / MACA)

Kernel Design Agents is a repeatable loop for agent-driven implementation
work. The loop is useful when a task needs both exploration and
evidence-based promotion decisions.

This is the MACA port of the upstream KDA `docs/agent-flow.md`. The flow is
unchanged; only the toolchain references differ.

## Principle

Keep the reusable workflow separate from the task workspace. This repository
explains the flow. The task workspace owns code, tests, datasets, benchmark
scripts, private rules, and generated artifacts.

MLSys-style benchmark work is one possible application of this loop. The same
flow can also be used for compiler passes, runtime kernels, infrastructure
changes, or other performance-sensitive tasks.

## Minimal Loop

1. Define the task contract.
2. Let the agent inspect the local workspace.
3. Make the agent write `docs/draft.md`.
4. Convert the draft into an executable plan.
5. Implement the first candidate.
6. Validate correctness.
7. Measure the target metric when applicable.
8. Record evidence and decide whether to keep, revise, or reject the
   candidate.
9. Repeat until the promotion criteria are met or the remaining blockers are
   explicit.

## Task Contract

Each task should state:

- Objective.
- Inputs and outputs.
- Correctness requirements.
- Constraints on implementation language, dependencies, APIs, or deployment.
- Validation command.
- Evaluation command, if different from validation.
- Promotion criteria.

## Evidence Records

Use simple files in the task workspace:

- `docs/draft.md` for the first plan draft.
- `docs/plan.md` for the executable plan.
- `benchmark.csv` or another tabular log for measurable results.
- `candidates.jsonl` for candidate names, parent links, and status.
- `profile/` for profiler output or report summaries.
- `runs/` or `outputs/` for generated artifacts.

The exact format is less important than consistency. A future reader should be
able to reconstruct what changed, what was measured, and why a candidate was
promoted.

## Profiling on MACA

The evidence step runs against a different profiler than upstream, and the
difference is worth stating explicitly because it changes what an evidence
record can contain:

- **Tool:** `mcTracer` (single-pass API tracer, Chrome-trace JSON), not
  `nsight-compute`. One trace captures the whole run — every launch, every
  allocation, every sync.
- **Available:** launch geometry (grid/block), registers per thread,
  shared-memory usage, register/shared-memory-limited occupancy, and exact
  device-side duration per launch.
- **Not available:** stall reasons, per-PC hotspots, cache hit rates, DRAM
  throughput, tensor-core pipe activity, register-spill counts, and the NCU
  rule engine with its `Est. Speedup` figures.

When a promotion decision needs a signal in the second list, substitute a
controlled experiment (vary the knob, re-time, record the delta) and cite the
experiment. See
[`../skills/mctracer-report-skill/reference/11-maca-proxies.md`](../skills/mctracer-report-skill/reference/11-maca-proxies.md).

An evidence record on MACA that says "not measurable, established by
experiment E2" is complete. One that invents a counter value is not.

## Promotion Rule

Promote a candidate only when it satisfies the task contract and has
evidence that it improves or preserves the target metric. If a candidate is
rejected, record the reason instead of silently discarding it.
