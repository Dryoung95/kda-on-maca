#!/usr/bin/env python3
"""Correctness gate for a KDA-MACA candidate, via the migrated KernelBench.

Profiling and correctness are separate concerns on this stack: mcTracer tells
you launch geometry and timing, KernelBench tells you whether the kernel
produces the right numbers. Run this before promoting any candidate.

Usage:
    source /data/cuda-harness-migration/env.sh   # needs KernelBench on PYTHONPATH
    python3 correctness_gate.py --level 1 --problem 1 --candidate my_kernel.py

The candidate file must define a `ModelNew` nn.Module — the KernelBench
convention. If it defines a custom CUDA kernel through
torch.utils.cpp_extension, that path is already verified working on this host.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from kernelbench.eval import eval_kernel_against_ref
except ImportError:
    sys.exit("kernelbench not importable — source /data/cuda-harness-migration/env.sh first")

DATASET_ROOT = Path("/data/cuda-harness-migration/KernelBench/KernelBench")


def problem_path(level: int, problem: int) -> Path:
    lvl = DATASET_ROOT / f"level{level}"
    matches = sorted(lvl.glob(f"{problem}_*.py"))
    if not matches:
        sys.exit(f"no problem file matching level{level}/{problem}_*.py under {lvl}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--problem", type=int, required=True)
    ap.add_argument("--candidate", required=True,
                    help="Path to a .py defining class ModelNew")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref_file = problem_path(args.level, args.problem)
    ref = ref_file.read_text()
    cand = Path(args.candidate).read_text()
    if "ModelNew" not in cand:
        sys.exit(f"{args.candidate} does not define class ModelNew — "
                 "KernelBench requires that name")

    res = eval_kernel_against_ref(
        original_model_src=ref,
        custom_model_src=cand,
        verbose=args.verbose,
    )
    print(f"problem:  {ref_file.name}")
    print(f"compiled: {res.compiled}")
    print(f"correct:  {res.correctness}")
    sys.exit(0 if (res.compiled and res.correctness) else 1)


if __name__ == "__main__":
    main()
