#!/usr/bin/env python3
"""Run the KernelBench correctness gate across a whole level on C500/MACA.

This measures ecosystem compatibility, not optimization skill: every problem is
run with its OWN reference implementation as the candidate (the `Model` class
renamed to `ModelNew`). A pass means "CUDA-syntax source compiles and produces
correct results through cucc + MACA without source modification."

That is the first-order adaptation question — before asking whether an agent
can optimize a kernel on C500, we ask whether the kernel corpus even runs here.

Usage:
    source /data/kda-maca/env.sh
    source /data/cuda-harness-migration/env.sh
    python3 scripts/run_level.py --level 1 [--out results/level1.csv]
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path("/data/kda-maca/scripts/correctness_gate.py")
DATASET = Path("/data/cuda-harness-migration/KernelBench/KernelBench")


def rename_to_model_new(src: str) -> str:
    """KernelBench ships `class Model`; the gate requires `ModelNew`.

    Rename every reference, not just the class definition: these files call
    `super(Model, self)` in __init__, and leaving that pointing at the old
    name raises `obj must be an instance or subtype of type`."""
    src = re.sub(r"\bclass\s+Model\b", "class ModelNew", src, count=1)
    return re.sub(r"\bsuper\(\s*Model\s*,", "super(ModelNew,", src)


def classify_failure(blob: str) -> str:
    """A failed problem is either a capacity limit or a real defect.

    The distinction changes the conclusion: an OOM problem recovers on a
    bigger board, a numeric problem does not. The allocator message in the
    gate's stderr is the only reliable place to split the two."""
    if "OutOfMemoryError" in blob or "out of memory" in blob.lower():
        return "oom"
    return "numeric"


def run_one(level: int, problem: int, timeout: int = 300):
    lvl = DATASET / f"level{level}"
    matches = sorted(lvl.glob(f"{problem}_*.py"))
    if not matches:
        return {"level": level, "problem": problem, "name": "<missing>",
                "compiled": False, "correct": False, "error": "no matching file"}
    ref_file = matches[0]
    src = rename_to_model_new(ref_file.read_text())

    with tempfile.TemporaryDirectory() as td:
        cand = Path(td) / f"cand_{problem}.py"
        cand.write_text(src)
        try:
            r = subprocess.run(
                ["python3", str(GATE), "--level", str(level),
                 "--problem", str(problem), "--candidate", str(cand)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"level": level, "problem": problem, "name": ref_file.stem,
                    "compiled": False, "correct": False, "error": "timeout"}

        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        # Gate prints "compiled: {b}" and "correct:  {b}" — note the two spaces
        # after `correct:` in the gate's f-string. Parse by line prefix instead
        # of substring, or a pass reads as a fail.
        def field(name):
            for line in out.splitlines():
                if line.startswith(name + ":"):
                    return line.split(":", 1)[1].strip()
            return ""
        compiled = field("compiled").lower() == "true"
        correct = field("correct").lower() == "true"
        # Keep the failure reason short — full stderr goes to the verbose log.
        reason = ""
        if not compiled or not correct:
            lines = [l for l in out.splitlines() if "compiled:" in l or "correct:" in l]
            reason = "; ".join(lines) if lines else (err.splitlines() or [""])[0][:200]
        # class distinguishes "board too small" from "actually wrong". Without
        # it the two collapse into one 40-row failure list and the headline
        # conclusion ("only 13 are real") becomes unverifiable from the CSV.
        row = {"level": level, "problem": problem, "name": ref_file.stem,
               "compiled": compiled, "correct": correct,
               "class": classify_failure(out + "\n" + err) if not correct else "pass",
               "error": reason}
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--out", default=None, help="csv path (default results/levelN.csv)")
    ap.add_argument("--verbose-log", default=None, help="optional file for full stderr")
    args = ap.parse_args()

    out_path = Path(args.out or f"results/level{args.level}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lvl = DATASET / f"level{args.level}"
    problems = sorted({int(p.name.split("_")[0]) for p in lvl.glob("*.py")})
    if not problems:
        sys.exit(f"no problems under {lvl}")

    verbose = open(args.verbose_log, "w") if args.verbose_log else None

    rows = []
    passed = 0
    for i, p in enumerate(problems, 1):
        row = run_one(args.level, p)
        rows.append(row)
        if row["compiled"] and row["correct"]:
            passed += 1
        status = "PASS" if (row["compiled"] and row["correct"]) else "FAIL"
        print(f"[{i:>3}/{len(problems)}] L{args.level} P{p:<3} {status:4}  {row['name']}"
              + (f"   ({row['error'][:80]})" if status == "FAIL" else ""),
              flush=True)
        if verbose and row["error"]:
            verbose.write(f"=== L{args.level} P{p} {row['name']}\n{row['error']}\n\n")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "problem", "name",
                                          "compiled", "correct", "class", "error"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nlevel {args.level}: {passed}/{len(problems)} passed "
          f"({100*passed/len(problems):.1f}%)")
    print(f"wrote {out_path}")
    if verbose:
        verbose.close()


if __name__ == "__main__":
    main()
