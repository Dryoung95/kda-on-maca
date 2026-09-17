#!/usr/bin/env python3
"""Rerun the L1 correctness gate with inputs resized to fit 16.3 GB.

The stock KernelBench L1 shapes are sized for 48 GB boards. On the C500's
15.22 GiB that turns 39 of 100 problems into allocator failures, and the gate
reports those as compiled=True, correctness=False -- indistinguishable from a
real numeric bug. This script asks the question those failures were silently
answering: does the kernel itself compute the right answer when it fits?

Strategy: patch each problem's get_inputs/get_init_inputs at import time to
scale shapes down by a factor chosen from the *input* byte cost, then run the
same gate. Structure is preserved -- channels stay divisible by num_groups,
predictions and targets keep a shared batch size -- because a shape that
violates an op's constraints fails for the wrong reason.

Usage:
    source /data/kda-maca/env.sh
    source /data/cuda-harness-migration/env.sh
    python3 scripts/run_level_resized.py --budget-gib 4 --out results/level1_resized.csv
"""

import argparse
import csv
import gc
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

GATE = Path("/data/kda-maca/scripts/correctness_gate.py")
DATASET = Path("/data/cuda-harness-migration/KernelBench/KernelBench")
GPU_BUDGET_BYTES_DEFAULT = 4 * 2**30

# torch.backends.cudnn.allow_tf32 defaults to True on C500 and its ~1e-3 error
# breaks the gate's 1e-4 fp32 tolerance (Conv1d: 9.5e-4 on vs exactly 0.0 off).
# Measuring real fp32 agreement is the point, so it goes off by default.
TF32_OFF_DEFAULT = True


def patched_source(src: str, factor: float) -> str:
    """Scale the module-level shape constants by `factor`.

    Every KernelBench L1 file declares plain module-level ints (batch_size,
    dim, N, ...) and consumes them inside get_inputs(). Scaling the constants
    keeps get_inputs()' own construction logic untouched, so relative shapes
    and dtypes survive.

    Shape constants may be plain ints or tuples (`input_shape = (32768,)`).
    Both are scaled, tuple members individually.

    Structural constraints are handled here, not by the caller: some constants
    are not sizes. Channels/groups/kernel sizes are structural parameters --
    scaling them changes what the op is, and num_groups must stay a divisor of
    the channel count. Leave them alone and shrink only the batch and spatial
    dims, which carry the byte cost.
    """
    # Names that must not be scaled. KernelBench L1 uses these for channel
    # counts, group counts, and kernel extents -- all structural, none of them
    # the thing that makes a problem too big to run.
    STRUCTURAL = {
        "features", "num_features", "num_groups", "channels", "in_channels",
        "out_channels", "num_classes", "num_heads", "kernel_size",
        "depth", "embedding_dimension", "num_layers", "embedding_dim",
        "hidden_dim", "sequence_length",
    }
    # `dim` is ambiguous: in the activation problems it is the tensor width
    # (huge, must scale), in the reduction problems it is the axis index
    # (tiny, must not). Decide by magnitude -- a reduction axis is never
    # larger than the rank, so anything big is a size.
    for name, val in re.findall(r"^(\w+)\s*=\s*(\d+)\s*(?:#.*)?$", src, re.M):
        if name == "dim" and int(val, 0) <= 8:
            STRUCTURAL.add("dim")
    out = src
    # int constants: name = N  (optional trailing comment, which the earlier
    # version's `(\d+)$` anchor choked on, silently leaving batch_size unscaled)
    for name, val in re.findall(r"^(\w+)\s*=\s*(\d+)\s*(?:#.*)?$", src, re.M):
        if name in STRUCTURAL:
            continue
        new = max(2, int(round(int(val, 0) * factor)))
        out = re.sub(rf"^(\s*{re.escape(name)}\s*=\s*){re.escape(val)}(\s*(?:#.*)?)$",
                     rf"\g<1>{new}\g<2>", out, count=1, flags=re.M)
    # expression constants: `M = 16384 * 4`. Scale the *first* factor and keep
    # the multiplier, so the shape stays a product of the same structure.
    for name, a, b in re.findall(r"^(\w+)\s*=\s*(\d+)\s*\*\s*(\d+)\s*(?:#.*)?$", src, re.M):
        if name in STRUCTURAL:
            continue
        new = max(2, int(round(int(a) * factor)))
        out = re.sub(rf"^(\s*{re.escape(name)}\s*=\s*){re.escape(a)}(\s*\*\s*{re.escape(b)}\s*(?:#.*)?)$",
                     rf"\g<1>{new}\g<2>", out, count=1, flags=re.M)
    # tuple constants: name = (a, b, ...) -- members are ints or int constants
    for name, body in re.findall(r"^(\w+)\s*=\s*\(([^()]*)\)\s*(?:#.*)?$", src, re.M):
        members = [t.strip() for t in body.split(",")]
        parts = []
        for t in members:
            m = re.fullmatch(r"(\d+)", t)
            if m:
                parts.append((t, str(max(2, int(round(int(m.group(1)) * factor))))))
            else:
                # reference to another constant -- leave it; the referenced
                # constant is scaled by its own match above
                parts.append((t, t))
        new_body = ", ".join(p[1] for p in parts)
        out = re.sub(rf"^(\s*{re.escape(name)}\s*=\s*\(){re.escape(body)}(\)\s*(?:#.*)?)$",
                     rf"\g<1>{new_body}\g<2>", out, count=1, flags=re.M)
    return out


def input_bytes(src: str) -> int:
    """Estimate input byte cost by resolving the shape constants.

    Static rather than executed: get_inputs() on the big problems allocates
    8-16 GiB, and calling it just to learn the shape defeats the purpose.

    Shape constants can be indirect -- `torch.rand(batch_size, *input_shape)`
    where input_shape is itself a tuple. Follow one level of indirection and
    treat the tuple's members as the real dims.
    """
    consts = {}
    for name, val in re.findall(r"^(\w+)\s*=\s*(.+)$", src, re.M):
        # strip trailing comments -- `batch_size = 112  # scaled up` is common
        val = val.split("#")[0].strip()
        try:
            consts[name] = int(val, 0)
        except ValueError:
            # expression constants: `M = 16384 * 4`. Evaluate the product so
            # the matmul problems' real byte cost is what sets the factor.
            m = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", val)
            if m:
                consts[name] = int(m.group(1)) * int(m.group(2))
            continue
    # names are case-sensitive in Python but KernelBench is inconsistent
    # (M/K/N in some files, m/k/n in others); index both spellings
    lower = {k.lower(): v for k, v in consts.items()}
    tuples = {}
    pending = []
    for line in src.splitlines():
        m = re.fullmatch(r"\s*(\w+)\s*=\s*\(([^()]*)\)\s*(?:#.*)?", line)
        if m:
            pending.append((m.group(1), m.group(2)))
    # a tuple may reference a constant declared later in the file, so resolve
    # in passes until the references close
    unresolved = list(pending)
    for _ in range(3):
        still = []
        for name, val in unresolved:
            parts = [t.strip() for t in val.split(",") if t.strip()]
            if all(p in consts for p in parts):
                tuples[name] = [consts[p] for p in parts]
            elif all(p in consts or p.isdigit() for p in parts):
                tuples[name] = [consts.get(p, int(p)) for p in parts]
            else:
                still.append((name, val))
        unresolved = still

    def resolve(tok):
        # `*input_shape` unpacks a tuple into dims; strip any number of stars
        while tok.strip().startswith("*"):
            tok = tok.strip()[1:].strip()
        tok = tok.strip()
        if tok in consts:
            return [consts[tok]]
        if tok.lower() in lower and tok.lower() not in ("dim",):
            return [lower[tok.lower()]]
        if tok in tuples:
            return list(tuples[tok])
        if tok.isdigit():
            return [int(tok)]
        return None

    gi = src.split("def get_inputs")[1].split("def get_init_inputs")[0] if "def get_inputs" in src else ""
    total = 0
    # Every input tensor in these problems comes from one call in get_inputs.
    # Extract each call, then its shape: either an inner tuple literal, a
    # bare-arg list, or `x.shape` (inheriting the previous tensor, which the
    # loop tracks so a derived tensor still counts).
    prev_dims = None
    for call in re.findall(r"torch\.(?:rand|randn|randint)\([^)]*\)", gi):
        args = call.split("(", 1)[1].rsplit(")", 1)[0]
        dims = []
        ok = True
        inner = re.findall(r"\(([^()]*)\)", args)
        toks = inner[0].split(",") if inner else args.split(",")
        for t in toks:
            t = t.strip()
            if not t:
                continue
            # `randint(0, 2, x.shape)` puts range bounds where shape args go;
            # only names and tuple refs are dims, never bare small literals
            if t.lstrip("-").isdigit():
                continue
            if ".shape" in t:
                if prev_dims is not None:
                    dims = list(prev_dims)
                else:
                    ok = False
                break
            r = resolve(t)
            if r is None:
                ok = False
                break
            dims.extend(r)
        if ok and dims:
            prev_dims = dims
        elif "x.shape" in args or ".shape" in args:
            # derived tensor: same cost as the one it was built from
            if prev_dims is not None:
                dims = list(prev_dims)
                ok = True
        if not ok or not dims:
            continue
        n = 1
        for d in dims:
            n *= d
        total += n * 4
    return total


def forward_cost(src: str) -> int:
    """Rough first guess at the byte cost of one forward pass.

    Some problems have tiny inputs and enormous outputs: P7 and P9 take
    16 MB and return a 16384x16384 matrix (1 GiB); P63 takes 1 GiB in and
    emits an 8 GiB feature map; P95's output is a scalar. The ratio between
    input and output cost spans 0x to 512x across this level, so no single
    multiplier can predict it. This returns the input cost as a starting
    point only -- the caller retries with progressively smaller factors when
    the attempt OOMs, which is what actually handles the spread.
    """
    return input_bytes(src)


def _attempt(level, problem, cand_path, timeout):
    """Run the gate once on a prepared candidate. Returns (row, oom)."""
    try:
        r = subprocess.run(
            ["python3", str(GATE), "--level", str(level),
             "--problem", str(problem), "--candidate", str(cand_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, False
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    blob = out + "\n" + err
    oom = "OutOfMemoryError" in blob or "out of memory" in blob.lower()

    def field(name):
        for line in out.splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    compiled = field("compiled").lower() == "true"
    correct = field("correct").lower() == "true"
    if not compiled:
        cls = "compile_fail"
    elif correct:
        cls = "pass"
    elif oom:
        cls = "oom"
    else:
        cls = "numeric"
    reason = ""
    if not compiled or not correct:
        lines = [l for l in out.splitlines() if "compiled:" in l or "correct:" in l]
        reason = "; ".join(lines) if lines else (err.splitlines() or [""])[0][:200]
    return {"compiled": compiled, "correct": correct, "class": cls, "error": reason}, oom


def run_one(level: int, problem: int, budget: int, tf32_off: bool, timeout: int = 600):
    lvl = DATASET / f"level{level}"
    matches = sorted(lvl.glob(f"{problem}_*.py"))
    if not matches:
        return {"level": level, "problem": problem, "name": "<missing>",
                "compiled": False, "correct": False, "class": "error",
                "factor": "", "error": "no matching file"}
    ref_file = matches[0]
    raw = ref_file.read_text()
    cost = forward_cost(raw)
    # Peak GPU use is a multiple of the raw cost -- the gate holds the
    # reference and candidate outputs plus their difference. 4x is the
    # observed worst case for conv, not a derived bound.
    factor = min(1.0, budget / (cost * 4)) if cost else 1.0
    # Output can dwarf the input by 512x (P7: 16 MB in, 1 GiB out), and no
    # static estimate covers that spread. Retry with smaller factors on OOM,
    # which handles it without a shape inference.
    factors = [factor]
    if factor >= 1.0:
        factors += [0.5, 0.25, 0.125, 0.05]
    else:
        factors += [factor * 0.5, factor * 0.25, factor * 0.125]

    last = None
    for f in factors:
        scaled = patched_source(raw, f)
        scaled = re.sub(r"\bclass\s+Model\b", "class ModelNew", scaled, count=1)
        scaled = re.sub(r"\bsuper\(\s*Model\s*,", "super(ModelNew,", scaled)
        if tf32_off:
            scaled = ("import torch as _t\n"
                      "_t.backends.cudnn.allow_tf32 = False\n"
                      "_t.backends.cuda.matmul.allow_tf32 = False\n" + scaled)
        with tempfile.TemporaryDirectory() as td:
            cand = Path(td) / f"cand_{problem}.py"
            cand.write_text(scaled)
            result, oom = _attempt(level, problem, cand, timeout)
        if result is None:
            last = {"compiled": False, "correct": False, "class": "timeout",
                    "error": "timeout"}
            continue
        last = result
        if result["class"] != "oom":
            break
        # a compile error or a numeric mismatch would not be fixed by
        # shrinking further; only OOM is worth another attempt
    if last is None:
        last = {"compiled": False, "correct": False, "class": "timeout",
                "error": "timeout"}
        f = factors[-1]
    return {"level": level, "problem": problem, "name": ref_file.stem,
            "compiled": last["compiled"], "correct": last["correct"],
            "class": last["class"], "factor": f"{f:.3f}",
            "error": last["error"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--budget-gib", type=float, default=4.0,
                    help="target total input bytes per problem (default 4)")
    ap.add_argument("--tf32-off", type=int, default=int(TF32_OFF_DEFAULT),
                    help="1 (default) disables TF32 to measure real fp32")
    ap.add_argument("--only", default="",
                    help="comma-separated problem numbers, else the whole level")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    budget = int(args.budget_gib * 2**30)
    out_path = Path(args.out or f"results/level{args.level}_resized.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lvl = DATASET / f"level{args.level}"
    problems = sorted({int(p.name.split("_")[0]) for p in lvl.glob("*.py")})
    if args.only:
        want = {int(x) for x in args.only.split(",") if x.strip()}
        problems = [p for p in problems if p in want]
    if not problems:
        sys.exit(f"no problems under {lvl}")

    rows = []
    passed = 0
    for i, p in enumerate(problems, 1):
        row = run_one(args.level, p, budget, bool(args.tf32_off))
        rows.append(row)
        if row["compiled"] and row["correct"]:
            passed += 1
        print(f"[{i:>3}/{len(problems)}] L{args.level} P{p:<3} "
              f"{row['class']:<12} f={row['factor']}  {row['name']}"
              + (f"   ({row['error'][:70]})" if row["class"] != "pass" else ""),
              flush=True)
        gc.collect()

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "problem", "name",
                                          "compiled", "correct", "class",
                                          "factor", "error"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    tally = Counter(r["class"] for r in rows)
    print(f"\nlevel {args.level} (resized to {args.budget_gib} GiB budget, "
          f"tf32_off={bool(args.tf32_off)}): {passed}/{len(problems)} passed")
    for k in ("pass", "oom", "numeric", "compile_fail", "timeout"):
        if tally.get(k):
            print(f"  {k}: {tally[k]}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
