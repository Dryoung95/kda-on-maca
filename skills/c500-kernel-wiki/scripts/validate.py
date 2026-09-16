#!/usr/bin/env python3
"""Validate the c500-kernel-wiki knowledge base.

Enforces the contract the wiki claims in references/schema.md: every page
carries provenance frontmatter, every measured page names a probe, and every
cited probe actually exists and compiles.

This is the C500 counterpart of upstream KernelWiki's scripts/validate.py,
scaled down to what this knowledge base actually contains — no PR corpus, no
artifact bundles, no version-claim registry, because none of those data
sources exist for this board. The provenance contract is the part that
 transfers, and it is the part that keeps the wiki honest.

Usage:
    python3 scripts/validate.py [--no-compile]
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = SKILL_ROOT / "wiki"
REFERENCES_DIR = SKILL_ROOT / "references"
PROBES_DIR = SKILL_ROOT / "probes"
QUERIES_DIR = SKILL_ROOT / "queries"

REQUIRED_FIELDS = ("title", "tags", "type", "provenance", "architecture", "confidence")
VALID_PROVENANCE = ("measured", "header", "derived")
VALID_CONFIDENCE = ("high", "medium", "low")
VALID_TYPE = ("hardware", "technique", "migration", "pattern")

_FM_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)
_PROBE_REF_RE = re.compile(r"probes/([A-Za-z0-9_.]+\.cu)")
_MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)(?:[^)]*)\)")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def extract_frontmatter(path):
    """Parse the YAML frontmatter block. Our frontmatter is flat key/value
    pairs, so a minimal parser is enough — no nested structures, no flow
    sequences beyond inline `[a, b]` lists, which we do handle."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"read error: {e}"
    m = _FM_RE.match(content)
    if not m:
        return None, None
    fm = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # key: value   (value may be an inline list or quoted)
        if ":" not in line:
            return None, f"malformed line: {line!r}"
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1]
            fm[key] = [
                p.strip().strip("\"'") for p in inner.split(",") if p.strip()
            ]
        elif raw:
            fm[key] = raw.strip("\"'")
        else:
            fm[key] = None
    return fm, None


def read_body(path):
    content = path.read_text(encoding="utf-8")
    m = _FM_RE.match(content)
    return content[m.end():] if m else content


def parse_date(value):
    if not isinstance(value, str):
        return None
    if not _DATE_RE.match(value):
        return None
    try:
        datetime.date.fromisoformat(value)
        return value
    except ValueError:
        return None


def probe_compiles(probe_path):
    """Compile one probe with cucc. Returns (ok, message). MACA_CUCC_FLAGS must
    be present in the environment; without it the include path for
    __macro_mxcc.h is missing and every build fails for the wrong reason."""
    flags = os.environ.get("MACA_CUCC_FLAGS")
    cucc = os.environ.get("MCTRACER")  # not the compiler; see below
    # Prefer a PATH lookup — env.sh puts cucc on PATH.
    from shutil import which

    cucc_bin = which("cucc")
    if cucc_bin is None:
        return None, "cucc not on PATH (source env.sh first)"
    if not flags:
        return None, "MACA_CUCC_FLAGS unset (source env.sh first)"
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "probe.out"
        try:
            r = subprocess.run(
                [cucc_bin, str(probe_path), "-O2", "-std=c++17"] + flags.split() + ["-o", str(out)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return False, "compile timeout"
        if r.returncode != 0:
            detail = (r.stderr or "").strip().splitlines()
            snippet = detail[-3:] if detail else ["no stderr"]
            return False, "; ".join(snippet)
        return True, None


def validate_page(path, errors, warnings, compile_probes, compiled_cache):
    rel = path.relative_to(SKILL_ROOT)
    is_reference = path.parent == REFERENCES_DIR
    fm, err = extract_frontmatter(path)

    if fm is None and err is None:
        if is_reference:
            # reference docs only get link/probe-ref checks
            fm = {}
        else:
            errors.append(f"{rel}: missing frontmatter (schema requires title/tags/type/provenance/architecture/confidence)")
            return
    if err:
        errors.append(f"{rel}: frontmatter parse error: {err}")
        return
    if not isinstance(fm, dict):
        errors.append(f"{rel}: frontmatter must be a mapping")
        return

    if not is_reference:
        for field in REQUIRED_FIELDS:
            if field not in fm or fm[field] is None:
                errors.append(f"{rel}: missing required field '{field}'")

    provenance = fm.get("provenance")
    if provenance not in VALID_PROVENANCE and not is_reference:
        errors.append(
            f"{rel}: provenance {provenance!r} not in {VALID_PROVENANCE}"
        )

    confidence = fm.get("confidence")
    if confidence not in VALID_CONFIDENCE and not is_reference:
        errors.append(f"{rel}: confidence {confidence!r} not in {VALID_CONFIDENCE}")

    ptype = fm.get("type")
    if ptype not in VALID_TYPE and not is_reference:
        errors.append(f"{rel}: type {ptype!r} not in {VALID_TYPE}")

    arch = fm.get("architecture")
    if isinstance(arch, list) and "c500" not in arch and not is_reference:
        errors.append(f"{rel}: architecture {arch} does not include 'c500'")

    # Provenance-specific contract.
    if provenance == "measured":
        measured = fm.get("measured")
        if not measured:
            errors.append(f"{rel}: provenance 'measured' requires a 'measured:' date")
        elif not parse_date(measured):
            errors.append(f"{rel}: 'measured:' must be YYYY-MM-DD, got {measured!r}")
        probe = fm.get("probe")
        if not probe:
            errors.append(f"{rel}: provenance 'measured' requires a 'probe:' path")
        elif not str(probe).startswith("probes/") and not str(probe).startswith("torch"):
            errors.append(
                f"{rel}: probe {probe!r} must be a probes/ path (or a torch.benchmark note)"
            )
    elif provenance == "header":
        # Header facts are authoritative for this toolchain version: a
        # medium/low confidence header claim is a contradiction.
        if confidence is not None and confidence != "high":
            errors.append(
                f"{rel}: provenance 'header' implies confidence 'high' "
                f"(header content is authoritative for this SDK), got {confidence!r}"
            )

    # Probe resolution + optional compile check.
    probe = fm.get("probe")
    if isinstance(probe, str) and probe.startswith("probes/"):
        probe_path = SKILL_ROOT / probe
        if not probe_path.is_file():
            errors.append(f"{rel}: probe '{probe}' does not exist")
        elif compile_probes:
            key = str(probe_path)
            if key not in compiled_cache:
                compiled_cache[key] = probe_compiles(probe_path)
            ok, msg = compiled_cache[key]
            if ok is False:
                errors.append(f"{rel}: probe '{probe}' does not compile: {msg}")
            elif ok is None:
                warnings.append(f"{rel}: probe compile check skipped ({msg})")

    # Probe references in the body must resolve.
    body = read_body(path)
    for name in _PROBE_REF_RE.findall(body):
        if not (PROBES_DIR / name).is_file():
            errors.append(f"{rel}: body references probes/{name}, which does not exist")

    # Internal markdown links must resolve.
    base = path.parent
    for m in _MD_LINK_RE.finditer(body):
        target = m.group(1)
        if target.startswith(("http", "/")):
            continue
        resolved = (base / target).resolve()
        # Allow escaping the skill dir only for the sibling mctracer skill.
        if not resolved.is_file():
            errors.append(f"{rel}: link '{target}' does not resolve")


def collect_pages():
    pages = []
    if WIKI_DIR.is_dir():
        pages.extend(sorted(WIKI_DIR.rglob("*.md")))
    if REFERENCES_DIR.is_dir():
        # primer.md and schema.md are prose companions, not knowledge pages.
        # Validate their links and probe references (those can still rot), but
        # do not require the provenance contract from them.
        for name in ("primer.md", "schema.md"):
            p = REFERENCES_DIR / name
            if p.is_file():
                pages.append(p)
    return pages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="skip the probe compile check (use when the MACA toolchain is unavailable)",
    )
    args = ap.parse_args()

    compile_probes = not args.no_compile
    errors = []
    warnings = []
    compiled_cache = {}

    pages = collect_pages()
    for path in pages:
        validate_page(path, errors, warnings, compile_probes, compiled_cache)

    # queries/ is generated output; it must not be validated as a knowledge page.
    n_probes = len(list(PROBES_DIR.glob("*.cu"))) if PROBES_DIR.is_dir() else 0

    print(f"Validated {len(pages)} pages, {n_probes} probes in skill tree")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    if errors:
        print(f"\n{len(errors)} errors found:\n")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    print("All pages valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
