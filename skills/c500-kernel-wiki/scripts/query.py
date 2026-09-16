#!/usr/bin/env python3
"""Query the c500-kernel-wiki knowledge base.

Keyword search over page titles, tags, and bodies, with filters by type,
provenance, and confidence. The C500 counterpart of upstream KernelWiki's
query.py — no alias map (this wiki's vocabulary is small) and no repo filter
(there is no PR corpus), just the retrieval that a browsing agent needs.

Usage:
    python3 scripts/query.py "shared memory"
    python3 scripts/query.py --type hardware
    python3 scripts/query.py --tag mma --paths-only
    python3 scripts/query.py --provenance measured --confidence high
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import (
    SKILL_ROOT,
    WIKI_DIR,
    extract_frontmatter,
    read_body,
)

_FM_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.DOTALL)


def load_pages():
    pages = []
    for md in sorted(WIKI_DIR.rglob("*.md")):
        fm, err = extract_frontmatter(md)
        if err or not isinstance(fm, dict):
            continue
        pages.append(
            {
                "path": md.relative_to(SKILL_ROOT).as_posix(),
                "fm": fm,
                "body": read_body(md),
            }
        )
    return pages


def match(page, keywords, args):
    fm = page["fm"]

    if args.type and fm.get("type") != args.type:
        return False
    if args.provenance and fm.get("provenance") != args.provenance:
        return False
    if args.confidence and fm.get("confidence") != args.confidence:
        return False
    if args.tag:
        tags = fm.get("tags")
        tags = tags if isinstance(tags, list) else ([tags] if tags else [])
        if args.tag not in [str(t).lower() for t in tags]:
            return False

    if not keywords:
        # Filters alone are a valid query ("list all hardware pages").
        return True

    haystack_parts = [str(fm.get("title", "")), " ".join(map(str, fm.get("tags", []) or []))]
    if args.search_body:
        haystack_parts.append(page["body"])
    haystack = "\n".join(haystack_parts).lower()

    return all(kw.lower() in haystack for kw in keywords)


def main():
    ap = argparse.ArgumentParser(description="Query the C500 kernel wiki")
    ap.add_argument("query", nargs="*", help="free-text keywords (all must match)")
    ap.add_argument("--type", help="hardware | technique | migration | pattern")
    ap.add_argument("--tag", help="filter by a frontmatter tag")
    ap.add_argument("--provenance", help="measured | header | derived")
    ap.add_argument("--confidence", help="high | medium | low")
    ap.add_argument("--search-body", action="store_true", help="also search page bodies (slower)")
    ap.add_argument("--limit", type=int, default=10, help="max results (default 10)")
    ap.add_argument("--compact", action="store_true", help="one line per result")
    ap.add_argument("--paths-only", action="store_true", help="just paths, for scripting")
    args = ap.parse_args()

    pages = load_pages()
    hits = [p for p in pages if match(p, args.query, args)]

    if not hits:
        print("No pages matched.", file=sys.stderr)
        sys.exit(1)

    hits = hits[: args.limit]

    if args.paths_only:
        for p in hits:
            print(p["path"])
        return

    for p in hits:
        fm = p["fm"]
        tags = ", ".join(map(str, fm.get("tags", []) or []))
        if args.compact:
            print(f"{p['path']}  —  {fm.get('title', '?')}  [{fm.get('type')}/{fm.get('provenance')}/{fm.get('confidence')}]")
        else:
            print(f"  {fm.get('title', 'Untitled')}")
            print(f"    path:        {p['path']}")
            print(f"    type:        {fm.get('type')}")
            print(f"    provenance:  {fm.get('provenance')}"
                  + (f"  (measured {fm.get('measured')})" if fm.get("measured") else ""))
            print(f"    confidence:  {fm.get('confidence')}")
            if tags:
                print(f"    tags:        {tags}")
            if fm.get("probe"):
                print(f"    probe:       {fm['probe']}")
            print()


if __name__ == "__main__":
    main()
