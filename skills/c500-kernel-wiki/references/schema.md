# C500 Kernel Wiki — Schema & Conventions

How knowledge pages in this wiki are structured and maintained. Modeled on the
upstream KDA KernelWiki schema, with one substantive change: **every page carries
provenance**, because on this platform the toolchain ships no authoritative
performance documentation, so a fact is only as good as how it was established.

## Page frontmatter

Every page starts with a YAML block:

```yaml
---
title: Human-readable title
tags: [warp64, shared-memory, barriers]   # lowercase kebab
architecture: [c500]                       # xcore1000; future boards add their own
type: hardware | technique | migration | pattern
provenance: measured | header | derived    # how the fact was established
measured: 2026-09-15                        # date, only when provenance=measured
probe: probes/probe_smem_mixed.cu           # reproducer, only when provenance=measured
confidence: high | medium | low
---
```

### provenance values

| Value | Meaning | Trust level |
|---|---|---|
| `measured` | Microbenchmark on this board | High, but re-verify after SDK upgrades |
| `header` | Read from a toolchain header or fatbinary | Authoritative for this toolchain version |
| `derived` | Inferred from measured + header facts | Medium; the inference can be wrong even if inputs are right |

`confidence` is the author's judgment and may be lower than provenance implies —
for example a measurement made at one block size and generalized.

## Why provenance is load-bearing here

Upstream KernelWiki cites upstream PRs: a vLLM commit, a CUTLASS PR, a blog post.
Those are durable, reviewable sources. On C500 there is no equivalent corpus —
the SDK samples contain zero warp-64 or bank-conflict content, and the online
documentation is not reliably reachable. So this wiki's sources are probes run on
this board. A page without provenance would be indistinguishable from a guess.

Practical rule: **if a page's fact would change between SDK versions, its
frontmatter must say which SDK version and probe established it.**

## Directory layout

```
c500-kernel-wiki/
├── SKILL.md                     entry point — triggers and query paths
├── references/
│   ├── schema.md                this file
│   └── primer.md                topic map: symptom → page
├── probes/                      the microbenchmarks behind `measured` pages
└── wiki/
    ├── hardware/                facts about the silicon and toolchain
    ├── techniques/              how to optimize for a specific mechanism
    ├── migration/               CUDA→C500 porting knowledge
    └── patterns/                symptom → diagnosis → fix (the playbook layer)
```

## Writing a page

A knowledge page answers one question and links to its neighbors. Structure:

1. **The claim** in the first paragraph — a reader who stops there has the answer.
2. **The evidence** — the measured numbers, the header lines, whatever grounds it.
3. **What to do** — the actionable part, with code.
4. **What not to do** — the misconception the page exists to correct.
5. **Links** to related pages and to the probe that produced it.

Pages that only describe ("C500 has 104 SMs") belong in a table on the hardware
index. Pages that correct an assumption ("stride-32 slowdown is not a bank
conflict") are the valuable kind and get full treatment.

## Quality bar

- **No unsourced numbers.** Every figure is either in a probe's output or in a
  header the compiler ships.
- **State the negative findings.** "cooperative_groups grid sync does not work"
  is more useful than another page about something that does.
- **Name the misconception.** If a fact contradicts what a CUDA-trained author
  would assume, say what the assumption is and how it fails. Silent failures get
  extra emphasis because they produce wrong answers, not errors.
- **Date the measurements.** `measured:` is mandatory for `provenance: measured`.
- **Link the probe.** A `probe:` field that cannot be run is a broken link.

## Maintenance

These facts are versioned against MACA SDK 3.3.0.15. When the SDK is upgraded,
re-run the probes in `probes/` and update any page whose numbers moved. The
`measured:` date on each page tells you which are at risk of staleness; pages
marked `header` need re-checking only if the referenced header changed.

The toolchain enforces this rather than trusting it:

```bash
python3 scripts/validate.py            # provenance contract + probe compile
python3 scripts/generate-indices.py    # rebuild queries/, incl. the probe reverse index
```

`queries/by-probe.md` maps each probe to every page that cites its numbers —
that is the re-verification worklist after an SDK upgrade.

Do not promote a page from `confidence: low` to `high` because it reads well.
Promote it when a second probe or a real kernel result corroborates it.
