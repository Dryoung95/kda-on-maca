#!/usr/bin/env bash
# Link the KDA-MACA skills into ~/.claude/skills so Claude Code discovers them.
# Equivalent to the upstream `ln -s skills/... ~/.claude/skills/...` step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="${HOME}/.claude/skills"
mkdir -p "$SKILL_DIR"

for skill in mctracer-report-skill c500-kernel-wiki; do
    src="$HERE/skills/$skill"
    dst="$SKILL_DIR/$skill"
    if [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "skip: $dst already exists"
        continue
    fi
    ln -s "$src" "$dst"
    echo "linked: $dst -> $src"
done

cat <<'NOTE'
Skills installed. In a new Claude Code session, invoke with:
  /mctracer-report-skill
  /c500-kernel-wiki
or let the model trigger them when you ask to profile a kernel or ask why a
C500 kernel is slow.

NOTE
