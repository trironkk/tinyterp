#!/bin/bash
set -euo pipefail

# Only needed in the ephemeral remote environment (Claude Code on the web),
# where a fresh clone leaves git submodules uninitialized. Local clones are
# assumed to manage submodules themselves.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Initialize/refresh git submodules to the commits pinned by this repo.
# This populates vendor/mattpocock-skills (Matt Pocock's skills) at the
# pinned commit; the grilling skill is symlinked from there
# into .claude/skills. Idempotent.
git -C "$CLAUDE_PROJECT_DIR" submodule update --init --recursive

# Register the nbstripout git clean filter so notebook OUTPUTS are stripped
# on every commit (the repo never source-controls generated cell outputs).
# .gitattributes routes *.ipynb through filter=nbstripout; here we point that
# filter at the tool via uvx so no full project sync is required. Idempotent.
git -C "$CLAUDE_PROJECT_DIR" config filter.nbstripout.clean "uvx nbstripout"
git -C "$CLAUDE_PROJECT_DIR" config filter.nbstripout.smudge cat
git -C "$CLAUDE_PROJECT_DIR" config diff.ipynb.textconv "uvx nbstripout -t"
