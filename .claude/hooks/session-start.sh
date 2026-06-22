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
# pinned commit; the grill-me / grilling skills are symlinked from there
# into .claude/skills. Idempotent.
git -C "$CLAUDE_PROJECT_DIR" submodule update --init --recursive
