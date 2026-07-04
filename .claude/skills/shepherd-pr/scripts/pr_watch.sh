#!/usr/bin/env bash
# One-shot PR watcher, gated on a server-provided timestamp.
#
# Usage: pr_watch.sh <owner/repo> <pr> [SINCE]
#
# Run it detached (a background process). It polls the PR's `updated_at` quietly
# and exits on the first value strictly newer than SINCE — and a detached
# background process exiting re-invokes the session with this script's final
# line. Idle polling therefore costs the session no context; only a real wake
# does. The session pulls the details itself on wake.
#
# SINCE (ISO8601 UTC) defaults to `updated_at` read from the server at launch,
# so a watcher armed right after the session acts ignores that action — its own
# change sits at the gate. `updated_at` is GitHub's clock (no local skew) and
# bumps on comments, reviews, commits, edits, merge and close, but NOT on CI
# check-runs, so CI churn never wakes the session. ISO8601 UTC sorts lexically
# = chronologically.
set -u

REPO="$1"
PR="$2"
SINCE="${3:-}"
INTERVAL="${WATCH_INTERVAL:-30}"
API="https://api.github.com/repos/$REPO/pulls/$PR"

# Token is optional: outside a sandbox, GITHUB_TOKEN/GH_TOKEN or `gh auth token`
# authenticates; inside a sandbox the proxy injects credentials for bare curl.
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
[ -z "$TOKEN" ] && TOKEN="$(gh auth token 2>/dev/null || true)"

gh_curl() {
  if [ -n "$TOKEN" ]; then
    curl -sS -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$@"
  else
    curl -sS -H "Accept: application/vnd.github+json" "$@"
  fi
}

updated_at() {
  gh_curl "$API" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["updated_at"])' 2>/dev/null
}

# Default the gate to the server's current updated_at at launch.
while [ -z "$SINCE" ]; do SINCE=$(updated_at); [ -z "$SINCE" ] && sleep 10; done
echo "watching $REPO#$PR for activity after $SINCE"

while true; do
  sleep "$INTERVAL"
  now=$(updated_at)
  if [ -n "$now" ] && [[ "$now" > "$SINCE" ]]; then
    echo "$REPO#$PR: activity after $SINCE (updated_at=$now) — wake session, pull details."
    exit 0
  fi
done
