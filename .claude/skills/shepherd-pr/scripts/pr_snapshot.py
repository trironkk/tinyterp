#!/usr/bin/env python3
"""Emit a stable, compact snapshot of a pull request's reviewable state.

Usage: pr_snapshot.py <owner/repo> <pr-number>

Every human-meaningful update (CI checks, reviews, comments, merge state, new
commits) changes at least one line, so a plain diff against the prior snapshot
detects activity. api.github.com auth is injected by the sandbox proxy, so no
token is needed.
"""

import json
import sys
import urllib.request


def get(api, path):
    req = urllib.request.Request(
        f"{api}{path}", headers={"Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main(repo, pr):
    api = f"https://api.github.com/repos/{repo}"
    p = get(api, f"/pulls/{pr}")
    head = p["head"]["sha"]
    checks = get(api, f"/commits/{head}/check-runs").get("check_runs", [])
    reviews = get(api, f"/pulls/{pr}/reviews")

    lines = [
        f"state={p['state']}",
        f"merged={p['merged']}",
        f"mergeable_state={p.get('mergeable_state')}",
        f"head_sha={head}",
        f"comments={p['comments']}",
        f"review_comments={p['review_comments']}",
        f"commits={p['commits']}",
        "checks="
        + (
            ";".join(
                sorted(f"{c['name']}={c['status']}/{c['conclusion']}" for c in checks)
            )
            or "none"
        ),
        "reviews="
        + (";".join(f"{r['user']['login']}:{r['state']}" for r in reviews) or "none"),
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: pr_snapshot.py <owner/repo> <pr-number>")
    main(sys.argv[1], sys.argv[2])
