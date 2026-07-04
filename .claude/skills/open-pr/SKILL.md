---
name: open-pr
description: Push a branch and open its PR, then hand off to shepherd-pr. Use to ship committed work, or when a skill reaches its ship step (e.g. notebook-finalize).
---

Opening a PR is outward-facing and hard to reverse, so do it only on an explicit go-ahead — one
the invoking skill may already carry (e.g. notebook-finalize's ship go-ahead).

- Confirm the branch has a commit ahead of its base and a clean working tree; if not, stop.
- Push the branch, then open the PR with a title and body. Auth and push mechanics (ssh vs an
  HTTPS token, `gh` vs the REST API) live in `shepherd-pr/reference.md`.
- Hand off to `shepherd-pr` with the PR number to carry it from open to merge.

Done when the PR is open and handed to `shepherd-pr`.
