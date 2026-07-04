---
name: open-pr
description: Push a branch and open its PR, then hand off to shepherd-pr. Use to ship committed work, or when a skill reaches its ship step (e.g. notebook-finalize).
---

Opening a PR is outward-facing and hard to reverse, so do it only on an explicit go-ahead.

- Push the branch: `git push -u origin <branch>`.
- Open the PR: `gh pr create` with a title and body.
- Hand off to `shepherd-pr` to carry it from open to merge.

Done when the PR is open and shepherd-pr is armed.
