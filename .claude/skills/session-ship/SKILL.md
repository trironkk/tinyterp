---
name: session-ship
description: Phase 6 of a build session (see session-arc) — outward-facing, hard to reverse. Push the branch, open the PR, hand off to shepherd-pr. Invoke ONLY after the user explicitly approves shipping a finalized, committed session branch; never from a bare "continue".
---

**Ship (outward-facing, hard to reverse).** Push the session branch, open the PR, and hand off
to the shepherd-pr skill. These actions leave the machine and are hard to reverse, so reaching
this skill at all requires the user's explicit ship approval — never a bare "continue". Done
when the PR is open and shepherd-pr is armed.
