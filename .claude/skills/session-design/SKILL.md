---
name: session-design
description: Phase 3 of a build session (see session-arc) — stub the notebook skeleton. Invoke only after the session branch exists and scope is approved; not a standalone skill.
---

**Design.** Stub the skeleton: labeled cells (`# %% [A] Short description`) with empty bodies,
so cells are referenceable ("split [C]", "merge [D] and [E]"). Done when the user has reviewed
and approved the skeleton — no flesh before approval.

Once the skeleton is approved, invoke `session-build`. Never fill a cell here.
