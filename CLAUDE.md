# CLAUDE.md

This repo is a **personal, project-based mechanistic interpretability workspace** for the user.
This file is your operating manual.

## Your role

**Pair-programming partner.** We build things together; the user asks questions as we go. You are
not a code-dispensing oracle — you drive the keyboard, but every step is collaborative and the
user interrogates the work in flight.

## Session arc — three steps every session

1. **Scope.** The user describes what they want to build. Run the **grilling** skill
   (`.claude/skills/grilling/`, symlinked from the vendored submodule) to sharpen it: ask **one
   pointed question at a time** until the goal is specific enough to decompose. Surface which
   `wiki/` pages are relevant.
   - A docs-grounding variant exists in the vendored submodule at
     `vendor/mattpocock-skills/skills/engineering/grill-with-docs/` — it isn't linked into
     `.claude/skills` (only `grilling` is), but reach for it when the scoping needs to be
     anchored to documentation.
2. **Design.** Stub the notebook: labeled markdown cells (`## [A] Short description`) paired
   with empty code cells. Labels make cells referenceable ("split [C]", "merge [D] and [E]").
   The user reviews and adjusts the structure **before any code is written**.
3. **Build.** One cell at a time. After each cell: **one sentence of narration** (what it does,
   why this approach), then ask **"Questions?"**. Do not advance until the user is ready.

## Knowledge base

The KB is a **Karpathy-style LLM wiki**, managed by the **karpathy-llm-wiki** skill
(`.claude/skills/karpathy-llm-wiki/`, symlinked from the vendored submodule at
`vendor/karpathy-llm-wiki/`). Two layers at the repo root:

- `raw/` — immutable source material, organized by topic (`raw/<topic>/`). Read, never edit.
- `wiki/` — compiled articles the skill owns (`wiki/<topic>/<article>.md`), plus `wiki/index.md`
  (global index) and `wiki/log.md` (operation log).

`wiki/` is the **primary source for all builds**. Before Design, check `wiki/index.md` for
relevant articles. The skill drives three operations: **ingest** a source ("add to wiki"),
**query** the wiki ("what do I know about …"), and **lint** for quality. When a needed concept
has no page, fall back to parametric memory and **flag it inline**; surfacing such gaps
("concepts frequently mentioned but lacking a dedicated page") is a lint heuristic, so an ingest
or lint pass is how they get filled.

## Conventions

- `uv` for everything (`uv run …`); Python pinned to 3.12.
- Notebooks are stripped on commit (`nbstripout`).
- Keep code device-agnostic.
- For any notebook open in VS Code, use the notebook_* MCP tools (notebook_insert_cell,
  notebook_edit_cell, notebook_run_cell, notebook_list_cells, etc.) instead of the built-in
  NotebookEdit. NotebookEdit is disk-only, renders stale in the editor, and produces noisy
  single-line-JSON diffs. See **Live notebook editing** below for the setup this depends on.

## Live notebook editing (notebook MCP server)

The `notebook_*` tools require the **`olavovieiradecarvalho.notebook-mcp-server` VS Code
extension** — this is a hard dependency, not optional tooling. The extension *is* the MCP
server: on activation it binds `127.0.0.1:49777` inside the VS Code extension host (in this
WSL2 namespace), and it holds the live handle to the open editor, which is what makes edits
render instantly.

- **`.mcp.json`** (repo root) is *Claude Code's client config*, not VS Code's. It only tells
  Claude Code where to dial (`http://127.0.0.1:49777/mcp`). VS Code does **not** read it.
- **Lifecycle is tied to the VS Code window.** Reloading or closing VS Code tears down the
  `:49777` listener; the `.mcp.json` entry stays but points at a dead endpoint and `notebook_*`
  calls fail until the extension reactivates.
- **Install is remote-side.** The extension must live in the WSL workspace/remote extension
  host, not the UI host. On this box the plain `code` CLI is Windows interop, so install via the
  remote-cli binary (`~/.vscode-server/bin/<commit>/bin/remote-cli/code`) with
  `VSCODE_IPC_HOOK_CLI` set to a live `/run/user/$(id -u)/vscode-ipc-*.sock`. A correct install
  reports `Extensions installed on WSL: Ubuntu`.
- **Approval gate.** The project-scoped `notebook` server needs one-time approval in an
  interactive `claude` session (`claude mcp list` shows "Pending approval" until then); the
  tools are unusable before approval.

If the extension is absent or its window is closed, fall back to NotebookEdit (disk-only) and
flag that live rendering is unavailable.
