"""Verify each notebook has a runs/ record whose code matches the source.

A record is the markdown produced by `make record`: nbconvert embeds every
code cell verbatim, so comparing the record's fenced python blocks against
the notebook's code cells detects missing or stale records. Stdlib only, so
CI can run it without installing the project environment.

Usage: check_runs.py [notebook.py ...]   (defaults to all of notebooks/)
"""

import re
import sys
from pathlib import Path

FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def notebook_code_cells(path: Path) -> list[str]:
    cells: list[str] = []
    current: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            if current:
                cells.append("\n".join(current).strip())
            current = [] if "[markdown]" not in line else None
        elif current is not None:
            current.append(line)
    if current:
        cells.append("\n".join(current).strip())
    return [c for c in cells if c]


def record_code_cells(path: Path) -> list[str]:
    return [m.strip() for m in FENCE.findall(path.read_text())]


def latest_record(notebook: Path) -> Path | None:
    records = sorted(Path("runs").glob(f"{notebook.stem}.*.md"))
    return records[-1] if records else None


def main() -> int:
    notebooks = [Path(a) for a in sys.argv[1:]] or sorted(Path("notebooks").glob("*.py"))
    failures = 0
    for nb in notebooks:
        record = latest_record(nb)
        if record is None:
            print(f"MISSING: no runs/ record for {nb} (make record NB={nb})")
            failures += 1
        elif notebook_code_cells(nb) != record_code_cells(record):
            print(f"STALE: {record} does not match {nb} (make record NB={nb})")
            failures += 1
        else:
            print(f"ok: {nb} matches {record}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
