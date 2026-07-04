# Ergonomics for the notebook workflow. Recipes in README.md show the
# underlying commands; these targets are the day-to-day entry points.

.PHONY: record check hooks

# Record an executed notebook run into runs/ (usage: make record NB=notebooks/<file>.py)
record:
	uv run jupytext --to ipynb --output - $(NB) | \
	  uv run jupyter nbconvert --stdin --execute --to markdown \
	    --output-dir runs --output "$(notdir $(basename $(NB))).$$(date +%F)"
	uv run --only-group lint mdformat runs/

# Verify run records match notebook sources, and markdown is formatted
check:
	python3 scripts/check_runs.py
	git ls-files '*.md' | xargs uv run --only-group lint mdformat --check

# Install the pre-commit hook that blocks committing a notebook without a fresh record
hooks:
	git config core.hooksPath .githooks
