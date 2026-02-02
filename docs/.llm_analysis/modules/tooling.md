# Repository Tooling & Hygiene

## Purpose
Captures repo-level developer tooling and hygiene conventions (formatting, ignore rules, and packaging surface) that affect code quality and contributor experience.

## Key Components

### Formatting (`.pre-commit-config.yaml`)
- `black` and `isort` are configured via `pre-commit`, with `isort` using `--profile=black` for compatibility.
- Hook versions are pinned, improving reproducibility across contributors and CI.

### Ignore Rules (`.gitignore`)
- Standard Python ignores: bytecode, `__pycache__/`, virtualenvs, packaging outputs, and test/tool caches.
- Includes `.DS_Store` ignore for macOS.

### Packaging (`setup.py`)
- Uses `setuptools` with console-script entrypoints for the CLI surface (`mlx_lm.*` commands).

## Observations / Opportunities
- The formatter-only pre-commit setup is intentionally low-friction; adding basic hygiene hooks (whitespace/EOF) and optional linting (e.g., Ruff) could catch more issues automatically if desired.
- Ignored artifacts can still accumulate locally (e.g., `.DS_Store`, `__pycache__/`); periodic cleanup scripts or developer docs can help keep working trees tidy.
