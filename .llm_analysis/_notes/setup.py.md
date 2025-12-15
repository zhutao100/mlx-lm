# setup.py

## Purpose
- Build configuration and package metadata.
- Defines CLI entry points (consoles scripts) for `mlx_lm.*` commands.

## Key Observations
- Dependencies: `mlx`, `numpy`, `transformers`, `sentencepiece`, etc.
- Entry points map to modules like `mlx_lm.generate:main`.
- Standard `setuptools` setup.
