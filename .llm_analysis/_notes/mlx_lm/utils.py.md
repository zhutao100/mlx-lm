# mlx_lm/utils.py

## Purpose
- Core utilities for model loading, saving, and quantization.
- Handles interactions with Hugging Face Hub (download/upload).

## Key Functions
- `_download`: Ensures a repo/path is available locally (HF Hub `snapshot_download`, optional ModelScope via `MLXLM_USE_MODELSCOPE=true`), using an allowlist of files.
- `load_config`: Reads `config.json` and merges `generation_config.json` (notably `eos_token_id`) when present.
- `load_model`: Main logic for loading weights and initializing model classes.
- `sharded_load` / `pipeline_load`: Handles distributed/sharded loading.
- `save_model`: Saves weights in safetensors format.
- `save_config` / `save`: Writes MLX-style folder layout including config + tokenizer and auxiliary files.
- `quantize_model` / `dequantize_model`: Applies/removes quantization.
- `upload_to_hub` / `create_model_card`: Hub publishing helpers.
- `_get_classes`: Maps configuration `model_type` to `mlx_lm.models.*` classes.

## Code Quality
- **High Complexity**: `load_model` does a lot (config loading, weight loading, quantization, sanitization).
- **Registry**: `MODEL_REMAPPING` hardcodes mapping from HF model types to internal names.
- **Dependency**: Heavy reliance on `mlx.core` and `mlx.nn`.
- **Good Practice**: explicit resource limit setting for opening many files.
- **Correctness risks**:
  - `load_model()` calls `logging.error(...)` without importing stdlib `logging` (can raise `NameError` if the “no safetensors found” strict path is hit).
  - `save()` currently ignores its `donate_model` parameter and always donates (`save_model(..., donate_model=True)`), which is surprising/possibly unintended API behavior.
