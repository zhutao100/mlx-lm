# mlx_lm/models/llama.py

## Purpose
- Implementation of Llama model architecture.

## Key Components
- `ModelArgs`: Configuration specific to Llama.
- `Attention`, `MLP`, `TransformerBlock`: Standard Transformer components.
- `Model`: Wrapper class matching the `mlx_lm` interface (includes `shard`, `sanitize`, `make_cache`).

## Code Quality
- **Standard**: Follows a standard pattern seen in other MLX examples.
- **Sharding**: Explicit `shard` method for distributed execution.
- **Sanitization**: `sanitize` removes unused weights (like rotary freq buffers) from HF checkpoints.
