# mlx_lm/models/base.py

## Purpose
- Base utilities and classes for model implementations.

## Key Components
- `BaseModelArgs`: Dataclass base.
- `create_attention_mask`: Generates causal and padding masks.
- `scaled_dot_product_attention`: Dispatches to `mx.fast.sdpa` or a custom quantized implementation.

## Code Quality
- **Reuse**: Centralizes critical attention logic, reducing duplication in individual models.
- **Quantization Support**: Explicitly handles quantized SDPA.
