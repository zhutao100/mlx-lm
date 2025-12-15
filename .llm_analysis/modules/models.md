# Models Module Analysis (`mlx_lm.models`)

> Scope note: detailed per-architecture review of `mlx_lm/models/` is deferred for now (per instruction). This document captures the shared structure and common patterns.

## Purpose
Contains the implementations of various Large Language Model architectures adapted for MLX.

## Key Components

### 1. Base Classes (`base.py`)
-   `BaseModelArgs`: Dataclass for configuration.
-   `create_attention_mask`: Generates causal and padding masks.
-   `scaled_dot_product_attention`: Core attention mechanism (supports quantized and standard).

### 2. Caching (`cache.py`)
Critical for efficient autoregressive generation.
-   `KVCache`: Standard append-only cache.
-   `RotatingKVCache`: For infinite generation (overwrites old tokens).
-   `QuantizedKVCache`: Compresses KV pairs to reduce memory bandwidth.
-   `BatchKVCache`: Handles cache for batched inputs with different padding.

### 3. Layers
-   `SwitchLayers` (`switch_layers.py`): Implements Mixture-of-Experts (MoE) routing logic (`SwitchGLU`, `SwitchLinear`).
-   `RoPE` (`rope_utils.py`): Rotary Positional Embeddings implementation.

### 4. Model Architectures
Implements `Model`, `ModelArgs`, and `TransformerBlock` for each family.
-   **Llama**: Standard architecture (`llama.py`).
-   **Mixtral**: Sparse MoE architecture (`mixtral.py`).
-   **Qwen/DeepSeek**: Variants with specific attention/MoE implementations.
-   **Common Pattern**:
    -   `ModelArgs` defines hyperparameters.
    -   `Model` class ties embeddings, layers, and norm.
    -   `sanitize` method cleans weights (e.g., removing unused rotary freqs) during loading/saving.

## Interface
All models expose:
-   `__call__(inputs, cache=None)`
-   `from_dict(config)`
-   `sanitize(weights)`
