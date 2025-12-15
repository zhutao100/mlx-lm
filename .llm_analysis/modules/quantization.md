# Quantization Module Analysis (`mlx_lm.quant`)

## Purpose
Provides tools for reducing model size and improving inference speed through quantization.

## Key Components

### 1. Utils (`utils.py`)
-   `load_data`: Loads calibration datasets (e.g., Wikitext) for quantization sensitive to activation distributions.

### 2. AWQ (`awq.py`)
-   **Algorithm**: Activation-aware Weight Quantization.
-   **Logic**:
    -   `search_best_scale`: Finds optimal scaling factors to minimize quantization error.
    -   `search_best_clip`: Finds optimal clipping thresholds.
    -   `awq_quantize`: Applies the transformation layer-by-layer.
-   **Config**: `AWQConfig` defines specific settings for models (Llama, DeepSeek, etc.).

### 3. GPTQ (`gptq.py`)
-   **Algorithm**: Generative Pre-trained Transformer Quantization.
-   **Logic**:
    -   Uses Hessian information (`Catcher` class accumulates inputs) to determine weight importance.
    -   Updates weights using the inverse Hessian to compensate for quantization error.
    -   `quantize`: Performs the actual bit-packing.

### 4. Dynamic/Sensitivity (`dynamic_quant.py`)
-   **Logic**:
    -   `estimate_sensitivities`: Measures how sensitive each layer is to quantization error (using KL divergence).
    -   `estimate_threshold`: Determines a cutoff to decide which layers to quantize more aggressively.
    -   Mixed-precision approach: Critical layers keep higher precision.

### 5. DWQ (`dwq.py`)
-   **Algorithm**: Distilled Weight Quantization.
-   **Purpose**: Optimizes quantization parameters (scales and biases) by distilling from a teacher model (original or high-precision).
-   **Logic**:
    -   Unfreezes `scales` and `biases` of quantized layers.
    -   Uses KL divergence loss to match student logits to teacher logits on a calibration dataset.
    -   Supports multi-process training (`mx.distributed`).

## Workflow
1.  Load Model (Float16/32).
2.  Load Calibration Data.
3.  Calculate Scales/Hessians (AWQ/GPTQ).
4.  Apply Quantization.
5.  Save Quantized Weights.
