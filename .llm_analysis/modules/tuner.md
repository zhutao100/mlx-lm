# Tuner Module Analysis (`mlx_lm.tuner`)

## Purpose
Facilitates fine-tuning of LLMs using Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA and QLoRA.

## Key Components

### 1. LoRA Layers (`lora.py`)
-   **Wrappers**:
    -   `LoRALinear`: Wraps `nn.Linear`. Adds low-rank `lora_a` and `lora_b` matrices.
    -   `LoRASwitchLinear`: Support for MoE layers.
    -   `LoRAEmbedding`: Support for embedding layers.
-   **Functionality**:
    -   `from_base`: Factory to create LoRA layer from existing layer.
    -   `fuse`: Merges LoRA weights back into the base layer for export.

### 2. DoRA Layers (`dora.py`)
-   **Concept**: Weight-Decomposed Low-Rank Adaptation. Decomposes weights into magnitude and direction.
-   **Components**:
    -   `DoRALinear`: Adapts `nn.Linear`. Updates direction via LoRA and magnitude via a learned scalar `scale`.
    -   `DoRAEmbedding`: Adapts `nn.Embedding` similarly.
-   **Integration**: Similar API to LoRA (`from_base`, `fuse`). Note: `DoRAEmbedding` currently lacks quantization support.

### 3. Trainer (`trainer.py`)
-   **Loop**: `train()` function manages the training loop.
-   **Features**:
    -   `grad_checkpoint`: Gradient checkpointing to save VRAM.
    -   `iterate_batches`: Efficient batching with sorting by length.
    -   `evaluate`: Validation loop.
    -   Adapter saving: Periodically saves `adapters.safetensors`.

### 3. Datasets (`datasets.py`)
-   **Abstractions**:
    -   `TextDataset`: Plain text.
    -   `ChatDataset`: Handles chat templates (`{"messages": [...]}`).
    -   `CompletionsDataset`: Prompt/Completion pairs.
-   **Loading**: Supports loading from local JSONL or Hugging Face Hub.

### 4. Losses (`losses.py`)
-   **Custom Kernels**:
    -   `kl_div_loss`: Optimized Metal kernel for KL Divergence.
    -   `js_div_loss`: Jensen-Shannon Divergence.
-   Used for distillation or specific fine-tuning objectives.

## Workflow
1.  User configures `TrainingArgs`.
2.  `load()` model with `adapter_path` (optional init).
3.  Wrap layers with `LoRALinear`.
4.  `train()` executes loop -> Updates LoRA weights.
5.  Save adapters.
