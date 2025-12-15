# Core Module Analysis (`mlx_lm`)

## Purpose
The `mlx_lm/` package is the orchestration layer for the project: it exposes the public Python API, provides CLI entrypoints, and implements model loading, text generation (including batching and speculative decoding), conversion/export utilities, evaluation helpers, and an OpenAI-compatible HTTP server.

## Entry Points
- `mlx_lm/__init__.py`: Public API exports (`load`, `generate`, `stream_generate`, `batch_generate`, `convert`), and sets `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`.
- `mlx_lm/__main__.py`: CLI multiplexer (`python -m mlx_lm <subcommand> ...`) that imports `mlx_lm.<subcommand>` and calls `main()`.

## Key Components

### 1. Model Loading, Saving, and Quantization (`mlx_lm/utils.py`)
- **Download & caching**: `_download()` uses `huggingface_hub.snapshot_download` (or ModelScope if `MLXLM_USE_MODELSCOPE=true`) with a file allowlist; `hf_repo_to_path()` supports `local_files_only`.
- **Config merge**: `load_config()` merges `generation_config.json` into `config.json` (notably `eos_token_id`).
- **Architecture selection**: `_get_classes()` dynamically imports `mlx_lm.models.<model_type>` (with `MODEL_REMAPPING` for HF compatibility).
- **Loading**: `load_model()` reads `model*.safetensors`, applies optional `sanitize()`, and supports both current `quantization` and legacy `quantization_config` flows.
- **Sharding**: `sharded_load()` supports pipeline parallelism and tensor parallelism; pipeline mode requires `model.safetensors.index.json` (MLX-converted models).
- **Saving/publishing**: `save_model()` shards weights (default 5GB) and writes `model.safetensors.index.json`; `save_config()` normalizes/sorts config; `save()` copies tokenizer + auxiliary files and generates a model card; `upload_to_hub()` uploads a folder.

### 2. Generation Engine (`mlx_lm/generate.py`)
- **Core loops**:
  - `generate_step()`: autoregressive decoding with prompt prefill chunking, async eval, optional KV-cache quantization (`kv_bits`, `quantized_kv_start`), and optional `input_embeddings` (gated by `does_model_support_input_embeddings()`).
  - `speculative_generate_step()`: draft-model speculative decoding using trimmable prompt caches.
  - `stream_generate()`: yields `GenerationResponse` objects with timing and memory stats, wrapping plain HF tokenizers in `TokenizerWrapper` when needed.
- **Batching**: `BatchGenerator` batches compatible requests with left/right padding and cache merging; it only supports a subset of cache types and manages `mx` wired-limit settings for throughput.
- **CLI**: `main()` wires together model loading, optional prompt-cache file loading, chat-template application, and generation parameters.

### 3. Tokenization & Streaming Detokenization (`mlx_lm/tokenizer_utils.py`)
- **Detokenizers**: `NaiveStreamingDetokenizer` (universal but O(T²) worst-case), plus fast paths for SentencePiece (`SPMStreamingDetokenizer`) and BPE (`BPEStreamingDetokenizer`) based on `tokenizer.json` decoder structure.
- **`TokenizerWrapper`**: unifies HF tokenizer + streaming detokenizer, supports extra EOS token IDs, and detects “thinking” / “tool_call” tokens (tokenizer-dependent).
- **Custom tokenizer modules**: `load()` supports `tokenizer_config.json` with `tokenizer_type`, loading `mlx_lm.tokenizers.<tokenizer_type>` to override chat templating logic.

### 4. Sampling Helpers (`mlx_lm/sample_utils.py`)
- `make_sampler()` builds a composable sampler chain (top-p / min-p / top-k / XTC / argmax).
- `make_logits_processors()` supports logit bias and repetition penalty.
- Several operations are `mx.compile`’d with explicit random state handling for performance.

### 5. Serving (`mlx_lm/server.py`)
- **HTTP server**: uses `ThreadingHTTPServer`, but generation is coordinated by a single background `ResponseGenerator` thread that can batch compatible requests via `BatchGenerator`.
- **Endpoints**: `/v1/completions`, `/v1/chat/completions` (streaming via SSE), `/v1/models` (scans local HF cache), and `/health`.
- **Prompt caching**: `LRUPromptCache` stores reusable prompt KV caches and can trim longer caches when supported.
- **Protocol details**: stop sequences (token and text) are handled carefully for streaming; basic tool-calling is supported when the tokenizer provides appropriate control tokens.

### 6. Conversion and Publishing (`mlx_lm/convert.py`, `mlx_lm/upload.py`, `mlx_lm/gguf.py`)
- **Conversion**: `convert.py` loads lazily, optionally casts dtypes (via `cast_predicate`), supports quantize/dequantize flows, and saves a portable MLX folder layout.
- **Quant recipes**: supports a small set of mixed-bit recipes (Llama-style) via `mixed_quant_predicate_builder`.
- **Publishing**: can generate a model card and upload artifacts to the Hub; `gguf.py` provides GGUF export for select model families.

### 7. Evaluation and Benchmarking (`mlx_lm/evaluate.py`, `mlx_lm/perplexity.py`, `mlx_lm/benchmark.py`)
- `evaluate.py` adapts models to `lm-evaluation-harness` via a custom `MLXLM` implementation and uses prompt caches for throughput.
- `perplexity.py` and `benchmark.py` provide CLI utilities for PPL and performance measurement.

## Data Flow
User (Python/CLI) → `utils.load()` → `_download()`/`snapshot_download()` → `load_model()` → `TokenizerWrapper` → `generate.stream_generate()` / `BatchGenerator` → output text.

For serving: Client → `APIHandler` → `ResponseGenerator` → (`BatchGenerator` if batchable, else `stream_generate`) → SSE/JSON response.

## Code Quality Notes (Core)
- **High-centrality modules**: `mlx_lm/utils.py`, `mlx_lm/generate.py`, and `mlx_lm/server.py` are large and multi-responsibility; refactoring boundaries could reduce change-risk.
- **Concrete issues spotted**:
  - `mlx_lm/utils.py` calls `logging.error(...)` in `load_model()` without importing stdlib `logging` (can raise `NameError` on that path).
  - `mlx_lm/utils.py:save()` ignores its `donate_model` argument (always donates), which is surprising API surface.
  - `mlx_lm/sample_utils.py:make_sampler()` uses a mutable default (`xtc_special_tokens=[]`).
