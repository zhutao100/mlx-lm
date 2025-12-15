# mlx_lm/chat.py

## Purpose
- Simple CLI for chatting with models.

## Key Observations
- Wraps `stream_generate`.
- Handles chat template application.
- Uses `mlx.distributed` to support multi-GPU inference via `sharded_load`.
