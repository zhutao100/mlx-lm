# Quantized Rotating KV Cache

## Contract

- `RotatingKVCache.to_quantized(group_size=64, bits=4)` returns `QuantizedRotatingKVCache`, a `QuantizedKVCache` subclass compatible with quantized attention dispatch.
- Conversion preserves the global token offset, retained prefix, sliding-window size, ring write position, and attention masks.
- Conversion quantizes only used entries and trims transient oversized prefill buffers before quantization.
- Single-token updates use bounded ring storage; multi-token updates retain temporal order and can temporarily exceed the window so each token has sufficient context.
- Rewinding is supported only before the sliding window overwrites history.

## State and Storage

- `state` contains the full quantized buffers and all scalar fields: `(keys, values, offset, keep, max_size, _idx, group_size, bits)`.
- `from_state()` and prompt-cache save/load reconstruct empty, partially filled, and rotated caches without a separate metadata setter.
- `keys_and_values()` returns only the active KV entries. Use it instead of unpacking `state` for attention inputs.
- `size()` reports the logical window length; `nbytes` uses the inherited quantized-buffer accounting.
- This follows upstream's full-state cache protocol. The earlier split between array state and string metadata is obsolete.

## Coverage

- Prompt-cache round trips preserve quantization and rotation parameters and subsequent updates.
- Conversion preserves window masks and dequantized values within quantization tolerance.
- Full-state restoration covers empty caches, unused capacity, and oversized prefill buffers.
- Conversion does not quantize unused capacity.

## Scope

- Generation uses this cache through the existing KV quantization path.
- Batched rotating-cache quantization is not implemented.
