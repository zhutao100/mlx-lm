
Regarding the task

```markdown
The `to_quantized(..)` method in the class `RotatingKVCache` at `mlx_lm/models/cache.py` is unimplemented.

Propose plan of implementing the method.

```

---

**Goal / Success Criteria**
- `RotatingKVCache.to_quantized()` returns a cache that can replace the float cache *in-place* during generation and still supports `max_kv_size` semantics (`keep`, `_idx` ring behavior, and `make_mask()`), while switching attention to the quantized KV path (`hasattr(cache, "bits")`).

**Compatibility / Dispatch (do this first)**
- Verify the quantized-attention dispatch: `mlx_lm/models/base.py:scaled_dot_product_attention()` uses `hasattr(cache, "bits")` to select quantized SDPA.
- To reduce integration risk, implement `QuantizedRotatingKVCache` as a subclass of `QuantizedKVCache` (and keep `bits` / `group_size` attributes), so it is a drop-in “quantized cache” by both attribute and type.

**1) Add `QuantizedRotatingKVCache`**
- Location: `mlx_lm/models/cache.py`, next to the other cache classes.
- Data model:
  - `keys`, `values`: quantized tuples `(q, scales, zeros)` like `QuantizedKVCache`.
  - Scalars: `keep`, `max_size`, `offset` (global), `_idx` (ring write pointer / temporal end), `group_size`, `bits`.
- Required interface parity with existing caches:
  - `update_and_fetch()`, `make_mask()`, `__len__`, `trim()`/`is_trimmable()`, `state`/`meta_state`, `from_state()` compatibility.

**2) Reuse Rotating logic via tuple-safe helpers**
- Implement `_trim(trim_size, v, append=None)` and `_temporal_order(v)` analogous to `RotatingKVCache`, but operating on quantized tuples via `tree_map`.
- Use the time axis consistently as `axis=-2` (and `shape[-2]`) to match `QuantizedKVCache` conventions and avoid hard-coding axis `2`.
- Copy `RotatingKVCache.make_mask()` verbatim (it depends on `offset`, `_idx`, `keep`, `max_size`, `window_size`).

**3) Implement `QuantizedRotatingKVCache.update_and_fetch()`**
- Dispatch like float rotating:
  - If `keys.shape[2] == 1`: `_update_in_place()`
  - Else: `_update_concat()`

**3a) `_update_in_place(keys, values)` (decode, S==1)**
- Ensure bounded allocation up to `max_size` (mirror the float growth behavior, but allocate quant buffers):
  - Initialize / expand quant buffers in chunks up to `max_size` using the same packing math as `QuantizedKVCache` (`el_per_int = 8 * mx.uint32.size // bits`).
- If current buffer length exceeds `max_size` (possible right after a prior concat update), trim down to `max_size` with `_trim()` and set `_idx = max_size` (matching float’s pre-rotation state and preserving the `make_mask(N==1, window_size<max_size)` clamp behavior).
- Rotate write pointer: if `_idx == max_size`, set `_idx = keep`.
- Quantize incoming `keys/values` and assign each tuple element into `[..., _idx:_idx+1, :]`.
- Update `offset += 1`, `_idx += 1`.
- Return:
  - If `offset < max_size`: slice time axis to `offset` (avoid exposing unused capacity).
  - Else return full buffers (ring order is fine; order doesn’t matter for N==1, and past order is irrelevant for N>1).

**3b) `_update_concat(keys, values)` (prefill blocks, S>1)**
- Put existing cache in temporal order (`_temporal_order`) to preserve context, then set `_idx = current_len`.
- Trim to `max_size - 1` “past” tokens (respecting `keep`) with `trim_size = _idx - max_size + 1`.
- Quantize incoming block and append via `_trim(..., append=q_in)`.
- Update `offset += S`, `_idx = new_len`.
- Return the concatenated quantized buffers (size will be `max_size + S - 1`, matching float rotating behavior).

**3c) `trim()` / `is_trimmable()` parity (don’t change semantics)**
- Match `RotatingKVCache` behavior: `is_trimmable()` should be true only while `offset < max_size` (once history has been overwritten, rewinding is not safe).
- `trim(n)` should adjust both `offset` and `_idx` consistently, mirroring `RotatingKVCache.trim()`.

**4) Serialization support**
- `state`: return `(keys, values)` (slice to the *used* length when appropriate, similar to existing caches).
- `meta_state`: include both rotating + quant params, e.g. `(keep, max_size, offset, _idx, group_size, bits)` as strings.
- Implement corresponding setters so `load_prompt_cache()` can reconstruct via `from_state()`.

**5) Implement `RotatingKVCache.to_quantized()`**
- Construct a `QuantizedRotatingKVCache(max_size=self.max_size, keep=self.keep, group_size=..., bits=...)`.
- Copy `offset`.
- If empty: return it.
- Decide what to quantize (avoid quantizing transient oversized concat buffers):
  - Start from float used data: `k, v = self.state` (drops unused capacity).
  - If `k.shape[2] > max_size`: trim to `max_size` using the same `_trim()` rule as float `_update_in_place` would apply (`trim_size = k_len - max_size`). Set `idx = max_size` (this preserves the `idx>=max_size -> idx=0` branch in `make_mask(N==1, window_size<max_size)` and keeps the next in-place update’s rotation behavior aligned).
  - Else: keep `idx = self._idx` (clamp to `k.shape[2]` defensively).
- Quantize `k, v` with `mx.quantize(...)`, assign into the new cache, set `_idx = idx`, return it.

**6) Tests to add (high signal, minimal)**
- Prefer non-brittle invariants over “exact token identity” (quantization can change sampling decisions).
- `test_rotating_to_quantized_preserves_window_mask_after_concat`: reproduce the `S>1` concat state then compare `make_mask(N=1, window_size<max_size)` before/after conversion (validates `_idx` handling and the `idx>=max_size -> 0` clamp behavior).
- `test_rotating_to_quantized_dequantize_close_to_float`: construct a controlled cache state, convert, dequantize (`mx.dequantize`) and compare against the float buffers that were actually quantized (post-trim if oversized), using tolerances appropriate for `bits`.
- `test_save_load_quantized_rotating_cache`: roundtrip via `save_prompt_cache`/`load_prompt_cache`, then do a single-token update and assert returned (quantized) buffers match and metadata (`offset`, `_idx`, `keep`, `max_size`, `bits`, `group_size`) is preserved.
- Optional smoke test: run `generate_step(..., max_kv_size=..., kv_bits=..., kv_group_size=...)` to ensure no crash.

**Optional follow-up**
- Implement `BatchRotatingKVCache.to_quantized()` only if you also plan to support kv-cache quantization in `BatchGenerator` (currently kv-cache quantization is only wired into `generate_step` / `speculative_generate_step`).
