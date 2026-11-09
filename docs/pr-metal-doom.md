Expose the persistent Doom worker and make texture cache boundaries deterministic

The standalone Doom loop can now be called from one SIMD group inside a larger
kernel. `DOOM_EMBEDDED` omits the standalone entry point; the original entry
forwards to the same loop. Buffer ownership and benchmark source transforms are
preserved and documented.

A longer E1M1 replay exposed short-column samples reaching zone allocator
metadata, yielding different pixels on native 64-bit and WASM 32-bit heaps.
Generated native and GPU C copies now clear composite texture holes and reserve
zero-filled cache tails. Original vendored source is unchanged. Undefined tail
pixels have a documented deterministic value.

Validation on Apple M4 / macOS 26.5.1:

- `tests/test_cache_boundaries.py`: serial and cooperative Metal each matched
  native C across the captured 2,056-frame schedule and all 32,896 diagnostics.
- `tests/test_persistent.py`: 650 frames matched, including 1/8-tick batches,
  viewport/menu/map/pause changes, live controls, and stop while unacknowledged.
- Benchmark shader source transformations still succeed.
- A separate metal-megakernel run validated training, generation, and Doom in one
  96-thread persistent dispatch for a minute, matching all 2,055 Doom frames.

Native reference builds apply the same documented cache padding, independently
of WASM/Metal lowering. No IWAD assets are committed. Please leave this PR unmerged.
