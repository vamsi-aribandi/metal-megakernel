# Doom in the combined kernel

The third SIMD group now calls the real Doom engine from the pinned
[metal-doom submodule](../third_party/metal-doom). The model remains the same
small character MLP: 30,432 stored values, including 28,896 learned weights.
The earlier maze has been removed from the current GPU kernel.

## One submission

The host compiles the generated Doom engine, its persistent protocol and callable
loop, then the model configuration and outer kernel into one Metal library.
`DOOM_EMBEDDED` omits Doom's standalone kernel entry. All translated visible
functions are linked into the pipeline with a call stack depth of 32.

`src/main.mm` encodes and commits once. One threadgroup has 96 threads, split
into three independent SIMD groups. The third calls `doom_persistent_job` with
all 32 lanes. Doom initialization, IWAD copy into its linear memory, gameplay,
and rendering all happen within that invocation. There is no separate Doom
queue or per-frame compute dispatch. The CPU does not run a WASM interpreter.

The Doom instance uses a 96 MiB memory image, a separate zeroed state buffer,
immutable IWAD bytes, input and output mailboxes, and an unused replay buffer
in normal live mode. The CPU uploads the initial image and assets before commit.
Generated build files and IWADs are ignored by Git.

## Live ownership and shutdown

The host provides time and key state. Doom accumulates time at 35 ticks/second,
with at most eight catch-up ticks per render. It publishes a complete snapshot
and leaves it immutable until the host acknowledges its sequence. The host
copies every published frame, diagnostics, and inputs before acknowledging.

`doom-frames.jsonl` records the initial tick-zero frame and each later batch's
tick, sampled keys, render count, SHA256 of raw pixels, sixteen signed diagnostics,
command status, and elapsed time. `doom-frame.bgra` retains the engine's original
`0x00RRGGBB` pixels. `frame.rgba` is only a host-side channel/alpha conversion for
display. The native window never redraws the game scene on the CPU.

Training and inference observe Doom's render counter while the dispatch is
running. Doom records their final progress on return. Live telemetry independently
checks intervals in which all three counters advance before command completion.
Model snapshot checksums and per-character weight versions remain in use.

The host always stops both the model and Doom control buffers. It also stops
all workers when a requested training step limit or a Doom failure occurs.
A held frame acknowledgement does not prevent Doom from observing stop.
`--seconds` starts counting play time at the first Doom snapshot; the clock
sent to the GPU stays monotonic through startup.

## What validation means

`tests/numerics.py` compares the same GPU training path against scalar Python at
1, 4, and 32 updates. Its explicit `--ml-test` mode skips Doom and is not evidence
of combined execution.

`tests/validate_doom.py` checks single submission/dispatch, live overlapping
progress, decreasing training loss, changing and checked weight snapshots,
complete text publication, game controls, and clean shutdown. It then runs the
captured tick/key/render schedule through independently compiled native Doom C,
checking every diagnostic and complete framebuffer hash. Replaying the exact
render schedule matters because rendering also changes some engine state.
The report records the shader, host source, binary, IWAD, and reference-library
hashes; the validator rejects mismatched current host/shader source.

Both native C and GPU build copies apply `engine/renderer_compat.py`. This
zeroes composite texture holes and adds zero-filled 128-byte cache tails. A
longer E1M1 test exposed vanilla column reads into zone allocator metadata,
producing different pixels on 32-bit and 64-bit heaps despite identical game
state and texture coordinates. The guard gives those undefined reads stable
pixels. This reference is independent of WASM and Metal lowering, with that
same documented cache preparation fix. Original vendored Doom source is retained.

The submodule's `tests/test_cache_boundaries.py` replays the exact 2,056-frame
schedule that found the problem against serial and cooperative Metal rendering.
It stores only tick/input pairs, with no game assets. The ordinary persistent
suite also covers viewport changes, menu, map, pause, low detail, and stopping
while a frame is unacknowledged.

## Recorded result

The [saved sixty-second run](evidence/doom-60s/README.md) completed 525,642 training
steps, 3,585 generated characters, and 2,055 Doom frames across 2,099 ticks.
Loss fell from 4.55 to 0.086; snapshot errors and lost tokens were zero.
All 32,880 game diagnostics and all 2,055 framebuffer hashes matched native C.
All three counters advanced in 547 sampled intervals while the command was still
running. Stop took 19 ms. The regression replay separately matched 2,056 frames
in each renderer, and model update errors stayed below 1.5e-8.

The AppKit window/event loop ran during the minute test. The Mac was locked
when computer-use inspection was attempted, so a visual window check remains
unverified; no screenshot is claimed.

## Limits

This uses experimental `coherent(system)` and system-scope fences, tested on an
M4 Mac mini with 16 GB, 10 GPU cores, and macOS 26.5.1 (25F80). Results establish
only the recorded durations on this machine. Doom traversal and gameplay are
mostly serial, rendering partly parallel, and audio is absent. The program
uses three SIMD groups rather than filling the GPU.

The combined program is GPL-2.0-or-later. Earlier MIT component notices remain
in `third_party/licenses`; the Doom submodule retains its upstream credits.
Assets are supplied separately. Both integration pull requests are kept unmerged.
