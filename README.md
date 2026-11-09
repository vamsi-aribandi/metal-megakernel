# metal-megakernel

One persistent Metal GPU kernel that **trains a small language model, generates
text from its changing weights, and runs real Doom at the same time** on an Apple
Silicon Mac.

Inspired by Jerry Zhu and Arnold Xu's
[open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes).
Doom comes from our [metal-doom](https://github.com/vamsi-aribandi/metal-doom)
submodule, built from the original game C into native Metal functions.
See [credits and licenses](CREDITS.md).

**Experimental:** live CPU/GPU communication uses the undocumented Metal compiler
feature `coherent(system)`. A future compiler or macOS release could break it.
[Details](docs/host-gpu-coherence.md).

## Run it

Requires an Apple Silicon Mac with unified memory, Apple's Command Line Tools,
Python 3.9+, and the Doom build tools. On a Homebrew Mac:

```sh
xcode-select --install  # only if Command Line Tools are missing
brew install llvm lld wabt
git clone --recurse-submodules --branch integrate-metal-doom \
  https://github.com/vamsi-aribandi/metal-megakernel.git
cd metal-megakernel
./run.sh --wad /path/to/doom1.wad
```

Supply your own compatible Doom IWAD; game assets are not bundled. You can also
set `DOOM_WAD`, or omit the path in window mode to use the native file picker.
The app starts fresh each launch, on E1M1. It shows training loss, generated
characters, and Doom at 320×200. The first build and Metal compilation take time.

| Keys | Action |
|---|---|
| W / S or Up / Down | Forward / backward |
| A / D or Left / Right | Turn |
| Q / E | Strafe |
| Space / Enter | Fire / use |
| Shift / 1–7 | Run / select weapon |
| Tab / M / Z | Map / game menu / menu enter |
| Esc or close window | Stop all three GPU workers |

After building, `build/Metal Megakernel.app` can also open from Finder. Keep it
inside the build folder so it can find the source, corpus, and generated engine.

## How it works

The app submits **one command buffer, one dispatch, one threadgroup of 96 GPU
threads**. Three independent SIMD groups of 32 stay inside that invocation:

1. **Train:** predict the next character, calculate gradients, update weights.
2. **Generate:** copy complete, checked snapshots of the changing weights and
   sample more text.
3. **Play:** initialize Doom, read controls, advance the game, and render pixels.

Doom gameplay and scene traversal run on the game group's owner lane; its 32
lanes cooperate on opaque rendering and palette conversion. The CPU uploads
assets before launch, supplies time and keyboard state, and copies published
frames to the native window. It does not simulate Doom or draw the game scene.
All workload barriers are scoped to a SIMD group. A slow Doom frame does not
require the training or generation groups to wait at a shared barrier.

The model and outer kernel use **Metal Shading Language**. The native app uses
**Objective-C++**, **Metal**, and **AppKit/Cocoa**. Doom starts as **C**, built
through WASM and wasm2c into native Metal source; WASM is a build intermediate,
with no runtime interpreter. Tests use **Python**; shell scripts build and run it.

## Differences from upstream

This follows upstream's small character model, rather than its later
162-million-parameter transformer:

| | Upstream small demo | This project |
|---|---:|---:|
| Context | 8 characters | 8 characters |
| Hidden layer | 512 units | 128 units |
| Learned weights | 115,296 | 28,896 |
| Vocabulary | 96 | 96 |
| GPU API | CUDA | Metal |
| Game | Doom | Doom via metal-doom |

There are also 1,536 fixed embedding values. The model learns letter patterns
from the included short corpus; it is not a general conversation model. Fixed
worker groups keep scheduling simple but do not fill the whole GPU. The earlier
version used a raycast maze while the native Metal Doom port was being built;
the current integration replaces that maze with the actual game engine.

## Validate it

```sh
DOOM_WAD=/path/to/doom1.wad ./test.sh
```

This compares training updates against an independent scalar Python model, then
records all three running together for twelve seconds. Every published Doom
frame is replayed in the original native C engine with the exact same inputs,
tick batches, and render schedule. All sixteen game diagnostics and every
framebuffer hash must match. Both builds apply the documented zero-filled cache
padding described below. The C reference runs only in tests.

For a longer recording:

```sh
./build/metal-megakernel --headless --wad /path/to/doom1.wad \
  --seconds 60 --test-input --output results/doom-60s
python3 tests/validate_doom.py results/doom-60s --wad /path/to/doom1.wad \
  --min-seconds 60 --scripted
```

`--seconds` measures play time after Doom's first frame. Omit `--headless` for the
window, or `--test-input` for manual controls. `Ctrl-C` stops a headless run.
`--corpus FILE` supplies printable ASCII text. `--ml-test --headless --steps N`
is an explicit numerical test mode that omits Doom; it does not prove combined
execution. `--help` lists normal options.

A recorded sixty-second run completed **525,642 training steps, 3,585 characters,
and 2,055 Doom frames**, with every Doom frame matching native C and zero
snapshot errors. See [saved evidence](docs/evidence/doom-60s/README.md).

[Integration and validation details](docs/doom-integration.md) explain the buffer
ownership and evidence. Older maze results in [docs/validation.md](docs/validation.md)
are historical and do not validate the Doom integration.

## Limits and source

- Live system coherence uses experimental compiler internals.
- Only three SIMD groups run the workloads; this is not tuned for maximum GPU use.
- Doom's gameplay and traversal are mostly serial, with a 96 MiB linear memory
  image. Audio is not implemented. Consult the Doom submodule for engine limits.
- The CPU supplies pacing time. A stalled host pauses Doom publication and text
  pacing; training can continue. A long host stall can lose old generated text,
  which the app counts. Doom uses acknowledgement ownership to preserve frames.
- Persistent kernels may encounter OS limits. Tests establish their measured
  durations on the tested machine, not indefinite execution on all Macs.

Main files: [`src/Persistent.metal`](src/Persistent.metal),
[`src/main.mm`](src/main.mm), [`src/Config.h`](src/Config.h), and
[`third_party/metal-doom`](third_party/metal-doom).
[Upstream study](docs/original-research.md) ·
[Metal scheduling notes](docs/metal-research.md).

The combined program is **GPL-2.0-or-later**. Component notices are retained in
[CREDITS.md](CREDITS.md) and the submodule. Game assets are separate.
