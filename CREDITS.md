# Credits

## Original idea and upstream project

This project is inspired by **[open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes)** by **Jerry Zhu and Arnold Xu**. That project demonstrated training, generation, and Doom inside one persistent CUDA kernel. It is itself an independent recreation of the Flapping Airplanes demonstration.

We studied upstream at commit [`9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1`](https://github.com/Bobliuuu/open-flapping-airplanes/commit/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1). Its small character model informed our model design, and its persistent workers informed the architecture. The implementation here uses Metal, a smaller hidden layer, fixed worker groups, complete weight snapshots, and real Doom through the metal-doom submodule. Game assets are supplied separately.

Upstream is MIT licensed, copyright 2026 Jerry Zhu. Its [license notice](third_party/licenses/open-flapping-airplanes-MIT.txt) is retained here. See [the source study](docs/original-research.md) for the exact upstream files.

## Live CPU/GPU communication

The system-coherent pointer and fence spelling was informed by **Liu Liu's [ccv](https://github.com/liuliu/ccv)**, specifically its [Metal/ANE implementation](https://github.com/liuliu/ccv/blob/baa0ef3178c31a5f42befe50d5dd4a53cedbc84a/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L368), studied at commit `baa0ef3178c31a5f42befe50d5dd4a53cedbc84a`.

This project applies that compiler-internal spelling in its own mailbox protocol and probes. ccv's applicable BSD 3-Clause [license notice](third_party/licenses/ccv-BSD-3-Clause.txt) is retained here. This is experimental compiler behavior, not an Apple-supported public synchronization API.

**[ArrowMetal](https://github.com/singhpratech/ArrowMetal)** and its [resident-worker investigation](https://github.com/singhpratech/ArrowMetal/blob/main/docs/RESIDENT.md) helped explain why ordinary shared-memory polling was unreliable. The local probes tested the behavior independently on the available M4 Mac.

Apple's [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) supplies the documented GPU scheduling and synchronization rules used by the implementation.

These acknowledgments do not imply endorsement by the original authors or Apple.

## Doom engine and Metal port

[metal-doom](https://github.com/vamsi-aribandi/metal-doom), by Vamsi Aribandi,
provides the native Metal Doom engine and persistent game loop in
[`third_party/metal-doom`](third_party/metal-doom). It builds the original Doom C
through a portable WASM/wasm2c intermediate into ordinary native Metal functions;
there is no bytecode interpreter at runtime. The submodule retains Doom,
DoomGeneric, Chocolate Doom, libc, and other upstream notices in its CREDITS.md
and vendor/licenses directory. Doom's authors are id Software.

The combined program is **GPL-2.0-or-later**, because it incorporates the GPL Doom
engine. The earlier MIT component notice is preserved in
[metal-megakernel-original-MIT.txt](third_party/licenses/metal-megakernel-original-MIT.txt).
The original component licenses and author notices remain applicable. No IWAD
assets are included or relicensed. Clone with submodules to obtain the complete
source; binary redistribution must provide corresponding source including the
pinned Doom submodule and its build tools under the GPL.
