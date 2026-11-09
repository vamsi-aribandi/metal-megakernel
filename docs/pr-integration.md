Run real Doom, training, and generation in one persistent Metal dispatch

Replace the maze worker with the native Doom engine from the metal-doom git
submodule. One command buffer launches one 96-thread threadgroup: three SIMD
groups initialize/run Doom, train the small character model, and generate text
from checked model snapshots. The host supplies time/input and displays published
frames. It does not simulate Doom or launch per-frame GPU work.

Add IWAD selection, complete Doom frame records, independent native-C replay,
clean shutdown of all workers, build instructions, upstream attribution, and GPL
licensing for the combined program. The original MIT component notice remains.
The submodule pins the unmerged embed-persistent-job branch commit, which also
fixes allocator-dependent tail pixels in short textures.

Validation on Apple M4 / macOS 26.5.1:

- `test.sh`: independent model math at 1/4/32 updates and twelve-second combined replay passed.
- One-minute native AppKit run: 525,642 training steps, 3,585 characters, 2,055
  Doom frames / 2,099 ticks; every frame hash and all 32,880 diagnostics matched
  native C with documented cache padding. Zero snapshot errors or lost tokens;
  loss 4.55 to 0.086; stop 19 ms.
- Standalone Doom regressions: 2,056 frames per renderer, plus 650 persistent
  control/render-state frames, all matched native C.
- Missing/malformed IWAD and CLI error paths checked.

Evidence is under docs/evidence/doom-60s. The Mac was locked during attempted
computer-use inspection, so no visual screenshot check is claimed. System
coherence remains experimental and undocumented. Game assets are not bundled.
Please leave this PR unmerged.
