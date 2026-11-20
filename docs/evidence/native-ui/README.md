# Native window and startup verification — November 19, 2026

Verified the packaged macOS app visually using native computer control:
Doom, training loss/graph, generated text, movement, turning, fire, use, Escape,
and closing the stopped window. Also opened the app without a WAD argument,
selected an IWAD in the native picker, and verified the game appeared and stopped.
Picker cancellation was checked too.

The recorded final bundled build ran for twenty seconds with native keyboard
input (W, D, Space, Return), without the app's `--test-input` mode. All 647
published Doom frames and 10,352 diagnostics matched independently compiled
native C with documented cache padding. It completed 160,814 training steps
and 1,210 characters with no snapshot errors or lost tokens. Stop took 28 ms.
The validator required the observed key mask 201 (forward, right, fire, use).

The bundle now contains a real executable instead of an external symlink, with
resource lookup supporting both CLI and bundle paths. macOS launch completes
before the modal file picker, and the game window is explicitly created after
selection. This fixes native app discovery and startup without changing the
compiled GPU source; its SHA256 matches the earlier minute recording.

`cli-validation.json` and `numerics.json` are the final `test.sh` rerun after
these startup fixes. The older minute recording belongs to host source at
commit `460a601`; its historical metadata is retained in ../doom-60s.

No IWAD or game image bytes are committed. The records contain input, state,
text, and hashes. Screenshots and raw frames remain in the local results folders.
