# Validation on an M4 Mac

Tested during development on 2026-11-02: **M4 Mac mini, 16 GB memory, 10 GPU cores, macOS 26.5.1 (25F80)**. The shader targets Metal Shading Language 3.2.

These results establish the tested runs on this machine. They do not establish unlimited runtime or support on other Macs and OS versions.

## Live application runs

Each run used one command buffer, one compute dispatch, and one threadgroup of 96 threads. All three GPU jobs made progress while that command was still running. Each job also recorded seeing its peers make progress.

| Measurement | 60-second headless run | 120-second native GUI run |
|---|---:|---:|
| Actual duration | 60.008 s | 120.023 s |
| Training steps | 518,429 | 857,309 |
| Generated characters | 3,211 | 7,115 |
| Game frames | 1,607 | 3,471 |
| First loss | 4.5529 | 4.5529 |
| Final loss average | 0.09448 | 0.20251 |
| Live intervals with all three advancing | 584 | 1,092 |
| Checked inference snapshots | 3,211 | 6,606 |
| Snapshot checksum mismatches | 0 | 0 |
| Lost characters | 0 | 0 |
| GPU stop latency | 6.39 ms | 16.77 ms |
| Normal worker exits | 3/3 | 3/3 |

Saved evidence:

- [Complete 60-second run](evidence/headless-60s): summary, live measurements, per-character model versions, generated text, framebuffer, and validation result.
- [120-second GUI summary](evidence/gui-120s-summary.json) and [validation result](evidence/gui-120s-validation.json).

The headless test sent forward, backward, left, and right controls. The validator checked that the GPU saw each input and moved or turned the player while training and generation continued. The native window was also inspected while running; held W, D, and Q keys exercised movement, turning, and strafing.

The GUI accepted 3,467 complete frame copies. One copy overlapped a GPU frame update and was discarded, as intended. The validator checks the minimap, player marker, crosshair, opaque pixels, and nonuniform image content. A close-up wall view can legitimately contain very few colors.

Generation used 3,211 distinct trained snapshots in the headless run and 6,606 in the GUI run. Each character records its model version. Versions never went backward. The consumer checksum is calculated from the weights actually copied into its model, rather than repeating the trainer's checksum.

A separate untimed GUI run was stopped with Escape after 17.736 seconds. It completed 136,999 training steps, 1,046 characters, and 518 frames, with no snapshot errors. All workers exited normally, the window displayed STOPPED, and closing it exited the app. [Result](evidence/gui-stop-validation.json).

## Training calculations

[The numerical test](../tests/numerics.py) compares actual GPU weight updates with an independent scalar Python reference. It checks the loss, resulting updates to every trained weight, and unchanged fixed embeddings. This CPU model runs only in the test.

| GPU updates | Random seed | Maximum weight error | Final loss-average error |
|---|---:|---:|---:|
| 1 | 42 | 7.45×10⁻⁹ | 7.87×10⁻⁸ |
| 4 | 0, treated as 1 | 7.45×10⁻⁹ | 3.69×10⁻⁷ |
| 32 | 12,345 | 1.49×10⁻⁸ | 2.72×10⁻⁶ |

[Saved report](evidence/numerics.json). Python uses higher-precision intermediate calculations; the GPU uses float32 and parallel reductions, so exact bit equality is not expected.

## Live-memory test

Ordinary shared-buffer atomics did not reliably exchange live input and output before the kernel completed. The selected experimental `coherent(system)` mechanism passed a separate test with three independent GPU groups and separate input/output buffers.

During a 10.001-second run, each group published 3,985 acknowledged 64-word messages, consumed 200 changing input values, and observed peer progress. There were zero message errors. All groups saw the stop flag and exited normally in 0.120 ms. [Probe log](../probes/visibility-system-10s.log) and [explanation](host-gpu-coherence.md).

The shader must run a system-scope fence on every polling iteration, including idle paths. An earlier probe stalled when the idle path skipped that fence. The successful probes and application use the corrected loop.

## Reproduce the checks

```sh
./test.sh
./build/metal-megakernel --headless --seconds 60 --test-input --output results/reproduce
python3 tests/validate.py results/reproduce --min-seconds 60 --scripted
```

The full test command builds the app, runs the three numerical cases, and checks a ten-second live run. It has been executed successfully on the tested Mac. Fresh output goes into `results/`, which is ignored by Git.

You can also check the saved 60-second evidence without launching GPU work:

```sh
python3 tests/validate.py docs/evidence/headless-60s --min-seconds 60 --scripted
```

The validator checks command completion, repeated live progress by every job, peer observations, weight versions and snapshot integrity, loss reduction, text, pixels, and optional input response. It also checks that the application source has one compute dispatch and one command submission. The live command status is 3 while scheduled and incomplete, and 4 after successful completion.

This validates the small-model, simpler-game demonstration. It does not measure general language ability, prove a Doom port, show full-GPU occupancy, or make the private memory feature a public Metal guarantee.
