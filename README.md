# metal-megakernel

One persistent Metal GPU kernel that **trains a language model, generates text from its changing weights, and runs an interactive maze at the same time** on an Apple Silicon Mac.

Inspired by Jerry Zhu and Arnold Xu's [open-flapping-airplanes](https://github.com/Bobliuuu/open-flapping-airplanes). This project carries its central idea over to Metal with a smaller model and a simple game. See [credits](CREDITS.md) for the original work and other sources that helped.

**Experimental:** live communication between the CPU and the running GPU kernel uses an undocumented Metal compiler feature, `coherent(system)`. It works on the tested M4 Mac, but a future macOS or compiler update could break it. [Details](docs/host-gpu-coherence.md).

## Run it

Tested on an **M4 Mac mini with 16 GB memory, 10 GPU cores, and macOS 26.5.1**. You need Apple's Command Line Tools. Install them with `xcode-select --install` if necessary. Python 3 is needed for the tests.

```sh
git clone https://github.com/vamsi-aribandi/metal-megakernel.git
cd metal-megakernel
./run.sh
```

The window shows the loss, generated text, and game. Everything starts from scratch on each launch.

| Keys | Action |
|---|---|
| W / S or Up / Down | Move forward / backward |
| A / D or Left / Right | Turn |
| Q / E | Strafe |
| Esc | Stop the GPU kernel |

Closing the window also stops the kernel. After building, you can open `build/Metal Megakernel.app` from Finder. Keep that app inside the build folder; it reads the shader and corpus from the project.

No model download or Python packages are needed. The GPU code is compiled at launch through Metal, so a separate `metal` compiler executable is not required.

## How it works

The app launches **one threadgroup of 96 GPU threads, once**. Those threads form three independent groups of 32:

1. **Train:** read a text sample, predict the next character, calculate the error, and update the model's weights.
2. **Generate:** take a fresh snapshot of the weights and use it to produce more text.
3. **Play:** read the controls, move the player, check walls, and draw the maze pixels.

All three groups stay inside the same GPU invocation until stopped. They use only local group barriers, so one job does not wait for the others at a shared barrier. The CPU initializes resources, sends keyboard state and wall time, and displays the results. It does not calculate training updates, generated characters, game movement, or the game image.

Training and generation use separate weight buffers. A small mailbox passes complete snapshots between them. Generation checks each copied snapshot and records which model version produced each character. The game also publishes complete frames so the CPU can avoid showing a half-written image.

The GPU code is written directly in **Metal Shading Language**. The Mac app uses **Objective-C++**, **Metal**, and **AppKit/Cocoa**. Tests use **Python**, and shell scripts build and run the project.

## What changed from upstream?

The upstream repository has both a small character model and a later 162-million-parameter GPT transformer. This project follows the **small character model**:

| | Upstream small demo | This project |
|---|---:|---:|
| Previous text the model sees | 8 characters | 8 characters |
| Hidden layer | 512 units | 128 units |
| Learned weights | 115,296 | 28,896 |
| Character vocabulary | 96 | 96 |
| GPU API | CUDA | Metal |
| Game | Doom | Raycast maze |

Our model has about a quarter as many learned weights as upstream's small model. It also has 1,536 fixed embedding values. It learns letter patterns from the short included corpus, so its text can be repetitive or incoherent. This is a demonstration of concurrent GPU work, not a general conversation model.

Upstream uses work queues and, in its larger version, partitions work across many NVIDIA GPU cores. This version gives each job a fixed group of 32 threads in one threadgroup. That keeps scheduling simple and uses Metal's documented guarantee that these groups can make independent progress.

**Why not Doom?** Upstream compiles Doom into NVIDIA GPU code and merges it into the CUDA kernel. That build path cannot be reused with Metal. Porting the engine and its runtime support would be a substantial separate task. The maze provides movement, collision, controls, and GPU rendering with much less code.

## Test it

```sh
./test.sh
```

This builds the app, compares GPU training updates against an independent Python reference, then runs and checks a ten-second live demo. The Python model calculations exist only in the test.

For a longer recorded run:

```sh
./build/metal-megakernel --headless --seconds 60 --test-input --output results/my-run
python3 tests/validate.py results/my-run --min-seconds 60 --scripted
```

Omit `--headless` to show the window. Omit `--test-input` for manual controls. Use `--corpus FILE` to supply printable ASCII text. `--help` lists the options. A normal launch runs until stopped; `Ctrl-C` stops a headless run.

A recorded two-minute GUI run completed **857,309 training steps, 7,115 characters, and 3,471 frames**, with zero snapshot checksum mismatches. The loss fell from 4.55 to 0.20. See [validation results and saved evidence](docs/validation.md).

## Limits

- The undocumented system-coherence feature is the main compatibility risk. Ordinary shared-buffer atomics did not provide reliable live CPU/GPU communication in our tests.
- Only three groups of 32 threads run the workloads. This is not tuned to fill all ten GPU cores or match upstream's speed.
- Long-running kernels may encounter OS timeouts. The recorded runs prove their tested durations, not unlimited runtime or support on other Macs.
- The CPU supplies wall time for pacing. If its event loop stalls, training continues while text and game pacing wait for a clock update. A long display stall can also lose old text from the output ring; lost characters are counted.
- The small corpus makes learning easy to see, but falling training loss does not establish general language ability.

## Files and further reading

- [`src/Persistent.metal`](src/Persistent.metal): all three GPU workloads.
- [`src/main.mm`](src/main.mm): setup, the one dispatch, input/output, and native window.
- [`src/Config.h`](src/Config.h): shared sizes and buffer layout.
- [`docs/original-research.md`](docs/original-research.md): the upstream design we studied, with pinned source links.
- [`docs/metal-research.md`](docs/metal-research.md): Metal's scheduling and synchronization rules.
- [`docs/host-gpu-coherence.md`](docs/host-gpu-coherence.md): the experimental live-memory mechanism.

MIT licensed. Credits and third-party notices are in [CREDITS.md](CREDITS.md) and [third_party/licenses](third_party/licenses).
