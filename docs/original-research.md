# Original project architecture

Inspected the public [Bobliuuu/open-flapping-airplanes repository](https://github.com/Bobliuuu/open-flapping-airplanes), commit [`9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1`](https://github.com/Bobliuuu/open-flapping-airplanes/commit/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1), retrieved 2026-11-02 local time. The source identifies itself as an independent reconstruction of Flapping Airplanes' video, not the company's implementation. No original CUDA workload was executed on this Mac.

## Two implementations worth distinguishing

The [small demo](https://github.com/Bobliuuu/open-flapping-airplanes/blob/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1/a100-demo/demo.cu) uses a character language model: eight preceding characters, a fixed random embedding of width 16, a 512-unit tanh hidden layer, and a 96-way softmax. Its 115,296 trained scalars are the two affine layers and their biases; embeddings are fixed. Batch size is 256, and learning uses cross-entropy and SGD at 0.012. Thus the original already demonstrates the architectural idea without a transformer.

Training comprises hidden activations, softmax/loss, then backward/update tasks. Inference computes hidden units, samples a character on the GPU, appends it to a ring buffer, and updates its context. These tasks read the same live weights that training modifies; the source does not establish a coherent checkpoint per generated token.

The persistent `vm` workers use ready-task counters and prioritize game, then inference, then training. Game/inference use sharded tickets with CAS claims; training phases use fetch-add claims. There is one long-running launch for the demonstration, with separate setup and benchmark kernels elsewhere in the file.

The [later GPT implementation](https://github.com/Bobliuuu/open-flapping-airplanes/blob/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1/llm/ofa_fast/ofa.cu) uses persistent worker blocks statically assigned by CUDA SM identifier. Its training and decode jobs contain phase tables, dependency edges, ticket counters, and completion counters. Workers fetch a task, acquire-check its dependencies, execute cooperatively, fence writes, then increment completion. The host uploads the task graph before dispatch. Decode weights point directly into the training BF16 weight buffer. Completion of training signals the other jobs to exit.

The [reference model](https://github.com/Bobliuuu/open-flapping-airplanes/blob/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1/llm/baseline/common.py) specifies GPT S: 12 layers, width 768, 12 attention heads, MLP width 3072, context 512, batch 8, padded vocabulary 100,352, tied token/output embeddings, learned positional embeddings, pre-layer normalization, causal attention, and GELU. It uses AdamW with warmup/cosine learning rate, FP32 master weights, and BF16 computation in the corresponding megakernel. The repository reports approximately 162.4M parameters and trains on TinyStories tokenized with cl100k.

## Game and host boundary

The [Doom integration](https://github.com/Bobliuuu/open-flapping-airplanes/blob/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1/llm/ofa_fast/doom_job.cuh) executes the actual engine's tick on GPU thread zero, with 255 drawing workers. Five additional tasks copy the indexed framebuffer and palette. Game time advances in 35 Hz ticks; device timing controls pacing. The host loads assets, allocates mapped pinned buffers, and consumes output. It does not calculate game physics or neural-network operations.

The [build explanation](https://github.com/Bobliuuu/open-flapping-airplanes/blob/9dd7dd0d3c1fc0245b43897eacd8d62aa096c4c1/llm/ofa_fast/README.md) describes compiling Doom to NVPTX and merging its PTX into the megakernel module. This is CUDA-specific compiler infrastructure, not portable game code that can simply be included in Metal. Doom also needs a libc substitute, recursion support, and substantial stack space. A GPU-native raycaster or small arena is a sensible initial simplification.

## Architectural requirements to preserve on Metal

1. Submit the workload once; keep model training, token generation, and game state progression inside that invocation.
2. Give all three jobs independently progressing GPU execution contexts. A host loop launching each job or repeated short kernels does not reproduce this mechanism.
3. Keep model parameters, gradients, optimizer state, inference context/RNG, and simulation state GPU-owned after initialization.
4. Publish only inputs and observability through a defined mailbox protocol: controls in; loss, versioned tokens, and pixels out.
5. Prove progression while the command buffer is still executing, with one dispatch count and advances in all three job counters. Verify gradient correctness and loss reduction separately from visual presentation.

Implementation recommendations, rather than claims about Metal guarantees: start with the original-style tiny neural language model; use a few persistent workers; avoid a grid-wide barrier or dependency on unscheduled threadgroups; replace SM identifiers with logical worker roles. Prefer immutable GPU-published weight snapshots and versioned output over the original's unsynchronized live reads. Use one GPU worker for simulation and rendering initially, then parallelize only when needed. CPU output formatting, image display, input collection, and resource initialization are compatible with the architecture; CPU model inference, training, or game updates are not.

Metal residency, forward progress, device synchronization, and host-visible memory behavior require separate documentation review and actual device probes. This source study alone does not establish that CUDA synchronization can be translated directly.
