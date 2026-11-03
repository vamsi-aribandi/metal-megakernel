# Live CPU/GPU communication

The demo needs keyboard input, text and frames to cross between the CPU and GPU while one Metal dispatch keeps running. It uses `coherent(system)` pointers and system-scope fences to do this. **These are undocumented compiler internals.** They work on the tested M4 with macOS 26.5.1, but Apple does not promise compatibility across devices or OS/compiler updates.

## Why shared storage is not enough

Apple's [shared-storage documentation](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) requires synchronization before the other processor accesses modified data. Unified physical memory does not guarantee that a running shader sees CPU writes, or that the CPU sees shader writes before completion. Public device-scope fences do not provide a documented system-scope guarantee.

Our ordinary shared-buffer atomic probe did not exchange updates during execution. An independent [ArrowMetal M4 Max study](https://github.com/singhpratech/ArrowMetal/blob/main/docs/RESIDENT.md) also found unreliable visibility with public qualifiers and barriers. It reported that CPU and GPU writes sharing a cache line could overwrite each other. The demo therefore uses separate input and output buffers, each with a single owning processor.

## The experimental mechanism

The [ccv ANE implementation](https://github.com/liuliu/ccv/blob/baa0ef3178c31a5f42befe50d5dd4a53cedbc84a/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L368) provides the precedent for enabling Metal compiler internals, using `coherent(system)`, and defining a system fence scope through `__METAL_MEMORY_SCOPE_SYSTEM__`, with fallback value 3. The demo applies that mechanism to its host mailbox.

All host-visible shader accesses use the coherence qualifier. Payload writes are fenced before publishing a completion counter. Every polling iteration also executes a system fence, including iterations waiting for input or an acknowledgement. Omitting the fence from a waiting path caused a probe to stall; the qualifier alone was insufficient.

The CPU uses atomic accesses to mailbox words. GPU-only model snapshots use separate storage and the device synchronization described in [Metal research](metal-research.md). The host mailbox remains an experimentally verified protocol, not a portable C++/MSL memory-model guarantee.

## Reproduce the visibility test

From the repository root:

```sh
./probes/run-visibility-system.sh 10 26000000
```

The arguments are duration in seconds and a finite GPU polling limit. The [probe](../probes/visibility-system.mm) submits one threadgroup of 96 threads: three independent SIMD groups. It uses separate CPU-owned input and GPU-owned output buffers. Each group publishes 64-word payloads containing its sequence and observed input. The CPU checks every word, then acknowledges the payload before the GPU may overwrite it. No additional dispatches, blits or cache-eviction sweeps run during the test.

The [recorded 10-second run](../probes/visibility-system-10s.log) on the M4 completed with:

| Measurement | Result |
|---|---:|
| Duration | 10.001 s |
| Verified payloads per SIMD group | 3,985 |
| Input changes observed per group | 200 |
| Payload mismatches | 0 |
| Groups observing peer progress | 3/3 |
| Groups exiting on host stop | 3/3 |
| Observed stop completion delay | 0.120 ms |

The log records progress with command status 3, before completion. Status 4 and no command error confirm successful exit. The complete application also passed 60-second headless and 120-second GUI runs, including changing model weights, generated text and responsive game input; see [validation](validation.md). These runs establish behavior on the tested environment, not indefinite operation or public API support.
