# Metal architecture and limits

The demo uses one persistent dispatch with three SIMD groups in one threadgroup. Metal documents their independent progress and GPU memory synchronization. Live communication with the CPU uses an experimental compiler feature, described in [the coherence note](host-gpu-coherence.md).

## Tested hardware and tools

On 2026-11-02, `sw_vers`, `system_profiler` and the Metal device query identified an **Apple M4 Mac mini, 10 GPU cores, 16 GB unified memory, macOS 26.5.1 (25F80)**. The available Command Line Tools could compile the Objective-C++ host, but did not include the standalone `metal` compiler. The project compiles its shader at runtime with `newLibraryWithSource`, targeting MSL 3.2.

## Why three SIMD groups share one threadgroup

The [Apple MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), 2026-06-04 edition, supplies the relevant rules:

- **§5.2.3.6, p.157:** SIMD groups in one threadgroup execute concurrently and make independent forward progress without intervening threadgroup barriers.
- **§6.10.1, pp.215–217:** SIMD-group barriers synchronize that group. All participating lanes must encounter them; memory flags select which accesses they order.
- **§4.8, p.123:** ordinary device memory has threadgroup coherence. MSL 3.2 adds `coherent(device)` for communication across threadgroups.
- **§6.16.1–3, pp.299–302:** MSL 3.2 supports sequentially consistent fences on Apple silicon. Ordinary atomic operations at this language version use relaxed ordering. Public fence scopes stop at the GPU device.

The host checks `threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`, following Apple's [threadgroup sizing guide](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes). The implementation requires a SIMD width of 32 and dispatches 96 threads. Each SIMD group owns one workload, and the shader uses no full-threadgroup barrier. It therefore avoids relying on simultaneous residency of separate threadgroups.

## Sharing model weights inside the kernel

The trainer owns mutable weights; inference owns a separate copy. They exchange snapshots through one slot:

1. The trainer observes EMPTY, fences, fills the slot, fences, then publishes READY.
2. Inference observes READY, fences, copies the weights and verifies their checksum, then fences before returning EMPTY.
3. Training continues even when the slot is occupied. Inference keeps its copied weights for a complete token.

State flags are relaxed atomics, paired with this MSL 3.2 fence:

```metal
atomic_thread_fence(mem_flags::mem_device,
                    memory_order_seq_cst, thread_scope_device);
```

Within each role, lanes take the same publication branch and use SIMD barriers to finish their cooperative copies before the leader changes the state. Ownership prevents the trainer from overwriting a snapshot while inference reads it.

## Communication with the CPU

Apple's [shared-storage documentation](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) and [coherence rules](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Mem-Obj/Mem-Obj.html) guarantee the usual boundary: CPU writes before commit, GPU results after completion. They do not guarantee polling during a running dispatch. `volatile` alone does not solve cache visibility.

Two approaches worked in local experiments:

| Approach | Evidence | Limitation |
|---|---|---|
| Public MSL with a 64 MiB GPU cache-eviction sweep and host cache operations | [10-second log](../probes/probe-10s.log): all three roles advanced, saw changing input and stopped | Uses substantial memory bandwidth; eviction has no documented visibility bound |
| Compiler-internal `coherent(system)` and system fences | [10-second integrity probe](../probes/visibility-system-10s.log): 3,985 verified 64-word payloads per role, zero mismatches | Undocumented and dependent on the compiler/OS |

The application uses the second approach, based on the [ccv implementation](https://github.com/liuliu/ccv/blob/baa0ef3178c31a5f42befe50d5dd4a53cedbc84a/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L368). The [ArrowMetal study](https://github.com/singhpratech/ArrowMetal/blob/main/docs/RESIDENT.md) provides independent evidence of the public API limitation. Neither source is an Apple support guarantee for the private mechanism.

Inputs and outputs occupy separate buffers with one owning processor each. Host-visible shader accesses use system coherence, and every polling iteration fences, including idle paths. Model computation, game state and rendering remain in the persistent kernel.

## Execution duration and validation

Apple documents [command-buffer timeouts](https://developer.apple.com/documentation/metal/mtlcommandbuffererror-swift.struct/timeout) for work that exceeds the system's allowed duration, without specifying a universal limit. The demo's stop is cooperative; indefinite execution and compatibility with other hardware are unproven.

The application passed **60-second headless and 120-second GUI runs** on the tested Mac. Live traces show all three workloads advancing while the same command remained incomplete; numerical tests check GPU training against an independent CPU reference. Snapshot checks, versioned text, GPU-generated frames and keyboard-response checks are recorded in [validation](validation.md).

Public [`MTLSharedEvent`](https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu) signaling and managed-resource [`synchronize(resource:)`](https://developer.apple.com/documentation/metal/mtlblitcommandencoder/synchronize(resource:)) operate through encoded commands. They do not provide the shader-side, mid-dispatch host mailbox used here.
