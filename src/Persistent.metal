// Inspired by open-flapping-airplanes; see CREDITS.md for sources and notices.
// Config.h is prepended by the host. The system-coherent mailbox spelling is
// an experimental compiler internal; see docs/host-gpu-coherence.md.
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant thread_scope system_scope = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
constant float learning_rate = 0.025f;
constant char initial_context[9] = "the gpu ";
inline uint random_step(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
inline void system_fence() {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, system_scope);
}
inline void device_fence() {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
// sync[0]: snapshot ready (0 producer owns; 1 consumer owns).
// sync[1]: snapshot version; sync[2]: bitwise additive snapshot hash.
// sync[3]: training ended. Producer/consumer never wait for each other.
kernel void persistent(
    const volatile coherent(system) device uint *ctrl [[buffer(0)]],
    volatile coherent(system) device uint *out [[buffer(1)]],
    device float *model [[buffer(2)]],
    coherent(device) device float *snapshot [[buffer(3)]],
    device float *infer_model [[buffer(4)]],
    const device uint *corpus [[buffer(5)]],
    constant uint &corpus_length [[buffer(6)]],
    device atomic_uint *sync [[buffer(7)]],
    device u8 *doomMemory [[buffer(8)]],
    device DoomState *doomState [[buffer(9)]],
    const device u8 *doomWad [[buffer(10)]],
    const volatile coherent(system) device DoomHostControl *doomHost [[buffer(11)]],
    volatile coherent(system) device DoomLiveOutput *doomOut [[buffer(12)]],
    constant DoomPersistentConfig &doomConfig [[buffer(13)]],
    const device DoomReplayOperation *doomOperations [[buffer(14)]],
    uint role [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Every barrier below is SIMD-group scoped. A slow game frame or token
    // never prevents either of the other two execution contexts progressing.
    threadgroup float train_x[INPUTS], train_h[HIDDEN], train_p[VOCAB], train_dh[HIDDEN];
    threadgroup float infer_h[HIDDEN], infer_p[VOCAB];
    threadgroup uint context[CONTEXT];
    const ulong guard_limit = 1000000000000ul;

    if (role == 0) {
        uint rng = ctrl[C_SEED]; if (rng == 0) rng = 1;
        uint step = 0;
        float moving_loss = 0.0f;
        for (ulong guard = 0; guard < guard_limit; ++guard) {
            system_fence();
            uint stop = lane == 0 ? ctrl[C_STOP] : 0;
            stop = simd_broadcast(stop, 0);
            if (stop != 0) break;
            uint maximum = lane == 0 ? ctrl[C_MAX_STEPS] : 0;
            maximum = simd_broadcast(maximum, 0);
            if (maximum != 0 && step >= maximum) break;

            rng = random_step(rng);
            uint offset = rng % (corpus_length - CONTEXT);
            uint target = corpus[offset + CONTEXT];
            for (uint i = lane; i < INPUTS; i += 32)
                train_x[i] = model[W_EMBED + corpus[offset + i / EMBED] * EMBED + i % EMBED];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint h = lane; h < HIDDEN; h += 32) {
                float z = model[W_HBIAS + h];
                for (uint i = 0; i < INPUTS; ++i)
                    z += model[W_HIDDEN + h * INPUTS + i] * train_x[i];
                train_h[h] = tanh(z);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float local_max = -INFINITY;
            for (uint v = lane; v < VOCAB; v += 32) {
                float z = model[W_OBIAS + v];
                for (uint h = 0; h < HIDDEN; ++h)
                    z += model[W_OUTPUT + v * HIDDEN + h] * train_h[h];
                train_p[v] = z; local_max = max(local_max, z);
            }
            float maximum_logit = simd_max(local_max), local_sum = 0.0f;
            for (uint v = lane; v < VOCAB; v += 32) {
                float p = exp(train_p[v] - maximum_logit);
                train_p[v] = p; local_sum += p;
            }
            float sum = simd_sum(local_sum);
            for (uint v = lane; v < VOCAB; v += 32) train_p[v] /= sum;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float loss = -log(max(train_p[target], 1e-20f));
            for (uint h = lane; h < HIDDEN; h += 32) {
                float dh = 0.0f;
                for (uint v = 0; v < VOCAB; ++v)
                    dh += model[W_OUTPUT + v * HIDDEN + h] * (train_p[v] - (v == target ? 1.0f : 0.0f));
                float hval = train_h[h];
                train_dh[h] = dh * (1.0f - hval * hval);
            }
            // All hidden derivatives read the old output weights first.
            simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            for (uint i = lane; i < VOCAB * HIDDEN; i += 32) {
                uint v = i / HIDDEN, h = i % HIDDEN;
                model[W_OUTPUT + i] -= learning_rate * (train_p[v] - (v == target ? 1.0f : 0.0f)) * train_h[h];
            }
            for (uint v = lane; v < VOCAB; v += 32)
                model[W_OBIAS + v] -= learning_rate * (train_p[v] - (v == target ? 1.0f : 0.0f));
            for (uint i = lane; i < HIDDEN * INPUTS; i += 32)
                model[W_HIDDEN + i] -= learning_rate * train_dh[i / INPUTS] * train_x[i % INPUTS];
            for (uint h = lane; h < HIDDEN; h += 32)
                model[W_HBIAS + h] -= learning_rate * train_dh[h];
            simdgroup_barrier(mem_flags::mem_device);
            ++step;
            moving_loss = step == 1 ? loss : 0.98f * moving_loss + 0.02f * loss;

            uint publish = 0;
            if (lane == 0 && (step == 1 || (step % 128) == 0))
                publish = atomic_load_explicit(sync + 0, memory_order_relaxed) == 0;
            publish = simd_broadcast(publish, 0);
            if (publish) {
                device_fence();
                uint hash = 0;
                for (uint p = lane; p < PARAMS; p += 32) {
                    float value = model[p]; snapshot[p] = value; hash += as_type<uint>(value);
                }
                hash = simd_sum(hash);
                device_fence();
                simdgroup_barrier(mem_flags::mem_device);
                if (lane == 0) {
                    atomic_store_explicit(sync + 1, step, memory_order_relaxed);
                    atomic_store_explicit(sync + 2, hash, memory_order_relaxed);
                    device_fence();
                    atomic_store_explicit(sync + 0, 1u, memory_order_relaxed);
                    out[O_SNAPSHOT_VERSION] = step; out[O_TRAIN_HASH] = hash;
                }
            }
            if (lane == 0) {
                if (step == 1) out[O_FIRST_LOSS] = as_type<uint>(loss);
                out[O_LOSS] = as_type<uint>(moving_loss);
                out[O_TRAIN_SEES_TOKENS] = out[O_TOKENS];
                out[O_TRAIN_SEES_FRAMES] = doomOut->renders;
                system_fence(); out[O_STEPS] = step; system_fence();
            }
        }
        if (lane == 0) {
            device_fence(); atomic_store_explicit(sync + 3, 1u, memory_order_relaxed);
            out[O_TRAIN_DONE] = 1; system_fence();
        }
    } else if (role == 1) {
        if (lane < CONTEXT) context[lane] = uint(initial_context[lane]) - 32;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint rng = ctrl[C_SEED] ^ 0x9e3779b9u; if (rng == 0) rng = 1;
        uint tokens = 0, version = 0, hash = 0, snapshot_errors = 0, snapshot_copies = 0;
        uint last_token_ms = lane == 0 ? ctrl[C_TIME_MS] : 0;
        last_token_ms = simd_broadcast(last_token_ms, 0);
        for (ulong guard = 0; guard < guard_limit; ++guard) {
            system_fence();
            uint stop = lane == 0 ? (ctrl[C_STOP] | atomic_load_explicit(sync + 3, memory_order_relaxed)) : 0;
            stop = simd_broadcast(stop, 0);
            if (stop != 0) break;
            uint now = lane == 0 ? ctrl[C_TIME_MS] : 0;
            now = simd_broadcast(now, 0);
            if (now - last_token_ms < 16) continue;
            last_token_ms = now;
            uint ready = lane == 0 ? atomic_load_explicit(sync + 0, memory_order_relaxed) : 0;
            ready = simd_broadcast(ready, 0);
            if (ready) {
                device_fence();
                uint v = lane == 0 ? atomic_load_explicit(sync + 1, memory_order_relaxed) : 0;
                uint advertised_hash = lane == 0 ? atomic_load_explicit(sync + 2, memory_order_relaxed) : 0;
                version = simd_broadcast(v, 0); advertised_hash = simd_broadcast(advertised_hash, 0);
                uint copied_hash = 0;
                for (uint p = lane; p < PARAMS; p += 32) {
                    float value = snapshot[p]; infer_model[p] = value;
                    copied_hash += as_type<uint>(value);
                }
                hash = simd_sum(copied_hash);
                ++snapshot_copies;
                if (hash != advertised_hash) ++snapshot_errors;
                device_fence(); simdgroup_barrier(mem_flags::mem_device);
                if (lane == 0) {
                    out[O_SNAPSHOT_ERRORS] = snapshot_errors;
                    out[O_SNAPSHOT_COPIES] = snapshot_copies;
                    atomic_store_explicit(sync + 0, 0u, memory_order_relaxed);
                }
            }
            for (uint h = lane; h < HIDDEN; h += 32) {
                float z = infer_model[W_HBIAS + h];
                for (uint i = 0; i < INPUTS; ++i)
                    z += infer_model[W_HIDDEN + h * INPUTS + i] *
                         infer_model[W_EMBED + context[i / EMBED] * EMBED + i % EMBED];
                infer_h[h] = tanh(z);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float local_max = -INFINITY;
            for (uint v = lane; v < VOCAB; v += 32) {
                float z = infer_model[W_OBIAS + v];
                for (uint h = 0; h < HIDDEN; ++h)
                    z += infer_model[W_OUTPUT + v * HIDDEN + h] * infer_h[h];
                z /= 0.8f;
                infer_p[v] = z; local_max = max(local_max, z);
            }
            float max_logit = simd_max(local_max), local_sum = 0.0f;
            for (uint v = lane; v < VOCAB; v += 32) {
                float p = exp(infer_p[v] - max_logit); infer_p[v] = p; local_sum += p;
            }
            float total = simd_sum(local_sum);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            rng = random_step(rng);
            if (lane == 0) {
                float pick = float(rng & 0xffffffu) * (1.0f / 16777216.0f) * total;
                float cdf = 0.0f; uint chosen = VOCAB - 1;
                for (uint v = 0; v < VOCAB; ++v) {
                    cdf += infer_p[v]; if (pick < cdf) { chosen = v; break; }
                }
                for (uint c = 0; c + 1 < CONTEXT; ++c) context[c] = context[c + 1];
                context[CONTEXT - 1] = chosen;
                out[O_TOKEN_BASE + tokens % TOKEN_CAP] = chosen;
                out[O_TOKEN_VERSION_BASE + tokens % TOKEN_CAP] = version;
                out[O_INFER_VERSION] = version; out[O_INFER_HASH] = hash;
                out[O_INFER_SEES_STEPS] = out[O_STEPS];
                out[O_INFER_SEES_FRAMES] = doomOut->renders;
                system_fence(); out[O_TOKENS] = tokens + 1; system_fence();
            }
            ++tokens;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) { out[O_INFER_DONE] = 1; system_fence(); }
    } else if (role == 2) {
        // Test-only numerical mode omits Doom; normal runs always have an IWAD.
        if (doomConfig.wadBytes)
            doom_persistent_job(doomMemory, doomState, doomWad, doomHost,
                                doomOut, doomConfig, doomOperations, lane, 32);
        if (lane == 0) {
            out[O_FRAMES] = doomOut->renders;
            out[O_GAME_SEES_STEPS] = out[O_STEPS];
            out[O_GAME_SEES_TOKENS] = out[O_TOKENS];
            out[O_GAME_DONE] = 1;
            system_fence();
        }
    }
}
