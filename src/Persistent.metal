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
constant char maze[257] =
    "################"
    "#......#.......#"
    "#......#.......#"
    "#..##..........#"
    "#...#...####...#"
    "#...#..........#"
    "#.......#......#"
    "###.#####..##..#"
    "#..............#"
    "#..###..#####..#"
    "#....#.........#"
    "#....#..###....#"
    "#..............#"
    "#..#####..###..#"
    "#..............#"
    "################";

inline uint random_step(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
inline void system_fence() {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, system_scope);
}
inline void device_fence() {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
inline bool wall_at(int x, int y) {
    return x < 0 || y < 0 || x >= 16 || y >= 16 || maze[y * 16 + x] == '#';
}
inline bool can_stand(float x, float y) {
    const float radius = 0.18f;
    return !wall_at(int(floor(x-radius)), int(floor(y-radius))) &&
           !wall_at(int(floor(x+radius)), int(floor(y-radius))) &&
           !wall_at(int(floor(x-radius)), int(floor(y+radius))) &&
           !wall_at(int(floor(x+radius)), int(floor(y+radius)));
}
inline uint rgba(uint r, uint g, uint b) {
    return r | (g << 8) | (b << 16) | 0xff000000u;
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
    uint role [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {

    // Every barrier below is SIMD-group scoped. A slow game frame or token
    // never prevents either of the other two execution contexts progressing.
    threadgroup float train_x[INPUTS], train_h[HIDDEN], train_p[VOCAB], train_dh[HIDDEN];
    threadgroup float infer_h[HIDDEN], infer_p[VOCAB];
    threadgroup uint context[CONTEXT];
    threadgroup float game_state[3];
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
                out[O_TRAIN_SEES_FRAMES] = out[O_FRAMES];
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
                out[O_INFER_SEES_FRAMES] = out[O_FRAMES];
                system_fence(); out[O_TOKENS] = tokens + 1; system_fence();
            }
            ++tokens;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) { out[O_INFER_DONE] = 1; system_fence(); }
    } else if (role == 2) {
        if (lane == 0) { game_state[0] = 2.5f; game_state[1] = 2.5f; game_state[2] = 0.2f; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint frames = 0;
        uint last_frame_ms = lane == 0 ? ctrl[C_TIME_MS] : 0;
        last_frame_ms = simd_broadcast(last_frame_ms, 0);
        for (ulong guard = 0; guard < guard_limit; ++guard) {
            system_fence();
            uint stop = lane == 0 ? (ctrl[C_STOP] | atomic_load_explicit(sync + 3, memory_order_relaxed)) : 0;
            stop = simd_broadcast(stop, 0); if (stop != 0) break;
            uint now = lane == 0 ? ctrl[C_TIME_MS] : 0;
            now = simd_broadcast(now, 0);
            if (frames > 0 && now - last_frame_ms < 33) continue;
            uint keys = lane == 0 ? ctrl[C_KEYS] : 0;
            keys = simd_broadcast(keys, 0);
            float dt = min(float(now - last_frame_ms) * 0.001f, 0.08f);
            last_frame_ms = now;
            if (lane == 0) {
                float x = game_state[0], y = game_state[1], a = game_state[2];
                a += (float((keys & KEY_RIGHT) != 0) - float((keys & KEY_LEFT) != 0)) * 2.2f * dt;
                a = fmod(a + 6.283185307f, 6.283185307f);
                float forward = float((keys & KEY_FORWARD) != 0) - float((keys & KEY_BACKWARD) != 0);
                float strafe = float((keys & KEY_STRAFE_RIGHT) != 0) - float((keys & KEY_STRAFE_LEFT) != 0);
                float dx = (cos(a) * forward - sin(a) * strafe) * 2.5f * dt;
                float dy = (sin(a) * forward + cos(a) * strafe) * 2.5f * dt;
                if (can_stand(x + dx, y)) x += dx;
                if (can_stand(x, y + dy)) y += dy;
                game_state[0] = x; game_state[1] = y; game_state[2] = a;
                out[O_FRAME_SEQ] = frames * 2 + 1; system_fence();
            }
            simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
            float px = game_state[0], py = game_state[1], angle = game_state[2];
            float dirx = cos(angle), diry = sin(angle);
            for (uint sx = lane; sx < WIDTH; sx += 32) {
                float camera = (2.0f * (float(sx) + 0.5f) / WIDTH - 1.0f) * 0.66f;
                float rx = dirx - diry * camera, ry = diry + dirx * camera;
                int mx = int(px), my = int(py), stepx = rx < 0 ? -1 : 1, stepy = ry < 0 ? -1 : 1;
                float deltax = abs(1.0f / (abs(rx) < 1e-6f ? 1e-6f : rx));
                float deltay = abs(1.0f / (abs(ry) < 1e-6f ? 1e-6f : ry));
                float sidex = (rx < 0 ? px - float(mx) : float(mx + 1) - px) * deltax;
                float sidey = (ry < 0 ? py - float(my) : float(my + 1) - py) * deltay;
                bool side = false;
                for (uint j = 0; j < 64; ++j) {
                    if (sidex < sidey) { sidex += deltax; mx += stepx; side = false; }
                    else { sidey += deltay; my += stepy; side = true; }
                    if (wall_at(mx, my)) break;
                }
                float distance = max(side ? sidey - deltay : sidex - deltax, 0.05f);
                float wall_height = float(HEIGHT) / distance;
                int top = int(float(HEIGHT) * 0.5f - wall_height * 0.5f);
                int bottom = int(float(HEIGHT) * 0.5f + wall_height * 0.5f);
                float hit = side ? px + distance * rx : py + distance * ry;
                hit -= floor(hit);
                float shade = (side ? 0.68f : 1.0f) / (1.0f + distance * 0.10f);
                for (uint sy = 0; sy < HEIGHT; ++sy) {
                    uint color;
                    if (int(sy) < top) color = rgba(17 + sy / 5, 25 + sy / 4, 42 + sy / 3);
                    else if (int(sy) > bottom) {
                        uint floor_shade = 20 + (sy - HEIGHT / 2) / 3;
                        color = rgba(floor_shade, floor_shade + 5, floor_shade + 6);
                    } else {
                        float height_fraction = (float(int(sy) - top) / max(wall_height, 1.0f));
                        bool mortar = fract(height_fraction * 8.0f) < 0.045f || fract(hit * 6.0f + (int(height_fraction * 8.0f) & 1) * 0.5f) < 0.045f;
                        float light = shade * (mortar ? 0.45f : 1.0f);
                        bool accent = ((mx + my) % 5) == 0;
                        color = accent ? rgba(uint(65 * light), uint(126 * light), uint(137 * light)) :
                                         rgba(uint(151 * light), uint(100 * light), uint(75 * light));
                    }
                    if (sx >= 8 && sx < 72 && sy >= 8 && sy < 72) {
                        uint mapx = (sx - 8) / 4, mapy = (sy - 8) / 4;
                        color = wall_at(int(mapx), int(mapy)) ? rgba(105, 122, 135) : rgba(12, 18, 24);
                        float dx = float(sx - 8) - px * 4.0f, dy = float(sy - 8) - py * 4.0f;
                        if (dx * dx + dy * dy < 5.0f) color = rgba(255, 116, 77);
                    }
                    if ((sx == WIDTH / 2 && abs(int(sy) - HEIGHT / 2) < 4) ||
                        (sy == HEIGHT / 2 && abs(int(sx) - WIDTH / 2) < 4)) color = rgba(215, 228, 219);
                    out[O_PIXEL_BASE + sy * WIDTH + sx] = color;
                }
            }
            system_fence(); simdgroup_barrier(mem_flags::mem_device);
            ++frames;
            if (lane == 0) {
                out[O_X] = as_type<uint>(px); out[O_Y] = as_type<uint>(py); out[O_ANGLE] = as_type<uint>(angle);
                out[O_KEYS_SEEN] = keys; out[O_GAME_SEES_STEPS] = out[O_STEPS]; out[O_GAME_SEES_TOKENS] = out[O_TOKENS];
                out[O_FRAMES] = frames; system_fence(); out[O_FRAME_SEQ] = frames * 2; system_fence();
            }
        }
        if (lane == 0) { out[O_GAME_DONE] = 1; system_fence(); }
    }
}
