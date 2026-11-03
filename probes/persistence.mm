#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <libkern/OSCacheControl.h>

// This layout matches Metal's 32-bit atomic_uint layout on the tested device.
// Host/device concurrent access is a hardware capability probe, not a claim that
// Metal documents C++/MSL system-scope atomics or a portable CPU/GPU memory model.
struct alignas(128) Cell { uint32_t word; uint32_t padding[31]; };
struct Shared {
    Cell stop, input;
    Cell ticks[3], seen[3], reasons[3], checksums[3], peerTicks[3];
};
static bool cpuFlush = true;
static uint32_t readWord(Cell *p) { if (cpuFlush) sys_dcache_flush(p, sizeof(*p)); return __atomic_load_n(&p->word, __ATOMIC_SEQ_CST); }
static void writeWord(Cell *p, uint32_t v) { __atomic_store_n(&p->word, v, __ATOMIC_SEQ_CST); if (cpuFlush) sys_dcache_flush(p, sizeof(*p)); }

static NSString *source = @R"METAL(
#include <metal_stdlib>
using namespace metal;
struct alignas(128) Cell { atomic_uint word; uint padding[31]; };
struct Shared {
    Cell stop, input;
    Cell ticks[3], seen[3], reasons[3], checksums[3], peerTicks[3];
};
kernel void persistent_probe(volatile device Shared &s [[buffer(0)]],
                             constant uint &limit [[buffer(1)]],
                             device uint *sweep [[buffer(2)]],
                             constant uint &sweepWords [[buffer(3)]],
                             uint group [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint quit[3];
    uint value = 1234567u + group * 101u + lane;
    uint iteration = 0;
    for (; iteration < limit; ++iteration) {
#if USE_FENCE
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
#endif
        if (lane == 0) quit[group] = atomic_load_explicit(&s.stop.word, memory_order_relaxed);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (quit[group] != 0) break;
        uint input = atomic_load_explicit(&s.input.word, memory_order_relaxed);
        for (uint j = 0; j < 64; ++j) {
            value ^= value << 13; value ^= value >> 17; value ^= value << 5;
            value += input;
        }
        if (lane == 0) {
            atomic_store_explicit(&s.checksums[group].word, value, memory_order_relaxed);
            atomic_store_explicit(&s.seen[group].word, input, memory_order_relaxed);
            atomic_store_explicit(&s.ticks[group].word, iteration + 1, memory_order_relaxed);
            atomic_store_explicit(&s.peerTicks[group].word, atomic_load_explicit(&s.ticks[(group + 1) % 3].word, memory_order_relaxed), memory_order_relaxed);
        }
        simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        for (uint k = group * 32u + lane; k < sweepWords; k += 96u) {
            value ^= sweep[k];
            sweep[k] = value + k + iteration;
        }
    }
    if (lane == 0)
        atomic_store_explicit(&s.reasons[group].word, iteration == limit ? 2u : 1u,
                              memory_order_relaxed);
}
)METAL";

int main(int argc, char **argv) {
    @autoreleasepool {
        double seconds = argc > 1 ? atof(argv[1]) : 1.0;
        uint32_t limit = argc > 2 ? strtoul(argv[2], nullptr, 10) : 1000000;
        uint32_t sweepMiB = argc > 3 ? strtoul(argv[3], nullptr, 10) : 0;
        uint32_t sweepWords = sweepMiB * 1024 * 1024 / 4;
        cpuFlush = argc > 4 ? atoi(argv[4]) != 0 : true;
        bool writeCombined = argc > 5 ? atoi(argv[5]) != 0 : true;
        bool useFence = argc > 6 ? atoi(argv[6]) != 0 : true;
        if (seconds < 0.1 || seconds > 15 || limit == 0) {
            fprintf(stderr, "usage: %s [seconds 0.1..15] [iteration-limit]\n", argv[0]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 2; }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_2;
        options.preprocessorMacros = @{@"USE_FENCE": @(useFence)};
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 2; }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"persistent_probe"] error:&error];
        if (!pipeline) { fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 2; }
        id<MTLBuffer> buffer = [device newBufferWithLength:sizeof(Shared) options:MTLResourceStorageModeShared | (writeCombined ? MTLResourceCPUCacheModeWriteCombined : MTLResourceCPUCacheModeDefaultCache)];
        Shared *shared = (Shared *)buffer.contents;
        id<MTLBuffer> sweepBuffer = [device newBufferWithLength:MAX(4, sweepWords * 4) options:MTLResourceStorageModeShared];
        memset(shared, 0, sizeof(*shared));
        writeWord(&shared->input, 1);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        command.label = @"ONE persistent compute dispatch, one group with three SIMD groups";
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBytes:&limit length:sizeof(limit) atIndex:1];
        [encoder setBuffer:sweepBuffer offset:0 atIndex:2];
        [encoder setBytes:&sweepWords length:sizeof(sweepWords) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];
        [encoder endEncoding];
        auto start = std::chrono::steady_clock::now();
        printf("device=%s unified=%d seconds=%.1f limit=%u groups=1 threads=96 simdgroups=3 dispatches=1\n",
               device.name.UTF8String, device.hasUnifiedMemory, seconds, limit);
        printf("sweepMiB=%u layout=128B-separated cpuFlush=%d writeCombined=%d deviceFence=%d\n", sweepMiB, cpuFlush, writeCombined, useFence);
        fflush(stdout);
        [command commit];
        uint32_t previous[3] = {}, progressSamples[3] = {};
        bool allObservedBeforeCompletion = false, inputObservedBeforeCompletion = false;
        unsigned samples = 0;
        double lastPrint = -1;
        uint32_t latestInput = 1;
        for (;;) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            auto status = command.status;
            uint32_t ticks[3], seen[3];
            bool all = true, inputAll = true;
            for (int i = 0; i < 3; ++i) {
                ticks[i] = readWord(&shared->ticks[i]);
                seen[i] = readWord(&shared->seen[i]);
                if (ticks[i] > previous[i] && status < MTLCommandBufferStatusCompleted) ++progressSamples[i];
                previous[i] = ticks[i];
                all &= ticks[i] != 0;
                inputAll &= seen[i] == latestInput;
            }
            if (status < MTLCommandBufferStatusCompleted && all) allObservedBeforeCompletion = true;
            if (status < MTLCommandBufferStatusCompleted && latestInput > 1 && inputAll) inputObservedBeforeCompletion = true;
            if (elapsed - lastPrint >= 0.5 || status >= MTLCommandBufferStatusCompleted) {
                printf("t=%.3f status=%ld ticks=%u,%u,%u input=%u seen=%u,%u,%u\n", elapsed, (long)status,
                       ticks[0], ticks[1], ticks[2], latestInput, seen[0], seen[1], seen[2]);
                fflush(stdout); lastPrint = elapsed;
            }
            ++samples;
            if (elapsed >= seconds || status >= MTLCommandBufferStatusCompleted) break;
            latestInput = 1 + (uint32_t)(elapsed * 4);
            writeWord(&shared->input, latestInput);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        writeWord(&shared->stop, 1);
        auto stopStart = std::chrono::steady_clock::now();
        [command waitUntilCompleted];
        double stopMillis = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stopStart).count();
        double runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        bool stopped = true, progressed = true;
        for (int i = 0; i < 3; ++i) {
            stopped &= readWord(&shared->reasons[i]) == 1;
            progressed &= progressSamples[i] >= 3;
        }
        bool pass = command.status == MTLCommandBufferStatusCompleted && allObservedBeforeCompletion &&
                    inputObservedBeforeCompletion && stopped && progressed && runtime >= seconds;
        printf("%s status=%ld runtime=%.3fs stop=%.3fms samples=%u liveProgress=%u,%u,%u liveAll=%d liveInput=%d reasons=%u,%u,%u checksums=%u,%u,%u\n",
               pass ? "PASS" : "FAIL", (long)command.status, runtime, stopMillis, samples,
               progressSamples[0], progressSamples[1], progressSamples[2], allObservedBeforeCompletion,
               inputObservedBeforeCompletion, readWord(&shared->reasons[0]), readWord(&shared->reasons[1]),
               readWord(&shared->reasons[2]), readWord(&shared->checksums[0]), readWord(&shared->checksums[1]),
               readWord(&shared->checksums[2]));
        if (command.error) fprintf(stderr, "Metal error: %s\n", command.error.description.UTF8String);
        printf("peerTicks=%u,%u,%u finalTicks=%u,%u,%u\n", readWord(&shared->peerTicks[0]), readWord(&shared->peerTicks[1]), readWord(&shared->peerTicks[2]), readWord(&shared->ticks[0]), readWord(&shared->ticks[1]), readWord(&shared->ticks[2]));
        return pass ? 0 : 1;
    }
}
