#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#include "Config.h"
#include "../third_party/metal-doom/app/persistent_protocol.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static volatile sig_atomic_t interrupted = 0;
static void signalStop(int) { interrupted = 1; }
static uint32_t load(const uint32_t *p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
static void store(uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
static float asFloat(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t random32(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
static std::string readFile(const std::string &p) {
    std::ifstream f(p); if (!f) throw std::runtime_error("Cannot read " + p);
    return {std::istreambuf_iterator<char>(f), {}};
}
static void writeBinary(const std::string &path, const void *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write((const char *)data, bytes)) throw std::runtime_error("Cannot write " + path);
}
static std::string digest(const void *data, size_t bytes) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, (CC_LONG)bytes, hash);
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned char b : hash) { result += hex[b >> 4]; result += hex[b & 15]; }
    return result;
}
static uint32_t little32(const unsigned char *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static std::string readWad(const std::string &path) {
    auto bytes = std::filesystem::file_size(path);
    if (bytes < 12 || bytes > 64 * 1024 * 1024) throw std::runtime_error("IWAD must be between 12 bytes and 64 MiB");
    std::string data = readFile(path);
    if (data.size() != bytes || memcmp(data.data(), "IWAD", 4)) throw std::runtime_error("Expected an IWAD base game file");
    const auto *b = (const unsigned char *)data.data();
    uint32_t count = little32(b + 4), offset = little32(b + 8);
    if (!count || offset < 12 || uint64_t(offset) + uint64_t(count) * 16 > bytes)
        throw std::runtime_error("Invalid IWAD directory");
    for (uint32_t i = 0; i < count; ++i) {
        const auto *entry = b + offset + uint64_t(i) * 16;
        if (uint64_t(little32(entry)) + little32(entry + 4) > bytes)
            throw std::runtime_error("Invalid IWAD lump bounds");
    }
    return data;
}
struct Options {
    bool headless = false, scripted = false, mlTest = false;
    double seconds = 0;
    uint32_t steps = 0, seed = 42;
    std::string root, output, dump, corpus, wad;
};
struct Sample {
    double t; uint32_t steps, tokens, frames, version, keys, requestedKeys;
    float loss, x, y, angle; long status;
};

class Engine {
public:
    Options options;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command;
    id<MTLBuffer> control, output, model, snapshot, inferModel, corpus, sync;
    id<MTLBuffer> doomMemory, doomState, doomWad, doomControl, doomOutput, doomOperations;
    DoomHostControl *dh = nullptr;
    DoomLiveOutput *dout = nullptr;
    DoomPersistentConfig doomConfig{};
    uint32_t *c = nullptr, *o = nullptr;
    uint32_t doomSequence = 0, doomTick = 0, doomRenders = 0, doomKeys = 0;
    int32_t doomDiagnostics[16] = {};
    std::vector<uint32_t> doomPixels = std::vector<uint32_t>(WIDTH * HEIGHT);
    std::string wadSHA256, shaderSHA256, hostSourceSHA256, binarySHA256;
    double readyTime = -1;
    std::vector<Sample> samples;
    std::vector<float> losses;
    std::vector<uint32_t> pixels;
    std::string text;
    uint32_t readTokens = 0, keyMask = 0;
    uint32_t liveProgress[3] = {}, versions = 0, lastVersion = 0;
    uint32_t frameCopies = 0, rejectedFrames = 0, skippedTokens = 0;
    uint32_t tokenVersionRegressions = 0, previousTokenVersion = 0;
    std::set<uint32_t> tokenVersions;
    double lastSample = -1, lastPrint = -1, stopTime = -1;
    bool stopSent = false, finalized = false;
    Clock::time_point start;
    std::ofstream log;
    std::ofstream tokenLog;
    std::ofstream doomLog;

    explicit Engine(Options opt) : options(opt), pixels(WIDTH * HEIGHT, 0xff111111) {
        device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) throw std::runtime_error("Requires an Apple Silicon Metal GPU with unified memory.");
        std::string doomRoot = options.root + "/third_party/metal-doom";
        std::string wadBytes = options.mlTest ? "" : readWad(options.wad);
        if (!options.mlTest) wadSHA256 = digest(wadBytes.data(), wadBytes.size());
        std::string src = readFile(doomRoot + "/build/doom-engine.metal") + "\n" +
            readFile(doomRoot + "/app/persistent_protocol.h") + "\n#define DOOM_EMBEDDED 1\n" +
            readFile(doomRoot + "/app/persistent.metal") + "\n" +
            readFile(options.root + "/src/Config.h") + "\n" + readFile(options.root + "/src/Persistent.metal");
        shaderSHA256 = digest(src.data(), src.size());
        std::string hostSource = readFile(options.root + "/src/main.mm");
        hostSourceSHA256 = digest(hostSource.data(), hostSource.size());
        std::string binary = readFile(options.root + "/build/metal-megakernel");
        binarySHA256 = digest(binary.data(), binary.size());
        NSError *error = nil;
        MTLCompileOptions *compile = [MTLCompileOptions new];
        compile.languageVersion = MTLLanguageVersion3_2;
        compile.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:compile error:&error];
        if (!lib) throw std::runtime_error("Metal compiler (requires experimental coherent(system)): " + std::string(error.localizedDescription.UTF8String));
        NSMutableArray<id<MTLFunction>> *visible = [NSMutableArray array];
        for (NSString *name in lib.functionNames) {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (f.functionType == MTLFunctionTypeVisible) [visible addObject:f];
        }
        MTLLinkedFunctions *links = [MTLLinkedFunctions new]; links.functions = visible;
        MTLComputePipelineDescriptor *desc = [MTLComputePipelineDescriptor new];
        desc.computeFunction = [lib newFunctionWithName:@"persistent"];
        desc.linkedFunctions = links; desc.maxCallStackDepth = 32;
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:desc
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!pipeline) throw std::runtime_error(std::string("Metal pipeline: ") + error.localizedDescription.UTF8String);
        if (pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 96)
            throw std::runtime_error("This implementation requires SIMD width32 and a96-thread threadgroup.");
        auto buffer = [&](NSUInteger n, NSString *name) {
            id<MTLBuffer> b = [device newBufferWithLength:n options:MTLResourceStorageModeShared];
            if (!b) throw std::runtime_error("Metal allocation failed");
            b.label = name; memset(b.contents, 0, n); return b;
        };
        control = buffer(C_WORDS * 4, @"CPU owns controls; GPU reads");
        output = buffer(O_WORDS * 4, @"GPU owns ML telemetry and tokens; CPU reads");
        model = buffer(PARAMS * 4, @"GPU training weights");
        snapshot = buffer(PARAMS * 4, @"GPU snapshot mailbox");
        inferModel = buffer(PARAMS * 4, @"GPU inference weights");
        sync = buffer(32 * 4, @"GPU-only snapshot ownership");
        c = (uint32_t *)control.contents; o = (uint32_t *)output.contents;
        std::string initial = options.mlTest ? "" : readFile(doomRoot + "/build/initial-memory.bin");
        if (!options.mlTest && initial.size() != 100663296) throw std::runtime_error("Invalid Doom memory image; rebuild the engine");
        doomMemory = buffer(std::max<size_t>(4, initial.size()), @"Doom linear memory");
        if (!initial.empty()) memcpy(doomMemory.contents, initial.data(), initial.size());
        doomState = buffer(1024 * 1024, @"Doom engine instance");
        doomWad = buffer(std::max<size_t>(4, wadBytes.size()), @"Read-only Doom IWAD");
        if (!wadBytes.empty()) memcpy(doomWad.contents, wadBytes.data(), wadBytes.size());
        doomControl = buffer(sizeof(DoomHostControl), @"CPU owns Doom controls and frame acknowledgements");
        doomOutput = buffer(sizeof(DoomLiveOutput), @"GPU owns Doom snapshots");
        doomOperations = buffer(sizeof(DoomReplayOperation), @"Unused live-mode replay storage");
        dh = (DoomHostControl *)doomControl.contents; dout = (DoomLiveOutput *)doomOutput.contents;
        doomConfig.wadBytes = (uint32_t)wadBytes.size();
        std::string raw = readFile(options.corpus.empty() ? options.root + "/assets/corpus.txt" : options.corpus);
        std::vector<uint32_t> ids;
        for (unsigned char ch : raw) {
            if (ch == '\n') ids.push_back(95);
            else if (ch >= 32 && ch <= 126) ids.push_back(ch - 32);
        }
        if (ids.size() <= CONTEXT) throw std::runtime_error("Corpus needs more than8 printable ASCII characters.");
        uint32_t corpusLength = (uint32_t)ids.size();
        corpus = buffer(ids.size() * 4, @"Read-only character corpus");
        memcpy(corpus.contents, ids.data(), ids.size() * 4);
        float *weights = (float *)model.contents;
        uint32_t rng = 1234567;
        for (int i = 0; i < PARAMS; ++i) {
            float r = (float)(random32(rng) & 0xffffff) / 16777216.f * 2.f - 1.f;
            weights[i] = r * (i < W_HIDDEN ? .5f : .08f);
            if ((i >= W_HBIAS && i < W_OUTPUT) || i >= W_OBIAS) weights[i] = 0;
        }
        memcpy(inferModel.contents, model.contents, PARAMS * 4);
        store(c + C_MAX_STEPS, options.steps); store(c + C_SEED, options.seed);
        if (!options.dump.empty()) {
            writeBinary(options.dump + "-initial.bin", weights, PARAMS * 4);
            writeBinary(options.dump + "-corpus.bin", ids.data(), ids.size() * 4);
        }
        if (!options.output.empty()) {
            std::filesystem::create_directories(options.output);
            log.open(options.output + "/samples.jsonl");
            tokenLog.open(options.output + "/tokens.jsonl");
            if (!log) throw std::runtime_error("Cannot create samples log");
            if (!tokenLog) throw std::runtime_error("Cannot create token log");
            doomLog.open(options.output + "/doom-frames.jsonl");
            if (!doomLog) throw std::runtime_error("Cannot create Doom frame log");
        }
        queue = [device newCommandQueue]; queue.label = @"Only workload queue";
        command = [queue commandBuffer]; command.label = @"ONE persistent dispatch: train + infer + Doom";
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        NSArray<id<MTLBuffer>> *buffers = @[control, output, model, snapshot, inferModel, corpus];
        for (NSUInteger i = 0; i < buffers.count; ++i) [enc setBuffer:buffers[i] offset:0 atIndex:i];
        [enc setBytes:&corpusLength length:4 atIndex:6];
        [enc setBuffer:sync offset:0 atIndex:7];
        NSArray<id<MTLBuffer>> *doomBuffers = @[doomMemory, doomState, doomWad, doomControl, doomOutput];
        for (NSUInteger i = 0; i < doomBuffers.count; ++i) [enc setBuffer:doomBuffers[i] offset:0 atIndex:i+8];
        [enc setBytes:&doomConfig length:sizeof(doomConfig) atIndex:13];
        [enc setBuffer:doomOperations offset:0 atIndex:14];
        // The application's ONLY compute dispatch. No GPU work is launched by poll() or drawing.
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(96, 1, 1)];
        [enc endEncoding];
        printf("device=%s unified=%d simd=32 threads=96 groups=1 dispatches=1 parameters=%d trained=%d corpus=%u\n",
               device.name.UTF8String, device.hasUnifiedMemory, PARAMS, PARAMS - W_HIDDEN, corpusLength);
        printf("Experimental Metal compiler internals: coherent(system), system-scope fence. No portable public API guarantee.\n");
        fflush(stdout);
        start = Clock::now(); [command commit];
    }
    ~Engine() {
        if (command && !done()) { stop(); [command waitUntilCompleted]; }
    }
    double elapsed() const { return std::chrono::duration<double>(Clock::now() - start).count(); }
    bool done() const { return command.status >= MTLCommandBufferStatusCompleted; }
    void stop() {
        if (!stopSent) {
            stopTime = elapsed(); stopSent = true;
            store(c + C_STOP, 1); store(&dh->stop, 1);
        }
    }
    void poll() {
        double t = elapsed();
        // Wall-clock and keyboard input only: this does not select, schedule, or execute GPU tasks.
        store(c + C_TIME_MS, (uint32_t)(t * 1000));
        uint32_t keys = keyMask;
        if (options.scripted) {
            double phase = readyTime < 0 ? -1 : fmod(t - readyTime, 12.);
            keys = phase < 0 ? 0 : phase < 4 ? KEY_FORWARD : phase < 6 ? KEY_RIGHT :
                   phase < 8 ? 64 : phase < 9 ? 128 : phase < 10 ? KEY_BACKWARD : phase < 11 ? KEY_LEFT : 0;
        }
        store(c + C_KEYS, keys);
        store(&dh->timeMS, (uint32_t)(t * 1000)); store(&dh->keys, keys);
        double runSeconds = options.mlTest ? t : readyTime < 0 ? 0 : t - readyTime;
        if (interrupted || (options.seconds > 0 && runSeconds >= options.seconds) ||
            (options.steps && load(o + O_TRAIN_DONE)) || load(&dout->done) ||
            (!options.mlTest && readyTime < 0 && t > 60)) stop();
        uint32_t count = load(o + O_TOKENS);
        if (count - readTokens > TOKEN_CAP) { skippedTokens += count - readTokens - TOKEN_CAP; readTokens = count - TOKEN_CAP; }
        std::vector<std::pair<uint32_t,uint32_t>> batch;
        for (uint32_t index = readTokens; index < count; ++index)
            batch.emplace_back(load(o + O_TOKEN_BASE + index % TOKEN_CAP),
                               load(o + O_TOKEN_VERSION_BASE + index % TOKEN_CAP));
        // Recheck after copying: a suspended UI may have allowed the producer to
        // lap us. Discard any possibly overwritten slot, including the one the
        // producer could currently be writing before it publishes its new head.
        uint32_t after = load(o + O_TOKENS);
        for (auto entry : batch) {
            uint32_t index = readTokens++;
            if (after >= TOKEN_CAP && index <= after - TOKEN_CAP) { ++skippedTokens; continue; }
            uint32_t ch = entry.first, version = entry.second;
            if (ch < VOCAB) text += ch == 95 ? '\n' : (char)(ch + 32);
            tokenVersionRegressions += version < previousTokenVersion;
            previousTokenVersion = version; tokenVersions.insert(version);
            if (tokenLog) tokenLog << "{\"index\":" << index << ",\"token\":" << ch << ",\"version\":" << version << ",\"t\":" << t << "}\n";
        }
        if (tokenLog) tokenLog.flush();
        uint32_t seq = load(&dout->sequence);
        if (seq && seq != doomSequence) {
            // GPU promises this payload stays immutable until ack is stored below.
            for (size_t i = 0; i < doomPixels.size(); ++i) doomPixels[i] = load(dout->pixels + i);
            for (int i = 0; i < 16; ++i) doomDiagnostics[i] = (int32_t)load(dout->diagnostics + i);
            doomSequence = seq; doomTick = load(&dout->tick);
            doomRenders = load(&dout->renders); doomKeys = load(&dout->keys);
            if (readyTime < 0) {
                readyTime = t;
                uint32_t n = std::min(load(&dout->logBytes), 4096u);
                std::string console(n, '\0');
                for (uint32_t i = 0; i < n; ++i) console[i] = char(load(dout->log + i / 4) >> (8 * (i % 4)));
                fprintf(stderr, "Doom ready after %.3fs\n%.*s\n", t, (int)n, console.data());
            }
            for (size_t i = 0; i < pixels.size(); ++i) {
                uint32_t v = doomPixels[i];
                pixels[i] = 0xff000000u | ((v >> 16) & 255) | (v & 0xff00) | ((v & 255) << 16);
            }
            ++frameCopies;
            if (doomLog) {
                doomLog << "{\"sequence\":" << seq << ",\"tick\":" << doomTick << ",\"keys\":" << doomKeys
                    << ",\"renders\":" << doomRenders << ",\"digest\":\"" << digest(doomPixels.data(), doomPixels.size() * 4)
                    << "\",\"diagnostics\":[";
                for (int i = 0; i < 16; ++i) doomLog << (i ? "," : "") << doomDiagnostics[i];
                doomLog << "],\"status\":" << (long)command.status << ",\"t\":" << t << "}\n";
                doomLog.flush();
            }
            store(&dh->ack, seq);
        }
        if (t - lastSample < .1 && !done()) return;
        Sample s{t, load(o + O_STEPS), count, doomRenders, load(o + O_INFER_VERSION),
            doomKeys, keys, asFloat(load(o + O_LOSS)), doomDiagnostics[4] / 65536.f,
            doomDiagnostics[5] / 65536.f, float(uint32_t(doomDiagnostics[7]) * (6.283185307179586 / 4294967296.)), (long)command.status};
        if (s.status == MTLCommandBufferStatusScheduled && !samples.empty()) {
            liveProgress[0] += s.steps > samples.back().steps;
            liveProgress[1] += s.tokens > samples.back().tokens;
            liveProgress[2] += s.frames > samples.back().frames;
        }
        if (s.version != lastVersion) { ++versions; lastVersion = s.version; }
        samples.push_back(s); lastSample = t;
        if (s.steps) losses.push_back(s.loss);
        if (log) {
            log << "{\"t\":" << t << ",\"status\":" << s.status << ",\"steps\":" << s.steps
                << ",\"tokens\":" << s.tokens << ",\"frames\":" << s.frames << ",\"version\":" << s.version
                << ",\"loss\":" << s.loss << ",\"x\":" << s.x << ",\"y\":" << s.y << ",\"angle\":" << s.angle
                << ",\"keys\":" << s.keys << ",\"requestedKeys\":" << keys << "}\n";
            log.flush();
        }
        if (t - lastPrint >= 1 || done()) {
            printf("t=%6.2f status=%ld steps=%u loss=%.4f tokens=%u version=%u frames=%u keys=%u pos=(%.3f,%.3f)\n",
                   t, s.status, s.steps, s.loss, s.tokens, s.version, s.frames, s.keys, s.x, s.y);
            fflush(stdout); lastPrint = t;
        }
    }
    void finish() {
        if (finalized) return;
        stop(); [command waitUntilCompleted]; poll(); finalized = true;
        double stopMs = (elapsed() - stopTime) * 1000;
        if (command.error) fprintf(stderr, "Metal command error: %s\n", command.error.description.UTF8String);
        if (!options.mlTest && (!doomSequence || load(&dout->result) || load(&dout->fault) || load(&dout->done) != 1))
            fprintf(stderr, "Doom failed: publications=%u result=%u fault=%u done=%u\n", doomSequence,
                    load(&dout->result), load(&dout->fault), load(&dout->done));
        if (!options.dump.empty()) writeBinary(options.dump + "-final.bin", model.contents, PARAMS * 4);
        NSDictionary *report = @{
            @"device":device.name, @"os":NSProcessInfo.processInfo.operatingSystemVersionString,
            @"shaderSHA256":[NSString stringWithUTF8String:shaderSHA256.c_str()],
            @"hostSourceSHA256":[NSString stringWithUTF8String:hostSourceSHA256.c_str()],
            @"binarySHA256":[NSString stringWithUTF8String:binarySHA256.c_str()],
            @"game":options.mlTest ? @"none" : @"doom", @"submissions":@1,
            @"doomFrameCount":@(frameCopies), @"doomTick":@(doomTick), @"doomRenders":@(doomRenders),
            @"doomResult":@(load(&dout->result)), @"doomFault":@(load(&dout->fault)), @"doomDone":@(load(&dout->done)),
            @"wadSHA256":[NSString stringWithUTF8String:wadSHA256.c_str()],
            @"startupSeconds":@(readyTime), @"runSeconds":@(readyTime < 0 ? 0 : elapsed() - readyTime),
            @"dispatches":@1, @"threadgroups":@1, @"threads":@96, @"simdGroups":@3,
            @"experimentalSystemCoherence":@YES, @"status":@((long)command.status),
            @"error":command.error ? command.error.description : @"", @"seconds":@(elapsed()), @"stopMilliseconds":@(stopMs),
            @"steps":@(load(o + O_STEPS)), @"tokens":@(load(o + O_TOKENS)), @"frames":@(doomRenders),
            @"firstLoss":@(asFloat(load(o + O_FIRST_LOSS))), @"finalLoss":@(asFloat(load(o + O_LOSS))),
            @"observedVersions":@(versions), @"inferenceVersion":@(load(o + O_INFER_VERSION)),
            @"liveProgressSamples":@[@(liveProgress[0]), @(liveProgress[1]), @(liveProgress[2])],
            @"doneFlags":@[@(load(o + O_TRAIN_DONE)), @(load(o + O_INFER_DONE)), @(load(o + O_GAME_DONE))],
            @"peerProgress":@[@(load(o + O_TRAIN_SEES_TOKENS)), @(load(o + O_TRAIN_SEES_FRAMES)),
                @(load(o + O_INFER_SEES_STEPS)), @(load(o + O_INFER_SEES_FRAMES)),
                @(load(o + O_GAME_SEES_STEPS)), @(load(o + O_GAME_SEES_TOKENS))],
            @"coherentFrameCopies":@(frameCopies), @"rejectedFrameCopies":@(rejectedFrames), @"skippedTokens":@(skippedTokens),
            @"trainingHash":@(load(o + O_TRAIN_HASH)), @"inferenceHash":@(load(o + O_INFER_HASH)),
            @"snapshotErrors":@(load(o + O_SNAPSHOT_ERRORS)), @"snapshotVerifiedCopies":@(load(o + O_SNAPSHOT_COPIES)),
            @"tokenWeightVersions":@(tokenVersions.size()), @"tokenVersionRegressions":@(tokenVersionRegressions),
            @"text":[NSString stringWithUTF8String:text.c_str()] ?: @""
        };
        if (!options.output.empty()) {
            NSError *err = nil;
            NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&err];
            if (!json) throw std::runtime_error(err.localizedDescription.UTF8String);
            [json writeToFile:[NSString stringWithUTF8String:(options.output + "/summary.json").c_str()] atomically:YES];
            writeBinary(options.output + "/frame.rgba", pixels.data(), pixels.size() * 4);
            if (!options.mlTest) writeBinary(options.output + "/doom-frame.bgra", doomPixels.data(), doomPixels.size() * 4);
            std::ofstream out(options.output + "/generated.txt"); out << text;
        }
        printf("Finished: status=%ld dispatches=1 liveProgress=%u,%u,%u versions=%u stop=%.2fms\n",
               (long)command.status, liveProgress[0], liveProgress[1], liveProgress[2], versions, stopMs);
    }
};

static NSColor *rgb(float r, float g, float b) { return [NSColor colorWithCalibratedRed:r green:g blue:b alpha:1]; }
static void label(NSString *s, NSRect r, CGFloat size, NSColor *color, bool mono = false) {
    NSFont *font = mono ? [NSFont monospacedSystemFontOfSize:size weight:NSFontWeightRegular] : [NSFont systemFontOfSize:size weight:NSFontWeightMedium];
    [s drawInRect:r withAttributes:@{NSFontAttributeName:font, NSForegroundColorAttributeName:color}];
}
@interface DemoView : NSView
@property(nonatomic, assign) Engine *engine;
@end
@implementation DemoView
- (BOOL)isFlipped { return YES; }
- (BOOL)acceptsFirstResponder { return YES; }
- (void)drawRect:(NSRect)dirty {
    (void)dirty;
    Engine &e = *_engine;
    [rgb(.045,.055,.075) setFill]; NSRectFill(self.bounds);
    label(@"ONE KERNEL / THREE WORLDS", NSMakeRect(28,22,900,40),26,rgb(.94,.95,.99));
    NSString *status = e.done() ? @"STOPPED" : @"RUNNING";
    label([NSString stringWithFormat:@"%@   •   %@   •   1 dispatch / 3 independent SIMD groups   •   %.1f s", status,e.device.name,e.elapsed()],NSMakeRect(30,64,1000,30),13,rgb(.55,.65,.75),true);
    CGFloat w = self.bounds.size.width, h = self.bounds.size.height;
    CGFloat left = 330, gameW = w - left - 82, gameH = gameW * HEIGHT / WIDTH;
    [rgb(.075,.09,.12) setFill];
    [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(24,110,left,h-150) xRadius:12 yRadius:12] fill];
    [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(left+42,110,gameW+16,h-150) xRadius:12 yRadius:12] fill];
    uint32_t steps=load(e.o+O_STEPS), tokens=load(e.o+O_TOKENS), frames=e.doomRenders;
    label(@"01   TRAIN",NSMakeRect(42,128,280,25),14,rgb(.48,.81,.99),true);
    label([NSString stringWithFormat:@"%.3f",asFloat(load(e.o+O_LOSS))],NSMakeRect(40,162,250,50),38,rgb(.95,.97,1),true);
    label([NSString stringWithFormat:@"loss EMA   •   %u steps",steps],NSMakeRect(42,214,290,25),12,rgb(.6,.69,.78),true);
    if (e.losses.size()>1) {
        NSBezierPath *line=[NSBezierPath bezierPath];
        size_t begin=e.losses.size()>500 ? e.losses.size()-500 : 0;
        for (size_t i=begin;i<e.losses.size();++i) {
            CGFloat x=42+(i-begin)*290./std::max<size_t>(1,e.losses.size()-begin-1);
            CGFloat y=325-std::min(6.f,std::max(0.f,e.losses[i]))/6.f*78;
            if(i==begin)[line moveToPoint:NSMakePoint(x,y)]; else [line lineToPoint:NSMakePoint(x,y)];
        }
        [rgb(.48,.81,.99) setStroke]; line.lineWidth=2; [line stroke];
    }
    label(@"02   GENERATE",NSMakeRect(42,353,290,25),14,rgb(.8,.65,1),true);
    label([NSString stringWithFormat:@"%u tokens • weights @ step %u",tokens,load(e.o+O_INFER_VERSION)],NSMakeRect(42,387,290,34),11,rgb(.6,.69,.78),true);
    std::string recent = e.text.substr(e.text.size()>650 ? e.text.size()-650 : 0);
    label([NSString stringWithUTF8String:recent.c_str()] ?: @"",NSMakeRect(42,425,292,h-485),13,rgb(.87,.87,.94),true);
    label(@"03   PLAY",NSMakeRect(left+60,128,gameW,25),14,rgb(.52,.91,.65),true);
    label([NSString stringWithFormat:@"%u GPU frames • Doom / E1M1",frames],NSMakeRect(left+60,159,gameW,25),12,rgb(.6,.69,.78),true);
    unsigned char *data=(unsigned char *)e.pixels.data();
    NSBitmapImageRep *rep=[[NSBitmapImageRep alloc] initWithBitmapDataPlanes:&data pixelsWide:WIDTH pixelsHigh:HEIGHT bitsPerSample:8 samplesPerPixel:4 hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace bytesPerRow:WIDTH*4 bitsPerPixel:32];
    NSImage *img=[[NSImage alloc] initWithSize:NSMakeSize(WIDTH,HEIGHT)]; [img addRepresentation:rep];
    NSGraphicsContext.currentContext.imageInterpolation=NSImageInterpolationNone;
    [img drawInRect:NSMakeRect(left+50,200,gameW,gameH) fromRect:NSZeroRect operation:NSCompositingOperationCopy fraction:1 respectFlipped:YES hints:nil];
    label(@"W / S move   A / D turn   Q / E strafe\nSpace fire · Enter use · Shift run · 1–7 weapons. Esc stops.",NSMakeRect(left+60,gameH+220,gameW-20,50),13,rgb(.72,.8,.84),true);
    label(@"Character MLP • 28,896 trained weights • GPU-owned model & world",NSMakeRect(28,h-30,w-56,22),11,rgb(.43,.51,.6),true);
}
- (uint32_t)bitFor:(NSEvent *)event {
    switch(event.keyCode) { case 13:case 126:return KEY_FORWARD;case 1:case 125:return KEY_BACKWARD;
        case 0:case 123:return KEY_LEFT;case 2:case 124:return KEY_RIGHT;case 12:return KEY_STRAFE_LEFT;case 14:return KEY_STRAFE_RIGHT;
        case 49:return 64;case 36:return 128;case 6:return 256;case 48:return 131072;case 46:return 262144;default:break; }
    NSString *chars = event.charactersIgnoringModifiers;
    if (chars.length == 1) { unichar ch = [chars characterAtIndex:0]; if (ch >= '1' && ch <= '7') return 1u << (10 + ch - '1'); }
    return 0;
}
- (void)flagsChanged:(NSEvent *)event {
    if (event.modifierFlags & NSEventModifierFlagShift) _engine->keyMask |= 512;
    else _engine->keyMask &= ~512u;
}
- (void)keyDown:(NSEvent *)event { if(event.keyCode==53)_engine->stop();else _engine->keyMask|=[self bitFor:event]; }
- (void)keyUp:(NSEvent *)event { _engine->keyMask&=~[self bitFor:event]; }
@end
@interface DemoDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, assign) Engine *engine;
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) DemoView *view;
@property(nonatomic, strong) NSTimer *timer;
@end
@implementation DemoDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)note {
    (void)note;
    self.window=[[NSWindow alloc] initWithContentRect:NSMakeRect(100,100,1120,800) styleMask:NSWindowStyleMaskTitled|NSWindowStyleMaskClosable|NSWindowStyleMaskMiniaturizable backing:NSBackingStoreBuffered defer:NO];
    self.window.title=@"Metal Megakernel — one kernel, three workloads"; self.window.delegate=self;
    self.view=[[DemoView alloc] initWithFrame:self.window.contentView.bounds];self.view.engine=self.engine;
    self.window.contentView=self.view;[self.window makeFirstResponder:self.view];[self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    self.timer=[NSTimer scheduledTimerWithTimeInterval:1./60 repeats:YES block:^(NSTimer *t){
        self.engine->poll();[self.view setNeedsDisplay:YES];
        if(self.engine->done()) { [t invalidate]; self.engine->finish(); [self.view setNeedsDisplay:YES];
            if(self.engine->options.seconds>0 || self.engine->options.steps>0) [NSApp terminate:nil]; }
    }];
}
- (BOOL)windowShouldClose:(NSWindow *)sender { (void)sender; self.engine->stop();return YES; }
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { (void)sender;return YES; }
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender { (void)sender;self.engine->finish();return NSTerminateNow; }
- (void)applicationDidResignActive:(NSNotification *)n { (void)n;self.engine->keyMask=0; }
@end

int main(int argc, char **argv) { @autoreleasepool { try {
    Options options;
    options.root=std::filesystem::canonical(argv[0]).parent_path().parent_path().string();
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto value=[&](){if(i+1>=argc)throw std::runtime_error("Missing value after "+a);return std::string(argv[++i]);};
        if(a=="--headless")options.headless=true;else if(a=="--ml-test")options.mlTest=true;else if(a=="--wad")options.wad=value();else if(a=="--test-input")options.scripted=true;
        else if(a=="--seconds")options.seconds=std::stod(value());else if(a=="--steps")options.steps=(uint32_t)std::stoul(value());
        else if(a=="--seed")options.seed=(uint32_t)std::stoul(value());else if(a=="--output")options.output=value();
        else if(a=="--dump-prefix")options.dump=value();else if(a=="--corpus")options.corpus=value();
        else if(a=="--help") {printf("Usage: %s [--wad FILE] [--headless] [--seconds N] [--test-input] [--steps N] [--seed N] [--output DIR] [--corpus FILE] [--dump-prefix PATH]\nDefault opens an interactive window and runs ONE dispatch until Esc/close. --seconds 0 means no time limit.\n",argv[0]);return 0;}
        else throw std::runtime_error("Unknown argument: "+a);
    }
    if(options.seconds<0 || !std::isfinite(options.seconds))throw std::runtime_error("seconds must be finite and nonnegative");
    if (options.mlTest && (!options.headless || !options.steps)) throw std::runtime_error("--ml-test requires --headless and --steps N");
    if (options.wad.empty() && getenv("DOOM_WAD")) options.wad = getenv("DOOM_WAD");
    if (!options.mlTest && options.wad.empty()) {
        if (options.headless) throw std::runtime_error("Supply --wad /path/to/doom1.wad or set DOOM_WAD (assets are not bundled)");
        [NSApplication sharedApplication]; [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular]; [NSApp activateIgnoringOtherApps:YES];
        NSOpenPanel *panel = [NSOpenPanel openPanel]; panel.title = @"Select your Doom IWAD";
        panel.message = @"Choose doom1.wad or a compatible IWAD. Game assets are supplied separately.";
        panel.canChooseDirectories = NO; panel.allowsMultipleSelection = NO;
        if ([panel runModal] != NSModalResponseOK) return 0;
        options.wad = panel.URL.path.UTF8String;
    }
    signal(SIGINT,signalStop);signal(SIGTERM,signalStop);
    Engine engine(options);
    if(options.headless) { while(!engine.done()) {engine.poll();std::this_thread::sleep_for(std::chrono::milliseconds(5));}engine.finish(); }
    else { [NSApplication sharedApplication]; [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        DemoDelegate *delegate=[DemoDelegate new];delegate.engine=&engine;NSApp.delegate=delegate;[NSApp run];engine.finish(); }
    return engine.command.status==MTLCommandBufferStatusCompleted &&
        (options.mlTest || (engine.doomSequence > 0 && load(&engine.dout->done) == 1 &&
         !load(&engine.dout->result) && !load(&engine.dout->fault))) ? 0 : 1;
} catch(const std::exception &e) {fprintf(stderr,"Error: %s\n",e.what());return 2;} } }
