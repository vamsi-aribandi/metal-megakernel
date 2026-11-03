#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>

// Experiment: coherent(system) and thread scope 3 are undocumented internals.
// CPU writes only Control; GPU writes only Output. CPU acknowledges a complete
// snapshot after reading, so its payload cannot be overwritten during the read.
struct alignas(128) Control { uint32_t stop, input, ack[3], pad[27]; };
struct alignas(128) Role { uint32_t sequence, seen, peer, reason, polls, pad[27]; uint32_t payload[64]; };
struct Output { Role roles[3]; };
static uint32_t rd(uint32_t *p) { return __atomic_load_n(p,__ATOMIC_SEQ_CST); }
static void wr(uint32_t *p,uint32_t v) { __atomic_store_n(p,v,__ATOMIC_SEQ_CST); }
static uint32_t expected(uint32_t seq,uint32_t input,uint32_t group,uint32_t k) {
    return seq*1664525u+input*1013904223u+group*2654435761u+k*2246822519u;
}
static NSString *shader=@R"MSL(
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;
constant thread_scope system_scope=static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
struct Control {uint stop,input,ack[3],pad[27];};
struct Role {uint sequence,seen,peer,reason,polls,pad[27];uint payload[64];};
struct Output {Role roles[3];};
kernel void probe(volatile coherent(system) device Control *ctrl [[buffer(0)]],
                  volatile coherent(system) device Output *out [[buffer(1)]],
                  constant uint &limit [[buffer(2)]],
                  uint group [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    uint seq=0,polls=0;
    for(;polls<limit;++polls) {
        atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,system_scope);
        uint stop=simd_broadcast_first(lane==0 ? ctrl->stop : 0);
        if(stop!=0) break;
        uint ack=simd_broadcast_first(lane==0 ? ctrl->ack[group] : 0);
        if(ack!=seq) continue;
        uint input=simd_broadcast_first(lane==0 ? ctrl->input : 0);
        ++seq;
        for(uint k=lane;k<64;k+=32)
            out->roles[group].payload[k]=seq*1664525u+input*1013904223u+group*2654435761u+k*2246822519u;
        atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,system_scope);
        simdgroup_barrier(mem_flags::mem_device);
        if(lane==0) {
            out->roles[group].seen=input;
            out->roles[group].peer=out->roles[(group+1)%3].sequence;
            out->roles[group].polls=polls;
            atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,system_scope);
            out->roles[group].sequence=seq;
            atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,system_scope);
        }
    }
    if(lane==0) {
        out->roles[group].reason=polls==limit ? 2 : 1;
        atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,system_scope);
    }
}
)MSL";
int main(int argc,char **argv) { @autoreleasepool {
    double duration=argc>1 ? atof(argv[1]) : 1;
    uint32_t limit=argc>2 ? strtoul(argv[2],nullptr,10) : 2000000;
    if(duration<0.1 || duration>15 || limit==0) return 2;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError *error=nil;if(!dev)return 2;
    MTLCompileOptions *opts=[MTLCompileOptions new];opts.languageVersion=MTLLanguageVersion3_2;
    id<MTLLibrary> lib=[dev newLibraryWithSource:shader options:opts error:&error];
    if(!lib){fprintf(stderr,"compile: %s\n",error.description.UTF8String);return 2;}
    id<MTLComputePipelineState> pipeline=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&error];
    if(!pipeline){fprintf(stderr,"pipeline: %s\n",error.description.UTF8String);return 2;}
    id<MTLBuffer> cb=[dev newBufferWithLength:sizeof(Control) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ob=[dev newBufferWithLength:sizeof(Output) options:MTLResourceStorageModeShared];
    Control *ctrl=(Control*)cb.contents;Output *out=(Output*)ob.contents;
    memset(ctrl,0,sizeof(*ctrl));memset(out,0,sizeof(*out));wr(&ctrl->input,1);
    id<MTLCommandQueue> queue=[dev newCommandQueue];id<MTLCommandBuffer> cmd=[queue commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];[enc setBuffer:cb offset:0 atIndex:0];[enc setBuffer:ob offset:0 atIndex:1];[enc setBytes:&limit length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(96,1,1)];[enc endEncoding];
    auto start=std::chrono::steady_clock::now();[cmd commit];
    uint32_t valid[3]={},changes[3]={},lastInput[3]={},peers[3]={},mismatches=0,samples=0;double lastPrint=-1;bool liveAll=false;
    printf("device=%s dispatches=1 threadgroups=1 threads=96 roles=3 wordsPerSnapshot=64 duration=%.1f limit=%u\n",dev.name.UTF8String,duration,limit);
    while(cmd.status<MTLCommandBufferStatusCompleted) {
        double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        for(uint32_t g=0;g<3;++g) {
            Role *r=&out->roles[g];uint32_t seq=rd(&r->sequence);if(seq==rd(&ctrl->ack[g]))continue;
            uint32_t input=rd(&r->seen);bool okay=true;
            for(uint32_t k=0;k<64;++k)if(rd(&r->payload[k])!=expected(seq,input,g,k)){okay=false;++mismatches;}
            if(okay)++valid[g];if(input!=lastInput[g]&&input>1)++changes[g];lastInput[g]=input;
            if(rd(&r->peer)>0)++peers[g];wr(&ctrl->ack[g],seq);
        }
        ++samples;liveAll|=valid[0]>0&&valid[1]>0&&valid[2]>0&&cmd.status<MTLCommandBufferStatusCompleted;
        if(t>=duration){wr(&ctrl->stop,1);break;}wr(&ctrl->input,2+(uint32_t)(t*20));
        if(t-lastPrint>=0.5){printf("t=%.3f status=%ld valid=%u,%u,%u inputChanges=%u,%u,%u peers=%u,%u,%u mismatches=%u\n",t,(long)cmd.status,valid[0],valid[1],valid[2],changes[0],changes[1],changes[2],peers[0],peers[1],peers[2],mismatches);fflush(stdout);lastPrint=t;}
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto stopped=std::chrono::steady_clock::now();[cmd waitUntilCompleted];
    double stopMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-stopped).count();
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    bool pass=cmd.status==MTLCommandBufferStatusCompleted&&elapsed>=duration&&liveAll&&mismatches==0;
    for(int g=0;g<3;++g)pass&=valid[g]>10&&changes[g]>5&&peers[g]>10&&out->roles[g].reason==1;
    printf("%s elapsed=%.3f stopMs=%.3f status=%ld samples=%u valid=%u,%u,%u inputChanges=%u,%u,%u peers=%u,%u,%u mismatches=%u reasons=%u,%u,%u error=%s\n",pass?"PASS":"FAIL",elapsed,stopMs,(long)cmd.status,samples,valid[0],valid[1],valid[2],changes[0],changes[1],changes[2],peers[0],peers[1],peers[2],mismatches,out->roles[0].reason,out->roles[1].reason,out->roles[2].reason,cmd.error.description.UTF8String);
    return pass?0:1;
}}
