// src/hip/engine_bridge.cpp - the extern "C" entries of the HIP-built engine stage code.  Built by the HIP
// build ONLY as part of strata_engine SHARED (see CMakeLists.txt, `if(STRATA_MULTI_BRAND AND STRATA_ENABLE_HIP)`):
// the CUDA engine opens that library and drives the AMD stages through these entries, so the AMD card runs
// the kernels ROCm's clang compiled for its own arch - native speed, which is Track B's point.  Neither
// toolchain sees the other's objects, headers or name mangling: only this table crosses.
//
// Why one library for the whole stage and not one entry per kernel: the engine's stage code (src/core/
// layer.cpp, src/core/verify.cpp, src/core/native_head.cpp, src/core/expert_cache.cpp, ...) is 18 translation
// units and every one of them is a DIRECT member of this target, so all of them are in the library, and the
// calls they make into strata_kernels drag the kernel objects in at link time.  A stage whose kernel was
// missing therefore fails to link - it cannot run half a layer and answer wrong from token 4000.
//
// These entries are the smallest honest version, and they measure the SHAPE before any layer runs: what this
// library is, which AMD cards it sees from inside its own HIP runtime, and that the stage entry (and the
// kernels behind it) really is in the library.  Routing real layers through it is the next step.
#include "strata/core/hip_engine_abi.hpp"
#include "strata/core/device.hpp"
#include "strata/core/layer.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>

#if defined(_WIN32)
#define STRATA_HIP_ENGINE_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_HIP_ENGINE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace strata::core {

// The engine's stage entries the CUDA engine will call: src/core/layer.cpp's MoE layer and its GDN layer,
// both declared in include/strata/core/layer.hpp (so they are outside the anonymous namespaces inside that
// .cpp, and another translation unit can name them), both taking the GPU stream as a plain `void*` and
// answering through a std::string out-param - the shape a cross-toolchain table can carry.  Taking their
// addresses is a LINK-TIME request: this library holds layer.cpp as a member, and layer.cpp's own calls into
// strata_kernels (and strata_prefill's GEMMs) pull those objects in.  Absent code here means an unresolved
// symbol at link, not a silent gap at run.
static void* const stage_entry = (void*) (std::uintptr_t) &moe_layer;
static void* const head_entry = (void*) (std::uintptr_t) &gdn_layer;

}  // namespace strata::core

namespace {

// The block this library is holding on an AMD card, if any.  It is held here - not on the caller's side - so
// the CUDA engine can look at the card's free memory while the block is alive, and then ask for it back.  The
// arena's own destructor is what gives the memory to the card again.
std::unique_ptr<strata::core::DeviceArena> g_arena;

void fill_engine_probe(void* out) {
    StrataEngineProbe* p = (StrataEngineProbe*) out;
    std::memset(p, 0, sizeof(StrataEngineProbe));
    const char* a = strata::core::compiled_gpu_archs();
    std::snprintf(p->archs, sizeof(p->archs), "%s", a ? a : "");
    p->hip_devices = strata::core::device_count();  // HIP's own count, from INSIDE this library
    p->stage_entry = (unsigned long long) (std::uintptr_t) strata::core::stage_entry;
    p->stage_linked = (p->stage_entry != 0 && strata::core::head_entry != nullptr) ? 1 : 0;
}

}  // namespace

STRATA_HIP_ENGINE_EXPORT long long strata_engine_abi() {
    // The version AND the size of the only block, so a header drift on either side stops the library.
    return (long long) STRATA_HIP_ENGINE_ABI_VERSION * 1000000 + (long long) sizeof(StrataEngineProbe);
}

STRATA_HIP_ENGINE_EXPORT int strata_engine_probe(void* out) {
    if (out == nullptr) return -1;
    fill_engine_probe(out);
    return 0;
}

// Hold ONE block on an AMD card through the ENGINE's own arena, from inside this library, and give the 16
// floats back into a buffer the CALLER owns.  Nothing here is a stand-in: DeviceArena is src/core/device.cu
// (it allocates, and with poison=true its poison_kernel writes NaNs), so this run exercises the engine's
// allocator, the engine's kernel, this card's memory, and a device-to-host copy - which is what a layer's
// hand-off needs, before any model or weight is involved.
//
// Return: 0 = held and measured. -1 = null argument or the two sides disagree on the block size. -2 = the
// engine's arena refused (its own CudaError). Any other negative = the HIP error code of the step that
// failed, so the caller prints which step died.
STRATA_HIP_ENGINE_EXPORT int strata_hip_stage_hold(long long bytes, int device, void* out, int expect_size,
                                                   float* host_dst) {
    StrataStageRun* o = (StrataStageRun*) out;
    if (o == nullptr || host_dst == nullptr || (int) sizeof(StrataStageRun) != expect_size) return -1;
    std::memset(o, 0, sizeof(StrataStageRun));
    o->bytes = bytes;
    o->device = device;

    hipError_t e = hipSetDevice(device);
    if (e != hipSuccess) return -(int) e;
    size_t total = 0;
    if (hipMemGetInfo(&o->free_before, &total) == hipSuccess) o->total_bytes = total;

    try {
        g_arena = std::make_unique<strata::core::DeviceArena>((uint64_t) bytes, device, /*poison=*/true);
        o->arena_ok = 1;
        o->held_bytes = g_arena->capacity();
    } catch (const strata::core::CudaError&) {
        return -2;
    } catch (const std::exception&) {
        return -2;
    }
    hipMemGetInfo(&o->free_after, &total);

    if (hipMemcpy(host_dst, g_arena->base(), 16 * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        return -3;
    int n_nan = 0;
    for (int i = 0; i < 16; ++i) {
        if (host_dst[i] != host_dst[i]) ++n_nan;  // a NaN is the only value that is not equal to itself
    }
    o->read_back_ok = (n_nan == 16) ? 1 : 0;
    return 0;
}

// Let go: the arena's own destructor gives the card's memory back. The CUDA side does this once per run, so
// what is measured is "held while CUDA holds its own", not "two pools that never end".
STRATA_HIP_ENGINE_EXPORT void strata_hip_stage_free() { g_arena.reset(); }
