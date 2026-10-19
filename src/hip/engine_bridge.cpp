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

#include <cstdint>
#include <cstdio>
#include <cstring>

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
