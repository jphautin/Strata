// include/strata/core/hip_engine_abi.hpp - the interface between the CUDA engine and the HIP-built ENGINE
// stage code: strata_hip_engine, the DLL a multi-brand HIP build makes from the same objects as its own
// strata_engine (see CMakeLists.txt), whose entries live in src/hip/engine_bridge.cpp.  This is B.1 at its
// widest.  The narrow boundary (hip_kernels_abi.hpp) carries one AMD kernel; this one carries the STAGE,
// because the AMD side must keep the kernels ROCm's clang built for its own arch (gfx1201 / gfx1200, wave32,
// its LDS), and the tree has 482 __global__ kernels: a table with one entry per kernel is not something
// anyone maintains, while the engine's stage code is 18 translation units this build compiles already.
//
// Bump the version when an entry's meaning changes.  Every block is plain data both sides compile from this
// header, and the ABI check carries its size, so a drift here stops the library rather than reading garbage
// across the boundary (the same rule strata_hip_kernels_abi.hpp and mmq_kq_abi.hpp use).
#pragma once

#include <string>

#define STRATA_HIP_ENGINE_ABI_VERSION 1

#if defined(_WIN32)
#define STRATA_HIP_ENGINE_LIB_NAME "strata_hip_engine.dll"
#else
#define STRATA_HIP_ENGINE_LIB_NAME "libstrata_hip_engine.so"
#endif

// What the HIP-built engine library says about itself BEFORE any stage runs: the arches its device code was
// built for, the AMD cards it sees from inside its own HIP runtime, and proof that the stage entry - and the
// kernels it calls - are actually in this library.  That last field is the measurement of the shape: the
// stage's own calls into strata_kernels pull the kernel objects in at link time, so an absent kernel fails
// the link rather than running half a layer at token 4000.
struct StrataEngineProbe {
    char archs[64] = "";                  // STRATA_HIP_ARCHS this library was compiled for ("gfx1201,gfx1200")
    int hip_devices = 0;                  // AMD cards HIP enumerates from inside this library
    int stage_linked = 0;                 // 1 = the stage entry below resolved inside this library
    unsigned long long stage_entry = 0;   // its address here (0 = not linked); the CUDA engine never dereferences it
};

extern "C" {
typedef long long (*strata_engine_abi_fn)();
typedef int (*strata_engine_probe_fn)(void* out);  // out IS a StrataEngineProbe*
}
