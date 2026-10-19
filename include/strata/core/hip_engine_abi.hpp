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

// One stage block the library held on the AMD card, as the library reports it.  Plain data again: the CUDA
// engine must not call HIP, so it gets numbers back rather a handle.  The free-VRAM pair is this AMD card's,
// read from INSIDE the library, so a change there cannot be confused with the CUDA card's own.
struct StrataStageRun {
    long long bytes = 0;                     // what the caller asked the library to hold
    long long held_bytes = 0;                // what the engine's arena actually took (0 = it refused)
    int device = -1;                         // the AMD card it lives on
    int arena_ok = 0;                        // 1 = the engine's arena was built inside the library
    int read_back_ok = 0;                     // 1 = the 16 floats the CALLER owns came back NaN
    unsigned long long free_before = 0;      // this card's free VRAM before the hold
    unsigned long long free_after = 0;       // and after it, so "held" cannot mean "nothing allocated"
    unsigned long long total_bytes = 0;      // the card's total, for the print
};

extern "C" {
typedef long long (*strata_engine_abi_fn)();
typedef int (*strata_engine_probe_fn)(void* out);  // out IS a StrataEngineProbe*

// The smallest real STAGE the library can run, and its measurement: the engine's OWN arena (src/core/device.cu:
// it allocates and its poison_kernel writes NaNs) lives on the AMD card, INSIDE this library, in a process whose
// CUDA runtime is already up.  Then 16 floats go back into a host buffer the CALLER owns: the layer split's
// hand-off moves activations through host RAM, so this is the shape a layer needs, without a model or weights.
//
// expect_size is the caller's own sizeof(StrataStageRun): the library refuses rather than writing a block whose
// size the two sides do not agree on (the same rule as hip_kernels_abi.hpp, checked where it matters).
typedef int (*strata_hip_stage_hold_fn)(long long bytes, int device, void* out, int expect_size,
                                        float* host_dst);  // out IS a StrataStageRun*; host_dst holds 16 floats
typedef void (*strata_hip_stage_free_fn)();
}

namespace strata::core {
// The whole measurement, from the engine's side (src/core/hip_kernels_loader.cpp): open the library, hold a
// block on the AMD card through it, read the NaNs back into this process's own RAM, print what was measured,
// then let go.  true = the engine's arena, the engine's kernel, this card's memory and the read-back all
// worked in one process that already owns CUDA.  Declared HERE (not in device.hpp) because only a multi-brand
// build can answer it, and this header is what both sides of the join already share.
bool stage_hold_gate(long long bytes, int device);
}  // namespace strata::core
