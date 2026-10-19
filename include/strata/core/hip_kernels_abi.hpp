// include/strata/core/hip_kernels_abi.hpp - the interface between the engine and strata_hip_kernels
// (src/hip/kernels_probe.cu), the library with the AMD kernels.  This is B.1's join (Track B: one NVIDIA and
// one AMD card in ONE process): the CUDA engine cannot compile AMD device code, and the HIP engine (ROCm's
// clang) cannot compile CUDA, so the AMD half arrives as a library built by the HIP build and opened at run
// time by the CUDA engine.  Same shape as strata_kq (include/strata/prefill/mmq_kq_abi.hpp).
//
// Bump the version when a function's meaning changes.  The probe block is plain data both sides compile from
// this header, and the ABI check carries its size, so a drift here stops the library rather than reading
// garbage across the boundary.
#pragma once

#include <string>

#define STRATA_HIP_KERNELS_ABI_VERSION 1

#if defined(_WIN32)
#define STRATA_HIP_KERNELS_LIB_NAME "strata_hip_kernels.dll"
#else
#define STRATA_HIP_KERNELS_LIB_NAME "libstrata_hip_kernels.so"
#endif

// One HIP device as the library reports it.  Fixed-size plain data: cl, clang and nvcc give this the same
// layout (char arrays, then the 64-bit sizes, then the ints), and the size is checked at load.
struct StrataHipDevices {
    int n = 0;                                    // devices the library filled in
    char archs[64] = "";                          // arches this library has code for ("gfx1201,gfx1200")
    struct Dev {
        char name[64] = "";
        char arch[16] = "";                       // gcnArchName without its feature suffix ("gfx1201")
        long long wave = 0;                       // 32 for the cards Strata's HIP kernels need
        unsigned long long total_bytes = 0, free_bytes = 0;
    } d[8];
};

// One AMD kernel run by this library, plus the card's free VRAM before and after.  This is the measurement
// that gates the NEXT step (the AMD layers' real kernels): an AMD code object has to load and run in a
// process whose CUDA runtime is already up, and its card's memory has to stay this runtime's own.
struct StrataHipTouch {
    unsigned long long free_before = 0, free_after = 0;
    int kernel_ok = 0;        // 1 = the launch, the sync and the read-back all returned
    int sample_is_nan = 0;    // 1 = the first float read back IS a NaN, so the kernel wrote it
};

extern "C" {
typedef int (*strata_hip_kernels_abi_fn)();
typedef int (*strata_hip_kernels_init_fn)(int device);
typedef int (*strata_hip_kernels_probe_fn)(void* out);  // out IS a StrataHipDevices*
typedef int (*strata_hip_kernels_touch_fn)(int device, void* out);  // out IS a StrataHipTouch*
}

// The device id of a mixed rig ("cuda:0", "hip:1", or a plain ordinal) lives with the device facts in
// include/strata/core/device.hpp, not here: this header is the library boundary only, and it must stay
// includable by a build that has no engine device code.
