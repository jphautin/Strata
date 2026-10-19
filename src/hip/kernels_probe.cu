// src/hip/kernels_probe.cu - strata_hip_kernels: the library with the AMD half of a mixed rig, built by the
// HIP build (ROCm's clang) and opened at run time by the CUDA engine (src/core/hip_kernels_loader.cpp).  This
// is B.1's join: CMake 4.3's Windows-Clang-HIP module refuses one configure that holds cl (the engine's MSVC
// host) and clang (HIP's) at once, so the two halves are SEPARATE CMake builds and the only thing between
// them is this `extern "C"` entry table - the shape src/prefill/mmq_kq_lib.cu already ships.  Neither toolchain
// has to see the other's objects, headers or name mangling.
//
// M1 of Track B is the enumeration, because the blocker was narrow: one runtime lists only one brand
// (src/core/device.cu asks cudaGetDeviceCount), while the layer split itself is brand-agnostic - its hand-off
// goes through pinned host RAM and needs no NVLink and no peer-to-peer access (docs/MULTI_GPU.md:10-12).
// The kernel routing for the AMD layers adds entry points to this same table.
#include "strata/core/hip_kernels_abi.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>

#if defined(_WIN32)
#define STRATA_HIP_KERNELS_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_HIP_KERNELS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
// "gfx1201:sramecc-:xnack-" -> "gfx1201": the same rule src/core/device.cu applies to gcnArchName.
void copy_base_arch(char* out, size_t cap, const char* gcn) {
    size_t i = 0;
    for (; gcn[i] && i + 1 < cap; ++i) {
        if (gcn[i] == ':') break;
        out[i] = gcn[i];
    }
    out[i] = '\0';
}
}  // namespace

// the interface version and the probe block's size: both sides compile the same header, so a drift here
// stops the library instead of reading garbage across the boundary
STRATA_HIP_KERNELS_EXPORT int strata_hip_kernels_abi() {
    return STRATA_HIP_KERNELS_ABI_VERSION * 100000 + (int) sizeof(StrataHipDevices);
}

// Starts this library's HIP runtime on `device` in the engine's context: hipSetDevice plus a first touch, so
// the code objects for this library load now, while the VRAM for them is still free - the same reason strata_kq
// opens before the expert cache is sized.  0 = ready, else the hipError_t.
STRATA_HIP_KERNELS_EXPORT int strata_hip_kernels_init(int device) {
    hipError_t e = hipSetDevice(device);
    if (e == hipSuccess) e = hipFree(nullptr);
    return (int) e;
}

// Lists every device THIS library's HIP runtime sees, into the engine's StrataHipDevices.  The CUDA engine
// cannot ask HIP for this: amdhip64 hands back an empty list once CUDA owns the process, which is the whole
// reason a mix of NVIDIA and AMD cards is "not supported" (docs/MULTI_GPU.md:53).  Returns the count written.
STRATA_HIP_KERNELS_EXPORT int strata_hip_kernels_probe(void* out) {
    StrataHipDevices* o = (StrataHipDevices*) out;
    o->n = 0;
#if defined(STRATA_HIP_ARCHS)
    std::snprintf(o->archs, sizeof(o->archs), "%s", STRATA_HIP_ARCHS);
#else
    o->archs[0] = '\0';
#endif
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) {  // HIP reports an error, not 0, with no usable device
        hipGetLastError();
        return 0;
    }
    const int cap = (int) (sizeof(o->d) / sizeof(o->d[0]));
    if (count > cap) count = cap;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t p{};
        if (hipSetDevice(i) != hipSuccess || hipGetDeviceProperties(&p, i) != hipSuccess) {
            hipGetLastError();
            continue;
        }
        StrataHipDevices::Dev& d = o->d[o->n];
        std::snprintf(d.name, sizeof(d.name), "%s", p.name);
        copy_base_arch(d.arch, sizeof(d.arch), p.gcnArchName);
        d.wave = p.warpSize;
        d.total_bytes = d.free_bytes = 0;
        if (hipMemGetInfo(&d.free_bytes, &d.total_bytes) != hipSuccess) {
            hipGetLastError();
            d.free_bytes = d.total_bytes = 0;
        }
        ++o->n;
    }
    return o->n;
}
