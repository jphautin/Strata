// src/core/hip_kernels_loader.cpp - opens strata_hip_kernels (src/hip/kernels_probe.cu), the library with the
// AMD kernels, from the engine's own folder, and joins its device list to this process's own.  This is B.1's
// join (Track B: one NVIDIA and one AMD card in ONE process at native speed).  It is a join and not a link:
// CMake 4.3's Windows-Clang-HIP module raises "mixes Clang and MSVC ... not supported" when HIP (clang) and
// the engine's MSVC host share one configure, and clang-as-host breaks the engine's MSVC-only /wd flags - so
// the CUDA build and the HIP build are SEPARATE CMake builds, and the only thing between them is this loader
// and the library's `extern "C"` table.  Same shape as src/prefill/mmq_kq_loader.cpp: LoadLibraryA / dlopen,
// symbols by name, an ABI check, and "no library = keep the path this build already had".
#include "strata/core/device.hpp"
#include "strata/core/hip_kernels_abi.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

std::string own_folder() {
#if defined(_WIN32)
    char p[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s(p, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#else
    char p[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", p, sizeof(p) - 1);
    std::string s(p, n > 0 ? (size_t) n : 0);
    const size_t k = s.find_last_of('/');
    return k == std::string::npos ? std::string() : s.substr(0, k + 1);
#endif
}

void* sym(void* h, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress((HMODULE) h, name));
#else
    return dlsym(h, name);
#endif
}

struct Hip {
    bool ok = false;
    int n = 0;
    StrataHipDevices devices{};
    std::string path;
};

Hip open_hip_lib() {
    Hip L;
#if defined(STRATA_USE_HIP)
    // This build IS the AMD side: its own runtime already enumerated every AMD card, so the library is not
    // opened here - a second HIP runtime in one process buys nothing.  strata_hip_kernels is for the CUDA
    // engine build, which is the one that cannot enumerate AMD cards itself.
    std::fprintf(stderr, "strata: this engine was built for AMD cards (HIP): the AMD brand is native here; "
                         "strata_hip_kernels belongs to the CUDA engine build\n");
    return L;
#else
    L.path = own_folder() + STRATA_HIP_KERNELS_LIB_NAME;
#if defined(_WIN32)
    void* h = (void*) LoadLibraryA(L.path.c_str());
#else
    void* h = dlopen(L.path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (h == nullptr) {
        std::fprintf(stderr, "strata: %s is not next to the engine: this process sees its own brand only\n",
                     STRATA_HIP_KERNELS_LIB_NAME);
        return L;
    }
    const auto abi = (strata_hip_kernels_abi_fn) sym(h, "strata_hip_kernels_abi");
    const auto init = (strata_hip_kernels_init_fn) sym(h, "strata_hip_kernels_init");
    const auto probe = (strata_hip_kernels_probe_fn) sym(h, "strata_hip_kernels_probe");
    const int a = abi ? abi() : 0;
    if (init == nullptr || probe == nullptr || a / 100000 != STRATA_HIP_KERNELS_ABI_VERSION ||
        a % 100000 != (int) sizeof(StrataHipDevices)) {
        std::fprintf(stderr, "strata: %s is from another engine version, or its probe block is another size: "
                             "this process sees its own brand only\n",
                     L.path.c_str());
        return L;
    }
    L.n = probe(&L.devices);
    if (L.n <= 0) {
        std::fprintf(stderr, "strata: %s sees no AMD device: this process keeps its own brand only\n",
                     L.path.c_str());
        return L;
    }
    // Start HIP on every card it lists, while the VRAM for this library's kernels is still free - the order
    // strata_kq's loader uses, so a card that will not start is out of the list rather than a late crash.
    int usable = 0;
    for (int d = 0; d < L.n; ++d) {
        const int e = init(d);
        if (e != 0) {
            std::fprintf(stderr, "strata: %s did not start on AMD device %d (HIP error %d): that card is not "
                                 "usable by this engine\n",
                         L.path.c_str(), d, e);
            continue;
        }
        if (usable != d) L.devices.d[usable] = L.devices.d[d];
        ++usable;
    }
    L.n = usable;
    if (usable == 0) return L;
    L.ok = true;
    std::fprintf(stderr, "strata: AMD kernels from %s (STRATA_MULTI_BRAND=1): %d AMD device(s), code for %s\n",
                 L.path.c_str(), L.n, L.devices.archs);
    return L;
#endif
}

const Hip& hip_lib() {
    static const Hip L = open_hip_lib();
    return L;
}

}  // namespace

bool hip_kernels_ready() { return hip_lib().ok; }

// ONE process, both brands: this build's own runtime first (upstream's numbering), then the AMD cards the
// library listed from INSIDE its own runtime.  `device_info` tags each of its own cards with the runtime it
// actually is, so the list says which brand each ordinal belongs to.
std::vector<DeviceInfo> union_device_list() {
    std::vector<DeviceInfo> out;
    const int own = device_count();
    for (int d = 0; d < own; ++d) {
        try {
            DeviceInfo i = device_info(d);
            out.push_back(i);
        } catch (const CudaError& e) {
            std::fprintf(stderr, "strata: this engine cannot run GPU %d (%s)\n", d, e.what());
        }
    }
    const Hip& L = hip_lib();
    for (int d = 0; d < L.n; ++d) {
        const StrataHipDevices::Dev& h = L.devices.d[d];
        DeviceInfo i;
        i.ordinal = d;
        i.runtime = DeviceRuntime::HIP;
        i.name = h.name;
        i.arch = h.arch;
        i.total_bytes = h.total_bytes;
        i.free_bytes = h.free_bytes;
        out.push_back(i);
    }
    return out;
}

}  // namespace strata::core
