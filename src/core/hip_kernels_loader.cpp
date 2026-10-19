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
#include "strata/core/hip_engine_abi.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
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
    void* h = nullptr;                                // the open library, for the entry points used later
    strata_hip_kernels_touch_fn touch = nullptr;      // one AMD kernel + this card's free VRAM, measured
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
    const auto touch = (strata_hip_kernels_touch_fn) sym(h, "strata_hip_kernels_touch");
    const int a = abi ? abi() : 0;
    if (init == nullptr || probe == nullptr || touch == nullptr ||
        a / 1000000 != STRATA_HIP_KERNELS_ABI_VERSION ||
        (a / 100) % 10000 != (int) sizeof(StrataHipDevices) || a % 100 != (int) sizeof(StrataHipTouch)) {
        std::fprintf(stderr, "strata: %s is from another engine version, or its argument blocks are another "
                             "size: this process sees its own brand only\n",
                     L.path.c_str());
        return L;
    }
    L.h = h;
    L.touch = touch;
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

// The HIP-built ENGINE library (strata_hip_engine: the DLL a multi-brand HIP build makes from the same objects
// as its own strata_engine; its entries live in src/hip/engine_bridge.cpp).  This is B.1 at its widest: the
// AMD side keeps the engine's STAGE code and the kernels ROCm's clang built for its own arch, so what crosses
// the boundary is a stage and not one of the tree's 482 kernels.  Opening it is separate from opening
// strata_hip_kernels on purpose: the small library gates the runtime join (M1/M2), this one carries the
// layers, and the two are measured apart.
struct Eng {
    bool ok = false;
    StrataEngineProbe p{};
    std::string path;
    void* h = nullptr;
    strata_engine_probe_fn probe = nullptr;
};

Eng open_engine_lib() {
    Eng L;
#if defined(STRATA_USE_HIP)
    // This build IS the AMD side: the stage code is compiled into this executable already, so there is nothing
    // to open.  Only the CUDA engine build needs the library.
    std::fprintf(stderr, "strata: this engine was built for AMD cards (HIP): the stage code is native here; "
                         "strata_engine belongs to the CUDA engine build\n");
    return L;
#else
    L.path = own_folder() + STRATA_HIP_ENGINE_LIB_NAME;
#if defined(_WIN32)
    void* h = (void*) LoadLibraryA(L.path.c_str());
#else
    void* h = dlopen(L.path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (h == nullptr) {
        std::fprintf(stderr, "strata: %s is not next to the engine: no AMD stage code in this process\n",
                     STRATA_HIP_ENGINE_LIB_NAME);
        return L;
    }
    const auto abi = (strata_engine_abi_fn) sym(h, "strata_engine_abi");
    const auto probe = (strata_engine_probe_fn) sym(h, "strata_engine_probe");
    const long long a = abi ? abi() : 0;
    if (probe == nullptr || a / 1000000 != STRATA_HIP_ENGINE_ABI_VERSION ||
        a % 1000000 != (long long) sizeof(StrataEngineProbe)) {
        std::fprintf(stderr, "strata: %s is from another engine version, or its stage block is another "
                             "size: no AMD stage code in this process\n",
                     L.path.c_str());
        return L;
    }
    // Ask the library what it is BEFORE any stage runs: which arches its device code holds, how many AMD cards
    // its own HIP runtime sees, and that the stage entry (and the kernels behind it) really linked in.
    if (probe(&L.p) != 0) {
        std::fprintf(stderr, "strata: %s refused its stage probe\n", L.path.c_str());
        return L;
    }
    if (L.p.stage_linked != 1) {
        std::fprintf(stderr, "strata: %s has no stage entry: it cannot run a layer in this process\n",
                     L.path.c_str());
        return L;
    }
    L.h = h;
    L.probe = probe;
    L.ok = true;
    std::fprintf(stderr, "strata: AMD stage code from %s (STRATA_MULTI_BRAND=1): code for %s, %d AMD device(s) "
                         "seen from inside its own HIP runtime\n",
                 L.path.c_str(), L.p.archs, L.p.hip_devices);
    return L;
#endif
}

const Eng& engine_lib() {
    static const Eng L = open_engine_lib();
    return L;
}

}  // namespace

bool hip_kernels_ready() { return hip_lib().ok; }
bool engine_stage_ready() { return engine_lib().ok; }

// One line saying what the HIP-built engine library is: its path, the arches its device code holds, and how
// many AMD cards its OWN HIP runtime sees.  Empty when there is no such library in this process (the normal
// case for a single-brand build or an engine with no library next to it).
const std::string& engine_stage_detail() {
    static const std::string s = [] {
        const Eng& L = engine_lib();
        if (!L.ok) return std::string();
        return L.path + ": code for " + L.p.archs + ", " + std::to_string(L.p.hip_devices) +
               " AMD device(s) seen from inside its own HIP runtime";
    }();
    return s;
}

// The card order the user chose at setup (setup.py asks which card is the MAIN one and keeps the answer in the
// run config: "gpu" with the main card first, plus "gpu_order": "as_given").  The launcher passes that answer
// down in the same style as its other settings (serve/server.py's child_env sets CUDA_VISIBLE_DEVICES /
// HIP_VISIBLE_DEVICES and STRATA_PRIMARY_DEVICE as environment variables): STRATA_GPU_ORDER="hip:0,cuda:0",
// brand-tagged ids as parse_device_id reads them, or plain ordinals for this build's own runtime.  The MAIN
// card is the one a stage goes on FIRST; then the second card; then the RAM tier for the experts.  This does
// NOT rank the cards: no unit is shared by an NVIDIA card and an RDNA card, so "more powerful" is never decided
// here - only the order the user already named is applied.  An order that does not name every card this process
// has is not applied at all, and the reason is printed, so a typo cannot silently put the layers on another
// card than the one you chose.
static bool order_applied = false;
static std::string order_text;

static void apply_chosen_order(std::vector<DeviceInfo>& cards) {
    order_applied = false;
    order_text.clear();
    if (cards.size() < 2) return;                  // one card: there is nothing to order
    const char* raw = std::getenv("STRATA_GPU_ORDER");
    if (raw == nullptr) return;                    // nobody chose: the runtime's own order stands
    std::vector<std::string> toks;
    std::string tok;
    for (const char c : std::string(raw)) {
        if (c == ',' || c == ';' || c == ' ') {
            if (!tok.empty()) toks.push_back(tok);
            tok.clear();
        } else if (c != '\0') {
            tok.push_back(c);
        }
    }
    if (!tok.empty()) toks.push_back(tok);
    if (toks.empty()) return;                      // an empty value is the same as none set

    const DeviceRuntime own_rt = cards[0].runtime;  // this build's own runtime, which lists its cards first
    std::vector<DeviceInfo> rest = cards;
    std::vector<DeviceInfo> head;
    std::string why;
    for (const std::string& t : toks) {
        DeviceRuntime r = DeviceRuntime::OWN;
        int ord = -1;
        if (!parse_device_id(t, r, ord)) {
            why = t + " is not a card id (use cuda:N, hip:N, or a plain number for this engine's own runtime)";
            break;
        }
        const DeviceRuntime want = r == DeviceRuntime::OWN ? own_rt : r;
        size_t hit = rest.size();
        for (size_t k = 0; k < rest.size(); ++k) {
            if (rest[k].ordinal == ord && rest[k].runtime == want) {
                hit = k;
                break;
            }
        }
        if (hit == rest.size()) {
            why = t + " names a card this process does not have";
            break;
        }
        head.push_back(rest[hit]);
        rest.erase(rest.begin() + hit);
    }
    if (why.empty() && head.size() != cards.size())
        why = "the list does not name every card this process has";
    if (!why.empty()) {
        std::fprintf(stderr, "strata: STRATA_GPU_ORDER \"%s\": %s. The card order is NOT applied: the layers "
                             "stay in the order this process's runtime lists them in\n",
                     raw, why.c_str());
        return;
    }
    cards = head;
    order_applied = true;
    order_text = raw;
    std::fprintf(stderr, "strata: card order from STRATA_GPU_ORDER \"%s\": main card first, then the second "
                         "card\n",
                 raw);
}

bool card_order_chosen() { return order_applied; }
const std::string& card_order_text() { return order_text; }

// BOTH cards in ONE process, measured.  This gates the NEXT step (the AMD layers' real kernels route through
// this library): a code object that loads but cannot run, or an AMD card whose memory HIP can claim while
// CUDA still holds its own, would make that step pointless.  Prints what it measured; true = the AMD kernel
// ran and wrote the value it read back.
bool cross_runtime_probe() {
    const Hip& L = hip_lib();
    if (!L.ok) {
        std::fprintf(stderr, "strata: the cross-runtime probe needs strata_hip_kernels next to the engine\n");
        return false;
    }
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    auto own_free = [GiB](int ordinal) {
        try {
            const DeviceInfo i = device_info(ordinal);
            return (double) i.free_bytes / GiB;
        } catch (const CudaError&) {
            return -1.0;  // this build cannot run that card; the reason is reported by the listing
        }
    };

    // Hold VRAM on this build's own card the whole time, so a "free VRAM unchanged" line cannot be bought by
    // nothing being allocated at all.
    std::unique_ptr<DeviceArena> hold;
    const double f0 = own_free(0);
    try {
        hold = std::make_unique<DeviceArena>(256ull << 20, 0, false);  // one cudaMalloc of 256 MiB
    } catch (const CudaError& e) {
        std::fprintf(stderr, "strata: could not hold VRAM on GPU 0 (%s)\n", e.what());
    }
    const double f1 = own_free(0);
    std::printf("cuda:0 free %.3f -> %.3f GiB while this engine holds %s (%s)\n", f0, f1,
                hold ? "256 MiB" : "nothing",
                hold ? "one cudaMalloc" : "the hold failed");

    bool hip_ran = false;
    for (int d = 0; d < L.n; ++d) {
        StrataHipTouch t{};
        const int e = L.touch(d, &t);
        std::printf("hip:%d touch error %d, kernel %d, read-back NaN %d, free %.3f -> %.3f GiB (total %.3f, "
                    "wave %d)\n",
                    d, e, t.kernel_ok, t.sample_is_nan, (double) t.free_before / GiB,
                    (double) t.free_after / GiB, (double) L.devices.d[d].total_bytes / GiB,
                    (int) L.devices.d[d].wave);
        if (t.kernel_ok == 1 && t.sample_is_nan == 1) hip_ran = true;
    }

    const double f2 = own_free(0);
    std::printf("cuda:0 free after the AMD kernel: %.3f GiB (was %.3f while held)\n", f2, f1);
    std::fflush(stdout);
    std::fprintf(stderr, "strata: cross-runtime probe: the AMD kernel %s in a process whose CUDA is already up "
                         "and holds %s\n",
                 hip_ran ? "RAN" : "did NOT run", hold ? "256 MiB" : "nothing");
    return hip_ran;
}

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
    apply_chosen_order(out);   // the user's order from the config, if one was given (STRATA_GPU_ORDER)
    return out;
}

}  // namespace strata::core
