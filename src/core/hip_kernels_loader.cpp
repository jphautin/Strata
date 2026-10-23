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
#include "strata/core/pinned.hpp"   // arena_pin_cap_gib(): C.4 reads the pin cap on THIS side too - the cap is per arena, not per process
#include "strata/core/expert_cache.hpp"   // read_expert_profile: C.5 ranks the pairs the helper tier takes
#include "strata/core/expert_source.hpp"  // ArenaExpertSource: the RAM arena re-laid WITH holes
#include "strata/core/remote_experts.hpp" // RemoteExperts: the helper tier, filled from experts.bin by a reader
#include "strata/kernels/cpu/expert_layout.hpp"   // present(), expert_layout_drop: the single RAM gate

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <utility>

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

// The shape a layer's hand-off needs, measured without a model or a weight: the library builds the ENGINE's OWN
// arena on the AMD card (src/core/device.cu: it allocates, and with poison=true its poison_kernel writes NaNs),
// and 16 floats come back into a host buffer THIS process owns.  CUDA holds its own 256 MiB the whole time, so
// the AMD hold cannot be read as "both runtimes share one pool", and a free-VRAM change cannot be bought by
// nothing being allocated.  true = the arena was built, the engine's kernel wrote it, and the read-back is NaN.
bool stage_hold_gate(long long bytes, int device) {
    const Eng& L = engine_lib();
    if (!L.ok) {
        std::fprintf(stderr, "strata: the stage-hold gate needs strata_hip_engine next to the engine\n");
        return false;
    }
    const auto hold = (strata_hip_stage_hold_fn) sym(L.h, "strata_hip_stage_hold");
    const auto release = (strata_hip_stage_free_fn) sym(L.h, "strata_hip_stage_free");
    if (hold == nullptr || release == nullptr) {
        std::fprintf(stderr, "strata: %s has no stage entries: it cannot hold a block on the AMD card\n", L.path.c_str());
        return false;
    }

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    auto own_free = [GiB](int ordinal) {
        try {
            const DeviceInfo i = device_info(ordinal);
            return (double) i.free_bytes / GiB;
        } catch (const CudaError&) {
            return -1.0;  // this build cannot run that card; the listing says why
        }
    };

    std::unique_ptr<DeviceArena> cuda_hold;
    const double c0 = own_free(0);
    try {
        cuda_hold = std::make_unique<DeviceArena>(256ull << 20, 0, false);  // one cudaMalloc of 256 MiB
    } catch (const CudaError& e) {
        std::fprintf(stderr, "strata: could not hold VRAM on GPU 0 (%s)\n", e.what());
    }
    const double c1 = own_free(0);
    std::printf("cuda:0 free %.3f -> %.3f GiB while this engine holds %s (one cudaMalloc)\n", c0, c1,
                cuda_hold ? "256 MiB" : "nothing");
    std::fflush(stdout);

    StrataStageRun r{};
    float host[16] = {};
    const int e = hold(bytes, device, &r, (int) sizeof(StrataStageRun), host);
    int n_nan = 0;
    for (int i = 0; i < 16; ++i) {
        if (host[i] != host[i]) ++n_nan;  // a NaN is the only value that is not equal to itself
    }
    std::printf("hip:%d engine arena %s, asked %lld B, held %lld B, free %.3f -> %.3f GiB (total %.3f), arena %d, "
                "read-back NaN %d of 16, code %d\n",
                device, e == 0 ? "held" : "refused", (long long) r.bytes, r.held_bytes,
                (double) r.free_before / GiB, (double) r.free_after / GiB, (double) r.total_bytes / GiB,
                r.arena_ok, n_nan, e);
    std::fflush(stdout);

    // Let go, then ask the SAME library instance for the card again: the arena's own destructor must give the
    // memory back, and the next small hold shows the card's free went up.  Two different HIP runtimes are not
    // compared here - both numbers come from the instance that did the hold.
    release();
    StrataStageRun again{};
    float tiny_host[16] = {};
    const long long small = 64ll << 20;
    const int e2 = hold(small, device, &again, (int) sizeof(StrataStageRun), tiny_host);
    std::printf("hip:%d after the release, a fresh %lld MiB hold sees %.3f GiB free (it was %.3f while the "
                "arena above was held), code %d\n",
                device, small >> 20, (double) again.free_before / GiB, (double) r.free_after / GiB, e2);
    std::fflush(stdout);
    release();

    const double c2 = own_free(0);
    std::printf("cuda:0 free after all of that: %.3f GiB (was %.3f while only CUDA held)\n", c2, c1);
    std::fflush(stdout);

    const bool ok = (e == 0 && r.arena_ok == 1 && r.held_bytes > 0 && r.read_back_ok == 1 && n_nan == 16);
    std::fprintf(stderr, "strata: stage-hold gate: %s.  The lines above are what this measured: the ENGINE's own "
                         "arena, the ENGINE's own kernel, the AMD card's memory, and a device-to-host copy into "
                         "a buffer the CUDA engine owns, all while CUDA holds its own 256 MiB\n",
                 ok ? "PROVEN" : "NOT PROVEN");
    return ok;
}

// C.1's measurement, taken on the ENGINE's side: ask the HIP-built library to build and run ONE real engine
// stage - its own weight table, its own native projections of that layer range, its own session carve, its own
// VRAM expert tier and its own Verifier, all made on the AMD card inside the library - and take the hand-off
// payload it wrote into RAM THIS process owns.  The library prints what it measured (GiB and seconds per step);
// this side prints the bytes it RECEIVED and its OWN hash of them, so the crossing is measured at the byte
// level rather than by a shape both sides agreed to write.  true = the stage ran, every expert of it was in the
// tier (so no CPU expert pool was asked across the join, which is what C.5 still has to wire), and the hash of
// the floats this process holds is the hash the library reported over the floats it wrote.
bool stage_run_gate(const StrataStageArgs& args) {
    const Eng& L = engine_lib();
    if (!L.ok) {
        std::fprintf(stderr, "strata: the stage gate needs strata_hip_engine next to the engine\n");
        return false;
    }
    const auto run = (strata_hip_stage_run_fn) sym(L.h, "strata_hip_stage_run");
    const auto take = (strata_hip_stage_payload_fn) sym(L.h, "strata_hip_stage_payload");
    if (run == nullptr || take == nullptr) {
        std::fprintf(stderr, "strata: %s has no stage entries: it cannot run a stage in this process\n",
                     L.path.c_str());
        return false;
    }

    StrataStageOut o{};
    const int e = run(&args, (int) sizeof(StrataStageArgs), &o, (int) sizeof(StrataStageOut));
    if (e != 0) {
        std::fprintf(stderr, "strata: the stage refused (code %d, step %d): %s\n", e, o.refused_step, o.err);
        return false;
    }

    // the payload, as the library reported it: rows of handoff_floats floats each.  This side sizes its OWN
    // buffer from those numbers and receives the floats by name - there is no handle across the join.
    const long long want = o.payload_bytes / (long long) sizeof(float);
    if (want < 1 || want > (long long) 1 << 28) {
        std::fprintf(stderr, "strata: the stage reported a payload of %lld floats: nothing to take\n", want);
        return false;
    }
    std::vector<float> host((size_t) want, 0.0f);
    const int got = take(host.data(), (int) want);
    if (got != (int) want) {
        std::fprintf(stderr, "strata: the stage wrote %lld floats and %d came back: the crossing did not carry "
                             "the hand-off\n", want, got);
        return false;
    }

    auto fnv = [](const void* pv, size_t n) {
        unsigned long long h = 1469598125565353939ull;
        for (const unsigned char* b = (const unsigned char*) pv, * e2 = b + n; b != e2; ++b)
            h = (((unsigned long long) (*b)) ^ h) * 10995116277794481ull;
        return h;
    };
    const unsigned long long here = fnv(host.data(), host.size() * sizeof(float));

    int nan = 0;
    for (int i = 0; i < 8; ++i) {
        if (host[i] != host[i]) ++nan;      // a NaN is the only value that is not equal to itself
    }

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::printf("stage: layers [%d, %d) ran INSIDE %s on hip:%d (%d AMD cards, code for %s), of %d layers\n",
                o.lb, o.le, L.path.c_str(), o.device, o.hip_devices, o.archs, o.n_layers);
    std::printf("stage: arena %.3f GiB + native %.3f GiB + Q8 %.3f GiB + tier %.3f GiB (%d experts) + session %.1f "
                "MiB + token table %.1f MiB host; card %.3f -> %.3f GiB free; load %.1f s, tier fill %.1f s, "
                "window %.1f ms\n",
                (double) o.arena_bytes / GiB, (double) o.dense_bytes / GiB, (double) o.hc_q8_bytes / GiB,
                (double) o.cache_bytes / GiB, o.resident_experts, (double) o.session_bytes / 1048576.0,
                (double) o.embed_bytes / 1048576.0, (double) o.free_before / GiB, (double) o.free_after / GiB,
                o.build_s, o.fill_s, o.run_ms);
    std::printf("stage: the hand-off crossed: %d floats per row, the buffer held %d rows, the window wrote %lld "
                "of them = %lld B; the library's hash of what it wrote %016llx, THIS process's hash of what it "
                "received %016llx (RAM at %p)\n",
                o.handoff_floats, o.rows,
                (o.handoff_floats > 0 ? o.payload_bytes / ((long long) o.handoff_floats * (long long) sizeof(float)) : 0LL),
                o.payload_bytes, o.payload_hash, here, (void*) host.data());
    std::printf("stage: the first values as they came back: ");
    for (int i = 0; i < 8; ++i) std::printf("%.6g ", (double) host[i]);
    std::printf("  (the library read %d, NaN %d of 8 here and %d there)\n", o.read_back_ok, nan, o.nan_count);
    std::fflush(stdout);

    const bool ok = (e == 0 && o.zero_doorbell == 1 && o.read_back_ok == 1 && nan == 0 && o.nan_count == 0 &&
                     here == o.payload_hash && o.resident_experts > 0 && o.run_ms >= 0.0);
    std::fprintf(stderr, "strata: stage gate: %s.  The lines above are what this measured: the stage built and "
                         "ran on the AMD card INSIDE the HIP-built library, in a process whose CUDA runtime is "
                         "already up, with no CPU expert pool across the join, and the payload's bytes hashed the "
                         "same on both sides\n", ok ? "PROVEN" : "NOT PROVEN");
    if (!ok) {
        if (here != o.payload_hash)
            std::fprintf(stderr, "strata:   the two hashes differ: the crossing did not carry the same bytes\n");
        if (o.zero_doorbell != 1)
            std::fprintf(stderr, "strata:   the window asked a CPU expert pool: the experts are not all in the "
                                 "tier (C.5's shape)\n");
    }
    return ok;
}

// ---- half (b) unit 2a: the SHIM - one call per window, payload glued to the step so a mixed serve can never
// read a stale window between two entries.  The stage is HELD in the library between build and free; a serve
// builds once, answers every window through step, and frees at the end of the run.
bool engine_stage_build(const StrataStageArgs& args, StrataStageOut* out) {
    static const auto fn = (strata_hip_stage_build_fn) sym(engine_lib().h, "strata_hip_stage_build");
    if (fn == nullptr) {
        std::fprintf(stderr, "strata: the engine library has no stage-build entry: no library stage in this process\n");
        return false;
    }
    StrataStageOut dummy{};
    StrataStageOut& o = out ? *out : dummy;              // the entry NEVER takes a null out: a caller that wants
        if (out == nullptr) std::memset(&dummy, 0, sizeof(dummy));  //   no report gets a private reading
    const int e = fn(&args, (int) sizeof(StrataStageArgs), &o, (int) sizeof(StrataStageOut));
    if (e != 0) {
        std::fprintf(stderr, "strata: the library stage refused at build (code %d, step %d): %s\n",
                     e, o.refused_step, o.err);
        return false;
    }
    return true;                                        // the stage is HELD inside the library now
}

long long engine_stage_step(const char* tokens, float* payload_out, int payload_floats, StrataStepOut* report) {
    static const auto step_fn = (strata_hip_stage_step_fn) sym(engine_lib().h, "strata_hip_stage_step");
    static const auto take_fn = (strata_hip_stage_payload_fn) sym(engine_lib().h, "strata_hip_stage_payload");
    if (step_fn == nullptr || take_fn == nullptr) {
        std::fprintf(stderr, "strata: the engine library has no step/payload pair: no per-window stage call\n");
        return -1;
    }
    StrataStepArgs sa{};
    std::snprintf(sa.tokens, sizeof(sa.tokens), "%s", tokens ? tokens : "");
    StrataStepOut dummy{};
    StrataStepOut& r = report ? *report : dummy;
    if (report == nullptr) std::memset(&dummy, 0, sizeof(dummy));
    const int e = step_fn(&sa, (int) sizeof(StrataStepArgs), &r, (int) sizeof(StrataStepOut));
    if (e != 0) {
        std::fprintf(stderr, "strata: the library stage step refused (code %d): %s\n", e, r.err);
        return -1;
    }
    const long long held = r.payload_bytes / (long long) sizeof(float);   // floats the window WROTE
    if (payload_out == nullptr || payload_floats < 1) return held;        // the caller only wants the report
    const int got = take_fn(payload_out, payload_floats);                 // min(wrote, capacity) - one step,
    if ((long long) got != held) {                                        //   one take: no stale window
        std::fprintf(stderr, "strata: the window wrote %lld floats, %d came back: the payload did not cross\n",
                     held, got);
        return -1;
    }
    return (long long) got;
}

void engine_stage_free() {   // the held stage goes back; a serve does this once per run, like stage_hold did
    static const auto fn = (strata_hip_stage_free_fn) sym(engine_lib().h, "strata_hip_stage_free");
    if (fn == nullptr) return;
    fn();
}

// Half (b): the SAME stage, TWO windows.  C.1's entry rebuilt everything for each window, which measures the
// crossing but cannot serve a decode: a real run replays the SAME built stage every window.  The split entries
// hold the stage in the library between build and free, so this gate proves the contract a decode needs:
// build ONCE inside the library, replay the same window TWICE on the held stage, take each window's payload
// back into RAM this process owns, and each window's bytes must hash the same on both sides - and the second
// window carries no load at all: the library's step line says which step it is, and its wait is only the gap
// between windows, never the stage's seconds.  Whether the two replays' payloads are EQUAL is printed as
// information, not demanded: a session that advanced between windows may answer differently, and the decode
// does not require them to agree.  true = built once, replayed twice, both windows crossed at the byte level.
bool stage_persist_gate(const StrataStageArgs& args) {
    const Eng& L = engine_lib();
    if (!L.ok) {
        std::fprintf(stderr, "strata: the persisted-stage gate needs strata_hip_engine next to the engine\n");
        return false;
    }
    // the gate drives the SHIM (unit 2a): one call per window, the payload glued to the step so the gate
    // itself cannot read a stale window between two entries

    StrataStageOut o{};
    if (!engine_stage_build(args, &o)) {
        return false;                                   // the shim named the refusal (and released what it took)
    }
    std::printf("persist: the stage BUILT inside %s on hip:%d: layers [%d, %d) of %d: tier %.3f GiB with %d experts, "
                "arena %.3f GiB, native %.3f GiB, Q8 %.3f GiB, session %.1f MiB, token table %.1f MiB host; card "
                "%.3f GiB free before, %.3f GiB while HELD; load %.1f s, tier fill %.1f s\n",
                L.path.c_str(), o.device, o.lb, o.le, o.n_layers,
                (double) o.cache_bytes / 1073741824.0, o.resident_experts,
                (double) o.arena_bytes / 1073741824.0, (double) o.dense_bytes / 1073741824.0,
                (double) o.hc_q8_bytes / 1073741824.0, (double) o.session_bytes / 1048576.0,
                (double) o.embed_bytes / 1048576.0,
                (double) o.free_before / 1073741824.0, (double) o.free_after / 1073741824.0,
                o.build_s, o.fill_s);
    std::fflush(stdout);

    auto fnv = [](const void* pv, size_t n) {
        unsigned long long h = 1469598125565353939ull;
        for (const unsigned char* b = (const unsigned char*) pv, * e2 = b + n; b != e2; ++b)
            h = (((unsigned long long) (*b)) ^ h) * 10995116277794481ull;
        return h;
    };

    StrataStepArgs sa{};
    std::snprintf(sa.tokens, sizeof(sa.tokens), "%s", args.tokens);   // the same window, replayed twice

    bool ok = true;
    bool got1 = false;
    unsigned long long hash1 = 0;
    for (int w = 0; w < 2; ++w) {
        StrataStepOut so{};
        const long long cap = (long long) o.handoff_floats * 8;   // kVerifyMaxT rows of this stage's hand-off
        std::vector<float> host((size_t) cap, 0.0f);
        const long long got = engine_stage_step(sa.tokens, host.data(), (int) cap, &so);
        if (got < 0) {
            std::fprintf(stderr, "strata: the persisted-stage gate refused at window %d: %s\n", w + 1, so.err);
            ok = false;
            break;
        }
        const long long want = so.payload_bytes / (long long) sizeof(float);
        const unsigned long long here = fnv(host.data(), (size_t) got * sizeof(float));
        if (got != want || here != so.payload_hash) {
            std::fprintf(stderr, "strata: window %d's payload did not cross: %lld of %lld floats back, the "
                                 "hashes %016llx here against %016llx there\n",
                         w + 1, (long long) got, want, here, so.payload_hash);
            ok = false;
            break;
        }
        if (w == 0) {
            hash1 = here;
            got1 = true;
        } else if (got1 && here != hash1) {
            std::printf("persist: the two replays differ at the payload: the session moved between windows - "
                        "information only, the decode does not demand they agree\n");
        }
        std::printf("persist: window %d of the HELD stage: %lld B of hand-off back in RAM I own, both sides "
                    "hash %016llx, the window %.1f ms (%.1f ms since the last ended - the load is NOT in here)\n",
                    w + 1, (long long) so.payload_bytes, here, so.run_ms, so.wait_ms);
        std::fflush(stdout);
    }
    engine_stage_free();                                // the held stage goes back; the card is free again

    std::fprintf(stderr, "strata: persisted-stage gate: %s.  One stage built inside the HIP-built library "
                         "answered TWO windows without being rebuilt: both windows' payload bytes hashed the "
                         "same on the library's side and on this process's side, and the library says which "
                         "step each window was\n", ok ? "PROVEN" : "NOT PROVEN");
    return ok;
}

// C.4's measurement: ONE ARENA PER RUNTIME, not one arena per process.  A mixed process has two runtimes up,
// and today one arena serves only the build's own: this gate makes each runtime hold and release ITS OWN
// DeviceArena on ITS OWN card - this process's arena on the card its runtime numbers 0 (printed first), and
// the library's arena inside its HIP runtime through the two arena entries above (printed second) - and each
// line carries ONLY that arena's asked/held/free-before-after/total, plus the arena_pin_cap_gib() value THAT
// side read: the WDDM cap (docs/MULTI_GPU.md:340-343) is per arena, so each side's line shows its own, asked
// BEFORE the hold rather than assumed.  A refusal names its card with that card's numbers, in generate.cpp:164's
// shape ` (%llu MiB of %llu MiB VRAM free on this GPU)`.  A HIP-only build has no library half: its line says
// "not applicable", and the gate is then this engine's own arena alone.  true = every arena that applies held
// and gave the memory back (the release line prints the card free again, on both sides).
bool per_runtime_arenas_gate(long long bytes) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    auto cap_text = [](long long c) -> std::string {
        if (c == -2) return "auto";
        if (c == -1) return "unset";
        if (c == 0) return "the whole arena";
        return std::to_string(c) + " GiB";
    };
    auto card0 = [](DeviceInfo& i) {
        try {
            i = device_info(0);
            return true;
        } catch (const CudaError&) {
            i = DeviceInfo{};
            return false;
        }
    };

    // ---- 1. the arena of THIS process's runtime, on the card IT numbers 0
    DeviceInfo c0{};
    if (!card0(c0)) {
        std::fprintf(stderr, "strata: the arenas gate cannot read this runtime's card 0: no arena was measured\n");
        return false;
    }
    const bool hip_here = (c0.runtime == DeviceRuntime::HIP);
    const char* brand = hip_here ? "hip" : "cuda";
    const long long cap_here = arena_pin_cap_gib();   // read on THIS side, printed as-is on this line
    std::unique_ptr<DeviceArena> own;
    long long held_own = 0;
    try {
        own = std::make_unique<DeviceArena>((unsigned long long) bytes, 0, /*poison=*/false);
        held_own = (long long) own->capacity();
    } catch (const CudaError&) {
        own.reset();   // the card refused; the arena does not exist
    }
    DeviceInfo c1{};
    if (!card0(c1)) c1 = c0;
    if (held_own > 0)
        std::printf("%s%d (%s) own arena: held, asked %lld B, held %lld B, free %.3f -> %.3f GiB (total %.3f GiB), "
                    "pin cap read here: %s\n",
                    brand, 0, c0.name.c_str(), (long long) bytes, held_own,
                    (double) c0.free_bytes / GiB, (double) c1.free_bytes / GiB,
                    (double) c0.total_bytes / GiB, cap_text(cap_here).c_str());
    else
        std::printf("%s%d (%s) own arena: REFUSED the ask of %lld B (%llu MiB of %llu MiB VRAM free on this GPU), "
                    "pin cap read here: %s\n",
                    brand, 0, c0.name.c_str(), (long long) bytes,
                    (unsigned long long) (c1.free_bytes >> 20), (unsigned long long) (c0.total_bytes >> 20),
                    cap_text(cap_here).c_str());
    std::fflush(stdout);
    own.reset();   // the arena's own destructor gives the card's memory back
    DeviceInfo c2{};
    if (!card0(c2)) c2 = c1;
    std::printf("%s%d own arena released: free back to %.3f GiB (was %.3f while it was held)\n",
                brand, 0, (double) c2.free_bytes / GiB, (double) c1.free_bytes / GiB);
    std::fflush(stdout);
    const bool own_ok = (held_own > 0 && c2.free_bytes > c1.free_bytes);

    // ---- 2. the arena INSIDE the library, on the card ITS HIP runtime numbers 0
    bool lib_ok = false, lib_na = false;
    const Eng& L = engine_lib();
    if (!L.ok) {
#ifdef STRATA_USE_HIP
        std::printf("library arena: not applicable (this engine IS the HIP side; the CUDA engine build opens %s)\n",
                    STRATA_HIP_ENGINE_LIB_NAME);
        lib_na = true;
        std::fflush(stdout);
#else
        std::fprintf(stderr, "strata: the arenas gate needs %s next to the engine: no library arena in this process\n",
                     STRATA_HIP_ENGINE_LIB_NAME);
#endif
    } else {
        const auto hold = (strata_hip_arena_hold_fn) sym(L.h, "strata_hip_arena_hold");
        const auto release = (strata_hip_arena_free_fn) sym(L.h, "strata_hip_arena_free");
        if (hold == nullptr || release == nullptr) {
            std::fprintf(stderr, "strata: %s has no arena entries: it cannot hold its own arena inside its HIP runtime\n",
                         L.path.c_str());
        } else {
            StrataArenaReport r{};
            const int e = hold(bytes, 0, &r, (int) sizeof(StrataArenaReport));
            if (e == 0 && r.held_bytes > 0) {
                std::printf("hip%d (%s) library arena: held, asked %lld B, held %lld B, free %.3f -> %.3f GiB "
                            "(total %.3f GiB), pin cap read THERE: %s\n",
                            r.device, r.card, (long long) bytes, r.held_bytes,
                            (double) r.free_before / GiB, (double) r.free_after / GiB,
                            (double) r.total_bytes / GiB, cap_text(r.cap_gib).c_str());
                std::fflush(stdout);
                release();
                StrataArenaReport again{};
                const long long small = 64ll << 20;
                const int e2 = hold(small, 0, &again, (int) sizeof(StrataArenaReport));
                std::printf("hip%d library arena released: a fresh %lld MiB hold inside the library sees %.3f GiB "
                            "free (it was %.3f while the arena above was held), code %d\n",
                            r.device, small >> 20, (double) again.free_before / GiB,
                            (double) r.free_after / GiB, e2);
                std::fflush(stdout);
                if (e2 == 0) release();
                lib_ok = true;
            } else if (e == -2) {
                std::printf("hip%d %s library arena: REFUSED the ask of %lld B (%llu MiB of %llu MiB VRAM free "
                            "on this GPU), pin cap read THERE: %s, code %d\n",
                            r.device, r.card[0] ? r.card : "(the card could not be named)", (long long) bytes,
                            (unsigned long long) (r.free_before >> 20), (unsigned long long) (r.total_bytes >> 20),
                            cap_text(r.cap_gib).c_str(), e);
                std::fflush(stdout);
            } else {
                std::fprintf(stderr, "strata: the library arena entry refused (%d): no card was named, the blocks "
                                     "disagree or the device does not exist in that runtime\n", e);
            }
        }
    }

    const bool ok = own_ok && (lib_ok || lib_na);
    std::fprintf(stderr, "strata: per-runtime arenas gate: %s.  The lines above are what this measured: %s, "
                         "each arena with its OWN numbers and the pin cap read ON ITS SIDE, never the other's\n",
                 ok ? "PROVEN" : "NOT PROVEN", lib_na ? "one arena, this engine's runtime only" : "two arenas, one per runtime");
    return ok;
}

// CODE.md C.5: the expert tiers across the join, one byte-stream at THREE sites.  The helper tier (NVIDIA,
// CUDA's own runtime here) is filled from experts.bin through a `reader` BEFORE the RAM arena exists - its
// blobs cannot come from the arena, the arena is about to be re-laid WITHOUT exactly these pairs.  The RAM
// arena (this process) gets holes at those pairs and loads the rest from experts.bin.  Dropped pairs come
// BACK through RemoteExperts::fetch, wired to the arena's aux_fetch.  The library's OWN expert tier (AMD,
// inside its HIP runtime, filled from experts.bin by its own objects) hashes the SAME named pairs: three
// FNV-1a numbers per pair - RAM arena (present), NVIDIA tier fetched back (dropped), AMD tier in the
// library (both).  Every line carries only the numbers of the side that measured it.
bool expert_tier_gate(const std::string& pack, const std::string& native, const std::string& profile_path,
                      long long layers, long long experts, int tier_pairs, int helper_device) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::string err;

    // In a mixed process the primary session runs INSIDE the HIP library, on the AMD card: a card of the
    // CUDA runtime here is a helper.  Left at its default (ordinal 0), the helper-cannot-be-primary rule
    // would refuse this rig's only CUDA card - the default describes the CUDA-only world, not this one.
    set_primary_device(-1);

    if (!strata::kernels::cpu::expert_layout_load(pack, layers, experts, err)) {
        std::fprintf(stderr, "strata: expert tier gate: the pack's layout refused: %s\n", err.c_str());
        return false;
    }
    auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) {
        std::fprintf(stderr, "strata: expert tier gate: a holed RAM arena needs a native pack (per-layer blobs): %s\n",
                     pack.c_str());
        return false;
    }

    // ---- 1. the helper tier, BEFORE the arena: experts.bin read pair by pair (piece (a))
    std::vector<std::pair<int32_t, int32_t>> ranked;
    int64_t pslots = 0;
    if (!read_expert_profile(profile_path, lay.n_layers, lay.n_expert, ranked, pslots, err)) {
        std::fprintf(stderr, "strata: expert tier gate: the profile refused: %s\n", err.c_str());
        return false;
    }
    double free_gib = 0.0;
    if (!RemoteExperts::preflight(helper_device, free_gib, err)) {
        std::fprintf(stderr, "strata: expert tier gate: helper preflight refused: %s\n", err.c_str());
        return false;
    }
    DeviceInfo hd{};
    try { hd = device_info(helper_device); } catch (const CudaError&) { hd = DeviceInfo{}; }
    // The tier's `reader`: expert by expert from the PACK, BEFORE the RAM arena exists.  This rig's pack
    // is a native GGUF one (no experts.bin - native_experts.txt names the tensors in the shards), so its
    // own FileExpertSource is what the reader reads through; a pack with experts.bin is the same reader
    // over that file.  Either way: these bytes come from the PACK, never from the arena to be.
    FileExpertSource tier_file;
    tier_file.set_gguf(native);
    if (!tier_file.open(pack, lay.n_layers, lay.n_expert, err)) {
        std::fprintf(stderr, "strata: expert tier gate: the pack's expert file refused: %s\n", err.c_str());
        return false;
    }
    // the reader's OWN buffers: verify_slot re-reads the first pair's bytes AFTER the fill loop, so
    // nothing may recycle them (the arena's own bytes do not exist for these pairs at all)
    std::map<std::pair<int32_t, int32_t>, std::vector<uint8_t>> held;
    auto reader = [&](int32_t l, int32_t e) -> const uint8_t* {
        const auto key = std::make_pair(l, e);
        const auto it = held.find(key);
        if (it != held.end()) return it->second.data();
        const uint8_t* b = tier_file.blob_stable(l, e);
        if (b == nullptr) return nullptr;
        std::vector<uint8_t> v(b, b + (size_t) lay.blob_bytes(l));
        return (held.emplace(key, std::move(v)).first)->second.data();
    };
    RemoteExperts helpers;
    std::vector<uint8_t> claimed((size_t) lay.n_layers * lay.n_expert, 0);
    ExpertCache no_primary;      // the real primary tier is the LIBRARY's, on the AMD card: nothing here owns it
    FileExpertSource no_source;  // left closed: a set `reader` is where these pairs' blobs come from
    if (!helpers.open(helper_device, tier_pairs, lay.n_layers, lay.n_expert, ranked,
                      no_primary, no_source, claimed, err, /*auto_size=*/false, reader)) {
        std::fprintf(stderr, "strata: expert tier gate: cuda%d tier refused: %s\n", helper_device, err.c_str());
        return false;
    }
    long long tier_count = 0;
    double tier_bytes = 0.0;
    for (size_t i = 0; i < claimed.size(); ++i)
        if (claimed[i]) { ++tier_count; tier_bytes += (double) lay.blob_bytes((long long) (i / lay.n_expert)); }
    std::printf("cuda%d (%s) helper tier BEFORE the arena: %lld pairs read pair by pair from the pack by the reader, "
                "%.3f GiB of blobs, %.3f GiB free on that card before\n",
                helper_device, hd.name.c_str(), tier_count, tier_bytes / GiB, free_gib);
    std::fflush(stdout);

    // ---- 2. the RAM arena re-laid WITHOUT exactly those pairs (piece (b)): claimed IS the mask
    const uint64_t file_bytes_before = lay.total;
    std::string derr;
    if (!strata::kernels::cpu::expert_layout_drop(claimed, derr)) {
        std::fprintf(stderr, "strata: expert tier gate: the drop refused: %s\n", derr.c_str());
        return false;
    }
    std::printf("expert layout dropped: the RAM arena is now %.3f GiB of the pack's %.3f GiB: %.3f GiB of "
                "holes, %lld pairs held by cuda%d alone\n",
                (double) lay.total / GiB, (double) lay.file_total / GiB,
                (double) (file_bytes_before - lay.total) / GiB, tier_count, helper_device);
    std::fflush(stdout);

    // ---- 3. the holed arena, loaded pair by pair around the holes (its own lines print its own words)
    ArenaExpertSource arena;
    arena.set_gguf(native);   // the pack is a native GGUF one: the arena reads the shards it names
    const auto t0 = std::chrono::steady_clock::now();
    if (!arena.open(pack, lay.n_layers, lay.n_expert, 6, err)) {
        std::fprintf(stderr, "strata: expert tier gate: the holed arena refused: %s\n", err.c_str());
        return false;
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!arena.ram_warning().empty())
        std::fprintf(stderr, "strata: expert tier gate note: %s\n", arena.ram_warning().c_str());
    std::printf("arena holed: %.3f GiB of experts loaded in %.1f s (%.2f GiB/s wall, registration included)\n",
                (double) lay.total / GiB, load_s, (double) lay.total / GiB / (load_s > 0 ? load_s : 1.0));
    std::fflush(stdout);

    // ---- 4. fetch-back: the dropped pairs have NO RAM copy; the arena's aux_fetch (the helper's fetch)
    // is the only way their bytes reach a host buffer.  Two dropped and two present pairs carry the list.
    arena.aux_fetch = [&](int64_t l, int64_t e, uint8_t* dst) -> bool { return helpers.fetch(l, e, dst); };
    int dropped_l[2] = {-1, -1}, dropped_e[2] = {-1, -1}, present_l[2] = {-1, -1}, present_e[2] = {-1, -1};
    for (long long l = 0, d = 0, p = 0; l < lay.n_layers && (d < 2 || p < 2); ++l)
        for (long long e = 0; e < lay.n_expert && (d < 2 || p < 2); ++e) {
            const bool held_pair = claimed[(size_t) (l * lay.n_expert + e)] != 0;
            if (held_pair && d < 2) { dropped_l[(size_t) d] = (int) l; dropped_e[(size_t) d++] = (int) e; }
            if (!held_pair && p < 2) { present_l[(size_t) p] = (int) l; present_e[(size_t) p++] = (int) e; }
        }
    unsigned long long host_hash[4] = {0, 0, 0, 0};
    std::printf("dropped pick (%d,%d) and (%d,%d): the arena calls them %s and %s\n",
                dropped_l[0], dropped_e[0], dropped_l[1], dropped_e[1],
                arena.transient(dropped_l[0], dropped_e[0]) ? "transient" : "NOT transient",
                arena.transient(dropped_l[1], dropped_e[1]) ? "transient" : "NOT transient");
    std::fflush(stdout);
    const bool fetch_ok = dropped_l[1] >= 0 && arena.transient(dropped_l[0], dropped_e[0])
        && arena.transient(dropped_l[1], dropped_e[1]);
    bool copy_ok = fetch_ok;
    for (int i = 0; i < 2 && copy_ok; ++i) {
        std::vector<uint8_t> buf((size_t) lay.blob_bytes(dropped_l[i]), 0);
        const bool one = arena.copy_blob(dropped_l[i], dropped_e[i], buf.data());
        copy_ok = one && !buf.empty();
        std::printf("fetch-back of (%d, %d): copy_blob %s, %lld B into the host buffer\n",
                    dropped_l[i], dropped_e[i], one ? "ran" : "refused", (long long) buf.size());
        std::fflush(stdout);
        if (copy_ok) host_hash[i] = fnv1a64(buf.data(), buf.size());
    }
    bool ram_ok = fetch_ok && copy_ok;
    for (int i = 0; i < 2 && ram_ok; ++i) {
        const uint8_t* b = arena.blob_at(present_l[i], present_e[i]);
        std::printf("present pick (%d, %d): the RAM arena answers %s\n",
                    present_l[i], present_e[i], b != nullptr ? "a pointer into itself" : "nothing");
        std::fflush(stdout);
        if (b == nullptr) { ram_ok = false; break; }
        host_hash[2 + i] = fnv1a64(b, (uint64_t) lay.blob_bytes(present_l[i]));
    }
    if (!fetch_ok || !copy_ok || !ram_ok) {
        std::fprintf(stderr, "strata: expert tier gate: a site gave nothing: dropped pair found no fetchable bytes, "
                             "or the RAM hashes could not be taken\n");
        return false;
    }
    std::printf("fetched back %d dropped pairs from cuda%d: the arena calls them transient, aux_fetch = RemoteExperts::fetch, "
                "%.3f GiB into host buffers hashed HERE\n",
                2, helper_device,
                ((double) lay.blob_bytes(dropped_l[0]) + (double) lay.blob_bytes(dropped_l[1])) / GiB);
    std::fflush(stdout);

    // ---- 4b. the PACK itself, hashed a fourth time through the gate's own reader (tier_file): which site,
    // if any, is the odd one out
    unsigned long long file_hash[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const int l = i < 2 ? dropped_l[i] : present_l[i - 2];
        const int e = i < 2 ? dropped_e[i] : present_e[i - 2];
        const uint8_t* b = tier_file.blob_stable(l, e);
        file_hash[i] = b ? fnv1a64(b, (uint64_t) lay.blob_bytes(l)) : 0ull;
    }

    // ---- 5. the library's OWN tier (AMD) hashes the SAME four pairs, inside its HIP runtime
    const Eng& L = engine_lib();
    if (!L.ok) {
        std::fprintf(stderr, "strata: expert tier gate: no %s next to the engine: no AMD tier to hash in this process\n",
                     STRATA_HIP_ENGINE_LIB_NAME);
        return false;
    }
    const auto hashfn = (strata_hip_expert_hash_fn) sym(L.h, "strata_hip_expert_hash");
    if (hashfn == nullptr) {
        std::fprintf(stderr, "strata: %s has no expert hash entry: the library cannot answer the join\n", L.path.c_str());
        return false;
    }
    StrataExpertPairs asked{};
    std::snprintf(asked.pack, sizeof asked.pack, "%s", pack.c_str());
    std::snprintf(asked.native, sizeof asked.native, "%s", native.c_str());
    asked.n_layers = lay.n_layers;
    asked.n_expert = lay.n_expert;
    asked.n_pairs = 4;
    for (int i = 0; i < 2; ++i) {
        asked.pairs[2 * i] = dropped_l[i];
        asked.pairs[2 * i + 1] = dropped_e[i];
        asked.pairs[2 * (2 + i)] = present_l[i];
        asked.pairs[2 * (2 + i) + 1] = present_e[i];
    }
    asked.device = 0;   // the AMD ordinal inside the library's own HIP runtime, as --list-runtimes numbers it
    StrataExpertOut got{};
    const int e2 = hashfn(&asked, (int) sizeof(StrataExpertPairs), &got, (int) sizeof(StrataExpertOut));
    if (e2 != 0) {
        std::fprintf(stderr, "strata: expert tier gate: the library tier entry refused (%d): %s\n", e2, got.err);
        return false;
    }
    std::printf("hip%d (%s) library tier: archs %s, %.3f GiB of VRAM cache, 4 pairs (2 dropped + 2 present) filled "
                "from experts.bin INSIDE the library in %.2f s, slots read back and hashed THERE\n",
                0, got.card, got.archs, (double) got.cache_bytes / GiB, got.fill_s);
    std::fflush(stdout);

    // ---- 6. the join's measurement: one byte-stream, three sites, two numbers per pair
    bool all_equal = got.ran == 1;
    for (int i = 0; i < 4; ++i) {
        const bool equal = host_hash[i] == got.hash[i] && got.blob_bytes[i] > 0;
        all_equal = all_equal && equal;
        std::printf("pair (%d, %d): %s: host %016llx vs library %016llx vs the pack-file %016llx over %lld B: %s\n",
                    i < 2 ? dropped_l[i] : present_l[i - 2], i < 2 ? dropped_e[i] : present_e[i - 2],
                    i < 2 ? "dropped, fetched from the NVIDIA tier" : "present, read from the RAM arena",
                    host_hash[i], got.hash[i], file_hash[i], (long long) got.blob_bytes[i],
                    equal ? "equal" : "NOT EQUAL");
        std::fflush(stdout);
    }
    std::fprintf(stderr, "strata: expert tier gate: %s.  The lines above are what this measured: a helper tier "
                         "filled from experts.bin BEFORE the arena, the RAM arena re-laid WITHOUT those pairs, "
                         "the dropped ones fetched back, and the same bytes hashed by the library's AMD tier\n",
                 all_equal ? "PROVEN" : "NOT PROVEN");
    return all_equal;
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
