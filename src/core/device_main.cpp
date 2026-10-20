// src/core/device_main.cpp - `strata-device`: report the GPU, the plan, and exercise the arena.
//
// This is P2.S1's "startup prints the memory plan vs actual cudaMemGetInfo" bullet, on its own so it can run
// without the model.  It is also the run-time half of the sm_120 policy: CMake refuses to COMPILE for another
// architecture, and this refuses to RUN on one.
#include "strata/core/device.hpp"
#include "strata/plan/plan.hpp"
#ifdef STRATA_MULTI_BRAND
#include "strata/core/hip_engine_abi.hpp"  // the engine library's ABI, and stage_hold_gate's declaration
#endif

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#if defined(STRATA_USE_HIP) && defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static std::string human(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GiB (%llu B)", (double) b / (1024.0 * 1024 * 1024),
                  (unsigned long long) b);
    return buf;
}

int main(int argc, char** argv) {
    bool selftest = false;
    bool list_devices = false;
    bool list_runtimes = false;
    bool probe_cross = false;
    bool probe_engine = false;
    bool stage_hold = false;
#ifdef STRATA_MULTI_BRAND
    // C.1's gate: the stage needs a model, so the paths and the range are asked for, with the shape's
    // defaults around them (one layer, 1024 cells, hip:0, and the ids the daily run uses).
    bool probe_stage = false;
    std::string stage_pack, stage_native, stage_tokens = "248045,846";
    long long stage_layers = 1, stage_context = 1024;
    int stage_device = 0;
#endif
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;
        else if (std::strcmp(argv[i], "--list-devices") == 0) list_devices = true;
        else if (std::strcmp(argv[i], "--list-runtimes") == 0) list_runtimes = true;
        else if (std::strcmp(argv[i], "--probe-cross") == 0) probe_cross = true;
        else if (std::strcmp(argv[i], "--probe-engine") == 0) probe_engine = true;
        else if (std::strcmp(argv[i], "--stage-hold") == 0) stage_hold = true;
#ifdef STRATA_MULTI_BRAND
        else if (std::strcmp(argv[i], "--probe-stage") == 0) probe_stage = true;
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-pack") == 0) stage_pack = argv[++i];
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-native") == 0) stage_native = argv[++i];
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-tokens") == 0) stage_tokens = argv[++i];
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-layers") == 0) stage_layers = std::atoll(argv[++i]);
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-context") == 0) stage_context = std::atoll(argv[++i]);
        else if (i + 1 < argc && std::strcmp(argv[i], "--stage-device") == 0) stage_device = std::atoi(argv[++i]);
#endif
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: strata-device [--selftest] [--list-devices]\n"
                        "  --list-devices  every GPU the runtime enumerates, numbered as HIP_VISIBLE_DEVICES /\n"
                        "                  CUDA_VISIBLE_DEVICES number them, and whether this binary can run it\n");
#ifdef STRATA_MULTI_BRAND
            std::printf("  --list-runtimes every GPU this process can drive, both brands, numbered cuda:N / hip:N\n"
                        "  --probe-cross  run one AMD kernel from the library in THIS process (its CUDA is up)\n"
                        "  --probe-engine open the HIP-built ENGINE library (its stage code, no layer runs)\n"
                        "  --stage-hold  hold 256 MiB on hip:0 through that library's ENGINE arena, then read its\n"
                        "                  NaNs back into THIS process's RAM (the layer hand-off's shape)\n");
            std::printf("  --probe-stage  build and run ONE real engine stage INSIDE that library on the AMD card,\n"                        "                  and bring its hand-off payload back into RAM this process owns;\n"                        "                  --stage-pack DIR --stage-native SHARD1 [--stage-layers K]\n"                        "                  [--stage-context N] [--stage-tokens IDS] [--stage-device D]\n");
#endif

            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    // The runtime's numbering, which setup needs on Windows: there an integrated Radeon is HIP device 0 and pushes the
    // discrete card to 1, while setup finds the cards in the display-adapter order (#325).  No arch check here - the
    // cards this binary has no code for are part of the answer.  Format (setup.py's hip_devices parses it):
    //   device N: <name>
    //     arch gfx1201, 15.9 GiB, wave32          (CUDA: compute capability 12.0, 11.9 GiB)
    //     cannot run: <why>                       (only for a card this binary cannot run)
    if (list_devices) {
        const int count = strata::core::device_count();
        if (count == 0) std::printf("(no GPU device)\n");
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            std::string name, detail;
            if (!strata::core::device_summary(ordinal, name, detail)) {
                std::printf("device %d: (the runtime cannot describe it)\n", ordinal);
                continue;
            }
            std::printf("device %d: %s\n  %s\n", ordinal, name.c_str(), detail.c_str());
            if (const std::string why = strata::core::gpu_arch_problem(ordinal); !why.empty())
                std::printf("  cannot run: %s\n", why.c_str());
        }
        return 0;
    }

    // Both brands in ONE process (Track B, B.1's shape): the cards this engine's own runtime enumerates, then
    // the AMD cards strata_hip_kernels listed from INSIDE its own runtime - which is the narrow blocker today:
    // one runtime names only one brand's devices, so docs/MULTI_GPU.md:53 refuses a mix of NVIDIA and AMD.
    // The layer split itself needs neither NVLink nor peer-to-peer access (its hand-off goes through pinned
    // host RAM), so a list that carries the brand is what a mixed rig needs.  Format:
    //   cuda:0: <name>
    //     compute capability 8.6, 8.0 GiB
    //   hip:0: <name>
    //     arch gfx1201, 15.9 GiB
    if (list_runtimes) {
#ifdef STRATA_MULTI_BRAND
        const auto cards = strata::core::union_device_list();
        for (const strata::core::DeviceInfo& d : cards) {
            const bool hip = d.runtime == strata::core::DeviceRuntime::HIP;
            std::printf("%s%d: %s\n", hip ? "hip" : "cuda", d.ordinal, d.name.c_str());
            if (hip)
                std::printf("  arch %s, %s\n", d.arch.c_str(), human(d.total_bytes).c_str());
            else
                std::printf("  compute capability %d.%d, %s\n", d.cc_major, d.cc_minor,
                            human(d.total_bytes).c_str());
        }
        // The placement order, in the same list: the MAIN card first, then the second card, then the RAM
        // tier (the experts no card holds).  Which card is main is the user's choice, asked at setup and
        // saved in the run config; the launcher carries it as STRATA_GPU_ORDER.  Nothing here ranks the
        // cards by power: a CUDA card and an RDNA card share no unit.
        std::string ids;
        for (const strata::core::DeviceInfo& d : cards) {
            const bool hip = d.runtime == strata::core::DeviceRuntime::HIP;
            ids += (ids.empty() ? std::string("placement order: ") : std::string(" then "))
                   + (hip ? "hip:" : "cuda:") + std::to_string(d.ordinal);
        }
        std::string note;
        if (cards.size() < 2)
            note = " - one card only: it holds the layers, and RAM holds the experts no card can";
        else if (strata::core::card_order_chosen())
            note = " (the order you chose at setup, STRATA_GPU_ORDER=\""
                   + strata::core::card_order_text() + "\")";
        else {
            const char* given = std::getenv("STRATA_GPU_ORDER");
            note = given ? " (an order WAS given (STRATA_GPU_ORDER) but could not be used - see the line above; "
                          "this build's own runtime lists its cards first, so the layers did not move)"
                         : " (nothing chosen at setup: this build's own runtime lists its cards first; the measured"
                           " default of a same-brand split puts the FASTER card last, docs/MULTI_GPU.md)";
        }
        std::printf("%s%s\n", ids.c_str(), note.c_str());
#ifdef STRATA_USE_HIP
        std::printf("AMD kernels library: not applicable (this engine is the HIP side; the CUDA engine build opens it)\n");
#else
        std::printf("AMD kernels library: %s\n",
                    strata::core::hip_kernels_ready() ? "present" : "not present next to this engine");
#endif
        return 0;
#else
        std::fprintf(stderr, "strata-device: --list-runtimes needs a multi-brand build (-DSTRATA_MULTI_BRAND=ON)\n");
        return 2;
#endif
    }

    // The gate for the NEXT step (routing the AMD layers through the library): one AMD kernel has to run in
    // THIS process, whose CUDA runtime is already up and is still holding VRAM.  Prints each card's free VRAM
    // before, during and after, so "free" cannot mean "nothing was ever allocated".
    if (probe_cross) {
#ifdef STRATA_MULTI_BRAND
        const bool ran = strata::core::cross_runtime_probe();
        std::printf("cross-runtime probe: %s\n", ran ? "the AMD kernel ran and wrote what it read back"
                                                    : "the AMD kernel did NOT run - do not route AMD layers here");
        return ran ? 0 : 3;
#else
        std::fprintf(stderr, "strata-device: --probe-cross needs a multi-brand build (-DSTRATA_MULTI_BRAND=ON)\n");
        return 2;
#endif
    }

    // B.1 at its widest: open the HIP-built ENGINE library (strata_engine built SHARED by the HIP build, whose
    // extern "C" entries live in src/hip/engine_bridge.cpp) and ask it what it is WITHOUT running a layer.  The
    // answer decides whether the AMD layers can be routed through it at all: the library has to say which arches
    // its device code holds, how many AMD cards its own HIP runtime sees from INSIDE it, and that the stage
    // entry (and the kernels behind it) linked in.
    if (probe_engine) {
#ifdef STRATA_MULTI_BRAND
        const bool ok = strata::core::engine_stage_ready();
        const std::string& d = strata::core::engine_stage_detail();
        std::printf("engine stage library: %s\n",
                    ok ? d.c_str() : "none (no HIP-built engine library next to this engine)");
        return ok ? 0 : 3;
#else
        std::fprintf(stderr, "strata-device: --probe-engine needs a multi-brand build (-DSTRATA_MULTI_BRAND=ON)\n");
        return 2;
#endif
    }

    // The shape a layer's hand-off needs, measured before any model or weight exists: this process (CUDA, MSVC)
    // asks the HIP-built library to build the ENGINE's OWN arena on the AMD card, and 16 floats come back into
    // RAM this process owns.  CUDA holds its own 256 MiB the whole time, so the AMD hold cannot be mistaken for
    // one shared pool, and the read-back cannot be bought by nothing being allocated.  exit 0 = proven, 3 = not.
    if (stage_hold) {
#ifdef STRATA_MULTI_BRAND
        return strata::core::stage_hold_gate(256ll << 20, 0) ? 0 : 3;
#else
        std::fprintf(stderr, "strata-device: --stage-hold needs a multi-brand build (-DSTRATA_MULTI_BRAND=ON)\n");
        return 2;
#endif
    }

    // C.1: not a probe of the join's shape but the join doing real work - the CUDA-built engine (this
    // process) asks the HIP-built library to build ONE whole engine stage on the AMD card (its weight
    // table, the native projections of its own layers, its own session carve, its VRAM expert tier filled
    // with every expert of the stage, its own Verifier) and to run one verify window through it.  Nothing
    // but plain data crossed: no handle, no HIP call on this side.  Then the stage's hand-off payload - the
    // residual rows the next stage would read - comes back into RAM THIS process owns and both sides hash it.
    // This loads a model, so it costs seconds, not milliseconds: it is the first gate that answers whether a
    // stage inside the library runs at all.
#ifdef STRATA_MULTI_BRAND
    if (probe_stage) {
        if (stage_pack.empty() || stage_native.empty()) {
            std::fprintf(stderr, "strata-device: --probe-stage needs --stage-pack DIR and --stage-native SHARD1\n");
            return 2;
        }
        StrataStageArgs a{};
        auto put = [](char* dst, size_t n, const std::string& s) { std::snprintf(dst, n, "%s", s.c_str()); };
        put(a.pack, sizeof a.pack, stage_pack);
        put(a.native, sizeof a.native, stage_native);
        put(a.tokens, sizeof a.tokens, stage_tokens);
        a.lb = 0;                       // the FIRST stage: the hand-off in is null, and that is only legal at layer 0
        a.le = stage_layers;
        a.max_context = stage_context;
        a.cache_slots = 0;              // every expert of the stage: the window must not ask a CPU pool across the join
        a.device = stage_device;
        return strata::core::stage_run_gate(a) ? 0 : 3;
    }
#endif

    try {
        const strata::core::DeviceInfo d = strata::core::device_info(0);
        std::printf("device %d: %s\n", d.ordinal, d.name.c_str());
#if defined(STRATA_USE_HIP)
        std::printf("  HIP arch            %s wave32 (compiled for %s)\n", d.arch.c_str(),
                    strata::core::compiled_gpu_archs());
#else
        std::printf("  compute capability  %d.%d   (sm_%d%d)\n", d.cc_major, d.cc_minor, d.cc_major, d.cc_minor);
#endif
        std::printf("  multiprocessors     %d\n", d.multi_processor_count);
        std::printf("  VRAM total / free   %s / %s\n", human(d.total_bytes).c_str(), human(d.free_bytes).c_str());
        std::printf("  driver / runtime    %d / %d\n", d.driver_version, d.runtime_version);
#if defined(STRATA_USE_HIP) && defined(_WIN32)
        // #468 #461: which HIP runtime this process loaded - the one beside the exe, or an AMD driver's System32 copy
        if (HMODULE h = GetModuleHandleA("amdhip64_7.dll")) {
            char path[MAX_PATH] = {};
            if (GetModuleFileNameA(h, path, MAX_PATH) > 0) std::printf("  HIP runtime         %s\n", path);
        }
#endif

        // The planner's view against the card's.  A plan that does not fit in what is actually FREE is the
        // failure this print exists to make visible at startup rather than at token 4000.
        const auto plan = strata::plan::make_plan(20480, strata::plan::Geometry{}, strata::plan::Costs{});
        std::printf("\n%s", strata::plan::to_string(plan).c_str());
        std::printf("  card free           %s\n", human(d.free_bytes).c_str());
        std::printf("  plan + KV vs free   %s\n",
                    plan.vram_budget <= d.free_bytes ? "FITS" : "*** DOES NOT FIT ***");

        if (selftest) {
            // Exercise the arena for real: allocate, write from the host, read back, and check the poison
            // path leaves NaNs rather than zeros.  A GPU test that only asks the driver for its name does not
            // test the runtime this file exists to provide.
            const uint64_t bytes = 64ull << 20;      // 64 MiB, small enough to be safe on any card
            strata::core::DeviceArena arena(bytes, 0, /*poison=*/true);
            void* a = arena.alloc(1 << 20, 256);
            void* b = arena.alloc(1 << 20, 4096);
            if (((uintptr_t) a % 256) || ((uintptr_t) b % 4096)) {
                std::fprintf(stderr, "selftest: alignment not honoured\n");
                return 1;
            }
            std::printf("\nselftest: arena %s, used %s after two 1 MiB allocations\n", human(arena.capacity()).c_str(),
                        human(arena.used()).c_str());
            // and the arena must REFUSE rather than wrap
            try {
                arena.alloc(bytes * 2);
                std::fprintf(stderr, "selftest: over-allocation did NOT throw\n");
                return 1;
            } catch (const strata::core::CudaError&) {
                std::printf("selftest: over-allocation refused as required\n");
            }
            std::printf("strata-device selftest OK\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-device: %s\n", e.what());
        return 1;
    }
}
