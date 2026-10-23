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
#include "strata/core/pinned.hpp"   // arena_pin_cap_gib(): C.4 reads the pin cap INSIDE this library too - the cap is per arena, not per process
#include "strata/core/layer.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

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
// what is measured is "held while CUDA holds its own", not "two pools that never end".  half (b): this entry
// is DEFINED below, beside the held-stage machinery, because it now releases a HELD stage too - the forward
// call at its old place would have named a function in the wrong scope.

// ================================ C.1: one REAL engine stage, inside this library ================================
//
// Not a probe: this builds one engine stage for a layer range the caller names, runs one verify window through
// it on the AMD card, and leaves the hand-off payload where the CUDA engine can take it into RAM IT owns.  The
// entry builds the model objects ITSELF, on the AMD card, and that is the measured reason this entry is not a
// shim: Verifier::init refuses without real data - output.weight must be in the table (verify.cpp:687-689),
// the VRAM expert tier must exist and be filled (verify.cpp:664-670), the geometry guard and the fused-gr check
// follow (:676-680) - and tests\** has not one reference to Verifier, SessionState, GdnBuffers, MoEBuffers or
// BlockBuffers (tests\hip\handoff.cpp exercises the hand-off at kernel level only), so the tree has no
// synthetic stage to build: the pack, the shards and the expert bytes are needed.  They are built here because
// the CUDA side could not build them at all: its cudaMalloc asks the NVIDIA card.
//
// The order below is the AMD-only program's own for one stage, call for call (src/program/generate.cpp):
//   :3233  expert_layout_load                :3406-3412 the geometry from the model's own metadata
//   :3501-3543 the native embedding, the skip set   :3601-3645 the weight arena and WeightTable::load
//   :3660-3662 NativeDense                    :4963-5000 session_bytes / session_init / session_zero
//   :5971  cache.open                         :6081-6095 admit + fill_slot_blocking from the pack
//   :6100  cache.verify_slot                  :7002-7024 the residency table (slot_of per pair, then ONE upload)
//   :8375-8399 the VerifyHits it wires        :8473-8475 set_stage / set_handoff_rows   :8493-8506 init
// What it is NOT is the program's plan: there is no VRAM search, no --expert-cache auto, no page-file
// complement and no CPU expert pool.  The caller asks for the slots it wants, a card that cannot hold them
// refuses here in the open, and a stage whose experts are NOT all in the VRAM tier is refused too (step 8) -
// hiding a CPU pool across the join would measure nothing.
namespace {

// The hand-off buffer this library owns, and the payload the last stage wrote into it.  Mapped pinned host
// memory: the engine's kernels read and write it through its DEVICE alias from inside a captured graph
// (verify.cpp:1076-1083 reads, :2090-2099 writes), and host RAM is the only channel the two runtimes share
// (hip_kernels_abi.hpp, docs/MULTI_GPU.md:63-65).  It lives here rather than on the caller's side so the
// CUDA engine can ask, after the stage is gone, for the floats the stage produced.
struct Payload {
    bool held = false;
    float* host = nullptr;    // the mapped host side: readable once the window's sync is back
    float* dev = nullptr;     // its device alias: what set_stage is given
    long long floats = 0;     // the floats the last window actually wrote (rows written, not rows held)
};
Payload g_payload;

void release_payload() {
    if (g_payload.held) {
        hipHostFree(g_payload.host);
        g_payload = Payload{};
    }
}

unsigned long long fnv1a64(const void* p, size_t n) {
    unsigned long long h = 1469598125565353939ull;
    for (const unsigned char* b = (const unsigned char*) p, * e = b + n; b != e; ++b)
        h = (((unsigned long long) (*b)) ^ h) * 10995116277794481ull;
    return h;
}

long long now_ms() {   // the clock the entry's own timings are taken from
    return (long long) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct OwnDevice {   // the hipMallocs this entry owns: freed whichever way the entry leaves
    std::vector<void*> p;
    ~OwnDevice() { for (void* v : p) hipFree(v); }
};
struct OwnHost {     // the mapped pinned buffers this entry owns, UNLESS the stage handed one over
    std::vector<void*> p;
    bool keep = false;
    ~OwnHost() { if (keep) return; for (void* v : p) hipHostFree(v); }
};

}  // namespace

// ---- half (b): the stage objects HELD between stage_build and stage_free -------------------------------
// C.1's entry built everything, ran ONE window and let go: it measures the crossing but cannot serve a decode,
// which replays the SAME built stage every window.  So the stage lives HERE, in this library, between the two
// entries - one stage at a time, released by strata_hip_stage_free - and the old body's order is kept exactly,
// split around its middle, so C.1's gate sees the same bytes, the same prints and the same card readings.
namespace {
struct StageHeld {
    OwnDevice own;                  // the arena, the session carve, the residency table: freed when the stage goes
    OwnHost own_host;               // the pinned hand-off; keep=true only while a payload of it is owed to a caller
    std::unique_ptr<strata::core::WeightTable> wt;
    std::unique_ptr<strata::core::NativeEmbed> embed;
    std::unique_ptr<strata::core::NativeDense> dense;
    std::unique_ptr<strata::core::SessionState> ss;
    std::unique_ptr<strata::core::ExpertCache> cache;
    std::unique_ptr<strata::core::Verifier> ver;
    strata::core::VerifyHits vh;               // points INTO the unique_ptrs: it lives exactly as long as they do
    strata::core::ModelGeometry g;             // the Verifier keeps a POINTER to it (verify.hpp:511): it must live
                                               //   with the stage, not on the build's stack - a held replay reads
                                               //   it every window, and a stack geometry died with the build
    std::vector<int32_t> host_res, picks;
    int32_t* d_res = nullptr;
    float* hb = nullptr;            // the mapped pinned hand-off buffer, held between windows
    float* hb_dev = nullptr;
    int HB = 0;                     // the stage's handoff_floats (Verifier::handoff_floats of its geometry)
    int n_expert = 0, device = -1;
    long long n_layers = 0, lb = 0, le = 0;
    int steps = 0;
    long long last_step_ms = 0;     // when the last step ended: the NEXT step's wait
    bool built = false;
};
StageHeld* g_stage = nullptr;

void stage_delete_partial() {       // a build/step that refused halfway: what it took goes back, the stage is gone
    delete g_stage;
    g_stage = nullptr;
}

// half (b): the held stage goes back on request - strata_hip_stage_free, DEFINED HERE because it must see
// the stage.  A payload still owed to the caller keeps its pinned buffer (the payload entry already set
// keep when it copied, this is the second sight of it); the arena and the session carve always go.
STRATA_HIP_ENGINE_EXPORT void strata_hip_stage_free() {
    if (g_stage) {
        if (g_payload.held && g_payload.host == g_stage->hb) g_stage->own_host.keep = true;
        delete g_stage;
        g_stage = nullptr;
    }
    g_arena.reset();
}
}  // anonymous

static int stage_build_into(const void* args, int args_size, void* out, int out_size);
static int stage_step_into(const void* args, int args_size, void* out, int out_size);
static void stage_measure_and_release(StrataStageOut* o);

// C.1's one-call entry, unchanged in what it measures: build, ONE window, the old card reading and the old
// prints, and let go - now produced by the two halves below, in the old order.
STRATA_HIP_ENGINE_EXPORT int strata_hip_stage_run(const void* args, int args_size, void* out, int out_size) {
    if (args == nullptr || out == nullptr ||
        (int) sizeof(StrataStageArgs) != args_size || (int) sizeof(StrataStageOut) != out_size)
        return -1;                        // the two sides do not agree on the block: nothing is read
    const int b = stage_build_into(args, args_size, out, out_size);
    if (b != 0) return b;
    const StrataStageArgs& a = *(const StrataStageArgs*) args;
    StrataStepArgs sa;                    // the same window's ids the args block named, as C.1 ran them
    std::snprintf(sa.tokens, sizeof(sa.tokens), "%s", a.tokens);
    StrataStepOut so;
    std::memset(&so, 0, sizeof(so));
    const int s = stage_step_into(&sa, (int) sizeof(sa), &so, (int) sizeof(so));
    if (s != 0) {
        stage_delete_partial();           // the old body left scope here too: objects back, no payload held
        return s;
    }
    StrataStageOut* o = (StrataStageOut*) out;
    o->window_runs = 1;
    o->rows = 8;                              // kMaxT: the buffer's capacity, printed as C.1 printed it
    o->handoff_floats = g_stage->HB;
    o->payload_bytes = so.payload_bytes;
    o->payload_hash = so.payload_hash;
    o->run_ms = so.run_ms;
    o->nan_count = 0;
    for (int i = 0; i < 8; ++i) {
        o->first[i] = g_stage->hb[i];
        if (o->first[i] != o->first[i]) ++o->nan_count;   // a NaN is the only value not equal to itself
    }
    o->read_back_ok = 1;
    stage_measure_and_release(o);         // the old last act: objects let go, the card read, then the prints
    return 0;
}

// half (b)'s entries: the stage is HELD between build and free, so a window costs only its replay.  A step
// without a built stage returns -2 with the reason in the block; so does a second build over a held stage.
STRATA_HIP_ENGINE_EXPORT int strata_hip_stage_build(const void* args, int args_size, void* out, int out_size) {
    if (g_stage) return -2;               // one stage at a time: the caller frees it first
    const int b = stage_build_into(args, args_size, out, out_size);
    if (b != 0 && g_stage) stage_delete_partial();
    return b;
}
STRATA_HIP_ENGINE_EXPORT int strata_hip_stage_step(const void* args, int args_size, void* out, int out_size) {
    if (!g_stage || !g_stage->built) {
        if (out) {
            StrataStepOut* o = (StrataStepOut*) out;
            std::memset(o, 0, sizeof(StrataStepOut));
            std::snprintf(o->err, sizeof(o->err), "no built stage: strata_hip_stage_build comes first");
        }
        return -2;
    }
    return stage_step_into(args, args_size, out, out_size);
}

// The build: C.1's body up to and including its step-8 check, objects in the held stage.  The window, the
// payload and the old card reading/prints are NOT here - the step and stage_measure_and_release keep them,
// in the old order, so the one-call entry above measures exactly as C.1 measured.
static int stage_build_into(const void* args, int args_size, void* out, int out_size) {
    if (args == nullptr || out == nullptr ||
        (int) sizeof(StrataStageArgs) != args_size || (int) sizeof(StrataStageOut) != out_size)
        return -1;                        // the two sides do not agree on the block: nothing is read
    const StrataStageArgs& a = *(const StrataStageArgs*) args;
    StrataStageOut* o = (StrataStageOut*) out;
    std::memset(o, 0, sizeof(StrataStageOut));
    const char* arch = strata::core::compiled_gpu_archs();
    std::snprintf(o->archs, sizeof(o->archs), "%s", arch ? arch : "");
    o->hip_devices = strata::core::device_count();
    release_payload();                    // a payload left over from an earlier call is not carried

    auto refuse = [&](int st, const std::string& why) {
        o->refused_step = st;
        std::snprintf(o->err, sizeof(o->err), "%s", why.c_str());
        std::fprintf(stderr, "strata hip engine: the stage refused at step %d: %s\n", st, why.c_str());
        return -100 - st;
    };

    const long long lb = a.lb, le = a.le;
    const int device = a.device;
    if (lb < 0 || le <= lb) return refuse(0, "the stage's layer range is empty or reversed");
    if (lb != 0) return refuse(0, "this entry builds the FIRST stage only: hand_in is null and C.1 runs layers [0, K)");
    size_t total = 0;
    if (hipSetDevice(device) != hipSuccess) return refuse(10, "this library's HIP runtime has no device " + std::to_string(device));
    o->device = device;
    hipMemGetInfo(&o->free_before, &total);

    std::string err;
    int step = 1;
    g_stage = new StageHeld;                    // the held stage: every allocation from here belongs to it
    OwnDevice& own = g_stage->own;
    OwnHost& own_host = g_stage->own_host;
    float*& hb = g_stage->hb;
    float*& hb_dev = g_stage->hb_dev;

    // the window's ids are the STEP's now (half (b)): the build never looks at them, so a stage can be built
    // once and replayed every window.  The buffer still holds kMaxT rows, as C.1 sized it.
    const int kMaxT = 8;                        // strata::kernels::kVerifyMaxT, include/strata/kernels/verify_kernels.hpp:21

    strata::core::ModelGeometry& g = g_stage->g;
    std::vector<std::string> shards;
    int64_t K = 0;
    try {
        shards = strata::gguf_split_paths(a.native);
    } catch (const std::exception& e) {
        return refuse(1, e.what());
    }
    if (shards.empty()) return refuse(1, "no model shards beside " + std::string(a.native));
    try {
        const strata::GgufFile meta(shards.front());
        std::string gerr;
        if (!strata::core::geometry_from_gguf(meta, g, K, gerr)) return refuse(1, gerr);
        const std::string mismatch = strata::core::compiled_geometry_mismatch(g);
        if (!mismatch.empty()) return refuse(1, mismatch);
    } catch (const std::exception& e) {
        return refuse(1, e.what());
    }
    using namespace strata::core;
    o->n_layers = (int) g.n_layers;
    o->n_expert = (int) g.n_expert;
    o->hc = (int) g.hc;
    o->n_embd = (int) g.n_embd;
    o->lb = (int) lb;
    o->le = (int) le;
    if (le > g.n_layers) return refuse(1, "the range runs past the model's trunk");

    // ---- the arithmetic contract the window reproduces.  Verifier::init refuses unless the process-wide
    // switches name ONE complete configuration (layer_verify_compatible, src/core/layer.cpp:483-492, called at
    // verify.cpp:672) - and every one of these defaults to OFF in the library, because the PROGRAM sets them:
    // this list is the AMD-only run's, call for call (src/program/generate.cpp:3034-3037 what --native turns on,
    // and :3670-3689 what it hands to the setters), so the stage below runs the SAME per-token arithmetic as
    // build-hip-mb\strata.exe does on the same pack - which is what C.2's parity check needs, and not a shape
    // the library made up.  gr_set_fp32_activations stays off as in that run: gr_set_native_mmvf implies it
    // (include/strata/kernels/gr.hpp:34), and the preset never turns on the <=256-token attention adapter
    // (:464 "never the <=256-token attention adapter").
    strata::kernels::gr_set_fp32_activations(false);
    strata::kernels::gr_set_native_mmvf(true);
    strata::core::layer_set_fast_attn(true);
    strata::core::layer_set_publish_kernel(true);
    strata::core::layer_set_fused_gdn(true);
    strata::core::layer_set_fast_select(true);
    strata::core::layer_set_fused_gr(true);      // gr_native_mmvf and none of the four opt-outs, as the daily run
    strata::core::layer_set_native_bf16(true);
    strata::core::layer_set_native_flash_attn_short(false);
    strata::kernels::ple_set_native_bf16(true);
    strata::kernels::shared_expert_set_native_bf16(true);
    strata::kernels::native_moe_combine_set_enabled(true);
    strata::kernels::native_gdn_set_enabled(true);
    strata::kernels::native_router_set_enabled(true);
    strata::kernels::native_qsa_set_enabled(true);
    strata::kernels::native_qsa_indexer_set_enabled(true);
    strata::kernels::native_rope_set_enabled(true);
    strata::kernels::ple_set_native_postops(true);

    const long long t_build0 = now_ms();
    try {
        // ---- 2. where this pack's experts live: a native pack's blobs differ per layer
        step = 2;
        if (!strata::kernels::cpu::expert_layout_load(a.pack, g.n_layers, g.n_expert, err)) return refuse(2, err);
        const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();

        // ---- 3. the model's own objects, on this card
        step = 3;
        std::set<std::string> skip;
        if (!NativeDense::served_names(shards, true, skip, err)) return refuse(3, err);
        (void) NativeDense::keep_unquantized_ple_key(a.pack, skip, err);
        err.clear();
        if (lay.native) skip.insert("token_embd.weight");     // the canonical row is IQ: the embedding below reads the GGUF
        skip.insert("output.weight");                         // the head belongs to the LAST stage; this one never runs it
        const bool hc_q8 = NativeDense::hc_q8_requested();    // auto = on for gfx12: mirror the AMD-only run
        if (hc_q8) NativeDense::add_hc_skip(a.pack, skip);

        g_stage->embed = std::make_unique<NativeEmbed>();
        NativeEmbed& embed = *g_stage->embed;
        if (!embed.load(shards, g.n_embd, g.n_vocab, err)) return refuse(3, err);
        o->embed_bytes = (long long) embed.bytes();
        set_native_embed(&embed);                             // verify.cpp:1084 asks for it: an IQ row is not readable otherwise

        uint64_t pool_bytes = 0;
        if (!WeightTable::pool_bytes(a.pack, pool_bytes, err, &skip)) return refuse(3, err);
        void* arena = nullptr;
        if (hipMalloc(&arena, (size_t) pool_bytes) != hipSuccess) return refuse(10, "the pack's dense weights do not fit on this card");
        own.p.push_back(arena);
        o->arena_bytes = (long long) pool_bytes;
        g_stage->wt = std::make_unique<WeightTable>();
        WeightTable& wt = *g_stage->wt;
        if (!wt.load(a.pack, arena, pool_bytes, err, &skip)) return refuse(3, err);
        // WeightTable::load_hc_q8 (the STRATA_HC_REQ8 shape, generate.cpp:3594 and :3649-3651) is NOT
        // mirrored: that env is unset on this rig, and the two shapes are alternatives - converting the
        // row here to int8 + fp32/32 scales AND skipping its bf16 copy leaves NativeDense a row that is
        // neither the arena's bf16 nor the pack's, and it refuses (measured: bytes 3686400 for a row of
        // 6553600 bf16 bytes, resident 1, and "is not a bf16 matrix").  STRATA_HC_Q8's own shape is the
        // add_hc_skip above plus NativeDense quantizing down / up from the pack file (:429-435).
        o->hc_q8_bytes = 0;             // the REQ8 copy: not asked for, so nothing of that form exists

        // ---- 4. the native projections of THIS stage's layers only (the rest belong to another stage)
        step = 4;
        NativeDense::set_layer_range((int) lb, (int) le);
        g_stage->dense = std::make_unique<NativeDense>();
        NativeDense& dense = *g_stage->dense;
        if (!dense.load(shards, wt, err, true, lb, le)) return refuse(4, err);
        o->dense_bytes = (long long) dense.allocated_bytes();

        // ---- 5. the session carve of [lb, le): this stage's own QSA and GDN state, not the model's
        step = 5;
        const int64_t cells = a.max_context > 0 ? a.max_context : 1024;
        const uint64_t sb = session_bytes(g, cells, K, lb, le);
        void* sbuf = nullptr;
        if (hipMalloc(&sbuf, (size_t) sb) != hipSuccess) return refuse(10, "the session carve does not fit on this card");
        own.p.push_back(sbuf);
        o->session_bytes = (long long) sb;
        g_stage->ss = std::make_unique<SessionState>();
        SessionState& ss = *g_stage->ss;
        session_init(g, cells, K, sbuf, ss, lb, le);
        session_zero(ss, g, nullptr, nullptr);

        // ---- 6. the VRAM expert tier, and EVERY expert of the stage in it
        step = 6;
        const long long want = (le - lb) * g.n_expert;
        const long long n_slots = a.cache_slots > 0 ? a.cache_slots : want;
        if (n_slots < want) return refuse(6, "asked for fewer slots than the stage's experts: the window would need a CPU pool");
        g_stage->cache = std::make_unique<ExpertCache>();
        ExpertCache& cache = *g_stage->cache;
        if (!cache.open(n_slots, g.n_layers, g.n_expert, (int64_t) lay.max_blob, err)) return refuse(6, err);
        o->cache_bytes = cache.bytes();
        FileExpertSource src;
        src.set_gguf(a.native);
        if (!src.open(a.pack, g.n_layers, g.n_expert, err)) return refuse(6, err);
        const long long f0 = now_ms();
        for (long long l = lb; l < le; ++l) {
            for (long long e = 0; e < g.n_expert; ++e) {
                const int32_t slot = cache.admit(l, e);
                if (slot == kNotResident) return refuse(6, "the tier filled up before the stage's experts did");
                const uint8_t* b = src.blob_stable(l, e);
                if (b == nullptr) return refuse(6, "the pack does not hold expert (" + std::to_string(l) + ", " +
                                                    std::to_string(e) + ")");
                if (!cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(l))) return refuse(6, err);
                if (l == lb && e == 0 && !cache.verify_slot(slot, b, err, (int64_t) lay.blob_bytes(l)))
                    return refuse(6, err);            // one slot is read back from the card, like the AMD-only fill does
                ++o->resident_experts;
            }
        }
        o->fill_s = (double) (now_ms() - f0) / 1000.0;

        // ---- 7. the residency table the window's plans come from, and the Verifier that runs the stage
        step = 7;
        std::vector<int32_t>& host_res = g_stage->host_res;
        host_res.assign((size_t) g.n_layers * (size_t) g.n_expert, kNotResident);
        for (long long l = lb; l < le; ++l)
            for (long long e = 0; e < g.n_expert; ++e)
                host_res[(size_t) l * g.n_expert + e] = cache.slot_of(l, e);
        int32_t*& d_res = g_stage->d_res;
        if (hipMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != hipSuccess) return refuse(10, "the residency table does not fit");
        own.p.push_back(d_res);
        if (hipMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), hipMemcpyHostToDevice) != hipSuccess)
            return refuse(10, "the residency table would not upload");
        VerifyHits& vh = g_stage->vh;
        vh.d_res = d_res;
        vh.h_res = host_res.data();
        vh.cache_base = cache.device_slot(0);
        vh.slot_off = cache.slot_offsets();
        vh.n_slots = cache.slots();
        vh.blob = (int64_t) lay.max_blob;

        // ---- the hand-off buffer: the library owns it, mapped pinned, and hands its device alias to the stage
        const long long HB = Verifier::handoff_floats(g);
        const long long hb_bytes = (long long) kMaxT * HB * sizeof(float);
        if (hipHostMalloc((void**) &hb, (size_t) hb_bytes, hipHostMallocMapped | hipHostMallocPortable) != hipSuccess)
            return refuse(10, "the hand-off buffer does not fit in pinned host memory");
        own_host.p.push_back(hb);
        if (hipHostGetDevicePointer((void**) &hb_dev, hb, 0) != hipSuccess)
            return refuse(10, "the hand-off buffer has no device alias on this card");

        g_stage->ver = std::make_unique<Verifier>();
        Verifier& ver = *g_stage->ver;
        ver.set_stage(lb, le, nullptr, hb_dev);       // the FIRST stage: no hand-off in, and its own buffer out
        if (!ver.init(wt, g, ss, vh, nullptr, kMaxT, err)) return refuse(7, err);
        g_stage->HB = (int) HB;                       // the step's payload size comes from here on
        g_stage->n_layers = g.n_layers;
        g_stage->lb = lb;
        g_stage->le = le;
        g_stage->device = o->device;

        // ---- 8. is the stage really all in VRAM?  If not, the window asks a host pool the join does not have.
        step = 8;
        o->zero_doorbell = ver.ar_live() ? 1 : 0;
        if (!o->zero_doorbell)
            return refuse(8, "not every expert of the stage is in the VRAM tier: the window would ask a CPU expert "
                             "pool across the join, and that is C.5's, not something this gate may hide");

        // the window is the STEP's now (half (b)): the build ends with its step-8 check above, HOLDING the
        // stage, so a real run replays these graphs every window instead of rebuilding them.
        g_stage->built = true;
        o->handoff_floats = (int) HB;
        o->rows = 8;                          // kMaxT: the buffer's capacity
        o->build_s = (double) (now_ms() - t_build0) / 1000.0;
        {
            size_t held_total = 0;
            hipMemGetInfo(&o->free_after, &held_total);   // the card WHILE the stage is held: what a decode would
        }                                                 //   start from
        return 0;
    } catch (const std::exception& e) {
        return refuse(step, e.what());
    }
}

// One window replayed on the HELD stage: the ids cross in, the payload's bytes get hashed HERE (the C.1 rule,
// per window), and the caller takes them through strata_hip_stage_payload as before.
static int stage_step_into(const void* args, int args_size, void* out, int out_size) {
    if (args == nullptr || out == nullptr ||
        (int) sizeof(StrataStepArgs) != args_size || (int) sizeof(StrataStepOut) != out_size)
        return -1;                        // the two sides do not agree on the block: nothing is read
    const StrataStepArgs& a = *(const StrataStepArgs*) args;
    StrataStepOut* o = (StrataStepOut*) out;
    std::memset(o, 0, sizeof(StrataStepOut));
    if (!g_stage || !g_stage->built) {
        std::snprintf(o->err, sizeof(o->err), "no built stage: strata_hip_stage_build comes first");
        return -2;
    }
    release_payload();                    // the last window's payload is not carried into this one
    std::vector<int32_t> toks;
    for (const char* q = a.tokens; q != nullptr && *q != '\0'; ) {
        char* end = nullptr;
        const long long v = std::strtoll(q, &end, 10);
        if (end == nullptr || end == q) break;
        toks.push_back((int32_t) v);
        q = end;
        while (*q == ',' || *q == ' ') ++q;
    }
    const int kMaxT = 8;                  // strata::kernels::kVerifyMaxT, as the build sized the buffer
    const int T = toks.size() > (size_t) kMaxT ? kMaxT : (int) toks.size();
    if (T < 1) {
        std::snprintf(o->err, sizeof(o->err), "no token ids in the step block");
        return -100;
    }
    auto& gs = *g_stage;
    std::vector<int32_t>& picks = gs.picks;
    picks.assign((size_t) T, 0);
    std::string err;
    const long long w0 = now_ms();
    if (!gs.ver->run(T, toks.data(), 0, nullptr, nullptr, picks.data(), err)) {
        std::snprintf(o->err, sizeof(o->err), "%s", err.c_str());
        std::fprintf(stderr, "strata hip engine: the step refused: %s\n", err.c_str());
        return -109;
    }
    o->run_ms = (double) (now_ms() - w0);            // raw ms, as C.1's window was timed
    o->wait_ms = gs.last_step_ms ? (double) (w0 - gs.last_step_ms) : 0.0;
    gs.last_step_ms = now_ms();
    o->steps_total = ++gs.steps;
    o->rows = T;
    const long long floats = (long long) T * gs.HB;
    o->payload_bytes = floats * (long long) sizeof(float);
    o->payload_hash = fnv1a64(gs.hb, (size_t) (floats * (long long) sizeof(float)));
    g_payload.held = true;
    g_payload.host = gs.hb;
    g_payload.dev = gs.hb_dev;
    g_payload.floats = floats;
    std::fprintf(stderr, "strata hip engine:   step %lld on the held stage: %d rows, %lld B of hand-off, the "
                         "library's hash %016llx, the window %.1f ms, %0.1f ms since the last one ended\n",
                 (long long) o->steps_total, o->rows, (long long) o->payload_bytes, o->payload_hash,
                 o->run_ms, o->wait_ms);
    return 0;
}

// C.1's last act, kept in the old order for the one-call entry: the stage's objects let go first (their own
// hipFrees), THEN the card is read with the arena and the session carve STILL held (own gives them back at
// the delete below, where the old body gave them back at return), then the old prints.  The pinned hand-off
// stays because the payload still owes it to the caller.
static void stage_measure_and_release(StrataStageOut* o) {
    g_stage->ver.reset();
    g_stage->cache.reset();
    g_stage->ss.reset();
    g_stage->dense.reset();
    g_stage->embed.reset();
    g_stage->wt.reset();
    strata::core::set_native_embed(nullptr);   // the embedding is about to leave: the process-wide pointer must not dangle
    strata::core::NativeDense::set_layer_range(-1, -1);  // the process-wide range too
    size_t total = 0;
    hipMemGetInfo(&o->free_after, &total);    // the stage's objects are gone: "built" cannot be silent
    g_stage->own_host.keep = true;      // the payload's buffer: owed to the caller until it has copied it
    std::fprintf(stderr, "strata hip engine: the stage ran INSIDE the library on device %d of %d AMD cards, arches %s: "
                         "layers [%lld, %lld) of %lld\n", o->device, o->hip_devices, o->archs,
                         g_stage->lb, g_stage->le, g_stage->n_layers);
    std::fprintf(stderr, "strata hip engine:   the weights took %.3f GiB of arena + %.3f GiB of native projections "
                         "(this stage only) + %.3f GiB of Q8 hyper-connections; the expert tier %.3f GiB, %d experts; "
                         "the session carve %.3f MiB; the token table %.1f MiB of host\n",
                         (double) o->arena_bytes / 1073741824.0, (double) o->dense_bytes / 1073741824.0,
                         (double) o->hc_q8_bytes / 1073741824.0, (double) o->cache_bytes / 1073741824.0,
                         o->resident_experts, (double) o->session_bytes / 1048576.0,
                         (double) o->embed_bytes / 1048576.0);
    std::fprintf(stderr, "strata hip engine:   the card: %.3f GiB free before the stage, %.3f GiB after it; the load "
                         "%.1f s, the tier fill %.1f s, the window %.1f ms\n",
                         (double) o->free_before / 1073741824.0, (double) o->free_after / 1073741824.0,
                         o->build_s, o->fill_s, o->run_ms);
    std::fprintf(stderr, "strata hip engine:   the hand-off: %d floats per row, the mapped buffer holds %d rows, "
                         "the window wrote %lld of them = %lld B, the library's hash of those bytes %016llx\n",
                         o->handoff_floats, o->rows,
                         (o->handoff_floats > 0 ? o->payload_bytes / ((long long) o->handoff_floats * (long long) sizeof(float)) : 0LL),
                         o->payload_bytes, o->payload_hash);
    stage_delete_partial();               // the arena and the session carve go back to the card here
}

// Take the payload into RAM the CALLER owns.  The library's mapped buffer is its own: the CUDA engine never
// sees it, only the floats this copies.  Returns the number of floats copied (negative: nothing is held), and
// lets go of the buffer, so a second call finds nothing - the gate hashes the copy it received and compares it
// against the hash the library reported, which is the crossing measured at the byte level.
STRATA_HIP_ENGINE_EXPORT int strata_hip_stage_payload(float* dst, int dst_floats) {
    if (dst == nullptr || !g_payload.held || dst_floats < 1) return -1;
    const long long n = g_payload.floats < (long long) dst_floats ? g_payload.floats : (long long) dst_floats;
    std::memcpy(dst, g_payload.host, (size_t) n * sizeof(float));
    if (g_stage && g_payload.host == g_stage->hb) g_stage->own_host.keep = true;  // the held stage's pinned buffer:
    release_payload();                                                            // freed here, and stage_free must not
    return (int) n;                                                               // free it a second time
}

// ================================ C.4: ONE ARENA PER RUNTIME ================================
//
// A mixed process has TWO runtimes up, and one arena serves ONE of them: the CUDA engine's DeviceArena
// belongs to the CUDA runtime, this library's belongs to the HIP runtime. So the library must be able to
// hold and release a block in ITS own arena on ITS own AMD card, through these entries - the CUDA loader
// cannot call HIP, so it asks. No poison: this entry measures the arena and the card's memory, not the
// stage; the stage entry above already proved the allocator and the kernel on this same card.
//
// The card is NAMED before the hold (device_summary): a refusal must be able to print which card said
// no, with that card's free/total - generate.cpp:164's shape ` (%llu MiB of %llu MiB VRAM free on this
// GPU)`. And cap_gib is what THIS side read: the WDDM cap (docs/MULTI_GPU.md:340-343, STRATA_ARENA_PIN_GIB)
// is per arena, so each side's line carries its own.
STRATA_HIP_ENGINE_EXPORT int strata_hip_arena_hold(long long bytes, int device, void* out, int expect_size) {
    StrataArenaReport* o = (StrataArenaReport*) out;
    if (o == nullptr || (int) sizeof(StrataArenaReport) != expect_size) return -1;
    std::memset(o, 0, sizeof(StrataArenaReport));
    o->bytes = bytes;
    o->device = device;
    o->cap_gib = strata::core::arena_pin_cap_gib();   // read INSIDE this library, printed as-is
    size_t total = 0;
    if (hipSetDevice(device) != hipSuccess) return -2;   // this runtime has no such device
    std::string name, detail;
    if (strata::core::device_summary(device, name, detail))   // named BEFORE the hold, so a refusal can name it
        std::snprintf(o->card, sizeof(o->card), "%s", name.c_str());
    if (hipMemGetInfo(&o->free_before, &total) == hipSuccess) o->total_bytes = (unsigned long long) total;
    try {
        g_arena = std::make_unique<strata::core::DeviceArena>((uint64_t) bytes, device, /*poison=*/false);
        o->arena_ok = 1;
        o->held_bytes = g_arena->capacity();
    } catch (const strata::core::CudaError&) {
        hipMemGetInfo(&o->free_after, &total);   // the refusal still carries this card's numbers
        return -2;
    } catch (const std::exception&) {
        hipMemGetInfo(&o->free_after, &total);
        return -2;
    }
    hipMemGetInfo(&o->free_after, &total);
    return 0;
}

// Let go: the arena's own destructor gives the card's memory back - what is measured is then the card again,
// so "held" cannot be silent, same rule as strata_hip_stage_free above.
STRATA_HIP_ENGINE_EXPORT void strata_hip_arena_free() { g_arena.reset(); }

// CODE.md C.5: hash the named expert pairs INSIDE this library - the library's OWN expert tier on the AMD
// card, filled from the pack by the library's OWN objects.  Nothing of the caller's RAM arena is read here:
// the CUDA host named these pairs BEFORE its own side existed, and this library's copy of the layout stays
// the FULL one (the drop is applied to the host's copy only - experts.bin keeps its own layout, `file_total`).
// Each slot's bytes come back through this runtime's hipMemcpy and are hashed in this library, so the number
// can only be the AMD tier's - the host compares it against its RAM arena (present pairs) and against what
// it fetched from the NVIDIA tier (dropped pairs).
STRATA_HIP_ENGINE_EXPORT int strata_hip_expert_hash(const void* args, int args_size, void* out, int out_size) {
    const StrataExpertPairs* a = (const StrataExpertPairs*) args;
    StrataExpertOut* o = (StrataExpertOut*) out;
    if (a == nullptr || (int) sizeof(StrataExpertPairs) != args_size || o == nullptr ||
        (int) sizeof(StrataExpertOut) != out_size) return -1;
    std::memset(o, 0, sizeof(StrataExpertOut));
    const char* arch = strata::core::compiled_gpu_archs();
    std::snprintf(o->archs, sizeof o->archs, "%s", arch ? arch : "");
    o->n_pairs = a->n_pairs;
    if (a->n_pairs <= 0 || a->n_pairs > 32) {
        std::snprintf(o->err, sizeof o->err, "expert hash: asked for %d pairs, 1..32 possible", a->n_pairs);
        return -100;
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(a->pack, a->n_layers, a->n_expert, err)) {
        std::snprintf(o->err, sizeof o->err, "%s", err.c_str());
        return -100;
    }
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    o->n_layers = (long long) lay.n_layers;
    o->n_expert = (long long) lay.n_expert;
    if (hipSetDevice(a->device) != hipSuccess) {
        std::snprintf(o->err, sizeof o->err, "this runtime has no such AMD card: %d", a->device);
        return -100;
    }
    std::string name, detail;
    if (strata::core::device_summary(a->device, name, detail))   // named BEFORE anything took the card
        std::snprintf(o->card, sizeof o->card, "%s", name.c_str());
    size_t total = 0;
    hipMemGetInfo(&o->free_before, &total);

    {   // the tier's whole life: free/after is read with the card given back
        strata::core::ExpertCache cache;
        if (!cache.open(a->n_pairs, lay.n_layers, lay.n_expert, (int64_t) lay.max_blob, err)) {
            std::snprintf(o->err, sizeof o->err, "%s", err.c_str());
            return -100;
        }
        o->cache_bytes = cache.bytes();
        strata::core::FileExpertSource src;
        src.set_gguf(a->native);
        if (!src.open(a->pack, lay.n_layers, lay.n_expert, err)) {
            std::snprintf(o->err, sizeof o->err, "%s", err.c_str());
            return -100;
        }
        std::vector<uint8_t> back;
        const long long f0 = now_ms();
        for (int i = 0; i < a->n_pairs; ++i) {
            const int32_t l = a->pairs[2 * i], e = a->pairs[2 * i + 1];
            const int32_t slot = cache.admit(l, e);
            if (slot == strata::core::kNotResident) {
                std::snprintf(o->err, sizeof o->err, "the tier refused expert (%d, %d)", l, e);
                return -100;
            }
            const uint8_t* b = src.blob_stable(l, e);
            if (b == nullptr) {
                std::snprintf(o->err, sizeof o->err, "the pack does not hold expert (%d, %d)", l, e);
                return -100;
            }
            const long long bb = (long long) lay.blob_bytes(l);
            if (!cache.fill_slot_blocking(slot, b, err, bb) || !cache.verify_slot(slot, b, err, bb)) {
                std::snprintf(o->err, sizeof o->err, "%s", err.c_str());
                return -100;
            }
            o->blob_bytes[i] = bb;
            back.assign((size_t) bb, 0);
            if (hipMemcpy(back.data(), cache.device_slot(slot), (size_t) bb, hipMemcpyDeviceToHost) != hipSuccess) {
                std::snprintf(o->err, sizeof o->err, "the slot would not come back from the card");
                return -100;
            }
            // NOT the bridge's local fnv1a64 above: that one multiplies by 10995116277794481 and seeds
            // 1469598125565353939 (it is the STAGE parity's pair - the loader's local lambda matches it).
            // The C.5 sites on the loader hash with pinned.hpp's fnv1a64 (seed 1469598103934665603, times
            // 1099511628211): the SAME hash FUNCTION is the claim here, so THIS library calls THAT function.
            o->hash[i] = strata::core::fnv1a64(back.data(), back.size());
        }
        o->fill_s = (double) (now_ms() - f0) / 1000.0;
    }
    hipMemGetInfo(&o->free_after, &total);
    o->ran = 1;
    return 0;
}
