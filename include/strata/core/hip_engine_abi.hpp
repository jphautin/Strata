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

#define STRATA_HIP_ENGINE_ABI_VERSION 5   // 5 = the stage entry split into BUILD and STEP (half (b): a stage must
                                          //   persist across decode windows, and C.1's one-call entry rebuilt and ran
                                          //   one window; v4 and older libraries refuse)

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

// C.4: what ONE ARENA of a runtime says about one hold, as plain data.  A mixed build now has one arena PER
// runtime - this process's own DeviceArena, and the library's own inside its HIP runtime - so a report can
// only ever carry the numbers of the arena that FILLED it, never the other side's.  The card is named BEFORE
// the hold (device_summary), so an allocation refusal can print the card and that card's free/total, in
// generate.cpp:164's shape `(N of M MiB VRAM free on this GPU)`.  cap_gib is the value arena_pin_cap_gib()
// read ON THAT SIDE - the WDDM cap is per arena, not per process (docs/MULTI_GPU.md:340-343).
struct StrataArenaReport {
    long long bytes = 0;                     // what was asked of that arena
    long long held_bytes = 0;                // what that arena actually took (0 = it refused)
    int device = -1;                         // the card it lives on, as THAT runtime numbers its cards
    int arena_ok = 0;                        // 1 = that arena exists in that runtime
    long long cap_gib = -1;                  // arena_pin_cap_gib() read there: -1 unset / -2 auto / 0 the whole arena / N GiB
    char card[48] = "";                      // the card's name, filled BEFORE the hold, so a refusal can name it
    unsigned long long free_before = 0;      // that card's free VRAM before the hold
    unsigned long long free_after = 0;       // and after (0 = the hold never happened)
    unsigned long long total_bytes = 0;      // the card's total, for the print
};

// C.5: the expert bytes ONE hash may ask about, as plain data.  The CUDA host named these pairs BEFORE it
// built its own holed RAM arena and NVIDIA tier, so the library cannot pick the easy ones: the list carries
// dropped and present pairs on purpose.  `native` is empty when the pack carries experts.bin.
struct StrataExpertPairs {
    char pack[256];
    char native[512];
    long long n_layers, n_expert;  // the geometry the CALLER read from the model's metadata: the library's layout load must agree
    int32_t pairs[64];       // (layer, expert) pairs, flat; up to 32 of them
    int n_pairs;
    int device;              // the AMD ordinal INSIDE the library's own HIP runtime
};

// What the library measured, as numbers only: each pair's FNV-1a over the bytes ITS OWN expert tier held
// (filled from experts.bin inside the library, on the AMD card), that card's name and free/total, and the
// cache's size.  The host compares these hashes against its RAM arena's bytes (present pairs) and against
// what it fetched back from the NVIDIA tier (dropped pairs): the SAME expert bytes, hashed at three sites.
struct StrataExpertOut {
    char archs[64];                  // the arches this library's device code holds
    char card[48];                   // the AMD card it ran on, by name
    char err[256];                   // what refused, in the library's own words (empty = it ran)
    int ran;                         // 1 = the tier was built and EVERY named pair hashed
    int n_pairs;                     // pairs hashed back (the caller's n, echoed)
    unsigned long long hash[32];     // FNV-1a per pair, over the slot's bytes read back from the card
    long long blob_bytes[32];        // each pair's byte count, so the host can size its own compare
    long long cache_bytes;           // the library's ExpertCache::bytes()
    long long n_layers, n_expert;    // the layout the library loaded for the pack
    unsigned long long free_before, free_after;   // the card's free VRAM around the tier
    double fill_s;                   // the tier's fill, from experts.bin
};

// ================================ C.1: ONE REAL ENGINE STAGE, INSIDE THIS LIBRARY ================================
//
// What the CUDA engine asks the library to do, as plain data: where the model is, which layer range this stage
// owns, the window's ids, and which AMD card.  Everything else the stage needs is built INSIDE the library, on
// the AMD card, by the library's own objects - the weight table, the native projections of ITS layers, the
// session carve of that range, the VRAM expert tier filled from the pack, and the Verifier whose graphs it then
// replays.  Those objects hold device pointers and std::strings, and a CUDA-side cudaMalloc would ask the
// NVIDIA card, so not one of them may be made on the caller's side and read here: only this block crosses.
//
// The shape is the tree's own, read before writing (CODE.md Appendix A): a stage IS a Verifier
// (include/strata/core/verify.hpp:79) plus set_stage(lb, le, hand_in, hand_out) (:173), and the payload per row
// is hc*n_embd + n_embd + hc floats - the residual R, the previous stage's pending write bo, and inject
// (handoff_floats, :194).  Both hand-off pointers are read and written INSIDE a captured graph as device-
// visible addresses (verify.cpp:1076-1083 reads, :2090-2099 writes), so the library owns one mapped pinned
// buffer, and the payload comes back into the caller's RAM by a copy the library does after the window - the
// only channel two runtimes share is host RAM (hip_kernels_abi.hpp and docs/MULTI_GPU.md:63-65).
struct StrataStageArgs {
    char pack[256];          // the pack dir (index.txt, dense.bin, native_experts.txt)
    char native[512];        // shard 1 of the model GGUF, as --native names it (gguf_split_paths finds the rest)
    char tokens[64];         // the window's token ids, comma-separated as --tokens writes them
    long long lb, le;        // the stage's layers [lb, le); C.1 runs the FIRST one, so lb = 0 and hand_in is null
    long long max_context;   // cells of session state to carve for that range
    long long cache_slots;   // expert slots; 0 = every expert of the stage, (le - lb) * n_expert
    int device;              // the ordinal INSIDE the library's own HIP runtime, as --list-runtimes numbers them
};

// What the stage measured, as numbers only: the library's arches, its card, the bytes each object took, the
// time each step cost, and the hand-off payload's size and hash.  Nothing here is a handle - the CUDA engine
// must not call HIP (the rule at :36-38 above), so it gets a report, and the floats themselves come back
// through strata_hip_stage_payload below, into RAM the engine owns and hashes.
struct StrataStageOut {
    char archs[64];                  // the arches this library's device code holds ("gfx1201,gfx1200")
    char err[256];                   // what refused, in the engine's own words (empty = the stage ran)
    long long arena_bytes;           // cudaMalloc for the pack's dense weights, inside the library
    long long dense_bytes;           // NativeDense's own allocations: this stage's layers only
    long long hc_q8_bytes;           // the arena's int8 + fp32/32-scale copy (STRATA_HC_REQ8): 0 here, as that
    // the env is unset on this rig; STRATA_HC_Q8's own copy is NativeDense's and lands in dense_bytes
    long long embed_bytes;           // NativeEmbed's token table, in mapped HOST memory (costs no VRAM)
    long long session_bytes;         // the session carve of [lb, le) only - not the whole model's
    long long cache_bytes;           // ExpertCache::bytes(): the VRAM expert tier this stage runs against
    long long payload_bytes;         // rows * handoff_floats * 4 the stage wrote into its mapped buffer
    unsigned long long free_before;  // this AMD card's free VRAM before any of it was built
    unsigned long long free_after;   // and after the stage's objects were released, so "built" cannot be silent
    long long window_runs;           // windows this entry ran (the gate runs one)
    int hip_devices;                 // AMD cards the library sees from inside its own HIP runtime
    int device;                      // the card it ran on
    int n_layers, n_expert;          // the geometry the library read from the model's own metadata
    int hc, n_embd;
    int lb, le;                      // the range it actually ran
    int rows;                        // rows of hand-off the mapped buffer holds (kVerifyMaxT unless asked)
    int handoff_floats;              // floats per row (verify.hpp:194): the caller sizes its own buffer from this
    int resident_experts;            // (layer, expert) pairs this stage admitted into the VRAM tier
    int zero_doorbell;               // 1 = ar_live(): EVERY expert of the stage is in VRAM, so no CPU pool was asked
    int read_back_ok;                // 1 = the library read its own payload back; it is what got hashed
    int refused_step;                // 0 ran; see the list below in strata_hip_stage_run's comment
    int nan_count;                   // NaNs among the first 8 payload floats (0 expected: a stage writes numbers)
    unsigned long long payload_hash; // FNV-1a over the payload bytes, computed by the LIBRARY
    double build_s;                  // the load: weights, projections, session
    double fill_s;                   // the expert tier filled from the pack
    double run_ms;                   // the window itself
    float first[8];                  // the first eight payload floats, printed as they came back
};

// C.6 half (b): the tokens of ONE window, as the caller writes them (decode: one id; a draft tree: up to
// strata::kernels::kVerifyMaxT).  A step never builds: it replays the graphs of an ALREADY built stage - the
// per-token contract the mixed serve needs, and one call per window keeps each window's payload measurable.
struct StrataStepArgs {
    char tokens[128];      // the window's ids, comma-separated as --tokens writes them (8 ids of 6 digits fit)
};

// What ONE step measured: the rows it ran, the payload's size and hash (the LIBRARY's own FNV over the bytes
// it wrote, the caller hashes its copy - the C.1 rule per window), and the times.
struct StrataStepOut {
    char err[256];                   // what refused, in the library's own words (empty = the step ran)
    int ran;                         // 1 = the window replayed
    int rows;                        // rows this step ran (its tokens)
    long long payload_bytes;         // rows * handoff_floats * 4, the bytes written into the mapped buffer
    unsigned long long payload_hash; // FNV-1a over those bytes, computed HERE
    double run_ms;                   // the window itself
    double wait_ms;                  // the wait since the previous step (0 before the first): the build's cost is
                                     // NOT in the step's run_ms, so a persisting stage shows up as ~0 build here
    long long steps_total;           // steps this built stage has run, this one included
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

// The REAL stage entry, on the in-tree template's contract (strata_kq_run at src/prefill/mmq_kq_lib.cu:46
// and its loader side strata_kq_run_fn run, resolved at mmq_kq_loader.cpp:73): the args block as a void*,
// the CALLER's own sizeof of it, and the same for the out block - the library refuses before touching either
// block when a size disagrees.  Return: 0 = the stage ran; -1 = a null argument or a size disagreement;
// -100 - step = the stage refused at that step (1 the GGUF geometry, 2 the expert layout, 3 the weights,
// 4 the native projections, 5 the session, 6 the expert tier, 7 the Verifier, 8 NOT every expert of the
// stage resident - the window would ask a CPU expert pool, and the pool across the join is not wired yet, so
// it is refused rather than hidden, 9 the window itself, 10 an allocation or the card).
typedef int (*strata_hip_stage_run_fn)(const void* args, int args_size, void* out, int out_size);
// The payload's way back: the floats the stage wrote, copied into a buffer the CUDA engine OWNS.  Returns the
// number of floats copied, or negative when there is no payload left.  The caller sizes its buffer from
// StrataStageOut's handoff_floats and rows, hashes what it receives, and compares that against payload_hash:
// two hashes of the same bytes, computed on the two sides of the join, is the measurement that the crossing
// carried the activations and not a copy of the same buffer.
typedef int (*strata_hip_stage_payload_fn)(float* dst, int dst_floats); 

// C.6 half (b): the stage entry SPLIT so a stage can persist across the windows of a real run - C.1's one-call
// entry rebuilt the WHOLE stage and ran ONE window, which measures the crossing but cannot serve a decode.
// build takes the SAME StrataStageArgs (FIRST stage [0,K) only, hand_in null - unchanged shape) and holds the
// stage's objects in the library's own file-static; step replays the built graphs for one window's tokens
// (StrataStepArgs) and leaves that window's payload for strata_hip_stage_payload; stage_free releases the stage.
// A step with no built stage returns -2 with the reason in out's err; so does a SECOND build over an
// already-held stage - one stage at a time, released by strata_hip_stage_free.
// build: 0 = built (out is a StrataStageOut; its window fields stay 0, build_s carries the load); step: 0 ran.
typedef int (*strata_hip_stage_build_fn)(const void* args, int args_size, void* out, int out_size);
typedef int (*strata_hip_stage_step_fn)(const void* args, int args_size, void* out, int out_size);

// C.4: hold a block in the arena of THIS library (the engine's own DeviceArena inside the HIP runtime, on
// the AMD card), then let go.  The CUDA loader holds ITS own arena in ITS runtime, same call, and each side
// prints only its arena's numbers.  Returns 0 = held; -1 = a null argument or a size disagreement; -2 = the
// card refused the allocation, out STILL carrying that card's free/total so the refusal can name the card
// with generate.cpp:164's shape `(N of M MiB VRAM free on this GPU)`.
typedef int (*strata_hip_arena_hold_fn)(long long bytes, int device, void* out, int expect_size);
typedef void (*strata_hip_arena_free_fn)();

// C.5: hash the named expert pairs INSIDE this library - its own expert tier on the AMD card, filled from
// experts.bin by the library's own objects, nothing of the caller's read here.  Same block rule as the
// stage entry: each side carries its own sizeof and the library refuses before touching either on a drift.
// Return: 0 = every pair hashed into out; -1 = null argument or size disagreement; -100 = the library
// refused (out's err names why).
typedef int (*strata_hip_expert_hash_fn)(const void* args, int args_size, void* out, int out_size);

}

namespace strata::core {
// The whole measurement, from the engine's side (src/core/hip_kernels_loader.cpp): open the library, hold a
// block on the AMD card through it, read the NaNs back into this process's own RAM, print what was measured,
// then let go.  true = the engine's arena, the engine's kernel, this card's memory and the read-back all
// worked in one process that already owns CUDA.  Declared HERE (not in device.hpp) because only a multi-brand
// build can answer it, and this header is what both sides of the join already share.
bool stage_hold_gate(long long bytes, int device);
// C.1's measurement, taken on the engine's side: ask the library to run ONE real stage, take its hand-off
// payload into RAM this process owns, and true = the stage ran, the payload crossed, and the hash of the
// bytes THIS side holds equals the hash the LIBRARY reported over the bytes it wrote.  Refuses (false) with
// the reason printed when there is no such library, when the stage refused, or when the two hashes disagree.
bool stage_run_gate(const StrataStageArgs& args);
bool stage_persist_gate(const StrataStageArgs& args);   // half (b): build once inside the library, replay twice

// half (b) unit 2a: the SHIM in strata_core's loader (hip_kernels_loader.cpp) - the per-window contract a
// mixed serve uses, glued into ONE call so no STALE window can be read between two entries: step answers
// one window AND takes its payload into RAM the caller owns.  build: true = the stage is held in the library
// (out, when given, is the caller's own StrataStageOut reading); step: floats copied (>0) or negative, with
// the window's report in `report` when the caller wants it; free releases the held stage.
bool engine_stage_build(const StrataStageArgs& args, StrataStageOut* out);
long long engine_stage_step(const char* tokens, float* payload_out, int payload_floats, StrataStepOut* report);
void engine_stage_free();

// C.4's measurement: ONE arena per runtime, not one arena per process.  The host holds its OWN DeviceArena
// (this process's runtime) and the library holds its OWN inside its HIP runtime through the two entries
// above; each printed line carries ONLY its arena's asked/held/free-before-after/total, plus the
// arena_pin_cap_gib() value THAT side read - the WDDM cap is per arena, not per process.  A HIP-only build
// has no library half and prints "not applicable" there.  false = one arena refused; a refusal names its
// card with generate.cpp:164's shape `(N of M MiB VRAM free on this GPU)`.
bool per_runtime_arenas_gate(long long bytes);

// C.5's measurement, taken on the engine's side: the NVIDIA tier filled from experts.bin BEFORE the RAM
// arena exists (its own `reader`), the RAM arena re-laid WITHOUT the pairs that tier holds (its holed load
// lines), the dropped pairs fetched back into this process's RAM and hashed HERE, and the library's own
// AMD tier hashing the SAME pairs - one number at three sites.  false = one site refused or the hashes
// disagree; every printed line carries only the numbers of the side that measured it.
bool expert_tier_gate(const std::string& pack, const std::string& native, const std::string& profile,
                      long long layers, long long experts, int tier_pairs, int helper_device);

}  // namespace strata::core
