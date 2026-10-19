// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

// Which runtime a device belongs to.  Upstream runs ONE runtime, so it sees ONE brand and a plain ordinal in
// --gpus is enough; OWN keeps that meaning.  CUDA and HIP name a brand, which is what a mixed rig needs:
// "--gpus cuda:0,hip:0" (docs/MULTI_GPU.md:53 still refuses a mix of NVIDIA and AMD cards today).  Track B
// keeps each card's native kernels, so a stage must know the runtime of the card it runs on.
enum class DeviceRuntime { OWN, CUDA, HIP };

// A device id on a mixed rig: "cuda:0" (NVIDIA), "hip:1" (AMD), or a plain ordinal ("2" = this engine's own
// runtime, upstream's behavior).  false when the text is not a device id; the out-params then keep what they had.
inline bool parse_device_id(const std::string& text, DeviceRuntime& runtime, int& ordinal) {
    auto digits = [](const std::string& s, size_t from, int& out) {
        if (from >= s.size()) return false;
        int v = 0, seen = 0;
        for (size_t i = from; i < s.size(); ++i) {
            if (s[i] < '0' || s[i] > '9') return false;
            v = v * 10 + (s[i] - '0');
            ++seen;
        }
        if (seen == 0) return false;
        out = v;
        return true;
    };
    auto ci = [](const std::string& s, size_t at, const char* word) {
        for (size_t i = 0; word[i]; ++i) {
            if (at + i >= s.size()) return false;
            const char c = s[at + i] >= 'A' && s[at + i] <= 'Z' ? char(s[at + i] - 'A' + 'a') : s[at + i];
            if (c != word[i]) return false;
        }
        return true;
    };
    if (text.empty()) return false;
    if (ci(text, 0, "cuda:") && digits(text, 5, ordinal)) { runtime = DeviceRuntime::CUDA; return true; }
    if (ci(text, 0, "hip:") && digits(text, 4, ordinal)) { runtime = DeviceRuntime::HIP; return true; }
    return digits(text, 0, ordinal) ? (runtime = DeviceRuntime::OWN, true) : false;
}

struct DeviceInfo {
    int ordinal = -1;
    DeviceRuntime runtime = DeviceRuntime::CUDA;   // set by src/core/device.cu: which runtime this card is on
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // HIP: gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
};

// HIP builds: whether GPU `ordinal` can run this binary - its architecture must be one the binary was COMPILED
// for (STRATA_HIP_ARCHS, set by cmake/hip_backend.cmake) and it must run wave32.  "" when it can (or when there
// is no such device: the caller's own device errors apply), else the reason in a sentence.  A binary carried to
// another card would otherwise fail later with "invalid device function".  CUDA builds: always "".
std::string gpu_arch_problem(int ordinal);

// The GPU architectures this binary was compiled for ("gfx1100,gfx1201"); "" on CUDA builds.
const char* compiled_gpu_archs();

// How many devices the runtime enumerates, numbered as HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES number them; 0 when
// there is none (or no usable runtime).  `device_info` throws on a card this binary cannot run, so a caller that LISTS
// every card - the ones it has no code for included (`strata-device --list-devices`) - counts them with this first.
int device_count();

// One line per device for `strata-device --list-devices`, without the arch check that `device_info` applies:
// "arch gfx1201, 15.9 GiB" (HIP) or "compute capability 12.0, 11.9 GiB" (CUDA), plus the device's name.  false when
// the runtime cannot describe it.
bool device_summary(int ordinal, std::string& name, std::string& detail);

// The cards ONE process can drive, in ONE list, in the EXISTING order: this build's own runtime first
// (upstream's order), then the AMD cards that strata_hip_kernels lists from inside its own runtime.  Nothing
// here re-ranks them: a CUDA process cannot ask HIP for its cards (amdhip64 answers with an empty list once
// CUDA owns the process), so the AMD part comes from the library that carries the HIP runtime
// (src/hip/kernels_probe.cu), opened by src/core/hip_kernels_loader.cpp.
//
// Which card is the MAIN one (the layers then go: main card, second card, then the CPU tier - the expert pool
// in RAM) is the user's choice, asked at setup and saved in the run config: setup.py asks and writes the
// order into "gpu", and writes "gpu_order": "as_given" when the answer is not the recommended order, so
// serve/server.py's ordered_gpus() keeps it.  That is upstream's own key, with upstream's measured default
// (the faster card LAST for an auto split, #1352; skipped when it would end on less VRAM, #1576) still in
// force when the user did not choose.  There is no cross-brand power table here: a CUDA card and an RDNA
// card share no power unit, so this seam does not invent a ranking - it keeps the ids and their order stable.
std::vector<DeviceInfo> union_device_list();
bool hip_kernels_ready();

// What union_device_list() did the last time it ran: `card_order_chosen()` is true when the order came from
// STRATA_GPU_ORDER (the order the user chose at setup, see apply_chosen_order in src/core/hip_kernels_loader.cpp),
// and `card_order_text()` is the value that was applied.  false / "" when the process kept its runtime's own
// order - either nothing was given, or what was given did not name every card and was refused, with the reason
// printed.
bool card_order_chosen();
const std::string& card_order_text();

// The measurement that gates routing the AMD layers here: run one AMD kernel from the library in THIS process,
// whose CUDA runtime is already up and is still holding VRAM, and print each card's free VRAM around it.
bool cross_runtime_probe();

// STRATA_PRIMARY_DEVICE (generate.cpp): the ordinal the engine's own session runs on; 0 = the runtime's first
// visible device (upstream's behavior).  The Windows HIP runtime enumerates the cards in a fixed order no
// HIP_VISIBLE_DEVICES list can change (it only filters), so a PC whose faster card comes second could not run
// the model on it while the first card served as a helper.  The helper expert caches and the peer tier must
// live on another device.  Set once, before any context exists.
void set_primary_device(int ordinal);
int primary_device();

// Throws when there is no CUDA device.  The engine targets sm_120 specifically and must say so rather than
// run slowly on something else: `CMakeLists.txt` already refuses to COMPILE for another architecture, and
// this is the matching check at run time (a binary can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

/// "" when this build has device code for the current device, else CUDA's error: a build for other GPUs would
/// otherwise fail at its first kernel launch, with nothing that names the cause.
std::string device_code_error();

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
