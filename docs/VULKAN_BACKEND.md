# Cross-brand multi-GPU (NVIDIA + AMD) — feasibility, open questions, and plan

Status: **analysis / design note** (not implemented). Generated from the local clone at
`D:\Development\temp\strata` (project version `0.1.44.1`).

## Goal (the real aim)

The PC has **two GPUs of different brands — one NVIDIA + one AMD — plus the CPU.** Today Strata
cannot use both together: the layer-split runs on **one backend's runtime**, which only sees one
brand's cards (`docs/MULTI_GPU.md:53` "Not supported: ... a mix of NVIDIA and AMD cards"). The
user wants to **remove that same-brand limitation** so both cards split one model.

Vulkan is the right *conceptual* fit: it is the **only single API both NVIDIA and AMD drivers
expose**, so `vkEnumeratePhysicalDevices` returns both cards and one backend could drive both.

## Key insight — the blocker is narrower than "rewrite the engine"

The existing layer-split is **already brand-agnostic**:
- The cross-card hand-off is **host-mediated** — "a few hundred KB through pinned RAM ... No
  NVLink or peer-to-peer access is needed" (`docs/MULTI_GPU.md:9`–`12`). It is pipeline/layer
  parallelism, not P2P.
- The split is driven by `cudaGetDeviceCount` + `--split-device D` + `CUDA<d>` ids
  (`src/program/generate.cpp:2279`, `:2360`), and each stage keeps its own expert cache.

So the **only** thing that prevents NVIDIA+AMD today is that **one runtime API enumerates only
one brand's devices.** If a single API saw both cards, the existing split logic would mostly
already work. That is what makes Vulkan attractive — and what makes a **cheaper alternative**
possible (see Track B below).

## Short answer

- **Vulkan = yes, feasible**, but it is a **full port** (new runtime layer + kernels rewritten as
  SPIR-V compute shaders), and it **loses the NVIDIA/AMD fast paths** (tensor cores, nvfp4, dp4a)
  → expect a **slower** engine than native CUDA/HIP. Good for "use both cards", not for "match
  CUDA speed".
- **Cheaper first move (Track B): let CUDA and HIP coexist in one binary** and route each layer-split
  stage to the right brand's runtime. This **keeps native speed** and reuses the already host-mediated
  split. It is a runtime/allocator/kernel-loading refactor, not a new shader language.

---

## 1. How backends are wired today (grounded in code)

The engine is one CUDA-shaped source tree: `src/**/*.cu` (kernels, prefill, engine, core).
Backends are selected in the top-level `CMakeLists.txt` (`CMakeLists.txt:58`–`96`):

| Backend | How it is built | Mechanism |
| --- | --- | --- |
| CUDA | `STRATA_ENABLE_CUDA` → `enable_language(CUDA)`, `find_package(CUDAToolkit)` | native nvcc |
| HIP (AMD) | `STRATA_ENABLE_HIP` → `cmake/hip_backend.cmake` | **same `.cu` files re-tagged as HIP** + `hip_compat/` headers that `#define cudaMalloc hipMalloc`, `cudaMemcpy hipMemcpy`, `cudaDeviceSynchronize hipDeviceSynchronize`, cublas→hipblas, WMMA, dot4 |
| HIP gfx906 | `STRATA_HIP_GFX906` → forces `STRATA_ENABLE_CUDA` on, CUDA targets aliased to HIP | same shim trick |
| Metal (Apple) | `metal/` project | `metal_compat/cuda_runtime.h` shims the CUDA API to Metal |
| SYCL (Intel Arc) | `STRATA_ENABLE_SYCL` → `add_subdirectory(sycl)` | **separate project** that re-takes `src/` sources, `.cu`→`.dp.cpp`, own `sycl_queue.hpp`/`sycl_math.hpp`/`sycl_doorbell.hpp`, `setup_intel.py`, `tools/migrate.sh`, `tools/merge_upstream.py` |

The decisive fact: **HIP and Metal work as shims because their GPU model is close to CUDA's** —
raw device pointers, `hipMalloc`/`hipMemcpy`, launch syntax, streams, graphs, WMMA, `dot4`.
A header of `#define cudaXxx hipXxx` is enough (`include/strata/hip_compat/cuda_runtime.h:18`–`120`).

Vulkan's compute model is **not** close to CUDA's, so the shim trick cannot carry over.

## 2. What the kernels actually depend on (the blockers)

Representative CUDA-specific usage found in the tree:

- Launch + sync + runtime API: `__global__`, `kernel<<<blocks,threads>>>(...)`,
  `cudaDeviceSynchronize`, `cudaMalloc`/`cudaMemcpy`/`cudaMemset` (`src/kernels/cuda/s2_gemv.cu:19`–`60`).
- One big **arena** `cudaMalloc`'d first, expert cache resident in VRAM, SSD lookup table,
  layer-split across GPUs, pinned host memory, peer access
  (`src/core/device.cu`, `src/core/vmm.cpp`, `src/core/expert_cache.cpp`, `src/core/copy_queue.cpp`).
- Device introspection: `cudaDeviceGetAttribute`, `cudaMemGetInfo`, `cudaDeviceProp`
  (`src/prefill/ggml_cuda_host.cu:62`–`80`, `src/kernels/cuda/decode_cluster_parity.cpp`).
- **NVIDIA-only instructions**: WMMA/MMA tensor cores (`src/prefill/wmma_gemm.cu`, `moe_mmq.cu`),
  `nvfp4` FP4 (`src/prefill/mmq_nvfp4_w4a4.cu`, `mmq_nvfp4_w4a8.cu`), `__dp4a`
  (`include/strata/kernels/dp4a.hpp`, `tools/sm60_dp4a_check.cu`), PDL / programmatic dependent
  launch, cooperative launch (`supports_cooperative_launch`), `__nanosleep`, warp shuffles.
- cuBLAS for GEMM (`CUDA::cublas` target; `src/prefill/gemm.cu`).
- ggml-cuda host symbols for the MMQ prefill path (`src/prefill/ggml_cuda_host.cu`, gated by `GGML_USE_HIP`).

Mapping to Vulkan:

| CUDA feature | Vulkan equivalent | Verdict |
| --- | --- | --- |
| `cudaMalloc` arena | `VkDeviceMemory`/`VkBuffer` + VMA allocator | OK, but shaders index by **bound buffer + offset**, not raw pointers |
| `<<<>>>` launch | `vkCmdDispatch` of a precompiled SPIR-V pipeline + descriptor sets | rewrite per kernel |
| `cudaMemcpy`/pinned | staging buffer + `vkCmdCopyBuffer`, or host-mapped `VkBuffer` | OK with a copy layer |
| `cudaDeviceSynchronize` | `vkQueueWaitIdle` / timeline fence | OK |
| `cudaMemGetInfo` | `VkPhysicalDeviceMemoryProperties` / VMA stats | OK |
| warp shuffles | GLSL `subgroupShuffle*` (subgroup ext) | mostly OK |
| `__dp4a` | emulate in GLSL (no int8 dot4 in core Vulkan) | slower, manual |
| WMMA / tensor cores | **no Vulkan tensor-core API** | must write plain compute → perf loss |
| `nvfp4` (NVIDIA FP4) | **not a Vulkan feature** | must dequant in shader |
| cooperative launch / grid sync | **not available** | must restructure algorithms |
| cuBLAS GEMM | write Vulkan GEMM shaders (or reuse ggml-vulkan) | new code |

## 3. Two things that already exist and help

- **ggml already ships a Vulkan backend** (llama.cpp `ggml-vulkan`). It is vendored in
  `third_party/ggml` and its CMake knobs appear in build logs (`GGML_VULKAN:BOOL=OFF` in
  `bench/results/2026-10-10-community-rx-9060-xt/build.json`). It could back the **dense
  projections**, but Strata's MoE expert kernels, QSA, router, KV, sampler are custom CUDA, not
  ggml — so ggml-vulkan alone is **not** the engine.
- **Vulkan is already used for the image/diffusion runner** (sd.cpp), not the LLM engine:
  `tools/image/qimage.py` has `BACKENDS = ("cuda","hip","vulkan","cpu")`, `-DSD_VULKAN=ON`,
  and downloads the LunarG Vulkan SDK (`docs/IMAGES.md`). Useful as a build/SDK precedent, but
  it does not touch the inference kernels.

## 4. Verdict — two ways to remove the brand limitation

**Track A — Vulkan single backend (cross-vendor).** One API sees both cards; the existing
host-mediated split then drives them. Cost: full port (section 2 table), and **perf loss** on
the tensor-core / nvfp4 / dp4a paths. Best when you want *one* code path that runs on any
vendor, or when a card has no native Strata backend (e.g. Intel).

**Track B — CUDA + HIP coexist in one binary, per-stage router (recommended first).** Keep the
native fast kernels for each brand; make the engine able to load **both** CUDA and HIP kernels in
one process and assign each layer-split stage to the brand that owns the card. The split already
copies activations through pinned host RAM, so the hard part is only:
- today CUDA and HIP are **mutually exclusive builds** (`CMakeLists.txt:91` FATAL_ERROR) — they
  must coexist (two runtime libs, two kernel-loading models, symbol/name clashes to resolve);
- the device list must be a **union** of CUDA and HIP devices, and `--split-device`/`CUDA<d>`
  ids must map to the right runtime per stage;
- the arena/pinned/copy layer must talk to both runtimes.

Track B keeps native speed and is the smaller change for a 2-card NVIDIA+AMD rig. Track A is the
bigger change but the only single-API, any-vendor answer.

Both are **additive**: keep CUDA as the parity reference.

---

## 5. Open questions (need answers before/while implementing)

0. **Aim check.** Is the goal "use my NVIDIA + AMD together at any speed", or "match native
   speed"? If the former, Track B (CUDA+HIP coexist) is the smaller win; if you also want Intel
   or a single any-vendor path, Track A (Vulkan).
0b. **Which card owns which layers?** For a 2-card NVIDIA+AMD split, do you want auto layer
   placement (by free VRAM) or a fixed split point?
1. **Target hardware & driver coverage.** Which GPUs must the backend run on — Intel Arc, AMD
   APUs/iGPUs (780M, Strix/Halo), AMD-under-Windows, or also discrete AMD/NVIDIA? This decides
   which Vulkan features we may rely on (subgroups, 64-bit addressing,
   `VK_KHR_cooperative_matrix`, `VK_KHR_fp8`, buffer sizes > 4 GB).
2. **Is "instead of CUDA" or "in addition to CUDA" intended?** The repo keeps CUDA as the
   reference for parity tests. Recommend *additive* (keep CUDA) so parity tests stay meaningful.
3. **Shader authoring path.** Hand-written GLSL compiled with `glslc`, or generate SPIR-V, or
   reuse `ggml-vulkan` shaders? Affects the whole kernel plan.
4. **The arena / expert-cache VRAM model.** Can we keep the single-arena + SSD-mapped design
   under Vulkan (VMA + `VkBuffer`s + host-mapped or staged copies), and does layer-split /
   multi-GPU peer access have a Vulkan equivalent?
5. **Tensor-core loss is acceptable?** Without WMMA/MMA, prefill speed may drop sharply. Do we
   accept a slower Vulkan path (like the SYCL/XMX path), or must it match CUDA?
6. **nvfp4 / K-quant experts.** How to represent `nvfp4` and Q4_K/Q5_K/Q6_K experts on Vulkan
   (dequant-in-shader vs. requantize at load)?
7. **dp4a / int8 router & QSA.** Emulate `__dp4a` in GLSL, or move router/QSA to fp32?
8. **Build integration.** New `option(STRATA_ENABLE_VULKAN)` + `cmake/vulkan_backend.cmake`, or
   a separate `vulkan/` project like `sycl/`? Which does the team prefer?
9. **SDK/deployment.** Vendor LunarG SDK headers + `glslc` (already done for images) vs.
   system `vulkan-sdk`/`vma`? Windows vs Linux packaging.
10. **Parity & validation.** Reuse the existing `*_parity.cpp` harnesses against a Vulkan
    reference, and add a `tests/vulkan/` set like `tests/hip/`?
11. **Effort/scope.** How many kernels are in scope (the full `src/kernels/cuda/*` set is ~40+
    files plus `src/prefill/*`)? Is a subset (decode-only, no prefill speed) acceptable first?

---

## 6. Proposed plan

### Track B first — CUDA + HIP coexist in one binary (recommended for a 2-card NVIDIA+AMD rig)

**B0 — Decide.** Confirm the aim (Q0) and the split policy (Q0b). Keep CUDA as the parity
reference.

**B1 — Coexist build.** Drop the `CMakeLists.txt:91` "enable only one" guard so CUDA and HIP can
both be configured in one build. Resolve the two runtime libs (`CUDA::cudart`/`CUDA::cublas` and
`hip::host`/`roc::hipblas`) and the `hip_compat/` force-include so they don't collide (namespace
the HIP shim, or compile the two kernel sets into separate translation units / libraries).

**B2 — Unified device list.** Make `cudaGetDeviceCount`/`--split-device`/`CUDA<d>` a **union** of
CUDA and HIP devices, each tagged with its runtime. `setup.py`/`setup.sh` already lists cards;
extend it to admit a mixed `--gpus` (e.g. `--gpus cuda:0,hip:0`).

**B3 — Per-stage router.** Each layer-split stage runs its kernels on the runtime that owns its
card. The cross-stage hand-off already goes through pinned host RAM (`docs/MULTI_GPU.md:9`–`12`),
so no P2P is needed across brands.

**B4 — Validate.** Reuse the existing split parity checks (`tools/split_mtp_batch_parity.py`,
`tests/hip/`, `tests/cuda/`) on a mixed rig; add a `tests/mixed/` set.

### Track A — Vulkan single backend (only if you want one any-vendor path / Intel)

**A0 — Scope.** Answer Q1–Q3, Q8, Q11. Pick "additive" + "separate `vulkan/` project" + "subset
first".

**A1 — Runtime layer.** `option(STRATA_ENABLE_VULKAN)` + `cmake/vulkan_backend.cmake` (or a
`vulkan/` subdir like `sycl/`). Vulkan runtime abstraction over the CUDA API surface used by
`src/core/*`: arena (VMA + `VkDeviceMemory`/`VkBuffer`), copy/pinned (staging + `vkCmdCopyBuffer`),
sync (timeline fence / `vkQueueWaitIdle`), device info (`VkPhysicalDeviceFeatures2`/`Properties`),
dispatch (launch a SPIR-V pipeline with bound buffers). `vkEnumeratePhysicalDevices` returns **both**
cards → the existing split logic drives them.

**A2 — Kernel port, subset first.** Decode-critical kernels as GLSL→SPIR-V (`s2_gemv`, `s_gemv`,
`iq_*`, `kv_*`, `rope`, `router_top10`, `sampler`, `elementwise`, `dequant_*`); reuse `ggml-vulkan`
for dense GEMM. Validate against the `*_parity.cpp` harnesses (add `tests/vulkan/`).

**A3 — MoE / prefill.** Port `native_moe`, `s2_expert_grouped`, `moe_fused*`, `mmq_*`; decide
nvfp4/K-quant representation; accept tensor-core loss or add `VK_KHR_cooperative_matrix`.

**A4 — Engine + docs.** Wire `src/program/generate.cpp`, expert cache, layer-split, conversation
state to the Vulkan runtime (mirror `sycl/src/core/*`); add `setup_vulkan.py` like
`sycl/setup_intel.py`; `docs/VULKAN.md` + community bench reports.

### Effort / risk
- **Track B:** medium — runtime/allocator/kernel-loading refactor; **keeps native speed**; smallest
  change for a 2-card NVIDIA+AMD rig.
- **Track A:** high — ~40+ CUDA kernel files + prefill + runtime rewritten as SPIR-V; **perf loss**
  on tensor-core/nvfp4/dp4a paths; the only single-API, any-vendor answer.
- Both additive: CUDA stays the parity reference.

### Recommendation
For this PC (one NVIDIA + one AMD), **start with Track B** (CUDA+HIP coexist + per-stage router):
it removes the brand limitation at native speed and reuses the already host-mediated split.
Choose **Track A (Vulkan)** only if you also need Intel GPUs or a single any-vendor code path —
and land a **decode-only Vulkan subset** first before the tensor-core-heavy prefill.

---

## 7. Track B — implementation map (grounded in code)

"B is fast" = fast **at runtime** (native CUDA + native HIP, no math rewrite). The cost is in the
**build/link plumbing**, because today the backend is fixed at **link time**, not chosen at runtime:

- **Kernels are linked directly, no dispatch table.** `include/strata/kernels/s2_gemv.hpp:22` declares
  `strata::kernels::s2_gemv(...)`; `src/kernels/cuda/s2_gemv.cu` defines it; the engine calls it. The
  only runtime-loaded lib today is the optional MMQ (`src/prefill/mmq_kq_loader.cpp:64` `dlopen`) —
  which is the **precedent** for shipping each backend's kernels as a shared library.
- **One GPU runtime per binary.** The whole engine is written against CUDA-named symbols
  (`cudaGetDeviceCount`, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceProp`, `cudaMemGetInfo`); the HIP build
  swaps them with a **global** `#define cudaXxx hipXxx` force-include
  (`include/strata/hip_compat/cuda_runtime.h:18`–`120`, `cmake/hip_backend.cmake:104`–`112`).
- **Device enumeration is single-runtime.** `src/core/device.cu:78` `device_count()` = `cudaGetDeviceCount`;
  the split sizes `n_dev` from it (`src/program/generate.cpp:2358`–`2360`) and `--split-device`/`CUDA<d>`
  index that one list.
- **Arena/pinned/copy are on that one runtime.** `include/strata/core/device.hpp:5` `DeviceArena` =
  "ONE cudaMalloc per region + bump alloc"; `src/core/vmm.cpp`, `src/core/pinned.cu`, `src/core/copy_queue.cpp`,
  `src/core/expert_cache.cpp`.
- **Build forbids coexist.** `CMakeLists.txt:91` FATAL_ERROR "enable only one" of CUDA/HIP.

### What Track B must change (the seam)
1. **Relax `CMakeLists.txt:91`** so CUDA and HIP can both be configured; link both runtime libs
   (`CUDA::cudart`/`CUDA::cublas` + `hip::host`/`roc::hipblas`).
2. **Make the `hip_compat` rewrite per-TU, not global** — it currently rewrites every `cudaXxx`; with
   real CUDA also present that would corrupt the CUDA TUs. Compile CUDA kernels with NVIDIA headers,
   HIP kernels with the shim, in separate translation units.
3. **Namespace or library-per-backend the kernels** to avoid the `strata::kernels::*` symbol clash
   (mirror the `mmq_kq_loader.cpp` dlopen pattern → `strata_cuda_kernels` + `strata_hip_kernels`).
4. **Union device list + backend tag.** Extend `device.hpp`'s `DeviceInfo`/`device_count()` to merge
   CUDA and HIP devices; `--gpus cuda:0,hip:0`; each `CUDA<d>`/`--split-device` id carries its runtime.
5. **Per-stage dispatch.** Each layer-split stage calls the kernel set of the runtime that owns its
   card; the cross-stage hand-off already goes through pinned host RAM (`docs/MULTI_GPU.md:9`–`12`),
   so no cross-brand P2P is needed.
6. **Arena/copy per runtime.** `DeviceArena`/`pinned.cu`/`copy_queue.cpp` allocate on the owning
   runtime; the shared expert arena in host RAM stays shared.

### Honest risk
- Two toolchains (nvcc + clang/HIP) and two runtime libs in one binary is a real build refactor; the
  global `hip_compat` `#define` is the trickiest part (it must stop leaking into CUDA TUs).
- The fast path is preserved (native kernels), so this is the better choice for a 2-card NVIDIA+AMD rig
  **if** the coexist build can be made to link cleanly. If that plumbing proves too invasive, fall back
  to Track A (Vulkan), which is a clean single-runtime model but slower.

### First concrete experiment (cheap, before any refactor)
Confirm the two cards are even *visible* to one API and that the split math is brand-agnostic:
- enumerate both cards via Vulkan (`vkEnumeratePhysicalDevices`) just to prove one API sees NVIDIA+AMD;
- run the existing split on **two same-brand** cards to confirm the host-mediated hand-off is the
  only coupling — that is the exact seam Track B has to bridge.

### Toolchain status on this PC (grounded, Oct 2026)
- **Vulkan sees both cards** (proven via `vulkaninfo.exe --summary`): GPU0 `vendorID 0x10de`
  **NVIDIA RTX 3070 Laptop GPU** (Vulkan 1.4.329), GPU1 `vendorID 0x1002` **AMD RX 9070 XT** (Vulkan 1.4.349).
  Both `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`. So one API already enumerates both brands — the seam is real.
- **HIP 7.2.60201** at `C:\Program Files\AMD\ROCm\7.2` (`hipcc`/`clang`/`hipblas`/`rocblas` + `include` + `cmake`):
  a usable compiler/runtime. **CUDA 13.3** present; **CMake** on PATH; **Python 3.12**; **Git**.
- **MSVC Build Tools 17.14 installed** at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools`
  (note the `(x86)` path). `cl.exe`/`link.exe` present at `VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`.
  Windows SDK is a **shared** component at `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um\Windows.h`
  (not under the product folder); `vcvars64.bat` adds it to `%INCLUDE%`, so `cl.exe`/`nvcc` find `<windows.h>`.
  So the MSVC side is complete. **Still missing: `ninja`** (the build uses `-G Ninja`, `build_windows.bat:68`)
  — add a standalone `ninja.exe` on PATH, or configure with `-G "Visual Studio 2022"`.
- The project's HIP build expects **ROCm 10.2 TheRock wheels** (`build_windows.bat:48`–`49`: `rocm-sdk.exe`,
  `%ROCM%\lib\llvm\bin\clang++.exe`), not the 7.2 SDK layout (clang in `bin`). The 7.2 SDK is not what
  `build_windows.bat` consumes.

### What I changed (Track B, gated)
- `CMakeLists.txt`: added `option(STRATA_CUDA_HIP_COEXIST ... OFF)` and relaxed the "enable only one"
  guard to allow CUDA+HIP when that flag is ON. **Default OFF → every existing build is byte-identical.**
  This removes the literal `CMakeLists.txt:91` blocker but is **necessary, not sufficient** (see the 4 items
  in the comment above the guard).

### Cheaper Track B variant — two cooperating processes (no coexist build, no symbol clash)
Because the cross-card hand-off is already **host-mediated** (`docs/MULTI_GPU.md:9`–`12`, "a few hundred KB
through pinned RAM ... No NVLink or peer-to-peer"), the two brands never need to share a GPU address space.
So instead of one binary with two runtimes, run **two engine processes**: a CUDA build owns the NVIDIA card's
layer range, a HIP build owns the AMD card's range, and they exchange the per-token activations through a
shared host-RAM buffer (or a pipe). This sidesteps the CUDA(cl.exe) vs HIP(clang) host-compiler conflict and
the `strata::kernels::*` symbol clash entirely — each process is a normal single-backend build. Cost: process-
boundary latency on the hand-off (small: a few hundred KB per verify window). This is a new engine mode, not
an existing one — it needs the engine to run as two cooperating processes that share the conversation/KV state.

### What still blocks a full coexist build (Track B, one binary)
1. **No MSVC** → can't compile/link either backend locally (install VS Build Tools).
2. **Toolchain mix**: CUDA host = `cl.exe`, HIP host = `clang` (ROCm); CMake refuses to mix them
   (`build_windows.bat:9`–`11`). Needs per-target host compilers or two sub-builds linked together.
3. **Symbol clash**: CUDA and HIP kernels define the same `strata::kernels::*` names → library-per-backend
   (mirror `src/prefill/mmq_kq_loader.cpp:64` `dlopen`).
4. **Union device list + per-stage dispatch + arena-per-runtime** (the seam in section 7 above).
