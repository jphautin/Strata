# RISKS — cross-brand multi-GPU

Each risk: the cause, then the step that reduces it.

## R1. Vulkan (Track A) is slower than CUDA / HIP.

The current kernels use hardware that Vulkan does not expose. So Vulkan uses
slower paths. The effect is large on GEMM and attention.
- Vulkan has no tensor-core or WMMA API. GEMM and attention use plain
  multiply-add instead of HMMA / MMA. This is the largest loss.
- Vulkan has no DP4A (one int8 dot-product). The int8 router and QSA do
  scalar byte work instead.
- Vulkan has no NVFP4 or MXFP4 (FP4) feature. The shader decodes FP4 experts
  with generic ALU.
- Vulkan has no cuBLAS or hipBLAS. You have to write Vulkan GEMM, or use
  `ggml-vulkan`.
- The driver compiles SPIR-V with less tuning. NVCC and AMD LLVM tune the
  exact card (sm_86, gfx1201).
- Vulkan shaders index by a bound buffer and offset. They cannot use raw
  device pointers. This adds bounds checks and extra copy steps.
Net: the native CUDA / HIP path stays faster. Use Vulkan only for the Intel or
any-vendor case. Step that reduces it: Track B keeps the native kernels.

## R2. The one-binary coexist build (Track B, B.1).

- Two host compilers in one binary. CUDA uses `cl.exe`. HIP uses `clang`
  (ROCm). CMake refuses the mix (`build_windows.bat:9`–`11`).
- The global `#define cudaXxx hipXxx` shim rewrites every `cuda*` symbol.
  It would corrupt the CUDA units. It has to become per-unit.
- CUDA and HIP kernels define the same `strata::kernels::*` names. They
  clash. Fix: one library per backend (mirror the `dlopen` pattern at
  `src/prefill/mmq_kq_loader.cpp:64`).
- One process then holds two runtime arenas and two pinned/copy layers.
Step that reduces it: build two linked sub-targets, or use B.2 (two
processes).

## R3. Output drift between brands.

NVIDIA uses nvfp4. AMD uses its own quant path and a signed dot4. The two
cards can give different numerics for the same layer. A split model can then
change token choices.
Step that reduces it: run the parity harnesses on a mixed rig
(`tools/split_mtp_batch_parity.py`, `tests/hip/`, `tests/cuda/`).

## R4. ROCm SDK layout.

The build scripts expect ROCm 10.2 TheRock wheels (`rocm-sdk.exe`,
`%ROCM%\lib\llvm\bin\clang++.exe`). This PC has the 7.2 SDK. Its compiler is in
`bin\clang++`. The device bitcode path also differs.
Step that reduces it: point `CMAKE_HIP_COMPILER` at `bin\clang++`, set
`CMAKE_HIP_COMPILER_ROCM_ROOT` and `CMAKE_PREFIX_PATH` to the 7.2 root, and set
`CMAKE_HIP_FLAGS`.

## R5. Layer-split correctness across brands.

The split sizes `n_dev` from one `cudaGetDeviceCount`. A mixed rig needs a
union device list with a runtime tag, and each stage has to call the owning
runtime.
Step that reduces it: extend `include/strata/core/device.hpp` `DeviceInfo` /
`device_count()`, and route each stage.

## R6. Build fragility.

CMake can cache empty HIP flags and drop the kernels to `-O0`
(`cmake/hip_backend.cmake:46`–`57`). One failed configure poisons the build
folder.
Step that reduces it: use a fresh build folder for each configure.

## R7. The Vulkan enumeration proof is not a fast path.

`vkEnumeratePhysicalDevices` sees both cards, but a Vulkan backend that uses
them is a large port. The proof only shows the seam is real. It does not give a
fast path. Step that reduces it: do not choose Vulkan on this proof alone. Run
Track B first.

## R8. The coexist build does not compile as one project.

- `<hip/hip_runtime.h>` will not host-compile under the CUDA toolchain, so a cl.exe / nvcc
  unit must not `#include` it. But HIP's host calls (`hipGetDeviceCount`,
  `hipGetDeviceProperties`, `hipMemGetInfo`) are a plain C API: declare them `extern "C"`
  and link `amdhip64`, then call them from the CUDA-host unit. So B3 needs NO clang
  TU. Only the HIP kernel device code needs clang (that is B4).
- The engine's host is the CUDA branch. `hip::host` and `roc::hipblas` are found but
  not linked to the engine under a CUDA-primary coexist configure. Step that reduces
  it: link them onto the kernel/library targets (PLAN-B B4).
- Corrected (Oct 2026): the mix did NOT stop the configure. nvcc took MSVC (`cl.exe`) as its CUDA host, and
  CMake took clang for the project host and for HIP; both `enable_language(CUDA)` and `enable_language(HIP)`
  detected cleanly. The real host problem: the engine and ggml add MSVC-only flags (`/wdNNNN`, `/utf-8`, `/Zi`).
  If the project host is clang, clang reads those as file names and the build fails. Step that reduces it: set
  the host to `cl` (`-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`) and keep clang only for the HIP language.
  The remaining risk is the `strata::kernels::*` name clash and the two host ABIs; keep it in one binary
  (library-per-backend, B4) or split into two processes (B.2).
- Confirmed (Oct 2026): CMake 4.3's Windows-Clang-HIP platform module (`Windows-Clang.cmake:187`) raises
  "mixes Clang and MSVC … not supported" when HIP (clang) and the CUDA/engine (MSVC) share one configure.
  Step that reduces it: build the CUDA engine and the HIP kernel library in TWO separate CMake builds and join
  them at link time (library or `dlopen`), or use B.2 (two processes).

## R9. Two processes split the policies that see the whole model (B.2).

The kernels are unchanged, so B.2 loses no kernel speed. The loss comes from the parts that today sit in ONE
process and decide work for the WHOLE model. Full table: PLAN-B.md §7.

- The hand-off must keep the OVERLAPPED form, not the serial default. Upstream 0.1.45 documents both
  (`MULTI_GPU.md:218-244`). The default syncs stage 1, then stages and launches stage 2 (`:220-226`).
  `STRATA_SPLIT_OVERLAP=1` publishes with a GPU flag (`handoff_publish` writes with volatile stores and raises
  it; `wait_flag_ge` waits for it in the next stage's graph) and stages that graph on a helper thread.
  Measured on 2x RX 7900 XTX: decode 53 -> 70.1 tok/s on code and 45 -> 55.9 on prose, so the serial form
  costs about a third (`:228-237`). B.2 cannot use `wait_flag_ge` in the next stage's graph: there is no P2P,
  and one process cannot write the other process's GPU memory. Step: publish into a shared pinned page plus a
  flag word; the next process spins on that word, copies one flag word to its own GPU with a tiny H2D, then
  stages on a helper thread. Keep the bounded wait and the error word (`STRATA_SPLIT_WAIT_MS` `:239-241`) and
  the drop hook (`STRATA_TEST_HANDOFF_DROP=N` `:243`) across processes: a dead peer must fail the window,
  never read stale rows.
- The prompt-chunk overlap (a later card reads chunk c while the first reads c+1, `MULTI_GPU.md:212-215`) becomes
  cross-process. A sync hand-off loses up to -18/-20% on prompts (`:350`) and the +36% at 16K on two stages
  (`:212-215`). Step that reduces it: keep a hand-off ring, and size the chunks for the pipeline (`--prefill-pipe`,
  `:245-247`).
- `--pipeline-windows 2` is the default on a two-or-more-stage gfx12 split (`MULTI_GPU.md:361-364`) and is worth
  +16% mean decode and +21% on code (the `--pipeline-windows` measured rows). The guessed-window rollback copies
  the first card's recurrent state (the "Cost" row, `:400`); that state must stay visible to both processes, or
  the gain is lost.
- The host expert arena is mapped once and shared by all cards (`MULTI_GPU.md:210`). Two processes must share ONE
  mapping, cut by layer range (the arena is indexed by (layer, expert) pair), and each must run `--adapt-every`
  over its own layers only. If they fight over the same slots, the CPU pool is hit more, and that regime is the
  cliff: the resident-RAM table on one 32 GB PC spans 25, 32 and 71 tok/s, and a helper card's per-layer round
  trip cost more than the CPU pool needed. Step: partition the arena by layer range at open time, and share one
  resident-set ledger.
- The CPU pool, the aux threads and the host thread share one CPU budget (`--aux-cpus`, and `note_owned_thread`
  in `src/program/generate.cpp`). Two processes oversubscribe the cores. Step: give each process its own CPU set.
  On this rig the effect is small, because both cards' caches hold nearly every routed expert
  (`MULTI_GPU.md:5-7`).
- RAM and pinning: the arena must not be copied (one test rig locks 22 GiB of experts on 32 GB of RAM), and two
  GPU contexts already push the WDDM 8 GiB pin cap (`MULTI_GPU.md:332`). Step: one shared file mapping, and
  `cudaHostRegister` each physical range in one process only.
- `--layer-split auto` tries the placements over the whole model ("auto tries every placement", `MULTI_GPU.md`).
  In B.2 the split points come from outside, so auto needs a planner step or an explicit point. Step: run the
  auto search in the launcher once, then pass the same points to both processes.
- `STRATA_PRIMARY_DEVICE` (new in 0.1.45) picks the ordinal the engine's session runs on, and `generate.cpp`
  refuses it with a layer split ("not supported with a layer split yet"). Do not lean on it for B.2 card
  ownership: it does not cover a split. Its reason is recorded there too — on Windows the HIP runtime enumerates
  the visible cards in an order `HIP_VISIBLE_DEVICES` can filter but not reorder. Step: the AMD process reports
  its own enumerated devices to the launcher, and the launcher places the split on that report.
- No in-repo precedent for a child engine process. The `dlopen` precedent (`src/prefill/mmq_kq_loader.cpp:64`)
  stays in one process. B.2 needs a launcher that starts and pairs two engines and owns the shared hand-off
  buffer. That is new plumbing, not a speed cost.
