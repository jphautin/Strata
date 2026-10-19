# PLAN-B — Track B: CUDA + HIP coexist for a cross-brand layer split

Repo: `D:\Development\temp\strata`. Version 0.1.44.1. See PLAN.md for the goal and SESSIONS.md for the log.

## 0. Status

Configure verified on this PC. With the MSVC environment active (`vcvars64.bat`) and the HIP compiler at
the 7.2 SDK, the coexist configure completes: CUDA 13.3 (nvcc + MSVC host) and HIP (clang 21) both register,
`hip::host` is found, ggml is fetched. `ninja` is present at `d:\Development\ninja\ninja.exe`.

Build verified (partial). `cmake --build ... --target strata-device` linked `strata-device.exe` clean under
coexist: `device.cu` and `pinned.cu` compile as CUDA (nvcc), host `.cpp` compile as clang, only deprecation
warnings. So B1 and B2 do not break the CUDA half.

The full `--target strata` build then failed: the host C++ was clang, and the engine and ggml add MSVC-only
flags (`/wdNNNN`, `/utf-8`, `/Zi`), which clang reads as file names. Forcing the host to `cl` (MSVC) instead
makes CMake's own HIP guard fail (`Windows-Clang.cmake:187`). So one coexist configure cannot hold both tool
chains on Windows. See §5 and §6.

Not done: B3 (union device list), B4 (library-per-backend kernels), B5 (per-stage dispatch), B6 (arena per
runtime), B7 (mixed-rig validation).

## 1. Open questions — closed

| Q | Question | Answer (grounded) |
| --- | --- | --- |
| Aim | Use both cards at any speed, or match native? | Both cards at native speed. Track B keeps the native CUDA and HIP kernels. |
| Which card owns which layers? | Auto or fixed? | Reuse the existing `--layer-split auto` (free VRAM) and `--layer-split K,K` (fixed). No new policy. `docs/MULTI_GPU.md:41`, `:63`. |
| Additive or replace CUDA? | Keep CUDA as reference? | Additive. CUDA stays the parity reference. `docs/MULTI_GPU.md:240`. |
| Target hardware | Which cards? | NVIDIA RTX 3070 Laptop (sm_86) + AMD RX 9070 XT (gfx1200/gfx1201). Windows. |
| Hand-off across brands | Is a GPU address shared? | No. The hand-off is host-mediated through pinned RAM. No NVLink, no peer-to-peer. `docs/MULTI_GPU.md:9`–`12`. |
| Vulkan features | Needed? | No. Track B needs no Vulkan feature. Vulkan (Track A) is the fallback only. |
| Device ids | How to name a mixed rig? | `--gpus cuda:0,hip:0`. Each id carries its runtime. New `parse_device_id` in `device.hpp`. |
| Kernel linking | How to avoid the name clash? | One library per backend: CUDA kernels and HIP kernels in separate TUs/libs. Mirror the `dlopen` in `src/prefill/mmq_kq_loader.cpp`. |

## 2. Prior art (what exists, what does not)

- No attempt at NVIDIA+AMD in one process. `docs/MULTI_GPU.md:53` says a mix is "Not supported".
- AMD cards share a model among themselves: `./setup.sh --backend hip --gpus 1,0` (`docs/AMD_HIP.md`).
- `STRATA_HIP_GFX906` (`CMakeLists.txt:64`–`155`) runs AMD as HIP by mapping CUDA names to HIP with a global
  `#define` shim, then forcing CUDA on. It is ONE runtime, not two.
- Precedents for "separate build": `sycl/` is its own project (`CMakeLists.txt:15`–`22`). `hip_compat` and
  `metal_compat` are global `#define` shims. `src/prefill/mmq_kq_loader.cpp` loads one kernel set with
  `dlopen` at run time.

## 3. The seam (files)

- `device.hpp` / `device.cu`: one-runtime device facts. `device_count()` = one `cudaGetDeviceCount`
  (`device.cu:78`); `DeviceArena` = one `cudaMalloc` (`device.hpp:72`).
- `generate.cpp`: the split sizes `n_dev` from `cudaGetDeviceCount` (`:2279`, `:2360`, `:2488`) and indexes
  `--split-device` / `CUDA<d>` ids into that one list.
- Kernels: `include/strata/kernels/*.hpp` declare `strata::kernels::name`; `src/kernels/cuda/*.cu` define it.
  CUDA and HIP use the same names → clash in one binary.
- HIP build: `hip_compat/cuda_runtime.h` `#define`s every `cuda*` to `hip*`; `hip_backend.cmake:104`–`126`
  force-includes it and re-tags every `.cu` as HIP.

## 4. Steps

Status key: done = coded this session; next = code it next; blocked = needs a build/toolchain first.

- B0 — Decide. Done. Aim, split policy, additive: closed in §1.
- B1 — Coexist build gate. Done (safe). `CMakeLists.txt`: added `option(STRATA_CUDA_HIP_COEXIST)` and a
  require check; it defines `STRATA_CUDA_HIP_COEXIST=1`. `hip_backend.cmake`: gated the global shim + `.cu`
  →HIP re-tag off under coexist, so CUDA TUs keep NVIDIA headers. `hip::host` + `roc::hipblas` still found/linked.
- B2 — Device-id seam in the host layer. Done (safe). `device.hpp`: `DeviceRuntime` enum, `DeviceInfo.runtime`,
  inline `parse_device_id("cuda:0"|"hip:0"|n)`, and `union_device_list()` declaration.
- B3 — Union device list. Next. Fill `union_device_list()` = CUDA devices then HIP devices, each with its runtime.
  The HIP device probe lives INSIDE the HIP-built library (`strata_hip_probe()`), exported as `extern "C"` in the
  shape of `strata_kq_lib.cu` (§8 anchor table). The CUDA host then never links `amdhip64` and never includes
  `<hip/hip_runtime.h>`; it calls one C function. This supersedes the earlier "declare them, then link `amdhip64`"
  idea, which made the engine carry the HIP import library. Under B.2 this is simpler: each process enumerates its
  own cards, and the launcher reads the AMD process's device report to place the split.
- B4 — One shared library per backend, at a C-ABI boundary. Next. `strata_cuda_kernels` (nvcc) and
  `strata_hip_kernels` (clang), each exporting an `extern "C"` entry table, copying the shape of
  `src/prefill/mmq_kq_lib.cu`: `STRATA_KQ_EXPORT` `:15-19`, `strata_kq_abi()` `:33` (interface version plus the
  argument block's size), `strata_kq_init(int device)` `:38` (start the runtime in the host's context), and
  `strata_kq_run(device, type, args, stream)` `:46` (the host's stream is passed in). The C ABI makes the cl and
  clang name mangling irrelevant; the DLL boundary hides the two toolchains; the ABI check rejects a mismatched
  pair at load; and `init(device)` loads the code objects while the VRAM for it is still free. The engine picks
  the set by `DeviceInfo.runtime` per stage, and loads it with `LoadLibraryA`/`dlopen` — `mmq_kq_loader.cpp:62`
  /`:64`, symbols by name `:71-73`. Required, not stylistic: it fixes the `strata::kernels::*` clash and the
  clang/MSVC mix.
- B5 — Per-stage dispatch. Next. In `generate.cpp`, place each `--layer-split` stage on the runtime that owns
  its card. The hand-off already uses pinned host RAM, so no P2P change is needed.
- B6 — Arena per runtime. Next. `DeviceArena`, `pinned.cu`, `copy_queue.cpp` allocate on the owning runtime.
  The shared expert arena in host RAM stays shared (`docs/MULTI_GPU.md:194`).
- B7 — Validate. Next. Run the split parity harnesses on the mixed rig (`tools/split_mtp_batch_parity.py`,
  `tests/hip/`, `tests/cuda/`). Add `tests/mixed/`.

## 5. The configure and first build (Oct 2026, on this PC)

Neither host works in ONE configure. Two needs fight over the same CMake compiler setting:
- Host = clang: the coexist CONFIGURE passes and `strata-device` links, but the engine and ggml add MSVC-only
  flags (`/wdNNNN`, `/utf-8`, `/Zi`) that clang reads as file names. The full `strata` build fails.
- Host = `cl` (MSVC): `enable_language(HIP)` raises `Windows-Clang.cmake:187` — "mixes Clang and MSVC … not
  supported. Use either Clang or MSVC as the compiler for all of C, C++, and/or HIP." Configure fails.

So CMake 4.3's Windows-Clang-HIP platform module forbids clang (HIP) and MSVC (cl) in one configure. Workable
B.1 is TWO CMake builds joined at link time by a per-backend kernel library / `dlopen` (see §6). The two builds:

```bat
set "PATH=d:\Development\ninja;%PATH%"
# Build 1 — engine host: MSVC + CUDA. vcvars puts cl.exe on PATH, which nvcc needs as its host.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_CUDA_HIP_COEXIST=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=86
# Build 2 — HIP kernel library: host clang + HIP, no CUDA language.
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_CUDA_HIP_COEXIST=ON ^
  -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1200" ^
  -DCMAKE_PREFIX_PATH="C:/Program Files/AMD/ROCm/7.2" ^
  -DCMAKE_HIP_COMPILER="C:/Program Files/AMD/ROCm/7.2/bin/clang++.exe" ^
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="C:/Program Files/AMD/ROCm/7.2"
# then link build-hip's kernel library into build-cuda's engine (the mmq_kq_loader.cpp dlopen pattern).
```

Findings (Oct 2026, this PC):
- clang-as-host configure passed and `--target strata-device` linked clean; `--target strata` then failed on the
  engine/ggml MSVC `/wd` flags read by clang.
- cl-as-host configure failed at `enable_language(HIP)` (`Windows-Clang.cmake:187`). This is the authoritative
  one-configure blocker.
- So do not seek a single configure with both backends. Use the two-build link (B.1) or two processes (B.2).

## 6. Decision: one binary vs two processes

- Confirmed (Oct 2026): ONE coexist configure cannot hold both toolchains on Windows. CMake's Windows-Clang-HIP
  guard forbids clang (HIP) + MSVC (cl) together, and clang-as-host breaks the engine's MSVC `/wd` flags.
- B.1 (one process) is therefore TWO CMake builds joined at link time by a per-backend kernel library or `dlopen`
  (see §5): a CUDA build = the engine host (MSVC + CUDA), a HIP build = the AMD kernel library (clang + HIP).
  Native speed, one process at run time. This matches the `sycl/`-separate-project and the `mmq_kq_loader.cpp`
  `dlopen` precedent.
- B.2 two processes = a CUDA process owns the NVIDIA layers, a HIP process owns the AMD layers, and activations
  pass through shared host RAM. No link join, no symbol clash, no toolchain mix. Each is a normal single-backend
  build. Cost: the hand-off itself is small (a few hundred KB per window, already pinned today). The costs that
  matter are the cross-card chunk overlap, the shared expert arena, and the pipeline. See §7 and RISKS.md R9.
- Rule: land B.2 first. It reuses two normal builds and needs no link surgery. Reconsider B.1 only if you need
  one process. Vulkan (Track A) stays the slower last resort.

## 7. What B.2 costs in speed (grounded, Oct 2026)

The kernels do not change. Each half still runs its native path at native speed: NVIDIA keeps the tensor-core,
nvfp4 and dp4a paths, AMD keeps the RDNA4 path. So B.2 does not lose the kind of speed Track A (Vulkan) loses
(RISKS R1).

The hand-off is not per token. `docs/MULTI_GPU.md:10-12`: a token crosses cards "once per verify window (a few
hundred KB through pinned RAM), not twice per layer", and needs no NVLink and no peer-to-peer access. The
same-brand split already pays the pinned D2H + H2D pair.

Upstream 0.1.45 documents the hand-off in two FORMS (`docs/MULTI_GPU.md:218-244`), and names the parts:
- Default, serial (`:220-226`): the host syncs stage 1, then stages and launches stage 2's graph, and so on.
- `STRATA_SPLIT_OVERLAP=1` (`:221-226`, stages on different cards only): the earlier stage's GPU writes the
  hand-off with volatile stores and raises a flag (`handoff_publish`); the next stage's graph is staged and
  launched ON A HELPER THREAD while the earlier stage still runs, and waits for the flag ON ITS OWN GPU
  (`wait_flag_ge`). This takes "one stream sync and one graph launch per window off the critical path".
- Measured, 2x RX 7900 XTX (`:228-237`): decode on a code prompt 53 -> 70.1 tok/s (**+32%**); prose 45 -> 55.9
  (+24%). The greedy output is byte-identical with the flag at 0 and 1. "Not measured on NVIDIA yet, where a
  sync and a graph launch cost less."

So the boundary is NOT about 1%. On a two-card AMD pair the serial form costs about a third of decode. B.2
must reproduce the OVERLAPPED form across processes, or it pays that.

Line refs below are read at `477ac13c` (0.1.45). Where a line was not re-checked, the row names the passage.

| What B.2 changes | Cost if you keep it | Cost if you drop it | What keeps it |
|---|---|---|---|
| The hand-off publish | about the same as today. One shared host page, one flag word. | A serial hand-off: up to a third of decode (`MULTI_GPU.md:228-237`: 53 -> 70.1 tok/s on 2x gfx1100). | A's GPU writes the payload into the shared pinned page, then sets the flag. B's host spins on it (us), then issues the H2D and stages B's graph on a helper thread — the shape at `:223-224`. |
| The GPU-side flag wait | one extra tiny H2D per window. | a hang, or a stale hand-off, if the peer dies. | B cannot use `wait_flag_ge` in its own graph: A cannot write a flag into B's GPU, and there is no P2P. So B copies one flag word to its GPU after its host sees the shared flag. Keep the bounded-wait + error-word discipline of `STRATA_SPLIT_WAIT_MS` (`:239-241`) and the drop test hook `STRATA_TEST_HANDOFF_DROP=N` (`:243`), across processes. |
| Prompt chunk overlap (card B chunk c ∥ card A chunk c+1) | none. | prompts -18/-20% (`MULTI_GPU.md:350`), and the +36% at 16K on 2 stages (`:212-215`). | Keep a hand-off ring, and size the chunks for the pipeline (`--prefill-pipe`, `:245-247`). |
| `--pipeline-windows 2` (default on a gfx12 split, `:361-364`) | none. | the measured +16% mean decode, +21% on code (the `--pipeline-windows` rows). | Keep the first card's recurrent-state copies (`:400`) visible to both processes, or keep the guessing stage whole in one process. |
| Expert residency (`--adapt-every`) | none. | the CPU-pool regime: the resident-RAM table spans 25, 32 and 71 tok/s on one PC; a helper card's per-layer round trip cost more than the CPU pool needed. | One arena mapping (`MULTI_GPU.md:210`: "the expert arena once, shared by all cards"), cut by layer range at open time; each process runs its swaps only over its own layers; one shared resident-set ledger. |
| CPU pool, aux threads, host thread | none, on this rig: both caches hold nearly every routed expert (`MULTI_GPU.md:5-7`). | the CPU-heavy paths slow on oversubscribed cores. | Give each process its own `--aux-cpus` set in `src/program/generate.cpp`. |
| Host RAM and pinning | none. | OOM: the arena is 22 GiB of locked RAM on a 32 GB rig; and two GPU contexts already push the WDDM 8 GiB pin cap (`MULTI_GPU.md:332`). | One shared file mapping; `cudaHostRegister` each physical range in ONE process only. |
| Start-up | seconds, one time. | none in steady state. | n/a — two contexts, two PCIe probes (`:11`), two profile loads, two autotuners. |

Bottom line for this rig (RTX 3070 Laptop sm_86 + RX 9070 XT gfx1200/1201, two cards, both caches nearly full):
- Keep the overlapped hand-off, the arena cut by layer range, and the pipeline. Then B.2's steady state is
  within a few percent of B.1.
- Use the SERIAL hand-off, and B.2 can lose about a third of decode. That number is upstream's own measured
  gain for the overlap (+32% on 2x gfx1100), not a guess.
- B.1 pays nothing in the publish row: its boundary is one `dlopen`-resolved call, no IPC, and it can keep
  `wait_flag_ge` inside the graph, because one process owns both cards' memory. That is the measured reason
  to prefer B.1 when the two-build link works.
- Measure before you spend B.2 on it. On the SAME-BRAND pair first, read `STRATA_SPLIT_OVERLAP=1` off and on:
  that puts the ~30% on your own box. Then read the B.2 rig against the same-brand split table
  (`docs/MULTI_GPU.md:350` area) and the bit-exact hand-off check `--layer-split K --split-device 0`
  (`:116`: "the bit-exact check of the hand-off, not a speed mode").

## 8. Rebase (Oct 2026): fork resync, branch map, and the anchors at 477ac13c

Resync: `fork/main` was an ancestor of `upstream/main` (264 commits forward, 0 back), so a plain
`git push fork main:main` was enough. No force-push. The tag `v0.1.45` was pushed to the fork too.
`main` == `fork/main` == `upstream/main` == `477ac13c`.

Branch map:
- `main` — never carries local work. It stays equal to upstream, so every later resync stays trivial.
- `wip/trackb-seam-296b59d2` (tag `seam-wip-on-296b59d2`; base marker tag `wip-seam-20261019` = `296b59d2`) —
  the seam and the design log exactly as read at `296b59d2` (0.1.44.1). It is the reference for the re-derivation.
  It is not a build source.
- `trackA-vulkan` — Track A: one Vulkan backend. The slower fallback (`RISKS.md` R1).
- `trackB1-one-binary` — Track B.1: two CMake builds joined by a kernel library / `dlopen`. One process at run time.
- `trackB2-two-processes` — Track B.2: a CUDA process and a HIP process, joined in shared host RAM. Landing branch.
The three solution branches start at `477ac13c`. The design log is the same on all three, so the log and the
decisions stay shared while the code does not.

Do NOT re-apply the seam as a patch. Upstream changed the same files (+217 lines in `CMakeLists.txt`, +11 in
`cmake/hip_backend.cmake`, +8 in `include/strata/core/device.hpp`). Re-derive it. Anchors at `477ac13c`:

| Seam point | Where it is now |
|---|---|
| The "enable only one" guard | `CMakeLists.txt:100-101`: `if(STRATA_ENABLE_CUDA AND STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)`. Add `AND NOT STRATA_CUDA_HIP_COEXIST`. |
| Backend options | `CMakeLists.txt:58` (`STRATA_ENABLE_CUDA`), `:64` (`STRATA_HIP_GFX906`), `:71` (`STRATA_ENABLE_HIP`). |
| Which branch of the tree runs | `if(STRATA_HIP_GFX906)` `:137-156` / `elseif(STRATA_ENABLE_CUDA)` `:157-…` (`enable_language(CUDA)` `:165`, `find_package(CUDAToolkit REQUIRED)` `:168`) / `elseif(STRATA_ENABLE_HIP)` `:212`. Only ONE branch runs. A coexist build needs one branch that does the HIP setup AND the CUDA setup. |
| HIP enable + SDK finds | `cmake/hip_backend.cmake:45` (`enable_language(HIP)`), `:58-59` (`find_package(hip/hipblas CONFIG REQUIRED)`), `:72-89` (target checks), `:95-108` (`strata_hip_runtime`, links `hip::host`). |
| The global CUDA→HIP force-include shim | `cmake/hip_backend.cmake:109-121`, over `include/strata/hip_compat/cuda_runtime.h`. Gate it OFF under coexist: it renames CUDA to HIP in every TU. |
| The `.cu` → HIP re-tag | `cmake/hip_backend.cmake:123-135`. Gate it OFF under coexist: under coexist the `.cu` files stay CUDA. |
| Host device seam | `include/strata/core/device.hpp`: `DeviceInfo` `:21`, `device_count()` `:44`, `device_summary` `:49`, `set_primary_device`/`primary_device` `:56-57`, `DeviceArena` `:80`. |
| Device side | `src/core/device.cu:78-80` (`device_count()`, `cudaGetDeviceCount` — HIP reports an error, not 0), `:227`, `:258`. |
| Split sizing sites | `src/program/generate.cpp`: `--split-device` `:2489`, `n_dev` `:2671-2673` and `:2763-2764`, the stage/device build `:2781-2807`, visible counts `:2843`, `:2910`, `:3268`, a HIP-side probe `:3615`. |
| The library template to copy | `src/prefill/mmq_kq_lib.cu`: `STRATA_KQ_EXPORT = extern "C" __declspec(dllexport)` `:15-19`, `strata_kq_abi()` `:33` (interface version + the argument block's size), `strata_kq_init(int device)` `:38`, `strata_kq_run(device, type, args, stream)` `:46`. `src/prefill/mmq_kq_loader.cpp`: `LoadLibraryA` `:62` / `dlopen` `:64`, symbols looked up by name `:71-73`. |
| The HIP-on-Windows build recipe | `tools/hip/build_windows.bat`: the toolchain-mix note `:10-11` ("CMake refuses to mix cl.exe with Clang for HIP"), the `rocm-sdk.exe` wheel root `:45-51`, `vcvars64.bat` `:58`, `-G Ninja` `:68`, the three compiler settings `:71-72`. |

Four upstream facts that change the design:
1. The hand-off already has an overlapped form, and it is named and measured. B.2 must reproduce it. See §7.
2. Upstream added `STRATA_PRIMARY_DEVICE` (`device.hpp:56-57`, `generate.cpp:2740-2749`, `:3262-3274` with
   `cudaSetDevice`). The comment at `generate.cpp:2740` records the reason: on Windows the HIP runtime
   enumerates the visible cards in an order that `HIP_VISIBLE_DEVICES` can only filter, not reorder. That is
   exactly B.2's "which card does my AMD process get". But `generate.cpp:2815-2817` refuses it with a layer
   split ("not supported with a layer split yet"), so a split still needs our own per-process choice.
3. There are TWO compat dirs now. `include/strata/platform/hip_compat/` is the gfx906 CUDA-name → HIP layer.
   `include/strata/hip_compat/` is the wave32 RDNA backend (`STRATA_USE_HIP=1`). Do not mix them in the seam.
4. `STRATA_HIP_GFX906` is a compat layer, NOT a coexist: it forces `STRATA_ENABLE_CUDA` ON and refuses
   `STRATA_ENABLE_HIP` (`CMakeLists.txt:66-68`), then makes `CUDA::cudart`/`CUDA::cublas` aliases of
   `hip::host`/`roc::hipblas` with the shim headers first (`:144-152`). CUDA and HIP are still mutually
   exclusive, and `docs/MULTI_GPU.md:53` still says a mix of NVIDIA and AMD cards is not supported. The
   premise of §5 and §6 stands.

Stale build dirs: `build-coexist-win` and `build-coexist-clhost` were configured against `296b59d2`. Delete
them before any build on the new base.
