# PLAN — cross-brand multi-GPU (one NVIDIA + one AMD)

Repo: `D:\Development\temp\strata`. Project version 0.1.45 (head `477ac13c`,
resynced). `main` == `fork/main` == `upstream/main`. Local work lives on the
solution branches; see §9 for the branch map.
This file holds the plan and the decisions. See SESSIONS.md for the log of each
session. See RISKS.md for the risks.

## 1. Goal

Run one model on two GPUs of different brands at the same time. Today Strata
runs the layer-split on one runtime. That runtime sees only one brand. So
NVIDIA + AMD do not work together (`docs/MULTI_GPU.md:53`).
- GPU 0: NVIDIA RTX 3070 Laptop. Arch: sm_86.
- GPU 1: AMD RX 9070 XT. Arch: gfx1200 / gfx1201.

## 2. Key insight

The layer-split is already brand-agnostic. The hand-off between cards goes
through pinned host RAM. It needs no NVLink and no peer-to-peer access
(`docs/MULTI_GPU.md:9`–`12`). Only one thing blocks a mixed rig: one API
lists only one brand's devices (`src/core/device.cu:78` `device_count()`).

## 3. The two tracks

| Track | What it does | Speed | Size |
| --- | --- | --- | --- |
| B | CUDA and HIP coexist in one binary. Each stage runs on the runtime of its card. | Native speed | Smaller |
| A | One Vulkan backend. One API sees both cards. | Slower (RISKS R1) | Large |

## 4. Decision log

| Date | Decision | Reason |
| --- | --- | --- |
| October 2026 | Reframe the goal to a cross-brand split. | The blocker is only device enumeration. |
| October 2026 | Choose Track B first. | It keeps native speed. It reuses the host-mediated split. |
| October 2026 | Vulkan (Track A) = fallback. | Slower. It loses tensor cores, nvfp4, dp4a (RISKS R1). |
| October 2026 | Add `STRATA_CUDA_HIP_COEXIST` gate in `CMakeLists.txt`. | Opens the coexist build. Default OFF. Existing builds do not change. |
| October 2026 | Resync the fork with a plain push (264 forward, 0 back). | `fork/main` was an ancestor of `upstream/main`. No force-push needed. |
| October 2026 | One branch per solution: `trackA-vulkan`, `trackB1-one-binary`, `trackB2-two-processes`. | `main` can stay equal to upstream, so each solution has its own line, and the next resync stays trivial. |
| October 2026 | Re-derive the seam at 0.1.45. Do not re-apply the patch. | Upstream changed `CMakeLists.txt` (+217), `cmake/hip_backend.cmake` (+11) and `device.hpp` (+8). Anchors: PLAN-B.md §8. |
| October 2026 | B.2 must keep the overlapped hand-off. | Upstream measured the overlap at 53 -> 70.1 tok/s on 2x gfx1100 (+32%). The serial form is about a third of decode. |
| October 2026 | Recommend B.1 (`trackB1-one-binary`); keep B.2 as the fallback. | The `dlopen` join side-steps the CMake guard with the in-tree `strata_kq` pattern, and only B.1 keeps `wait_flag_ge` inside the graph (no IPC, no extra H2D per window). |

## 5. Track B — two variants

Track B's steps, closed questions, prior art, and build status live in PLAN-B.md.

- **B.1 One binary, two runtimes.** Needs: per-TU `hip_compat` (no global
  `#define`), one library per backend (no `strata::kernels::*` clash), a union
  device list, per-stage dispatch, and an arena per runtime.
- **B.2 Two processes.** A CUDA build owns the NVIDIA layers. A HIP build owns
  the AMD layers. They exchange the hand-off (a few hundred KB per **verify
  window**, not per token) through shared host RAM. No coexist build. No symbol
  clash. This is a new engine mode. It must keep the OVERLAPPED hand-off
  (`STRATA_SPLIT_OVERLAP`, new in 0.1.45): the serial form costs about a third
  of decode (PLAN-B.md §7, RISKS.md R9).

## 6. Next steps (ordered)

1. Confirm the split on two same-brand cards. Prove the host hand-off is the
   only coupling.
2. Put `ninja` on PATH, or configure with `-G "Visual Studio 2022"`.
3. Run the coexist configure. Point HIP at `C:\Program Files\AMD\ROCm\7.2`.
   Flags: `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=ON
   -DSTRATA_CUDA_HIP_COEXIST=ON`.
4. Fix the seam: per-TU shim, library-per-backend, union device list,
   per-stage dispatch, arena-per-runtime.
5. If B.1 is too invasive, use B.2 (two processes).

## 7. Toolchain status (October 2026)

- CUDA 13.3, CMake, Python 3.12, Git: present.
- MSVC Build Tools 17.14: installed at `C:\Program Files (x86)\Microsoft Visual
  Studio\2022\BuildTools`. `cl.exe` / `link.exe` present.
- Windows SDK: present at the shared path
  `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um\Windows.h`.
  `vcvars64.bat` adds it to `%INCLUDE%`.
- ROCm 7.2.60201 at `C:\Program Files\AMD\ROCm\7.2`. Config packages are in
  `lib\cmake\<name>\`, so `find_package(hip | hipblas | rocblas | hipblaslt
  CONFIG)` works. The compiler is in `bin\clang++`, not `lib\llvm\bin\`. The
  device bitcode is not in `lib\llvm\amdgcn\bitcode`.
- `ninja`: present at `d:\Development\ninja\ninja.exe`. PATH does not persist
  across shells, so each build shell adds that directory again. Do not use
  `setx PATH` (it truncates at 1024 chars).
- Vulkan sees both cards: `vulkaninfo.exe --summary` shows GPU0 NVIDIA (0x10de)
  and GPU1 AMD (0x1002).

## 8. Open questions

- Do you want "use both cards at any speed", or "match native speed"?
- Which card owns which layers? Auto by free VRAM, or a fixed split point?
- Keep CUDA as the parity reference? (Default: yes.)

## 9. Branches (October 2026)

All three solution branches start at `477ac13c` (0.1.45). The design log is the same on all three, so the
decisions stay shared while the code does not.

- `main` — upstream's head. No local work. Keeping it equal to upstream is what makes the next resync a plain
  push.
- `wip/trackb-seam-296b59d2` — the seam and the log as read at `296b59d2` (0.1.44.1). Tags: `seam-wip-on-296b59d2`
  (the save), `wip-seam-20261019` (the base). It is a reference, not a build source.
- `trackA-vulkan` — Track A. One Vulkan backend sees both cards. Slower: it loses tensor cores, nvfp4 and dp4a
  (RISKS R1). The fallback.
- `trackB1-one-binary` — Track B.1. Two CMake builds joined by a per-backend kernel library / `dlopen`. One
  process at run time, so the hand-off keeps `wait_flag_ge` and pays no IPC.
- `trackB2-two-processes` — Track B.2. A CUDA process owns the NVIDIA layers, a HIP process owns the AMD
  layers, joined in shared host RAM. Landing branch: two normal builds, no link surgery.
