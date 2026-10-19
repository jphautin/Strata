# SESSIONS — log

One entry per session. Newest entry at the bottom. Each entry has: date, goal,
actions, result, decision.

## Session 1 — October 2026 — Fork sync

- Goal: align the fork `jphautin/Strata` with upstream `Niko1121/Strata`.
- Actions: fork main HEAD `99f3dbd` and upstream `296b59d` had no common
  ancestor. Overwrote the fork: `git push --force origin upstream/main:main`.
- Result: fork main = upstream.
- Decision: the fork now tracks upstream.

## Session 2 — October 2026 — Design note

- Goal: answer whether Strata can use NVIDIA + AMD together.
- Actions: wrote `docs/VULKAN_BACKEND.md`. Mapped backends, kernel features,
  and blockers to Vulkan.
- Result: the doc lists feasibility, open questions, and a two-track plan.
- Decision: analysis stage complete.

## Session 3 — October 2026 — Vulkan enumeration probe

- Goal: test whether one API sees both cards.
- Actions: ran `vulkaninfo.exe --summary`. GPU0 `0x10de` NVIDIA RTX 3070
  Laptop (Vulkan 1.4.329). GPU1 `0x1002` AMD RX 9070 XT (Vulkan 1.4.349).
  Both report `PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`.
- Result: one API lists both brands. The blocker is only device enumeration.
- Decision: reframed the goal to the cross-brand split. Chose Track B first
  (native speed). Vulkan is the fallback (slower, RISKS R1).

## Session 4 — October 2026 — Track B toolchain setup

- Goal: prepare the local Track B coexist build.
- Actions:
  - Added `option(STRATA_CUDA_HIP_COEXIST ... OFF)` and relaxed the "enable
    only one" guard in `CMakeLists.txt`. Default OFF.
  - The user installed MSVC Build Tools 2022 with bare
    `winget install --id Microsoft.VisualStudio.2022.BuildTools`. It landed at
    the `(x86)` path. `cl.exe` / `link.exe` present.
  - Found ROCm 7.2.60201 at `C:\Program Files\AMD\ROCm\7.2`. clang in `bin`.
    Config packages in `lib\cmake`.
  - Windows SDK: confirmed present at the shared path
    `Windows Kits\10\Include\10.0.26100.0\um\Windows.h`. `vcvars64.bat` adds it
    to `%INCLUDE%`.
- Result: MSVC, Windows SDK, CUDA, and the HIP SDK are present. `ninja` is
  missing. The ROCm 7.2 layout differs from the 10.2 wheels the scripts expect.
- Decision: Track B is ready to try. Need `ninja` or the VS generator, then run
  the coexist configure.

## Session 5 — October 2026 — Track B: design closure and first seam edits

- Goal: focus on Track B. Close the open questions. Write PLAN-B.md. Check for a prior
  attempt. Start the implementation.
- Actions:
  - Prior art: no NVIDIA+AMD attempt. `docs/MULTI_GPU.md:53` says a mix is "Not supported".
    Closest precedents: `sycl/` (separate project), the `hip_compat`/`metal_compat` global
    `#define` shims, and the `dlopen` in `src/prefill/mmq_kq_loader.cpp`.
  - Wrote PLAN-B.md. Closed the open questions: aim = both cards at native speed; reuse
    `--layer-split auto`/fixed; additive (CUDA stays the parity reference); `--gpus cuda:0,hip:0`.
  - Seam edits, all gated (default OFF → existing builds identical):
    - `CMakeLists.txt`: added the `STRATA_CUDA_HIP_COEXIST` require check + `STRATA_CUDA_HIP_COEXIST=1` define.
    - `cmake/hip_backend.cmake`: gated the global shim force-include (`:104`–`112`) and the `.cu`→HIP
      re-tag (`:114`–`126`) off under coexist, so CUDA TUs keep NVIDIA headers. `hip::host` + `roc::hipblas`
      still found and linked.
    - `include/strata/core/device.hpp`: added `DeviceRuntime` enum, `DeviceInfo.runtime`, inline
      `parse_device_id("cuda:0"|"hip:0"|n)`, and `union_device_list()` declaration.
- Result: B0, B1, B2 done and safe. B3–B7 next. Not compiled: `ninja` not on PATH, and
  the coexist configure has not run.
- Decision: try B.1 (one binary) first. If the toolchain mix (`build_windows.bat:9`–`11`)
  will not link, fall back to B.2 (two processes). Vulkan (Track A) stays the slower last resort.

## Session 6 — October 2026 — coexist configure and first build

- Goal: run the Track B coexist configure and capture the real errors.
- Actions:
  - `ninja` installed at `d:\Development\ninja\ninja.exe` (added to PATH for the session).
  - First configure failed at `CMakeLists.txt:174` `enable_language(CUDA)`: `nvcc fatal: Cannot find compiler
    'cl.exe' in PATH`. Cause: the MSVC environment was not active.
  - Re-ran inside `vcvars64.bat` + ROCm 7.2 compiler paths. Configure completed: CUDA 13.3 (nvcc + MSVC host
    19.44), HIP clang 21, `hip::host` found, ggml fetched, "Configuring done".
  - Built `--target strata-device`: 9 units; `device.cu` and `pinned.cu` compiled as CUDA (nvcc), host `.cpp`
    as clang, `strata-device.exe` linked. Only deprecation warnings.
  - Built `--target strata`: failed. `clang++: error: no such file or directory: '/wd4005' ... '/utf-8'`.
    Cause: the host C++ was clang, but the engine and ggml add MSVC flags. Fix: `-DCMAKE_C_COMPILER=cl
    -DCMAKE_CXX_COMPILER=cl`.
- Result: B1 and B2 verified. The coexist configures and the CUDA half builds clean. B3–B7 not done.
- Decision: force the project host to MSVC and keep clang only for HIP. Then re-build `strata`. If the engine
  links, do B3 (union device list) in a clang host unit that can include `<hip/hip_runtime.h>`.
- Addendum: forcing the host to `cl` made `enable_language(HIP)` raise `Windows-Clang.cmake:187` — CMake 4.3
  forbids Clang (HIP) + MSVC (cl) in one configure. So no single coexist configure holds both toolchains on
  Windows. New decision: land B.2 (two processes), or B.1 as two CMake builds joined by a kernel library /
  `dlopen`. See PLAN-B.md §5 and §6.

## Session 7 — October 2026 — fork resync, one branch per solution, re-baseline

- Goal: resync the fork, then give each solution its own branch.
- Resync: `fork/main` was an ancestor of `upstream/main` (264 commits forward, 0 back), so a plain `git push fork
  main:main` was enough. No force-push was needed. The tag `v0.1.45` went to the fork too. `main` == `fork/main`
  == `upstream/main` == `477ac13c` (0.1.45).
- Save first: the working state (the design log plus the gated seam) went to `wip/trackb-seam-296b59d2`, tagged
  `seam-wip-on-296b59d2`. The older tag `wip-seam-20261019` marks the base `296b59d2`.
- Branches, all at `477ac13c`: `trackA-vulkan`, `trackB1-one-binary`, `trackB2-two-processes` (landing). `main`
  carries no local work, so every later resync stays trivial. The design log is identical on the three branches.
- Re-baseline check: the blocker is still there. `docs/MULTI_GPU.md:53` still says "a mix of NVIDIA and AMD cards"
  is not supported. The premise of Track B holds.
- The seam was NOT re-applied as a patch. Upstream changed the same files (+217 `CMakeLists.txt`, +11
  `cmake/hip_backend.cmake`, +8 `include/strata/core/device.hpp`). PLAN-B.md §8 now holds the anchors at
  `477ac13c` for the re-derivation.
- Upstream 0.1.45 facts that change the plan: the hand-off has a named, measured OVERLAPPED form
  (`STRATA_SPLIT_OVERLAP`, `handoff_publish`, `wait_flag_ge`; 53 -> 70.1 tok/s on 2x RX 7900 XTX), so the
  boundary is worth about a third of decode, not ~1%; `STRATA_PRIMARY_DEVICE` exists but is refused with a
  layer split; there are two `hip_compat` dirs; and `strata_kq` is an ABI-checked `extern "C"` shared library,
  which is the template for B.4.
- Housekeeping: `ninja` lives at `d:\Development\ninja\ninja.exe`, and PATH does not persist across shells.
  `build-coexist-win` and `build-coexist-clhost` were configured against `296b59d2` and are stale; delete them
  before any build on the new base.
- Decision: keep B.2 as the landing branch, and correct the perf claim: the hand-off boundary costs about a third
  of decode unless it stays overlapped. Next: re-derive the seam on the B branches, then the B.2 launcher.
