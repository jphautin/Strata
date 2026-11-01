# PLAN-C — the speed gain: the chain on the AMD card, CUDA0 as an expert holder

Branch: `trackC-speed` - cut from `trackB1-one-binary` at `d8674782` (2c2), because this plan changes
the seam and has to carry it. Repo: `D:\Development\temp\strata`, tree 0.1.45. Authored against 0.1.45
text: the app repo `D:\Development\Strata` is now 0.1.55 (`c62d5e3a`), so PLAN.md §10 and `CODE.md`'s `RE-DERIVE at 0.1.55` section (measured 2026-11-01) says
RE-DERIVE there, never re-apply.
This file is the SPEED plan. PLAN.md holds the goal and the decisions; PLAN-B.md holds
the PROVEN unit lines of the seam. Nothing here is PROVEN until a gate prints its numbers.

## 0b. ANCHOR DRIFT AT 0.1.55 (measured 2026-11-01, read-only sweep - this file's own anchors)

Every inline `file:NNNN` in this file is a line of the TEMP tree (0.1.45, `D:\Development\temp\strata`).
It is NOT a line of the app tree (0.1.55, `D:\Development\Strata` at `c62d5e3a`), where this plan must be
re-derived. Re-locate by TEXT, then use the measured 0.1.55 line in the second column.

| file | anchor | 0.1.45 -> 0.1.55 | delta |
| --- | --- | --- | --- |
| `src/program/generate.cpp` | `CUDA0'S ARENA IS BUILT BEFORE THE SPLIT IS KNOWN` | 3555 -> 4480 | +925 |
| `src/program/generate.cpp` | `--trim-stage-weights` | 899 -> 1045 | +146 |
| `src/program/generate.cpp` | `RouterLookahead` | 3569 -> 4494 | +925 |
| `src/program/generate.cpp` | `one distinct GPU per K` | MEASURED 0 HITS IN EVERY TREE - the name is a false premise | - |
| `src/program/generate.cpp` | `STRATA_SPLIT_OVERLAP` | 8448 -> 10050 | +1602 |
| `src/program/generate.cpp` | `set_handoff_flags` | 8450 -> 10052 | +1602 |
| `src/program/generate.cpp` | `STRATA_SPLIT_WAIT_MS` | MEASURED 0 HITS IN EVERY TREE - the name is a false premise | - |
| `src/program/generate.cpp` | `--layer-split` | 649 -> 784 | +135 |
| `src/core/verify.cpp` | `wait_flag_ge_or(m_flagA_` | 1981 -> 2249 | +268 |
| `src/core/verify.cpp` | `wait_flag_ge(m_flag_` | 1132 -> 1361 | +229 |
| `src/core/verify.cpp` | `mapped host memory` | 468 -> 591 | +123 |
| `src/kernels/cuda/verify_kernels.cu` | `wait_flag_ge_kernel` | 1165 -> 1224 | +59 |
| `src/kernels/cuda/verify_kernels.cu` | `handoff_publish_kernel` | 1200 -> 1259 | +59 |
| `src/kernels/cuda/verify_kernels.cu` | `wait_flag_ge_or_kernel` | 1425 -> 1496 | +71 |
| `src/kernels/cuda/verify_kernels.cu` | `wait_flag_ge_stamped_kernel` | 1223 -> 1282 | +59 |
| `docs/MULTI_GPU.md` | `Not supported` | 44 -> 84 | +40 |
| `docs/MULTI_GPU.md` | `Intel GPUs` | 53 -> 90 | +37 |
| `apps/host/src/lib/gpu-roles.ts` | `8 GB or less` | NEW at 0.1.55, line 92 - COEXIST with it | - |
| `apps/host/src/lib/gpu-roles.ts` | `gpus_by` | MEASURED 0 HITS IN EVERY TREE - the name is a false premise | - |
| `cmake/hip_backend.cmake` | `COEXIST` | MEASURED 0 HITS IN EVERY TREE - the name is a false premise | - |
| `CMakeLists.txt` | `hip_backend` | 104 -> 152 | +48 |

The files this plan edits, measured (0.1.45 base -> 0.1.55 app):
- `apps/host/src/lib/gpu-roles.ts`: ABSENT -> 278 lines
- `docs/MULTI_GPU.md`: 467 -> 314 lines
- `serve/server.py`: 7160 -> 9590 lines
- `src/core/expert_cache.cpp`: 808 -> 903 lines
- `src/core/expert_source.cpp`: 5291 -> 6566 lines
- `src/core/remote_experts.cpp`: 484 -> 502 lines
- `src/core/verify.cpp`: 4211 -> 5091 lines
- `src/kernels/cuda/fused_gr.cu`: 3035 -> 3325 lines
- `src/program/generate.cpp`: 16002 -> 19053 lines

PREMISE MEASURED AT 0.1.55 (`git grep` over `src include cmake serve docs`): `COEXIST` 0 / `STRATA_MULTI_BRAND` 0
/ `hip_engine` 0 / `hip:0` 0 / `cuda:0` 0 / `STRATA_GPU_ORDER` 0 / `cross-brand` 0 - the 0.1.55 tree still has NO
cross-brand machinery, so this plan's premise stands; only its anchors moved. The two builds are still joined by
`CMakeLists.txt` naming `cmake/hip_backend.cmake` at 104 (base) -> 152 (app).

## 0. Rules (inherited, unchanged)

- Measure, never guess. A number in this file is either cited to a `RUNS\runN` / a log / a
  code line, or it is tagged **PROJECTED** with its arithmetic shown.
- One commit per unit, only when its gate is green, `git commit -F <file>`. Each unit NAMES
  its own files. Nothing is pushed.
- Heavy gates run in the RAM window only: the daily serve session is stopped first, ONE
  command `cmd /c "D:\Development\temp\strata\<bat>"`, then the session restarts.
- A unit is PROVEN only at the level of its own gate. Speed claims need the pair measured
  on the SAME split and the SAME cache pin (`--vram-reserve-mib N`, RUNS\run14's confounder).

## 1. The claim under test

**Keep the layer chain mostly on the AMD card, and use CUDA0 as an EXPERT holder instead of
as a layer worker** — its own dense weights trimmed to its own layers, its freed VRAM holding
experts, and the AMD stage reading those experts through C.5's fetch. Estimated band
**+0 % to +20 % against the AMD-only 52.01 tok/s**, and that band is a model, not a
measurement. It is the ONLY shape on this rig where a speed gain is not arithmetically dead,
because the second card here is the slower one. Which EXPERT lives on which card, by utility, is
PLAN-D.md: C-c3 uses whatever placement PLAN-D measures green, it does not invent one.

## 2. The measured anchors

| configuration | decode | resident experts | PCIe line |
| --- | --- | --- | --- |
| AMD only, daily config (`RUNS\run2`, 2026-10-20) | **52.01 tok/s** median (50.41 / 58.68 / 52.01), prefill 452.90 @4K / 645.32 @32K | 3,580 slots / 6.82 GiB (the `run8` sitting) | 0.38 |
| AMD only, second point (`RUNS\run1`) | 50.0, prefill 422.8 / 601.1 | — | cache hits **41.8 / 47.5 / 52.4 %** |
| CUDA only (`RUNS\run14`, C.2 parity) | 27.72 / 28.03 tok/s | 3,208 / 3,186 slots = 6.13 / 6.09 GiB | 0.42 / 0.44 |
| CUDA only, before the join (`RUNS\run7`, C.5's AIM) | — | **107 / 55 slots** | — |
| mixed, HEAD `d8674782` (2c2 gate, K=14) | 11.9–12.5 tok/s — **answer WRONG, not a speed measure** | hip 3,584 experts HELD (tier 8.887 GiB) + CUDA0 283 slots (571 MiB) = 3,867 | `pcie_frac` 0.10–0.18 |
| upstream, 2x gfx1100 same brand | 53 -> **70.1 tok/s (+32 %)** with the overlap | — | — |

Measured per-card cost, one card running all 48 layers: AMD 19.2 ms/token, CUDA 35.7 ms/token
(the two tok/s pairs above). **PROJECTED: the CUDA card is 1.86x slower per layer on this rig.**

## 3. Why a gain can exist at all (the three levers, measured)

1. **Residency is the win, not parallelism.** The chain is sequential
   (`docs/MULTI_GPU.md:9-12`: the hand-off is pinned host RAM, no peer-to-peer). Moving a
   layer to another card moves work, it does not cut it. What a second card can buy is **VRAM
   for experts**, so decode stops re-reading them over PCIe. Measured: CUDA alone held 107/55
   slots while the AMD held ~3,580; the AMD-only hit ladder 41.8 -> 52.4 % is that lever's
   own effect.
2. **The per-card trim is a measured residency lever.** `generate.cpp:3707-3723`: the tree's
   own words are "every GPU holds only the dense weights of ITS layers (the PLE tensors stay
   everywhere)", `stage_trim = trim_asked && multi_gpu && !split_auto && !split_at.empty()`.
   On the 4-way IQ3_S rig it bought **CUDA2 1241 -> 2485 slots** and **CUDA3 533 -> 1772**
   (near doubling). On THIS rig it is measured NEUTRAL so far: at K=20 every CUDA0 number was
   identical to K=14 (2.39 GiB free after weights, 283 slots, 0 MiB free) because CUDA0 keeps
   **300 native projection matrices / 2,018.88 MiB** and a ~3.4 GiB full dense copy — the
   DEBT. `--trim-stage-weights` (PR #559) == `STRATA_STAGE_TRIM=1` (PR #639), and CUDA0's
   arena is built BEFORE the split is known (`generate.cpp:3717-3719`), so only an explicit
   `--layer-split` can trim it.
3. **The overlap hides the hand-off.** C.7 (NOT WRITTEN / NOT MEASURED): the flag words are
   MAPPED HOST MEMORY (`verify.cpp:468-470`, a host function may only raise them), the CUDA
   side has `wait_flag_ge_kernel` `verify_kernels.cu:1165`, `handoff_publish_kernel` `:1200`,
   `wait_flag_ge_or_kernel` `:1425`, host launchers `:1237 / :1450 / :1487 / :1494`; `verify.cpp`
   calls `wait_flag_ge(m_flag_, 1, cs)` at `:1132` and `wait_flag_ge_or` at `:1981`; the opt-in
   block is `generate.cpp:8448-8522` (`set_handoff_flags` at `:8477`, off beside `--pipeline-windows`
   at `:8458`). `STRATA_SPLIT_WAIT_MS` (default 30000) must still bound a stuck hand-off.

## 4. The arithmetic that bounds the shape (PROJECTED, from the measured unit costs)

- AMD expert cost: **0.6348 GiB/layer, 2.539 MiB/expert** (measured, `CODE.md` C.6 2c3 attempt 4).
  AMD Windows budget 15,447 MiB. At K=14: tier 8.887 GiB, 5.720 GiB free while HELD.
  **PROJECTED** K_max while HELD = (15.447 - 8.887 - 5.720 -> 0.840 GiB of other allocations)
  => 14.607 / 0.6348 = **K ~ 23**. Measured: K=20 HELD 12.695 GiB / 1.539 GiB slack; **K=22
  OFF THE TABLE** (the HIP runtime and its buffers take ~1.2 GiB beyond the tier).
- Therefore **"[0,K) HELD with K ~ 40" is arithmetically dead**: 40 x 0.6348 = 25.4 GiB > the
  card's 15.9 GiB. The chain can only sit on the AMD at K ~ 40 if the AMD's OWN held tier stays
  a budget-limited SUBSET (~3,580 slots of its 10,240 experts) and the rest streams.
- CUDA0 budget 8.00 GiB (WDDM). Measured today: ~7.9 GiB of dense for 48 layers (0.165 GiB/layer)
  => 2.39 GiB free after weights, 283 slots. **PROJECTED** with the trim, a card owning 8 layers
  carries 8 x 0.165 = 1.32 GiB of dense, leaving ~6 GiB => **~2,400 expert slots** at 2.539 MiB.
  That is the +20 % upper end of the band, and it is the number C-c1 must MEASURE.
- Residency total, projected at K=40 with the trim: ~3,580 (AMD, VRAM-bound) + ~2,400 (CUDA0)
  = ~5,980 slots vs 3,580 AMD-only = **+67 % residency**. Whether the extra residency converts
  into tok/s, and how much, is the whole question: **measure it, do not extrapolate the hit
  ladder.**

## 5. The units (ordered, each with its gate)

**C-c1. Close the debt: CUDA0 keeps only ITS layers' dense weights.**
Files: `src/core/native_dense.cpp`, `src/core/layer.cpp`, `src/program/generate.cpp`, plus the
pack site. Gate `t24c1`: the SAME two splits (K=14 and K=40) must print DIFFERENT CUDA0 numbers.
Green shape: `300 native projection matrices / 2,018.88 MiB` falls to the card's own share;
free-after-weights rises; `expert cache slots` rises at cache time. Refused before that: the
`--trim-stage-weights` line stays neutral, then the unit is NOT green and the trim's gate
(`generate.cpp:3723`) is the suspect. CHECK both brands (the trim is HIP-side too, `:3709`).

**C-c2. Raise K with the library holding a BUDGETED subset HELD.**
Files: `src/hip/engine_bridge.cpp`, `src/core/hip_kernels_loader.cpp`, `include/strata/core/hip_engine_abi.hpp`,
`src/program/generate.cpp`. Gate `t24c2`: `MIXED LOAD` at K = 34 and 40 printing (a) the AMD's
held tier NOT growing past its VRAM budget (~3,580 slots, the HELD tier chosen by budget and not
by layer count), (b) CUDA0's slots from C-c1, (c) NO refusal. **DESIGN QUESTION to answer by
measurement first**: the HELD hold is per stage over `[lb,le)` (`stage_build_into`,
`engine_bridge.cpp:470-590`, `session_init(...lb,le)`) — if the tier cannot be budgeted
independently of the layer range, THIS unit owes a second stage range (compute on the AMD, hold
less) rather than a new knob. Do not add an env knob to a gate that is meant to run the product.

**C-c3. CUDA0 as the expert holder, read through the join.**
Files: `src/core/remote_experts.cpp`, `src/core/remote_expert_opt.cu`, `src/core/expert_cache.cpp`,
`src/core/peer_experts.cpp`, the CUDA side of the bridge. This is C.5 PROVEN (`RUNS\run16`, 2026-10-23)
turned AROUND: C.5 put the static tier on the NVIDIA helper and the arena got holes at exactly the
pairs it holds (`expert_layout_drop` -> `pos[]`, `present()` the single "is this expert in RAM?"
gate), and the bytes were measured EQUAL at three sites (RAM arena PRESENT / helper tier fetched
back DROPPED / the library's own tier, both). Gate `t24c3`: the arena's `loaded 23.42 GiB` falls by
the tier's bytes, the hit-rate line rises against the AMD-only ladder (41.8 / 47.5 / 52.4 %), and
the SAME expert bytes hash EQUAL at the three sites. The measured trap to respect: after the arena,
a fresh PINNED host allocation is refused (WDDM commit) — C.5 needed the daily session stopped, and
its pass-10 WDDM2 direct host read CRASHED `0xC0000005` (the small `RemoteMeta` is host-visible, the
GiB cudaMalloc arena is NOT). Staged pinned copy stays.

**C-c4. The overlap across the brand boundary (C.7).**
Files: `verify_kernels.cu`, `verify.cpp`, `generate.cpp`. Gate `t24c4`: `STRATA_SPLIT_OVERLAP=1`
and off printed as a PAIR on the SAME split, plus the stuck shape printing its own line rather
than hanging (`STRATA_SPLIT_WAIT_MS` 30000). Rule from C.7's own words: the flags are host-mapped
words, so the CUDA host raises them and the HIP side may only READ them — never a CUDA kernel on a
HIP stream or the reverse.

**C-c5. The benchmark, and the decision (C.10).**
Files: `bench/`, the user's `bench/results/2026-10-19-rx9070xt-coder/`. Gate: the tok/s pair on the
daily config against the two bars — **AMD only 52.01** and **CUDA only 27.72** — with the cache PINNED.
C.10's aim is the decision itself: "keep both brands in one binary" against "run the AMD card alone".
If the mixed pair is under 52.01, that decision LOSES on this rig and `docs/MULTI_GPU.md:53` (C.9)
must say the mix is for packs that do not fit the AMD alone, not a speed promise.

## 6. Do not re-try (measured dead ends, from the 2c3 cut)

- The KEEPHB control (`STRATA_2C3_KEEPHB`): the payloads are BYTE-IDENTICAL across the free/keep
  control (`handoff-A.p0..p3` equal `handoff-B.p0..p3`, SHA256 at all four windows) — the hand-off
  free is not the fault.
- `TEMP-2c3-PFLEG` (C.1's prompt leg through the HELD stage): it BROKE window 0 (w0 mean KL 16.404764
  against the yardstick's 0.000091) and double-counted the seat (bridge seat 2 while the host ASKS
  pos 1); run A died `copy_i32_from_mapped launch: invalid argument` (the library refuses an
  un-captured T: only the leg's T 2 and the decode's T 1 graphs existed).
- Raising K with the debt OPEN is not a lever on CUDA0: at K=20 every CUDA0 number was identical to
  K=14. K buys AMD tightness only.
- The absolute numbers drift every sitting (`expert_slots` 33/118/447/728, `vram_free_mib` 0/236/654,
  `pcie_frac` 0.10/0.18): only the REPRODUCED pair (the 0.223310 jump at position 2, the 3,584-expert
  / 5.720 GiB lines) is a finding.

## 7. State and resume

STATE 2026-10-31 (day closed): HEAD `d8674782` = 2c2; C.6 half (b) unit 2c3 still owed its fix — the
mixed answer's head is `198 2523 ...` against C.1's `198 1 248046`, mean KL 0.000091 at w0 and
0.223310 / 10.737312 at w1 / w2, payloads byte-EQUAL across the join. **PLAN-C starts only AFTER 2c3
is green**: without a correct mixed run, no speed number means anything.
resume = `CODE.md` C.6 STATE, then C-c1 (the debt) as the first gate, and its first measurement is
the two CUDA0 print pairs at K=14 and K=40.

BRANCH LINE (2026-11-01): `trackC-speed`, cut at `d8674782` (2c2), carries 2c2's seam plus ONE `docs:`
commit (`PLAN-C.md`). No code unit is committed here: no PLAN-C gate is green. Pushed to `fork`
(jphautin/Strata) only - never `main`, never `upstream`.
