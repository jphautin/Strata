# PLAN-D — the experts divided by UTILITY: which expert lives on which card

Branch: `trackD-placement` - cut from `trackB1-one-binary` at `d8674782` (2c2), because the placement
re-points the seam's own mechanism (`expert_layout_drop` -> `pos[]`, `present()`, `fetch`). Repo:
`D:\Development\temp\strata`, tree 0.1.45. Authored against 0.1.45 text: the app repo
`D:\Development\Strata` is now 0.1.55 (`c62d5e3a`), so PLAN.md §10 and `CODE.md`'s `RE-DERIVE at 0.1.55` section (measured 2026-11-01) says RE-DERIVE there.
This is the PLACEMENT plan. PLAN.md = goal and decisions. PLAN-B.md = the seam's PROVEN units.
PLAN-C.md = the speed shape (the chain on the AMD card, CUDA0 trimmed). PLAN-D.md decides WHICH
EXPERTS sit on WHICH card, and it must be measured before PLAN-C's C-c3 picks a policy.

## 0b. ANCHOR DRIFT AT 0.1.55 (measured 2026-11-01, read-only sweep - this file's own anchors)

Every inline `file:NNNN` in this file is a line of the TEMP tree (0.1.45, `D:\Development\temp\strata`).
It is NOT a line of the app tree (0.1.55, `D:\Development\Strata` at `c62d5e3a`), where this plan must be
re-derived. Re-locate by TEXT, then use the measured 0.1.55 line in the second column.

| file | anchor | 0.1.45 -> 0.1.55 | delta |
| --- | --- | --- | --- |
| `src/program/generate.cpp` | `expert_cache_remote` | 546 -> 677 | +131 |
| `src/core/expert_cache.cpp` | `one counter shared by all 48` | 648 -> 739 | +91 |
| `src/core/expert_cache.cpp` | `read_expert_profile` | 74 -> 163 | +89 |
| `src/core/expert_source.cpp` | `the vote-ranked top M` | 2293 -> 3132 | +839 |
| `src/core/expert_source.cpp` | `Raise --resident-budget-gib` | 3280 -> 4195 | +915 |
| `src/core/remote_experts.cpp` | `a dropped expert has NO RAM copy` | SEAM-OWNED: line 98 at head-2c2, 0 hits at 0.1.55 - RE-ADD it | - |
| `src/core/remote_experts.cpp` | `two copies fewer per layer` | 223 -> 234 | +11 |
| `src/core/remote_experts.cpp` | `STRATA_REMOTE_ZEROCOPY` | 223 -> 234 | +11 |
| `src/core/remote_experts.cpp` | `STRATA_REMOTE_SPIN` | 83 -> 85 | +2 |
| `src/core/remote_experts.cpp` | `fill_slot_blocking` | 189 -> 194 | +5 |
| `src/core/remote_experts.cpp` | `original_row_` | 127 -> 132 | +5 |

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

## 0. Rules (inherited)

Measure, never guess; a number is cited to a log / `RUNS\runN` / a code line, or tagged **PROJECTED**.
One commit per unit when its gate is green, `git commit -F`, the unit NAMES its own files. Nothing pushed.
Heavy gates (anything that loads the 23.42 GiB arena) run in the RAM window: daily serve stopped first,
ONE command `cmd /c "D:\Development\temp\strata\<bat>"`, session restarted after.

## 1. The claim

The layer split decides **which card computes which layer** — that is a SPEED axis, and on this rig
the slow card must compute as little as possible (PLAN-C). The EXPERT POOL is a SECOND, independent
axis: each of the 12,288 experts (48 layers x 256) may live in the RAM arena, resident on the AMD,
or resident on CUDA0. Divide it by **utility**: the ranked (hottest) experts resident on the card that
COMPUTES their layer, the cold ones in the arena, so the PCIe reads fall only where they buy back
VRAM that is actually free.

**Nothing here is a gain yet.** The gain is a measured consequence of two things the tree already does
(C.5 PROVEN, and the ranked profile already read) applied at once, and the one cost that can kill it
(the fetch through host RAM instead of a PCIe read).

## 2. The measured anchors

| quantity | measured |
| --- | --- |
| experts | 48 layers x 256 = **12,288** |
| one expert | **2.539 MiB** (so 1 GiB = ~402 slots, 6 GiB = ~2,400) |
| expert arena | `loaded 23.42 GiB at 0.88-1.36 GiB/s`, `locked 16115 MiB`, `17 slices pinned (7 GiB)` |
| AMD-only residency | **~3,580 slots / 6.82 GiB** (`run8` sitting); hits **41.8 / 47.5 / 52.4 %** (`run1`); PCIe **0.38** |
| CUDA-only residency | **3,208 / 3,186 slots = 6.13 / 6.09 GiB** (`run14`); PCIe **0.42 / 0.44** |
| CUDA alone, before the join | **107 / 55 slots** (`run7`, C.5's AIM) |
| mixed, K=14 (HEAD `d8674782`) | hip **3,584 experts HELD** (tier 8.887 GiB, 5.720 GiB slack) + CUDA0 **283 slots (571 MiB)** = 3,867 |
| counter trap | `expert_cache.cpp:648`: "one counter shared by all 48 - which is what confined the measured hit rate to **2.97 %**" |

## 3. The mechanism already in the tree (read, not written)

1. **A ranked expert list exists and is read.** `expert_cache.cpp:75-169` `read_expert_profile`: header
   `version, nl, ne, want, n_ranked`, then `ranked[r] = {order[r] / n_expert, order[r] % n_expert}` —
   the pairs are ALREADY in rank order.
2. **The rank is utility, not position.** `expert_source.cpp:2293`: "the vote-ranked top M (the experts
   most tokens predict)". `:3280`: "Raise `--resident-budget-gib` (the lend region comes after the
   ranked experts)" — residency is already handed out in rank order, budget-first.
3. **A card's static tier is filled from that list, with a slot count.** `remote_experts.cpp:189-220`
   wants "the device, ranked experts and positive slot count" (`STRATA_REMOTE_RANK_START`,
   `STRATA_REMOTE_DROP_RESERVE_MIB` default 1024).
4. **The RAM arena gets holes at exactly the pairs the tier holds.** `expert_layout_drop` builds `pos[]`,
   `present()` is the single "is this expert in RAM?" gate (`expert_layout.hpp:18-42` has NO `pos`/`present`
   at `5c90c5ee` — C.5 PORTED these onto this tree, and it is PROVEN 2026-10-23 in `RUNS\run16`).
5. **The read-back.** `remote_experts.cpp:98`: "a dropped expert has NO RAM copy - its bytes live in
   this tier's VRAM"; `aux_fetch`, `RemoteExperts::fetch`, `copy_blob`.
6. **The equality already measured.** C.5: the SAME expert bytes hash EQUAL at THREE sites — the RAM
   arena (a PRESENT pair), the helper tier fetched back (a DROPPED pair), the tier inside the library
   (both). That is the placement's correctness marker, already earned once.
7. **Upstream's knobs are CUDA-only and do NOT cross brands**: `STRATA_REMOTE_DROP=1`,
   `STRATA_REMOTE_RANK_START`, `STRATA_REMOTE_DROP_RESERVE_MIB` (`remote-drop.patch` reads CUDA-to-CUDA
   only: `cudaGetDeviceCount`, one helper `remote_dev[0]`, no layer split, refuses `STRATA_ARENA_MMAP`).
   `expert_cache_remote[0] > 0` also refuses WITH `--layer-split` (`generate.cpp:3303-3323`): one or the
   other, never both. **That refusal is the seam PLAN-D must open.**

## 4. The constraints a placement must respect

- **Compute locality.** One layer is computed by ONE card. An expert of that layer on the OTHER card is
  fetched per use through pinned host staging — `remote_experts.cpp:293-294` counts it: "two copies fewer
  per layer, each of which is a PCIe round trip", `STRATA_REMOTE_ZEROCOPY=0` copies, `:421-425`
  `STRATA_REMOTE_SPIN`. Same-brand zero-copy was measured; **cross-brand zero-copy is NOT a thing to
  assume** — see the next bullet.
- **No peer path across brands.** `peer_experts.cpp` never compiled on HIP and `cudaMemcpyPeer` /
  `cudaDeviceDisablePeerAccess` are refused on HIP (commit `e6ad6c81`, the #2666 probe). The ONLY
  cross-brand channel is host RAM. So the placement's cost model is: resident-on-this-card = free read;
  resident-on-the-other-card = a D2H through pinned memory; in the arena = a PCIe/file read.
- **The profile is static, the router is per token.** A placement tuned to the daily config is not tuned
  to another prompt; the AMD-only ladder (41.8 / 47.5 / 52.4 %) is that ladder's OUTCOME. Off-workload, a
  ranked tier can LOSE hits. `expert_cache.cpp:648`'s 2.97 % counter trap says per-layer counters are
  required before any per-card claim is believed.
- **CUDA0's room is small today**: 283 slots (571 MiB) inside 8.000 GiB, because it carries the dense
  weights of ALL 48 layers (2,018.88 MiB of native projections + a ~3.4 GiB full copy). The trim
  (PLAN-C C-c1) is what makes the room real: **PROJECTED ~2,400 slots** once CUDA0 keeps only its own
  layers. So a placement on CUDA0 is a TARGETED lever (hundreds to ~2,400 of 12,288), never a
  redistribution.
- **The WDDM traps C.5 already hit and must not be re-invented**: after the arena a fresh PINNED
  host allocation is refused (commit exhausted; error 98 shares the "out of memory" string); the
  WDDM2 direct host read of a cudaMalloc'd tier CRASHED `0xC0000005` (the small `RemoteMeta` is
  host-visible, the GiB arena is NOT); a leftover unexited `strata.exe` holding a 23.6 GiB working
  set made every fetch say out of memory. Staged `cudaHostAlloc` at open, sized to `lay.max_blob`, stays.

## 5. The units (ordered, each with its gate)

**D-d1. Print the placements side by side.**
Files: `src/core/expert_source.cpp`, `src/core/expert_cache.cpp`, `src/program/generate.cpp`.
Gate `t24d1`, the SAME pack and the SAME split, THREE placements of the CUDA0 tier: **P0** none dropped
(today's mixed: 283 slots), **P1** the TOP ranked pairs of the layers CUDA0 itself computes, **P2** the
top ranked pairs of the AMD's layers. Each must print: resident slots, the hit-rate line, the PCIe line,
the arena `loaded`. GREEN = the three print DIFFERENT slots/hits/PCIe AND P1 >= P0 (the placement that
keeps each expert on its computing card wins). If P2 > P1, the placement axis is real but the
fetch cost already beats locality, and D-d3 is the blocker, not a win.

**D-d2. Open the refusal: ranked tier + `--layer-split` together, cross-brand.**
Files: `src/core/remote_experts.cpp`, `src/core/remote_expert_opt.cu`, `src/core/expert_cache.cpp`,
`src/core/peer_experts.cpp`, the CUDA side of the bridge. Today `expert_cache_remote[0] > 0` refuses
WITH `--layer-split` (`generate.cpp:3303-3323`). Gate `t24d2`: the tier fills on CUDA0 BEFORE the arena,
the arena re-laid with holes at EXACTLY those pairs, and the SAME expert bytes hash EQUAL at the three
sites (C.5's marker, roles reversed). This is the correctness gate: no tok/s is read in it.

**D-d3. Measure the fetch cost against the PCIe read it replaces.**
Files: `remote_experts.cpp`, `remote_expert_opt.cu`. Gate `t24d3`: on ONE split and ONE placement, print
the tok/s pair with `STRATA_REMOTE_ZEROCOPY=1` and `=0`, and the per-window copy accounting; the D2H
count per window against the PCIe reads the drop removed. A placement is kept ONLY if its PCIe line
falls AND its tok/s pair does not. Pin the cache (`--vram-reserve-mib N`) — `run14` showed the AUTO
cache is a confounder (3,208 / 3,186 slots inside one sitting).

**D-d4. The workload check on the profile.**
Files: `expert_cache.cpp` (the profile read), the tools that build it. Gate `t24d4`: the SAME placement,
THREE prompt families (the daily config, the one-shot 16-token shape, an off-profile prompt); the hit
rates must be printed per family. Green = the ranked tier helps where the profile was built AND the
off-profile family does not silently lose more than it gains; otherwise the placement is OFF by default
and the docs say so.

**D-d5. Feed the decision.**
Gate = C.10's pair (the daily config against **AMD only 52.01 tok/s** and **CUDA only 27.72**) with the
cache pinned. If the placement's pair is under 52.01 on the coder pack, PLAN-D's honest conclusion is
that it is an `iq3_xxs` (75.84 GB) and larger-pack lever, not a coder-pack speed win, and C.9's line 53
says exactly that.

## 6. Do not re-try (measured dead ends)

- `expert_cache_remote` and `--layer-split` at once: refused today, one or the other (`generate.cpp:3303-3323`).
- Same-brand +32 % (53 -> 70.1 tok/s, 2x gfx1100) as a bar for THIS rig: it cannot be made here (one NVIDIA
  8.000 GiB + one AMD 15.922 GiB, PLAN.md §4).
- `peer_experts.cpp` / `cudaMemcpyPeer` across brands: does not compile on HIP (`e6ad6c81`).
- The absolute drift: `expert_slots` 33 / 118 / 447 / 728, `vram_free_mib` 0 / 236 / 654, `pcie_frac`
  0.10 / 0.18 move every sitting. Only a REPRODUCED pair is a finding.

## 7. State and resume

STATE 2026-10-31 (day closed): 2c3 (C.6 half (b), the mixed WINDOW ENTRIES) is still owed its fix — the
mixed answer is `198 2523 ...` against C.1's `198 1 248046`, mean KL 0.000091 at w0, 0.223310 at position
2, 10.737312 at w2, payloads byte-EQUAL across the join. **PLAN-D runs only after 2c3 is green**: a
placement measured on a run whose answer is wrong measures nothing.
resume = `CODE.md` C.6 STATE, then C-c1 (PLAN-C, the CUDA0 range-cut) as the enabler, then D-d1 as the
first placement gate, whose first measurement is the THREE slot/hit/PCIe print pairs at one split.

BRANCH LINE (2026-11-01): `trackD-placement`, cut at `d8674782` (2c2), carries 2c2's seam plus ONE `docs:`
commit (`PLAN-D.md`). No code unit is committed here: no PLAN-D gate is green. Pushed to `fork`
(jphautin/Strata) only - never `main`, never `upstream`. D-d2 and PLAN-C's C-c3 are the SAME seam work:
run it once, as D-d2, and cross out C-c3 in `PLAN-C.md`.
