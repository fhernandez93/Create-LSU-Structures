# 20260622_lsu_hyperuniform_N1000_ends.txt

N=1000, box=11.44, d0=0.8. A **near-hyperuniform amorphous trivalent network** produced by the
recipe discovered in the 2026-06-22 investigation. It **reproduces the reference's S(k) /
hyperuniformity** but its angle/ring distribution is **broader** than the gold reference (see caveat).
Format: tab-delimited rod endpoints (x1 y1 z1 x2 y2 z2), PBC-duplicated face-crossing rods — same
convention as `lsu_example_ends.txt`.

## How it was made (the recipe)
1. **Energy fix (now the production default).** `lsu_network.energy_components` was using simplified
   f1/f2 (harmonic `(L-d0)^2` + normalized `(cosθ+1/2)^2`), which made the energy ~6-8x too
   angle-dominated (bonds too soft) → its minimum lost hyperuniformity. Switched to the literal
   length-coupled **Keating** forms `f1=Σ(L²-d0²)²`, `f2=Σ(r_ij·r_ik + d0²/2)²` (weights unchanged
   0.7/0.7/0.3/0.4; f3/f4 unchanged). `_KEATING_F1F2` defaults ON; set `LSU_KEATING_F1F2=0` to revert.
2. **Near-hyperuniform seed PLACEMENT.** S(k0)=0.041 cannot be created by the WWW anneal — it must be
   SUPPLIED by the vertex placement. `Claude_Helpers/_hyperuniform_seed.py` generates points by
   collective-coordinate optimization (suppress low-k S(k) + soft hard-core) → seed S(k0)≈0.04,
   amorphous, valid deg-3 build.
3. **WWW anneal under Keating + uniformity penalty** (w=30, kmax=2) to HOLD the supplied low-k through
   topology annealing: `Claude_Helpers/_run_hyperuniform.py hyperu_w30 60000 10000 42`, T 0.06→0.012.

## Validated metrics (recomputed from this file) vs reference
| metric | this structure | reference | status |
|---|---|---|---|
| S_low_k2 | 0.049 | 0.053 | ✓ matched |
| S(k0) | 0.074 | 0.041 | ✓ low (~1.8x) |
| α (hyperuniformity slope) | +2.59 | +1.51 | ✓ hyperuniform |
| S_v peak / Bragg | 1.16 / none | 1.82 / none | ✓ amorphous |
| bond-angle mean | 119.97° | 119.98° | ✓ |
| **bond-angle std** | **11.63°** | **8.41°** | ✗ too broad |
| dihedral entropy | 0.860 | 0.796 | ✗ too disordered |
| Phi_22 (LSU) | 0.844 | 0.889 | ✗ low |
| rings 6/7/8/9 (%) | 13.4/30.3/38.9/15.4 | 7.6/10.0/59.7/20.9 | ✗ under-ordered |
| ring mean / girth | 7.62 / 6 | 7.99 / 6 | partial |

## CAVEAT — what this is and is NOT
- **IS:** a genuinely **hyperuniform, amorphous** trivalent network reproducing the reference's
  long-wavelength structure factor (S_low, α) — the primary target.
- **IS NOT:** a full reproduction. The **local geometry order is broader/lower** than the reference
  (angle-std 11.6° vs 8.41°, 8-rings 39% vs 60%). This is a **reachability limit** of the local WWW
  anneal starting from a DISORDERED seed: the reference's local order is *held* by the corrected energy
  (anneal-from-reference is stable) but *not reached* from a random start, even with a perfect-density
  seed (confirmed: pure-WWW with full annealing freedom also plateaus angle-std at ~11°).
- Closing the angle/ring gap likely needs a **seed that starts with the reference's local order**
  (gyroid-like / crystal_srs melt) or a **multi-stage protocol** — not more weight/schedule tuning.

See `claude_plans/N1000_investigation_results.md` and memory `lsu-energy-keating-balance-fix` for the
full investigation. Do NOT confuse this with the gold `lsu_example_ends.txt` (untouched).
