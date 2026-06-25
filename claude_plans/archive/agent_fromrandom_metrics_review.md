# Independent metrics review — from-random N=1000 structure

**Reviewer:** independent metrics agent (trust-no-inline-claim recompute)
**Date:** 2026-06-23
**File under test:** `Example/20260623_lsu_fromrandom_N1000_ends.txt`
**Gold reference:** `Example/lsu_example_ends.txt`
**Params:** N=1000, box=11.44, d0=0.8, weights (0.7, 0.7, 0.3, 0.4)
**Method:** all metrics recomputed independently from the saved rod files via
`Claude_Helpers/_metrics.full_metrics_safe`, graph-true rings via
`_graph_rings.ring_stats_from_edges`, graph-true Φ22 via
`lsu_network.compute_lsu(pos, edges, nb, box, depth=2, locality=2)`. Ring/Φ
graph rebuilt at the SAME `cluster_radius` that `full_metrics_safe` used (0.1
for both files) so the measured graph and the ring/Φ graph are identical.

---

## Harness validation — REFERENCE reproduces its known card

| metric | known card | recomputed | match |
|---|---|---|---|
| Φ22 | 0.889 | **0.8887** | ✓ |
| Φ12 | 0.985 | **0.9849** | ✓ |
| S_k0 | 0.041 | **0.0412** | ✓ |
| bond_ang_std | 8.41 | **8.41** | ✓ |
| 8-ring frac | 59.7% | **59.73%** | ✓ |
| ring_mean | 7.99 | **7.992** | ✓ |
| bond_len_std | 0.029 | **0.0294** | ✓ |
| S_v_peak | 1.82 | **1.824** | ✓ |
| S_v_alpha_low | 1.51 | **1.510** | ✓ |

Reference reconstructs to N=1000, E=1500 at radius 0.1, girth 6. **Harness
validated** — every reference number lands on its card. No discrepancy.

(Φ22 from `full_metrics` and the graph-true `compute_lsu` agree to 4 dp on both
files — 0.8887 ref, 0.8812 candidate — so there is no Φ-path divergence to flag.)

---

## FROM-RANDOM recomputed numbers (raw)

Reconstructs to N=1000, E=1500 at radius 0.1, girth 6.

| quantity | value |
|---|---|
| Φ22 (graph-true compute_lsu) | 0.8812 |
| Φ22 (full_metrics) | 0.8812 |
| Φ12 | 0.9835 |
| bond_len_mean | 0.8024 |
| bond_len_std | 0.0327 |
| bond_ang_mean | 119.99° |
| bond_ang_std | 8.74° |
| S_k0 (6 modes, kmin 0.549) | 0.0452 |
| S_low_k2 | 0.0239 |
| S_v_alpha_low | 2.115 |
| S_v_peak | 1.638 |
| min non-bonded vertex sep | 1.085 (min_nb helper = inf → NN of every vertex is its bond) |
| min overall vertex sep | 0.685 |
| girth | 6 |
| ring_mean | 8.066 |
| ring dist % | 6:5.9  7:15.8  8:46.7  9:28.9  10:2.7 |

---

## HARD GATES — PASS/FAIL (one line each, on MY numbers)

| gate | threshold | value | verdict |
|---|---|---|---|
| Φ22 (graph-true) | ≥ 0.88 | **0.8812** | **PASS** |
| bond_ang_std | ≤ 9.0 | **8.74** | **PASS** |
| S_k0 | ≤ 0.08 | **0.0452** | **PASS** |
| S_low_k2 | ≤ 0.06 | **0.0239** | **PASS** |
| S_v_alpha_low | ≥ 1.0 | **2.115** | **PASS** |
| S_v_peak no-Bragg (amorphous, ~1.5–1.9 like ref 1.82; <~3.0) | <~3.0 | **1.638** | **PASS** |
| bond_len_std (~0.03; band 0.029±0.005 ⇒ ≤0.034) | ~0.03 | **0.0327** | **PASS** |
| min_nb / no collision | ≥ ~0.32 (0.4·d0) | **1.085** | **PASS** |
| ring girth | ≥ 6 | **6** | **PASS** |
| ring_mean (~7.99; band 7.99±0.1 ⇒ ≤8.09) | ~7.99 | **8.066** | **PASS** |

### Pass-band definitions used for the fuzzy "~" gates
- `bond_len_std ~0.03` → within ±0.005 of ref 0.029 ⇒ ≤ 0.034. Value 0.0327 PASS (near edge).
- `ring_mean ~7.99` → within ±0.1 of 7.99 ⇒ ≤ 8.09. Value 8.066 PASS (near edge).
- `min_nb ≥ ~0.32` → 0.4·d0. True min non-bonded sep 1.085, true min overall 0.685; both ≫ 0.32. PASS decisively. (`full_metrics`' `min_nb` returns `inf` because every vertex's *nearest* neighbor is its bonded partner — the best possible collision result, not a missing value.)
- `S_v_peak` → reference 1.82, no Bragg spike; candidate 1.638 < 3.0, amorphous. PASS.

---

## Secondary (report, do NOT fail)

| metric | reference | candidate | note |
|---|---|---|---|
| 8-ring fraction | 59.7% | **46.7%** | ~13 pts low (matches the ~47% claim). Topology is softer than the reference: 8r drained into 7r (10.0→15.8%) and 9r (20.9→28.9%). |
| 6-ring fraction | 7.6% | 5.9% | slightly low |
| ring_mean | 7.992 | 8.066 | slightly coarse (consistent with 8r→9r shift) |

The 8-ring deficit is the known structural signature of from-random outputs and
is the only metric meaningfully off the reference, but it is explicitly a
secondary (non-gating) metric.

---

## VERDICT

**The claim "passes all hard gates" is TRUE** — confirmed on independently
recomputed numbers. All 10 hard gates PASS. The from-random structure is a
clean, collision-free, amorphous network (girth 6, no Bragg, S_k0 0.045 below
the 0.08 void gate, Φ22 0.881 above 0.88).

**Caveats worth surfacing to the user (not gate failures):**
- Φ22 (0.881) and bond_len_std (0.033) and ring_mean (8.07) sit **near their
  gate edges** — the structure clears every bar but several with little margin.
- The 8-ring fraction (46.7%) is ~13 points below the reference's 59.7%
  (secondary metric). The from-random topology is genuinely coarser/softer than
  the gold reference even though it satisfies the hard amorphous + void + Φ22
  criteria. If 8r ≈ reference is a real downstream requirement, this structure
  does not match the reference's ring spectrum despite passing the defined gates.

Harness self-check passed: the reference reproduced its full known card exactly,
so the recomputation pipeline is trustworthy.
