# Independent Stats Review — r10 candidate

Recomputed from saved rod files (`_verify_r10.py`), not trusting the results table.
S(k0) hand-computed (`|Σ exp(ik·r)|²/N` over the 6 ⟨100⟩ modes) AND cross-checked
vs `tools._vertex_structure_factor` — exact match. Bragg test uses RAW (unbinned)
S(k) over hkl∈[-8,8]³ to avoid the binning blind spot.

## Candidate r10 — `Structures/20260618_r10_coldT_w30_40k.txt`  (box=9.152, d0=0.8)

| Metric | Reported | Recomputed | Verdict |
|---|---|---|---|
| N (vertices) | 512 | 512 | PASS |
| E (edges) | — | 768 (=3N/2) | PASS |
| degrees | — | all = 3 (no deg-4 errors) | PASS |
| Φ_12 | 0.986 | 0.9863 | PASS |
| Φ_22 | 0.869 | 0.8687 | PASS |
| bond std | 0.104 | 0.1036 | PASS |
| ring mean | 8.16 | 8.1641 | PASS |
| 6-ring % | 3.1% | 3.12% (24/768) | PASS |
| S(k₀) | 0.144 | 0.1441 | PASS (but MARGINAL — see below) |
| S_low_k2 | 0.061 | 0.0608 | PASS |
| α(k<2) | +1.53 | +1.533 (n=4) | PASS |
| amorphous | yes | yes (raw S max=9.05 ≪ N=512) | PASS |

Internal consistency: ring_mean matches ring_distribution; 6-ring% = count/E. OK.
Density: 512/9.152³ = 0.667916 = 1000/11.44³ exactly. Box matching correct.

## Measurement-error checks (all clean)
- rods_in_file=871 → reconstructs to exactly 512 vertices / 768 edges (PBC duplicate
  rods collapse correctly).
- Min pairwise vertex distance (PBC-aware) = 0.488 > cluster_radius 0.1 → NO coincident
  /over-merged vertices.
- All 512 vertices are degree-3 → no round-trip / degree-4 artifacts.
- Box and density verified identical to reference.

## S(k₀) scrutiny — MARGINAL, not robust
6 modes at |k|=0.6865 (=2π/L), high variance as warned:
  per-mode = [0.2112, 0.2112, 0.0346, 0.0346, 0.1865, 0.1865]
  (the (±1,0,0) and (0,0,±1) pairs are ~0.19–0.21; the (0,±1,0) pair is ~0.035)
  mean=0.1441, std=0.0781, max=0.2112.
The <0.15 gate is MET but barely (0.1441 vs 0.15) and the mean is dominated by 2 of 3
axes; the worst single mode (0.211) exceeds the gate. Verdict: MARGINAL — passes on the
shell mean but is anisotropic and one resampled seed could tip it over 0.15.

## Amorphous claim — CONFIRMED
Raw unbinned S_v(k) global max = 9.05 (at |k|/k0≈5.9), top values [9.05, 8.09, 6.53…].
All O(1), far below O(N)=512. No sharp Bragg peak. Genuinely amorphous.

## Reference sanity (harness correct)
REF reproduces: N=1000, E=1500, all deg-3, Φ_12=0.9849, Φ_22=0.8887, bond std=0.0294,
ring_mean=7.992, 6-ring=7.60%, S(k₀)=0.0412 (hand=tool match), S_low_k2=0.0530,
α=1.510. Matches reported reference row. Pipeline is sound.

## OVERALL VERDICT
The reported r10 numbers HOLD UP — every metric reproduces within <1% (no discrepancy
>5%). Only caveat: S(k₀)=0.144 passes its <0.15 gate MARGINALLY and anisotropically,
not robustly. The bond-std gap vs reference (0.104 vs 0.029) is a real candidate
property, not a measurement error.
