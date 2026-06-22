# Independent stats/metrics review — N=1000 investigation

Reviewer: independent stats/metrics agent. Every network reloaded FROM its saved
rod file and recomputed via `Claude_Helpers/_metrics.full_metrics_safe(box=11.44,
d0=0.8)` (CPU only). Audit driver: `Claude_Helpers/_audit_n1000.py`; raw dump:
`claude_plans/_audit_n1000_raw.json`. The harness's `S_k0` was independently
cross-checked against a from-scratch mean of the 6 lowest |k| modes computed on
the SAME reconstructed positions/merge radius — they agree to 4 decimals on every
file, so the table is internally consistent with the harness.

## Reconstruction integrity (degree-3 / N / E / density) — ALL PASS

| file | merge r | N | E | rods | density | degree hist | min_nb |
|---|---|---|---|---|---|---|---|
| REFERENCE | 0.1 | 1000 | 1500 | 1653 | 0.6679 | {3:1000} | inf |
| baseline_ak1000 | 0.1 | 1000 | 1500 | 1656 | 0.6679 | {3:1000} | 0.374 |
| A1 | **0.04** | 1000 | 1500 | 1649 | 0.6679 | {3:1000} | **0.062** |
| S1_s42 | 0.1 | 1000 | 1500 | 1652 | 0.6679 | {3:1000} | 0.243 |
| S1_s7 | 0.1 | 1000 | 1500 | 1656 | 0.6679 | {3:1000} | 0.237 |
| probe_w0_100k | 0.1 | 1000 | 1500 | 1675 | 0.6679 | {3:1000} | 0.265 |

- Density N/box^3 = 1000/11.44^3 = **0.66792** (= claimed 0.668) on every file. PASS.
- Every vertex is degree 3 after correct reconstruction — `np.bincount(edges)` is
  exactly {3:1000} for all six. No degree-4 over-merge artifact survives.
- A1's collision is real and confirmed: min_nb = **0.062** (0.077·d0). The default
  0.1 merge radius DOES wrongly fuse it (the harness auto-retries to 0.04, which I
  reproduce). Only A1 trips it; baseline/S1/probe all have min_nb ≥ 0.24 and use
  the default 0.1. Reference min_nb = inf (every vertex's nearest neighbor is its
  bond partner — healthy, expected, NOT a flag).
- rods-in-file ~1650 > E=1500 on every file: expected PBC-split rod segments. Not a flag.

## Recomputed vs results-table claims — confirm/correct

Format: my recompute (results-table claim). |Δ|>5% flagged.

| metric | REFERENCE | baseline | A1 | S1_s42 | S1_s7 | probe_w0 |
|---|---|---|---|---|---|---|
| Φ_12 | 0.985 (0.985) | 0.985 (0.985) | 0.982 (0.982) | 0.983 (0.983) | 0.980 (0.980) | 0.984 (—) |
| Φ_22 | 0.889 (0.889) | 0.865 (0.865) | 0.850 (0.850) | 0.857 (0.857) | 0.848 (0.848) | 0.863 (0.863) |
| bond mean | 0.800 | 0.818 | 0.817 | 0.819 | 0.832 | 0.826 |
| bond std | 0.029 (0.029) | 0.112 (0.112) | 0.118 (0.118) | 0.120 (0.120) | 0.131 (0.131) | 0.116 (—) |
| ring mean | 7.99 (7.99) | 8.20 (8.20) | 8.06 (8.06) | 8.23 (8.23) | 8.27 (8.27) | 8.34 (8.34) |
| 6-ring % | 7.6 (7.6) | 0.4 (0.4) | 3.5 (3.5) | **1.6** (1.6) | **2.0** (2.0) | 3.2 (3.2) |
| 7-ring % | 10.0 | 17.9 | 18.7 | 12.1 | 16.1 | 10.7 |
| 8-ring % | 59.7 (59.7) | 48.5 (48.5) | 50.6 (50.6) | 53.5 (53.5) | 43.9 (43.9) | 43.3 (43.3) |
| 9-ring % | 20.9 | 28.1 | 23.0 | 28.0 | 29.8 | 35.7 |
| S(k₀) | 0.041 (0.041) | 0.128 (0.128) | 0.120 (0.120) | 0.164 (0.164) | 0.145 (0.145) | 0.540 (0.540) |
| S_low_k2 | 0.053 (0.053) | 0.095 (0.095) | 0.117 (0.117) | 0.071 (0.071) | 0.065 (0.065) | 0.498 (0.498) |
| α(k<2) | +1.51 (+1.51) | +0.99 (+0.99) | +1.25 (+1.25) | +1.93 (+1.93) | +2.03 (+2.03) | −1.18 (−1.18) |
| dih entropy | 0.796 (0.796) | 0.875 (0.875) | 0.901 (0.901) | 0.887 (—) | 0.899 (—) | 0.866 (—) |

**PASS column: every numeric claim in the results table is CONFIRMED.** Across all
six files and all reported metrics I found **zero discrepancies >5%** vs the table.
The reference reproduces the prompt's gold-standard targets exactly (Φ12 0.985,
Φ22 0.889, rings 6:7.6/7:10.0/8:59.7/9:20.9/10:1.7, S_k0 0.041, S_low 0.053, α
+1.51, min_nb inf). No box/density error, no degree-4 artifact, no harness fault.

## Per-file 6-lowest-mode S(k) breakdown (k₀ shell, |k|=0.549; the crux of claim b)

For real positions S(k)=S(−k), so the 6 modes are 3 Cartesian ± pairs. I list the
3 pair-values per file; the **anomalous direction is bold**:

| file | x-pair (±1,0,0) | y-pair (0,±1,0) | z-pair (0,0,±1) | mean = S(k₀) |
|---|---|---|---|---|
| REFERENCE | 0.084 | 0.033 | 0.006 | 0.041 |
| A1 | 0.032 | **0.255** | 0.073 | 0.120 |
| S1_s42 | **0.414** | 0.029 | 0.049 | 0.164 |
| S1_s7 | **0.243** | 0.032 | 0.159 | 0.145 |
| baseline | 0.049 | 0.127 | 0.210 | 0.128 |
| probe_w0 | 0.674 | 0.325 | 0.621 | 0.540 |

- **Claim (b) "S1's 6 lowest modes [0.41,0.41,0.029,0.029,0.049,0.049], 4/6 at
  reference, only the x-pair anomalous" — CONFIRMED exactly** (S1_s42). A1's bad
  direction is **y** (0.255); S1_s42's is **x** (0.414); S1_s7's is **x** (0.243)
  with z also elevated (0.159). So the anomalous direction is **seed/run-dependent,
  not a fixed lab axis** — consistent with a single slow global density wave that
  freezes into whichever direction the run happened to leave it.
- Note the reference is NOT perfectly isotropic either: its x-pair (0.084) is 14×
  its z-pair (0.006). The reference's S(k₀)=0.041 is low because all three pairs are
  small; our networks have ONE pair an order of magnitude too large.

## Stress-test of the three KEY CLAIMS

**(a) 6-ring fraction robustly <5% — CONFIRMED.** A1 3.5%, S1_s42 1.6%, S1_s7 2.0%,
baseline 0.4%, probe 3.2%. Every generated network fails the ≥5% gate; the reference
alone is 7.6%. The 6-ring hard-floor claim is sound on my independent ring counts.

**(b) "Void essentially solved modulo one anisotropic mode" — PARTIALLY CONFIRMED,
but the framing OVERSTATES it. CORRECTION on the mechanism.**
- CONFIRMED: S1 α passes (+1.93 / +2.03 ≥ ref +1.51), S_low_k2 is near-gate
  (0.071 / 0.065 ≈ 0.06), and S(k₀) fails (0.164 / 0.145) driven by ONE k₀ pair.
- CORRECTION (S_low_k2) — "S_low_k2 passes WHILE S(k₀) fails" is NOT independent
  evidence the void is solved. I read the code: `S_low_k2 = low_k_structure_factor(
  pos, box, kmax=2)` averages over `_low_k_hkl(2)` = ALL integer modes with
  h²+k²+l²≤4 = the shells at |k|²=1/2/3/4 → **32 modes** (6+12+8+6), which
  **includes the 6-mode k₀ (±1,0,0) shell**. The arithmetic closes exactly:
  (6×0.164 + 26×0.049)/32 = 0.071 = the reported S_low_k2. So S_low_k2 and S(k₀)
  are the SAME density field at two averaging widths — S_low_k2 is just the bad
  k₀ pair diluted across 26 well-behaved higher modes. They are NOT two independent
  gates; passing the diluted one while failing the concentrated one is exactly what
  a single large defect looks like, not a sign the void is "solved."
- α (separate mechanism — verified, NOT the same dilution charge). I printed the
  actual k≤2 bins the α fit sees (S1_s42): the lowest 24-bin is centered at k=0.696
  and **pre-blends the k₀ shell with the √2 shell into S=0.101** — the raw 0.164 k₀
  value never enters the fit. α=+1.93 is positive because the HIGHER k<2 bins rise
  steeply (k=1.29/1.58/1.87 → S=0.346/0.341/0.363) above that low first bin;
  dropping the first bin makes α STEEPER (+3.64), proving the k₀-blended bin is the
  flattest point, not the driver. So α passes by a genuinely different route than
  S_low_k2: it reflects S(k) *rising* with k across the window (real suppression of
  the modes it resolves), with the one bad pair pre-averaged into the first bin
  before the slope is taken. α corroborates "everything except the one mode is
  clean" — it does NOT independently certify the k₀ residual is gone.
- Honest restatement: S1's void is **reduced to one slow anisotropic density wave**
  that is real and large (S(k₀) 3.5–4× ref). The original "modulo one anisotropic
  mode" wording is fair as far as it goes; the OVERREACH is citing S_low_k2 as a
  passing gate (it passes only by averaging the defect away) and implying the void
  is "essentially solved." It is **reduced but not solved**: the single k₀ pair is
  the whole remaining gap, and S(k₀) — the metric that isolates it — should govern
  the verdict, not the broader-window S_low_k2.

**(c) All outputs AMORPHOUS, no Bragg — CONFIRMED.** Binned S_v peak is 1.4–1.8 on
every generated network (ref 1.82), in the stated amorphous 1.4–1.8 band, nowhere
near a crystal's ~8.5. I also checked the **raw** (un-binned) single-mode max as a
sharper Bragg test (advisor point 4): every file's raw max is 4.8–8.7 and sits at
**k≈3.1–3.8** — this is the amorphous first-sharp-diffraction / ring-scale peak,
present in the REFERENCE itself (raw max 8.74 @ k=3.805, the LARGEST of all six).
A genuine Bragg peak would be O(N)~hundreds and at a lattice k; nothing of the kind
appears. No mode ≫ the binned peak. AMORPHOUS confirmed on all outputs.

## Verdict (trustworthiness of the results table)

The results table is **trustworthy on its numbers**: every metric I independently
recomputed from the saved rod files — degrees, N/E/density, Φ, rings, bond stats,
S(k₀), S_low_k2, α, dihedral entropy, min_nb — matches the table within <5% (in
fact to the printed precision), and the harness's S(k₀) reproduces a from-scratch
mean of the 6 lowest modes exactly. Reconstruction is clean degree-3 everywhere
(A1's 0.062 collision correctly handled at r=0.04). The two negative headlines —
6-ring hard floor (<5% across all seeds) and pure-WWW void non-emergence (α<0) —
are fully supported by my recompute. The ONE place I correct the *interpretation*
(not a number): claim (b)'s "void essentially solved" cites S_low_k2 as a passing
gate, but S_low_k2(kmax=2) is a 32-mode average that *contains* the 6-mode k₀ shell
— it passes (0.071) only by diluting the same anomalous k₀ pair S(k₀) exposes
(arithmetic closes: (6×0.164+26×0.049)/32=0.071). It is NOT an independent gate. α
is a genuinely different signal (a slope over k≤2 bins whose first bin pre-blends
the k₀ shell down to 0.101; dropping it steepens α to +3.64) — α legitimately
corroborates "everything except the one mode is clean," but does not certify the k₀
residual gone. So the void should be reported as **reduced to a single large slow
anisotropic mode, not solved**; the S(k₀) failure is the real residual, and the
seed-variable bad direction (y on A1, x on S1_s42, x on S1_s7) confirms it is one
global density wave, not noise.
