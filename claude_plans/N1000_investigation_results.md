# N=1000 investigation — results table (durable state)

Start: 2026-06-19 16:00 CEST. Hard stop 16h → 2026-06-20 08:00. Interpreter:
`/home/francisco/miniconda3/envs/lsu_project/bin/python` (JAX 0.10.0, CUDA RTX 4080).
Harness: `Claude_Helpers/_run_n1000.py` (N=1000, box=11.44, d0=0.8; recomputes metrics
FROM the saved file). Per-run timeout 5400s (Arm A) / 10800s (Arm B long).

## Reference (gold standard, N=1000, box=11.44, d0=0.8) — recomputed this session
| metric | value |
|---|---|
| Φ_12 / Φ_22 | 0.9849 / 0.8887 |
| bond mean / std | 0.800 / 0.0294 |
| bond ang std | 8.41° |
| rings | 6:7.6% 7:10.0% 8:59.7% 9:20.9% 10:1.7% |
| ring mean | 7.99 |
| **S(k₀)** lowest shell | **0.041** (kmin=0.549, 6 modes) |
| S_low_k2 | 0.053 |
| S_v α(k<2) | +1.51 |
| dihedral entropy | 0.796 |
| voxel_std4 | 2.25 |

## Success gates @ N=1000
- 6-ring ≥5% (ref 7.6%); ring mean 7.8–8.1; 8-ring dominant.
- S(k₀)/S_low_k2 ≤ ~0.06 (ref 0.041/0.053); α(k<2) ≥ ~1.3 (ref 1.51).
- Φ_22 ≥ 0.88 (don't over-drive to 1.0); Φ_12 ≈ 0.985.
- bond std tracked (ref 0.029, secondary); amorphous: S_v peak ≈ ref, Bragg(>3)=0.

## FAILING BASELINE — notebook cell 5 (20260619_ak1000), verified independently this session
Config: w=20, T 0.045→0.015, **100k iters**, seed 59, lsu_degree_22=0.88,
seed_jitter_sigma=0.5 (**no-op for random_bm2000 — only applies to crystal_srs**).
| metric | ref | baseline | status |
|---|---|---|---|
| 6-ring | 7.6% | **0.4%** | ✗ catastrophic |
| 7-ring | 10.0% | 17.9% | excess |
| 8-ring | 59.7% | 48.5% | low |
| 9/10/11-ring | 20.9/1.7/0 | 28.1/5.0/0.1 | coarse excess |
| ring mean | 7.99 | 8.20 | coarsened |
| S(k₀) | 0.041 | 0.128 | ✗ marginal |
| S_low_k2 | 0.053 | 0.095 | ✗ |
| α(k<2) | +1.51 | +0.99 | ✗ |
| Φ_22 | 0.889 | 0.865 | low |
| bond std | 0.029 | 0.112 | high (secondary) |
| dih entropy | 0.796 | 0.875 | too disordered |
| S_v peak/Bragg | 1.82/0 | 1.44/0 | amorphous ✓ |
Diagnosis: baseline is **too broad/disordered** (excess 7/9/10/11 rings, high dih entropy),
NOT crystalline. PRIMARY N=1000 failure = RINGS (0.4% 6-rings); void is only marginal.

## Seed health — random_bm2000, N=1000, seed=42 (regenerated this session)
Perfect 3-regular; girth **5**; rings 5:21.5% 6:16.8% 7:17.1% 8:14.9% 9:15.5% 10:9.1% …;
ring mean 7.36; bond mean 1.099 std 0.236 max 2.19·d0 (raw, pre-settle); **S(k₀)=0.141**
(already near the <0.15 gate). HEALTHY — rich 6-ring tail (16.8%) to retain ~7.6%.

## Runtime calibration (2k-iter pilot, w=30, cold T, seed 42)
130s / 2k iters = 65 ms/iter (inflated by JIT compile + cold-seed relax; amortizes).
At 2k iters: 5-rings already 21.5%→3.3% (die fast), 6-rings 16.8%→18.9% (preserved),
Φ_22 0.75 (climbing), S(k₀) 0.38 (void grows early, penalty pulls back over more iters).
**→ window hypothesis:** an intermediate iteration count kills 5-rings (girth→6) + raises Φ
while retaining 6-rings; the notebook's 100k (100 moves/vertex vs N=512 sweet-spot ~29) overshot.

## ===== ROOT CAUSE FOUND (2026-06-22): an f1/f2 ENERGY-FORM BUG supersedes everything below =====
**The whole investigation's wall — "6-ring floor", the temperature tradeoff, the void, the basin
coarsening — traces to ONE bug: `energy_components` uses SIMPLIFIED f1/f2 (harmonic `(L−d0)²` +
normalized `(cosθ+½)²`) instead of the literal length-coupled Keating forms. This makes the energy
~6–8× too ANGLE-dominated (bonds too soft) → its minimum spreads bonds + loses hyperuniformity.**
Demonstrated 3 ways (reweight f1, stiffen f1, swap to Keating — all pull the min onto the reference);
f3/f4 are FAITHFUL (Eq.3/4) and become MORE influential under Keating. Details + the env-gated test
flag (`LSU_KEATING_F1F2`, default OFF) in memory `lsu-energy-keating-balance-fix`. Evidence:
- **Holds the reference:** cold anneal-from-reference under Keating keeps 6r 7.6→7.6, S_k0 0.041→0.048,
  α 1.5→1.18, 8r 59.7→57, Φ 0.889→0.890 — where the OLD energy DESTROYED it (6r→1.2, S_k0→0.12, α→0.26).
  And it's genuinely ACTIVE not just cold-inertia: from the random seed Keating FAVORS 6-rings (13%)
  and SUPPRESSES 9-ring coarsening (9r=18 vs old ~30).
- **DOES NOT (yet) reach the reference from a random seed** (50k, 0.1→0.015): 6r=13 (overshoot), 8r=38
  (undersharp), S_k0=0.47/α=−0.19 (void unfixed — warm start opened it; metrics still moving at 50k).
  So reachability is UNPROVEN IN BOTH DIRECTIONS (50k<Sellers' 100k; warm start confounds the void).
**CONSEQUENCE:** essentially EVERY conclusion below (6r floor falsification, temperature sweep,
tradeoff, basin/coarsening, seed audit "void=seed") was measured under the BROKEN energy and needs
RE-VALIDATION under Keating. Everything below this line is PRE-energy-fix and largely superseded.

## ===== FINAL OUTCOME (2026-06-22) =====
**1. ROOT CAUSE (durable deliverable):** the f1/f2 energy-form bug above — found, fixed (literal
Keating forms), adopted as production default. The answer to "why couldn't we reproduce the reference."
**2. S(k)/HYPERUNIFORMITY — REPRODUCED.** Recipe = Keating energy + near-hyperuniform seed PLACEMENT
(`Claude_Helpers/_hyperuniform_seed.py`, collective-coordinate low-k suppression) + uniformity penalty
to hold it through annealing. Saved structure `Example/20260622_lsu_hyperuniform_N1000_ends.txt`
(+README): S_low=0.049 (ref 0.053 ✓), α=+2.6, amorphous (no Bragg). User accepted this as the S(k) win.
**3. ANGLE/LOCAL-GEOMETRY — OPEN (reachability limit).** angle-std 11.6° (ref 8.41°), 8r 39% (ref 60%)
plateau short from a disordered seed (same with w=0 — NOT a penalty artifact). The reference's local
order is HELD by Keating but not REACHED from random. Needs a gyroid-like seed or multi-stage protocol,
NOT more weight/schedule sweeps. See memory `lsu-energy-keating-balance-fix` for the full chain.

## ===== CORRECTION (2026-06-21): the "6-RING HARD FLOOR" is FALSIFIED =====
**The advisor flagged that temperature — the dominant ring lever at N=512 — was never swept at
N=1000 (all runs sat cold, T≤0.045). An intermediate-T w=0 probe (T 0.2→0.02 slow cool, seed42,
`Structures/20260621_probew0_Thi_*`, log `claude_plans/log_probe_Thi.txt`) OVERTURNS the floor:**
- At HOT T the penalty-free 6r is FAR above the gate: **T=0.136 → 6r=8.3%** (>ref 7.6%);
  T=0.093 → 6r=6.3%. The "~3% floor" was a COLD-SCHEDULE ARTIFACT (the system froze before 6r
  could equilibrate). 6r is strongly T-dependent, not a thermodynamic floor.
- **But a clean 6r↔Φ tradeoff vs T replaces it:** hot T gives 6r but LOW Φ_22 (0.785, disordered);
  cold T gives higher Φ (0.855) but kills 6r (→0.8%). Reference needs BOTH high (6r 7.6% @ Φ 0.889).
  Whether a SUSTAINED HOLD at an intermediate T (~0.08–0.10) can equilibrate 8r/Φ UP while keeping
  6r≥5% is the NEW open question (the continuous cool passed through that band too fast to tell).
- **STILL VALID from below:** α stays NEGATIVE at EVERY temperature (−0.65 hot → −1.37 cold), so the
  void/anti-hyperuniformity is a SEPARATE problem temperature does not fix — it needs the penalty.
  The Arm B "pure-WWW does not emerge hyperuniform" result stands (now confirmed across the full T range).
- **NET:** the headline negative is DOWNGRADED from "hard floor, settled" to "open — temperature
  breaks the 6r floor; the live question is whether 6r and Φ can co-exist via a sustained hold +
  penalty for the void." Pursuing this now (user-directed temperature sweep + N=1000/N=4000).

## ===== RESOLVED (2026-06-21): the wall is the BASIN (8r ceiling + 6r↔8r anti-correlation) =====
**The sustained-hold test answers the open question — NO, the reference corner is not reachable.**
Run `sustain085_w30_N1000` (sustained cool→hold@T=0.085→cool, w=30 kmax=2, seed42, 70k, checkpointed;
`Structures/20260621_sustain085_w30_N1000*`, log `claude_plans/log_sustain085.txt`):
- **6r ↔ 8r anti-correlation is a TREND across the T-sweep, NOT a proven thermodynamic law.**
  Across schedules, hot states pair high-6r/low-8r and cold states low-6r/high-8r. CAVEAT [advisor]:
  the sharp 30k→40k drop (6r 4.4→0.4 while 8r 40.9→48.8) is NOT proof — BOTH checkpoints are at
  CONSTANT T=0.085 (the hold spans iters ~10.5k–49k; the final cool starts ~49k). At fixed T, 6r
  craters WHILE S_k0 blows up (0.075→0.199) AND α drops (1.18→0.69) all at once = a single stochastic
  VOID-FORMING event (local-shell relax blind to box-scale voids), not a temperature-driven law. The
  reference corner (6r 7.6% AND 8r 59.7%) is still not realized in any single state we produced — that
  stands on the whole dataset, not this one transition.
- **The basin COARSENS toward 9-rings — sharper than just an "8r ceiling."** Even the best-balanced
  checkpoint (30k: void 0.075✓, 6r 4.4) sat at 8r=40.9 / 9r=32.1, ring mean ~8.3–8.4 vs reference
  **7.99**. Too many 9-rings, never a sharp 8-peak: 8r has **never reached ≥55%** across EVERY
  run/T/weight (all-time max = S1's 53.5%; ref 59.7%). The fingerprint is coarsening (excess 9s),
  which points cleanly at the seed/basin, not the schedule.
- **The void reopens on the final cool EVEN WITH w=30:** S_k0 0.075 (30k, near gate) → 0.358 (70k);
  α 1.48 → 0.61. The penalty cannot hold the void through cooling at this weight/schedule.
- **Final ring vector FAILS every axis:** 6r 2.4 (7.6), 7r 11.9 (10.0), 8r 45.1 (59.7), 9r 33.7
  (20.9), Φ 0.866 (0.889), S_k0 0.358 (0.041), α 0.61 (1.51). NOT saved — sub-reference on all gates.
- **VERDICT:** no schedule/weight/temperature in the explored space reproduces the reference from the
  random_bm2000 seed. The blocker is the SEED/BASIN: (a) an 8r-sharpness ceiling, (b) a 6r↔8r
  anti-correlation that forbids the reference corner, (c) a void that reopens on cooling. The
  mandated seed (the investigation's all-along dominant caveat) is now the PRIME SUSPECT and the
  next real lever — but it is HARD-CONSTRAINED by the original prompt, so changing it is a user call.
  N=4000 reframes as a finite-size test (does a bigger box raise the 8r ceiling / improve α?), NOT a
  rescue — unlikely to break a basin/thermodynamic wall, but it isolates finite-size from fundamental.

## ===== SUPERSEDED SETTLED CONCLUSION (cold-runs only; kept for the record) =====
**The single decisive blocker is a 6-RING HARD FLOOR, not a tunable tradeoff.**
- The penalty-FREE (w=0) equilibrium 6-ring fraction is **~3%** (probe seed42: 1.2–3.6% across
  10k–100k, a PLATEAU, NEVER ≥5%). By the FIRST w=0 checkpoint (10k) 6r is ALREADY 3.6% (<5%) and
  stays there — the seed's 16.8% decays below the gate within the first ~10k iters. (The precise
  sub-10k crossing is UNOBSERVED at w=0; the 18.9%@2k anchor is the w=30 PILOT, a different weight —
  do not attribute that timing to w=0. [causal review correction]). The 6r floor is **WEIGHT-ROBUST,
  NOT a penalty artifact**: w=0 ~3%, A1 w=30 3.5%, S1 w=35 1.6%; at MATCHED 30k w=30≈w=0 within
  noise → the penalty does NOT monotonically lower 6r (the N=512 "penalty kills 6r" does not cleanly
  replicate at N=1000). **6r ≥5% is unreachable at N=1000 across ALL tested weights including zero.**
  Reference 7.6% ≈ 2.5× the best-case equilibrium → NOT an equilibrium of (random_bm2000 + Sellers
  energy + any tested schedule/weight). Φ↔6r SIGN FLIP corroborates a different basin: our runs pair
  higher Φ with LOWER 6r (A1 0.850/3.5%, S1 0.857/1.6%); reference has BOTH high (0.889/7.6%).
- **A1 and S1 are two PARETO points of ONE distribution, neither passes 6r:** A1 = max-6r endpoint
  (6r 3.5%, 8r 50.6%, S(k0) 0.120); S1 = max-sharpness/min-void endpoint (6r 1.6%, 8r 53.5%,
  S_low 0.071, α 1.93). S1 is NOT a "winner" — its 6r is worse than A1's.
- **Void is REDUCED but NOT solved; the 6r floor (not the void) is the fundamental blocker.** The
  penalty shrinks the broad low-k excess to ONE large anisotropic residual mode (S1 6 lowest =
  [0.41,0.41, 0.029,0.029, 0.049,0.049]: 4/6 at reference, one pair anomalous; A1's bad pair is y,
  S1's is x — seed-variable ⇒ one slow global density wave). S(k0)=0.145–0.164 (3–4× ref) GENUINELY
  FAILS the void gate. CAVEAT [stats review]: S_low_k2(kmax=2) and α are NOT independent of S(k0) —
  they contain/average the SAME k0 shell, so they "pass" only by diluting the one bad mode among ~18
  good modes; **S(k0) isolates the residual and should govern the void verdict.** Net: the void is a
  genuine (single-mode) failure, but the **6r HARD FLOOR (fails even at w=0, weight-robust)** is the
  decisive blocker.
- **Arm B (pure-WWW emergence): REFUTED at N=1000, full 100k** — α<0 (anti-hyperuniform) throughout,
  8r stuck ~43% (coarsens to 9r), Φ plateaus 0.863. Settles the N=512 40k-caveat decisively.
- **CONFOUND (flag for causal review):** S1-vs-A1 changed schedule + iters (30k→50k) + weight (30→35)
  simultaneously → "sustained-T sharpens" is SUGGESTIVE ONLY, not an isolated lever. The negative
  headline does NOT depend on it — it rests on the clean w=0 probe.
- **Honest fidelity framing:** move + energy + Metropolis + Vink-threshold + local-shell relax are
  FAITHFUL (audit), yet pure WWW voids where Sellers claims emergence. Result = **unreachable within
  the MANDATED random_bm2000 seed + the tested schedules**; UNFALSIFIED escape hatches = Sellers'
  Poisson seed (different basin), Sellers' exact (unpublished) schedule, and system size N≫1000
  (their 100k may be few moves/vertex). NOT a claim the protocol is universally incapable.

## KEY DIAGNOSIS (updated after A1 + probe ck=10k)
The reference is a SHARP distribution: **7-ring 10.0%, 8-ring 59.7%**. Our networks have **EXCESS
7-rings** (A1 18.7%, pure-WWW 23.6%) and **deficient 8-rings** (50.6% / 42%). 7-rings are odd-ring
defects the reference annealed out (Φ 0.889) but ours retain (Φ stuck 0.85). → the gap is
UNDER-CONVERGENCE of ring topology (too many 7-ring defects), not just the void.
- The **penalty SHARPENS toward 8-rings** (A1 w=30 8r=50.6% > pure-WWW 8r=42%) — confirms advisor:
  uniformization narrows the distribution (supplies 8-ring dominance) but costs the 6-ring tail.
- The **void NEEDS the penalty** at N=1000 (pure WWW S_low=0.722 catastrophic; w=30→0.117).
- Lever-3 direction: a schedule that climbs Φ + eliminates 7-rings (sustained moderate-T to cross
  the 7→8 barrier) WHILE the penalty holds the void.

## Runs
| # | tag | key params | iters | Φ12 | Φ22 | bond std | ring mean | 5r% | 6r% | 8r% | S(k₀) | S_low_k2 | α | elapsed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | REFERENCE | — | — | 0.985 | 0.889 | 0.029 | 7.99 | 0 | 7.6 | 59.7 | 0.041 | 0.053 | +1.51 | — | gold |
| base | ak1000 notebook | w=20,100k,seed59 | 100k | 0.985 | 0.865 | 0.112 | 8.20 | 0 | 0.4 | 48.5 | 0.128 | 0.095 | +0.99 | ~1h | FAIL rings+void |
| pilot | pilot2k | w=30,2k,seed42 | 2k | 0.935 | 0.753 | 0.195 | 7.62 | 3.3 | 18.9 | 26.1 | 0.377 | 0.345 | +1.01 | 130s | (calibration only) |
| A1 | A1_w30_30k_s42 | w=30,kmax2,coldT,seed42 | 30k | 0.982 | 0.850 | 0.118 | 8.06 | 0.0 | 3.5 | 50.6 | 0.120 | 0.117 | +1.25 | 1171s | girth6✓ rings 9× notebook but 6r<5%✗; void✗(worse@big box); collision min_nb=0.062 |

### Run 2 — PURE-WWW reachability probe (w=0, cold geometric 0.045→0.015 over 100k, checkpoints/10k, seed42)
DECISIVE NEGATIVE on ring-shape reachability. Trajectory (pure Sellers, NO penalty):
| iter | Φ22 | 6r | 7r | 8r | 9r | ringmean | S_k0 | S_low | α | acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 10k | 0.819 | 3.6 | 23.6 | 42.0 | 22.9 | 8.089 | 0.445 | 0.722 | −0.42 | 12.7% |
| 20k | 0.840 | 3.2 | 13.8 | 40.3 | 34.5 | 8.327 | 0.756 | 1.022 | −1.30 | 4.6% |
| 30k | 0.850 | 1.2 | 16.1 | 42.8 | 32.7 | 8.298 | 1.162 | 0.843 | −1.55 | 3.0% |
| 50k | 0.859 | 1.9 | 10.5 | 43.5 | 39.3 | 8.353 | 0.584 | 0.670 | −1.32 | 1.6% |
| 70k | 0.863 | 3.2 | 9.0 | 42.9 | 39.1 | 8.358 | 0.597 | 0.544 | −1.25 | 0.6% |
| **100k** | **0.863** | **3.2** | **10.7** | **43.3** | **35.7** | 8.336 | 0.540 | **0.498** | **−1.18** | 0.4% |
- **8-rings PLATEAU at ~43%** the entire run (never approach ref 59.7%); 7-ring defects anneal out
  (23.6→10.7=ref) but convert to **9-rings (22.9→35.7, COARSENING)**, not 8-rings. Ring mean 8.09→8.34.
- **Φ_22 plateaus 0.863** (not 0.889). **Void stays anti-hyperuniform** the whole run: α ≈ −1.2 at 100k
  (ref +1.51), S_low decreasing slowly (1.02→0.498) but S_k0=0.54 (ref 0.041), nowhere near emerging.
- **→ Sellers' "emergent near-hyperuniformity from ~100k pure WWW" is REFUTED in this code at N=1000
  AT THE FULL SELLERS BUDGET** — the void never emerges (α<0 throughout), the rings coarsen to 9r.
  Settles the N=512 "40k caveat": even at 100k it does not emerge. DECISIVE NEGATIVE (Arm B).
- **Mechanism:** local-shell relax can't feel a box-scale void (locally strain-free), so void-creating
  SW moves are accepted → density coarsens into 9-rings + void. The PENALTY is the only thing that
  both suppresses the void AND sharpens toward 8-rings (rejects 9-ring coarsening) — at the cost of 6r.
- **Verdict:** the reference's sharp-8r-WITH-6r-tail is NOT on the pure-WWW trajectory and is not an
  equilibrium of (Sellers energy + symmetric low-k penalty) at any weight. Path to best config = the
  PENALTY, tuned for 8r-sharpness + void while retaining max 6r.

### Low-k S(k) shell profiles (k0=0.549) — kmax lever ruled out
| network | S(0.55) | S(0.78) | S(0.95) | S(1.10) | S(1.23) |
|---|---|---|---|---|---|
| REFERENCE | 0.041 | 0.045 | 0.069 | 0.060 | 0.054 |
| A1 (w30 kmax2) | 0.120 | 0.099 | 0.166 | 0.087 | 0.438 |
| probe w0 100k | 0.540 | 0.317 | 0.763 | 0.464 | 0.153 |
| notebook (w20 100k) | 0.128 | 0.097 | 0.086 | 0.066 | 0.284 |
- Reference is **uniformly low across ALL low-k shells** (0.04–0.07). Our void is a **broad low-k
  excess** (every shell 2–4× ref), NOT a single-k0 feature. → **kmax=1 (k0-only) is the WRONG lever**
  (leaves the higher-shell excess unpenalized). The void needs **more penalty strength** (higher w
  and/or kmax≥2), which costs MORE 6-rings → the tradeoff TIGHTENS, not loosens.
- Iteration dependence of void: notebook (w20,100k) S_low 0.095 < A1 (w30,30k) 0.117 → void wants
  MORE iters; 6r wants FEWER. Fully opposed in (w, iters). → revised S2 = higher-weight frontier run
  (not kmax=1) to bracket the best-achievable void vs the 6r cost.

| S1 | S1_sustain_w35_50k_s42 | sustained T0.045→hold0.025→0.012, w=35,kmax2,seed42 | 50k | 0.983 | 0.857 | 0.120 | 8.23 | 0.0 | **1.6** | 53.5 | 0.164 | **0.071** | **+1.93** | 1721s | sharpened (8r↑ 7r↓ α↑ S_low↓ near-gate, NO collision) but 6r↓1.6% & S(k0)=0.164 residual; void now k0-concentrated |
| S1b | S1_sustain_w35_50k_s7 | same as S1, **seed=7** (multi-seed confirm) | 50k | 0.980 | 0.848 | 0.131 | 8.27 | 0.0 | **2.0** | 43.9 | 0.145 | **0.065** | **+2.03** | 1781s | CONFIRMS: 6r 2.0%<5 (robust fail); S_low 0.065/α2.03 (void mostly-solved robust); S(k0)0.145 1-mode; NO collision; 8r seed-variable 43.9 vs 53.5 |

### Batch S1b — multi-seed confirmation (S1 on seeds 42 & 7) of the NEGATIVE result
- **6-ring gate FAILS robustly:** 1.6% (s42) / 2.0% (s7), both ≪5% gate. The hard floor is seed-robust.
- **Void robustly "mostly-solved-modulo-one-mode":** S_low_k2 0.071/0.065 (≈gate), α +1.93/+2.03
  (pass), S(k0) 0.164/0.145 (fails on a single anisotropic k0 pair both times).
- **Sustained-T robustly avoids the collision** (min_nb 0.30/0.24, vs A1 cold-geom's 0.062).
- **8r-sharpness is seed-variable** (53.5 vs 43.9%), NEVER reaches ref 59.7%; Φ_22 0.848–0.857.
- → The negative result (no config passes all gates; 6r hard floor) is **multi-seed confirmed**.

### Batch S1 — schedule-shape SHARPENS but deepens the 6-ring loss; residual void is k0-only
- **Sustained-moderate-T works in the predicted direction**: vs A1, 8r 50.6→53.5, 7r 18.7→12.1
  (toward ref 10), Φ_22 0.850→0.857, **S_low_k2 0.117→0.071** (near the ≤0.06 gate), **α +1.25→+1.93**
  (exceeds gate, near ref). NO collision (min_nb 0.30). The schedule lever is REAL.
- **BUT 6-rings dropped 3.5→1.6%** — sharpening narrows the distribution, killing the 6r tail FURTHER.
  Same tradeoff, pushed along the sharpening axis. 6r and 8r-sharpness are anti-correlated here.
- **Residual void is now k0-CONCENTRATED**: S(0.55)=0.164 but S(0.78..1.10)=0.070/0.028/0.037 (≈ref!).
  The kmax=2 penalty crushed the easy local modes but left the hard GLOBAL k0 mode high (S_low_k2
  passes-ish but S(k0) fails). → reopens a k0-targeted (kmax=1) follow-up to crush the void floor.
- **Tradeoff MAP (6r vs 8r-sharpness):** pure-WWW (6r~3, 8r43) → A1 w30 (6r3.5, 8r50.6) → S1 sustain
  w35 (6r1.6, 8r53.5). As 8r sharpens, 6r dies. The reference (6r7.6 AND 8r59.7) is OFF this curve.

### Batch A1 — void HARDER at N=1000; tradeoff replicates  [over-run claim REFUTED, see correction]
- **~~Over-run hypothesis CONFIRMED~~ → REFUTED by the causal review (agent_causal_review_N1000).**
  My original A1-vs-notebook claim (30k 6r 3.5% vs 100k 6r 0.4% = "9× over-run") is CONFOUNDED
  (iters AND weight 20→30 AND seed 59→42 AND target differ). The CLEAN iters-only evidence (w=0 probe,
  seed42) shows 6r PLATEAUS ~3% from 10k→100k — it does NOT keep decaying with iterations. At MATCHED
  100k, probe w=0 gives 6r 3.2% vs notebook 0.4% → the notebook catastrophe is a WEIGHT(20)/SEED(59)
  artifact, NOT over-run. Corrected conclusion: iteration count is NOT the ring lever I claimed; the
  6r floor (~3%) is reached by ~10k iters and is iteration- and weight-robust thereafter.
- **A1 replicates the N=512 nominee at N=1000** (6r 3.5% vs 3.3%; Φ_22 0.850 vs 0.85; ring mean
  8.06 vs 8.09) → the penalty↔6-ring tradeoff is N-independent, holds at N=1000.
- **Void is WORSE at the bigger N=1000 box:** S(k₀)=0.120, S_low=0.117 (vs N=512 w=30's 0.063 PASS).
  kmin=0.549 (vs 512's 0.687) → the box-scale void is at longer wavelength; w=30 under-suppresses it.
  → at N=1000 the void gate FAILS at w=30. Iteration tension: rings want FEW iters (retain 6r),
  void wants MANY (notebook 100k got S_low 0.095 < A1 30k's 0.117). Opposed in the iter dimension.
- **Distribution stays BROAD + Φ stuck 0.850** (dih entropy 0.901 > notebook 0.875 > ref 0.796).
  Under-converged toward the sharp reference (ref: tight 8-ring 59.7%, dih 0.796). → advisor's
  "broad + Φ stuck" signal → next lever = schedule SHAPE (sustained moderate-T) and/or kmax-narrow.
- **Collision:** min_nb=0.062 (0.077·d0) — a near-coincident non-bonded pair (the degree-4 round-trip
  signal). Metrics recomputed with cluster_radius=0.04. Notebook had none; watch if penalty/fast-cool
  specific. (Harness now collision-robust: `full_metrics_safe`.)

---

## SUMMARY (deliverable) — 2026-06-19, ~2.8h of 16h budget, 5 configs (of 12 max)

**STATUS: DECISIVE NEGATIVE, multi-seed confirmed.** No config in the explored space
(mandated random_bm2000 seed + faithful Sellers energy + low-k penalty + cold/sustained schedules)
passes all gates at N=1000. The blocker is a **6-ring HARD FLOOR**, not a tunable tradeoff. This
CONFIRMS and STRENGTHENS the N=512 finding at full reference scale + 100k (the prompt's "excuses
off the table" budget). The prior "needs more iterations / finite size" get-outs are closed.

### The single decisive finding: 6-ring ≥5% is unreachable across ALL tested weights (incl. zero)
- Penalty-free (w=0) equilibrium 6-ring fraction ≈ 3% (probe seed42: 1.2–3.6% across 10k–100k, a
  PLATEAU, never ≥5%). 6r is ALREADY 3.6% (<5%) by the first w=0 checkpoint (10k) and stays there —
  the seed's 16.8% decays below the gate within the first ~10k iters, while girth-6 ✓ but Φ 0.819,
  S_low 0.722, α −0.42 are all still bad. (Sub-10k crossing UNOBSERVED at w=0; the 18.9%@2k is the
  w=30 pilot — different weight. [causal correction])
- The 6r floor is **WEIGHT-ROBUST, not a penalty artifact**: w=0 ~3%, w=30 (A1) 3.5%, w=35 (S1) 1.6%;
  at matched 30k w=30≈w=0 within noise. The penalty does NOT monotonically lower 6r at N=1000 (unlike
  N=512). → 6r≥5% unreachable across all tested weights.
- Reference 7.6% ≈ 2.5× the best-case equilibrium ⇒ NOT an equilibrium of this (seed+energy+schedule).
- Φ↔6r SIGN FLIP confirms a different basin: our runs pair higher Φ with LOWER 6r
  (A1 0.850/3.5%, S1 0.857/1.6%); reference has BOTH high (0.889/7.6%).

### Best-achievable configs (two Pareto endpoints; NEITHER is a winner — both fail 6r)
| config | what | 6r | 8r | ring mean | Φ22 | S(k0) | S_low | α | min_nb | file |
|---|---|---|---|---|---|---|---|---|---|---|
| REFERENCE | gold | 7.6 | 59.7 | 7.99 | 0.889 | 0.041 | 0.053 | +1.51 | inf | Example/lsu_example_ends.txt |
| **A1** (max-6r) | w=30 kmax2 cold-geom 30k | **3.5** | 50.6 | 8.06 | 0.850 | 0.120 | 0.117 | +1.25 | 0.062✗ | Structures/20260619_A1_w30_30k_s42.txt |
| **S1** (max-sharp/min-void) | w=35 kmax2 sustained-T 50k | 1.6 | **53.5** | 8.23 | 0.857 | 0.164 | **0.071** | **+1.93** | 0.30✓ | Structures/20260619_S1_sustain_w35_50k_s42.txt |
| S1 seed7 | (confirm) | 2.0 | 43.9 | 8.27 | 0.848 | 0.145 | 0.065 | +2.03 | 0.24✓ | Structures/20260619_S1_sustain_w35_50k_s7.txt |
Gates passed: girth-6 ✓, ring-mean ✓, 8r-dominant ✓, amorphous (S_v peak 1.4–1.5, 0 Bragg) ✓,
Φ_12 ✓, α ✓ (S1). Gates FAILED: 6r≥5% ✗ (all), S(k0)≤0.06 ✗ (one anisotropic mode), Φ_22≥0.88 ✗
(plateau 0.85), bond-std ✗ (secondary, seed-conditioner floor ~0.12).

### The void is REDUCED to one anisotropic mode but NOT solved (the 6r floor is the real blocker)
The penalty crushes the broad low-k excess to ONE large anisotropic residual: S1's 6 lowest |k|
modes = [0.41,0.41, 0.029,0.029, 0.049,0.049] — 4/6 at reference, one pair anomalous (A1 bad-pair y,
S1 bad-pair x, seed-variable ⇒ one slow global density wave). **S(k0)=0.145–0.164 (3–4× ref) GENUINELY
FAILS the void gate.** CAVEAT [stats review]: S_low_k2(kmax=2)=0.065–0.071 and α +1.9–2.0 "pass" only
because they AVERAGE/contain the same k0 shell — diluting the one bad mode among ~18 good ones; they
are NOT independent of S(k0). S(k0) isolates the residual and should govern. So the void is reduced,
not solved; but it is the **6r hard floor** (fails even at w=0) that decisively blocks all configs.
(kmax=1 is still the WRONG lever for the broad excess en route; the residual is one mode at the end.)

### Which levers mattered (before → after)
1. **Iteration count: NOT the ring lever I first claimed [causal correction].** The w=0 probe shows
   6r PLATEAUS ~3% from 10k→100k (does not keep decaying). At MATCHED 100k, probe w=0 6r=3.2% vs
   notebook 6r=0.4% → the notebook's ring catastrophe is a WEIGHT(20)/SEED(59) artifact, NOT over-run.
   What iters DO control: the first ~10k decays 6r from the seed's 16.8% to the ~3% floor (the only
   real iteration effect; sub-10k crossing of the 5% gate unobserved at w=0).
2. **Uniformity penalty (the only void+sharpening tool):** pure-WWW 8r 43% / S_low 0.72 → A1 w=30
   8r 50.6% / S_low 0.117. Sharpens toward 8-rings (rejects 9-ring coarsening) AND fixes the void.
   It does NOT monotonically kill 6r at N=1000 (w=30 6r 3.5% ≈ w=0 ~3% at matched iters) — the 6r
   floor is weight-robust. Void is HARDER at the bigger N=1000 box (kmin 0.549 vs 512's 0.687) →
   w=30 leaves S_low 0.117 (vs 512's 0.063 pass); needs stronger/sustained penalty (S1: S_low 0.071).
3. **Sustained-moderate-T schedule (SUGGESTIVE, confounded):** A1→S1 (8r 50.6→53.5, 7r 18.7→12.1,
   α 1.25→1.93, S_low 0.117→0.071, NO collision) — but 6r 3.5→1.6. Changed schedule+iters+weight
   together, so "sustained-T sharpens" is suggestive, not isolated. Notably it ROBUSTLY avoids the
   vertex collision that cold-geometric+penalty produces.

### Arm A vs Arm B verdict
- **Arm B (pure-WWW emergence): REFUTED at full 100k.** Did near-hyperuniformity AND the ring
  distribution emerge on their own? NO to BOTH. α stays NEGATIVE the whole run (−0.42→−1.18;
  anti-hyperuniform), 8-rings stuck ~43% (7-ring defects convert to 9-rings = COARSENING, not 8r),
  Φ_22 plateaus 0.863. The decisive negative the prompt anticipated. Mechanism: local-shell relax
  is BLIND to box-scale voids (locally strain-free) → void-creating SW moves accepted → coarsening.
  **SEED-ROBUST THROUGH 20k (Batch B2, addresses causal-review scope gap):** the 2nd-seed (seed 7)
  w=0 probe lands in the SAME basin at 10k/20k — 6r 3.5%→3.1% (below the 5% gate, matching seed-42's
  ~3% floor), α NEGATIVE (−1.30, −1.33; anti-hyperuniform like seed-42), 8r plateau ~43% coarsening
  to 9r (23.9→32.4%). The SIGN of α and the 6r floor replicate on a 2nd seed; the trajectory locks
  by ~20k on both. CAVEAT: seed-7 stopped cleanly at 20k (no crash, budget-limited) — only seed-42
  carried to the full 100k plateau, so the *100k* verdict itself is single-seed; the *20k* basin is
  2-seed.
- **Arm A (break the 6-ring floor): NO lever breaks it.** Late-penalty-staging is futile (6r doesn't
  regrow; advisor). kmax=1 is wrong (void is broad). Sustained-T sharpens 8r but does not raise 6r.
  The 6r floor (~3%) is below the gate even at w=0 AND weight-robust, so no penalty/schedule passes it.

### WWW Sellers-fidelity audit conclusion (agent-confirmed vs the PDFs)
SW move, Metropolis acceptance, the threshold-energy early-reject identity (BM2000 Eq.3 — the
literal E_t=E_b−T·ln(s), "exactly equivalent to Metropolis"), the four-term energy U (Supp Eq.2),
and the depth-4 local-shell relax (Vink/MB) are all **FAITHFUL** (independently PDF-verified). The
only deviations are documented EXTENSIONS: the uniformity_weight low-k penalty (Metropolis objective
only) and the settle_seed_with_repulsion conditioner. **The decisive negative is EXTENSION-FREE** —
it rests on the w=0 (strictly Sellers Eq.2) probe. DOMINANT unfalsified caveat = the MANDATED
random_bm2000 seed (girth≥5 Hamiltonian-scaffold) vs Sellers' unspecified "random seed pattern";
a different seed ring-spectrum could alone explain non-emergence. SCOPE the negative as **"the
reference is unreachable within the mandated random_bm2000 seed + the tested schedules,"** NOT "the
protocol is universally incapable." Escape hatches: seed kind, Sellers' exact (unpublished) anneal
schedule, system size N≫1000.

### What I'd try next (not run — outside the mandated-seed scope)
1. **Vary the seed** (the dominant caveat): a Poisson/configuration-model random seed (Sellers'
   literal start) — its different ring-spectrum may sit in the reference's basin. The mandate fixed
   random_bm2000; relaxing it is the highest-value next experiment.
2. **A void-aware acceptance that is NOT symmetric uniformity** (penalize only the global k0
   anisotropic mode that S1 leaves) to close S(k0) without further ring narrowing — but this won't
   fix the 6r floor.
3. The 6r floor itself appears intrinsic to (this seed + Sellers energy); reaching ref 7.6% likely
   needs a different seed or an energy/MC ingredient that stabilizes 6-rings against decay.

### Files
- Runs: `Structures/20260619_{A1_w30_30k_s42, S1_sustain_w35_50k_s42, S1_sustain_w35_50k_s7,
  probew0_s42_ck*}.txt` (+ `.kwargs.json`). Logs: `claude_plans/log_{A1,probe_s42,S1,S1_s7}.txt`.
- Harness: `Claude_Helpers/{_run_n1000.py, _run_sched.py, _probe_w0_checkpoints.py, _metrics.py
  (full_metrics_safe, collision-robust)}`.
- Reviews: `claude_plans/archive/agent_{stats,causal,fidelity}_review_N1000.md`.
