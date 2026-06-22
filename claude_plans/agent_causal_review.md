# Causal-attribution review — N≈512 random_bm2000 ablation

Reviewer: causal-attribution agent. Source: `N500_validation_results.md` + `log_r1..r11.txt`
(config JSON at top of each log). All configs cross-checked field-by-field.

## Two independent axes (kept separate per verdict)
1. **Config-confound** — does the run pair vary exactly ONE knob?
2. **Statistical power** — every run is `seed=42` (identical seed line in all logs:
   `mean=1.066 std=0.181 max=1.691`, 2-opt 330 swaps). A pair can be CONFOUND-FREE on
   config yet NEEDS-MORE-DATA on seed, especially for the high-variance S(k₀)
   (6 modes on the lowest shell; the md itself calls r9's value "noisy").

A claim is only fully clean when BOTH axes pass.

## Cross-cutting caveats (apply to several claims)
- **Single seed everywhere.** Every void / S(k₀) / S_low magnitude is one draw. Claims that
  hinge on S(k₀) ordering or absolute void value (2, 4b) inherit a power caveat regardless of
  config cleanliness.
- **The "full-N relax of a finished network = NO-OP (max move 0.003·d0)" evidence is NOT in any
  r1–r11 log.** It is asserted only in `N500_validation_results.md` (CONFOUND-FOUND box). It could
  not be reproduced from the run artifacts provided. Claims 3 and 5 lean on it to argue
  "topological, baked into the chain, not a final-relax artifact"; that attribution is
  correspondingly weaker until the diagnostic is logged.
- **`check_lsu_every` semantics (lsu_network.py:2288–2336).** Controls (a) progress logging and
  (b) the early-exit. For these RANDOM (ascending-Φ) seeds the exit fires only when Φ ≥ target
  0.889. **No run reached target** (all undershot at Φ≈0.83–0.87; last logged iter = nominal
  iters−500 in every run). So `check_lsu_every=0` vs `500` never changed the iteration count.
  Residual effect: `compute_lsu(..., rng=rng)` (line 2296) shares the SAME generator that drives
  `stone_wales_propose` (2338) and the Metropolis draw (2346). A firing check consumes rng draws,
  so check=500 vs check=0 produces a divergent move stream — a real but **second-order** drift.

---

## Claim 1 — "Temperature is the dominant lever for the RING distribution"
Runs: **r1** (T=0.5 default, no T keys) vs **r5** (T=0.045→0.015). r1 ring mean 8.46 / 6-ring 2.3%;
r5 7.996 / 5.5%.

**Config diff:** r1 sets neither `initial_temperature` nor `final_temperature` (→ generate default
T=0.5→0.001); r5 sets 0.045→0.015. One *other* field differs: `check_lsu_every` 0→500.

**Verdict: CONFOUND-FREE (modulo a second-order rng caveat).**
The check_lsu_every difference is inert on iteration count (no early-exit fired) and affects
dynamics only through shared-rng move-stream drift — second-order against a 10× temperature change.
The accidental T=0.5 in r1 legitimately supplies the "hot" endpoint; r1/r5 are otherwise matched
(same seed, iters, weight=0, geometry). The md's own CONFOUND-FOUND box correctly caught that
**batch 1 (r1, r2) ran at T=0.5 by accident** and **correctly re-baselined to cold-T r5/r6 before
drawing the temperature conclusion** — so the confound was flagged and retired, not buried.
Power caveat: single seed, but the ring effect (2.3%→5.5%, ring mean 8.46→7.996) is large and the
mechanism (hot T → near-uniform acceptance ignores energy) is sound.

## Claim 2 — "Uniformity penalty is the void lever; monotonic trade-off vs 6-rings"
Runs: **r5 (w0) / r6 (w10) / r7 (w30)**, all cold-T. S(k₀) 0.92→0.26→0.12; 6-ring 5.5%→3.0%→2.3%.

**Config diff:** ONLY `uniformity_weight` (0/10/30) varies. T, iters (15k), seed (42),
local_shell_depth (4), relax_local_iters (100), check_lsu_every (500), geometry — all identical.

**Verdict: CONFOUND-FREE on config; NEEDS-MORE-DATA on seed.**
This is the cleanest three-point sweep in the study. The directional claims (penalty lowers void,
penalty lowers 6-ring fraction) are properly isolated. BUT the headline quantities are single-seed
S(k₀) over 6 modes — exactly the quantity the md flags as noisy in r9. The *monotonicity* (a clean
0.92→0.26→0.12 and 5.5→3.0→2.3) is more robust than any single value, but whether the trade-off is
genuinely monotonic (vs. seed scatter) needs a 2nd seed. The "no single weight passes both gates"
conclusion is plausible but rests on one seed for the S(k₀)<0.15 gate.

## Claim 3 — "Full-N relaxation during the chain does NOT fix bond-std or the void"
Runs: **r8** (local_shell_depth=None, relax_local_iters=800, 10k iters) vs **r5** (depth=4,
iters=100, 15k iters).

**Config diff:** r8 differs in THREE fields, not one — `local_shell_depth` (null vs 4),
`relax_local_iters` (800 vs 100), AND `n_www_iterations` (10k vs 15k).

**Verdict: CONFOUNDED.**
Not a one-variable test. Direction analysis: r8 ran *fewer* iters (10k<15k) yet produced *lower*
bond std (0.112 vs 0.131) and *lower* 6-rings (1.6% vs 5.5%) — so the iter deficit and the relax
change push some metrics the same way and the full-N effect cannot be cleanly attributed. What
DOES survive the confound: the "bond std never reaches 0.045" observation (every variant clusters
0.10–0.15 regardless), because that is robust to the iter difference. What does NOT survive:
"full-N relax not worth it / mechanism not operative" — r8 is 2.4× slower per iter and was not
iter-matched, so the cost/benefit verdict is confounded by budget. A fair test needs full-N at
15k iters, depth=4-equivalent runtime.

## Claim 4 — "Long pure-WWW (w=0) does NOT emergently produce hyperuniformity; it coarsens"
Runs: **r11** (w0, 40k) vs **r5** (w0, 15k). 6-ring 5.5%→0.0%; void S_low 0.59→0.41, S(k₀) 0.92→0.74.

**Config diff:** ONLY `n_www_iterations` (15k→40k). Clean iters ablation.

**Verdict: SPLIT.**
- **(4a) Ring coarsening — CONFOUND-FREE.** r5→r11 isolates iterations; 6-rings decay 5.5%→0.0%
  with longer pure WWW. Clean and well-supported (also matches the seed→15k→40k transient 17%→5.5%→0%).
- **(4b) Hyperuniformity non-emergence — NEEDS-MORE-DATA.** The void is still *monotonically
  decreasing* with iters at w0 (S_low 0.59→0.41, S(k₀) 0.92→0.74) — i.e. on a downward trajectory
  consistent with SLOW emergence, not a plateau. And 40k is ~40% of the Sellers "~100k" timescale
  the emergence claim explicitly targets. You cannot refute emergence from a still-improving trend
  truncated well short of the claimed timescale. Plus single seed. "REFUTES emergence in this code"
  is overstated for the void; only the ring-coarsening half is established.

## Claim 5 — "bond-std ≈0.11 is under-equilibration, not a few long bonds; robust across relax/weight"
Evidence spans r1/r5/r6/r7/r8/r11.

**Verdict: PARTLY SUPPORTED / NEEDS-MORE-DATA on mechanism.**
Supported:
- "Robust across relax-locality and weight" — bond std 0.11–0.13 at w0/w10/w30 (r5/r6/r7) and at
  depth=4 vs None (r5 vs r8). True within the study (one seed each, but consistent).
- "Broad spread, not a few long bonds" — r7 rods span 0.49–1.23 around mean 0.81; that is a genuinely
  broad distribution, not a thin tail. Supported. (Side note: the md's "bond_max ≈1.33·d0 in EVERY
  output" is loose — observed bond_max ranges 1.19–1.71·d0 across r5/r6/r7/r8/r11 — but the broad-spread
  point does not depend on it.)

Not yet supported — **internal contradiction the study has not reconciled:** "under-equilibration"
predicts more iters → lower std, yet Batch 5 reports 15k→40k "bought ≈nothing … 0.12 is the
equilibrium at this schedule" (r6→r9: std 0.124→0.121). Within the tested range the equilibration
mechanism is contradicted by the study's own data. The competing mechanism (seed conditioner;
repulsion-settle leaves seed std 0.207) is untested. The md itself proposes the discriminant
(record post-settle seed std vs final std per seed across multiple seeds) — until that runs, the
*magnitude* claim (0.11) is robust but the *causal label* ("under-equilibration") is NEEDS-MORE-DATA.

---

## Summary verdict table
| Claim | Config | Power (seed) | Overall |
|---|---|---|---|
| 1 Temperature → rings (r1 vs r5) | CONFOUND-FREE (2nd-order rng caveat; T=0.5 confound correctly re-baselined) | single-seed, large effect | CONFOUND-FREE |
| 2 Penalty → void, trade-off (r5/r6/r7) | CONFOUND-FREE (only weight varies) | single-seed noisy S(k₀) | NEEDS-MORE-DATA (2nd seed) |
| 3 Full-N relax no fix (r8 vs r5) | CONFOUNDED (3 fields: depth+relax_iters+iters) | — | CONFOUNDED |
| 4a Long WWW coarsens rings (r11 vs r5) | CONFOUND-FREE (only iters) | single-seed | CONFOUND-FREE |
| 4b Hyperuniformity non-emergence (r11) | clean ablation but void still decreasing & 40k≪100k | single-seed | NEEDS-MORE-DATA |
| 5 bond-std = under-equilibration | robust across weight/relax; but Batch 5 (iters≈no-op) contradicts the equilibration label | single-seed | NEEDS-MORE-DATA (mechanism) |
