# Causal-attribution review — N=1000 investigation

Independent reviewer. Scope: are the causal claims in `N1000_investigation_results.md`
("SETTLED CONCLUSION" block) isolated one-variable-at-a-time and seed-robust?
Evidence: run configs `Structures/20260619_*.kwargs.json`, harnesses
`Claude_Helpers/_probe_w0_checkpoints.py` + `_run_sched.py`, probe trajectory
`claude_plans/log_probe_s42.txt`, results table. No re-runs needed (effect sizes large,
trajectory JSON intact).

Key harness facts established:
- **The w=0 probe is a SINGLE continuous anneal** (`_probe_w0_checkpoints.py`): one shared
  geometric T-schedule over the full 100k (`T_full`, lines 55–58), `pos`/`edges` carried
  across chunks (lines 84–96). So its checkpoints isolate ONLY iteration count, with
  seed=42, w=0, schedule all held fixed. Genuinely extension-free / clean.
- **The probe's first checkpoint is 10k.** There is NO w=0 datapoint below 10k. The only
  sub-10k 6-ring number in the project (2k = 18.9%) is the `pilot2k` run, which is **w=30**,
  not w=0 (`20260619_pilot2k.kwargs.json`: `uniformity_weight: 30.0`).
- **Notebook (w=20, 100k) and probe (w=0, 100k) share the SAME schedule + iter count**
  (both geometric 0.045→0.015 over 100k). They differ in weight (20 vs 0), seed (59 vs 42),
  target (0.88 vs 0.889). This is the closest thing to a matched-iters pair in the set.

---

## Check 1 — "Over-run is a ring cause" (notebook 100k 6r=0.4% vs A1 30k 6r=3.5%)
**Verdict: CONFOUNDED — and the clean data argues AGAINST the over-run mechanism.**

The notebook↔A1 pair changes 4 variables at once: iters (100k→30k), weight (20→30),
seed (59→42), target (0.88→0.889). "Over-run" is not isolable from this pair.

The clean iters-only evidence (the w=0 probe, seed42, one schedule) **refutes** rather than
supports the mechanism: across the 10k→100k window 6r is a **plateau, not a decay**
(10k 3.6 / 30k 1.2 / 50k 1.9 / 70k 3.2 / 100k 3.2). Running longer does not coarsen 6-rings
away. Moreover the notebook (100k, w=20) and probe (100k, w=0) share schedule + iters yet
give 6r 0.4% vs 3.2% — at **matched iters**, so the notebook's catastrophic 0.4% is a
**weight/seed/target** effect, NOT over-run.

Clean run to isolate: rerun the notebook config at 30k AND 100k holding w/seed/target fixed
(or read the probe's own 10k vs 100k, already done → plateau). What 6r decay there is happens
in the *first* ~10k (seed 16.8% → ~3% floor), and that crossing is only ever seen through the
w=30 pilot 2k anchor — never at w=0. **This is an over-statement to flag, but the SETTLED
CONCLUSION does not depend on it** (it rests on the w=0 floor, not on over-run).

## Check 2 — "Sustained-T sharpens 8r" (S1 vs A1)
**Verdict: CONFOUNDED comparison — but the doc's DISCLOSURE is VALID.**

Confirmed: S1 vs A1 changes schedule SHAPE + iters (30k→50k) + weight (30→35) simultaneously
(`S1...kwargs.json` vs `A1...kwargs.json`), so "sustained-T sharpens" is not an isolated lever.
The results doc already discloses this explicitly (line 79–81: "changed schedule + iters +
weight simultaneously → SUGGESTIVE ONLY, not an isolated lever") and correctly notes the
headline negative does not depend on it. The downgrade is honest and correct. No fix needed
beyond what is stated. (Clean run to isolate would be sustained-vs-geometric at matched
iters+weight.)

## Check 3 — The decisive negative (6r ≥5% unreachable even at w=0)
**Verdict: VALID (headline) — with one timing caveat to soften.**

The core negative rests on the clean, extension-free w=0 probe (seed42, full 100k): 6r bounces
1.2–3.6% from 10k onward and never reaches 5%. That is the right kind of evidence and it is
unconfounded for "w=0 equilibrium 6r ≈ 3% ≪ 5%." Solid.

Caveat to soften in the doc: the specific phrasing "6r decays through the 5% gate by ~10k
iters, BEFORE girth-6/Φ/void are acceptable" leans on the **w=30 pilot 2k=18.9% anchor**, not
on w=0 data — the 5%-crossing window (5k–10k) is **unobserved at w=0**. The probe's first w=0
point (10k) is already at 3.6%, i.e. below the gate. So the *destination* (<5% at w=0) is
clean; the *timing/path* ("by ~10k", "before void acceptable") borrows a different-weight run.
Recommend rewording to "the w=0 equilibrium 6r is ~3% (≪5%); the seed's 16.8% is lost during
early annealing (pilot, w=30, shows >5% still present at 2k)."

## Check 4 — Arm A vs Arm B not confounded by iters / seed
**Verdict: CONFOUNDED-but-robust-by-magnitude (no clean weight-only pair exists).**

Both arms use seed 42 (good — removes seed as a confound). But iteration count is NOT matched
cleanly to T: the probe's geometric schedule is stretched over 100k while A1's is over 30k, so
at the same iter index the temperatures differ (probe@30k T≈0.032 still cooling; A1@30k T=0.015
cold). There is no run that varies ONLY weight on a fixed schedule+iters. The penalty→void
effect (S_low 0.84→0.12, α −1.55→+1.25 between probe and A1) survives only because the **effect
size dwarfs the schedule confound**, not because it is isolated. Rate the cross-run story
("pure WWW coarsens to 9r+void; penalty fixes void + sharpens 8r") as directionally sound but
formally confounded. Clean isolating run: w=30 on the *same* 100k geometric schedule as the
probe (weight-only delta).

Side finding (contradicts a stated mechanism): the SETTLED CONCLUSION says "the penalty only
lowers 6r FURTHER." The cleanest matched pair (both seed42, both 30k) shows the **opposite**:
A1 (w=30) 6r=3.5% > probe (w=0) 6r=1.2%. The penalty did not lower 6r here. The negative result
survives regardless (every tested weight gives <5%), so the conclusion holds via "all weights
<5%," NOT via "penalty monotonically lowers 6r." Flag the mechanism sentence as over-stated.

## Check 5 — Multi-seed support
**Verdict: VALID for the 6r negative; NEEDS-MORE-EVIDENCE (single-seed) for Arm B + the w=0 floor.**

- 6r < 5% is genuinely **2-seed**: probe s42 (3.2%), A1 s42 (3.5%), S1 s42 (1.6%), S1b s7 (2.0%).
  The hard floor is seed-robust. Void "mostly-solved-modulo-one-mode" is also 2-seed (S1 α 1.93 /
  S1b α 2.03; S_low 0.071 / 0.065). Solid.
- **Arm B refutation (α<0 throughout, 9r coarsening) and the w=0 ~3% equilibrium are
  probe-s42 ONLY = single seed.** Defensible given the large effect (α stays −1.2 to −1.5 the
  whole 100k) and the N=512 prior, but the doc should state the single-seed scope rather than
  imply multi-seed robustness for Arm B.
- 8r seed variability (53.5% s42 vs 43.9% s7) is **not** a problem for the 6r negative or the
  void result. It only makes "sustained-T sharpens 8r toward ref" seed-fragile — and neither
  seed approaches ref 59.7% anyway, so the headline (8r never reaches ref) is unharmed.

---

## VERDICT on the SETTLED CONCLUSION
**The headline causal claims are SOUND:** (a) 6-ring hard floor <5% at N=1000 is clean
(w=0 probe) and multi-seed (s42, s7); (b) the void needs the penalty (pure-WWW S_low
catastrophic, α<0 the whole 100k); (c) Arm B (emergent near-hyperuniformity from pure 100k
WWW) is refuted at full budget. None of these is threatened by the confounds found.

**Over-stated / to correct (none block the headline):**
1. **"Over-run is a ring cause" (Check 1)** — confounded, and the clean iters-only probe
   *refutes* it (6r plateaus 10k→100k; notebook 0.4% is a weight/seed effect at matched iters).
   Downgrade from "CONFIRMED" to "not supported; notebook catastrophe is not over-run."
2. **"The penalty only lowers 6r FURTHER" (Check 4)** — contradicted by the matched 30k pair
   (w=30 6r=3.5% > w=0 6r=1.2%). The negative holds via "all tested weights <5%," not via this
   monotonic mechanism. Reword.
3. **"6r decays through the 5% gate by ~10k" (Check 3)** — the gate-crossing window is
   unobserved at w=0; the 2k=18.9% anchor is a w=30 run. The *destination* is clean; the
   *timing* should be attributed to the pilot's different weight.
4. **Arm B single-seed scope (Check 5)** — state that the Arm-B refutation and w=0 floor are
   probe-s42 only.

Checks 2 (disclosure) and the Check 3 headline are fine as written.
