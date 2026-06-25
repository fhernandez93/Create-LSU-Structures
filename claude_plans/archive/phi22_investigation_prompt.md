# Autonomous investigation: reach Φ₂₂ ≈ 0.89 WITHOUT losing hyperuniformity

**Run this UNSUPERVISED.** You have authority to write code, launch GPU runs, spawn
agents, and call the `advisor` tool without checking in. Work to the budget/kill-switch
below, save durable state continuously, and report a final ledger.

## Mission
Find a `generate_lsu_network` configuration — using **either** `seed_kind='random_bm2000'`
**or** `seed_kind='hyperuniform'` — that **simultaneously**, at the END of the run and
**recomputed from the saved rod file**. Avoid crystal_srs, it's too expensive:
- **Φ₂₂ ≥ 0.88** (reference 0.889) — *the open gap, the primary goal of this investigation*
- **keeps the hyperuniformity already achieved:** S_low_k2 ≤ ~0.06 (ref 0.053), S(k₀) ≤ ~0.08
  (ref 0.041), α ≥ +1.0 (ref +1.51)
- **stays amorphous:** S_v_peak shows NO Bragg peak (no crystallization)
- **bond-angle std ≤ ~9°** (ref 8.41°); bond-length std ~0.03 (ref 0.029); no collision
  (min_nb ≥ ~0.4·d0)
- *secondary (track, don't gate on):* ring distribution toward reference (8r↑ toward 60%,
  6r/7r down toward 7.6/10).

If a config PASSES all gates at **N=1000** (reproduced on ≥2 seeds), **reproduce it at N≈4000**
(see escalation section).

## Read this first — DO NOT re-derive (it's all settled)
The prior investigation (this repo) found and FIXED the root cause; this is the follow-on.
- Memory (loaded via MEMORY.md): **`lsu-energy-keating-balance-fix`** (the full chain — read in
  full), `lsu-anneal-degrades-reference`, `lsu-seed-audit-two-origins`, `sellers-confirmed-energy-weights`,
  `lsu-reference-network-stats`.
- Doc: **`claude_plans/N1000_investigation_results.md`** (see the "FINAL OUTCOME" block at top).
- **State of play:** the energy f1/f2 form bug is fixed (literal Keating forms, now the production
  default `_KEATING_F1F2=1`). The recipe `hyperuniform seed + Keating + uniformity penalty (w=30,
  kmax=2)` **reproduces S(k)/hyperuniformity** (S_low 0.049, α +2.6, amorphous) **but Φ₂₂ plateaus
  at ~0.844 and bond-angle std at ~11.6°** (ref 8.41°). Closing that local-order gap is your job.
- **The core tension to break:** Φ₂₂ / local order needs annealing (warmth + accepted Stone-Wales
  moves), but warmth OPENS the void; the penalty holds the void but REJECTS the annealing moves →
  Φ₂₂ plateaus. Confirmed both with w=0 (full freedom → angstd plateaus ~11°, void opens) and w=30
  (void held → Φ₂₂/angstd plateau). The reference's local order is HELD by the Keating energy
  (anneal-from-reference is stable) but NOT REACHED from a disordered seed.
- **Entry points:** `generate_lsu_network(seed_kind='hyperuniform', hyperuniform_kmax=3, ...)` or
  `seed_kind='random_bm2000'`. Helpers: `Claude_Helpers/_run_hyperuniform.py` (checkpointed runner +
  angle-std), `Claude_Helpers/_hyperuniform_seed.py` (`hyperuniform_points`), `Claude_Helpers/_metrics.py`
  (`full_metrics_safe`, collision-tolerant). `lsu_network.hyperuniform_placement(...)` is the in-code
  version. Best structure so far: `Example/20260622_lsu_hyperuniform_N1000_ends.txt` (+README).

## Hypotheses / levers — prioritized (break the warm-vs-void tension)
1. **MULTI-STAGE protocol (most promising).** Separate the conflicting phases instead of fighting
   them in one schedule. E.g. Stage A: anneal local order WARM (no/low penalty) to drive Φ₂₂ up and
   narrow angles; Stage B: restore the low-k either by re-running the collective-coordinate
   `hyperuniform_placement`-style optimization **on the annealed vertex positions** (re-hyperuniformize
   the geometry the topology already has), and/or ramping the penalty on a final cold hold. Possibly
   ALTERNATE A/B. Watch that Stage B doesn't broaden the angles back (relax under Keating after).
2. **Schedule shape:** a warm HOLD (T≈0.08–0.12) long enough to build Φ₂₂, then slow cool with the
   penalty ON to hold the void. Tune hold-T, hold length, and penalty ramp timing.
3. **Late/ramped penalty:** apply uniformity_weight only AFTER the warm local-order phase (so early
   annealing isn't blocked), ramping up as you cool. (Keep kmax=2 — see memory for why; don't drop to 1.)
4. **Seed with better starting local order:** `seed_kind='crystal_srs'` (the gyroid crystal — starts
   with sharp 8-rings + ordered angles → high Φ₂₂) + a melt to kill Bragg + the penalty to hold the
   void. This trades Sellers-faithfulness for reaching the target; flag it but it may be the fastest
   path to Φ₂₂≈0.89. Compare against the random/hyperuniform routes.
5. **Longer anneal (100k):** note Φ₂₂ plateaued by ~60k under the penalty, so more iters ALONE is
   likely insufficient — verify, don't assume.
6. Secondary: relax depth, threshold-relax (c_f), relax_local_iters.

## Hard constraints (NEVER violate)
- `energy_weights` FIXED: alpha=0.7, beta=0.7, gamma=0.3, delta=0.4 (Sellers-confirmed — never change).
- Keep the **Keating energy** (default `LSU_KEATING_F1F2=1`); do NOT revert f1/f2 to the old forms.
- Never delete/overwrite `Example/lsu_example_ends.txt` or any file you did not create. Only ADD
  dated output files. Save winning structures to `Example/` as new dated files + a README.
- GPU = **ONE run at a time** (RTX 4080, 12 GB — parallel runs OOM). CPU-only analysis:
  `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.
- Env = conda `lsu_project` (JAX 0.10.0 CUDA). For live logs use
  `conda run --no-capture-output -n lsu_project python -u ...` (plain `conda run` buffers stdout).
- d0 (edge_length) = 0.8, BOX = 11.44 at N=1000 (density 0.668).

## Validation protocol (do this for EVERY run)
- **Recompute every metric independently FROM the saved rod file** — never trust an inline table.
  Use `Claude_Helpers/_metrics.full_metrics_safe` + the angle-std computation in `_run_hyperuniform.py`.
  Report: Φ₂₂, Φ₁₂, S(k₀), S_low_k2, α (S_v_alpha_low), **S_v_peak (Bragg/amorphous check)**,
  bond-angle mean/std, bond-length std, dihedral entropy, ring distribution + girth, min_nb (collision).
- **Checkpoint every run** (e.g. every 10k iters) so a crash/collision can't lose the trajectory;
  keep measurements collision-tolerant (skip-and-continue — see `_run_hyperuniform.py`).
- Reference (recompute once from `Example/lsu_example_ends.txt` to confirm the harness): Φ₂₂ 0.889,
  Φ₁₂ 0.985, S(k₀) 0.041, S_low 0.053, α +1.51, bond-angle 120.0°/std 8.41°, dih_ent 0.796,
  rings 6:7.6 7:10.0 8:59.7 9:20.9, ring mean 7.99, S_v_peak ~1.82, min_nb inf.

## Cross-review with agents & decision CONVERGENCE (REQUIRED — do not skip)
Every **significant decision or claim** must be cross-reviewed by **≥2 independent agents** and must
**converge** before you act on it or report it as settled. This is mandatory, not optional — the prior
investigation repeatedly caught its own errors this way (a wrong "energy is broken" leap, an
over-claimed "2-seed confirmed", a stale memory, a miscounted reference citation).
- **What must be cross-reviewed:** (a) any claim that a config PASSES/FAILS the gates; (b) the
  attributed CAUSE of a plateau or a result ("Φ₂₂ plateaus because X"); (c) the choice of "winner"
  config; (d) any non-trivial code change (multi-stage runner, O(N²)→O(N) hyperuniformizer, a new seed
  path) — agents verify correctness AND that it respects the hard constraints; (e) the final ledger.
- **Use independent lenses** (spawn separate agents), e.g.: a **metrics/validation** agent (recompute
  every number from the saved file, confirm ≤ some tolerance), a **causal-attribution** agent (is the
  stated cause supported, or confounded?), and a **fidelity/constraints** agent (weights fixed? Keating
  kept? amorphous? Example untouched?). Each writes a review file in `claude_plans/agent_*_review.md`.
- **CONVERGENCE rule:** proceed only when the independent reviews AGREE. If they DIVERGE, you MUST
  reconcile to convergence BEFORE building on the decision — re-examine the artifacts, re-run if
  needed, and put the conflict to the **`advisor`** (surface both sides + your evidence). Do not
  silently pick a side, and do not advance on a non-converged decision. Record the reconciliation.
- Treat agent corrections as binding unless you have primary-source evidence against them (then
  reconcile via one more advisor pass). Update the durable docs/memory whenever a review corrects you.

## Process & autonomy
- Use the **`advisor`** tool at least: once before committing to the ablation order, once at each
  divergence to reconcile agent reviews, and once before declaring a result final. Give it serious
  weight; surface conflicts with your own data rather than silently switching.
- Smoke-test every code change before a long run; gate trusting it on the agent cross-review above.
- **Budget / kill-switch:** ~16 h wall, ≤ ~12 configs, per-run timeout. If a clear plateau or negative
  emerges (e.g. Φ₂₂ and S(k) provably can't co-exist in the explored space) AND the agents converge on
  that, STOP and report the honest negative — do NOT spiral into endless weight/schedule sweeps.
- Save durable state continuously: a results `.md` in `claude_plans/` updated after every run, and
  update memory (extend `lsu-energy-keating-balance-fix` or add a new memory) with what you learn.

## N≈4000 escalation (only if a good N=1000 config is found)
If a config passes ALL gates at N=1000 on ≥2 seeds, reproduce it at **N≈4000**, density-matched:
`box = (N/1000)**(1/3) * 11.44` → for N=4000, box ≈ 18.16.
- **Watch:** `hyperuniform_placement` has an O(N²) soft hard-core term — at N=4000 (~1.6e7 pairs) it
  may be slow / memory-heavy. Test runtime first; if needed, switch the hard-core to a neighbor-list /
  cutoff (O(N)) or chunk it. (The low-k S(k) term is cheap — O(N·M).)
- N=4000 at ~100k iters is multi-hour: checkpoint, GPU one-at-a-time.
- Validate the N=4000 sample on the SAME gates (recomputed from file) and save to `Example/` as a new
  dated file + README if it passes.

## Deliverables
1. If a good config is found: the N=1000 (and N≈4000) structure(s) saved to `Example/` as new dated
   files + a README (recipe, validated metrics vs reference, caveats) — and **the exact
   `generate_lsu_network(...)` call** that reproduces it.
2. A results writeup in `claude_plans/` + updated memory.
3. A final honest ledger: did Φ₂₂ reach ~0.89? what (if anything) it cost on S(k)/amorphousness, and
   whether random vs hyperuniform vs crystal_srs seed was the winner. If the targets are mutually
   unreachable in the explored space, say so plainly with the evidence.
