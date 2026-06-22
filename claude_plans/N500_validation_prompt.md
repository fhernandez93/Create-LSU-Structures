# Autonomous validation prompt — fix 6-ring deficit + low-k S(k) at N~512 (random_bm2000)

## Mission
Find a `generate_lsu_network` configuration that produces a **random_bm2000-seeded** degree-3
LSU network whose statistics match the reference gold standard, then hand me a validated config
plus a predicted-runtime plan for scaling to **N≈5000**. Iterate autonomously at **N≈512** until
the success criteria below are met. Do **not** launch the N≈5000 run — produce the recipe for it.

## Hard constraints (do not violate)
- **Seed must be `seed_kind='random_bm2000'`.** Crystal-seed annealing is impractical at N~5000,
  so the crystal path is off the table. Confirm the random seed is healthy before blaming it
  (last session: raw seed had 6-rings 17%, 5-rings 22%, ring mean 7.13, S_low_k1 0.11 — it is fine;
  the WWW chain is what coarsens it).
- `d0 = edge_length = 0.8`. `BOX = (N/1000 * 11.44**3)**(1/3)` (density-matched to reference 0.668).
- Energy weights are FIXED: `alpha=0.7, beta=0.7, gamma=0.3, delta=0.4` (Sellers-group confirmed —
  never change these).
- Work in `/home/francisco/Documents/Create LSU Structures  - Claude`. Code: `lsu_network.py`,
  `tools.py`. Reference network: `Example/lsu_example_ends.txt` (N=1000, box=11.44).

## Two problems to solve (the whole point)
1. **6-ring deficit / ring coarsening.** Bad run gave 6-rings 0.8% and ring mean 8.33 vs reference
   7.6% and 7.99 (distribution shifted to 9-rings).
2. **Low-k S(k) mismatch.** Bad run gave S(k₀)=0.815 at the lowest mode and S_low_k2=0.327 vs
   reference S(k₀)≈0.04 and S_low_k2=0.053 (the S_v slope flips negative = NOT hyperuniform).

## Established diagnosis (verified last session — start here, don't re-derive)
- The bad run used `lsu_degree_22=0.9999` → this becomes the literal stop target
  `target_lsu=float(lsu_degree_22)` (lsu_network.py:3621). 0.9999 ≈ the crystalline limit; the
  reference Φ_22 is only **0.889**. Over-driving Φ narrows the ring distribution and kills the
  5/6-ring tails. **Prime suspect for the 6-ring problem.**
- The bad run used `uniformity_weight=0`. Sellers prescribes no S(k) penalty (hyperuniformity is
  meant to be emergent), but over a long bonded-only run a box-scale void grows. **Prime suspect
  for the low-k problem** — it's the documented guard (see `low_k_structure_factor` docstring).
- A full **global relax of the final bad config was a no-op** (max vertex move 0.0023 d0, S_low
  unchanged). So the void and bond-length spread are **baked into the topology**, not incomplete
  relaxation — post-hoc relaxing does NOT fix it. The levers are the Φ-target + long-wavelength
  control during the chain, not coordinate relaxation.
- Relaxation faithfulness (`local_shell_depth=4` = local-only, vs Sellers' full-N) is a real
  fidelity gap but is a SEPARATE issue; test whether full-N relax DURING the chain (it changes
  which moves get accepted — frozen far-field can make void-creating moves look cheap) actually
  reduces the void. Plumbing already supports `local_shell_depth=None`; if you use it, also raise
  the per-move budget (per-move relax uses `relax_local_iters`, ~150 is too few to converge a
  full-N relax — use 600–1000, or it's a fake fix).
- Background context lives in memory files: `lsu-cold-www-coarsening`, `lsu-www-faithfulness-audit`,
  `lsu-sellers-protocol-random-pure-www`, `lsu-reference-network-stats`, `sellers-confirmed-energy-weights`.

## Levers to ablate (change ONE at a time, attribute cause)
1. **Φ target → reference value.** Set `lsu_degree_22=0.889` (or `lsu_degree_12=0.985`), NOT ~1.0.
2. **uniformity_weight > 0** (try 5, 10, 15) with **`uniformity_kmax=2`** (note: 1.5 truncates to
   one shell — a bug-ish setting; use integer 2).
3. **Relaxation fidelity:** `local_shell_depth=None` + `relax_local_iters≈800`, vs the default
   local-only. Compare void outcome and runtime.
4. **Temperature vs T_melt:** capture the printed `T_melt` and ensure the WWW temperature is hot
   enough to actually equilibrate rings (the prior 0.045→0.015 may be too cold). Don't run so cold
   that acceptance collapses, nor so hot it never settles.
5. **Burn-in melt** (`burn_in_n_heat/cool/quench`): default 8000/16000/4000. Test on vs off for the
   random seed. It's a Hemmann (non-Sellers) step; it may help homogenize density but adds cost.
   Treat as optional, last.

## Validation protocol
- Use **short N=512 runs** (e.g. `n_www_iterations=10_000–25_000`, `check_lsu_every=500`,
  `verbose=True`). Always pass `seed=42` first for reproducibility, then confirm with 1–2 other seeds.
- Capture the FULL verbose stdout each run (T_melt, acceptance %, early-reject %, Φ trajectory) —
  the prior run's log was never inspected and that left acceptance/early-reject deviations untested.
- Measure every run with `tools.analyze_network(rods, box=BOX, d0=0.8)` and compare per-shell low-k
  S(k) with the existing `_cmp_sk.py` (especially **S(k₀), the single lowest mode** — the robust
  void signal). Reference loads from `Example/lsu_example_ends.txt`.
- Note: if `tools.rods_to_network` raises a degree-4 error on round-trip, work from the in-memory
  positions/edges or check for coincident vertices (it's a real clumping signal).

## Success criteria at N≈512 (finite-size-aware; reference is N=1000)
Target the reference column, accepting finite-size slack on the few-mode low-k band:
- 6-ring fraction **≥ 5%** (ref 7.6%); ring mean **7.8–8.1** (ref 7.99); 8-ring the dominant ring.
- bond-length std **< 0.045** (ref 0.029); bond mean ≈ 0.80.
- **S(k₀) (lowest mode) < ~0.15** and **S_low_k2 < ~0.10** (ref 0.041 / 0.053); S_v slope not
  negative. Treat the S(k₀) spike as the pass/fail gate for the void.
- Φ_12 ≈ 0.985 and Φ_22 ≈ 0.889 (match reference, do NOT exceed toward 1.0).
- AMORPHOUS check passes (no Bragg peaks), as the existing comparison script reports.
Run the same comparison harness the user already used (the metric table + ring-length table +
GLOBAL-AMORPHOUS S_v check) so results are directly comparable to the reference column.

## Autonomy rules (this will run unsupervised — do not stop to ask)
- Work the plan end-to-end without pausing for confirmation. Make decisions, run them, measure,
  adjust. Prefer **background** execution for long runs and poll for completion.
- Keep each exploratory run time-bounded (short iteration counts); don't burn hours on one config.
  Budget: aim to converge the N=512 recipe in a handful of short runs, not one giant run.
- **Ablate one variable at a time** and keep a running results table (params → metrics) in
  `claude_plans/N500_validation_results.md`. Append every run; never overwrite.
- Call the **advisor** tool: once before committing to the ablation order, and once before
  declaring the recipe validated. Give its guidance real weight.
- **Findings must be verified with agents — do not trust a single pass.** Before declaring the
  recipe validated (and before writing the N≈5000 recommendation), spawn 2–3 independent review
  agents to cross-check the conclusions against the data and the primary sources. Give each a
  distinct scope, e.g.: (a) **stats/metrics agent** — re-load the best output independently and
  confirm the reported metrics (rings, bond stats, S(k₀)/S_low, Φ, amorphous check) actually match
  what the results table claims, with no measurement/box/density-matching error; (b) **causal-attribution
  agent** — confirm the ablation genuinely isolates which lever fixed each problem (one variable at a
  time, no confound), and that the claimed cause survives the "global relax was a no-op → void is
  topological" evidence; (c) **Sellers-fidelity agent** — confirm the winning config's deviations
  from strict Sellers (uniformity_weight>0, target value, relaxation locality, burn-in) are correctly
  characterized as faithful vs engineering-crutch, against the PDFs in `LSU Literature/`. Treat an
  agent's contradiction seriously: reconcile it (re-run the check, or surface the conflict via the
  advisor) before locking the conclusion — do not silently override a reviewer. Spawning agents is
  expected here and counts against neither the 12-config nor the time budget's exploration limit, but
  still respect the 4-hour hard cap.
- Save the best network to `Structures/` or `Example/` with a dated name; save its exact kwargs.
- Update memory (`MEMORY.md` + a file) with what was validated/refuted. Correct the existing
  hypotheses if the data disagrees — the data wins over stored beliefs.
- Do **not** run N≈5000. When N=512 passes, produce: (a) the final validated kwargs, (b) which
  lever(s) actually mattered (with before/after numbers), (c) a predicted-runtime + recommended
  config for N≈5000 — including whether full-N relax is affordable there or whether the Vink
  hybrid (local relax + periodic global relax) is the practical fidelity path at that size.

## Kill switch / budget (hard stops — an unsupervised run must not spin forever)
- **Total wall-clock budget: 4 hours.** Record the start time in your first action. Before
  launching any new run, check elapsed time; if ≥ 4 h, STOP exploring, write up the best result
  so far, and exit — even if criteria are not met.
- **Per-run timeout:** wrap every generation run in a hard timeout (e.g. `timeout 1800 python ...`
  = 30 min). If a single config exceeds it, kill it, log "TIMEOUT" in the results table, and move
  on — never retry the same config unchanged.
- **Max configs: 12.** If 12 ablation runs have not met the success criteria, STOP and report the
  best config + what you'd try next. Do not keep inventing variations past 12.
- **No-progress stop:** if 4 consecutive runs fail to improve the gating metric (S(k₀)) over the
  best-so-far, STOP and report — you're not converging; surface it for me rather than burning budget.
- **Hard safety rails:** never delete or overwrite `Example/lsu_example_ends.txt` (the reference) or
  any existing file you did not create this session; only ever ADD dated output files. Never edit
  the energy weights. Never start the N≈5000 run regardless of remaining budget.
- **Checkpoint:** after every run, the results table in `claude_plans/N500_validation_results.md`
  IS the durable state — write to it immediately after each run so a stop at any point leaves a
  complete record and the best config recovered.
- On ANY stop (budget hit, max configs, no-progress, or success): write a final
  `## SUMMARY` block to the results file with the best kwargs, its metrics vs reference, which
  levers mattered, and the N≈5000 recommendation. This is the deliverable whether or not you converged.

## First steps
1. Read the memory files named above and the prior results so you don't re-derive.
2. Regenerate just the random_bm2000 seed (N=512, seed=42) and confirm it's healthy.
3. Run baseline-corrected config: `lsu_degree_22=0.889`, `uniformity_weight=10`, `uniformity_kmax=2`,
   `n_www_iterations=15_000`, capture full log, measure, compare. Then ablate from there.
