# Autonomous investigation prompt — fix low-k S(k) mismatch + 6-ring depletion at N=1000 (the notebook example)

## Mission
The example run in `Create_LSU_Function.ipynb` (N=1000, full reference scale) is supposed to
reproduce the gold-standard statistics of `Example/lsu_example_ends.txt`. It is **globally
amorphous** (the decisive gate passes), but it still **misses the low-k S(k) and the 6-ring
distribution**. Find a `generate_lsu_network` configuration that closes both gaps **at full
N=1000**, confirm the WWW relaxation is **Sellers-faithful**, and have every conclusion
cross-checked by independent review agents. Iterate autonomously. This is a meta-goal too:
the prior N≈512 work concluded no config passed both gates and that more iterations / finite
size were the get-out — but this failure is at **N=1000 and 100k iterations**, so those
excuses are off the table. Settle it.

## Hard constraints (do not violate)
- **Seed must be `seed_kind='random_bm2000'`.** The crystal path is impractical at scale and
  off the table. The raw seed is healthy (it has a rich 5/6-ring tail); the WWW chain is what
  coarsens it. Confirm seed health before blaming it.
- `d0 = edge_length = 0.8`. `BOX = (N/1000 * 11.44**3)**(1/3)` = 11.44 at N=1000 (density-matched
  to reference 0.668).
- Energy weights are FIXED: `alpha=0.7, beta=0.7, gamma=0.3, delta=0.4` (Sellers-group confirmed —
  **never change these**).
- **Run at full N=1000 only.** No N=512 ablation. Every run is reference-scale (~1h on GPU);
  budget configs accordingly (see kill switch).
- Work in `/home/francisco/Documents/Create LSU Structures  - Claude`. Code: `lsu_network.py`,
  `tools.py`. Reference network: `Example/lsu_example_ends.txt` (N=1000, box=11.44).
- Never delete/overwrite `Example/lsu_example_ends.txt` or any file you did not create this
  session. Only ADD dated output files to `Structures/`.

## The two problems to solve (the whole point)
The last notebook run (20260619: `uniformity_weight=20`, T 0.045→0.015, 100k WWW, seed 59):

| Metric | Reference | This run | Status |
|---|---|---|---|
| **6-ring fraction** | 7.6% | **0.4%** | severe depletion ✗ |
| 8-ring fraction | 59.7% | 48.5% | low |
| ring mean | 7.99 | 8.20 | shifted to 9-rings |
| **S_low_k2** | 0.053 | **0.095** | low-k void excess ✗ |
| **α(k<2) hyperuniformity** | +1.51 | **0.99** | slope too shallow ✗ |
| Φ_22 | 0.8887 | 0.8647 | 0.024 low |
| bond std (·d0) | 0.029 | 0.112 | high (secondary) |
| S_v peak / Bragg(>3) | 1.82 / 0 | 1.44 / 0 | amorphous ✓ |

**Problem 1 — 6-ring depletion / ring coarsening:** the distribution lost its 6-ring tail and
shifted toward 9/10-rings.
**Problem 2 — low-k S(k) mismatch:** S_low_k2 and the hyperuniformity slope α(k<2) are both
off — long-wavelength density is not suppressed the way the reference's is.
Bond-std and Φ_22 are **secondary symptoms**, not primary targets — report them, don't chase them.

## Established diagnosis (verified — do NOT re-derive, start from here)
From the N≈512 validation (`claude_plans/N500_validation_results.md`,
`nominee_config_N512.json`), the three prior agent reviews
(`claude_plans/agent_{stats,causal,fidelity}_review.md`), and memory files
(`lsu-n512-validation-temperature-and-tradeoff`, `lsu-cold-www-coarsening`,
`lsu-www-faithfulness-audit`, `lsu-sellers-protocol-random-pure-www`,
`lsu-reference-network-stats`, `sellers-confirmed-energy-weights`):
- **Temperature is the dominant ring lever.** Cold T preserves 6-rings; hot T coarsens them.
  The default `initial_temperature=0.5` is too hot; the notebook already uses cold 0.045→0.015.
- **The core tradeoff:** `uniformity_weight>0` fixes the low-k void but **depletes 6-rings**.
  At N=512, *no single config passed both gates* — penalty buys void at the cost of rings.
  This is the central tension this investigation must confront head-on at N=1000.
- **`local_shell_depth=4` IS faithful** — it is the canonical Vink/Mousseau-Barkema
  4th-neighbour-shell cluster relax, NOT a shortcut. The N=512 fidelity review confirmed this,
  and full-N relax gave no benefit and is not the void cause. The void is **baked into the
  topology**, not a coordinate-relaxation artifact (a post-hoc global relax was a no-op).
- **The `uniformity_weight` penalty is a non-Sellers extension** (absent from Eq. 2).
  Sellers/Hejna claim near-hyperuniformity is *emergent* from long pure WWW — but that has
  only ever been tested out to ~40k (where it coarsens, not emerges). The ~100k pure-WWW
  emergence claim is **untested** and is exactly Arm B below.
- The BM2000 seed reproduces girth≥5 but drops the BC→AB+AC loop-expansion move (by design;
  ring shaping is left to the anneal).

## Two parallel investigation arms
Run BOTH. They answer different questions; compare their outcomes at the end.

### Arm A — break the penalty↔6-ring tradeoff
Goal: a config that passes BOTH gates jointly at N=1000. The penalty-vs-T frontier is already
mapped and doesn't pass both — so explore levers that decouple void control from ring
coarsening. Ablate **one at a time**, attribute cause:
1. **Late penalty annealing** — start `uniformity_weight` high to kill the box-scale void early,
   then ramp it to ~0 over the final fraction of the run so the rings can re-equilibrate under
   pure Sellers energy. (Check whether `generate_lsu_network`/`www_anneal` support a per-iter
   weight schedule; if not, approximate via staged runs that restart from a saved network.)
2. **Penalty k-window** — `uniformity_kmax` (1 shell vs 2). A tighter window may suppress the
   void with less ring collateral.
3. **Φ_22 target = reference 0.889** (not over-driven toward 1.0) — over-driving Φ narrows the
   ring distribution; confirm the notebook's 0.88 target isn't itself coarsening rings.
4. **Temperature fine-tune around the cold band** — capture the printed `T_melt`; ensure T is
   cold enough to keep 6-rings but hot enough that acceptance doesn't collapse.
5. **Seed bond-std conditioning** — the ~0.11 bond-std floor was traced to seed-settle strain at
   N=512; check whether it persists at N=1000 and whether a gentler seed conditioner helps.

### Arm B — pure-Sellers emergence (the fidelity baseline)
Goal: test the untested Sellers-faithful hypothesis directly. Set `uniformity_weight=0`,
faithful WWW (local-shell relax depth 4, geometric T schedule, Sellers weights), and run
**≥100k iterations** (push further if budget allows). Measure whether near-hyperuniformity
AND the ring distribution emerge on their own, as Sellers/Hejna claim. This is the honest
control: if Arm B emerges both, the penalty is an unnecessary crutch; if it doesn't even at
≥100k, that is itself a publishable, decisive negative result about the protocol at this scale.

## WWW Sellers-fidelity audit (required, in addition to the runs)
Independently re-verify that the relaxation/anneal is faithful to Sellers (Nat. Commun. 2017,
Methods + Vink 2001 / Mousseau-Barkema 2001) against the code AND the primary PDFs in
`LSU Literature/`. For each item, classify as **faithful / extension / shortfall** with a
one-line justification (reuse the prior fidelity-review vocabulary):
- **Bond move:** Stone-Wales transposition `(i,c)+(j,d)→(i,d)+(j,c)` — `stone_wales_propose`
  (`lsu_network.py:2044`), `stone_wales_apply` (`:2101`). Confirm no spurious extra moves in the
  anneal loop and that revert is an exact inverse.
- **Metropolis + Vink threshold identity:** `www_anneal` (`:2136`), acceptance `s < exp(-ΔE/T)`
  and the early-reject ≡ rejection threshold scheme. Confirm `s` reuse logic is sound.
- **Temperature schedule:** geometric T0→T_final.
- **Energy components f1..f4 and weights:** `energy_components` (`:1541`), `total_energy` (`:1598`)
  — bonds, angles (cos→−1/2), dihedral, skew; weights 0.7/0.7/0.3/0.4.
- **Local-shell relax:** `compute_local_shell_mask` (`:1467`, depth 4), `relax` (`:1897`).
  Confirm gradient masking holds out-of-shell vertices fixed and depth-4 = Vink canonical.
- **Uniformity penalty:** `_acceptance_objective` (`:1451`) — confirm it is applied only to the
  Metropolis objective, NOT inside the L-BFGS strain relax (so the local geometry stays pure
  Sellers). Classify as extension.
- **Seed:** `random_seed_network_bm2000` (`:474`) — girth≥5, dropped loop-expansion move.

## Validation protocol (every run)
- Save rods to `Structures/` with a dated, tagged name; save the exact kwargs alongside.
- Capture the FULL verbose stdout (`verbose=True`, `check_lsu_every=500`): `T_melt`, acceptance %,
  early-reject %, Φ trajectory. The notebook log's acceptance/early-reject behavior was never
  inspected — inspect it.
- Recompute the full metric set **independently from the saved rod file** with
  `tools.analyze_network(rods, box=BOX, d0=0.8)` — reuse the patterns in
  `Claude_Helpers/_metrics.py` and `_verify_r10.py`. **Never trust an inline/printed table**;
  re-load and recompute. Compare per-shell low-k S(k) including **S(k₀), the single lowest mode**
  (the robust void signal). Reference loads from `Example/lsu_example_ends.txt`.
- If `tools.rods_to_network` raises a degree-4 round-trip error, work from in-memory
  positions/edges or check for coincident vertices (a real clumping signal).
- Append every run to a results table in `claude_plans/N1000_investigation_results.md`
  immediately after it finishes. Never overwrite. This file IS the durable state.

## Success criteria at N=1000 (reference is N=1000 — no finite-size slack this time)
A passing config must hit the reference column, not approximate it:
- **6-ring fraction ≥ 5%** (ref 7.6%); ring mean **7.8–8.1** (ref 7.99); 8-ring the dominant ring.
- **S_low_k2 ≤ ~0.06** (ref 0.053) **and α(k<2) ≥ ~1.3** (ref 1.51); S_v slope clearly positive.
  Treat S(k₀)/S_low_k2 as the void pass/fail gate.
- Φ_22 ≥ 0.88 (ref 0.889; do not over-drive toward 1.0); Φ_12 ≈ 0.985.
- bond std tracked (ref 0.029) — report, secondary.
- AMORPHOUS check passes: S_v peak ≈ ref, **Bragg(>3) = 0**.
Run the same comparison harness the notebook uses (metric table + ring-length table +
GLOBAL-AMORPHOUS S_v check) so results sit directly next to the reference column.

## Multi-agent review (required — do not trust a single pass)
Before declaring any result final, spawn **three independent review agents**, each with a
distinct scope (mirror the prior `agent_{stats,causal,fidelity}_review.md`). Each writes its
own file: `claude_plans/agent_stats_review_N1000.md`, `agent_causal_review_N1000.md`,
`agent_fidelity_review_N1000.md`.
- **stats/metrics agent** — independently re-load the best output(s) from the saved rod files and
  confirm every reported metric (rings, bond stats, S(k₀)/S_low_k2/α, Φ, amorphous check)
  actually matches the results table, with no box/density/measurement error.
- **causal-attribution agent** — confirm the ablation isolates which lever fixed each problem
  (one variable at a time, no confound), with ≥2-seed support for any winner; check that Arm A
  vs Arm B conclusions aren't confounded by iteration count or seed.
- **Sellers-fidelity agent** — confirm the WWW-fidelity audit verdicts against the PDFs in
  `LSU Literature/`, and characterize the winning config's deviations (penalty, target,
  relaxation locality, schedule) as faithful vs engineering-crutch.
Take any contradiction seriously: reconcile it (re-run the check, or surface via the advisor)
before locking the conclusion. Do not silently override a reviewer.

## Autonomy rules (runs unsupervised — do not stop to ask)
- Work the plan end-to-end. Make decisions, run them, measure, adjust. Use **background**
  execution for the long N=1000 runs and poll for completion.
- **Ablate one variable at a time** within each arm; keep the running results table current.
- Call the **advisor** tool: once before committing to the ablation order across both arms, and
  once before declaring a result final. Give its guidance real weight.
- Save the best network(s) to `Structures/` with a dated name + exact kwargs.
- Update memory (`MEMORY.md` + a file) with what was validated/refuted. If the N=1000 data
  contradicts a stored N=512 belief (e.g. "no config passes both gates"), the **data wins** —
  correct the memory.

## Kill switch / budget (hard stops — must not spin forever)
- **Total wall-clock budget: 16 hours.** Record start time in your first action. Before launching
  any run, check elapsed; if ≥16h, STOP, write up the best result, exit — even if criteria unmet.
- **Per-run timeout:** wrap every generation run in a hard timeout (`timeout 5400 python ...` =
  90 min for a ~100k N=1000 run; scale up only for the deliberate ≥100k Arm B long run, capped
  at `timeout 10800`). On timeout: kill, log "TIMEOUT", move on — never retry unchanged.
- **Max configs: 12** total across both arms (≈ Arm A 7 / Arm B 5; reallocate as needed). If 12
  runs haven't met criteria, STOP and report the best + what you'd try next.
- **No-progress stop:** if 3 consecutive runs fail to improve the gating metrics (6-ring% and
  S_low_k2) over best-so-far, STOP and report — you're not converging.
- **Multi-seed confirmation:** any claimed winner must be reproduced on ≥2 seeds before it counts
  (this consumes config budget — plan for it).
- **Checkpoint:** the results table is durable state — write to it immediately after each run.
- On ANY stop (budget, max configs, no-progress, success): write a final `## SUMMARY` block to
  `claude_plans/N1000_investigation_results.md` with best kwargs, metrics vs reference, which
  levers mattered (before/after numbers), the Arm A vs Arm B verdict (did pure WWW emerge both?),
  and the WWW-fidelity audit conclusion. This is the deliverable whether or not you converged.

## First steps
1. Read the memory files and prior results named above so you don't re-derive.
2. Record start time. Regenerate just the random_bm2000 seed (N=1000, seed=42) and confirm its
   ring tail and bond stats are healthy.
3. Re-load the existing notebook output (`Structures/`/`Example/20260619_*lsu_generated.txt`) and
   independently confirm the failing numbers in the table above — establish the baseline you're
   beating.
4. Launch **Arm A run 1** (start from `nominee_config_N512.json` scaled to N=1000:
   `lsu_degree_22=0.889`, `uniformity_kmax=2`, cold T, plus the first Arm A lever) in background,
   and queue **Arm B run 1** (`uniformity_weight=0`, ≥100k, faithful WWW). Measure, compare, ablate.
