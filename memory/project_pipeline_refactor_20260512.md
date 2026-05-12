---
name: 2026-05-12 pipeline refactor (Hemmann triangular burn-in + Vink/MB threshold relax + BM2000 random seed)
description: Major refactor of lsu_network.py end-to-end pipeline — adds Vink Eq. 5 / BM2000 Eq. 3-4 early-rejection relax, Hemmann triangular burn-in, BM2000 random seed alternative, and a cluster-diagnostic helper. Notebook is wired to the new API. Supersedes the constant-T burn-in section of [[seed-network-rewrite]] (2026-05-11) but keeps everything else there valid.
metadata:
  type: project
---

## What changed and why

The user asked to make the pipeline **literature-faithful for Type-2 amorphous gyroid networks** (Φ_22 ≈ 0.89, matching `Example/lsu_example_ends.txt`). Two gaps in the prior code were identified and fixed:

1. **Burn-in was constant-T at auto-calibrated 20% acceptance.** Hemmann/Saba 2026 § 2.3 + Figure 2 actually prescribes a **triangular profile** (heat 0→T_max, cool T_max→0, quench at T=0) with T_max calibrated against T_melt via Hemmann Eq. 5. The constant-T version had no quench leg and no melting-temperature anchor.
2. **`relax` ran a single L-BFGS pass per SW move.** Sellers cites refs [13]=Vink 2001 and [14]=Mousseau-Barkema 2001 for the relaxation refinements; both prescribe **threshold-energy early rejection** — `E_t = E_b − T·ln(s)`, abort relax when `E − c_f|F|² > E_t` (BM2000 Eq. 3-4). Hemmann § 2.1 quotes the same recipe with the 5-cycle anharmonic warm-up and local→global promotion at cycle 10.

The user explicitly chose (AskUserQuestion 2026-05-12): "Do both, and don't take any shortcuts nor simplifications" for the seed pipeline (keep crystal-srs *and* add BM2000 random) and "Yes, full Vink/MB scheme" for the relax.

**Plan file:** `/home/francisco/.claude/plans/we-are-going-to-twinkling-nest.md` (executed plan; useful as a chronological record of intent).

## New functions in `lsu_network.py`

### `cluster_diagnostics(positions, edges, neighbors, box, d0, probe_grid=12)` — `lsu_network.py:1238`
Read-only diagnostic. Returns dict with Hemmann/Saba metrics:
- `r_nn`, `r_u` (Hemmann's nearest-uncoordinated-neighbour distance, target ≥ 1.0·d0)
- `delta_c` (critical pore radius via probe grid, Hemmann target ≤ 0.5·d0)
- `min_non_bonded`, `n_close_pairs` (cluster guards)
- `bond_len_{mean,std,min,max}`, `voxel_std_4`, `S_low_k2`

Used at three checkpoints in `generate_lsu_network`: post-seed, post-initial-relax, post-burn-in, and at the very end. Hard-fails if `min_non_bonded < 0.4·d0` after the initial relax (the Sellers energy has no non-bonded repulsion — a collapsed pair would never be pushed apart).

### `random_seed_network_bm2000(N, box, d0, rng, ...)` — `lsu_network.py:442`
The Sellers-literally-cited random seed (refs [13,14]). Algorithm:
1. **Placement.** Poisson-disk with min-sep `0.98·d0` (BM2000 ratio 2.3/2.35 Å). Falls back to lower min-sep in 0.02 steps if it deadlocks.
2. **Hamiltonian cycle.** Greedy nearest-neighbour traversal under PBC, gives every vertex deg=2.
3. **Loop expansion to Z=3.** Pair deg-2 vertices: nearest valid partner within `rc`; grow `rc` if no progress. N=1000/L=11.44/d0=0.8 takes 0.2 s, rc grows to ~3.5 in the worst case.
4. **Force-pair fallback** for the last 0-4 stragglers (typical at N=1000): nearest pair regardless of `rc`, guarantees termination.

We deviated from BM2000's literal "swap-eq.2" move because BM2000 was for Z=4 (swap converts deg-2 to deg-4); for Z=3 the move would over-coordinate. Hamiltonian cycle + direct pairing is the natural Z=3 analogue and produces all the BM2000 invariants.

**Anti-cluster guarantees:** at N=1000/L=11.44, gives `min_non_bonded ≈ 0.8·d0` and `r_u ≈ 1.1·d0` post-relax — no cluster trouble.

## Refactored functions

### `relax(positions, ctx, max_iter, *, E_threshold=inf, c_f=0.5, cycle_size=None, on_global_promote=None)` — `lsu_network.py:1257`
**Return shape changed: now 3-tuple `(positions, E, info)`.** Every call site updated.

- `E_threshold=inf` → identical behaviour to legacy single L-BFGS pass. `info["early_rejected"]=False`.
- Finite `E_threshold` → chunked L-BFGS in cycles of `max(5, max_iter//25)` (so ~25 cycles per move per Hemmann § 2.1).
  - At each cycle boundary read `E, |F|² = ctx.value_and_grad(x)`; `E_est = E − c_f·|F|²` (BM2000 Eq. 4).
  - If `cycle > 5` AND `E_est > E_threshold` → abort, return with `info["early_rejected"]=True`.
  - If `cycle == 10` AND moving mask is set AND `|E − E_threshold| < 0.1` → drop mask via `ctx.set_moving_mask(None)`, fire `on_global_promote(positions)` callback. Threshold checks continue but on full-N gradient.

`info` dict carries `n_iter_done`, `force_norm_final`, `promoted_to_global`, `early_rejected`, `E_estimate_at_abort`.

### `www_anneal` — `lsu_network.py:1612`
New kwargs: `threshold_energy_relax=True`, `c_f=0.5`, `cycle_size=None`, `temperatures: Optional[ndarray]=None`, `log_tag="WWW"`.

Inner loop:
1. Snapshot `E_b = E_curr`, draw `s = rng.random()`, compute `E_t = E_b − T·ln(s)` (Vink Eq. 5).
2. Apply SW, refresh topology, set moving-shell mask.
3. `relax(..., E_threshold=E_t, c_f=c_f, on_global_promote=...)`.
4. If `info["early_rejected"]`: revert + `continue`. **No Metropolis roll** — Vink's identity says this is exact.
5. Else: compute acceptance objective (with `uniformity_weight` term), reuse the same `s` to decide accept/reject.

Acceptance rate is now reported alongside `early=...%` and `promote=...` in the verbose log.

**`temperatures` kwarg** overrides the geometric T schedule; pass an `(n_iterations,)` array for arbitrary profiles. Used by `topology_burn_in` to feed the triangular schedule.

**LSU check hoisted to top of loop** (commit after 2026-05-12 smoke run): previously the LSU check was *after* the move-handling block, so every `continue` (proposal failure, disconnect, early-rejection) skipped it. With early-reject rates ~90%, prints only fired ~10% of the time. The check now runs at the top of each iteration; `last_force_norm` is cached at the end of each iteration for the next-iter check to read.

### `topology_burn_in` — `lsu_network.py:2169`
**New triangular-profile signature** (no positional `n_moves`, `T` anymore — all kwargs). Old kwargs are deprecation aliases in `generate_lsu_network`, see below.

```python
topology_burn_in(positions, edges, neighbors, box, d0, weights, rng, *,
    n_heat=8_000, n_cool=16_000, n_quench=4_000,
    T_max=None, T_max_over_T_melt=1.15, P_melt=0.001,
    T_melt_probe_moves=600, T_melt_probe_T=5.0,
    threshold_energy_relax=True, c_f=0.5,
    relax_local_iters=100, relax_global_iters=500,
    local_shell_depth=4, global_fallback_threshold=inf,
    uniformity_weight=10.0, uniformity_kmax=2,
    target_accepts_per_vertex=None, ...)
```

Algorithm:
1. **Calibrate T_max** (if not user-supplied). Run `T_melt_probe_moves` SW moves at flat `probe_T`; collect mean uphill ΔE over **all proposed uphill moves** (not just accepted — important fix: the accepted subset is biased toward small ΔEs and would under-estimate T_melt by ~30%). `T_melt = <ΔE_up> / ln(1/P_melt)` (Hemmann Eq. 5). `T_max = T_max_over_T_melt * T_melt`.
2. Build `(n_heat + n_cool + n_quench,)` linear-then-linear-then-zero schedule.
3. Call `www_anneal` in chunks of ≥500 iters with `temperatures=` slice.

Returns `info = {T_max_used, T_melt, P_melt, n_heat, n_cool, n_quench, moves, accepted, proposed, early_rejected, cluster_after, probe_mean_uphill_dE}`.

### `generate_lsu_network` — `lsu_network.py:2901`
New kwargs (added; old ones kept as deprecation aliases):

| new kwarg | default | purpose |
|---|---|---|
| `seed_kind` | `"crystal_srs"` | `"crystal_srs"` (Hemmann) or `"random_bm2000"` (Sellers literally-cited) |
| `threshold_energy_relax` | `True` | Wire Vink/MB threshold abort into every relax |
| `c_f` | `0.5` | BM2000 Eq. 4 coefficient; 1.0 is permissive, 0.3 is strict |
| `cycle_size` | `None` | Auto = `max(5, relax_local_iters // 25)` |
| `bm2000_min_separation_frac` | `0.98` | Poisson-disk min separation for random seed |
| `bm2000_rc_start_frac` | `1.30` | Initial bonding cutoff for random seed |
| `bm2000_rc_grow_frac` | `0.05` | Cutoff growth step |
| `bm2000_rc_max_frac` | `6.00` | Cutoff cap; force-pair beyond this |
| `burn_in_n_heat` | `8_000` | Heat phase length |
| `burn_in_n_cool` | `16_000` | Cool phase length |
| `burn_in_n_quench` | `4_000` | Quench phase length |
| `burn_in_T_max` | `None` | Auto-calibrate via T_melt |
| `burn_in_T_max_over_T_melt` | `1.15` | Hemmann hyperuniform regime; raise to 2.0 for crystal_srs to escape srs basin |
| `burn_in_P_melt` | `0.001` | Hemmann Eq. 5 melting probability anchor |
| `burn_in_T_melt_probe_moves` | `600` | Calibration probe length |
| `burn_in_T_melt_probe_T` | `5.0` | Calibration probe temperature |
| `burn_in_target_accepts_per_vertex` | `None` | Optional accepted-moves cap |

Deprecation aliases (each emits `DeprecationWarning`):
- `topology_burn_in_moves` → split 1/5 heat, 3/5 cool, 1/5 quench
- `topology_burn_in_T` → `burn_in_T_max`
- `topology_burn_in_target_accepts_per_vertex` → `burn_in_target_accepts_per_vertex`

## Notebook (`Create_LSU_Function.ipynb`)

`full-run` cell rewritten with the new kwargs targeting Φ_22=0.89, N=1000, L=11.44, d0=0.8:
- `seed_kind='crystal_srs'`, `burn_in_T_max_over_T_melt=2.0`, `burn_in_n_heat=8000`, `burn_in_n_cool=16000`, `burn_in_n_quench=4000`
- `n_www_iterations=80_000`, `initial_temperature=1.0`, `final_temperature=1e-5`
- `threshold_energy_relax=True`, `c_f=0.5`, `relax_local_iters=120`, `local_shell_depth=4`
- `energy_weights={'alpha':20, 'beta':5, 'gamma':1, 'delta':0.5}` (kept from prior run)

The user requested raising `T_max_over_T_melt` to 2.0 explicitly because 1.15 (Hemmann's recommended hyperuniform regime) is too cold to escape the perfect srs basin — they tested it and got 0.5% acceptance with `T_max/T_melt=1.15` vs 8.9% at 2.0.

## Open issues / tuning frontier

These came up during smoke tests but are NOT bugs:

### Burn-in over-disorders for Type-2 targets
Φ_22 after burn-in at `T_max/T_melt=2.0` lands around 0.81 (with 2492 accepted moves, ~2.5 per vertex). For Φ_22=0.89 target this is too disordered — production WWW must climb back up to 0.89. Smoke runs show Φ_22 creeps +0.001 per 1000 iters at T~0.3, so the schedule needs ~80k iters to recover.

**Tuning recommendation if user revisits this:** for Type-2 (Φ_22 ≥ 0.85), use `T_max_over_T_melt=1.30` (top of Hemmann's hyperuniform regime) with a *longer cool phase* (`n_cool=20_000+`). The system should land at ~0.88 after burn-in, requiring only fine polishing in production WWW.

### Production WWW acceptance ~3% at moderate T
At `T0=0.4`, T cools through 0.4→0.18 over 7500 iters. Typical uphill ΔE ≈ 5; Metropolis acceptance = exp(−5/0.3) = 1e−7. The 3% accept rate is almost entirely downhill or near-isoenergetic moves. Early-rejection at 97% is correct behaviour, not a bug.

If user wants faster Φ_22 polishing: raise `c_f` to 1.0 (more permissive threshold → more moves reach Metropolis → more accepted moves at the cost of more compute per move) OR raise `T_final` from 1e-5 to ~0.01 so the schedule stays warm enough for SW moves throughout.

### Strict-Sellers path (no triangular burn-in)
The triangular profile is Hemmann's, not Sellers's. Sellers's literal recipe is "random network + simulated annealing" (supplement line 89), no separate burn-in. To run strict Sellers:
```python
seed_kind='random_bm2000',
burn_in_n_heat=0, burn_in_n_cool=0, burn_in_n_quench=0,
n_www_iterations=100_000,
initial_temperature=1.0, final_temperature=1e-5,
```
This bypasses `topology_burn_in` entirely.

### `min_non_bonded` drift during burn-in
End-of-burn-in `cluster_after` at `T_max/T_melt=2.0` shows `min_non_bonded ≈ 0.5·d0` with 1 `close<0.7·d0` pair — a stray vertex pair that drifted close because the Sellers energy has no non-bonded repulsion. Above the hard-fail threshold (0.4·d0) but worth watching. Cross-references [[lsu-known-issues]] bond-collapse mechanism (same root cause).

## Critical files

- [lsu_network.py](../lsu_network.py) — all functions updated; see line numbers above
- [Create_LSU_Function.ipynb](../Create_LSU_Function.ipynb) — `full-run` cell rewritten
- [we-are-going-to-twinkling-nest.md](/home/francisco/.claude/plans/we-are-going-to-twinkling-nest.md) — approved plan (executed)
- [Example/lsu_example_ends.txt](../Example/lsu_example_ends.txt) — N=1000/E=1500 reference

## What `cluster_diagnostics` looks like for a good run

Reference (`lsu_example_ends.txt`): `voxel_std_4 ≈ 3.65`, corner/centre ≈ 1.21.

Smoke run end-state at `T_max/T_melt=2.0`, 7000 burn-in + 2000 WWW: `Φ_22 = 0.8820`, `voxel_std_4=2.09`, `min_non_bonded=1.18·d0`, `r_u=1.32·d0`, `n_close_pairs=0`. Already in tolerance of 0.89 target with a fraction of the full schedule.

Full schedule (28k burn-in + 80k WWW) is currently the notebook default. Expect ~60-90 min wall-clock at N=1000 on CPU+JAX-JIT.
