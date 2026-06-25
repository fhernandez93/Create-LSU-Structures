# Independent metrics-verification review — energy diagnostic

Reviewer: independent recompute FROM SAVED FILES (no inline claim trusted).
Config: box=11.44, d0=0.8, N=1000, weights=(0.7,0.7,0.3,0.4).
Runtime check: `lsu._KEATING_F1F2 == True` (confirmed at import). Keating f1/f2 forms active.

All scripts import `tools`/`lsu_network` from the repo root; they failed with
`ModuleNotFoundError: No module named 'tools'` under `conda run` until I set
`PYTHONPATH=<repo root>`. Numbers below are with that fix; nothing else altered.

---

## CLAIM 1 — Energy diagnostic (the load-bearing claim): **CONFIRMED**

`Claude_Helpers/_energy_compare.py`, deep-relaxed 2000 iters, weighted Keating total:

| structure        | E_total | E/atom  | f1    | f2     | f3     | f4    |
|------------------|---------|---------|-------|--------|--------|-------|
| REFERENCE        | 34.484  | 0.03448 | 2.405 | 14.825 | 16.006 | 1.248 |
| refHold_ck50k    | 36.195  | 0.03619 | 2.512 | 15.601 | 16.810 | 1.271 |
| coldDis_ck50k    | 62.126  | 0.06213 | 4.553 | 26.565 | 28.290 | 2.718 |
| hyperuniform     | 55.456  | 0.05546 | 3.456 | 24.541 | 24.826 | 2.633 |

- REFERENCE E/atom = **0.03448** (claim ≈0.0345 — exact match).
- coldDis_ck50k E/atom = **0.06213** (claim ≈0.062 — exact match).
- **Ratio = 0.06213 / 0.03448 = 1.80x** (claim ~1.8x — exact). Reference is clearly
  the deeper basin.
- The gap is uniform across ALL FOUR components (f1 1.9x, f2 1.8x, f3 1.8x, f4 2.2x),
  not a single-term artifact. refHold (reference re-annealed and held) stays at 0.0362,
  near the reference — so the reference basin is *holdable*; the from-random plateau is not.

**Robustness (advisor-flagged convergence check):**
- coldDis E/atom is **identical at relax 2000 and 6000 iters (0.06213 = 0.06213)** —
  fully converged; the favorable ratio is NOT a under-relaxation artifact. Ratio is
  1.802x at both depths.
- Reference as-loaded E/atom 0.03494 → post-relax 0.03448 (only 1.3% drop): the
  reference is already sitting in its deep basin essentially as-saved. Directly supports
  "reference is a deep low-E basin."
- Asymmetry that makes claim 1 unfalsifiable-by-relaxation: more iterations can only push
  energies LOWER. coldDis is already converged, and any residual under-relaxation of the
  reference would only push ref lower and WIDEN the ratio. Under-relaxation cannot break
  claim 1 in either direction — it can only strengthen it.

**ctx.energy is the WEIGHTED Keating total (apples-to-apples across topologies): CONFIRMED from code.**
`_RelaxContext.energy` → `value_and_grad` → `_energy_jax_full`, which returns
`a*f1 + b*f2 + g*f3 + d*f4` (lsu_network.py:1782-1783). The NumPy fallback
(`total_energy`, lsu_network.py:1719-1720) returns the identical weighted sum. The
moving-mask is applied to the GRADIENT only (lsu_network.py:1801), never to the energy —
so `ctx.energy` always reports the full-N weighted total regardless of which vertices are
masked. Comparison across different topologies is therefore valid.

Verdict: the conclusion "reference is a deep low-E basin, reachable in principle; the
anneal under-anneals" is supported. The energy DOES favour the reference (lower by 1.8x);
the gap is kinetic, not an objective/energy mismatch.

---

## CLAIM 2 — Graph ring-counter validity: **CONFIRMED**

`Claude_Helpers/_graph_rings.py Example/lsu_example_ends.txt` (shortest-cycle-per-edge,
edge-list only, no rod round-trip):

```
N=1000 E=1500 girth=6 ring_mean=7.99
dist%: 6:7.6  7:10.0  8:59.73  9:20.93  10:1.73
```
Reproduces the reference distribution **6:7.6 7:10 8:59.7 9:20.9 mean 7.99** exactly.
The collision-proof 8r gate is validated.

mq1 8r trajectory (latest `Structures/*_mq1_ck*_edges.npy`):

| checkpoint | girth | ring_mean | 8r%   |
|------------|-------|-----------|-------|
| mq1_ck0    | 5     | 7.36      | 14.9  |
| mq1_ck10k  | 3     | 7.30      | 25.0  |
| mq1_ck20k  | 5     | 7.47      | 30.0  |

8r is **climbing monotonically 14.9 → 25.0 → 30.0** from the ~15 seed. Confirmed climbing.
(Caveat, not part of the claim: ck10k shows girth=3 i.e. a transient triangle — a small
fraction of sub-5 rings appears mid-run before being healed by ck20k. 8r at 30% is still
far from the reference 59.7%, consistent with the under-anneal narrative.)

---

## CLAIM 3 — Relaxation-bias refutation: **CONFIRMED**

`Claude_Helpers/_relax_bias_test.py` on coldDis_ck50k (60 valid SW moves, shell=4,
local=100, full=1500):

```
baseline trapped E/atom=0.06213 (ref 0.0345)
dE_local : mean +0.6240  min -0.0581  (looks-uphill: 59/60)
dE_full  : mean +0.5750  min -0.1002  (truly-downhill: 1/60)
full reveals MORE downhill than local (df<dl): 59/60
*** WRONGLY-LOOKING moves (dE_local>0 but dE_full<0): 0/60 ***
mean gap (dE_local - dE_full): +0.0490  max gap +0.1281
VERDICT: relaxation bias WEAK
```

- **0/60** moves are wrongly rejected (local-uphill but full-downhill). The local depth-4
  masked relax is NOT hiding good moves.
- Only 1/60 moves is genuinely downhill even under full relax — the trapped plateau has
  very few accessible improving SW moves. Mean local↔full gap is small (+0.049).
- Verdict matches the claim: the trap is a search/barrier problem (needs hotter/slower
  schedule or more moves), not an under-relaxation artifact.

---

## Summary

| # | claim | result |
|---|-------|--------|
| 1 | ref E/atom ≈0.0345, ~1.8x below coldDis ≈0.062; ctx.energy = weighted Keating total | **CONFIRMED** — 0.03448 vs 0.06213, ratio 1.802x, converged, weighted-total verified from code |
| 2 | graph counter reproduces ref (8r 59.7), mq1 8r climbing from seed | **CONFIRMED** — exact ref match; 8r 14.9→25.0→30.0 |
| 3 | local depth-4 relax ≈ full-N relax; anneal not hiding good moves | **CONFIRMED** — 0/60 wrongly-rejected; verdict bias WEAK |

No load-bearing number is materially off. All three claims hold with the reported numbers.
