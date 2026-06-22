# WWW Sellers-fidelity audit (N=1000 investigation) — my pass

Audited against `lsu_network.py` (this session) + the Sellers supplement facts verified in
prior sessions ([[lsu-sellers-protocol-random-pure-www]], [[lsu-www-faithfulness-audit]],
[[sellers-confirmed-energy-weights]]). Sellers et al. Nat. Commun. 8, 14439 (2017) Methods +
Supp Eq. 1–2; refs [13] Vink 2001, [14] Mousseau-Barkema 2001 (BM2000). Verdict vocabulary:
**faithful** / **extension** (adds non-Sellers machinery) / **shortfall** (deviates from Sellers).
(The independent Sellers-fidelity AGENT will re-verify these against the PDFs in `LSU Literature/`.)

| # | Item | Code | Verdict | Justification |
|---|---|---|---|---|
| 1 | Stone-Wales bond move | `stone_wales_propose` 2044, `_apply` 2101, `_revert` 2119 | **faithful** | Canonical transposition `(i,c),(j,d)→(i,d),(j,c)`; guards reject duplicate/degenerate edges & disconnection; rows written in canonical (min,max) order so `_revert` is an exact array-level inverse. No spurious extra moves in the anneal loop (one propose per iter, 2338). |
| 2 | Metropolis + Vink threshold | `www_anneal` 2344-2428 | **faithful** | `E_t = E_b − T·ln(s)` (Vink 2001 Eq.5, line 2348). Early-reject when relax can't drop below `E_t` ⇒ revert with **no re-draw** (Vink identity: aborting on the same `s` ≡ Metropolis rejection, 2384-2391). Final accept `dE≤0 or s<exp(−dE/T)` reuses the same `s` (2420). For w=0 this is exact Metropolis; the threshold is in strain-energy units only (penalty excluded). |
| 3 | Temperature schedule | `www_anneal` 2278-2281 | **faithful** | Geometric `T = T0·exp(ln(T_final/T0)·it/(n−1))`. Optional `temperatures=` override (used by my probe to share one schedule across chunks) — does not alter the per-step Metropolis. |
| 4 | Energy f1–f4 + weights | `energy_components` 1541, `total_energy` 1598 | **faithful** | f1 `Σ(L−d0)²` bonds; f2 `Σ(cosθ+½)²` angles→120°; f3 `Σ(|cos dih|−⅓)²` dihedral; f4 `Σ(skew_i²+skew_j²)`. `U = α f1+β f2+γ f3+δ f4`, weights (0.7,0.7,0.3,0.4) — Sellers-group-confirmed, never changed. Raw sum (not per-bond normalized); Metropolis uses ΔE so scale is consistent. JAX path (`_energy_jax_full` 1629) matches the NumPy path term-by-term. |
| 5 | Local-shell relax (depth 4) | `compute_local_shell_mask` 1467, `relax` 1897 | **faithful** | Depth-4 graph-hop shell around the SW seed {i,c,j,d}; out-of-shell vertices held fixed by gradient masking. Chunked L-BFGS; BM2000 Eq.4 estimator `E_f_est=E−c_f|F|²`; early-rejection only in cycles (5,10] (BM2000 no-reject first 5; Hemmann §2.1 none after 10); Vink §IV.B local→global rescue at cycle 10 within `promote_margin` above threshold. `|F|²` over moving DOFs only. Canonical Vink/MB refinement, NOT a shortcut (N=512 fidelity review + full-N gave no benefit). |
| 6 | Uniformity penalty | `_acceptance_objective` 1451, applied 2408-2411 | **extension** | `objective = E + w·S_low(kmax)` added ONLY to the Metropolis objective — NOT inside the L-BFGS strain relax and NOT in the Vink threshold `E_t` (local geometry stays pure Sellers Eq.2). Non-Sellers (absent from Eq.2). With w>0 the `s`-reuse on the final accept becomes a documented approximation (threshold speedup remains strain-only). Set w=0 for strict Sellers. |
| 7 | Seed `random_bm2000` | `random_seed_network_bm2000` 474; settle 1151 | **faithful (rough seed)** | Hamiltonian-cycle scaffold + chord-matching to Z=3 with BM2000 §II.A min-separation. Drops BM2000's BC→AB+AC loop-expansion move (ring shaping left to the anneal). This session's seed42 N=1000: perfect 3-regular, **girth 5**, rich ring tail (6-rings 16.8%). `settle_seed_with_repulsion` is an extension (soft-sphere conditioner for the long-bond tail; WWW still sees pure Eq.2). |

## Verdict summary
The WWW move set, Metropolis acceptance, Vink threshold identity, temperature schedule, energy
functional, and local-shell relaxation are all **faithful** to Sellers/Vink/MB. The only
deviations are the two documented **extensions**: the `uniformity_weight` low-k penalty (Metropolis
objective only) and the `settle_seed_with_repulsion` seed conditioner — neither touches the per-move
strain energy that selects topology. → A wrong equilibrium ring distribution, if found, is NOT
attributable to an energy/move fidelity bug; it would point to schedule/temperature or the penalty's
ring-narrowing side-effect.

## Best-achievable-config deviation characterization (no config passed all gates)
There is no "winning" config — A1 and S1 are two Pareto endpoints of one tradeoff, neither passing
the 6-ring gate. Their deviations from strict Sellers Eq. 1–2 + pure WWW:
- **A1** (`w=30, kmax=2`, cold geometric 0.045→0.015, 30k): deviation = the `uniformity_weight`
  low-k penalty (EXTENSION, item 6) + the `settle_seed_with_repulsion` conditioner (EXTENSION,
  item 7). Schedule = standard geometric (faithful). This is the max-6r endpoint.
- **S1** (`w=35, kmax=2`, SUSTAINED-moderate-T via `temperatures=` override, 50k): same two
  extensions PLUS a **custom non-geometric temperature profile** (cool→hold 0.025→cool). The
  sustained hold is an ENGINEERING schedule choice — still within the WWW/Metropolis framework
  (each step is faithful Metropolis at the prescribed T), but NOT the simple geometric anneal of
  the Sellers Methods. Classify: **extension (schedule engineering), not a fidelity violation** —
  it changes only the T(it) profile fed to faithful acceptance.
- **The decisive negative does NOT rest on any extension.** It rests on the w=0 (penalty-free,
  strictly-Sellers-Eq.2) 100k probe, which gives 6r ~3% (< gate) and α<0 (no emergence). So the
  failure is intrinsic to (mandated random_bm2000 seed + faithful Sellers energy + tested schedules),
  not introduced by the penalty or the sustained schedule.
- **Open fidelity question (honest):** faithful move+energy+Metropolis+relax, yet pure WWW voids
  (α<0) where Sellers reports emergent near-hyperuniformity. Unfalsified differences vs Sellers:
  (a) seed kind (mandated random_bm2000 Hamiltonian-scaffold vs Sellers' Poisson/config-model),
  (b) Sellers' exact (unpublished) anneal schedule, (c) system size (Sellers' 100k may be at
  N≫1000 = few moves/vertex). The local-shell relax is faithful (Vink/MB) but is BLIND to box-scale
  voids (locally strain-free) — a plausible mechanism for why this implementation voids where a
  global-relax or larger-N run might not. Stated scope: **unreachable within the mandated seed +
  tested schedules**, NOT "the protocol is universally incapable."

## Independent fidelity-agent verification (PDF-grounded)

Re-verified each item against `lsu_network.py` AND the primary PDFs in `LSU Literature/`. All seven verdicts stand.

1. **SW move — AGREE.** Code `(i,c),(j,d)→(i,d),(j,c)` (2056/2111) keeps central bond i–j, swaps outer
   partners c,d → the WWW 1985 "local rearrangement of bonds" (PRL 54, 1392, Fig.1: "tetrahedral
   bonding preserved … five- and sevenfold rings introduced"); = BM2000's ABCD/AB,BC,CD switch (62,4985 §II).
2. **Metropolis+threshold — AGREE (citation sharpened).** The literal `E_t = E_b − T·ln(s)`, s∈[0,1] (code 2348)
   is **BM2000 Eq.3** (audit's "Vink Eq.5" = same with `ln(1−r)`, distributionally identical), which states it is
   *"exactly equivalent to the usual Metropolis procedure"* and licenses rejecting once the threshold is shown
   unreachable — validating the NO-re-draw early-reject identity (2384–2391). Final accept reuses the same `s`
   (2420) ⇒ pure Metropolis at w=0. **CONFIRMED valid per BM2000/Vink.**
3. **Geometric T — AGREE.** Code 2279 geometric; Sellers Supp only says *"gradually reducing the temperature"*
   (no form mandated); WWW 1985 uses staged kT 0.40→0.25 eV. Geometric + optional `temperatures=` override are
   faithful instantiations; the override does not touch per-step acceptance.
4. **Energy f1–f4 — AGREE (identical to Supp Eq.2).** Code f1 Σ(L−d0)², f2 Σ(cosθ+½)², f3 Σ(|n̂_i·n̂_j|−⅓)²
   (1589, `_DIH_TARGET=1/3`), f4 Σ(r̂_ij·n̂_i)²+(r̂_ij·n̂_j)² (1591–1593) = **Supp Eq.2/3/4 term-for-term**
   (f4 "three edges about a vertex coplanar"); weights (0.7,0.7,0.3,0.4) Sellers-confirmed. JAX path matches.
5. **Local-shell relax depth 4 — AGREE.** Vink (PRB 64,245214) §II: cluster = **fourth neighbor shell**, local→
   global switch *"when … within 0.1 eV of the threshold"* = code `promote_margin=0.1` at cycle 10 (1934–1943).
   BM2000 Eq.4 `E_f≈E−c_f|F|²` (62,4985) = code `E_est=E−c_f·F_sq` (1983); no-reject-first-5 = `cycle_skip=5`.
6. **Uniformity penalty — AGREE (EXTENSION).** `_acceptance_objective` adds `w·S_low` ONLY to the Metropolis
   objective (2408–2411); `E_t` (2348) and `relax` (`E_threshold=E_t`, 2377–2382) see strain energy only.
   **Absent from Supp Eq.2** — confirmed non-Sellers, isolated from threshold and L-BFGS. `w≤0` early-returns (1459).
7. **Seed `random_bm2000` — AGREE (rough seed).** Docstring (489–508) is a *Z=3 adaptation* of BM2000 §II.A
   (Poisson placement + Hamiltonian cycle + girth≥5 expansion); it is NOT BM2000's own 4-fold diamond-randomized
   seed, and Sellers only says *"random seed patterns"* (Supp §Methods) — neither is provably Sellers' seed.

**Special-attention confirmations:** Supp lines 100–112 state amorphous gyroid *"generated from random seed
patterns after around 100,000 WWW iterations"* with **no** structure-factor/low-k term — the only energy is Eq.2.
Near-hyperuniformity is emergent (Hejna ref [9], lines 61/147: CRNs *"not strictly hyperuniform"*), never a
penalty ⇒ the item-6 penalty is genuinely non-Sellers.

**(a) Decisive negative — verdict.** Conditional on the reported w=0 statistics (6r~3%, α<0; not re-run, no GPU),
the failure is attributable to the **faithful core + mandated seed, NOT any extension**: the w=0 path is provably
extension-free (penalty off at 1459; pure Metropolis; faithful energy/move/relax). Its only global relaxation is
Vink's in-relax local→global promotion (defaults `relax_global_every=0`, `global_fallback_threshold=inf`), so the
relax's blindness to box-scale voids is **inherited from faithful Vink** (same 0.1-eV switch), not a fidelity bug.

**(b) Key caveat.** The decisive confound is the **seed**: `random_bm2000` is a Hamiltonian-scaffold Z=3 adaptation
with imposed girth≥5, whereas Sellers' actual "random seed pattern" is unspecified. A wrong seed ring-spectrum is
not falsifiable by code+PDF and could alone explain non-emergence — so the negative is scoped to *this* seed, not
to faithful WWW in general.
