# Sellers-Fidelity Review — N=512 random_bm2000 winning config

Reviewer: sellers-fidelity-review agent. Date: 2026-06-18.

Config under review (validation nominee r10):
`uniformity_weight=30, uniformity_kmax=2, lsu_degree_22=0.889 (target),
initial_temperature=0.045, final_temperature=0.015, n_www_iterations=40000,
local_shell_depth=4 (local-only relax), burn_in=OFF, weights α=0.7 β=0.7 γ=0.3 δ=0.4,
seed_kind=random_bm2000.`

Primary sources (all in `LSU Literature/`):
- Sellers et al., Nat. Commun. 8, 14439 (2017) — `ncomms14439.pdf` + ESM `41467_2017_BFncomms14439_MOESM1815_ESM.pdf`
- Barkema & Mousseau, Phys. Rev. B 62, 4985 (2000) — `PhysRevB.62.4985.pdf`
- Vink & Barkema, Phys. Rev. B 64, 245214 (2001) — `PhysRevB.64.245214.pdf`

The authoritative algorithm description is the **Sellers ESM "Amorphous Gyroid
Simulated Annealing"** section (txt lines 384–455). The main-text Methods only
covers diffraction/PBG modelling, not the WWW schedule.

---

## 1. uniformity_weight=30 (low-k S(k) penalty in the Metropolis objective)

**VERDICT: JUSTIFIED-CRUTCH (correctly characterized as non-Sellers; load-bearing
at the tested 40k budget; "required" is untested at Sellers' 100k budget).**

Primary source — Sellers ESM defines the *entire* accepted-move objective as a
four-term bonded energy:

> "U = αf1({d}) + βf2({θ}) + γf3({φ}) + δf4({χ})" — ESM Eq. (2), line 401–402,
> where f1/f2 are Keating edge-length and 120° bond-angle terms, f3 favours the
> gyroid dihedral arcos(±1/3), f4 favours trihedral coplanarity (lines 419–423).

Acceptance is strict Metropolis on this U: "Pa = e^{(E0−Ef)/T}" (ESM Eq. 1, line
392). **No S(k), no structure-factor, no hyperuniformity, no long-wavelength
density term appears anywhere in the accepted objective.** The only place near-
hyperuniformity is mentioned is as an *emergent* outcome, not a driver:

> "we note the emergence of a hyperuniform-like exclusion domain around k=0 in
> the structure factor of networks with significant LSU" — main text, line 1038–1041.
> (Hejna/Steinhardt/Torquato is cited in the ESM, ref. line 591, for the same
> "nearly hyperuniform emerges from WWW" picture.)

The code itself documents this is off-Sellers: `_acceptance_objective`
(`lsu_network.py:1451`) adds `uniformity_weight * low_k_structure_factor(...)`
straight onto the strain energy that feeds the Metropolis roll
(`lsu_network.py:2408–2411`), and the `www_anneal` docstring states it should be
"0.0 for strict Sellers Eq. 2 acceptance" (`lsu_network.py:2198–2201`). So
**calling weight>0 a "non-Sellers crutch" is CORRECT** — it is provably absent
from Eq. 2; that part is airtight.

Whether it is *necessary* is the more careful claim. The validation's emergence-
refutation (run r11) was at **40k** iterations, where w=0 coarsens (6-rings
5.5%→0.0%, void S_low stays 0.41, α=−1.06) instead of emerging. But Sellers'
emergence is stated explicitly at **~100,000 iterations** (ESM lines 410 and
455). You cannot refute emergence at 40k when Sellers' own budget is 100k — the
failure of emergence here is **confounded with deviation #5 (iteration count)**.
Defensible statement: the penalty is load-bearing *at the 40k budget actually run*
in this local-relax code; "genuinely required" remains untested at 100k. This
matches the standing memory note ("emergence claim is ~100k — caveat stands").
Cross-reference deviation #5.

---

## 2. Relaxation locality: local_shell_depth=4 (local-only per-move relax)

**VERDICT: FAITHFUL. The prompt's framing ("vs Sellers/Vink full-network
relaxation") is BACKWARDS — local relaxation IS the Vink/BM2000 method, and
local_shell_depth=4 matches Vink's 4th-neighbour-shell cluster exactly.**

This is the sharpest finding and it corrects the task premise. Vink & Barkema's
entire contribution is making relaxation *local*, not global:

> "A local-relaxation procedure is used whenever possible. Immediately after a
> bond transposition, only a small cluster of atoms... experiences a significant
> force. This cluster consists of the atoms directly involved... and of nearby
> atoms, typically **up to the fourth neighbor shell** of the four transposition
> atoms. The number of atoms in such a cluster is about 80. It, therefore,
> suffices to calculate the force locally... rather than globally." — Vink lines
> 170–180.

> "The four atoms... and all atoms up to the **fourth neighbor shells**... are
> grouped into a cluster. These atoms, about 80, are allowed to move." — Vink
> lines 285–291.

BM2000 use the third shell for the same purpose: "we relax only locally in the
first ten relaxation steps after a bond transposition (up to the third neighbor
shell)" (BM2000 lines 173–177). Sellers explicitly inherits these: the ESM says
they followed WWW "incorporating many of the refinements subsequently developed
for modelling large amorphous silicon networks13,14" (lines 384–385) — refs 13/14
are BM2000 and Vink. So `local_shell_depth=4` is the canonical Vink cluster, not
a shortcut.

**Important nuance — the accept/reject still uses GLOBAL energy, and the code is
faithful here too.** Vink requires that, after local relaxation, the *total*
Keating energy be evaluated for the final accept/reject (an O(N) op): "to make
the final accept/reject decision on the proposed move, the total Keating energy
of the system has to be calculated... a switch must be made from local to global
relaxation, usually after about ten local relaxation steps" (Vink lines 186–191).
The code does exactly this: the moving mask zeroes only the *gradient* of frozen
vertices (`_RelaxContext.set_moving_mask`, `lsu_network.py:1742`), while
`value_and_grad`/`energy` always return `total_energy` summed over ALL
edges/triples/quads (`lsu_network.py:1792–1811, 1598–1611`). The Metropolis dE at
`lsu_network.py:2411` is therefore a *global* energy difference. It also
implements Vink's local→global rescue (`relax` docstring lines 1934–1943;
`lsu_network.py:2017`, fallback at 2398–2403).

The validation's empirical result reinforces this: full-N relax during the chain
(run r8, `local_shell_depth=None`) did NOT improve bond-std (0.112, same as
local), helped void only moderately, hurt 6-rings, and cost 2.4× runtime. So the
local choice is both faithful AND empirically the right call.

---

## 3. lsu_degree_22=0.889 used as early-stop/objective vs fixed ~100k iterations

**VERDICT: SPLIT — the VALUE is FAITHFUL; the early-stop MECHANISM deviates but
was INERT in the nominee.**

*As a target value:* 0.889 matches the Sellers type-2 LSU. The main-text Methods
(Ps. marsyas reflectance modelling) state: "Models... were generated using
type-2 amorphous gyroid networks with F22's around 0.88" (line ~1657), and the
body text gives type-2 "F22's around 0.88" / type-1 "F22 values around 0.72"
(lines 593–595). So aiming at Φ22≈0.889 is descriptively faithful to the
published target window — FAITHFUL.

*As an early-stop criterion replacing fixed iterations:* Sellers ran a **fixed
~100,000 WWW iterations** ("after around 100,000 WWW iterations", ESM lines 410,
455) — there is no Φ22-triggered early termination in the paper. Using Φ22=0.889
to stop early therefore deviates from the protocol. HOWEVER, the validation
explicitly set `check_lsu_every=0` (full-length, "no early-exit confound" — see
`N500_validation_results.md` line 29), so in the nominee run r10 the early-stop
was **inert** — the target functioned only as a comparison gauge, not a control
knob. Judge the mechanism: deviates in principle, but did not act in the nominee.
(Additionally moot here because Φ plateaued ~0.869 < 0.889 — it never reached the
stop value regardless.)

---

## 4. burn-in OFF for the random seed

**VERDICT: FAITHFUL. No caveat.**

Burn-in (`topology_burn_in`) exists to destroy crystalline Bragg memory in an
ordered seed (`lsu_network.py:117–118`). The random_bm2000 seed has no crystalline
order and no Bragg peaks to melt, so there is nothing for burn-in to do. This
matches Sellers exactly: the ESM protocol starts from a **random seed** and applies
**pure WWW simulated annealing** — "The basic process involves the simulated
annealing of a random network" (line 386); "amorphous gyroid networks were
successfully generated from random seed patterns" (line 455). No melt/burn-in
pre-stage is described. Turning burn-in OFF for a random seed is the faithful choice.

---

## 5. n_www_iterations=40000 vs Sellers' ~100,000

**VERDICT: UNDER-EQUILIBRATION SHORTFALL (does not fit FAITHFUL / CRUTCH /
MISCHARACTERIZED cleanly — it is an honestly-flagged budget gap).**

Sellers state the iteration count twice and unambiguously: "producing only type-1
networks after ∼100,000 WWW iterations" (ESM line 410) and "High-quality
amorphous gyroid networks were successfully generated from random seed patterns
after around **100,000 WWW iterations**" (ESM line 455). 40k is 0.4× that.

This is not "faithful" (it is 2.5× short of the stated budget) and not a "crutch"
(it removes capability rather than adding a non-Sellers prop) — it is a deliberate
under-equilibration imposed by the validation's 4-hour / 1800s-per-run compute
budget. The validation flags this honestly and ties multiple residual failures to
it: Φ22 plateaus at 0.869 < 0.889, bond-std floors at ~0.10–0.12 vs reference
0.029, and 6-rings sit below the 5% gate — all consistent with a network that has
not reached the ~100k-iteration equilibrium. The notes explicitly say "bond std
<0.045 likely needs ~100k iters (Sellers)" (`N500_validation_results.md` lines 88,
118).

**Coupling to #1:** because 40k ≠ 100k, the "emergence REFUTED / penalty required"
conclusion (run r11, w=0 @ 40k) cannot claim to test Sellers' actual emergence
budget. The penalty is shown necessary *at 40k*; whether pure WWW would emerge to
near-hyperuniformity at 100k in this local-relax code is untested. The honest
verdict on #1 must carry this caveat, and #5 is its root.

---

## Summary table

| # | Deviation | Verdict | Key primary source |
|---|---|---|---|
| 1 | uniformity_weight=30 (S(k) penalty in Metropolis) | JUSTIFIED-CRUTCH (non-Sellers, load-bearing @40k; "required" untested @100k) | ESM Eq. 2 line 401–402 (no S(k) term); main text 1038–41 (emergent only) |
| 2 | local_shell_depth=4 (local-only relax) | FAITHFUL (prompt premise is backwards) | Vink 170–191, 285–291; BM2000 173–177; global energy for accept/reject confirmed in code |
| 3 | Φ22=0.889 target/early-stop vs fixed 100k | VALUE faithful; early-stop deviates but INERT in nominee | Main text ~1657, 593–595 (type-2 F22≈0.88); ESM 410/455 (fixed 100k) |
| 4 | burn-in OFF (random seed) | FAITHFUL | ESM 386, 455 (random seed + pure WWW, no melt) |
| 5 | 40k iterations vs ~100k | UNDER-EQUILIBRATION SHORTFALL (budget gap, honestly flagged) | ESM 410, 455 (~100,000 iterations) |
