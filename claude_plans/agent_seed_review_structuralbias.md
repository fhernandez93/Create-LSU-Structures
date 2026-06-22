# Seed-origin structural-bias review: does `random_seed_network_bm2000` cause the coarsening and the void?

Read-only code + literature analysis. No code modified, nothing run on GPU.

**Scope:** judge whether the random Z=3 seed generator
(`lsu_network.py:474` `random_seed_network_bm2000`, `lsu_network.py:1331`
`_poisson_disk_pbc`, `lsu_network.py:1151` `settle_seed_with_repulsion`)
has a structural bias or shortcut that CAUSES the two measured downstream
defects:

- **COARSENING** — annealed ring mean ~8.3–8.4 vs reference 7.99; 8-ring
  fraction caps ~45–53% (ref 59.7%); excess 9-rings ~28–34% (ref 20.9%); no
  sharp 8-ring peak.
- **VOID / not hyperuniform** — annealed S(k0) ~0.07–0.16 (ref 0.041);
  hyperuniformity slope α ≈ 0 or negative (ref +1.51); a box-scale density
  void keeps forming.

**Reference (gold standard):** `Example/lsu_example_ends.txt` — Sellers's OWN
published network, made from a DIFFERENT random seed. Verified stats
(memory `lsu-reference-network-stats`): rings {6:114, 7:150, 8:896, 9:314,
10:26} over 1500 → 6:7.6%, 7:10.0%, **8:59.7%**, 9:20.9%, 10:1.7%; ring mean
7.99; rod-midpoint 4³ voxel std = 3.65.

**The seed itself (measured this session, N=1000, raw/pre-settle):** perfectly
3-regular, girth 5, ring mean **7.36**, distribution heavy in BOTH small and
large rings (5:21.5% 6:16.8% 7:17.1% 8:14.9% 9:15.5% 10:9.1%),
**S(k0)=0.141**, bond mean 1.10·d0, std 0.24, max 2.19·d0.

---

## The two literature facts that drive the whole analysis

These are quoted from the PDFs the code cites and are load-bearing for every
verdict below.

**FACT 1 — the faithful seed uses PURE-RANDOM placement (no min-separation
beyond a hard contact), NOT Poisson-disk.**
BM2000 §II.A (`LSU Literature/PhysRevB.62.4985.pdf`, p.4986):
"we randomly place all the atoms in a cubic box with periodic boundary
conditions, under the constraint that **no two atoms are closer than 2.3 Å**."
That single contact rule (2.3 / 2.35 ≈ 0.979·d0) is a *hard-core overlap
exclusion*, not a Poisson-disk uniformizing rule — it only forbids near-
coincidences and otherwise leaves the field Poissonian. Vink 2001 (the paper
Sellers cites for the random-seed route, `LSU Literature/PhysRevB.64.245214.pdf`,
p.2, summary of BM2000 point 1) is even blunter: "Starting point for the
relaxation in this case is a **truly random configuration whereby the atoms are
placed at random locations in a periodic box at the crystalline density**. This
guarantees that the resulting network is not contaminated by some memory of the
crystalline state." BM2000 p.4987 reinforces the intent: "this method leads to
highly strained initial configurations ... it has the advantage that it
contains absolutely no trace of crystallinity."

**FACT 2 — WWW relaxation is LOCAL, so it does not move long-wavelength
density.** BM2000 §II.B point 3 (p.4986): "We use a local/nonlocal relaxation
procedure ... we relax only locally in the first ten relaxation steps after a
bond transposition (**up to the third neighbor shell**)." Vink 2001 §III.B
(p.2–3): "only atoms inside the **fourth neighbor shell** allowed to move ...
the atoms in the fifth neighbor shell ... are not allowed to move." Sellers
inherits this ("incorporating many of the refinements ... for ... amorphous
silicon networks," supplement Supplementary Methods p.6, citing Vink [13] and
Mousseau-Barkema [14]). The WWW MOVE itself is a local bond transposition
(BM2000 Fig. 1). A relaxation/anneal made of local moves + a local Keating/
LSU energy has **no term acting on the lowest reciprocal modes** — the code
even acknowledges this in `low_k_structure_factor` (`lsu_network.py:1424`):
"the bonded Sellers / Keating-like energy ... has no term suppressing
long-wavelength density fluctuations." Consequence: **the anneal has no
restoring force toward box-scale homogeneity — it neither systematically
removes the seed's void nor reliably preserves the seed value; it scatters
S(k0) around the seed without driving it toward zero.** (The measured evidence:
seed S(k0)=0.141, annealed S(k0) lands anywhere in ~0.07–0.16 — *both above and
below* the seed — and never approaches the reference 0.041. The 2× best-case
drop and the values above 0.141 show the low-k mode is perturbed in both
directions, not frozen; but there is no drive toward 0.041.)

---

## Per-cause analysis

### (a) Poisson-disk placement at min_sep 0.98·d0 creates clumps+gaps → a long-wavelength void the local relax can't remove. — UNLIKELY (direction is backwards)

The candidate as stated is the opposite of what the code does. A
minimum-separation rejection rule (`_poisson_disk_pbc`, `lsu_network.py:1351`,
accept candidate only if `dist >= min_dist`, where `min_dist =
0.98·d0`) **suppresses** density fluctuations relative to pure random: it
forbids two points within 0.98·d0, which removes the short-range clumping that a
Poisson field has. A Poisson-disk (a.k.a. RSA / hard-sphere) field has
S(k)<1 at all k and is *more* homogeneous than Poisson (whose S(k)≈1
everywhere). So Poisson-disk does not *manufacture* clumps-and-gaps; it damps
them.

Critically, the faithful BM2000/Vink seed uses *pure random* placement
(FACT 1). Pure random placement is *less* homogeneous (higher S(k0)) than the
code's Poisson-disk placement. So if anything the code's placement is **more
faithful-than-faithful on homogeneity** — it cannot be the origin of a void
that the faithful (less-homogeneous) construction supposedly does not produce.
The void must therefore come from a stage *after* placement, or from a property
that Poisson-disk placement does NOT control.

That last clause is the real point: a 0.98·d0 hard-core rule controls the
*near-contact* (high-k, first-shell) structure but does essentially nothing to
the *box-scale* (k0) modes — RSA at this modest packing fraction is close to
Poissonian at small k. So the placement is neither the villain (it does not add
clumps) nor the hero (it does not deliver the box-scale uniformity the
reference has). The seed's S(k0)=0.141 (high — Poisson-like at low k) is
consistent with this: placement gave good local spacing and left the
long-wavelength density essentially random. **Verdict: (a) is not the cause;
the placement rule is the wrong knob for box-scale density.**

### (b) Greedy nearest-neighbour Hamiltonian cycle → long-range correlations / large loops biasing toward coarsening. — UNLIKELY as a *coarsening* cause; mildly plausible as a *void* contributor

Two separate sub-claims.

*Coarsening sub-claim — not supported by the seed's own ring stats.* The seed
ring mean is **7.36, BELOW the reference 7.99**, and the seed distribution is
*broad* (5-rings 21.5% all the way to 10-rings 9.1%), not *shifted-large*. A
topology biased "toward large rings / coarsening" would show a seed ring mean
*above* reference and a depleted small-ring tail; the seed shows the opposite
(small-ring-RICH). So whatever the greedy cycle does, it does not pre-bias the
seed toward the large-ring/coarse end. The coarsening appears during the anneal,
not in the seed topology. (This also matches the prior finding in memory
`lsu-random-bm2000-seed-faithfulness` and three prior agent reviews that the
Hamiltonian-cycle scaffold is acceptable and that the *girth rule*, not the
scaffold, was the load-bearing fix. I am not reopening that.)

*Void sub-claim — mildly plausible but not the primary mechanism.* A greedy
nearest-neighbour Hamiltonian cycle (`lsu_network.py:638`–674) is a
space-filling-like traversal: it always hops to the nearest unvisited vertex.
Such traversals are known to leave a few long "return" chords (the cycle has to
eventually jump back across the box; the closing edge `lsu_network.py:669`–674
and the rc-grown / force-paired chords `lsu_network.py:733`–860 are exactly
these). Those long bonds are then mostly removed by the 2-opt pass
(`lsu_network.py:937`–953) and by `soft_start_seed_relax` /
`settle_seed_with_repulsion`. The residual effect on box-scale density is
second-order compared with the fact that *neither the greedy cycle nor pure
random gives a hyperuniform vertex field*. The cycle is a topology builder, not
a density equilibrator; it inherits the placement's S(k0) and does not improve
it. **Verdict: not the coarsening cause; at most a minor void contributor that
the cleanup stages already target.**

### (c) Nearest-partner loop expansion biases the ring spectrum toward large rings or toward inhomogeneity. — UNLIKELY (and note: the code's mechanism is NOT BM2000's loop expansion)

First, a faithfulness note that matters for interpreting this candidate. The
code does **not** implement BM2000's loop-expansion MOVE. BM2000 §II.A
(p.4986, Fig. 2) expands a single growing loop by *insertion*: pick three atoms
A,B,C with B–C bonded, A within rc of both but bonded to neither, then "the
bond BC is then replaced by bonds AB and AC, expanding the loop by one step."
The code instead builds a full Hamiltonian cycle and then raises every deg-2
vertex to deg-3 by **global ascending-distance greedy matching of the deg-2
vertices** (`lsu_network.py:690`–728: collect all pairs within rc, sort by PBC
distance, accept the shortest valid (deg<3, not bonded, girth≥5) pair first).
So this candidate is really about the *short-chord matching*, not BM's move.

Does shortest-chord matching bias toward large rings? It does the opposite of
what a large-ring bias would need. Matching the *nearest* available deg-2
partners adds *short* extra bonds, which tend to close *small* rings (a short
chord across the Hamiltonian path closes a short cycle). That is consistent with
the seed being small-ring-RICH (5-rings 21.5%), i.e., the matching pushes the
seed *small*, not large. The only large-ring/inhomogeneity contribution is the
straggler force-pair fallback (`lsu_network.py:820`–860, up to √3·L/2) and
rc-grown chords, which are few (typically 0–4 stragglers per the docstring) and
are cleaned by 2-opt. **Verdict: not a large-ring bias; if anything a
small-ring bias, which the anneal then has to coarsen away.**

### (d) The seed is simply too inhomogeneous/coarse to begin with, vs what BM2000/Sellers would produce. — SPLIT: the "coarse" half is FALSE; the "inhomogeneous (high S(k0))" half is TRUE and is the void's origin

Split this into the two metrics, because they point in opposite directions.

*"Too coarse" (ring spectrum) — FALSE.* The seed ring mean (7.36) is *below*
reference (7.99) and its spectrum is broad/small-ring-rich. The seed is not
coarse; it is broad-and-fine. BM2000's own seeds (Table I, p.4987) start at a
~30° bond-angle spread and are heavily strained but are *also* not
ring-coarse — the final ring distribution (their Conf. A: 5:0.472 6:0.761
7:0.507 8:0.125 9:0.034 per atom) is *emergent from the anneal*, not present in
the seed. So the coarsening fingerprint is not inherited from this seed's
topology.

*"Too inhomogeneous" (high S(k0)=0.141) — TRUE, and decisive for the void.* By
FACT 2, the local WWW anneal has no restoring force toward box-scale
homogeneity. The annealed networks scatter at S(k0) ~0.07–0.16 — straddling the
seed value 0.141 — and **never descend to the reference 0.041**, because no
term in the anneal drives the lowest reciprocal modes toward zero. The seed
supplies a high-S(k0) starting point and the anneal cannot pull it down to the
reference; the seed's box-scale density fluctuation is therefore the origin of
the residual void. **Verdict: the seed's high S(k0) is the void cause; its ring
spectrum is NOT the coarsening cause.**

---

## Conclusion

The two defects have **different origins**, and the honest read of the evidence
(including this project's own N=1000 result) is that only ONE of them is truly
seed-origin.

### VOID — single most likely seed-origin cause: the seed's box-scale density inhomogeneity (S(k0)=0.141), which local WWW cannot remove.

The chain is clean and literature-grounded:
1. Placement + cycle + matching deliver good *local* spacing (Poisson-disk at
   0.98·d0) but leave the *box-scale* density essentially random — S(k0)=0.141,
   Poisson-like at low k (cause-(a)/(d) analysis above).
2. WWW relaxation and the WWW anneal are **local** (FACT 2: BM2000 3rd-shell /
   Vink 4th-5th-shell relax; local bond-transposition move) and the LSU/Keating
   energy has no long-wavelength term (`lsu_network.py:1424` docstring). So the
   anneal has no restoring force toward box-scale homogeneity.
3. Therefore the final S(k0) scatters around the seed value with no drive toward
   zero — annealed ~0.07–0.16 (straddling seed 0.141), and the reference 0.041
   is never reached. The void is seed-supplied and anneal-unremovable.

This is the strongest defensible seed-origin claim in the whole analysis. Note
it is NOT the Poisson-disk rule that is at fault (that rule helps local
homogeneity); it is that *nothing in the seed construction targets box-scale /
near-hyperuniform density*, and the local anneal can never fix that afterwards.

**The key unknown I cannot resolve read-only, and must state plainly:** if the
anneal has no drive toward low S(k0) and *neither* Poisson-disk *nor*
pure-random placement yields 0.041, **how did Sellers's reference reach 0.041?** The supplement does not
describe the seed construction beyond "random seed patterns ... ~100,000 WWW
iterations" (Supplementary Methods, p.6). Two possibilities, both consistent
with the literature: (i) Sellers's (unpublished) seed was near-hyperuniform by
construction or by a density-equilibration step they did not document; or
(ii) near-hyperuniformity emerges over a route the code does not reproduce
(memory `lsu-n512-validation-temperature-and-tradeoff` records that full-N
relaxation does NOT fix the low-k void either, which argues against pure
emergence via more global relaxation). Sellers cites Hejna 2013
(`...MOESM1815_ESM.pdf` ref 9, "Nearly hyperuniform network models of amorphous
silicon," supplement p.7) precisely on this property — that CRNs are *nearly*
hyperuniform — so a near-hyperuniform seed is the literature-sanctioned lever.

### COARSENING — NOT cleanly seed-origin; the evidence says it is anneal-intrinsic.

I will not name a seed cause for coarsening, because the seed's own topology
argues against every seed-origin candidate (ring mean 7.36 < ref 7.99; broad,
small-ring-rich spectrum), AND because this project's own decisive N=1000
result refutes a seed explanation. Memory `lsu-n1000-tradeoff-hardfloor`
(verified at full scale + 100k iterations): "6-ring≥5% is a HARD FLOOR
(unreachable even at w=0...); pure-WWW emergence REFUTED at 100k (α<0, coarsens
to 9-rings)." Coarsening to 9-rings happens *even at pure WWW with no
uniformity penalty at 100k* — i.e., it is produced by the anneal dynamics
(temperature, move acceptance, energy functional), not by a topological bias
imprinted on the seed. Moreover, **no seed feature pushes toward coarsening**:
the seed (ring mean 7.36 < ref 7.99) is small-ring-rich, and the only directional
seed influence — the *small-ring* bias of nearest-partner matching (cause c) —
points the *opposite* way (it makes the anneal have to coarsen *more*, not less).
So the seed actively *resists* coarsening; coarsening is **wholly
anneal-origin**.

This asymmetry — void = seed-origin, coarsening = anneal-origin — is itself a
result: it tells you to fix the seed for the void and to look at the anneal
(temperature schedule / energy functional, per the N512 and N1000 memories) for
the coarsening, not to keep re-engineering the seed topology for the rings.

---

## Concrete, literature-grounded fix to test (seed construction)

**Target: the VOID only** (the seed fix will not touch coarsening per the above).

**Fix: replace the homogeneity-blind placement with a density-equilibrated /
near-hyperuniform vertex field before topology is built.** Two literature-anchored
options, in increasing strength:

1. **Minimal — lower the seed's S(k0) by equilibrating the point field, not just
   excluding contacts.** The current `_poisson_disk_pbc` (`lsu_network.py:1331`)
   only enforces a hard core; it leaves S(k0)≈0.14. Add a short collective
   point-pattern relaxation of the *placed vertices* under a soft pair potential
   *before* the Hamiltonian cycle (this is a placement step, distinct from the
   bonded `settle_seed_with_repulsion` at `lsu_network.py:1151`, which only acts
   once bonds exist and so cannot drive box-scale modes). A few sweeps of an
   inverse-power or Yukawa repulsion drives an RSA field toward a more
   uniform/stealthy state and *measurably lowers* S(k0). This mirrors the
   "repulsive equilibration that Sellers's cited random-seed refs (Vink 2001;
   Mousseau-Barkema 2001) apply before WWW" already invoked in the
   `settle_seed_with_repulsion` docstring (`lsu_network.py:1169`–1182) — but
   applied to the *bare points* so it can actually move long wavelengths.

2. **Strong — generate a stealthy/near-hyperuniform point pattern as the seed
   skeleton** (the Hejna-2013 lever Sellers cites). Hejna, Steinhardt & Torquato
   2013 (supplement ref 9) show CRN models of a-Si are *nearly hyperuniform*;
   the principled way to inherit S(k0)→0.041 through a local anneal is to *start*
   from a near-hyperuniform field. Build the vertices by collective-coordinate /
   stealthy-hyperuniform optimization (suppress S(k) for k below a cutoff k_c)
   and *then* run the existing cycle + girth≥5 matching + 2-opt on those points.
   Because the anneal conserves low-k (FACT 2), a seed with S(k0)≈0.04 should
   *retain* it, directly closing the gap to the reference.

**Why this is the right fix and not the others:** the void is inherited and the
anneal cannot remove it (FACT 2), so it MUST be fixed in the seed; the seed's
problem is box-scale density, which only a *collective* (long-wavelength) point
operation can change — the hard-core placement rule and the bonded settle both
act locally and demonstrably leave S(k0)≈0.14. Do NOT pursue the
`uniformity_weight` Metropolis penalty (`low_k_structure_factor`,
`lsu_network.py:1424`) as the *primary* fix: per memory
`lsu-sellers-protocol-random-pure-www` it is a non-Sellers crutch, and it fights
the local anneal rather than supplying the missing initial condition. A
near-hyperuniform seed is the faithful, literature-grounded route.

**Primary risk to this fix — state it up front, it is the most likely failure
mode and the data already points to it.** The annealed S(k0) does not merely
"track" the seed: it scatters in ~0.07–0.16 from a 0.141 start, i.e. there is a
*band* the anneal pulls toward, roughly centred near ~0.1. A seed at S(k0)≈0.04
sits *below* that band, so the most probable outcome is that the anneal
**drifts it UP toward ~0.1**, not that it holds at 0.04. If the anneal has a
low-k fixed-point basin near ~0.1, *no* seed below it survives and the seed fix
fails outright. The empirical evidence for this risk is already in hand and
needs no GPU: annealed values appear *both above and below* the seed's 0.141,
which is the signature of an attracting band rather than a conserved value. The
fix is worth testing precisely because this is falsifiable.

**Test plan (read-only-safe to specify):** generate the near-hyperuniform seed,
measure its S(k0) (expect ≈0.04 for option 2), run the existing pure-WWW route
(`uniformity_weight=0`, local-shell relax, `check_lsu_every=0` so the Φ
early-exit does not truncate), and measure whether the annealed S(k0) **holds
near 0.04 or drifts up into the ~0.07–0.16 band**. Holding low ⇒ the void is
seed-fixable (fix succeeds). Drifting up ⇒ the anneal has a low-k basin and the
void cannot be fixed in the seed alone (fix fails; the void becomes a joint
seed+anneal problem). Either way, coarsening is predicted *unaffected* by the
seed change — that is the second falsifiable prediction, and it is what
distinguishes the void (seed-supplied) cause from the coarsening (anneal) cause.

---

## What is NOT in the PDFs (stated rather than guessed)

- **The Sellers supplement does not describe seed construction.** It says only
  "random seed patterns ... after around 100,000 WWW iterations" (Supplementary
  Methods, p.6) and cites Vink [13] / Mousseau-Barkema [14] for the
  *relaxation* refinements. It does NOT say whether their seed was
  pure-random, Poisson-disk, density-equilibrated, or hyperuniform. So I cannot
  claim the reference's 0.041 came from a hyperuniform seed — I can only show
  (FACT 2) that a hyperuniform seed is the consistent way to reach it.
- **BM2000/Vink do not target hyperuniformity at all.** Their quality metrics
  are bond-angle spread, strain/atom, ρ/ρ0, and ring statistics (BM2000 Table I,
  p.4987; Vink Table I, p.4). Neither paper reports S(k0) or any
  hyperuniformity measure. The hyperuniformity framing comes only from Hejna
  2013 (Sellers ref 9), which I read only via its citation in the supplement,
  not as a full PDF in this repo.
- **No PDF states that the local anneal *cannot* lower S(k0).** That is my
  inference from the explicitly local relaxation scope (FACT 2) plus the absence
  of any long-wavelength energy term — corroborated empirically by this
  project's measurements (annealed S(k0) stuck near the seed value) and by
  memory `lsu-n512-validation-temperature-and-tradeoff` (full-N relax does not
  fix it). It is a strong inference, not a literature quote.
