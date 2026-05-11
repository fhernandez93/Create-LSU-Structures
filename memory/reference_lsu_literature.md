---
name: LSU literature anchors
description: External-literature pointers for tuning / extending the Sellers energy U = α f1 + β f2 + γ f3 + δ f4 and for matching the Sellers-cited Vink/Mousseau-Barkema relaxation scheme.
type: reference
---

**No published values for {α, β, γ, δ}** for the Sellers four-term energy exist. Verified absent from: the 2017 Nat. Commun. paper + supplement, patent WO2017134424A1, Surrey dataset DOI 10.15126/surreydata.00813094 (metadata-only), no public code release on GitHub/Zenodo, no follow-up from the Florescu group republishing the weights. Don't waste time searching for them again.

**Sellers-prescribed relaxation scheme (reading the actual supplement, p. 6).** The Sellers paper says only "the network is then relaxed to minimise an energy function" and explicitly delegates the relaxation *refinements* to refs [13] = Vink, Barkema, Stijnman, Bisseling 2001 PRB 64, 245214 and [14] = Mousseau, Barkema 2001 Curr. Opin. Solid State Mater. Sci. 5. The Vink/MB recipe is **spatially-local relaxation restricted to the 4-neighbour shell of the SW move, with full-network L-BFGS only as a rare fallback** (when local relax fails to lower energy). Hemmann/Saba 2026 p. 4 quotes the same refs to the same effect. Our `lsu_network.py` originally did full-N L-BFGS for both the post-SW relax and the periodic polish (distinguished only by `max_iter`); that was the mechanism producing the corner-void clustering, see `project_known_issues.md`.

**Hemmann, Glauser, Steiner, Saba (2026)** — arXiv:2601.10333 / Adv. Funct. Mater. DOI 10.1002/adfm.202600037. Independent re-derivation of WWW for arbitrary-coordination disordered networks, with amorphous gyroid (srs, Z=3) as one application case. The PDF is in `LSU Literature/Adv Funct Materials - 2026 - Hemmann ...pdf`. **Key facts (corrected 2026-05-07):**
- Their energy (Eq. 4): `E = (3/16) Σ_(ij) (r_ij²−1)² + (3/8) β Σ_(jik) (r_ij·r_ik · cos θ_jik + 1)²`. Two-body bond stretch + three-body **bonded** bond bending. **No non-bonded vertex-repulsion term.**
- Their "angle-dependent repulsion" is between *bond directions sharing a vertex* (equilibrium angle 180° for arbitrary Z), not between non-bonded vertices. For Z=3 specifically the ground-state geometry is the same as Sellers's f2 with target 120° (frustrated equilateral triangle); the Hemmann modification gains nothing structural for our Z=3 case.
- They explicitly observe and characterise vertex/void clustering with dedicated metrics (`r_u`, `δ_c`); Figure 8c shows critical pore radius growing monotonically with accepted MC moves *for all β values*. Their fix is **algorithmic**: operate near the melting transition (1.0 ≲ T_max/T_melt ≲ 1.3) with relatively few accepted moves, plus the Vink/MB local-shell relaxation scheme.
- For the Z=3 gyroid case they use the periodic `srs` net as the crystalline
  starting point. `srs` is the correct ordered parent for Sellers amorphous
  gyroids; a generic Z=3 crystal is less faithful even if it is connected and
  trivalent.
- Because they note that local Keating-like energies do not directly control
  pore size / hyperuniformity, the implementation now includes an optional
  low-k structure-factor penalty in Metropolis acceptance (`uniformity_weight`,
  default 10.0). This is a deliberate long-range uniformity guard; set it to
  0.0 for strict Sellers Eq. 2 acceptance.
- They do **not** use the Sellers LSU statistic Φ_nl. They use 12 quantifiers across direct space (`σ_r`, `σ_θ`, `r_nn`, `r_u`, `δ_c`, ring statistics) and reciprocal space (structure factor, hyperuniformity exponent α). LSU evaluation in our pipeline remains independent of their work.
- (Earlier memory entry incorrectly said Hemmann adds non-bonded vertex repulsion. They don't. Cite this paper as published precedent for the Vink/MB local-shell relaxation, not for an added energy term.)

**Vink, Barkema, Stijnman, Bisseling (2001)** — PRB 64, 245214. Standard Si-Keating ratio β/α ≈ 0.285 (bond-bend ≈ 3.5× weaker than bond-stretch). The classical prior for picking β when α is fixed; informs the recommended {α=10, β≈2} starting point. Also the canonical source for the local-shell relaxation refinement Sellers cites.

**Synthesised weight prior** (no published Sellers values, derived from Si Keating ratio): with α as unit, β ≈ 0.2–0.3, γ ≈ 0.1–0.2, δ ≈ 0.05–0.1. So with α=10: β ≈ 2, γ ≈ 1.5, δ ≈ 1. Earlier weights {10, 2, 5, 5} or {80, 5, 1, 0.5} both produced void clustering — but the dominant cause was the relax scheme, not the weights.

**How to apply:** Use the literature-prior weights when no specific reason to deviate. For void clustering / hyperuniformity issues the first-line fix is the relaxation scheme, not the energy weights. Cite Vink/MB 2001 (the Sellers-cited reference) for the spatially-local relax, and Hemmann/Saba 2026 as recent independent confirmation.
