# Physics & Correctness Review — LSU Network Generator

## Context

This project implements the Wooten-Winer-Weaire (WWW) simulated-annealing algorithm for
generating periodic 3D amorphous trivalent networks with prescribed **Local Self-Uniformity
(LSU)**, following:

> Sellers, Man, Sahba & Florescu, "Local self-uniformity in photonic networks",
> *Nature Communications* **8**, 14439 (2017).

The implementation also draws from:
- Vink & Barkema, *Phys. Rev. B* **63**, 245214 (2001) — threshold-energy relax
- Mousseau & Barkema, *Phys. Rev. B* **62**, 4985 (2000) — BM2000 seed / local relax
- Hemmann & Saba, *Adv. Funct. Mater.*, 2026 — triangular burn-in & hyperuniformity

The full Sellers (2017) paper PDF, its supplement, and all supporting papers are in:

    LSU Literature/

The main implementation is in `lsu_network.py` (3 291 lines).
The notebook `Create_LSU_Function.ipynb` is the user-facing entry point.
The analysis helper is `tools.py`.

---

## Authoritative energy weights from Sellers

The user (who communicated directly with the Sellers group) confirmed that the
correct energy weights for the Sellers Eq. 2 energy functional are:

```python
energy_weights = {'alpha': 0.7, 'beta': 0.7, 'gamma': 0.3, 'delta': 0.4}
```

These are the **exact values provided by Sellers** and must be used without change.

The energy is:

    U = alpha * f1 + beta * f2 + gamma * f3 + delta * f4

where (Supplement Eq. 2–4):
- **f1**: Keating-like edge-length deviation, Σ (L_ij − d₀)²
- **f2**: Bond-angle deviation from 120°, Σ (cos θ + 0.5)²
- **f3**: Gyroid-dihedral deviation, Σ (|cos φ_ij| − 1/3)² [Supplement Eq. 3]
- **f4**: Trihedron coplanarity, Σ (r̂_ij · n̂_i)² + (r̂_ij · n̂_j)² [Supplement Eq. 4]

---

## Reference example

The gold-standard output lives in `Example/lsu_example_ends.txt`:
- 1 653 rod lines (1 500 unique edges + 153 PBC-image duplicates)
- N = 1 000 vertices, E = 1 500 unique edges, periodicity L = 11.44 µm
- Target bond length d₀ = 0.8 µm
- Φ_12 ≈ 0.99, Φ_22 ≈ 0.89
- Rod length: mean ≈ 0.800 µm, std ≈ 0.029 µm, q5 ≈ 0.752, med ≈ 0.801, q95 ≈ 0.846

Any generated network should closely match these statistics to be considered correct.

---

## What to review

You have **read access to all files**. Please:

### 1. Physics correctness

Cross-check every formula in `lsu_network.py` against the Sellers 2017 paper and its
supplement (Supplement Eq. 2–4 for the energy; Supplement Methods for the LSU Φ_{nl}
definition):

- `energy_components()` at line 984: NumPy implementation of f1, f2, f3, f4.
- `_energy_jax_full()` at line 1072: JAX implementation of the same.
- Are the two implementations **exactly equivalent**?
- **f3 dihedral**: Is `_DIH_TARGET = 1/3` and taking `|cos φ|` correct per the Sellers
  supplement? Or should it be `cos²φ` = 1/9, or `cos φ` (without absolute value)?
- **f4 coplanarity**: Does `r̂_ij · n̂_i` correctly penalise non-coplanar trihedrals?
  Verify sign convention and whether the sum-of-squares form matches Supplement Eq. 4.
- **f2 bond angle**: The target is cos θ = −1/2 (120°). Correct for both paths?
- **f1 edge length**: Are PBC minimum-image displacements applied before computing L?

### 2. LSU statistic Φ_{nl}

The LSU computation lives in `compute_lsu()` (line 2596), `phi_ab()` (line 2582),
`_phi_for_permutation()` (line 2553), `_align_two_trees()` (line 2437).

Key questions:
- Does `phi_ab(tree_a, tree_a)` return **exactly 1.0** for any tree (self-comparison)?
  If not, the normalisation is wrong. The code averages over all 3! = 6 permutations
  of root edges and divides by γ_fact = 6. A non-symmetric tree will give < 1.0 for
  non-identity permutations, making the self-comparison < 1.0. Does the Sellers
  paper define Φ as a group-average or as a maximum over permutations?
- Does `compute_lsu()` correctly collect pairs at **exactly** locality-l hops vs
  **up to** locality-l hops? The BFS adds ALL vertices within l hops. Check if the
  Sellers paper computes Φ_{nl} over pairs at exactly l hops or up to l hops.
- The `_overlap_score()` (line 2520) computes `(r_a · r_b) / mean(|r_a|, |r_b|)²`.
  Verify this matches the Sellers normalisation. Does using `mean(|r_a|, |r_b|)` vs
  `|r_a| * |r_b|` or `|r_a|² ` matter for the final value?
- For `depth=2`, child edges are stored as displacement vectors from the INTERMEDIATE
  vertex (not from the root). When the root-level rotation R is applied to `ce_b_local`
  in `_phi_for_permutation()` (line 2573), is that the correct rotation frame?

### 3. Stone-Wales topology move

The bond transposition is in `stone_wales_apply()` (line 1526) and
`stone_wales_revert()` (line 1541).

- Verify the move: edges (i,c) and (j,d) become (i,d) and (j,c), keeping (i,j)
  unchanged. Check all 4 neighbor-table updates are correct and that revert is a
  true inverse.
- Does `stone_wales_propose()` (line 1469) ever generate a self-loop or multi-edge?
  The existing checks for `c == d`, `d in nbr_set[i]`, `c in nbr_set[j]` should
  prevent these. Are they sufficient?
- After the move, the graph should still be 3-regular. Is that guaranteed?

### 4. Metropolis / Vink threshold-energy scheme

The main annealing loop is in `www_anneal()` (line 1558) and `relax()` (line 1340).

- Vink 2001 Eq. 5: threshold energy `E_t = E_b − T·ln(s)`, where s ∈ (0,1) and E_b
  is the energy BEFORE the SW move. The code sets this at line 1749:
  `E_t = E_b - T * math.log(max(s, 1e-12))`. Verify this is correct.
- The threshold is only applied to the **strain energy** E_new, not to the full
  `objective_new` (which includes the low-k uniformity penalty). Is this correct?
  The comment says it is intentional — verify it matches Vink/BM2000.
- The Metropolis roll uses the SAME `s` that set E_t: `s < math.exp(-dE / T)` for
  the objective (line 1821). Verify this is the correct Vink identity (early rejection
  on strain energy + Metropolis on objective both use the same s).
- The `c_f` BM2000 estimator: `E_est = E - c_f * |F|²` approximates the final energy.
  Is `c_f = 0.5` the correct value from BM2000 Eq. 4?

### 5. Hemmann triangular burn-in

In `topology_burn_in()` (line 2171) and `_calibrate_T_melt()` (line 2041):

- Is T_melt calibrated correctly via Hemmann Eq. 5: `T_melt = <ΔE_up> / ln(1/P_melt)`?
- The code collects uphill ΔEs from ALL proposed moves (not just accepted ones). Is
  that the correct interpretation of Hemmann's `<ΔE_up>`?
- The triangular schedule: heat 0→T_max, cool T_max→0, quench T=0. Does the code
  build this exactly?

### 6. Seed networks

- `crystal_seed_network()` (line 282): the srs lattice is defined at line 135.
  Verify the 12 bonds and 8 site coordinates match the I4₁32 srs net (Wyckoff 8a,
  x=1/8). All bonds should have length `a·√2/4`. Are there any self-loops or missing
  bonds in the tiling loop?
- `random_seed_network_bm2000()` (line 466): BM2000 §II.A Hamiltonian-cycle +
  loop-expansion. Verify the Hamiltonian cycle gives exactly deg=2 everywhere before
  the loop expansion, and that the loop expansion terminates correctly.

### 7. Comparison with reference output

Load `Example/lsu_example_ends.txt` via `tools.analyze_network()` and compare the
key statistics to what a fresh run of the notebook produces. Specifically:

- Φ_12 ≈ 0.99, Φ_22 ≈ 0.89
- Bond length: mean ≈ 0.800, std ≈ 0.029
- Bond angle: roughly centred on 120° (Sellers Fig. 4b)
- Ring distribution: centred on 7–8 rings (amorphous gyroid)
- Voxel std (4³ grid) ≈ 3.65, corner/centre ratio ≈ 1.21

If the generated output deviates significantly from these numbers, identify which part
of the algorithm is responsible.

---

## Instructions

1. **Read all papers** in `LSU Literature/` (especially `ncomms14439.pdf` and its
   supplement `41467_2017_BFncomms14439_MOESM1815_ESM.pdf`) before drawing any
   conclusions about correct physics. Do NOT rely on memory of the paper — read the
   actual PDFs.

2. **Find and fix all bugs** you identify. Fix them directly in `lsu_network.py`.

3. **Do not skip any step** of the Sellers algorithm. If the code is missing a step
   that the paper requires, add it.

4. **Do not change the energy weights**: alpha=0.7, beta=0.7, gamma=0.3, delta=0.4
   are Sellers-confirmed values.

5. **Launch parallel sub-agents** to investigate independent subsystems concurrently
   (e.g. one agent for the energy/physics, one for the LSU statistic, one for the
   Stone-Wales + topology logic, one for the seed networks). Synthesize their findings
   and apply all fixes.

6. After fixing bugs, verify the notebook still runs and the output statistics approach
   the reference values from `Example/lsu_example_ends.txt`.

7. Summarise every bug found (what it was, where, why it was wrong, what the fix is).

---

## Working directory

    /home/francisco/Documents/Create LSU Structures  - Claude/

Key files:
- `lsu_network.py`      — main algorithm (3 291 lines)
- `Create_LSU_Function.ipynb` — entry-point notebook
- `tools.py`            — analysis / comparison helpers
- `Example/lsu_example_ends.txt` — reference output (1 653 rods)
- `LSU Literature/ncomms14439.pdf` — Sellers 2017 paper
- `LSU Literature/41467_2017_BFncomms14439_MOESM1815_ESM.pdf` — Sellers supplement
- `LSU Literature/PhysRevB.62.4985.pdf` — Mousseau-Barkema 2000
- `LSU Literature/Sci. 5, 497–502 (2001).pdf` — Vink 2001 (threshold-energy relax)
- `LSU Literature/Adv Funct Materials - 2026 - Hemmann ....pdf` — Hemmann 2026
