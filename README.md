# WWW Algorithm + LSU — Equation Reference

Following Sellers et al., Nat. Commun. 8, 14439 (2017) and its supplement.

## 1. Network model
- A continuous random network (CRN) of trivalent (γ=3) vertices in 3D.
- Each vertex i has exactly 3 neighbors. A graph with N vertices has E = 3N/2 edges
  (so N must be even, and num_rods must be a multiple of 3).
- The cell is periodic; all vector quantities use minimum-image PBC.

## 1.5. Seed network — srs crystalline seed + controlled WWW burn-in
The current implementation follows the Hemmann/Saba 2026 reading of WWW:
start from a periodic crystalline network with the desired coordination, then
use Stone-Wales moves to disorder topology. For amorphous gyroids the natural
Z=3 parent is the single-network gyroid (`srs`) net, not a generic trivalent
crystal.

Implemented in `crystal_seed_network` / `topology_burn_in`:

1. **srs crystalline seed**: 8 vertices per cubic cell, all bonds length
   `a·sqrt(2)/4`, and three 120° bonds at every vertex. For the Sellers
   reference size (`N=1000`, `L=11.44`, `d0=0.8`) this gives initial bonds of
   ~0.809 µm, close to the target length.
2. **Initial L-BFGS**: settles jitter and pulls the seed exactly onto `d0`.
3. **Controlled burn-in**: constant-T WWW with no LSU target. Temperature is
   auto-calibrated to modest acceptance near melting, then burn-in stops once
   accepted Stone-Wales moves have involved each vertex a capped number of
   times (`topology_burn_in_target_accepts_per_vertex`, default 4.0).
4. **Long-wavelength uniformity guard**: during Metropolis acceptance,
   `uniformity_weight` penalizes low-k vertex structure factor so density
   modes that produce large voids are rejected. L-BFGS still minimizes the
   Sellers local geometry energy.

## 2. Energy (Supplement, Eq. 2)
    U = α f1({d}) + β f2({θ}) + γ f3({φ}) + δ f4({χ})

- **f1** — Keating-like edge length term:
      f1 = Σ_edges (|r_ij| − d0)^2
  Drives all bonds toward target length d0.

- **f2** — Bond angle term, target 120° (cos = −1/2):
      f2 = Σ_vertices Σ_{pairs (a,b) of neighbors} (cos θ_ab + 1/2)^2

- **f3** — Dihedral term (Supplement Eq. 3):
      f3 = Σ_edges (|n̂_{i1,i2} · n̂_{j1,j2}| − 1/3)^2
  where i and j share an edge; (i1,i2) are the other neighbors of i, and
  (j1,j2) are the other neighbors of j; n̂_{a,b} is the unit normal to the
  plane spanned by (r_{ia}, r_{ib}). This favors gyroid-like dihedrals
  arccos(±1/3) ≈ 70.53° / 109.47°.

- **f4** — Skew angle / coplanarity (Supplement Eq. 4):
      f4 = Σ_edges (r̂_ij · n̂_{i1,i2})^2 + (r̂_ij · n̂_{j1,j2})^2
  Penalizes the central edge being out of the plane of the trihedron at
  either endpoint.

The paper does not print numeric weights; the values confirmed directly by
the Sellers group are α=0.7, β=0.7, γ=0.3, δ=0.4, and these are the
`generate_lsu_network` defaults.

## 3. WWW iteration (Supplement, Eq. 1 + Methods)
1. Pick a random edge (i, j).
2. Pick a random neighbor c of i (c ≠ j) and a random neighbor d of j (d ≠ i,
   d ≠ c, must not be already adjacent to either).
3. Stone-Wales / bond transposition: remove edges (i, c) and (j, d); add edges
   (i, d) and (j, c). This preserves trivalence.
4. Locally relax positions of vertices within the Vink/Mousseau-Barkema
   fourth-neighbour shell of the Stone-Wales move to minimize U for the new
   topology.
5. Compute ΔE from the relaxed Sellers energy plus the optional low-k
   uniformity penalty.
6. Accept with probability P_a = min(1, exp(−ΔE / T)) (Eq. 1 of supplement).
7. Cool T according to a schedule (geometric decay).

Fixed-schedule global relaxations are disabled by default because they
re-introduce void drift under the bonded-only local energy.

## 4. LSU statistic
For each vertex pair (a, b) with b within `locality` edges of a:

(a) Build n-trees T_n^a, T_n^b by breadth-first traversal of depth n from a, b.
(b) Translate so root vertices coincide.
(c) For each permutation σ of root edges of T_n^b:
    - Rotate T_n^b around root vertex to maximally align its 1st root edge with
      σ-image of 1st root edge of T_n^a, then a second rotation around that
      first edge to align the 2nd, etc. (3 alignment steps for trihedral; 4 for
      tetrahedral with reflection — Supplement Methods).
    - Recursively pair non-root edges depth-first to maximize overlap (greedy:
      at each interior vertex, pick the assignment of (γ−1) child edges that
      maximizes sum of overlap dot products).
    - Score the alignment via Eq. 3:
        f(T^a, T^b; σ) = (1 / (|T_n^a| − 1)) · Σ_pairs (r^a · r^b) / (mean(|r^a|,|r^b|))^2
      (|T_n^a| − 1 = number of edges in the tree: 3 at depth 1, 9 at depth 2.)
(d) ϕ_ab = (1 / γ!) · Σ_σ f(T_n^a, T_n^b; σ)            (Eq. 1)

Φ_nl = mean of {ϕ_ab : b within l edges of a, over all a}. The first
subscript is the tree depth n, the second the locality l (Sellers Eq. 2;
Fig. 3b plots Φ_12, Φ_22, Φ_32 — all at locality 2).

For our trivalent case:
- Φ_12: γ=3, depth 1, locality 2. Each tree has 3 edges; 3! = 6 permutations.
- Φ_22: γ=3, depth 2, locality 2. Each tree has 3 + 6 = 9 edges; same 6 root
  permutations (interior pairings handled greedily by the depth-first heuristic).

## 5. PBC unwrapping for output
Each rod's two endpoints can straddle a periodic boundary. The example file
stores every rod at full length: each face-crossing edge appears twice, once
anchored at each endpoint's canonical-box image (the two rows are the same
segment translated by a lattice vector), so either endpoint of a row may lie
up to one rod length outside `[-L/2, L/2]^3`. We mirror this convention in
the output array (`pbc_duplicate_boundary_rods=True`,
`clip_endpoints_to_box=False`).

## 6. Periodic supercell vs. visible window
The example has periodicity L = 11.44 µm, so the canonical box is
`[-5.72, 5.72]^3`. Visible coordinates extend to ~±6.48 µm because rods
crossing a face of the canonical box have one endpoint outside. The
"≈13 µm size" in the example caption refers to that visible bounding box,
not the true period.
