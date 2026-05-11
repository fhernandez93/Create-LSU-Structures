# WWW Algorithm + LSU — Equation Reference

Following Sellers et al., Nat. Commun. 8, 14439 (2017) and its supplement.

## 1. Network model
- A continuous random network (CRN) of trivalent (γ=3) vertices in 3D.
- Each vertex i has exactly 3 neighbors. A graph with N vertices has E = 3N/2 edges
  (so N must be even, and num_rods must be a multiple of 3).
- The cell is periodic; all vector quantities use minimum-image PBC.

## 1.5. Seed network — srs crystalline Z=3 seed + controlled WWW burn-in
The pipeline now follows the Hemmann/Saba 2026 recipe (Adv. Funct. Mater.,
PDF in `LSU Literature/`): start from a **periodic Z=3 crystalline net**
and run a high-T constant-temperature **WWW Stone-Wales "burn-in"** to lose
crystalline memory before the production LSU-targeted annealing begins.
This matches the same ensemble Sellers's refs [27] (Wooten-Winer-Weaire
1985) and [28] (Mousseau-Barkema 2001) describe — once burn-in is long
enough, the topology distribution is determined by the Sellers energy,
not the seed. The crystalline starting point gives a robust entry to
that landscape: every initial bond is the same length, connectivity is
by construction, every vertex is exactly trivalent, and no long-distance
chord stragglers can drift vertices around during the initial L-BFGS.

The legacy Barkema-Mousseau random seeder (`bm_initial_network`,
`_build_trivalent_proximity_graph`, `poisson_disk_pbc`) was removed:
it shortcut BM 2000's loop expansion with a Hamiltonian-cycle + greedy
nearest-pair chord matching, which left 3–5·d0 stragglers for the few
isolated degree-2 vertices and dragged vertices across the cell during
relax — the void-clustering mechanism documented in
`memory/project_known_issues.md`.

### Implementation: `crystal_seed_network` + `topology_burn_in`

1. **Build the crystalline lattice** (`crystal_seed_network`): pick the
   tile (nx, ny, nz) so that `vertices_per_cell · nx·ny·nz ≈ N` (rounds
   to nearest valid N; pass `strict_tiling=True` to raise instead).
   Default lattice `srs`: the single-network gyroid net (I4_132, 8a,
   x=1/8), 8 vertices / cubic cell, all bond lengths `a·sqrt(2)/4`, and
   three coplanar 120° bonds at every vertex. This is the ordered parent
   of the amorphous gyroid and matches Hemmann/Saba's Z=3 gyroid starting
   point. The older `diamond3` option remains available for diagnostics,
   but it is no longer the default: at N=1000, L=11.44, d0=0.8 its seed
   bonds are ~0.991 µm, forcing a distorting initial relax.
2. **Position jitter**: Gaussian noise of std `jitter_sigma · d0`
   (default 0.10) breaks exact symmetry so the first SW moves see a
   non-degenerate Hessian.
3. **Initial relax** (full-N L-BFGS, `relax_global_iters` iters): pulls
   bonds from `a·sqrt(3)/4` to `d0` via the Keating f1 term and absorbs
   the jitter. For N=1000 / box=11.44 / d0=0.8 this lands all bonds at
   ~d0 with std 0.0 (perfect lattice), φ_22 = 1.0000.
4. **Topology burn-in** (`topology_burn_in`): constant-T WWW with no
   LSU target. T is auto-calibrated by a 200-move probe sweep to a modest
   acceptance near melting (~20% by default), or user-supplied via
   `topology_burn_in_T`. The burn-in also stops when accepted moves have
   involved each vertex `topology_burn_in_target_accepts_per_vertex`
   times on average (default 4.0; each accepted move counts four vertex
   involvements). This replaces the older 70%-acceptance / 20k-move
   maximum-randomization burn-in, which could overrun into Hemmann's
   large-pore regime.
5. **Long-wavelength uniformity guard**: `uniformity_weight` adds a
   low-k vertex structure-factor penalty to the Metropolis acceptance
   objective. The L-BFGS relaxation still minimizes the Sellers local
   geometry energy; this acceptance term rejects topology moves that
   amplify box-scale density fluctuations / voids. Set
   `uniformity_weight=0.0` for strict Sellers Eq. 2 acceptance.

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

Default weights from the paper: not explicitly given. We use α=β=γ=δ=1 as a
reasonable starting point; user can tune.

## 3. WWW iteration (Supplement, Eq. 1 + Methods)
1. Pick a random edge (i, j).
2. Pick a random neighbor c of i (c ≠ j) and a random neighbor d of j (d ≠ i,
   d ≠ c, must not be already adjacent to either).
3. Stone-Wales / bond transposition: remove edges (i, c) and (j, d); add edges
   (i, d) and (j, c). This preserves trivalence.
4. Locally relax positions of vertices in the 1- or 2-neighborhood of {i, j} to
   minimize U for the new topology.
5. Compute ΔE = U_new − U_old.
6. Accept with probability P_a = min(1, exp(−ΔE / T)) (Eq. 1 of supplement).
7. Cool T according to a schedule (geometric decay).

Periodic global relaxations every K accepted moves clean up accumulated drift.

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
        f(T^a, T^b; σ) = (1 / |T_n^a|) · Σ_pairs (r^a · r^b) / (mean(|r^a|,|r^b|))^2
(d) ϕ_ab = (1 / γ!) · Σ_σ f(T_n^a, T_n^b; σ)            (Eq. 1)

Φ_nl = mean of {ϕ_ab : b within l edges of a, over all a}.

For our trivalent case:
- Φ_12: γ=3, depth 1, locality 1. Each tree has 3 edges; 3! = 6 permutations.
- Φ_22: γ=3, depth 2, locality 2. Each tree has 3 + 6 = 9 edges; same 6 root
  permutations (interior pairings handled greedily by the depth-first heuristic).

## 5. PBC unwrapping for output
Each rod's two endpoints can straddle a periodic boundary. The example file
keeps one endpoint inside the canonical box `[-L/2, L/2]^3` and stores the
second endpoint as `endpoint_inside + minimum_image_displacement`, which can
extend outside the box by up to one rod length. We mirror this convention in
the output array.

## 6. Periodic supercell vs. visible window
The example has periodicity L = 11.44 µm, so the canonical box is
`[-5.72, 5.72]^3`. Visible coordinates extend to ~±6.48 µm because rods
crossing a face of the canonical box have one endpoint outside. The
"≈13 µm size" in the example caption refers to that visible bounding box,
not the true period.
