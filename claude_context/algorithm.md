# WWW Algorithm + LSU — Equation Reference

Following Sellers et al., Nat. Commun. 8, 14439 (2017) and its supplement.

## 1. Network model
- A continuous random network (CRN) of trivalent (γ=3) vertices in 3D.
- Each vertex i has exactly 3 neighbors. A graph with N vertices has E = 3N/2 edges
  (so N must be even, and num_rods must be a multiple of 3).
- The cell is periodic; all vector quantities use minimum-image PBC.

## 1.5. Seed network — Barkema-Mousseau (PRB 62, 4985, 2000) §II.A
The Sellers supplement Methods cites Vink 2001 / Mousseau-Barkema 2001, both built
on the Barkema-Mousseau 2000 procedure. The configuration model (random pairing of
labelled stubs, then positions assigned uniformly afterwards) is what BM explicitly
reject — it leaves long-range "memory-less" bonds that produce empty regions and a
heavy-tailed bond-length distribution after relaxation.

BM §II.A targets *tetravalent* (Si) networks via a single loop visiting each atom
twice. For our trivalent case the loop visits each atom once and a chord matching
lifts every vertex from degree 2 to degree 3.

Implemented in `bm_initial_network` / `_build_trivalent_proximity_graph`:

1. **Hard-core placement** (`poisson_disk_pbc`, BM §II.A): N vertices placed
   uniformly in [-L/2, L/2]^3 with PBC under a minimum-image separation
   `r_min = 0.7·d0` (the Si analogue is 2.3 Å for d0 = 2.35 Å). The constraint
   softens by 5% on excess rejection rate.
2. **Stage A — Hamiltonian-like cycle via BM loop expansion**:
   - Seed with a triangle of 3 mutually-close vertices.
   - Iteratively insert each remaining vertex `A` into an existing edge
     `(B, C)` when `dist(A, B), dist(A, C) ≤ r_cut = 1.7·d0`, replacing
     `(B, C)` with `(A, B), (A, C)`. This is the BM elementary "+1 edge" move
     (their Fig. 1) — A goes from degree 0 to 2; B, C stay at 2.
   - Fallback: if no valid `(B, C)` exists for any remaining `A`, bond `A` to
     its two nearest neighbours that still have degree < 3. May saturate
     those neighbours to degree 3 a step early; Stage B then issues fewer
     chords.
3. **Stage B — Chord matching to degree 3**: repeatedly add the shortest
   unbonded pair of degree-2 vertices. If the shortest valid pair lies
   beyond `r_cut`, take it anyway — relaxation will pull it in.
4. Outer loop in `bm_initial_network`: for each `r_cut` value try
   `layouts_per_cutoff = 4` independent Poisson-disk layouts before
   widening `r_cut` by 8%.

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
