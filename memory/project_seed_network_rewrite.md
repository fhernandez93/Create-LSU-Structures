---
name: Seed-network rewrite (2026-05-11)
description: The BM-style random seeder was replaced with a crystalline-seed + WWW burn-in pipeline following Hemmann/Saba 2026 and Sellers refs [27,28].
type: project
---

The legacy `bm_initial_network` / `_build_trivalent_proximity_graph` /
`poisson_disk_pbc` / `random_3regular_graph` seeders are gone as of
2026-05-11. The user found the BM-style seeds produced bad-quality
samples for downstream FDTD work and asked for an entirely new function.

**New pipeline** (in `lsu_network.py`):
1. `crystal_seed_network` — tiles a periodic Z=3 crystalline lattice
   (default `'diamond3'`: cubic diamond minus a direction-balanced
   perfect matching of 4 bonds, 8 atoms/cubic cell, all bonds equal).
   Asserts Z=3 + 3D connectivity invariants.
2. Initial full-N L-BFGS pulls bonds to `d0` (e.g. 0.99 → 0.81 for the
   N=1000, L=11.44, d0=0.8 reference).
3. `topology_burn_in` — constant-T WWW Stone-Wales loop with no LSU
   target. T auto-calibrated to ~70% acceptance via a 200-move probe
   sweep, or user-supplied. Stops early when the 4³-voxel-density std
   plateaus.
4. Production WWW (existing `www_anneal`) targets the user's Φ.

**Why:** Refs [27] (WWW 1985) and [28] (Mousseau-Barkema 2001) — the
two papers Sellers cites for "random seed networks" — both end the
seed procedure with many SW transpositions; the seed only sets where
you enter the energy landscape. The Hemmann/Saba 2026 paper does
exactly this with a crystalline starting point and reports clean
amorphous Z=3 networks. The crystalline seed gives every initial bond
the same length, perfect Z=3, and 3D connectivity — none of which the
old BM seeder could guarantee (it could leave 3–5·d0 chord stragglers).

**How to apply:**
- For new runs, just call `generate_lsu_network(...)` with the new
  default kwargs (`seed_lattice='diamond3'`, `topology_burn_in_moves=
  20_000`).
- If the user reports a regression vs the old BM behaviour, the
  documented fallback is `topology_burn_in_moves=0` (skip burn-in,
  use the bare crystal — useful only for diagnostics; produces
  Φ_22 ≈ 1.0).
- The seed assertion `crystal_seed_network: lattice 'X' is not
  3D-connected at tiling (...)` means the user picked a lattice or
  tile size that produces disconnected slabs; the diamond3 default is
  verified to be connected at all tile sizes ≥ (2,2,2).
- A common warning is `seed bond length X differs from d0 by more
  than 20%` — for the diamond3 lattice, bond length is
  `(box[0]/nx) * sqrt(3)/4`. If this is too far from d0, the initial
  L-BFGS will distort the lattice; the user should adjust
  (num_vertices, bounds_microns, edge_length) so the geometry is
  consistent.

**Adding new lattices:** add an entry to `_LATTICE_LIBRARY` with
`sites_frac`, `bonds` (with PBC offsets), `cell_aspect`,
`target_bond_frac`, `vertices_per_cell`. The `crystal_seed_network`
runtime checks (Z=3, edge count = 3N/2, 3D connectivity) catch
bad lattice definitions before the user wastes a burn-in run.
