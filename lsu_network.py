"""
lsu_network.py
==============

Generate periodic 3D amorphous trivalent networks with prescribed
Local Self-Uniformity (LSU), following

    Sellers, Man, Sahba & Florescu,
    "Local self-uniformity in photonic networks",
    Nature Communications 8, 14439 (2017).

Algorithm: Wooten-Winer-Weaire (WWW) simulated annealing on a 3-regular graph
with periodic boundary conditions, using the modified amorphous-gyroid energy

    U = alpha * f1 + beta * f2 + gamma * f3 + delta * f4

(Supplement Eq. 2). f1 is a Keating-like edge length term, f2 is a 120-degree
bond-angle term, f3 is the gyroid-like dihedral term (Supplement Eq. 3), and
f4 is the trihedron coplanarity term (Supplement Eq. 4).

Public entry point: ``generate_lsu_network``.

The module also exposes the building blocks (energy, relaxation, Stone-Wales
move, LSU statistic, output formatting) so they can be reused or replaced.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from itertools import permutations
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

try:
    import jax
    # L-BFGS-B is sensitive to gradient precision; enable float64 globally.
    # NB: this is process-wide once set. Any other JAX code in the same
    # interpreter will also see float64 as the default.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import grad, jit

    HAS_JAX = True
except Exception:
    HAS_JAX = False

try:
    from jaxopt import LBFGS as _JaxoptLBFGS
    HAS_JAXOPT = True
except Exception:
    HAS_JAXOPT = False


# --------------------------------------------------------------------------- #
# PBC helpers
# --------------------------------------------------------------------------- #
def coerce_box(bounds: Union[float, Tuple[float, float, float], np.ndarray]) -> np.ndarray:
    if np.isscalar(bounds):
        return np.array([bounds, bounds, bounds], dtype=np.float64)
    box = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if box.size != 3:
        raise ValueError(f"bounds_microns must be scalar or length-3, got {bounds}")
    return box


def pbc_displacement(d: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image displacement (vectorized over leading axes)."""
    return d - box * np.round(d / box)


# --------------------------------------------------------------------------- #
# Topology — random 3-regular graph
# --------------------------------------------------------------------------- #
def is_connected(N: int, edges: np.ndarray) -> bool:
    nbrs: list[list[int]] = [[] for _ in range(N)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    visited = np.zeros(N, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        u = stack.pop()
        for v in nbrs[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return bool(visited.all())


def build_neighbors(N: int, edges: np.ndarray) -> np.ndarray:
    """Return an (N, 3) array of neighbor indices for a 3-regular graph."""
    out = np.full((N, 3), -1, dtype=np.int64)
    fill = np.zeros(N, dtype=np.int64)
    for a, b in edges:
        out[a, fill[a]] = b
        fill[a] += 1
        out[b, fill[b]] = a
        fill[b] += 1
    if not np.all(fill == 3):
        raise ValueError("Not all vertices have degree 3.")
    return out


# --------------------------------------------------------------------------- #
# Crystalline Z=3 seed networks
# --------------------------------------------------------------------------- #
# A Z=3 crystalline seed gives a known-good starting graph for WWW annealing:
# every initial bond has the same length, connectivity is by construction,
# every vertex is exactly trivalent, and no long-distance chord stragglers
# can drift vertices around during the initial L-BFGS. A subsequent high-T
# WWW "burn-in" (see ``topology_burn_in``) destroys the crystalline memory;
# with enough accepted moves the topology distribution is determined by the
# Sellers energy, not by the seed. This is the Hemmann/Saba 2026 recipe;
# combined with the burn-in it samples the same ensemble Sellers's refs
# [27] (Barkema-Mousseau 2000) and [28] (WWW 1985) describe from a random
# seed, but with markedly better-conditioned initial geometry.

# Lattice entries each provide:
#   sites_frac:       (n_sites, 3) fractional coords inside the unit cell
#   bonds:            (n_bonds, 5) rows [vi, vj, dx, dy, dz] — bond connects
#                     site vi in cell (i,j,k) to site vj in cell
#                     (i+dx, j+dy, k+dz), PBC-wrapped
#   cell_aspect:      (cx, cy, cz) ratios of unit-cell edges
#   target_bond_frac: bond length in units of `ax` (the Cartesian x-edge
#                     of the tile; valid for orthorhombic lattices with
#                     symmetric aspect)
#   vertices_per_cell

_LATTICE_LIBRARY: Dict[str, Dict] = {
    "srs": {
        # The single-network gyroid / srs net (I4_132, Wyckoff 8a with
        # x=1/8), used by Hemmann/Saba for the Z=3 gyroid case and by
        # Sellers as the ordered parent of amorphous gyroids. It is the
        # natural crystalline seed here: 8 vertices per cubic cell, all
        # bonds have length a*sqrt(2)/4, and every vertex has three
        # coplanar bonds at 120 degrees (cos theta = -1/2), exactly matching
        # the Sellers f2 target before disorder is introduced.
        "sites_frac": np.array([
            [1, 1, 1],
            [3, 7, 5],
            [7, 5, 3],
            [5, 3, 7],
            [5, 5, 5],
            [7, 3, 1],
            [3, 1, 7],
            [1, 7, 3],
        ], dtype=np.float64) / 8.0,
        "bonds": np.array([
            # Nearest-neighbour srs bonds. Format [vi, vj, dx, dy, dz],
            # connecting vi in the current cell to vj in the offset cell.
            [0, 5, -1,  0,  0],
            [0, 6,  0,  0, -1],
            [0, 7,  0, -1,  0],
            [1, 4,  0,  0,  0],
            [1, 6,  0,  1,  0],
            [1, 7,  0,  0,  0],
            [2, 4,  0,  0,  0],
            [2, 5,  0,  0,  0],
            [2, 7,  1,  0,  0],
            [3, 4,  0,  0,  0],
            [3, 5,  0,  0,  1],
            [3, 6,  0,  0,  0],
        ], dtype=np.int64),
        "cell_aspect": (1.0, 1.0, 1.0),
        "target_bond_frac": math.sqrt(2.0) / 4.0,
        "vertices_per_cell": 8,
    },
    "diamond3": {
        # 8 atoms per cubic unit cell. The diamond (Z=4) lattice with a
        # perfect matching of 4 bonds removed -> Z=3 everywhere.
        # Sublattice A = sites 0..3, B = sites 4..7 (offset by 1/4 along
        # the cubic diagonal). The matching pairs each A_i with a unique
        # B_j AND uses each of the 4 tetrahedral directions
        # {d1=(1,1,1)/4, d2=(1,-1,-1)/4, d3=(-1,1,-1)/4, d4=(-1,-1,1)/4}
        # exactly once: {A0-B0 (d1), A1-B3 (d3), A2-B1 (d4), A3-B2 (d2)}.
        # This direction-balanced removal keeps the remaining 12 bonds
        # 3D-connected across cells (verified at runtime by ``is_connected``).
        # Using fewer than 4 directions leaves slab-like disconnected
        # components (e.g. {A0-B0, A1-B2, A2-B1, A3-B3} splits the
        # (5,5,5) tile into 5 disconnected 200-vertex slabs).
        # All remaining bonds have Cartesian length a*sqrt(3)/4.
        "sites_frac": np.array([
            [0.00, 0.00, 0.00],   # A0
            [0.00, 0.50, 0.50],   # A1
            [0.50, 0.00, 0.50],   # A2
            [0.50, 0.50, 0.00],   # A3
            [0.25, 0.25, 0.25],   # B0
            [0.25, 0.75, 0.75],   # B1
            [0.75, 0.25, 0.75],   # B2
            [0.75, 0.75, 0.25],   # B3
        ], dtype=np.float64),
        "bonds": np.array([
            # 12 bonds = (16 diamond bonds) - (4 removed). Format:
            # [vi, vj, dx, dy, dz]. Removed bonds (commented out below
            # the table) are A0-B0(0,0,0), A1-B3(-1,0,0), A2-B1(0,-1,0),
            # A3-B2(0,0,-1) -- each in a unique tetrahedral direction.
            [0, 5,  0, -1, -1],  # A0-B1 (d2)
            [0, 6, -1,  0, -1],  # A0-B2 (d3)
            [0, 7, -1, -1,  0],  # A0-B3 (d4)
            [1, 4,  0,  0,  0],  # A1-B0 (d2)
            [1, 5,  0,  0,  0],  # A1-B1 (d1)
            [1, 6, -1,  0,  0],  # A1-B2 (d4)
            [2, 4,  0,  0,  0],  # A2-B0 (d3)
            [2, 6,  0,  0,  0],  # A2-B2 (d1)
            [2, 7,  0, -1,  0],  # A2-B3 (d2)
            [3, 4,  0,  0,  0],  # A3-B0 (d4)
            [3, 5,  0,  0, -1],  # A3-B1 (d3)
            [3, 7,  0,  0,  0],  # A3-B3 (d1)
        ], dtype=np.int64),
        "cell_aspect": (1.0, 1.0, 1.0),
        "target_bond_frac": math.sqrt(3.0) / 4.0,
        "vertices_per_cell": 8,
    },
}


def _pick_tile_dims(N: int, box: np.ndarray, lattice_key: str,
                    strict: bool = False) -> Tuple[int, int, int, int]:
    """Pick (nx, ny, nz) tiling factors so that vertices_per_cell * nx*ny*nz
    is as close to ``N`` as possible while keeping per-axis cell size
    roughly equal (matching the box aspect).

    Returns (nx, ny, nz, N_actual). Warns if N_actual != N; raises if
    ``strict``.
    """
    lat = _LATTICE_LIBRARY[lattice_key]
    n_per_cell = lat["vertices_per_cell"]
    cax, cay, caz = lat["cell_aspect"]

    target_cells = max(1, int(round(N / n_per_cell)))

    V_unit = cax * cay * caz
    V_box = float(box[0] * box[1] * box[2])
    a = (V_box / (target_cells * V_unit)) ** (1.0 / 3.0)
    nx = max(1, int(round(box[0] / (a * cax))))
    ny = max(1, int(round(box[1] / (a * cay))))
    nz = max(1, int(round(box[2] / (a * caz))))

    # Adjust to hit target_cells exactly.
    safety = 0
    while nx * ny * nz != target_cells and safety < 100:
        ratios = (nx * cax / box[0], ny * cay / box[1], nz * caz / box[2])
        if nx * ny * nz < target_cells:
            idx = int(np.argmin(ratios))
            if idx == 0:
                nx += 1
            elif idx == 1:
                ny += 1
            else:
                nz += 1
        else:
            idx = int(np.argmax(ratios))
            if idx == 0 and nx > 1:
                nx -= 1
            elif idx == 1 and ny > 1:
                ny -= 1
            elif idx == 2 and nz > 1:
                nz -= 1
            else:
                break
        safety += 1

    N_actual = nx * ny * nz * n_per_cell
    if N_actual != N:
        msg = (
            f"crystal_seed_network: requested N={N} not exactly tilable "
            f"with lattice '{lattice_key}' (vertices_per_cell={n_per_cell}); "
            f"using N={N_actual} via tiling ({nx},{ny},{nz})."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
    return nx, ny, nz, N_actual


def crystal_seed_network(
    N: int,
    box: Union[float, Tuple[float, float, float], np.ndarray],
    d0: float,
    rng: np.random.Generator,
    lattice: str = "srs",
    jitter_sigma: float = 0.10,
    strict_tiling: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build a periodic Z=3 crystalline seed network.

    Returns
    -------
    positions : (N_actual, 3) float64
        Vertex positions wrapped into [-L/2, L/2]^3, plus optional
        Gaussian jitter of std ``jitter_sigma * d0`` per Cartesian
        component (added after the Z=3 / connectivity invariants are
        checked).
    edges : (E, 2) int64
        E = 3 * N_actual / 2 unique edges, lexicographically sorted.
    meta : dict
        Keys: ``tile``, ``lattice_constant``, ``seed_bond_length``,
        ``lattice``, ``N_actual``.

    Notes
    -----
    Seed bond length is set by tiling geometry, not by ``d0``: after
    tiling, the mean bond length is ``a * target_bond_frac`` where
    ``a = box[0] / nx`` (e.g. ``a * sqrt(2)/4`` for the default
    ``srs`` gyroid net).
    The caller's initial L-BFGS pulls bonds toward ``d0`` via the
    Keating ``f1`` term; if ``seed_bond_length / d0`` is far from 1
    the relax may distort the lattice substantially, so this routine
    warns when ``|ratio - 1| > 0.2``.

    The default ``srs`` lattice is the single-network gyroid parent used
    for Z=3 gyroid WWW evolution: 8 vertices per cubic cell, equal bond
    lengths, and 120-degree bond angles. ``diamond3`` remains available
    for diagnostics; it is cubic diamond (Z=4) with a perfect matching
    of 4 bonds removed per cubic cell to drop to Z=3, so its bond angles
    are tetrahedral (109.47 degrees) rather than Sellers's 120-degree
    f2 target.
    """
    if lattice not in _LATTICE_LIBRARY:
        raise ValueError(
            f"Unknown lattice '{lattice}'. Available: "
            f"{list(_LATTICE_LIBRARY.keys())}"
        )
    lat = _LATTICE_LIBRARY[lattice]
    box_arr = coerce_box(box)
    nx, ny, nz, N_actual = _pick_tile_dims(N, box_arr, lattice, strict_tiling)
    sites_frac = lat["sites_frac"]
    bonds_template = lat["bonds"]
    n_per_cell = sites_frac.shape[0]

    ax = box_arr[0] / nx
    ay = box_arr[1] / ny
    az = box_arr[2] / nz
    seed_bond_length = ax * lat["target_bond_frac"]
    if abs(seed_bond_length / d0 - 1.0) > 0.2:
        warnings.warn(
            f"crystal_seed_network: seed bond length "
            f"{seed_bond_length:.3g} differs from d0={d0:.3g} by more "
            f"than 20%. Initial relax will distort the lattice; consider "
            f"adjusting (num_vertices, bounds_microns, edge_length) so "
            f"the seed bond length matches d0 within ~20%.",
            stacklevel=2,
        )

    # Tile positions.
    positions = np.empty((N_actual, 3), dtype=np.float64)
    flat_idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([ix * ax, iy * ay, iz * az])
                for s in range(n_per_cell):
                    positions[flat_idx, 0] = base[0] + sites_frac[s, 0] * ax
                    positions[flat_idx, 1] = base[1] + sites_frac[s, 1] * ay
                    positions[flat_idx, 2] = base[2] + sites_frac[s, 2] * az
                    flat_idx += 1
    # Centre and wrap into canonical box.
    positions -= box_arr / 2.0
    positions -= box_arr * np.round(positions / box_arr)

    # Build edges.
    n_bonds_template = bonds_template.shape[0]
    edges_buf = np.empty((nx * ny * nz * n_bonds_template, 2), dtype=np.int64)
    edge_count = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                src_cell_base = ((ix * ny + iy) * nz + iz) * n_per_cell
                for b in range(n_bonds_template):
                    vi = int(bonds_template[b, 0])
                    vj = int(bonds_template[b, 1])
                    dx = int(bonds_template[b, 2])
                    dy = int(bonds_template[b, 3])
                    dz = int(bonds_template[b, 4])
                    dst_ix = (ix + dx) % nx
                    dst_iy = (iy + dy) % ny
                    dst_iz = (iz + dz) % nz
                    src_atom = src_cell_base + vi
                    dst_atom = ((dst_ix * ny + dst_iy) * nz + dst_iz) * n_per_cell + vj
                    if src_atom == dst_atom:
                        raise RuntimeError(
                            f"crystal_seed_network: bond template "
                            f"(vi={vi}, vj={vj}, ofs=({dx},{dy},{dz})) "
                            f"is a self-loop at cell ({ix},{iy},{iz}); "
                            f"lattice '{lattice}' is invalid (too small "
                            f"a tiling for the given offset?)."
                        )
                    if src_atom < dst_atom:
                        edges_buf[edge_count, 0] = src_atom
                        edges_buf[edge_count, 1] = dst_atom
                    else:
                        edges_buf[edge_count, 0] = dst_atom
                        edges_buf[edge_count, 1] = src_atom
                    edge_count += 1
    edges = np.unique(edges_buf[:edge_count], axis=0)

    # Invariants.
    expected_E = (3 * N_actual) // 2
    if edges.shape[0] != expected_E:
        raise RuntimeError(
            f"crystal_seed_network: built {edges.shape[0]} edges "
            f"(expected {expected_E} = 3N/2). Lattice '{lattice}' "
            f"bonds list has duplicates or missing entries."
        )
    deg = np.zeros(N_actual, dtype=np.int64)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    if not np.all(deg == 3):
        bad = int(np.flatnonzero(deg != 3)[0])
        raise RuntimeError(
            f"crystal_seed_network: vertex {bad} has degree {deg[bad]} "
            f"(expected 3). Lattice '{lattice}' is not Z=3."
        )
    if not is_connected(N_actual, edges):
        raise RuntimeError(
            f"crystal_seed_network: lattice '{lattice}' is not "
            f"3D-connected at tiling ({nx},{ny},{nz})."
        )

    # Position jitter (after invariant checks).
    if jitter_sigma > 0:
        sigma = jitter_sigma * d0
        positions = positions + rng.normal(0.0, sigma, size=positions.shape)
        positions -= box_arr * np.round(positions / box_arr)

    meta = {
        "tile": (nx, ny, nz),
        "lattice_constant": (ax, ay, az),
        "seed_bond_length": seed_bond_length,
        "lattice": lattice,
        "N_actual": N_actual,
    }
    return positions, edges, meta


# --------------------------------------------------------------------------- #
# Random Z=3 seed network — BM2000 § II.A loop expansion
# --------------------------------------------------------------------------- #
# This is the seed recipe Sellers (Nat. Commun. 8, 14439, 2017) literally
# cites for the random-network start ("simulated annealing of a random
# network", supplement Methods, refs [13] Vink 2001 and [14] Mousseau-Barkema
# 2001). It produces a connected Z=3 network from random points + a
# Hamiltonian-cycle scaffold + loop expansion, with the BM2000
# minimum-separation constraint preventing the near-collinear vertex
# clusters that the Sellers Eq. 2 energy cannot spontaneously remove (it has
# no non-bonded repulsion).
#
# BM2000's published algorithm targets Z=4 amorphous Si; BM's literal
# loop-expansion move (replace bond BC with AB+AC) is intrinsically a Z=4
# recipe, so for Z=3 we reproduce BM's *properties* rather than its exact
# move:
#   - Place N points with min-separation `min_separation_frac * d0` (BM2000:
#     2.3 Å vs Si–Si d=2.35 Å, i.e. ≈ 0.979).  [faithful]
#   - Build a Hamiltonian cycle through all N points using nearest-neighbour
#     traversal. Every vertex starts at degree 2.  [a Z=3 scaffold standing
#     in for BM's "loop visiting all atoms"; not the literal BC->AB+AC move]
#   - Loop expansion to Z=3: add N/2 more edges by direct pairing of deg-2
#     vertices within a growing `rc`, REJECTING any pairing that would close
#     a 3- or 4-ring (girth >= 5). This re-imports BM2000 §II.B's "no
#     four-membered rings" rule that the bare scaffold otherwise lacked;
#     5-rings are allowed (a-Si CRNs are 5-ring-rich) and the WWW anneal
#     shapes the final 5-vs-6 distribution. We grow `rc` when pairing stalls.
#   - BM2000's close-pair/short-bond cleanup is a POST-relaxation step
#     ("in the beginning of this first quench"), so it is intentionally NOT
#     done here; the downstream L-BFGS + WWW anneal handle it.
#   - Final invariants identical to `crystal_seed_network`: deg==3
#     everywhere, `is_connected`, edges canonicalised; plus zero triangles.
def random_seed_network_bm2000(
    N: int,
    box: Union[float, Tuple[float, float, float], np.ndarray],
    d0: float,
    rng: np.random.Generator,
    min_separation_frac: float = 0.98,
    rc_start_frac: float = 1.30,
    rc_grow_frac: float = 0.05,
    rc_max_frac: float = 6.00,
    max_outer_passes: int = 10_000,
    long_bond_frac: float = 1.5,
    max_2opt_passes: int = 400,
    twoopt_k: int = 24,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build a random Z=3 seed network via BM2000 (Phys. Rev. B 62, 4985) § II.A.

    Algorithm (Z=3 adaptation of BM2000):

    1. **Placement.** Poisson-disk: place ``N`` vertices in
       ``[-L/2, L/2]^3`` rejecting candidates within
       ``min_separation_frac * d0`` of an existing vertex. If placement
       deadlocks, lower the min-separation by 0.02 until 0.85, raising
       ``RuntimeError`` beyond that.
    2. **Hamiltonian cycle.** Greedy nearest-neighbour traversal under
       PBC, starting from vertex 0. Every vertex ends at degree 2. This
       is BM2000's "loop visiting all atoms" starting point.
    3. **Loop expansion to Z=3.** Pair deg-2 vertices: for each
       under-coordinated vertex ``i`` in random order, bond it to its
       nearest deg-2 partner ``j`` within ``rc`` such that ``(i, j)`` is
       not already an edge AND closes no 3- or 4-ring (girth >= 5,
       re-importing BM2000 §II.B's no-four-ring rule). When no progress is
       made at the current ``rc``, grow ``rc`` by ``rc_grow_frac * d0`` and
       retry. Continues until every vertex has degree 3 or ``rc`` exceeds
       ``rc_max_frac * d0``.
    4. **Invariants.** Same as ``crystal_seed_network``: ``deg == 3``
       everywhere, ``is_connected``, edges canonicalised ``(min, max)``
       and lex-sorted.

    Returns
    -------
    positions : (N, 3) float64
    edges : (E, 2) int64  with E = 3N/2
    meta : dict
        Keys ``N_actual``, ``lattice``, ``seed_bond_length`` (mean PBC
        edge length), ``rc_final``, ``outer_passes``,
        ``min_separation_frac``, ``cycle_max_step``.
    """
    if N % 2 != 0:
        raise ValueError(
            f"random_seed_network_bm2000: N must be even (got {N}); "
            f"3-regular graph needs 2E = 3N with N even."
        )
    box_arr = coerce_box(box)

    # --- Step 1: Poisson-disk placement ---------------------------------- #
    placement_min_frac = float(min_separation_frac)
    positions: Optional[np.ndarray] = None
    while placement_min_frac >= 0.85:
        positions = _poisson_disk_pbc(
            N, box_arr, placement_min_frac * d0, rng,
            max_tries=max(50_000, 200 * N),
        )
        if positions is not None:
            break
        if verbose:
            print(
                f"[bm2000 seed] placement at min_sep={placement_min_frac:.3f}*d0 "
                f"deadlocked; lowering by 0.02"
            )
        placement_min_frac -= 0.02
    if positions is None:
        raise RuntimeError(
            f"random_seed_network_bm2000: could not place {N} vertices in "
            f"box {box_arr.tolist()} even at min_separation 0.85*d0={0.85 * d0:.3g}. "
            f"Lower N or enlarge the box."
        )
    if verbose:
        print(
            f"[bm2000 seed] placed {N} vertices at "
            f"min_sep={placement_min_frac:.3f}*d0"
        )

    # Wrap into canonical box (idempotent guard).
    positions = positions - box_arr * np.round(positions / box_arr)

    nbr_sets: list[set[int]] = [set() for _ in range(N)]
    edges_list: list[Tuple[int, int]] = []

    def _add_edge(a: int, b: int) -> None:
        u, v = (a, b) if a < b else (b, a)
        edges_list.append((u, v))
        nbr_sets[a].add(b)
        nbr_sets[b].add(a)

    def _degree(i: int) -> int:
        return len(nbr_sets[i])

    def _remove_edge(a: int, b: int) -> None:
        nbr_sets[a].discard(b)
        nbr_sets[b].discard(a)

    def _bond(a: int, b: int) -> None:
        nbr_sets[a].add(b)
        nbr_sets[b].add(a)

    def _connected_now() -> bool:
        seen = np.zeros(N, dtype=bool)
        stack = [0]
        seen[0] = True
        cnt = 1
        while stack:
            uu = stack.pop()
            for ww in nbr_sets[uu]:
                if not seen[ww]:
                    seen[ww] = True
                    cnt += 1
                    stack.append(ww)
        return cnt == N

    def _dist(a: int, b: int) -> float:
        return float(np.linalg.norm(
            pbc_displacement(positions[a] - positions[b], box_arr)
        ))

    def _would_make_short_ring(a: int, b: int, max_ring: int) -> bool:
        """True if bonding (a, b) would close a ring of size <= max_ring.

        BM2000 (Phys. Rev. B 62, 4985) §II.B disallows 4-membered rings;
        3-membered rings (triangles) never occur in a CRN (the reference
        gyroid is girth-6). Only the 2-neighbourhood is inspected, O(deg^2)
        <= 9 ops for Z<=3. ``max_ring=3`` forbids only triangles; ``4`` also
        forbids squares. 5-rings are intentionally allowed (a-Si CRNs are
        5-ring-rich; the WWW anneal shapes the final 5-vs-6 distribution).
        """
        na = nbr_sets[a]
        nb = nbr_sets[b]
        if na & nb:
            return True                       # common neighbour -> 3-ring
        if max_ring >= 4:
            for x in na:
                if nbr_sets[x] & nb:
                    return True               # a-x-y-b path -> 4-ring
        return False

    def _tree() -> cKDTree:
        pos_shift = positions + box_arr / 2.0
        pos_shift = np.clip(pos_shift, 0.0, box_arr - 1e-12)
        return cKDTree(pos_shift, boxsize=box_arr)

    # --- Step 2: Hamiltonian cycle (nearest-neighbour traversal) -------- #
    # Pick a random starting vertex; greedily walk to the nearest unvisited
    # vertex under PBC; close the cycle by connecting the last vertex back
    # to the start. This gives every vertex deg==2.
    tree = _tree()
    start = int(rng.integers(0, N))
    visited = np.zeros(N, dtype=bool)
    cycle_order = np.empty(N, dtype=np.int64)
    cycle_order[0] = start
    visited[start] = True
    cycle_max_step = 0.0
    current = start
    pos_shift_all = positions + box_arr / 2.0
    pos_shift_all = np.clip(pos_shift_all, 0.0, box_arr - 1e-12)
    for step in range(1, N):
        # Query enough neighbours to find at least one unvisited.
        k_query = min(N, max(8, step // 64 + 8))
        while True:
            dists, idxs = tree.query(pos_shift_all[current], k=k_query)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            # unvisited candidates among the k nearest (tree.query returns them
            # in ascending-distance order, so cand_uv[0] is the nearest).
            cand_uv = [(float(d_ij), int(j)) for d_ij, j in zip(dists, idxs)
                       if int(j) != current and not visited[int(j)]]
            if cand_uv:
                if _RANDOM_SEED_CONSTRUCT:
                    # BM2000-faithful: RANDOM among nearby unvisited (kept within
                    # the k nearest so edges stay local/bounded), NOT greedy-nearest.
                    d_nxt, nxt = cand_uv[int(rng.integers(0, len(cand_uv)))]
                else:
                    d_nxt, nxt = cand_uv[0]  # greedy nearest (original)
                break
            if k_query >= N:
                raise RuntimeError(
                    "random_seed_network_bm2000: cycle traversal could "
                    "not find an unvisited vertex; geometry is degenerate."
                )
            k_query = min(N, k_query * 2)
        cycle_order[step] = nxt
        visited[nxt] = True
        if d_nxt > cycle_max_step:
            cycle_max_step = d_nxt
        _add_edge(current, nxt)
        current = nxt
    # Close the cycle.
    d_close = float(np.linalg.norm(
        pbc_displacement(positions[start] - positions[current], box_arr)
    ))
    if d_close > cycle_max_step:
        cycle_max_step = d_close
    _add_edge(current, start)
    if verbose:
        print(
            f"[bm2000 seed] Hamiltonian cycle built: "
            f"max_step={cycle_max_step:.3f} ({cycle_max_step/d0:.2f}*d0)"
        )

    # All vertices should be deg==2 now.
    deg = np.array([_degree(i) for i in range(N)], dtype=np.int64)
    if not (deg == 2).all():
        bad = int(np.flatnonzero(deg != 2)[0])
        raise RuntimeError(
            f"random_seed_network_bm2000: Hamiltonian cycle did not give "
            f"deg=2 everywhere (vertex {bad} has deg {deg[bad]})."
        )

    # --- Step 3: loop expansion to Z=3 ---------------------------------- #
    # Global ascending-distance greedy matching of the deg-2 vertices: collect
    # every candidate pair within rc, sort by PBC distance, and accept the
    # shortest valid (deg<3, not already bonded, girth >= 5) pair first. This
    # is far better than per-vertex greedy -- it minimises long chords by
    # construction, so rc rarely has to grow and few stragglers remain.
    rc = rc_start_frac * d0
    outer_passes = 0

    while outer_passes < max_outer_passes:
        outer_passes += 1
        progress = False

        if (deg == 3).all():
            break

        # All vertex pairs within rc (PBC); keep those with both ends deg<3.
        cand = tree.query_pairs(r=rc, output_type="ndarray")
        if cand.size:
            is_d2 = deg < 3
            cand = cand[is_d2[cand[:, 0]] & is_d2[cand[:, 1]]]
        if cand.size:
            if _RANDOM_SEED_CONSTRUCT:
                # BM2000-faithful: RANDOM order among the in-rc candidate pairs
                # (still bounded by rc, but not shortest-first which biases the
                # seed's ring/topology spectrum toward spatial locality).
                order = rng.permutation(cand.shape[0])
            else:
                dvec = pbc_displacement(
                    positions[cand[:, 0]] - positions[cand[:, 1]], box_arr
                )
                order = np.argsort(np.linalg.norm(dvec, axis=1))
            for idx in order:
                u = int(cand[idx, 0])
                v = int(cand[idx, 1])
                if _degree(u) >= 3 or _degree(v) >= 3:
                    continue  # consumed earlier this pass
                if v in nbr_sets[u]:
                    continue
                if _would_make_short_ring(u, v, 4):
                    continue  # girth >= 5: reject 3- and 4-rings (BM2000 §II.B)
                _add_edge(u, v)
                deg[u] += 1
                deg[v] += 1
                progress = True

        if (deg == 3).all():
            break

        if not progress:
            if rc >= rc_max_frac * d0:
                break  # fall through to the force-pair fallback below
            rc += rc_grow_frac * d0
            if verbose:
                print(
                    f"[bm2000 seed] outer={outer_passes} grew rc to "
                    f"{rc:.3f} ({rc/d0:.2f}*d0) "
                    f"(deg_lt3={int((deg < 3).sum())})"
                )

    # Force-pair fallback: if any deg-2 vertices remain (typically 0-4), pair
    # them with each other irrespective of rc. The PBC max distance in a
    # cubic box is sqrt(3)*L/2, which is always reachable. This guarantees
    # termination at the cost of a few very-long bonds; the subsequent
    # L-BFGS and burn-in shrink them back to ~d0.
    leftover = np.flatnonzero(deg < 3).tolist()
    if leftover:
        if verbose:
            print(
                f"[bm2000 seed] {len(leftover)} stragglers; "
                f"force-pairing them across PBC"
            )
        # Greedily pair the closest two stragglers, repeat. Prefer pairings
        # that close no 3- or 4-ring (girth >= 5); if none exist among the
        # remaining stragglers, relax to allow a 4-ring but NEVER a triangle
        # (BM2000 §II.B tolerates the few 4-rings, which the WWW anneal then
        # removes). 3-rings are forbidden unconditionally.
        def _augment_pair(i: int, j: int) -> bool:
            """Raise two deg-2 stragglers (i, j) to deg-3 when a direct (i, j)
            bond would close a triangle. Augmenting move: remove an existing
            edge (x, y) near them and add {(i, x), (j, y)} or {(i, y), (j, x)}.
            This keeps x, y at deg-3 and pushes no deficit forward, while
            respecting girth >= 5 and connectivity. Returns True on success."""
            cand: set[int] = set()
            for src in (i, j):
                _, idxs = tree.query(pos_shift_all[src], k=min(N, 32))
                cand.update(int(t) for t in np.atleast_1d(idxs))
            cand.discard(i)
            cand.discard(j)
            best_score = float("inf")
            best_move = None  # (x, y, (a1, b1), (a2, b2))
            for x in cand:
                for y in tuple(nbr_sets[x]):
                    if len({i, j, x, y}) != 4:
                        continue
                    for (a1, b1), (a2, b2) in (
                        ((i, x), (j, y)),
                        ((i, y), (j, x)),
                    ):
                        if b1 in nbr_sets[a1] or b2 in nbr_sets[a2]:
                            continue
                        score = _dist(a1, b1) + _dist(a2, b2)
                        if score >= best_score:
                            continue
                        # Transactional girth check (old edge removed).
                        _remove_edge(x, y)
                        ok = not _would_make_short_ring(a1, b1, 4)
                        if ok:
                            _bond(a1, b1)
                            ok = not _would_make_short_ring(a2, b2, 4)
                            _remove_edge(a1, b1)
                        _bond(x, y)
                        if ok:
                            best_score = score
                            best_move = (x, y, (a1, b1), (a2, b2))
            if best_move is None:
                return False
            x, y, (a1, b1), (a2, b2) = best_move
            _remove_edge(x, y)
            _bond(a1, b1)
            _bond(a2, b2)
            if not _connected_now():
                # Revert and reject this move.
                _remove_edge(a1, b1)
                _remove_edge(a2, b2)
                _bond(x, y)
                return False
            deg[i] += 1
            deg[j] += 1  # x, y net unchanged
            if verbose:
                print(
                    f"[bm2000 seed] augment-relinked stragglers ({i}, {j}) "
                    f"by breaking edge ({x}, {y})"
                )
            return True

        while len(leftover) >= 2:
            best = (-1, -1, float("inf"))
            for max_ring in (4, 3):
                for ii, a in enumerate(leftover):
                    for b in leftover[ii + 1:]:
                        if b in nbr_sets[a]:
                            continue
                        if _would_make_short_ring(a, b, max_ring):
                            continue
                        d_ab = float(np.linalg.norm(
                            pbc_displacement(
                                positions[a] - positions[b], box_arr
                            )
                        ))
                        if d_ab < best[2]:
                            best = (a, b, d_ab)
                if best[0] >= 0:
                    break  # found a legal pair at this girth level
            if best[0] < 0:
                # Every remaining straggler pair would close a triangle.
                # Augment-relink the first two stragglers via a broken edge
                # instead of emitting a 3-ring or re-seeding.
                i, j = int(leftover[0]), int(leftover[1])
                if not _augment_pair(i, j):
                    raise RuntimeError(
                        f"random_seed_network_bm2000: stragglers "
                        f"{leftover} cannot be paired or augment-relinked "
                        f"without a triangle; retry with a different `seed`."
                    )
                leftover = [v for v in leftover if deg[v] < 3]
                continue
            a, b, d_ab = best
            _add_edge(a, b)
            deg[a] += 1
            deg[b] += 1
            if verbose:
                print(
                    f"[bm2000 seed] force-paired ({a}, {b}) at "
                    f"d={d_ab:.3f} ({d_ab/d0:.2f}*d0)"
                )
            leftover = [v for v in leftover if deg[v] < 3]
        # If a single leftover remains (odd parity — impossible for N even,
        # but guard anyway), abort.
        if leftover:
            raise RuntimeError(
                f"random_seed_network_bm2000: odd straggler {leftover} "
                f"after pairing; check N is even."
            )

    if not (deg == 3).all():
        bad = int(np.flatnonzero(deg != 3)[0])
        raise RuntimeError(
            f"random_seed_network_bm2000: vertex {bad} has degree {deg[bad]} "
            f"(expected 3) after force-pair fallback."
        )

    # --- Step 3b: 2-opt long-bond cleanup ------------------------------- #
    # The Hamiltonian closing edge and rc-grown / force-paired chords can
    # leave bonds several d0 long. A long bond is a *topological* defect: no
    # position relaxation can shorten it, because Sellers Eq. 2 has no
    # non-bonded repulsion -- contracting it would drag the endpoints through
    # neighbouring vertices and collapse them (the min_non_bonded < 0.4*d0
    # hard fail in generate_lsu_network). Remove long bonds at the source
    # with degree-preserving 2-opt swaps: replace a long edge (u, v) and a
    # nearby edge (x, y) with the shorter reconnection {(u, x), (v, y)} or
    # {(u, y), (v, x)}, keeping deg==3 everywhere, girth >= 5 (no 3-/4-rings,
    # BM2000 §II.B) and global connectivity. Longest-first, first-improvement,
    # iterated until no edge exceeds long_bond_frac*d0 or no improving swap
    # exists. Each accepted swap strictly shortens the worst bond, so the
    # loop terminates.
    L_long = float(long_bond_frac) * d0
    tree2 = _tree()

    def _try_swap(u: int, v: int, d_uv: float) -> bool:
        """Apply the best valid 2-opt that shortens edge (u, v). Greedy
        first-improvement: returns True and leaves nbr_sets mutated on the
        first accepted swap; False if none found."""
        cand_verts: set[int] = set()
        for src in (u, v):
            _, idxs = tree2.query(pos_shift_all[src], k=min(N, twoopt_k))
            cand_verts.update(int(t) for t in np.atleast_1d(idxs))
        cand_verts.discard(u)
        cand_verts.discard(v)
        for x in cand_verts:
            for y in tuple(nbr_sets[x]):
                if len({u, v, x, y}) != 4:
                    continue
                # Two degree-preserving reconnections of {u,v,x,y}.
                for (a1, b1), (a2, b2) in (
                    ((u, x), (v, y)),
                    ((u, y), (v, x)),
                ):
                    if b1 in nbr_sets[a1] or b2 in nbr_sets[a2]:
                        continue  # would duplicate an existing edge
                    new_max = max(_dist(a1, b1), _dist(a2, b2))
                    if new_max >= d_uv:
                        continue  # must strictly shorten the long bond
                    # Transactional validity check: remove old, test girth on
                    # the new pair, then connectivity; revert on any failure.
                    _remove_edge(u, v)
                    _remove_edge(x, y)
                    ok = not _would_make_short_ring(a1, b1, 4)
                    if ok:
                        _bond(a1, b1)
                        ok = not _would_make_short_ring(a2, b2, 4)
                        if ok:
                            _bond(a2, b2)
                            if _connected_now():
                                return True  # accept; leave applied
                            _remove_edge(a2, b2)
                        _remove_edge(a1, b1)
                    # revert
                    _bond(u, v)
                    _bond(x, y)
        return False

    n_2opt_swaps = 0
    for _pass in range(max_2opt_passes):
        edge_lengths = sorted(
            ((_dist(i, j), i, j) for i in range(N) for j in nbr_sets[i] if i < j),
            reverse=True,
        )
        if not edge_lengths or edge_lengths[0][0] <= L_long:
            break
        progressed = False
        for d_uv, u, v in edge_lengths:
            if d_uv <= L_long:
                break
            if _try_swap(u, v, d_uv):
                n_2opt_swaps += 1
                progressed = True
                break  # re-sort after each accepted swap
        if not progressed:
            break  # no improving swap for any long edge

    # Rebuild the edge list from nbr_sets (Step 3b mutated the adjacency in
    # place; edges_list is now stale).
    edges_list = [(i, j) for i in range(N) for j in nbr_sets[i] if i < j]
    if verbose:
        residual = sorted(
            (_dist(i, j) for (i, j) in edges_list), reverse=True
        )
        print(
            f"[bm2000 seed] 2-opt cleanup: {n_2opt_swaps} swaps, "
            f"max bond now {residual[0] / d0:.2f}*d0 "
            f"(target <= {long_bond_frac:.2f}*d0)"
        )

    # --- Step 4: invariants --------------------------------------------- #
    edges_arr = np.array(edges_list, dtype=np.int64)
    edges_arr = np.unique(np.sort(edges_arr, axis=1), axis=0)
    if edges_arr.shape[0] != (3 * N) // 2:
        raise RuntimeError(
            f"random_seed_network_bm2000: built {edges_arr.shape[0]} edges "
            f"(expected {(3 * N) // 2})."
        )
    if not is_connected(N, edges_arr):
        raise RuntimeError(
            "random_seed_network_bm2000: final network is disconnected."
        )

    # Girth-guard verification: the loop-expansion/fallback guard forbids
    # triangles everywhere, so a clean build has zero 3-rings (BM2000 §II.B;
    # the reference gyroid is girth-6). A non-zero count is a guard bug.
    n_triangles = 0
    for a, b in edges_arr:
        n_triangles += len(nbr_sets[int(a)] & nbr_sets[int(b)])
    n_triangles //= 3
    if n_triangles != 0:
        raise RuntimeError(
            f"random_seed_network_bm2000: built {n_triangles} triangle(s); "
            f"the girth guard failed (internal bug)."
        )

    # Compute mean bond length (PBC minimum image).
    bond_d = pbc_displacement(
        positions[edges_arr[:, 1]] - positions[edges_arr[:, 0]], box_arr
    )
    bond_L = float(np.linalg.norm(bond_d, axis=1).mean())

    meta = {
        "N_actual": N,
        "lattice": "random_bm2000",
        "seed_bond_length": bond_L,
        "rc_final": float(rc),
        "outer_passes": int(outer_passes),
        "min_separation_frac": float(placement_min_frac),
        "cycle_max_step": float(cycle_max_step),
        "n_triangles": int(n_triangles),
    }
    return positions, edges_arr, meta


def soft_start_seed_relax(
    positions: np.ndarray,
    edges: np.ndarray,
    box: Union[float, Tuple[float, float, float], np.ndarray],
    d0: float,
    *,
    r_rep_frac: float = 0.9,
    k_rep: float = 1.0,
    n_outer: int = 12,
    inner_iter: int = 60,
    target_min_nb_frac: float = 0.6,
    bond_max_frac: float = 1.6,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Spread a raw random seed before the (repulsion-free) Sellers relax.

    The BM2000 loop-expansion can leave bonds several ``d0`` long (the
    girth->=5 pairing grows ``rc`` up to its cap, and the straggler fallback
    stitches pairs across the box). The Sellers Eq. 2 energy has **no
    non-bonded repulsion** (see the module note above
    ``random_seed_network_bm2000``), so the first L-BFGS relax contracts
    those long bonds straight *through* neighbouring vertices and collapses
    them into near-coincident clusters -- the ``min_non_bonded < 0.4*d0``
    hard fail in ``generate_lsu_network``.

    This pre-relax uses a self-contained potential -- harmonic bond springs
    toward ``d0`` plus a one-sided soft-sphere repulsion on non-bonded pairs
    closer than ``r_rep_frac * d0`` -- to contract the long bonds *without*
    letting vertices cross. It mirrors the repulsive/Keating equilibration
    that Sellers's cited random-seed refs (Vink 2001; Mousseau-Barkema 2001)
    apply before WWW. It runs ONLY on the seed; the downstream relax and WWW
    anneal still see the pure Eq. 2 functional, so the production protocol is
    unchanged.

    The non-bonded pair list is rebuilt each outer pass (cKDTree under PBC)
    so pairs that newly close as long bonds contract are caught; each inner
    L-BFGS solve sees a frozen pair list (a smooth objective).

    Returns ``(positions, info)`` where ``info`` reports the min non-bonded
    separation before/after, the final max bond length, and the outer-pass
    count.
    """
    box_arr = coerce_box(box)
    N = positions.shape[0]
    pos = positions.astype(np.float64).copy()
    e0 = edges[:, 0].astype(np.int64)
    e1 = edges[:, 1].astype(np.int64)
    edge_set = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in edges}
    r_rep = float(r_rep_frac) * float(d0)
    target_min_nb = float(target_min_nb_frac) * float(d0)

    def _tree(p: np.ndarray) -> cKDTree:
        shift = np.clip(p + box_arr / 2.0, 0.0, box_arr - 1e-12)
        return cKDTree(shift, boxsize=box_arr)

    def _min_non_bonded(p: np.ndarray) -> float:
        tree = _tree(p)
        for r_q in (0.7 * d0, 1.2 * d0, 2.0 * d0):
            cand = tree.query_pairs(r=r_q, output_type="ndarray")
            if cand.size == 0:
                continue
            cand = np.sort(cand, axis=1)
            keep = np.array([(int(a), int(b)) not in edge_set for a, b in cand])
            cand = cand[keep]
            if cand.shape[0] == 0:
                continue
            dd = np.linalg.norm(
                pbc_displacement(p[cand[:, 0]] - p[cand[:, 1]], box_arr), axis=1
            )
            return float(dd.min())
        return float("inf")

    def _rep_pairs(p: np.ndarray) -> np.ndarray:
        cand = _tree(p).query_pairs(r=r_rep, output_type="ndarray")
        if cand.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        cand = np.sort(cand, axis=1)
        keep = np.array([(int(a), int(b)) not in edge_set for a, b in cand])
        return cand[keep]

    def _bond_lengths(p: np.ndarray) -> np.ndarray:
        return np.linalg.norm(pbc_displacement(p[e1] - p[e0], box_arr), axis=1)

    def _energy_grad(x: np.ndarray, rep: np.ndarray):
        p = x.reshape(N, 3)
        g = np.zeros((N, 3), dtype=np.float64)
        # Harmonic bonds toward d0.
        dvec = pbc_displacement(p[e1] - p[e0], box_arr)
        L = np.linalg.norm(dvec, axis=1)
        Lsafe = np.maximum(L, 1e-12)
        E = float(np.sum((L - d0) ** 2))
        coef = (2.0 * (L - d0) / Lsafe)[:, None] * dvec  # dE/dp[e1]
        np.add.at(g, e1, coef)
        np.add.at(g, e0, -coef)
        # One-sided soft-sphere repulsion on close non-bonded pairs.
        if rep.shape[0]:
            ra, rb = rep[:, 0], rep[:, 1]
            rvec = pbc_displacement(p[rb] - p[ra], box_arr)
            Lr = np.linalg.norm(rvec, axis=1)
            overlap = r_rep - Lr
            active = overlap > 0.0
            if np.any(active):
                E += float(k_rep * np.sum(overlap[active] ** 2))
                Lrsafe = np.maximum(Lr, 1e-12)
                gco = (-2.0 * k_rep * overlap / Lrsafe)[:, None] * rvec
                gco[~active] = 0.0
                np.add.at(g, rb, gco)
                np.add.at(g, ra, -gco)
        return E, g.reshape(-1)

    min_nb_before = _min_non_bonded(pos)
    outer = 0
    for outer in range(1, n_outer + 1):
        rep = _rep_pairs(pos)
        res = minimize(
            _energy_grad, pos.reshape(-1), args=(rep,), jac=True,
            method="L-BFGS-B", options={"maxiter": inner_iter, "gtol": 1e-8},
        )
        pos = res.x.reshape(N, 3)
        pos = pos - box_arr * np.round(pos / box_arr)
        cur = _min_non_bonded(pos)
        bond_max = float(_bond_lengths(pos).max())
        if verbose:
            print(
                f"[soft-start] outer={outer} min_nb={cur / d0:.3f}d0 "
                f"bond_max={bond_max / d0:.2f}d0 rep_pairs={rep.shape[0]}"
            )
        if cur >= target_min_nb and bond_max <= bond_max_frac * d0:
            break
    info = {
        "min_nb_before": float(min_nb_before),
        "min_nb_after": float(_min_non_bonded(pos)),
        "bond_max_after": float(_bond_lengths(pos).max()),
        "outer_passes": int(outer),
    }
    return pos, info


def settle_seed_with_repulsion(
    positions: np.ndarray,
    ctx: "_RelaxContext",
    edges: np.ndarray,
    box: Union[float, Tuple[float, float, float], np.ndarray],
    d0: float,
    *,
    r_rep_frac: float = 0.95,
    r_detect_frac: float = 1.6,
    lambda0: float = 4.0,
    lambda_end_frac: float = 0.3,
    n_stages: int = 8,
    inner_iter: int = 40,
    final_pure_iters: int = 0,
    target_min_nb_frac: float = 0.6,
    max_corrections: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Initial seed settle under Sellers Eq. 2 PLUS a decaying soft-sphere
    repulsion, annealed to zero.

    A pure repulsion-free relax of a random seed contracts every bond toward
    ``d0``; at scale (random degree-3 at this density wants mean bond ~1.3 d0,
    ~25% of bonds >1.5 d0) that contraction drags endpoints through neighbours
    and collapses dense regions (the ``min_non_bonded < 0.4*d0`` hard fail).
    Pre-spreading alone does not help -- the subsequent pure relax simply
    undoes it. The robust fix is to keep a non-bonded repulsion *inside* the
    settle and anneal it to zero, so the contraction reaches ``d0`` from a
    well-spaced basin rather than crushing coincidences. This mirrors the
    repulsive equilibration Sellers's own cited random-seed refs (Vink 2001;
    Mousseau-Barkema 2001) apply before WWW; the **WWW anneal still sees the
    pure Eq. 2 functional**, so emergent properties remain WWW-driven.

    Requires the JAX backend (``ctx.use_jax``) for the analytic Sellers
    gradient; callers fall back to ``soft_start_seed_relax`` + plain relax on
    the NumPy path.

    Schedule: ``n_stages`` L-BFGS solves with the repulsion weight ramped
    linearly ``lambda0 -> 0``; the non-bonded pair list is rebuilt each stage
    (so newly-closing pairs are caught) but frozen within a stage (smooth
    objective). A final ``final_pure_iters`` pure-Eq.2 polish (weight exactly
    0) lands the seed on a genuine Sellers minimum.

    Returns ``(positions, info)`` with ``info`` reporting min non-bonded
    separation before/after and final max bond length.
    """
    box_arr = coerce_box(box)
    N = positions.shape[0]
    # Activation floor (push apart below this) vs detection radius (track all
    # near pairs, so one frozen per-stage list still catches a pair that
    # collapses *during* a solve -- the failure mode of a tight-radius list).
    r_rep = float(r_rep_frac) * float(d0)
    r_detect = float(r_detect_frac) * float(d0)
    e0 = edges[:, 0].astype(np.int64)
    e1 = edges[:, 1].astype(np.int64)
    edge_set = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in edges}

    def _tree(p):
        shift = np.clip(p + box_arr / 2.0, 0.0, box_arr - 1e-12)
        return cKDTree(shift, boxsize=box_arr)

    def _rep_pairs(p):
        cand = _tree(p).query_pairs(r=r_detect, output_type="ndarray")
        if cand.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        cand = np.sort(cand, axis=1)
        keep = np.array([(int(a), int(b)) not in edge_set for a, b in cand])
        return cand[keep]

    def _min_non_bonded(p):
        tree = _tree(p)
        for r_q in (0.7 * d0, 1.2 * d0, 2.0 * d0):
            cand = tree.query_pairs(r=r_q, output_type="ndarray")
            if cand.size == 0:
                continue
            cand = np.sort(cand, axis=1)
            keep = np.array([(int(a), int(b)) not in edge_set for a, b in cand])
            cand = cand[keep]
            if cand.shape[0] == 0:
                continue
            dd = np.linalg.norm(
                pbc_displacement(p[cand[:, 0]] - p[cand[:, 1]], box_arr), axis=1
            )
            return float(dd.min())
        return float("inf")

    def _bond_max(p):
        return float(np.linalg.norm(
            pbc_displacement(p[e1] - p[e0], box_arr), axis=1
        ).max())

    def _rep_energy_grad(x, rep, k):
        """One-sided soft-sphere repulsion energy + flat gradient."""
        if not rep.shape[0] or k == 0.0:
            return 0.0, np.zeros_like(x)
        p = x.reshape(N, 3)
        g = np.zeros((N, 3), dtype=np.float64)
        ra, rb = rep[:, 0], rep[:, 1]
        rvec = pbc_displacement(p[rb] - p[ra], box_arr)
        Lr = np.linalg.norm(rvec, axis=1)
        overlap = r_rep - Lr
        active = overlap > 0.0
        if not np.any(active):
            return 0.0, g.reshape(-1)
        E = float(k * np.sum(overlap[active] ** 2))
        Lrsafe = np.maximum(Lr, 1e-12)
        gco = (-2.0 * k * overlap / Lrsafe)[:, None] * rvec
        gco[~active] = 0.0
        np.add.at(g, rb, gco)
        np.add.at(g, ra, -gco)
        return E, g.reshape(-1)

    if not ctx.use_jax:
        raise RuntimeError(
            "settle_seed_with_repulsion requires the JAX backend for the "
            "analytic Sellers gradient; use soft_start_seed_relax on NumPy."
        )

    min_nb_before = _min_non_bonded(positions)
    x = positions.reshape(-1).astype(np.float64).copy()
    lambda_end = lambda_end_frac * lambda0
    target = target_min_nb_frac * d0

    def _stage(xv, lam, maxit):
        rep = _rep_pairs(xv.reshape(N, 3)) if lam > 0.0 else np.empty((0, 2), np.int64)

        def _vg(xx, _rep=rep, _lam=lam):
            Es, gs = ctx.value_and_grad(xx)
            gs = np.asarray(gs, dtype=np.float64)
            Er, gr = _rep_energy_grad(xx, _rep, _lam)
            return float(Es) + Er, gs + gr

        res = minimize(_vg, xv, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxit, "gtol": 1e-8})
        xv = res.x
        xv = (xv.reshape(N, 3)
              - box_arr * np.round(xv.reshape(N, 3) / box_arr)).reshape(-1)
        if verbose:
            p = xv.reshape(N, 3)
            print(
                f"[settle-rep] lambda={lam:.2f}: "
                f"min_nb={_min_non_bonded(p) / d0:.3f}d0 "
                f"bond_max={_bond_max(p) / d0:.2f}d0 rep_pairs={rep.shape[0]}"
            )
        return xv

    # Ramp the repulsion weight lambda0 -> lambda_end over n_stages (a small
    # nonzero floor, NOT zero: a full pure-Eq.2 polish would re-contract the
    # network to the collapse edge and erase the spacing the repulsion just
    # bought). A brief lambda=0 polish then removes gross repulsion artifacts
    # without crushing the margin.
    for s in range(n_stages):
        lam = lambda0 + (lambda_end - lambda0) * s / max(1, n_stages - 1)
        x = _stage(x, lam, inner_iter)
    if final_pure_iters > 0:
        x = _stage(x, 0.0, final_pure_iters)

    # Self-correct: if the worst pair is still below target, re-spread (the
    # repulsion provably lifts min_nb) with a short ramp. Bounded.
    corrections = 0
    while _min_non_bonded(x.reshape(N, 3)) < target and corrections < max_corrections:
        corrections += 1
        if verbose:
            print(f"[settle-rep] correction {corrections}: re-spreading "
                  f"(min_nb below {target_min_nb_frac:.2f}*d0)")
        for s in range(3):
            lam = lambda0 * (1.0 - s / 3.0)
            x = _stage(x, max(lam, lambda_end), inner_iter)

    pos = x.reshape(N, 3)
    info = {
        "min_nb_before": float(min_nb_before),
        "min_nb_after": float(_min_non_bonded(pos)),
        "bond_max_after": float(_bond_max(pos)),
        "n_stages": int(n_stages),
        "corrections": int(corrections),
    }
    return pos, info


def _poisson_disk_pbc(
    N: int,
    box: np.ndarray,
    min_dist: float,
    rng: np.random.Generator,
    max_tries: int = 200_000,
) -> Optional[np.ndarray]:
    """Sequentially-rejected Poisson-disk placement under PBC.

    Returns ``None`` if it cannot place ``N`` points within ``max_tries``
    attempts; the caller is expected to retry with a smaller ``min_dist``.
    """
    positions = np.empty((N, 3), dtype=np.float64)
    placed = 0
    tries = 0
    # Lazy tree rebuild: rebuild when the placed count grows by a fixed factor.
    tree: Optional[cKDTree] = None
    rebuild_threshold = max(8, N // 64)
    last_built_at = 0

    while placed < N and tries < max_tries:
        tries += 1
        cand = (rng.random(3) - 0.5) * box
        if placed == 0:
            positions[placed] = cand
            placed += 1
            continue
        # Rebuild tree if stale.
        if tree is None or placed - last_built_at >= rebuild_threshold:
            pos_shift = positions[:placed] + box / 2.0
            pos_shift = np.clip(pos_shift, 0.0, box - 1e-12)
            tree = cKDTree(pos_shift, boxsize=box)
            last_built_at = placed
        cand_shift = cand + box / 2.0
        cand_shift = np.clip(cand_shift, 0.0, box - 1e-12)
        dist, _ = tree.query(cand_shift, k=1)
        # Account for the few points placed since the last rebuild that
        # are not yet in the tree:
        if last_built_at < placed:
            recent = positions[last_built_at:placed]
            r_diff = pbc_displacement(recent - cand, box)
            r_min = float(np.linalg.norm(r_diff, axis=1).min())
            dist = min(float(dist), r_min)
        if dist >= min_dist:
            positions[placed] = cand
            placed += 1

    if placed < N:
        return None
    return positions


def _voxel_density_std(positions: np.ndarray, box: np.ndarray,
                       ngrid: int = 4) -> float:
    """Std of vertex count in an ngrid^3 voxel grid over the box.

    Cheap surrogate for vertex-uniformity / void-clustering. Lower is
    more uniform; for the Sellers reference network ngrid=4 gives
    ~2.79. Used by ``topology_burn_in`` as a plateau-detector signal.
    """
    pos = positions - box * np.round(positions / box)
    fracs = (pos + box / 2.0) / box  # in [0, 1)
    cells = np.floor(fracs * ngrid).astype(np.int64) % ngrid
    counts = np.zeros((ngrid, ngrid, ngrid), dtype=np.int64)
    np.add.at(counts, (cells[:, 0], cells[:, 1], cells[:, 2]), 1)
    return float(counts.std())


_LOW_K_CACHE: Dict[int, np.ndarray] = {}


def _low_k_hkl(kmax: int) -> np.ndarray:
    """Integer reciprocal vectors with |hkl| <= kmax, excluding zero."""
    kmax = int(kmax)
    if kmax <= 0:
        return np.empty((0, 3), dtype=np.float64)
    cached = _LOW_K_CACHE.get(kmax)
    if cached is not None:
        return cached
    hkl = []
    k2_max = kmax * kmax
    for h in range(-kmax, kmax + 1):
        for k in range(-kmax, kmax + 1):
            for l in range(-kmax, kmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if h * h + k * k + l * l <= k2_max:
                    hkl.append((h, k, l))
    out = np.asarray(hkl, dtype=np.float64)
    _LOW_K_CACHE[kmax] = out
    return out


def low_k_structure_factor(
    positions: np.ndarray,
    box: np.ndarray,
    kmax: int = 2,
) -> float:
    """Mean low-k vertex structure factor S(k) over integer shells.

    This is a cheap PBC-aware large-scale homogeneity metric. The bonded
    Sellers / Keating-like energy controls local bond geometry but has no
    term suppressing long-wavelength density fluctuations; Hemmann/Saba
    likewise note that pore size and hyperuniformity are not directly
    controlled by the local strain energy. Penalising the lowest reciprocal
    modes during Metropolis acceptance discourages the corner/face voids seen
    in long WWW runs without changing the local L-BFGS geometry relax.
    """
    hkl = _low_k_hkl(kmax)
    if hkl.size == 0:
        return 0.0
    box = np.asarray(box, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    pos = pos - box * np.round(pos / box)
    phases = 2.0 * math.pi * (pos / box) @ hkl.T
    amp = np.exp(1j * phases).sum(axis=0)
    S = (np.abs(amp) ** 2) / max(1, pos.shape[0])
    return float(S.mean())


def _acceptance_objective(
    strain_energy: float,
    positions: np.ndarray,
    box: np.ndarray,
    uniformity_weight: float,
    uniformity_kmax: int,
) -> Tuple[float, float]:
    """Return (Metropolis objective, low-k S metric)."""
    if uniformity_weight <= 0.0:
        return float(strain_energy), 0.0
    s_low = low_k_structure_factor(positions, box, kmax=uniformity_kmax)
    return float(strain_energy) + float(uniformity_weight) * s_low, s_low




def compute_local_shell_mask(
    seed_vertices: np.ndarray,
    neighbors: np.ndarray,
    depth: int,
    N: int,
) -> np.ndarray:
    """Boolean (N,) mask of vertices within `depth` graph-edge steps of any seed.

    Used to restrict L-BFGS to a spatially local patch around a Stone-Wales
    move, per the Vink / Mousseau-Barkema relaxation refinement that the
    Sellers supplement (Methods, refs [13,14]) cites. Out-of-shell vertices
    are held fixed by gradient-masking inside `relax`.
    """
    mask = np.zeros(N, dtype=bool)
    mask[np.asarray(seed_vertices, dtype=np.int64)] = True
    for _ in range(int(depth)):
        true_idx = np.flatnonzero(mask)
        if true_idx.size == 0:
            break
        nbr_idx = neighbors[true_idx].reshape(-1)
        # Drop -1 sentinels just in case (degree-3 invariant should prevent any).
        nbr_idx = nbr_idx[nbr_idx >= 0]
        mask[nbr_idx] = True
    return mask


def build_angle_triples(neighbors: np.ndarray) -> np.ndarray:
    """For each vertex, return the 3 (vertex, nbr_a, nbr_b) angle triples.

    Returns shape (3*N, 3) — three triples per vertex, one per pair of
    neighbours (n0,n1), (n0,n2), (n1,n2).
    """
    N = neighbors.shape[0]
    triples = np.empty((3 * N, 3), dtype=np.int64)
    pair_idx = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    triples[:, 0] = np.repeat(np.arange(N), 3)
    triples[:, 1] = neighbors[:, pair_idx[:, 0]].reshape(-1)
    triples[:, 2] = neighbors[:, pair_idx[:, 1]].reshape(-1)
    return triples


def build_dihedral_quads(edges: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """For each edge (i, j), return (i, i1, i2, j, j1, j2) where
    i1, i2 are the two neighbours of i other than j (likewise j1, j2).

    Returns shape (E, 6) int64.
    """
    E = edges.shape[0]
    out = np.empty((E, 6), dtype=np.int64)
    for k in range(E):
        i, j = edges[k]
        i_nbrs = neighbors[i]
        j_nbrs = neighbors[j]
        i_others = i_nbrs[i_nbrs != j]
        j_others = j_nbrs[j_nbrs != i]
        if i_others.size != 2 or j_others.size != 2:
            raise ValueError(
                f"Edge ({i},{j}) does not connect two trivalent vertices."
            )
        out[k, 0] = i
        out[k, 1] = i_others[0]
        out[k, 2] = i_others[1]
        out[k, 3] = j
        out[k, 4] = j_others[0]
        out[k, 5] = j_others[1]
    return out


# --------------------------------------------------------------------------- #
# Energy — NumPy implementation (analytic, vectorised)
# --------------------------------------------------------------------------- #
_DIH_TARGET = 1.0 / 3.0  # |cos(70.53°)| = |cos(109.47°)| = 1/3

# Use the literal length-coupled Keating forms for f1/f2:
#   f1 = Σ(L²−d0²)²        (bond, vs the old simplified harmonic (L−d0)²)
#   f2 = Σ(r_ij·r_ik + d0²/2)²   (angle, vs the old normalized (cosθ+1/2)²)
# ADOPTED AS DEFAULT 2026-06-22: the old simplified forms made the energy
# ~6-8x too angle-dominated (bonds too soft), so its minimum spread bonds and
# lost the reference's hyperuniformity. The Keating forms (more faithful to
# Sellers' "Keating energy-like", Supp Eq.2) make the reference a stable fixed
# point of the anneal; f3/f4 are unchanged (faithful to Eq.3/4). Weights stay
# fixed (0.7/0.7/0.3/0.4). Set LSU_KEATING_F1F2=0 to revert to the old forms
# (regression/comparison only). See memory lsu-energy-keating-balance-fix.
_KEATING_F1F2 = os.environ.get("LSU_KEATING_F1F2", "1") == "1"

# EXPERIMENTAL (default OFF): BM2000-faithful RANDOM seed construction — random
# cycle traversal + random in-rc pairing instead of greedy-nearest / shortest-
# first selection (which inject the spatial-locality "crystallinity" bias that
# BM2000 §II.A explicitly avoids: "absolutely no trace of crystallinity"). This
# is a CONNECTIVITY change only; it does NOT alter Poisson-disk placement or the
# seed's intrinsic vertex S(k0). Under multi-agent verification 2026-06-22.
_RANDOM_SEED_CONSTRUCT = os.environ.get("LSU_RANDOM_SEED_CONSTRUCT", "0") == "1"


def energy_components(
    positions: np.ndarray,
    edges: np.ndarray,
    triples: np.ndarray,
    quads: np.ndarray,
    box: np.ndarray,
    d0: float,
) -> Tuple[float, float, float, float]:
    """Return the four energy components (f1, f2, f3, f4)."""
    pos = positions

    # --- f1: edge lengths ------------------------------------------------- #
    p_a = pos[edges[:, 0]]
    p_b = pos[edges[:, 1]]
    d_ab = pbc_displacement(p_b - p_a, box)
    L = np.linalg.norm(d_ab, axis=1)

    # --- f2: bond angles (target cos = -1/2) ------------------------------ #
    p_v = pos[triples[:, 0]]
    p_n1 = pos[triples[:, 1]]
    p_n2 = pos[triples[:, 2]]
    e1 = pbc_displacement(p_n1 - p_v, box)
    e2 = pbc_displacement(p_n2 - p_v, box)
    n1 = np.linalg.norm(e1, axis=1)
    n2 = np.linalg.norm(e2, axis=1)
    if _KEATING_F1F2:
        # literal length-coupled Keating forms
        f1 = np.sum((L ** 2 - d0 ** 2) ** 2)
        f2 = np.sum((np.einsum("ij,ij->i", e1, e2) + d0 ** 2 / 2) ** 2)
    else:
        f1 = np.sum((L - d0) ** 2)
        cos_t = np.einsum("ij,ij->i", e1, e2) / np.maximum(n1 * n2, 1e-12)
        f2 = np.sum((cos_t + 0.5) ** 2)

    # --- f3, f4: dihedrals + skew ----------------------------------------- #
    i, i1, i2, j, j1, j2 = (quads[:, k] for k in range(6))
    r_i_i1 = pbc_displacement(pos[i1] - pos[i], box)
    r_i_i2 = pbc_displacement(pos[i2] - pos[i], box)
    r_j_j1 = pbc_displacement(pos[j1] - pos[j], box)
    r_j_j2 = pbc_displacement(pos[j2] - pos[j], box)
    r_ij = pbc_displacement(pos[j] - pos[i], box)

    n_i = np.cross(r_i_i1, r_i_i2)
    n_j = np.cross(r_j_j1, r_j_j2)
    n_i_mag = np.linalg.norm(n_i, axis=1)
    n_j_mag = np.linalg.norm(n_j, axis=1)
    r_ij_mag = np.linalg.norm(r_ij, axis=1)

    n_i_hat = n_i / np.maximum(n_i_mag, 1e-12)[:, None]
    n_j_hat = n_j / np.maximum(n_j_mag, 1e-12)[:, None]
    r_ij_hat = r_ij / np.maximum(r_ij_mag, 1e-12)[:, None]

    dih = np.abs(np.einsum("ij,ij->i", n_i_hat, n_j_hat))
    f3 = np.sum((dih - _DIH_TARGET) ** 2)

    skew_i = np.einsum("ij,ij->i", r_ij_hat, n_i_hat)
    skew_j = np.einsum("ij,ij->i", r_ij_hat, n_j_hat)
    f4 = np.sum(skew_i ** 2 + skew_j ** 2)

    return float(f1), float(f2), float(f3), float(f4)


def total_energy(
    positions_flat: np.ndarray,
    N: int,
    edges: np.ndarray,
    triples: np.ndarray,
    quads: np.ndarray,
    box: np.ndarray,
    d0: float,
    weights: Tuple[float, float, float, float],
) -> float:
    pos = positions_flat.reshape(N, 3)
    f1, f2, f3, f4 = energy_components(pos, edges, triples, quads, box, d0)
    a, b, g, d = weights
    return a * f1 + b * f2 + g * f3 + d * f4


# --------------------------------------------------------------------------- #
# Energy — JAX path (used only when HAS_JAX)
# --------------------------------------------------------------------------- #
# Strategy: a single jit-compiled value_and_grad function that takes
# `edges, triples, quads` as RUNTIME arguments (not static). Their shapes
# (n_edges, n_triples=3*N, n_quads=n_edges) are invariant under Stone-Wales
# moves — only indices change — so JAX caches the compilation once per
# (N, n_edges) and reuses it for the entire run.
# --------------------------------------------------------------------------- #
if HAS_JAX:
    from jax import value_and_grad

    def _pbc_jax(d, box):
        return d - box * jnp.round(d / box)

    def _energy_jax_full(pos_flat, edges, triples, quads, box, d0, w):
        n_total = pos_flat.shape[0]
        pos = pos_flat.reshape(n_total // 3, 3)

        # f1: edge lengths
        d_ab = _pbc_jax(pos[edges[:, 1]] - pos[edges[:, 0]], box)
        L = jnp.linalg.norm(d_ab, axis=1)

        # f2: bond angles, target cos = -1/2
        e1 = _pbc_jax(pos[triples[:, 1]] - pos[triples[:, 0]], box)
        e2 = _pbc_jax(pos[triples[:, 2]] - pos[triples[:, 0]], box)
        n1 = jnp.linalg.norm(e1, axis=1)
        n2 = jnp.linalg.norm(e2, axis=1)
        if _KEATING_F1F2:
            f1 = jnp.sum((L ** 2 - d0 ** 2) ** 2)
            f2 = jnp.sum((jnp.sum(e1 * e2, axis=1) + d0 ** 2 / 2) ** 2)
        else:
            f1 = jnp.sum((L - d0) ** 2)
            cos_t = jnp.sum(e1 * e2, axis=1) / jnp.maximum(n1 * n2, 1e-12)
            f2 = jnp.sum((cos_t + 0.5) ** 2)

        # f3, f4: dihedrals + skew
        i = quads[:, 0]; i1 = quads[:, 1]; i2 = quads[:, 2]
        j = quads[:, 3]; j1 = quads[:, 4]; j2 = quads[:, 5]
        r_i1 = _pbc_jax(pos[i1] - pos[i], box)
        r_i2 = _pbc_jax(pos[i2] - pos[i], box)
        r_j1 = _pbc_jax(pos[j1] - pos[j], box)
        r_j2 = _pbc_jax(pos[j2] - pos[j], box)
        r_ij = _pbc_jax(pos[j] - pos[i], box)

        ni = jnp.cross(r_i1, r_i2)
        nj = jnp.cross(r_j1, r_j2)
        ni_mag = jnp.linalg.norm(ni, axis=1)
        nj_mag = jnp.linalg.norm(nj, axis=1)
        rij_mag = jnp.linalg.norm(r_ij, axis=1)
        ni_h = ni / jnp.maximum(ni_mag, 1e-12)[:, None]
        nj_h = nj / jnp.maximum(nj_mag, 1e-12)[:, None]
        rij_h = r_ij / jnp.maximum(rij_mag, 1e-12)[:, None]

        dih = jnp.abs(jnp.sum(ni_h * nj_h, axis=1))
        f3 = jnp.sum((dih - _DIH_TARGET) ** 2)
        f4 = jnp.sum(jnp.sum(rij_h * ni_h, axis=1) ** 2
                     + jnp.sum(rij_h * nj_h, axis=1) ** 2)

        a, b, g, d = w[0], w[1], w[2], w[3]
        return a * f1 + b * f2 + g * f3 + d * f4

    # Compile once at module load. The closure compiles per unique input shape;
    # since shapes are constant across the run (n_edges, n_triples, n_quads,
    # 3*N positions), this happens exactly once on first call.
    _value_and_grad_jit = jit(value_and_grad(_energy_jax_full, argnums=0))

    def _jax_value_and_grad(pos_flat, edges_j, triples_j, quads_j,
                            box_j, d0, w_j, mask_flat_j=None):
        e, g = _value_and_grad_jit(jnp.asarray(pos_flat),
                                   edges_j, triples_j, quads_j,
                                   box_j, d0, w_j)
        g_arr = np.asarray(g, dtype=np.float64)
        if mask_flat_j is not None:
            # Gradient masking implements the Vink/Mousseau-Barkema local relax
            # that Sellers cites: out-of-shell vertices have zero gradient and
            # L-BFGS doesn't move them. Applied host-side after the JIT call so
            # the kernel signature (and JIT cache) stays unchanged.
            g_arr = g_arr * np.asarray(mask_flat_j, dtype=np.float64)
        return float(e), g_arr

    def _jaxopt_value_and_grad(x_flat, edges_j, triples_j, quads_j,
                               box_j, d0_j, w_j, mask_flat_j):
        # Module-level wrapper kept identity-stable so that jaxopt's internal
        # JIT cache stays warm across Stone-Wales updates. Topology arrays
        # are passed as run-time arguments (not closure captures) to avoid
        # retracing when their *values* change.
        e, g = _value_and_grad_jit(x_flat, edges_j, triples_j, quads_j,
                                   box_j, d0_j, w_j)
        return e, g * mask_flat_j


# --------------------------------------------------------------------------- #
# Relaxation
# --------------------------------------------------------------------------- #
class _RelaxContext:
    """Holds JIT-compiled callables and JAX-resident topology arrays.

    Constructing one of these allocates JAX device arrays for the topology and
    binds them to the cached `_value_and_grad_jit`. All subsequent `relax`
    calls within a WWW loop share the same compiled kernel; only the index
    *values* change after a Stone-Wales move (shapes are invariant), so JAX
    does not retrace.
    """

    def __init__(self, N, box, d0, weights, use_jax, use_jaxopt=False):
        self.N = N
        self.box = np.asarray(box, dtype=np.float64)
        self.d0 = float(d0)
        self.weights = tuple(float(w) for w in weights)
        self.use_jax = bool(use_jax and HAS_JAX)
        self.use_jaxopt = bool(self.use_jax and use_jaxopt and HAS_JAXOPT)
        # Per-vertex moving mask. None ⇒ unmasked (all vertices move).
        # When set to a (N,) bool array, gradient is zeroed for frozen
        # components so L-BFGS keeps them fixed. See compute_local_shell_mask
        # and the Vink/Mousseau-Barkema relaxation that Sellers cites.
        self._mask_flat: Optional[np.ndarray] = None
        self._moving_idx: Optional[np.ndarray] = None
        if self.use_jax:
            self._box_j = jnp.asarray(self.box)
            self._w_j = jnp.asarray(self.weights)
            self._d0_j = jnp.float64(self.d0)
            self._mask_flat_j = jnp.ones((3 * self.N,), dtype=jnp.float64)
            # Cache jaxopt LBFGS instances keyed by (maxiter, tol). Each
            # instance lazily JIT-compiles its inner while-loop on first
            # call; reusing the same instance across WWW iterations keeps
            # the cache warm and avoids the seconds-per-call recompile we'd
            # otherwise pay. Only used on the opt-in jaxopt path; on CPU
            # scipy + jit'd autodiff is ~50-150x faster (jaxopt's per-call
            # host dispatch overhead is ~2s regardless of problem size).
            self._jaxopt_solvers: dict = {}

    def set_moving_mask(self, mask: Optional[np.ndarray]) -> None:
        """Set the per-vertex boolean mask. None ⇒ all vertices move (full-N).

        For the JAX path the mask is broadcast to (N, 3) and applied to the
        gradient. For the NumPy path the moving sub-vector indices are cached
        for use by `relax`.
        """
        if mask is None:
            self._mask_flat = None
            self._moving_idx = None
            if self.use_jax:
                self._mask_flat_j = jnp.ones((3 * self.N,), dtype=jnp.float64)
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.N,):
            raise ValueError(f"mask shape {mask.shape} != ({self.N},)")
        flat = np.broadcast_to(mask[:, None], (self.N, 3)).reshape(-1)
        self._mask_flat = flat.astype(np.float64)
        self._moving_idx = np.flatnonzero(flat)
        if self.use_jax:
            self._mask_flat_j = jnp.asarray(self._mask_flat)

    def get_jaxopt_solver(self, max_iter: int, tol: float):
        if not self.use_jaxopt:
            return None
        key = (int(max_iter), float(tol))
        solver = self._jaxopt_solvers.get(key)
        if solver is None:
            solver = _JaxoptLBFGS(
                fun=_jaxopt_value_and_grad,
                value_and_grad=True,
                maxiter=key[0],
                tol=key[1],
                jit=True,
            )
            self._jaxopt_solvers[key] = solver
        return solver

    def update_topology(self, edges, neighbors):
        """Recompute triples/quads after a topology change. O(E)."""
        self.edges = edges
        self.triples = build_angle_triples(neighbors)
        self.quads = build_dihedral_quads(edges, neighbors)
        if self.use_jax:
            # Push topology arrays to device. Shapes match across SW moves,
            # so the JIT-compiled kernel doesn't retrace.
            self._edges_j = jnp.asarray(self.edges, dtype=jnp.int32)
            self._triples_j = jnp.asarray(self.triples, dtype=jnp.int32)
            self._quads_j = jnp.asarray(self.quads, dtype=jnp.int32)

    def value_and_grad(self, positions_flat):
        if self.use_jax:
            return _jax_value_and_grad(
                positions_flat,
                self._edges_j, self._triples_j, self._quads_j,
                self._box_j, self._d0_j, self._w_j,
                mask_flat_j=self._mask_flat_j,
            )
        # NumPy path: compute energy and use scipy's finite-difference gradient.
        e = total_energy(positions_flat, self.N, self.edges, self.triples,
                         self.quads, self.box, self.d0, self.weights)
        return float(e), None  # gradient: None → scipy uses finite differences

    def energy(self, positions_flat):
        if self.use_jax:
            e, _ = self.value_and_grad(positions_flat)
            return e
        return float(total_energy(positions_flat, self.N, self.edges,
                                  self.triples, self.quads,
                                  self.box, self.d0, self.weights))


def _relax_single_lbfgs(positions, ctx, max_iter, tol):
    """One scipy/jaxopt L-BFGS pass, identical to the legacy relax behaviour.

    Returns ``(new_positions, E_final)``. Used both as the inner work of
    the threshold-aware ``relax`` and as a fast path when no threshold is
    set.
    """
    if ctx.use_jaxopt:
        solver = ctx.get_jaxopt_solver(max_iter, tol)
        x0 = jnp.asarray(positions.reshape(-1))
        res = solver.run(
            x0,
            ctx._edges_j, ctx._triples_j, ctx._quads_j,
            ctx._box_j, ctx._d0_j, ctx._w_j, ctx._mask_flat_j,
        )
        return (
            np.asarray(res.params, dtype=np.float64).reshape(ctx.N, 3),
            float(res.state.value),
        )
    if ctx.use_jax:
        def fun(x):
            return ctx.value_and_grad(x)
        res = minimize(fun, positions.reshape(-1), jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": tol})
        return res.x.reshape(ctx.N, 3), float(res.fun)

    # NumPy path with optional moving-mask sub-vector.
    if ctx._moving_idx is not None:
        full = positions.reshape(-1).copy()
        moving = ctx._moving_idx

        def fun_sub(x_sub):
            full[moving] = x_sub
            return ctx.energy(full)

        x0 = full[moving].copy()
        res = minimize(fun_sub, x0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": tol})
        full[moving] = res.x
        return full.reshape(ctx.N, 3), float(res.fun)

    def fun(x):
        return ctx.energy(x)
    res = minimize(fun, positions.reshape(-1), method="L-BFGS-B",
                   options={"maxiter": max_iter, "gtol": tol})
    return res.x.reshape(ctx.N, 3), float(res.fun)


def _force_norm_sq_moving(ctx, positions_flat) -> Tuple[float, float]:
    """Return (E, |F|^2) over the moving DOFs, using whichever backend ctx has.

    For the JAX-on path this calls the JIT'd value_and_grad once and squares
    the masked gradient. For the NumPy path, scipy's finite-difference gradient
    is too slow to query mid-relax; we approximate |F|^2 by re-running a tiny
    L-BFGS step and reading off its residual gradient norm. In practice we
    never reach this branch from production runs (use_jax defaults to True),
    so the approximation is harmless.
    """
    if ctx.use_jax:
        e, g = ctx.value_and_grad(positions_flat)
        g = np.asarray(g, dtype=np.float64)
        if ctx._mask_flat is not None:
            g = g * ctx._mask_flat
        return float(e), float(np.dot(g, g))
    # NumPy fallback: cheap finite-difference estimate over moving DOFs.
    e = ctx.energy(positions_flat)
    if ctx._moving_idx is None:
        return float(e), float("nan")
    moving = ctx._moving_idx
    eps = 1e-5
    full = positions_flat.copy()
    g_sq = 0.0
    for k in moving:
        full[k] += eps
        e_plus = ctx.energy(full)
        full[k] -= 2 * eps
        e_minus = ctx.energy(full)
        full[k] += eps
        g_k = (e_plus - e_minus) / (2 * eps)
        g_sq += g_k * g_k
    return float(e), float(g_sq)


def relax(
    positions: np.ndarray,
    ctx: _RelaxContext,
    max_iter: int,
    tol: float = 1e-8,
    *,
    E_threshold: float = float("inf"),
    threshold_check_cycle_skip: int = 5,
    threshold_local_to_global_cycle: int = 10,
    promote_margin: float = 0.1,
    c_f: float = 0.5,
    cycle_size: Optional[int] = None,
    on_global_promote: Optional[callable] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """L-BFGS relaxation with optional Vink/MB threshold-energy early rejection.

    Backwards-compatible API for callers that don't set ``E_threshold``:
    when ``E_threshold == inf`` the function runs a single L-BFGS pass and
    returns ``(positions, E, info)`` with ``info["early_rejected"]=False``,
    matching the legacy single-pass behaviour exactly except for the new
    third return slot.

    When ``E_threshold`` is finite, runs L-BFGS in chunks of ``cycle_size``
    iterations (auto-defaulted to ``max(5, max_iter // 25)`` so a typical
    relax does ~25 cycles, per Hemmann § 2.1). At each cycle boundary:

      - Query energy ``E`` and gradient ``g`` from ``ctx.value_and_grad``.
      - Compute ``|F|^2`` over moving DOFs only (Vink local-relax restricts
        the threshold check to the cluster).
      - BM2000 Eq. 4 estimator: ``E_f_est ≈ E - c_f * |F|^2``.
      - Early rejection window (BM2000 / Hemmann § 2.1): abort with
        ``info["early_rejected"]=True`` when ``E_f_est > E_threshold``,
        but only for cycles in
        ``(threshold_check_cycle_skip, threshold_local_to_global_cycle]``.
        BM2000: "we do not reject any move during the first five steps of
        relaxation" (anharmonic warm-up); Hemmann § 2.1: "After 10 cycles,
        relaxation continues ... without early rejections."
      - Local→global rescue (Vink PRB 64, 245214 § IV.B): at cycle
        ``threshold_local_to_global_cycle``, if a moving mask is set and
        the energy sits in the band ``[E_threshold,
        E_threshold + promote_margin)`` — i.e. the frozen-shell boundary
        strain is all that keeps the move above threshold — drop the mask
        via ``ctx.set_moving_mask(None)`` and continue full-N ("we switch
        from local to global relaxation when, during local relaxation, the
        energy comes to within 0.1 eV of the threshold energy"). Moves
        already below threshold stay local (Hemmann § 2.1 production runs
        are entirely local for system-size independence).
    """
    if cycle_size is None:
        cycle_size = max(5, int(max_iter) // 25)
    info: Dict = {
        "n_iter_done": 0,
        "force_norm_final": 0.0,
        "promoted_to_global": False,
        "early_rejected": False,
        "E_estimate_at_abort": float("nan"),
    }

    if not math.isfinite(E_threshold):
        new_pos, E_final = _relax_single_lbfgs(positions, ctx, max_iter, tol)
        info["n_iter_done"] = int(max_iter)
        # Cheap force-norm reading on the JAX path; skip on NumPy to avoid
        # the FD cost.
        if ctx.use_jax:
            _, F_sq = _force_norm_sq_moving(ctx, new_pos.reshape(-1))
            info["force_norm_final"] = math.sqrt(F_sq)
        return new_pos, E_final, info

    # Threshold-aware chunked relax.
    pos_flat = positions.reshape(-1).copy()
    cycles_done = 0
    iters_done = 0
    promoted = False
    E = float("inf")
    F_sq = float("inf")
    while iters_done < max_iter:
        this_chunk = min(cycle_size, max_iter - iters_done)
        new_pos, E_after = _relax_single_lbfgs(
            pos_flat.reshape(ctx.N, 3), ctx, this_chunk, tol
        )
        pos_flat = new_pos.reshape(-1)
        iters_done += this_chunk
        cycles_done += 1
        # Re-read energy + force norm at the current point.
        E, F_sq = _force_norm_sq_moving(ctx, pos_flat)
        # BM2000 Eq. 4 estimator.
        E_est = E - c_f * F_sq
        # Threshold check. Active only inside the window
        # (warm-up, local_to_global]: BM2000 forbid rejections during the
        # first five cycles (anharmonicities at the SW defect make the
        # harmonic estimator unreliable), and Hemmann § 2.1 prescribes no
        # early rejections after cycle 10 ("After 10 cycles, relaxation
        # continues ... without early rejections").
        if (
            threshold_check_cycle_skip
            < cycles_done
            <= threshold_local_to_global_cycle
            and E_est > E_threshold
        ):
            info.update({
                "n_iter_done": iters_done,
                "force_norm_final": math.sqrt(F_sq),
                "promoted_to_global": promoted,
                "early_rejected": True,
                "E_estimate_at_abort": E_est,
            })
            return pos_flat.reshape(ctx.N, 3), E, info
        # Local→global rescue (Vink § IV.B): only when the energy is stuck
        # in the band within `promote_margin` ABOVE the threshold — i.e.
        # boundary strain held by the frozen shell is plausibly all that
        # blocks acceptance. Moves already below threshold stay local
        # (Hemmann § 2.1 keeps production relaxation entirely local).
        if (
            not promoted
            and cycles_done == threshold_local_to_global_cycle
            and ctx._mask_flat is not None
            and 0.0 <= (E - E_threshold) < promote_margin
        ):
            if on_global_promote is not None:
                on_global_promote(pos_flat.reshape(ctx.N, 3))
            ctx.set_moving_mask(None)
            promoted = True
        # Convergence check: tiny |F| means L-BFGS won't move further.
        if F_sq < tol * tol:
            break

    info.update({
        "n_iter_done": iters_done,
        "force_norm_final": math.sqrt(F_sq) if math.isfinite(F_sq) else 0.0,
        "promoted_to_global": promoted,
        "early_rejected": False,
        "E_estimate_at_abort": float("nan"),
    })
    return pos_flat.reshape(ctx.N, 3), E, info


# --------------------------------------------------------------------------- #
# Stone-Wales bond switch
# --------------------------------------------------------------------------- #
def _replace_neighbor(neighbors_row: np.ndarray, old: int, new: int) -> None:
    for k in range(neighbors_row.shape[0]):
        if neighbors_row[k] == old:
            neighbors_row[k] = new
            return
    raise ValueError(f"old neighbour {old} not found in row {neighbors_row}")


def stone_wales_propose(
    edges: np.ndarray,
    neighbors: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 30,
) -> Optional[Tuple[int, Tuple[int, int, int, int], int]]:
    """Propose a valid Stone-Wales bond transposition.

    Pick a random edge (i, j); pick a neighbour c of i other than j; pick a
    neighbour d of j other than i (and not already adjacent to i so the new
    edge (i, d) doesn't duplicate). The move is

        (i, c), (j, d)  →  (i, d), (j, c)

    Returns
    -------
    (edge_index_to_remove1, (i, c, j, d), edge_index_to_remove2)
    or None if no valid move was found in `max_tries` attempts.
    """
    E = edges.shape[0]
    nbr_set = [set(int(x) for x in row) for row in neighbors]

    for _ in range(max_tries):
        ek = int(rng.integers(0, E))
        i, j = int(edges[ek, 0]), int(edges[ek, 1])

        # other neighbours of i and j
        i_others = [int(x) for x in neighbors[i] if int(x) != j]
        j_others = [int(x) for x in neighbors[j] if int(x) != i]
        c = i_others[int(rng.integers(0, 2))]
        d = j_others[int(rng.integers(0, 2))]

        if c == d:
            continue
        # New edges must not already exist
        if d in nbr_set[i] or c in nbr_set[j]:
            continue
        if c == j or d == i:
            continue

        # Find edge indices for (i, c) and (j, d)
        ek1 = _find_edge_index(edges, i, c)
        ek2 = _find_edge_index(edges, j, d)
        if ek1 is None or ek2 is None or ek1 == ek2:
            continue
        return ek1, (i, c, j, d), ek2

    return None


def _find_edge_index(edges: np.ndarray, a: int, b: int) -> Optional[int]:
    mask = ((edges[:, 0] == a) & (edges[:, 1] == b)) | \
           ((edges[:, 0] == b) & (edges[:, 1] == a))
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def stone_wales_apply(
    edges: np.ndarray,
    neighbors: np.ndarray,
    move: Tuple[int, Tuple[int, int, int, int], int],
) -> None:
    ek1, (i, c, j, d), ek2 = move
    # edges (i,c) and (j,d) become (i,d) and (j,c). Rows are written in
    # canonical (min, max) order so that every writer preserves the seed
    # invariant "edge rows sorted" and `stone_wales_revert` is an exact
    # array-level inverse (not merely a graph-level one).
    edges[ek1] = (i, d) if i < d else (d, i)
    edges[ek2] = (j, c) if j < c else (c, j)
    _replace_neighbor(neighbors[i], c, d)
    _replace_neighbor(neighbors[j], d, c)
    _replace_neighbor(neighbors[c], i, j)
    _replace_neighbor(neighbors[d], j, i)


def stone_wales_revert(
    edges: np.ndarray,
    neighbors: np.ndarray,
    move: Tuple[int, Tuple[int, int, int, int], int],
) -> None:
    ek1, (i, c, j, d), ek2 = move
    edges[ek1] = (i, c) if i < c else (c, i)
    edges[ek2] = (j, d) if j < d else (d, j)
    _replace_neighbor(neighbors[i], d, c)
    _replace_neighbor(neighbors[j], c, d)
    _replace_neighbor(neighbors[c], j, i)
    _replace_neighbor(neighbors[d], i, j)


# --------------------------------------------------------------------------- #
# WWW main loop
# --------------------------------------------------------------------------- #
def www_anneal(
    positions: np.ndarray,
    edges: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray,
    d0: float,
    weights: Tuple[float, float, float, float],
    n_iterations: int,
    T0: float,
    T_final: float,
    rng: np.random.Generator,
    target_lsu: Optional[float] = None,
    target_depth: int = 1,
    target_locality: int = 1,
    target_tolerance: float = 0.005,
    relax_local_iters: int = 100,
    relax_global_iters: int = 500,
    relax_global_every: int = 0,
    global_fallback_threshold: float = float("inf"),
    local_shell_depth: Optional[int] = 4,
    check_lsu_every: int = 500,
    uniformity_weight: float = 10.0,
    uniformity_kmax: int = 2,
    threshold_energy_relax: bool = True,
    c_f: float = 0.5,
    cycle_size: Optional[int] = None,
    temperatures: Optional[np.ndarray] = None,
    use_jax: bool = False,
    use_jaxopt: bool = False,
    verbose: bool = True,
    log_tag: str = "WWW",
):
    """Run WWW simulated annealing. Returns (positions, edges, neighbors, history).

    Sellers's supplement (Methods, refs [13,14]) follows Vink/Mousseau-Barkema.
    The Stone-Wales loop here implements that recipe end-to-end:

      1. Snapshot ``E_b = E_curr``.
      2. Draw ``s ∈ (0, 1)`` and set the Metropolis threshold
         ``E_t = E_b - T * ln(s)`` (Vink 2001 Eq. 5).
      3. Apply the SW move, refresh topology.
      4. Build the moving-shell mask (``local_shell_depth``, default 4),
         held fixed via gradient masking.
      5. Call ``relax(..., E_threshold=E_t)``: chunked L-BFGS with BM2000
         Eq. 4 ``E_f_est = E - c_f * |F|^2`` early-rejection checks active
         only for cycles 6..10 (BM2000: no rejections during the first 5
         cycles; Hemmann § 2.1: none after cycle 10), plus the Vink
         local→global rescue at cycle 10 when E sits within 0.1 *above*
         the threshold.
      6. If the relax aborted early (``info["early_rejected"]``), revert
         topology and positions and continue. **No Metropolis roll** —
         Vink's identity says this is exactly equivalent to a Metropolis
         rejection at the same ``s``.
      7. Else, compute the acceptance objective with the optional
         ``uniformity_weight`` low-k S(k) penalty; reuse the same ``s`` to
         decide accept/reject via ``s < exp(-dE / T)``.

    ``temperatures`` overrides the geometric T schedule built from
    ``T0``/``T_final`` — pass an array of length ``n_iterations`` to use a
    custom profile (e.g. ``topology_burn_in``'s triangular ramp). When set,
    ``T0`` and ``T_final`` are ignored.

    ``uniformity_weight`` adds a low-k structure-factor penalty to the
    Metropolis objective (not to the L-BFGS relax, and not to the Vink
    threshold ``E_t`` — that lives in strain-energy units alone). Set to
    0.0 for strict Sellers Eq. 2 acceptance.

    The legacy ``global_fallback_threshold`` block stays as a separate
    safety net (default ``inf``, effectively off) — it triggers only on
    *converged* local relaxes where ΔE > threshold, after the new
    threshold scheme has already had its chance to abort.
    """
    if relax_global_every:
        warnings.warn(
            "`relax_global_every` runs an unconditional full-N L-BFGS that "
            "re-introduces void clustering under the bonded-only Sellers "
            "energy. The Vink/Mousseau-Barkema scheme uses fallback-gated "
            "global relax instead — set `relax_global_every=0` and tune "
            "`global_fallback_threshold` (default inf/off).",
            DeprecationWarning, stacklevel=2,
        )

    N = positions.shape[0]
    ctx = _RelaxContext(N, box, d0, weights, use_jax=use_jax, use_jaxopt=use_jaxopt)
    ctx.update_topology(edges, neighbors)
    E_curr = ctx.energy(positions.reshape(-1))
    objective_curr, S_low_curr = _acceptance_objective(
        E_curr, positions, box, uniformity_weight, uniformity_kmax
    )
    # Seed's starting LSU, used to make the early-exit direction-aware: a
    # crystalline seed (Phi ~ 1.0) descends toward the target as it disorders,
    # whereas a random seed (Phi ~ 0.5) ascends toward it as it anneals. The
    # exit must trigger when Phi reaches the target from whichever side it
    # began on; a single fixed sign mis-fires for one of the two directions.
    phi_start: Optional[float] = None
    if target_lsu is not None and check_lsu_every > 0:
        phi_start = compute_lsu(
            positions, edges, neighbors, box,
            depth=target_depth, locality=target_locality,
            max_pairs=2000, rng=rng,
        )
    history: Dict = {
        "iter": [], "T": [], "E": [], "objective": [],
        "uniformity_S": [], "lsu": [],
        "accepted": 0, "proposed": 0, "global_fallbacks": 0,
        "early_rejected": 0, "local_to_global": [],
        "force_norm_history": [],
    }

    accepted = 0
    proposed = 0
    fallback_count = 0
    early_reject_count = 0
    promote_count = 0
    if temperatures is None:
        log_ratio = math.log(T_final / T0) if T0 > 0 and T_final > 0 else 0.0
    else:
        temperatures = np.asarray(temperatures, dtype=np.float64).reshape(-1)
        if temperatures.shape[0] != n_iterations:
            raise ValueError(
                f"temperatures length {temperatures.shape[0]} != "
                f"n_iterations {n_iterations}"
            )
    t_start = time.time()

    def _make_promote_cb(it_now):
        def _cb(_positions):
            nonlocal promote_count
            promote_count += 1
            history["local_to_global"].append(int(it_now))
        return _cb

    # Cache the last relax info so the periodic LSU check at the *top* of
    # each iteration (which runs before any `continue`) can include the
    # latest force_norm even when the previous iteration aborted early.
    last_force_norm = 0.0
    early_exit_triggered = False

    for it in range(n_iterations):
        if temperatures is not None:
            T = float(temperatures[it])
        else:
            if T0 > 0 and T_final > 0:
                T = T0 * math.exp(log_ratio * it / max(1, n_iterations - 1))
            else:
                T = float(T0 if it == 0 else T_final)

        # Periodic LSU check at the *start* of each iteration. Hoisted here
        # (rather than after the move) because every `continue` in the move
        # logic below — proposal failure, disconnection, early-rejection —
        # would otherwise skip the check. With early_rejected rates of
        # ~90 %, the check would only fire ~10 % as often as requested.
        if (
            check_lsu_every > 0
            and it > 0
            and it % check_lsu_every == 0
        ):
            phi = compute_lsu(
                positions, edges, neighbors, box,
                depth=target_depth, locality=target_locality,
                max_pairs=2000, rng=rng,
            )
            history["iter"].append(it)
            history["T"].append(T)
            history["E"].append(E_curr)
            history["objective"].append(objective_curr)
            history["uniformity_S"].append(S_low_curr)
            history["lsu"].append(phi)
            history["force_norm_history"].append(float(last_force_norm))
            if verbose:
                acc_rate = accepted / max(1, proposed)
                fb_rate = fallback_count / max(1, proposed)
                er_rate = early_reject_count / max(1, proposed)
                uniformity_msg = (
                    f"  S_low={S_low_curr:.4g}"
                    if uniformity_weight > 0.0 else ""
                )
                print(
                    f"[{log_tag} it={it:6d}] T={T:.4g}  E={E_curr:.4g}  "
                    f"Obj={objective_curr:.4g}{uniformity_msg}  "
                    f"phi_{target_depth}{target_locality}={phi:.4f}  "
                    f"acc={acc_rate:.2%}  early={er_rate:.2%}  "
                    f"fb={fb_rate:.2%}  promote={promote_count}  "
                    f"elapsed={time.time()-t_start:.1f}s"
                )
            if target_lsu is not None:
                reached = abs(phi - target_lsu) <= target_tolerance
                if phi_start is not None and phi_start >= target_lsu:
                    # Descending (crystalline seed): stop once Phi has fallen
                    # to/through the target.
                    reached = reached or (phi <= target_lsu)
                elif phi_start is not None:
                    # Ascending (random seed): stop once Phi has risen
                    # to/through the target.
                    reached = reached or (phi >= target_lsu)
                if reached:
                    if verbose:
                        print(f"[{log_tag}] target LSU {target_lsu} reached "
                              f"(measured {phi:.4f}); stopping.")
                    early_exit_triggered = True
                    break

        move = stone_wales_propose(edges, neighbors, rng)
        if move is None:
            continue
        proposed += 1
        _ek1, (sw_i, sw_c, sw_j, sw_d), _ek2 = move

        # Snapshot E_b and draw s up front (Vink Eq. 5: E_t = E_b - T*ln(s)).
        E_b = E_curr
        s = rng.random()
        if T > 0:
            E_t = E_b - T * math.log(max(s, 1e-12))
        else:
            # T=0 quench phase: only accept moves with dE <= 0.
            E_t = E_b

        # Snapshot positions for revert.
        pos_before = positions.copy()
        stone_wales_apply(edges, neighbors, move)

        # Connectivity: SW preserves degrees but can disconnect graph; reject.
        if not is_connected(N, edges):
            stone_wales_revert(edges, neighbors, move)
            continue

        # Topology changed → refresh triples/quads (and JAX device arrays).
        ctx.update_topology(edges, neighbors)

        # Vink/Mousseau-Barkema local relax: only vertices within
        # `local_shell_depth` graph-edge hops of the SW seed {i, c, j, d} are
        # allowed to move. Out-of-shell vertices are held fixed via gradient
        # masking. Setting `local_shell_depth=None` falls back to full-N.
        if local_shell_depth is not None and local_shell_depth > 0:
            seed_verts = np.array([sw_i, sw_c, sw_j, sw_d], dtype=np.int64)
            shell = compute_local_shell_mask(seed_verts, neighbors,
                                             local_shell_depth, N)
            ctx.set_moving_mask(shell)
        else:
            ctx.set_moving_mask(None)

        E_threshold_use = E_t if threshold_energy_relax else float("inf")
        new_pos, E_new, relax_info = relax(
            positions, ctx, max_iter=relax_local_iters,
            E_threshold=E_threshold_use, c_f=c_f, cycle_size=cycle_size,
            on_global_promote=_make_promote_cb(it),
        )

        if relax_info["early_rejected"]:
            # Vink identity: aborting on the same s as the Metropolis roll
            # is exactly equivalent to a Metropolis rejection. No re-draw.
            stone_wales_revert(edges, neighbors, move)
            ctx.update_topology(edges, neighbors)
            positions = pos_before
            early_reject_count += 1
            continue

        strain_dE = E_new - E_curr

        # Legacy safety net (default off): one full-N polish if local relax
        # converged but with a huge dE. Independent of the in-relax local→
        # global promotion above.
        if strain_dE > global_fallback_threshold:
            ctx.set_moving_mask(None)
            new_pos, E_new, _ = relax(
                new_pos, ctx, max_iter=relax_global_iters,
                E_threshold=float("inf"),
            )
            new_pos = new_pos - box * np.round(new_pos / box)
            strain_dE = E_new - E_curr
            fallback_count += 1

        objective_new, S_low_new = _acceptance_objective(
            E_new, new_pos, box, uniformity_weight, uniformity_kmax
        )
        dE = objective_new - objective_curr

        # Metropolis acceptance on the (possibly fallback-improved)
        # objective. Reuse the same `s` to keep tight equivalence with the
        # in-relax threshold abort (only relevant when uniformity_weight=0;
        # with uniformity_weight>0 the objective differs from E and the
        # reuse becomes an approximation, which is what we want — the
        # threshold scheme remains an algorithmic speedup on strain energy
        # only).
        if dE <= 0 or s < math.exp(-dE / max(T, 1e-12)):
            positions = new_pos
            E_curr = E_new
            objective_curr = objective_new
            S_low_curr = S_low_new
            accepted += 1
        else:
            stone_wales_revert(edges, neighbors, move)
            ctx.update_topology(edges, neighbors)
            positions = pos_before

        # Record the force norm from this move's relax so the LSU check at
        # the *top* of the next iteration can read it. Out-of-iteration
        # state — `continue` does NOT skip this assignment unless one of
        # the upstream `continue`s skipped the relax call itself.
        last_force_norm = float(relax_info.get("force_norm_final", 0.0))

    history["accepted"] = accepted
    history["proposed"] = proposed
    history["global_fallbacks"] = fallback_count
    history["early_rejected"] = early_reject_count
    return positions, edges, neighbors, history


# --------------------------------------------------------------------------- #
# Cluster / void / homogeneity diagnostics
# --------------------------------------------------------------------------- #
def cluster_diagnostics(
    positions: np.ndarray,
    edges: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray,
    d0: float,
    probe_grid: int = 12,
) -> Dict[str, float]:
    """Read-only health metrics for a Z=3 amorphous network.

    Collects the direct-space / reciprocal-space order metrics that
    Hemmann/Saba 2026 (Adv. Funct. Mater. DOI 10.1002/adfm.202600037) use to
    diagnose vertex clustering and large-pore formation:

    - ``r_nn``  : median nearest-neighbour distance over all vertex pairs
                 (PBC, includes bonded neighbours).
    - ``r_u``   : median nearest-uncoordinated-neighbour distance per vertex,
                 i.e. the distance to the closest vertex *not* already
                 bonded. Hemmann targets ``r_u >~ 1.0 d0``; smaller values
                 indicate vertex clustering.
    - ``delta_c``: critical pore radius (largest empty-sphere radius), the
                  pore-percolation surrogate; estimated on a ``probe_grid``
                  x3 jittered probe array. Hemmann targets
                  ``delta_c <~ 0.5 d0``.
    - ``min_non_bonded``: minimum non-bonded vertex separation. Hard fail
                          for ``< 0.4 d0`` (the Sellers energy has no
                          non-bonded repulsion, so vertices closer than
                          this would never be pushed apart by L-BFGS).
    - ``n_close_pairs``: count of non-bonded pairs with separation
                         ``< 0.7 d0`` (a softer cluster surrogate).
    - ``bond_len_{mean,std,min,max}``.
    - ``voxel_std_4`` : ``_voxel_density_std(positions, box, 4)``.
    - ``S_low_k2``    : ``low_k_structure_factor(positions, box, 2)``.

    PBC handled via ``scipy.spatial.cKDTree(boxsize=box)``. All neighbour
    queries are O(N log N); the probe-grid pore estimate is O(P log N)
    with P = probe_grid^3.
    """
    pos = np.asarray(positions, dtype=np.float64)
    box_arr = np.asarray(box, dtype=np.float64).reshape(3)
    N = pos.shape[0]
    # cKDTree's `boxsize` requires positions in [0, L)^3, not [-L/2, L/2)^3.
    pos_shift = pos - box_arr * np.floor((pos + box_arr / 2.0) / box_arr)
    pos_shift = pos_shift + box_arr / 2.0  # now in [0, L)
    # Clamp to box-(epsilon) to satisfy cKDTree's strict bound.
    pos_shift = np.clip(pos_shift, 0.0, box_arr - 1e-12)

    tree = cKDTree(pos_shift, boxsize=box_arr)

    # --- bond lengths (PBC minimum image) --------------------------------- #
    p_a = pos[edges[:, 0]]
    p_b = pos[edges[:, 1]]
    bond_d = pbc_displacement(p_b - p_a, box_arr)
    bond_L = np.linalg.norm(bond_d, axis=1)

    # --- r_nn : nearest-neighbour distance, any pair --------------------- #
    # Query the 2 closest tree points; index 0 is the vertex itself.
    nn_dists, _ = tree.query(pos_shift, k=2)
    r_nn = float(np.median(nn_dists[:, 1]))

    # --- r_u : nearest *uncoordinated* neighbour distance ---------------- #
    # Convert neighbours to a set per vertex for fast membership test.
    nbr_sets = [set(int(x) for x in row) for row in neighbors]
    # Query enough k that there's at least one unbonded neighbour. With
    # Z=3 in 3D, k=8 is comfortably above the expected coordination
    # shell; if even k=8 is all bonded (impossible in practice for Z=3)
    # we fall back to a sequential scan.
    k_query = min(8, N)
    _, nn_idx = tree.query(pos_shift, k=k_query)
    r_u_per_vertex = np.empty(N, dtype=np.float64)
    for i in range(N):
        chosen = -1
        for j in nn_idx[i]:
            j = int(j)
            if j == i:
                continue
            if j in nbr_sets[i]:
                continue
            chosen = j
            break
        if chosen < 0:
            # Fallback: brute force on PBC distance to non-bonded vertices.
            d = pbc_displacement(pos - pos[i:i + 1], box_arr)
            dists = np.linalg.norm(d, axis=1)
            dists[i] = np.inf
            for nb in nbr_sets[i]:
                dists[nb] = np.inf
            chosen = int(np.argmin(dists))
        r_u_per_vertex[i] = float(np.linalg.norm(
            pbc_displacement(pos[chosen:chosen + 1] - pos[i:i + 1], box_arr)
        ))
    r_u = float(np.median(r_u_per_vertex))

    # --- delta_c : critical pore radius via probe grid ------------------- #
    # Sample probe points on a jittered grid through the canonical box,
    # query the nearest vertex distance, take the max.
    rng_probe = np.random.default_rng(12345)
    grid_axis = (np.arange(probe_grid) + 0.5) / probe_grid - 0.5
    gx, gy, gz = np.meshgrid(grid_axis, grid_axis, grid_axis, indexing="ij")
    probes = np.stack(
        [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=1
    ) * box_arr
    probes += (rng_probe.random(probes.shape) - 0.5) * (box_arr / probe_grid)
    probes_shift = probes + box_arr / 2.0
    probes_shift = np.clip(probes_shift, 0.0, box_arr - 1e-12)
    pore_d, _ = tree.query(probes_shift, k=1)
    delta_c = float(pore_d.max())

    # --- min_non_bonded and n_close_pairs -------------------------------- #
    # Query all neighbours within 0.7*d0; then strip out the bonded edges.
    pairs = tree.query_pairs(r=0.7 * d0, output_type="ndarray")
    edge_set = set()
    for a, b in edges:
        a = int(a); b = int(b)
        edge_set.add((a, b) if a < b else (b, a))
    if pairs.size:
        non_bonded_mask = np.array([
            tuple(sorted((int(a), int(b)))) not in edge_set
            for a, b in pairs
        ])
        non_bonded_pairs = pairs[non_bonded_mask]
    else:
        non_bonded_pairs = np.empty((0, 2), dtype=np.int64)
    n_close_pairs = int(non_bonded_pairs.shape[0])

    # min_non_bonded is the global minimum non-bonded separation; the
    # cheap way is to query for a larger ball that should contain it.
    # We loop with growing radius until we find at least one non-bonded
    # pair; if a generous cap fails, fall back to brute force.
    min_non_bonded: float = float("inf")
    for r_query in (0.7 * d0, 1.0 * d0, 1.5 * d0, 2.5 * d0):
        cand = tree.query_pairs(r=r_query, output_type="ndarray")
        if cand.size == 0:
            continue
        cand_sorted = np.sort(cand, axis=1)
        keep_mask = np.array([
            (int(a), int(b)) not in edge_set for a, b in cand_sorted
        ])
        cand_non_bonded = cand_sorted[keep_mask]
        if cand_non_bonded.shape[0] == 0:
            continue
        diffs = pbc_displacement(
            pos[cand_non_bonded[:, 0]] - pos[cand_non_bonded[:, 1]], box_arr
        )
        dists = np.linalg.norm(diffs, axis=1)
        min_non_bonded = float(dists.min())
        break
    if not math.isfinite(min_non_bonded):
        # Brute O(N^2) sweep; only triggered when every non-bonded vertex
        # pair is > 2.5*d0 apart, which is unusual but consistent.
        all_d = pbc_displacement(
            pos[:, None, :] - pos[None, :, :], box_arr
        )
        all_dist = np.linalg.norm(all_d, axis=-1)
        # Mask out self + bonded.
        all_dist[np.eye(N, dtype=bool)] = np.inf
        for a, b in edges:
            all_dist[int(a), int(b)] = np.inf
            all_dist[int(b), int(a)] = np.inf
        min_non_bonded = float(all_dist.min())

    return {
        "r_nn": r_nn,
        "r_u": r_u,
        "delta_c": delta_c,
        "min_non_bonded": min_non_bonded,
        "n_close_pairs": n_close_pairs,
        "bond_len_mean": float(bond_L.mean()),
        "bond_len_std": float(bond_L.std()),
        "bond_len_min": float(bond_L.min()),
        "bond_len_max": float(bond_L.max()),
        "voxel_std_4": _voxel_density_std(pos, box_arr, ngrid=4),
        "S_low_k2": low_k_structure_factor(pos, box_arr, kmax=2),
    }


def _format_cluster_diagnostics(diag: Dict[str, float], d0: float) -> str:
    """One-line summary string for ``cluster_diagnostics`` output."""
    return (
        f"r_nn={diag['r_nn']:.3f} r_u={diag['r_u']:.3f} "
        f"delta_c={diag['delta_c']:.3f} "
        f"min_nb={diag['min_non_bonded']:.3f} "
        f"close<0.7d0={diag['n_close_pairs']} "
        f"bond_len(mean={diag['bond_len_mean']:.3f}, "
        f"std={diag['bond_len_std']:.3f}) "
        f"voxel_std4={diag['voxel_std_4']:.3f} "
        f"S_low={diag['S_low_k2']:.4g}"
    )


# --------------------------------------------------------------------------- #
# Topology burn-in (triangular-profile WWW phase to lose crystalline memory)
# --------------------------------------------------------------------------- #
def _calibrate_T_melt(
    positions: np.ndarray,
    edges: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray,
    d0: float,
    weights: Tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    probe_moves: int,
    probe_T: float,
    relax_local_iters: int,
    local_shell_depth: Optional[int],
    threshold_energy_relax: bool,
    c_f: float,
    uniformity_weight: float,
    uniformity_kmax: int,
    use_jax: bool,
    use_jaxopt: bool,
    verbose: bool,
) -> Tuple[float, int, int]:
    """Estimate ΔE_min for Hemmann Eq. 5: the relaxed energy cost of the
    *energetically lowest* uphill bond switch of the CURRENT network.

    Hemmann (Adv. Funct. Mater. 2026) § 2.3: "We define T_melt as the
    temperature at which the energetically lowest bond switch and
    relaxation are accepted with probability P_accept > P_melt := 0.1%.
    Isolating T in the Metropolis acceptance probability yields the
    melting temperature" — i.e. ``T_melt = ΔE_min / ln(1/P_melt)`` with
    ``ΔE_min`` the smallest positive relaxed ΔE among candidate bond
    switches of the *initial* configuration.

    The probe therefore samples ``probe_moves`` random Stone-Wales
    switches FROM THE UNCHANGED INPUT STATE: each candidate is applied,
    locally relaxed (no threshold abort), measured, and reverted — never
    accepted, no Metropolis evolution. ``probe_T`` is retained in the
    signature for backwards compatibility but is no longer used.

    Returns ``(min_uphill_dE, n_uphill, proposed)``. The caller computes
    ``T_melt = min_uphill_dE / ln(1/P_melt)``.
    """
    del probe_T  # unused — kept for backwards-compatible signature
    # Snapshot inputs so the probe does not mutate the caller's state.
    pos0 = positions.copy()
    edges_probe = edges.copy()
    neighbors_probe = neighbors.copy()
    probe_seed = int(rng.integers(0, 2**31 - 1))

    N = pos0.shape[0]
    ctx = _RelaxContext(N, box, d0, weights, use_jax=use_jax, use_jaxopt=use_jaxopt)
    ctx.update_topology(edges_probe, neighbors_probe)
    E0 = ctx.energy(pos0.reshape(-1))
    obj0, _ = _acceptance_objective(
        E0, pos0, box, uniformity_weight, uniformity_kmax
    )
    probe_rng = np.random.default_rng(probe_seed)
    uphill_dE_samples: list = []
    proposed = 0
    for _ in range(probe_moves):
        move = stone_wales_propose(edges_probe, neighbors_probe, probe_rng)
        if move is None:
            continue
        proposed += 1
        _ek1, (sw_i, sw_c, sw_j, sw_d), _ek2 = move
        stone_wales_apply(edges_probe, neighbors_probe, move)
        if not is_connected(N, edges_probe):
            stone_wales_revert(edges_probe, neighbors_probe, move)
            continue
        ctx.update_topology(edges_probe, neighbors_probe)
        if local_shell_depth is not None and local_shell_depth > 0:
            shell = compute_local_shell_mask(
                np.array([sw_i, sw_c, sw_j, sw_d], dtype=np.int64),
                neighbors_probe, local_shell_depth, N,
            )
            ctx.set_moving_mask(shell)
        else:
            ctx.set_moving_mask(None)
        # Full local relax — no threshold abort; we need the true relaxed ΔE.
        new_pos, E_new, _ = relax(
            pos0, ctx, max_iter=relax_local_iters,
            E_threshold=float("inf"),
        )
        obj_new, _ = _acceptance_objective(
            E_new, new_pos, box, uniformity_weight, uniformity_kmax
        )
        dE = obj_new - obj0
        if dE > 1e-12:
            uphill_dE_samples.append(dE)
        # Always revert: every candidate is probed from the same state.
        stone_wales_revert(edges_probe, neighbors_probe, move)
        ctx.update_topology(edges_probe, neighbors_probe)
    ctx.set_moving_mask(None)

    if uphill_dE_samples:
        min_uphill = float(np.min(uphill_dE_samples))
    else:
        # No uphill candidate found (already-disordered state where every
        # sampled switch is downhill). Fall back to a generic estimate
        # using the energy scale ``E0`` itself.
        min_uphill = max(1.0, abs(E0) * 0.1)
        if verbose:
            print(
                f"[burn-in calibration] WARNING: no uphill switch among "
                f"{proposed} probes; using fallback ΔE estimate {min_uphill:.3g}"
            )
    if verbose:
        print(
            f"[burn-in calibration] probed {proposed} switches from fixed "
            f"state: uphill={len(uphill_dE_samples)}  "
            f"min_uphill_dE={min_uphill:.3g} (Hemmann Eq. 5 basis)"
        )
    return min_uphill, len(uphill_dE_samples), proposed


def topology_burn_in(
    positions: np.ndarray,
    edges: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray,
    d0: float,
    weights: Tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    n_heat: int = 8_000,
    n_cool: int = 16_000,
    n_quench: int = 4_000,
    T_max: Optional[float] = None,
    T_max_over_T_melt: float = 1.15,
    P_melt: float = 0.001,
    T_melt_probe_moves: int = 600,
    T_melt_probe_T: float = 5.0,
    relax_local_iters: int = 100,
    relax_global_iters: int = 500,
    local_shell_depth: Optional[int] = 4,
    global_fallback_threshold: float = float("inf"),
    threshold_energy_relax: bool = True,
    c_f: float = 0.5,
    uniformity_weight: float = 10.0,
    uniformity_kmax: int = 2,
    target_accepts_per_vertex: Optional[float] = None,
    use_jax: bool = False,
    use_jaxopt: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Triangular-profile WWW burn-in (Hemmann § 2.3, Figure 2).

    Replaces the legacy constant-T burn-in. The temperature schedule has
    three phases:

      - **Heat** (``n_heat`` moves): linearly ramp T from 0 to ``T_max``.
      - **Cool** (``n_cool`` moves): linearly ramp T from ``T_max`` to 0.
      - **Quench** (``n_quench`` moves): T = 0; only downhill moves accepted.

    The schedule is fed into ``www_anneal`` via its ``temperatures`` kwarg.
    The relax inside each SW move follows the Vink/MB threshold-energy
    early rejection (Vink 2001 Eq. 5 + BM2000 Eq. 3/4), with the
    5-cycle anharmonic warm-up and local→global promotion at cycle 10.

    ``T_max`` is either user-supplied or auto-calibrated against the
    melting temperature ``T_melt = ΔE_min / ln(1/P_melt)`` (Hemmann
    Eq. 5, with ``ΔE_min`` the relaxed cost of the energetically lowest
    uphill bond switch of the initial network — see
    ``_calibrate_T_melt``). With ``T_max_over_T_melt = 1.15`` the
    schedule lands in the hyperuniform regime Hemmann Figure 8c,d
    identifies (``1.0 ≲ T_max/T_melt ≲ 1.3``).

    Returns
    -------
    positions, edges, neighbors, info
        ``info`` keys: ``T_max_used``, ``T_melt``, ``P_melt``, ``n_heat``,
        ``n_cool``, ``n_quench``, ``moves``, ``accepted``, ``proposed``,
        ``early_rejected``, ``cluster_after``, ``probe_min_uphill_dE``.
    """
    n_total = int(n_heat) + int(n_cool) + int(n_quench)
    if n_total <= 0:
        return positions, edges, neighbors, {
            "T_max_used": None, "T_melt": None, "P_melt": P_melt,
            "n_heat": n_heat, "n_cool": n_cool, "n_quench": n_quench,
            "moves": 0, "accepted": 0, "proposed": 0, "early_rejected": 0,
            "cluster_after": cluster_diagnostics(
                positions, edges, neighbors, box, d0
            ),
            "probe_min_uphill_dE": None,
        }

    # --- T_max calibration ---------------------------------------------- #
    T_melt: Optional[float] = None
    probe_uphill: Optional[float] = None
    if T_max is None:
        probe_uphill, _, _ = _calibrate_T_melt(
            positions, edges, neighbors, box, d0, weights, rng,
            probe_moves=T_melt_probe_moves,
            probe_T=T_melt_probe_T,
            relax_local_iters=relax_local_iters,
            local_shell_depth=local_shell_depth,
            threshold_energy_relax=threshold_energy_relax,
            c_f=c_f,
            uniformity_weight=uniformity_weight,
            uniformity_kmax=uniformity_kmax,
            use_jax=use_jax,
            use_jaxopt=use_jaxopt,
            verbose=verbose,
        )
        T_melt = float(probe_uphill / math.log(1.0 / max(P_melt, 1e-12)))
        T_max_used = float(T_max_over_T_melt) * T_melt
        if verbose:
            print(
                f"[burn-in] T_melt={T_melt:.3g} (Hemmann Eq.5, P_melt={P_melt}); "
                f"T_max={T_max_used:.3g} = {T_max_over_T_melt}*T_melt"
            )
    else:
        T_max_used = float(T_max)
        if verbose:
            print(f"[burn-in] T_max={T_max_used:.3g} (user-supplied)")

    # --- Build triangular temperature schedule ------------------------- #
    schedule = np.empty(n_total, dtype=np.float64)
    if n_heat > 0:
        schedule[:n_heat] = T_max_used * (np.arange(n_heat) + 1) / n_heat
    if n_cool > 0:
        schedule[n_heat:n_heat + n_cool] = (
            T_max_used * (1.0 - np.arange(n_cool) / n_cool)
        )
    if n_quench > 0:
        schedule[n_heat + n_cool:] = 0.0

    if verbose:
        print(
            f"[burn-in] schedule: heat 0→{T_max_used:.3g} ({n_heat} moves), "
            f"cool →0 ({n_cool} moves), quench (T=0, {n_quench} moves)"
        )

    # Optional accepted-moves cap.
    max_accepted: Optional[int] = None
    if (
        target_accepts_per_vertex is not None
        and target_accepts_per_vertex > 0
    ):
        max_accepted = int(math.ceil(
            float(target_accepts_per_vertex) * positions.shape[0] / 4.0
        ))

    # --- Run www_anneal in chunks so we can honour the accepted-moves cap.
    t_start = time.time()
    moves_done = 0
    accepted_total = 0
    proposed_total = 0
    early_rejected_total = 0
    chunk_size = max(500, n_total // 20)
    while moves_done < n_total:
        chunk = min(chunk_size, n_total - moves_done)
        chunk_temps = schedule[moves_done:moves_done + chunk]
        positions, edges, neighbors, hist_chunk = www_anneal(
            positions, edges, neighbors, box, d0, weights,
            n_iterations=chunk,
            T0=float(chunk_temps[0]),  # ignored when temperatures is set
            T_final=float(chunk_temps[-1]),
            rng=rng,
            target_lsu=None,
            relax_local_iters=relax_local_iters,
            relax_global_iters=relax_global_iters,
            relax_global_every=0,
            global_fallback_threshold=global_fallback_threshold,
            local_shell_depth=local_shell_depth,
            uniformity_weight=uniformity_weight,
            uniformity_kmax=uniformity_kmax,
            threshold_energy_relax=threshold_energy_relax,
            c_f=c_f,
            temperatures=chunk_temps,
            check_lsu_every=0,
            use_jax=use_jax, use_jaxopt=use_jaxopt,
            verbose=False,
            log_tag="burn-in",
        )
        moves_done += chunk
        accepted_total += hist_chunk["accepted"]
        proposed_total += hist_chunk["proposed"]
        early_rejected_total += hist_chunk["early_rejected"]
        if verbose:
            acc_chunk = hist_chunk["accepted"] / max(1, hist_chunk["proposed"])
            T_lo = float(chunk_temps.min())
            T_hi = float(chunk_temps.max())
            cap_msg = ""
            if max_accepted is not None:
                involvements = 4.0 * accepted_total / positions.shape[0]
                cap_msg = (
                    f"  acc_per_vertex={involvements:.2f}/"
                    f"{target_accepts_per_vertex:.2f}"
                )
            print(
                f"[burn-in] moves={moves_done}/{n_total}  "
                f"T={T_lo:.3g}-{T_hi:.3g}  acc_chunk={acc_chunk:.2%}  "
                f"early={hist_chunk['early_rejected']/max(1, hist_chunk['proposed']):.2%}  "
                f"voxel_std4={_voxel_density_std(positions, box, 4):.3f}"
                f"{cap_msg}  elapsed={time.time()-t_start:.1f}s"
            )
        if max_accepted is not None and accepted_total >= max_accepted:
            if verbose:
                involvements = 4.0 * accepted_total / positions.shape[0]
                print(
                    f"[burn-in] accepted-move cap reached: {accepted_total} "
                    f"accepts ({involvements:.2f} per vertex)"
                )
            break

    diag = cluster_diagnostics(positions, edges, neighbors, box, d0)
    if verbose:
        print(f"[burn-in] post-burn-in cluster diag: "
              f"{_format_cluster_diagnostics(diag, d0)}")

    info = {
        "T_max_used": T_max_used,
        "T_melt": T_melt,
        "P_melt": P_melt,
        "n_heat": int(n_heat),
        "n_cool": int(n_cool),
        "n_quench": int(n_quench),
        "moves": int(moves_done),
        "accepted": int(accepted_total),
        "proposed": int(proposed_total),
        "early_rejected": int(early_rejected_total),
        "cluster_after": diag,
        "probe_min_uphill_dE": probe_uphill,
    }
    return positions, edges, neighbors, info


# --------------------------------------------------------------------------- #
# LSU statistic
# --------------------------------------------------------------------------- #
def _build_tree(
    root: int,
    depth: int,
    neighbors: np.ndarray,
    positions: np.ndarray,
    box: np.ndarray,
) -> dict:
    """Build a depth-`depth` tree rooted at `root`.

    The tree is stored as nested lists of edge vectors (in the unwrapped
    local frame where the root sits at the origin).

    Returns dict with:
      'root_edges': (3, 3) array of vectors from root to its 3 neighbours
      'child_edges': list of (3,3) arrays — for each root edge, the edges
                     leaving its endpoint (excluding the back-edge to root).
                     Only present when depth >= 2.
      'paths': list of (path_idx, edge_vec) for ALL edges in the tree, where
               path_idx is the chain of root-edge indices to reach this edge.
    """
    tree = {"root": int(root), "depth": depth}
    pos_root = positions[root]

    nbrs = [int(x) for x in neighbors[root]]
    root_edges = np.empty((3, 3))
    for k, nb in enumerate(nbrs):
        d = pbc_displacement(positions[nb] - pos_root, box)
        root_edges[k] = d
    tree["root_edges"] = root_edges
    tree["root_neighbors"] = np.array(nbrs, dtype=np.int64)

    if depth >= 2:
        child_edges = []
        child_neighbors = []
        for k, nb in enumerate(nbrs):
            # endpoint of root edge k is at pos_root + root_edges[k]  (in local frame)
            # In *world* coordinates this endpoint is positions[nb] (modulo PBC),
            # but we work in local frame to avoid PBC ambiguity.
            local_endpoint = pos_root + root_edges[k]
            grand_nbrs = [int(x) for x in neighbors[nb] if int(x) != root]
            ce = np.empty((2, 3))
            for m, gn in enumerate(grand_nbrs):
                d = pbc_displacement(positions[gn] - positions[nb], box)
                ce[m] = (local_endpoint + d) - pos_root - root_edges[k]
                # = displacement from endpoint to grandchild, in local frame
            child_edges.append(ce)
            child_neighbors.append(np.array(grand_nbrs, dtype=np.int64))
        tree["child_edges"] = child_edges
        tree["child_neighbors"] = child_neighbors
    return tree


def _align_two_trees(
    tree_a: dict,
    tree_b: dict,
    sigma: Tuple[int, ...],
) -> np.ndarray:
    """Return rotation R (3,3) that maps tree_b's root edges (after permutation
    sigma) onto tree_a's root edges as best as possible.

    Concretely:
      1. Translate so root vertices coincide (already done — both trees stored
         with root at origin).
      2. Rotate so root edge sigma(0) of B aligns with root edge 0 of A.
      3. Rotate around the new edge-0 axis so root edge sigma(1) of B (now in
         the plane containing root edge 0 of A and root edge 1 of A) aligns
         best with root edge 1 of A.
      4. (Reflection step is needed for tetrahedral; trihedral/3-edge case is
         covered after these two rotations.)

    For trivalent trees the residual mismatch in the third edge is what makes
    LSU < 1.
    """
    Ea = tree_a["root_edges"]                      # (3,3)
    Eb = tree_b["root_edges"][list(sigma)]         # permute B's roots

    # Step 1: rotate B so edge 0 lines up with A's edge 0
    R1 = _rotation_aligning(Eb[0], Ea[0])

    # Apply R1
    Eb1 = (R1 @ Eb.T).T

    # Step 2: rotate around Ea[0] axis to align Eb1[1] with Ea[1] in the plane.
    axis = Ea[0] / max(np.linalg.norm(Ea[0]), 1e-12)
    # project Eb1[1] and Ea[1] onto plane perpendicular to axis
    def _proj_perp(v, axis):
        return v - axis * np.dot(v, axis)
    p_b = _proj_perp(Eb1[1], axis)
    p_a = _proj_perp(Ea[1], axis)
    nb = np.linalg.norm(p_b)
    na = np.linalg.norm(p_a)
    if nb < 1e-9 or na < 1e-9:
        R2 = np.eye(3)
    else:
        cos_t = np.dot(p_b, p_a) / (nb * na)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        sin_t = np.dot(np.cross(p_b, p_a), axis) / (nb * na)
        sin_t = np.clip(sin_t, -1.0, 1.0)
        theta = math.atan2(sin_t, cos_t)
        R2 = _rotation_about_axis(axis, theta)

    return R2 @ R1


def _rotation_aligning(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit-direction(u) -> unit-direction(v)."""
    nu = max(np.linalg.norm(u), 1e-12)
    nv = max(np.linalg.norm(v), 1e-12)
    a = u / nu
    b = v / nv
    c = np.dot(a, b)
    if c > 1.0 - 1e-9:
        return np.eye(3)
    if c < -1.0 + 1e-9:
        # 180 deg around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= max(np.linalg.norm(axis), 1e-12)
        return _rotation_about_axis(axis, math.pi)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s + 1e-18))


def _rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    n = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([[0, -n[2], n[1]],
                  [n[2], 0, -n[0]],
                  [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _overlap_score(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
    pairing: Tuple[int, ...],
) -> float:
    """Sum of (r_a · r_b) / mean(|r_a|, |r_b|)^2 for the given pairing."""
    s = 0.0
    for i, j in enumerate(pairing):
        ra = edges_a[i]
        rb = edges_b[j]
        denom = ((np.linalg.norm(ra) + np.linalg.norm(rb)) / 2.0) ** 2
        if denom < 1e-18:
            continue
        s += float(np.dot(ra, rb) / denom)
    return s


def _best_pairing(
    edges_a: np.ndarray,
    edges_b: np.ndarray,
) -> Tuple[Tuple[int, ...], float]:
    """Brute-force best pairing of equal-size edge sets (size <= 3)."""
    n = edges_a.shape[0]
    best = None
    best_s = -float("inf")
    for perm in permutations(range(n)):
        s = _overlap_score(edges_a, edges_b, perm)
        if s > best_s:
            best_s = s
            best = perm
    return best, best_s


def _phi_for_permutation(
    tree_a: dict,
    tree_b: dict,
    sigma: Tuple[int, ...],
) -> Tuple[float, int]:
    """Apply alignment for permutation sigma, then sum overlap scores
    across all edges in the tree. Returns (sum_score, n_edges)."""
    R = _align_two_trees(tree_a, tree_b, sigma)

    # Score root edges
    Ea_root = tree_a["root_edges"]
    Eb_root_perm = tree_b["root_edges"][list(sigma)]
    Eb_root_aligned = (R @ Eb_root_perm.T).T
    score = _overlap_score(Ea_root, Eb_root_aligned, tuple(range(Ea_root.shape[0])))
    n_edges = Ea_root.shape[0]

    # Depth-2 children: greedy depth-first pairing.
    if tree_a.get("depth", 1) >= 2 and "child_edges" in tree_a and "child_edges" in tree_b:
        for k in range(Ea_root.shape[0]):
            ce_a = tree_a["child_edges"][k]                 # (2,3)
            ce_b_local = tree_b["child_edges"][sigma[k]]    # (2,3) in tree_b local frame
            ce_b_aligned = (R @ ce_b_local.T).T
            best_perm, best_s = _best_pairing(ce_a, ce_b_aligned)
            score += best_s
            n_edges += ce_a.shape[0]

    return score, n_edges


def phi_ab(tree_a: dict, tree_b: dict) -> float:
    gamma_fact = math.factorial(tree_a["root_edges"].shape[0])
    total = 0.0
    n_edges_norm = tree_a["root_edges"].shape[0]
    if tree_a.get("depth", 1) >= 2:
        n_edges_norm += sum(arr.shape[0] for arr in tree_a["child_edges"])

    for sigma in permutations(range(tree_a["root_edges"].shape[0])):
        s, _ = _phi_for_permutation(tree_a, tree_b, sigma)
        total += s / n_edges_norm

    return total / gamma_fact


def compute_lsu(
    positions: np.ndarray,
    edges: np.ndarray,
    neighbors: np.ndarray,
    box: np.ndarray,
    depth: int = 1,
    locality: int = 1,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute mean Φ_{depth,locality} over (a, b) pairs with b within
    `locality` edges of a (excluding a == b).

    For very large networks pass max_pairs to subsample uniformly.
    """
    N = positions.shape[0]
    rng = rng or np.random.default_rng(0)

    # Build lookup of locality-l neighborhoods (BFS).
    pairs = []
    for a in range(N):
        visited = {a}
        frontier = {a}
        for _ in range(locality):
            new_front = set()
            for v in frontier:
                for w in neighbors[v]:
                    w = int(w)
                    if w not in visited:
                        visited.add(w)
                        new_front.add(w)
            frontier = new_front
        for b in visited:
            if b == a:
                continue
            pairs.append((a, b))

    pairs = np.array(pairs, dtype=np.int64)
    if max_pairs is not None and pairs.shape[0] > max_pairs:
        idx = rng.choice(pairs.shape[0], size=max_pairs, replace=False)
        pairs = pairs[idx]

    # Cache trees.
    tree_cache: dict[int, dict] = {}

    def get_tree(v: int) -> dict:
        if v not in tree_cache:
            tree_cache[v] = _build_tree(v, depth, neighbors, positions, box)
        return tree_cache[v]

    phi_vals = np.empty(pairs.shape[0], dtype=np.float64)
    for k, (a, b) in enumerate(pairs):
        phi_vals[k] = phi_ab(get_tree(int(a)), get_tree(int(b)))

    return float(np.mean(phi_vals))


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def _clip_second_endpoint_to_box(
    p1: np.ndarray, p2: np.ndarray, box: np.ndarray
) -> np.ndarray:
    """Vectorised: for each rod with p1 inside [-L/2, L/2]^3 and p2 possibly
    outside, return p2' on the line p1→p2 such that p2' lies on the nearest
    box face along the rod direction. Rods entirely inside the box are
    returned unchanged.
    """
    half = box / 2.0
    direction = p2 - p1                          # shape (E, 3)
    n_rods = p1.shape[0]
    t_min = np.ones(n_rods, dtype=p1.dtype)      # default: keep p2

    eps = 1e-12
    safe_dir = np.where(np.abs(direction) > eps, direction, np.inf)
    # For each axis, compute t where line crosses ±half[axis]
    for axis in range(3):
        for face in (half[axis], -half[axis]):
            t_face = (face - p1[:, axis]) / safe_dir[:, axis]
            # Take the smallest positive t (first face hit going from p1)
            t_min = np.where((t_face > eps) & (t_face < t_min), t_face, t_min)

    # Only clip rods whose p2 is actually outside the box
    outside = np.any(np.abs(p2) > half + 1e-9, axis=1)
    t_eff = np.where(outside, t_min, np.ones_like(t_min))
    return p1 + t_eff[:, None] * direction


def network_to_rods(
    positions: np.ndarray,
    edges: np.ndarray,
    box: np.ndarray,
    pbc_duplicate_boundary_rods: bool = True,
    clip_endpoints_to_box: bool = True,
) -> np.ndarray:
    """Convert (positions, edges) → rod endpoints array shape (R, 6).

    Each periodic-cell edge becomes one or two rods depending on whether it
    crosses a box face:

    - For an edge entirely inside the canonical box, one rod is emitted —
      `(a_canon, a_canon + min_image(b - a))`.
    - For an edge crossing ≥1 face, the same rule applied from b's
      canonical position yields a different rod — `(b_canon, b_canon +
      min_image(a - b))`. Both are emitted, matching the convention of the
      Sellers reference file `lsu_example_ends.txt` (1500 unique edges →
      1653 rendered rods, 153 duplicates from face-crossing edges).

    The dual rendering is required when downstream code (e.g.
    ``create_permittivity_grid_penlike``) draws each rod as a literal
    cylinder without applying PBC: without the second image, structure
    crossing the +x face is drawn extending past +x but is missing on the
    -x side, breaking the periodicity of the resulting permittivity grid.

    Parameters
    ----------
    pbc_duplicate_boundary_rods : bool
        Set to False to suppress duplication and get one rod per unique
        edge (output shape ``(E, 6)``).
    clip_endpoints_to_box : bool
        Default True. The second endpoint of each rod (``p2``) is clipped
        to land on the nearest box face along the rod direction whenever
        ``p2`` lies outside the canonical box ``[-L/2, L/2]^3``. The
        first endpoint is always inside (it's the canonical-box image of
        a vertex). Clipping does **not** change the structure rendered
        by ``create_permittivity_grid_penlike`` (which clips at grid
        bounds anyway), but it makes a centerline visualisation in
        ParaView fit cleanly inside the cube outline. Set to False to
        keep the historical behaviour where rods can extend up to one
        bond length beyond the cube.

    Returns
    -------
    rods : ndarray, shape ``(R, 6)``
        ``R = E`` if ``pbc_duplicate_boundary_rods=False`` else
        ``R = E + (number of face-crossing edges)`` ≤ ``2E``.
    """
    pos_canon = positions - box * np.round(positions / box)

    a_canon = pos_canon[edges[:, 0]]
    b_canon = pos_canon[edges[:, 1]]
    d_ab = pbc_displacement(b_canon - a_canon, box)

    # Render 1: a_canon as the in-box endpoint.
    render_a = np.concatenate([a_canon, a_canon + d_ab], axis=1)

    if pbc_duplicate_boundary_rods:
        # Render 2: b_canon as the in-box endpoint. Distinct from render 1
        # iff the edge crosses ≥1 face (i.e. a_canon + d_ab != b_canon).
        render_b = np.concatenate([b_canon, b_canon - d_ab], axis=1)
        crosses = np.any(np.abs(a_canon + d_ab - b_canon) > 1e-9, axis=1)
        out = np.concatenate([render_a, render_b[crosses]], axis=0)
    else:
        out = render_a

    if clip_endpoints_to_box:
        # Clip the second endpoint to land on the nearest box face. The
        # first endpoint is the canonical-box image of a vertex (always
        # inside [-L/2, L/2]^3), so we only need to clip p2.
        p1 = out[:, :3]
        p2 = out[:, 3:]
        p2_clipped = _clip_second_endpoint_to_box(p1, p2, box)
        out = np.concatenate([p1, p2_clipped], axis=1)

    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_lsu_network(
    lsu_degree_12: Optional[float] = None,
    lsu_degree_22: Optional[float] = None,
    num_rods: Optional[int] = None,
    num_vertices: Optional[int] = None,
    bounds_microns: Union[float, Tuple[float, float, float]] = 11.44,
    edge_length: float = 0.8,
    n_www_iterations: int = 20_000,
    initial_temperature: float = 0.5,
    final_temperature: float = 1e-3,
    energy_weights: Optional[Dict[str, float]] = None,
    target_tolerance: float = 0.01,
    check_lsu_every: int = 500,
    relax_global_every: int = 0,
    relax_local_iters: int = 100,
    relax_global_iters: int = 500,
    global_fallback_threshold: float = float("inf"),
    local_shell_depth: Optional[int] = 4,
    threshold_energy_relax: bool = True,
    c_f: float = 0.5,
    cycle_size: Optional[int] = None,
    seed_kind: str = "crystal_srs",
    seed_lattice: str = "srs",
    seed_jitter_sigma: float = 0.10,
    strict_tiling: bool = False,
    bm2000_min_separation_frac: float = 0.98,
    bm2000_rc_start_frac: float = 1.30,
    bm2000_rc_grow_frac: float = 0.05,
    bm2000_rc_max_frac: float = 6.00,
    burn_in_n_heat: int = 8_000,
    burn_in_n_cool: int = 16_000,
    burn_in_n_quench: int = 4_000,
    burn_in_T_max: Optional[float] = None,
    burn_in_T_max_over_T_melt: float = 1.15,
    burn_in_P_melt: float = 0.001,
    burn_in_T_melt_probe_moves: int = 600,
    burn_in_T_melt_probe_T: float = 5.0,
    burn_in_target_accepts_per_vertex: Optional[float] = None,
    # Deprecated aliases (constant-T burn-in API) ---------------------------
    topology_burn_in_moves: Optional[int] = None,
    topology_burn_in_T: Optional[float] = None,
    topology_burn_in_target_accepts_per_vertex: Optional[float] = None,
    uniformity_weight: float = 10.0,
    uniformity_kmax: int = 2,
    seed: int = 42,
    use_jax: Optional[bool] = None,
    use_jaxopt: bool = False,
    pbc_duplicate_boundary_rods: bool = True,
    clip_endpoints_to_box: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Generate a 3D periodic amorphous trivalent network with prescribed LSU.

    Implements the Wooten-Winer-Weaire (WWW) simulated annealing algorithm
    of Sellers et al. (Nat. Commun. 8, 14439, 2017) on a 3-regular graph in
    a periodic box. Returns the rod endpoints as a NumPy array of shape
    ``(num_rods, 6)``.

    Parameters
    ----------
    lsu_degree_12, lsu_degree_22 : float, optional
        Target Φ_{1,2} or Φ_{2,2} (Sellers Eq. 2 convention: first
        subscript = tree depth n, second = locality l, i.e. root vertices
        within l edges of one another). Provide exactly one of the two.
    num_rods : int, optional
        Number of unique periodic-cell edges. Must be divisible by 3 so that
        the corresponding number of trivalent vertices V = 2·num_rods/3
        is an integer (and even). The rendered output may contain more rows
        than ``num_rods`` because edges crossing box faces are emitted twice
        when ``pbc_duplicate_boundary_rods=True`` (default). Provide exactly
        one of ``num_rods`` or ``num_vertices``.
    num_vertices : int, optional
        Number of trivalent vertices in the periodic cell. Must be even.
        Equivalent to setting ``num_rods = 3 * num_vertices // 2``. Use this
        when matching a known-N reference (e.g. the Sellers `lsu_example_ends.txt`
        is N=1000 / E=1500, rendered as 1653 rod lines). Provide exactly one
        of ``num_rods`` or ``num_vertices``.
    bounds_microns : float or 3-tuple
        Side length(s) of the periodic cubic/orthorhombic box.
    edge_length : float
        Target rod length d0 (in the same units as ``bounds_microns``).
    n_www_iterations : int
        Maximum number of WWW outer iterations. The loop exits early when
        the measured LSU is within ``target_tolerance`` of the target.
    initial_temperature, final_temperature : float
        Geometric temperature schedule for the Metropolis acceptance.
    energy_weights : dict, optional
        Mapping with keys ``alpha, beta, gamma, delta`` for the four energy
        terms. Default: the Sellers-group-confirmed weights
        ``alpha=0.7, beta=0.7, gamma=0.3, delta=0.4``.
    target_tolerance : float
        Tolerance for early exit.
    check_lsu_every : int
        How often (in WWW iterations) to measure the current LSU.
    relax_global_every : int
        **Deprecated, default 0.** Legacy fixed-schedule full-N L-BFGS polish.
        With the bonded-only Sellers energy this re-introduces void clustering
        because vertices anywhere in the cell can drift toward each other every
        K iterations. The Vink/Mousseau-Barkema scheme used by Sellers's cited
        refs [13,14] runs global relax only as a fallback when local relax
        fails to lower energy — see ``global_fallback_threshold``. Setting
        a non-zero value emits a ``DeprecationWarning``; behaviour is also
        retained as no-op (it is **not** wired back into the loop).
    relax_local_iters : int
        L-BFGS-B iteration cap for the local relaxation after each SW move
        (default 100). **Critical for convergence**: benchmarking on N=1102
        shows ΔE is positive for all moves at 30 iters (1% acceptance) and
        crosses into negative territory between 30 and 100 iters. Values below
        ~80 effectively prevent phi from rising regardless of iteration count.
    relax_global_iters : int
        L-BFGS-B iteration cap for each fallback full-N polish (default 500).
        Used when the per-move local relax fails the
        ``global_fallback_threshold`` check.
    global_fallback_threshold : float
        Energy increase ΔE = E_new - E_curr above which a single full-N L-BFGS
        polish is run on the post-SW positions before the Metropolis decision.
        Default ``float('inf')`` (off). Lower to a finite value to enable the
        Vink/Mousseau-Barkema fallback. Pick the threshold large enough that
        only "stalled" local relaxes trigger it: shell-constrained relax
        leaves residual ΔE > 0 even for good moves, so a value below the
        typical post-shell residual fires on every uphill move and
        re-introduces the void drift this gate is meant to avoid. Reasonable
        starting values are 5.0–20.0 in the same units as the energy.
    local_shell_depth : int or None
        Depth (in graph edges) of the moving shell around each Stone-Wales
        defect. Vertices farther than this are held fixed during the post-SW
        relaxation. Default 4, matching the Vink/Mousseau-Barkema scheme that
        Sellers's supplement (Methods, refs [13,14]) cites. Set to ``None`` or
        0 to disable shell masking and run a full-N L-BFGS for every relax
        (the previous behaviour; produces corner/edge void clustering at high
        iteration counts because vertices anywhere in the cell can drift
        toward each other under the bonded-only Sellers energy).
    seed_lattice : str
        Crystalline Z=3 lattice used as the WWW seed. Currently supports
        ``'srs'`` (default): the single-network gyroid net, with 8 vertices
        per cubic cell, 120-degree bond angles, and bond length
        ``a*sqrt(2)/4``. This is the crystalline parent of the amorphous
        gyroid and matches Hemmann/Saba's Z=3 gyroid starting point.
        ``'diamond3'`` is also available: cubic diamond topology with a perfect
        matching of 4 bonds removed per cubic cell to drop from Z=4 to
        Z=3; 8 vertices per cubic cell, all bond lengths equal, but with
        tetrahedral 109.47-degree angles. Replaces the legacy
        Barkema-Mousseau random seeder which produced long chord stragglers
        (3–5·d0) and seeded void clustering. See
        ``crystal_seed_network`` for details.
    seed_jitter_sigma : float
        Std-dev of Gaussian position jitter added to the crystalline
        seed, in units of ``edge_length``. Default 0.10. Breaks exact
        lattice symmetry so the first SW moves see a non-degenerate
        Hessian.
    strict_tiling : bool
        If True, raise ValueError when the requested ``num_vertices``
        cannot be exactly tiled by ``seed_lattice``'s unit cell.
        Default False: emit a warning and use the nearest valid N.
    topology_burn_in_moves : int
        Number of constant-temperature WWW Stone-Wales moves run before
        the production annealing to destroy crystalline memory of the
        seed. Default 20_000. Set to 0 to skip burn-in (only useful for
        diagnostic runs where you want to inspect the bare crystalline
        seed). The burn-in uses ``www_anneal`` with no LSU target and
        T0 = T_final = ``topology_burn_in_T``; it stops early when the
        4^3-voxel-density std plateaus. See ``topology_burn_in``.
    topology_burn_in_T : float or None
        Temperature for the burn-in. None (default) auto-calibrates via
        a short probe sweep to modest acceptance near melting. A user-supplied
        value skips calibration.
    topology_burn_in_target_accepts_per_vertex : float or None
        Stop the burn-in once accepted Stone-Wales moves have involved each
        vertex this many times on average (each accepted move involves four
        vertices). Default 4.0 prevents the high-T burn-in from over-randomising
        the network into the large-pore regime Hemmann/Saba diagnose. Set
        ``None`` to use ``topology_burn_in_moves`` as the only cap.
    uniformity_weight : float
        Weight of a low-k vertex structure-factor penalty added to the
        Metropolis acceptance objective. The local Sellers geometry energy is
        still used for L-BFGS relaxation; this term only rejects topology moves
        that amplify long-wavelength density fluctuations / voids. Default 10.0.
        Set 0.0 for strict Sellers Eq. 2 acceptance.
    uniformity_kmax : int
        Integer reciprocal-space shell used by the uniformity penalty. Default
        2 averages over 32 lowest nonzero modes in a cubic box.
    seed : int
        Random seed for reproducibility.
    use_jax : bool, optional
        If True, accelerate energy & gradient with JAX (autodiff + JIT).
        Default: True if JAX is installed, False otherwise. The L-BFGS
        driver is still ``scipy.optimize.minimize`` — only the gradient
        evaluation is JIT-compiled.
    use_jaxopt : bool
        If True (default False), drive the inner L-BFGS loop with
        ``jaxopt.LBFGS`` instead of scipy. Benchmarked: slower than
        scipy+JAX for N≤2000 on current hardware (~1.3 s vs ~84 ms for
        N=1102, 100 iters). Only consider enabling for very large N on GPU.
    pbc_duplicate_boundary_rods : bool
        If True (default), emit each face-crossing edge twice — once
        anchored at each canonical-box endpoint — matching the convention
        of the Sellers reference file `lsu_example_ends.txt`. Required for
        downstream code (e.g. ``create_permittivity_grid_penlike``) that
        draws rods as literal cylinders without applying PBC: without the
        duplicates the rendered structure is not periodic across box
        faces. Set to False to suppress duplication and emit one rod per
        unique edge.
    clip_endpoints_to_box : bool
        Default **False**, which matches the Sellers reference file
        convention (``lsu_example_ends.txt`` stores full-length rods whose
        endpoints extend up to one bond length beyond the cube). If True,
        the second endpoint of each rod (``p2``) is clipped to the nearest
        box face along the rod direction whenever it would extend outside
        the canonical box; the first endpoint is always inside. Clipping
        does **not** change the structure rendered by
        ``create_permittivity_grid_penlike`` (which clips at grid bounds
        anyway), but it makes a centerline visualisation in ParaView fit
        cleanly inside the cube outline.
    verbose : bool
        Print progress every ``check_lsu_every`` iterations.

    Returns
    -------
    rods : ndarray, shape (R, 6), dtype float64
        Each row is ``[x1, y1, z1, x2, y2, z2]``. ``R`` equals the unique
        edge count ``E = 3N/2`` if ``pbc_duplicate_boundary_rods=False``,
        otherwise ``R`` = ``E`` plus the number of edges that cross at
        least one box face.
    """
    if (lsu_degree_12 is None) == (lsu_degree_22 is None):
        raise ValueError(
            "Provide exactly one of `lsu_degree_12` or `lsu_degree_22`.")

    if (num_rods is None) == (num_vertices is None):
        raise ValueError(
            "Provide exactly one of `num_rods` or `num_vertices`.")
    if num_vertices is not None:
        if num_vertices % 2 != 0:
            raise ValueError(
                f"num_vertices must be even (got {num_vertices}); a "
                f"3-regular graph requires 2*E = 3*N with N even.")
        N = int(num_vertices)
        num_rods = (3 * N) // 2
    else:
        if num_rods % 3 != 0:
            raise ValueError(
                f"num_rods must be divisible by 3 (got {num_rods}); for a "
                f"trivalent graph 2*E = 3*V."
            )
        N = (2 * num_rods) // 3
        if N % 2 != 0:
            raise ValueError(
                f"num_rods={num_rods} ⇒ N={N} vertices, which is odd; "
                f"a 3-regular graph needs N even."
            )

    box = coerce_box(bounds_microns)
    use_jax = HAS_JAX if use_jax is None else (use_jax and HAS_JAX)
    weights_dict = energy_weights or {}
    # Defaults are the energy weights confirmed by the Sellers group for
    # the Eq. 2 functional: alpha=0.7, beta=0.7, gamma=0.3, delta=0.4.
    weights = (
        float(weights_dict.get("alpha", 0.7)),
        float(weights_dict.get("beta", 0.7)),
        float(weights_dict.get("gamma", 0.3)),
        float(weights_dict.get("delta", 0.4)),
    )

    if lsu_degree_12 is not None:
        target_lsu = float(lsu_degree_12)
        # Sellers Eq. 2: Φ_nl = depth-n trees, root vertices within l
        # edges. Φ_12 ⇒ depth 1, locality 2 (Fig. 3b plots Φ_12, Φ_22,
        # Φ_32 — all at locality 2).
        target_depth, target_locality = 1, 2
    else:
        target_lsu = float(lsu_degree_22)
        target_depth, target_locality = 2, 2

    rng = np.random.default_rng(seed)

    use_jaxopt_eff = bool(use_jaxopt and use_jax and HAS_JAXOPT)
    if verbose:
        print(f"[gen] N={N} vertices, E={num_rods} rods, box={box.tolist()}, "
              f"d0={edge_length}, target phi_{target_depth}{target_locality}={target_lsu}, "
              f"seed_kind={seed_kind}, "
              f"jax={'on' if use_jax else 'off'}, "
              f"jaxopt={'on' if use_jaxopt_eff else 'off'}")

    # --- Deprecation: map legacy topology_burn_in_* kwargs to burn_in_* ---
    if topology_burn_in_moves is not None:
        warnings.warn(
            "`topology_burn_in_moves` is deprecated. Use the new "
            "triangular-profile kwargs `burn_in_n_heat`, `burn_in_n_cool`, "
            "`burn_in_n_quench` (default 8_000/16_000/4_000). The legacy "
            "value is split 1/5 heat, 3/5 cool, 1/5 quench for back-compat.",
            DeprecationWarning, stacklevel=2,
        )
        legacy_total = int(topology_burn_in_moves)
        burn_in_n_heat = max(1, legacy_total // 5)
        burn_in_n_cool = max(1, 3 * legacy_total // 5)
        burn_in_n_quench = max(1, legacy_total - burn_in_n_heat - burn_in_n_cool)
    if topology_burn_in_T is not None:
        warnings.warn(
            "`topology_burn_in_T` is deprecated. Use `burn_in_T_max` "
            "(the triangular peak temperature) or leave it None to "
            "auto-calibrate against T_melt.",
            DeprecationWarning, stacklevel=2,
        )
        if burn_in_T_max is None:
            burn_in_T_max = float(topology_burn_in_T)
    if topology_burn_in_target_accepts_per_vertex is not None:
        warnings.warn(
            "`topology_burn_in_target_accepts_per_vertex` is deprecated. "
            "Use `burn_in_target_accepts_per_vertex` (same semantics).",
            DeprecationWarning, stacklevel=2,
        )
        if burn_in_target_accepts_per_vertex is None:
            burn_in_target_accepts_per_vertex = float(
                topology_burn_in_target_accepts_per_vertex
            )

    # --- Seed network ---------------------------------------------------- #
    if seed_kind == "crystal_srs":
        # Crystalline Z=3 seed (default: gyroid/srs). Hemmann/Saba 2026
        # precedent. Every initial bond has the same length and connectivity
        # is by construction.
        positions, edges, seed_meta = crystal_seed_network(
            N, box, edge_length, rng,
            lattice=seed_lattice,
            jitter_sigma=seed_jitter_sigma,
            strict_tiling=strict_tiling,
        )
        if seed_meta["N_actual"] != N:
            # Tiling rounded N. Re-derive num_rods to stay consistent.
            N = seed_meta["N_actual"]
            num_rods = (3 * N) // 2
        neighbors = build_neighbors(N, edges)
        if verbose:
            seed_lengths = np.linalg.norm(
                pbc_displacement(
                    positions[edges[:, 1]] - positions[edges[:, 0]], box
                ),
                axis=1,
            )
            print(f"[gen] crystal seed lattice='{seed_lattice}' "
                  f"tile={seed_meta['tile']} "
                  f"a={seed_meta['lattice_constant'][0]:.3f}: "
                  f"bond length mean={seed_lengths.mean():.3f}, "
                  f"std={seed_lengths.std():.3f}, "
                  f"min={seed_lengths.min():.3f}, "
                  f"max={seed_lengths.max():.3f}")
    elif seed_kind == "random_bm2000":
        # Sellers's literally-cited random seed (refs [13,14] = Vink 2001
        # / Mousseau-Barkema 2001). Hamiltonian-cycle scaffold + chord
        # matching loop expansion to Z=3, with BM2000 § II.A min-separation.
        positions, edges, seed_meta = random_seed_network_bm2000(
            N, box, edge_length, rng,
            min_separation_frac=bm2000_min_separation_frac,
            rc_start_frac=bm2000_rc_start_frac,
            rc_grow_frac=bm2000_rc_grow_frac,
            rc_max_frac=bm2000_rc_max_frac,
            verbose=verbose,
        )
        neighbors = build_neighbors(N, edges)
        if verbose:
            seed_lengths = np.linalg.norm(
                pbc_displacement(
                    positions[edges[:, 1]] - positions[edges[:, 0]], box
                ),
                axis=1,
            )
            print(f"[gen] random BM2000 seed: "
                  f"min_sep={seed_meta['min_separation_frac']:.3f}*d0, "
                  f"rc_final={seed_meta['rc_final']:.3f}, "
                  f"outer_passes={seed_meta['outer_passes']}, "
                  f"bond length mean={seed_lengths.mean():.3f}, "
                  f"std={seed_lengths.std():.3f}, "
                  f"min={seed_lengths.min():.3f}, "
                  f"max={seed_lengths.max():.3f}")
    else:
        raise ValueError(
            f"unknown seed_kind={seed_kind!r}; expected "
            f"'crystal_srs' or 'random_bm2000'."
        )

    if verbose:
        diag_seed = cluster_diagnostics(positions, edges, neighbors, box, edge_length)
        print(
            f"[gen] post-seed cluster diag: "
            f"{_format_cluster_diagnostics(diag_seed, edge_length)}"
        )

    # Long-bond guard + initial settle, packaged as a reusable closure so a
    # collapsed random seed can be rebuilt (see the seed-increment fallback
    # below). A raw random seed (random_bm2000) carries a broad bond-length
    # tail (mean ~1.3 d0, a few bonds past 2 d0; intrinsic to degree-3 +
    # girth-5 on Poisson points -- a crystal seed is monodisperse at ~1.0 d0).
    # The Sellers Eq. 2 relax has no non-bonded repulsion, so contracting a
    # long bond can drag its endpoints through a neighbour and collapse them.
    # Pre-spread with a self-contained soft-sphere potential, then relax; if
    # the relax still lands an at-risk pair, re-spread harder and retry.
    def _soft_start_and_settle(pos, edges_l, neighbors_l):
        ctx = _RelaxContext(N, box, edge_length, weights,
                            use_jax=use_jax, use_jaxopt=use_jaxopt_eff)
        ctx.update_topology(edges_l, neighbors_l)
        bond_max = float(np.linalg.norm(
            pbc_displacement(pos[edges_l[:, 1]] - pos[edges_l[:, 0]], box),
            axis=1,
        ).max())
        # A long-bond tail (random_bm2000) needs the repulsion guard; a
        # monodisperse crystal seed (bond_max ~1.0 d0) does not and takes the
        # plain relax path.
        needs_repulsion = bond_max > 1.5 * edge_length
        if needs_repulsion and ctx.use_jax:
            # Robust, size-scalable settle: Sellers Eq. 2 PLUS a decaying
            # soft-sphere repulsion annealed to a small floor. A pure
            # repulsion-free relax collapses large random seeds (the
            # contraction itself is the hazard, and seed-side spreading is
            # undone by it); keeping repulsion *inside* the settle and ramping
            # it down reaches d0 without crushing coincidences. The WWW anneal
            # still sees pure Eq. 2.
            pos, srep = settle_seed_with_repulsion(
                pos, ctx, edges_l, box, edge_length, verbose=verbose,
            )
            diag = cluster_diagnostics(
                pos, edges_l, neighbors_l, box, edge_length
            )
            if verbose:
                print(
                    f"[gen] repulsion-settle: min_nb "
                    f"{srep['min_nb_before'] / edge_length:.3f}d0 -> "
                    f"{srep['min_nb_after'] / edge_length:.3f}d0, "
                    f"bond_max {srep['bond_max_after'] / edge_length:.2f}d0, "
                    f"corrections={srep['corrections']}; "
                    f"{_format_cluster_diagnostics(diag, edge_length)}"
                )
            return pos, None, diag
        # NumPy fallback (no JAX): soft-start spread then plain relax with a
        # re-spread retry. Works for small N; large N needs the JAX path above.
        if needs_repulsion:
            pos, soft_info = soft_start_seed_relax(
                pos, edges_l, box, edge_length, verbose=verbose,
            )
            if verbose:
                print(
                    f"[gen] soft-start seed spread: min_nb "
                    f"{soft_info['min_nb_before'] / edge_length:.3f}d0 -> "
                    f"{soft_info['min_nb_after'] / edge_length:.3f}d0, "
                    f"bond_max {soft_info['bond_max_after'] / edge_length:.2f}d0, "
                    f"{soft_info['outer_passes']} passes"
                )
        diag = None
        E_final = None
        max_init_attempts = 4
        for _attempt in range(max_init_attempts):
            pos, E_final, _ = relax(pos, ctx, max_iter=relax_global_iters)
            pos = pos - box * np.round(pos / box)
            diag = cluster_diagnostics(
                pos, edges_l, neighbors_l, box, edge_length
            )
            if verbose:
                prl = np.linalg.norm(
                    pbc_displacement(
                        pos[edges_l[:, 1]] - pos[edges_l[:, 0]], box
                    ),
                    axis=1,
                )
                print(
                    f"[gen] initial relax (attempt {_attempt + 1}): "
                    f"E={E_final:.4g}, bond length mean={prl.mean():.3f} "
                    f"(target d0={edge_length}); "
                    f"{_format_cluster_diagnostics(diag, edge_length)}"
                )
            if (
                diag["min_non_bonded"] >= 0.6 * edge_length
                or _attempt == max_init_attempts - 1
            ):
                break
            if verbose:
                print(
                    f"[gen] initial relax landed min_nb="
                    f"{diag['min_non_bonded'] / edge_length:.3f}d0 (< 0.6 d0); "
                    f"re-spreading harder and retrying "
                    f"({_attempt + 1}/{max_init_attempts - 1})"
                )
            pos, _ = soft_start_seed_relax(
                pos, edges_l, box, edge_length,
                r_rep_frac=0.95, k_rep=2.0 + _attempt,
                n_outer=24, target_min_nb_frac=0.7, verbose=verbose,
            )
        return pos, E_final, diag

    positions, E0, diag_initrelax = _soft_start_and_settle(
        positions, edges, neighbors
    )

    # Seed-increment fallback (random_bm2000, NumPy path only): the rare
    # residual collapse is a property of one specific seed's bond-length tail,
    # not of the protocol. On the JAX path settle_seed_with_repulsion handles
    # it directly, so this is skipped there -- otherwise a (systematic) large-N
    # collapse would grind through several full seed rebuilds before raising.
    # On the NumPy path (small N, no repulsion-settle) a fresh RNG resolves it
    # while leaving the pure-Eq.2 relax/anneal untouched.
    seed_attempt = 0
    max_seed_attempts = 6
    while (
        seed_kind == "random_bm2000"
        and not use_jax
        and diag_initrelax["min_non_bonded"] < 0.4 * edge_length
        and seed_attempt < max_seed_attempts
    ):
        seed_attempt += 1
        rng = np.random.default_rng(int(seed) + 7919 * seed_attempt)
        if verbose:
            print(
                f"[gen] initial relax collapsed (min_nb="
                f"{diag_initrelax['min_non_bonded'] / edge_length:.3f}d0); "
                f"rebuilding random seed "
                f"(attempt {seed_attempt}/{max_seed_attempts})"
            )
        positions, edges, seed_meta = random_seed_network_bm2000(
            N, box, edge_length, rng,
            min_separation_frac=bm2000_min_separation_frac,
            rc_start_frac=bm2000_rc_start_frac,
            rc_grow_frac=bm2000_rc_grow_frac,
            rc_max_frac=bm2000_rc_max_frac,
            verbose=verbose,
        )
        neighbors = build_neighbors(N, edges)
        positions, E0, diag_initrelax = _soft_start_and_settle(
            positions, edges, neighbors
        )

    # Hard fail only if the retries + seed rebuilds could not lift the worst
    # pair out of the collapse zone.
    if diag_initrelax["min_non_bonded"] < 0.4 * edge_length:
        raise RuntimeError(
            f"generate_lsu_network: min non-bonded vertex distance "
            f"{diag_initrelax['min_non_bonded']:.3g} < 0.4*d0="
            f"{0.4 * edge_length:.3g} after the initial relax. The seed "
            f"or the relax produced a near-coincident vertex pair that "
            f"Sellers Eq. 2 cannot resolve (no non-bonded repulsion). "
            f"Try a different seed (seed_kind={seed_kind!r}, seed={seed}) "
            f"or a smaller seed_jitter_sigma."
        )
    if diag_initrelax["min_non_bonded"] < 0.6 * edge_length:
        warnings.warn(
            f"min non-bonded vertex distance "
            f"{diag_initrelax['min_non_bonded']:.3g} is in "
            f"[0.4*d0, 0.6*d0]; cluster diagnostics suggest the seed "
            f"contains near-coincident pairs. The burn-in should clean "
            f"this up via SW moves, but watch for f1 = 0 residuals.",
            stacklevel=2,
        )

    # Topology burn-in: triangular Hemmann profile to lose crystalline /
    # random-seed memory.
    if (burn_in_n_heat + burn_in_n_cool + burn_in_n_quench) > 0:
        positions, edges, neighbors, burn_info = topology_burn_in(
            positions, edges, neighbors, box, edge_length, weights, rng,
            n_heat=burn_in_n_heat,
            n_cool=burn_in_n_cool,
            n_quench=burn_in_n_quench,
            T_max=burn_in_T_max,
            T_max_over_T_melt=burn_in_T_max_over_T_melt,
            P_melt=burn_in_P_melt,
            T_melt_probe_moves=burn_in_T_melt_probe_moves,
            T_melt_probe_T=burn_in_T_melt_probe_T,
            relax_local_iters=relax_local_iters,
            relax_global_iters=relax_global_iters,
            local_shell_depth=local_shell_depth,
            global_fallback_threshold=global_fallback_threshold,
            threshold_energy_relax=threshold_energy_relax,
            c_f=c_f,
            uniformity_weight=uniformity_weight,
            uniformity_kmax=uniformity_kmax,
            target_accepts_per_vertex=burn_in_target_accepts_per_vertex,
            use_jax=use_jax, use_jaxopt=use_jaxopt_eff,
            verbose=verbose,
        )
        if verbose:
            acc = burn_info["accepted"] / max(1, burn_info["proposed"])
            print(
                f"[gen] burn-in done: T_max={burn_info['T_max_used']:.3g} "
                f"T_melt={burn_info['T_melt']} "
                f"moves={burn_info['moves']} "
                f"accepted={burn_info['accepted']} ({acc:.1%}) "
                f"early-rejected={burn_info['early_rejected']}"
            )

    # WWW annealing ----------------------------------------------------------
    positions, edges, neighbors, history = www_anneal(
        positions, edges, neighbors, box, edge_length, weights,
        n_iterations=n_www_iterations,
        T0=initial_temperature, T_final=final_temperature,
        rng=rng,
        target_lsu=target_lsu,
        target_depth=target_depth, target_locality=target_locality,
        target_tolerance=target_tolerance,
        relax_local_iters=relax_local_iters,
        relax_global_iters=relax_global_iters,
        relax_global_every=relax_global_every,
        global_fallback_threshold=global_fallback_threshold,
        local_shell_depth=local_shell_depth,
        uniformity_weight=uniformity_weight,
        uniformity_kmax=uniformity_kmax,
        threshold_energy_relax=threshold_energy_relax,
        c_f=c_f,
        cycle_size=cycle_size,
        check_lsu_every=check_lsu_every,
        use_jax=use_jax, use_jaxopt=use_jaxopt_eff, verbose=verbose,
    )

    # Final clean-up: one short full-N polish to settle bond-length residual.
    final_ctx = _RelaxContext(N, box, edge_length, weights,
                              use_jax=use_jax, use_jaxopt=use_jaxopt_eff)
    final_ctx.update_topology(edges, neighbors)
    positions, _, _ = relax(positions, final_ctx,
                            max_iter=min(relax_local_iters, 50))
    positions = positions - box * np.round(positions / box)

    # Connectivity sanity check
    if not is_connected(N, edges):
        raise RuntimeError("Final network is disconnected. This should not "
                           "happen if SW moves rejected disconnections; "
                           "please report.")

    rods = network_to_rods(positions, edges, box,
                           pbc_duplicate_boundary_rods=pbc_duplicate_boundary_rods,
                           clip_endpoints_to_box=clip_endpoints_to_box)
    if verbose:
        # Final LSU
        phi_final = compute_lsu(positions, edges, neighbors, box,
                                depth=target_depth, locality=target_locality,
                                max_pairs=2000, rng=rng)
        print(f"[gen] final phi_{target_depth}{target_locality} = {phi_final:.4f} "
              f"(target {target_lsu}, tol {target_tolerance})")
        rod_lengths = np.linalg.norm(rods[:, 3:6] - rods[:, 0:3], axis=1)
        print(f"[gen] rod lengths: mean={rod_lengths.mean():.3f}, "
              f"std={rod_lengths.std():.3f}, "
              f"min={rod_lengths.min():.3f}, max={rod_lengths.max():.3f}")
        diag_final = cluster_diagnostics(
            positions, edges, neighbors, box, edge_length
        )
        print(
            f"[gen] final cluster diag: "
            f"{_format_cluster_diagnostics(diag_final, edge_length)}"
        )
        if pbc_duplicate_boundary_rods:
            n_extra = rods.shape[0] - num_rods
            print(f"[gen] rendered {rods.shape[0]} rods "
                  f"({num_rods} unique edges + {n_extra} PBC-image duplicates "
                  f"for face-crossing edges)")
        else:
            print(f"[gen] rendered {rods.shape[0]} rods (one per unique edge; "
                  f"PBC duplication disabled)")
    return rods
