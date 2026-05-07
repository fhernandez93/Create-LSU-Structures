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
import time
from itertools import permutations
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

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
def poisson_disk_pbc(
    N: int,
    box: np.ndarray,
    r_min: float,
    rng: np.random.Generator,
    soften_factor: float = 0.95,
    soften_after_failures: int = 200,
    max_total_tries: Optional[int] = None,
) -> np.ndarray:
    """Random hard-core sampling of N points in a periodic box.

    Implements the placement step of Barkema & Mousseau (PRB 62, 4985, 2000)
    section II.A: uniform random positions in [-L/2, L/2]^3 (PBC) with the
    minimum-image distance between any two points constrained to be >= r_min.
    If rejection rate becomes pathological the constraint is softened by
    `soften_factor`, mirroring BM's tolerance to a few too-close pairs in
    the seed configuration.
    """
    box = np.asarray(box, dtype=np.float64).reshape(3)
    half = box / 2.0
    if max_total_tries is None:
        max_total_tries = N * 2000

    positions = np.empty((N, 3), dtype=np.float64)
    placed = 0
    r_min_curr = float(r_min)
    consecutive_failures = 0
    total_tries = 0

    while placed < N and total_tries < max_total_tries:
        candidate = rng.uniform(-half, half)
        total_tries += 1
        if placed == 0:
            positions[0] = candidate
            placed = 1
            continue
        diff = candidate - positions[:placed]
        diff -= box * np.round(diff / box)
        if np.min(np.linalg.norm(diff, axis=1)) >= r_min_curr:
            positions[placed] = candidate
            placed += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= soften_after_failures:
                r_min_curr *= soften_factor
                consecutive_failures = 0

    if placed < N:
        raise RuntimeError(
            f"poisson_disk_pbc: placed only {placed}/{N} after {total_tries} "
            f"tries (final r_min={r_min_curr:.4g}, initial {r_min:.4g})."
        )
    return positions


def _build_trivalent_proximity_graph(
    positions: np.ndarray,
    box: np.ndarray,
    r_cut: float,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Build a 3-regular edge set on `positions` using BM-style proximity bonds.

    Adapted from Barkema-Mousseau (PRB 62, 4985, 2000) §II.A, which targets
    tetravalent (Si) networks via a single loop visiting each atom twice.
    For our trivalent (γ = 3) case the loop visits each atom once and we
    add a chord matching to lift degree from 2 to 3.

    Stage A — Hamiltonian-like cycle (every vertex at deg = 2):
      - Seed with a triangle of 3 mutually-close vertices.
      - Iteratively insert an unbonded vertex A into an existing edge
        (B, C) when dist(A,B), dist(A,C) ≤ r_cut: remove (B, C), add
        (A, B), (A, C). This is the BM elementary move (their Fig. 1)
        for the +1-edge step. A goes from deg 0 to deg 2; B, C stay at 2.

    Stage B — Chord matching (every vertex at deg = 3):
      - Repeatedly add the shortest available edge between two vertices
        both still at deg 2 that are not already bonded.

    Returns the edge array (E, 2) on success or None if the cutoff is too
    tight (caller should widen r_cut and retry).
    """
    N = positions.shape[0]
    box = np.asarray(box, dtype=np.float64).reshape(3)

    diff = positions[:, None, :] - positions[None, :, :]
    diff -= box * np.round(diff / box)
    dists = np.linalg.norm(diff, axis=-1)

    deg = np.zeros(N, dtype=np.int64)
    adj = np.zeros((N, N), dtype=bool)
    edge_set: set = set()  # (min(i,j), max(i,j))

    def add_edge(i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        adj[i, j] = True; adj[j, i] = True
        edge_set.add((a, b))
        deg[i] += 1; deg[j] += 1

    def remove_edge(i: int, j: int) -> None:
        a, b = (i, j) if i < j else (j, i)
        adj[i, j] = False; adj[j, i] = False
        edge_set.discard((a, b))
        deg[i] -= 1; deg[j] -= 1

    # ===== Stage A: Hamiltonian-like cycle via BM loop expansion =====
    # Seed: triangle of 3 mutually-close vertices.
    iu, ju = np.triu_indices(N, k=1)
    pair_d = dists[iu, ju]
    seed_idx = int(np.argmin(pair_d))
    i_seed, j_seed = int(iu[seed_idx]), int(ju[seed_idx])
    score = np.maximum(dists[i_seed], dists[j_seed]).copy()
    score[i_seed] = np.inf; score[j_seed] = np.inf
    k_seed = int(np.argmin(score))
    add_edge(i_seed, j_seed)
    add_edge(j_seed, k_seed)
    add_edge(k_seed, i_seed)

    inserted = np.zeros(N, dtype=bool)
    inserted[[i_seed, j_seed, k_seed]] = True

    # Vectorised per-iteration: build (|R|, |E|) cost matrix, take argmin.
    # Per-iter cost dominated by the (|R|, |E|) ≈ (N, 1.5N) numpy slice +
    # fancy indexing (~milliseconds at N=10³). Falls back to a (|R|, N)
    # nearest-pair scan when no standard insertion is available.
    while not inserted.all():
        remaining_arr = np.where(~inserted)[0]
        edges_arr = np.fromiter(
            (idx for ab in edge_set for idx in ab),
            dtype=np.int64, count=2 * len(edge_set),
        ).reshape(-1, 2)
        B_arr = edges_arr[:, 0]; C_arr = edges_arr[:, 1]

        d_to_B = dists[remaining_arr[:, None], B_arr[None, :]]
        d_to_C = dists[remaining_arr[:, None], C_arr[None, :]]
        d_BC = dists[B_arr, C_arr]
        valid = (d_to_B < r_cut) & (d_to_C < r_cut)
        valid &= (deg[B_arr] < 3)[None, :] & (deg[C_arr] < 3)[None, :]
        valid &= ~adj[remaining_arr[:, None], B_arr[None, :]]
        valid &= ~adj[remaining_arr[:, None], C_arr[None, :]]

        cost = np.where(valid, d_to_B + d_to_C - d_BC[None, :], np.inf)
        flat = int(np.argmin(cost))
        i_r, j_e = divmod(flat, edges_arr.shape[0])
        if np.isfinite(cost[i_r, j_e]):
            A = int(remaining_arr[i_r])
            B = int(B_arr[j_e]); C = int(C_arr[j_e])
            remove_edge(B, C)
            add_edge(A, B)
            add_edge(A, C)
            inserted[A] = True
            continue

        # Fallback: each remaining A picks its 2 nearest available
        # neighbours (deg < 3, not already adjacent). May saturate those
        # neighbours to deg 3 a step early; Stage B issues fewer chords.
        d_R = dists[remaining_arr]
        valid_n = (d_R < r_cut) & (d_R > 0)
        valid_n &= (deg < 3)[None, :]
        valid_n &= ~adj[remaining_arr]
        masked_d = np.where(valid_n, d_R, np.inf)
        if masked_d.shape[1] < 2:
            return None
        partn = np.argpartition(masked_d, 1, axis=1)[:, :2]
        top2_d = np.take_along_axis(masked_d, partn, axis=1)
        rows_invalid = ~np.isfinite(top2_d).all(axis=1)
        costs = np.where(rows_invalid, np.inf, top2_d.sum(axis=1))
        best_r = int(np.argmin(costs))
        if not np.isfinite(costs[best_r]):
            return None
        A = int(remaining_arr[best_r])
        B = int(partn[best_r, 0]); C = int(partn[best_r, 1])
        add_edge(A, B); add_edge(A, C)
        inserted[A] = True

    # ===== Stage B: chord matching to deg = 3 =====
    # Per-iteration: argmin on the (|D2|, |D2|) submatrix of `dists` masked
    # by the (|D2|, |D2|) sub-block of `adj`. Vectorised — no Python loops
    # over neighbour sets.
    while not np.all(deg == 3):
        deg2_idx = np.where(deg == 2)[0]
        if deg2_idx.size == 0:
            break
        sub = dists[np.ix_(deg2_idx, deg2_idx)].copy()
        sub_adj = adj[np.ix_(deg2_idx, deg2_idx)]
        np.fill_diagonal(sub, np.inf)
        sub[sub_adj] = np.inf
        flat = int(np.argmin(sub))
        ii, jj = divmod(flat, deg2_idx.size)
        if not np.isfinite(sub[ii, jj]):
            return None
        i, j = int(deg2_idx[ii]), int(deg2_idx[jj])
        add_edge(i, j)

    edges = np.array(sorted(edge_set), dtype=np.int64)
    return edges


def bm_initial_network(
    N: int,
    box: Union[float, Tuple[float, float, float], np.ndarray],
    d0: float,
    rng: np.random.Generator,
    r_min_ratio: float = 0.7,
    r_cut_ratio: float = 1.7,
    layouts_per_cutoff: int = 4,
    max_cutoff_widenings: int = 8,
    r_cut_widen: float = 1.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Barkema-Mousseau-style random initial trivalent network.

    Places `N` vertices uniformly inside the periodic cube under a hard-core
    minimum separation `r_min = r_min_ratio * d0`, then builds a connected
    3-regular graph whose edges connect physically nearby vertices using a
    greedy nearest-neighbour pass plus BM loop-expansion repair (see
    ``_build_trivalent_proximity_graph``).

    For each candidate cutoff `r_cut` we try `layouts_per_cutoff` independent
    Poisson-disk layouts before giving up and widening `r_cut`. This keeps
    seed bonds short (close to `d0`) by preferring a fresh position draw
    over a permissive cutoff, since widening the cutoff lets repair
    edges bridge long distances.

    Returns
    -------
    positions : (N, 3) float64
        Vertex coordinates centred in [-L/2, L/2]^3.
    edges : (E, 2) int64
        Edge list with E = 3*N // 2.

    Raises
    ------
    RuntimeError
        If a connected 3-regular network cannot be built within the attempt
        budget. Either lower the density (raise box size) or raise
        ``r_cut_ratio``.
    """
    if N < 4 or N % 2 != 0:
        raise ValueError(f"N must be even and >= 4, got {N}")

    box_arr = coerce_box(box)
    r_min = r_min_ratio * d0
    r_cut = r_cut_ratio * d0

    last_err = None
    for widen_step in range(max_cutoff_widenings):
        for layout in range(layouts_per_cutoff):
            try:
                positions = poisson_disk_pbc(N, box_arr, r_min, rng)
            except RuntimeError as e:
                r_min *= 0.95
                last_err = e
                continue
            edges = _build_trivalent_proximity_graph(positions, box_arr, r_cut, rng)
            if edges is None:
                continue
            if not is_connected(N, edges):
                continue
            return positions, edges
        r_cut *= r_cut_widen
    raise RuntimeError(
        f"bm_initial_network: failed after {max_cutoff_widenings} cutoff "
        f"widenings × {layouts_per_cutoff} layouts (N={N}, "
        f"box={box_arr.tolist()}, d0={d0}, final r_min={r_min:.4g}, "
        f"r_cut={r_cut:.4g}). Last placement error: {last_err}"
    )


def random_3regular_graph(N: int, rng: np.random.Generator,
                          max_attempts: int = 200) -> np.ndarray:
    """Generate a connected simple 3-regular graph on N vertices.

    Uses the configuration model with rejection: 3 stubs per vertex are
    randomly paired; pairings producing a self-loop or a parallel edge are
    discarded and the procedure restarts.

    Note
    ----
    Retained for reference. ``generate_lsu_network`` no longer uses this
    seed; it calls :func:`bm_initial_network`, which produces a seed with
    short, proximity-correlated bonds (Barkema-Mousseau 2000).

    Returns
    -------
    edges : (E, 2) int64 array  with E = 3*N // 2
    """
    if N < 4 or N % 2 != 0:
        raise ValueError(f"N must be even and >= 4, got {N}")

    for _ in range(max_attempts):
        stubs = np.repeat(np.arange(N, dtype=np.int64), 3)
        rng.shuffle(stubs)
        edges = stubs.reshape(-1, 2)

        # Reject self-loops
        if np.any(edges[:, 0] == edges[:, 1]):
            continue
        # Reject parallel edges
        canon = np.sort(edges, axis=1)
        keys = canon[:, 0].astype(np.int64) * N + canon[:, 1]
        if len(np.unique(keys)) != len(keys):
            continue
        # Connectivity
        if is_connected(N, edges):
            return edges
    raise RuntimeError(
        f"Could not generate a connected simple 3-regular graph on {N} "
        f"vertices in {max_attempts} attempts."
    )


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
    f1 = np.sum((L - d0) ** 2)

    # --- f2: bond angles (target cos = -1/2) ------------------------------ #
    p_v = pos[triples[:, 0]]
    p_n1 = pos[triples[:, 1]]
    p_n2 = pos[triples[:, 2]]
    e1 = pbc_displacement(p_n1 - p_v, box)
    e2 = pbc_displacement(p_n2 - p_v, box)
    n1 = np.linalg.norm(e1, axis=1)
    n2 = np.linalg.norm(e2, axis=1)
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
        f1 = jnp.sum((L - d0) ** 2)

        # f2: bond angles, target cos = -1/2
        e1 = _pbc_jax(pos[triples[:, 1]] - pos[triples[:, 0]], box)
        e2 = _pbc_jax(pos[triples[:, 2]] - pos[triples[:, 0]], box)
        n1 = jnp.linalg.norm(e1, axis=1)
        n2 = jnp.linalg.norm(e2, axis=1)
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
                               box_j, d0_j, w_j):
        # Module-level wrapper kept identity-stable so that jaxopt's internal
        # JIT cache stays warm across Stone-Wales updates. Topology arrays
        # are passed as run-time arguments (not closure captures) to avoid
        # retracing when their *values* change.
        return _value_and_grad_jit(x_flat, edges_j, triples_j, quads_j,
                                   box_j, d0_j, w_j)


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
            self._mask_flat_j = None
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
                self._mask_flat_j = None
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


def relax(
    positions: np.ndarray,
    ctx: _RelaxContext,
    max_iter: int,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """L-BFGS relaxation using the cached JIT'd kernel in `ctx`.

    If `ctx.set_moving_mask(mask)` has been called with a non-None mask, the
    relaxation is restricted to the masked vertices: out-of-shell positions
    are held fixed (Vink/Mousseau-Barkema scheme that Sellers cites). On the
    JAX path this is implemented by zeroing the gradient for frozen
    components inside `_jax_value_and_grad`. On the NumPy path the moving
    DOFs are passed to scipy as a sub-vector, with frozen positions held
    constant inside the closure.

    Default JAX path: ``scipy.optimize.minimize(method="L-BFGS-B")`` driving
    the JIT-compiled ``value_and_grad`` (single host→device call per
    gradient eval, ~26 µs on CPU). Optional ``ctx.use_jaxopt=True`` swaps
    in ``jaxopt.LBFGS`` which keeps the L-BFGS loop on-device — only worth
    it on a GPU; on CPU jaxopt's per-call dispatch overhead (~2 s) makes
    it 50–150× slower than scipy+JIT regardless of problem size. The
    jaxopt path does NOT honour the moving mask (cf. set_moving_mask) yet;
    use the default scipy+JIT path for local-shell relaxation.
    """
    if ctx.use_jaxopt:
        solver = ctx.get_jaxopt_solver(max_iter, tol)
        x0 = jnp.asarray(positions.reshape(-1))
        res = solver.run(
            x0,
            ctx._edges_j, ctx._triples_j, ctx._quads_j,
            ctx._box_j, ctx._d0_j, ctx._w_j,
        )
        new_pos = np.asarray(res.params, dtype=np.float64).reshape(ctx.N, 3)
        return new_pos, float(res.state.value)
    if ctx.use_jax:
        # Mask (if any) is applied inside ctx.value_and_grad — frozen
        # components have zero gradient and L-BFGS leaves them fixed.
        def fun(x):
            return ctx.value_and_grad(x)
        res = minimize(fun, positions.reshape(-1), jac=True, method="L-BFGS-B",
                       options={"maxiter": max_iter, "gtol": tol})
        return res.x.reshape(ctx.N, 3), float(res.fun)

    # NumPy path. If a mask is set, optimise only the moving DOFs as a
    # sub-vector; scipy's finite-difference gradient then perturbs only those.
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
    # edges (i,c) and (j,d) become (i,d) and (j,c)
    edges[ek1] = (i, d)
    edges[ek2] = (j, c)
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
    edges[ek1] = (i, c)
    edges[ek2] = (j, d)
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
    relax_global_every: int = 200,
    local_shell_depth: Optional[int] = 4,
    check_lsu_every: int = 500,
    use_jax: bool = False,
    use_jaxopt: bool = False,
    verbose: bool = True,
):
    """Run WWW simulated annealing. Returns (positions, edges, neighbors, history).
    """
    N = positions.shape[0]
    ctx = _RelaxContext(N, box, d0, weights, use_jax=use_jax, use_jaxopt=use_jaxopt)
    ctx.update_topology(edges, neighbors)
    E_curr = ctx.energy(positions.reshape(-1))
    history = {"iter": [], "T": [], "E": [], "lsu": [], "accepted": 0,
               "proposed": 0}

    accepted = 0
    proposed = 0
    log_ratio = math.log(T_final / T0) if T0 > 0 else 0.0
    t_start = time.time()

    for it in range(n_iterations):
        T = T0 * math.exp(log_ratio * it / max(1, n_iterations - 1))

        move = stone_wales_propose(edges, neighbors, rng)
        if move is None:
            continue
        proposed += 1
        _ek1, (sw_i, sw_c, sw_j, sw_d), _ek2 = move

        # Snapshot positions for revert
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
        # masking. Setting `local_shell_depth=None` falls back to the old
        # full-N relaxation (kept for diagnostics, not recommended).
        if local_shell_depth is not None and local_shell_depth > 0:
            seed_verts = np.array([sw_i, sw_c, sw_j, sw_d], dtype=np.int64)
            shell = compute_local_shell_mask(seed_verts, neighbors,
                                             local_shell_depth, N)
            ctx.set_moving_mask(shell)
        else:
            ctx.set_moving_mask(None)

        new_pos, E_new = relax(positions, ctx, max_iter=relax_local_iters)

        dE = E_new - E_curr
        # Metropolis acceptance
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            positions = new_pos
            E_curr = E_new
            accepted += 1
        else:
            # Reject: revert topology and positions
            stone_wales_revert(edges, neighbors, move)
            ctx.update_topology(edges, neighbors)
            positions = pos_before

        # Periodic global relax — full-network polish (no shell mask).
        if relax_global_every > 0 and (it + 1) % relax_global_every == 0:
            ctx.set_moving_mask(None)
            positions, E_curr = relax(positions, ctx,
                                      max_iter=relax_global_iters)
            # L-BFGS preserves energy under global PBC translation, so
            # positions can drift outside [-L/2, L/2]^3 over many iterations.
            # Wrap back into the canonical box; this is idempotent and keeps
            # downstream rod-export visualisation centred on the unit cell.
            positions = positions - box * np.round(positions / box)

        # Periodic LSU check + early exit
        if check_lsu_every > 0 and (it + 1) % check_lsu_every == 0:
            phi = compute_lsu(
                positions, edges, neighbors, box,
                depth=target_depth, locality=target_locality,
                max_pairs=2000, rng=rng,
            )
            history["iter"].append(it + 1)
            history["T"].append(T)
            history["E"].append(E_curr)
            history["lsu"].append(phi)
            if verbose:
                acc_rate = accepted / max(1, proposed)
                print(
                    f"[WWW it={it+1:6d}] T={T:.4g}  E={E_curr:.4g}  "
                    f"phi_{target_depth}{target_locality}={phi:.4f}  "
                    f"acc={acc_rate:.2%}  "
                    f"elapsed={time.time()-t_start:.1f}s"
                )
            if verbose and (it + 1) == check_lsu_every:
                acc_rate = accepted / max(1, proposed)
                if acc_rate < 0.05:
                    print(
                        f"[WWW] WARNING: acceptance rate {acc_rate:.1%} after "
                        f"{it+1} iterations — nearly all moves are rejected. "
                        f"relax_local_iters={relax_local_iters} is likely too small: "
                        f"L-BFGS does not converge to the new local minimum, so ΔE "
                        f"appears positive even for good topology changes. "
                        f"Raise relax_local_iters to ≥100 (benchmark showed ΔE crosses "
                        f"zero between 30 and 100 iterations for N=1102)."
                    )
            if target_lsu is not None and abs(phi - target_lsu) <= target_tolerance:
                if verbose:
                    print(f"[WWW] target LSU {target_lsu} reached "
                          f"(measured {phi:.4f}); stopping.")
                break

    history["accepted"] = accepted
    history["proposed"] = proposed
    return positions, edges, neighbors, history


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
def network_to_rods(
    positions: np.ndarray,
    edges: np.ndarray,
    box: np.ndarray,
    pbc_duplicate_boundary_rods: bool = True,
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

    Set ``pbc_duplicate_boundary_rods=False`` to suppress duplication and
    get one rod per unique edge (the prior behaviour). Output shape is
    then exactly ``(E, 6)``.

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

    if not pbc_duplicate_boundary_rods:
        return render_a

    # Render 2: b_canon as the in-box endpoint. Distinct from render 1 iff
    # the edge crosses ≥1 face (i.e. a_canon + d_ab != b_canon).
    render_b = np.concatenate([b_canon, b_canon - d_ab], axis=1)

    # An edge crosses a face iff a_canon + d_ab differs from b_canon. Use
    # a generous tolerance so floating-point noise on interior edges
    # doesn't get duplicated.
    crosses = np.any(np.abs(a_canon + d_ab - b_canon) > 1e-9, axis=1)
    return np.concatenate([render_a, render_b[crosses]], axis=0)


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
    relax_global_every: int = 200,
    relax_local_iters: int = 100,
    relax_global_iters: int = 500,
    local_shell_depth: Optional[int] = 4,
    seed: int = 42,
    use_jax: Optional[bool] = None,
    use_jaxopt: bool = False,
    pbc_duplicate_boundary_rods: bool = True,
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
        Target Φ_{1,1} or Φ_{2,2}. Provide exactly one of the two.
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
        terms. Default: all 1.0.
    target_tolerance : float
        Tolerance for early exit.
    check_lsu_every : int
        How often (in WWW iterations) to measure the current LSU.
    relax_global_every : int
        How often to perform a full L-BFGS-B relaxation (default 200).
        Frequent global polishes keep the system near its geometric local
        minimum so that ΔE between accepted moves reflects topology quality
        rather than residual geometric strain.
    relax_local_iters : int
        L-BFGS-B iteration cap for the local relaxation after each SW move
        (default 100). **Critical for convergence**: benchmarking on N=1102
        shows ΔE is positive for all moves at 30 iters (1% acceptance) and
        crosses into negative territory between 30 and 100 iters. Values below
        ~80 effectively prevent phi from rising regardless of iteration count.
    relax_global_iters : int
        L-BFGS-B iteration cap for the global relaxation (default 500).
    local_shell_depth : int or None
        Depth (in graph edges) of the moving shell around each Stone-Wales
        defect. Vertices farther than this are held fixed during the post-SW
        relaxation. Default 4, matching the Vink/Mousseau-Barkema scheme that
        Sellers's supplement (Methods, refs [13,14]) cites. Set to ``None`` or
        0 to disable shell masking and run a full-N L-BFGS for every relax
        (the previous behaviour; produces corner/edge void clustering at high
        iteration counts because vertices anywhere in the cell can drift
        toward each other under the bonded-only Sellers energy).
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
    weights = (
        float(weights_dict.get("alpha", 10.0)),
        float(weights_dict.get("beta", 1.0)),
        float(weights_dict.get("gamma", 1.0)),
        float(weights_dict.get("delta", 1.0)),
    )

    if lsu_degree_12 is not None:
        target_lsu = float(lsu_degree_12)
        target_depth, target_locality = 1, 1
    else:
        target_lsu = float(lsu_degree_22)
        target_depth, target_locality = 2, 2

    rng = np.random.default_rng(seed)

    use_jaxopt_eff = bool(use_jaxopt and use_jax and HAS_JAXOPT)
    if verbose:
        print(f"[gen] N={N} vertices, E={num_rods} rods, box={box.tolist()}, "
              f"d0={edge_length}, target phi_{target_depth}{target_locality}={target_lsu}, "
              f"jax={'on' if use_jax else 'off'}, "
              f"jaxopt={'on' if use_jaxopt_eff else 'off'}")

    # Seed network -----------------------------------------------------------
    # Barkema-Mousseau (PRB 62, 4985, 2000) §II.A: hard-core uniform vertex
    # placement in the periodic box, then bonds grown along proximity. This
    # gives short, near-d0 seed bonds and avoids the empty-region artifact
    # of the configuration-model seed (random_3regular_graph).
    positions, edges = bm_initial_network(N, box, edge_length, rng)
    edges = edges.copy()
    neighbors = build_neighbors(N, edges)
    if verbose:
        seed_lengths = np.linalg.norm(
            pbc_displacement(positions[edges[:, 1]] - positions[edges[:, 0]], box),
            axis=1,
        )
        print(f"[gen] BM seed: rod length mean={seed_lengths.mean():.3f}, "
              f"std={seed_lengths.std():.3f}, "
              f"min={seed_lengths.min():.3f}, max={seed_lengths.max():.3f}")

    # Initial global relaxation
    init_ctx = _RelaxContext(N, box, edge_length, weights,
                             use_jax=use_jax, use_jaxopt=use_jaxopt_eff)
    init_ctx.update_topology(edges, neighbors)
    positions, E0 = relax(positions, init_ctx, max_iter=relax_global_iters)
    positions = positions - box * np.round(positions / box)
    if verbose:
        print(f"[gen] initial relax: E={E0:.4g}")

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
        local_shell_depth=local_shell_depth,
        check_lsu_every=check_lsu_every,
        use_jax=use_jax, use_jaxopt=use_jaxopt_eff, verbose=verbose,
    )

    # Final clean-up: full relaxation
    final_ctx = _RelaxContext(N, box, edge_length, weights,
                              use_jax=use_jax, use_jaxopt=use_jaxopt_eff)
    final_ctx.update_topology(edges, neighbors)
    positions, _ = relax(positions, final_ctx, max_iter=relax_global_iters * 2)
    positions = positions - box * np.round(positions / box)

    # Connectivity sanity check
    if not is_connected(N, edges):
        raise RuntimeError("Final network is disconnected. This should not "
                           "happen if SW moves rejected disconnections; "
                           "please report.")

    rods = network_to_rods(positions, edges, box,
                           pbc_duplicate_boundary_rods=pbc_duplicate_boundary_rods)
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
        if pbc_duplicate_boundary_rods:
            n_extra = rods.shape[0] - num_rods
            print(f"[gen] rendered {rods.shape[0]} rods "
                  f"({num_rods} unique edges + {n_extra} PBC-image duplicates "
                  f"for face-crossing edges)")
        else:
            print(f"[gen] rendered {rods.shape[0]} rods (one per unique edge; "
                  f"PBC duplication disabled)")
    return rods
