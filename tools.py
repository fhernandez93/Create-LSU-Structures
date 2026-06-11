"""Post-generation analysis tools for LSU networks.

Usage:
    import numpy as np
    import tools

    rods = np.loadtxt('Example/lsu_example_ends.txt')
    report = tools.analyze_network(rods, box=11.44, d0=0.8, label='reference')

    # Or load and analyze in one call:
    tools.analyze_network('Example/lsu_example_ends.txt', label='reference')

    # Compare multiple networks side-by-side:
    tools.compare_networks([
        ('Example/lsu_example_ends.txt', 'reference'),
        ('Example/lsu_generated_2.txt',   'mine'),
    ])

The metrics mirror Hemmann/Saba 2026 § 3 (Order Metrics) plus Sellers's LSU
Φ_nl. Their meaning, in one line each:

  Φ_12, Φ_22         Sellers LSU statistics. Local self-uniformity of n-edge
                     trees up to locality `l`. 1.0 = perfect local symmetry
                     (crystal); reference Type-2 amorphous gyroid sits at
                     Φ_22 ≈ 0.89.
  bond_len_mean/std  Edge length statistics under PBC.
  bond_ang_mean/std  Bond-angle statistics over all (v, n1, n2) triples.
  trihedral_|det|    |det| of the 3×3 matrix of unit bond vectors at each
                     vertex. 0 = coplanar trihedral, ~0.4 = tetrahedral.
  ring_n_fraction    Fraction of edges whose shortest containing cycle has
                     length n. srs has 100 % 10-rings; amorphous gyroids
                     centre on 7-/8-rings.
  voxel_std_4        Std of vertex count over a 4³-voxel grid in the
                     canonical box (Hemmann's homogeneity metric).
  r_nn, r_u, δ_c     Hemmann §3.2 cluster-diagnostic radii in units of d0.
                     `r_u` < 1 indicates vertex clustering; `δ_c` is the
                     critical pore radius.
  dihedral_entropy   Hemmann §3.1 normalised entropy of the dihedral-angle
                     histogram (0 = single peak, 1 = uniform).
  S_v(k) alpha       Hyperuniformity exponent fit S_v(k) ~ k^α at low k.
                     α > 1: hyperuniform; α = 0: random; α < 0: clustered.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import scipy.sparse.csgraph as csg

from lsu_network import build_neighbors, cluster_diagnostics, compute_lsu, crystal_seed_network


# ----------------------------- graph reconstruction ------------------------ #


def rods_to_network(
    rods: np.ndarray, box: np.ndarray, cluster_radius: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct (positions, edges) from a (R, 6) rod-endpoint array.

    Rod endpoints that map to the same canonical-box image (within
    `cluster_radius`) are merged into one vertex. PBC-image duplicates of
    face-crossing rods collapse to the same edge.
    """
    L = float(box[0])
    p1 = rods[:, :3]; p2 = rods[:, 3:]
    endpoints = np.vstack([p1, p2])
    wrapped_pos = (endpoints - L * np.floor(endpoints / L + 0.5)) + L / 2
    tree = cKDTree(wrapped_pos, boxsize=L)
    pairs = tree.query_pairs(r=cluster_radius, output_type='ndarray')
    n = len(wrapped_pos)
    if len(pairs):
        rows = pairs[:, 0]; cols = pairs[:, 1]
        g = coo_matrix((np.ones(len(pairs)), (rows, cols)), shape=(n, n))
        _, labels = csg.connected_components(g, directed=False)
    else:
        labels = np.arange(n)
    nlabels = int(labels.max()) + 1
    positions = np.zeros((nlabels, 3))
    for cid in range(nlabels):
        members = np.where(labels == cid)[0]
        pts = wrapped_pos[members]
        anchor = pts[0]
        diffs = pts - anchor
        diffs -= L * np.round(diffs / L)
        positions[cid] = anchor + diffs.mean(axis=0)
    positions = ((positions + L / 2) % L) - L / 2
    R = len(p1)
    edges_full = np.stack([labels[:R], labels[R:]], axis=1)
    edges = np.unique(np.sort(edges_full, axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return positions, edges


def srs_crystal_rods(
    num_vertices: int = 1000,
    box: Union[float, np.ndarray] = 11.44,
    d0: float = 0.8,
    jitter_sigma: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Build the pristine srs / single-network-gyroid crystal as rod endpoints.

    Returns ``(E, 6)`` array suitable to feed into :func:`analyze_network` as
    a reference for the Φ_22 = 1.0, all-10-rings, fully Bragg-peaked limit.
    The second endpoint of each rod is set to the nearest PBC image of the
    first so that ``rods_to_network`` reconstructs the graph cleanly.

    Parameters
    ----------
    num_vertices
        Requested vertex count; the tiler picks the cubic dimension nearest
        to ``(num_vertices / 8)**(1/3)`` (srs has 8 sites per cubic cell).
    box, d0
        Box side (µm) and target bond length (µm); the seed bond comes out
        to ``a * sqrt(2)/4`` for ``a = box / nx``.
    jitter_sigma
        Gaussian jitter on positions in units of ``d0`` (default 0 = no jitter).
    seed
        RNG seed for jitter.
    """
    rng = np.random.default_rng(seed)
    box_arr = np.asarray([box, box, box], dtype=float) if not hasattr(box, "__len__") else np.asarray(box, dtype=float)
    positions, edges, _ = crystal_seed_network(
        N=num_vertices, box=box_arr, d0=d0, rng=rng,
        lattice="srs", jitter_sigma=float(jitter_sigma),
    )
    L = float(box_arr[0])
    p1 = positions[edges[:, 0]]
    p2 = positions[edges[:, 1]]
    d = p2 - p1
    d -= L * np.round(d / L)
    return np.hstack([p1, p1 + d])


# ----------------------------- metric helpers ------------------------------ #


def _bond_lengths(positions: np.ndarray, edges: np.ndarray, L: float) -> np.ndarray:
    d = positions[edges[:, 1]] - positions[edges[:, 0]]
    d -= L * np.round(d / L)
    return np.linalg.norm(d, axis=1)


def _angle_and_planarity(
    positions: np.ndarray, edges: np.ndarray, L: float
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(positions)
    nbrs: List[List[int]] = [[] for _ in range(N)]
    for i, j in edges:
        nbrs[i].append(j); nbrs[j].append(i)
    angles: List[float] = []; planarities: List[float] = []
    for v in range(N):
        if len(nbrs[v]) != 3:
            continue
        a, b, c = nbrs[v]
        ra = positions[a] - positions[v]; ra -= L * np.round(ra / L); ra /= np.linalg.norm(ra)
        rb = positions[b] - positions[v]; rb -= L * np.round(rb / L); rb /= np.linalg.norm(rb)
        rc = positions[c] - positions[v]; rc -= L * np.round(rc / L); rc /= np.linalg.norm(rc)
        planarities.append(abs(np.linalg.det(np.column_stack([ra, rb, rc]))))
        for u, w in [(ra, rb), (ra, rc), (rb, rc)]:
            angles.append(np.degrees(np.arccos(np.clip(u @ w, -1.0, 1.0))))
    return np.asarray(angles), np.asarray(planarities)


def _skew_angles(
    positions: np.ndarray, edges: np.ndarray, L: float
) -> np.ndarray:
    """Trihedral skew angles χ in degrees (Sellers et al. 2017, Fig. 4d).

    For each 3-valent vertex with bond unit-vectors (e_a, e_b, e_c), χ is the
    angle between one bond and the unit normal of the plane spanned by the
    other two: χ_i = arccos(e_i · (e_j × e_k)/|e_j × e_k|). A flat trihedron
    (all three bonds coplanar) yields χ = 90°. Three values returned per
    vertex.
    """
    N = len(positions)
    nbrs: List[List[int]] = [[] for _ in range(N)]
    for i, j in edges:
        nbrs[i].append(j); nbrs[j].append(i)
    chis: List[float] = []
    for v in range(N):
        if len(nbrs[v]) != 3:
            continue
        a, b, c = nbrs[v]
        ra = positions[a] - positions[v]; ra -= L * np.round(ra / L); ra /= np.linalg.norm(ra)
        rb = positions[b] - positions[v]; rb -= L * np.round(rb / L); rb /= np.linalg.norm(rb)
        rc = positions[c] - positions[v]; rc -= L * np.round(rc / L); rc /= np.linalg.norm(rc)
        for primary, o1, o2 in [(ra, rb, rc), (rb, ra, rc), (rc, ra, rb)]:
            n = np.cross(o1, o2)
            nn = float(np.linalg.norm(n))
            if nn < 1e-12:
                continue
            n /= nn
            chis.append(np.degrees(np.arccos(np.clip(float(primary @ n), -1.0, 1.0))))
    return np.asarray(chis)


def _shortest_ring_per_edge(N: int, edges: np.ndarray) -> np.ndarray:
    adj: List[List[int]] = [[] for _ in range(N)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    out: List[int] = []
    for u, v in edges:
        u, v = int(u), int(v)
        dist = {u: 0}
        q: deque = deque([u])
        found = -1
        while q and found < 0:
            cur = q.popleft()
            for nb in adj[cur]:
                if cur == u and nb == v:
                    continue
                if cur == v and nb == u:
                    continue
                if nb in dist:
                    continue
                dist[nb] = dist[cur] + 1
                if nb == v:
                    found = dist[nb] + 1
                    break
                q.append(nb)
        out.append(found if found > 0 else 0)
    return np.asarray(out)


def _dihedral_entropy(
    positions: np.ndarray, edges: np.ndarray, L: float, nbins: int = 18
) -> Tuple[np.ndarray, float]:
    N = len(positions)
    nbrs: List[List[int]] = [[] for _ in range(N)]
    for a, b in edges:
        nbrs[a].append(b); nbrs[b].append(a)
    phis: List[float] = []
    for i, k in edges:
        ri = positions[i]; rk = positions[k]
        for j in nbrs[i]:
            if j == k:
                continue
            for ll in nbrs[k]:
                if ll == i:
                    continue
                rij = positions[j] - ri; rij -= L * np.round(rij / L)
                rik = rk - ri; rik -= L * np.round(rik / L)
                rkl = positions[ll] - rk; rkl -= L * np.round(rkl / L)
                n1 = np.cross(rij, rik); n2 = np.cross(rik, rkl)
                norm_rik = max(float(np.linalg.norm(rik)), 1e-12)
                phi = np.degrees(np.arctan2(
                    float(np.dot(rik, np.cross(n1, n2)) / norm_rik),
                    float(np.dot(n1, n2)),
                ))
                phi = abs(phi)
                if phi > 180:
                    phi = 360 - phi
                phis.append(phi)
    phis_arr = np.asarray(phis)
    if len(phis_arr) == 0:
        return phis_arr, 0.0
    edges_bins = np.linspace(0, 180, nbins + 1)
    h, _ = np.histogram(phis_arr, bins=edges_bins)
    p = h / h.sum()
    p_nz = p[p > 0]
    entropy = -float(np.sum(p_nz * np.log(p_nz)) / np.log(nbins))
    return phis_arr, entropy


def _vertex_structure_factor(
    positions: np.ndarray, box: np.ndarray, k_modes_max: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    L = float(box[0])
    k0 = 2.0 * np.pi / L
    N = len(positions)
    hkl: List[Tuple[int, int, int]] = []
    for h in range(-k_modes_max, k_modes_max + 1):
        for k in range(-k_modes_max, k_modes_max + 1):
            for l in range(-k_modes_max, k_modes_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl.append((h, k, l))
    kvecs = np.asarray(hkl, dtype=np.float64) * k0
    kmag = np.linalg.norm(kvecs, axis=1)
    phases = kvecs @ positions.T
    re = np.cos(phases).sum(axis=1)
    im = np.sin(phases).sum(axis=1)
    return kmag, (re ** 2 + im ** 2) / N


def _bin_structure_factor(
    kmag: np.ndarray, S: np.ndarray, nbins: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(kmag.min(), kmag.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(nbins); counts = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        m = (kmag >= edges[i]) & (kmag < edges[i + 1])
        if m.sum() > 0:
            means[i] = S[m].mean(); counts[i] = int(m.sum())
    return centers, means, counts


def _vertex_structure_factor_2d_slice(
    positions: np.ndarray,
    box: np.ndarray,
    k_modes_max: int = 8,
    nbins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sellers-Fig.-4h style 2D slice of S_v(k_x, k_y), averaged over k_z.

    Returns (bin_centers_1d, grid_2d) with shape (nbins,) and (nbins, nbins).
    The grid is indexed as ``grid[iy, ix]`` so it can be passed directly to
    ``imshow`` with ``origin='lower', extent=(-kmax, kmax, -kmax, kmax)``.
    Empty bins are 0.
    """
    L = float(box[0])
    k0 = 2.0 * np.pi / L
    N = len(positions)
    if nbins is None:
        nbins = 2 * k_modes_max + 1  # one bin per integer-multiple of k0
    hkl: List[Tuple[int, int, int]] = []
    for h in range(-k_modes_max, k_modes_max + 1):
        for k in range(-k_modes_max, k_modes_max + 1):
            for l in range(-k_modes_max, k_modes_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl.append((h, k, l))
    kvecs = np.asarray(hkl, dtype=np.float64) * k0
    phases = kvecs @ positions.T
    S = (np.cos(phases).sum(axis=1) ** 2 + np.sin(phases).sum(axis=1) ** 2) / N
    edge_kmax = (k_modes_max + 0.5) * k0
    edges = np.linspace(-edge_kmax, edge_kmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ix = np.clip(np.digitize(kvecs[:, 0], edges) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(kvecs[:, 1], edges) - 1, 0, nbins - 1)
    grid_sum = np.zeros((nbins, nbins))
    grid_cnt = np.zeros((nbins, nbins), dtype=int)
    np.add.at(grid_sum, (iy, ix), S)
    np.add.at(grid_cnt, (iy, ix), 1)
    grid = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), 0.0)
    return centers, grid


def _fit_hyperuniformity(
    k_centers: np.ndarray, S_means: np.ndarray, kmax_fit: float
) -> Tuple[float, int]:
    m = (k_centers > 0) & (k_centers <= kmax_fit) & (S_means > 0)
    if m.sum() < 3:
        return float("nan"), 0
    slope, _ = np.polyfit(np.log(k_centers[m]), np.log(S_means[m]), 1)
    return float(slope), int(m.sum())


# ----------------------------- main entry points --------------------------- #


def analyze_network(
    rods_or_path: Union[np.ndarray, str, "os.PathLike"],
    box: Union[float, np.ndarray] = 11.44,
    d0: float = 0.8,
    label: Optional[str] = None,
    verbose: bool = True,
    k_modes_max: int = 8,
    k_modes_max_2d: int = 16,
) -> Dict[str, Any]:
    """Analyse a single LSU network and return a dict of metrics.

    Parameters
    ----------
    rods_or_path
        Either a ``(R, 6)`` rod-endpoint array (e.g. as returned by
        ``generate_lsu_network``) or a path to a file readable by
        ``np.loadtxt`` in the 6-column tab-separated format.
    box, d0
        Box side length (µm) and target bond length (µm). Defaults match the
        reference example.
    label
        Optional label for the verbose printout. Defaults to the file stem if
        a path was given, else "network".
    verbose
        Print a one-screen report.

    Returns
    -------
    dict
        All scalar metrics plus the binned vertex structure factor under
        keys ``S_v_k_centers`` and ``S_v_k_means``.
    """
    if isinstance(rods_or_path, (str, os.PathLike)):
        path = os.fspath(rods_or_path)
        rods = np.loadtxt(path)
        if label is None:
            label = os.path.splitext(os.path.basename(path))[0]
    else:
        rods = np.asarray(rods_or_path, dtype=float)
        if label is None:
            label = "network"

    box_arr = np.asarray(box if hasattr(box, "__len__") else [box, box, box], dtype=float)
    L = float(box_arr[0])

    positions, edges = rods_to_network(rods, box_arr)
    N = len(positions)
    neighbors = build_neighbors(N, edges)

    bl = _bond_lengths(positions, edges, L)
    angles, planarities = _angle_and_planarity(positions, edges, L)
    chis = _skew_angles(positions, edges, L)
    rings = _shortest_ring_per_edge(N, edges)
    diag = cluster_diagnostics(positions, edges, neighbors, box_arr, d0, probe_grid=12)
    # Sellers Eq. 2 convention: Φ_nl = depth-n trees compared for root
    # vertices within l edges. Φ_12 ⇒ depth 1, locality 2.
    phi12 = compute_lsu(positions, edges, neighbors, box_arr, depth=1, locality=2)
    phi22 = compute_lsu(positions, edges, neighbors, box_arr, depth=2, locality=2)
    phis_arr, h_dih = _dihedral_entropy(positions, edges, L)
    kmag, S = _vertex_structure_factor(positions, box_arr, k_modes_max=k_modes_max)
    kc, sm, _ = _bin_structure_factor(kmag, S, nbins=24)
    sk2d_centers, sk2d_grid = _vertex_structure_factor_2d_slice(
        positions, box_arr, k_modes_max=k_modes_max_2d
    )
    alpha_low, n_low = _fit_hyperuniformity(kc, sm, kmax_fit=2.0)
    alpha_lower, n_lower = _fit_hyperuniformity(kc, sm, kmax_fit=3.0)

    ring_vals, ring_cnts = np.unique(rings, return_counts=True)
    ring_dist = {int(v): int(c) for v, c in zip(ring_vals, ring_cnts)}

    result: Dict[str, Any] = {
        "label": label,
        "N": N,
        "E": len(edges),
        "rods_in_file": len(rods),
        "phi12": phi12, "phi22": phi22,
        "bond_len_mean": float(bl.mean()), "bond_len_std": float(bl.std()),
        "bond_len_min": float(bl.min()),  "bond_len_max": float(bl.max()),
        "bond_ang_mean": float(angles.mean()), "bond_ang_std": float(angles.std()),
        "bond_ang_min": float(angles.min()),   "bond_ang_max": float(angles.max()),
        "trihedral_det_mean": float(planarities.mean()),
        "trihedral_coplanar_frac": float(np.mean(planarities < 0.05)),
        "ring_distribution": ring_dist,
        "ring_mean": float(rings[rings > 0].mean()),
        "ring_8_fraction": float((rings == 8).sum() / len(rings)),
        "voxel_std_4": float(diag["voxel_std_4"]),
        "S_low_k2": float(diag["S_low_k2"]),
        "r_nn": float(diag["r_nn"] / d0),
        "r_u": float(diag["r_u"] / d0),
        "delta_c": float(diag["delta_c"] / d0),
        "min_non_bonded": float(diag["min_non_bonded"] / d0),
        "dihedral_entropy": h_dih,
        "S_v_alpha_low": alpha_low, "S_v_alpha_n_low": n_low,
        "S_v_alpha_wide": alpha_lower, "S_v_alpha_n_wide": n_lower,
        "S_v_k_centers": kc.tolist(),
        "S_v_k_means": sm.tolist(),
        "theta_angles_deg": angles,
        "phi_angles_deg": phis_arr,
        "chi_angles_deg": chis,
        "S_v_2d_k_centers": sk2d_centers,
        "S_v_2d_grid": sk2d_grid,
        "box_L": L,
    }

    if verbose:
        _print_report(result)
    return result


def compare_networks(
    inputs: List[Union[Tuple[Union[np.ndarray, str], str], str]],
    box: Union[float, np.ndarray] = 11.44,
    d0: float = 0.8,
) -> List[Dict[str, Any]]:
    """Analyse multiple networks and print a side-by-side comparison table.

    Each entry can be either a path string (label = file stem) or a
    ``(rods_or_path, label)`` tuple.
    """
    results: List[Dict[str, Any]] = []
    for item in inputs:
        if isinstance(item, tuple):
            r, lbl = item
        else:
            r, lbl = item, None
        results.append(analyze_network(r, box=box, d0=d0, label=lbl, verbose=False))
    _print_comparison(results)
    return results


# ----------------------------- pretty-printing ----------------------------- #


def _print_report(r: Dict[str, Any]) -> None:
    print()
    print(f"=== {r['label']} (N={r['N']}, E={r['E']}, rods={r['rods_in_file']}) ===")
    print(f"  LSU:               Φ_12={r['phi12']:.4f}   Φ_22={r['phi22']:.4f}")
    print(f"  bond length (µm):  mean={r['bond_len_mean']:.4f}  std={r['bond_len_std']:.4f}  "
          f"min={r['bond_len_min']:.3f}  max={r['bond_len_max']:.3f}")
    print(f"  bond angle (°):    mean={r['bond_ang_mean']:.2f}  std={r['bond_ang_std']:.2f}  "
          f"min={r['bond_ang_min']:.1f}  max={r['bond_ang_max']:.1f}")
    print(f"  trihedral |det|:   mean={r['trihedral_det_mean']:.4f}  "
          f"coplanar(|det|<0.05)={r['trihedral_coplanar_frac']:.3f}")
    rd = r['ring_distribution']
    ring_str = "  ".join(f"{n}={c}({100*c/r['E']:.1f}%)" for n, c in sorted(rd.items()))
    print(f"  shortest rings:    {ring_str}")
    print(f"  ring mean={r['ring_mean']:.2f}   8-ring fraction={r['ring_8_fraction']:.3f}")
    print(f"  homogeneity:       voxel_std_4={r['voxel_std_4']:.3f}   "
          f"S_low_k2={r['S_low_k2']:.4f}")
    print(f"  cluster radii:     r_nn={r['r_nn']:.3f}·d0   r_u={r['r_u']:.3f}·d0   "
          f"δ_c={r['delta_c']:.3f}·d0   min_non_bonded={r['min_non_bonded']:.3f}·d0")
    print(f"  dihedral entropy:  h_φ={r['dihedral_entropy']:.4f}  (1.0 = uniform)")
    if not np.isnan(r['S_v_alpha_low']):
        print(f"  S_v(k) ~ k^α:      α(k<2.0)={r['S_v_alpha_low']:.3f} "
              f"(n={r['S_v_alpha_n_low']}),  α(k<3.0)={r['S_v_alpha_wide']:.3f}")


def _print_comparison(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    labels = [r['label'] for r in results]
    width = max(max(len(lbl) for lbl in labels), 14) + 2
    rows: List[Tuple[str, str]] = [
        ("Φ_12",                "phi12"),
        ("Φ_22",                "phi22"),
        ("bond len mean",       "bond_len_mean"),
        ("bond len std",        "bond_len_std"),
        ("bond ang mean (°)",   "bond_ang_mean"),
        ("bond ang std (°)",    "bond_ang_std"),
        ("bond ang min (°)",    "bond_ang_min"),
        ("bond ang max (°)",    "bond_ang_max"),
        ("trihedral |det|",     "trihedral_det_mean"),
        ("coplanar frac",       "trihedral_coplanar_frac"),
        ("ring mean",           "ring_mean"),
        ("8-ring fraction",     "ring_8_fraction"),
        ("voxel_std_4",         "voxel_std_4"),
        ("r_u (·d0)",           "r_u"),
        ("δ_c (·d0)",           "delta_c"),
        ("dihedral entropy",    "dihedral_entropy"),
        ("S_v α (k<2)",         "S_v_alpha_low"),
    ]

    header = "metric".ljust(22) + "".join(lbl.center(width) for lbl in labels)
    print()
    print(header)
    print("-" * len(header))
    for label, key in rows:
        row = label.ljust(22)
        for r in results:
            val = r.get(key, float("nan"))
            row += f"{val:.4f}".center(width) if isinstance(val, float) and not np.isnan(val) else "—".center(width)
        print(row)

    # Ring distribution sub-table (only edge-count rings 4..12).
    ring_keys = sorted({k for r in results for k in r['ring_distribution']}.union(range(4, 13)))
    ring_keys = [k for k in ring_keys if 4 <= k <= 12]
    if ring_keys:
        print()
        print("ring length".ljust(22) + "".join(lbl.center(width) for lbl in labels))
        print("-" * len(header))
        for n in ring_keys:
            row = f"  n={n}".ljust(22)
            for r in results:
                pct = 100 * r['ring_distribution'].get(n, 0) / r['E']
                row += f"{pct:.1f}%".center(width)
            print(row)


# ----------------------------- plotting ------------------------------------ #


def plot_comparison(
    results: List[Dict[str, Any]],
    ring_range: Tuple[int, int] = (4, 12),
    sk_logy: bool = True,
    sk_logx: bool = False,
    angle_bins: int = 36,
    sk_2d_log: bool = True,
    sk_2d_shared_clim: bool = True,
    sk_2d_interp: str = "bilinear",
    crystal_indices: Optional[List[int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
):
    """Multi-panel comparison: radial S_v(k), ring distribution, θ/φ/χ angle
    distributions (Sellers Fig. 4d), plus one 2D S_v(k_x, k_y) slice per
    network (Sellers Fig. 4h).

    Layout (2 rows for purely amorphous comparison; 3 rows when crystals are
    present — crystals get their own bottom row to avoid Bragg-peak dynamic
    range squashing the amorphous traces):
      row 0:  [ S_v(k) | rings | θ/φ/χ ]                — amorphous only
      row 1:  [ 2D S_v slice per amorphous network ]    — amorphous only
      row 2:  [ S_v(k) | rings | θ/φ/χ | 2D slices ]    — crystal(s) only

    Parameters
    ----------
    results
        List of dicts returned by :func:`analyze_network`. The 2D slice
        resolution is set by ``k_modes_max_2d`` at analysis time (default
        16 → 33×33 grid).
    ring_range
        Inclusive (min, max) ring lengths to show in the bar chart.
    sk_logy, sk_logx
        Log axes on the radial S_v(k) panel.
    angle_bins
        Histogram bins for θ / φ / χ (0–180°).
    sk_2d_log
        Plot the 2D slices on a log color scale.
    sk_2d_shared_clim
        Use a single (vmin, vmax) across all 2D slices so colors are directly
        comparable. Set False to autoscale each independently.
    sk_2d_interp
        Matplotlib imshow interpolation for the 2D slices.
        ``'bilinear'`` (default), ``'nearest'``, ``'bicubic'`` etc.
    crystal_indices
        Indices into ``results`` that should be drawn in the dedicated
        bottom "crystal" row rather than overlaid in the amorphous panels.
        If None, any entry with ``phi22 > 0.999`` is auto-classified as a
        crystal. Pass ``[]`` to force every entry into the amorphous panels.
    figsize
        Defaults to ``(4.4 * max(3, n_networks), 9.0)``.
    save_path
        If given, the figure is saved here at 150 dpi.

    Returns
    -------
    (fig, axes_dict) with keys ``sk``, ``rings``, ``angles``, ``sk2d``
    (amorphous), and — when crystals are present — ``crystal_sk``,
    ``crystal_rings``, ``crystal_angles``, ``crystal_sk2d``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    if crystal_indices is None:
        crystal_indices = [i for i, r in enumerate(results) if r.get("phi22", 0.0) > 0.999]
    crystal_set = set(crystal_indices)
    amorph = [r for i, r in enumerate(results) if i not in crystal_set]
    crystals = [results[i] for i in crystal_indices]
    has_crystal_row = len(crystals) > 0

    n_amorph = len(amorph)
    n_crystal = len(crystals)
    # cols: top row needs 3; amorph 2D row needs n_amorph; crystal row needs 3 + n_crystal
    crystal_row_cols = 3 + n_crystal if has_crystal_row else 0
    n_cols = max(3, n_amorph, crystal_row_cols)
    n_rows = 3 if has_crystal_row else 2
    if figsize is None:
        figsize = (4.4 * n_cols, 4.5 * n_rows)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.34, wspace=0.34)

    def _plot_sk(ax, networks, title):
        for r in networks:
            kc = np.asarray(r["S_v_k_centers"])
            sm = np.asarray(r["S_v_k_means"])
            m = sm > 0 if sk_logy else np.ones_like(sm, dtype=bool)
            ax.plot(kc[m], sm[m], marker="o", lw=1.4, ms=4, label=r["label"])
        ax.set_xlabel(r"$k$  (1/µm)"); ax.set_ylabel(r"$S_v(k)$")
        ax.set_title(title)
        if sk_logy:
            ax.set_yscale("log")
        if sk_logx:
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    def _plot_rings(ax, networks, title):
        ns = list(range(ring_range[0], ring_range[1] + 1))
        n_series = max(len(networks), 1)
        bar_w = 0.8 / n_series
        x = np.arange(len(ns))
        for i, r in enumerate(networks):
            rd = r["ring_distribution"]
            pct = [100.0 * rd.get(n, 0) / r["E"] for n in ns]
            offset = (i - (n_series - 1) / 2.0) * bar_w
            ax.bar(x + offset, pct, width=bar_w, label=r["label"])
        ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("ring length n"); ax.set_ylabel("% of edges")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    def _plot_angles(ax, networks, title):
        bin_edges = np.linspace(0.0, 180.0, angle_bins + 1)
        centers_a = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_w = 180.0 / angle_bins
        colors = {"theta": "#d62728", "phi": "#2ca02c", "chi": "#1f77b4"}
        linestyles = ["-", "--", ":", "-."]
        for i, r in enumerate(networks):
            ls = linestyles[i % len(linestyles)]
            for key, arr_key, lbl in [
                ("theta", "theta_angles_deg", r"$\theta$  bond"),
                ("phi",   "phi_angles_deg",   r"$\varphi$  dihedral"),
                ("chi",   "chi_angles_deg",   r"$\chi$  skew"),
            ]:
                a = r.get(arr_key)
                if a is None or len(a) == 0:
                    continue
                h, _ = np.histogram(a, bins=bin_edges)
                density = h / (h.sum() * bin_w)
                label = f"{lbl} ({r['label']})" if len(networks) > 1 else lbl
                ax.plot(centers_a, density, color=colors[key], ls=ls, lw=1.6, label=label)
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
        ax.set_xlabel("angle (°)"); ax.set_ylabel("frequency density (1/°)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _plot_2d_row(row_idx, col_start, networks, label):
        if not networks:
            return [], None
        grids = [np.asarray(r["S_v_2d_grid"]) for r in networks]
        if sk_2d_shared_clim and sk_2d_log:
            all_pos = np.concatenate([g.ravel() for g in grids])
            all_pos = all_pos[all_pos > 0]
            vmin = max(float(all_pos.min()), 1e-3) if all_pos.size else 1e-3
            vmax = float(all_pos.max()) if all_pos.size else 1.0
            shared_norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            shared_norm = None
        axes_out = []
        last_im_local = None
        for i, r in enumerate(networks):
            ax = fig.add_subplot(gs[row_idx, col_start + i])
            centers_2d = np.asarray(r["S_v_2d_k_centers"])
            grid = grids[i]
            kmax = float(centers_2d[-1])
            extent = (-kmax, kmax, -kmax, kmax)
            if sk_2d_log:
                if shared_norm is None:
                    pos = grid[grid > 0]
                    lvmin = max(float(pos.min()), 1e-3) if pos.size else 1e-3
                    lvmax = float(grid.max()) if grid.max() > 0 else 1.0
                    this_norm = LogNorm(vmin=lvmin, vmax=lvmax)
                else:
                    this_norm = shared_norm
                im = ax.imshow(
                    np.where(grid > 0, grid, np.nan),
                    origin="lower", extent=extent, cmap="inferno", norm=this_norm,
                    interpolation=sk_2d_interp, aspect="equal",
                )
            else:
                im = ax.imshow(
                    grid, origin="lower", extent=extent, cmap="inferno",
                    interpolation=sk_2d_interp, aspect="equal",
                )
            ax.set_xlabel(r"$k_x$  (1/µm)"); ax.set_ylabel(r"$k_y$  (1/µm)")
            ax.set_title(rf"$S_v(k_x, k_y)$ — {r['label']}")
            if not (sk_2d_shared_clim and shared_norm is not None):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$S_v$")
            axes_out.append(ax)
            last_im_local = im
        if sk_2d_shared_clim and shared_norm is not None and last_im_local is not None:
            fig.colorbar(last_im_local, ax=axes_out, fraction=0.025, pad=0.02, label=r"$S_v$")
        return axes_out, last_im_local

    # --- Row 0: amorphous S(k), rings, angles ---
    ax_sk = fig.add_subplot(gs[0, 0])
    ax_rings = fig.add_subplot(gs[0, 1])
    ax_ang = fig.add_subplot(gs[0, 2])
    _plot_sk(ax_sk, amorph or results, "Vertex structure factor (radial)")
    _plot_rings(ax_rings, amorph or results, "Shortest-ring distribution")
    _plot_angles(ax_ang, amorph or results, r"Angle distributions  $\theta$ / $\varphi$ / $\chi$")

    # --- Row 1: 2D slice per amorphous network ---
    ax_2d_list, _ = _plot_2d_row(1, 0, amorph or results, "amorphous")

    out = {"sk": ax_sk, "rings": ax_rings, "angles": ax_ang, "sk2d": ax_2d_list}

    # --- Row 2 (only if crystals present): crystal S(k), rings, angles, 2D ---
    if has_crystal_row:
        ax_c_sk = fig.add_subplot(gs[2, 0])
        ax_c_rings = fig.add_subplot(gs[2, 1])
        ax_c_ang = fig.add_subplot(gs[2, 2])
        _plot_sk(ax_c_sk, crystals, "Crystal: $S_v(k)$")
        _plot_rings(ax_c_rings, crystals, "Crystal: rings")
        _plot_angles(ax_c_ang, crystals, r"Crystal: $\theta$ / $\varphi$ / $\chi$")
        ax_c_2d_list, _ = _plot_2d_row(2, 3, crystals, "crystal")
        out["crystal_sk"] = ax_c_sk
        out["crystal_rings"] = ax_c_rings
        out["crystal_angles"] = ax_c_ang
        out["crystal_sk2d"] = ax_c_2d_list

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, out


# ----------------------------- CLI for quick use --------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse LSU network rod-endpoint file(s).")
    parser.add_argument("paths", nargs="+", help="One or more rod-endpoint files")
    parser.add_argument("--box", type=float, default=11.44)
    parser.add_argument("--d0", type=float, default=0.8)
    parser.add_argument("--compare", action="store_true",
                        help="Print side-by-side comparison (default for >1 file)")
    args = parser.parse_args()

    if len(args.paths) == 1 and not args.compare:
        analyze_network(args.paths[0], box=args.box, d0=args.d0)
    else:
        compare_networks([(p, None) for p in args.paths], box=args.box, d0=args.d0)
