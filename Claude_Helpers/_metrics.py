"""Shared metrics for the N=512 random_bm2000 validation ablation.

Computes the full reference-comparison metric set, including S(k0) = the
single lowest-mode structure factor (mean over the first |k|-shell), which is
the robust void/hyperuniformity gate. Dimensionless, so valid to compare across
boxes (ref box 11.44, candidate box 9.152).
"""
import numpy as np
import tools
import lsu_network as lsu


def s_k0(positions, box):
    """Mean S_v(k) over the lowest non-zero |k| shell (the 6 (+-1,0,0) modes)."""
    kmag, S = tools._vertex_structure_factor(positions, np.asarray(box, float),
                                             k_modes_max=4)
    kmin = kmag.min()
    shell = np.isclose(kmag, kmin, rtol=1e-6)
    return float(S[shell].mean()), float(kmin), int(shell.sum())


def s_k_lowshells(positions, box, n_shells=4):
    """Per-shell mean S_v(k) for the lowest n_shells distinct |k| values."""
    kmag, S = tools._vertex_structure_factor(positions, np.asarray(box, float),
                                             k_modes_max=4)
    uk = np.unique(np.round(kmag, 6))
    out = []
    for k in uk[:n_shells]:
        shell = np.isclose(kmag, k, rtol=1e-6)
        out.append((float(k), float(S[shell].mean()), int(shell.sum())))
    return out


def full_metrics(rods_or_path, box, d0=0.8, label="net", cluster_radius=None):
    """Full metric dict via tools.analyze_network plus S(k0) and ring fractions.

    cluster_radius: if set, temporarily overrides rods_to_network's merge radius
    (default 0.1). Use a value < the smallest genuine non-bonded separation (e.g.
    0.04) when a network has a near-coincident vertex pair (min_nb < 0.1) that the
    default radius would wrongly fuse into a degree-4 vertex (a real clumping
    signal — see prompt). PBC-image duplicates are at PBC-distance ~0 so any
    radius > ~1e-6 still merges them correctly.
    """
    import functools
    _orig_r2n = tools.rods_to_network
    if cluster_radius is not None:
        tools.rods_to_network = functools.partial(_orig_r2n, cluster_radius=cluster_radius)
    try:
        r = tools.analyze_network(rods_or_path, box=box, d0=d0, label=label, verbose=False)
        # rebuild positions to compute S(k0) directly
        if isinstance(rods_or_path, str):
            rods = np.loadtxt(rods_or_path)
        else:
            rods = np.asarray(rods_or_path, float)
        box_arr = np.asarray([box, box, box], float) if not hasattr(box, "__len__") else np.asarray(box, float)
        positions, edges = tools.rods_to_network(rods, box_arr)
    finally:
        tools.rods_to_network = _orig_r2n
    sk0, kmin, nmodes = s_k0(positions, box_arr)
    shells = s_k_lowshells(positions, box_arr)

    # min non-bonded vertex separation (PBC) — collision/clumping signal
    from scipy.spatial import cKDTree
    L = float(box_arr[0])
    wp = (positions + L / 2) % L
    tree = cKDTree(wp, boxsize=L)
    dists, idxs = tree.query(wp, k=2)
    nn_d, nn_j = dists[:, 1], idxs[:, 1]
    bonded = set(map(tuple, (np.sort(edges, axis=1)).tolist()))
    min_nb = np.inf
    for a in range(len(positions)):
        b = int(nn_j[a])
        if tuple(sorted((a, b))) not in bonded:
            min_nb = min(min_nb, float(nn_d[a]))
    r["min_nb"] = float(min_nb)
    r["cluster_radius_used"] = float(cluster_radius) if cluster_radius is not None else 0.1

    E = r["E"]
    rd = r["ring_distribution"]
    def frac(n):
        return rd.get(n, 0) / E
    r["S_k0"] = sk0
    r["S_k0_kmin"] = kmin
    r["S_k0_nmodes"] = nmodes
    r["low_shells"] = shells
    r["ring_6_fraction"] = frac(6)
    r["ring_5_fraction"] = frac(5)
    r["ring_7_fraction"] = frac(7)
    r["ring_9_fraction"] = frac(9)
    # amorphous check: peak of binned S_v(k) (Bragg => large peak)
    sm = np.asarray(r["S_v_k_means"], float)
    r["S_v_peak"] = float(np.nanmax(sm[np.isfinite(sm)]))
    return r


def full_metrics_safe(rods_or_path, box, d0=0.8, label="net",
                      radii=(None, 0.04, 0.02, 0.01, 0.005, 0.002)):
    """full_metrics that retries with tighter merge radii on the degree-4
    round-trip error (a near-coincident vertex pair the default 0.1 radius
    wrongly fuses). Returns (metrics_dict, radius_used). Records min_nb so the
    clumping is always visible."""
    last_err = None
    for cr in radii:
        try:
            m = full_metrics(rods_or_path, box, d0=d0, label=label, cluster_radius=cr)
            return m, (cr if cr is not None else 0.1)
        except (IndexError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(f"full_metrics_safe failed at all radii {radii}: {last_err}")


def print_metrics(r, ref=None):
    rd = r["ring_distribution"]
    ring_str = "  ".join(f"{n}={c}({100*c/r['E']:.1f}%)" for n, c in sorted(rd.items()))
    print(f"--- {r['label']} (N={r['N']}, E={r['E']}) ---")
    print(f"  Phi_12={r['phi12']:.4f}  Phi_22={r['phi22']:.4f}")
    print(f"  bond mean={r['bond_len_mean']:.4f}  std={r['bond_len_std']:.4f}")
    print(f"  bond ang mean={r['bond_ang_mean']:.2f}  std={r['bond_ang_std']:.2f}")
    print(f"  rings: {ring_str}")
    print(f"  ring_mean={r['ring_mean']:.3f}  6-ring={100*r['ring_6_fraction']:.1f}%  "
          f"8-ring={100*r['ring_8_fraction']:.1f}%")
    print(f"  S(k0)={r['S_k0']:.4f} (kmin={r['S_k0_kmin']:.3f}, {r['S_k0_nmodes']} modes)  "
          f"S_low_k2={r['S_low_k2']:.4f}")
    print(f"  low shells S(k): " + ", ".join(f"S({k:.2f})={s:.3f}" for k, s, n in r['low_shells']))
    print(f"  S_v_alpha(k<2)={r['S_v_alpha_low']:.3f}  S_v_peak={r['S_v_peak']:.2f}  "
          f"dih_ent={r['dihedral_entropy']:.3f}  voxel_std4={r['voxel_std_4']:.3f}")


if __name__ == "__main__":
    import sys
    ref = full_metrics("Example/lsu_example_ends.txt", box=11.44, d0=0.8, label="REFERENCE")
    print_metrics(ref)
    for path in sys.argv[1:]:
        m = full_metrics(path, box=9.152, d0=0.8, label=path)
        print_metrics(m)
