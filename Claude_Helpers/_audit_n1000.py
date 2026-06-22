"""Independent stats/metrics audit for the N=1000 investigation.

Reloads every network FROM its saved rod file and recomputes all metrics via
full_metrics_safe (box=11.44, d0=0.8). Adds:
  - explicit degree histogram from the reconstructed edge list (degree-3 check)
  - per-mode S(k) for the 6 lowest |k| modes, from the SAME reconstructed
    positions / merge radius the harness used (cross-checks S_k0 = mean of 6)
  - identifies which Cartesian direction carries the anomalous k0 pair
  - raw-mode Bragg check (max over individual k_modes_max=4 modes, not binned)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from Claude_Helpers._metrics import full_metrics_safe

BOX = 11.44
D0 = 0.8

FILES = [
    ("REFERENCE", "Example/lsu_example_ends.txt"),
    ("baseline_ak1000", "Example/20260619_ak1000_lsu_generated.txt"),
    ("A1", "Structures/20260619_A1_w30_30k_s42.txt"),
    ("S1_s42", "Structures/20260619_S1_sustain_w35_50k_s42.txt"),
    ("S1_s7", "Structures/20260619_S1_sustain_w35_50k_s7.txt"),
    ("probe_w0_100k", "Structures/20260619_probew0_s42_ck100k.txt"),
]


def reconstruct(path, cluster_radius):
    """Rebuild positions/edges exactly as the harness does, given the merge radius."""
    import functools
    rods = np.loadtxt(path)
    box_arr = np.asarray([BOX, BOX, BOX], float)
    orig = tools.rods_to_network
    try:
        if cluster_radius is not None and cluster_radius != 0.1:
            tools.rods_to_network = functools.partial(orig, cluster_radius=cluster_radius)
        positions, edges = tools.rods_to_network(rods, box_arr)
    finally:
        tools.rods_to_network = orig
    return positions, edges, box_arr


def six_lowest_modes(positions, box_arr):
    """Return list of (h,k,l, |k|, S) for the 6 lowest |k| modes (the k0 shell)."""
    L = float(box_arr[0])
    k0 = 2.0 * np.pi / L
    rows = []
    for h in range(-1, 2):
        for k in range(-1, 2):
            for l in range(-1, 2):
                if h == 0 and k == 0 and l == 0:
                    continue
                kvec = np.array([h, k, l], float) * k0
                kmag = np.linalg.norm(kvec)
                phases = positions @ kvec
                re = np.cos(phases).sum()
                im = np.sin(phases).sum()
                S = (re * re + im * im) / len(positions)
                rows.append((h, k, l, kmag, S))
    # k0 shell = the 6 with smallest |k| (the +-1,0,0 perms)
    rows.sort(key=lambda r: r[3])
    return rows[:6]


def raw_mode_max(positions, box_arr, kmax=4):
    kmag, S = tools._vertex_structure_factor(positions, box_arr, k_modes_max=kmax)
    i = int(np.argmax(S))
    return float(S[i]), float(kmag[i])


def main():
    out = {}
    for label, rel in FILES:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)
        m, cr = full_metrics_safe(path, box=BOX, d0=D0, label=label)
        positions, edges, box_arr = reconstruct(path, cr)
        N = len(positions)
        E = len(edges)
        deg = np.bincount(edges.ravel(), minlength=N)
        deghist = {int(d): int((deg == d).sum()) for d in np.unique(deg)}
        modes = six_lowest_modes(positions, box_arr)
        sk0_recomputed = float(np.mean([r[4] for r in modes]))
        rawmax, rawmax_k = raw_mode_max(positions, box_arr, kmax=4)
        density = N / (BOX ** 3)

        rec = {
            "file": rel,
            "cluster_radius_used": cr,
            "N": N, "E": E, "rods_in_file": int(m["rods_in_file"]),
            "density": density,
            "degree_hist": deghist,
            "phi12": m["phi12"], "phi22": m["phi22"],
            "bond_mean": m["bond_len_mean"], "bond_std": m["bond_len_std"],
            "bond_ang_std": m["bond_ang_std"],
            "ring_dist": {int(k): int(v) for k, v in m["ring_distribution"].items()},
            "ring_mean": m["ring_mean"],
            "ring_5_frac": m["ring_5_fraction"],
            "ring_6_frac": m["ring_6_fraction"],
            "ring_7_frac": m["ring_7_fraction"],
            "ring_8_frac": m["ring_8_fraction"],
            "ring_9_frac": m["ring_9_fraction"],
            "S_k0_harness": m["S_k0"],
            "S_k0_kmin": m["S_k0_kmin"],
            "S_k0_nmodes": m["S_k0_nmodes"],
            "S_k0_recomputed_mean6": sk0_recomputed,
            "six_modes": [[int(r[0]), int(r[1]), int(r[2]), float(r[3]), float(r[4])] for r in modes],
            "S_low_k2": m["S_low_k2"],
            "S_v_alpha_low": m["S_v_alpha_low"],
            "S_v_peak_binned": m["S_v_peak"],
            "raw_mode_max": rawmax, "raw_mode_max_k": rawmax_k,
            "dihedral_entropy": m["dihedral_entropy"],
            "min_nb": m["min_nb"],
            "low_shells": [[float(k), float(s), int(n)] for k, s, n in m["low_shells"]],
        }
        out[label] = rec
        print(f"=== {label} ({rel}) cr={cr} ===")
        print(f"  N={N} E={E} rods={rec['rods_in_file']} density={density:.4f} deg={deghist}")
        print(f"  Phi12={m['phi12']:.4f} Phi22={m['phi22']:.4f} bond {m['bond_len_mean']:.4f}/{m['bond_len_std']:.4f} angstd={m['bond_ang_std']:.2f}")
        rs = "  ".join(f"{k}:{100*v/E:.1f}%" for k, v in sorted(rec['ring_dist'].items()) if k>0)
        print(f"  rings {rs}  mean={m['ring_mean']:.3f}")
        print(f"  6r={100*m['ring_6_fraction']:.1f}% 8r={100*m['ring_8_fraction']:.1f}%")
        print(f"  S_k0(harness)={m['S_k0']:.4f}  mean6(recompute)={sk0_recomputed:.4f}  kmin={m['S_k0_kmin']:.3f} nmodes={m['S_k0_nmodes']}")
        print(f"  6 modes (h,k,l|k||S):")
        for r in modes:
            print(f"     ({r[0]:+d},{r[1]:+d},{r[2]:+d}) |k|={r[3]:.3f}  S={r[4]:.4f}")
        print(f"  S_low_k2={m['S_low_k2']:.4f}  alpha={m['S_v_alpha_low']:.3f}  S_v_peak_binned={m['S_v_peak']:.3f}  raw_mode_max={rawmax:.3f}@k={rawmax_k:.3f}")
        print(f"  dih={m['dihedral_entropy']:.3f}  min_nb={m['min_nb']:.4f}")
        print()

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "claude_plans", "_audit_n1000_raw.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE claude_plans/_audit_n1000_raw.json")


if __name__ == "__main__":
    main()
