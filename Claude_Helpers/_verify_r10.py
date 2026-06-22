"""Independent verification of r10 candidate metrics. Does NOT trust the table."""
import numpy as np
import tools

CAND = "Structures/20260618_r10_coldT_w30_40k.txt"
REF = "Example/lsu_example_ends.txt"
BOX_C = 9.152
BOX_R = 11.44
D0 = 0.8


def hand_sk(positions, box, hkls):
    """Direct |sum exp(i k.r)|^2 / N for explicit integer (h,k,l) triples."""
    L = float(box)
    k0 = 2 * np.pi / L
    N = len(positions)
    out = []
    for hkl in hkls:
        kv = np.array(hkl, float) * k0
        ph = positions @ kv
        re = np.cos(ph).sum()
        im = np.sin(ph).sum()
        out.append((re * re + im * im) / N)
    return np.array(out)


def degree_check(N, edges):
    deg = np.zeros(N, int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    return deg


def analyze(path, box, label):
    print(f"\n===== {label}  ({path}) =====")
    rods = np.loadtxt(path)
    box_arr = np.array([box, box, box], float)
    L = float(box)
    positions, edges = tools.rods_to_network(rods, box_arr)
    N = len(positions)
    E = len(edges)
    deg = degree_check(N, edges)
    print(f"rods_in_file={len(rods)}  N(vertices)={N}  E(edges)={E}  3N/2={3*N/2}")
    print(f"degree: min={deg.min()} max={deg.max()} mean={deg.mean():.4f}  "
          f"frac_deg3={np.mean(deg==3):.4f}  n_deg!=3={np.sum(deg!=3)}")
    density = N / L**3
    print(f"box={L}  density N/L^3 = {density:.6f}")

    # coincident-vertex check (PBC-aware min pairwise distance)
    from scipy.spatial import cKDTree
    wp = ((positions + L/2) % L)
    tree = cKDTree(wp, boxsize=L)
    d, _ = tree.query(wp, k=2)
    mind = d[:, 1].min()
    print(f"min pairwise vertex dist (PBC) = {mind:.4f}  (cluster_radius=0.1; "
          f"{'OK' if mind > 0.1 else 'COINCIDENT!'})")

    # bond lengths
    bl = tools._bond_lengths(positions, edges, L)
    print(f"bond: mean={bl.mean():.4f} std={bl.std():.4f} min={bl.min():.4f} "
          f"max={bl.max():.4f}  (mean/d0={bl.mean()/D0:.3f})")

    # phi
    nbrs = tools.build_neighbors(N, edges)
    phi12 = tools.compute_lsu(positions, edges, nbrs, box_arr, depth=1, locality=2)
    phi22 = tools.compute_lsu(positions, edges, nbrs, box_arr, depth=2, locality=2)
    print(f"Phi_12={phi12:.4f}  Phi_22={phi22:.4f}")

    # rings
    rings = tools._shortest_ring_per_edge(N, edges)
    vals, cnts = np.unique(rings, return_counts=True)
    rd = {int(v): int(c) for v, c in zip(vals, cnts)}
    ring_mean = rings[rings > 0].mean()
    print(f"ring_dist={rd}")
    print(f"ring_mean={ring_mean:.4f}  6-ring={100*rd.get(6,0)/E:.2f}%  "
          f"(count6={rd.get(6,0)}/E={E})")

    # ---- S(k0): hand-computed 6 lowest modes ----
    lowest = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    sk_hand = hand_sk(positions, L, lowest)
    # cross-check against tools._vertex_structure_factor
    kmag, S = tools._vertex_structure_factor(positions, box_arr, k_modes_max=4)
    kmin = kmag.min()
    shell = np.isclose(kmag, kmin, rtol=1e-6)
    sk_tool = S[shell]
    print(f"\nS(k0) lowest shell |k|={kmin:.4f} (=2pi/L={2*np.pi/L:.4f}), "
          f"n_modes={shell.sum()}")
    print(f"  hand modes (100-type): {np.array2string(sk_hand, precision=4)}")
    print(f"  hand mean={sk_hand.mean():.4f} std={sk_hand.std():.4f} max={sk_hand.max():.4f}")
    print(f"  tool shell: {np.array2string(np.sort(sk_tool), precision=4)}")
    print(f"  tool mean={sk_tool.mean():.4f}  "
          f"(hand-vs-tool match: {np.allclose(np.sort(sk_hand), np.sort(sk_tool))})")

    # ---- Bragg / amorphous: RAW unbinned S over wide k range ----
    kmagW, SW = tools._vertex_structure_factor(positions, box_arr, k_modes_max=8)
    imax = np.argmax(SW)
    k0u = 2*np.pi/L
    # recover the (h,k,l) of the global max
    print(f"\nRAW S_v(k) over hkl in [-8,8]^3 (excl 0):")
    print(f"  global max S={SW[imax]:.3f} at |k|={kmagW[imax]:.3f} "
          f"(|k|/k0={kmagW[imax]/k0u:.2f})   [N={N}, crystal would be O(N)]")
    # distribution of high peaks
    order = np.argsort(SW)[::-1][:8]
    print(f"  top-8 S values: {np.array2string(SW[order], precision=2)}")
    print(f"  amorphous if max << N ({N}); Bragg if max ~ O(N)")

    return dict(N=N, E=E, phi12=phi12, phi22=phi22, bond_std=bl.std(),
                ring_mean=ring_mean, ring6=100*rd.get(6,0)/E,
                sk0=sk_hand.mean(), smax=SW[imax])


if __name__ == "__main__":
    analyze(CAND, BOX_C, "CANDIDATE r10")
    analyze(REF, BOX_R, "REFERENCE")
    print(f"\nDensity equality check: 512/9.152^3={512/9.152**3:.6f}  "
          f"1000/11.44^3={1000/11.44**3:.6f}")
