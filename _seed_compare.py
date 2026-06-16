"""Does the random seed start hyperuniform? (Decides the seed recommendation.)

Computed directly on the real (positions, edges) graph — no lossy rod
round-trip. Crystal srs is the hyperuniform Z=3 parent of the amorphous
gyroid: S(k->0)=0 (but topologically ordered: 1 ring size, dihedral entropy
low). A Poisson/random seed (random_bm2000) is topologically disordered but
has Poisson long-wavelength density fluctuations: S(k->0)~const, alpha~0.
Reference is BOTH amorphous AND hyperuniform.
"""
import numpy as np
import lsu_network as lsu
import tools

BOX, D0, N = 9.152, 0.8, 512
box = np.array([BOX, BOX, BOX])

def stats(positions, edges, label):
    neighbors = lsu.build_neighbors(len(positions), edges)
    diag = lsu.cluster_diagnostics(positions, edges, neighbors, box, D0)
    rings = tools._shortest_ring_per_edge(len(positions), edges)
    rv, rc = np.unique(rings[rings > 0], return_counts=True)
    ring_mean = float(rings[rings > 0].mean())
    _, h_dih = tools._dihedral_entropy(positions, edges, BOX)
    kmag, S = tools._vertex_structure_factor(positions, box, k_modes_max=8)
    kc, sm, _ = tools._bin_structure_factor(kmag, S, nbins=24)
    alpha, nlow = tools._fit_hyperuniformity(kc, sm, kmax_fit=2.0)
    sk_peak = float(np.nanmax(sm[np.isfinite(sm)]))
    n_ring_sizes = len(rv)
    print(f"{label:<16} S_low_k2={diag['S_low_k2']:.4f}  alpha(k<2)={alpha:+.3f}  "
          f"voxel_std4={diag['voxel_std_4']:.3f}  S_v_peak={sk_peak:.2f}  "
          f"dih_ent={h_dih:.3f}  ring_mean={ring_mean:.2f}  #ring_sizes={n_ring_sizes}")
    rd = {int(v): int(c) for v, c in zip(rv, rc)}
    print(f"{'':16} rings: {rd}")

print(f"N={N} box={BOX}  (matched to reference density / srs bond=0.81)\n")
print(f"{'seed':<16} hyperuniformity (S_low,alpha,voxel) + crystallinity (dih_ent,rings)")
print("-" * 100)

pos_c, edg_c, _ = lsu.crystal_seed_network(N, BOX, D0, np.random.default_rng(0),
                                            lattice="srs", jitter_sigma=0.10)
stats(pos_c, edg_c, "crystal_srs")

try:
    pos_r, edg_r, _ = lsu.random_seed_network_bm2000(
        N, BOX, D0, np.random.default_rng(0), verbose=False)
    stats(pos_r, edg_r, "random_bm2000")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"random_bm2000 FAILED: {type(e).__name__}: {e}")

print("\nreference target: S_low_k2=0.053  alpha=+1.51  dih_ent=0.80  ring_mean=7.99  (peaked at 8-rings)")
