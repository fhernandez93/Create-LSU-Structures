"""Decisive diagnostic: is the reference (8r-rich) LOWER Keating energy than the
disorder plateau (8r 38)? If lower -> my plateau is a KINETIC trap (fixable by
better annealing, consistent with BM-WWW reaching good CRNs from random). If the
reference is HIGHER energy -> my energy doesn't favour it and no anneal reaches it
(the gap is the energy/objective, not kinetics)."""
import os, numpy as np, tools, lsu_network as lsu

box = np.array([11.44] * 3); D0 = 0.8; W = (0.7, 0.7, 0.3, 0.4); N = 1000


def load(path):
    rods = np.loadtxt(path)
    pos, edges = tools.rods_to_network(rods, box)
    return pos, edges


def relaxed_energy(pos, edges, deep=2000):
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
    ctx.update_topology(edges, nb)
    p, _, _ = lsu.relax(pos, ctx, max_iter=deep)
    E = float(ctx.energy(p.ravel()))
    tri = lsu.build_angle_triples(nb); quad = lsu.build_dihedral_quads(edges, nb)
    f1, f2, f3, f4 = lsu.energy_components(p, edges, tri, quad, box, D0)
    a, b, g, d = W
    return E, (a * float(f1), b * float(f2), g * float(f3), d * float(f4))


items = [('REFERENCE', 'Example/lsu_example_ends.txt'),
         ('refHold_ck50k', 'Structures/20260622_refHold_ck50k.txt'),
         ('coldDis_ck50k', 'Structures/20260622_coldDis_ck50k.txt'),
         ('hyperuniform', 'Example/20260622_lsu_hyperuniform_N1000_ends.txt')]

print(f"KEATING={lsu._KEATING_F1F2}  weights={W}  (deep-relaxed 2000 iters)")
print("weighted total Keating energy (lower = more favoured by the energy):")
for name, path in items:
    if not os.path.exists(path):
        print(f"  {name:16s}: MISSING"); continue
    pos, edges = load(path)
    E, comps = relaxed_energy(pos, edges)
    print(f"  {name:16s}: E_total={E:9.3f}  per_atom={E/N:.5f}  "
          f"f1={comps[0]:.3f} f2={comps[1]:.3f} f3={comps[2]:.3f} f4={comps[3]:.3f}")
