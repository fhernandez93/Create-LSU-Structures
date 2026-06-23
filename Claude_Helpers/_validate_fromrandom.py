"""Final validation of a from-random mq2 checkpoint, with optional Stage-B void
restoration. Loads the checkpoint, deep-relaxes under Keating, measures ALL gates
graph-true; then applies Stage-B (minimise Keating + lambda*S_low at FIXED topology)
to drive S_k0 -> reference at zero ring/angle cost (Finding 1), re-measures, and
reports PASS/FAIL vs the reference. Saves the best Stage-B structure.

Usage: python -m Claude_Helpers._validate_fromrandom <ckpt_rodfile.txt> [out_tag]
"""
import sys, os, math
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
import tools, lsu_network as lsu
from Claude_Helpers._graph_rings import ring_stats_from_edges
from Claude_Helpers._metrics import full_metrics_safe, s_k0

BOX = 11.44; D0 = 0.8; box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4); N = 1000
KMAX = 2
PATH = sys.argv[1]
OUT_TAG = sys.argv[2] if len(sys.argv) > 2 else None
LAMBDAS = [0.0, 1.0, 5.0, 20.0]

REF = dict(phi22=0.889, phi12=0.985, S_k0=0.041, S_low=0.053, alpha=1.51, angstd=8.41,
           bstd=0.029, svpeak=1.82, r8=59.7)

rods = np.loadtxt(PATH)
pos, edges = tools.rods_to_network(rods, box)
# prefer saved edges if present (graph-true, collision-proof)
ep = PATH.replace('.txt', '_edges.npy')
edges_g = np.load(ep) if os.path.exists(ep) else edges
nb = lsu.build_neighbors(N, edges)
ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
ctx.update_topology(edges, nb)

hkl = lsu._low_k_hkl(KMAX); kvec = jnp.asarray(2.0*math.pi*(hkl/box))
_slow_vg = jax.jit(jax.value_and_grad(
    lambda x: (jnp.abs(jnp.exp(1j*(x.reshape(N,3) @ kvec.T)).sum(0))**2).sum()/N))


def combined(xf, lam):
    ek, gk = ctx.value_and_grad(xf)
    if lam == 0.0:
        return ek, gk
    vs, gs = _slow_vg(jnp.asarray(xf))
    return ek + lam*float(vs), gk + lam*np.asarray(gs, np.float64)


def angstd(p, e):
    nbb = lsu.build_neighbors(N, e); tri = lsu.build_angle_triples(nbb)
    v=p[tri[:,0]];a=p[tri[:,1]];b=p[tri[:,2]]
    da=lsu.pbc_displacement(a-v,box); db=lsu.pbc_displacement(b-v,box)
    da/=np.linalg.norm(da,axis=1,keepdims=True); db/=np.linalg.norm(db,axis=1,keepdims=True)
    return float(np.degrees(np.arccos(np.clip((da*db).sum(1),-1,1))).std())


def measure(p, tag):
    p = p - box*np.round(p/box)
    d,m,c,g = ring_stats_from_edges(edges_g, N)
    astd = angstd(p, edges)
    phi22 = float(lsu.compute_lsu(p, edges, nb, box, depth=2, locality=2))
    phi12 = float(lsu.compute_lsu(p, edges, nb, box, depth=1, locality=2))
    sk0,_,_ = s_k0(p, box); slow = float(lsu.low_k_structure_factor(p, box, kmax=KMAX))
    rods_o = lsu.network_to_rods(p, edges, box, pbc_duplicate_boundary_rods=True, clip_endpoints_to_box=False)
    out = f"Structures/_valfr_{tag}.txt"; np.savetxt(out, rods_o, fmt="%.6f", delimiter="\t")
    try:
        mm,_ = full_metrics_safe(out, box=BOX, d0=D0, label=tag)
        alpha=mm["S_v_alpha_low"]; svpk=mm["S_v_peak"]; bstd=mm["bond_len_std"]; minnb=mm["min_nb"]
    except RuntimeError:
        alpha=float('nan'); svpk=float('nan'); bstd=float('nan'); minnb=float('nan')
    return dict(rings=d, mean=m, girth=g, angstd=astd, phi22=phi22, phi12=phi12,
                S_k0=float(sk0), S_low=slow, alpha=alpha, svpeak=svpk, bstd=bstd, min_nb=minnb), p


print(f"=== VALIDATE from-random {PATH} ===", flush=True)
# baseline: deep relax under Keating (lambda=0)
p0,_,_ = lsu.relax(pos, ctx, max_iter=1500)
base,_ = measure(p0, "base")
print(f"[base/Keating-relax] 8r={base['rings'].get(8,0):.1f} 7r={base['rings'].get(7,0):.1f} "
      f"mean={base['mean']:.2f} girth={base['girth']} | angstd={base['angstd']:.2f} "
      f"Phi22={base['phi22']:.4f} Phi12={base['phi12']:.4f} | S_k0={base['S_k0']:.3f} "
      f"S_low={base['S_low']:.3f} a={base['alpha']:+.2f} svpk={base['svpeak']:.2f} "
      f"bstd={base['bstd']:.4f} min_nb={base['min_nb']:.3f}", flush=True)

best = None
for lam in LAMBDAS:
    res = minimize(lambda x: combined(x, lam), p0.ravel().astype(np.float64), jac=True,
                   method="L-BFGS-B", options={"maxiter": 1000})
    r, p = measure(res.x.reshape(N,3), f"lam{lam:g}")
    # gate check (relaxed-amorphous reproduction)
    g = (r['phi22']>=0.88 and r['S_low']<=0.06 and r['S_k0']<=0.08 and r['angstd']<=9.0
         and (not math.isnan(r['svpeak']) and r['svpeak']<3.0))
    print(f"[Stage-B lam={lam:5.1f}] 8r={r['rings'].get(8,0):.1f} mean={r['mean']:.2f} | "
          f"angstd={r['angstd']:.2f} Phi22={r['phi22']:.4f} | S_k0={r['S_k0']:.4f} "
          f"S_low={r['S_low']:.4f} a={r['alpha']:+.2f} svpk={r['svpeak']:.2f} "
          f"bstd={r['bstd']:.4f} [{'PASS' if g else '.'}]", flush=True)
    if best is None or (abs(r['S_k0']-REF['S_k0']) < abs(best[1]['S_k0']-REF['S_k0']) and r['angstd']<=9.5):
        best = (lam, r, p)

print(f"\nREF: 8r 59.7 mean 7.99 | angstd 8.41 Phi22 0.889 | S_k0 0.041 S_low 0.053 a +1.51 svpk 1.82 bstd 0.029", flush=True)
if OUT_TAG and best is not None:
    lam, r, p = best
    rods_o = lsu.network_to_rods(p - box*np.round(p/box), edges, box,
                                 pbc_duplicate_boundary_rods=True, clip_endpoints_to_box=False)
    outp = f"Example/{OUT_TAG}_ends.txt"; np.savetxt(outp, rods_o, fmt="%.6f", delimiter="\t")
    print(f"SAVED best (lam={lam}) -> {outp}", flush=True)
