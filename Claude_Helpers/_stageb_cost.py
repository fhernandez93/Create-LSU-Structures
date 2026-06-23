"""CROSS-REVIEW DATA for the claim: 'Stage B (re-hyperuniformising the GEOMETRY at
fixed topology) can only COST local order (angle-std), never build it.'

Method: take the saved best structure (good S(k), broad angles). At FIXED topology,
minimise  J(pos) = E_Keating(pos) + lambda * S_low_k(pos)  for a sweep of lambda.
 - lambda=0 is a pure Keating relax = the angstd-MINIMISING geometry at this topology
   (Keating f2/f4 drive 120deg/planar). That value is the topology's angstd floor.
 - lambda>0 pulls S_k0 DOWN (toward ref 0.041). If angstd rises (or at best holds) as
   S_k0 falls, then a low-k geometry perturbation cannot improve local order -> Stage B
   can only cost it. The magnitude of the cost is the deliverable.

CPU only:  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python -m Claude_Helpers._stageb_cost [path]
"""
import sys, os, math
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
import tools
import lsu_network as lsu
from Claude_Helpers._metrics import full_metrics_safe

PATH = sys.argv[1] if len(sys.argv) > 1 else "Example/20260622_lsu_hyperuniform_N1000_ends.txt"
BOX = 11.44; D0 = 0.8
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
KMAX = 2                                   # same shells as the uniformity penalty
LAMBDAS = [0.0, 2.0, 10.0, 40.0, 150.0]

print(f"=== STAGE-B COST  path={PATH}  KEATING={lsu._KEATING_F1F2} kmax={KMAX} ===", flush=True)

rods = np.loadtxt(PATH)
positions, edges = tools.rods_to_network(rods, box)
N = len(positions)
neighbors = lsu.build_neighbors(N, edges)
print(f"loaded N={N} E={len(edges)}", flush=True)

ctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx.update_topology(edges, neighbors)

# --- cheap low-k S(k) term (no hard-core; topology+Keating keep vertices apart) ---
hkl = lsu._low_k_hkl(KMAX)
kvec = jnp.asarray(2.0 * math.pi * (hkl / box))


def _slow_vg(xf):
    def e(x):
        p = x.reshape(N, 3)
        rho = jnp.exp(1j * (p @ kvec.T)).sum(axis=0)
        return (jnp.abs(rho) ** 2).sum() / N
    v, g = jax.value_and_grad(e)(jnp.asarray(xf))
    return float(v), np.asarray(g, np.float64)


_slow_vg_j = jax.jit(jax.value_and_grad(
    lambda x: (jnp.abs(jnp.exp(1j * (x.reshape(N, 3) @ kvec.T)).sum(0)) ** 2).sum() / N))


def combined(xf, lam):
    ek, gk = ctx.value_and_grad(xf)
    if lam == 0.0:
        return ek, gk
    vs, gs = _slow_vg_j(jnp.asarray(xf))
    return ek + lam * float(vs), gk + lam * np.asarray(gs, np.float64)


def angle_std(p, edges):
    nb = lsu.build_neighbors(N, edges); tri = lsu.build_angle_triples(nb)
    v = p[tri[:, 0]]; a = p[tri[:, 1]]; b = p[tri[:, 2]]
    da = lsu.pbc_displacement(a - v, box); db = lsu.pbc_displacement(b - v, box)
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1)))
    return float(ang.mean()), float(ang.std())


def evaluate(pos, tag):
    pos = pos - box * np.round(pos / box)
    amean, astd = angle_std(pos, edges)
    rods_o = lsu.network_to_rods(pos, edges, box, pbc_duplicate_boundary_rods=True,
                                 clip_endpoints_to_box=False)
    tmp = f"Structures/_stageb_{tag}.txt"
    np.savetxt(tmp, rods_o, fmt="%.6f", delimiter="\t")
    m, cr = full_metrics_safe(tmp, box=BOX, d0=D0, label=f"stageb_{tag}")
    return dict(angstd=astd, angmean=amean, S_k0=m["S_k0"], S_low=m["S_low_k2"],
                phi22=m["phi22"], alpha=m["S_v_alpha_low"], bstd=m["bond_len_std"],
                svpeak=m["S_v_peak"], min_nb=m["min_nb"])


# baseline: as-loaded (no relax)
b = evaluate(positions.copy(), "loaded")
print(f"[loaded ]                S_k0={b['S_k0']:.4f} S_low={b['S_low']:.4f} "
      f"a={b['alpha']:+.2f} angstd={b['angstd']:.2f} Phi22={b['phi22']:.4f} "
      f"bstd={b['bstd']:.4f} min_nb={b['min_nb']:.3f}", flush=True)
print("REF: S_k0 0.041 angstd 8.41 Phi22 0.889 bstd 0.029", flush=True)
print("-" * 96, flush=True)

x0 = positions.ravel().astype(np.float64)
for lam in LAMBDAS:
    res = minimize(lambda x: combined(x, lam), x0.copy(), jac=True,
                   method="L-BFGS-B", options={"maxiter": 800})
    p = res.x.reshape(N, 3)
    r = evaluate(p, f"lam{lam:g}")
    print(f"[lam={lam:7.1f}] S_k0={r['S_k0']:.4f} S_low={r['S_low']:.4f} "
          f"a={r['alpha']:+.2f} angstd={r['angstd']:.2f} Phi22={r['phi22']:.4f} "
          f"bstd={r['bstd']:.4f} min_nb={r['min_nb']:.3f} svpk={r['svpeak']:.2f}", flush=True)

print("-" * 96, flush=True)
print("READ: lam=0 (pure Keating) = the angstd FLOOR at this topology. If angstd does "
      "not drop below it as S_k0 falls, Stage-B can't build local order.", flush=True)
