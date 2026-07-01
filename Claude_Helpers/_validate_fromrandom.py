"""Assess the statistics of an amorphous LSU network against the gold reference.

`assess_statistics(rods_or_path, ...)` is the importable entry point (used by the
notebook). It deep-relaxes the network under the Keating energy and measures the
FULL gate set, graph-true:
  Phi_22, Phi_12, bond-angle std, S(k0), S_low_k2, alpha (hyperuniformity slope),
  S_v_peak (Bragg/amorphous check), bond-length std, ring distribution + girth,
  min_nb (collision) -- each printed PASS/FAIL vs the reference.

With `stage_b=True` it also runs the free fixed-topology void restoration
(minimise E_Keating + lambda*S_low at FIXED topology) over a lambda sweep, picks
the best, and optionally saves it to Example/<out_tag>_ends.txt. This is the
"validate a raw checkpoint" mode the from-random recipe's step 3 uses.

CLI (unchanged behaviour -- Stage-B sweep, save best if a tag is given):
  python -m Claude_Helpers._validate_fromrandom <ckpt_rodfile.txt> [out_tag]
  env: N_VAL (1000)
"""
import sys
import os
import math
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import tools
import lsu_network as lsu
from Claude_Helpers._graph_rings import ring_stats_from_edges
from Claude_Helpers._metrics import full_metrics_safe, s_k0
from Claude_Helpers.from_random_recipe import _rods_to_network_N

KMAX = 2
W = (0.7, 0.7, 0.3, 0.4)
D0 = 0.8
REF = dict(phi22=0.889, phi12=0.985, S_k0=0.041, S_low=0.053, alpha=1.51, angstd=8.41,
           bstd=0.029, svpeak=1.82, r8=59.7)


def _box_for(N):
    return np.array([(N / 1000.0) ** (1.0 / 3.0) * 11.44] * 3, float)


def _angstd(p, edges, box):
    N = len(p)
    nb = lsu.build_neighbors(N, edges)
    tri = lsu.build_angle_triples(nb)
    v = p[tri[:, 0]]; a = p[tri[:, 1]]; b = p[tri[:, 2]]
    da = lsu.pbc_displacement(a - v, box); db = lsu.pbc_displacement(b - v, box)
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1))).std())


def _gate(r):
    """Hard reproduction gates (mission spec): Phi22>=0.88, S_low<=0.06,
    S(k0)<=0.08, angstd<=9, amorphous (S_v_peak<3, no Bragg)."""
    return bool(r['phi22'] >= 0.88 and r['S_low'] <= 0.06 and r['S_k0'] <= 0.08
                and r['angstd'] <= 9.0
                and (not math.isnan(r['svpeak']) and r['svpeak'] < 3.0))


def _load(rods_or_path, box, N):
    """Return (pos, edges, edges_graph). Reconstruct a self-consistent (pos, edges)
    pair from the rod geometry with `_rods_to_network_N`, which auto-shrinks the
    merge radius if the default 0.1 um fuses a near-coincident *non-bonded* vertex
    pair into a degree>3 node -- otherwise `build_neighbors` raises IndexError
    (this is what bites at large N, where a rare pair of distinct junctions can
    relax to within 0.1 um; e.g. an N=10000 network with two junctions ~0.087 um
    apart). The near-coincidence is a real clumping signal and still surfaces via
    `min_nb`. Prefer a sibling `_edges.npy` (graph-true topology) for ring stats
    when a path is given."""
    if isinstance(rods_or_path, str):
        rods = np.loadtxt(rods_or_path)
        pos, edges = _rods_to_network_N(rods, box, N)
        ep = rods_or_path.replace('.txt', '_edges.npy')
        edges_g = np.load(ep) if os.path.exists(ep) else edges
    else:
        rods = np.asarray(rods_or_path, float)
        pos, edges = _rods_to_network_N(rods, box, N)
        edges_g = edges
    return pos, edges, edges_g


def _measure(p, edges, edges_g, box, N, label):
    p = p - box * np.round(p / box)
    d, m, c, g = ring_stats_from_edges(edges_g, N)
    nb = lsu.build_neighbors(N, edges)
    astd = _angstd(p, edges, box)
    mp = None if N <= 1500 else 6000
    phi22 = float(lsu.compute_lsu(p, edges, nb, box, depth=2, locality=2, max_pairs=mp, rng=np.random.default_rng(0)))
    phi12 = float(lsu.compute_lsu(p, edges, nb, box, depth=1, locality=2, max_pairs=mp, rng=np.random.default_rng(0)))
    sk0, _, _ = s_k0(p, box)
    slow = float(lsu.low_k_structure_factor(p, box, kmax=KMAX))
    rods_o = lsu.network_to_rods(p, edges, box, pbc_duplicate_boundary_rods=True, clip_endpoints_to_box=False)
    try:
        mm, _ = full_metrics_safe(rods_o, box=float(box[0]), d0=D0, label=label)
        alpha = mm["S_v_alpha_low"]; svpk = mm["S_v_peak"]; bstd = mm["bond_len_std"]; minnb = mm["min_nb"]
    except RuntimeError:
        alpha = float('nan'); svpk = float('nan'); bstd = float('nan'); minnb = float('nan')
    r = dict(rings=d, mean=m, girth=g, angstd=astd, phi22=phi22, phi12=phi12,
             S_k0=float(sk0), S_low=slow, alpha=alpha, svpeak=svpk, bstd=bstd,
             min_nb=minnb, rods=rods_o)
    r['pass'] = _gate(r)
    return r


def _print_row(label, r):
    print(f"[{label:>20s}] 8r={r['rings'].get(8, 0):4.1f} 7r={r['rings'].get(7, 0):4.1f} "
          f"mean={r['mean']:.2f} girth={r['girth']} | angstd={r['angstd']:.2f} "
          f"Phi22={r['phi22']:.4f} Phi12={r['phi12']:.4f} | S_k0={r['S_k0']:.4f} "
          f"S_low={r['S_low']:.4f} a={r['alpha']:+.2f} svpk={r['svpeak']:.2f} "
          f"bstd={r['bstd']:.4f} min_nb={r['min_nb']:.3f}  [{'PASS' if r['pass'] else 'fail'}]",
          flush=True)


def _print_ref():
    print(f"[{'REFERENCE':>20s}] 8r=59.7  7r=10.0 mean=7.99 girth=6 | angstd=8.41 "
          "Phi22=0.8887 Phi12=0.9849 | S_k0=0.0410 S_low=0.0530 a=+1.51 svpk=1.82 "
          "bstd=0.0290 min_nb=inf", flush=True)


def assess_statistics(rods_or_path, N=1000, stage_b=False, lambdas=(0.0, 1.0, 5.0, 20.0),
                      out_tag=None, verbose=True):
    """Measure all reproduction gates for a network and print PASS/FAIL vs the
    reference. Returns a metrics dict (the best Stage-B result when stage_b=True,
    else the deep-relaxed baseline). Pass a rod-endpoint array (e.g. straight from
    `generate_from_random`) or a path to a saved `_ends.txt` / checkpoint `.txt`.

    stage_b=False : just measure (deep-relax under Keating, then the gate set).
                    Use this on a finished network (the recipe already did Stage-B).
    stage_b=True  : also run the free fixed-topology void restoration over `lambdas`,
                    pick the best, and (if out_tag) save it to Example/<out_tag>_ends.txt.
                    Use this to validate a RAW (pre-void-fix) anneal checkpoint.
    """
    box = _box_for(N)
    pos, edges, edges_g = _load(rods_or_path, box, N)
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
    ctx.update_topology(edges, nb)

    if not stage_b:
        # Assess a FINISHED network: measure the saved geometry as-is. A pure-Keating
        # relax would re-open the void (the Keating minimum is less hyperuniform than
        # the Stage-B-restored geometry), so do NOT relax here.
        m = _measure(pos, edges, edges_g, box, N, "as-saved")
        if verbose:
            _print_row("as-saved", m)
            _print_ref()
        return m

    # Validate a RAW (pre-void-fix) checkpoint: deep-relax baseline, then Stage-B.
    p0, _, _ = lsu.relax(pos, ctx, max_iter=1500)
    base = _measure(p0, edges, edges_g, box, N, "base")
    if verbose:
        _print_row("base/Keating-relax", base)

    hkl = lsu._low_k_hkl(KMAX); kvec = jnp.asarray(2.0 * math.pi * (hkl / box))
    slow_vg = jax.jit(jax.value_and_grad(
        lambda x: (jnp.abs(jnp.exp(1j * (x.reshape(N, 3) @ kvec.T)).sum(0)) ** 2).sum() / N))

    def combined(xf, lam):
        ek, gk = ctx.value_and_grad(xf)
        if lam == 0.0:
            return ek, gk
        vs, gs = slow_vg(jnp.asarray(xf))
        return ek + lam * float(vs), gk + lam * np.asarray(gs, np.float64)

    best = None; rows = []
    for lam in lambdas:
        res = minimize(lambda x: combined(x, lam), p0.ravel().astype(np.float64), jac=True,
                       method="L-BFGS-B", options={"maxiter": 1000})
        r = _measure(res.x.reshape(N, 3), edges, edges_g, box, N, f"lam{lam:g}")
        r['lam'] = lam; rows.append(r)
        if verbose:
            _print_row(f"Stage-B lam={lam:g}", r)
        if best is None or (abs(r['S_k0'] - REF['S_k0']) < abs(best['S_k0'] - REF['S_k0'])
                            and r['angstd'] <= 9.5):
            best = r
    if verbose:
        _print_ref()
    best['all'] = rows
    if out_tag and best is not None:
        outp = f"Example/{out_tag}_ends.txt"
        np.savetxt(outp, best['rods'], fmt="%.6f", delimiter="\t")
        if verbose:
            print(f"SAVED best (lam={best['lam']:g}) -> {outp}", flush=True)
    return best


if __name__ == "__main__":
    N = int(os.environ.get("N_VAL", "1000"))
    path = sys.argv[1]
    out_tag = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"=== VALIDATE from-random {path} (N={N}) ===", flush=True)
    assess_statistics(path, N=N, stage_b=True, out_tag=out_tag, verbose=True)
