"""USER REDIRECT: target ONLY (Phi22 >= 0.88) + (good low S(k)), dropping the angstd/ring gates.

Question: with S(k) HELD low by the uniformity penalty, how high can Phi22 be pushed in an AMORPHOUS
network? Finding 1 showed Phi22 is topology-bound (geometry/S(k) tricks don't move it) and `target_lsu`
is only an early-exit (not an acceptance force), so the only lever is the SCHEDULE. This runs the
hyperuniform seed (low-S(k) start) + Keating + penalty (hold S(k)) on a Phi22-maximising schedule and
reports the peak amorphous Phi22 reached with S_low held. If it tops out < 0.88, the relaxed target hits
the SAME topology wall as the full target (Phi22 and local order are the same quantity).

GRAPH-TRUE measurement (compute_lsu Phi22, s_k0 from positions). Checkpointed.

Usage: python -m Claude_Helpers._run_phi22max <tag> <T0> <TF> [n_total] [chunk] [seed]
env: HU_W (30 penalty), HU_KMAX_SEED (3), HU_KMAX_PEN (2), DEEP_RELAX (600)
"""
import sys, os, json, time, datetime, math
import numpy as np
import tools
import lsu_network as lsu
from Claude_Helpers._hyperuniform_seed import hyperuniform_points
from Claude_Helpers._metrics import full_metrics_safe, s_k0

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "phi22max"
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
TF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
n_total = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 42
W = float(os.environ.get("HU_W", "30"))
KSEED = int(os.environ.get("HU_KMAX_SEED", "3"))
KPEN = int(os.environ.get("HU_KMAX_PEN", "2"))
DEEP = int(os.environ.get("DEEP_RELAX", "600"))

print(f"=== PHI22-MAX (relaxed target: Phi22>=0.88 + low S(k))  tag={tag} seed=hyperuniform "
      f"T {T0}->{TF} n={n_total} w={W} kpen={KPEN} seed={SEED} KEATING={lsu._KEATING_F1F2} ===",
      flush=True)

rng = np.random.default_rng(SEED)
hp = hyperuniform_points(N, box, D0, np.random.default_rng(SEED), kmax=KSEED, maxiter=400, verbose=True)
_orig = lsu._poisson_disk_pbc
lsu._poisson_disk_pbc = lambda n, b, md, r, max_tries=0: hp.copy()
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
lsu._poisson_disk_pbc = _orig
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[p22] hyperuniform seed built+settled: E={len(edges)} "
      f"min_nb={srep['min_nb_after']/D0:.3f}d0", flush=True)

g = np.arange(n_total)
T_full = T0 * np.exp(math.log(TF / T0) * g / max(1, n_total - 1))


def angle_std(p, edges):
    nb = lsu.build_neighbors(N, edges); tri = lsu.build_angle_triples(nb)
    v = p[tri[:, 0]]; a = p[tri[:, 1]]; b = p[tri[:, 2]]
    da = lsu.pbc_displacement(a - v, box); db = lsu.pbc_displacement(b - v, box)
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1)))
    return float(ang.mean()), float(ang.std())


def measure(pos, edges, label):
    nb = lsu.build_neighbors(N, edges)
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, nb)
    p2, _, _ = lsu.relax(pos, fctx, max_iter=DEEP)
    p2 = p2 - box * np.round(p2 / box)
    amean, astd = angle_std(p2, edges)
    phi22 = float(lsu.compute_lsu(p2, edges, nb, box, depth=2, locality=2))
    sk0, kmin, nm = s_k0(p2, box)
    slow = float(lsu.low_k_structure_factor(p2, box, kmax=KPEN))
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    np.save(f"Structures/{date}_{tag}_{label}_edges.npy", edges)
    rec = dict(angstd=astd, phi22=phi22, S_k0=float(sk0), S_low=slow)
    try:
        m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
        rd = m["ring_distribution"]; E = m["E"]
        rec.update(alpha=m["S_v_alpha_low"], svpeak=m["S_v_peak"], bstd=m["bond_len_std"],
                   r8=100*rd.get(8,0)/E, ring_mean=m["ring_mean"], girth=min(rd) if rd else 0)
    except RuntimeError:
        rec.update(alpha=float("nan"), svpeak=float("nan"), bstd=float("nan"),
                   r8=float("nan"), ring_mean=float("nan"), girth=0)
    return out, rec


t0 = time.time(); traj = []; done = 0; best = {"phi22": 0.0}
_, m0 = measure(pos, edges, "ck0")
print(f"[p22 ck=     0] SEED Phi22={m0['phi22']:.4f} S_k0={m0['S_k0']:.4f} S_low={m0['S_low']:.4f} "
      f"angstd={m0['angstd']:.2f} 8r={m0.get('r8',float('nan')):.1f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=Tslice[0], T_final=Tslice[-1], temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=W, uniformity_kmax=KPEN, check_lsu_every=0,
        use_jax=True, use_jaxopt=False, verbose=False)
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    _, rec = measure(pos, edges, f"ck{done//1000}k")
    rec.update(iter=done, T=float(Tslice[-1]), acc=acc)
    traj.append(rec)
    # PASS for the RELAXED target = Phi22>=0.88 AND low S(k) (S_low<=0.06, S_k0<=0.08) AND amorphous
    amorph = (not math.isnan(rec.get("svpeak", float("nan")))) and rec["svpeak"] < 3.0
    relaxed_pass = (rec["phi22"] >= 0.88 and rec["S_low"] <= 0.06 and rec["S_k0"] <= 0.08 and amorph)
    if rec["phi22"] > best["phi22"]:
        best = dict(rec)
    tag_str = "RELAXED-PASS" if relaxed_pass else "."
    print(f"[p22 ck={done:6d}] T={rec['T']:.4f} | Phi22={rec['phi22']:.4f} [{tag_str}] | "
          f"S_k0={rec['S_k0']:.3f} S_low={rec['S_low']:.3f} a={rec.get('alpha',float('nan')):+.2f} "
          f"Svpk={rec.get('svpeak',float('nan')):.2f} | angstd={rec['angstd']:.2f} "
          f"8r={rec.get('r8',float('nan')):.1f} bstd={rec.get('bstd',float('nan')):.3f} "
          f"acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print("P22_JSON:", json.dumps({k:(None if isinstance(v,float) and math.isnan(v) else v)
                                   for k,v in rec.items()}), flush=True)

print(f"\n=== BEST Phi22={best['phi22']:.4f} @ S_low={best.get('S_low'):.3f} S_k0={best.get('S_k0'):.3f} "
      f"angstd={best.get('angstd'):.2f} (target Phi22>=0.88) ===", flush=True)
print(f"=== REF Phi22 0.889 | relaxed target = Phi22>=0.88 + S_low<=0.06 + amorphous ===", flush=True)
print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
