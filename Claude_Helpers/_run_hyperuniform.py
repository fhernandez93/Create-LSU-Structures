"""DECISIVE RECIPE TEST: near-hyperuniform seed PLACEMENT + Keating energy.

The 2026-06-22 chain: (1) energy f1/f2 form bug fixed (Keating, now default) ->
the anneal HOLDS a good S(k0) but cannot CREATE one; (2) S(k0)=0.041 must be
SUPPLIED by vertex placement; (3) collective-coordinate optimisation gives a seed
with S(k0)~0.04, amorphous (no Bragg), valid deg-3 build. This runs the anneal
from that seed and asks: does Keating HOLD S(k0)~0.04 while the topology anneals
toward the reference's ANGLE distribution (ref bond-angle std 8.41 deg) and S(k)?

Primary targets (user): S(k) [S_k0 ref 0.041, alpha ref +1.51] and the bond-angle
+ dihedral distributions [ang std 8.41, dih_ent 0.796, Phi22 0.889]. Rings 2ndary.

Usage: python -m Claude_Helpers._run_hyperuniform <tag> [n_total] [chunk] [seed]
env: HU_KMAX (3), HU_ITERS (400), PROBE_T0 (0.06), PROBE_TF (0.012), HU_W (0 uniformity weight)
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
import tools
from Claude_Helpers._hyperuniform_seed import hyperuniform_points
from Claude_Helpers._metrics import full_metrics_safe

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "hyperu"
n_total = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
KMAX = int(os.environ.get("HU_KMAX", "3"))
HITERS = int(os.environ.get("HU_ITERS", "400"))
T0 = float(os.environ.get("PROBE_T0", "0.06"))
TF = float(os.environ.get("PROBE_TF", "0.012"))
W = float(os.environ.get("HU_W", "0"))

REF = dict(S_k0=0.041, alpha=1.51, angstd=8.41, dih=0.796, phi22=0.889, r6=7.6, r8=59.7)
print(f"=== HYPERUNIFORM-SEED + KEATING  tag={tag} n={n_total} chunk={chunk} "
      f"kmax={KMAX} hiters={HITERS} T={T0}->{TF} w={W} seed={SEED} "
      f"KEATING={lsu._KEATING_F1F2} ===", flush=True)

# --- near-hyperuniform placement -> seed topology on it -> settle ---
hp = hyperuniform_points(N, box, D0, np.random.default_rng(SEED), kmax=KMAX,
                         maxiter=HITERS, verbose=True)
_orig = lsu._poisson_disk_pbc
lsu._poisson_disk_pbc = lambda n, b, md, rng, max_tries=0: hp.copy()
rng = np.random.default_rng(SEED)
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
lsu._poisson_disk_pbc = _orig
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[hu] seed built+settled: E={len(edges)} bond_max={srep['bond_max_after']/D0:.2f}d0 "
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
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, lsu.build_neighbors(N, edges))
    p2, _, _ = lsu.relax(pos, fctx, max_iter=50)
    p2 = p2 - box * np.round(p2 / box)
    amean, astd = angle_std(p2, edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    try:
        m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
    except RuntimeError as e:
        print(f"[hu {label}] metrics skipped (collision): {e}", flush=True)
        return out, None
    m["_angstd"] = astd; m["_angmean"] = amean
    return out, m


t0 = time.time(); traj = []; done = 0
b0, m0 = measure(pos, edges, "ck0")
if m0:
    print(f"[hu ck=     0] SEED  S_k0={m0['S_k0']:.4f} S_low={m0['S_low_k2']:.4f} "
          f"a={m0['S_v_alpha_low']:+.2f} angstd={m0['_angstd']:.2f} "
          f"Phi22={m0['phi22']:.4f} dih={m0['dihedral_entropy']:.3f} "
          f"Svpeak={m0['S_v_peak']:.2f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=Tslice[0], T_final=Tslice[-1], temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=W, uniformity_kmax=2, check_lsu_every=0,
        use_jax=True, use_jaxopt=False, verbose=False)
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    _, m = measure(pos, edges, f"ck{done//1000}k")
    if m is None:
        print(f"[hu ck={done:6d}] T={float(Tslice[-1]):.4f} acc={acc:.1%} skipped", flush=True)
        continue
    rd = m["ring_distribution"]; E = m["E"]
    def fr(n): return 100 * rd.get(n, 0) / E
    rec = dict(iter=done, T=float(Tslice[-1]), S_k0=m["S_k0"], S_low=m["S_low_k2"],
               alpha=m["S_v_alpha_low"], angstd=m["_angstd"], phi22=m["phi22"],
               dih=m["dihedral_entropy"], svpeak=m["S_v_peak"], r6=fr(6), r8=fr(8),
               ring_mean=m["ring_mean"], girth=min(rd) if rd else 0, acc=acc)
    traj.append(rec)
    print(f"[hu ck={done:6d}] T={rec['T']:.4f} | S_k0={rec['S_k0']:.4f} "
          f"S_low={rec['S_low']:.4f} a={rec['alpha']:+.2f} | angstd={rec['angstd']:.2f} "
          f"Phi22={rec['phi22']:.4f} dih={rec['dih']:.3f} | 6r={rec['r6']:.1f} "
          f"8r={rec['r8']:.1f} mean={rec['ring_mean']:.2f} Svpk={rec['svpeak']:.2f} "
          f"acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print("HU_JSON:", json.dumps(rec), flush=True)

print(f"\n=== REF: S_k0 0.041 a +1.51 angstd 8.41 Phi22 0.889 dih 0.796 6r 7.6 8r 59.7 ===", flush=True)
print("HU_TRAJ:", json.dumps(traj), flush=True)
print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
