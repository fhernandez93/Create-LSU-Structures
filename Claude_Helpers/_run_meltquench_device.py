"""INTEGRATION-PARITY test + fast runner: the mq2 recipe via the ON-DEVICE anneal
(www_anneal_device, BB relax). Same seed/schedule as `_run_meltquench mq2 ...`.
Go/no-go: must track the scipy mq2 seed-42 trajectory within run-to-run noise
(8r +-3, angstd +-0.6, Phi22 +-0.005). If it lands 8r ~40 / Phi22 ~0.85 -> BB
under-relaxation compounds -> NO-GO (escalate to L-BFGS).

Usage: python -m Claude_Helpers._run_meltquench_device <tag> <T_hot> <T_cold> [n_total] [chunk] [seed]
env: HU_W (0), DEEP_RELAX (600), RELAX_ITERS (150)
"""
import sys, os, json, time, datetime, math
import numpy as np
import tools, lsu_network as lsu
from Claude_Helpers._anneal_device import www_anneal_device
from Claude_Helpers._graph_rings import ring_stats_from_edges
from Claude_Helpers._metrics import s_k0

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "mqd"
T_HOT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.09
T_COLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.028
n_total = int(sys.argv[4]) if len(sys.argv) > 4 else 250000
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 25000
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 42
W_UNI = float(os.environ.get("HU_W", "0"))
DEEP = int(os.environ.get("DEEP_RELAX", "600"))
RELAX_ITERS = int(os.environ.get("RELAX_ITERS", "150"))

print(f"=== MELT-QUENCH DEVICE  tag={tag} T {T_HOT}->{T_COLD} n={n_total} chunk={chunk} "
      f"w={W_UNI} relax_iters={RELAX_ITERS} seed={SEED} ===", flush=True)
print("SCIPY mq2 seed42 reference: 100k Phi22 0.859/8r46/angstd10.2; 175k Phi22 0.876/8r57/angstd9.1; "
      "250k Phi22 0.883/8r45/angstd8.6", flush=True)

rng = np.random.default_rng(SEED)
pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False); ctx0.update_topology(edges, neighbors)
pos, _ = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)

g = np.arange(n_total)
T_full = T_HOT * np.exp(math.log(T_COLD / T_HOT) * g / max(1, n_total - 1))


def angstd(p, e):
    nb = lsu.build_neighbors(N, e); tri = lsu.build_angle_triples(nb)
    v = p[tri[:,0]]; a = p[tri[:,1]]; b = p[tri[:,2]]
    da = lsu.pbc_displacement(a-v, box); db = lsu.pbc_displacement(b-v, box)
    da /= np.linalg.norm(da,axis=1,keepdims=True); db /= np.linalg.norm(db,axis=1,keepdims=True)
    return float(np.degrees(np.arccos(np.clip((da*db).sum(1),-1,1))).std())


def measure(pos, edges, label):
    nb = lsu.build_neighbors(N, edges)
    fctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False); fctx.update_topology(edges, nb)
    p2, _, _ = lsu.relax(pos, fctx, max_iter=DEEP); p2 = p2 - box*np.round(p2/box)
    epa = float(fctx.energy(p2.ravel()))/N
    d, m, c, girth = ring_stats_from_edges(edges, N)
    sk0, _, _ = s_k0(p2, box)
    np.save(f"Structures/{date}_{tag}_{label}_edges.npy", edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True, clip_endpoints_to_box=False)
    np.savetxt(f"Structures/{date}_{tag}_{label}.txt", rods, fmt="%.6f", delimiter="\t")
    return dict(epa=epa, angstd=angstd(p2, edges),
                phi22=float(lsu.compute_lsu(p2, edges, nb, box, depth=2, locality=2)),
                S_k0=float(sk0), r8=d.get(8,0), r7=d.get(7,0), mean=m, girth=girth)


t0 = time.time(); done = 0
m0 = measure(pos, edges, "ck0")
print(f"[mqd ck=     0] SEED E/atom={m0['epa']:.4f} Phi22={m0['phi22']:.4f} angstd={m0['angstd']:.2f} 8r={m0['r8']:.1f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done+n_this]
    pos, edges, neighbors, hist = www_anneal_device(
        pos, edges, neighbors, box, D0, W, n_this, Tslice, rng,
        relax_iters=RELAX_ITERS, local_shell_depth=4, uniformity_weight=W_UNI)
    done += n_this
    acc = hist["accepted"]/max(1, hist["proposed"])
    r = measure(pos, edges, f"ck{done//1000}k"); r.update(iter=done, T=float(Tslice[-1]), acc=acc)
    print(f"[mqd ck={done:6d}] T={r['T']:.4f} | E/atom={r['epa']:.4f} Phi22={r['phi22']:.4f} "
          f"angstd={r['angstd']:.2f} | 8r={r['r8']:.1f} 7r={r['r7']:.1f} mean={r['mean']:.2f} girth={r['girth']} "
          f"S_k0={r['S_k0']:.3f} acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print("MQD_JSON:", json.dumps(r), flush=True)

print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
