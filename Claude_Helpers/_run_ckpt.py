"""CHECKPOINTED production runner — schedule + weight + per-chunk FULL RING VECTOR.

Combines _run_sched's schedule flexibility (geometric / sustained cool->hold->cool)
and uniformity penalty with _probe_w0_checkpoints's per-chunk measurement, so a
WITH-PENALTY run can be judged on the FULL ring vector -> reference at every
checkpoint (advisor: do NOT judge on "6r>=5% & Phi high"; judge 8r->60% w/ 7r->10%
AND where 6r dies on the final cool = kinetic-vs-thermodynamic diagnostic).

Faithful preamble: random_bm2000 seed -> settle_seed_with_repulsion (JAX) ->
www_anneal in chunks sharing ONE schedule -> 50-iter polish on the final state.
Saves final rods + kwargs (production / Example candidate).

Usage: python -m Claude_Helpers._run_ckpt <tag> '<json>'
json keys: N, n_iters, weight, kmax, seed, chunk, schedule{...}
  schedule: {"kind":"geometric","T0":..,"Tf":..}
            {"kind":"sustained","T0":..,"T_hold":..,"Tf":..,"f_cool1":..,"f_hold":..}
"""
import sys, json, time, datetime, math
import numpy as np
import lsu_network as lsu
import tools
from Claude_Helpers._metrics import full_metrics_safe

WEIGHTS = (0.7, 0.7, 0.3, 0.4)
D0 = 0.8
date = datetime.date.today().strftime("%Y%m%d")

# reference ring vector + key scalars (recomputed gold standard, N=1000)
REF = dict(r6=7.6, r7=10.0, r8=59.7, r9=20.9, r10=1.7, phi22=0.889,
           S_k0=0.041, alpha=1.51)

tag = sys.argv[1]
cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
N = int(cfg.get("N", 1000))
BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)
box = np.array([BOX, BOX, BOX], float)
n_iters = int(cfg.get("n_iters", 60000))
weight = float(cfg.get("weight", 30.0))
kmax = int(cfg.get("kmax", 2))
SEED = int(cfg.get("seed", 42))
chunk = int(cfg.get("chunk", 10000))
sched = cfg.get("schedule", {"kind": "sustained", "T0": 0.15, "T_hold": 0.085,
                             "Tf": 0.02, "f_cool1": 0.2, "f_hold": 0.5})


def build_T(sched, n):
    k = sched.get("kind", "geometric")
    if k == "geometric":
        T0, Tf = sched["T0"], sched["Tf"]
        g = np.arange(n)
        return T0 * np.exp(math.log(Tf / T0) * g / max(1, n - 1))
    if k == "sustained":
        T0, Th, Tf = sched["T0"], sched["T_hold"], sched["Tf"]
        f1 = sched.get("f_cool1", 0.2); fh = sched.get("f_hold", 0.5)
        n1 = int(n * f1); nh = int(n * fh); n2 = n - n1 - nh
        a = T0 * np.exp(np.linspace(0, math.log(Th / T0), n1))
        b = np.full(nh, Th)
        c = Th * np.exp(np.linspace(0, math.log(Tf / Th), max(1, n2)))
        return np.concatenate([a, b, c])[:n]
    raise ValueError(k)


print(f"=== CKPT RUN tag={tag} N={N} BOX={BOX:.4f} n={n_iters} w={weight} "
      f"kmax={kmax} seed={SEED} chunk={chunk} sched={sched} ===", flush=True)

t0 = time.time()
rng = np.random.default_rng(SEED)
pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[ckpt] settle: bond_max {srep['bond_max_after']/D0:.2f}d0 "
      f"min_nb {srep['min_nb_after']/D0:.3f}d0", flush=True)

T_full = build_T(sched, n_iters)
print(f"[ckpt] T: start={T_full[0]:.4f} hold~{np.median(T_full):.4f} "
      f"end={T_full[-1]:.4f}", flush=True)


def measure(pos, edges, label, save=False):
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, lsu.build_neighbors(N, edges))
    p2, _, _ = lsu.relax(pos, fctx, max_iter=50)
    p2 = p2 - box * np.round(p2 / box)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
    m["_radius_used"] = cr; m["_path"] = out
    return m


traj = []
done = 0
while done < n_iters:
    n_this = min(chunk, n_iters - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=Tslice[0], T_final=Tslice[-1], temperatures=Tslice,
        rng=rng, target_lsu=None,
        relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=weight, uniformity_kmax=kmax,
        check_lsu_every=0, use_jax=True, use_jaxopt=False, verbose=False,
    )
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    m = measure(pos, edges, f"ck{done//1000}k")
    rd = m["ring_distribution"]; E = m["E"]
    def fr(n): return 100 * rd.get(n, 0) / E
    rec = dict(iter=done, T=float(Tslice[-1]), phi12=m["phi12"], phi22=m["phi22"],
               r5=fr(5), r6=fr(6), r7=fr(7), r8=fr(8), r9=fr(9), r10=fr(10),
               ring_mean=m["ring_mean"], girth=min(rd) if rd else 0,
               S_k0=m["S_k0"], S_low_k2=m["S_low_k2"], alpha=m["S_v_alpha_low"],
               min_nb=m.get("min_nb", float("nan")), acc=acc, elapsed=time.time() - t0)
    traj.append(rec)
    # full ring vector + delta-to-reference, so the run is judged correctly
    print(f"[ckpt {done:6d}] T={rec['T']:.4f} Phi22={rec['phi22']:.4f} girth={rec['girth']} "
          f"| 5r={rec['r5']:.1f} 6r={rec['r6']:.1f} 7r={rec['r7']:.1f} 8r={rec['r8']:.1f} "
          f"9r={rec['r9']:.1f} 10r={rec['r10']:.1f} "
          f"| dRING(6/7/8/9)={rec['r6']-REF['r6']:+.1f}/{rec['r7']-REF['r7']:+.1f}/"
          f"{rec['r8']-REF['r8']:+.1f}/{rec['r9']-REF['r9']:+.1f} "
          f"| S_k0={rec['S_k0']:.3f} a={rec['alpha']:.2f} acc={acc:.1%} "
          f"t={rec['elapsed']:.0f}s", flush=True)
    print("CKPT_JSON:", json.dumps(rec), flush=True)

# final state already measured as last checkpoint; persist kwargs + a clean final copy
final = traj[-1]
with open(f"Structures/{date}_{tag}.kwargs.json", "w") as f:
    json.dump(dict(N=N, n_iters=n_iters, weight=weight, kmax=kmax, seed=SEED,
                   schedule=sched, final=final), f, indent=2)

print("\n=== FULL-RING-VECTOR VERDICT (ref: 6r7.6 7r10.0 8r59.7 9r20.9 | Phi22 .889 "
      "S_k0 .041 a +1.51) ===", flush=True)
print(f"{'iter':>7} {'T':>6} {'Phi22':>6} {'g':>2} {'6r':>5} {'7r':>5} {'8r':>5} "
      f"{'9r':>5} {'S_k0':>6} {'a':>5}")
for r in traj:
    print(f"{r['iter']:>7} {r['T']:>6.3f} {r['phi22']:>6.3f} {r['girth']:>2} "
          f"{r['r6']:>5.1f} {r['r7']:>5.1f} {r['r8']:>5.1f} {r['r9']:>5.1f} "
          f"{r['S_k0']:>6.3f} {r['alpha']:>5.2f}")
print("CKPT_TRAJ_JSON:", json.dumps(traj), flush=True)
print(f"=== done {time.time()-t0:.0f}s; final saved {final['iter']//1000}k "
      f"8r={final['r8']:.1f} 6r={final['r6']:.1f} Phi22={final['phi22']:.3f} "
      f"S_k0={final['S_k0']:.3f} ===", flush=True)
