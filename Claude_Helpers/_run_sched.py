"""Flexible N=1000 WWW runner with a CUSTOM temperature schedule + weight + kmax.

Faithful to generate_lsu_network's pipeline (random_bm2000 seed ->
settle_seed_with_repulsion JAX path -> www_anneal -> 50-iter polish ->
network_to_rods), but lets me pass an arbitrary temperatures[] array (e.g. a
sustained moderate-T hold) and any uniformity_weight/kmax. A "geometric"
schedule here is identical to generate_lsu_network.

Usage: python _run_sched.py <tag> '<json>'
json keys: n_iters, weight, kmax, seed, schedule{...}, lsu_target(0.889 stop, or null)
schedule kinds:
  {"kind":"geometric","T0":0.045,"Tf":0.015}
  {"kind":"sustained","T0":0.045,"T_hold":0.025,"Tf":0.012,
   "f_cool1":0.45,"f_hold":0.35}   # cool->hold->cool; f_cool2 = 1-f_cool1-f_hold
"""
import sys, json, time, datetime, math
import numpy as np
import lsu_network as lsu
import tools
from Claude_Helpers._metrics import full_metrics_safe, print_metrics

WEIGHTS = (0.7, 0.7, 0.3, 0.4)
D0 = 0.8
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1]
cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
N = int(cfg.get("N", 1000))                       # density-matched box from N
BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)
box = np.array([BOX, BOX, BOX], float)
n_iters = int(cfg.get("n_iters", 40000))
weight = float(cfg.get("weight", 30.0))
kmax = int(cfg.get("kmax", 2))
SEED = int(cfg.get("seed", 42))
sched = cfg.get("schedule", {"kind": "geometric", "T0": 0.045, "Tf": 0.015})
lsu_target = cfg.get("lsu_target", None)  # None = no early exit


def build_T(sched, n):
    k = sched.get("kind", "geometric")
    if k == "geometric":
        T0, Tf = sched["T0"], sched["Tf"]
        g = np.arange(n)
        return T0 * np.exp(math.log(Tf / T0) * g / max(1, n - 1))
    if k == "sustained":
        T0, Th, Tf = sched["T0"], sched["T_hold"], sched["Tf"]
        f1 = sched.get("f_cool1", 0.45); fh = sched.get("f_hold", 0.35)
        n1 = int(n * f1); nh = int(n * fh); n2 = n - n1 - nh
        a = T0 * np.exp(np.linspace(0, math.log(Th / T0), n1))           # cool T0->Th
        b = np.full(nh, Th)                                              # hold
        c = Th * np.exp(np.linspace(0, math.log(Tf / Th), max(1, n2)))   # cool Th->Tf
        return np.concatenate([a, b, c])[:n]
    raise ValueError(k)


print(f"=== SCHED RUN tag={tag} N={N} BOX={BOX:.4f} n={n_iters} w={weight} "
      f"kmax={kmax} seed={SEED} sched={sched} ===", flush=True)

t0 = time.time()
rng = np.random.default_rng(SEED)
pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[sched] settle: bond_max {srep['bond_max_after']/D0:.2f}d0 "
      f"min_nb {srep['min_nb_after']/D0:.3f}d0", flush=True)

T_arr = build_T(sched, n_iters)
print(f"[sched] T: start={T_arr[0]:.4f} min={T_arr.min():.4f} end={T_arr[-1]:.4f}", flush=True)

pos, edges, neighbors, hist = lsu.www_anneal(
    pos, edges, neighbors, box, D0, WEIGHTS,
    n_iterations=n_iters, T0=T_arr[0], T_final=T_arr[-1], temperatures=T_arr,
    rng=rng,
    target_lsu=(float(lsu_target) if lsu_target is not None else None),
    target_depth=2, target_locality=2, target_tolerance=0.01,
    relax_local_iters=100, local_shell_depth=4,
    uniformity_weight=weight, uniformity_kmax=kmax,
    check_lsu_every=(1000 if lsu_target is not None else 0),
    use_jax=True, use_jaxopt=False, verbose=True,
)
acc = hist["accepted"] / max(1, hist["proposed"])
er = hist["early_rejected"] / max(1, hist["proposed"])
print(f"[sched] anneal done: acc={acc:.2%} early={er:.2%} "
      f"elapsed={time.time()-t0:.1f}s", flush=True)

# final polish (matches generate)
fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
fctx.update_topology(edges, lsu.build_neighbors(N, edges))
pos, _, _ = lsu.relax(pos, fctx, max_iter=50)
pos = pos - box * np.round(pos / box)
rods = lsu.network_to_rods(pos, edges, box, pbc_duplicate_boundary_rods=True,
                           clip_endpoints_to_box=False)
out = f"Structures/{date}_{tag}.txt"
np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
with open(f"Structures/{date}_{tag}.kwargs.json", "w") as f:
    json.dump(dict(n_iters=n_iters, weight=weight, kmax=kmax, seed=SEED,
                   schedule=sched, lsu_target=lsu_target), f, indent=2)
elapsed = time.time() - t0
print(f"=== elapsed {elapsed:.1f}s ({elapsed/n_iters*1000:.2f} ms/iter); saved {out} ===", flush=True)

m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=tag)
print(f"### radius_used={cr} min_nb={m['min_nb']:.4f} ({m['min_nb']/D0:.3f} d0) ###")
print_metrics(m)
rd = m["ring_distribution"]; E = m["E"]
def fr(n): return 100 * rd.get(n, 0) / E
summ = dict(tag=tag, phi12=m["phi12"], phi22=m["phi22"], bond_std=m["bond_len_std"],
           ring_mean=m["ring_mean"], r5=fr(5), r6=fr(6), r7=fr(7), r8=fr(8), r9=fr(9), r10=fr(10),
           girth=min(rd), S_k0=m["S_k0"], S_low_k2=m["S_low_k2"], alpha=m["S_v_alpha_low"],
           dih=m["dihedral_entropy"], S_v_peak=m["S_v_peak"], min_nb=m["min_nb"], radius=cr,
           acc=acc, early=er, elapsed=elapsed)
print("SCHED_SUMMARY:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in summ.items()}), flush=True)
