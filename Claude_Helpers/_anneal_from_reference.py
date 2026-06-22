"""DECISIVE diagnostic: does OUR pure-WWW anneal PRESERVE or DESTROY the reference?

Both seed-audit agents leave the SAME open question: if the local WWW anneal has
no restoring force toward low S(k0) (agent: "anneal can't lower S(k0)"), how did
Sellers reach 0.041? And is the COARSENING anneal-origin or seed-origin? This
test answers both at once by SEEDING FROM THE GOLD-STANDARD REFERENCE itself
(Example/lsu_example_ends.txt: S(k0)=0.041, 8r=59.7%, ring mean 7.99) and running
our pure-Sellers WWW (w=0) on the SAME cold schedule that coarsens random_bm2000.

Discriminator (checkpointed):
  HOLDS  (S(k0)~0.04, 8r~60, ring mean ~8.0) -> our move+energy+relax are faithful;
          the reference is a stable fixed point of our anneal -> the ENTIRE gap is
          the seed (we just can't REACH this basin from random_bm2000).
  DEGRADES (S(k0) climbs, 8r falls, 9r grows, ring mean rises) -> our anneal/schedule
          actively DESTROYS reference quality -> coarsening+void are ANNEAL-origin
          (a faithfulness bug in the anneal, NOT just the seed) -> stop re-engineering
          the seed; fix the anneal.

Read-only w.r.t. the reference (loads, never writes Example/). Saves only dated
diagnostic rods to Structures/.

Usage: python -m Claude_Helpers._anneal_from_reference [n_total] [chunk] [seed]
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
import tools
from Claude_Helpers._metrics import full_metrics_safe

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
# T-band env-overridable: cold default coarsens random_bm2000; warm slow-cool
# (0.13->0.02) is the advisor's discriminating test vs the random-seed probe.
T0 = float(os.environ.get("REF_T0", "0.045"))
TF = float(os.environ.get("REF_TF", "0.015"))
date = datetime.date.today().strftime("%Y%m%d")

n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42
tag = os.environ.get("REF_TAG", "annealref")

REF = dict(r6=7.6, r7=10.0, r8=59.7, r9=20.9, ring_mean=7.99, S_k0=0.041,
           phi22=0.889, alpha=1.51)

print(f"=== ANNEAL-FROM-REFERENCE  n_total={n_total} chunk={chunk} w=0 "
      f"T={T0}->{TF} seed={SEED} ===", flush=True)

# --- load the gold reference AS A NETWORK (read-only) ---
rods = np.loadtxt("Example/lsu_example_ends.txt")
pos, edges = tools.rods_to_network(rods, box, cluster_radius=0.1)
Nref = len(pos)
neighbors = lsu.build_neighbors(Nref, edges)
deg = np.array([len(neighbors[i]) for i in range(Nref)])
print(f"[ref] loaded {rods.shape[0]} rods -> N={Nref} E={len(edges)} "
      f"deg(min/max)={deg.min()}/{deg.max()} all3={np.all(deg==3)}", flush=True)
if not np.all(deg == 3) or Nref != N:
    print(f"[ref] WARNING: expected N=1000 all-deg-3; got N={Nref}", flush=True)


def measure(pos, edges, label):
    fctx = lsu._RelaxContext(Nref, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, lsu.build_neighbors(Nref, edges))
    p2, _, _ = lsu.relax(pos, fctx, max_iter=50)
    p2 = p2 - box * np.round(p2 / box)
    rd = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                             clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rd, fmt="%.6f", delimiter="\t")
    m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
    return p2, m


def rowstr(it, T, m):
    r = m["ring_distribution"]; E = m["E"]
    def fr(n): return 100 * r.get(n, 0) / E
    return (f"[{tag} {it:6d}] T={T:.4f} Phi22={m['phi22']:.4f} "
            f"girth={min(r) if r else 0} | 6r={fr(6):.1f} 7r={fr(7):.1f} "
            f"8r={fr(8):.1f} 9r={fr(9):.1f} | ringmean={m['ring_mean']:.3f} "
            f"S_k0={m['S_k0']:.3f} a={m['S_v_alpha_low']:.2f} "
            f"| dS_k0={m['S_k0']-REF['S_k0']:+.3f} d8r={fr(8)-REF['r8']:+.1f} "
            f"dmean={m['ring_mean']-REF['ring_mean']:+.2f}")


# --- baseline: reference under OUR energy (50-iter relax, no topology change) ---
rng = np.random.default_rng(SEED)
_, m0 = measure(pos, edges, "ck0")
print("BASELINE (reference relaxed on our energy, 0 WWW moves):", flush=True)
print(rowstr(0, T0, m0), flush=True)
print("CKPT_JSON:", json.dumps(dict(iter=0, S_k0=m0["S_k0"], phi22=m0["phi22"],
      ring_mean=m0["ring_mean"], alpha=m0["S_v_alpha_low"],
      r8=100*m0["ring_distribution"].get(8,0)/m0["E"])), flush=True)

# --- shared cold geometric schedule over the full run ---
g = np.arange(n_total)
T_full = T0 * np.exp(math.log(TF / T0) * g / max(1, n_total - 1))

t0 = time.time()
done = 0
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=Tslice[0], T_final=Tslice[-1], temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=0.0, check_lsu_every=0,
        use_jax=True, use_jaxopt=False, verbose=False,
    )
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    pos, m = measure(pos, edges, f"ck{done//1000}k")
    print(rowstr(done, float(Tslice[-1]), m) + f" acc={acc:.1%} t={time.time()-t0:.0f}s",
          flush=True)
    r = m["ring_distribution"]; E = m["E"]
    print("CKPT_JSON:", json.dumps(dict(iter=done, T=float(Tslice[-1]),
          S_k0=m["S_k0"], phi22=m["phi22"], ring_mean=m["ring_mean"],
          alpha=m["S_v_alpha_low"], r6=100*r.get(6,0)/E, r8=100*r.get(8,0)/E,
          r9=100*r.get(9,0)/E, acc=acc)), flush=True)

print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
