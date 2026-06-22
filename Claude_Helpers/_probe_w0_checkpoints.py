"""Run 2 — RING-SHAPE REACHABILITY PROBE (advisor-designed).

Pure-Sellers WWW (uniformity_weight=0), cold T 0.045->0.015 over a TOTAL of
n_total iterations, run in chunks of `chunk` iters that SHARE one geometric
T-schedule (so it is a single continuous anneal). After each chunk, dump the
network and measure the full ring SHAPE + S(k0) + Phi + dihedral entropy.

Discriminator: does 8-ring% ever reach ~55-60% while 6-ring% is still >=5%?
  YES  -> the reference ring shape IS on the pure-WWW trajectory (void is the
          only real problem; the penalty's job is just the void).
  NO   -> reference shape is unreachable with this energy/schedule
          (decisive negative result).

Faithful to generate_lsu_network preamble: random_bm2000 seed ->
settle_seed_with_repulsion (JAX path) -> www_anneal. Burn-in OFF.
NOTE: chunking rebuilds the JAX ctx per chunk and (with check_lsu_every=0)
consumes no rng inside the loop for LSU checks, so this is a faithful INSTANCE
of the protocol, not bit-identical to a monolithic 100k call (different rng
realization only). The reachability conclusion is robust to rng realization.

Usage: python _probe_w0_checkpoints.py <tag> [n_total] [chunk]
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
import tools
from Claude_Helpers._metrics import full_metrics, full_metrics_safe, print_metrics

N = int(os.environ.get("PROBE_N", "1000"))
D0 = 0.8
BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
# Temperature band is env-overridable so the SAME faithful probe can sweep the
# intermediate-T regime (the cold default skipped it). Cold default = 0.045->0.015.
T0 = float(os.environ.get("PROBE_T0", "0.045"))
TF = float(os.environ.get("PROBE_TF", "0.015"))

tag = sys.argv[1] if len(sys.argv) > 1 else "probew0"
n_total = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
date = datetime.date.today().strftime("%Y%m%d")

print(f"=== PROBE tag={tag} N={N} BOX={BOX:.4f} n_total={n_total} chunk={chunk} "
      f"w=0 T={T0}->{TF} seed={SEED} ===", flush=True)

# --- faithful preamble: seed + settle ---
rng = np.random.default_rng(SEED)
pos, edges, seed_meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[probe] settle: bond_max {srep['bond_max_after']/D0:.2f}d0, "
      f"min_nb {srep['min_nb_after']/D0:.3f}d0", flush=True)

# --- shared geometric T schedule over the FULL run ---
g = np.arange(n_total)
log_ratio = math.log(TF / T0)
T_full = T0 * np.exp(log_ratio * g / max(1, n_total - 1))


def measure(pos, edges, label):
    """Save rods + compute full metrics from the saved file (parity w/ harness)."""
    # light final polish (matches generate's 50-iter cleanup) for geometry parity
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, lsu.build_neighbors(N, edges))
    p2, _, _ = lsu.relax(pos, fctx, max_iter=50)
    p2 = p2 - box * np.round(p2 / box)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    try:
        m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
    except RuntimeError as e:
        # transient non-bonded collision the rod round-trip can't reconstruct to
        # deg-3 (no effect on the live network state) — skip metrics, keep annealing
        print(f"[probe {label}] metrics SKIPPED (collision): {e}", flush=True)
        return out, None
    m["_radius_used"] = cr
    return out, m


t0 = time.time()
traj = []
done = 0
chunk_idx = 0
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this,
        T0=T0, T_final=TF,                  # ignored when temperatures= given
        temperatures=Tslice,
        rng=rng,
        target_lsu=None,                    # NO early exit — see full trajectory
        relax_local_iters=100,
        local_shell_depth=4,
        uniformity_weight=0.0,              # PURE Sellers
        check_lsu_every=0,                  # no rng-consuming LSU checks inside loop
        use_jax=True, use_jaxopt=False, verbose=False,
    )
    done += n_this
    chunk_idx += 1
    acc = hist["accepted"] / max(1, hist["proposed"])
    er = hist["early_rejected"] / max(1, hist["proposed"])
    out, m = measure(pos, edges, f"ck{done//1000}k")
    if m is None:
        print(f"[probe ck={done:6d}] T={float(Tslice[-1]):.4f} acc={acc:.1%} "
              f"-- checkpoint metrics skipped (collision), annealing continues", flush=True)
        continue
    rd = m["ring_distribution"]
    E = m["E"]
    def fr(n): return 100 * rd.get(n, 0) / E
    rec = dict(iter=done, T=float(Tslice[-1]),
               phi12=m["phi12"], phi22=m["phi22"],
               r5=fr(5), r6=fr(6), r7=fr(7), r8=fr(8), r9=fr(9), r10=fr(10),
               ring_mean=m["ring_mean"], girth=min(rd) if rd else 0,
               S_k0=m["S_k0"], S_low_k2=m["S_low_k2"], alpha=m["S_v_alpha_low"],
               dih_ent=m["dihedral_entropy"], bond_std=m["bond_len_std"],
               min_nb=m.get("min_nb", float("nan")), radius=m.get("_radius_used", 0.1),
               acc=acc, early=er, elapsed=time.time() - t0)
    traj.append(rec)
    print(f"[probe ck={done:6d}] T={rec['T']:.4f} Phi22={rec['phi22']:.4f} "
          f"girth={rec['girth']} 5r={rec['r5']:.1f} 6r={rec['r6']:.1f} "
          f"7r={rec['r7']:.1f} 8r={rec['r8']:.1f} 9r={rec['r9']:.1f} "
          f"ringmean={rec['ring_mean']:.3f} S_k0={rec['S_k0']:.3f} "
          f"S_low={rec['S_low_k2']:.3f} a={rec['alpha']:.2f} "
          f"dih={rec['dih_ent']:.3f} acc={acc:.1%} er={er:.1%} "
          f"t={rec['elapsed']:.0f}s", flush=True)
    print("CKPT_JSON:", json.dumps(rec), flush=True)

print("\n=== TRAJECTORY (does 8r reach ~55-60% while 6r>=5%?) ===", flush=True)
print(f"{'iter':>7} {'Phi22':>6} {'girth':>5} {'5r':>5} {'6r':>5} {'7r':>5} "
      f"{'8r':>6} {'9r':>5} {'mean':>5} {'S_k0':>6} {'a':>5} {'dih':>5}")
for r in traj:
    print(f"{r['iter']:>7} {r['phi22']:>6.3f} {r['girth']:>5} {r['r5']:>5.1f} "
          f"{r['r6']:>5.1f} {r['r7']:>5.1f} {r['r8']:>6.1f} {r['r9']:>5.1f} "
          f"{r['ring_mean']:>5.2f} {r['S_k0']:>6.3f} {r['alpha']:>5.2f} "
          f"{r['dih_ent']:>5.3f}")
print("PROBE_TRAJ_JSON:", json.dumps(traj), flush=True)
