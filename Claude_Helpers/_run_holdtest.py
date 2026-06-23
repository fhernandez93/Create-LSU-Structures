"""HYSTERESIS GATE: two-sided constant-T hold to test whether the reference's
8-ring-rich topology is KINETICALLY UNREACHABLE from disorder (a genuine trap) vs
merely melted by warmth.

Per advisor: a warm (T~0.10) plateau is ambiguous (it may just be the warm
equilibrium -- warm-from-reference degrades 8r 60->47). The discriminating test is a
constant hold at ONE cold, stability-preserving T (~0.045, where memory says the
reference is HELD: 8r 59.7->57), run from BOTH ends:
  - reference-seeded hold  -> stays ordered (~8r 57)?  [order is stable here]
  - disorder-seeded  hold  -> climbs toward it, or freezes?  [reachable or trapped]
If order is stable at this T yet disorder can't reach it => HYSTERESIS = kinetic trap
= an airtight, mechanism-level negative.

Measurement is GRAPH-TRUE (immune to the hot-checkpoint vertex collisions that break
the rod round-trip and inflate angstd): deep Keating relax (DEEP_RELAX iters) of the
in-memory edge list, then angstd from edges, Phi22 via compute_lsu, S_k0 from
positions. Ring distribution / void-slope via a best-effort rod round-trip (skipped
on collision). Saves edges per checkpoint for offline re-analysis.

Usage: python -m Claude_Helpers._run_holdtest <tag> <T_hold> <seed_source> [n_total] [chunk] [seed]
  seed_source = 'random_bm2000'  OR  a path to a rod-endpoints .txt (e.g. the reference)
env: HU_W (0), HU_KMAX (2), DEEP_RELAX (600)
"""
import sys, os, json, time, datetime, math
import numpy as np
import tools
import lsu_network as lsu
from Claude_Helpers._metrics import full_metrics_safe, s_k0

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "hold"
T_HOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.045
SRC = sys.argv[3] if len(sys.argv) > 3 else "random_bm2000"
n_total = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 42
W = float(os.environ.get("HU_W", "0"))
KMAX = int(os.environ.get("HU_KMAX", "2"))
DEEP = int(os.environ.get("DEEP_RELAX", "600"))

print(f"=== HOLD-TEST  tag={tag} T_hold={T_HOLD} src={SRC} n={n_total} chunk={chunk} "
      f"w={W} seed={SEED} deep_relax={DEEP} KEATING={lsu._KEATING_F1F2} ===", flush=True)

rng = np.random.default_rng(SEED)
if SRC == "random_bm2000":
    pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
    neighbors = lsu.build_neighbors(N, edges)
    ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    ctx0.update_topology(edges, neighbors)
    pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
    print(f"[ht] random_bm2000 seed built+settled: E={len(edges)} "
          f"bond_max={srep['bond_max_after']/D0:.2f}d0 min_nb={srep['min_nb_after']/D0:.3f}d0",
          flush=True)
else:
    rods = np.loadtxt(SRC)
    pos, edges = tools.rods_to_network(rods, box)
    assert len(pos) == N, f"loaded N={len(pos)} != {N}"
    neighbors = lsu.build_neighbors(N, edges)
    print(f"[ht] loaded seed from {SRC}: N={len(pos)} E={len(edges)}", flush=True)


def angle_std(p, edges):
    nb = lsu.build_neighbors(N, edges); tri = lsu.build_angle_triples(nb)
    v = p[tri[:, 0]]; a = p[tri[:, 1]]; b = p[tri[:, 2]]
    da = lsu.pbc_displacement(a - v, box); db = lsu.pbc_displacement(b - v, box)
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1)))
    return float(ang.mean()), float(ang.std())


def measure(pos, edges, label):
    """GRAPH-TRUE deep-relax read. angstd/Phi22/S_k0 never touch the rod round-trip."""
    nb = lsu.build_neighbors(N, edges)
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, nb)
    p2, _, _ = lsu.relax(pos, fctx, max_iter=DEEP)
    p2 = p2 - box * np.round(p2 / box)
    amean, astd = angle_std(p2, edges)
    phi22 = lsu.compute_lsu(p2, edges, nb, box, depth=2, locality=2)
    phi12 = lsu.compute_lsu(p2, edges, nb, box, depth=1, locality=2)
    sk0, kmin, nm = s_k0(p2, box)
    slow = float(lsu.low_k_structure_factor(p2, box, kmax=KMAX))
    # save edges + a rod render for offline re-analysis
    np.save(f"Structures/{date}_{tag}_{label}_edges.npy", edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    rec = dict(angstd=astd, angmean=amean, phi22=float(phi22), phi12=float(phi12),
               S_k0=float(sk0), S_low=slow)
    # best-effort full metrics (rings, void slope, S_v_peak) -- may skip on collision
    try:
        m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
        rd = m["ring_distribution"]; E = m["E"]
        rec.update(alpha=m["S_v_alpha_low"], svpeak=m["S_v_peak"],
                   bstd=m["bond_len_std"], min_nb=m["min_nb"],
                   ring_mean=m["ring_mean"], girth=min(rd) if rd else 0,
                   r6=100*rd.get(6,0)/E, r7=100*rd.get(7,0)/E,
                   r8=100*rd.get(8,0)/E, r9=100*rd.get(9,0)/E)
    except RuntimeError:
        rec.update(alpha=float("nan"), svpeak=float("nan"), bstd=float("nan"),
                   min_nb=float("nan"), ring_mean=float("nan"), girth=0,
                   r6=float("nan"), r7=float("nan"), r8=float("nan"), r9=float("nan"))
    return out, rec


t0 = time.time(); traj = []; done = 0
_, m0 = measure(pos, edges, "ck0")
print(f"[ht ck=     0] SEED angstd={m0['angstd']:.2f} Phi22={m0['phi22']:.4f} "
      f"S_k0={m0['S_k0']:.3f} 8r={m0.get('r8',float('nan')):.1f} "
      f"mean={m0.get('ring_mean',float('nan')):.2f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = np.full(n_this, T_HOLD, float)
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=T_HOLD, T_final=T_HOLD, temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=W, uniformity_kmax=KMAX, check_lsu_every=0,
        use_jax=True, use_jaxopt=False, verbose=False)
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    _, rec = measure(pos, edges, f"ck{done//1000}k")
    rec.update(iter=done, T=T_HOLD, acc=acc)
    traj.append(rec)
    gate = "PASS" if (rec['angstd'] <= 9.0 and rec['phi22'] >= 0.88) else "."
    print(f"[ht ck={done:6d}] T={T_HOLD:.3f} | angstd={rec['angstd']:.2f} "
          f"Phi22={rec['phi22']:.4f} [{gate}] | 8r={rec.get('r8',float('nan')):.1f} "
          f"7r={rec.get('r7',float('nan')):.1f} mean={rec.get('ring_mean',float('nan')):.2f} | "
          f"S_k0={rec['S_k0']:.3f} S_low={rec['S_low']:.3f} bstd={rec.get('bstd',float('nan')):.3f} "
          f"acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print("HT_JSON:", json.dumps({k:(None if isinstance(v,float) and math.isnan(v) else v)
                                  for k,v in rec.items()}), flush=True)

print(f"\n=== REF: angstd 8.41 Phi22 0.889 8r 59.7 mean 7.99 S_k0 0.041 ===", flush=True)
print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
