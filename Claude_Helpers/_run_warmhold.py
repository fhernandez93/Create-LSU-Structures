"""GATE EXPERIMENT: clean w=0 sustained-HOLD topology reachability from random_bm2000.

The decisive question after the 2026-06-22 energy fix: the angle-std (11.6 deg vs
ref 8.41) and Phi22 (0.844 vs 0.889) gaps are TOPOLOGY-bound (the saved structure's
angstd is already Keating-relax-minimal at its topology, so re-hyperuniformising the
GEOMETRY can only cost local order, never build it). So the multi-stage protocol
lives or dies on whether ANNEALING FREED FROM THE VOID PENALTY (w=0) can build the
reference's topology (8r->60%, angstd->8.4) at all.

This is the one schedule shape never cleanly isolated: a CONSTANT-T sustained hold at
w=0 from random_bm2000 (richest topology seed: if IT can't reach the corner with full
freedom, nothing disordered will). The void WILL open (w=0) -- that's expected; we
measure LOCAL order only. Report the TRAJECTORY (plateaued vs still-climbing), not
just the endpoint -- the prior 6r "hard floor" was a cold-schedule artifact.

DISCRIMINATOR: does any w=0 state reach  angstd <= 9 deg  AND  Phi22 >= 0.88  together?

Usage: python -m Claude_Helpers._run_warmhold <tag> <T_hold> [n_total] [chunk] [seed]
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
from Claude_Helpers._metrics import full_metrics_safe

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float)
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "warmhold"
T_HOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
n_total = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 42
W = float(os.environ.get("HU_W", "0"))           # gate = w=0
KMAX = int(os.environ.get("HU_KMAX", "2"))

REF = dict(S_k0=0.041, alpha=1.51, angstd=8.41, dih=0.796, phi22=0.889, r6=7.6, r8=59.7)
print(f"=== WARM-HOLD GATE  tag={tag} seed_kind=random_bm2000 T_hold={T_HOLD} "
      f"n={n_total} chunk={chunk} w={W} seed={SEED} KEATING={lsu._KEATING_F1F2} ===",
      flush=True)

# --- random_bm2000 seed (richest topology start) -> settle ---
rng = np.random.default_rng(SEED)
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[wh] seed built+settled: E={len(edges)} bond_max={srep['bond_max_after']/D0:.2f}d0 "
      f"min_nb={srep['min_nb_after']/D0:.3f}d0", flush=True)


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
        print(f"[wh {label}] metrics skipped (collision): {e}", flush=True)
        return out, None
    m["_angstd"] = astd; m["_angmean"] = amean
    return out, m


t0 = time.time(); traj = []; done = 0
b0, m0 = measure(pos, edges, "ck0")
if m0:
    print(f"[wh ck=     0] SEED  S_k0={m0['S_k0']:.4f} S_low={m0['S_low_k2']:.4f} "
          f"a={m0['S_v_alpha_low']:+.2f} angstd={m0['_angstd']:.2f} "
          f"Phi22={m0['phi22']:.4f} dih={m0['dihedral_entropy']:.3f} "
          f"Svpeak={m0['S_v_peak']:.2f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = np.full(n_this, T_HOLD, float)         # CONSTANT-T hold
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=T_HOLD, T_final=T_HOLD, temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=100, local_shell_depth=4,
        uniformity_weight=W, uniformity_kmax=KMAX, check_lsu_every=0,
        use_jax=True, use_jaxopt=False, verbose=False)
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    _, m = measure(pos, edges, f"ck{done//1000}k")
    if m is None:
        print(f"[wh ck={done:6d}] T={T_HOLD:.4f} acc={acc:.1%} skipped", flush=True)
        continue
    rd = m["ring_distribution"]; E = m["E"]
    def fr(n): return 100 * rd.get(n, 0) / E
    rec = dict(iter=done, T=T_HOLD, S_k0=m["S_k0"], S_low=m["S_low_k2"],
               alpha=m["S_v_alpha_low"], angstd=m["_angstd"], phi22=m["phi22"],
               dih=m["dihedral_entropy"], svpeak=m["S_v_peak"],
               r6=fr(6), r7=fr(7), r8=fr(8), r9=fr(9),
               ring_mean=m["ring_mean"], girth=min(rd) if rd else 0,
               bondstd=m["bond_len_std"], min_nb=m["min_nb"], acc=acc)
    traj.append(rec)
    gate = "PASS" if (rec['angstd'] <= 9.0 and rec['phi22'] >= 0.88) else "."
    print(f"[wh ck={done:6d}] T={rec['T']:.3f} | angstd={rec['angstd']:.2f} "
          f"Phi22={rec['phi22']:.4f} [{gate}] | 6r={rec['r6']:.1f} 7r={rec['r7']:.1f} "
          f"8r={rec['r8']:.1f} 9r={rec['r9']:.1f} mean={rec['ring_mean']:.2f} | "
          f"S_k0={rec['S_k0']:.3f} a={rec['alpha']:+.2f} dih={rec['dih']:.3f} "
          f"Svpk={rec['svpeak']:.2f} bstd={rec['bondstd']:.3f} acc={acc:.1%} "
          f"t={time.time()-t0:.0f}s", flush=True)
    print("WH_JSON:", json.dumps(rec), flush=True)

print(f"\n=== REF: angstd 8.41 Phi22 0.889 | 6r 7.6 7r 10 8r 59.7 9r 20.9 | "
      f"S_k0 0.041 a +1.51 dih 0.796 ===", flush=True)
print("WH_TRAJ:", json.dumps(traj), flush=True)
print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
