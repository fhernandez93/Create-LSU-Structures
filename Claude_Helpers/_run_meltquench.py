"""CRACK THE KINETIC TRAP: proper MELT-QUENCH from random_bm2000, w=0, energy objective.

Diagnostic (_energy_compare.py) proved the reference (Keating E/atom 0.0345, 8r 60) is ~1.8x LOWER
energy than my disorder plateau (0.062, 8r 38) -> the energy WANTS the reference; my anneal was just
kinetically trapped. The literature (Barkema-Mousseau WWW) reaches well-relaxed CRNs from LIQUID-LIKE
starts with a proper schedule + many moves/atom. My prior runs were constant holds (0.045 frozen / 0.10
liquid) or a modest cool from 0.06 -- never a true melt-quench cooled SLOWLY through the ordering window
(Tc ~0.06-0.09, bracketed by warmDis=liquid / coldDis=frozen).

This: random_bm2000 seed (the literal random/liquid start) + w=0 (crack TOPOLOGY only; S(k) is restorable
for free post-hoc per Finding 1) + a melt-quench schedule (hot liquid -> slow cool through the window ->
cold). OBJECTIVE METRIC = Keating E/atom (scalar, monotonic; success = drop 0.062 -> toward 0.0345).

Usage: python -m Claude_Helpers._run_meltquench <tag> <T_hot> <T_cold> [n_total] [chunk] [seed]
env: HU_W (0), DEEP_RELAX (600), RELAX_LOCAL (100), CF (0.5), SHELL_DEPTH (4), HOLD_FRAC (0.0 = pure
     geometric cool; >0 = fraction of iters spent isothermal just-below-Tc at T_HOLDISO before cooling)
     T_HOLDISO (0.06)
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

tag = sys.argv[1] if len(sys.argv) > 1 else "mq"
T_HOT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12
T_COLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
n_total = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 42
W = float(os.environ.get("HU_W", "0"))
DEEP = int(os.environ.get("DEEP_RELAX", "600"))
RELAX_LOCAL = int(os.environ.get("RELAX_LOCAL", "100"))
CF = float(os.environ.get("CF", "0.5"))
SHELL = int(os.environ.get("SHELL_DEPTH", "4"))
HOLD_FRAC = float(os.environ.get("HOLD_FRAC", "0.0"))
T_HOLDISO = float(os.environ.get("T_HOLDISO", "0.06"))
REF_EPA = 0.0345

print(f"=== MELT-QUENCH  tag={tag} seed=random_bm2000 T {T_HOT}->{T_COLD} hold_frac={HOLD_FRAC}"
      f"@{T_HOLDISO} n={n_total} w={W} relaxlocal={RELAX_LOCAL} cf={CF} shell={SHELL} "
      f"seed={SEED} KEATING={lsu._KEATING_F1F2} ===", flush=True)

rng = np.random.default_rng(SEED)
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, srep = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"[mq] random_bm2000 seed built+settled: E={len(edges)} min_nb={srep['min_nb_after']/D0:.3f}d0",
      flush=True)

# --- melt-quench temperature schedule ---
# optional isothermal sit just-below-Tc for HOLD_FRAC of the run, then geometric cool to T_COLD.
g = np.arange(n_total)
if HOLD_FRAC > 0.0:
    n_hold = int(HOLD_FRAC * n_total)
    n_cool = n_total - n_hold
    gc = np.arange(n_cool)
    T_cool = T_HOLDISO * np.exp(math.log(T_COLD / T_HOLDISO) * gc / max(1, n_cool - 1))
    T_full = np.concatenate([np.full(n_hold, T_HOLDISO), T_cool])
else:
    T_full = T_HOT * np.exp(math.log(T_COLD / T_HOT) * g / max(1, n_total - 1))


def angle_std(p, edges):
    nb = lsu.build_neighbors(N, edges); tri = lsu.build_angle_triples(nb)
    v = p[tri[:, 0]]; a = p[tri[:, 1]]; b = p[tri[:, 2]]
    da = lsu.pbc_displacement(a - v, box); db = lsu.pbc_displacement(b - v, box)
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((da * db).sum(1), -1, 1)))
    return float(ang.std())


def measure(pos, edges, label):
    nb = lsu.build_neighbors(N, edges)
    fctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=True, use_jaxopt=False)
    fctx.update_topology(edges, nb)
    p2, _, _ = lsu.relax(pos, fctx, max_iter=DEEP)
    p2 = p2 - box * np.round(p2 / box)
    epa = float(fctx.energy(p2.ravel())) / N        # *** objective: Keating E/atom ***
    astd = angle_std(p2, edges)
    phi22 = float(lsu.compute_lsu(p2, edges, nb, box, depth=2, locality=2))
    sk0, _, _ = s_k0(p2, box)
    np.save(f"Structures/{date}_{tag}_{label}_edges.npy", edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    out = f"Structures/{date}_{tag}_{label}.txt"
    np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
    rec = dict(epa=epa, angstd=astd, phi22=phi22, S_k0=float(sk0))
    try:
        m, cr = full_metrics_safe(out, box=BOX, d0=D0, label=f"{tag}_{label}")
        rd = m["ring_distribution"]; E = m["E"]
        rec.update(r6=100*rd.get(6,0)/E, r7=100*rd.get(7,0)/E, r8=100*rd.get(8,0)/E,
                   r9=100*rd.get(9,0)/E, ring_mean=m["ring_mean"], svpeak=m["S_v_peak"])
    except RuntimeError:
        for k in ("r6","r7","r8","r9","ring_mean","svpeak"): rec[k] = float("nan")
    return rec


t0 = time.time(); traj = []; done = 0
m0 = measure(pos, edges, "ck0")
print(f"[mq ck=     0] SEED E/atom={m0['epa']:.5f} (ref {REF_EPA}) Phi22={m0['phi22']:.4f} "
      f"angstd={m0['angstd']:.2f} 8r={m0.get('r8',float('nan')):.1f}", flush=True)
while done < n_total:
    n_this = min(chunk, n_total - done)
    Tslice = T_full[done:done + n_this]
    pos, edges, neighbors, hist = lsu.www_anneal(
        pos, edges, neighbors, box, D0, WEIGHTS,
        n_iterations=n_this, T0=Tslice[0], T_final=Tslice[-1], temperatures=Tslice,
        rng=rng, target_lsu=None, relax_local_iters=RELAX_LOCAL, local_shell_depth=SHELL,
        uniformity_weight=W, uniformity_kmax=2, check_lsu_every=0, c_f=CF,
        use_jax=True, use_jaxopt=False, verbose=False)
    done += n_this
    acc = hist["accepted"] / max(1, hist["proposed"])
    rec = measure(pos, edges, f"ck{done//1000}k")
    rec.update(iter=done, T=float(Tslice[-1]), acc=acc)
    traj.append(rec)
    print(f"[mq ck={done:6d}] T={rec['T']:.4f} | E/atom={rec['epa']:.5f} (ref {REF_EPA}) | "
          f"Phi22={rec['phi22']:.4f} angstd={rec['angstd']:.2f} | 8r={rec.get('r8',float('nan')):.1f} "
          f"7r={rec.get('r7',float('nan')):.1f} mean={rec.get('ring_mean',float('nan')):.2f} "
          f"S_k0={rec['S_k0']:.3f} acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print("MQ_JSON:", json.dumps({k:(None if isinstance(v,float) and math.isnan(v) else v)
                                  for k,v in rec.items()}), flush=True)

best = min(traj, key=lambda r: r["epa"]) if traj else m0
print(f"\n=== BEST E/atom={best['epa']:.5f} (ref {REF_EPA}) @ Phi22={best.get('phi22'):.4f} "
      f"angstd={best.get('angstd'):.2f} 8r={best.get('r8')} ===", flush=True)
print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
