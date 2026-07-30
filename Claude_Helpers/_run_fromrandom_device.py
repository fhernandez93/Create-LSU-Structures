"""N-parametrized, checkpointed from-random device runner (for #1 N=4000 escalation
and reusable at N=1000). Schedule = slow-cool t_hot->t_hold over n_cool, then hold
t_hold for n_hold. On-device anneal (BB relax). Stage-B is applied separately via
_validate_fromrandom on a chosen checkpoint.

box = (N/1000)**(1/3) * 11.44 (density-matched). NOTE: fast/device path is
parity-validated at N=1000+this schedule; re-check parity at N=4000 before trusting
(run a short device-vs-scipy comparison).

Usage: python -m Claude_Helpers._run_fromrandom_device <N> <tag> <t_hot> <t_hold> <n_cool> <n_hold> [chunk] [seed]
env: DEEP_RELAX (600), RELAX_ITERS (150),
     BOX ("L" cube or "Lx,Ly,Lz" slab, e.g. "100,100,20"; overrides the
     density-matched cube — check N gives the reference density N/V ~= 0.668/um^3)
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
from Claude_Helpers._anneal_device import www_anneal_device
from Claude_Helpers._graph_rings import ring_stats_from_edges
from Claude_Helpers._metrics import s_k0

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
tag = sys.argv[2] if len(sys.argv) > 2 else f"frd{N}"
t_hot = float(sys.argv[3]) if len(sys.argv) > 3 else 0.09
t_hold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.040
n_cool = int(sys.argv[5]) if len(sys.argv) > 5 else 600000
n_hold = int(sys.argv[6]) if len(sys.argv) > 6 else 200000
chunk = int(sys.argv[7]) if len(sys.argv) > 7 else 50000
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 42
D0 = 0.8
_box_env = os.environ.get("BOX")
if _box_env:
    _v = [float(x) for x in _box_env.replace(",", " ").split()]
    if len(_v) not in (1, 3):
        raise ValueError(f"BOX env needs 1 or 3 values, got {_box_env!r}")
    box = np.array(_v * 3 if len(_v) == 1 else _v, float)
    _rho = N / float(np.prod(box))
    _rho_ref = 1000.0 / 11.44 ** 3
    if abs(_rho / _rho_ref - 1.0) > 0.02:
        print(f"WARNING: density N/V={_rho:.4f} is {100*(_rho/_rho_ref-1):+.1f}% off "
              f"the reference {_rho_ref:.4f}/um^3 the schedule is calibrated for.",
              flush=True)
else:
    box = np.array([(N / 1000.0) ** (1.0 / 3.0) * 11.44] * 3, float)
W = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")
DEEP = int(os.environ.get("DEEP_RELAX", "600"))
RELAX_ITERS = int(os.environ.get("RELAX_ITERS", "150"))

n_total = n_cool + n_hold
gc = np.arange(n_cool)
T_cool = t_hot * np.exp(math.log(t_hold / t_hot) * gc / max(1, n_cool - 1))
T_full = np.concatenate([T_cool, np.full(n_hold, t_hold)])
print(f"=== FROM-RANDOM DEVICE  N={N} box={np.round(box, 3).tolist()}  tag={tag} cool {t_hot}->{t_hold}x{n_cool} "
      f"+ hold x{n_hold} = {n_total} ({n_total/N:.0f}/atom)  seed={SEED} ===", flush=True)

import glob, re, tools
# --- AUTO-RESUME: continue from the latest saved checkpoint for this tag if any
# (crash-robust: the device anneal can segfault at the CUDA level on long runs;
# a restart-on-crash wrapper re-invokes this and we pick up where we left off) ---
resume_iter = 0
cks = glob.glob(f"Structures/*_{tag}_ck*k_edges.npy")
if cks:
    def _it(p):
        m = re.search(r'_ck(\d+)k_edges', p); return int(m.group(1)) * 1000 if m else 0
    latest = max(cks, key=_it); resume_iter = _it(latest)
    rod = latest.replace('_edges.npy', '.txt')
    rods0 = np.loadtxt(rod); pos, edges = tools.rods_to_network(rods0, box)
    assert len(pos) == N, f"resume N mismatch {len(pos)} != {N}"
    neighbors = lsu.build_neighbors(N, edges)
    rng = np.random.default_rng(SEED + resume_iter)   # fresh stream from here
    print(f"[{tag}] RESUMING from {latest} at iter {resume_iter}", flush=True)
else:
    rng = np.random.default_rng(SEED)
    pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
    neighbors = lsu.build_neighbors(N, edges)
    ctx0 = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx0.update_topology(edges, neighbors)
    pos, _ = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)


def angstd(p, e):
    nb = lsu.build_neighbors(N, e); tri = lsu.build_angle_triples(nb)
    v=p[tri[:,0]];a=p[tri[:,1]];b=p[tri[:,2]]
    da=lsu.pbc_displacement(a-v,box); db=lsu.pbc_displacement(b-v,box)
    da/=np.linalg.norm(da,axis=1,keepdims=True); db/=np.linalg.norm(db,axis=1,keepdims=True)
    return float(np.degrees(np.arccos(np.clip((da*db).sum(1),-1,1))).std())


def measure(pos, edges, label):
    nb = lsu.build_neighbors(N, edges)
    fctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); fctx.update_topology(edges, nb)
    p2,_,_ = lsu.relax(pos, fctx, max_iter=DEEP); p2 = p2 - box*np.round(p2/box)
    d,m,c,girth = ring_stats_from_edges(edges, N)
    sk0,_,_ = s_k0(p2, box)
    np.save(f"Structures/{date}_{tag}_{label}_edges.npy", edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True, clip_endpoints_to_box=False)
    np.savetxt(f"Structures/{date}_{tag}_{label}.txt", rods, fmt="%.6f", delimiter="\t")
    return dict(epa=float(fctx.energy(p2.ravel()))/N, angstd=angstd(p2,edges),
                phi22=float(lsu.compute_lsu(p2,edges,nb,box,depth=2,locality=2,max_pairs=4000,rng=np.random.default_rng(0))),
                S_k0=float(sk0), r8=d.get(8,0), r7=d.get(7,0), mean=m, girth=girth)


t0=time.time(); done=resume_iter
if resume_iter == 0:
    m0=measure(pos,edges,"ck0")
    print(f"[{tag} ck=      0] SEED 8r={m0['r8']:.1f} S_k0={m0['S_k0']:.3f} Phi22={m0['phi22']:.4f}", flush=True)
while done < n_total:
    n_this=min(chunk, n_total-done); Tslice=T_full[done:done+n_this]
    pos,edges,neighbors,hist = www_anneal_device(pos,edges,neighbors,box,D0,W,n_this,Tslice,rng,
                                                 relax_iters=RELAX_ITERS,local_shell_depth=4,uniformity_weight=0.0)
    done+=n_this; acc=hist["accepted"]/max(1,hist["proposed"])
    r=measure(pos,edges,f"ck{done//1000}k"); r.update(iter=done,T=float(Tslice[-1]),acc=acc)
    print(f"[{tag} ck={done:7d}] T={r['T']:.4f} | 8r={r['r8']:.1f} 7r={r['r7']:.1f} mean={r['mean']:.2f} girth={r['girth']} | "
          f"S_k0={r['S_k0']:.3f} Phi22={r['phi22']:.4f} angstd={r['angstd']:.2f} E/atom={r['epa']:.4f} "
          f"acc={acc:.1%} t={time.time()-t0:.0f}s", flush=True)
    print(f"{tag.upper()}_JSON:", json.dumps(r), flush=True)

print(f"=== done {time.time()-t0:.0f}s ===", flush=True)
