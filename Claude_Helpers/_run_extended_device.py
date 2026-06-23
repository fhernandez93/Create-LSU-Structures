"""#3 BOUNDED PROBE (on-device, fast): can FAR more annealing push 8r past ~55 and
make hyperuniformity EMERGE from pure WWW (raw S_k0 < ~0.1, no Stage-B)?

Schedule: slow-cool t_hot->t_hold over n_cool, then a LONG hold at t_hold (the
ordering sweet-spot) for n_hold. w=0 (pure WWW). Watch graph-true 8r + raw S_k0.
STOP (advisor): if ~3-5x the deliverable's moves doesn't push 8r past ~55 or drop
raw S_k0 below ~0.1, conclude + document (8r ~50 is the equilibrium; 60 is the
reference's high tail; void may genuinely need the low-k objective).

Usage: python -m Claude_Helpers._run_extended_device <tag> <t_hot> <t_hold> <n_cool> <n_hold> [chunk] [seed]
env: DEEP_RELAX (600), RELAX_ITERS (150)
"""
import sys, os, json, time, datetime, math
import numpy as np
import lsu_network as lsu
from Claude_Helpers._anneal_device import www_anneal_device
from Claude_Helpers._graph_rings import ring_stats_from_edges
from Claude_Helpers._metrics import s_k0

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
date = datetime.date.today().strftime("%Y%m%d")

tag = sys.argv[1] if len(sys.argv) > 1 else "ext"
t_hot = float(sys.argv[2]) if len(sys.argv) > 2 else 0.09
t_hold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.040
n_cool = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
n_hold = int(sys.argv[5]) if len(sys.argv) > 5 else 850000
chunk = int(sys.argv[6]) if len(sys.argv) > 6 else 50000
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 42
DEEP = int(os.environ.get("DEEP_RELAX", "600"))
RELAX_ITERS = int(os.environ.get("RELAX_ITERS", "150"))

n_total = n_cool + n_hold
gc = np.arange(n_cool)
T_cool = t_hot * np.exp(math.log(t_hold / t_hot) * gc / max(1, n_cool - 1))
T_full = np.concatenate([T_cool, np.full(n_hold, t_hold)])
print(f"=== EXTENDED DEVICE #3  tag={tag} cool {t_hot}->{t_hold} x{n_cool} + hold {t_hold} x{n_hold} "
      f"= {n_total} moves  seed={SEED} ===", flush=True)
print("TARGET: 8r>55 and/or raw S_k0<0.1 (else conclude: 8r~50 equilibrium / void needs Stage-B)", flush=True)

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
                phi22=float(lsu.compute_lsu(p2,edges,nb,box,depth=2,locality=2)),
                S_k0=float(sk0), r8=d.get(8,0), r7=d.get(7,0), mean=m, girth=girth)


t0=time.time(); done=0; best8r=0.0; bestsk0=1.0
m0=measure(pos,edges,"ck0")
print(f"[ext ck=      0] SEED 8r={m0['r8']:.1f} S_k0={m0['S_k0']:.3f} Phi22={m0['phi22']:.4f}", flush=True)
while done < n_total:
    n_this=min(chunk, n_total-done); Tslice=T_full[done:done+n_this]
    pos,edges,neighbors,hist = www_anneal_device(pos,edges,neighbors,box,D0,W,n_this,Tslice,rng,
                                                 relax_iters=RELAX_ITERS,local_shell_depth=4,uniformity_weight=0.0)
    done+=n_this; acc=hist["accepted"]/max(1,hist["proposed"])
    r=measure(pos,edges,f"ck{done//1000}k"); r.update(iter=done,T=float(Tslice[-1]),acc=acc)
    best8r=max(best8r,r['r8']); bestsk0=min(bestsk0,r['S_k0'])
    flag = "8r>55!" if r['r8']>55 else ("Sk0<0.1!" if r['S_k0']<0.1 else "")
    print(f"[ext ck={done:7d}] T={r['T']:.4f} | 8r={r['r8']:.1f} (best {best8r:.1f}) 7r={r['r7']:.1f} mean={r['mean']:.2f} "
          f"girth={r['girth']} | S_k0={r['S_k0']:.3f} (best {bestsk0:.3f}) Phi22={r['phi22']:.4f} angstd={r['angstd']:.2f} "
          f"E/atom={r['epa']:.4f} acc={acc:.1%} {flag} t={time.time()-t0:.0f}s", flush=True)
    print("EXT_JSON:", json.dumps(r), flush=True)

print(f"=== done {time.time()-t0:.0f}s  best 8r={best8r:.1f}  best raw S_k0={bestsk0:.3f} ===", flush=True)
