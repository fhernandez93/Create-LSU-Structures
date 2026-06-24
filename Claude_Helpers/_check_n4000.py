"""Focused N=4000 check: per-move cost (device vs scipy) + E/atom parity, building
the seed ONCE and printing as it goes (so a timeout still shows partial results).
Informs the #1 cost decision + whether fast=True is parity-valid at N=4000."""
import time
import numpy as np
import lsu_network as lsu
from Claude_Helpers._anneal_device import www_anneal_device

N = 4000; D0 = 0.8; BOX = (N/1000)**(1/3)*11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
T = 0.05; NDEV = 400; NSCI = 200

t0 = time.time(); rng = np.random.default_rng(42)
pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
nb = lsu.build_neighbors(N, edges)
ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx.update_topology(edges, nb)
pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
print(f"seed+settle: {time.time()-t0:.1f}s  (N={N}, box={BOX:.2f})", flush=True)


def epa(p, e):
    n2 = lsu.build_neighbors(N, e); c = lsu._RelaxContext(N, box, D0, W, use_jax=True); c.update_topology(e, n2)
    pr, _, _ = lsu.relax(p, c, max_iter=400)
    return float(c.energy(pr.ravel()))/N


# device warmup (compile) then timed
print("device warmup (compile)...", flush=True)
pw, ew, nw, _ = www_anneal_device(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
                                  30, np.full(30, T), np.random.default_rng(9), relax_iters=150)
print("device warmup done; timing...", flush=True)
t1 = time.time()
pd, ed, nbd, hd = www_anneal_device(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
                                   NDEV, np.full(NDEV, T), np.random.default_rng(2), relax_iters=150)
t_dev = (time.time()-t1)/NDEV*1000
print(f"DEVICE: {t_dev:.1f} ms/move  acc={hd['accepted']/max(1,hd['proposed']):.1%}  E/atom={epa(pd,ed):.4f}", flush=True)

t1 = time.time()
ps, es, nbs, hs = lsu.www_anneal(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
                                 n_iterations=NSCI, T0=T, T_final=T, temperatures=np.full(NSCI, T),
                                 rng=np.random.default_rng(2), relax_local_iters=100, local_shell_depth=4,
                                 uniformity_weight=0.0, check_lsu_every=0, use_jax=True, verbose=False)
t_sci = (time.time()-t1)/NSCI*1000
print(f"SCIPY : {t_sci:.1f} ms/move  acc={hs['accepted']/max(1,hs['proposed']):.1%}  E/atom={epa(ps,es):.4f}", flush=True)
print(f"\nspeedup={t_sci/t_dev:.1f}x  | full-run estimate at 300/atom = {300*N} moves @ device "
      f"{t_dev:.0f}ms = {300*N*t_dev/1000/3600:.1f}h", flush=True)
