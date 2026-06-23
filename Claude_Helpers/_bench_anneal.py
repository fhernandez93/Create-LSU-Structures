"""End-to-end wall-time benchmark + sanity: on-device anneal (www_anneal_device,
BB relax) vs lsu.www_anneal (scipy), same seed/schedule/move-budget. Reports
ms/move and final E/atom (must be comparable; big divergence => the device path
under-relaxes and is NOT usable)."""
import sys, time
import numpy as np
import lsu_network as lsu
from Claude_Helpers._anneal_device import www_anneal_device

NMOVES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
D0 = 0.8; BOX = (N / 1000.0) ** (1.0 / 3.0) * 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)


def fresh():
    rng = np.random.default_rng(42)
    pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False); ctx.update_topology(edges, nb)
    pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
    return pos, edges, nb


def epa(pos, edges):
    nb = lsu.build_neighbors(N, edges); ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx.update_topology(edges, nb)
    p, _, _ = lsu.relax(pos, ctx, max_iter=600)
    return float(ctx.energy(p.ravel())) / N


Tarr = np.full(NMOVES, T)
# warmup device path (compile)
pos0, e0, nb0 = fresh()
print("warming up device path (compile)...", flush=True)
www_anneal_device(pos0.copy(), e0.copy(), lsu.build_neighbors(N, e0), box, D0, W,
                  200, np.full(200, T), np.random.default_rng(9), relax_iters=150)

# DEVICE
pos, edges, nb = fresh()
t0 = time.time()
pd, ed, nbd, hd = www_anneal_device(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
                                    NMOVES, Tarr, np.random.default_rng(2), relax_iters=150)
t_dev = time.time() - t0
acc_d = hd["accepted"]/max(1, hd["proposed"])

# SCIPY (lsu.www_anneal)
pos, edges, nb = fresh()
t0 = time.time()
ps, es, nbs, hs = lsu.www_anneal(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
                                 n_iterations=NMOVES, T0=T, T_final=T, temperatures=Tarr,
                                 rng=np.random.default_rng(2), relax_local_iters=100, local_shell_depth=4,
                                 uniformity_weight=0.0, check_lsu_every=0, use_jax=True, verbose=False)
t_sci = time.time() - t0
acc_s = hs["accepted"]/max(1, hs["proposed"])

print(f"\n{'path':>10} {'wall(s)':>9} {'ms/move':>9} {'acc':>7} {'E/atom':>9}", flush=True)
print(f"{'device':>10} {t_dev:>9.1f} {1000*t_dev/NMOVES:>9.2f} {acc_d:>7.1%} {epa(pd,ed):>9.4f}", flush=True)
print(f"{'scipy':>10} {t_sci:>9.1f} {1000*t_sci/NMOVES:>9.2f} {acc_s:>7.1%} {epa(ps,es):>9.4f}", flush=True)
print(f"\nSPEEDUP = {t_sci/t_dev:.1f}x   (move budget {NMOVES} @ T={T})", flush=True)
