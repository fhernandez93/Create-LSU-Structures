"""Profile where per-move WWW time goes (GPU), to target the right optimization for
the genuinely-local relax. Runs a short www_anneal under cProfile."""
import cProfile, pstats, io, time
import numpy as np
import lsu_network as lsu

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float); W = (0.7, 0.7, 0.3, 0.4)
NMOVES = 2000

rng = np.random.default_rng(42)
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
ctx0.update_topology(edges, neighbors)
pos, _ = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)
print(f"seed ready. profiling {NMOVES} moves at T=0.05 w=0 ...", flush=True)

# warm up JIT
lsu.www_anneal(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
               n_iterations=50, T0=0.05, T_final=0.05,
               temperatures=np.full(50, 0.05), rng=np.random.default_rng(1),
               relax_local_iters=100, local_shell_depth=4, uniformity_weight=0.0,
               check_lsu_every=0, use_jax=True, verbose=False)

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
lsu.www_anneal(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
               n_iterations=NMOVES, T0=0.05, T_final=0.05,
               temperatures=np.full(NMOVES, 0.05), rng=np.random.default_rng(2),
               relax_local_iters=100, local_shell_depth=4, uniformity_weight=0.0,
               check_lsu_every=0, use_jax=True, verbose=False)
pr.disable()
elapsed = time.time() - t0
print(f"\n{NMOVES} moves in {elapsed:.1f}s = {1000*elapsed/NMOVES:.2f} ms/move\n", flush=True)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(28)
print(s.getvalue())
# also by tottime (self time)
s2 = io.StringIO()
pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(20)
print("=== BY SELF-TIME (tottime) ===")
print(s2.getvalue())
