"""Profile said the per-move bottleneck is host<->device TRANSFER (jax _value 38%,
device_put, asarray), NOT compute. So the fix is on-device L-BFGS (jaxopt) or CPU
(no transfers), NOT a genuinely-local relax. Time 500 moves under each config."""
import sys, time
import numpy as np
import lsu_network as lsu

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX, BOX, BOX], float); W = (0.7, 0.7, 0.3, 0.4)
NMOVES = int(sys.argv[2]) if len(sys.argv) > 2 else 500
USE_JAXOPT = sys.argv[1] == "jaxopt" if len(sys.argv) > 1 else False
label = sys.argv[1] if len(sys.argv) > 1 else "jax"

rng = np.random.default_rng(42)
pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
neighbors = lsu.build_neighbors(N, edges)
ctx0 = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=USE_JAXOPT)
ctx0.update_topology(edges, neighbors)
pos, _ = lsu.settle_seed_with_repulsion(pos, ctx0, edges, box, D0, verbose=False)

# warm up JIT
lsu.www_anneal(pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
               n_iterations=30, T0=0.05, T_final=0.05, temperatures=np.full(30, 0.05),
               rng=np.random.default_rng(1), relax_local_iters=100, local_shell_depth=4,
               uniformity_weight=0.0, check_lsu_every=0, use_jax=True, use_jaxopt=USE_JAXOPT,
               verbose=False)

t0 = time.time()
p, e, nb, hist = lsu.www_anneal(
    pos.copy(), edges.copy(), lsu.build_neighbors(N, edges), box, D0, W,
    n_iterations=NMOVES, T0=0.05, T_final=0.05, temperatures=np.full(NMOVES, 0.05),
    rng=np.random.default_rng(2), relax_local_iters=100, local_shell_depth=4,
    uniformity_weight=0.0, check_lsu_every=0, use_jax=True, use_jaxopt=USE_JAXOPT,
    verbose=False)
el = time.time() - t0
acc = hist["accepted"] / max(1, hist["proposed"])
print(f"CONFIG={label} jaxopt={USE_JAXOPT}: {NMOVES} moves in {el:.1f}s = "
      f"{1000*el/NMOVES:.2f} ms/move  acc={acc:.1%}", flush=True)
