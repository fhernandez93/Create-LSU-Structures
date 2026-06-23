"""Decisive test of the advisor's hypothesis 4: are defect-healing bond-switch moves
WRONGLY REJECTED because the anneal's LOCAL depth-4 masked relax under-evaluates their
true (full-relaxation) downhill dE?

For a trapped state (coldDis ck50k, 8r 38, E/atom 0.062), propose many valid SW moves;
for each compute dE under (i) the anneal's local masked relax (depth `SHELL`, `LOCAL` iters)
vs (ii) a full-N deep relax. If many moves are 'uphill local / downhill full', the
acceptance is biased against healing moves => fuller relaxation is the unlock. If
dE_local ~= dE_full, relaxation is fine and the trap is a genuine search/barrier problem.

CPU: CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python -m Claude_Helpers._relax_bias_test [path] [nmoves]
"""
import sys, os, numpy as np
import tools, lsu_network as lsu

box = np.array([11.44] * 3); D0 = 0.8; W = (0.7, 0.7, 0.3, 0.4); N = 1000
PATH = sys.argv[1] if len(sys.argv) > 1 else "Structures/20260622_coldDis_ck50k.txt"
NMOVES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SHELL = int(os.environ.get("SHELL_DEPTH", "4"))
LOCAL = int(os.environ.get("RELAX_LOCAL", "100"))
FULL = int(os.environ.get("RELAX_FULL", "1500"))

rng = np.random.default_rng(0)
rods = np.loadtxt(PATH); pos, edges = tools.rods_to_network(rods, box)
edges = edges.copy(); neighbors = lsu.build_neighbors(N, edges)
ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
ctx.update_topology(edges, neighbors)
# baseline: full relax of the trapped state
pos, _, _ = lsu.relax(pos, ctx, max_iter=FULL)
E0 = float(ctx.energy(pos.ravel()))
print(f"=== RELAX-BIAS TEST  {PATH}  shell={SHELL} local={LOCAL} full={FULL} ===", flush=True)
print(f"baseline trapped E/atom={E0/N:.5f} (ref 0.0345)", flush=True)

rows = []
tries = 0
while len(rows) < NMOVES and tries < NMOVES * 20:
    tries += 1
    move = lsu.stone_wales_propose(edges, neighbors, rng, max_tries=30)
    if move is None:
        continue
    _ek1, (i, c, j, d), _ek2 = move
    pos_before = pos.copy()
    lsu.stone_wales_apply(edges, neighbors, move)
    if not lsu.is_connected(N, edges):
        lsu.stone_wales_revert(edges, neighbors, move); continue
    ctx.update_topology(edges, neighbors)
    # (i) LOCAL masked relax (as the anneal does), no threshold -> converged-ish local E
    seed = np.array([i, c, j, d], dtype=np.int64)
    ctx.set_moving_mask(lsu.compute_local_shell_mask(seed, neighbors, SHELL, N))
    p_loc, E_loc, _ = lsu.relax(pos_before.copy(), ctx, max_iter=LOCAL, E_threshold=float("inf"))
    dE_local = float(E_loc) - E0
    # (ii) FULL-N relax from the same post-move positions
    ctx.set_moving_mask(None)
    p_full, E_full, _ = lsu.relax(p_loc.copy(), ctx, max_iter=FULL, E_threshold=float("inf"))
    dE_full = float(E_full) - E0
    rows.append((dE_local, dE_full))
    # revert
    lsu.stone_wales_revert(edges, neighbors, move)
    ctx.update_topology(edges, neighbors)
    pos = pos_before

rows = np.array(rows)
dl, df = rows[:, 0], rows[:, 1]
print(f"\n{len(rows)} valid SW moves on the trapped state:", flush=True)
print(f"  dE_local  : mean {dl.mean():+.4f}  min {dl.min():+.4f}  (>0 i.e. looks-uphill: {(dl>0).sum()}/{len(dl)})", flush=True)
print(f"  dE_full   : mean {df.mean():+.4f}  min {df.min():+.4f}  (<0 i.e. truly-downhill: {(df<0).sum()}/{len(df)})", flush=True)
print(f"  full reveals MORE downhill than local (df<dl): {(df < dl - 1e-6).sum()}/{len(df)}", flush=True)
wrong = ((dl > 0) & (df < 0)).sum()
print(f"  *** WRONGLY-LOOKING moves (dE_local>0 but dE_full<0): {wrong}/{len(df)} ***", flush=True)
print(f"  mean gap (dE_local - dE_full): {(dl - df).mean():+.4f}  max gap {(dl-df).max():+.4f}", flush=True)
best_full = df.min()
print(f"\nBest single-move dE_full = {best_full:+.4f} (E/atom step {best_full/N:+.6f}); "
      f"a healthy anneal needs many such downhill moves accessible.", flush=True)
if wrong >= max(2, len(df)//10):
    print("VERDICT: relaxation bias PRESENT -> local masked relax hides downhill moves "
          "=> fuller relaxation / global-fallback is a real unlock.", flush=True)
else:
    print("VERDICT: relaxation bias WEAK -> local relax ~ full; trap is a search/barrier "
          "problem (needs hotter/slower schedule or more moves), not relaxation.", flush=True)
