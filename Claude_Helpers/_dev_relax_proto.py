"""Prototype + UNIT-validate an on-device masked relax (jitted Adam in one
lax.fori_loop) against the current scipy L-BFGS relax. Profile said the per-move
bottleneck is host<->device transfer (~189 round-trips/move); an on-device loop
collapses that to ~1. Gate: same post-SW config -> same relaxed energy minimum
(under-relax prevents Phi rising, so the bar is tight), and frozen atoms stay put.

CPU or GPU. Usage: python -m Claude_Helpers._dev_relax_proto [n_test_moves]
"""
import sys, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
import lsu_network as lsu

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
NTEST = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SHELL = 4; LOCAL_ITERS = 100

vg = lsu._value_and_grad_jit  # (pos_flat, edges, triples, quads, box, d0, w)->(e,g)


@partial(jax.jit, static_argnames=("n_iters",))
def relax_adam(pos_flat, edges, triples, quads, box_j, d0_j, w_j, mask, n_iters, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8

    def body(i, carry):
        x, m, v = carry
        e, g = vg(x, edges, triples, quads, box_j, d0_j, w_j)
        g = g * mask
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        t = i.astype(x.dtype) + 1.0
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        x = x - lr * mhat / (jnp.sqrt(vhat) + eps)
        return (x, m, v)

    z = jnp.zeros_like(pos_flat)
    x, m, v = jax.lax.fori_loop(0, n_iters, body, (pos_flat, z, z))
    e, _ = vg(x, edges, triples, quads, box_j, d0_j, w_j)
    return x, e


def build():
    rng = np.random.default_rng(42)
    pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
    ctx.update_topology(edges, nb)
    pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
    return pos, edges, nb, ctx


pos, edges, nb, ctx = build()
# deep-relax to a near-minimum config: this is the regime during ANNEALING (each
# SW move is a small perturbation from a near-optimal structure), not the raw seed.
ctx.set_moving_mask(None)
pos, _, _ = lsu.relax(pos, ctx, max_iter=1500)
print(f"(starting from deep-relaxed config, E/atom={float(ctx.energy(pos.ravel()))/N:.4f})", flush=True)
rng = np.random.default_rng(1)
boxj = jnp.asarray(box); d0j = jnp.float64(D0); wj = jnp.asarray(W)

print(f"=== UNIT energy-parity: scipy L-BFGS(100) vs on-device Adam ===", flush=True)
print(f"{'move':>4} {'E_scipy':>11} {'E_adam(n,lr)':>26} {'dE':>10} {'frozen_moved':>13}", flush=True)
results = []
for k in range(NTEST):
    move = lsu.stone_wales_propose(edges, nb, rng, max_tries=40)
    if move is None:
        continue
    _ek1, (i, c, j, d), _ek2 = move
    pos_before = pos.copy()
    lsu.stone_wales_apply(edges, nb, move)
    if not lsu.is_connected(N, edges):
        lsu.stone_wales_revert(edges, nb, move); continue
    ctx.update_topology(edges, nb)
    shell = lsu.compute_local_shell_mask(np.array([i, c, j, d]), nb, SHELL, N)
    ctx.set_moving_mask(shell)
    mask_flat = jnp.asarray(np.broadcast_to(shell[:, None], (N, 3)).reshape(-1).astype(np.float64))
    edges_j = jnp.asarray(ctx.edges, jnp.int32); tri_j = jnp.asarray(ctx.triples, jnp.int32); quad_j = jnp.asarray(ctx.quads, jnp.int32)

    # scipy reference (current path)
    p_sci, E_sci, _ = lsu.relax(pos_before, ctx, max_iter=LOCAL_ITERS, E_threshold=float("inf"))

    # on-device adam: try a couple of (n,lr) configs
    best = None
    for (ni, lr) in [(300, 3e-3), (600, 3e-3), (1000, 2e-3)]:
        x0 = jnp.asarray(pos_before.ravel())
        x, E = relax_adam(x0, edges_j, tri_j, quad_j, boxj, d0j, wj, mask_flat, ni, lr)
        E = float(E)
        if best is None or E < best[0]:
            best = (E, ni, lr, np.asarray(x).reshape(N, 3))
    E_ad, ni, lr, p_ad = best
    # frozen atoms moved?
    frozen = ~shell
    fro_disp = float(np.abs(lsu.pbc_displacement(p_ad[frozen] - pos_before[frozen], box)).max()) if frozen.any() else 0.0
    print(f"{k:>4} {E_sci:>11.4f} {('%.4f (%d,%.0e)'%(E_ad,ni,lr)):>26} {E_ad-E_sci:>+10.4f} {fro_disp:>13.2e}", flush=True)
    results.append((E_sci, E_ad))
    lsu.stone_wales_revert(edges, nb, move); ctx.update_topology(edges, nb); pos = pos_before

if results:
    r = np.array(results)
    dE = r[:, 1] - r[:, 0]
    print(f"\nmean dE (adam-scipy) = {dE.mean():+.4f}  (adam HIGHER energy = under-relaxed if >0)", flush=True)
    print(f"adam reaches <= scipy on {int((dE<=1e-3).sum())}/{len(dE)} moves", flush=True)
