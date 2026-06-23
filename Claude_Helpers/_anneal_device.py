"""On-device WWW anneal: the per-move relax runs entirely on the GPU in ONE jitted
call (Barzilai-Borwein masked GD, parity-validated vs scipy L-BFGS), and POSITIONS
STAY ON DEVICE across moves. This collapses the ~189 host<->device round-trips/move
(the profiled bottleneck) to a few small topology transfers.

Faithfulness vs lsu.www_anneal: same Stone-Wales move, same Keating energy, same
depth-`local_shell_depth` local relax (gradient-masked), same Metropolis acceptance.
DROPS the Vink early-reject shortcut -> does a FULL local relax every move then plain
Metropolis (still exactly Metropolis, just without the equivalent early-abort speedup).

Use this ONLY after integration-parity vs www_anneal is confirmed (gate-metric
distributions match within run-to-run noise).
"""
from functools import partial
import math
import numpy as np
import jax, jax.numpy as jnp
import lsu_network as lsu

vg = lsu._value_and_grad_jit


@partial(jax.jit, static_argnames=("n_iters",))
def _relax_bb(x0, edges, triples, quads, box, d0, w, mask, n_iters, alpha0):
    """Masked Barzilai-Borwein GD relax, fully on-device. Returns (x, E)."""
    _, g0 = vg(x0, edges, triples, quads, box, d0, w); g0 = g0 * mask
    x1 = x0 - alpha0 * g0

    def body(i, c):
        xp, gp, x = c
        _, g = vg(x, edges, triples, quads, box, d0, w); g = g * mask
        s = x - xp; y = g - gp
        sy = jnp.sum(s * y); yy = jnp.sum(y * y); ss = jnp.sum(s * s)
        alpha = jnp.where((sy > 1e-30) & (yy > 1e-30), ss / jnp.maximum(sy, 1e-30), alpha0)
        alpha = jnp.clip(alpha, 1e-7, 0.5)
        return (x, g, x - alpha * g)

    xp, gp, x = jax.lax.fori_loop(0, n_iters, body, (x0, g0, x1))
    e, _ = vg(x, edges, triples, quads, box, d0, w)
    return x, e


def www_anneal_device(pos, edges, neighbors, box, d0, weights, n_iterations,
                      temperatures, rng, relax_iters=150, bb_alpha0=1e-3,
                      local_shell_depth=4, uniformity_weight=0.0, uniformity_kmax=2):
    N = pos.shape[0]
    ctx = lsu._RelaxContext(N, box, d0, weights, use_jax=True, use_jaxopt=False)
    ctx.update_topology(edges, neighbors)
    box_j, d0_j, w_j = ctx._box_j, ctx._d0_j, ctx._w_j

    x = jnp.asarray(pos.ravel())                       # positions live ON DEVICE
    # current energy + (optional) uniformity term
    E_curr = float(vg(x, ctx._edges_j, ctx._triples_j, ctx._quads_j, box_j, d0_j, w_j)[0])
    if uniformity_weight > 0:
        S_curr = float(lsu.low_k_structure_factor(np.asarray(x).reshape(N, 3), box, kmax=uniformity_kmax))
    else:
        S_curr = 0.0
    obj_curr = E_curr + uniformity_weight * S_curr
    accepted = 0; proposed = 0

    for it in range(n_iterations):
        T = float(temperatures[it])
        move = lsu.stone_wales_propose(edges, neighbors, rng, max_tries=30)
        if move is None:
            continue
        proposed += 1
        _ek1, (si, sc, sj, sd), _ek2 = move
        x_before = x                                   # immutable device array
        edges_before = edges.copy()
        lsu.stone_wales_apply(edges, neighbors, move)
        if not lsu.is_connected(N, edges):
            lsu.stone_wales_revert(edges, neighbors, move); edges = edges_before
            continue
        ctx.update_topology(edges, neighbors)          # rebuild triples/quads + push to device
        shell = lsu.compute_local_shell_mask(np.array([si, sc, sj, sd], dtype=np.int64),
                                             neighbors, local_shell_depth, N)
        ctx.set_moving_mask(shell)
        x_new, E_new = _relax_bb(x_before, ctx._edges_j, ctx._triples_j, ctx._quads_j,
                                 box_j, d0_j, w_j, ctx._mask_flat_j, relax_iters, bb_alpha0)
        E_new = float(E_new)
        if uniformity_weight > 0:
            S_new = float(lsu.low_k_structure_factor(np.asarray(x_new).reshape(N, 3), box, kmax=uniformity_kmax))
        else:
            S_new = 0.0
        obj_new = E_new + uniformity_weight * S_new
        dE = obj_new - obj_curr
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
            x = x_new; E_curr = E_new; S_curr = S_new; obj_curr = obj_new
            accepted += 1
        else:
            lsu.stone_wales_revert(edges, neighbors, move)
            ctx.update_topology(edges, neighbors)      # restore topology arrays
            x = x_before

    pos_out = np.asarray(x).reshape(N, 3)
    pos_out = pos_out - box * np.round(pos_out / box)
    return pos_out, edges, neighbors, {"accepted": accepted, "proposed": proposed}
