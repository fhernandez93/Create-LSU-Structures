"""Benchmark the on-device relax SPEED vs scipy, to confirm the approach is worth
building the full anneal loop around. Times per-call relax (warmed up)."""
import time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
import lsu_network as lsu

N = 1000; D0 = 0.8; BOX = 11.44
box = np.array([BOX]*3, float); W = (0.7, 0.7, 0.3, 0.4)
SHELL = 4
vg = lsu._value_and_grad_jit


@partial(jax.jit, static_argnames=("n_iters",))
def relax_adam(x0, edges, triples, quads, box_j, d0_j, w_j, mask, n_iters, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    def body(i, carry):
        x, m, v = carry
        e, g = vg(x, edges, triples, quads, box_j, d0_j, w_j)
        g = g * mask
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g
        t = i.astype(x.dtype)+1.0
        x = x - lr * (m/(1-b1**t)) / (jnp.sqrt(v/(1-b2**t))+eps)
        return (x, m, v)
    z = jnp.zeros_like(x0)
    x, _, _ = jax.lax.fori_loop(0, n_iters, body, (x0, z, z))
    e, _ = vg(x, edges, triples, quads, box_j, d0_j, w_j)
    return x, e


rng = np.random.default_rng(42)
pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
nb = lsu.build_neighbors(N, edges)
ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False)
ctx.update_topology(edges, nb)
pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
boxj = jnp.asarray(box); d0j = jnp.float64(D0); wj = jnp.asarray(W)
edges_j = jnp.asarray(ctx.edges, jnp.int32); tri_j = jnp.asarray(ctx.triples, jnp.int32); quad_j = jnp.asarray(ctx.quads, jnp.int32)
mask = jnp.ones((3*N,))  # full (worst case for adam cost); local mask is cheaper-equal

# a representative post-SW mask
move = lsu.stone_wales_propose(edges, nb, rng, max_tries=40)
_ek1,(i,c,j,d),_ek2 = move
shell = lsu.compute_local_shell_mask(np.array([i,c,j,d]), nb, SHELL, N)
ctx.set_moving_mask(shell)
mask_loc = jnp.asarray(np.broadcast_to(shell[:,None],(N,3)).reshape(-1).astype(np.float64))
x0 = jnp.asarray(pos.ravel())

# warmup (compile)
for ni in (300, 1000):
    x,e = relax_adam(x0, edges_j, tri_j, quad_j, boxj, d0j, wj, mask_loc, ni, 3e-3); x.block_until_ready()
p,_,_ = lsu.relax(pos, ctx, max_iter=100)

def t_adam(ni, reps=50):
    t0=time.time()
    for _ in range(reps):
        x,e=relax_adam(x0, edges_j, tri_j, quad_j, boxj, d0j, wj, mask_loc, ni, 3e-3);
    x.block_until_ready()
    return 1000*(time.time()-t0)/reps

def t_scipy(reps=50):
    t0=time.time()
    for _ in range(reps):
        p,e,_=lsu.relax(pos, ctx, max_iter=100, E_threshold=float("inf"))
    return 1000*(time.time()-t0)/reps

print(f"scipy L-BFGS(100):        {t_scipy():.2f} ms/relax", flush=True)
for ni in (300, 600, 1000, 2000):
    print(f"on-device Adam({ni:>4}):     {t_adam(ni):.2f} ms/relax", flush=True)
