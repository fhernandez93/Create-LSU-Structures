"""One-call reproduction of the from-RANDOM-seed amorphous LSU network.

The recipe (validated 2026-06-23; see Example/20260623_lsu_fromrandom_N1000_README.md):
  1. random_bm2000 seed (random/liquid start)
  2. extended slow-cool pure WWW (w=0, Keating, ~250 moves/atom) -> clears the
     local-order plateau (Phi22->0.88, bond-angle std->ref level)
  3. sustained hold at the ordering T=0.04 -> settles angles + holds 8-rings
  4. Stage-B: free fixed-topology low-k optimisation -> restores hyperuniformity

Returns the rod-endpoint array. `fast=True` uses the on-device anneal (BB relax,
~3.4x faster). CAUTION: fast=True is integration-parity-gated to N=1000 + this
schedule (validated via _run_meltquench_device, seed 42). It is NOT validated for
N=4000 or other schedules -- keep fast=False (the default) there until re-checked.
"""
import math
import numpy as np
import lsu_network as lsu


def _schedule(t_hot, t_cold, n):
    g = np.arange(n)
    return t_hot * np.exp(math.log(t_cold / t_hot) * g / max(1, n - 1))


def _anneal(pos, edges, nb, box, D0, W, temps, rng, w_uni, fast, relax_iters):
    if fast:
        from Claude_Helpers._anneal_device import www_anneal_device
        return www_anneal_device(pos, edges, nb, box, D0, W, len(temps), temps, rng,
                                 relax_iters=relax_iters, local_shell_depth=4,
                                 uniformity_weight=w_uni)
    p, e, n2, _ = lsu.www_anneal(pos, edges, nb, box, D0, W, n_iterations=len(temps),
                                 T0=temps[0], T_final=temps[-1], temperatures=temps, rng=rng,
                                 relax_local_iters=100, local_shell_depth=4,
                                 uniformity_weight=w_uni, check_lsu_every=0, use_jax=True,
                                 verbose=False)
    return p, e, n2, None


def stage_b_void_fix(pos, edges, box, D0, W, kmax=2, lam=1.0, maxiter=1000):
    """Restore hyperuniformity at FIXED topology: minimise E_Keating + lam*S_low_k."""
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    N = len(pos); nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True, use_jaxopt=False); ctx.update_topology(edges, nb)
    hkl = lsu._low_k_hkl(kmax); kvec = jnp.asarray(2.0 * math.pi * (hkl / box))
    slow_vg = jax.jit(jax.value_and_grad(
        lambda x: (jnp.abs(jnp.exp(1j * (x.reshape(N, 3) @ kvec.T)).sum(0)) ** 2).sum() / N))
    p0, _, _ = lsu.relax(pos, ctx, max_iter=1500)

    def fun(xf):
        ek, gk = ctx.value_and_grad(xf)
        vs, gs = slow_vg(jnp.asarray(xf))
        return ek + lam * float(vs), gk + lam * np.asarray(gs, np.float64)

    res = minimize(fun, p0.ravel().astype(np.float64), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    p = res.x.reshape(N, 3)
    return p - box * np.round(p / box)


def generate_from_random(N=1000, seed=42, t_hot=0.09, t_cold=0.028, n_cool=250000,
                         n_hold=50000, t_hold=0.04, stage_b=True, fast=False,
                         relax_iters=150, verbose=True):
    """Reproduce the from-random amorphous LSU network. Returns rod endpoints (M,6).

    NOTE: this is a long computation (~250-300 moves/atom of WWW annealing):
    ~hours on the scipy path, ~1/3 of that with fast=True (on-device). For a quick
    smoke set n_cool/n_hold small (the result will be under-annealed).
    """
    box = np.array([(N / 1000 * 11.44 ** 3) ** (1 / 3)] * 3, float); D0 = 0.8
    W = (0.7, 0.7, 0.3, 0.4)
    rng = np.random.default_rng(seed)
    pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx.update_topology(edges, nb)
    pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
    if verbose:
        print(f"[recipe] seed built (N={N}); slow-cool {t_hot}->{t_cold} x{n_cool} (fast={fast})...", flush=True)

    temps = _schedule(t_hot, t_cold, n_cool)
    pos, edges, nb, _ = _anneal(pos, edges, nb, box, D0, W, temps, rng, 0.0, fast, relax_iters)

    if n_hold > 0:
        if verbose: print(f"[recipe] sustained hold T={t_hold} x{n_hold}...", flush=True)
        temps_h = np.full(n_hold, t_hold)
        pos, edges, nb, _ = _anneal(pos, edges, nb, box, D0, W, temps_h, rng, 0.0, fast, relax_iters)

    if stage_b:
        if verbose: print("[recipe] Stage-B void restoration...", flush=True)
        pos = stage_b_void_fix(pos, edges, box, D0, W)

    rods = lsu.network_to_rods(pos, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    if verbose: print(f"[recipe] done: {len(rods)} rods, E={len(edges)} edges.", flush=True)
    return rods
