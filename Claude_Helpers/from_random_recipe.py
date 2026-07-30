"""One-call reproduction of the from-RANDOM-seed amorphous LSU network.

The recipe (validated 2026-06-23; see Example/20260623_lsu_fromrandom_N1000_README.md):
  1. random_bm2000 seed (random/liquid start)
  2. extended slow-cool pure WWW (w=0, Keating, ~250 moves/atom) -> clears the
     local-order plateau (Phi22->0.88, bond-angle std->ref level)
  3. sustained hold at the ordering T=0.04 -> settles angles + holds 8-rings
  4. Stage-B: free fixed-topology low-k optimisation -> restores hyperuniformity

Returns the rod-endpoint array. `fast=True` uses the on-device anneal (BB relax,
~3.4x faster). CAUTION: fast=True is integration-parity-gated to N=1000 + this
schedule (parity-validated against the scipy path at N=1000, seed 42). It is NOT
validated for N=4000 or other schedules -- keep fast=False (the default) there
until re-checked.

CHECKPOINTING: pass `checkpoint_every > 0` (+ a `checkpoint_tag`) to save the
annealing trajectory to `checkpoint_dir` (default Structures/) every N moves as
`<date>_<tag>_ck<k>k.txt` (+ `_edges.npy`), exactly the format the
`_validate_fromrandom` / `_run_fromrandom_device` tools read. With `resume=True`
the run auto-continues from the latest checkpoint for that tag (crash-robust: the
long anneal can segfault at the CUDA level on multi-hour runs). The checkpoint
schedule is identical to the one-shot path -- checkpointing does not change the
result, only adds restartable save points.
"""
import os
import glob
import re
import math
import datetime
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


# --------------------------------------------------------------------------- #
# Checkpointing (save/resume the annealing trajectory)
# --------------------------------------------------------------------------- #
def _save_checkpoint(pos, edges, box, D0, W, tag, it, ckpt_dir, deep_relax_iters=600):
    """Deep-relax under Keating, then save <date>_<tag>_ck<k>k.txt (+ _edges.npy).

    The saved rods are physical (relaxed), so any checkpoint can be fed straight
    to `assess_statistics` / `_validate_fromrandom`. Returns the rod-file path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    N = len(pos)
    nb = lsu.build_neighbors(N, edges)
    ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx.update_topology(edges, nb)
    p2, _, _ = lsu.relax(pos, ctx, max_iter=deep_relax_iters)
    p2 = p2 - box * np.round(p2 / box)
    date = datetime.date.today().strftime("%Y%m%d")
    base = os.path.join(ckpt_dir, f"{date}_{tag}_ck{it // 1000}k")
    np.save(base + "_edges.npy", edges)
    rods = lsu.network_to_rods(p2, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    np.savetxt(base + ".txt", rods, fmt="%.6f", delimiter="\t")
    return base + ".txt"


def _rods_to_network_N(rods, box, N, radii=(0.1, 0.04, 0.02, 0.01, 0.005, 0.002)):
    """rods_to_network that retries with a tighter merge radius until it recovers
    exactly N vertices. A deep-relaxed checkpoint can leave a near-coincident
    non-bonded pair (the degree-4 round-trip) that the default 0.1 radius wrongly
    fuses into N-1 vertices -- same guard as _metrics.full_metrics_safe."""
    import tools
    last = None
    for cr in radii:
        pos, edges = tools.rods_to_network(rods, box, cluster_radius=cr)
        last = (pos, edges)
        if len(pos) == N:
            return pos, edges
    pos, edges = last
    raise ValueError(f"resume N mismatch: recovered {len(pos)} vertices (expected {N}) "
                     f"at all merge radii {radii}")


def _find_latest_checkpoint(tag, ckpt_dir, box, N):
    """Return (resume_iter, pos, edges) from the highest-iter checkpoint for `tag`,
    or None if there is none. Positions come from the rod file (collision-robust
    reconstruction); edges from rods_to_network are self-consistent with them."""
    cks = glob.glob(os.path.join(ckpt_dir, f"*_{tag}_ck*k_edges.npy"))
    if not cks:
        return None

    def _it(p):
        m = re.search(r"_ck(\d+)k_edges", p)
        return int(m.group(1)) * 1000 if m else 0

    latest = max(cks, key=_it)
    resume_iter = _it(latest)
    rod = latest.replace("_edges.npy", ".txt")
    rods0 = np.loadtxt(rod)
    pos, edges = _rods_to_network_N(rods0, box, N)
    return resume_iter, pos, edges


def generate_from_random(N=1000, seed=42, t_hot=0.09, t_cold=0.028, n_cool=250000,
                         n_hold=50000, t_hold=0.04, stage_b=True, fast=False,
                         relax_iters=150, checkpoint_every=0, checkpoint_tag=None,
                         checkpoint_dir="Structures", resume=False, verbose=True,
                         bounds=None):
    """Reproduce the from-random amorphous LSU network. Returns rod endpoints (M,6).

    Box shape (`bounds`):
      bounds=None (default) -> cubic box density-matched to the reference:
        L = (N/1000)**(1/3) * 11.44, i.e. the validated behaviour.
      bounds=(Lx, Ly, Lz)   -> orthorhombic box, e.g. (100, 100, 20) for a slab.
        Pass N=None to auto-derive N from the box volume at the reference
        density 1000/11.44^3 ~= 0.668 vertices/um^3 (rounded to even). If you
        pass both N and bounds, a density more than 2% off the reference
        raises a warning: the annealing T-schedule and d0=0.8 are calibrated
        at the reference density, so off-density runs are unvalidated.
        The whole pipeline (PBC, KD-trees, S(k) modes) is per-axis, so
        anisotropic boxes need no other changes; note the low-k penalty /
        Stage-B modes are integer (h,k,l) shells of the box, so along a short
        axis (Lz=20) the lowest mode sits at 5x larger |k| than along Lx=100.

    NOTE: this is a long computation (~250-300 moves/atom of WWW annealing):
    ~hours on the scipy path, ~1/3 of that with fast=True (on-device). For a quick
    smoke set n_cool/n_hold small (the result will be under-annealed).

    Checkpointing (optional, crash-robust):
      checkpoint_every : save the trajectory every this many WWW moves (0 = off).
      checkpoint_tag   : filename tag (required when checkpoint_every>0).
      checkpoint_dir   : directory for the .txt/.npy checkpoints (default Structures/).
      resume           : if True, auto-continue from the latest checkpoint for the tag.
    Checkpoints are written as <date>_<tag>_ck<k>k.txt (+ _edges.npy) -- the same
    format assess_statistics / _validate_fromrandom read, so you can validate any
    intermediate state (for a non-cubic box pass bounds=... to assess_statistics,
    or BOX env to the _validate_fromrandom CLI -- otherwise they assume the
    density-matched cube). Checkpointing does NOT change the final result.
    Returns: 
    - pos — vertex (node) coordinates. Shape (N, dims), which is why N = len(pos). These are the physical positions of the network's atoms/nodes in the box. _save_checkpoint deep-relaxes them
        under the Keating potential (lsu.relax), wraps them back into the box (p2 - box*round(p2/box)), and writes them out.
    - edges — the bond list / connectivity. An array of vertex-index pairs (i, j) saying which nodes are bonded. It's pure topology (indices into pos), carrying no coordinates. It's used to
        build the neighbor table (lsu.build_neighbors(N, edges)), to set the relax topology, and to convert the relaxed network into rods (lsu.network_to_rods(p2, edges, ...)). It's saved
        alongside the rod file as _edges.npy.
    - rods end to end point of the actual rods

    """
    D0 = 0.8
    RHO_REF = 1000.0 / 11.44 ** 3  # reference vertex density (per um^3)
    if bounds is None:
        if N is None:
            raise ValueError("N is required when bounds is not given")
        box = np.array([(N / 1000 * 11.44 ** 3) ** (1 / 3)] * 3, float)
    else:
        box = lsu.coerce_box(bounds)
        V = float(np.prod(box))
        if N is None:
            N = 2 * int(round(RHO_REF * V / 2.0))  # nearest even N at ref density
        else:
            rho = N / V
            if abs(rho / RHO_REF - 1.0) > 0.02:
                import warnings
                warnings.warn(
                    f"generate_from_random: density N/V={rho:.4f} deviates "
                    f"{100 * (rho / RHO_REF - 1):+.1f}% from the reference "
                    f"{RHO_REF:.4f}/um^3 the recipe (d0=0.8, T-schedule) is "
                    f"calibrated for. Pass N=None to auto-match density.",
                    stacklevel=2)
        if min(box) < 4 * D0:
            raise ValueError(
                f"bounds {box.tolist()}: the shortest axis must be >= 4*d0="
                f"{4 * D0} for the periodic minimum-image convention to make sense")
    W = (0.7, 0.7, 0.3, 0.4)
    if checkpoint_every > 0 and not checkpoint_tag:
        raise ValueError("checkpoint_every>0 requires a checkpoint_tag")

    # Move counts may arrive as floats (e.g. n_cool=275*N*1.1 in the notebook);
    # the schedule arrays need exact integer lengths.
    n_cool = int(round(n_cool)); n_hold = int(round(n_hold))
    checkpoint_every = int(round(checkpoint_every))

    # Full schedule: slow-cool t_hot->t_cold over n_cool, then hold t_hold for n_hold.
    # (Built as one array so a chunked/checkpointed run is identical to the one-shot
    # run -- www_anneal just reads temperatures[it] per move.)
    n_total = n_cool + n_hold
    T_full = np.concatenate([_schedule(t_hot, t_cold, n_cool), np.full(n_hold, t_hold)])

    # --- seed, or resume from the latest checkpoint -----------------------------
    resume_iter = 0
    resumed = None
    if resume and checkpoint_every > 0 and checkpoint_tag:
        resumed = _find_latest_checkpoint(checkpoint_tag, checkpoint_dir, box, N)
    if resumed is not None:
        resume_iter, pos, edges = resumed
        nb = lsu.build_neighbors(N, edges)
        rng = np.random.default_rng(seed + resume_iter)   # fresh stream from here
        if verbose:
            print(f"[recipe] RESUMING tag={checkpoint_tag} at iter {resume_iter}/{n_total}", flush=True)
    else:
        rng = np.random.default_rng(seed)
        pos, edges, _ = lsu.random_seed_network_bm2000(N, box, D0, rng, verbose=False)
        nb = lsu.build_neighbors(N, edges)
        ctx = lsu._RelaxContext(N, box, D0, W, use_jax=True); ctx.update_topology(edges, nb)
        pos, _ = lsu.settle_seed_with_repulsion(pos, ctx, edges, box, D0, verbose=False)
        if verbose:
            print(f"[recipe] seed built (N={N}); slow-cool {t_hot}->{t_cold} x{n_cool} "
                  f"+ hold {t_hold} x{n_hold} (fast={fast}, checkpoint_every={checkpoint_every})...",
                  flush=True)
        if checkpoint_every > 0:
            _save_checkpoint(pos, edges, box, D0, W, checkpoint_tag, 0, checkpoint_dir)

    # --- annealing loop (chunked iff checkpointing) -----------------------------
    step = checkpoint_every if checkpoint_every > 0 else n_total
    done = resume_iter
    while done < n_total:
        n_this = min(step, n_total - done)
        Tslice = T_full[done:done + n_this]
        pos, edges, nb, _ = _anneal(pos, edges, nb, box, D0, W, Tslice, rng, 0.0, fast, relax_iters)
        done += n_this
        if checkpoint_every > 0:
            path = _save_checkpoint(pos, edges, box, D0, W, checkpoint_tag, done, checkpoint_dir)
            if verbose:
                T_now = float(Tslice[-1])
                print(f"[recipe] checkpoint {done}/{n_total} (T={T_now:.4f}) -> {path}", flush=True)

    # --- Stage-B free fixed-topology void restoration ---------------------------
    if stage_b:
        if verbose: print("[recipe] Stage-B void restoration...", flush=True)
        pos = stage_b_void_fix(pos, edges, box, D0, W)

    rods = lsu.network_to_rods(pos, edges, box, pbc_duplicate_boundary_rods=True,
                               clip_endpoints_to_box=False)
    if verbose: print(f"[recipe] done: {len(rods)} rods, E={len(edges)} edges.", flush=True)
    return rods, pos, edges
