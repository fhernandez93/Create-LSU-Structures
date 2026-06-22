"""Near-hyperuniform vertex PLACEMENT for the LSU seed (Hejna-2013 route Sellers
cites). The decisive 2026-06-22 finding: S(k0)=0.041 cannot be created by the WWW
anneal (even corrected-Keating) from a disordered seed — it must be SUPPLIED by
the seed's vertex placement, which the Keating energy then HOLDS.

This generates points by COLLECTIVE-COORDINATE optimization: minimize the low-k
structure factor S(k) for |k| below a cutoff (a "stealthy"/near-hyperuniform
objective) plus a soft hard-core penalty so the points stay ~d0 apart (so the
existing seed topology builder can connect them). Returns positions in the
canonical box [-L/2, L/2]^3.

Usage (standalone proof): python -m Claude_Helpers._hyperuniform_seed [N] [kmax] [seed]
"""
import sys
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
import lsu_network as lsu


def _low_k_modes(box, kmax):
    """Reciprocal vectors k = 2*pi*hkl/L for integer shells up to kmax (the same
    shells low_k_structure_factor penalises, extended to suppress a broad low-k
    band)."""
    hkl = lsu._low_k_hkl(kmax)              # (M,3) integer triples, excludes 0
    return 2.0 * np.pi * (hkl / np.asarray(box))   # (M,3)


def hyperuniform_points(N, box, d0, rng, kmax=3, r_floor_frac=0.85,
                        w_rep=20.0, maxiter=1500, verbose=True):
    box = np.asarray(box, float)
    L = float(box[0])
    kvec = jnp.asarray(_low_k_modes(box, kmax))     # (M,3)
    r_floor = r_floor_frac * d0

    # start from a hard-core (Poisson-disk) field so we begin spaced
    pos0 = lsu._poisson_disk_pbc(N, box, 0.9 * d0, rng, max_tries=max(50_000, 200 * N))
    if pos0 is None:
        pos0 = (rng.random((N, 3)) - 0.5) * box
    boxj = jnp.asarray(box)

    def energy(xf):
        p = xf.reshape(N, 3)
        # --- low-k structure factor: S(k) = |sum_j e^{i k.r_j}|^2 / N, summed ---
        phases = p @ kvec.T                          # (N,M)
        rho = jnp.exp(1j * phases).sum(axis=0)       # (M,)
        Sk = (jnp.abs(rho) ** 2).sum() / N           # want small (-> hyperuniform)
        # --- soft hard-core so points stay >~ r_floor apart (PBC) ---
        diff = p[:, None, :] - p[None, :, :]
        diff = diff - boxj * jnp.round(diff / boxj)
        r2 = (diff ** 2).sum(-1) + jnp.eye(N) * 1e6
        r = jnp.sqrt(r2)
        pen = (jnp.clip(r_floor - r, 0.0, None) ** 2).sum() * 0.5
        return Sk + w_rep * pen

    vg = jax.jit(jax.value_and_grad(energy))

    def f(xf):
        v, g = vg(jnp.asarray(xf))
        return float(v), np.asarray(g, float)

    res = minimize(f, np.asarray(pos0, float).ravel(), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    pos = res.x.reshape(N, 3)
    pos = pos - box * np.round(pos / box)            # wrap to canonical box

    if verbose:
        sk0 = lsu.low_k_structure_factor(pos, box, kmax=1)
        sk2 = lsu.low_k_structure_factor(pos, box, kmax=2)
        mn = _min_sep(pos, box) / d0
        print(f"[hyperuniform] N={N} kmax={kmax}: S_k0={sk0:.4f} S_low_k2={sk2:.4f} "
              f"min_sep={mn:.3f} d0 (converged={res.success}, U={res.fun:.3f})", flush=True)
    return pos


def _min_sep(pos, box):
    from scipy.spatial import cKDTree
    L = float(box[0])
    w = (pos - L * np.floor(pos / L + 0.5)) + L / 2
    tree = cKDTree(w, boxsize=L)
    d, _ = tree.query(w, k=2)
    return float(d[:, 1].min())


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    D0 = 0.8
    BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)
    box = np.array([BOX, BOX, BOX], float)
    rng = np.random.default_rng(seed)
    # baseline poisson-disk S(k0) for comparison
    pd = lsu._poisson_disk_pbc(N, box, 0.98 * D0, rng, max_tries=max(50_000, 200 * N))
    print(f"baseline Poisson-disk: S_k0={lsu.low_k_structure_factor(pd, box, kmax=1):.4f} "
          f"min_sep={_min_sep(pd, box)/D0:.3f} d0", flush=True)
    for km in (2, 3, 4):
        hyperuniform_points(N, box, D0, np.random.default_rng(seed), kmax=km,
                            maxiter=1200, verbose=True)
