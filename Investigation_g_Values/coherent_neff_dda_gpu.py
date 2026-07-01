"""
GPU (JAX) port of the coupled-dipole solve for the coherent-transmission n_eff measurement.
CPU BiCGSTAB stagnates + is ~3 s/iter at n=2.93; GPU FFTs make iterations ~50x cheaper.
Re-VALIDATE against the uniform-slab anchor (CPU gave n=1.5 -> 1.489) before trusting network numbers.

Build the Green-tensor comps in numpy (one-time, exact, complex128), cast to complex64 on GPU.
matvec & BiCGSTAB run entirely on GPU.
"""
import os, sys, time
sys.path.insert(0, "/home/francisco/Documents/Create LSU Structures  - Claude")
import numpy as np
import scattering_g as sg
import jax, jax.numpy as jnp
from functools import partial

print("jax devices:", jax.devices())
CDT = jnp.complex64


def build_gpu_operators(occ, eps_grid, d, k, khat, epol):
    """Return jitted matvec, Einc (jnp), and bookkeeping for a GPU coupled-dipole solve."""
    Gshape = occ.shape
    comps_np, Pshape = sg._interaction_fft(Gshape, d, k)        # numpy complex128, one-time
    comps = {kk: jnp.asarray(v, CDT) for kk, v in comps_np.items()}
    del comps_np
    coords = (np.stack(np.meshgrid(*[np.arange(s) for s in Gshape], indexing='ij'), -1)).astype(float)*d
    khat = np.asarray(khat, float); epol = np.asarray(epol, float)
    phase = k*(coords @ khat)
    Einc = np.zeros(Gshape + (3,), complex)
    for c in range(3):
        Einc[..., c] = epol[c]*np.exp(1j*phase)
    Einc *= occ[..., None]
    alpha = np.zeros(Gshape, complex)
    for ev in np.unique(eps_grid[occ]):
        alpha[occ & (eps_grid == ev)] = sg.polarizability_LDR(complex(ev), k, d, khat, epol)
    inv_alpha = np.zeros(Gshape, complex); inv_alpha[occ] = 1.0/alpha[occ]

    occ_j = jnp.asarray(occ.astype(np.float32))[..., None]            # (G,3broadcast)
    inv_alpha_j = jnp.asarray(inv_alpha, CDT)[..., None]
    Gx, Gy, Gz = Gshape
    cxx, cxy, cxz = comps['xx'], comps['xy'], comps['xz']
    cyy, cyz, czz = comps['yy'], comps['yz'], comps['zz']

    @jax.jit
    def apply_interaction(Pv):
        def F(field):
            f = jnp.zeros(Pshape, CDT).at[:Gx, :Gy, :Gz].set(field)
            return jnp.fft.fftn(f)
        Fx, Fy, Fz = F(Pv[..., 0]), F(Pv[..., 1]), F(Pv[..., 2])
        yx = jnp.fft.ifftn(cxx*Fx+cxy*Fy+cxz*Fz)[:Gx, :Gy, :Gz]
        yy = jnp.fft.ifftn(cxy*Fx+cyy*Fy+cyz*Fz)[:Gx, :Gy, :Gz]
        yz = jnp.fft.ifftn(cxz*Fx+cyz*Fy+czz*Fz)[:Gx, :Gy, :Gz]
        return jnp.stack([yx, yy, yz], -1)

    @jax.jit
    def matvec(Pv):
        Pv = Pv*occ_j
        return (inv_alpha_j*Pv - apply_interaction(Pv))*occ_j

    return matvec, jnp.asarray(Einc, CDT), alpha, inv_alpha


def bicgstab_gpu(matvec, b, tol=2e-4, maxiter=2000, every=100):
    x = jnp.zeros_like(b); r = b - matvec(x); r0 = r.copy()
    dot = lambda a, c: jnp.sum(a*c)
    rho_prev = alpha_s = omega = jnp.array(1.0+0j, CDT)
    p = jnp.zeros_like(b); v = jnp.zeros_like(b)
    bnorm = jnp.linalg.norm(b); t0 = time.time(); hist = []
    for it in range(maxiter):
        rho = dot(r0.conj(), r); beta = (rho/rho_prev)*(alpha_s/omega)
        p = r + beta*(p - omega*v); v = matvec(p)
        alpha_s = rho/dot(r0.conj(), v); s = r - alpha_s*v; t = matvec(s)
        omega = dot(t.conj(), s)/dot(t.conj(), t)
        x = x + alpha_s*p + omega*s; r = s - omega*t; rho_prev = rho
        res = float(jnp.linalg.norm(r)/bnorm); hist.append(res)
        if it % every == 0 or res < tol:
            print(f"      iter {it:4d}  res={res:.2e}  ({time.time()-t0:.1f}s)", flush=True)
        if res < tol or not np.isfinite(res):
            break
    return x, hist


def solve_neff(occ, eps_grid, d, lam, n_guess=1.5, tol=2e-4, maxiter=2000, trim_frac=0.25):
    k0 = 2*np.pi/lam
    mv, Einc, alpha, inv_alpha = build_gpu_operators(occ, eps_grid, d, k0, [0,0,1], [1,0,0])
    P, hist = bicgstab_gpu(mv, Einc, tol=tol, maxiter=maxiter)
    P = np.asarray(P)                     # back to CPU for the small coherent fit
    Nz = occ.shape[2]; z = np.arange(Nz)*d
    Pc = P[..., 0].reshape(-1, Nz); oc = occ.reshape(-1, Nz); cnt = oc.sum(0)
    Pz = np.where(cnt > 0, (Pc*oc).sum(0)/np.maximum(cnt, 1), 0.0)
    from neff_estimator import fit_two_wave_keff
    z0, z1 = z[int(trim_frac*Nz)], z[int((1-trim_frac)*Nz)]
    kc, coef, rel = fit_two_wave_keff(z, Pz, k_guess=k0*n_guess, zfit=(z0, z1))
    n_eff = float(np.real(kc)/k0)
    ell = 1.0/(2*np.imag(kc)) if np.imag(kc) > 1e-9 else np.inf
    print(f"    => n_eff={n_eff:.4f}  Im(k)={np.imag(kc):.4f} (ell_ub={ell:.1f} um)  resid={rel:.3f}  res_solve={hist[-1]:.1e}")
    return dict(n_eff=n_eff, k_eff=complex(kc), ell_ub=ell, resid=rel, z=z, Pz=Pz, hist=hist)


if __name__ == "__main__":
    from neff_estimator import make_uniform_slab
    lam = 2.0
    print("=== GPU re-validation: uniform slab n=1.5 (CPU ref 1.489) ===")
    occ, epsg = make_uniform_slab(2.25, Lx=6.0, Ly=6.0, Lz=8.0, d=0.05)
    print(f"  grid {occ.shape} ndip={occ.sum()}")
    solve_neff(occ, epsg, 0.05, lam, n_guess=1.5, tol=2e-4, maxiter=1500)
