"""
De-risk the coherent-transmission n_eff estimator (agent-2 Eq.9) BEFORE wiring into the notebook.

Estimator: illuminate a slab (plane wave along +z), solve coupled-dipole (DDA), take the
transverse-averaged co-polarised dipole moment <P_x(z)>.  The COHERENT wave obeys
   <P_x(z)> = A exp(+i k_eff z) + B exp(-i k_eff z)         (forward + Fabry-Perot reflected)
Fit complex k_eff (inner linear solve for A,B at each k):
   Re(k_eff)=k0 n_eff ;  Im(k_eff)=1/(2 ell_s).

VALIDATION ANCHOR: a UNIFORM dielectric slab of index n must return n_eff = n (Im~0).
"""
import sys
sys.path.insert(0, "/home/francisco/Documents/Create LSU Structures  - Claude")
import numpy as np
import scattering_g as sg
from scipy.optimize import minimize


def fit_two_wave_keff(z, Pz, k_guess, zfit=None):
    """Fit <P(z)> = A e^{ikz}+B e^{-ikz} for complex k. Inner linear A,B; outer 2-param search on k."""
    z = np.asarray(z, float); Pz = np.asarray(Pz, complex)
    if zfit is not None:
        m = (z >= zfit[0]) & (z <= zfit[1]); z, Pz = z[m], Pz[m]
    def resid_for_k(kc):
        M = np.stack([np.exp(1j*kc*z), np.exp(-1j*kc*z)], 1)   # (n,2)
        coef, *_ = np.linalg.lstsq(M, Pz, rcond=None)
        return Pz - M @ coef, coef
    def cost(p):
        kc = p[0] + 1j*p[1]
        r, _ = resid_for_k(kc)
        return np.sum(np.abs(r)**2)
    res = minimize(cost, x0=[np.real(k_guess), np.imag(k_guess)], method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-16, maxiter=5000))
    kc = res.x[0] + 1j*res.x[1]
    r, coef = resid_for_k(kc)
    rel = np.sqrt(np.sum(np.abs(r)**2)/np.sum(np.abs(Pz)**2))
    return kc, coef, rel


def coherent_keff(P, occ, d, k0, n_guess, axis=2, pol=0, trim_frac=0.2, verbose=True):
    """Transverse-average co-pol dipole moment along propagation axis; fit k_eff."""
    G = occ.shape
    Pc = P[..., pol]
    # move propagation axis to last
    Pc = np.moveaxis(Pc, axis, -1); occm = np.moveaxis(occ, axis, -1)
    Nz = Pc.shape[-1]
    z = np.arange(Nz)*d
    # transverse mean over occupied sites only (per z-slice)
    num = Pc.reshape(-1, Nz)
    occflat = occm.reshape(-1, Nz)
    cnt = occflat.sum(0)
    Pz = np.where(cnt > 0, (num*occflat).sum(0)/np.maximum(cnt, 1), 0.0)
    z0, z1 = z[int(trim_frac*Nz)], z[int((1-trim_frac)*Nz)]
    kc, coef, rel = fit_two_wave_keff(z, Pz, k_guess=k0*n_guess + 0j*k0, zfit=(z0, z1))
    n_eff = np.real(kc)/k0
    ell_s = 1.0/(2*np.imag(kc)) if np.imag(kc) > 1e-9 else np.inf
    if verbose:
        print(f"    fit k_eff = {kc:.4f}  -> n_eff = {n_eff:.4f}  Im(k)={np.imag(kc):.4f} (ell_s={ell_s:.2f} um)"
              f"  |A|={abs(coef[0]):.3e} |B|={abs(coef[1]):.3e}  resid={rel:.3f}")
    return dict(k_eff=kc, n_eff=n_eff, ell_s=ell_s, z=z, Pz=Pz, A=coef[0], B=coef[1], resid=rel)


def dda_solve_verbose(occ, eps_grid, d, k, khat, epol, E0=1.0, tol=1e-4, maxiter=4000, every=40):
    """Same BiCGSTAB as sg.dda_solve but prints residual every `every` iters (flush)."""
    Gshape = occ.shape
    comps, Pshape = sg._interaction_fft(Gshape, d, k)
    coords = (np.stack(np.meshgrid(*[np.arange(s) for s in Gshape], indexing='ij'), -1)).astype(float)*d
    khat = np.asarray(khat, float); epol = np.asarray(epol, float)
    phase = k*(coords @ khat)
    Einc = np.zeros(Gshape + (3,), complex)
    for c in range(3):
        Einc[..., c] = E0*epol[c]*np.exp(1j*phase)
    Einc *= occ[..., None]
    alpha = np.zeros(Gshape, complex)
    for ev in np.unique(eps_grid[occ]):
        alpha[occ & (eps_grid == ev)] = sg.polarizability_LDR(complex(ev), k, d, khat, epol)
    inv_alpha = np.zeros(Gshape, complex); inv_alpha[occ] = 1.0/alpha[occ]
    def matvec(Pv):
        Pv = Pv*occ[..., None]
        return (inv_alpha[..., None]*Pv - sg._apply_interaction(comps, Pshape, Gshape, Pv))*occ[..., None]
    b = Einc.copy(); x = np.zeros_like(b); r = b - matvec(x); r0 = r.copy()
    dot = lambda a, c: np.sum(a*c)
    rho_prev = alpha_s = omega = 1.0
    p = np.zeros_like(b); v = np.zeros_like(b); bnorm = np.linalg.norm(b); hist = []
    import time, sys as _s; t0 = time.time()
    for it in range(maxiter):
        rho = dot(r0.conj(), r); beta = (rho/rho_prev)*(alpha_s/omega)
        p = r + beta*(p - omega*v); v = matvec(p)
        alpha_s = rho/dot(r0.conj(), v); ssv = r - alpha_s*v; t = matvec(ssv)
        omega = dot(t.conj(), ssv)/dot(t.conj(), t)
        x = x + alpha_s*p + omega*ssv; r = ssv - omega*t; rho_prev = rho
        res = np.linalg.norm(r)/bnorm; hist.append(res)
        if it % every == 0 or res < tol:
            print(f"      iter {it:4d}  res={res:.2e}  ({time.time()-t0:.0f}s)"); _s.stdout.flush()
        if res < tol:
            break
    return x, coords, alpha, Einc, occ, hist


def make_uniform_slab(eps, Lx, Ly, Lz, d):
    Gx, Gy, Gz = [int(round(L/d)) for L in (Lx, Ly, Lz)]
    occ = np.ones((Gx, Gy, Gz), bool)
    epsg = np.full((Gx, Gy, Gz), float(eps))
    return occ, epsg


if __name__ == "__main__":
    # ---------- VALIDATION: uniform slab, must recover n_eff = sqrt(eps) ----------
    lam = 2.0; k0 = 2*np.pi/lam
    for n_true in (1.3, 1.5):
        eps = n_true**2
        # resolution check |m| k0 d
        d = 0.05
        print(f"\n=== UNIFORM SLAB  n_true={n_true}  (eps={eps:.3f})  |m|k0 d = {n_true*k0*d:.3f} ===")
        occ, epsg = make_uniform_slab(eps, Lx=6.0, Ly=6.0, Lz=8.0, d=d)
        print(f"    grid {occ.shape}  ndip={occ.sum()}")
        P, coords, alpha, Einc, occ2, hist = sg.dda_solve(occ, epsg, d, k0,
                                                          khat=[0,0,1], epol=[1,0,0],
                                                          tol=1e-4, maxiter=3000)
        print(f"    DDA converged: iters={len(hist)} res={hist[-1]:.2e}")
        out = coherent_keff(P, occ, d, k0, n_guess=n_true, axis=2, pol=0, trim_frac=0.25)
        print(f"    => n_eff/n_true = {out['n_eff']/n_true:.4f}   (target 1.000)")
