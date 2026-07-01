"""
scattering_g.py  --  Scattering anisotropy factor g and mean free paths for the
                     amorphous LSU photonic networks, WITHOUT FDTD.

This is the engine behind ``Anisotropy_g_from_ends.ipynb``.  Three layers:

  (1) BORN / SPECTRAL-DENSITY  (primary, for g):
      The bulk single-scattering phase function of a continuous two-phase medium
      is set by the spectral density  chi_V(q) = |FFT[eps(r) - <eps>]|^2 .
      Subtracting <eps> removes the q=0 coherent (forward) term by construction,
      so what remains is exactly the diffuse scattering of the disorder.
          p(theta)  ~  chi_V(q) * F_pol(theta),   q = 2 k_av sin(theta/2)
          k_av = k0 * sqrt(eps_av),  eps_av = ff*eps_rod + (1-ff)*eps_bg   (Wiener vol. avg)
          F_pol(unpolarized) = (1 + cos^2 theta)/2     [vector-EM dipole]
          g = <cos theta> = INT p cos sin dtheta / INT p sin dtheta
      Ref: Vynck et al., Rev. Mod. Phys. 95, 045003 (2023), Eqs. 44/48/55;
           Debye-Bueche (1949); Torquato, Random Heterogeneous Materials (2002).
      In the Born limit g is INDEPENDENT of the contrast magnitude (Delta-eps
      cancels in the normalized phase function); contrast/ff enter the mean
      free paths.  g is faithful within weak single scattering; its angular
      SHAPE is robust even at high contrast.

  (2) MEAN FREE PATHS:
      - Born / weak-scattering scattering MFP (cross-check):
          1/ell_s = (k0^4 / 8pi) INT_0^pi F_pol(theta) chi_V(q(theta)) sin(theta) dtheta
        with chi_V in physical units [eps^2 * um^3].   ell_t = ell_s/(1-g).
      - Strong-contrast (Torquato/Kim) MFP  -- see strong_contrast.py / notebook;
        quantitative at high contrast where plain Born is not.

  (3) DDA (discrete-dipole / coupled-dipole), FFT-accelerated, LDR polarizability:
      A full-wave (no-FDTD) solver, VALIDATED here against analytic Mie.  It is
      used ONLY as a low-contrast end-to-end validator of the Born phase-function
      code -- it cannot give the *bulk* g at n~2.9 because a finite free-space
      cluster (n_av~1.3-1.6) produces a coherent forward "envelope" lobe that
      swamps the diffuse phase function (g -> 1 artifact) unless one does
      configurational <E> subtraction.
      Ref: Draine & Flatau, JOSA A 11, 1491 (1994); Draine & Goodman, ApJ 405, 685 (1993).

Validation anchors (run validate_*() in the notebook):
  - DDA  vs  analytic Mie (sphere)        -> kernel correct (optical theorem C_ext=C_sca)
  - Born g  vs  analytic RGD (sphere)     -> FFT/Ewald/polarization/integration correct
  - Rayleigh limit  x->0 : g -> 0

Units: lengths in micrometres (um), wavenumbers in um^-1, consistent with the
LSU box L=(N/1000)^(1/3)*11.44 um and bond length d0=0.8 um.
"""
from __future__ import annotations
import numpy as np

# =====================================================================
# 0.  Voxeliser  (verbatim from 20250903_create_h5_from_ends.ipynb)
# =====================================================================
def create_permittivity_grid_penlike(
    rod_endpoints, grid_size=128, minor_radius=0.1, aspect_ratio=None,
    aspect_ratio_hole=None, permittivity=3.42**2, permittivity_bg=1.0,
    hole_minor_radius=0.0, box_size=None, *, progress_every=20,
    dinamic_radius=False, sigma_inner=None, sigma_outer=None, mean_inner=None,
    mean_outer=None, create_hole=False, use_radius_array=False, b_list=None,
    b_h_list=None, verbose=False):
    """Rasterise rods (post-warp world coords) into a (G,G,G) eps grid.
    Returns (grid, b_array, filling_fraction).  aspect_ratio = global z warp s
    (elliptical cross-section); aspect_ratio=1 -> circular rods."""
    rods = np.asarray(rod_endpoints, dtype=np.float32)
    G = int(grid_size)
    grid = np.ones((G, G, G), dtype=np.float32) * permittivity_bg
    dx = float(box_size) / G
    grid_coords = (np.arange(G, dtype=np.float32) + 0.5) * dx - np.float32(box_size / 2.0)
    pts = rods.reshape(-1, 3); half = box_size / 2.0
    n_out = int(np.count_nonzero(np.any((pts < -half) | (pts > half), axis=1)))
    if n_out > 0 and verbose:
        print(f"[warn] {n_out}/{pts.shape[0]} rod endpoints fall outside the box.")
    if not dinamic_radius:
        b_array = np.full(len(rods), float(minor_radius), dtype=np.float32)
        b_h_array = np.full(len(rods), float(hole_minor_radius), dtype=np.float32)
    else:
        b_array = np.random.normal(mean_outer, sigma_outer, len(rods)).astype(np.float32)
        b_h_array = np.random.normal(mean_inner, sigma_inner, len(rods)).astype(np.float32)
    if use_radius_array:
        if b_list is None or len(b_list) != len(rods):
            raise ValueError("b_list must match rod_endpoints length.")
        b_array = np.asarray(b_list, dtype=np.float32)
        if b_h_list is not None:
            b_h_array = np.asarray(b_h_list, dtype=np.float32)
    s = float(aspect_ratio) if aspect_ratio is not None else 1.0
    s_in = float(aspect_ratio_hole) if aspect_ratio_hole is not None else 1.0
    k_in = s / s_in

    def idx_range_for_world(min_w, max_w, pad):
        i0 = int(np.searchsorted(grid_coords, min_w - pad, side='left'))
        i1 = int(np.searchsorted(grid_coords, max_w + pad, side='right') - 1)
        i0 = max(i0, 0); i1 = min(i1, G - 1)
        if i1 < i0:
            mid = 0.5 * (min_w + max_w)
            i0 = i1 = max(min(int(np.searchsorted(grid_coords, mid, side='left')), G - 1), 0)
        return i0, i1

    for i_rod, rod in enumerate(rods):
        if progress_every and (i_rod % progress_every == 0):
            print(f"[postwarp] rod {i_rod} / {len(rods)}")
        b = b_array[i_rod]; b_h = b_h_array[i_rod]
        use_hole = (b_h > 0.0) and (b_h < b)
        r_pad_world = b * max(1.0, s) + dx
        p1w = rod[:3].astype(np.float32); p2w = rod[3:].astype(np.float32)
        p1u = p1w.copy(); p1u[2] = p1w[2] / s
        p2u = p2w.copy(); p2u[2] = p2w[2] / s
        vu = p2u - p1u; L2u = float(np.dot(vu, vu))
        if L2u <= 0.0:
            continue
        Lu = float(np.sqrt(L2u)); nu = vu / Lu
        xmin, xmax = float(min(p1w[0], p2w[0])), float(max(p1w[0], p2w[0]))
        ymin, ymax = float(min(p1w[1], p2w[1])), float(max(p1w[1], p2w[1]))
        zmin, zmax = float(min(p1w[2], p2w[2])), float(max(p1w[2], p2w[2]))
        ix0, ix1 = idx_range_for_world(xmin, xmax, r_pad_world)
        iy0, iy1 = idx_range_for_world(ymin, ymax, r_pad_world)
        iz0, iz1 = idx_range_for_world(zmin, zmax, r_pad_world)
        xs = grid_coords[ix0:ix1+1]; ys = grid_coords[iy0:iy1+1]; zs = grid_coords[iz0:iz1+1]
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        Zu = Z / s
        RXu = X - p1u[0]; RYu = Y - p1u[1]; RZu = Zu - p1u[2]
        tu = RXu * nu[0] + RYu * nu[1] + RZu * nu[2]
        mask_len = (tu >= 0.0) & (tu <= Lu)
        rX = RXu - tu * nu[0]; rY = RYu - tu * nu[1]; rZ = RZu - tu * nu[2]
        outer_ok = (rX**2 + rY**2 + rZ**2) <= (b**2)
        if use_hole and create_hole:
            inner_ok = (rX**2 + rY**2 + (k_in * rZ)**2) <= (b_h**2)
            final_mask = mask_len & outer_ok & (~inner_ok)
        else:
            final_mask = mask_len & outer_ok
        if np.any(final_mask):
            sub = grid[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1]
            sub[final_mask] = permittivity
            grid[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] = sub
    grid_ff = grid.copy()
    grid_ff[grid_ff == permittivity_bg] = 0; grid_ff[grid_ff > 0] = 1
    return grid, b_array, float(grid_ff.mean())


def voxelize_network(rods, N=None, box_size=None, n_rod=2.9275, n_bg=1.0,
                     minor_radius=0.22, aspect_ratio=1.0, grid_size=220, verbose=False):
    """Convenience wrapper. Box defaults to LSU scaling (N/1000)^(1/3)*11.44 um.
    Returns dict with eps grid, ff, eps_av, n_av, box, dx."""
    if box_size is None:
        if N is None:
            raise ValueError("give N or box_size")
        box_size = (N / 1000.0 * 11.44**3) ** (1/3)
    eps, b, ff = create_permittivity_grid_penlike(
        rods, grid_size=grid_size, minor_radius=minor_radius, aspect_ratio=aspect_ratio,
        permittivity=n_rod**2, permittivity_bg=n_bg**2, box_size=box_size,
        progress_every=None, verbose=verbose)
    eps_av = ff * n_rod**2 + (1 - ff) * n_bg**2
    return dict(eps=eps, ff=ff, eps_av=eps_av, n_av=float(np.sqrt(eps_av)),
                box=float(box_size), dx=float(box_size / grid_size), grid_size=grid_size,
                n_rod=n_rod, n_bg=n_bg, minor_radius=minor_radius, aspect_ratio=aspect_ratio)


# =====================================================================
# 1.  Spectral density  chi_V(q)
# =====================================================================
def spectral_density(eps_grid, box_size, nbins=120):
    """Return (chi3d, q_axis, interp3d, q_centers, chi_radial) in PHYSICAL units.

    chi3d        : fftshifted |FFT[eps-<eps>]|^2 * dx^3 / G^3   [eps^2 * um^3], shape (G,G,G)
    q_axis       : fftshifted 1D q grid (um^-1) common to all 3 axes
    interp3d     : RegularGridInterpolator over (q,q,q) -> chi3d  (for directional g)
    q_centers    : radial bin centres (um^-1)
    chi_radial   : isotropic (radially-averaged) chi_V(|q|)  [eps^2 * um^3]
    """
    from scipy.interpolate import RegularGridInterpolator
    G = eps_grid.shape[0]; dx = box_size / G
    de = eps_grid.astype(np.float64) - float(eps_grid.mean())
    F = np.fft.fftn(de)
    norm = (dx**3) / (G**3)                       # |FFT|^2 -> physical chi_V [eps^2 um^3]
    chi3d = np.fft.fftshift(np.abs(F)**2) * norm
    q_axis = np.fft.fftshift(2*np.pi*np.fft.fftfreq(G, d=dx))
    interp3d = RegularGridInterpolator((q_axis, q_axis, q_axis), chi3d,
                                       bounds_error=False, fill_value=0.0)
    QX, QY, QZ = np.meshgrid(q_axis, q_axis, q_axis, indexing='ij')
    qmag = np.sqrt(QX**2 + QY**2 + QZ**2).ravel()
    edges = np.linspace(0, q_axis.max(), nbins+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    idx = np.clip(np.digitize(qmag, edges)-1, 0, nbins-1)
    sw = np.bincount(idx, weights=chi3d.ravel(), minlength=nbins)
    cw = np.bincount(idx, minlength=nbins)
    pop = cw > 0                                   # drop empty bins (bin width can be < 2pi/L):
    return chi3d, q_axis, interp3d, centers[pop], sw[pop]/cw[pop]   # avoids spurious interior zeros


def variance_check(chi_radial, q_centers, eps_grid):
    """Parseval: INT chi_V d^3q /(2pi)^3 should reproduce Var(eps)."""
    recon = np.trapezoid(4*np.pi*q_centers**2*chi_radial, q_centers)/(2*np.pi)**3
    return float(recon), float(np.var(eps_grid))


# =====================================================================
# 2.  Born anisotropy factor  g
# =====================================================================
def _fpol(theta, polarized=False):
    return 1.0 if polarized else (1.0 + np.cos(theta)**2)/2.0

def g_isotropic(q_centers, chi_radial, k_av, ntheta=3000, polarized=False):
    """g from the radially-averaged spectral density (statistically isotropic medium)."""
    th = np.linspace(0.0, np.pi, ntheta)
    q = 2*k_av*np.sin(th/2)
    chi = np.interp(q, q_centers, chi_radial, left=0.0, right=0.0)
    p = chi * _fpol(th, polarized) * np.sin(th)
    num = np.trapezoid(p*np.cos(th), th); den = np.trapezoid(p, th)
    return float(num/den) if den > 0 else float('nan')

def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n); th = np.pi*(1 + 5**0.5)*i
    return np.stack([np.sin(phi)*np.cos(th), np.sin(phi)*np.sin(th), np.cos(phi)], -1)

def g_directional(interp3d, k_av, u_in, ndir=6000, polarized=False):
    """g = <cos> for a specific incidence direction (full 3D Ewald-sphere sampling)."""
    u_in = np.asarray(u_in, float); u_in = u_in/np.linalg.norm(u_in)
    u_out = fib_sphere(ndir)
    cos_t = u_out @ u_in
    chi = interp3d(k_av*(u_out - u_in[None, :]))
    w = chi * (1.0 if polarized else (1 + cos_t**2)/2)
    s = np.sum(w)
    return float(np.sum(w*cos_t)/s) if s > 0 else float('nan')

def g_orientation_avg(interp3d, k_av, n_inc=24, ndir=4000, polarized=False):
    return float(np.mean([g_directional(interp3d, k_av, d, ndir, polarized)
                          for d in fib_sphere(n_inc)]))


# =====================================================================
# 3.  Mean free paths
# =====================================================================
def born_ell_s(q_centers, chi_radial, k_av, n_av, ntheta=4000, polarized=False):
    """Weak-scattering (Born) scattering mean free path ell_s [um].
       1/ell_s = (k0^4/8pi) INT_0^pi F_pol chi_V(q(theta)) sin(theta) dtheta."""
    th = np.linspace(0.0, np.pi, ntheta)
    q = 2*k_av*np.sin(th/2)
    chi = np.interp(q, q_centers, chi_radial, left=0.0, right=0.0)
    k0 = k_av/n_av
    inv = (k0**4/(8*np.pi))*np.trapezoid(_fpol(th, polarized)*chi*np.sin(th), th)
    return float(1.0/inv) if inv > 0 else float('inf')

def ell_t(ell_s, g):
    return ell_s/(1.0 - g)


# =====================================================================
# 4.  Analytic references (validation)
# =====================================================================
def mie_qsca_g(m, x, nmax=None):
    """Exact Mie Q_sca and asymmetry g (Bohren & Huffman), real relative index m."""
    from scipy.special import spherical_jn, spherical_yn
    m = float(m)
    if nmax is None:
        nmax = int(np.ceil(x + 4.05*x**(1/3) + 2)) + 10
    n = np.arange(1, nmax+1)
    def psi(z):
        jn = spherical_jn(n, z); jnp = spherical_jn(n, z, derivative=True)
        return z*jn, jn + z*jnp
    def xi(z):
        jn = spherical_jn(n, z); jnp = spherical_jn(n, z, derivative=True)
        yn = spherical_yn(n, z); ynp = spherical_yn(n, z, derivative=True)
        h = jn + 1j*yn; hp = jnp + 1j*ynp
        return z*h, h + z*hp
    pm, pmp = psi(m*x); px, pxp = psi(x); xx, xxp = xi(x)
    a = (m*pm*pxp - px*pmp)/(m*pm*xxp - xx*pmp)
    b = (pm*pxp - m*px*pmp)/(pm*xxp - m*xx*pmp)
    qsca = (2/x**2)*np.sum((2*n+1)*(np.abs(a)**2 + np.abs(b)**2))
    t1 = (n[:-1]*(n[:-1]+2)/(n[:-1]+1))*np.real(a[:-1]*np.conj(a[1:]) + b[:-1]*np.conj(b[1:]))
    t2 = ((2*n+1)/(n*(n+1)))*np.real(a*np.conj(b))
    gq = (4/x**2)*(np.sum(t1)+np.sum(t2))
    return float(qsca), float(gq/qsca)

def g_rgd_sphere(x, polarized=False, ntheta=4000):
    """Analytic Rayleigh-Gans-Debye asymmetry g for a sphere, x=kR."""
    th = np.linspace(1e-9, np.pi, ntheta)
    u = 2*x*np.sin(th/2)
    Gf = 3*(np.sin(u) - u*np.cos(u))/u**3
    p = (Gf**2)*_fpol(th, polarized)*np.sin(th)
    return float(np.trapezoid(p*np.cos(th), th)/np.trapezoid(p, th))


# =====================================================================
# 5.  DDA  (coupled-dipole, FFT-accelerated, LDR)  -- validation engine
# =====================================================================
def polarizability_LDR(eps, k, d, khat, epol):
    b1, b2, b3 = -1.8915316, 0.1648469, -1.7700004
    S = float(np.sum((np.real(khat)*np.real(epol))**2))
    a_cm = (3*d**3/(4*np.pi))*(eps-1)/(eps+2)
    kd = k*d
    denom = 1 + (a_cm/d**3)*((b1 + eps*b2 + eps*b3*S)*kd**2 - (2/3)*1j*kd**3)
    return a_cm/denom

def _interaction_fft(shape, d, k):
    G = np.asarray(shape, int); P = 2*G
    offs = []
    for ax in range(3):
        o = np.arange(P[ax]); o = np.where(o >= G[ax], o - P[ax], o); offs.append(o.astype(float))
    OX, OY, OZ = np.meshgrid(offs[0], offs[1], offs[2], indexing='ij')
    X, Y, Z = OX*d, OY*d, OZ*d
    r = np.sqrt(X*X+Y*Y+Z*Z); r_safe = np.where(r == 0, 1.0, r)
    pref = np.exp(1j*k*r)/r_safe
    c1 = k*k; c2 = (1.0/r_safe**2 - 1j*k/r_safe); inv_r2 = 1.0/r_safe**2
    comps = {}
    for (a, b, key) in [(X,X,'xx'),(X,Y,'xy'),(X,Z,'xz'),(Y,Y,'yy'),(Y,Z,'yz'),(Z,Z,'zz')]:
        nn = (a*b)*inv_r2; delta = 1.0 if key in ('xx','yy','zz') else 0.0
        A = pref*(c1*(delta-nn) + c2*(3*nn-delta)); A[r == 0] = 0.0
        comps[key] = np.fft.fftn(A)
    return comps, tuple(P)

def _apply_interaction(comps, Pshape, Gshape, Pvec):
    Gx, Gy, Gz = Gshape
    def F(field):
        f = np.zeros(Pshape, complex); f[:Gx, :Gy, :Gz] = field; return np.fft.fftn(f)
    Fx, Fy, Fz = F(Pvec[..., 0]), F(Pvec[..., 1]), F(Pvec[..., 2])
    cxx, cxy, cxz, cyy, cyz, czz = (comps[k] for k in ('xx','xy','xz','yy','yz','zz'))
    yx = np.fft.ifftn(cxx*Fx+cxy*Fy+cxz*Fz)[:Gx, :Gy, :Gz]
    yy = np.fft.ifftn(cxy*Fx+cyy*Fy+cyz*Fz)[:Gx, :Gy, :Gz]
    yz = np.fft.ifftn(cxz*Fx+cyz*Fy+czz*Fz)[:Gx, :Gy, :Gz]
    return np.stack([yx, yy, yz], -1)

def dda_solve(occ, eps_grid, d, k, khat, epol, E0=1.0, tol=1e-4, maxiter=4000):
    """Coupled-dipole solve on a cubic occupancy grid. Returns P, coords, alpha, Einc, occ, res_hist."""
    Gshape = occ.shape
    comps, Pshape = _interaction_fft(Gshape, d, k)
    coords = (np.stack(np.meshgrid(*[np.arange(s) for s in Gshape], indexing='ij'), -1)).astype(float)*d
    khat = np.asarray(khat, float); epol = np.asarray(epol, float)
    phase = k*(coords @ khat)
    Einc = np.zeros(Gshape + (3,), complex)
    for c in range(3):
        Einc[..., c] = E0*epol[c]*np.exp(1j*phase)
    Einc *= occ[..., None]
    alpha = np.zeros(Gshape, complex)
    for ev in np.unique(eps_grid[occ]):
        alpha[occ & (eps_grid == ev)] = polarizability_LDR(complex(ev), k, d, khat, epol)
    inv_alpha = np.zeros(Gshape, complex); inv_alpha[occ] = 1.0/alpha[occ]

    def matvec(Pv):
        Pv = Pv*occ[..., None]
        return (inv_alpha[..., None]*Pv - _apply_interaction(comps, Pshape, Gshape, Pv))*occ[..., None]

    b = Einc.copy(); x = np.zeros_like(b); r = b - matvec(x); r0 = r.copy()
    dot = lambda a, c: np.sum(a*c)
    rho_prev = alpha_s = omega = 1.0
    p = np.zeros_like(b); v = np.zeros_like(b); bnorm = np.linalg.norm(b); hist = []
    for it in range(maxiter):
        rho = dot(r0.conj(), r); beta = (rho/rho_prev)*(alpha_s/omega)
        p = r + beta*(p - omega*v); v = matvec(p)
        alpha_s = rho/dot(r0.conj(), v); s = r - alpha_s*v; t = matvec(s)
        omega = dot(t.conj(), s)/dot(t.conj(), t)
        x = x + alpha_s*p + omega*s; r = s - omega*t; rho_prev = rho
        res = np.linalg.norm(r)/bnorm; hist.append(res)
        if res < tol:
            break
    return x, coords, alpha, Einc, occ, hist

def dda_cross_sections_g(P, coords, alpha, Einc, occ, k, khat, E0=1.0, ndir=3000):
    """Return dict(C_ext, C_abs, C_sca, g) from a solved DDA dipole field."""
    sel = occ; pos = coords[sel]; Pv = P[sel]; Ein = Einc[sel]; inv_a = 1.0/alpha[sel]
    C_ext = 4*np.pi*k/abs(E0)**2*np.sum(np.imag(np.sum(np.conj(Ein)*Pv, axis=1)))
    term = (np.imag(np.sum(Pv*(np.conj(inv_a)[:, None]*np.conj(Pv)), axis=1))
            - (2/3)*k**3*np.sum(np.abs(Pv)**2, axis=1))
    C_abs = 4*np.pi*k/abs(E0)**2*np.sum(term)
    dirs = fib_sphere(ndir)
    F = np.exp(-1j*k*(dirs @ pos.T)) @ Pv
    Fperp = F - dirs*np.sum(dirs*F, axis=1, keepdims=True)
    dCdO = k**4/abs(E0)**2*np.sum(np.abs(Fperp)**2, axis=1)
    w = 4*np.pi/ndir; C_sca = np.sum(dCdO)*w
    g = np.sum(dCdO*(dirs @ np.asarray(khat, float)))*w/C_sca
    return dict(C_ext=float(C_ext), C_abs=float(C_abs), C_sca=float(C_sca), g=float(g))

def make_sphere(radius, d):
    G = int(np.ceil(2*radius/d)) + 2
    ax = (np.arange(G) - (G-1)/2.0)*d
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    occ = (X**2+Y**2+Z**2) <= radius**2
    a_eff = (3*int(occ.sum())*d**3/(4*np.pi))**(1/3)
    return occ, a_eff


# =====================================================================
# 6.  Strong-contrast effective permittivity -> scattering MFP  ell_s
#     Torquato & Kim, Phys. Rev. X 11, 021002 (2021), 3D scaled form.
#     Quantitative at high contrast where weak-scattering Born is not.
# =====================================================================
def maxwell_garnett_3d(eps_q, eps_p, phi_p):
    """Static Maxwell-Garnett: inclusion p (frac phi_p) in host q."""
    beta = (eps_p - eps_q)/(eps_p + 2*eps_q)
    return eps_q*(1 + 3*phi_p*beta/(1 - phi_p*beta))

def bruggeman_3d(eps1, eps2, phi2):
    """Symmetric Bruggeman effective permittivity (the connected/bicontinuous reference)."""
    phi1 = 1 - phi2
    b = 2*phi1*eps1 - phi1*eps2 + 2*phi2*eps2 - phi2*eps1
    return (b + np.sqrt(b**2 + 8*eps1*eps2))/4.0

def indicator_spectral_density(eps_grid, box_size, eps_bg=1.0, nbins=120):
    """Radial spectral density of the PHASE INDICATOR (S_2(r)-phi^2), in um^3.
       This is the chi_V used by the strong-contrast formulas (NOT the eps-field one)."""
    ind = (eps_grid > eps_bg + 1e-6).astype(np.float64)
    _, _, _, q_centers, chi_radial = spectral_density(ind, box_size, nbins=nbins)
    return q_centers, chi_radial

def _ImF_3d(Qgrid, q_centers, chi_ind, nq=4000):
    """Im[F](Q) = -Q/(2 (2pi)^{3/2}) INT_0^{2Q} q chi_ind(q) dq   (PRX Eq. 70, d=3)."""
    Qgrid = np.atleast_1d(np.asarray(Qgrid, float))
    pref = -1.0/(2*(2*np.pi)**1.5)
    qf = np.linspace(0.0, q_centers.max(), nq)
    integ = qf*np.interp(qf, q_centers, chi_ind, left=0, right=0)
    out = np.empty_like(Qgrid)
    for i, Q in enumerate(Qgrid):
        m = qf <= 2*Q
        out[i] = pref*Q*np.trapezoid(integ[m], qf[m]) if m.sum() > 1 else 0.0
    return out

def _ReF_3d_KK(Q, qg, ImF_g):
    """Re[F](Q) via Kramers-Kronig (PRX Eq. 71) with the p.v. subtraction trick."""
    g = np.where(qg > 0, ImF_g/np.where(qg == 0, 1, qg), 0.0)
    gQ = np.interp(Q, qg, g)
    with np.errstate(divide='ignore', invalid='ignore'):
        integ = (g - gQ)/(Q**2 - qg**2)        # regular at q=Q (pv of 1/(Q^2-q^2)=0)
    bad = ~np.isfinite(integ)
    if bad.any():
        integ[bad] = np.interp(qg[bad], qg[~bad], integ[~bad])
    return -(2*Q**2/np.pi)*np.trapezoid(integ, qg)

def strong_contrast_eps_eff(k0, q_centers, chi_ind, eps_p, eps_q, phi_p,
                            scaled=True, ref_scale=None):
    """Complex effective permittivity, 3D scaled 2-point strong-contrast (PRX Eq. 73).
       eps_p = inclusion phase, eps_q = reference phase, phi_p = inclusion fraction.
       ref_scale: permittivity used to scale the F-argument (default = static HS/MGA);
                  pass bruggeman_3d(...) for the connected-network (bicontinuous) reference."""
    beta = (eps_p - eps_q)/(eps_p + 2*eps_q)
    eps_HS = eps_q*(1 + 3*phi_p*beta/(1 - phi_p*beta)) if ref_scale is None else ref_scale
    k_q = np.sqrt(eps_q)*k0
    Q = np.sqrt(eps_HS/eps_q)*k_q if scaled else k_q
    qg = np.linspace(1e-6, q_centers.max(), 1200)
    ImF_g = _ImF_3d(qg, q_centers, chi_ind)
    F = _ReF_3d_KK(Q, qg, ImF_g) + 1j*float(np.interp(Q, qg, ImF_g))
    denom = phi_p*(1 - beta*phi_p) + np.sqrt(2*np.pi)*beta*F
    return eps_q*(1 + 3*beta*phi_p**2/denom)

def strong_contrast_ell_s(k0, q_centers, chi_ind, eps_p, eps_q, phi_p,
                          scaled=True, ref_scale=None):
    """Scattering (=extinction, lossless) mean free path ell_s [um] = 1/(2 Im k_e)."""
    eps_e = strong_contrast_eps_eff(k0, q_centers, chi_ind, eps_p, eps_q, phi_p, scaled, ref_scale)
    imk = abs((k0*np.sqrt(eps_e)).imag)
    return (1.0/(2*imk)) if imk > 0 else np.inf

def strong_contrast_ell_s_network(k0, q_centers, chi_ind, n_rod, n_bg, ff, reference='rods'):
    """Convenience: ell_s for the network with the recommended reference choice.
       reference='rods'  -> rod phase is the strong-contrast reference (best for the
                            connected/percolating high-index skeleton; air voids = inclusions).
       reference='air'   -> air is the reference, rods = inclusions (bracket; ratio 8.6 -> needs care).
       reference='bruggeman' -> rods-as-reference but scaled by the symmetric Bruggeman eps."""
    er, eb = n_rod**2, n_bg**2
    if reference == 'air':
        return strong_contrast_ell_s(k0, q_centers, chi_ind, eps_p=er, eps_q=eb, phi_p=ff)
    rs = bruggeman_3d(eb, er, ff) if reference == 'bruggeman' else None
    return strong_contrast_ell_s(k0, q_centers, chi_ind, eps_p=eb, eps_q=er, phi_p=1-ff, ref_scale=rs)


# =====================================================================
# 7.  Effective index n_eff -> contrast-dependent (beyond-Born) g
#     The ONLY contrast dependence of the NORMALISED single-scattering g is the
#     Ewald radius  q = 2 k_eff sin(theta/2),  k_eff = k0 * Re[n_eff].
#     (Vynck et al. RMP 95, 045003 (2023), Table I / Eqs. 55-79: the scalar local-
#     field / Clausius-Mossotti vertex cancels in normalised g exactly like Delta-eps;
#     a full angular DDA is non-viable for this connected, strongly-scattering medium
#     -- thin-vs-representative gap, Mishchenko config-averaging.)  So a faithful
#     beyond-Born g  <=>  pinning n_eff.  For a CONNECTED, chunky high-index skeleton
#     n_eff sits at the UPPER Hashin-Shtrikman bound (the high-index phase percolates
#     -> "high-index matrix" topology), NOT the isolated-inclusion MG (HS lower) and
#     NOT the Wiener volume average (which lies ABOVE HS-upper -> unphysical, isotropic).
#     VALIDATED full-wave: coherent-transmission DDA on a network slab gave Re[n_eff]=1.53
#     at lambda=6 um, vs strong-contrast rod-ref 1.55 -- two independent methods agree to
#     ~1.5% near HS-upper.  See Investigation_g_Values/ (coherent_neff_dda_gpu.py + the
#     <P(z)> profile).  Im[k_eff]/ell_s from that run is UNUSABLE (Fabry-Perot + residual
#     diffuse) -- keep ell_s from the strong-contrast expansion (Section 6).
# =====================================================================
def hs_bounds_n(n_rod, n_bg, ff):
    """Hashin-Shtrikman index bounds (isotropic two-phase; tighter than Wiener).
       Returns (n_lo, n_hi).
       n_lo: high-index = dispersed inclusion in low-index host  (== Maxwell-Garnett; HS lower).
       n_hi: low-index  = dispersed inclusion in high-index (connected) host (HS upper).
       For a CONNECTED, chunky high-index network the physical n_eff sits near n_hi."""
    er, eb = n_rod**2, n_bg**2
    e_lo = maxwell_garnett_3d(eb, er, ff)        # rods dispersed in air  -> HS lower
    e_hi = maxwell_garnett_3d(er, eb, 1 - ff)    # air dispersed in rods  -> HS upper
    return float(np.sqrt(e_lo)), float(np.sqrt(e_hi))


def strong_contrast_neff_network(k0, q_centers, chi_ind, n_rod, n_bg, ff,
                                 reference='rods', hs_cap=True):
    """Re[n_eff](k0) for the network, from the strong-contrast eps_e (Section 6 machinery).
       reference='rods' (the connected high-index phase as reference) is the physical choice.
       hs_cap=True clamps to the rigorous HS upper bound: the 2-point truncation can overshoot
       it by a few % (finite-frequency dispersion), but quasi-statically n_eff<=n_hi for an
       isotropic medium.  This is the Ewald-radius index used for the contrast-dependent g."""
    er, eb = n_rod**2, n_bg**2
    if reference == 'air':
        eps_e = strong_contrast_eps_eff(k0, q_centers, chi_ind, eps_p=er, eps_q=eb, phi_p=ff)
    else:
        rs = bruggeman_3d(eb, er, ff) if reference == 'bruggeman' else None
        eps_e = strong_contrast_eps_eff(k0, q_centers, chi_ind, eps_p=eb, eps_q=er,
                                        phi_p=1 - ff, ref_scale=rs)
    n = float(np.real(np.sqrt(eps_e)))
    if hs_cap:
        _, n_hi = hs_bounds_n(n_rod, n_bg, ff)
        n = min(n, n_hi)
    return n


def g_contrast_isotropic(q_centers, chi_radial, k0, n_eff, ntheta=3000, polarized=False):
    """Contrast-dependent (beyond-Born) g at the physical effective Ewald radius
       k_eff = k0 * Re[n_eff].  Identical machinery to g_isotropic; the contrast enters
       ONLY through n_eff (the scalar-vertex theorem above).  Reduces to the Born g as the
       contrast -> 0 (then every effective index, incl. n_eff and the Wiener n_av, -> n_bg)."""
    return g_isotropic(q_centers, chi_radial, k0 * n_eff, ntheta=ntheta, polarized=polarized)
