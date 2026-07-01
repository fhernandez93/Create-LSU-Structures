# Beyond-Born, contrast-dependent `g` — full-wave `n_eff` validation

Supporting evidence for **Section 7** of `Anisotropy_g_from_ends.ipynb` (the contrast-dependent
anisotropy factor `g(λ)`). The deliverable itself lives in the notebook + `scattering_g.py`
(`hs_bounds_n`, `strong_contrast_neff_network`, `g_contrast_isotropic`); these files are the
**one-time full-wave check** that pins the effective Ewald index.

## Why this matters
In the scalar-vertex framework (Vynck *et al.*, *Rev. Mod. Phys.* **95**, 045003 (2023), Table I,
Eqs. 55–79) the **only** way the index contrast enters the *normalized* single-scattering `g` is the
Ewald-sphere radius `q = 2·k_eff·sin(θ/2)`, with `k_eff = k0·Re[n_eff]`. The beyond-Born scalar
local-field / Clausius–Mossotti vertex cancels in the normalized `g` exactly like `Δε²`. So a faithful
beyond-Born `g` reduces to **pinning `n_eff`** (a full angular configurational-averaging DDA is
non-viable for this connected, strongly-scattering medium — the optically-thin and
correlation-representative cluster sizes are mutually exclusive where it matters).

The notebook's original Born `g` used the **Wiener volume average `n_av = 1.64`**, which is
**unphysical** — it lies *above* the Hashin–Shtrikman upper bound for an isotropic two-phase medium.

## What was measured
A **coherent-transmission DDA** on a slab cut from the N=4000 network: illuminate with a plane wave,
take the `k⊥=0` (transverse-mean) component of the induced polarization `⟨P(z)⟩`, and fit its complex
phase ramp → `k_eff` (phase slope = `Re k_eff = k0·n_eff`).

**Result: `Re[n_eff] = 1.53` at λ = 6 µm**, vs the strong-contrast (Torquato–Kim) rod-reference value
`1.55` — two *independent* methods agreeing to ~1.5 %, both at the **HS upper bound** (≈1.51). The
connected, *chunky* high-index skeleton (rod radius 0.335 vs bond 0.8, percolating in 3-D) behaves like
a *"high-index matrix"* topology, so `n_eff` sits at HS-upper — **not** the isolated-inclusion
Maxwell-Garnett value (= HS lower = 1.25) and **not** symmetric Bruggeman (1.33).

## Pitfalls found (each cost a solve — documented so they aren't repeated)
1. **Short λ fails:** the slab is optically thick (ℓ_s ≈ 0.4 µm ≪ depth), the ballistic/coherent field
   is extinguished, and the fit returns garbage (n≈2.5, fit-resid 0.77). **Measure at long λ** (weak
   scattering, ℓ_s ≫ slab depth).
2. **Under-averaging biases low:** a 5–6 µm transverse patch (~25 correlation areas) gives n≈1.345 with
   ~20 % diffuse residual. Use the **full transverse box (~16 µm)** → n≈1.53. (See the `⟨P(z)⟩` profile.)
3. **`Im[k]`/ℓ_s is unusable** from this measurement (Fabry–Pérot + residual diffuse → factor-3
   modulation of `|⟨P⟩|`). Only `Re[n_eff]` survives; keep ℓ_s from the strong-contrast expansion.

## Files
- `neff_estimator.py` — CPU coupled-dipole solve + the two-wave / phase-slope `k_eff` fit; the
  uniform-slab validation anchor (a slab of known index `n` must return `n_eff = n`; got 1.489 for n=1.5).
- `coherent_neff_dda_gpu.py` — JAX/GPU port of the solve (bit-identical to CPU on the uniform anchor;
  the CPU BiCGSTAB stagnates at res≈0.04 for n=2.93, the GPU pushes through to ~1e-3 in ~60 s).
  Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- `neff_fullwave_lam6.npz` — the saved `⟨P(z)⟩` profile and fit (λ=6, full transverse box).
- `neff_fullwave_Pz_profile_lam6.png` — the coherent amplitude + phase-ramp plot (the linear phase
  ramp is what licenses the `n_eff` read; the amplitude modulation is why `Im[k]` is discarded).
