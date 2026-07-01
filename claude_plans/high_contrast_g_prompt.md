# Investigation prompt — high-contrast (beyond-Born) corrections to the anisotropy factor g

## Mission
The notebook `Anisotropy_g_from_ends.ipynb` (+ engine `scattering_g.py`) computes the scattering
anisotropy factor **g = ⟨cosθ⟩** of the LSU amorphous networks from the first **Born / spectral-density**
approximation. That g is the *single-scattering phase-function shape* — and it is **independent of the
index-contrast magnitude** (the Δε² prefactor cancels in the normalized phase function). At the real
contrast (n_rod ≈ 2.9275, n_bg = 1) the material is a **strong scatterer** (k_av·ℓ_s ≈ 3.6 at the
scattering peak, near the Ioffe–Regel limit), so the *true* transport g carries **beyond-Born,
contrast-dependent corrections** (dependent/near-field scattering, Mie-type resonance of the local
structure) that the current g misses. **Goal: introduce those high-contrast corrections to g, faithfully,
WITHOUT FDTD**, and report a contrast-dependent g(λ) (and the corrected ℓ_t = ℓ_s/(1−g)) — validated so it
reduces to the existing Born g in the low-contrast limit.

This is a methodology task with a real cost-vs-rigor fork: **investigate, validate, then surface the
method choice to the user before building** (mirror how the original task used `advisor` + research agents
+ `AskUserQuestion`). Do not silently pick the heavy path.

## Read first (do NOT re-derive — start from here)
- Memory: `memory/lsu-anisotropy-g-notebook.md` (all decisions + pitfalls), `memory/MEMORY.md`.
- `scattering_g.py` — the engine. Already contains, **validated**:
  - Born/spectral-density g: `spectral_density`, `g_isotropic`, `g_directional`, `g_orientation_avg`.
  - Mean free paths: `born_ell_s` (weak-scatt), Torquato strong-contrast `strong_contrast_eps_eff` /
    `strong_contrast_ell_s_network` (gives ℓ_s only — **not** g).
  - **DDA (coupled-dipole, FFT-accelerated, LDR polarizability)**: `dda_solve`,
    `dda_cross_sections_g`, `make_sphere`, `polarizability_LDR`, `_interaction_fft`. **Validated
    against analytic Mie** (`mie_qsca_g`) — optical theorem C_ext=C_sca holds exactly. This is your
    full-wave (no-FDTD) kernel — REUSE it, don't reinvent.
  - Analytic anchors: `mie_qsca_g`, `g_rgd_sphere`.
  - `voxelize_network`, `create_permittivity_grid_penlike` (rods → periodic ε grid).
- `Anisotropy_g_from_ends.ipynb` — the deliverable; add to it, don't rebuild. Its Section 5 explains
  why Born g is contrast-independent; Section 9 is the convergence/error budget.

## Hard constraints
- **No FDTD / no time-domain solver.** Frequency-domain / integral-equation / effective-medium only.
- n_rod = 2.9275, n_bg = 1 (ε ratio 8.57); ff ≈ 0.10–0.21 (depends on rod radius/aspect); box
  L = (N/1000)^(1/3)·11.44 µm; bond d0 = 0.8 µm; FSDP q_peak ≈ 3.6 µm⁻¹.
- Env: conda `lsu_project` (py3.12): numpy/scipy/jax(CUDA, **12 GB** GPU)/h5py/matplotlib; **no miepython**
  (Mie is inline in `scattering_g.py`). Run with `JAX_PLATFORMS=cpu` if the GPU is contended (62 GB RAM).
- Never overwrite `Example/*` or files you didn't create. Add outputs to `Investigation_g_Values/`.

## Established facts & pitfalls (verified this project — do not repeat these mistakes)
1. **g (Born) is contrast-independent by construction.** For a binary medium ε=ε_bg+Δε·χ(r),
   dσ/dΩ ∝ Δε²·χ̃_V(q)·P(θ); Δε² cancels in g. Contrast enters only (a) the *strength* (ℓ_s ∝ 1/Δε²)
   and (b) weakly via the Ewald radius k_av = k0√ε_av (dispersion). So any *contrast-dependence of g*
   must come from **beyond-Born** physics — that is the whole task.
2. **A finite free-space DDA cluster does NOT give bulk g.** n_av ≈ 1.3–1.6 produces a coherent forward
   "envelope" lobe (∝ volume²) that swamps the diffuse phase function → g→1 artifact. The fix is
   **configurational averaging**: diffuse dσ/dΩ ∝ ⟨|F(n̂)|²⟩ − |⟨F(n̂)⟩|² over disorder realizations /
   cluster positions / orientations / incidence. This is the load-bearing subtlety — get it right.
3. **Connected network breaks form-factor × structure-factor factorization.** No isolated "scatterer";
   the rigorous Born object is the spectral density χ̃_V(q). Any motif/t-matrix decomposition is
   approximate for these networks — treat as a cross-check, not ground truth.
4. **Strong-contrast (Torquato/Kim) gives ℓ_e, not g.** Don't expect a phase function from it.
5. **q-resolution = 2π/L**; g converged for a/λ ≳ 0.25 (N1000 ≈ N4000); long-λ is finite-size-limited.
   System size & grid are already saturated — a bigger structure will NOT fix g (see Section 9).
6. **DDA at n=2.93 converges slowly**: needs |m|·k·d ≲ 0.2 for clean Mie agreement; ~10–30% error at
   coarser resolution. Cut clusters must balance resolution (fine d) vs FFT size (2R padded) on 12 GB GPU.

## Candidate methods (investigate, pin formulas with research agents, then offer the fork to the user)
Ordered cheap→rigorous. Each MUST be checked to reduce to the existing Born g as Δε→0.
- **(A) DDA + configurational averaging → diffuse phase function → contrast-dependent g** *(rigorous,
  full-wave, no-FDTD, expensive).* Cut representative clusters (spheres radius R ≳ correlation length
  ~1–2 µm), illuminate (plane wave, multiple incidences/polarizations), ensemble-average to subtract the
  coherent field, take the diffuse far-field → p(θ) → g. **Key design question (resolve with literature +
  advisor):** the cluster must be optically *thin* (single scattering dominant, so the diffuse pattern IS
  the phase function) yet large enough to carry the local correlations + near-field — i.e. extract the
  *effective single-scattering phase function* of the dense medium, not a multiple-scattering halo.
  This is the gold-standard answer; budget GPU time.
- **(B) Effective-wavenumber / distorted-wave Born (DWBA)** *(cheap first cut).* Replace k_av with the
  complex effective wavenumber k_eff = k0√ε_eff (from the existing strong-contrast `ε_eff`) in the Ewald
  sampling, and optionally renormalize the scattering vertex from bare Δε to the exact per-voxel
  polarizability (Clausius-Mossotti/LDR, already in code). Captures the dispersion + leading vertex
  correction; NOT the full angular beyond-Born physics. Good as a fast baseline / sanity bound.
- **(C) Exact-motif t-matrix × structure factor** *(semi-analytic cross-check).* DDA-compute the t-matrix
  of a representative motif (single rod, or a node + its 3 bonds) at the real contrast; combine
  dσ/dΩ = |t(q,n̂,n̂′)|²·S(q). Contrast-dependent via the exact motif scattering. Caveat: factorization
  is approximate for a connected network (fact #3) — use only to bracket (A).
- **(D) Quasi-crystalline approximation / dense-media radiative transfer (Tsang–Kong DMRT)**
  *(semi-analytic effective-field).* Gives both ℓ and a contrast-dependent phase function for correlated
  scatterers; discrete-scatterer assumption (same caveat). Consider if (A) is too costly.

## Validation anchors (decisive — a method that fails these is wrong)
- **Low-contrast limit:** as Δε→0 (e.g. n_rod = 1.05–1.2), the new contrast-dependent g MUST converge to
  the existing Born `g_isotropic`/`g_orientation_avg`. This is the single most important check — it ties
  the new method to the validated one.
- **Mie:** the DDA kernel is already Mie-validated; for method (A) re-confirm on a single dielectric
  sphere that the *diffuse* extraction of g reproduces the Mie g at the real contrast.
- **Sanity:** g ∈ [−1, 1]; the high-contrast correction should move g relative to Born in a physically
  defensible direction (near the scattering/Ioffe–Regel peak, where the correction is largest); report
  the magnitude of the correction vs the Born g.
- **Self-consistency:** if method (A) yields g, recompute ℓ_t = ℓ_s/(1−g) with the strong-contrast ℓ_s
  and check it stays physical (k·ℓ_t > 0, reasonable vs the Born/Ioffe–Regel anchors).

## Process (mirror what worked on this project)
1. Orient: read the memory + `scattering_g.py` + notebook Sections 5/6/9.
2. Call **advisor** before committing to an approach (it sees your transcript) — this caught two
   load-bearing errors last time (the envelope artifact; the gap-vs-Bragg mislabel).
3. Launch **research agents** to pin the exact beyond-Born formalism + literature for the
   contrast-dependent g / effective phase function of correlated/connected dense media (Vynck RMP 2023
   dependent-scattering sections; Tsang–Kong DMRT; Mishchenko on configurational averaging; the coherent/
   diffuse decomposition). Get equations + validity, then verify them yourself.
4. **Surface the cost-vs-rigor fork to the user** (`AskUserQuestion`): which method (A/B/C/D), and the
   operating point (single λ vs the gap sweep). Recommend, don't survey.
5. **De-risk the hard kernel in scratchpad first** (e.g. the configurational-averaging diffuse extraction
   on a sphere → match Mie) BEFORE wiring into the notebook — exactly as the DDA/Born kernels were proven.
6. Validate against the low-contrast Born limit. Only then integrate as a **new notebook section**
   (keep the Born g as the labeled low-contrast limit / cross-check).
7. Update `memory/lsu-anisotropy-g-notebook.md` + `MEMORY.md` with what was validated/refuted. If the data
   contradicts a stored belief, the data wins — correct the memory.

## First steps
1. Read the memory files and `scattering_g.py`; reproduce the current Born g + strong-contrast ℓ_s for the
   N=1000 reference to re-establish the baseline you're correcting.
2. Quick win for intuition: implement method (B)'s cheapest variant — swap k_av → Re(k_eff) (Maxwell-
   Garnett or strong-contrast effective index) in the existing Born g and quantify how much g shifts. This
   is the *dispersion-only* contrast effect and bounds the trivial part; it is NOT the deliverable.
3. Call the advisor with your proposed approach; launch the research agents; then ask the user the fork.
4. Prove the chosen kernel against Mie + the low-contrast Born limit in scratchpad before touching the notebook.

## Deliverable
A new, validated section in `Anisotropy_g_from_ends.ipynb` (and supporting functions in `scattering_g.py`)
that reports a **contrast-dependent g(λ)** at n_rod = 2.9275 with a clear statement of method, validity, and
the size of the correction relative to the Born g — plus the updated ℓ_t. Re-execute the notebook; save
results to `Structures/`.
