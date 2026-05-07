---
name: LSU known issues
description: Open issues in lsu_network.py that the user may run into and that future sessions should know are deliberately deferred, not bugs to re-fix.
type: project
---

**Bond collapse during WWW relaxation.** After many Stone-Wales iterations, L-BFGS occasionally settles into a local minimum where one bond's length ≈ 0 (e.g. 1e-6 to 0.05 µm). Min rod-length printouts of `0.000` at the end of `generate_lsu_network` are this.

**Why:** The Sellers energy (Supplement Eq. 2) has no explicit non-bonded vertex repulsion — only the Keating bond term `f1 = Σ(|r_ij| - d0)²`. A collapsed bond contributes only `(0 - 0.8)² = 0.64` to f1, easily masked by f2/f3/f4 terms. PBC-protected unit vectors via `jnp.maximum(L, 1e-12)` keep gradients finite but not informative near `L = 0`.

**How to apply:** Don't try to "fix" this in the relax — it's a property of the published energy. Mitigations are:
  - More WWW iterations (more chances to escape the bad basin via SW + reject)
  - Tighter `final_temperature` (fewer accepted bad moves late)
  - Larger `relax_global_iters` (more polish per global relax)
A real fix would add a soft repulsion term to U; that breaks parity with the Sellers paper. The local-relax fix for void clustering (below) also helps bond collapse as a side effect — out-of-shell vertices can't drift toward bonded neighbours and pinch a bond closed.

**jaxopt is slower than scipy+JIT on CPU**, despite being "the JAX-native option". Benchmarked at N=40/200/600: jaxopt ~2.1 s/relax independent of N (per-call dispatch overhead), scipy+JIT scales linearly 15–46 ms/relax. The `use_jaxopt=True` flag exists for GPU runs but defaults to False. Don't switch the default without re-benchmarking.

**Void clustering during WWW — root cause was the relaxation scheme, not the energy.** During 100k WWW iterations the vertex distribution drifts: 4³-voxel std grew from 2.98 (BM seed) → 7.31 (final) for the user's N=1102, L=14.3 µm run, vs reference 2.79. Manifests as rod depletion at the cubic-cell corners/faces (8-corner / 8-centre count ratio ~0.36 vs reference 0.59). **The BM initial network is not the cause** — it's uniform out of the box; the drift accumulates during WWW.

**Why (corrected 2026-05-07):** The earlier diagnosis blamed the missing non-bonded term in the Sellers energy. That gap is real but secondary. The primary cause was that our `relax()` did **full-N L-BFGS** for both the post-SW relax and the periodic global polish, distinguished only by `max_iter`. Sellers's Supplement Methods (p. 6) explicitly delegates the relaxation refinements to Vink 2001 and Mousseau-Barkema 2001, which prescribe **spatially-local relaxation up to the 4-neighbour shell of the SW move, with global L-BFGS only as a rare fallback**. Our full-N L-BFGS let vertices anywhere in the cell migrate toward each other under the no-repulsion energy — the mechanism producing corner voids. Hemmann/Saba 2026 (arXiv:2601.10333) uses the same Vink/MB local-only scheme and explicitly observes that void clustering still grows monotonically with accepted MC moves even with their angular-modified energy (their Figure 8c) — confirming the relaxation scheme is the dominant lever, not the energy term.

**How to apply:** As of 2026-05-07 the planned fix (`claude_plans/2026-05-07_local_relaxation_void_clustering.md`) is to restore the Vink/MB scheme: BFS the 4-neighbour shell of each SW move's seed vertices `{i, j, c, d}`, gradient-mask all out-of-shell positions to zero so L-BFGS doesn't move them. Keep the periodic global polish optional. If voxel std after this fix still doesn't drop to ≤3.5 the documented fall-back is to add `f5 = Σ_{(i,j) non-bonded} max(0, r_floor - |r_ij|)²` with `r_floor ≈ 0.5·d0` to both `energy_components` and `_energy_jax_full` — a deviation from Sellers Eq. 2 the user has approved.
