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
A real fix would add a soft repulsion term to U; that breaks parity with the Sellers paper, so the user has to decide whether to deviate from spec.

**jaxopt is slower than scipy+JIT on CPU**, despite being "the JAX-native option". Benchmarked at N=40/200/600: jaxopt ~2.1 s/relax independent of N (per-call dispatch overhead), scipy+JIT scales linearly 15–46 ms/relax. The `use_jaxopt=True` flag exists for GPU runs but defaults to False. Don't switch the default without re-benchmarking.
