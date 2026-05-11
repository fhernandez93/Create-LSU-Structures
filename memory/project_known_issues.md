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

**Void clustering during WWW — updated 2026-05-11 after the first crystal-seed rewrite was still too void-prone.** The BM seeder (`bm_initial_network`, `_build_trivalent_proximity_graph`, `poisson_disk_pbc`, `random_3regular_graph`) remains removed. The default crystalline seed is now true `srs` (single-network gyroid), not the earlier `diamond3` fallback. This matters at the reference size: `srs` gives a seed bond length ≈0.809 µm for N=1000/L=11.44/d0=0.8, while `diamond3` gave ≈0.991 µm and required a distorting initial relax. The burn-in is also capped by accepted move involvements per vertex and uses a low-k structure-factor Metropolis penalty (`uniformity_weight`) so long-wavelength density modes do not grow unchecked.

**Void clustering during WWW — historical root cause.** During 100k WWW iterations the vertex distribution drifts: 4³-voxel std grew from 2.98 (BM seed) → 7.31 (final) for the user's N=1102, L=14.3 µm run, vs reference 2.79. Manifests as rod depletion at the cubic-cell corners/faces (8-corner / 8-centre count ratio ~0.36 vs reference 0.59). **The BM initial network is not the cause** — it's uniform out of the box; the drift accumulates during WWW.

**Why (corrected 2026-05-07):** The earlier diagnosis blamed the missing non-bonded term in the Sellers energy. That gap is real but secondary. The primary cause was that our `relax()` did **full-N L-BFGS** for both the post-SW relax and the periodic global polish, distinguished only by `max_iter`. Sellers's Supplement Methods (p. 6) explicitly delegates the relaxation refinements to Vink 2001 and Mousseau-Barkema 2001, which prescribe **spatially-local relaxation up to the 4-neighbour shell of the SW move, with global L-BFGS only as a rare fallback**. Our full-N L-BFGS let vertices anywhere in the cell migrate toward each other under the no-repulsion energy — the mechanism producing corner voids. Hemmann/Saba 2026 (arXiv:2601.10333) uses the same Vink/MB local-only scheme and explicitly observes that void clustering still grows monotonically with accepted MC moves even with their angular-modified energy (their Figure 8c) — confirming the relaxation scheme is the dominant lever, not the energy term.

**How to apply (updated 2026-05-07, second iteration):** The local-shell mask landed in commit 8716139, but a 50k-iter run with `relax_global_every=1000` still produced void clustering (4³ voxel std 9.72 vs reference 3.65; corner/centre ratio 0.95 vs 1.21). The local-shell relax alone is necessary but not sufficient: the **unconditional periodic full-N polish** in `www_anneal` (lines 1020-1023 of the pre-fix code) re-introduced the same drift, ~50× per run with no Metropolis check. Sellers/Vink/MB explicitly say global relax should be a *rare fallback*, not on a fixed schedule.

The completed fix:
1. Removed the unconditional `relax_global_every` schedule branch from `www_anneal`. Default is now 0; non-zero emits a `DeprecationWarning`.
2. Added `global_fallback_threshold: float = float('inf')` kwarg — when local-relax `dE > threshold`, run a one-shot full-N polish before Metropolis. Default `inf` keeps the gate off; users opt in with finite values (e.g. 5.0–20.0). A 0.0 default fires on every uphill move (≈70% of moves in smoke test) which is exactly the void mechanism — verified by smoke test and avoided.
3. Reduced end-of-run polish in `generate_lsu_network` from `relax_global_iters * 2` to `min(relax_local_iters, 50)`. One short pass settles bond-length residual without re-introducing drift.
4. `local_shell_depth=4` remains the default (Sellers/Vink/MB shell depth).

If the gated fix alone still leaves voxel std > 4.0, the documented fall-back is `seed_network=` (load `lsu_example_ends.txt` as a known-uniform starting topology, deferred per AskUserQuestion 2026-05-07). Last-resort fall-back is the `f5 = Σ_{(i,j) non-bonded} max(0, r_floor - |r_ij|)²` non-bonded soft repulsion with `r_floor ≈ 0.5·d0` — a deviation from Sellers Eq. 2 the user has approved in principle.
