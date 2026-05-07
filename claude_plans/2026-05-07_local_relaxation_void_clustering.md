# Restore Sellers/Vink/Mousseau-Barkema local-only relaxation

## Context

Run with `α=80, β=5, γ=1, δ=0.5`, 100k WWW iters, N=1102 vertices, L=14.3 µm produced a network with corner/face void clustering: rod-midpoint 4×4×4 voxel std = 10.67 vs reference 2.79; 8-corner / 8-centre count ratio = 0.36 vs reference 0.59. The reference `lsu_example_ends.txt` is uniform.

### Why it happened

Sellers's supplement (`LSU Literature/41467_2017_BFncomms14439_MOESM1815_ESM.pdf`, Methods, p. 6) describes only "the network is then relaxed" and delegates the relaxation refinements to refs [13,14] — Vink et al. 2001 (PRB 64, 245214) and Mousseau-Barkema 2001. The Vink/MB recipe is **spatially-local relaxation up to the 4th-neighbour shell of the SW move, with global L-BFGS only as a rare fallback** (confirmed by Hemmann 2026 p. 4 quoting the same refs).

Our [`relax`](lsu_network.py#L703) does NOT do this. Both `relax(positions, ctx, max_iter=relax_local_iters)` at [`www_anneal:908`](lsu_network.py#L908) and the periodic call at [`www_anneal:924`](lsu_network.py#L924) run **full-N L-BFGS** over all vertices, distinguished only by `max_iter`. With the Sellers energy having no non-bonded vertex repulsion, full-N L-BFGS lets vertices anywhere in the cell migrate toward each other when it lowers bond/angle energy — which is the mechanism producing the corner voids.

The other candidates (initial network, weight tuning) are ruled out:
- BM seed is uniform (memory: voxel std ≈ 2.98 right after seeding); drift accumulates during WWW.
- No choice of `{α,β,γ,δ}` introduces a restoring force against non-bonded vertex clustering. Memory's Option-A trial `{10, 2, 5, 5}` already failed.

So the fix is to make `relax` actually local: freeze every vertex outside the 4-neighbour BFS shell of the SW seed. This is what Sellers cites and what Hemmann re-implements.

## Implementation plan

### Files to modify
- [`lsu_network.py`](lsu_network.py) — `_energy_jax_full` + JIT wrapper, `_RelaxContext`, `relax`, `www_anneal`, `generate_lsu_network`.
- [`Create_LSU_Function.ipynb`](Create_LSU_Function.ipynb) — example call updated to expose the new `local_shell_depth=4` parameter.

### Step 1 — BFS helper for the 4-shell
Add `compute_local_shell_mask(seed_vertices, neighbors, depth, N) -> bool[N]`:
- Initialise `mask = False[N]`; set `mask[seed_vertices] = True` (seed = the four SW-affected vertices `{i, j, c, d}` returned by `stone_wales_propose`).
- Repeat `depth` times: `mask |= any(mask[neighbors[i]] for each vertex i)` — vectorised by gathering `neighbors` for all True vertices and OR-ing.
- Cost: O(N × depth) booleans per SW move. Cheap (~µs for N=1102).
- Place near `build_neighbors` ([`lsu_network.py`](lsu_network.py) topology section).

### Step 2 — Gradient masking, not slicing
Keep all N positions as L-BFGS variables; multiply the gradient by the `(N, 3)` boolean mask so out-of-shell components are zero → L-BFGS doesn't move them. This preserves the JIT cache (shapes invariant; mask is a runtime arg of fixed shape `(N,)` or `(3N,)`).

Mathematically equivalent to constraining out-of-shell vertices: bonds connecting a moving vertex to a frozen one still contribute energy, but only the moving endpoint sees a non-zero gradient. Bonds entirely inside the frozen region are constant and contribute zero gradient.

### Step 3 — JAX kernel
[`_energy_jax_full`](lsu_network.py#L556) returns scalar U; the masking happens at the gradient level. Modify [`_value_and_grad_jit`](lsu_network.py#L602) wrapper:
```
def _jax_value_and_grad(pos_flat, edges_j, ..., w_j, mask_flat):
    e, g = _value_and_grad_jit(pos_flat, edges_j, ..., w_j)
    return float(e), np.asarray(g * mask_flat, dtype=np.float64)
```
where `mask_flat` is `mask[:, None].repeat(3, 1).reshape(-1)`. When `mask_flat` is all-ones, behaviour is identical to today.

JIT wrapper signature gains one runtime argument (`mask_flat`); shapes are constant; no retrace.

### Step 4 — `_RelaxContext` and `relax`
- Add `ctx._mask_flat_j` (JAX device array), default all-ones (full-N).
- Add `ctx.set_moving_mask(mask: np.ndarray | None)` that pushes the mask to device. `None` resets to all-ones.
- Modify [`relax`](lsu_network.py#L703) to thread the current mask through `value_and_grad`. No new parameter — caller controls via `ctx.set_moving_mask(...)` before invoking.

### Step 5 — `www_anneal`
At [`www_anneal:903`](lsu_network.py#L903) (after `update_topology`), before [`www_anneal:908`](lsu_network.py#L908):
```
seed = np.array([move.i, move.j, move.c, move.d])
mask = compute_local_shell_mask(seed, neighbors, depth=local_shell_depth, N=N)
ctx.set_moving_mask(mask)
new_pos, E_new = relax(positions, ctx, max_iter=relax_local_iters)
```
For the periodic global polish at [`www_anneal:924`](lsu_network.py#L924), set `ctx.set_moving_mask(None)` first so it relaxes the whole network as before.

Add new kwarg `local_shell_depth: int = 4` to `www_anneal` and thread up to `generate_lsu_network`.

### Step 6 — Vink/MB-style global fallback (optional, deferred)
Vink/MB also globally relax when local relax fails to lower energy. Implement only if step 1–5 don't fully fix the voids: add a check `if E_new > E_curr + threshold: ctx.set_moving_mask(None); new_pos, E_new = relax(positions, ctx, max_iter=relax_global_iters)`. Skip in v1.

### Step 7 — NumPy path
[`total_energy`](lsu_network.py#L525) is used only when `use_jax=False`, with finite-difference gradient via scipy. Easiest: in the NumPy branch of `relax`, hold out-of-shell positions fixed by passing a sliced sub-vector to `scipy.optimize.minimize`, reconstructing the full position vector inside the closure. Acceptable: this path is rarely used (slow on N=1102) and shape changes don't matter without JIT.

### Step 8 — Notebook
Update the call in [`Create_LSU_Function.ipynb`](Create_LSU_Function.ipynb) to keep current weights but add `local_shell_depth=4`. No other parameter changes — the whole point is that the original weights and iteration count should now produce a uniform network.

## Verification

1. **Smoke (~2 min):** N=40 system, 1000 WWW iters, `seed=42`, `local_shell_depth=4`. Confirm:
   - Acceptance rate ≥ 30% (was ~40-100%; should not regress badly).
   - Φ_12 reaches its previous level within ±0.02.
   - End-of-run min rod length > 0.5·d0.

2. **Mid-scale (~10 min):** N=600 vertices, 10000 WWW iters, `seed=42`. Compute rod-midpoint 4×4×4 voxel std. Target: ≤ 3.5 (vs reference 2.79).

3. **Full reproduce (~2.3 h on the user's machine):** original failing parameters (1653 rods, L=14.3, 100k iters, weights `{80, 5, 1, 0.5}`, `seed=42`) plus `local_shell_depth=4`. Regenerate via `create_permittivity_grid_penlike` to produce `Generated_Example_3.h5`. Acceptance criteria:
   - Rod-midpoint 4×4×4 corner-sum / centre-sum ratio ∈ [0.5, 0.7] (reference: 0.59).
   - Vertex 4×4×4 voxel std ≤ 3.5 (reference: 2.56; failing run: 6.92).
   - Φ_22 = 0.89 ± 0.01 (the LSU target — should still be reachable since each SW move can still propagate through subsequent moves).
   - End-of-run min rod length > 0.5·d0.
   - Visual comparison of `Generated_Example_3.h5` vs `lsu_example.h5` shows no corner voids.

4. **Regression check:** rerun the notebook's existing example with `local_shell_depth=None` (or whatever sentinel disables masking) and confirm output is bit-identical (within float tolerance) to the previous run with the same seed.

## Memory follow-up (after exiting plan mode)

Two memory entries are misleading and should be corrected:

- [`memory/project_known_issues.md`](memory/project_known_issues.md): the "Void clustering" entry attributes the issue to the missing non-bonded term in the Sellers energy. The deeper cause is that our full-N L-BFGS deviates from the Vink/Mousseau-Barkema relaxation that Sellers cited; the energy-functional gap is real but secondary. Update Option A/B framing accordingly and add Option C = restore local relax.
- [`memory/reference_lsu_literature.md`](memory/reference_lsu_literature.md): the Hemmann/Saba 2026 entry says they "add non-bonded repulsion via an angular trick". They don't — the 180°-target term is over **bonded** angle triples, just with a coordination-independent equilibrium. They also explicitly observe void clustering in their generated networks (Figure 8c of the paper) and don't solve it via the energy. Correct the entry to reflect the true mechanism.

## Out of scope

- Adding a non-bonded `f5` term. Reserve as fallback if local-relax alone doesn't reach the voxel-std target.
- Switching to Hemmann's 180°-equilibrium bond-angle term (no benefit for our Z=3 case).
- Reciprocal-space "collective coordinate control" hyperuniformity methods that Hemmann mentions as the principled fix — much larger rewrite, not needed for the LSU target.
- BM seeder, Stone-Wales selector, LSU computation, output formatting — none implicated.
