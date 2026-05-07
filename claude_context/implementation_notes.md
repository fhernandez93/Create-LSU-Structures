# `lsu_network.py` — Code Map and Performance Notes

## Module layout (top to bottom)
1. **Imports + JAX detection** — `HAS_JAX` flag. When JAX imports successfully
   we also call `jax.config.update("jax_enable_x64", True)` so L-BFGS-B sees
   float64 gradients.
2. **PBC helpers** — `pbc_displacement`, `coerce_box`.
3. **Topology** —
   - `poisson_disk_pbc` — hard-core uniform vertex placement under PBC
     (Barkema-Mousseau §II.A, the placement step).
   - `_build_trivalent_proximity_graph` — Stage A + Stage B builder (see
     `algorithm.md` §1.5). Stage A loop-expansion is the BM "+1 edge" move
     adapted for trivalent (single-visit Hamiltonian cycle); Stage B is the
     chord matching. Both stages are vectorised over a (|R|, |E|) cost
     matrix and a (|D2|, |D2|) chord matrix using a precomputed (N, N)
     boolean adjacency matrix `adj` — no Python loops over neighbour sets.
     Cost: ~22 s for N=1102 on CPU, dominated by Stage A's per-iteration
     fancy-indexing of `dists`. Both stages have fallbacks that allow
     longer-than-`r_cut` bonds rather than failing — the WWW relaxation
     then pulls them in.
   - `bm_initial_network` — outer wrapper that returns `(positions, edges)` for
     a connected, simple, 3-regular network with seed bonds clustered around
     `d0`. **This is the seed used by `generate_lsu_network`.**
   - `random_3regular_graph` — configuration model with rejection. Retained for
     reference / tests; **not used** in the default code path.
   - `is_connected` — BFS connectivity check.
   - `build_neighbors` — edge list → (N,3) neighbor index array.
   - `build_angle_triples`, `build_dihedral_quads` — index arrays for f2 / f3+f4.
4. **Energy** —
   - `total_energy(...)` — NumPy scalar U via vectorized ops (f1+f2+f3+f4).
   - `_energy_jax_full(...)` — same but in JAX, autodiffed and JIT-compiled.
   - `_value_and_grad_jit` — single module-level JIT'd value-and-gradient,
     keyed on shape. Shapes are invariant under Stone-Wales (n_edges, n_triples,
     n_quads stay constant), so JAX compiles **once** per (N, E) and reuses.
5. **Relaxation** —
   - `_RelaxContext` — caches device-side topology arrays and binds them to
     the JIT'd kernel. After a Stone-Wales move the WWW loop calls
     `ctx.update_topology(edges, neighbors)` to refresh the index arrays
     **without** triggering JAX retracing.
   - `relax(positions, ctx, max_iter)` — single L-BFGS routine.
     **Default JAX path:** scipy.optimize.minimize(L-BFGS-B) driving the
     JIT-compiled value_and_grad — host↔device cost is only ~26 µs per
     gradient, well below the L-BFGS-B per-iter overhead. **Opt-in
     `use_jaxopt=True`:** drives the inner loop with `jaxopt.LBFGS` and
     keeps everything on the JAX device. Topology arrays are passed as
     runtime args to `solver.run(...)` so the jaxopt JIT cache stays warm
     across Stone-Wales updates, and solver instances are cached on the
     ctx by `(maxiter, tol)` (each instance pays a 4–5 s compile on first
     call). Pure-NumPy path uses scipy with finite differences.
   - There's no separate "local" routine — vertices already at a local
     minimum have ~zero gradient and don't move, so a small `max_iter`
     after a SW move acts as an effective local relax while a large
     `max_iter` does a full polish.

   **CPU benchmark, BM-seeded N=40/200/600 vertices, 100-iter relax:**

   | N   | scipy + JIT | jaxopt LBFGS |
   | --- | ----------- | ------------ |
   |  40 |    15 ms    |   2060 ms    |
   | 200 |    26 ms    |   2160 ms    |
   | 600 |    46 ms    |   2200 ms    |

   Scipy + JIT scales linearly with N; jaxopt's per-call dispatch
   overhead is ~2 s regardless of size, so it loses by 50–150× on CPU at
   every scale. jaxopt is only worth turning on for GPU runs.
6. **Stone-Wales** —
   - `stone_wales_propose(neighbors, edges, rng)` — select valid bond switch.
   - `stone_wales_apply(...)` / `stone_wales_revert(...)`.
7. **WWW main loop** —
   - `www_anneal(positions, neighbors, edges, box, ...)` — Metropolis loop with
     geometric temperature schedule and periodic full relaxations. Uses one
     long-lived `_RelaxContext`.
8. **LSU computation** —
   - `_build_tree(vertex, depth, neighbors, positions, box)` — BFS to depth n,
     storing edge vectors in unwrapped local frame.
   - `_phi_for_permutation(...)` — Eq. 3 score for one root-edge permutation
     σ, with depth-first greedy interior pairing.
   - `phi_ab(tree_a, tree_b)` — average over all γ! permutations (Eq. 1).
   - `compute_lsu(positions, neighbors, edges, box, depth, locality, sample)`
     — average ϕ over (a,b) pairs.
9. **Output formatting** — `network_to_rods(positions, edges, box,
   pbc_duplicate_boundary_rods=True)`. With duplication on (default) each
   face-crossing edge is emitted twice — once anchored at each canonical-box
   endpoint — matching the Sellers reference convention and producing a
   periodic permittivity grid downstream. `pbc_duplicate_boundary_rods=False`
   gives one row per unique edge (legacy behaviour, kept for callers that
   PBC-tile downstream themselves).
10. **Public entry point** — `generate_lsu_network(...)`.

## Performance (measured, Windows 11, CPU only)
Benchmark: 60 rods / 40 vertices, 300 WWW iterations, identical seed.

| Path  | Wall time | Notes                                            |
| ----- | --------- | ------------------------------------------------ |
| JAX   |    2.4 s  | First-call JIT compile ~0.5 s, then ~5 ms/iter   |
| NumPy |  269 s    | scipy finite-difference gradient is the cost.    |

⇒ ~110× speedup with JAX even on a small system. Gap widens at full
scale because JAX cost is dominated by JIT-compiled vectorised ops while
NumPy finite-diff scales as O(3N · n_funeval) per L-BFGS step.

**relax_local_iters sensitivity (benchmarked on N=1102, GPU):**

| relax_local_iters | ΔE typical | acceptance | per-relax time |
| ----------------- | ---------- | ---------- | -------------- |
|  30               | +3 to +8   | ~1%        | 23 ms          |
| 100               | -4 to 0    | ~100%→40%  | 84 ms          |
| 200               | -20        | 100%       | 169 ms         |
| 500               | -60        | 100%       | 456 ms         |

**At 30 iters, ΔE is positive for ALL moves** — L-BFGS cannot escape the
SW-perturbed high-energy state in 30 steps, so Metropolis always rejects.
This is the root cause of the 1% acceptance rate and phi22 stuck at 0.49.
**Use relax_local_iters ≥ 100** (ΔE crosses zero between 30 and 100 iters).

**jaxopt vs scipy+JAX on GPU for N=1102:**

| solver       | 30 iters | 100 iters | 200 iters | 1000 iters |
| ------------ | -------- | --------- | --------- | ---------- |
| jaxopt GPU   | 1305 ms  | 1323 ms   | 1447 ms   | 1700 ms    |
| scipy+JAX    |   23 ms  |   84 ms   |  169 ms   |  744 ms    |

jaxopt has ~1.3 s fixed dispatch overhead for N=1102 — scipy+JAX is faster
at every iteration count. jaxopt only wins for very large N or very high
iteration counts (>2000). The note about "GPU use → jaxopt" is superseded
by this benchmark; leave use_jaxopt=False for this system.

Estimated full-case run (1653 rods / 1102 vertices, 100k WWW iterations,
relax_local_iters=100, scipy+JAX GPU):
- ~2.3 hours (95k relax calls × 84 ms each)
- NumPy: don't.

## Why retracing was the old bottleneck (and how it's fixed)
Earlier `relax_local` built a fresh closure each call (capturing per-move
`pos_full`, `moving_idx`, sliced topology arrays), then called `jit` on that
closure. JAX caches by Python function identity, not by content, so each call
recompiled the kernel — JIT overhead dwarfed the optimisation itself.

Fix: a single jit'd `value_and_grad(_energy_jax_full, argnums=0)` lives at
module scope. Topology and box are passed as *runtime* arguments (not
closure captures), and `_RelaxContext` keeps the device-side arrays stable
across calls. Shapes never change, so JAX compiles exactly once per Python
process per (N, n_edges) pair.

## Notes / gotchas
- `jax_enable_x64=True` is set process-wide. Any other JAX code in the same
  interpreter will also default to float64.
- BM initial network requires the box to support a hard-core packing of N
  vertices at separation `0.7·d0`. Density `N/L³` should be at most
  `~1/d0³` (gyroid-like packing). The Sellers example sits at
  N/L³ = 1102/1497 ≈ 0.74 vertices/µm³, well within the limit.
- **Bond collapse**: the Sellers energy (Eq. 2 in the supplement) has no
  explicit non-bonded repulsion, only the Keating bond-stretch term f1.
  After many SW moves L-BFGS sometimes settles into a local minimum where
  one bond's length is near zero. The collapsed bond gets weight in f1 of
  only `(0 - d0)² = 0.64` per bond — easily masked by the rest of the
  energy. Mitigation: run more WWW iterations / more global relaxes to
  give the chain more chances to escape the bad basin. Long-term fix
  would be adding a soft repulsion term to U; deferred to keep parity
  with the published energy.
- After every Stone-Wales (whether accepted or reverted) we call
  `ctx.update_topology(...)` to keep the device-side `edges_j/triples_j/quads_j`
  consistent with the host arrays.
- L-BFGS preserves energy under PBC translation, so positions can drift
  outside `[-L/2, L/2]^3` over many iterations. We wrap back into the
  canonical box after each global relax (in `www_anneal`) and at the end
  of `generate_lsu_network`.
- LSU computation has O(N · k · γ!) cost where k = average size of locality-l
  neighborhood. For large N pass `max_lsu_check_pairs` to subsample.
- The temperature schedule is geometric:
  T_k = T0 · (T_final / T0)^(k / n_iterations).
