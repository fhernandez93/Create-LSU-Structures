# Seed-Construction Change Correctness Review

**Scope:** Read-only correctness / production-safety review of the RANDOM
seed-construction change gated by `_RANDOM_SEED_CONSTRUCT`
(env `LSU_RANDOM_SEED_CONSTRUCT`, default `"0"` = OFF) inside
`random_seed_network_bm2000` (`lsu_network.py`).

**Verdict: PASS-WITH-CONCERNS.**
Flag-OFF (production) is byte-identical to the original algorithm — a clean
pass. Flag-ON (experimental, opt-in) is logically correct and provably
cannot emit an invalid graph, but its correctness rests on reasoning plus a
single smoke point; the concerns below are **non-blocking and scoped
entirely to the ON path**.

---

## 1. Logical correctness of the random branches

### Cycle traversal (`lsu_network.py:639-668`)
- `cand_uv` is built correctly (`:648-649`): it filters the k nearest
  (returned by `tree.query` in ascending-distance order) down to unvisited
  vertices `!= current`. `cand_uv[0]` is therefore the nearest unvisited.
- Random pick (`:654`): `cand_uv[int(rng.integers(0, len(cand_uv)))]`.
  `rng.integers(0, n)` yields `[0, n)`, a valid index into a length-`n`
  list. The branch is only entered when `cand_uv` is non-empty (`:650`), so
  `len(cand_uv) >= 1` and there is no empty-range / off-by-one risk.
- Deg-2 closure is preserved: every `step` calls `_add_edge(current, nxt)`
  (`:668`) and the post-loop closing edge `_add_edge(current, start)`
  (`:676`) is order-independent. The `deg == 2` assertion at `:684-690`
  guards this and is unchanged.
- No deadlock in the inner `while True`: if `cand_uv` is empty, `k_query`
  doubles up to `N` (`:658-663`); since exactly `step` of `N` vertices are
  visited, an unvisited one always exists, so a full-`N` query always
  populates `cand_uv`. The `RuntimeError` at `:659-662` only fires on a
  degenerate geometry, identical to the original.

### Loop-expansion pairing (`lsu_network.py:713-736`)
- Random order (`:718`): `order = rng.permutation(cand.shape[0])` is a valid
  permutation of `[0, cand.shape[0])`, the same index space the original
  `np.argsort(...)` produced — only the visitation order differs.
- All acceptance checks are **inside** the loop and unchanged
  (`:727-735`): `deg < 3` for both endpoints, `v not in nbr_sets[u]`
  (not-already-bonded), and `_would_make_short_ring(u, v, 4)` (girth >= 5,
  rejects 3- and 4-rings). Order changes *which* valid pairs are accepted
  first, never *whether* a pair is legal. Girth, dedup, and degree
  invariants hold identically under random order.

**Conclusion:** both random branches are logically correct. Indexing,
range, closure, and all validity checks are sound.

---

## 2. Production safety (flag OFF) — byte-identical

Confirmed against the uncommitted diff (`git diff HEAD -- lsu_network.py`):

- **Cycle, OFF branch (`:656`):** `d_nxt, nxt = cand_uv[0]`. The original
  code (shown in the diff) iterated `dists/idxs` in ascending-distance order
  and `break`-ed on the first `j != current and not visited[j]`. That first
  hit is exactly `cand_uv[0]`. Selection is identical.
- **Pairing, OFF branch (`:719-723`):** the `argsort` of PBC distances is the
  original code verbatim; the acceptance loop below it is untouched.
- **RNG-stream preservation:** this is the decisive point for "byte-identical"
  at the *stream* level, not just selection. Both OFF branches consume
  **zero** `rng` draws (just like the original). The random branches are the
  only new `rng` consumers, and they are unreachable when OFF. Therefore the
  downstream stochastic stages that share this same `rng` object —
  `topology_burn_in` (`:3942`) and `www_anneal` (`:3978`), plus the
  collapse-retry rebuild (`:3895`) — receive a bit-for-bit identical stream.
  Production behaviour is unchanged.

**No discrepancy found. Flag OFF reproduces the original exactly.**

---

## 3. Robustness / edge cases (flag ON)

The worst behaviours of random selection are real but **bounded by
deterministic, self-checking machinery that the flag never touches**:

- **Long closing edge.** Random-among-k-nearest can leave a near vertex
  unvisited and pick a farther one, and the final `current` may sit far from
  `start`, lengthening the closing edge (`:671-676`). This is a *topological*
  long bond, handled by the Step 3b 2-opt cleanup (`:884-961`), which is
  fully deterministic (`twoopt_k`-NN candidates, first-improvement, girth +
  connectivity guarded) and **flag-independent**. The smoke point
  (N=1000, seed=42) gave max edge 2.14·d0 vs the greedy 2.19·d0 — same
  regime, no blow-up.
- **More stragglers.** Random pairing order can leave more deg-2 stragglers
  than shortest-first. They drain through the force-pair fallback
  (`:828-868`) and, when a direct bond would force a triangle, the
  augment-relink move (`:769-826`). Both are deterministic and
  flag-independent. Force-pair across PBC is always reachable
  (max image distance `sqrt(3)·L/2`), guaranteeing termination.
- **Convergence is order-agnostic.** `rc` grows monotonically on no-progress
  and caps at `rc_max_frac·d0` → fallback (`:741-743`); `max_outer_passes`
  caps the outer loop (`:701`). Random order can slow convergence or lean
  harder on the fallbacks, but cannot prevent termination.
- **Output cannot be invalid.** Final guards are flag-independent and fire on
  any bad graph regardless of how it was built: edge-count `== 3N/2`
  (`:979`), `is_connected` (`:984`), and zero-triangle girth check
  (`:992-1000`). A clean ON build (smoke test) reached deg-3 everywhere and
  passed all three.

**This is the core safety argument:** random order can only produce a
*harder* instance for already-existing, self-validating fallbacks — never an
*invalid* graph.

**Residual concern (why PASS-WITH-CONCERNS, not PASS):** ON-path correctness
is established by reasoning plus a *single* empirical point (N=1000,
seed=42). Other N/seeds could in principle strand a longer closing edge or
leave many stragglers, and the Python force-pair / augment-relink fallbacks
are O(stragglers²). Under the no-run constraint this was not swept. The ON
path is sound-but-lightly-tested; the concern is non-blocking and does not
affect production (OFF).

---

## 4. Bugs / off-by-one / RNG misuse / stream interaction

- **No off-by-one or rng misuse.** `rng.integers(0, len(cand_uv))` and
  `rng.permutation(cand.shape[0])` use correct half-open ranges over the
  exact index spaces (verified §1).
- **RNG-stream interaction (ON path — inherent, not a defect).** The shared
  `rng` is reused downstream (`:3942`, `:3978`). Because the ON branches
  consume extra draws (the cycle pick consumes one integer per non-greedy
  step; `permutation` consumes draws per outer pass), an ON run's downstream
  burn-in / anneal stream differs from an OFF run's. This is the *expected*
  consequence of changing stochastic seed construction, not a bug, and it
  cannot affect production (OFF consumes zero draws — see §2). ON runs are
  simply not draw-comparable to OFF runs; the OFF stream is untouched.
- **No other defects found** in the reviewed range (`:474-1018`).

---

## Out-of-scope note for the caller

The same uncommitted working-tree diff also carries an **unrelated** energy
change, `_KEATING_F1F2` (`:1558`, `:1594`, `:1675`), which — unlike the seed
flag — defaults **ON**. It alters f1/f2 in `energy_components`. It was **not**
reviewed here (outside this task's scope); flagging only so the caller knows
the diff is not seed-construction-only.

---

## Verdict

**PASS-WITH-CONCERNS.**
- Flag OFF (production default): clean PASS — byte-identical selection *and*
  rng stream.
- Flag ON (experimental, opt-in): logically correct; provably cannot emit an
  invalid graph; concerns are limited to light empirical coverage (one
  N/seed) and the inherent rng-stream shift. All concerns are non-blocking
  and scoped entirely to the opt-in ON path.
