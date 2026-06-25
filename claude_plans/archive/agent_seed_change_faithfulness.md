# Seed-Change Faithfulness Verification — `_RANDOM_SEED_CONSTRUCT`

Read-only verification. No code modified, nothing run on the GPU. Builds on the prior
audit `claude_plans/agent_seed_review_faithfulness.md`.

**The change under test.** A module flag `_RANDOM_SEED_CONSTRUCT`
(`lsu_network.py:1566`, default OFF via `LSU_RANDOM_SEED_CONSTRUCT=0`) gates two —
and only two — branches inside `random_seed_network_bm2000`:

- **(a) cycle traversal** (`lsu_network.py:651`–`656`): instead of greedy-nearest
  (`cand_uv[0]`), pick a RANDOM unvisited vertex among the k nearest:
  `cand_uv[int(rng.integers(0, len(cand_uv)))]`.
- **(b) loop-expansion pairing** (`lsu_network.py:714`–`723`): instead of
  shortest-first (`np.argsort(...)`), iterate the in-`rc` candidate pairs in a
  RANDOM permutation: `order = rng.permutation(cand.shape[0])`.

`grep` confirms the flag appears **only** at lines `651`, `714`, `1566` — so every
other step (Poisson placement, cycle architecture, girth filter, force-pair,
augment-relink, 2-opt, settle) is byte-for-byte unchanged by the flag.

**Primary source (verified by `pdftotext`).** BM2000 = Barkema & Mousseau, *Phys.
Rev. B* **62**, 4985 (2000), `LSU Literature/PhysRevB.62.4985.pdf`, §II.A
"Generating random initial CRN's" (pp. 4986–4987), §II.C "Efficient quenching"
(p. 4987).

---

## 1. Does it remove the crystallinity bias (deviation #1)? — PARTIALLY

Deviation #1 was "greedy nearest-neighbour cycle + shortest-first deg-2 pairing
replacing BM2000's RANDOM selection." The change targets exactly the two selection
rules. Assessed per branch:

### Branch (b), the loop-expansion pairing — GENUINELY BM2000-faithful on locality.

BM2000 §II.A states (verbatim, extracted from the PDF):

> "The expansion of the loop is achieved by **randomly selecting** a group of three
> atoms A, B, and C, such that A is not fourfold coordinated and is **within a
> distance of r_c from B and C** but not bonded to either, while B and C are bonded."

The crucial point: **BM2000's own "random" is random among in-`r_c` candidates.**
The `r_c` window is not an extra bias the code injects — it is BM2000's defined
search neighbourhood. So `order = rng.permutation` over the in-`rc` candidate set
(`lsu_network.py:714`–`718`, built from `tree.query_pairs(r=rc, ...)` at `709`) is a
faithful reproduction of BM2000's random selection on the locality axis. The task's
framing — "restricting to within-rc still injects spatial-locality bias" — is
**incorrect on this point**: restricting to within-`rc` is precisely what BM2000 does.
The `rc`-growth mechanism (`736`/`744`) is also BM2000's ("set to some small value
like 3 Å, but then it is gradually increased until all atoms are fourfold
coordinated"). On selection-randomness within its own move class, branch (b) is faithful.

**BUT — the random pick is still filtered by girth≥5, and that filter is
flag-INDEPENDENT and anti-BM2000.** The `_would_make_short_ring(u, v, 4)` test sits at
`lsu_network.py:731`, *inside* the shared `for idx in order` loop (`724`–`736`). Only
`order` is gated by the flag (`714` vs `719`); the girth reject is not. So even with
the flag ON, the effective selection is "random among in-`rc` candidates **that also
close no 3- or 4-ring**." BM2000 §II.C says the opposite:

> "we find that especially for quenching the relaxation is significantly helped **by
> allowing for four-membered rings** … At the end of the quenching, **the few
> four-membered rings that are created can easily be removed one by one**."

So the residual non-randomness in the pick is **not** the `rc` window (that is faithful)
— it is the build-time girth≥5 filter (deviation #3), which the change does not touch.
This is the sharpest finding: deviation #3 is now the *only* thing keeping branch (b)'s
selection from being truly BM2000-random.

### Branch (a), the cycle traversal — "less greedy," but there is nothing to be faithful TO.

Randomizing the k-nearest pick (`651`–`654`) does reduce the greedy spatial-locality
ordering — strictly an improvement over `cand_uv[0]`. But the **Hamiltonian-cycle
architecture itself has no BM2000 counterpart** (deviation #2; BM2000 grows a 4-atom
loop by vertex insertion, it never builds a spanning cycle). You cannot call a step
"BM2000-faithful" when BM2000 has no analogous step. Two caveats sharpen this:

- The random pick is confined to the **k nearest** unvisited vertices, where
  `k = max(8, step//64 + 8)` (`lsu_network.py:641`). For most of the walk this is
  k≈8 — a much tighter window than BM2000's *growing* `r_c`. So branch (a)'s
  "random" is over a deliberately local candidate set chosen "to keep edges
  local/bounded" (comment `652`–`653`). That is a defensible engineering choice (it
  prevents wild long cycle edges) but it is not BM2000's procedure.
- The cycle supplies **N of the 1.5N edges (two-thirds)** of the seed. So two-thirds
  of the connectivity still comes from a non-BM2000 construction, regardless of the flag.

### Verdict on #1.

The change removes the *greedy/shortest-first* half of deviation #1: both selection
rules are now random. The pairing branch (1/3 of edges) is genuinely BM2000-random on
locality. But #1 is not fully resolved, because (i) the cycle branch (2/3 of edges)
randomizes a step BM2000 doesn't have, over a tight k≈8 window, and (ii) the random
pick is still filtered by the anti-BM2000 girth≥5 rule. **#1 is improved, not closed.**

---

## 2. Which audit deviations remain UNADDRESSED?

All confirmed untouched by the flag (grep: flag only at `651`/`714`/`1566`):

| # | Deviation | Lines | Still materially breaks faithfulness? |
|---|-----------|-------|----------------------------------------|
| **#2** | Hamiltonian-cycle-then-pair architecture vs BM2000 grow-a-4-atom-loop vertex insertion (BC→AB+AC) | `624`–`742` | **YES — dominant residual structural deviation.** 2/3 of edges come from this. BM2000's move class is vertex insertion into an existing bond (`731` audit quote: "the bond BC is replaced by bonds AB and AC"); the code never inserts a vertex, it directly bonds two deg-2 vertices (`733` `_add_edge(u, v)`). The change randomizes the *order* of this non-BM2000 move; it does not make the move BM2000's. |
| **#3** | Build-time girth≥5 enforcement | `599`–`617`, `731`, `984`–`992` | **YES — and now load-bearing.** As shown in §1, `_would_make_short_ring` at `731` filters even the random pick. Contradicts BM2000 §II.C ("allowing for four-membered rings … removed one by one" at the end). This is the specific reason branch (b)'s random selection is not truly BM2000-random. |
| **#4** | 2-opt long-bond repair + force-pair / augment-relink | `744`–`966` | Engineering additions, no literature trigger; bias is local and touches few edges. Unchanged. Materially a deviation but lower-impact than #2/#3. |
| **#5** | `settle_seed_with_repulsion` / `soft_start_seed_relax` non-bonded repulsion | `1013`–`1328` | Geometry-only; does not change the edge list. Lowest topological impact. Unchanged. |

So #2, #3, #4, #5 are all unaddressed. #2 and #3 still materially break faithfulness
even after the change; #4 and #5 are lower-impact (#4 local, #5 geometry-only).

---

## 3. Faithfulness vs the user's GOAL (S(k), angles) — the change is ORTHOGONAL.

The user's actual targets are the structure factor S(k) (hyperuniformity, reference
S(k0)=0.041) and angle distributions (reference bond-angle mean 120°, std 8.41°), NOT
ring statistics. The smoke test reports the new random seed has **S(k0)=0.1413 —
identical to the old seed**. This is expected, and in fact **bit-identical, not merely
"≈ identical"**:

- `low_k_structure_factor` (`lsu_network.py:1432`–`1456`) operates **only on vertex
  POSITIONS**: line `1453` `phases = 2π (pos/box) @ hkl.T`, then `amp =
  exp(1j·phases).sum`. It never reads the edge list. S(k) is a function of point
  placement, full stop.
- Placement is **Step 1** (`lsu_network.py:530`–`559`), which runs entirely *before*
  the flag-gated code at `651`/`714`. The placement RNG stream is therefore unchanged
  by the flag, so `positions` is byte-for-byte identical whether the flag is ON or OFF.
- The change is a **connectivity-only** change (the module comment at `1564`–`1565`
  says exactly this). Connectivity does not move Poisson-disk vertices.

Therefore S(k0) is provably invariant under this flag. The fix **cannot** help the
user's S(k) goal. It is a **pure faithfulness/topology improvement** (random vs greedy
selection → less seed crystallinity in the ring spectrum), not a hyperuniformity lever.

Likewise the bond-angle distribution depends on geometry (the post-relax positions
under the Keating/Sellers energy), so the connectivity-ordering change at seed time has
no direct, first-order effect on the angle targets either.

### If S(k0) is the goal, the actually-relevant levers are:

1. **Vertex placement** (`_poisson_disk_pbc`, `lsu_network.py:1331`–`1380`, called at
   `534`). S(k0) is set by the point pattern. Replacing sequential Poisson-rejection
   with a genuinely near-hyperuniform / stealthy point process (e.g. collective-
   coordinate / Fourier-space-constrained points) would lower the seed's intrinsic
   S(k0) toward the 0.041 target. This is where the lever actually lives.
2. **The uniformity penalty during the WWW run** (`_acceptance_objective`,
   `lsu_network.py:1459`–`1470`; `low_k_structure_factor` in the Metropolis objective):
   penalizing the lowest reciprocal modes during acceptance suppresses the corner/face
   voids. Project memory (N=512 validation) records this is the lever that fixes the
   low-k void — at a cost (it trades against 6-rings; temperature is the dominant ring lever).

Note (consistency with established findings): "emergence over long pure WWW" is **not**
offered as a live lever — project memory records pure-WWW emergence was **refuted at
N=1000** (α<0, coarsens to 9-rings; 6-ring≥5% hard floor). So the two real S(k) levers
are placement (1) and the uniformity penalty (2), not pure-WWW emergence.

---

## Verdict

**PARTIAL faithfulness improvement; does NOT achieve faithful BM2000 random
construction; orthogonal to the user's S(k) goal.**

- The change makes **both** seed selection rules random, eliminating the greedy/
  shortest-first bias that was deviation #1's substance. The **pairing branch is
  genuinely BM2000-faithful on locality** — its `rc` window is BM2000's own
  ("within a distance of r_c", §II.A), so "random among in-`rc` candidates" matches
  BM2000's "randomly selecting." The task's worry that the `rc`/k restriction "still
  injects spatial-locality bias" is **not** the real problem.
- It does **not** achieve faithful BM2000 construction because: **#2** the cycle-then-
  pair architecture (2/3 of edges, no BM2000 analogue) is untouched and is the dominant
  residual deviation; **#3** the random pick is still filtered by build-time girth≥5
  (`731`), which is flag-independent and contradicts BM2000 §II.C ("allowing for
  four-membered rings … removed one by one") — this is the actual reason the selection
  is not truly BM2000-random; **#4/#5** (2-opt/force-pair/settle) untouched (lower impact).
- It is **provably orthogonal** to the user's stated targets: S(k0) is byte-for-byte
  identical (0.1413) because placement precedes the gated code and `low_k_structure_factor`
  reads only positions. If S(k0)=0.041 is the goal, the levers are **placement**
  (near-hyperuniform points, `_poisson_disk_pbc:1331`) and the **uniformity penalty**
  (`_acceptance_objective:1459`), not seed connectivity.

**Bottom line:** a legitimate, correctly-motivated *faithfulness* refinement of the
seed's selection randomness — and the pairing half is genuinely faithful — but it
leaves the two highest-impact structural deviations (#2 architecture, #3 girth) in
place, so it is not a faithful BM2000 §II.A construction, and it has zero effect on the
user's hyperuniformity / angle objectives.
