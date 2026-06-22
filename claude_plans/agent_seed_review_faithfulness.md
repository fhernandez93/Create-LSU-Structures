# Seed-Builder Faithfulness Audit — `random_seed_network_bm2000`

Read-only audit. No code was modified, nothing was run on the GPU.

**Scope.** `lsu_network.py`:
- `random_seed_network_bm2000(...)` — `474`–`1010`
- `_poisson_disk_pbc(...)` — `1331`–`1380`
- `soft_start_seed_relax(...)` — `1013`–`1148` (NumPy seed pre-relax)
- `settle_seed_with_repulsion(...)` — `1151`–`1328` (JAX seed settle)
- `build_neighbors(...)` — `96`–`107`

**Literature read (verbatim quotes used below).**
- **BM2000** = Barkema & Mousseau, *Phys. Rev. B* **62**, 4985 (2000) — `LSU Literature/PhysRevB.62.4985.pdf`. §II.A "Generating random initial CRN's" (pp. 4986–4987); §II.B "Avoiding complete relaxation of trial configurations" (p. 4987, threshold-energy quench acceleration, Eqs. 3–4); §II.C "Efficient quenching" (p. 4987, where the four-membered-ring "allow-then-remove" discussion actually lives).
- **Sellers** = supplement `LSU Literature/41467_2017_BFncomms14439_MOESM1815_ESM.pdf`, "Supplementary Methods → Amorphous Gyroid Simulated Annealing" (p. 6).
- **WWW1985** = Wooten, Winer, Weaire, *PRL* **54**, 1392 (1985) — `LSU Literature/PhysRevLett.54.1392.pdf`.
- **MB2001** = Mousseau & Barkema, *Curr. Opin. Solid State Mater. Sci.* **5**, 497 (2001) — `LSU Literature/Sci. 5, 497–502 (2001).pdf` (this IS Sellers's cited ref [14]).

---

## 1. Step-by-step classification table

**Taxonomy legend** (mapping to the three requested buckets FAITHFUL / EXTENSION / SHORTCUT-or-DEVIATION): "ENGINEERING ADDITION" = a sub-type of **EXTENSION** — code with no literature counterpart, added to make the custom pipeline robust. "DEVIATION" = SHORTCUT-or-DEVIATION (departs from what the cited source specifies).

| # | Code step | Lines | Classification | Literature it is compared against |
|---|-----------|-------|----------------|-----------------------------------|
| 1 | Poisson-disk placement, reject within `min_sep*d0`, lower min-sep on deadlock | `529`–`558`, `1331`–`1380` | **FAITHFUL** | BM2000 §II.A p.4986: "we **randomly place** all atoms in a cubic box with periodic boundary conditions, under the constraint that **no two atoms be closer than 2.3 Å.**" MB2001 p.499 pt.1: "we start from a truly random configuration whereby the atoms are placed at random locations in a periodic box at the crystalline density." Code's min-separation rejection = BM2000's hard-core constraint; the default `min_separation_frac=0.98` (line `479`) matches BM2000's 2.3/2.35 = 0.979 ratio almost exactly. |
| 2a | Build a **Hamiltonian cycle** spanning all N as the deg-2 starting graph | `624`–`688` | **DEVIATION (architecture)** | BM2000 §II.A starts from "a loop visiting **four atoms**" and *grows* it; the code starts from a single loop already visiting **all** atoms. Cycle-then-pair is not BM2000's construction. See §2(a). |
| 2b | Cycle order = **greedy nearest-unvisited** under PBC | `638`–`667` | **DEVIATION (bias)** | BM2000 §II.A: loop expansion done by "**randomly selecting** a group of three atoms"; its whole point is "it contains absolutely **no trace of crystallinity**." Greedy-NN injects spatial-locality ordering that BM2000 explicitly avoids. See §2(a). |
| 3a | Loop expansion to Z=3 = global **shortest-first** matching of deg-2 pairs within `rc`; bond two under-coordinated vertices directly | `690`–`742` | **DEVIATION (move class + bias)** | BM2000 §II.A expands by **inserting** a vertex into an existing bond (`BC → AB + AC`), repeated until each atom is "visited exactly twice"; it **never directly bonds two under-coordinated vertices**, and selection is **random**, not shortest-first. See §2(b). |
| 3b | `rc` grows by `rc_grow_frac*d0` when a pass makes no progress, up to `rc_max` | `696`, `733`–`742` | **FAITHFUL (in spirit)** | BM2000 §II.A: "Initially, `r_c` is set to some small value like 3 Å, but then it is gradually increased until all atoms are fourfold coordinated." The growth *mechanism* matches; it is bolted onto a non-BM2000 pairing step. |
| 3c | Girth ≥ 5 enforced **at build time**: `_would_make_short_ring(u,v,4)` rejects 3- and 4-rings on every bond | `599`–`617`, `723`–`724` | **DEVIATION (misattributed; opposite stance)** | The docstring (`504`–`505`, `599`–`607`) cites "BM2000 §II.B's no-four-ring rule." There is **no four-ring rule in §II.B** (that section is threshold-energy quench acceleration). The four-ring discussion is in **§II.C**, is about *quenching* not construction, and **allows** 4-rings then removes them afterward. See §2(c). |
| 3d | Force-pair fallback: pair leftover stragglers across PBC (any distance); prefer girth≥5, else allow a 4-ring, never a triangle | `744`–`868` | **DEVIATION / engineering** | No BM2000 analogue. BM2000 lets `rc` grow until coordination completes; it has no "pair across the whole box" terminator. See §2(d). |
| 3e | `_augment_pair`: when stragglers can only form a triangle, break a nearby edge `(x,y)` and relink `{(i,x),(j,y)}` preserving deg-3 | `761`–`818` | **EXTENSION (borrowed move, new trigger)** | Move-type echoes BM2000 §II.A's artifact swap ("replace a bond of each of these atoms by a bond between these atoms and another bond between their neighbors, conserving four-fold coordination"). But BM2000 triggers it on *close-but-unbonded geometric artifacts*; here it is a *topological deadlock terminator*. Different purpose. See §2(d). |
| 3f | 2-opt long-bond repair: swap a long edge + nearby edge for the shorter reconnection, deg-preserving, girth≥5, connectivity-checked, longest-first | `876`–`966` | **DEVIATION / engineering** | Not in BM2000, WWW1985, Sellers, or MB2001. Same 2-out-2-in swap *shape* as BM2000's artifact swap and the WWW move, but used to repair long bonds the code itself created. See §2(d). |
| 4 | Invariants: E = 3N/2, connected, canonical/sorted edges, **zero-triangle** guard | `968`–`992` | **FAITHFUL (good practice)** / partly **EXTENSION** | Degree/connectivity invariants are standard. The explicit zero-triangle assertion enforces the build-time girth rule that is itself a deviation (3c). |
| 5 | `meta` (bond length, rc_final, passes, n_triangles) | `1000`–`1010` | **FAITHFUL (bookkeeping)** | n/a |
| S1 | `soft_start_seed_relax`: harmonic bonds → d0 **+ one-sided soft-sphere non-bonded repulsion** (NumPy path) | `1013`–`1148` | **ENGINEERING ADDITION** | Neither BM2000 (Keating, Eq. 2) nor Sellers (Eq. 2) has non-bonded repulsion. Docstring (`1040`–`1043`) justifies it via "Vink 2001; Mousseau-Barkema 2001 repulsive equilibration" — a **code-claimed** justification I did **not** verify (Vink2001 not read; MB2001 read but does not specify a repulsive seed equilibration). |
| S2 | `settle_seed_with_repulsion`: Sellers Eq. 2 **+ decaying soft-sphere repulsion** annealed toward a small floor (JAX path) | `1151`–`1328` | **ENGINEERING ADDITION** | Same as S1; same unverified Vink/MB2001 justification (`1178`–`1182`). Geometry-only (does not change topology). |
| B | `build_neighbors` adjacency (N,3) | `96`–`107` | **FAITHFUL (utility)** | n/a |
| P | `_poisson_disk_pbc` sequential-rejection sampler | `1331`–`1380` | **FAITHFUL** | Implements BM2000's hard-core random placement. Sequential rejection (not blue-noise dart-throwing) is an implementation choice, immaterial to topology. |

---

## 2. Targeted questions

### (a) Does BM2000 start from a Hamiltonian cycle, and is its traversal greedy-NN or random? Does the code match?

**BM2000 does NOT start from a Hamiltonian cycle.** §II.A (p.4986) verbatim:

> "We achieve this by **starting with a loop visiting four atoms** somewhere in the configuration, in such a way that each pair of atoms that are neighbors along the loop be not separated by more than a cutoff distance `r_c`. This loop is **gradually expanded until it visits each atom exactly twice**; the steps of the loop are then the bonds in our tetravalent network."

So BM2000 begins with a tiny 4-atom loop and grows it. "Visits each atom exactly twice" is the **Z=4** terminal condition (each atom is an interior vertex of the loop twice → degree 4). The code instead builds a **full N-vertex Hamiltonian cycle first** (every vertex deg-2), then pairs deg-2 vertices up to deg-3 (Z=3). This is a different architecture motivated by the Z=3 target.

**Traversal: BM2000 is RANDOM, the code is GREEDY nearest-neighbour.** BM2000 §II.A: expansion is by "**randomly selecting** a group of three atoms A, B, and C"; and the stated *purpose* of the whole random-seed method (§II.A, end): "Although this method leads to highly strained initial configurations, it has the advantage that it contains **absolutely no trace of crystallinity**." MB2001 p.499 pt.1 echoes: "a **truly random** configuration… not contaminated by some memory of the crystalline state."

Code, by contrast (`638`–`667`): `tree.query(...)` then "greedily walk to the **nearest** unvisited vertex." Comment at `691`–`695`: "Global **ascending-distance greedy matching**… accept the **shortest valid** pair first." **This is the opposite of random.** Shortest-first connection imposes exactly the spatial-locality bias BM2000 designed its random selection to avoid; it is the single most topology-biasing deviation.

**Verdict (a):** Code does **not** match BM2000 on either count — it uses a full Hamiltonian cycle (vs. a grown 4-atom loop) **and** greedy nearest-neighbour ordering (vs. random selection).

### (b) Does the code's "nearest available deg-2 partner within `rc`" pairing match BM2000's bond-insertion?

**No — different move class.** BM2000 §II.A expands the loop by **vertex insertion into an existing bond**:

> "The expansion of the loop is achieved by randomly selecting a group of three atoms A, B, and C, such that A is not fourfold coordinated and is within a distance of `r_c` from B and C but not bonded to either, while **B and C are bonded**. Next, **the bond BC is replaced by bonds AB and AC**, expanding the loop by one step."

(Illustrated in BM2000 Fig. 2.) BM2000 takes an existing bond `B–C`, breaks it, and inserts `A` between them. This raises `A`'s degree by 2 per insertion (it gains two bonds, AB and AC), which dovetails with the Z=4 target reached when "each atom is visited exactly twice."

The code (`716`–`728`) finds **two under-coordinated (deg-2) vertices** `u, v` and **bonds them directly** (`_add_edge(u, v)`), adding +1 degree to each. It never breaks an existing bond and never inserts a vertex. Fair context: BM2000's +2-per-step insertion is awkward for a Z=3 target (you would overshoot), which **motivates** the substitute pairing scheme — but motivation does not make it faithful. It is a different bond-insertion algorithm, and within it the *selection rule* (shortest-first) further departs from BM2000's random rule (see (a)).

### (c) The no-3-ring / no-4-ring (girth ≥ 5) rule — does BM2000 §II.B impose it at construction the way the code does?

**No — the code's "§II.B no-four-ring rule" citation is wrong on three counts.** BM2000 §II.B is titled "**Avoiding complete relaxation of trial configurations**" (p.4987): it is a *quench-acceleration* technique (threshold energy `E_t = E_b − k_B T ln(s)`, early move rejection — Eqs. 3–4) and **contains no four-ring discussion at all**. The four-ring text the docstring is presumably reaching for lives in the *next* section, **§II.C "Efficient quenching"** (p.4987, right column, after the bond-transposition-enumeration paragraph). Three distinct errors:

1. **Wrong section.** No four-ring rule exists in §II.B; the relevant passage is §II.C.

2. **Wrong phase.** Wherever it appears, that passage governs *quenching*, not *construction*. The code applies its girth check during *seed construction* (`723`–`724`, and in fallback `790`/`921`). BM2000's construction step is §II.A, which imposes **no ring rule at all** — it explicitly accepts "highly strained initial configurations."

3. **Opposite stance on 4-rings.** Where BM2000 *does* discuss four-membered rings (§II.C, p.4987) it **allows** them and removes them later:

   > "Following DTW, we find that especially for quenching the relaxation is significantly helped **by allowing for four-membered rings**, because of the large extra number of pathways available to the system. At the end of the quenching, **the few four-membered rings that are created can easily be removed one by one**…"

   So BM2000 *permits* 4-rings during the process and prunes them at the end. The code does the reverse: it **forbids** 4-rings (and 3-rings) at the moment each bond is inserted (`_would_make_short_ring(.,.,4)` rejects squares; the final invariant at `984`–`992` asserts zero triangles).

(Note re: project memory — the index entry claiming the random seed is "girth-3" is **stale**; the code as read enforces girth ≥ 5 at construction. Code wins.)

3-rings: BM2000 doesn't discuss triangles (a Keating CRN is effectively triangle-free by strain); the code forbidding them is reasonable but is still a build-time constraint BM2000 does not state.

### (d) The 2-opt long-bond repair and the `rc`-growth fallback — literature or engineering?

Three sub-parts, three different answers:

- **`rc` growth: FAITHFUL (in spirit).** BM2000 §II.A (p.4986–4987): "Initially, `r_c` is set to some small value like 3 Å, but then it is **gradually increased** until all atoms are fourfold coordinated." The code's `rc += rc_grow_frac*d0` (`736`) is this exact mechanism. (It is grafted onto a non-BM2000 pairing step, but the radius-growth idea itself is BM2000's.)

- **Force-pair-across-PBC fallback (`744`–`868`): ENGINEERING ADDITION.** BM2000 has no "pair the last stragglers across the whole box at any distance" terminator; it relies on `rc` growth alone. The code adds this because its direct-pairing scheme (deviation (b)) can strand a few deg-2 vertices that no legal in-`rc` partner can absorb. `_augment_pair` (`761`–`818`) is a degree-preserving edge-break-and-relink: the move-*shape* echoes BM2000 §II.A's coordination-preserving artifact swap ("replace a bond of each of these atoms by a bond between these atoms and another bond between their neighbors, conserving four-fold coordination"), but BM2000 uses that to fix **close-but-unbonded geometric artifacts**, whereas here it is a **topological deadlock terminator**. Borrowed move, different trigger.

- **2-opt long-bond repair (`876`–`966`): ENGINEERING ADDITION.** Not in BM2000, WWW1985, Sellers, or MB2001. The 2-out/2-in degree-preserving swap is the same *shape* as the WWW bond transposition (WWW1985 Fig. 1; MB2001 Fig. 1 `ABCD → ACBD`) and BM2000's artifact swap, but here it is deployed purely to shorten the over-long bonds the cycle-closure / `rc`-growth / force-pairing steps created — a self-inflicted-defect cleanup with no literature counterpart. The docstring at `876`–`889` is candid that this exists because Sellers Eq. 2 has no non-bonded repulsion to shorten long bonds geometrically.

### (e) Does Sellers specify HOW the random seed is made?

**No. Sellers leaves the seed construction entirely unspecified — this is itself a key finding.** The Sellers supplement (p.6, "Amorphous Gyroid Simulated Annealing") describes only the *WWW annealing*, and refers to the seed in just two phrases:

> "The basic process involves the simulated annealing of **a random network**, which is allowed to evolve through a multitude of geometric and topological configurations."

and the closing line:

> "High-quality amorphous gyroid networks were successfully generated **from random seed patterns** after around 100,000 WWW iterations."

That is the **entire** specification: "a random network" / "random seed patterns." There is **no** description of placement, no connection algorithm, no Hamiltonian cycle, no loop expansion, no girth rule, no degree-completion procedure. Sellers's cited methodological references are **[13] Vink et al. 2001** and **[14] Mousseau & Barkema 2001** for the WWW algorithm. MB2001 (ref [14], read here) likewise specifies only "atoms are placed at random locations in a periodic box at the crystalline density… truly random… not contaminated by memory of the crystalline state" (p.499 pt.1) and points to the BM2000-style loop method without re-detailing it.

**Important corollary [CORRECTED 2026-06-22 — user-flagged, verified against both PDFs]:** my
original claim that "BM2000 is absent from Sellers' reference list" was WRONG — it only checked the
*supplement's* reference numbering and missed the *main paper's*. The correct citation map:
- **Main paper `ncomms14439.pdf` (ref list p.12): ref 27 = Barkema & Mousseau, PRB 62, 4985 (2000) =
  BM2000** (the namesake), and **ref 28 = Wooten-Winer-Weaire (1985) = WWW.** So Sellers DOES cite
  BM2000.
- **Supplement (p.6): "we followed the protocol of WWW, incorporating many of the refinements
  subsequently developed for modelling large amorphous silicon networks [13,14]"**, where **supp
  ref 13 = Vink, Barkema, Stijnman & Bisseling, PRB 64, 245214 (2001) (Vink2001)** and **supp ref 14
  = Mousseau & Barkema, Curr. Opin. Solid State Mater. Sci. 5, 497 (2001) (MB2001).**

So the builder's BM2000 citation is **well-grounded** — BM2000 is genuinely one of Sellers' cited
CRN/seed-generation references. This does NOT rescue the implementation, and in fact SHARPENS the
critique: BM2000 (main 27), Vink2001 (supp 13) and MB2001 (supp 14) **all** specify *pure-random*
generation; the code's greedy-nearest-neighbour + shortest-first construction deviates from **all
three** of Sellers' cited generation references, not from a single "proxy." What remains true is that
Sellers does not give a *step-by-step* seed-construction recipe — only "random seed patterns" plus
these three citations — so the precise placement/connection procedure is still unspecified by Sellers.

---

## 3. Verdict

**This is a custom seed construction that cites the literature, not a faithful implementation of a published random-seed method.** It correctly borrows BM2000's two genuinely-specified ingredients — hard-core random Poisson placement (§II.A) and gradual `rc` growth — but the actual *connectivity* algorithm (full greedy-NN Hamiltonian cycle → shortest-first direct pairing of deg-2 vertices, with build-time girth≥5, force-pairing, augment-relink, and 2-opt repair) is the author's own and differs from BM2000 in move class, selection rule, and ring handling. Crucially, the primary reference (Sellers) **does not give a step-by-step seed recipe** ("a random network" / "random seed patterns") — though [CORRECTED] Sellers DOES cite the relevant generation literature: BM2000 (main paper ref 27), plus Vink2001 (supp 13) and MB2001 (supp 14). So "faithfulness to the literature" reduces to faithfulness to BM2000/Vink/MB2001 — all of which specify *pure-random* generation, which the code's greedy/shortest-first connectivity steps do not achieve.

**Deviations ranked by plausible bias to the resulting network's topology / ring statistics** (most → least):

1. **Greedy nearest-neighbour cycle + shortest-first pairing replacing BM2000's RANDOM selection** (`638`–`667`, `690`–`728`). *Highest impact.* BM2000's explicit design goal is "absolutely no trace of crystallinity"; greedy-shortest injects precisely the spatial-locality bias BM2000 avoids, correlating bond orientations and seeding short-loop / locally-ordered motifs before WWW ever runs. This is the deviation most likely to bias ring statistics and large-scale homogeneity.
2. **Hamiltonian-cycle-then-pair architecture** (`624`–`742`) replacing BM2000's grow-a-4-atom-loop vertex-insertion. Changes which loops/rings exist at seed time; a single spanning cycle plus chords is a structurally distinct starting topology.
3. **Build-time girth ≥ 5 enforcement** (`599`–`617`, `723`–`724`, `984`–`992`), opposite to BM2000 §II.B's "allow 4-rings, remove later." Directly suppresses small rings in the seed's ring spectrum the WWW anneal then inherits.
4. **2-opt long-bond repair + force-pair/augment-relink fallback** (`744`–`966`). Local topology surgery (edge swaps/relinks) with no literature trigger; biases local connectivity around the repaired regions, though it touches relatively few edges.
5. **`settle_seed_with_repulsion` / `soft_start_seed_relax` non-bonded repulsion** (`1013`–`1328`). *Lowest topological impact* — geometry-only; it moves vertex positions but does not change the edge list. The concern is fidelity of *justification*: the Vink/MB2001 "repulsive equilibration" rationale is code-claimed and was not verifiable from the sources read (Vink2001 not read; MB2001 specifies only random placement, not a repulsive seed settle).

**Caveat acknowledged:** Sellers's emergent-properties claim is that *after ~100,000 WWW iterations* the network is WWW-determined; if the WWW anneal fully equilibrates topology, some seed bias is washed out. But the project's own prior findings (N=1000 hard-floor, refuted pure-WWW emergence at 100k) indicate the anneal does **not** fully erase seed topology at production scale — which makes deviations 1–3 above materially relevant rather than cosmetic.
