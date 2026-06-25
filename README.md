# Amorphous LSU Networks

Generate 3D periodic **amorphous trivalent (degree‑3) networks** with prescribed
local self‑similarity (the "LSU" statistic), reproducing the reference network of
**Sellers et al., *Nat. Commun.* 8, 14439 (2017)** via Wooten–Winer–Weaire (WWW)
bond‑switching simulated annealing.

The gold reference is `Example/lsu_example_ends.txt` (N=1000 vertices, box
L=11.44 µm, bond length d0=0.8 µm). The validated production recipe grows a
network **from a random seed** that matches it on every hard gate (Φ₂₂, bond/angle
distributions, near‑hyperuniformity, amorphousness). See
[The recipe](#the-recipe) and the dated deliverables in `Example/`.

---

## Repository layout

| Path | What it is |
|---|---|
| `lsu_network.py` | Core library: `generate_lsu_network`, the WWW anneal, the Keating energy, seed builders, the LSU statistic. |
| `tools.py` | Analysis & plotting: `analyze_network`, `plot_comparison`, `rods_to_network`, `srs_crystal_rods`. |
| `Claude_Helpers/` | 6 helper modules — the production recipe + validation (see below). |
| `Create_LSU_Function.ipynb` | **Main notebook**: generate → save → assess statistics. |
| `20250903_create_h5_from_ends.ipynb` | Export a rod file to `.h5` / `.vtk` for visualization. |
| `Example/` | The gold reference + dated deliverables (each with a README + metrics) + grouped checkpoints. |
| `claude_plans/` | The investigation write‑ups (documentation only — see its README). |

`Claude_Helpers/` (everything needed to generate + validate networks):
- `from_random_recipe.py` — `generate_from_random(...)`: the one‑call production recipe (with checkpointing).
- `_anneal_device.py` — on‑device (GPU) WWW anneal, ~3.4× faster than the SciPy path.
- `_run_fromrandom_device.py` — N‑parametrized, checkpointed CLI runner (auto‑resume) for long/large runs.
- `_validate_fromrandom.py` — `assess_statistics(...)`: the full PASS/fail gate report vs the reference.
- `_metrics.py`, `_graph_rings.py` — metric helpers (collision‑tolerant S(k); graph‑true ring stats).

---

## Requirements & setup

- **Python 3.12** with **NumPy, SciPy, NetworkX, h5py, Matplotlib, Jupyter** and
  **[JAX](https://docs.jax.dev/en/latest/installation.html)** (+ `jaxopt`).
- **JAX is required** (the energy, relaxation and anneal are JAX‑accelerated). A
  CUDA GPU is recommended (the reference work used an RTX 4080, JAX 0.10.0 CUDA),
  but **JAX‑CPU also works** — just slower (use `fast=False`, below).

```bash
# create the environment (matching the reference env)
conda create -n lsu_project python=3.12 -y
conda activate lsu_project

# core scientific stack (versions pinned in requirements.txt)
pip install -r requirements.txt

# install JAX for your platform (NOT pinned in requirements.txt):
pip install -U "jax[cuda12]"     # NVIDIA GPU (CUDA 12)
#   or, CPU-only:
pip install -U jax
```

Quick check:

```bash
python -c "import jax, numpy, scipy, networkx; print('jax', jax.__version__, jax.devices())"
```

---

## Quick start

### Option A — the notebook (recommended)
Open `Create_LSU_Function.ipynb` and run the cells top to bottom:
1. **Setup** — sets `D0, N = 0.8, 1000`.
2. **Generate** — calls `generate_from_random(...)` (the recipe). It checkpoints to
   `Structures/` and auto‑resumes after a crash.
3. **Save** — writes the rod file to `Example/`.
4. **Verify** — inline coverage/uniformity checks **and** the full gate report via
   `assess_statistics(rods, N=N)`.

> A full‑schedule run is a **long computation** (~250–300 Stone‑Wales moves/atom).
> Budget ~hours on GPU; ~3× that on CPU. The pre‑computed result is already at
> `Example/20260623_lsu_fromrandom_N1000_ends.txt` — run the cell only to regenerate.

### Option B — from Python

```python
import numpy as np
from Claude_Helpers.from_random_recipe import generate_from_random
from Claude_Helpers._validate_fromrandom import assess_statistics

# Generate (checkpointed + resumable). fast=True uses the on-device path (N=1000).
rods = generate_from_random(
    N=1000, seed=42,
    t_hot=0.09, t_cold=0.040, n_cool=250_000,   # extended slow-cool, pure WWW
    n_hold=50_000, t_hold=0.040,                # sustained hold at the ordering T
    stage_b=True,                               # free fixed-topology void restoration
    fast=True,                                  # True -> GPU device path (N=1000)
    checkpoint_every=25_000,                    # save trajectory every 25k moves (0 = off)
    checkpoint_tag="fromrandom_N1000", resume=True,
)
np.savetxt("Example/my_network_N1000_ends.txt", rods, fmt="%.6f", delimiter="\t")

# Validate against the gold reference (prints a PASS/fail table, returns a dict)
assess_statistics(rods, N=1000)
```

A 6‑column tab‑separated rod file `x1 y1 z1 x2 y2 z2` is produced (face‑crossing
rods PBC‑duplicated, same convention as the reference — see [PBC](#5-pbc-unwrapping-for-output)).

---

## Generating networks

### The production recipe — `generate_from_random(...)`
`Claude_Helpers/from_random_recipe.py`. From a random ("liquid") seed:
1. **Slow‑cool pure WWW** (`t_hot→t_cold` over `n_cool`, uniformity weight = 0) under
   the Keating energy — clears the local‑order plateau (Φ₂₂→ref, bond‑angle std→ref).
2. **Sustained hold** at the ordering temperature `t_hold≈0.04` for `n_hold` — settles
   the angles and holds the 8‑rings (do **not** cool below ~0.04: the cold tail coarsens
   8‑rings into 9‑rings).
3. **Stage‑B void restoration** — a free fixed‑topology optimization (minimize
   `E_Keating + λ·S_low_k`) that restores near‑hyperuniformity at ~zero ring/angle cost.

**Checkpointing.** Set `checkpoint_every>0` + a `checkpoint_tag` to save the
trajectory to `Structures/<date>_<tag>_ck<k>k.txt` (+ `_edges.npy`) every N moves.
`resume=True` auto‑continues from the latest checkpoint for that tag (crash‑robust —
the device anneal can segfault at the CUDA level on multi‑hour runs). Any checkpoint
can be fed to `assess_statistics`. `checkpoint_every=0` (default) is the one‑shot run.

**Speed.** `fast=False` uses the SciPy/L‑BFGS relax (portable, CPU‑ok). `fast=True`
uses the on‑device Barzilai–Borwein relax (~3.4× faster) — parity‑validated at
**N=1000 + this schedule**; for other N/schedules keep `fast=False` until re‑checked.

### Larger sizes (N=4000, …) — the CLI runner
For long/large runs use the checkpointed, auto‑resuming runner (density‑matched box
`L = (N/1000)^(1/3)·11.44`):

```bash
# N=4000: cool 0.09->0.040 over 700k, hold 500k, checkpoint every 25k, seed 42
conda run --no-capture-output -n lsu_project \
  python -m Claude_Helpers._run_fromrandom_device 4000 frd4000 0.09 0.040 700000 500000 25000 42
# then apply Stage-B + validate a chosen checkpoint:
N_VAL=4000 python -m Claude_Helpers._validate_fromrandom \
  Structures/<date>_frd4000_ck1200k.txt 20260101_my_N4000
```

> **GPU = one run at a time** (a 12 GB card OOMs on parallel JAX runs). For CPU‑only
> analysis, prefix with `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.

### Alternative seed routes — `generate_lsu_network(seed_kind=...)`
`lsu_network.generate_lsu_network` also supports building from other seeds:
`'random_bm2000'` (the literal random start used by the recipe), `'crystal_srs'`
(srs/gyroid crystal + melt burn‑in), and `'hyperuniform'` (near‑hyperuniform vertex
placement + uniformity penalty). The from‑random recipe above is the validated route;
the others are documented in `claude_plans/`. **Energy weights are fixed** at
α=0.7, β=0.7, γ=0.3, δ=0.4 (confirmed by the Sellers group — do not change).

---

## Validating a network — `assess_statistics(...)`

`Claude_Helpers/_validate_fromrandom.py`. Recomputes **every** reproduction gate
graph‑true and prints PASS/fail vs the gold reference:

```python
from Claude_Helpers._validate_fromrandom import assess_statistics

# Assess a FINISHED network (already void-fixed) — measures the saved geometry as-is:
assess_statistics("Example/20260623_lsu_fromrandom_N1000_ends.txt", N=1000, stage_b=False)

# Assess a RAW (pre-Stage-B) checkpoint AND apply the void fix (lambda sweep);
# add out_tag to save the best to Example/<out_tag>_ends.txt:
assess_statistics("Structures/<date>_fromrandom_N1000_ck250k.txt",
                  N=1000, stage_b=True, out_tag="20260101_my_network_N1000")
```

> Use `stage_b=False` for a finished network — a Keating relax would re‑open the void.
> `stage_b=True` is for raw checkpoints (it deep‑relaxes a baseline then runs Stage‑B).

### Reference metrics & gates (recomputed from `Example/lsu_example_ends.txt`)

| metric | reference | gate |
|---|---|---|
| Φ₂₂ (LSU, depth 2 / locality 2) | 0.889 | ≥ 0.88 |
| Φ₁₂ (depth 1 / locality 2) | 0.985 | ≈ 0.985 |
| bond‑angle std | 8.41° | ≤ ~9° |
| bond‑length std | 0.029 | ~0.03 |
| S(k₀) lowest shell | 0.041 | ≤ ~0.08 |
| S_low_k2 | 0.053 | ≤ ~0.06 |
| α (hyperuniformity slope, k<2) | +1.51 | ≥ +1.0 |
| S_v peak (Bragg / amorphous check) | 1.82 (no Bragg) | no Bragg peak |
| rings 6/7/8/9 (%) | 7.6 / 10.0 / 59.7 / 20.9 | 8‑ring dominant, girth 6 |

The decisive property is **global amorphousness**: the vertex structure factor
S_v(k) must show **no Bragg peaks** (a crystal peaks ~8.5; the reference ~1.8). High
local order (Φ₂₂≈0.89) alone is not sufficient — a lightly jittered crystal can show
Φ≈0.89 yet stay crystalline.

---

## Exporting for visualization

`20250903_create_h5_from_ends.ipynb` converts a rod‑endpoint file into `.h5` and
`.vtk` (e.g. for ParaView). The reference render `Structures/lsu_example.{h5,vtk}` is
provided. `tools.plot_comparison(...)` (used in the notebook) plots ring
distributions and 1D/2D structure factors side‑by‑side against the reference.

---

## The recipe

The reference is reproduced from a **random seed** by extended WWW annealing under
the corrected energy plus a free void restoration. Two findings made this work:

1. **The energy form (root cause).** The `f1`/`f2` terms were originally simplified
   (harmonic + normalized‑cosine), which made the energy ~6–8× too angle‑dominated and
   destroyed the reference's hyperuniformity. They are now the literal length‑coupled
   **Keating** forms (Section 2), which make the reference a stable fixed point of the
   anneal. This is the default (`LSU_KEATING_F1F2=1`; set `=0` to revert, for
   regression only). Weights are unchanged.
2. **Reachability.** The local WWW anneal *holds* a good long‑wavelength structure but
   cannot *create* one; an extended slow‑cool + sustained hold reaches reference‑level
   local order (Φ₂₂, angles, 8‑rings) from random, and the explicit low‑k **Stage‑B**
   step supplies the near‑hyperuniformity at fixed topology.

Full investigation: `claude_plans/` (start with `claude_plans/README.md` and
`phi22_gap_results.md`).

---

# Algorithm & equation reference

Following Sellers et al., *Nat. Commun.* 8, 14439 (2017) and its supplement.

## 1. Network model
- A continuous random network (CRN) of trivalent (γ=3) vertices in 3D; every vertex
  has exactly 3 neighbors. N vertices ⇒ E = 3N/2 edges (N even, num_rods a multiple of 3).
- The cell is periodic; all vector quantities use minimum‑image PBC.
- Box and density: `L = (N/1000)^(1/3)·11.44` µm at d0=0.8 µm (density ≈ 0.668 vertices/µm³).

## 2. Energy (Supplement, Eq. 2)
    U = α·f1({d}) + β·f2({θ}) + γ·f3({φ}) + δ·f4({χ})

- **f1** — Keating bond‑length term (length‑coupled, the default):
      f1 = Σ_edges (|r_ij|² − d0²)²
  (Legacy harmonic form `Σ(|r_ij|−d0)²` available via `LSU_KEATING_F1F2=0`.)

- **f2** — Keating bond‑angle term (length‑coupled; target 120°, i.e. r_ij·r_ik = −d0²/2):
      f2 = Σ_vertices Σ_{pairs (a,b)} (r_ia·r_ib + d0²/2)²
  (Legacy normalized form `Σ(cos θ_ab + 1/2)²` available via `LSU_KEATING_F1F2=0`.)

- **f3** — Dihedral term (Supplement Eq. 3):
      f3 = Σ_edges (|n̂_{i1,i2} · n̂_{j1,j2}| − 1/3)²
  where i and j share an edge; (i1,i2) are the other neighbors of i, (j1,j2) of j;
  n̂_{a,b} is the unit normal to the plane spanned by (r_{ia}, r_{ib}). Favors
  gyroid‑like dihedrals arccos(±1/3) ≈ 70.53° / 109.47°.

- **f4** — Skew angle / coplanarity (Supplement Eq. 4):
      f4 = Σ_edges (r̂_ij · n̂_{i1,i2})² + (r̂_ij · n̂_{j1,j2})²
  Penalizes the central edge leaving the plane of the trihedron at either endpoint.

The paper does not print numeric weights; the values confirmed directly by the
Sellers group are **α=0.7, β=0.7, γ=0.3, δ=0.4** (the `generate_lsu_network` defaults).

## 3. WWW iteration (Supplement, Eq. 1 + Methods)
1. Pick a random edge (i, j).
2. Pick a neighbor c of i (c ≠ j) and a neighbor d of j (d ≠ i, d ≠ c, not already adjacent).
3. Stone‑Wales / bond transposition: remove (i, c) and (j, d); add (i, d) and (j, c)
   (preserves trivalence).
4. Locally relax the vertices within the Vink/Mousseau‑Barkema fourth‑neighbour shell
   of the move to minimize U for the new topology.
5. Compute ΔE from the relaxed Sellers energy plus the optional low‑k uniformity penalty.
6. Accept with probability P_a = min(1, exp(−ΔE / T)) (Eq. 1).
7. Cool T on a geometric schedule.

Fixed‑schedule global relaxations are disabled by default (they re‑introduce void
drift under the bonded‑only local energy).

## 4. LSU statistic
For each vertex pair (a, b) with b within `locality` edges of a:

(a) Build n‑trees T_n^a, T_n^b by breadth‑first traversal of depth n from a, b.
(b) Translate so root vertices coincide.
(c) For each permutation σ of root edges of T_n^b: rotate to maximally align root
    edges, recursively pair non‑root edges depth‑first to maximize overlap, and score
        f(T^a, T^b; σ) = (1 / (|T_n^a|−1)) · Σ_pairs (r^a·r^b) / (mean(|r^a|,|r^b|))²
    (|T_n^a|−1 = edges in the tree: 3 at depth 1, 9 at depth 2.)
(d) ϕ_ab = (1 / γ!) · Σ_σ f(T_n^a, T_n^b; σ)            (Eq. 1)

Φ_nl = mean of {ϕ_ab : b within l edges of a, over all a}. The first subscript is the
tree depth n, the second the locality l (Sellers Eq. 2; Fig. 3b plots Φ_12, Φ_22, Φ_32,
all at locality 2). For our trivalent case Φ_12 uses depth‑1 trees (3 edges, 3!=6 root
permutations); Φ_22 uses depth‑2 trees (3+6=9 edges, same 6 root permutations).

## 5. PBC unwrapping for output
Each rod's two endpoints can straddle a periodic boundary. The reference file stores
every rod at full length: each face‑crossing edge appears twice, once anchored at each
endpoint's canonical‑box image (the two rows are the same segment translated by a
lattice vector), so either endpoint of a row may lie up to one rod length outside
`[-L/2, L/2]³`. We mirror this convention in the output array
(`pbc_duplicate_boundary_rods=True`, `clip_endpoints_to_box=False`).

## 6. Periodic supercell vs. visible window
The reference has period L = 11.44 µm, so the canonical box is `[-5.72, 5.72]³`.
Visible coordinates extend to ~±6.48 µm because rods crossing a face have one endpoint
outside. The "≈13 µm size" in the example caption refers to that visible bounding box,
not the true period.
