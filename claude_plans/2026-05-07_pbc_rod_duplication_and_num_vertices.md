# Add PBC-image rod duplication and `num_vertices` API

## Context

`generate_lsu_network(num_rods=1653, bounds_microns=14.3, ...)` produces a
1102-vertex / 1653-edge periodic graph and writes 1653 rod lines. The Sellers
reference `lsu_example_ends.txt` has the same line count (1653) but a different
underlying topology: **N=1000 vertices, E=1500 unique edges**, with 153
PBC-image duplicates of edges crossing box faces inflating the rendered count
to 1653 (verified 2026-05-07 by deduplicating endpoints in PBC and counting
unique edges).

Two separate problems result:

1. **Wrong target size.** Setting `num_rods=1653` builds a denser network
   (1102 vs 1000 vertices) than the reference, which throws off downstream
   sample comparisons and the LSU/PBG analysis.

2. **Broken PBC rendering.** [`create_permittivity_grid_penlike`](20250903_create_h5_from_ends.ipynb)
   in the downstream notebook draws each rod as a literal cylinder at its
   coordinates — it does **not** apply PBC. The console already reports
   "187/3306 rod endpoints fall outside the box [-7.15, 7.15]^3" on a recent
   run. Without PBC-image rod duplicates a bond crossing the +x face is
   drawn extending past +x but is missing entirely on the -x side, so the
   permittivity grid is no longer periodic. The reference file solves this
   by emitting each face-crossing edge twice — once from each endpoint's
   canonical-box perspective.

The fix has two parts:
- Add `num_vertices=N` to `generate_lsu_network` (alternative to `num_rods`,
  exactly one required).
- Make `network_to_rods` emit PBC-image duplicates by default for any edge
  that crosses ≥1 face, matching the reference file convention and producing
  a properly periodic permittivity grid downstream.

## Files to modify

- [`lsu_network.py`](lsu_network.py)
  - [`network_to_rods`](lsu_network.py#L1346-L1369) — add PBC-image duplication
  - [`generate_lsu_network`](lsu_network.py#L1375) — add `num_vertices` and
    `pbc_duplicate_boundary_rods` parameters, thread through
- [`Create_LSU_Function.ipynb`](Create_LSU_Function.ipynb) — switch the example
  call to use `num_vertices=1000` (matches the reference topology) and document
  that the output rod count will be ~1500 + boundary duplicates, not exactly 1653
- [`claude_context/file_format.md`](claude_context/file_format.md) and
  [`claude_context/README.md`](claude_context/README.md) — update reference
  description (N=1000 / E=1500 / 1653 rendered rods) and document the new
  `num_vertices` parameter and PBC duplication

## Implementation

### Step 1 — PBC duplication rule in `network_to_rods`

Already verified rule from the reference file: for each edge (a, b),
- compute `a_canon` = wrap(positions[a]) into [-L/2, L/2)
- compute `b_canon` = wrap(positions[b])
- compute `d_ab` = minimum-image displacement from a_canon to b_canon
- **render 1**: `(a_canon, a_canon + d_ab)` — first endpoint inside the box
- **render 2**: `(b_canon, b_canon - d_ab)` — second endpoint inside the box
- if render 1 == render 2 (edge entirely inside box, no face crossing) emit
  one row; otherwise emit both rows

This produces 1500 + 153 = 1653 rendered rods on the reference network
(matches what's observed). Maximum multiplicity per edge is 2 (verified —
edges crossing 2 or 3 faces still produce only 2 distinct PBC-canonical
renderings, since each render starts from a single canonical endpoint).

New signature:

```python
def network_to_rods(
    positions, edges, box,
    pbc_duplicate_boundary_rods: bool = True,
) -> np.ndarray
```

When `pbc_duplicate_boundary_rods=False` the function keeps current behaviour
(single render per edge, output shape `(E, 6)`).

### Step 2 — `num_vertices` parameter in `generate_lsu_network`

Add `num_vertices: Optional[int] = None` and `pbc_duplicate_boundary_rods:
bool = True`. Validation:

```python
if (num_rods is None) == (num_vertices is None):
    raise ValueError("Provide exactly one of `num_rods` or `num_vertices`.")
if num_vertices is not None:
    if num_vertices % 2 != 0:
        raise ValueError(f"num_vertices must be even (got {num_vertices})")
    N = num_vertices
    num_rods = (3 * N) // 2  # E = 3N/2 for trivalent
```

The internal pipeline still uses `num_rods` as the unique-edge count (no
change to `bm_initial_network` / `www_anneal` / `total_energy`). Only the
final `network_to_rods` call gains the duplication step. Update the verbose
log line to print both the unique-edge count and the rendered-rod count.

### Step 3 — Notebook update

Change [`Create_LSU_Function.ipynb`](Create_LSU_Function.ipynb) cell `full-run`
to use `num_vertices=1000` instead of `num_rods=1653`. Other parameters
(`bounds_microns=14.3`, `edge_length=1.0`, `local_shell_depth=4`, weights)
unchanged. Note that `len(rods)` in the verify cell will now be ≈1500 +
boundary duplicates (statistical, ~150 for L=14.3, d=1.0), not exactly 1653.
Update the comment in the verify cell so the reader knows what to expect.

### Step 4 — Doc updates

- `claude_context/file_format.md` — document `num_vertices`, the new
  `pbc_duplicate_boundary_rods` flag, and that rendered rod count = unique
  edges + boundary duplicates.
- `claude_context/README.md` — fix the reference-file description: 1653
  lines but 1500 unique edges over 1000 vertices.

## Verification

1. **Reference round-trip.** Load `lsu_example_ends.txt` (1653 lines), dedupe
   to (positions, edges) with N=1000, E=1500. Call
   `network_to_rods(positions, edges, box, pbc_duplicate_boundary_rods=True)`.
   Assert output shape is `(1653, 6)` and the set of (sorted) endpoint pairs
   matches the original file (within 1e-3 PBC tolerance). This proves the
   duplication rule reproduces the reference convention exactly.

2. **Opt-out unchanged.** Same network, call with
   `pbc_duplicate_boundary_rods=False`. Assert output shape is `(1500, 6)` —
   one row per unique edge. Bit-for-bit identical to the current
   `network_to_rods` output for this input.

3. **`num_vertices` smoke test.** Run `generate_lsu_network(num_vertices=40,
   bounds_microns=4.0, ...)` for 200 WWW iters. Confirm:
   - Internal log reports N=40, E=60.
   - Output shape is roughly `(60 + boundary, 6)` (boundary count
     statistical, log it).
   - `num_rods` parameter still works: same call with `num_rods=60` gives the
     same output (modulo random seed alignment).

4. **End-to-end with permittivity grid.** Run the notebook with
   `num_vertices=1000`, then run `create_permittivity_grid_penlike` on the
   output. Confirm:
   - The "rod endpoints fall outside the box" warning still triggers (those
     are the duplicated PBC-image rods extending across the boundary — this
     is correct, the box-clipping then captures the in-box portion).
   - The resulting permittivity grid has matching density on opposite faces
     (sanity check: bin the grid into 1µm slabs along x and verify slab 0 ≈
     slab N-1 within statistical noise).

## Out of scope

- The `lsu_generated.txt` file (used by `Generated_Example_2.h5`) stays as-is
  — re-running the notebook with the new defaults will produce a new file
  using the new convention.
- LSU computation, Stone-Wales selector, BM seeder, energy terms — none
  affected.
- The opposite direction (loading external rod files into the engine) — not
  needed for this fix; only used for verification step 1.
