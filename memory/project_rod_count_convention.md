---
name: Rod count convention and PBC duplication
description: How `num_rods`, `num_vertices`, and rendered output count relate, and why the reference file has 1653 lines but only 1500 unique edges.
type: project
---

**Reference file `Example/lsu_example_ends.txt` is N=1000 / E=1500, not 1653 unique edges.** Verified 2026-05-07 by deduplicating the 3306 endpoint coordinates in PBC and counting unique unordered vertex pairs:
- 1000 unique trivalent vertices (after collapsing PBC images that wrap to the same canonical position)
- 1500 unique edges in the periodic cell
- 153 of those edges cross at least one box face and are emitted **twice** in the file (once anchored at each canonical-box endpoint), giving 1500 + 153 = 1653 lines

**Why the reference duplicates face-crossing edges.** [`create_permittivity_grid_penlike`](20250903_create_h5_from_ends.ipynb) draws each rod as a literal cylinder at its stored (p1, p2) coordinates and does **not** apply PBC. A bond crossing the +x face is drawn extending past +x but is missing on the -x side, breaking the periodicity of the rendered permittivity grid. The reference convention fixes this by emitting both PBC-image renderings of each face-crossing edge.

**API as of 2026-05-07.** `generate_lsu_network` takes:
- Either `num_rods` (unique periodic-cell edge count) or `num_vertices`, exactly one — they're equivalent, related by `num_rods = 3 * num_vertices // 2`. Use `num_vertices=1000` to match the reference topology.
- `pbc_duplicate_boundary_rods=True` (default) — emits face-crossing edges twice. Output shape is `(E + B, 6)` where `B` is the count of face-crossing edges (statistical, ~150 for L=14.3 / d=1.0).
- Set `pbc_duplicate_boundary_rods=False` for legacy single-render-per-edge behaviour, e.g. when downstream code PBC-tiles itself.

**How to apply.** When matching the reference, `num_vertices=1000` is the right choice — `num_rods=1653` builds a denser network (N=1102) than the reference. The rendered output count `len(rods)` will be ~1500 + boundary duplicates, close to but not exactly 1653 (depends on edge layout for the specific seed/box).

**How to load an external rod file as a network.** Wrap endpoints into the canonical box modulo L, cluster with `cKDTree(boxsize=L).query_ball_tree(..., r=1e-3)` and union-find, then build `edges = np.unique(np.sort(pair_keys, axis=1), axis=0)` to deduplicate PBC-image rods. Used in the LSU verification round-trip (verified 2026-05-07: Φ_22=0.8886 on the reconstructed reference).
