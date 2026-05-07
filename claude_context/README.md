# LSU Network Generation — Claude Context

This folder is a self-contained briefing so a future Claude session can pick up
work on this project without re-loading the original PDFs and example files.

## The task
Build a function `generate_lsu_network(...)` that produces a periodic 3D
amorphous trivalent network whose Local Self-Uniformity (LSU) matches a user
target value (Φ_12 or Φ_22).

## Key files
- `lsu_network.py` — the implementation module (sits in the project root).
- `Create_LSU_Function.ipynb` — notebook the user runs; calls `generate_lsu_network`.
- `Example/lsu_example_ends.txt` — reference output, **1500 unique edges over
  1000 trivalent vertices**, periodicity 11.44 µm, rod length ~0.8 µm,
  Φ_12=0.99, Φ_22=0.89. The file has 1653 lines (6 tab-separated columns
  `[x1, y1, z1, x2, y2, z2]`); the extra 153 lines are PBC-image duplicates
  of the edges that cross box faces, emitted twice (once anchored at each
  endpoint's canonical-box image). This duplication is required for
  `create_permittivity_grid_penlike` to draw a periodic structure. Verified
  2026-05-07: `compute_lsu` returns Φ_22=0.8886 / Φ_12=0.9849 on the
  PBC-deduplicated reconstruction.
- `Example/lsu_generated.txt` — generator output. The function emits a
  `(R, 6)` NumPy array, `R = E + (face-crossing edges)` when the new default
  `pbc_duplicate_boundary_rods=True` is used, or `R = E` when it's `False`.
- `LSU Literature/ncomms14439.pdf` — Sellers et al. 2017 (main paper).
- `LSU Literature/41467_2017_BFncomms14439_MOESM1815_ESM.pdf` — supplement with
  the actual algorithm (Supplementary Methods → Amorphous Gyroid Simulated Annealing).

## Layered docs in this folder
- `algorithm.md` — WWW algorithm + energy terms + LSU statistic, with equation
  references back to the paper. Read this before touching `lsu_network.py`.
- `implementation_notes.md` — code map for `lsu_network.py`: which function does
  what, performance tips, JAX/CPU paths, known limitations.
- `file_format.md` — input parameter conventions and output array format.

## Status / known limitations
- The implementation is correct in spirit but slow at full scale (~1100 vertices,
  100k WWW iterations). Pure-NumPy path runs in hours; JAX-jitted path is
  available for the energy + gradient and is much faster on CPU/GPU.
- LSU computation supports Φ_12 (depth 1) fully and Φ_22 (depth 2) using a
  greedy edge-alignment heuristic per the paper's depth-first search description.
- Initial seed: Barkema-Mousseau (PRB 62, 4985, 2000) hard-core uniform
  vertex placement followed by greedy proximity bonds with BM loop-expansion
  repair. Implemented in `bm_initial_network`. The legacy configuration-model
  path is in `random_3regular_graph` but is not used by default.
- Output: rods are written with one endpoint inside the canonical box and the
  other endpoint placed by minimum-image displacement, matching the example.

## How to extend
- Increase iterations / lower final temperature for higher LSU.
- Tune `energy_weights = {α, β, γ, δ}` to bias the network shape.
- For very large networks, port the relaxation step to GPU JAX; the WWW outer
  loop is sequential but each relaxation is independent.
