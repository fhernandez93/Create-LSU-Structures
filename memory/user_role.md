---
name: User role
description: PhD student working on photonic structures; technical, knows the LSU literature
type: user
---

PhD student (paco.alejandre@gmail.com). Builds 3D amorphous photonic networks (Sellers et al. 2017 LSU
gyroid family) for downstream FDTD-style simulations. Comfortable with the underlying physics
(Wooten-Winer-Weaire, Barkema-Mousseau, Keating potential, dihedral / skew terms), reads source PDFs,
and points to specific reference numbers when asking for changes — write at that level.

Project currently sits at `/home/francisco/Documents/Create LSU Structures  - Claude` in this workspace
(historically `h:\phd stuff\Create LSU Structures  - Claude` on Windows). User now reports GPU resources.
Pipeline: `lsu_network.generate_lsu_network` → `(N,6)` rod-endpoint array →
`create_permittivity_grid_penlike` (in `20250903_create_h5_from_ends.ipynb`) → HDF5.
