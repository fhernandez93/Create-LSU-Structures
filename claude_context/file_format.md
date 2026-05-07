# I/O Format

## Inputs to `generate_lsu_network`

| Parameter            | Type            | Notes                                                  |
| -------------------- | --------------- | ------------------------------------------------------ |
| `lsu_degree_12`      | float or None   | Target Φ_12 (depth 1). One of `lsu_degree_12` or       |
|                      |                 | `lsu_degree_22` must be provided.                      |
| `lsu_degree_22`      | float or None   | Target Φ_22 (depth 2).                                 |
| `num_rods`           | int or None     | Number of unique periodic-cell edges. Must be          |
|                      |                 | divisible by 3 (so V = 2·num_rods/3 is even).          |
|                      |                 | Provide exactly one of `num_rods` or `num_vertices`.   |
| `num_vertices`       | int or None     | Number of trivalent vertices in the periodic cell.     |
|                      |                 | Must be even (E = 3N/2 is then guaranteed integer).    |
|                      |                 | Use this to match a known-N reference (e.g. N=1000     |
|                      |                 | for `lsu_example_ends.txt`). Mutually exclusive with   |
|                      |                 | `num_rods`.                                            |
| `pbc_duplicate_boundary_rods` | bool   | If True (default), rendered output emits each          |
|                      |                 | face-crossing edge twice (once per canonical endpoint),|
|                      |                 | matching the Sellers reference convention. Required    |
|                      |                 | for downstream `create_permittivity_grid_penlike` to   |
|                      |                 | draw a periodic permittivity grid.                     |
| `bounds_microns`     | float / 3-tuple | Side length(s) of the periodic box.                    |
| `edge_length`        | float           | Target rod length d0 (default 0.8 µm).                 |
| `n_www_iterations`   | int             | Max WWW outer iterations.                              |
| `seed`               | int             | RNG seed for reproducibility.                          |
| `energy_weights`     | dict or None    | {'alpha':α, 'beta':β, 'gamma':γ, 'delta':δ}.           |
| `target_tolerance`   | float           | Stop early if measured LSU is within this of target.   |
| `check_lsu_every`    | int             | How often (in WWW iters) to measure LSU.               |

## Output

NumPy array with shape `(R, 6)`, dtype `float64`. Each row is
`[x1, y1, z1, x2, y2, z2]` for one rod, with `(x1,y1,z1)` placed inside
the canonical box `[-L/2, L/2]^3` and `(x2,y2,z2)` reached from
`(x1,y1,z1)` via the minimum-image displacement (so `(x2,y2,z2)` may
fall outside the box by up to one rod length).

The row count `R` depends on `pbc_duplicate_boundary_rods`:

- `pbc_duplicate_boundary_rods=False`: `R = E`, exactly one row per unique
  edge. Matches the legacy behaviour and the format of `Example/lsu_generated.txt`.
- `pbc_duplicate_boundary_rods=True` (default): `R = E + B` where `B` is
  the number of edges crossing at least one box face. Each such edge is
  emitted twice — once anchored at each endpoint's canonical-box image.
  Matches `Example/lsu_example_ends.txt` (E=1500, B=153, R=1653).

To save to disk in the 6-column tab-separated format compatible with
`np.loadtxt`:

    rods = generate_lsu_network(...)
    np.savetxt("out.txt", rods, fmt='\t'.join(["%.14g"]*6))
