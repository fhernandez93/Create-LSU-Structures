# I/O Format

## Inputs to `generate_lsu_network`

| Parameter            | Type            | Notes                                                  |
| -------------------- | --------------- | ------------------------------------------------------ |
| `lsu_degree_12`      | float or None   | Target Φ_12 (depth 1). One of `lsu_degree_12` or       |
|                      |                 | `lsu_degree_22` must be provided.                      |
| `lsu_degree_22`      | float or None   | Target Φ_22 (depth 2).                                 |
| `num_rods`           | int             | Number of edges. Must be divisible by 3                |
|                      |                 | (so V = 2·num_rods/3 is an integer and even).          |
| `bounds_microns`     | float / 3-tuple | Side length(s) of the periodic box.                    |
| `edge_length`        | float           | Target rod length d0 (default 0.8 µm).                 |
| `n_www_iterations`   | int             | Max WWW outer iterations.                              |
| `seed`               | int             | RNG seed for reproducibility.                          |
| `energy_weights`     | dict or None    | {'alpha':α, 'beta':β, 'gamma':γ, 'delta':δ}.           |
| `target_tolerance`   | float           | Stop early if measured LSU is within this of target.   |
| `check_lsu_every`    | int             | How often (in WWW iters) to measure LSU.               |

## Output

NumPy array with shape `(num_rods, 6)`, dtype `float64`. Each row is
`[x1, y1, z1, x2, y2, z2]` for one rod, with `(x1,y1,z1)` placed inside
the canonical box `[-L/2, L/2]^3` and `(x2,y2,z2)` reached from
`(x1,y1,z1)` via the minimum-image displacement (so `(x2,y2,z2)` may
fall outside the box by up to one rod length).

This matches the example file `Example/lsu_generated.txt` exactly. To save
to disk in the 7-column format of `Example/lsu_example_ends.txt` (1-based
index in column 0), do:

    rods = generate_lsu_network(...)
    indexed = np.column_stack([np.arange(1, len(rods)+1), rods])
    np.savetxt("out.txt", indexed, fmt="%d\t" + "\t".join(["%.14g"]*6))
