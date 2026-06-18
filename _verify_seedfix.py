import numpy as np
import lsu_network as lsu

D0 = 0.8
WEIGHTS = (0.7, 0.7, 0.3, 0.4)
Ns = [216, 250, 432, 512, 686, 1024]
SEEDS = list(range(42, 52))


def part_a(N, seed):
    box = np.full(3, (N / 0.668) ** (1 / 3))
    rng = np.random.default_rng(seed)
    pos, edges, meta = lsu.random_seed_network_bm2000(N, box, D0, rng)
    nbr = lsu.build_neighbors(N, edges)
    d = lsu.pbc_displacement(pos[edges[:, 0]] - pos[edges[:, 1]], box)
    bl = np.linalg.norm(d, axis=1)
    if bl.max() > 1.5 * D0:
        pos, _ = lsu.soft_start_seed_relax(pos, edges, box, D0)
    ctx = lsu._RelaxContext(N, box, D0, WEIGHTS, use_jax=False)
    ctx.update_topology(edges, nbr)
    p2, _, _ = lsu.relax(pos.copy(), ctx, max_iter=200)
    p2 = p2 - box * np.round(p2 / box)
    diag = lsu.cluster_diagnostics(p2, edges, nbr, box, D0)
    ratio = diag["min_non_bonded"] / D0
    return ratio, ratio >= 0.4


def part_b(N, seed):
    box = np.full(3, (N / 0.668) ** (1 / 3))
    try:
        lsu.generate_lsu_network(
            lsu_degree_22=0.9999, num_vertices=N, bounds_microns=box,
            edge_length=D0, seed_kind='random_bm2000',
            burn_in_n_heat=0, burn_in_n_cool=0, burn_in_n_quench=0,
            n_www_iterations=2, check_lsu_every=1,
            relax_local_iters=50, relax_global_iters=200,
            energy_weights={'alpha': 0.7, 'beta': 0.7, 'gamma': 0.3, 'delta': 0.4},
            seed=seed, use_jax=False, verbose=False,
        )
        return "OK"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    print("=== PART A ===")
    rows = []
    fails = []
    for N in Ns:
        if N % 2 != 0:
            continue
        for seed in SEEDS:
            try:
                ratio, ok = part_a(N, seed)
            except Exception as e:
                ratio, ok = float('nan'), False
                print(f"  EXC N={N} seed={seed}: {type(e).__name__}: {e}", flush=True)
            rows.append((N, seed, ratio, ok))
            tag = "PASS" if ok else "FAIL"
            if not ok:
                fails.append((N, seed, ratio))
            print(f"  N={N:<5} seed={seed}  min_nb/d0={ratio:.4f}  {tag}", flush=True)

    npass = sum(1 for r in rows if r[3])
    print(f"\nPART A pass rate: {npass}/{len(rows)} = {100*npass/len(rows):.1f}%")
    if fails:
        print("FAILS:")
        for N, seed, ratio in fails:
            print(f"  N={N} seed={seed} min_nb/d0={ratio:.4f}")
    else:
        print("FAILS: none")

    print("\n=== PART B (full path) ===")
    braises = []
    for N in [250, 512]:
        for seed in range(42, 48):
            res = part_b(N, seed)
            print(f"  N={N:<5} seed={seed}  {res}", flush=True)
            if res != "OK":
                braises.append((N, seed, res))
    print("\nPART B raises:")
    if braises:
        for N, seed, res in braises:
            print(f"  N={N} seed={seed}: {res}")
    else:
        print("  none")
    print("\nDONE")
