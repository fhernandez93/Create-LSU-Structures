import numpy as np
import lsu_network as lsu


def check(N, seed):
    box = np.full(3, (N / 0.668) ** (1 / 3))
    pos, edges, meta = lsu.random_seed_network_bm2000(
        N, box, 0.8, np.random.default_rng(seed)
    )
    # degrees
    deg = np.zeros(N, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    dmin, dmax = deg.min(), deg.max()
    n_edges = edges.shape[0]
    # connectivity
    conn = lsu.is_connected(N, edges)
    # triangles: any common neighbor among bonded pairs
    nbr = [set() for _ in range(N)]
    for a, b in edges:
        nbr[a].add(b); nbr[b].add(a)
    tri = 0
    for a, b in edges:
        if nbr[a] & nbr[b]:
            tri += 1
    return dmin, dmax, n_edges, conn, tri


fail = 0
for N in (250, 512, 1024):
    for seed in range(42, 47):
        dmin, dmax, ne, conn, tri = check(N, seed)
        ok = (dmin == 3 and dmax == 3 and ne == 3 * N // 2 and conn and tri == 0)
        status = "OK" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"N={N} seed={seed}: deg[{dmin},{dmax}] edges={ne} "
              f"(exp {3*N//2}) conn={conn} tri={tri} -> {status}")
print("TOTAL FAILURES:", fail)
