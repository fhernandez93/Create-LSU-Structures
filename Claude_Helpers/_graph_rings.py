"""Collision-proof, GRAPH-TRUE ring statistics from the edge list (no rod round-trip).

Shortest-cycle-per-edge (Guttman ring): for each edge (u,v), remove it, BFS the shortest
path u->v, ring size = path_len + 1. This is topology-only, immune to the vertex collisions
that break the rod-reconstruction ring counter at warm anneal checkpoints (8r=nan). Validated
to reproduce the reference distribution (6:7.6 7:10 8:59.7 9:20.9, mean 7.99).
"""
import numpy as np
from collections import deque


def ring_stats_from_edges(edges, N):
    """Return (dist dict size->fraction%, mean_ring, counts dict, girth)."""
    nbrs = [[] for _ in range(N)]
    for a, b in edges:
        a = int(a); b = int(b)
        nbrs[a].append(b); nbrs[b].append(a)
    sizes = []
    for a, b in edges:
        a = int(a); b = int(b)
        # shortest path a->b WITHOUT the direct edge (a,b)
        # BFS; skip the first direct hop a->b
        dist = {a: 0}
        dq = deque([a])
        found = -1
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du + 1 >= found and found > 0:
                break
            for w in nbrs[u]:
                if u == a and w == b:
                    continue  # forbid the direct edge
                if w == b:
                    found = du + 1
                    break
                if w not in dist:
                    dist[w] = du + 1
                    dq.append(w)
            if found > 0:
                break
        sizes.append((found + 1) if found > 0 else 0)
    sizes = np.array([s for s in sizes if s > 0])
    E = len(sizes)
    uniq, cnt = np.unique(sizes, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(uniq, cnt)}
    dist = {int(k): round(100.0 * v / E, 2) for k, v in zip(uniq, cnt)}
    mean = float(sizes.mean())
    girth = int(uniq.min()) if len(uniq) else 0
    return dist, mean, counts, girth


if __name__ == "__main__":
    import os, sys, tools, lsu_network as lsu
    # env BOX: "L" (cube) or "Lx,Ly,Lz" (slab); only used for .txt rod files,
    # where the graph is reconstructed under PBC. .npy edge input is box-free.
    _b = os.environ.get("BOX")
    if _b:
        _v = [float(x) for x in _b.replace(",", " ").split()]
        box = np.array(_v * 3 if len(_v) == 1 else _v, float)
    else:
        box = np.array([11.44] * 3)
    path = sys.argv[1] if len(sys.argv) > 1 else "Example/lsu_example_ends.txt"
    if path.endswith(".npy"):
        edges = np.load(path); N = int(edges.max()) + 1
    else:
        rods = np.loadtxt(path); pos, edges = tools.rods_to_network(rods, box); N = len(pos)
    dist, mean, counts, girth = ring_stats_from_edges(edges, N)
    print(f"{path}: N={N} E={len(edges)} girth={girth} ring_mean={mean:.2f}")
    print("  dist%:", dict(sorted(dist.items())))
    print("  8r=%.1f%%  ref(rod) 6:7.6 7:10 8:59.7 9:20.9 mean 7.99" % dist.get(8, 0.0))
