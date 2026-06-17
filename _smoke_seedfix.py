"""Smoke test for the random_seed_network_bm2000 girth-guard fix."""
import numpy as np
from collections import defaultdict
import lsu_network as lsu


def ring_counts(N, edges):
    nbr = defaultdict(set)
    for a, b in edges:
        nbr[int(a)].add(int(b))
        nbr[int(b)].add(int(a))
    # triangles
    tri = 0
    for a, b in edges:
        tri += len(nbr[int(a)] & nbr[int(b)])
    tri //= 3
    # 4-rings: pairs of common neighbours over non-adjacent vertex pairs / 2
    sq = 0
    verts = list(range(N))
    for u in verts:
        for v in verts:
            if v <= u or v in nbr[u]:
                continue
            c = len(nbr[u] & nbr[v])
            sq += c * (c - 1) // 2
    sq //= 2
    # girth probe: does any 5-ring exist? (cheap existence check)
    has5 = False
    for a in verts:
        if has5:
            break
        for b in nbr[a]:
            for c in nbr[b]:
                if c == a:
                    continue
                for d in nbr[c]:
                    if d in (a, b):
                        continue
                    for e in nbr[d]:
                        if e in (a, b, c):
                            continue
                        if a in nbr[e]:
                            has5 = True
                            break
                    if has5:
                        break
                if has5:
                    break
            if has5:
                break
    return tri, sq, has5


for N in (216, 512):
    box = np.full(3, (N / 0.668) ** (1.0 / 3.0))
    d0 = 0.8
    rng = np.random.default_rng(5151)
    pos, edges, meta = lsu.random_seed_network_bm2000(N, box, d0, rng, verbose=False)
    deg = np.bincount(edges.flatten(), minlength=N)
    blen = np.linalg.norm(
        lsu.pbc_displacement(pos[edges[:, 1]] - pos[edges[:, 0]], box), axis=1
    )
    tri, sq, has5 = ring_counts(N, edges)
    print(f"\n=== N={N}  box={box[0]:.3f} ===")
    print(f"deg min/max      : {deg.min()}/{deg.max()}  (expect 3/3)")
    print(f"edges            : {len(edges)}  (expect {3*N//2})")
    print(f"connected        : {lsu.is_connected(N, edges)}")
    print(f"meta n_triangles : {meta['n_triangles']}")
    print(f"recount triangles: {tri}  (MUST be 0)")
    print(f"4-rings (squares): {sq}")
    print(f"any 5-ring       : {has5}  (expect True -> girth 5 allowed)")
    print(f"rc_final         : {meta['rc_final']:.3f} ({meta['rc_final']/d0:.2f} d0)")
    print(f"outer_passes     : {meta['outer_passes']}")
    print(f"bond len mean/min/max: {blen.mean():.3f}/{blen.min():.3f}/{blen.max():.3f}")
    assert deg.min() == 3 and deg.max() == 3, "degree invariant broken"
    assert len(edges) == 3 * N // 2, "edge count wrong"
    assert lsu.is_connected(N, edges), "disconnected"
    assert tri == 0, "TRIANGLES PRESENT — guard failed"
    print("OK")

print("\nALL SMOKE CHECKS PASSED")
