import numpy as np
import lsu_network as lsu

rng = np.random.default_rng(0)
N = 60
L = (N / 0.668) ** (1 / 3)
box = np.full(3, L)
d0 = 0.8

# Build a real seed so edges/topology are valid.
pos, edges, _ = lsu.random_seed_network_bm2000(
    N, box, d0, np.random.default_rng(7)
)

# Reach into soft_start_seed_relax's local _energy_grad by reconstructing it.
# Instead, exercise the public function's internal objective by replicating
# the exact closure construction via monkey-inspection: easiest is to call a
# tiny reimplementation is risky -> instead FD-check by perturbing positions
# through the same math the function uses. We test the closure directly by
# pulling it out with a wrapper.

from scipy.spatial import cKDTree
from lsu_network import pbc_displacement, coerce_box

box_arr = coerce_box(box)
e0 = edges[:, 0].astype(np.int64)
e1 = edges[:, 1].astype(np.int64)
edge_set = {(int(a), int(b)) if a < b else (int(b), int(a)) for a, b in edges}
r_rep_frac, k_rep = 0.9, 1.0
r_rep = r_rep_frac * d0


def _tree(p):
    shift = np.clip(p + box_arr / 2.0, 0.0, box_arr - 1e-12)
    return cKDTree(shift, boxsize=box_arr)


def _rep_pairs(p):
    cand = _tree(p).query_pairs(r=r_rep, output_type="ndarray")
    if cand.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    cand = np.sort(cand, axis=1)
    keep = np.array([(int(a), int(b)) not in edge_set for a, b in cand])
    return cand[keep]


def _energy_grad(x, rep):
    p = x.reshape(N, 3)
    g = np.zeros((N, 3))
    dvec = pbc_displacement(p[e1] - p[e0], box_arr)
    Ln = np.linalg.norm(dvec, axis=1)
    Lsafe = np.maximum(Ln, 1e-12)
    E = float(np.sum((Ln - d0) ** 2))
    coef = (2.0 * (Ln - d0) / Lsafe)[:, None] * dvec
    np.add.at(g, e1, coef)
    np.add.at(g, e0, -coef)
    if rep.shape[0]:
        ra, rb = rep[:, 0], rep[:, 1]
        rvec = pbc_displacement(p[rb] - p[ra], box_arr)
        Lr = np.linalg.norm(rvec, axis=1)
        overlap = r_rep - Lr
        active = overlap > 0.0
        if np.any(active):
            E += float(k_rep * np.sum(overlap[active] ** 2))
            Lrsafe = np.maximum(Lr, 1e-12)
            gco = (-2.0 * k_rep * overlap / Lrsafe)[:, None] * rvec
            gco[~active] = 0.0
            np.add.at(g, rb, gco)
            np.add.at(g, ra, -gco)
    return E, g.reshape(-1)


# Perturb so some repulsion pairs are active.
x0 = (pos + 0.05 * d0 * rng.standard_normal(pos.shape)).reshape(-1)
rep = _rep_pairs(x0.reshape(N, 3))
print("active rep pairs (within r_rep):", rep.shape[0])

E0, g = _energy_grad(x0, rep)
fd = np.zeros_like(g)
h = 1e-6
for i in range(g.size):
    xp = x0.copy(); xp[i] += h
    xm = x0.copy(); xm[i] -= h
    Ep, _ = _energy_grad(xp, rep)
    Em, _ = _energy_grad(xm, rep)
    fd[i] = (Ep - Em) / (2 * h)

err = np.max(np.abs(g - fd))
print("max abs grad error:", err)
print("max abs grad mag:", np.max(np.abs(g)))
print("rel err:", err / max(np.max(np.abs(g)), 1e-30))
