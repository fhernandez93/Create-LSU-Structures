"""Run one generate_lsu_network ablation config at N=512 and report metrics.

Usage: python _run_config.py <tag> '<json_config_overrides>'
Saves rods to Structures/<date>_<tag>.txt and prints a JSON metrics block.
"""
import sys, json, time, datetime
import numpy as np
import lsu_network as lsu
from Claude_Helpers._metrics import full_metrics, print_metrics

N = 512
D0 = 0.8
BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)

tag = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

base = dict(
    num_vertices=N,
    bounds_microns=BOX,
    edge_length=D0,
    lsu_degree_22=0.889,
    n_www_iterations=15_000,
    initial_temperature=0.045,  # SANE T (advisor): default 0.5 is ~10x too hot -> near-uniform accept
    final_temperature=0.015,
    check_lsu_every=500,        # undershooting runs never hit target -> no early-exit; gives logging
    uniformity_weight=0.0,
    uniformity_kmax=2,
    seed_kind="random_bm2000",
    seed=42,
    # burn-in OFF: random seed has no Bragg peaks to melt (non-Sellers step)
    burn_in_n_heat=0, burn_in_n_cool=0, burn_in_n_quench=0,
    local_shell_depth=4,
    relax_local_iters=100,
    verbose=True,
)
cfg = {**base, **overrides}

print(f"=== RUN tag={tag} N={N} BOX={BOX:.4f} ===")
print("CONFIG:", json.dumps(cfg, default=str))
t0 = time.time()
rods = lsu.generate_lsu_network(**cfg)
elapsed = time.time() - t0
print(f"=== generate elapsed: {elapsed:.1f}s ===")

date = datetime.date.today().strftime("%Y%m%d")
out = f"Structures/{date}_{tag}.txt"
np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
print(f"saved {out} ({len(rods)} rods)")

m = full_metrics(rods, box=BOX, d0=D0, label=tag)
print_metrics(m)

summary = {k: m[k] for k in [
    "phi12", "phi22", "bond_len_mean", "bond_len_std", "bond_ang_std",
    "ring_mean", "ring_5_fraction", "ring_6_fraction", "ring_8_fraction",
    "S_k0", "S_low_k2", "S_v_alpha_low", "S_v_peak", "dihedral_entropy",
]}
summary["elapsed_s"] = elapsed
summary["ring_distribution"] = m["ring_distribution"]
print("METRICS_JSON:", json.dumps(summary))
