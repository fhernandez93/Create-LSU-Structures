"""Run one generate_lsu_network config at N=1000 (reference scale) and report metrics.

Usage: python _run_n1000.py <tag> '<json_config_overrides>'
Saves rods to Structures/<date>_<tag>.txt and prints a METRICS_JSON block.
Independently recomputes the full metric set FROM THE SAVED FILE.
"""
import sys, json, time, datetime
import numpy as np
import lsu_network as lsu
from Claude_Helpers._metrics import full_metrics, print_metrics

N = 1000
D0 = 0.8
BOX = (N / 1000 * 11.44 ** 3) ** (1 / 3)  # 11.44, density-matched to reference 0.668

tag = sys.argv[1]
overrides = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

base = dict(
    num_vertices=N,
    bounds_microns=BOX,
    edge_length=D0,
    lsu_degree_22=0.889,
    n_www_iterations=30_000,
    initial_temperature=0.045,   # cold T: the dominant ring lever (default 0.5 ~10x too hot)
    final_temperature=0.015,
    check_lsu_every=500,
    uniformity_weight=30.0,
    uniformity_kmax=2,
    seed_kind="random_bm2000",
    seed=42,
    burn_in_n_heat=0, burn_in_n_cool=0, burn_in_n_quench=0,  # random seed: no Bragg to melt
    local_shell_depth=4,
    relax_local_iters=100,
    energy_weights={"alpha": 0.7, "beta": 0.7, "gamma": 0.3, "delta": 0.4},
    use_jax=True,
    verbose=True,
)
cfg = {**base, **overrides}

print(f"=== RUN tag={tag} N={N} BOX={BOX:.4f} ===", flush=True)
print("CONFIG:", json.dumps(cfg, default=str), flush=True)
t0 = time.time()
rods = lsu.generate_lsu_network(**cfg)
elapsed = time.time() - t0
print(f"=== generate elapsed: {elapsed:.1f}s ({elapsed/cfg['n_www_iterations']*1000:.2f} ms/iter) ===", flush=True)

date = datetime.date.today().strftime("%Y%m%d")
out = f"Structures/{date}_{tag}.txt"
np.savetxt(out, rods, fmt="%.6f", delimiter="\t")
# save exact kwargs alongside
with open(f"Structures/{date}_{tag}.kwargs.json", "w") as f:
    json.dump({k: (v if not isinstance(v, float) or np.isfinite(v) else str(v)) for k, v in cfg.items()},
              f, default=str, indent=2)
print(f"saved {out} ({len(rods)} rods)", flush=True)

# ---- independent recompute FROM THE SAVED FILE ----
m = full_metrics(out, box=BOX, d0=D0, label=tag)
print_metrics(m)

summary = {k: m[k] for k in [
    "phi12", "phi22", "bond_len_mean", "bond_len_std", "bond_ang_std",
    "ring_mean", "ring_5_fraction", "ring_6_fraction", "ring_7_fraction",
    "ring_8_fraction", "ring_9_fraction",
    "S_k0", "S_low_k2", "S_v_alpha_low", "S_v_peak", "dihedral_entropy",
]}
summary["S_k0_nmodes"] = m["S_k0_nmodes"]
summary["low_shells"] = [(round(k, 3), round(s, 4)) for k, s, n in m["low_shells"]]
summary["ring_distribution"] = m["ring_distribution"]
summary["elapsed_s"] = elapsed
summary["tag"] = tag
print("METRICS_JSON:", json.dumps(summary), flush=True)
