"""Empirical baseline: how crystalline is each network?

Compares the Sellers reference, the bare srs crystal seed, and the prior
generated output. Focus on crystallinity discriminators:
  - dihedral entropy (1.0 = fully uniform/disordered)
  - S_v(k) peak height + radial profile (crystal => sharp Bragg peaks)
  - ring distribution spread (crystal => single ring size)
  - bond-angle std (crystal srs => ~0; amorphous => ~8 deg)
"""
import numpy as np
import tools
import lsu_network as lsu

BOX, D0 = 11.44, 0.8

# Bare srs crystal seed at the reference size (no jitter, no burn-in).
srs_rods = tools.srs_crystal_rods(num_vertices=1000, box=BOX, d0=D0)

sources = [
    (srs_rods,                               "srs-crystal"),
    ("Example/lsu_example_ends.txt",         "REFERENCE"),
    ("Example/20260611_lsu_generated.txt",   "prior-run"),
]

results = []
for src, lbl in sources:
    r = tools.analyze_network(src, box=BOX, d0=D0, label=lbl, verbose=False,
                              k_modes_max=8, k_modes_max_2d=16)
    # Peak / spread of the radial vertex structure factor (crystallinity).
    sk = np.asarray(r["S_v_k_means"], dtype=float)
    kc = np.asarray(r["S_v_k_centers"], dtype=float)
    sk_finite = sk[np.isfinite(sk)]
    r["_sk_max"] = float(np.nanmax(sk_finite)) if sk_finite.size else float("nan")
    r["_sk_mean"] = float(np.nanmean(sk_finite)) if sk_finite.size else float("nan")
    results.append(r)

tools._print_comparison(results)

print("\n=== Crystallinity discriminators ===")
hdr = f"{'metric':<22}" + "".join(f"{r['label']:>16}" for r in results)
print(hdr)
print("-" * len(hdr))
def row(name, key, fmt="{:.4f}"):
    line = f"{name:<22}"
    for r in results:
        v = r.get(key, float("nan"))
        line += f"{fmt.format(v):>16}" if isinstance(v, float) and np.isfinite(v) else f"{'-':>16}"
    print(line)

row("dihedral entropy",   "dihedral_entropy")
row("bond ang std (deg)", "bond_ang_std", "{:.2f}")
row("S_v(k) peak",        "_sk_max", "{:.2f}")
row("S_v(k) mean",        "_sk_mean", "{:.3f}")
row("S_v alpha(k<2)",     "S_v_alpha_low", "{:.3f}")
row("voxel_std_4",        "voxel_std_4", "{:.3f}")
row("S_low_k2",           "S_low_k2")
row("ring mean",          "ring_mean", "{:.3f}")
row("Phi_12",             "phi12")
row("Phi_22",             "phi22")

# Radial S_v(k) profile to eyeball Bragg peaks.
print("\n=== Radial S_v(k) profile (peak => crystallinity) ===")
for r in results:
    kc = np.asarray(r["S_v_k_centers"], dtype=float)
    sk = np.asarray(r["S_v_k_means"], dtype=float)
    top = np.argsort(np.nan_to_num(sk))[::-1][:4]
    peaks = ", ".join(f"S({kc[i]:.2f})={sk[i]:.2f}" for i in sorted(top))
    print(f"  {r['label']:<14} top modes: {peaks}")
