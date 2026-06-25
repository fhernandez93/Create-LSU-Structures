# claude_plans — investigation records

The **written record** of the LSU-network reproduction investigation (the science
behind the production recipe). Documentation only — no code imports this folder.
The runnable project lives in `lsu_network.py`, `tools.py`, `Claude_Helpers/`, and
the notebooks; see the top-level `README.md` to generate/validate a network.

## The investigation in one paragraph
Goal: reproduce the Sellers et al. (Nat. Commun. 8, 14439, 2017) amorphous
trivalent ("LSU") reference network `Example/lsu_example_ends.txt`. Two root
findings: **(1) an energy-form bug** — the f1/f2 terms were simplified (harmonic
`(L−d0)²` + normalized `(cosθ+½)²`) instead of the literal length-coupled
**Keating** forms `f1=Σ(L²−d0²)²`, `f2=Σ(r_ij·r_ik+d0²/2)²`, which made the energy
~6–8× too angle-dominated and destroyed hyperuniformity. Fixed and adopted as the
default (`LSU_KEATING_F1F2=1`). **(2)** with the corrected energy the reference is
reproduced **from a random seed** by extended slow-cool WWW annealing + a free
fixed-topology low-k "Stage-B" void restoration — packaged as the one-call recipe
`Claude_Helpers/from_random_recipe.generate_from_random`. Deliverables (N=1000 and
N=4000, each with a README + validated metrics vs the reference) are in `Example/`.

## Top-level docs (the consolidated record)
- **`phi22_gap_results.md`** — latest / most complete: the from-random recipe that
  closes the Φ22 / local-order gap while keeping hyperuniformity (the final route).
- `N1000_investigation_results.md` — the N=1000 phase (energy-form root cause;
  hyperuniform-seed + uniformity-penalty route; the local-order plateau analysis).
- `N500_validation_results.md` — the earlier N=512 validation phase.
- `2026-05-07_local_relaxation_void_clustering.md`,
  `2026-05-07_pbc_rod_duplication_and_num_vertices.md` — two early technical notes.
- `nominee_config_N512.json` — a saved N=512 candidate configuration.

## archive/
The task **prompts** (`*_prompt.md`) and the per-stage **agent cross-reviews**
(`agent_*.md`) that independently audited each major claim (metrics, causal
attribution, Sellers-fidelity). Kept for provenance; the results docs above already
incorporate their conclusions.

## Durable conclusions
The distilled findings live in the project memory (loaded each session):
`lsu-energy-keating-balance-fix`, `lsu-random-reachability-kinetic`,
`lsu-repo-cleanup-canonical-tooling`, `sellers-confirmed-energy-weights`.
Per-run raw logs were removed in the 2026-06-25 cleanup — their trajectories are
tabulated inside the results docs above.
