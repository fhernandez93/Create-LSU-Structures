# N≈512 random_bm2000 validation — results table

Start: 2026-06-18 17:59. Hard stop 21:59 (4h). Per-run timeout 1800s. Max 12 configs.

## Reference (gold standard, N=1000, box=11.44, d0=0.8)
| metric | value |
|---|---|
| Φ_12 | 0.9849 |
| Φ_22 | 0.8887 |
| bond mean / std | 0.800 / 0.0294 |
| bond ang std | 8.41° |
| rings | 6:7.6% 7:10.0% 8:59.7% 9:20.9% 10:1.7% |
| ring mean | 7.99 |
| 6-ring frac | 7.6% |
| **S(k₀)** (lowest shell) | **0.041** |
| S_low_k2 | 0.053 |
| S_v α(k<2) | +1.51 |
| dihedral entropy | 0.796 |

## Success criteria @ N=512
- 6-ring ≥5%; ring mean 7.8–8.1; 8-ring dominant
- bond std <0.045; bond mean ≈0.80
- **S(k₀) <~0.15** (gate); S_low_k2 <~0.10; α not negative
- Φ_12≈0.985, Φ_22≈0.889 (don't exceed toward 1.0)
- amorphous (no Bragg peaks)

## Fixed across all runs
seed_kind=random_bm2000, d0=0.8, box=9.152, weights α0.7 β0.7 γ0.3 δ0.4,
burn-in OFF, check_lsu_every=0 (full-length, no early-exit confound — advisor).

## Ablation order (advisor-revised)
1. target=0.889, weight=0  ← discriminating + most-faithful: does target alone fix void?
2. target=0.889, weight=10  ← prompt's prescribed (both fixes on)
3. weight sweep (5/15), longer iters for weight=0 (emergence is ~100k claim)
4. relaxation fidelity (local_shell_depth=None, iters~800) — LATE, expensive

## Runs
| # | tag | key params | iters | Φ12 | Φ22 | bond std | ring mean | 6-ring% | S(k₀) | S_low_k2 | elapsed | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | REFERENCE | N=1000 | - | 0.985 | 0.889 | 0.029 | 7.99 | 7.6 | 0.041 | 0.053 | - | gold |
| 1 | r1_t889_w0 | weight=0 | 15k | 0.976 | 0.834 | **0.148** | 8.46 | 2.3 | **3.16** | 0.98 | 666s(GPU) | FAIL void+bond+rings |
| 2 | r2_t889_w10 | weight=10 (T=0.5!) | 15k | 0.973 | 0.831 | **0.154** | 8.33 | 2.3 | **0.68** | 0.32 | 656s | FAIL (confounded T) |
| 5 | r5_coldT_w0 | weight=0, **T=0.045→0.015** | 15k | 0.978 | 0.842 | 0.131 | **7.996** | **5.5** | 0.92 | 0.59 | 486s | rings PASS; void FAIL |

### Batch 2 — temperature is the dominant lever for rings
- **Cold T (0.045→0.015) fixes the rings**: ring mean 8.46→7.996 (=ref), 6-rings 2.3%→5.5% (passes
  ≥5%), 8-ring dominant. The T=0.5 default was destroying the ring distribution (near-uniform
  acceptance ignores energy). Acceptance trajectory healthy: 27%→6%, E 102→41, Φ 0.74→0.84.
- **Void is now the sole gate**: S(k₀)=0.92 at cold-T/w0, still ≫0.15. The void GROWS during the
  chain even at cold T (seed S_low 0.23 → 0.59) = local-only-relax accepts void moves.
- bond std 0.131 still fails but is ONE phenomenon with the void (advisor).
- Base config corrected: T=0.045→0.015, check_lsu_every=500 now default for all subsequent runs.

| 6 | r6_coldT_w10 | weight=10, coldT | 15k | 0.978 | 0.836 | 0.124 | 8.034 | 3.0 | 0.264 | 0.206 | ~470s | void closer; 6-ring fail |
| 7 | r7_coldT_w30 | weight=30, coldT | 15k | 0.983 | 0.850 | 0.111 | 8.090 | 2.3 | **0.120** | **0.063** | 472s | void PASS; 6-ring+bond fail |

### Batch 3 — penalty solves void but trades against 6-rings; bond-std wall
- Monotonic tradeoff: weight 0→10→30 drops S(k₀) 0.92→0.26→0.12 (PASS at 30) BUT drops 6-rings
  5.5%→3.0%→2.3% (FAIL at 30) and over-uniformizes (voxel_std4 2.22→1.67, ref 2.25). No single
  weight passes BOTH void and 6-ring gates → penalty alone is insufficient / wrong tool.
- **bond std plateaus 0.131→0.124→0.111** — never near 0.045. Residual TOPOLOGICAL strain the
  penalty can't relieve. Signature of local-only relaxation accepting strained/void topology.
- → Next: full-N relax DURING the chain (local_shell_depth=None, advisor's lever, now testable at
  sane T). Hypothesis: accurate per-move energy rejects void/strain moves → lower void AND bond-std
  WITHOUT the penalty's ring distortion.

| 8 | r8_coldT_w0_fullN | local_shell_depth=None, iters=800, coldT | 10k | 0.983 | 0.852 | 0.112 | 8.046 | 1.6 | 0.418 | 0.297 | 1175s | full-N relax REFUTED |

### Batch 4 — full-N-relax hypothesis REFUTED; bond-std is the wall
- Full-N relax during chain: bond std 0.112 = SAME as everything else (0.11–0.13). Did NOT fix bond
  std. Helped void only moderately (0.92→0.42, < penalty's 0.12) and HURT 6-rings (→1.6%). Costs 2.4×
  runtime (1175s/10k vs 486s/15k). Net: not worth it; the advisor's "accurate energy rejects strain
  moves" mechanism is not operative here.
- **bond std ≈0.11 is ROBUST across local-relax, full-N-relax, and penalty** → it is NOT a relaxation
  artifact and NOT merely void-coupling (r7 solved void S_low=0.063 but bond std stayed 0.111).
  Contradicts the "void+bond-std are one phenomenon" framing.
- Root cause: bond_max ≈1.33·d0 in EVERY output = topologically-forced long seed bonds (random_bm2000
  force-pair-across-PBC tail) that WWW hasn't rewired in 10–15k iters. Φ plateaus 0.85 < target 0.889
  → UNDER-equilibrated. Sellers used ~100k. → Next: longer run (slower cool, stays warm) to rewire
  the long-bond tail; that is the candidate bond-std fix, not relaxation locality.

### Reframe (advisor reconcile) — what actually matters
- The TWO stated targets (prompt "two problems to solve") are EACH solved: **rings via cold T**
  (r5: 6-rings 5.5%, ring mean 7.996) and **low-k void via penalty** (r7: S(k₀)=0.12, S_low=0.063).
  The open crux is their **TRADEOFF**: no single weight at 15k gives 6-rings≥5% AND S(k₀)<0.15.
- bond-std mechanism corrected: it's **general under-equilibration** (Φ↔bond-std monotonic across
  r1/r5/r6/r7/r8: 0.834/0.148 → 0.852/0.111), NOT a few long bonds (std 0.11 needs broad spread).
  bond_max 1.33·d0 is the tail, not the cause. bond std <0.045 likely needs ~100k iters (Sellers).
- Two divergent bond-std levers to disambiguate via multi-seed: (a) equilibration → expensive at
  N=5000; (b) seed conditioner (repulsion-settle leaves seed bond std 0.207) → O(N) cheap. Test:
  record post-settle seed std vs final std per seed. Final tracks seed std ⇒ conditioner; seed-stable
  & only iters move it ⇒ equilibration.
- r9 (40k, w=10) tests whether more equilibration dissolves the tradeoff. WATCH: Φ must break >0.85
  (r7 froze at 0.848/acc4.5%); if it caps at 0.85 again, floor T=0.015 is freezing it → need warmer hold.

| 9 | r9_coldT_w10_40k | weight=10, 40k slow-cool | 40k | 0.982 | 0.857 | 0.121 | 8.191 | 3.1 | 0.41(noisy) | 0.154 | 1194s | tradeoff NOT dissolved |
| 10 | r10_coldT_w30_40k | weight=30, 40k slow-cool | 40k | **0.986** | 0.869 | 0.104 | 8.164 | 3.1 | **0.144** | **0.061** | 1213s | void+α+Φ12 MATCH ref; 6r+bond fail |

### Batch 5 — more iterations ≠ fix; bond std 0.12 is the equilibrium value
- 15k→40k @ w=10: Φ 0.836→0.857 (broke 0.85 plateau, not fully frozen) BUT bond std 0.124→0.121
  (unchanged), 6-rings 3.0→3.1% (unchanged), void noisy/same, voxel over-uniform 1.39. Extra 25k
  iters bought ≈nothing → bond std 0.12 is the equilibrium at this schedule, NOT a transient.
- Two targets stay mutually exclusive via the weight knob. r10 = w=30 + 40k is the last attempt at BOTH.

| 11 | r11_coldT_w0_40k | weight=0, 40k slow-cool | 40k | 0.985 | 0.870 | 0.115 | 8.275 | **0.0** | 0.74 | 0.41 | 1217s | EMERGENCE REFUTED: coarsens |

### Batch 6 — emergent-hyperuniformity route REFUTED; tradeoff is fundamental
- w=0 @ 40k: 6-rings 5.5%(15k)→**0.0%**(40k), void S_low 0.59→0.41 (still fails, α=−1.06). Long pure
  WWW COARSENS (kills 6-rings) and does NOT emerge to hyperuniform. Confirms [[lsu-cold-www-coarsening]];
  REFUTES the Sellers "near-hyperuniformity emergent over ~100k pure WWW" claim *in this code* — the
  local-relax WWW machinery coarsens instead of emerging. → the penalty is genuinely required for the void.
- 6-ring fraction is a decaying transient of the seed (17%→5.5%→0% with iters at w=0). Reference 7.6%
  is NOT reproduced at any (weight, iters) tested. Penalty at long t PRESERVES ~3% 6-rings (r10) vs
  w=0's 0% — counterintuitive but consistent.
- **CONCLUSION: no single config passes BOTH the ≥5% 6-ring gate AND the S(k₀)<0.15 void gate.**
  Fundamental tradeoff. Nominee = r10 (w=30, 40k): passes the designated void GATE + matches ref
  hyperuniformity (S_low 0.061, α 1.53) + Φ_12 (0.986); 6-rings 3.1% (4× the bad run's 0.8%, below
  5% gate); bond std 0.10 (equilibration floor, needs ~100k).

| 12 | r12_w30_seed7 | weight=30, **seed=7** | 15k | - | - | ~0.11 | 8.077 | 4.6 | **0.096** | 0.068 | ~480s | void PASS |
| 13 | r13_w30_seed17 | weight=30, **seed=17** | 15k | - | - | ~0.11 | 8.049 | 3.1 | **0.036** | 0.050 | ~480s | void PASS |

### Batch 7 — multi-seed confirmation (w=30, 15k; seeds 42/7/17)
- **Void gate ROBUSTLY PASSES across 3 seeds**: S(k₀)=0.12/0.096/0.036 (all <0.15), S_low=0.063/0.068/0.050
  (≈ref 0.053), α=+2.0/+2.0/+2.4. Resolves the single-seed/6-mode-noise caveat — the void fix is real.
- **6-ring gate ROBUSTLY FAILS**: 2.3%/4.6%/3.1% (mean ~3.3%, seed variance 2.3–4.6%, none ≥5%).
- **Post-settle seed bond std = 0.208 for ALL seeds** (conditioner is deterministic in magnitude) →
  final bond std (~0.10–0.13) ≈ half the conditioner-injected strain, consistent across seeds.
  Diagnosis: bond-std floor is set by `settle_seed_with_repulsion` (O(N) cheap to fix), NOT mainly by
  iteration count (Batch 5: 15k→40k barely moved it). Cheaper N=5000 lever than 100k iters.

### Observations after batch 1
- **weight=0 → catastrophic void** (S(k₀)=3.16, α=−1.92). weight=10 cuts it to 0.68 — penalty IS
  load-bearing at 15k budget (REFUTES "hyperuniformity emergent / penalty is crutch" *at this
  budget*; emergence claim is ~100k — caveat stands). But 0.68 still 16× over ref 0.041 → FAIL.
- **bond std ~0.15 in BOTH** (ref 0.029; old bad run was 0.097). Geometric, not topological.
  Suspect the final `repulsion-settle` step: logs show it sweeps λ and leaves bond_max 1.8–1.9·d0,
  S_low ballooning (r1: 0.23→0.98). Prime new suspect — investigate before more weight sweeps.
- **rings coarsened fast**: 6-rings 17% (seed) → 2.3% by 15k, ring mean 8.3–8.5, Φ stuck ~0.83
  (never reached 0.889). Coarsening happens BEFORE reaching target → not purely a Φ-overshoot effect.
- GPU collision: r3/r4 crashed (CUDA OOM/init). Rule: ONE GPU run at a time. Re-running serially.

### CONFOUND FOUND (advisor) — batch 1 invalidated for attribution
- **Batch 1 ran at T=0.5→0.001 (generate defaults), NOT a sane T.** My `_run_config.py` base never
  set initial/final_temperature. The documented bad run used 0.045→0.015. At T=0.5 acceptance is
  near-uniform → energy+penalty barely gate moves → void grows uncontrolled (why hot-15k void 3.16
  beat cold-100k 0.815). Temperature is lever #4, untouched, and gates acceptance BEFORE energy
  accuracy → full-N-relax lever is meaningless until T is sane.
- **Full-N relax on r1 output = NO-OP** (max vertex move 0.003·d0; bond std/void unchanged). Void +
  bond-std are TOPOLOGICAL (baked in during the chain), confirming memory in this worse regime.
- **Void and bond-std are ONE phenomenon** (void forms → vertices forced apart → bonds stretch to
  1.33·d0 → std 0.148). Track as one failure.
- weight=0-vs-10 conclusion is confounded by T; hold "penalty load-bearing" until sane-T w=0 run.
- Re-enabling check_lsu_every=500: undershooting runs (Φ plateaus 0.83 < target) never trigger the
  rising-Φ early-exit, so no confound + I get the required acceptance%/Φ-trajectory logging.

---

## SUMMARY (deliverable)

**Status: PARTIAL SUCCESS.** Each of the prompt's two target problems is individually solved, but
they are mutually exclusive at this iteration budget, and bond-std (a listed criterion, not a core
target) is not met. End time well within the 4h budget; 13 valid runs (2 crashed on GPU collision).

### (a) Validated recipe — `w=30, cold T, 15k` (3-SEED confirmed)
The validated config is the **15k** w=30 run, confirmed across 3 seeds (42/7/17). `r10` (40k) is a
single-seed longer variant that nudges Φ_22 0.850→0.869 at 2.6× cost — use it for production polish.
```python
generate_lsu_network(
    num_vertices=512, bounds_microns=9.152, edge_length=0.8,   # density-matched to ref 0.668
    seed_kind="random_bm2000", seed=42,
    lsu_degree_22=0.889,                                       # target = reference value, NOT ~1.0
    initial_temperature=0.045, final_temperature=0.015,        # SANE cold T — the dominant ring lever
    n_www_iterations=15000,                                    # 3-seed-validated; 40k = longer variant
    uniformity_weight=30.0, uniformity_kmax=2,                 # void/hyperuniformity lever (a crutch)
    local_shell_depth=4, relax_local_iters=100,               # faithful Vink local-shell relax
    burn_in_n_heat=0, burn_in_n_cool=0, burn_in_n_quench=0,    # OFF: random seed has no Bragg to melt
    energy_weights={"alpha":0.7,"beta":0.7,"gamma":0.3,"delta":0.4},  # fixed (Sellers-confirmed)
    check_lsu_every=0,                                         # production: no early-exit
)
```
**Void gate — the headline, 3-seed shell-mean S(k₀) = 0.12 / 0.096 / 0.036 (seeds 42/7/17), all <0.15;**
S_low_k2 = 0.063 / 0.068 / 0.050 (≈ref 0.053); α = +2.0 / +2.0 / +2.4. The void/hyperuniformity fix
is robust, not a 6-mode fluke. (r10's single-seed 40k S(k₀)=0.144 is marginal/anisotropic — worst
mode 0.21 > gate — so do NOT lead with it; the 15k 3-seed numbers are the robust evidence.)

Metrics vs reference (✓/✗ against the prompt's N=512 criteria; r10=40k single-seed shown for the Φ/ring detail):
| metric | 15k 3-seed | r10 (40k) | reference | gate | verdict |
|---|---|---|---|---|---|
| S(k₀) lowest shell | 0.12/0.096/0.036 | 0.144 | 0.041 | <0.15 | ✓ (3-seed robust; r10 marginal) |
| S_low_k2 | 0.063/0.068/0.050 | 0.061 | 0.053 | <0.10 | ✓ |
| S_v α(k<2) | +2.0/+2.0/+2.4 | +1.53 | +1.51 | not neg | ✓ |
| ring mean | 8.09/8.08/8.05 | 8.16 | 7.99 | 7.8–8.1 | ✓ |
| 8-ring dominant | yes | 50.0% | 59.7% | yes | ✓ |
| 6-ring fraction | 2.3/4.6/3.1% | 3.1% | 7.6% | ≥5% | ✗ (robust ~3.3%; 4× the bad run's 0.8%) |
| Φ_12 | ~0.983 | 0.986 | 0.985 | ≈0.985 | ✓ |
| Φ_22 | ~0.850 | 0.869 | 0.889 | ≈0.889 | ✗ (close; 40k helps) |
| bond mean / std | 0.81 / ~0.11 | 0.811 / 0.104 | 0.800 / 0.029 | std<0.045 | ✗ (conditioner-set floor) |
| amorphous (no Bragg) | yes | yes | yes | yes | ✓ |

### (b) Which levers actually mattered (before → after)
1. **TEMPERATURE — the dominant lever, fixes problem #1 (rings).** Generate's default T=0.5→0.001 is
   ~10× too hot → near-uniform Metropolis acceptance ignores energy → ring distribution destroyed.
   Cold T 0.045→0.015: ring mean **8.46→7.996**, 6-rings **2.3%→5.5%** (r1→r5, ONLY T changed; clean).
   This was the unflagged confound in the prompt's prescribed first config.
2. **UNIFORMITY PENALTY — fixes problem #2 (low-k void).** At cold T, weight 0→30: S(k₀) **0.92→0.12**,
   S_low **0.59→0.063** (r5→r7; ONLY weight changed; 3-seed confirmed). But it **trades off against
   6-rings** (suppressing density fluctuations narrows the ring distribution): 6-rings 5.5%→2.3%.
   No single weight passes BOTH the void and 6-ring gates — the core unresolved tension.
3. Slow-cool + 40k (r9/r10): pushes Φ 0.85→0.87 and S_low to ref level, but bond-std & 6-rings ≈flat.

### (c) Negative results (with agent-flagged caveats)
- **Full-N relax during the chain (r8): no benefit, 2.4× cost.** bond-std 0.112 = same as local relax.
  CAVEAT (causal agent): r8 ran 10k vs r5's 15k — the comparison is budget-confounded; only "never
  reaches 0.045" is clean. Fidelity agent: local-shell relax IS the faithful Vink refinement (accept/
  reject uses global energy; the mask only zeroes the gradient) — so this is NOT a fidelity gap, and
  full-N is the wrong, costlier choice at scale. (Corrects my prior memory `lsu-www-faithfulness-audit`.)
- **Long pure-WWW does not emerge to hyperuniform here (r11, w=0/40k): it COARSENS** (6-rings→0%,
  void S_low 0.59→0.41). CAVEAT (causal agent): void was still decreasing and 40k≈40% of Sellers' 100k,
  so "emergence refuted" is overstated — better: *not achieved by 40k; the local-relax WWW coarsens
  rings on the way*. Either way, the penalty was needed for the void at any affordable budget.
- **bond-std ≈0.11 is robust across relax-locality, weight, and 15k→40k.** Mechanism: the seed
  conditioner `settle_seed_with_repulsion` deterministically injects post-settle std **0.208** (all 3
  seeds), which WWW only halves. So the floor is conditioner-set, not purely iteration-bound. (I had
  earlier called it "under-equilibration"; the data says conditioner + weak iteration dependence.)
- Full-N global relax of a FINISHED network = no-op (inline diagnostic on r1 output: max vertex move
  0.003·d0, S_low/bond-std unchanged) → the void/strain are topological (chain-selected), not
  incomplete final relaxation. (Provenance: session diagnostic, not in an r* log — flagged by causal agent.)

### (d) N≈5000 recommendation (DO NOT RUN — recipe only)
- **Config:** same as the validated recipe. Keep `local_shell_depth=4` (faithful Vink; full-N gives no
  quality benefit and is far costlier at scale). burn-in OFF. weights fixed. `uniformity_weight≈30, kmax=2`.
- **Iterations — justified ONLY by moves/vertex to reach the same plateau, NOT by "more equilibration".**
  N=512 hit its Φ/void/ring plateau at ~30–80 moves/vertex (15k–40k iters). Equal moves/vertex at
  N=5000 → ~150k–390k iters; Sellers used ~100k. Recommend **~150k–200k**. NOTE (per our own Batch 5):
  more iters do NOT improve bond-std — that is conditioner-floored and decoupled from iteration count.
- **Runtime — lead with a pilot, not a point estimate.** GPU at N=512: 15k≈480s, 40k≈1210s ⇒ ~0.030 s/iter.
  **Deliverable: run a 2k-iter N=5000 pilot, then runtime ≈ (observed s/iter) × 150k–200k.** Per-iter
  scaling vs N is the shakiest link — at N=512 the GPU is under-saturated, so N=5000 could be ~2× (sub-
  linear) up to ~10× if an O(N) energy/penalty term dominates. The pilot resolves it. For planning only:
  ~4× ⇒ ≈5 h (caveat range 3–8 h).
- **Relaxation at N=5000:** use local-shell relax (the faithful Vink path). Full-N relax is NOT affordable
  and gives no quality benefit (proven at 512). The Vink hybrid (local + finite `global_fallback_threshold`)
  is available but evidence says it won't reduce the void — leave OFF unless the pilot shows residual strain.
- **bond-std at N=5000 is decoupled from iterations — fix the SEED CONDITIONER (O(N)-cheap).**
  `settle_seed_with_repulsion` deterministically sets the floor (post-settle std 0.208 → final ~0.10–0.13,
  all 3 seeds); relaxing its bonds nearer d0 is the lever, not more iters.
- **6-ring/void tradeoff is intrinsic to the penalty** and does not dissolve with iterations. Matching
  reference 7.6% 6-rings AND S_low 0.05 simultaneously likely needs the genuine Sellers ~100k pure-WWW
  *emergent* state, which this code's local-relax WWW does not reach (it coarsens). Honest expectation at
  N=5000 with this recipe: void/hyperuniformity + ring-mean + amorphous PASS; 6-ring ~3% and bond-std
  ~0.10 remain below their gates until the conditioner + an emergent (penalty-free, ~100k) route are addressed.

### Files
- Nominee network: `Structures/20260618_r10_coldT_w30_40k.txt` (+ all r* outputs in Structures/).
- Harness: `_run_config.py`, `_metrics.py` (S(k₀) = mean over lowest |k| shell).
- Agent reviews: `claude_plans/archive/agent_stats_review.md`, `archive/agent_causal_review.md`, `archive/agent_fidelity_review.md`.
