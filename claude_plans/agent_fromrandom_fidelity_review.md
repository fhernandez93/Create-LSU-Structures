# From-random fidelity + honesty review (independent)

Reviewer: independent agent, 2026-06-23. Scope of verification stated per item.
Files reviewed: `claude_plans/phi22_gap_results.md` (OUTCOME / MONOTONIC / OPTION-2 sections),
`Example/20260623_lsu_fromrandom_N1000_README.md`, `Claude_Helpers/_run_meltquench.py`,
`Claude_Helpers/_validate_fromrandom.py`. I **re-ran** `_validate_fromrandom.py` on the delivered
checkpoint `Structures/20260623_mq2_ck225k.txt` to ground the numbers (not taken on faith).

---

## AXIS 1 — Hard constraints — **SUPPORTED** (all five, verified from source)

| constraint | finding | evidence |
|---|---|---|
| energy_weights = (0.7,0.7,0.3,0.4), never changed | **SUPPORTED** | `_run_meltquench.py:27` `WEIGHTS=(0.7,0.7,0.3,0.4)`, passed verbatim to `_RelaxContext` (52,83) and `www_anneal` (115); `_validate_fromrandom.py:17` `W=(0.7,0.7,0.3,0.4)`. No reassignment anywhere. |
| anneal w=0 (pure WWW, no penalty during acceptance) | **SUPPORTED** | `_run_meltquench.py:36` `W=float(os.environ.get("HU_W","0"))` → default 0; line 118 `uniformity_weight=W`. README/recipe ran with default env, so w=0. |
| Keating energy on (`_KEATING_F1F2` default, never set 0) | **SUPPORTED** | `lsu_network.py:1634` `_KEATING_F1F2 = os.environ.get("LSU_KEATING_F1F2","1")=="1"` → default ON; neither script sets `LSU_KEATING_F1F2`. Runner even prints `KEATING={lsu._KEATING_F1F2}` (47) for an audit trail. |
| seed = `random_bm2000` (allowed random kind) | **SUPPORTED** | `_run_meltquench.py:50` `lsu.random_seed_network_bm2000(...)`; docstring (`lsu_network.py:491`) = Poisson-disk random/liquid seed, BM2000 §II.A, girth≥5. A random seed, not crystalline. |
| `Example/lsu_example_ends.txt` UNTOUCHED | **SUPPORTED** | mtime `2026-05-02 20:38`, predates today (2026-06-23). Reference gold file untouched. |

Axis 1 is clean. The settings fidelity is genuinely correct; nothing in the protocol secretly
changes weights, turns on a penalty, disables Keating, or uses a forbidden seed.

---

## AXIS 2 — Is "from a random seed" honest? — **PARTIALLY SUPPORTED / OVER-CLAIMED for the hyperuniformity gates**

The topology/local-order win is honest and pure-WWW. The hyperuniformity (S_k0/S_low/α) PASS on
the delivered artifact is **load-bearing on Stage-B**, a non-WWW geometric post-process — the doc's
"hyperuniformity emerged from the anneal" does NOT hold at the delivered checkpoint.

**Independently measured** (`_validate_fromrandom.py` on `mq2_ck225k.txt`, this review):

| gate | RAW anneal (Stage-B λ=0, pure WWW + Keating relax) | after Stage-B λ=1.0 | gate | passes only via Stage-B? |
|---|---|---|---|---|
| Φ22 | 0.8822 ✓ | 0.8811 ✓ | ≥0.88 | no — pure anneal |
| angstd | 8.58° ✓ | 8.74° ✓ | ≤9° | no — pure anneal |
| 8r | 46.7 | 46.7 | (secondary) | no — pure anneal (topology fixed) |
| **S(k₀)** | **0.252 ✗** | 0.045 ✓ | ≤0.08 | **YES — fails 3× without Stage-B** |
| **S_low** | **0.144 ✗** | 0.024 ✓ | ≤0.06 | **YES — fails 2.4× without Stage-B** |
| **α** | **−0.09 ✗** | +2.12 ✓ | ≥+1.0 | **YES — raw is ~Poisson, NOT hyperuniform** |

- **Topology / local order (Φ22, angstd, rings) is honestly "from random by pure WWW."** Stage-B is
  verified topology-invariant: `_validate_fromrandom.py` loads `edges` once (line 27), never
  reassigns it; `minimize` optimises only positions `x`, energy from a fixed-topology `ctx`. The
  λ-sweep above confirms 8r/Φ22 unchanged across λ — Stage-B does not touch topology. So for the
  gates the anneal actually solves, "pure WWW from random" is fair and well-supported.

- **Hyperuniformity is NOT honestly "from the pure anneal" on this config.** README line 19 claims
  "Hyperuniformity EMERGED from the anneal (S_k0 0.67→~0.08) with no penalty." The delivered 225k
  checkpoint's RAW S_k0 is **0.252** with **α = −0.09** (essentially uncorrelated, the opposite of
  hyperuniform). The S_k0/S_low/α gates pass ONLY after Stage-B (`E_Keating + λ·S_low` minimised over
  geometry). The README's own recipe line 21 admits "drives S_k0 0.25→0.045" — i.e. the void was NOT
  settled by the anneal on this artifact; Stage-B is doing the work for 3 of the gates.

  → The "0.67→~0.08" figure is a best-case TREND across runs/checkpoints, not the state of the
  delivered config. Stating it as "hyperuniformity emerged from the anneal" without "...but the
  delivered checkpoint needed Stage-B to reach the S(k) gate" is **OVER-CLAIMED** (the 0.08 figure is
  a real noisy checkpoint, not fabricated; the framing overstates it — and Stage-B is an allowed,
  disclosed step, so this is not a hard-constraint VIOLATION).

  **The obvious rebuttal — "the anneal built a hyperuniform-capable topology; Stage-B just realized
  it" — is killed by my own λ=0 row.** Pure Keating relax at the *fixed built topology* with NO
  penalty gives S_k0 = 0.252 / α = −0.09. So hyperuniformity is not a free consequence of energy
  minimisation even at the built topology — it requires the explicit `λ·S_low` term. The penalty was
  not eliminated; it was MOVED out of WWW acceptance into a fixed-topology post-process. That is the
  airtight form of the axis-2 finding.

**Verdict:** Calling the result "from-random" is fair for the hard *topology* gates. The headline
"passes ALL the hard reproduction gates from random by pure WWW" is OVER-CLAIMED: three of the
passing gates (S_k0, S_low, α) are produced by the non-WWW Stage-B geometric optimiser, not by the
anneal, on this specific deliverable.

---

## AXIS 3 — Is the success claim appropriately bounded? — **OVER-CLAIMED in the headlines; honest in the README caveat body**

**Credit (honest disclosures, README "CAVEATS"):**
- (a) ring distribution softer than gold is disclosed: 8r 46.7 vs 59.7 flagged "partial"; 6/7/9% labelled "softer". ✓
- (b) the peak-rings vs peak-angle-sharpness mismatch is disclosed: README states the 175k checkpoint hit 8r 56.9 but angstd 9.1 (over gate), "peak rings and peak angle-sharpness occurred at different points." ✓
- (c) single seed/run is disclosed: "Single seed (42), single run. Not yet the 2-seed confirmation the prompt asks for." ✓

**OVER-CLAIM / VIOLATION flags (the banners/headlines, which a reader sees first):**

1. **"negative OVERTURNED" conflates two distinct negatives.** The prior negative had two parts:
   (a) the Φ22/angstd local-order plateau (0.844 / 11.6°), and (b) the 8r-60 topology is kinetically
   unreachable. The new result **genuinely clears (a)** — 0.881 / 8.74° is a real, large win and the
   honest headline. It does **NOT** clear (b): the deliverable is 8r 46.7, and the only checkpoint
   near the reference topology (175k, 8r 57) FAILS angstd (9.1°) and is a *different* config. The
   ✅✅✅ "FROM-RANDOM REPRODUCTION ACHIEVED — negative OVERTURNED" banner (results-doc line 38) and
   the README "This OVERTURNS the ... topology negative" (line 6) **OVER-CLAIM**: the local-order
   plateau is overturned; the 8r-60 topology remains unreached. The honest statement is "the
   local-order plateau is overturned; the gold 8r-rich topology is approached but not reproduced."

2. **"MONOTONIC CONVERGENCE" papers over a within-run reversal.** The mq2 run is NON-monotonic in 8r:
   8r 56.9 @175k → 46.7 @225k (the delivered checkpoint). The doc reports the best number per axis
   from different checkpoints (8r-best @175k, angstd/balance @225k). README discloses the
   checkpoint-mixing (credit), but the results-doc section title "MONOTONIC CONVERGENCE" and the
   trend table (which uses @150k for mq2) hide that the chosen run reversed on its primary 8r axis
   after the peak. **OVER-CLAIM** in the section header.

3. **Single seed ≠ PASS by the prompt's own rule.** The prompt requires ≥2 seeds for a PASS; this is
   one seed, one run. The README *body* discloses this (honest), but the README *headline* ("passes
   all the hard reproduction gates") and the results-doc "REPRODUCTION ACHIEVED" banner assert
   settled reproduction without the qualifier. Headline honesty ≠ caveat honesty: grade the headline
   **OVER-CLAIMED**.

**Is the headline justified by passing the HARD gates?** Partly. The hard *local-order* gates (Φ22,
angstd) are passed by pure WWW from random — a real result worth stating. But "passes ALL hard
gates" leans on Stage-B for S_k0/S_low/α (axis 2), and "REPRODUCTION / negative OVERTURNED" leans on
the unreached 8r-60 topology + single seed (axis 3). The defensible headline is bounded:
"A random-seed pure-WWW anneal reaches the Φ22/angstd local-order gates (overturning the local-order
plateau); hyperuniformity needs a free fixed-topology Stage-B; the gold 8r-60 topology is approached
(57 at one checkpoint) but not simultaneously with the angle gate, on a single unreplicated seed."

---

## Verification scope (stated explicitly)
- Hard constraints (axis 1): verified directly from source code + file mtime. High confidence.
  w=0 confirmed as the env default (`HU_W` default "0") AND stated in the recipe; the actual mq2 run
  stdout/log was not inspected — audited-as-claimed, not read from the run log.
- Stage-B topology-invariance + the raw-vs-Stage-B gate split (axis 2): **independently re-measured**
  by re-running `_validate_fromrandom.py` on the saved `mq2_ck225k` checkpoint this session; numbers
  match the README's claimed deliverable metrics. High confidence.
- Trend/convergence history and the 175k=8r57 / angstd9.1 claim (axis 3): taken as reported in the
  doc/README (not independently re-run); the 225k reversal and the Stage-B dependence are measured.

---

## 6-line summary

1. AXIS 1 (hard constraints): **SUPPORTED** — weights (0.7,0.7,0.3,0.4) never changed, w=0 pure WWW, Keating default-on with no override, random_bm2000 seed, gold file untouched (mtime May 2). Verified from source.
2. AXIS 2: **OVER-CLAIM** — re-measured the delivered 225k checkpoint: raw (pre-Stage-B) S_k0=0.252, S_low=0.144, α=−0.09 all FAIL the gates; they pass ONLY via the non-WWW Stage-B geometric post-process. "Hyperuniformity emerged from the anneal" is false for this artifact.
3. AXIS 2 (positive): topology/local-order gates (Φ22 0.882, angstd 8.58) DO pass from pure WWW; Stage-B is verified topology-invariant. "From-random by pure WWW" is honest for those gates only.
4. AXIS 3: **OVER-CLAIM** — "negative OVERTURNED" conflates the (genuinely cleared) local-order plateau with the (un-reached) 8r-60 topology (deliverable 8r 46.7; the 8r-57 checkpoint fails angstd).
5. AXIS 3: **OVER-CLAIM** — "MONOTONIC CONVERGENCE" hides a within-run reversal (8r 57@175k → 47@225k); and single seed/run is NOT a PASS by the prompt's ≥2-seed rule (disclosed in caveats, asserted in headlines).
6. NET: hard-constraint fidelity is clean; the real win (local-order plateau overturned, pure WWW from random) is solid; the headlines OVER-CLAIM by (a) crediting the anneal for Stage-B's hyperuniformity, (b) reading "reproduction/overturned" past the unreached gold topology, and (c) headlining a single-seed result as settled.
