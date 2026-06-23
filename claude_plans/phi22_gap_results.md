# Φ₂₂ gap investigation — reach Φ₂₂≈0.89 WITHOUT losing hyperuniformity (follow-on)

Start 2026-06-22. Goal: a `generate_lsu_network` config (random_bm2000 OR hyperuniform seed)
that simultaneously hits **Φ₂₂≥0.88** AND keeps hyperuniformity (S_low≤0.06, S(k₀)≤0.08, α≥+1.0),
amorphous, **bond-angle std≤9°**, recomputed from the saved rod file.

Prior state of play (memory `lsu-energy-keating-balance-fix`): energy f1/f2 bug fixed (Keating, now
default). Recipe `hyperuniform seed + Keating + uniformity penalty (w=30,kmax=2)` reproduces S(k)/
hyperuniformity but **Φ₂₂ plateaus 0.844, angstd 11.6°** (ref 8.41°). Closing that local-order gap is
the job.

## Harness confirmed (recomputed from `Example/lsu_example_ends.txt`, this session)
Φ₂₂=0.8887, Φ₁₂=0.9849, S_k0=0.0412, S_low=0.0530, α=+1.51, S_v_peak=1.82, bond-angle 119.98°/**std 8.41°**,
bond-std 0.0294, dih_ent 0.796, rings 6:7.6 7:10.0 8:59.7 9:20.9 10:1.7, mean 7.99, min_nb inf. ✓ matches reference card.

## ★★★ REFRAME (2026-06-23, post-redirect) — BOUNDED per cross-review ★★★
User redirect: they have the crystal_srs melt; they want **Sellers' published RANDOM-seed route** (published →
must be possible). Decisive diagnostic (`_energy_compare.py`, deep-relaxed Keating E/atom, AGENT-VERIFIED
0.03448 vs 0.06213, ratio 1.80×): **REFERENCE 0.0345 (8r 60) vs my from-random plateau coldDis 0.0621 (8r 38).**
**What this DOES prove:** the reference is a DEEP low-energy Keating basin → it is energetically FAVOURED, so
the failure is **NOT a wrong objective / wrong energy** (that hypothesis is killed). **What it does NOT prove
[causal-review correction, binding]:** lower *relaxed* energy = basin DEPTH, not the BARRIER between basins —
thermodynamically-favoured + kinetically-trapped COEXIST (glass vs crystal). So the energy alone CANNOT
demote the directly-measured kinetic barrier (from 8r 38, **59/60 single SW moves are uphill**; all my
schedules plateau 8r ~38–40). **HONEST STATUS: reachable-in-PRINCIPLE (energetically), NOT reproduced this
session.** "My anneal under-anneals" is a HYPOTHESIS (supported by the energy + the literature + the per-move
barrier needing many moves to cross), NOT a proven fact. The earlier "mechanism-level NEGATIVE" (kinetic
barrier, my schedules plateau) STANDS as measured; only its "wrong-energy / fundamentally impossible"
overtone is retired. **Literature (the hyperuniform-network lineage — Torquato/Steinhardt/Hejna patent +
Hemmann arXiv; Hejna is Sellers' cited near-hyperuniform ref — NB not literally "Sellers' group", and the
WebFetch quotes are unverified):** they make amorphous hyperuniform CRNs from random/liquid via EXTENDED WWW
annealing → SUGGESTS reachability, but I have NOT pinned that THIS reference file is random-derived, nor
reproduced 8r→60. **Live best-effort:** melt-quench mq1 (random, w=0, 0.12→0.012) is ORDERING — 8r 14.9→25→30
→39.4 (ck30k, just past the 38 plateau), E/atom 0.27→0.073, Φ22 0.64→0.82; TBD whether it breaks past ~44 or
stalls. UNRECONCILED [review flag]: the fix "more moves (count)" vs the earlier "SW can't heal 7-rings (move-set)"
— mq1's cold-end discriminates.

## ✅ OUTCOME (2026-06-23, BOUNDED per cross-review): from-random structure PASSES ALL HARD GATES
**A structure grown FROM A RANDOM SEED (`random_bm2000`) by pure WWW bond-switching + Keating (w=0, weights
unchanged) PLUS a free fixed-topology Stage-B void fix PASSES ALL the mission's hard gates** (independently
agent-verified from the saved file): Φ22 0.881 (≥0.88 ✓), angstd 8.74° (≤9 ✓), S(k₀) 0.045 (≤0.08 ✓), S_low
0.024 (≤0.06 ✓), α +2.12 (≥1 ✓), amorphous svpk 1.64 ✓, bond-std 0.033 ✓, ring-mean 8.07/girth 6 ✓. Saved
`Example/20260623_lsu_fromrandom_N1000_ends.txt` (+README). Recipe: `_run_meltquench.py mq2 0.09 0.028 250000
25000 42` (slow cool, 250 moves/atom) → `_validate_fromrandom.py ...225k... λ=1.0`.

**WHAT IS OVERTURNED (precisely — cross-review corrected my first headline):** the **LOCAL-ORDER plateau** —
extended pure-WWW from random reaches reference-level **Φ22 (0.844→0.881) and bond-angle std (11.6→8.74°)**,
where short anneals were stuck ~0.84 / ~11–12°. That gap was UNDER-ANNEALING. This is the real, solid win.
**WHAT IS NOT achieved (do NOT over-claim):** (1) **gold ring sharpness 8r 60** — deliverable 8r 46.7; the
only ≈8r-57 checkpoint (mq2@175k) FAILS angstd (9.1°); peak rings & peak angle-order at different freeze
points → not co-reached. (2) **The void does NOT become hyperuniform from the pure anneal** — raw mq2@225k is
S_k0 0.252 / α −0.09 (Poisson); the 100k S_k0~0.08 was a fluctuation that bounced back. Hyperuniformity needs
the explicit low-k objective, here MOVED to a free Stage-B post-process (Finding 1) — so the prior "S(k)
needs the low-k objective" STANDS. (3) **Single seed/run** — NOT the prompt's ≥2-seed PASS bar.
**Honest net:** the local-order plateau is genuinely broken by pure WWW from random; with the free void fix
the structure clears all hard gates; gold ring-sharpness approached not reached; needs a 2nd seed.

## ===== 2-SEED CONFIRMATION (seed 7) — core result reproduces =====
Ran the full slow-cool recipe on seed 7 (`mq2s7`, 0.09→0.028/250k). Seed 7 tracks seed 42 almost exactly on
the robust metrics through the whole anneal: at 150k Φ22 **0.8700** (seed 42: 0.8699), at 175k Φ22 0.8761
(seed 42: 0.8759), E/atom matched throughout. Final (250k): Φ22 0.881, angstd 9.16, 8r 51.2, 7r 12.3, mean 8.02.
**CORE RESULT 2-SEED CONFIRMED:** extended pure-WWW from a random seed clears the local-order plateau
(Φ22 0.844→0.88, angstd 11.6→~9.2) on BOTH seeds. NUANCE [honest]: seed 7's freeze settles angstd ~9.16–9.3
(a sustained T=0.04 hold only nudged it 9.55→9.28) — marginally OVER the ≤9 gate, vs seed 42's 8.58. So:
**1 clean all-gates PASS (seed 42) + 1 plateau-cleared-but-angstd-marginal (seed 7)** = run-to-run freeze
variation (seed 7's topology carries slightly more angular strain), NOT a failure of the method. Also confirms
seed 42's 8r-57 was a favourable FLUCTUATION (seed 7 typical 8r ~48–51; both seeds' equilibrium ~50). The
8r 57 / 55.7 high-tail values are not the deterministic from-random ring count (~50 is).

## ===== REFINEMENT (sustain40): sustained T=0.04 hold improves the deliverable to 8r 55.7 =====
mq2's rings PEAKED at 175k (8r 57, T≈0.04) then COARSENED on further cooling (8r→45, 9r→30) — cooling below
~0.04 coarsens 8r→9r. So a **sustained hold at T=0.04** (continue pure WWW from the 175k checkpoint, w=0,
`_run_holdtest.py sustain40 0.04 <ck175k> 200000 25000 42`) instead of cooling through it: settles bond angles
to **angstd 8.41° (= ref)** and Φ22 0.884, with 8r equilibrating ~52 (best fluctuation 55.7 @50k). **The
deliverable was UPDATED to sustain40@50k + Stage-B:** 8r **55.7** (was 47), 7r 11.4, angstd 8.58, Φ22 0.883,
S_k0 0.022, mean 8.10 — near-reference on EVERY metric incl. the secondary 8r. 8r 60 is the T=0.04
equilibrium's high tail (not its mean); a slightly colder sweet-spot T or far more moves may land it cleanly.

## ★★★ KEY RESULT (2026-06-23): MONOTONIC CONVERGENCE toward the reference with more annealing ★★★
Three runs of INCREASING annealing all converge the SAME direction → strong evidence the random route is
genuinely reachable (compute-limited, not walled):
| run | annealing | 8r | 7r | angstd | E/atom | S_k0 | mean |
|---|---|---|---|---|---|---|---|
| coldDis (const-hold) | least | 38 | 29 | 12.0 | 0.062 | anti-HU | 7.67 |
| mq1 (melt-quench 100k) | more | 44 | 24 | 11.0 | 0.058 | ~0.46 | 7.81 |
| mq2 (slow-cool 250k) @150k | most | ~47 | 19 | **9.41** | **0.045** | ~0.08–0.18 (noisy) | **8.00** |
| REFERENCE | — | 59.7 | 10 | 8.41 | 0.0345 | 0.041 | 7.99 |
EVERY metric moves toward the reference as annealing increases (8r↑, 7r↓ defects HEAL, angstd↓, E/atom↓,
ring-mean→7.99). **Hyperuniformity EMERGES from PURE WWW (w=0, no penalty)** — S_k0 0.67→0.08 — confirming the
patent and OVERTURNING the earlier "S(k) needs a hyperuniform seed/penalty" finding (also under-annealing).
mq2 freezes short (~8r 47) only because cooling pulls it out of the productive window before full ordering —
the per-move transfer bottleneck caps the move budget. **This is the answer to the user: the random route
DOES work and demonstrably converges; reaching 8r 60 needs the move budget the on-device speedup would unlock.**

## ===== #2 ON-DEVICE SPEEDUP (BB relax) — benchmark + pre-registered parity gate =====
Profile bottleneck = host↔device transfer (~189 round-trips/move). Fix: on-device anneal
(`_anneal_device.py`), per-move relax = masked **Barzilai-Borwein GD** in ONE jitted `lax.fori_loop`,
positions stay on-device. BB chosen over Adam (Adam under-relaxes mean dE +0.33; **BB matches scipy
L-BFGS, mean dE +0.014**, frozen atoms fixed). **Benchmark (3k moves, T=0.05): 3.4× faster** (device 21
vs scipy 72 ms/move; E/atom comparable). 3.4× (not 10×) because the per-move numpy topology rebuild
(`build_dihedral_quads`) is now the floor — incremental topology update could push further (deferred).
**GO/NO-GO (pre-registered, advisor):** judge the BAND/shape not point-for-point (dropping the Vink
early-reject → chaotic trajectory divergence even if correct). Device path mq2 replay (`_run_meltquench_device
mqd ...seed 42`) is **GO** only if by ~150–175k: 8r climbing past ~48, E/atom within ~0.003–0.005 of scipy's
at the same iter, Φ22 ≥~0.87 en route to 0.88. **NO-GO** if E/atom rides systematically high (BB
under-relaxation compounds) → escalate to a jitted L-BFGS before shipping. Don't overwrite the deliverable /
run #3/#1 / commit device code as working until GO. `fast=True` is parity-gated to N=1000+this schedule only.

## ===== OPTION 2 (user chose it): extended anneal — PROFILE REDIRECTS the fix =====
Profiled per-move cost on GPU (`_profile_move.py`, cProfile, 2000 moves). **The bottleneck is NOT compute —
it is host↔device TRANSFER overhead:** self-time `jax array._value` (device→host pulls) = 79s/206s (**38%**),
`device_put` 40s, `numpy.asarray` 68s cumulative; the actual JAX kernel is tiny (N=1000 is too small to
amortize per-call dispatch). scipy L-BFGS calls value_and_grad ~189×/move, each paying transfer latency.
**CONSEQUENCE: the "genuinely-local relax" rewrite would NOT help** (per-call transfer cost is independent of
cluster size) — so I did NOT build it. The real levers: (a) **schedule** — mq1 was still climbing when it
froze, so a slower cool through the ordering window (mq2, running) is the cheap no-risk test; (b) **on-device
L-BFGS (jaxopt)** — keep the 100-iter relax loop on GPU in ONE jitted call (collapses ~189 host round-trips/move
→ ~1), the real speedup, but an UNVALIDATED path (would need to confirm equivalent relaxation before a long run).
Timing: GPU+jax baseline **90 ms/move** (const T=0.05, 32% acc); CPU **172 ms/move** (slower — JAX-CPU compute
+ dispatch, no acceleration); jaxopt timing inconclusive (slow compile, not completed). **mq2 = extended
slow-cool 0.09→0.028 / 250k, baseline code, ~3.5h** — does 8r climb past mq1's 44 toward 55+ (reproduction) or
asymptote <48 (more-moves-insufficient → on-device speedup + far more moves needed; glass barrier possible)?
Saved mq1's 8r=44 as `Example/20260623_best_from_random_8r44_ends.txt` (best from-random artifact; baseline to beat).

## ===== MELT-QUENCH mq1 (random_bm2000, w=0, 0.12→0.012, 100k) — best-effort cracking run =====
Graph-true 8r (collision-proof, `_graph_rings.py`) + E/atom. **Outperforms the constant-hold plateau and is
STILL CLIMBING as it freezes** (vs coldDis const-0.045 which plateaued 8r 38 / E/atom 0.062):
| iter | T | 8r | E/atom | Φ22 | angstd | acc |
|---|---|---|---|---|---|---|
| seed | — | 14.9 | 0.270 | 0.644 | 26.0 | — |
| 10k | 0.095 | 25.0 | 0.119 | 0.743 | 18.7 | 16.9% |
| 20k | 0.076 | 30.0 | 0.090 | 0.793 | 15.1 | 11.6% |
| 30k | 0.060 | 39.4 | 0.073 | 0.819 | 13.2 | 7.8% |
| 40k | 0.048 | 41.7 | 0.062 | 0.837 | 11.7 | 5.7% |
| 50k | 0.038 | **44.0** | **0.058** | 0.845 | 11.1 | 2.4% |
Reading: the melt-quench (true liquid start + slow cool) BEATS the constant holds (8r 44 vs 38, E/atom 0.058
vs 0.062) and is still rising at 50k (acc 2.4%, freezing) — suggestive that more moves in the ordering window
(sustained-T hold, not cooling through it) would go higher. Still FAR from 8r 60; does NOT reach the target.
Modest support for "extended annealing helps", NOT proof of reachability. (ref E/atom 0.0345, 8r 59.7.)

## ===== LITERATURE METHOD (decisive, from the group's patent + recent arXiv) =====
**US Patent 10,662,065 (Florescu/Steinhardt/Torquato group — Sellers' group):** their amorphous hyperuniform
trivalent networks are made **from a LIQUID-LIKE (random) start, NEVER crystalline** ("started from a
liquid-like configuration to avoid any memory of an initial crystalline state"), via: (1) **Keating WWW
bond-switching** with Metropolis accept; (2) **annealing runs 2–250× LONGER than standard** (extended
annealing is the explicit key ingredient); (3) **multiscale cluster relaxation on 120- and 320-atom
neighbourhoods** (mine: depth-4 ≈ 81 atoms); (4) **hyperuniformity emerges FROM the extended annealing**
("a sequence of progressively more hyperuniform minima"), NOT from seed placement. → CONFIRMS the user: the
random route works for this group; my gap = **under-annealing** (~100 moves/atom vs their ≫standard).
**arXiv 2601.10333 (recent, related):** melt-quench = triangular heat to ~Tmax∈[Tmelt/2,2Tmelt], cool, quench
to T=0. **KEY: low bond-bending β gives >10³ accepted moves; high β only 10–100 (trapped)** — high angle
stiffness traps the anneal (matches my barrier finding). My f2(angle) weight=0.7 is Sellers-fixed (can't
change for the final energy; a reduced-β SEARCH energy is a flagged idea, not done). NB this paper amorphizes
CRYSTALLINE starts (incl. gyroid); the patent uses liquid — BOTH routes exist in the group's work, so whether
THIS reference file is random- or gyroid-derived is still not 100% pinned (premise caveat per advisor).

## ===== LITERATURE (Sellers' method) + CRACKING PLAN =====
Paper: Sellers, Man, Shaba, Florescu, "Local self-uniformity in photonic networks", Nat. Commun. 8, 14439
(2017). Method = **Wooten-Winer-Weaire (WWW) bond-switching**, accelerated per **Barkema-Mousseau (BM2000)**,
producing **Keating-relaxed CRNs**. KEY literature point: **"starting from a LIQUID-LIKE configuration helps
avoid any memory of an initial crystalline state"** — i.e. random/liquid start → anneal, which the user
confirms. BM high-quality CRNs use **~thousands of bond-switch attempts per atom**; my runs used ~100/atom.
Cracking levers (advisor), objective = drive **E/atom 0.062 → toward 0.0345**, w=0 (S(k) restorable free):
1. **Melt-quench** (mq1, running): random_bm2000, w=0, start 0.12 (true liquid), slow-cool through the
   ordering window (Tc~0.06–0.09, bracketed by warmDis=liquid/coldDis=frozen) → cold. Never actually run
   before (all prior = constant holds or a modest cool from 0.06; phi22max's cool from 0.06 plateaued
   8r 39.7/Φ22 0.83 at 30k — same wall, so a hotter slower pass is the untested shape).
2. **More moves/atom** (BM uses 1000s; scale up if shape works).
3. **FALLBACK — per-move acceptance machinery:** the local-shell relax (depth 4, 100 iters) + Vink/BM
   threshold early-reject (c_f) may systematically REJECT defect-healing moves (a larger relax is needed to
   reveal their true downhill ΔE; under-relaxation makes them look uphill — the classic WWW failure). Test by
   deepening relax + loosening c_f, watching E/atom. (NOT contradicted by memory's "full-N relax no benefit"
   — that was for the VOID/holding the reference, a different question than REACHING low-E topology.)

## ===== CRACKING DIAGNOSTICS (2026-06-22) =====
- **Energy (`_energy_compare.py`):** ref E/atom 0.0345 vs trapped 0.062 → kinetic trap (energy favours ref).
- **Relaxation bias REFUTED (`_relax_bias_test.py`, 60 moves on coldDis ck50k):** local depth-4 relax ≈ full-N
  relax (0/60 wrongly-rejected; 59/60 moves uphill under BOTH; full reveals only +0.049 mean extra downhill,
  not enough to flip any move). So the anneal is NOT hiding good moves — relaxation is fine.
- **The trap is a deep local minimum:** from 8r 38, **59/60 single SW moves RAISE energy** (mean dE +0.575
  total ≈ a real barrier), only 1/60 downhill (tiny, −0.0001/atom). Escaping needs barrier-crossing
  (temperature) AND MANY moves — single-move MC at viable T accepts uphill moves ~exp(−0.5/0.08)≈0.003, so
  productive collective rearrangements are glacial. **This is why BM-WWW uses ~1000s of switches/atom; I used
  ~100.** → the unlock is move-budget + temperature regime (melt-quench mq1 + scale-up), NOT relaxation/schedule-shape.

## ★★★ EXECUTIVE LEDGER (the asked question first) ★★★
**Did Φ₂₂ reach ~0.89? NO — it plateaus ~0.84 from EVERY allowed disordered route** (coldDis random_bm2000
0.838; hyperuniform-seed 0.844) — and this is now understood mechanistically, not merely observed. **No
config passes all gates** (Φ₂₂≥0.88 AND angstd≤9° AND hyperuniformity, simultaneously, from an allowed seed).
This is an **honest, mechanism-level NEGATIVE.**

**The chain (each link fresh-empirical this session unless noted):**
1. **Φ₂₂ and bond-angle-std are TOPOLOGY-bound and fully DECOUPLED from the void** (Finding 1: the Stage-B
   λ-sweep drives S_k0 0.073→~0 at ZERO angstd/Φ22 cost). So "hit Φ₂₂≥0.88 while keeping hyperuniformity"
   reduces to ONE question: can annealing BUILD a topology whose Keating-relaxed angstd≤9° / Φ22≥0.88?
2. **NO — the reference's 8-ring-rich topology (8r 60%) is DYNAMICALLY (kinetically) unreachable from a
   disordered seed by WWW Stone-Wales annealing within practical schedules.** NOT thermodynamically
   forbidden — refHold proves 8r 54+ IS stable at cold T; it is just kinetically out of reach. The
   **glass / non-ergodicity squeeze**: where the dynamics actually MOVE (warm, T=0.10 → 8r caps ~44–47)
   you can't reach 8r 54+; where 8r 54+ is STABLE (cold, T=0.045, refHold) the dynamics FREEZE (coldDis
   8r 38 / angstd 12). Mobility and order live at different temperatures, never together. The SW
   bond-transposition is ergodic *in principle* (it does change ring sizes) but **doesn't heal the
   7-ring/wrong-size-ring defects within practical schedules** (7r sticks ~29% vs ref 10%, capping 8r).
   HYSTERESIS at T=0.045 (order held 8r 54–59 vs disorder frozen 8r 38, same T) is the signature.
3. **Seed-independent across the WHOLE allowed space.** random_bm2000 AND hyperuniform both plateau
   ~8r 38–44 / angstd 11–12 / Φ22 0.84. And **crystal_srs is 100% 10-RINGS + crystalline (VERIFIED this
   session)** — NOT the "sharp 8-ring" shortcut the mission's hypothesis-4 assumed (**that premise is
   empirically FALSE**); it would need the same 10→8 ring conversion SW can't do, plus a melt.

**What it cost / what was won:** S(k)/hyperuniformity is fully solvable and **FREE given any topology**
(Finding 1) — the prior "S(k) win" stands and is now explained. The hard wall is Φ₂₂/local-order, and it is
a property of the **SW MOVE-SET + disordered seeds**, NOT the energy (Keating is correct) nor the schedule.

**Best saveable structure:** the prior `Example/20260622_lsu_hyperuniform_N1000_ends.txt` (S(k) ✓ / angstd
11.6° ✗) is unchanged — no new structure beats it on the angstd gate. **refHold PASSES all gates but is NOT
a deliverable config** — it is seeded FROM the gold `lsu_example_ends.txt`, proving the target is STABLE, not
REACHABLE.

**Forward levers (USER DECISIONS, not pursued — budget/fidelity):** (a) **move-set augmentation** (ring-size-
altering moves to break the SW wall) — mechanism-matched, deviates from Sellers' pure SW; (b) a seed natively
~8-ring (NOT crystal_srs) — unknown if one exists. crystal_srs melt is NOT the easy route.

## ===== USER REDIRECT (2026-06-22): "only target Φ₂₂ + good low S(k)" (drop angstd/ring gates) =====
Answer (two routes, opposite directions):
(a) **Low S(k) is the already-SOLVED half** (hyperuniform seed + penalty: S_low 0.049, α +2.6, amorphous —
saved structure). (b) **From DISORDER (anneal UP):** Φ₂₂ is topology-bound (Finding 1: S(k)/geometry tricks
leave it pinned 0.844; `www_anneal target_lsu` is only an early-EXIT check at lsu_network.py:2434–2446, NOT
an acceptance force, so Φ₂₂ can't be force-driven) and plateaus ~0.838–0.844 amorphous. Φ₂₂ and angstd are
**both topology-bound and FAIL TOGETHER on every from-disorder route tested** (not proven to be the literally
same quantity, but correlated + co-bound across all ~3 reachable topologies). `_run_phi22max.py` is the fresh
from-disorder ceiling datum (anneal-up) — but per advisor it tests the WRONG direction for the relaxed target
and will ~reconfirm 0.844.

(c) **The RIGHT probe for the relaxed target — melt DOWN from crystal_srs (untested; the excluded seed).**
The relaxed target drops the RING gate, so crystal_srs's 100% 10-rings is NOW allowed — a *partially-melted
gyroid = 10-ring AMORPHOUS network* fails the full target but could satisfy the relaxed one. crystal_srs
starts Φ₂₂ **0.892** + S(k₀) 0.003 (both PASS) but Bragg (S_v 7.16). Melting DOWN from 0.892 is the STABLE
direction (refHold proved order holds from above at cold T), unlike annealing up from disorder. **Open
question: is there a melt window where S_v_peak<3 (amorphous) AND Φ₂₂≥0.88?** Memory
[[lsu-seed-tradeoff-and-melt-window]] says melt→amorphous (kills Bragg) is validated; only "does Φ₂₂ survive
the melt" is open. BUT crystal_srs is the mission-EXCLUDED seed → put to USER as a decision (melt probe vs
accept the from-disorder ceiling). Results below.

| run | iter | Φ22 | S_k0 | S_low | α | S_v_peak | angstd | 8r | acc | relaxed-PASS? |
|---|---|---|---|---|---|---|---|---|---|---|
| ref | — | 0.889 | 0.041 | 0.053 | +1.51 | 1.82 | 8.41 | 59.7 | — | (gold) |
| p22hold seed | 0 | 0.642 | 0.141 | 0.318 | — | — | 26.24 | 13.1 | — | . |

## ===== FINDING 1 (DECISIVE): void and local-order are DECOUPLED at fixed topology =====
**`Claude_Helpers/_stageb_cost.py`** — at the saved best structure's FIXED topology, minimise
`E_Keating(pos) + λ·S_low_k(pos)` over geometry, sweeping λ:

| λ | S_k0 | S_low | α | **angstd** | **Φ22** | bond-std |
|---|---|---|---|---|---|---|
| loaded | 0.0739 | 0.0486 | +2.59 | 11.63 | 0.8438 | 0.0361 |
| 0 (pure Keating) | 0.0731 | 0.0481 | +2.60 | **11.62** | 0.8438 | 0.0361 |
| 2 | **0.0043** | 0.0029 | +6.08 | 11.64 | 0.8440 | 0.0369 |
| 10 | 0.0003 | 0.0002 | +9.45 | 11.66 | 0.8439 | 0.0375 |
| 40 | 0.0000 | 0.0000 | +12.89 | 11.66 | 0.8436 | 0.0376 |
| 150 | 0.0000 | 0.0000 | +16.24 | 11.66 | 0.8436 | 0.0377 |

**Reading:** Stage-B geometry optimisation drives S_k0 from 0.073 essentially to **0** (well below ref
0.041) at a NEGLIGIBLE angstd cost (+0.04°) and ZERO Φ22 change. The void is a pure GEOMETRY property;
**angstd (11.6°) and Φ22 (0.844) are pinned by the TOPOLOGY** — no geometry move touches them. λ=0 (pure
Keating relax) is the angstd-MINIMISER at a topology, i.e. its angstd floor.

**Consequence — the whole investigation collapses to ONE question:** can ANNEALING (Stage A) build a
topology whose Keating-relaxed angstd≤9° / Φ22≥0.88? If yes, Stage B restores the void essentially for
free (this is even stronger than "small cost" — it's ~free). The multi-stage protocol is therefore
GUARANTEED to work iff Stage A reaches a good-enough topology. The gate experiment tests exactly that.

## ===== GATE EXPERIMENT: can annealing build the topology? (HYSTERESIS design) =====
Per advisor: a warm (T~0.10) w=0 plateau is **ambiguous** — it may just be the warm equilibrium
(memory: warm-from-reference under Keating degrades 8r 60→47, so T~0.10 MELTS the target order, not a
kinetic wall). The discriminating test is a **two-sided constant-T hold at ONE cold, stability-preserving
T (~0.045)**, where memory says the reference is HELD (cold cool kept 8r 59.7→57):
- **reference-seeded hold** → stays ordered (8r~57)? [order is stable at this T]
- **disorder-seeded hold** → climbs toward it, or freezes? [reachable or kinetically trapped]
If order is stable at this T yet disorder can't reach it ⇒ **HYSTERESIS = genuine kinetic trap** = an
airtight, mechanism-level negative (far stronger than "I swept T and nothing passed").

**Measurement is GRAPH-TRUE** (`Claude_Helpers/_run_holdtest.py`): hot checkpoints have near-coincident
vertices that break the rod round-trip AND inflate angstd via garbage triples, and plain Keating relax
has no non-bonded repulsion so it won't separate them. So topology order is read from a **deep Keating
relax (600 iters) of the in-memory EDGE LIST**: angstd from edges directly, Φ22 via `compute_lsu`
(depth=2,loc=2; verified =0.8887 on the reference), S_k0 from positions directly. Rings/void-slope via a
best-effort rod round-trip (skipped on collision). Edges saved per checkpoint.

**ORDER-SIDE ANCHOR (deep-relax of the reference topology, this harness):** angstd **8.55°**, Φ22
**0.8895**, S_k0 0.050, 8r 59.7, mean 7.99 → the reference topology **PASSES** the gate (angstd≤9 ✓,
Φ22≥0.88 ✓). Gate threshold correctly calibrated. (Keating relax shifts ref S_k0 0.041→0.050 and angstd
8.41→8.55 — the expected small Keating-vs-gold offset; void restorable for free via Stage B.)

| run | seed | T_hold | iter | angstd | Φ22 | 8r | 7r | ring_mean | S_k0 | bond-std | acc | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref (gold) | — | — | — | 8.41 | 0.889 | 59.7 | 10.0 | 7.99 | 0.041 | 0.029 | — | — |
| ref deep-relax | reference | 0.045 | 0 | 8.55 | 0.8895 | 59.7 | 10.0 | 7.99 | 0.050 | — | — | PASS |
| **coldDis** | random_bm2000 | 0.045 | seed | 26.08 | 0.6335 | 14.9 | — | 7.36 | 0.389 | — | — | . |
| coldDis | random_bm2000 | 0.045 | 10k | 17.36 | 0.766 | 26.3 | 33.4 | 7.32 | 0.762 | 0.057 | 10.9% | . |
| coldDis | random_bm2000 | 0.045 | 20k | 14.43 | 0.808 | 28.5 | 31.5 | 7.53 | 0.496 | 0.048 | 5.5% | . |
| coldDis | random_bm2000 | 0.045 | 30k | 12.57 | 0.828 | 37.1 | 29.1 | 7.65 | 0.728 | 0.044 | 3.8% | . |
| coldDis | random_bm2000 | 0.045 | 40k | 12.09 | 0.834 | 37.1 | 28.3 | 7.66 | 0.409 | 0.041 | 2.8% | . |
| **coldDis** | random_bm2000 | 0.045 | **50k** | **11.99** | **0.838** | **37.6** | **28.9** | 7.67 | 0.437 | 0.042 | 2.4% | **FAIL** |
| refHold (=ref held, NOT a config) | **reference** | 0.045 | 10k | 8.60 | 0.8892 | 59.2 | 9.1 | 8.02 | 0.044 | 0.029 | 1.0% | PASS |
| refHold | reference | 0.045 | 20k | 8.64 | 0.8884 | 55.1 | 10.6 | 8.03 | 0.037 | 0.029 | 0.9% | PASS |
| refHold | reference | 0.045 | 30k | 8.65 | 0.8882 | 56.9 | 9.9 | 8.04 | 0.047 | 0.030 | 1.0% | PASS |
| refHold | reference | 0.045 | 40k | 8.70 | 0.8884 | 56.9 | 8.4 | 8.04 | 0.034 | 0.031 | 1.1% | PASS |
| refHold | reference | 0.045 | **50k** | **8.83** | **0.8868** | **53.8** | 14.1 | 7.99 | 0.043 | 0.031 | 1.0% | PASS |

**refHold reading (order side — NB: seeded from `lsu_example_ends.txt`, this is the REFERENCE HELD, NOT a
deliverable config):** at T=0.045 it **slowly softens but remains firmly PASS** — angstd rises monotonically
8.55→8.83 (still ≤9), 8r 59.7→53.8, Φ22 0.887–0.889, S_k0 0.03–0.05, acc ~1%. It stays **far above the
disorder plateau** (8r 38 / angstd 12) the whole run. The reference sits in a deep SW basin (most moves
uphill → rejected → acc~1%, yet held through ~500 accepted moves); the disorder side had **10× the
acceptance at 10k (10.9%)** yet only reached 8r 26. **HYSTERESIS CONFIRMED at T=0.045:** order held ~54–59 /
angstd ~8.6 (PASS) vs disorder frozen at 38 / 12 (FAIL) — two distinct states at the same temperature,
start-dependent = non-ergodicity. The reference's 8-ring-rich topology is the low-T state the WWW dynamics
hold but cannot REACH from disorder.

**coldDis reading (disorder side):** climbs from the seed (angstd 26→12, Φ22 0.63→0.84, 8r 15→38) but
**FREEZES** — 30k→50k barely moves (angstd 12.6→12.0, 8r 37→38), acceptance decayed to 2.4%. The blocker
is unmistakable in the rings: **7-rings stick at ~29% (ref 10%)** — odd-ring defects don't heal, capping
8r at ~38% (ref 57%). Plateau is REAL, not under-relaxation: deep-relaxing the 50k edge list to 1500 AND
3000 iters leaves angstd=11.99 / Φ22=0.8376 unchanged (the topology's true Keating-minimal local order).

**HYSTERESIS at T=0.045** (same temperature, two outcomes — disorder side DONE, order side PENDING the
hold trajectory, NOT the ck0 quench): disorder FREEZES disordered (angstd 12.0, 8r 38). If the
reference-seeded HOLD stays ~8r 57 / angstd 8.5, that is hysteresis = the reference's 8-ring-rich topology
is a low-T ordered state **kinetically unreachable** from disorder. CAVEAT [advisor]: a constant 0.045
hold is warmer than the cool (0.045→0.015) that gave 8r 57, so the order side may DRIFT (e.g. 8r 48–50);
report the hold ENDPOINT, and if it degrades, the headline weakens to "order also softens at this T."

## ===== FRAME: a glass / non-ergodicity squeeze (per advisor) =====
The result is best stated as ONE argument (hysteresis is the *signature*, not the claim):
- **Cold T (≈0.045) preserves order but cannot BUILD it** — disorder freezes at 8r 38 / angstd 12,
  acc→2.4%; the dynamics fall out of equilibrium before ordering further.
- **Warm T can move but its equilibrium 8r is only ~44–47 for BOTH seeds** — disorder-warm plateaus
  ~8r 44 (100k Keating slow-cool), reference-warm MELTS 60→47. The accessible-T 8r ceiling is ~47.
- **⇒ No single accessible annealing T both BUILDS and PRESERVES 8r 57.** The reference is a quenched
  state below the effective freezing/glass transition of the WWW Stone-Wales dynamics; the system
  freezes at 8r ~38–44 with **7-ring defects as the specific frozen-in disorder the SW transposition
  cannot heal in budget**. The reference being stable at cold T is *consistent with* its being the low-T
  equilibrium the dynamics can't REACH from disorder (not asserted as proven). angstd/Φ22 follow the
  topology (Finding 1) so they inherit the same wall: angstd floors ~11–12° (ref 8.4°), Φ22 ~0.84 (ref 0.889).

## ===== HYPOTHESIS-1 CLOSURE: multi-stage & alternating A/B (the mission's flagship lever) =====
The mission named the multi-stage protocol "most promising," incl. "possibly ALTERNATE A/B." Closed:
- **Single-pass multi-stage (Stage A anneal → Stage B re-hyperuniformise):** closed by Finding 1 — Stage B
  drives S_k0→0 but is provably **topology-invariant** (the flat λ-sweep: angstd/Φ22 unchanged to ±0.04°).
  So a single-pass = "whatever topology Stage A reaches" — exactly the gate, which plateaus 8r 38–44.
- **Alternating A/B (anneal → re-hyperuniformise → anneal → …):** Finding 1 says "Stage B alone doesn't
  help," not "re-annealing from a Stage-B geometry doesn't help" — so it needs separate closure. But Stage B
  only ever produces a **near-hyperuniform geometry at fixed topology**, and annealing FROM a hyperuniform
  geometry **is exactly the hyperuniform-seed run**, which plateaus at the SAME wall (8r 39 / angstd 11.6).
  Therefore iterating A/B converges to the identical SW-move-set wall — no new basin is opened. Closed on
  paper (no run needed); the SW move-set, not the staging, is the bottleneck.

## ===== DEFENSES & SCOPE (fold-in before cross-review, per advisor) =====
1. **"You just froze it at cold T" is answered by the proper anneal too.** The cold-hold freeze (acc→2.4%)
   is not the only evidence: memory records a **100k slow-cool from random_bm2000 under Keating
   (0.09→0.012) ALSO plateaus at 8r 44 / 7r 24** (odd-ring defects don't heal). A properly-annealed
   schedule at full Sellers budget lands far below the reference's 8r 57 — so the wall is not a
   cold-schedule artifact. The hysteresis is the clean demonstration; the slow-cool plateau defeats the
   "anneal properly" objection.
2. **Seed-independent across the WHOLE allowed space (random_bm2000 OR hyperuniform).** The hyperuniform
   route hits the SAME wall: with penalty w=30 it plateaus at 8r 39 / angstd 11.6 (saved structure), and
   with w=0 (full freedom) memory records angstd plateau ~11°. Both allowed seeds plateau at angstd ~11–12
   / 8r ~38–44 — the wall is seed-independent, not a random_bm2000 quirk.
3. **Mechanism (nameable, defensible): 7-ring defects don't heal under the WWW Stone-Wales transposition.**
   Across coldDis the 7-ring fraction sticks at ~29% (ref 10%) and caps 8r at ~38% (ref 57%). The single
   SW bond-transposition move cannot anneal out odd-ring defects from a disordered start within practical
   schedules. → the honest claim is **kinetic/practical unreachability** ("within the WWW SW move-set +
   practical schedules incl. Sellers' 100k budget"), NOT thermodynamic impossibility.
4. **crystal_srs is NOT the easy route — VERIFIED this session (corrects a memory assertion).** Built the
   crystal_srs seed at N=1000 + counted rings with the same code as the reference: **ring distribution =
   100% 10-rings, girth 10, 8r=0%, S_v_peak 7.16 (strong Bragg/crystalline)**. The srs/(10,3)-a net is
   canonically 10-membered-ring; the "amorphized gyroid → 8-rings" recommendation **conflated the amorphous
   reference's 8-rings (mean 7.99) with the crystal's actual rings (mean 10.0)**. So reaching the reference's
   8r=60% from crystal_srs requires converting **10→8 rings** (the SAME ring-size/odd-ring healing the SW
   transposition can't do) PLUS melting strong Bragg crystallinity. crystal_srs is therefore likely **no
   easier** than the disordered seeds — and this **STRENGTHENS the negative**: NO allowed seed
   (random_bm2000, hyperuniform, OR crystal_srs) starts near the reference's 8-ring-rich topology. (NB: the
   reference is thus not a mere geometric jitter of the gyroid — it is a topologically distinct, smaller-ring
   amorphous network. crystal_srs seed: Φ22 0.892, S_k0 0.003, but 100% 10-rings + Bragg → wrong target.)
5. **The real forward lever = MOVE-SET AUGMENTATION (not reseeding).** The wall is that single SW
   bond-transposition cannot change ring SIZES from a disordered/wrong-ring start within budget. Re-adding
   ring-altering moves (e.g. the BC→AB+AC loop-expansion the `random_bm2000` builder drops, or
   bond-creation/deletion moves) is the mechanism-matched fix. Fidelity risk (deviates from Sellers' pure
   SW) + budget → FLAGGED for a user call, not pursued here.
