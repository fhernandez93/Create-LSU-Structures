# Independent causal-attribution + fidelity review

Reviewer: independent agent (2026-06-23). Scope: `claude_plans/phi22_gap_results.md` (MAJOR REFRAME,
LITERATURE METHOD, CRACKING DIAGNOSTICS) + memory `lsu-random-reachability-kinetic.md`. Each claim graded
SUPPORTED / OVER-CLAIMED / UNSUPPORTED with reasoning. Verdicts are evidence-based, not deferential to the
writeup's own framing.

---

## 1. Central causal claim — "reference reachable from random IN PRINCIPLE; my anneal merely UNDER-ANNEALS"

**Bounded version (reachable-in-principle / energetically favoured): SUPPORTED.**
**Headline / memory version ("my anneal just UNDER-ANNEALS"; "the topology wall / glass / hysteresis were
under-annealing artifacts"): OVER-CLAIMED.**

The energy diagnostic is real and correctly computed (`_energy_compare.py`: same box/D0/weights/Keating,
deep-relaxed 2000 iters): reference E/atom **0.0345** vs disorder plateau **0.0621**. That establishes the
reference is a **deeper Keating basin** — i.e. *thermodynamically favoured*. That conclusion is sound.

But the inference chain at lines 21–25 ("the energy WANTS the reference → my anneal just fails to REACH it →
the gap is KINETIC (annealing efficiency), NOT a topology wall") **does not follow from the energy result**,
and this is the central defect of the reframe:

- **Lower relaxed energy proves basin DEPTH, not the BARRIER between basins.** Reachability by WWW is governed
  by the barrier landscape connecting the disordered basin to the reference basin, which a single relaxed
  energy-per-atom comparison says nothing about. Thermodynamic favourability and kinetic inaccessibility are
  not in tension — **they coexist by definition in a trap.** The canonical counterexample is exactly this
  shape: a glass-former's crystal is far lower in energy and kinetically unreachable on any practical
  timescale. So "lower energy than my plateau" proves "energetically favored if you could get there," NOT
  "reachable from random by WWW." The reframe used an *orthogonal* (thermodynamic) result to license
  dismissing a *directly measured* (kinetic) one.

- **The session's own barrier diagnostic CORROBORATES the earlier kinetic-wall finding, not the reversal.**
  Line 72: "from 8r 38, 59/60 single SW moves RAISE energy (mean dE +0.575)." A low-lying target separated
  from a locally stable disordered basin by uphill barriers **is the textbook signature of a kinetic trap.**
  Properly combined, the new energy datum (deep target basin) + the barrier datum (uphill local moves)
  *strengthen* the earlier "kinetic wall / hysteresis / non-ergodicity squeeze" conclusion. The reframe
  presents them as overturning it.

- **The word "kinetic" silently changes meaning across the reversal.** Earlier: "real barriers the dynamics
  cannot cross within the WWW SW move-set + practical schedules." Reframe: "annealing *efficiency* — just
  needs more moves of the same kind." The energy diagnostic supports *neither* reading over the other. The
  only evidence genuinely pointing toward reach-in-practice is the **literature** (a different, better-
  localized, higher-budget implementation) — not the energy.

- **Not reproduced this session — so "merely under-anneals" is a hypothesis, not a result.** The writeup's own
  EXECUTIVE LEDGER (lines 78–81) is blunt: "Did Φ₂₂ reach ~0.89? NO — it plateaus ~0.84 from EVERY allowed
  disordered route." No from-random run this session reached 8r meaningfully above ~44 (coldDis froze at
  8r 38 / angstd 12.0 / Φ22 0.838 by 50k; mq1 in progress; phi22max climbing but plateauing ~0.84). With no
  run reaching the target, "the wall was an under-annealing artifact" overstates the evidence. The honest,
  *bounded* claim is the correct one: **energy + literature say reachable-in-principle, but it was NOT
  actually reproduced this session.** The memory headline ("the topology wall/glass/hysteresis were
  under-annealing artifacts") declares the wall dead on the strength of a diagnostic that cannot kill it.

**What the energy diagnostic legitimately DID accomplish (credit where due):** `_energy_compare.py`'s own
docstring frames a real fork — "if the reference is HIGHER energy, my energy doesn't favour it and the gap
is the energy/objective, not kinetics." The 0.0345 < 0.0621 result genuinely **refutes the wrong-objective
hypothesis**: the Keating energy is not mis-tuned against the target. That is a real, sound result. The error
is only in stretching it from "the objective favours the target" to "therefore my dynamics can reach it" —
the diagnostic adjudicates *objective correctness*, not *kinetic reachability*.

**Was the reversal justified, or did it flip too eagerly?** It flipped too eagerly **in rhetoric.** The
legitimate update is narrow and defensible: "not proven impossible; the energy favours the target; the
group's published method suggests a route my budget never tried." The illegitimate part is the memory file's
confident obituary for the kinetic wall. The writeup does NOT simply ignore the body-vs-reframe tension — it
explicitly **demotes** the EXECUTIVE LEDGER as "SUPERSEDED" (lines 28–30: "unreachable-by-my-schedules ≠
unreachable-in-principle"). But **that supersession is itself the over-claim.** A non-discriminating energy
diagnostic (basin depth) cannot license *retiring* a directly-measured negative — the 59/60-uphill barrier
and the frozen 8r-38 plateau. The writeup superseded the wrong conclusion: the LEDGER's mechanism-level
negative remains the honest core; the memory headline is the over-claim built on top of it.

---

## 2. Premise resolution — Patent 10,662,065 + arXiv 2601.10333 as evidence of a random/liquid route

**Honesty about residual ambiguity: SUPPORTED. But two citation-integrity caveats partly undercut the pillar.**

The writeup *is* honest about the residual ambiguity, and explicitly so (lines 44–46): it flags that the
arXiv paper amorphizes CRYSTALLINE starts (incl. gyroid), that the patent uses liquid, that "BOTH routes
exist in the group's work," and that "whether THIS reference file is random- or gyroid-derived is still not
100% pinned (premise caveat per advisor)." That is exactly the disclosure the question asks for — credit
where due.

However, the patent is the **single external pillar holding up the entire reversal**, and independent
verification surfaces problems the writeup does not flag:

- **Patent 10,662,065 exists but is mis-attributed.** It is real (Trustees of Princeton, granted 2020-05-26),
  and it does cover amorphous/near-hyperuniform trivalent random networks — topic match confirmed. BUT its
  inventors are **Steinhardt, Torquato, and Hejna** — *not* Florescu, and not obviously "Sellers' group."
  The writeup labels it "(Florescu/Steinhardt/Torquato group — Sellers' group)"; Florescu is not on it.
  Calling a Steinhardt/Torquato/Hejna Princeton patent "Sellers' group" is a stretch that **inflates how
  directly the patent speaks to THIS reference file's provenance.** The specific quoted phrases ("avoid any
  memory of an initial crystalline state", "annealing 2–250× LONGER", "120- & 320-atom neighbourhoods") could
  NOT be confirmed against the patent text by this reviewer and should be treated as **unverified in-session.**

- **arXiv 2601.10333 resolves, but to a different paper than described.** The ID is real and is
  "Computer Generation of Disordered Networks with Targeted Structural Properties" (Hemmann, Glauser, Steiner,
  Saba) — a WWW extension that controls disorder via **bond-bending force constant + temperature.** That
  *does* support the writeup's load-bearing point ("low bond-bending β → many accepted moves; high β → trapped").
  But the abstract does **not** confirm the "melt-quench, heat~Tmelt, cool, quench" framing nor the
  gyroid/crystalline-amorphization detail the writeup attributes to it; those specifics are unverified.

Net: the *premise honesty* the question asks about is genuinely present (SUPPORTED), but the external pillar
is weaker than the writeup implies — one mis-attributed patent with unverified quotes, one arXiv paper whose
specifics partly don't match. Since this literature is the *only* evidence genuinely pushing Q1 toward
"reachable in practice," these citation issues are load-bearing for Q1, not cosmetic. **Flag: the reversal
hinges on partly-unverified / mis-attributed citations.**

---

## 3. Proposed fix — "genuinely-local relaxation → more moves/atom"

**Mechanics / diagnosis of the bottleneck: SUPPORTED. Efficacy: predicated on the over-claimed Q1, and
internally contradicted by the writeup's own mechanism. GPU caveat: STATED (SUPPORTED).**

- **The mechanical premise is correct, verified in source.** `lsu_network.py:1790–1812`: the JAX path calls
  `_value_and_grad_jit(_energy_jax_full, ...)` over **all N** positions, then multiplies the gradient by the
  mask host-side (`g_arr = g_arr * mask_flat_j`). So the masked "local" relax pays **full-N energy/gradient
  cost every L-BFGS iteration** — frozen atoms are zeroed *after* the full compute, not skipped. The claim
  "each move is ~10× costlier than a genuinely-local 120–320-atom cluster relax → caps me at ~100 moves/atom"
  is a faithful description of the implementation. The reasoning is sound.

- **GPU caveat IS stated** (memory lines 38–39; writeup line 44 region): "On GPU the speedup may be 2–3× not
  10×, since full-N grad is already cheap there — scope before committing." Good. SUPPORTED.

- **BUT efficacy rests on the unproven move-COUNT hypothesis, which the writeup's own defense #5
  CONTRADICTS.** Defense #5 (lines 276–280) diagnoses the wall as the move-**set**: "single SW
  bond-transposition cannot change ring SIZES / heal 7-ring defects." A genuinely-local relax buys **more
  moves of the same SW type** — which only helps if move *count* (not move *set*) is the limiter. With 59/60
  local moves uphill from 8r 38, more-of-the-same SW is plausibly **necessary-but-not-sufficient.** The
  writeup holds two different diagnoses (count-limited vs set-limited) without reconciling them, and the
  proposed fix only addresses one. **Flag: fix mechanics sound, but its efficacy is predicated on the very
  reframe that is over-claimed (Q1), and contradicted by the writeup's own move-set mechanism.**

---

## 4. Hard constraints (fidelity) — all SUPPORTED (clean)

- **energy_weights = (0.7, 0.7, 0.3, 0.4), never changed in any run: SUPPORTED.** All 13 helper scripts in
  `Claude_Helpers/` define `WEIGHTS = (0.7, 0.7, 0.3, 0.4)` (or the dict form); grep finds zero deviations.
  No script reduces the f2/β weight — the "reduced-β SEARCH energy" idea is explicitly flagged as **not done**
  (writeup lines 44, 75), and the code confirms it was not done.

- **Keating energy kept (LSU_KEATING_F1F2 default on): SUPPORTED.** `lsu_network.py:1634`:
  `_KEATING_F1F2 = os.environ.get("LSU_KEATING_F1F2", "1") == "1"` — defaults ON. No script sets it to "0"
  (the only `=0` hit is the explanatory comment at line 1632). Every diagnostic prints `KEATING={...}` for
  audit, and `_energy_compare.py` runs with it on.

- **`Example/lsu_example_ends.txt` left UNTOUCHED: SUPPORTED.** `ls -la` shows mtime **May 2 20:38** —
  predates this session (2026-06-22/23) by weeks; committed 4701f3b. The energy diagnostic and gate anchors
  *read* it (load-only); nothing writes it. Untouched.

- **No forbidden crystal_srs ANNEAL run: SUPPORTED.** crystal_srs appears in exactly one helper,
  `_seed_compare.py`, which only calls `crystal_seed_network(...)` then `stats(...)` (ring count + S(k)
  diagnostics) — a cheap **seed + ring-count**, no `generate_lsu_network` / `www_anneal` / WWW loop. The
  output `Structures/_crystal_srs_seedcheck.txt` is a seed dump, not an anneal trajectory (no `_ckNk`
  checkpoints exist for it, unlike coldDis/refHold/mq1). The melt-DOWN-from-crystal_srs probe is correctly
  left as a flagged USER DECISION (lines 114–133), not executed. Allowed-only usage confirmed.

---

## Bottom line

The fidelity discipline (Q4) is intact and the premise honesty (Q2) is genuinely present. The failure is Q1:
a thermodynamic result (deeper basin) was used to overturn a kinetic result (measured barriers + frozen
plateau) that it cannot logically reach, and the memory file canonicalized that over-claim as a reversal —
contradicting the writeup's own EXECUTIVE LEDGER, which remains the honest mechanism-level negative.

---

### 6-line summary (with OVER-CLAIM flags)

1. **Q1 OVER-CLAIMED:** "lower energy ⇒ my anneal merely under-anneals / the wall was an artifact" is invalid —
   lower relaxed energy proves basin DEPTH, not the BARRIER; thermo-favoured + kinetically-trapped coexist by
   definition. Only the bounded "reachable-in-principle, NOT reproduced this session" is SUPPORTED.
2. **Q1 wrong supersession:** the writeup explicitly demotes the EXECUTIVE LEDGER ("honest mechanism-level
   NEGATIVE") as "SUPERSEDED" (lines 28–30), but that demotion IS the over-claim — a non-discriminating energy
   datum cannot retire the directly-measured negative (59/60 uphill + frozen 8r-38 plateau), which in fact
   *corroborates* the kinetic wall. (The energy DID legitimately kill the wrong-objective hypothesis — fair
   credit — it just cannot adjudicate kinetic reachability.) The reversal flipped too eagerly in rhetoric.
3. **Q2 SUPPORTED (honest about ambiguity) — but CITATION FLAG:** Patent 10,662,065 is real but mis-attributed
   (inventors Steinhardt/Torquato/Hejna, not Florescu; "Sellers' group" is a stretch) with unverified quotes;
   arXiv 2601.10333 resolves to Hemmann et al. (bond-bending/temperature claim supported, melt-quench/gyroid
   specifics unverified). The reversal's only "reachable-in-practice" pillar is partly unverified.
4. **Q3 mechanics SUPPORTED** (confirmed at lsu_network.py:1790–1812: full-N grad then host-side mask → masked
   relax pays full-N cost; GPU 2–3× caveat IS stated) — **but efficacy FLAG:** it fixes move-COUNT while the
   writeup's own defense #5 blames the move-SET (SW can't heal 7-ring defects); the two diagnoses are unreconciled.
5. **Q4 ALL SUPPORTED / CLEAN:** weights (0.7,0.7,0.3,0.4) in all 13 helpers, never changed; LSU_KEATING_F1F2
   default ON, never set to 0; `Example/lsu_example_ends.txt` untouched (mtime May 2, predates session);
   crystal_srs used only as seed+ring-count in `_seed_compare.py`, no forbidden anneal run.
6. **Net:** fidelity intact; the honest core is the EXECUTIVE LEDGER's negative; the headline reversal is the
   over-claim and should be demoted to "reachable-in-principle, unreproduced — hypothesis pending a run."
