# Experiment index

This is the curated public experiment record. The complete machine-generated
ledger, scheduler commands, local paths, and raw logs are preserved outside
Git. Frozen protocols and results remain the authoritative source for exact
claims.

## Released model path

- Docking: 50k early-time/t=0 replay checkpoint; see
  `EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md` and
  `EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md`.
- Confidence: raw+refined+crystal-anchor U70k checkpoint; see
  `S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md` and
  `weights/CONFIDENCE_MODEL_CARD.md`.
- External results: `BENCHMARK_RESULTS.md` and
  `S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md`.

The older entries below are retained as an append-only narrative of negative
and superseded studies. They do not define the released default.

Each entry: ID, date, hypothesis, config diff, result, conclusion, wandb link.

---

## CONF-SELECT-0001 — Pose-set loss fine-tuning screen

- Date: 2026-07-20
- Hypothesis: a setwise-success, all-pairs success-ranking, or RMSD-listwise
  objective improves PLINDER val512 top-1 <2A by at least 1.5 points over the
  retained extmatch step-42500 checkpoint without harming the frozen composite.
- Config: same 4,096 matched N80/S25/sigma0.5/pocket10 training complexes,
  val512, seed 43, fresh Muon+AdamW, 1,500 steps; one loss change per run plus
  an unchanged-loss control.
- Result: baseline success/frozen = 58.59/57.03%. Final success/frozen was
  57.81/56.45 control, 58.40/57.62 setwise, 58.20/57.03 pairwise, and
  57.62/57.81 RMSD-listwise. Every best checkpoint remained step 42500.
- Conclusion: hypothesis not met. Do not run external benchmarks or promote a
  new weight; abandon these loss weights and retain step 42500. Prefer broader
  matched training data or richer pose features for the next study.
- Jobs: H100 control `38824`, setwise `38831`, pairwise `38880`, RMSD-listwise
  `38830`; three prior 48 GB attempts OOMed before their first update and are
  recorded as operational failures.
- Artifacts: `docs/CONFIDENCE_SELECTION_STUDY.md`,
  `outputs/eff-dock/confidence-selection-v1/*/metrics.json`.

---

## VINA-GUIDE-0001 — Inference-time Vina+DG guidance

- Date: 2026-07-19
- Hypothesis: frozen late-time Vina+DG guidance improves official PoseBusters
  pass-all by >=3pp while selected RMSD<2A loses <=2pp.
- Config: `EFFDOCK-VINA-GUIDANCE-V1`; N80/S25/sigma0.5/pocket10, scale 0.05,
  start_t 0.5, linear ramp, force cap 10, velocity caps 5, DG weight 1.
- Result: official validity 54.87% -> 56.17% (+1.30pp; paired bootstrap 95%
  CI [-0.32,+3.25], McNemar p=0.289); selected RMSD<2A 72.73% -> 72.73%;
  oracle-80 94.81% -> 94.81%. Sampling and official evaluation both 308/308.
- Conclusion: accuracy guardrail passed but the +3pp primary target failed.
  Keep guidance opt-in; tune any next configuration on PLINDER validation only.
- Jobs: `38791`/`38797`/`38799`, H100 rescue `38801`, official validity `38802`.
- Artifacts: `docs/VINA_GUIDANCE_PROTOCOL.md`,
  `docs/VINA_GUIDANCE_RESULTS.md`, `docs/VINA_GUIDANCE_RESULTS.json`.

---

## CONF-BENCH-0001 — Retained extmatch confidence baseline

- Date: 2026-07-19
- Hypothesis: on matched N80/S25/sigma0.5/pocket10 candidates, the frozen
  confidence selector improves PoseBusters top-1 over same-candidate Vina+DG
  and reproduces historical Astex/PoseBusters within 2 percentage points.
- Config: `EFFDOCK-CONFIDENCE-EXTMATCH-N80-S25-V1`; geometry-FT step 100000,
  confidence step 42500, seed 42, frozen reference-defined centers.
- Result: Astex/PoseBusters frozen-composite <2A = 78.82/72.73%;
  same-candidate Vina+DG = 77.65/71.10%; pure confidence = 76.47/73.05%;
  oracle-80 = 95.29/94.81%.
- Validity/completeness: official PoseBusters pass-all excluding RMSD = 54.87%
  (169/308); 678/678 final rows, with two recorded Astex H100 numerical
  rescues and no unresolved failures.
- Historical evidence: single-run frozen selector Astex 81.18%, PoseBusters
  77.60%; hard-pair fine-tunes failed to beat step 42500 on validation.
- Conclusion: the primary PoseBusters improvement prediction passed (+1.62pp
  over Vina), but historical reproduction failed (-2.35pp Astex, -4.87pp
  PoseBusters versus the ±2pp criterion). Keep the trained checkpoint; retain
  the composite for reproducibility and treat selector recalibration as a new
  validation-only study. Pure confidence was slightly stronger overall.
- Jobs: inference `38752`/`38760`/`38761`, Astex rescues `38770`/`38771`,
  official validity `38781`; confidence prepare/train smokes `38773`/`38762`.
- Artifacts: `docs/CONFIDENCE_BENCHMARK_PROTOCOL.md`,
  `outputs/benchmarks/confidence/summary.json`,
  `outputs/benchmarks/raw/effdock-confidence-extmatch-n80-s25-v1-*`.

---

## BENCH-0001 — Retained EMA oracle-pocket redocking baseline

- Date: 2026-07-19
- Hypothesis: retained EMA oracle-40 remains >=85% on PoseBusters and fixed
  Vina+DG improves selected top-1 over first-pose order by >=5 percentage
  points without learned confidence.
- Config: `EFFDOCK-REDOCK-EMA-N40-S25-V1`; N=40, 25 ODE steps, sigma=1,
  late schedule power=3, reference-defined 8A pocket center, seed=42.
- Result: Astex/PoseBusters Vina+DG <2A = 64.71/64.29%; oracle-40 =
  94.12/91.56%; official PoseBusters validity = 60.71%.
- Conclusion: hypothesis passed. Keep the EMA sampler and fixed physical
  selector as the no-confidence compatibility baseline; prioritize selection
  and large/flexible-ligand behavior before a target-independent release.
- Artifacts: `docs/BENCHMARK_RESULTS.md`, `outputs/benchmarks/summary.json`.

---

## MIGRATION-0001 — EFF-Dock compatibility baseline

- Date: 2026-07-17
- Hypothesis: the fragment SE(3) docking path can be isolated from historical
  reranking experiments while retaining safe access to released model weights.
- Config: `configs/train.yaml`; AdamW default; explicit pocket definitions.
- Result: migration verification only; no new benchmark number claimed.
- Conclusion: keep as the architecture/checkpoint compatibility baseline.
- Next: freeze benchmark mapping and pocket-center manifests, generate a
  strict benchmark-excluded split, then run GPU train/inference smokes.

---
