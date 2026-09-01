# Frozen benchmark protocol

Protocol ID: `EFFDOCK-REDOCK-EMA-N40-S25-V1`

Frozen: 2026-07-19, before the new EFF-Dock GPU runs.

## Scope and non-claim

This is a reference-defined, oracle-pocket redocking diagnostic on Astex
Diverse (85) and PoseBusters v2 (308). The pocket center is
the centroid of receptor residue virtual nodes within 8 Angstrom of the frozen
reference ligand. The center is therefore target-derived. These values do not
measure target-independent pocket finding or prospective public inference and
must never be reported as such.

Reference ligand coordinates are used only to freeze this explicitly declared
benchmark site and to calculate RMSD. Sampling receives canonical SMILES, the
receptor with the reference ligand removed, and the frozen center. External
results are not used for training or further hyperparameter selection.

## Pre-registration

- Question: does the retained 200k EMA docking model still generate accurate
  poses after confidence/reranking code is removed, and can a fixed physical
  selector recover a useful top-1 pose?
- Hypothesis: the retained EMA sampler has high top-40 coverage, and the frozen
  EFF-Dock Vina+DG selector improves over sampling order without learned
  confidence.
- Prediction: PoseBusters v2 oracle top-40 success at RMSD <2A is at least 85%;
  Astex oracle top-40 is at least 90%; Vina+DG improves PoseBusters selected
  top-1 by at least 5 percentage points over the first-pose baseline.
- Baseline: the first sampled pose from the identical run. Historical
  FlowFrag/confidence results are context only and are not a selector baseline
  because their code and pocket manifests differ.
- Primary metric: PoseBusters v2 Vina+DG-selected top-1 symmetry-aware heavy
  atom RMSD <2A.
- Secondary metrics: first-pose and oracle top-40 RMSD success at <1/<2/<3/<5A,
  Astex selector results, failure rate, the fast DG/clash validity
  subset, and official PoseBusters validity for the PoseBusters Vina-selected
  poses.
- Failure threshold: invalidate a dataset run if more than 2% of frozen IDs
  fail preprocessing/sampling/scoring, or if any shard uses a different input,
  checkpoint, config, or center-manifest hash.
- Independent change: replace historical learned confidence selection with
  the fixed EFF-Dock torch Vina+DG selector; checkpoint and sampler remain the
  retained compatibility baseline.

## Frozen execution

- Checkpoint: `weights/effdock_legacy_flowfrag_200k_ema.pt`
  (`sha256:3ee604ec2338532532fa23a2ae91d0d540322defc32f5e453c8e7e12e389d36a`)
- Config: `configs/train.yaml`
  (`sha256:39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`)
- Samples: 40 independent poses per complex.
- ODE steps: 25; sigma: 1.0; time schedule: late, power 3.
- Pocket cutoff: 8A; center jitter: 0; refinement: none.
- Base seed: 42, deterministically offset by global sorted complex position.
- Selector: minimum EFF-Dock torch Vina energy plus DG strain, weight 1.0.
- Oracle: minimum RMSD among the same 40 poses; diagnostic only.
- Hardware/runtime metadata: Slurm job ID, GPU, CUDA, package lock, shard and
  stdout/stderr paths are retained with the outputs.

Input identities and per-complex receptor/reference hashes live in ignored
local manifests under `data/external_test/`. Final aggregate values and their
machine-readable summary are written to `docs/BENCHMARK_RESULTS.md` and
`outputs/benchmarks/summary.json`.

## Frozen source snapshots

| Dataset | Frozen IDs | Local archive SHA256 |
|---|---:|---|
| Astex Diverse | 85 | `c521c1cf5ef980211f44e60fa6d4e9f3e507dfad4957b682c7c2e16b16481fa6` |
| PoseBusters v2 | 308 | `49f38295155ae751aba3cdc31b7ed6bdeaeb8d730650ffb6c6b0d7ddff975f71` |

Astex and PoseBusters are the PoseBusters Zenodo benchmark snapshot. Every raw
receptor, cleaned receptor, ligand, and file hash is recorded in
`reference_redocking_manifest.json`.
