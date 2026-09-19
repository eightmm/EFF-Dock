# PoseX official-protocol EFF-Dock evaluation

This protocol adds an EFF-Dock row to the published PoseX evaluation without
rerunning the reported baselines. It is separate from the supplied-pocket
diagnostic reports.

- Official code: `CataAI/PoseX` commit
  `3d563881f9744e480e26799bbd9ebf88be1ff435`.
- Original `posex_cif.zip` is retained with SHA256
  `9fab878c2c972825b149e0ee88596f07cacc78bd540b654259ae124d97d7dcc6`.
  The upstream relaxer cannot process all supplied fixed-receptor PDBs because
  CIF-based sequence-gap reconstruction can create duplicate author-residue
  identifiers. The EFF-Dock relaxation condition therefore retains observed
  receptor residues, adds missing atoms/hydrogens, and uses the upstream
  OpenMM/OpenFF relaxation and evaluator unchanged. It is labelled
  `fixed-receptor relaxed`, not unmodified official relaxation.
- Cohorts: all 718 PoseX-SD and all 1,312 PoseX-CD items from the pinned
  `posex_set.zip` release.
- Inference: promoted U70k confidence checkpoint, N100/S10, sigma 2.0,
  late-power-3 schedule, unguided deterministic ODE, and pure confidence
  Top-1 ranking. It uses the supplied receptor and reference-ligand-derived
  pocket center; no oracle-best candidate may be selected.
- Repeats: three independent sampling seeds. Results are reported as mean and
  standard deviation across those runs.
- Export: each selected EFF-Dock SDF and unchanged supplied receptor are
  materialized as `ID_model_ligand.sdf` and `ID_model_protein.pdb`, the input
  schema expected by the unmodified official alignment/evaluation scripts.
- Evaluation: raw output is processed by the official alignment and `redock`
  PoseBusters evaluator. The optional official PoseX OpenMM/OpenFF relaxation
  condition is reported only after its strict per-case coverage gate passes;
  the relaxer can otherwise silently omit failed systems, which would change
  the benchmark denominator. The archived official evaluator environment is
  used unchanged.
- Metrics: Top-1 RMSD <=2 A and Top-1 RMSD <=2 A plus PB-valid. PoseX-CD is
  aggregated by official cross-docking `GROUP`, rather than by individual
  pair; PoseX-SD is per item.

The literature table must compare EFF-Dock only to PoseX's Pocket-Given
methods. Blind and co-folding methods remain separately labelled because they
do not receive an explicit pocket. The use of PoseX's external OpenMM
relaxation is evaluation-only and is not part of EFF-Dock inference.
