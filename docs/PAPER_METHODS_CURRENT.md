# EFF-Dock paper methods and experimental record

This is the manuscript source of truth for the released EFF-Dock stack. It separates deployed components from diagnostics and completed results from ongoing work. Linked protocols remain the immutable evidence.

## 1. Scope and released deployment contract

EFF-Dock performs supplied-pocket protein-ligand redocking. Inputs are a receptor structure, ligand chemistry, and an explicit pocket center; output is a ranked ligand-pose ensemble. It does not discover pockets, predict affinity, classify binders, or co-fold receptors and ligands.

| Component | Current paper identity |
|---|---|
| Docking model | `effdock_docking_early_time_t0p10_50k.pt`, 50k-update EMA, SHA256 `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6` |
| Confidence model | `effdock_confidence_s50_raw_refined_u70k.pt`, U70k, SHA256 `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638` |
| Generation | 100 poses, 10 ODE steps, translation-prior sigma 2.0 |
| Context | 10-A receptor crop; late time schedule, power 3 |
| Selection | Minimum predicted pose RMSD from U70k confidence |

The sampler emits raw poses. Any `refined` result is a separate labelled post-sampling evaluation condition and is not an implicit public API step.

## 2. Data, split, and molecular representation

Training structures originate from PLINDER 2024-06/v2. Coordinates are in Angstroms and the immutable sample key is `<system_id>__<ligand_instance_chain>`. Invalid structures are quarantined with an explicit reason instead of being zero-filled.

The released docking fine-tune used the preserved PLINDER compatibility split: 47,310 train and 1,076 validation identities before filtering, and 47,277 filtered train systems in the 50k run. This is released-checkpoint provenance; it predates the stricter current external-exclusion split contract.

For future replacement training, the strict split builder excludes canonical ligand SMILES in frozen external mappings and groups validation by `pocket_fident__70__community`. It requires train/validation disjointness by sample key, canonical SMILES, and pocket70 community.

Ligands are sanitized RDKit heavy-atom graphs. EFF-Dock cuts rotatable bonds that are single, non-ring, non-planar-conjugated, and have two nonterminal heavy-atom endpoints. Amide-like, ester-like, urea/carbamate, and sulfonamide bonds remain rigid. A deterministic greedy pass and singleton merging guarantee no one-atom fragment. Each fragment has a centroid, local atom coordinates, and a rotation; cut-bond, adjacency, and triangulation edges retain cross-fragment constraints.

Base training samples pocket crops uniformly from 6 to 12 A, applies uniform ligand rotation augmentation, and samples translation-prior sigmas `{0.5, 1, 2, 3, 4}` A with weights `{0.10, 0.25, 0.30, 0.25, 0.10}`. The deployed endpoint fixes sigma at 2.0 and crop at 10 A.

## 3. Fragment-level SE(3) flow model

One heterogeneous graph contains ligand atoms, ligand fragments, protein atoms, and protein residue virtual nodes. Ligand atoms encode element, charge, aromaticity, hybridization, ring, valence, chirality, and pharmacophore features. Protein atoms encode residue/atom identity and backbone, metal, and pharmacophore flags. Static edges encode ligand bonds, fragment ownership, cut bonds, triangulation, protein structure, and residue relations. Protein-ligand contact edges are rebuilt at a 5-A cutoff from evolving ligand coordinates.

The docking network contains six edge-type-specific O(3)-equivariant interaction layers, spherical harmonics through `l=2`, 32 radial basis functions, dropout 0.1, and a 128-dimensional time embedding. Node irreps contain 384 scalar, 32 odd-vector, 32 even-vector, 16 even rank-2, and 16 odd rank-2 channels.

The network predicts atom vector fields. Mean atom force yields fragment translation, while torque about the centroid is converted into angular velocity by rank-aware Newton-Euler aggregation. Unobservable rotation axes of single-atom or rank-deficient fragments are projected out. Rigid fragment transforms reconstruct full ligand coordinates.

Conditional flow matching regresses fragment translation and observable angular velocity. The base objective uses translation MSE, angular MSE (weight 8.0), atom auxiliary loss (0.3), and a one-step inter-fragment distance-geometry loss (3.0).

The released docking checkpoint starts from the preceding geometry model and uses `0.80 * SimpleFold + 0.10 * Uniform(0, 0.3) + 0.10 * delta(t=0)`. It ran 50,000 fresh AdamW updates on four GPUs with global batch 64, peak LR `2e-5`, weight decay 0.01, gradient clipping 1.0, EMA decay 0.999, 1,000-update warmup, and cosine decay from update 40,000 to final LR `2e-6`.

## 4. Sampling and confidence ranking

Fragment translations start from a pocket-centered Gaussian prior and follow a deterministic learned ODE. Main paper inference is N100/S10, sigma 2.0, late power-3 time grid, and no FK-SDE, Vina, or differentiable-energy guidance. All 100 candidate poses are persisted in sampling order with ensemble identity and confidence fields, enabling auditable label-blind reranking and Top-k/oracle analyses.

The confidence model uses the same graph and saved `t=1` ligand hidden representations from the paired docking model. It has four equivariant interaction layers with the same `l=2`, 32-RBF, and 5-A contact setup, followed by global-contact-attention pose readout. It predicts pose RMSD, pose success, atom displacement, and atom success.

U70k was warm-started from the terminal S50 symmetry-confidence state. Each complex contributes 32 raw sigma-2 poses, 32 deterministic-refinement poses, and one mapped crystal anchor. Labels are symmetry-aware no-alignment heavy-atom RDKit `CalcRMS`; crystal anchors have exact zero RMSD and are excluded from validation selection. The loss weights are atom 0.2, atom-success 0.2, pose 0.3, pose-success 0.4, rank 0.1, and success-listwise 1.0.

U70k was selected only on the fixed 1,035-complex PLINDER bank: 622/1,035 (60.10%) Top-1 below 2 A. U100k reached 617/1,035 (59.61%) and is not the default. Confidence is a within-ensemble ranking signal, not a calibrated cross-target RMSD, probability, or affinity estimate.

## 5. Evaluation and claims

The primary endpoint is selected Top-1 symmetry-aware no-alignment heavy-atom RMSD below 2 A. Secondary reports include Top-k and oracle success, official PoseBusters validity, and their same-pose conjunction. Crystal ligand coordinates are labels only; explicit frozen pocket centers are mandatory inputs and missing IDs fail before sampling.

Completed three-seed cohorts are Astex Diverse (85), PoseBusters v2 (308), PhiBench (206, including three reconstructed systems), FoldBench-Pocket full (558), and auxiliary OpenBind (925, including flagged systems and two noncovalent approximations of covalent systems). PhiBench and FoldBench are temporal checks; OpenBind is a dense single-protease auxiliary cohort. All headline rows are supplied-pocket redocking, not blind docking or co-folding. Astex and PoseBusters were opened during development, so they are descriptive rather than independent model-selection evidence. U70k was selected on the internal PLINDER bank, not external metrics. The uniform unguided main table and separate guided/budget ablations are generated in [the paper results](paper/20260919/RESULTS.md). FoldBench PB includes three explicitly disclosed energy-reference InChI-repair shards.

PoseX-SD/CD evaluation is complete for SD718/CD1312 and three seeds (101/202/303), including the original confidence and input-chirality+E/Z selection arms followed by the separately recorded PoseX relaxation protocol. SD and CD follow native evaluation/grouping; CD aggregates 109 groups and is not a simple per-case success percentage. The earlier fixed-receptor recovery attempts are historical, not interchangeable with the final recovered upstream results. See `POSEX_RECOVERED_EVALUATION.md`, `POSEX_STEREO_RELAX_CHAIN.md`, and the final CSV/summary identities when describing relaxation; do not describe EFF-Dock's own refinement as PoseX relaxation.

## 6. Paper map and boundaries

| Paper section | Current source |
|---|---|
| Task and data | This document sections 1-2; `DATA.md` |
| Architecture and objective | This document section 3; `MODEL.md`; `configs/train.yaml` |
| Inference and ranking | This document section 4; model cards; `EVALUATION.md` |
| Evaluation and results | This document section 5; `BENCHMARK_RESULTS.md` |
| Training provenance | `EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md`; `S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md` |

Do not claim blind docking, affinity prediction, calibrated confidence, or prospective screening. Do not merge raw and refined conditions. Do not present guidance, FK-SDE, Vina guidance, or fixed-receptor PoseX relaxation as the default sampler. Do not use opened external cohorts for post-hoc selection of a checkpoint, sigma, schedule, selector, or pocket center.

## 7. Evidence links

- `README.md`, `weights/MANIFEST.md`, and model cards: released identity.
- `DATA.md`, `MODEL.md`, `EVALUATION.md`: public method contracts.
- `EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md`: docking lineage.
- `S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md`: confidence training record.
- `BENCHMARK_RESULTS.md`: completed external results.
- `POSEX_OFFICIAL_PROTOCOL.md`: separate PoseX protocol; final recovered evaluation is complete.
