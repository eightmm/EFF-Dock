# Manuscript-ready analysis text

Draft prose below is evidence-bounded. It is not a claim of statistical
significance or independent benchmark generalization. Keep the limitations
with the tables rather than moving them out of view.

## Methods — evaluation design

We evaluated a fixed early-time S50 docking model and the internally selected
U70k confidence model using supplied receptor pockets. The main sampling
configuration generated 100 candidate poses using 10 ODE steps, translation
prior σ=2 Å, a 10 Å receptor crop, and the late-power-3 time schedule without
energy guidance. We evaluated raw and separately refined pose banks, with
and without an input-SMILES tetrahedral-chirality selection mask. Candidate
selection used minimum predicted RMSD; when no candidate satisfied the mask,
the original confidence Top-1 was retained. No reference structure, measured
RMSD, or PB-validity label entered candidate selection.

For Astex85, PoseBusters308, PhiBench206, FoldBench558 and OpenBind925, accuracy
was measured by symmetry-aware, no-alignment heavy-atom RMSD strictly below
2 Å. Selected-pose validity and its conjunction with RMSD success were
reported separately. All original systems were retained and each metric was
computed per repeat, then summarized by the mean and sample standard deviation
over three sampling seeds. The FoldBench evaluation includes three explicitly
documented PB energy-reference compatibility-repair shards. PhiBench contains
three reconstructed cases and OpenBind contains quality-flagged systems and
two noncovalent approximations of covalent ligands.

We used saved candidate-level confidence scores and symmetry-aware RMSD labels
to distinguish sampling coverage from selection performance. Ordinary Top-5
success required at least one near-native pose among the five lowest predicted
RMSDs; oracle coverage required at least one among all candidates. The
near-native candidate fraction was averaged equally over complexes. Underlying
score files, their candidate order, both raw/refined vectors and their hashes
were independently checked against the saved selection ledgers.

Training exposure was annotated against the executed S50 fine-tuning loader index, with
train and validation treated separately. Ligand identity used canonical
heavy-atom isomeric SMILES; nearest-training similarity used radius-2,
2,048-bit Morgan fingerprints without chirality and Tanimoto similarity.
The parseable reference index covered 47,276 of 47,277 executed train entries.
One unresolved valence-invalid training SMILES was retained as missing, not
repaired or treated as proof of no overlap. The executed index was verified
by reconstructing its cache key from the ordered preserved 47,310-system split
and loader filters, and by checking its content hash. This audit does not
reconstruct the union of all predecessor pretraining data.

## Results — postprocessing and guidance

With refinement and chirality-based selection, the unguided N100/S10 model
achieved joint RMSD/PB-valid success of 79.22±2.72% on Astex and 79.22±1.42% on
PoseBusters. The corresponding values were 60.19±3.36% on PhiBench,
72.82±0.63% on FoldBench and 52.58±0.61% on OpenBind. Refinement substantially
increased the mean joint success in all five saved cohorts; chirality selection
after refinement gave a smaller additional change. These observations concern
the evaluated frozen banks, not universal monotonic gains for every complex.

The paired Astex/PoseBusters ablation used exactly matching initial-prior
hashes within each pose/step budget. Guidance improved raw-bank joint success,
but the final refined-and-filtered differences were small: guided eta2 yielded
78.43±1.36% versus unguided 79.22±2.72% on Astex, and 79.55±1.49% versus
79.22±1.42% on PoseBusters. Thus the completed external experiment does not
establish that guidance is necessary for the final pipeline. N100/S10 had
higher final mean success than N40/S25 in both datasets, while N40/S25 was less
expensive in the observed end-to-end driver accounting.

## Results — candidate coverage and selection headroom

For the refined unguided N100/S10 bank, ordinary confidence Top-1/Top-5/oracle
RMSD success was 81.96/92.94/96.86% on Astex, 82.14/89.18/94.16% on
PoseBusters, 63.27/78.48/88.83% on PhiBench, 74.43/85.36/91.70% on FoldBench,
and 52.79/73.69/87.86% on OpenBind. These are RMSD-only endpoints before the
chirality-selection mask, not the joint endpoints in the main table.

The mean number of near-native candidates per 100 was 34.45, 31.69, 22.96,
26.02 and 6.32 respectively. In particular, OpenBind combined high oracle
coverage with sparse near-native candidates and a large selection gap.
This motivates distinguishing sampler coverage, near-native density and
confidence ranking rather than attributing all failure to any one component.
Oracle gaps quantify available headroom but do not guarantee that retraining
confidence will recover that headroom.

## Results — training exposure

The parseable preserved train index contained exact ligand identities for
24/85 Astex, 121/308 PoseBusters, 93/206 PhiBench, 419/558 FoldBench and 66/925
OpenBind systems. FoldBench also contained 341/558 exact receptor PDB-accession
matches. These observations preclude describing the full FoldBench cohort as
uniformly unseen-target evaluation under this checkpoint lineage.

Among systems with no observed exact ligand match in the parseable index,
refined-and-filtered joint success was 82.51% (Astex, n=61), 83.78%
(PoseBusters, n=187), 55.46% (PhiBench, n=113), 66.67% (FoldBench, n=139) and
54.17% (OpenBind, n=859). Full strata, denominators and repeat SD are given in
the exposure table. These are not “leakage-free” subsets: protein/pocket
similarity is unmeasured, one train ligand is unresolved, and stratum
composition differs. We report observed matches rather than deleting cases
post hoc or interpreting these differences as causal memorization effects.

## Results — cost and sensitivity

The matched unguided runs used approximately 1.84× (Astex) and 1.83×
(PoseBusters) summed shard-wall seconds per complex for N100/S10 relative to
N40/S25. Sampling CUDA allocated-memory peaks were approximately 2.53× and
2.56× higher. The wall-cost accounting includes refinement of all candidates,
both raw/refined confidence banks, model loading and I/O; it is not pure
sampling latency or campaign wall-clock duration. N100/S10 and N40/S25 share
1,000 learned pose steps but not the same postprocessing workload or memory.

Amortized pipeline cost per generated pose was approximately 1.039 s (Astex)
and 1.037 s (PoseBusters) for N100/S10, versus 1.409 s and 1.419 s for N40/S25.
These costs divide the measured per-complex pipeline wall accounting by N;
they are not single-pose latencies. Confidence scoring covers 2N raw/refined
pose evaluations and is normalized by 2N instead. Sampling-process GPU peaks
are never divided by candidate count because model and workspace memory are
shared. The full stage-specific table includes device and cohort denominators.

The previously completed guided-eta2 pocket/prior study showed lower joint
success with larger pocket-center perturbations and condition-dependent
trade-offs across receptor crop and translation-prior width. At cutoff10/prior2,
raising per-axis center-jitter σ from 0 to 2 Å changed mean joint success from
78.04 to 70.20% on Astex and from 78.90 to 65.58% on PoseBusters. This older
refined/confidence-only condition is distinct from the new chirality-filtered
unguided main table and does not select a new production prior or crop.

## Limitations and pending measurements

The preserved split differs from the later strict external-exclusion design;
training exposure and opened-benchmark reuse limit independent generalization
claims. No external-to-training protein/pocket nearest-neighbor matrix is
available here, and exact PDB-accession mismatch is not sequence novelty.
Official selected-pose PB labels cannot establish all-candidate PB validity,
joint-PB Top-5 or joint-PB oracle. Sampling-only wall time, host peak RSS and
refinement/confidence GPU peaks were not recorded. Three-repeat SD measures
sampling variability, not target-level statistical uncertainty. OpenBind is
an auxiliary dense single-protease cohort. PoseX uses separate relaxation,
threshold and grouping conventions and is reported separately.
