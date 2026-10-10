# EFF-Dock early-time/t=0 docking checkpoint

**Operator provenance:** these unchanged weights were trained with `legacy_rt_w` (Rᵀw), which does not make the full model exactly rotation-equivariant. The corrected `rw` inference option changes computation under these weights. See [the correction and limits](../docs/ORIENTATION_INJECTION.md).

- File: `effdock_docking_early_time_t0p10_50k.pt`
- SHA-256: `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`
- Model type: fragment-level flow matching on SE(3), with the historical orientation injection
- Training endpoint: 50,000-update early-time/t=0 replay fine-tune EMA
- Paired confidence checkpoint: `effdock_confidence_s50_raw_refined_u70k.pt`

## Intended use

Generate candidate ligand poses for a receptor, ligand chemistry, and explicit
binding-pocket center. The promoted inference preset uses 100 poses, 10 ODE
steps, translation sigma 2.0, a 10-Angstrom pocket crop, and a late-power-3
time grid. The paired U70k confidence model ranks the generated poses.

## Training intervention

The 50,000-update run initialized from the previous geometry checkpoint and
used the 47,277-sample filtered training loader from the preserved PLINDER split.
Its time distribution was `0.80 SimpleFold + 0.10 U(0,0.3) + 0.10 exact t=0`.
The run used fresh AdamW state, EMA decay 0.999, and a registered internal
PLINDER-validation endpoint.

Under the original training-time operator and validation protocol, internal
single-pose rollout success below 2 Angstrom increased from 192/1,076 at step 0
to 219/1,076 at step 50,000. These are checkpoint-selection records, rather than
results under the manuscript's inference-time coordinate conventions.

The manuscript evaluation keeps these weights frozen and explicitly uses `rw`
angular features, generation-consistent ligand references, and `contextual_v1`
confidence frames. Current external results are in the
[paired confidence model card](CONFIDENCE_MODEL_CARD.md) and
Supplementary Information. They do not establish a causal
benefit of individual architecture components or of the original fine-tuning intervention.

## Limitations

- Requires an explicit pocket; it does not discover pockets.
- Output depends on receptor preparation, ligand protonation/stereochemistry,
  pose count, sigma, ODE budget, crop, and seed.
- The model does not predict binding affinity or binder status.
- External redocking cohorts were opened during development and are
  descriptive rather than independent model-selection sets.
- The pinned public environment targets Linux and NVIDIA CUDA 13.

Exact sample IDs, exclusions and hashes are in
[training membership](../benchmarks/inputs/training_membership/README.md).
The public [training specification](../docs/methods/06_training_and_checkpoint_selection.md)
and current manuscript separate this
checkpoint's provenance from later three-repeat external characterization.
