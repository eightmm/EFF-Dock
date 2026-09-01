# Fragment-Template Headroom Stereo Sensitivity Protocol

Protocol ID: `EFFDOCK-FRAGMENT-TEMPLATE-SWAP-HEADROOM-SENSITIVITY-V1`

Status: frozen before the full sensitivity outputs were generated. The failed
completion status and successful-subset metrics of the primary
`EFFDOCK-FRAGMENT-TEMPLATE-SWAP-HEADROOM-V1` analysis were already open. This
protocol is therefore a labelled robustness analysis, not a replacement
confirmatory protocol and not a new production-model admission gate.

## Purpose

The primary analysis required a whole-molecule stereo-preserving atom map. It
completed 378 of 393 complexes and rejected 14 exact constitutional matches
whose input and crystal stereochemistry disagree. One additional apparent SDF
failure was traced to the RDKit indexed reader seeking incorrectly at a
4096-byte buffer boundary; the serialized record and its registered hash are
intact.

This analysis asks whether the 14 stereo-mismatch systems could hide enough
fragment-template headroom to change the practical decision. It deliberately
computes an optimistic upper-bound sensitivity that may include correction of
stereochemistry that RDKit-template fine-tuning cannot learn.

## Frozen inputs and reader repair

- Use the same 85 Astex and 308 PoseBusters sigma-2, N=100 saved ensembles,
  frozen SMILES, crystal references, seeds, hashes, and fragment decomposition
  as the primary protocol.
- Read SDFs sequentially with `ForwardSDMolSupplier`. Do not change, rewrite,
  or substitute any saved record. All registered input hashes remain exact.
- Require all 393 complexes and all 100 candidates per complex to complete.

## Mapping policy

1. Use the unchanged complete stereo-preserving mapping and symmetry policy
   wherever it succeeds.
2. Only when that policy rejects a mapping solely for stereochemistry after a
   complete element- and connectivity-preserving bijection has been verified,
   enumerate every exact full constitutional match with chirality disabled.
3. Reject atom-count, element, connectivity, or representation-only partial
   mappings. Truncated symmetry enumeration also fails closed.
4. Select the non-chiral match with the minimum production-template
   fragment-rigid floor, identically to the primary selection rule.
5. Mark every such row `stereo_preserving=false` and
   `sensitivity_only=true` in the output.

The fragment-local crystal template is still transported independently in each
saved fragment frame. Whole-ligand crystal position and inter-fragment crystal
geometry are never copied.

## Interpretation

Report the same K2 and threshold-crossing metrics as the primary protocol,
including a separate total for sensitivity-only rows. The frozen primary gate
is not retroactively changed: the V1 result remains incomplete because its
stereo requirement was not met.

The full sensitivity result is used only as an optimistic robustness bound:

- a weighted mean delta below `+1.0 K2/100` rules out strong endpoint headroom
  even after granting stereo correction;
- a weighted mean absolute delta below `0.25 K2/100` supports deprioritizing a
  long RDKit-local fine-tune, while the original complex-crossing rule is
  reported rather than silently redefined;
- no sensitivity outcome alone admits a production model or proves that
  RDKit-local fine-tuning will reproduce the post-hoc swap.

## Registered outputs

```text
outputs/analysis/fragment_template_swap_headroom_sensitivity_v1/astex_sigma2_n100.json
outputs/analysis/fragment_template_swap_headroom_sensitivity_v1/posebusters_sigma2_n100.json
```
