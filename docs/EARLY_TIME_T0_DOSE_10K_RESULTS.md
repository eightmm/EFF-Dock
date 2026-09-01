# Paired Exact-Zero Dose Continuation Results

## Decision

Do **not** increase exact `t=0` sampling from 10% to 15%, and do not launch a
20% dose or another continuation from this result. The registered paired audit
is invalid because the two nominally identical step-0 rollouts were not
bit-for-bit equal. Independently of that validity failure, the descriptive
endpoint direction is negative on the primary metric: the 15% treatment found
`215/1,076` complexes below 2 Angstrom versus `220/1,076` for the matched 10%
control, a difference of `-5` complexes (`-0.4647` percentage points). The
registered gate required at least `+11` complexes.

## Results

| Arm / update | Median RMSD (A) | Success below 2 A | Success below 5 A | Centroid distance (A) |
|---|---:|---:|---:|---:|
| Control 10%, S0 | 3.919521 | 219/1,076 | 675/1,076 | 1.434686 |
| Treatment 15%, S0 | 3.919987 | 219/1,076 | 674/1,076 | 1.435105 |
| Control 10%, S10k | 3.877671 | 220/1,076 | 671/1,076 | 1.455438 |
| Treatment 15%, S10k | 3.839773 | 215/1,076 | 679/1,076 | 1.442599 |

At S10k, treatment minus control was:

- success below 2 A: `-5` complexes (`-0.4647 pp`), failing the primary gate;
- success below 5 A: `+8` complexes (`+0.7435 pp`), passing the secondary
  non-inferiority bound;
- median RMSD: `-0.03790 A` (lower is better), passing the secondary bound.

The secondary improvements do not rescue the failed primary endpoint.
Treatment also ended below the nominal shared S0 `<2 A` count (`215` versus
`219`), failing the retention condition.

## Audit validity

Both producer tasks in array `53926` completed 10,000 updates and wrote frozen
config, split, common-initialization, code, and artifact hashes. The original
dependent audit `53927` failed before the Python audit because its live-tree
hash covered unrelated inference-only files changed after launch. A separately
identified recovery audit preserved all producer and artifact checks while
hashing the exact unchanged audit dependency closure.

That recovery correctly failed the unchanged registered audit on the S0
identity requirement:

```text
paired S0 mismatch for rollout/centroid_dist:
1.4346861839294434 != 1.4351050853729248
```

The S0 weights originate from the same common EMA initialization, but the
configuration is non-deterministic and the two GPU rollouts differ slightly in
centroid distance, median RMSD, success below 5 A, and validation loss. The
comparison must therefore remain descriptive rather than be relabelled a
passed paired audit. Weakening the frozen equality check or rerunning until it
passes would be invalid.

Future paired time-sampling studies should freeze and reuse identical rollout
priors/initial rotations (or save the complete rollout input tensors) and test
weight identity separately from deterministic metric identity.

## Provenance

- Protocol: `docs/EARLY_TIME_T0_DOSE_10K_PROTOCOL.md`
- Producer array: `53926_0` control, `53926_1` treatment; both
  `COMPLETED (0:0)`
- Original audit: `53927`, failed before the registered Python comparison
- Recovery attempts: `54215` exposed missing failure observability; `54217`
  preserved the exact S0 mismatch and failed as required
- Recovery envelope SHA256:
  `8cc746551275e248426342249d09bb38be261b29022b146658eacbc8c61ee8ec`
- Control S10k/latest SHA256:
  `066139a5ae2f5c65ee954403b54c514a186eaee7d362dc0227c394e392cb774a`
- Treatment S10k/latest SHA256:
  `d3216d483ba26fdca9586d4dab26afb0156ddd19be4bf42af305fd5926ea6253`
