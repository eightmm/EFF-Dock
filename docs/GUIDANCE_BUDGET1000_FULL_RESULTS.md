# Full-cohort unified-guidance fixed-budget results

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2`

Completed: 2026-07-31

## Decision

The full-cohort implementation goal passed: EFF-Dock audited and sampled all
`85/85` Astex Diverse and `308/308` PoseBusters v2 complexes, with no failed
complex, non-finite guidance trial, trust-region violation, OOM, or Slurm
failure. The official PoseBusters pass-all evaluation also completed for all
`2,358/2,358` oracle poses.

The performance hypothesis did not pass. Guidance preserved oracle RMSD
coverage but produced only small, mixed validity changes. No cell reached the
retained `+2 percentage point` joint fast-validity target, and every paired
95% interval for official pass-all validity included zero. Guidance therefore
remains diagnostic and is not admitted as the production sampler default.

## Why V1 did not use the full datasets

The missing coverage was not a CUDA-memory or ODE numerical failure. It came
from three independent admission/input problems:

1. The V1 launcher deliberately used the strict chemistry-eligible whitelist,
   so it admitted only `36/85` Astex and `94/308` PoseBusters complexes.
   Active cofactors, additives, and unresolved metal sites failed closed.
2. The Astex mapping for `1meh` supplied the nucleotide IMP, while the
   deposited ligand SDF is MOA. The frozen V2 input uses the MOA graph derived
   from that SDF; reference coordinates are still excluded from generation.
3. Three Astex inputs (`1owe`, `1uou`, and `1ygc`) and PoseBusters
   `8d5d_5dk` retained explicit stereochemical hydrogens under the
   generic loader. V2 applies `RemoveAllHs` only inside its frozen heavy-atom
   input contract. PoseBusters `5sak_zry` and `6zk5_imh` have a complete
   element/connectivity-preserving bijection but differ only in bond-order,
   tautomer, aromatic, or formal-charge representation; they are retained in
   an explicit representation-mismatch slice rather than matched by partial
   MCS or silently rewritten.

V2 removes the whitelist and introduces the separately named
`geometry_only` receptor policy. Standard protein chemistry is unchanged;
supported nonprotein heavy atoms are fixed repulsion-only obstacles, an
unparameterized atom is a bounded generic obstacle, and a metal site that
fails strict attraction admission is bounded repulsion-only. A fallback never
adds hydrogen-bond, charge, aromatic, hydrophobic, metal-attraction, or affinity
semantics. All runtime energy and gradients remain in-repository Torch code.

## Frozen setup

| Item | Value |
|---|---|
| Checkpoint | `effdock_geometry_ft_100k_best.pt`, step 100,000 |
| Checkpoint SHA-256 | `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db` |
| Input manifest SHA-256 | `99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668` |
| Combined audit SHA-256 | `d7321f847c8d6d08950e02d5f41ff42b62fd29ccea78072f27078aa039791c45` |
| Guidance implementation SHA-256 | `067bb59f3e5d377375aaa052a3c593b18417184b53e643bf92970c0abf226dd3` |
| Receptor policy | `geometry_only` |
| Prior | sigma `0.5 A`, pocket cutoff `10 A`, center jitter `0 A` |
| Arms | unguided scale `0`; guided scale `0.1` |
| Cells | N100/S10, N50/S20, N40/S25; each 1,000 learned pose-steps |
| Pairing | same global ID seed and exact 100-pose prior-pool hash |
| Confidence | disabled; all reported poses use the RMSD oracle |

The exact protocol and stopping rules are in
[`GUIDANCE_BUDGET1000_FULL_PROTOCOL.md`](GUIDANCE_BUDGET1000_FULL_PROTOCOL.md).

## Audit and numerical coverage

| Dataset | Discovered / audited / passed | Failed / non-finite | Chemistry slices: strict / nonprotein / metal / both | Ligand graph: exact / representation mismatch | Fixed obstacles | Repulsion-only metal fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Astex | 85 / 85 / 85 | 0 / 0 | 36 / 26 / 6 / 17 | 85 / 0 | 887 | 32 |
| PoseBusters v2 | 308 / 308 / 308 | 0 / 0 | 94 / 128 / 27 / 59 | 306 / 2 | 4,153 | 141 |
| Total | 393 / 393 / 393 | 0 / 0 | 130 / 154 / 33 / 76 | 391 / 2 | 5,040 | 173 |

Every reference-pose energy and coordinate gradient was finite. Total-energy
`min / Q1 / median / Q3 / max` was
`-43.993 / -29.125 / -22.412 / -17.579 / 9.753 kcal/mol` for Astex and
`-54.505 / -29.723 / -22.403 / -14.485 / 165.170 kcal/mol` for PoseBusters.
The strict-supported slice reproduced the V1 energy, gradient, and physical
system exactly (`36/36` Astex and `94/94` PoseBusters).

## Sampling results

The internal joint metric is
`any(fast-valid candidate with symmetry-aware RMSD < 2 A)`. Values below are
unguided to guided, with `Delta` defined as guided minus unguided. This is an
internal diagnostic; the next section reports the independent PoseBusters
implementation.

### Astex Diverse (85 complexes)

| Cell | Oracle <2 A | Oracle median RMSD (A) | Joint fast-valid and <2 A | Joint Delta (pp) |
|---|---:|---:|---:|---:|
| N100/S10 | 92.94 -> 92.94 | 0.801 -> 0.806 | 89.41 -> 89.41 | 0.00 |
| N50/S20 | 92.94 -> 92.94 | 0.792 -> 0.790 | 88.24 -> 88.24 | 0.00 |
| N40/S25 | 91.76 -> 91.76 | 0.832 -> 0.832 | 84.71 -> 85.88 | +1.18 |

### PoseBusters v2 (308 complexes)

| Cell | Oracle <2 A | Oracle median RMSD (A) | Joint fast-valid and <2 A | Joint Delta (pp) |
|---|---:|---:|---:|---:|
| N100/S10 | 92.86 -> 92.86 | 0.872 -> 0.866 | 82.47 -> 81.82 | -0.65 |
| N50/S20 | 92.21 -> 92.21 | 0.866 -> 0.866 | 77.92 -> 78.25 | +0.32 |
| N40/S25 | 92.86 -> 92.86 | 0.878 -> 0.877 | 78.90 -> 79.55 | +0.65 |

Guidance did not change the oracle `<2 A` count in any of the six cells. The
largest positive joint change was `+1.18 pp` on Astex N40/S25, below the
retained `+2 pp` descriptive guard. The changes are too small and mixed to
select a new step/pose allocation. N100/S10 remains the fixed-budget default
from V1; V2 does not provide evidence to replace it.

## Official PoseBusters 0.6.5 validity

Pass-all means all 27 non-RMSD `redock` checks pass for the saved RMSD-oracle
pose. RMSD itself is excluded from this validity label. Percentages are
unguided to guided; intervals are paired complex-ID bootstrap 95% intervals
for the percentage-point change (10,000 resamples, seed 20260731).

| Dataset | Cell | Valid count, unguided -> guided | Pass-all %, unguided -> guided | Delta (pp), 95% CI |
|---|---|---:|---:|---:|
| Astex | N100/S10 | 51 -> 50 / 85 | 60.00 -> 58.82 | -1.18 [-4.74, 2.35] |
| Astex | N50/S20 | 52 -> 51 / 85 | 61.18 -> 60.00 | -1.18 [-3.53, 0.00] |
| Astex | N40/S25 | 51 -> 50 / 85 | 60.00 -> 58.82 | -1.18 [-3.53, 0.00] |
| PoseBusters v2 | N100/S10 | 158 -> 158 / 308 | 51.30 -> 51.30 | 0.00 [-1.62, 1.62] |
| PoseBusters v2 | N50/S20 | 156 -> 158 / 308 | 50.65 -> 51.30 | +0.65 [0.00, 1.62] |
| PoseBusters v2 | N40/S25 | 161 -> 162 / 308 | 52.27 -> 52.60 | +0.32 [-0.65, 1.62] |

Within PoseBusters v2, the best guided official value is N40/S25 at `52.60%`,
but its paired change is only `+1/308` net complex and its interval includes
zero. It does not justify choosing N40/S25 or enabling guidance by default.

## Execution evidence

- CPU audit: Slurm `45371`; both dataset tasks completed with exit 0.
- GPU sampling: Slurm `45379`; all `48/48` array tasks completed with exit 0,
  yielding `96/96` arm/shard summaries and `2,358` complex-arm rows.
- Official validity: Slurm `45513` through `45524`; all `96/96` CPU tasks
  completed with exit 0 and evaluated `2,358/2,358` poses.
- Maximum CUDA allocated memory was `16.743 GiB`. Maximum reserved memory was
  `46.654 GiB` (allocator cache); there was no OOM or traceback.
- Maximum accepted atom displacement was `0.016884 A`, below the frozen
  `0.25 A` trust-region limit.
- The implementation gate passed `302` tests with `3` skipped; focused Ruff,
  shell syntax, and `git diff --check` gates also passed. Stderr contained only
  known Torch/Fx and RDKit/PoseBusters chemistry warnings.

Raw generated poses and per-shard output remain under the ignored
`outputs/benchmarks/guidance_budget1000_full_v2/` tree. Tracked, strict
machine-readable reports are:

- [`GUIDANCE_BUDGET1000_FULL_INPUTS.json`](GUIDANCE_BUDGET1000_FULL_INPUTS.json)
- [`GUIDANCE_BUDGET1000_FULL_COHORT.json`](GUIDANCE_BUDGET1000_FULL_COHORT.json)
- [`GUIDANCE_BUDGET1000_FULL_RESULTS.json`](GUIDANCE_BUDGET1000_FULL_RESULTS.json)
- [`GUIDANCE_BUDGET1000_FULL_POSEBUSTERS_RESULTS.json`](GUIDANCE_BUDGET1000_FULL_POSEBUSTERS_RESULTS.json)

## Claim boundary

These are reference-pocket redocking diagnostics, not an independent external
generalization estimate. V1 Astex/PoseBusters outcomes were already opened,
and the retained compatibility split overlaps ligand identities for `24/85`
Astex and `123/308` PoseBusters complexes; Astex `1meh` is an exact
entry-and-ligand overlap. The strict reports preserve matched/unmatched slices,
but no value in this study may tune a guidance term or support a production
admission claim.
