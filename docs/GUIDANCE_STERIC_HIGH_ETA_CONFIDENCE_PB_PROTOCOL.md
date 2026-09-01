# EFF-Dock steric high-eta confidence/PoseBusters protocol

Protocol ID: `EFFDOCK-UNIFIED-GUIDANCE-STERIC-HIGH-ETA-CONFIDENCE-PB-V1`

Status: frozen before sampling. This is a paired descriptive external-benchmark
dose-response study. It cannot select an eta, tune a guidance term or selector,
or admit guidance to the production sampler.

Infrastructure correction (2026-08-03): the initial
`20260802T084006Z` smoke attempt is invalid and excluded because inherited
`SBATCH_*` values silently placed its CUDA tasks on the `test` partition and
an RTX A5000. No full-sampling, official-PoseBusters, or report stage ran from
that root. The fresh rerun clears inherited submission overrides, pins GPU
stages to `6000ada` and CPU stages to `cpu_only`, and verifies RTX 6000 Ada
identity and visible memory before inference. These are execution safeguards;
all scientific arms, inputs, seeds, equations, coefficients, caps, selectors,
and outcome gates below remain unchanged. The fresh execution manifest is the
sole source/runtime identity for the rerun, and no artifact from the invalid
root may be merged into it.

Execution-capsule correction (2026-08-04): the later
`20260803T063226Z` attempt is also excluded before sampling. Its first GPU
array remained pending, but two manifest-covered source files changed after
submission, so the fail-closed gate would have rejected it before ODE
evaluation. Replacement runs execute from a per-run, read-only code capsule
under `.effdock_execution_capsules/`; the capsule source and Slurm stage files
are content-addressed in the execution manifest and imported ahead of the
editable environment. Large data, weights, and the Python environment remain
linked to their declared repository locations, while every selected frozen
input retains its existing exact hash gate. This is an execution-isolation
change only and does not change an arm, input, seed, term, cap, or outcome.

Scheduler-availability correction (2026-08-05): replacement CUDA smoke,
stress, and full-sampling arrays may use the ordered Slurm partition list
`6000ada,heavy`. Every task still requests exactly one GPU and requires at
least 48,000 MiB visible memory. Tasks placed on `6000ada` retain the strict
RTX 6000 Ada identity guard; `heavy` is restricted by cluster inventory to
H100 and RTX PRO 6000-class devices. The actual partition, GPU name, and
visible memory are recorded, and the audit retains the observed GPU-name and
memory sets. This is scheduler expansion only: eta arms, `N100/S10`, inputs,
seeds, equations, caps, confidence selectors, and all outcome gates remain
unchanged.

Post-smoke prior-hash audit amendment (2026-08-06): after the smoke outputs
were inspected, the high-eta integrity audit was revised transparently to
`EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2`, schema
`effdock.guidance_steric_high_eta_confidence_integrity.v2`. The exact
`sampling_seed`, prior-pool size `100`, and declared
`EFFDOCK_SHARED_PRIOR_V1` construction contract remain fail-closed. Each
per-row `prior_pool_sha256` must still be a valid digest, but equality of that
digest across eta arms is diagnostic rather than a gate. The prior is produced
by a CPU Torch trigonometric/normalization path. The observed digest difference
coincided with mixed node partition/GPU assignments, which is retained as
association-only provenance and does not establish GPU causation. Original
float32 prior tensors were not persisted, so their exact tensor delta cannot be
reconstructed. Every differing complex ID is retained with its hash set and
per-eta partition/GPU provenance. No selector, guidance setting, candidate
count, seed, or downstream outcome gate changes. The historical standalone
audit V1 remains strict and is not reinterpreted.
Each full official shard uses
`EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_OFFICIAL_BINDING_V2` to bind the exact V2
integrity-audit file, schema, policy, global ledger, and mismatch count to its
sampling/official artifacts.

## Question and arms

The experiment measures the behavior of the current default-on diagnostic
`protein_ligand_steric_barrier` profile at

`eta = {0.0, 0.5, 1.0, 1.5, 2.0}`.

The corresponding artifact tags are `eta0000`, `eta0500`, `eta1000`,
`eta1500`, and `eta2000`. The same-run `eta=0` arm is the only paired
baseline. Every arm, including `eta=0.5`, is generated afresh. Earlier
`eta=0.5` artifacts used another guidance parameter identity and are historical
context only.

## Frozen guidance identity

- Combined GuidanceEnergy SHA-256:
  `6621d17c41aeb6c9685075209155850018c5eb9882489ae209c7c30b8070e89f`
- Physical EFF-FF version/formula: `2.1.0` / `effff-diagnostic-2.1`
- Physical EFF-FF SHA-256:
  `079940d8b61ed777ea00c3ac9abb101996a618df461deacee3d5ab3189f5d674`
- Interaction parameter SHA-256:
  `b772d431e21bcaecf1648ce4e539b1448e4f9df8ff5bd0db0fdf6407fcd23f16`
- `geometry_only` receptor-policy SHA-256:
  `92adb215ccb77aae51ea14d8a2cc33319f70feb8548e9f0b07f500a5bcee1c20`

The runtime remains repository-native Torch code. Vina guidance is zero and
excluded. No external force-field, docking, minimization, or simulation engine
participates in ODE solving.

## Frozen inputs and sampling

- Cohorts: all 85 Astex Diverse and all 308 PoseBusters v2 complexes in the
  content-addressed full-cohort manifest.
- Docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`, SHA-256
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`.
- Confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`.
- Budget: `N100/S10`, or 1,000 learned-model pose-steps per complex and arm.
- Prior pool: 100 poses; `sigma=0.5`; base seed 42 plus the one-based position
  in the frozen, sorted dataset-ID inventory. Every CSV row must contain the
  exact resulting seed and canonical `prior_pool_size=100`.
- Pairing: each complex must have exactly the same `sampling_seed` in all five
  arms. Prior-pool size and construction contract remain exact. Cross-eta
  `prior_pool_sha256` differences are retained as structured diagnostics with
  counts, IDs, hash sets, and per-arm Slurm partition/GPU context; they do not
  fail audit V2.
- ODE schedule: late, power 3.0; `normalized_drift`; guidance starts at
  `t=0.5` with ramp power 1.0.
- Pocket cutoff 10 Angstrom; center jitter 0; protein shell 18 Angstrom;
  `geometry_only` receptor policy; no refinement.
- Force cap 20; translation and angular caps 5; maximum estimated atom
  displacement 0.25 Angstrom; maximum backtracks 8.
- Selector profile: `confidence_cluster_free`. Pure minimum predicted RMSD
  (`confidence`) is primary. The frozen cluster-free `confidence_filter` is
  diagnostic. First and oracle poses are evaluation-only and cannot select eta.

## Fail-closed execution chain

The output root is
`outputs/benchmarks/guidance_steric_high_eta_confidence_runs/<run-id>`.
Every stage uses an isolated task reservation and an `afterok` dependency:

1. All-arm GPU smoke: 10 tasks = 2 datasets x 5 eta values, one frozen complex
   per dataset.
2. Parent-free integrity audit of the smoke inventory, paired seeds, strict
   prior construction contract, and prior-hash provenance.
3. Official PoseBusters 0.6.5/redock smoke for both confidence selectors at
   `eta=2.0`.
4. A separate `eta=2.0` CUDA size stress on `8f4j_pho`, followed by a strict
   stress audit. This ID is frozen because it had the largest product of
   standard receptor-source heavy atoms and ligand input heavy atoms in the
   prior full-cohort resource inventory; RMSD and validity outcomes were not
   used.
5. Full GPU sampling: 80 tasks = 2 datasets x 5 eta values x 8 shards.
6. Full parent-free integrity audit.
7. Official selected-pose PoseBusters: 160 tasks = 2 selectors x 2 datasets x
   5 eta values x 8 shards.
8. Strict aggregate report with paired comparisons against same-run `eta=0`.

Required full inventory is 1,965 complex-arm rows and 196,500 candidate poses.
The official selected-pose inventory is 3,930 evaluations. Any missing,
duplicate, non-finite, failed, or hash-mismatched frozen input or persisted
artifact fails its stage; partial survivor denominators are forbidden. The one
exception is cross-eta equality of valid per-row prior-pool digests, which is
record-only under the V2 policy above.

## Numerical and resource gates

- Every nonzero arm must have eight finite active-step traces per complex.
- All non-finite counters must be zero. Zero raw-direction and zero
  reference-velocity counters remain explicit diagnostics rather than assumed
  failures.
- Reported post-cap maxima must be no larger than 5.000001 for translation and
  angular velocity and 0.250001 Angstrom for estimated atom displacement.
- Every 100-candidate confidence ledger must be finite, bounded, and reproduce
  both saved selector indices.
- Every row must bind the current `all_poses` SDF path and SHA-256 and the file
  must contain exactly 100 records. Those SDF coordinates are persisted at
  four decimal Angstrom precision and therefore cannot reconstruct the
  producer's original float32 `candidate_ensemble_sha256` digest.
- Each CUDA task must run on `6000ada` or `heavy` with exactly one visible
  allowed GPU and at least 48,000 MiB total visible memory. The allowed runtime
  names are RTX 6000 Ada, H100, and RTX PRO 6000-class devices. Allocated and
  reserved peaks must remain below the existing fixed 48 GiB cap and, for this
  profile, allocated peak must remain below 90% of that cap. OOM or a failed
  smoke gate blocks the full array.
- The source/runtime implementation identity, execution manifest, frozen input
  manifest, parameter identities, current protein/reference hashes, and saved
  selected-pose hashes must all match exactly.
- Jobs must run from the recorded read-only execution capsule. Queue-time
  edits to the live worktree must not alter imported EFF-Dock code or the
  submitted Slurm stage.

Nominal eta may differ strongly from effective guidance dose because all caps
remain fixed. The report therefore retains per-step model/applied/total
atom-speed RMS, applied/model ratios, model-guide cosine and parallel ratio,
cap-scale quantiles, individual/any/multiple-cap counts, path proxies,
post-cap maxima, zero-direction counters, and CUDA peaks. Caps are not changed
after outcomes are opened.

## Outcomes and claim boundary

For each dataset, eta, and frozen selector, report:

- selected-pose symmetry-aware RMSD `<2 Angstrom` count and percentage;
- selected-pose median RMSD;
- official pass-all PoseBusters validity over 27 non-RMSD redock checks;
- joint RMSD `<2 Angstrom` and PoseBusters-valid count and percentage;
- paired changes and 95% paired complex-ID bootstrap intervals versus `eta=0`.

Oracle RMSD remains an evaluation ceiling. No winner is selected. These are
descriptive reference-pocket redocking results for this exact profile, not an
independent generalization claim, affinity/free-energy estimate, molecular
dynamics result, or production-admission study. A future interaction-energy
plus confidence selector must first be frozen on internal PLINDER data; this
external run does not tune such a combination.
