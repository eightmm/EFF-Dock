# Full-cohort unified-guidance fixed-budget protocol

Protocol ID: `EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2`

Frozen: 2026-07-31, before any V2 full-cohort sampling or official
PoseBusters outcomes are opened.

## Question and claim boundary

Can the in-repository `geometry_only` receptor policy remove the chemistry
admission bottleneck and run the frozen unified-guidance experiment on every
discovered Astex Diverse and PoseBusters v2 complex without numerical or
sampling failures?

The primary claim is an implementation and numerical-safety claim:

- exact audit and sampling coverage is `85/85` Astex plus `308/308`
  PoseBusters (`393/393` total);
- every complex has a finite crystal energy and coordinate gradient in the
  preflight audit;
- every one of the 12 paired sampling cells has zero failed complexes and zero
  non-finite guidance trials; and
- maximum accepted atom displacement is at most `0.25 Å`.

The Astex and PoseBusters outcomes from V1 have already been opened. V2 is
therefore a coverage-policy completion and paired descriptive sensitivity
rerun, not a fresh external confirmation. Oracle, fast-validity, and official
PoseBusters effects are secondary descriptive results. They cannot tune or
admit an energy term, coefficient, scale, schedule, force cap, trust region,
fallback, checkpoint, or production default.

The retained checkpoint's legacy train/validation split was not frozen with an exact
entity-level exclusion against these benchmark structures. The frozen input
inventory finds ligand-identity matches to that compatibility split for 24/85 Astex and
123/308 PoseBusters IDs; Astex `1meh` also has the same entry and ligand in the
training split. These are outcome-independent disclosure slices, not exclusion
or tuning rules. Consequently no Astex or PoseBusters number in this protocol
is an independent external-generalization estimate, even before accounting for
the opened V1 outcomes. Results are reference-pocket redocking characterization
and must report both matched and unmatched disclosure slices.

## Falsifiable predictions and stopping rules

Before V2 sampling:

- the full-cohort audit will discover exactly 85 Astex and 308 PoseBusters IDs,
  audit each ID exactly once, and finish with no failure IDs;
- the conservative fallback will make every crystal energy and gradient
  finite without adding an attractive interaction to unsupported chemistry;
- the full sampling array will complete with no failed ID, no non-finite base
  pose or trial, and no accepted displacement above `0.25 Å`.

Any missing, duplicate, outside, failed, or non-finite ID stops aggregation.
The strict report never emits a survivor-only or complete-case paired effect.
If the audit is not `393/393`, sampling does not start. If a representative GPU
smoke fails, the full array does not start. Numerical fixes may be tested on
the smoke/audit inputs, but any change to the scientific energy or correction
settings creates a new protocol ID.

The V1 descriptive effect guard is retained only for comparability: report
whether guidance changes `any(fast-valid and RMSD <2 Å)` by at least `+2`
percentage points and whether oracle RMSD `<2 Å` falls by more than `2`
percentage points. Passing these thresholds would not be independent external
validation because the datasets were previously opened.

## Frozen model, data, and inference boundary

- Docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`
  (`sha256:6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`).
- Configuration: `configs/train.yaml`
  (`sha256:39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`).
- Confidence is disabled. This study measures sampling oracle coverage, not
  pose selection.
- Datasets are all complexes discovered by the frozen benchmark loader:
  Astex Diverse 85 and PoseBusters v2 308. The V1 chemistry-eligible ID list is
  neither read nor used as a whitelist.
- Exact ligand inputs are frozen in
  `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`
  (`sha256:99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668`).
  It content-addresses every raw and canonical heavy-atom SMILES, the local
  Astex/PoseBusters mapping manifests, PoseBusters membership, RDKit version,
  hydrogen policy, and the legacy-split overlap inventory. The launcher and
  strict reporter fail if this tracked file changes under the same protocol.
- The local source mapping hashes are Astex
  `ec53476f64fda163058541ede965278a36a18f47d2fb943d6cc1d0bf64953e67`,
  PoseBusters
  `f17b424fdf779ff5284f6be3e46df57d03244ce9e21575d2b59b7f65517b91c9`,
  and PoseBusters membership
  `a69a7b6b9a5a52531933078ef983e6c069e3a987a1d7a733bd7d72cbe1793de6`.
  They are provenance only after freezing; runtime discovery uses the tracked
  manifest.
- Astex `1meh` is the one declared source correction: the ignored mapping
  supplied IMP, while the deposited complex contains MOA. Its frozen SMILES is
  derived from the reference SDF graph only
  (`sha256:7a35262c9a5d88b47294107423cd765c1ad35c4c685fe235a9a8659599b6b553`);
  reference coordinates are forbidden in conformer generation and sampling.
- FULL-V2 alone applies seeded generic ligand loading followed by RDKit
  `RemoveAllHs`, because the frozen model graph is heavy-atom only. Legacy
  evaluation without this manifest keeps the original generic loader semantics.
- Immutable evaluation unit is one benchmark complex ID. Exact audit ID hashes
  use `EFFDOCK_SORTED_COMPLEX_IDS_V1`: SHA-256 over the ASCII prefix
  `EFFDOCK_SORTED_COMPLEX_IDS_V1\0`, followed by each lexicographically sorted
  UTF-8 ID and a NUL byte. The V1 newline-delimited eligibility hash is retained
  only as labeled historical provenance.
- Pocket centers are the frozen reference-pocket files. During sampling, the
  reference ligand coordinates are used only for RMSD after pose generation;
  they never enter sampling guidance, fallbacks, or pose selection. Separately,
  the declared outcome-blind CPU audit requires a complete, element- and
  connectivity-preserving heavy-atom bijection and evaluates energy and
  gradient on that reference-mapped pose solely as a numerical preflight.
  Atom counts and both index sets must be full permutations, corresponding
  atomic numbers and undirected connectivity must match, and partial MCS or a
  constitutional-connectivity change fails closed. A full bijection with only
  tautomer, formal-charge, aromatic, or bond-order representation differences
  is retained in the separately reported
  `same_connectivity_representation_mismatch` slice rather than changing the
  frozen sampling input to match the reference outcome representation.
- Ligand prior sigma is `0.5 Å`; pocket crop is `10 Å`; center jitter is
  `0 Å`; seed is 42 plus the global sorted-complex offset.
- A deterministic 100-pose translation/SO(3) prior pool is generated once per
  complex. Every arm and budget cell uses the same seed and exact prior-pool
  hash. N50 and N40 are prefixes of N100.

## Full-cohort audit manifest

Sampling provenance must point to an outcome-blind full-cohort audit manifest.
The manifest is generated from receptor/ligand chemistry, frozen centers, and
the 18 Å guidance shell before sampling outcomes. It records:

- protocol and receptor-policy identities and parameter hashes;
- discovered, audited, successful, and failed exact ID lists and hashes;
- one successful per-complex record with receptor provenance;
- the exact benchmark-input identity plus per-complex protein, reference-SDF,
  ligand-input, and physical-system hashes used again by the sampling gate;
- finite crystal energy/gradient preflight status;
- fixed-obstacle and metal-fallback counts and structured fallback reasons;
- mutually exclusive chemistry slices that partition the full dataset and are
  used for descriptive paired reporting.
- mutually exclusive ligand-representation slices, plus matched/unmatched
  legacy-split ligand-identity and exact-entry disclosure slices with counts,
  ID hashes, and paired descriptive effects.

Admission requires `success == audited == discovered`, `failed == 0`, exact
85/308 membership, empty failure IDs/codes, and all declared numerical
preflight checks finite. A combined
`docs/GUIDANCE_BUDGET1000_FULL_COHORT.json` may contain both dataset audits;
the launcher also accepts a dataset-specific audit path through
`EFFDOCK_COHORT_MANIFEST`.

After both dataset audits pass, create the self-contained combined manifest:

```bash
.venv/bin/python -m effdock.workflows.guidance_coverage_audit \
  --merge-audit docs/GUIDANCE_BUDGET1000_FULL_COHORT_ASTEX.json \
  --merge-audit docs/GUIDANCE_BUDGET1000_FULL_COHORT_POSEBUSTERS.json \
  --require-complete-success \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2 \
  --output docs/GUIDANCE_BUDGET1000_FULL_COHORT.json
```

Merge rejects differing implementation, parameter-set, receptor-policy, or
receptor-policy-identity records. Each dataset audit independently binds the
frozen benchmark-input identity. The sampling launcher revalidates these
identities and the tracked input-manifest SHA before touching a GPU.

## Receptor-policy intervention

The only scientific intervention relative to V1 is
`--unified-guidance-receptor-policy geometry_only`.

- Standard protein atoms keep the existing typed physical and interaction
  terms.
- Strict Zn/Mg directional attraction is used only when the existing
  coordination admission gates resolve the site.
- Any unresolved, clustered, embedded, identity-mismatched, or otherwise
  unsupported metal site is a bounded all-ligand repulsion-only site with a
  structured reason. It never falls back to Zn attraction.
- Supported active nonprotein atoms are fixed EFF-FF-v2 geometry obstacles
  with repulsion only.
- Atoms without an EFF-FF-v2 element parameter are fixed generic bounded
  steric obstacles with a separately versioned identity.
- No fallback supplies hydrogen-bond, charge, pi, hydrophobic, metal
  attraction, affinity, or free-energy semantics.

All coordinate-dependent energy and gradient operations remain in-repository
Torch code. No external force-field, docking, minimization, or simulation
engine is used at runtime. The policy is a conservative geometry prior, not a
complete cofactor force field.

## Arms and fixed learned-model budget

| Cell | ODE steps | Poses | Learned model pose-steps |
|---|---:|---:|---:|
| N100/S10 | 10 | 100 | 1,000 |
| N50/S20 | 20 | 50 | 1,000 |
| N40/S25 | 25 | 40 | 1,000 |

Each cell has paired `unguided` (`scale=0`) and `guided` (`scale=0.1`)
arms. The learned-model budget excludes guidance energy/gradient and
backtracking work, which is recorded separately.

All V1 corrector settings remain frozen:

- operator-split correction after the learned ODE proposal;
- late schedule, power `3`; guidance starts at `t=0.5` with linear ramp;
- physical soft core `1.5 Å -> 0.75 Å`;
- atom-force cap `20`; fragment translation/angular caps `5/5`;
- accepted atom-displacement limit `0.25 Å`;
- backtracking factor `0.5`, at most `8` reductions;
- receptor guidance shell `18 Å`;
- finite, energy-descent (`atol=rtol=1e-6`), and trust-region acceptance.

## Metrics

Primary engineering metrics:

- exact audit and sampling ID coverage;
- failed and non-finite complex/trial counts;
- accepted/rejected corrections, backtracks, energy evaluations, and maximum
  accepted atom displacement;
- fixed-obstacle, generic-obstacle, strict-metal, metal-fallback, and fallback
  reason counts from the audit manifest;
- wall time and CUDA peak allocated/reserved memory.

Secondary descriptive pose metrics:

- minimum symmetry-aware heavy-atom RMSD and oracle `<1/<2/<3/<5 Å`;
- fast-valid candidate count, RMSD-oracle fast validity,
  `any(fast-valid and RMSD <2 Å)`, and fast-valid oracle RMSD;
- paired guided-minus-unguided effects overall and on every non-empty,
  predeclared chemistry slice;
- paired fixed-budget comparisons using the exact common IDs;
- official PoseBusters 0.6.5 `redock` non-RMSD pass-all validity of the saved
  RMSD-oracle pose, including module-level checks.

Paired percentile-bootstrap 95% intervals resample complex IDs with seed
`20260731` and 10,000 resamples. Chemistry slices are mutually exclusive,
partition the full cohort, and are always reported with exact counts and ID
hashes; the full cohort remains the primary denominator.

## Execution and reporting gates

1. Run focused CPU tests for receptor-policy construction, finite
   energy/gradient, operator order, descent, and trust-region rejection.
2. Generate and validate the full 85/308 audit manifest. Do not sample if any
   ID fails.
3. Run `EFFDOCK_ONLY_ID` GPU smokes covering at least a standard receptor, a
   supported cofactor obstacle, a generic obstacle, a strict metal site, and a
   repulsion-only metal fallback. Guided/unguided prior hashes must match. A
   smoke submission must override Slurm with exactly one array task (for
   example `sbatch --array=0 ...` for an Astex N100/S10 smoke or
   `--array=24` for PoseBusters); the launcher rejects a 48-task smoke to avoid
   repeated writes and stores smoke output outside the strict-report `raw/`
   directory.
4. Submit the 48-task 6000ada array: 2 datasets x 3 budgets x 8 shards, with
   paired arms run serially inside each task.
5. Strict aggregation requires all 96 sampling shard summaries, exact audit
   hashes and IDs, consistent checkpoint/config/pocket/policy/parameter hashes,
   zero failures/non-finites, and the trust-region bound.
6. Run the existing official PoseBusters pipeline on each saved
   `poses/<run-name>/<dataset>/oracle/` directory and aggregate the same exact
   IDs. Official-check failures also invalidate paired reporting.

For each of the 12 run names, the existing official pipeline is compatible
with the V2 layout through explicit environment variables. Submit one 8-shard
CPU array per run (replace the example run name and dataset together):

```bash
run_name=effdock-guidance-budget1000-full-v2-astex-n100-s10-guided
EFFDOCK_NUM_SHARDS=8 \
EFFDOCK_PB_RUN_NAME="$run_name" \
EFFDOCK_PB_INPUT_DIR=outputs/benchmarks/guidance_budget1000_full_v2/raw \
EFFDOCK_PB_POSE_DIR="outputs/benchmarks/guidance_budget1000_full_v2/raw/poses/$run_name/astex/oracle" \
EFFDOCK_PB_OUTPUT_DIR="outputs/benchmarks/guidance_budget1000_full_v2/posebusters_official/$run_name" \
EFFDOCK_PB_SELECTOR=oracle \
sbatch --array=0-7 scripts/slurm/posebusters_array.sbatch
```

The official report requires exactly those 12 run directories, eight shard
summaries per directory, no failure rows, and exact audit ID coverage.

Sampling launcher:
`scripts/slurm/guidance_budget1000_full_array.sbatch`.

Sampling strict report:
`python -m effdock.workflows.guidance_budget_full_report`.

Official-validity strict report:
`python -m effdock.workflows.guidance_budget_full_posebusters_report`.
