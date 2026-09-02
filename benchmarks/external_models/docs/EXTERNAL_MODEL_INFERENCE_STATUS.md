# External model inference status

> Historical execution ledger. The audited acceptance state and corrected
> metrics are in [`EXTERNAL_MODEL_AUDIT_20260830.md`](EXTERNAL_MODEL_AUDIT_20260830.md).
> In particular, the Meeko receptor compatibility Vina arm below is rejected
> and must not be used as the Vina benchmark.
> Current reporting is also supplied-pocket-only; blind/full-receptor,
> pocket-prediction, and co-folding jobs listed here are archival diagnostics
> and are excluded from all result tables and plots.

Initial snapshot: 2026-08-28 15:05 KST
Latest recovery snapshot: 2026-08-28 17:18 KST

This is an execution ledger, not a benchmark result table. RMSD and official
PoseBusters metrics are computed only after inference coverage is frozen.

## Failed-arm recovery (authoritative latest state)

The earlier job chains listed below are retained as historical provenance. The
following focused retries and clean replacement chains supersede failed or
permanently blocked jobs; no partial result is silently admitted.

| Arm | Pre-recovery coverage / failure | Corrected jobs | State at snapshot |
|---|---|---|---|
| DiffBindFR + MDN | Astex complete; PB raw 308/308 but MDN 282/308 because shard 5 failed | PB shard-5 retry `60387` | accepted on `6000ada`, waiting for resources |
| SurfDock | Astex 83/85; PB 296/308 | focused Astex `60388`, PB `60389` | accepted on `6000ada`, waiting by priority |
| PoseBench DiffDock | Astex 77/85; PB 280/308 at released S20/actual-19, N5 protocol | focused Astex `60390`, PB `60391` | accepted on `6000ada`, waiting by priority |
| PoseBench DynamicBind | old 3-day jobs were invalid under their QoS and never ran | S2/N1 gate `60392`; dependent Astex `60393`, PB `60394`; coverage `60395`-`60396` | gate running; ESM2-650M acquisition completed and receptor/graph preprocessing active |
| PoseBench Vina | old 3-day jobs were invalid under their QoS; first recovery gate exposed Meeko/RDKit and bundled ADFR ABI incompatibilities | validated gate `60420`; dependent Astex `60421`, PB `60422`; coverage `60423`-`60424` | gate completed 1/1 with one SDF pose; both full arrays running on `cpu_only` |

The Vina gate record is
`outputs/external_models/runs/posebench_vina/recovery_20260828_v3/gate/coverage.json`.
It records the actual emitted pose count and receptor-writer provenance.

Recovery changes are deliberately narrow:

- DiffBindFR repairs only malformed multi-character chain IDs in exported
  runtime PDBs before MDN scoring; original raw structures remain unchanged.
- SurfDock restores missing double-bond stereo from supplied 3D coordinates,
  preserves a valid Bio.PDB pocket when an optional RDKit rewrite returns
  `None`, and falls back to the already-local 8 A surface only when the native
  3 A face crop is empty.
- DynamicBind and Vina now request QoS values that permit their declared wall
  times. CPU-only Vina work remains exclusively on `cpu_only` and does not
  request an explicit CPU count.
- PoseBench pins Meeko `0.6.0a3` while the environment contains RDKit `2025.9.6`.
  A runtime shim supplies only the removed `rdkit.six.StringIO` symbol.
- PoseBench's bundled Python-2 ADFR has an incompatible NumPy Unicode ABI on
  this cluster. The validated Vina arm therefore keeps PoseBench site,
  protonation pre-pass, ligand preparation, Vina engine, exhaustiveness, and
  output conversion, but writes the rigid receptor PDBQT with the already
  pinned Meeko installation from PoseBench's original temporary PDB. Coverage
  labels this explicitly as `meeko_0.6.0a3_original_pdb_compat`; it must not be
  described as byte-identical to the legacy ADFR writer.

## Frozen cohorts

| Dataset | Denominator | Input policy |
|---|---:|---|
| Astex Diverse | 85 | Full official cohort; failures remain in the denominator |
| PoseBusters Benchmark v2 | 308 | Full official cohort; failures remain in the denominator |

Exact manifests, shard membership, and preparation provenance are stored below
`outputs/external_models/inputs/` as ignored runtime data. Unless an arm says
otherwise, pocket-supplied models dock to PoseBench's released holo-aligned
predicted receptor and use the crystal ligand only to define the site. Blind
models receive no crystal site.

## Executed and active arms

| Arm | Native setting | Gate | Full inference | Runtime output root |
|---|---|---|---|---|
| PoseBench DiffDock | S20, actual 19, N5, confidence | job `59919` passed | PB job `59920`, Astex job `59921`, both running on `6000ada` | `outputs/external_models/runs/posebench_diffdock/full/` |
| DiffDock-Pocket | S30, N40, confidence | job `59942` passed, 1/1 and 5/5 | gate `59944`, then Astex `59946` and PB `59947` | `outputs/external_models/runs/diffdock_pocket/full/` |
| SigmaDock | S25, N40 independent seeds, Vinardo | environment `60172`; clean 1-target smoke `60188`; predicted-receptor parser smoke `60204` | final `test`/A5000/`veryshort` gates Astex `60237`, PB `60238`; arrays Astex `60240`, PB `60239`; coverage `60241`, `60242` | `outputs/external_models/runs/sigmadock/s25_n40_predicted_receptor_20260828/full_compat_v2/` |
| RLDiff native | S20, N40, confidence | job `59943` passed, 1/1 and 5/5 | gate `59945`, then Astex `59948` and PB `59949` | `outputs/external_models/runs/rldiff/full/` |
| RLDiff RL++ | S20, N40, native RL++ sampling; smina/GNINA CPU reranking | CPU gate `59904` queued | raw jobs `59916`-`59918` depend on the gate | `outputs/external_models/runs/rldiff/` |
| SurfDock | S20, N40, posepredict MDN, no force optimization | job `59964` depends on environment `59824` | Astex `59965`, PB `59966` | `outputs/external_models/runs/surfdock/` |
| DiffBindFR + MDN | declared S22, actual S20, N40, no smina correction | two-target job `59967` depends on environment `59883` | Astex `59968`, PB `59969` | `outputs/external_models/runs/diffbindfr/` |
| Interformer | 64 Monte Carlo repeats x 2000 steps, top 20 learned-energy poses | job `59970` depends on environment `59858` | Astex `59971`, PB `59972` | `outputs/external_models/runs/interformer/` |
| PoseBench FABind | deterministic one-pose post-optimized output | job `59952` depends on environment `59838` | Astex `59954`, PB `59955` | `outputs/external_models/runs/posebench_fabind/` |
| PoseBench DynamicBind | S20, N40, paper weights, no native relax | job `59953` depends on environment `59839` | Astex `59956`, PB `59957` | `outputs/external_models/runs/posebench_dynamicbind/` |
| PoseBench Vina | exhaustiveness 32, requested N40, supplied site | job `59958` depends on environment `59840` | Astex `59959`, PB `59960`, all on `cpu_only` | `outputs/external_models/runs/posebench_vina/` |

At this snapshot, completed clean DiffDock shards cover 49/85 Astex targets and
43/308 PoseBusters targets with the requested five poses. These are partial
coverage counts, not success rates. The arrays continue over the remaining
shards.

## Protocol details

- DiffDock-Pocket uses the predicted receptor and crystal ligand SDF. Its older
  protein-only smoke outputs are diagnostic and excluded.
- SigmaDock uses the official `v0.1.0-beta` checkpoint, 25 native Euler
  diffusion steps, one independent pose per seed, and 40 seeds. It receives the
  same predicted receptor and crystal-defined pocket as the other supplied-site
  arms. Native Vinardo scores are retained for Top-1 selection; PoseBusters is
  evaluated only after selection, not used to rank poses.
- RLDiff uses its exact released pocket selector before ESM embedding. This
  retains selected cofactors and avoids the ESM 1022-residue failure seen on
  raw PoseBusters target `7D6O_MTE`.
- SurfDock uses the official 8 A surface, ESM2-650M layer-33 embeddings, native
  diffusion, and native MDN ranker. Only the two PoseBusters chains longer than
  ESM's 1022-residue limit use local windows covering all supplied-pocket
  residues.
- DiffBindFR's current arm is the pure learned `DiffBindFR + MDN` arm. Its
  optional smina correction is not mixed into this run.
- Interformer uses an ETKDGv3/UFF start conformer. Its appended input conformer
  is retained on disk for provenance but explicitly excluded from pose counts,
  Oracle, and Top-1 evaluation. PyVina's native per-thread generators are seeded
  from the run seed over the scheduler-provided CPU affinity set.
- FABind is deterministic and therefore has no fabricated Oracle@40.
- Vina's actual emitted SDF record count will be frozen from the smoke output;
  the requested `num_modes=40` is not assumed to mean 40 written records.

## Coverage semantics

- Smoke jobs fail unless every requested pose is present.
- Full jobs continue past target-specific failures and record per-target pose
  counts.
- A missing target remains an explicit failure in the `N=85` or `N=308`
  denominator.
- `aggregate_inference_coverage.py` checks missing, duplicate, unexpected, and
  incomplete targets across shards.
- Every full arm records source commit, checkpoint/engine choice, seed, receptor
  policy, site policy, pose count, and model-specific exceptions.

## Corrections discovered by real inference

- PoseBench DiffDock exceeded the 11 GB test GPU on large receptors; clean full
  shards run on 48 GB `6000ada` and retain target-specific failures.
- DiffDock-Pocket's pinned CLI checks an undeclared filtering alias and eagerly
  imports an unused OpenFF relaxation stack. The compatibility layer supplies
  only the alias and unused import boundary; model and checkpoint paths remain
  unchanged.
- Raw RLDiff reached ESM's position limit on two 1285+ residue chains. The final
  arm applies the released pocket selector before ESM rather than truncating an
  arbitrary chain prefix.
- SurfDock hardcodes a developer-home binary path and references an ESM script
  directory absent from the pinned checkout. The runtime layer injects paths to
  its exact bundled binaries and calls the pinned official fair-esm API.
- Interformer's native sampler accepts only four-character PDB IDs and otherwise
  deletes the shared multi-target output directory on each target. Reversible
  PDB-prefix aliases and `continue_dock_index=0` preserve native sampling while
  preventing that deletion.
- SigmaDock's checkpoint was found in the upstream `v0.1.0-beta` prerelease.
  The downloaded 242,097,065-byte asset is accepted only when its SHA-256 is
  `db15427ca349e6f1e5f894bff841112c7360384886aa472667d8011307cad382`;
  GNINA 1.3.2 is likewise checksum-pinned. Co-folding models remain excluded
  from the primary comparison.
- SigmaDock environment job `60156` correctly failed its import gate before
  inference because transitive `torchvision` came from PyPI while Torch came
  from the cu126 index (`torchvision::nms` was unavailable). No pose job ran.
  The replacement environment pins both Torch and Torchvision to the same
  official cu126 index; jobs `60157`-`60161` were cancelled and replaced by the
  chain listed above.
- SigmaDock smoke `60173` generated its pose but failed the GNINA stage because
  the standalone binary could not locate model-local cuDNN. The runner now
  exports that exact library path; replacement smoke `60188` passed pose and
  Vinardo coverage. No result from `60173` is admitted.
- The first full SigmaDock Astex seed exposed an RDKit proximity-bond artifact
  in a holo-aligned predicted receptor. The compatibility fallback runs only
  after normal sanitization fails and removes nonstandard inter-residue bonds
  while retaining intra-residue, sequential peptide, and disulfide bonds.
  Target-specific smoke `60204` and the corrected full Astex seed both covered
  85/85 targets.
- The first PoseBusters seed `60192` was rejected at 298/308 coverage. Four
  predicted pockets contained the same nonstandard residue-link artifact, and
  one polyene lost double-bond stereo in SigmaDock's defensive SDF reader,
  making ETKDG pathologically slow. The final wrapper restores stereochemistry
  from the SDF's existing 3D coordinates before generating an independent
  conformer. Failed and mixed-compatibility outputs remain preserved but are
  excluded; final jobs use only `full_compat_v2` and fail closed at 85/85 or
  308/308 for every seed.
