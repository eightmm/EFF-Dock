# Repository structure

```text
EFF-Dock/
├── src/effdock/                 active Python package
│   ├── models/                  SE(3) docking network
│   ├── confidence/              trained confidence model, dataset, loss, runtime, selector
│   ├── inference/               sampler and single-complex docking
│   ├── training/                docking trainer/checkpoint resume
│   ├── evaluation/              RMSD, Vina+DG, and validity primitives
│   ├── guidance/                flat physical/interaction guidance boundary
│   ├── workflows/               CLI workflows and benchmark aggregation
│   ├── preprocess/              protein/ligand/fragment graph construction
│   └── data/                    active dataset interfaces
├── configs/                     docking and confidence training configs
├── benchmarks/                  benchmark adapters, jobs, and figure scripts
│   ├── external_models/         external-model prepare/run/evaluate adapters
│   │   └── slurm/               install, inference, selection, evaluation jobs
│   ├── figures/                 benchmark plotting/rendering code
│   └── results/                 compact public metrics and provenance
├── scripts/slurm/               training jobs plus benchmark compatibility links
├── tests/                       CPU/unit and focused smoke tests
├── weights/                     released docking/confidence pair and model cards
├── docs/                        paper index, contracts, protocols, and results
├── data/                        preserved local datasets; ignored by Git
├── outputs/                     preserved runs/benchmarks; ignored by Git
└── archive/flowfrag_legacy/     non-destructive historical code and experiments
```

The active boundary is `src/effdock`, `benchmarks`, `configs`, `scripts`,
`tests`, and the two public artifacts under `weights`. `archive` is evidence
and recovery material, not imported by the active package. `data` and
`outputs` are never deleted during reorganization and remain ignored so
large/private artifacts are not accidentally committed.

Benchmark-specific adapters and launchers have one canonical home under
`benchmarks/`. Reusable metric/runtime code remains under `src/effdock`.
Historical `scripts/external_models`, `scripts/figures`, external-model Slurm
paths, and `configs/external_models.json` are symlink compatibility aliases so
archived commands and already-submitted jobs keep working without duplicate
implementations.

The GitHub release includes Python code, benchmark adapters, pinned environment manifests,
protocol/config/seed provenance, exact coverage accounting, compact per-seed
and aggregate metrics, separated runtime summaries, and final figures. Installed
environments, upstream checkouts, raw data/logs/pose banks, private machine
paths, and artifacts without redistribution permission remain ignored.

`effdock.inference.DockingOptions` and `effdock.inference.dock` are the primary
public inference interface. The `eff-dock` command and Slurm launchers are thin
workflow wrappers retained for exact experiment reproduction. The repository
is run from its root so released config and weight paths remain explicit.

The documentation entry point is `docs/README.md`; paper-facing claims are
mapped in `docs/PAPER_EVIDENCE.md`. Detailed protocol/result files remain at
stable paths because they are immutable scientific records. The complete
machine run ledger and historical checkpoints remain in the ignored local
archive rather than the public documentation tree.

`guidance/physical.py` owns generic geometry and nonbonded energies;
`guidance/interaction.py` owns typed chemical motifs. By explicit user request,
all seven implemented terms—hydrophobic contact, idealized
missing-valence-cone heavy-atom hydrogen bond, screened formal-charge groups,
pi stacking, cation-pi, ligand-to-protein halogen bond, and
profile-dispatched metal coordination—are enabled in the default diagnostic
profile. This default is not production validation or sampler admission.
Standalone monatomic receptor ions are detected automatically: strict
one-vacancy Zn/CN4 and Mg/CN6 sites may attract a typed ligand donor, whereas
Ca/Mn/Fe/Co/Ni/Cu are repulsion-only with an explicit trace reason.
Independent sites are supported; clusters, cofactors, and identity mismatches
fail closed under the default receptor policy. The separately named
`geometry_only` full-cohort policy converts unsupported active receptor atoms
to fixed, bounded repulsion-only obstacles and records the fallback; it never
adds an attractive interaction.
`polar_unsatisfied_proxy` is trace-only and never contributes energy or force.
`guidance/runtime.py` combines the selected physical and interaction terms
into one `GuidanceEnergy`; production ODE coupling is not admitted. Vina
remains in legacy evaluation code but is excluded from this guidance path.
The frozen protocol and completed no-admission result are recorded in
`INTERACTION_PRIOR_PROBE_PROTOCOL.md` and
`INTERACTION_PRIOR_PROBE_RESULTS.md`; their V2 three-term baseline remains
historical and unchanged.
The full `85/85` Astex plus `308/308` PoseBusters fixed-budget coverage run is
recorded in `GUIDANCE_BUDGET1000_FULL_PROTOCOL.md` and
`GUIDANCE_BUDGET1000_FULL_RESULTS.md`; it passed coverage/numerical gates but
did not admit guidance as a production default.

The main runtime flow is:

```text
protein + ligand + explicit pocket
  -> preprocess heterogeneous graph
  -> EFF-Dock samples N poses
  -> (training data) label poses + cache t=1 ligand features
  -> confidence model can be trained/resumed on preserved shards
  -> learned confidence scores the same poses
  -> declared frozen confidence/filter policy orders poses
  -> SDF + results.pt + provenance
```
