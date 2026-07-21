# Repository structure

```text
EFF-Dock/
├── src/effdock/                 active Python package
│   ├── models/                  SE(3) docking network
│   ├── confidence/              trained confidence model, dataset, loss, runtime, selector
│   ├── inference/               sampler and single-complex docking
│   ├── training/                docking trainer/checkpoint resume
│   ├── evaluation/              RMSD, Vina+DG, and validity primitives
│   ├── workflows/               CLI workflows and benchmark aggregation
│   ├── preprocess/              protein/ligand/fragment graph construction
│   └── data/                    active dataset interfaces
├── configs/                     docking and confidence training configs
├── scripts/slurm/               benchmark, confidence training, validity jobs
├── tests/                       CPU/unit and focused smoke tests
├── weights/                     named, hashed retained model artifacts
├── docs/                        contracts, protocols, results, experiment log
├── data/                        preserved local datasets; ignored by Git
├── outputs/                     preserved runs/benchmarks; ignored by Git
└── archive/flowfrag_legacy/     non-destructive historical code and experiments
```

The active boundary is `src/effdock`, `configs`, `scripts`, `tests`, and
`weights`. `archive` is evidence and recovery material, not imported by the
active package. `data` and `outputs` are never deleted during reorganization and
remain ignored so large/private artifacts are not accidentally committed.

The main runtime flow is:

```text
protein + ligand + explicit pocket
  -> preprocess heterogeneous graph
  -> EFF-Dock samples N poses (optional late-time Torch Vina+DG guidance)
  -> (training data) label poses + cache t=1 ligand features
  -> confidence model can be trained/resumed on preserved shards
  -> Vina+DG and/or learned confidence score the same poses
  -> declared pure-confidence or frozen-composite policy orders poses
  -> SDF + results.pt + provenance
```
