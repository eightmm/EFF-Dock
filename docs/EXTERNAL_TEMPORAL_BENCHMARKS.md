# Recent external docking benchmarks

EFF-Dock keeps downloaded structures and generated benchmark inputs under
`data/`, which is gitignored. Only this source registry and the conversion code
belong in the public repository.

## Frozen evaluation cohorts

| EFF-Dock name | Local cohort | Intended use |
| --- | ---: | --- |
| `phibench` | 203 | Recent temporal, sequence-diverse pocket redocking |
| `foldbench` | 66 | Strict `initial_release_date > 2024-06-30` pocket-redocking adaptation |
| `openbind` | 860 | Clean non-covalent EV-A71/CVA16 2A redocking |

The counts are produced from the official raw releases by
`eff-dock data external`; the ignored JSON manifests freeze every selected ID,
filter, exclusion, source checksum, and ID-list hash.

### PhiBench

- Paper/code: [PhysDock](https://github.com/KexinZhangResearch/PhysDock)
- Data: [Zenodo record 15178859](https://zenodo.org/records/15178859), CC BY 4.0
- Official archive checksum: `md5:ad71e631eb439367667a89de8c41892e`
- EFF-Dock selection: retain official PhiBench members deposited from June to
  December 2024, require a ligand chain, collapse identical receptor sequences,
  then select one deterministic representative per connected component at
  99.5% global sequence identity.

The official archive contains 476 PhiBench files. Three contain no ligand
chain, and eight ligand references omit one to three heavy-atom coordinates.
The latter are excluded rather than filled with idealized coordinates. RCSB
currently returns dates for 425 of the 427 PDB IDs; the two unresolved IDs
remain eligible because they are members of the official PhiBench archive and
are explicitly marked `null` for their dates. After complete-heavy-reference
validation, the sequence procedure yields 203 systems. It is labelled an
**EFF-Dock derived cohort**: the PhysDock release does not expose an exact
206-ID author manifest, so we do not claim identity with the paper's hidden
curation list or force the count to match it.
New CCD IDs absent from PhysDock's bundled CCD pickle are fetched as official
RCSB Chemical Component CIF files; their URLs and SHA-256 hashes are frozen in
the local manifest rather than inferring bonds from coordinates.

### FoldBench P-L

- Project: [FoldBench](https://github.com/BEAM-Labs/FoldBench)
- Registry: [OpenFold benchmark portal](https://portal.openfold.omsf.io/benchmarks/fold-bench)
- Data license: CC BY 4.0
- Interface CSV checksum:
  `sha256:f0bf964ca1b9699e2036baa9bdfcc231e56181ec2a6df0f2eb24000e23cf3e0a`
- Ground-truth archive checksum:
  `sha256:69d72dbbddaa4a6b4005220b8eafc09d1a0f7575dcf3783686e7847655f3e1c9`
- EFF-Dock selection: 66 of the 558 official P-L interfaces have an RCSB
  `initial_release_date` strictly later than 2024-06-30.

This is an EFF-Dock pocket-redocking adaptation. Its RMSD/PB-valid numbers must
not be presented as the native FoldBench leaderboard metric, which evaluates a
different prediction contract and includes LDDT-PLI.

### OpenBind EV-A71/CVA16 2A

- Project: [OpenBind benchmark repository](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark)
- Data: [Zenodo DOI 10.5281/zenodo.20026661](https://doi.org/10.5281/zenodo.20026661), CC0 1.0
- Official archive checksum: `md5:860a4979d0ba9decaa2bfaa933c1d217`
- EFF-Dock selection: retain rows where `covalent=False`,
  `pb_valid_prepared=True`, `pb_valid_ref=True`, and
  `suspected_artefact=False`. This gives 860 of 925 structures.

The full 925-system raw release is retained locally; filtering affects only the
frozen evaluation cohort.

## Cohort relationships

The frozen PhiBench and FoldBench sets have no PDB-ID overlap with the local
Astex Diverse or PoseBusters v2 cohorts. PhiBench and FoldBench share one PDB
entry (`9jff`), so a combined cross-dataset total must deduplicate that entry;
separate per-dataset tables need no adjustment. OpenBind is intentionally a
dense enterovirus 2A-protease series rather than a target-diversity benchmark.

## Preparation and evaluation

After placing/extracting the official files at the paths shown by `--help`:

```bash
uv run eff-dock data external --dataset all
```

This creates:

```text
data/external_benchmarks/normalized/<dataset>/<id>/<id>_protein.pdb
data/external_benchmarks/normalized/<dataset>/<id>/<id>_ligand.sdf
data/external_benchmarks/manifests/<dataset>.json
data/external_test/<dataset>_smiles.json
data/external_test/<dataset>_reference_pocket_centers.json
```

One dataset can then be sampled with the ordinary evaluator, for example:

```bash
uv run eff-dock evaluate \
  --dataset phibench \
  --data-dir data/external_benchmarks/normalized/phibench \
  --pocket-centers data/external_test/phibench_reference_pocket_centers.json
```

Do not use `--skip-source-verification` for reported results. PhysDock's inputs
and CCD metadata are Python pickles; checksum verification is the trust boundary
before they are deserialized.

## Temporal cross-docking status

PhysDock describes a 25-protein/50-case temporal cross-docking experiment, but
the public benchmark archive does not expose an exact receptor-pair manifest or
the receptor-frame transformation needed for a correct cross-docking reference.
EFF-Dock therefore does not fabricate this benchmark. Add it only after the
authors publish the pair list and alignment contract (or provide them directly).
