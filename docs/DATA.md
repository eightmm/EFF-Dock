# Data contract

EFF-Dock uses PLINDER 2024-06/v2 complexes. Coordinates are Angstroms and the
immutable sample key is `<system_id>__<ligand_instance_chain>`.

## Local assets

- Curated pool: `data/plinder_pool.parquet` (50,297 rows at migration time).
- Processed tensors: `data/plinder_processed/` (50,213 sample directories at
  migration time).
- Preserved compatibility split: `data/splits/plinder.json` (47,310 train,
  1,076 validation). This split is usable for legacy compatibility work, but it
  predates the strict EFF-Dock external-exclusion contract.
- Raw data, processed tensors, splits, external structures, and manifests stay
  local and are ignored by Git. Cleanup must never delete or rewrite them.

Each processed sample contains protein, ligand, and metadata tensor mappings.
Invalid structures must be quarantined with an explicit reason; scientific
features are not silently zero-filled.

## New split contract

`eff-dock data split` intersects the pool with successfully processed samples,
canonicalizes ligands with RDKit, strictly removes canonical SMILES found in
frozen Astex Diverse and PoseBusters v2 mappings, then groups
validation by `pocket_fident__70__community`. Train and validation must be
disjoint on sample key, canonical SMILES, and pocket70 community.

The command intentionally fails until both files exist:

```text
data/external_test/astex_smiles.json
data/external_test/pb_smiles.json
```

This prevents an apparently strict split from being generated with an omitted
benchmark. Every publishable split must record source snapshot hashes, seed,
counts, exclusions, and preprocessing version in its manifest.

## Commands

```bash
uv run eff-dock data curate --help
uv run eff-dock data prepare --help
uv run eff-dock data split --help
```

Any source refresh, feature/schema change, split-policy change, or preprocessing
change invalidates the corresponding manifest and compatibility claim.
