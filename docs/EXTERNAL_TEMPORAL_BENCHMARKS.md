# Recent external docking benchmark registry

Downloaded structures, normalized complexes, and generated manifests remain
under ignored `data/` paths. This file records the public source, license,
source identity, and exact derivation used for each EFF-Dock cohort.

## Frozen cohorts

| EFF-Dock name | N | Intended use |
|---|---:|---|
| `phibench` | 203 | Recent, high-identity-deduplicated pocket redocking |
| `foldbench` | 66 | Post-2024-06-30 pocket-redocking adaptation |
| `openbind` | 860 | Clean non-covalent EV-A71/CVA16 2A redocking |

The ignored JSON manifests freeze every admitted ID, exclusion, source
checksum, and final ID-list hash. No benchmark structure or generated pose is
stored in Git.

## PhiBench

- Paper and code: [PhysDock](https://github.com/KexinZhangResearch/PhysDock)
- Data: [Zenodo record 15178859](https://zenodo.org/records/15178859), CC BY 4.0
- Official archive MD5: `ad71e631eb439367667a89de8c41892e`
- Selection: for official PhiBench members with an RCSB date record, require a
  deposit date from June through December 2024. Retain the two official-archive
  members for which RCSB returned no record as explicit frozen exceptions.
  Then require a ligand chain and complete heavy-atom reference, collapse
  identical receptor sequences, and select one deterministic representative
  per connected component at 99.5% global sequence identity.

The public archive contains 476 PhiBench files. Three have no ligand chain and
eight ligand references omit one to three heavy-atom coordinates; the latter
are excluded rather than repaired. RCSB supplied dates for 425 of 427 PDB IDs.
The unresolved entries are `9j9c` and `9jd2`; the frozen manifest records both
with null dates and records the official-archive exception count. Their dates
are therefore not verified, and the 203-system result is labelled an
**EFF-Dock-derived archive cohort**, not a strictly date-verified set. The
PhysDock release also does not
expose the exact 206-ID author curation, so these results are not claimed to
reproduce the paper's hidden cohort or to establish broad sequence diversity.

New CCD entries absent from PhysDock's released CCD metadata are retrieved as
official RCSB Chemical Component CIFs. Their URLs and SHA-256 values are frozen
in the local manifest; bonds are not inferred from coordinates.

## FoldBench P-L

- Project: [FoldBench](https://github.com/BEAM-Labs/FoldBench)
- Registry: [OpenFold benchmark portal](https://portal.openfold.omsf.io/benchmarks/fold-bench)
- License: CC BY 4.0
- Interface CSV SHA-256:
  `f0bf964ca1b9699e2036baa9bdfcc231e56181ec2a6df0f2eb24000e23cf3e0a`
- Ground-truth archive SHA-256:
  `69d72dbbddaa4a6b4005220b8eafc09d1a0f7575dcf3783686e7847655f3e1c9`
- Selection: 66 of 558 official protein-ligand interfaces have an RCSB
  `initial_release_date` strictly later than 2024-06-30.

This is an EFF-Dock pocket-redocking adaptation. Its RMSD and PB-valid values
must not be presented as native FoldBench leaderboard metrics, which use a
different prediction contract and include LDDT-PLI.

## OpenBind EV-A71/CVA16 2A

- Project: [OpenBind benchmark repository](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark)
- Data: [Zenodo DOI 10.5281/zenodo.20026661](https://doi.org/10.5281/zenodo.20026661), CC0 1.0
- Official archive MD5: `860a4979d0ba9decaa2bfaa933c1d217`
- Clean-cohort selection: require `covalent=False`,
  `pb_valid_prepared=True`, `pb_valid_ref=True`, and
  `suspected_artefact=False`, yielding 860 of 925 structures.

The 860-system clean cohort is the broad target-family characterization. The
separate 802-system official-style cohort additionally follows OpenBind's
filtered scaffold-only figure contract; the two denominators must not be
interchanged.

## Relationships and information boundary

PhiBench and FoldBench have no PDB-ID overlap with the frozen Astex Diverse or
PoseBusters v2 cohorts. PhiBench and FoldBench share one PDB entry (`9jff`), so
their per-dataset tables are valid but a combined count must deduplicate it.
OpenBind is intentionally a dense enterovirus 2A-protease series rather than a
target-diversity benchmark.

All evaluations are pocket-conditioned redocking. The frozen pocket center is
computed from the crystal complex as the centroid of receptor residue virtual
nodes within 8 A of the reference ligand, with the ligand centroid as fallback.
That derived three-vector, prepared receptor coordinates, and ligand chemistry
enter inference. Reference ligand atom coordinates do not otherwise enter the
model or GuidanceEnergy and are used directly after generation for RMSD. These
results do not establish blind pocket finding, cross-docking, or co-folding
performance.
