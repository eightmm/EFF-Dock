# Recovered PoseX official evaluation

Submitted 2026-09-15 after verifying complete relaxation coverage: SD 718 and
CD 1,312 cases for each of seeds 101, 202, 303. Recovery compatibility changes
are documented in `POSEX_METAL_CHAIN_RECOVERY.md` and
`POSEX_RELAX_RECOVERY_20260914.md`; other models use published numbers.

Job array 74738 (indices 0–5, concurrency 3) evaluates SD101/202/303 followed
by CD101/202/303. Job 74742 depends on successful completion of the whole
array and writes the three-seed mean and sample standard deviation. CPU-only
partition; no new docking or relaxation is performed.

`scripts/evaluate_posex_recovered.py` resolves each benchmark ID to exactly
one complete pair across base v1 and the two v2 recovery directories. It
copies final unaligned inputs into a separate evaluation workspace, retaining
source paths and SHA256 hashes. Missing/ambiguous cases fail before evaluation.
The original benchmark CSV is copied too, because upstream writes results
beside that CSV. Existing raw, fixed-receptor and upstream results remain intact.

The unchanged official alignment script runs first. Every case must produce
both aligned files. The unchanged official PoseBusters evaluator then runs
with `--relax true`, and result IDs must match the complete input set. Existing
summary code uses the official PDB_GROUP → GROUP means, RMSD ≤2 Å and joint
RMSD ≤2 Å / PB-valid fraction ≥0.5 criteria. SD groups are individual case IDs.

Per-run output: `outputs/benchmarks/posex_official_protocol/runs/{sd,cd}-seed{101,202,303}/upstream_recovered_evaluation_v1/`.
Each includes `inputs.json`, `processed/`, official result CSV and, only after
coverage validation, `summary.json`.

Final output: `outputs/benchmarks/posex_official_protocol/upstream_recovered_evaluation_v1_summary.json`.

Python syntax, shell syntax and changed-file lint passed. First three array
tasks observed RUNNING; final scientific results remain pending. The previous
repository-wide fast check had unrelated lint failures and was not repeated
for this isolated evaluation launcher.
