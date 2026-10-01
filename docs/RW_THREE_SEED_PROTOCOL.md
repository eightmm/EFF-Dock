# Rw five-benchmark evaluation

Status: complete. Current primary manuscript figures use the corrected Rw three-seed results. Historical ablations and the earlier single-seed diagnostic retain their provenance. See `benchmarks/results/paper/rw_report.json` for the full frozen-cohort summary.

Evaluate the released docking and confidence weights without retraining on Astex
Diverse Set (85), PoseBusters v2 (308), PhiBench (206), FoldBench (558), and OpenBind
(925). Run corrected `rw` with base seeds 42, 100042, and 200042. Fresh
`legacy_rt_w` controls on Astex and PoseBusters use identical seeds and priors.
This is 6,246 corrected and 1,179 control complex executions.

Question: how does the corrected orientation injection perform under frozen
weights across inference seeds? Accuracy may improve or degrade; no directional
prediction is assumed. The scientific arm difference is orientation injection
in both generation and fresh confidence feature extraction. Confidence weights
were trained with legacy features; corrected inference is explicitly a
cross-operator evaluation. Do not select a checkpoint, seed, or selector using
these external results. Existing overlap and repeated benchmark inspection limit
claims of independent generalization.

Use the historical N100/S10 unguided protocol: supplied pocket center, 10 Å crop,
sigma 2, late time schedule with power 3, generation chunks of 10, and fresh
confidence chunks of 20. Preserve rigid energy post-refinement (100-step cap,
0.1 Å displacement caps, 5° rotation cap, 12 backtracks; energy plateau absolute
0.02 kcal/mol, relative 0.001, patience 5, minimum 25 steps). Preserve all saved
candidates and the input-only tetrahedral chirality selector, falling back to
unfiltered minimum predicted RMSD if no candidate is eligible.

Each complex uses base seed plus its one-based position in the globally sorted
cohort. Preserve historical input conformer policies: Astex/PoseBusters generation
uses the frozen benchmark manifest and conformer seed 0; temporal cohorts use
their existing normalized inputs and the per-complex sampling seed. Refinement
and rescoring reconstruct inputs with the existing sampling-seed policy.
References enter RMSD, PoseBusters, and passive diagnostics only. No reference
coordinates or success labels enter selection or energy optimization.

Primary outcome is refined selected-pose success: all-heavy-atom receptor-frame
RMSD <2 Å and all 27 non-RMSD PoseBusters checks. Also report raw/refined RMSD-only
success, PB validity, oracle RMSD success, unfiltered selection, and three-seed
mean ± sample SD. RMSD is symmetry-aware where topology permits; preserve the
documented full-atom mapped fallback for representation mismatches. No atom
subset scoring or failed-case exclusion. Retain PhiBench reconstruction flags
and OpenBind's noncovalent approximation flags.

Use PoseBusters 0.6.5 through the historical cohort-specific paths: native
official redock for Astex/PoseBusters, and the previously audited InChI
energy-reference roundtrip compatibility adapter for the three temporal cohorts.
Use the same path for both operators and all repeats within a cohort. The adapter
does not edit candidate coordinates or relax molecular identity checks.

Execution amendment before full-cohort launch: the initial universal adapter
passed nine PB smoke cases but rejected a stereo identity change for the
8F4J_PHO energy-reference roundtrip. Preserve that failed attempt. Restore the
historical cohort-specific paths using the already saved candidates and selectors;
do not weaken the identity guard, alter ligand chemistry, drop the case, or
regenerate poses. Verify native selected-pose PB checks and freeze the recovery
sources before admitting the full study. Initial generation smoke outputs remain
unchanged. This amendment repairs evaluation compatibility, without selecting a
method or seed based on docking success.

Execution backends are recorded per shard. The completed study used validated RTX 6000 Ada, RTX A5000 and RTX PRO 6000 Blackwell Max-Q GPUs with identical software, precision, candidate chunks and scientific settings. Prespecified largest-ligand probes established finite output and memory headroom on each added backend; those probes do not enter estimates. Only pending tasks were reassigned, operator pairs shared one GPU and identical priors, and completed/running tasks were not restarted. User-authorized scheduling amendments removed the initial concurrency cap while retaining scheduler limits and minimum CPU allocations. Hardware-dependent numerical variation remains part of execution uncertainty; these results do not establish hardware-invariant predictions.

Freeze source, inputs, cohort IDs, checkpoint hashes, and commands before launch.
Require representative and largest-ligand smoke tests, finite complete 100-pose
banks, operator provenance, exact paired prior hashes, and full selected-pose PB
checks before full execution. Failure of an integrity gate stops downstream
aggregation; retain failed attempts and repair only identified execution faults.
Execution repeatability diagnostics do not select the retained realization or
exclude a complex. No favorable-seed retries. Report paired differences on
matching cases; single-seed diagnostic intervals do not establish seed stability.
