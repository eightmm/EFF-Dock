# Output retention inventory

Generated: 2026-08-10T05:04:47.579057+00:00

> Generating this inventory is non-destructive. Any previously applied recoverable archive move is listed below; no files were deleted. New cleanup candidates require a separate reference check and explicit approval.

Applied recoverable archive operations:

- `outputs/archive/20260810-transient/ARCHIVE_MANIFEST.json`
- `outputs/archive/20260810-smoke/ARCHIVE_MANIFEST.json`

## outputs

Total indexed size: 18.09 GiB across 228 directories.

Classification counts: retain=7, retain_active=1, retain_referenced=8, review=205, review_large_transient=7

| class | size | files | references | path | reason |
|---|---:|---:|---:|---|---|
| retain_active | 2.01 GiB | 136428 | 418 | `outputs/benchmarks` | currently active Slurm output root |
| retain_referenced | 1.60 GiB | 23 | 1 | `outputs/flowfrag_geometry_ft_100k_from_200k` | referenced by repository code or documentation |
| review_large_transient | 1.17 GiB | 4464 | 0 | `outputs/external_eval_pair_gate_density_rank_vote_full_foreground_20260714T102034Z-DCTN-0429113256-nogit-a6a24d` | transient-name pattern but too large for automatic archival |
| review | 0.87 GiB | 15 | 0 | `outputs/flowfrag_small_sigma_mixture_ft_50k` | no automatic retention decision |
| review | 0.87 GiB | 14 | 0 | `outputs/flowfrag_vp_flow_scratch_50k` | no automatic retention decision |
| review | 0.87 GiB | 13 | 0 | `outputs/flowfrag_vp_score_full_amin01_scratch_50k` | no automatic retention decision |
| retain | 0.77 GiB | 29 | 25 | `outputs/eff-dock` | named canonical/user-facing output root |
| review | 0.58 GiB | 10 | 0 | `outputs/flowfrag_local_refine_probe_30k` | no automatic retention decision |
| retain | 0.56 GiB | 224 | 21 | `outputs/guidance` | named canonical/user-facing output root |
| review | 0.51 GiB | 8 | 0 | `outputs/flowfrag_vp_score_full_scratch_50k` | no automatic retention decision |
| review | 0.36 GiB | 21 | 0 | `outputs/flowfrag_torsion_local_refine_50k` | no automatic retention decision |
| review | 0.27 GiB | 3942 | 0 | `outputs/external_benchmarks_nohidden_100k_sde` | no automatic retention decision |
| review | 0.27 GiB | 3943 | 0 | `outputs/external_benchmarks_nohidden_100k` | no automatic retention decision |
| retain_referenced | 0.25 GiB | 58 | 2 | `outputs/external_eval_extmatch_conf_n80_s25_conf_success_atom_q90_manual_20260713` | referenced by repository code or documentation |
| review | 0.24 GiB | 8665 | 0 | `outputs/external_eval_best_n80_s25_nolocal_atom_success_20260708_230820` | no automatic retention decision |
| review | 0.24 GiB | 8277 | 0 | `outputs/external_eval_local40x8_conf_localresample50k_best_20260708_100936` | no automatic retention decision |
| review | 0.22 GiB | 21 | 0 | `outputs/flowfrag_torsion_local_refine_matched_50k` | no automatic retention decision |
| review | 0.21 GiB | 524 | 0 | `outputs/external_eval_targeted_pb_final3_20260715` | no automatic retention decision |
| review | 0.19 GiB | 7 | 0 | `outputs/pose_confidence_extmatch_hardpair_ft_500step_20260715` | no automatic retention decision |
| review | 0.16 GiB | 10 | 0 | `outputs/selector_sweeps` | no automatic retention decision |
| review | 0.15 GiB | 6 | 0 | `outputs/pose_confidence_extmatch_n80_s25_sig0p5_pc10_muon_50k` | no automatic retention decision |
| review | 0.15 GiB | 3 | 0 | `outputs/pose_confidence_pocket_lighidden_local40x8_global_contact_attention_muon_50k` | no automatic retention decision |
| retain_referenced | 0.15 GiB | 1623 | 2 | `outputs/.pymol-venv` | referenced by repository code or documentation |
| review | 0.15 GiB | 3 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_global_contact_attention_muon_20k` | no automatic retention decision |
| review | 0.15 GiB | 3 | 0 | `outputs/pose_confidence_extmatch_hardpair_ft_500step_freshopt_20260715` | no automatic retention decision |
| review | 0.15 GiB | 3 | 0 | `outputs/flowfrag_robust_sigmix_jitter2_200k` | no automatic retention decision |
| review | 0.14 GiB | 6527 | 0 | `outputs/external_eval_pose_sdf_current_n80_s25_20260714T1220FILTER` | no automatic retention decision |
| review | 0.14 GiB | 1977 | 0 | `outputs/external_benchmarks_nohidden_64det_30k` | no automatic retention decision |
| review | 0.14 GiB | 1971 | 0 | `outputs/external_benchmarks_detconf_20k` | no automatic retention decision |
| review | 0.14 GiB | 1972 | 0 | `outputs/external_benchmarks_nn` | no automatic retention decision |
| review_large_transient | 0.13 GiB | 2 | 0 | `outputs/smoke_contact_attention_confidence_1step` | transient-name pattern but too large for automatic archival |
| review | 0.12 GiB | 7 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_nn_muon_200k` | no automatic retention decision |
| review | 0.12 GiB | 3 | 0 | `outputs/pose_confidence_pocket_lighidden_muon_200k` | no automatic retention decision |
| review | 0.12 GiB | 6 | 0 | `outputs/pose_confidence_pocket_nohidden_v2_nn_muon_100k` | no automatic retention decision |
| review | 0.12 GiB | 5 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_muon_200k` | no automatic retention decision |
| review | 0.12 GiB | 5 | 0 | `outputs/pose_confidence_pocket_nohidden_v2_clean_success_listwise_64pose_det_muon_30k` | no automatic retention decision |
| retain_referenced | 0.12 GiB | 4 | 1 | `outputs/pose_confidence_pocket_lighidden_v2_clean_success_listwise_64pose_det_muon_20k` | referenced by repository code or documentation |
| review | 0.12 GiB | 3 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_hardpair_muon_50k` | no automatic retention decision |
| review | 0.12 GiB | 4 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_clean_success_listwise_64pose_muon_50k` | no automatic retention decision |
| review | 0.12 GiB | 4 | 0 | `outputs/pose_confidence_pocket_lighidden_v2_clean_listwise_muon_50k` | no automatic retention decision |

## outputs/benchmarks

Total indexed size: 2.01 GiB across 20 directories.

Classification counts: retain=6, retain_active=1, retain_referenced=12, review=1

| class | size | files | references | path | reason |
|---|---:|---:|---:|---|---|
| retain_active | 0.00 GiB | 1 | 2 | `outputs/benchmarks/guidance_eta_cap_extension_runs` | currently active Slurm output root |
| retain | 0.66 GiB | 6350 | 10 | `outputs/benchmarks/guidance_steric_high_eta_confidence_runs` | named canonical/user-facing output root |
| retain | 0.52 GiB | 5094 | 18 | `outputs/benchmarks/guidance_sigma_sweep_eta2_runs` | named canonical/user-facing output root |
| retain_referenced | 0.33 GiB | 35882 | 14 | `outputs/benchmarks/guidance_eta_sweep_v2_runs` | referenced by repository code or documentation |
| retain_referenced | 0.19 GiB | 7386 | 4 | `outputs/benchmarks/guidance_eta_sweep_confidence_standalone_runs` | referenced by repository code or documentation |
| retain | 0.11 GiB | 3462 | 115 | `outputs/benchmarks/logs` | named canonical/user-facing output root |
| retain_referenced | 0.06 GiB | 28402 | 5 | `outputs/benchmarks/pocket_sensitivity_n80_s25_v2` | referenced by repository code or documentation |
| retain_referenced | 0.05 GiB | 9670 | 6 | `outputs/benchmarks/guidance_direct_drift_v1` | referenced by repository code or documentation |
| review | 0.04 GiB | 20074 | 0 | `outputs/benchmarks/pocket_sensitivity_n80_s25_v1` | no automatic retention decision |
| retain | 0.04 GiB | 9699 | 198 | `outputs/benchmarks/guidance_budget1000_full_v2` | named canonical/user-facing output root |
| retain_referenced | 0.01 GiB | 6855 | 10 | `outputs/benchmarks/raw` | referenced by repository code or documentation |
| retain_referenced | 0.01 GiB | 3473 | 5 | `outputs/benchmarks/guidance_budget1000_v1` | referenced by repository code or documentation |
| retain_referenced | 0.00 GiB | 20 | 2 | `outputs/benchmarks/plinder_guidance_validation_runs` | referenced by repository code or documentation |
| retain_referenced | 0.00 GiB | 1 | 5 | `outputs/benchmarks/plinder_guidance_validation` | referenced by repository code or documentation |
| retain | 0.00 GiB | 5 | 8 | `outputs/benchmarks/confidence` | named canonical/user-facing output root |
| retain_referenced | 0.00 GiB | 3 | 1 | `outputs/benchmarks/combined` | referenced by repository code or documentation |
| retain | 0.00 GiB | 16 | 1 | `outputs/benchmarks/confidence_posebusters_official` | named canonical/user-facing output root |
| retain_referenced | 0.00 GiB | 16 | 1 | `outputs/benchmarks/vina_guidance_posebusters_official` | referenced by repository code or documentation |
| retain_referenced | 0.00 GiB | 16 | 3 | `outputs/benchmarks/posebusters_official` | referenced by repository code or documentation |
| retain_referenced | 0.00 GiB | 0 | 2 | `outputs/benchmarks/guidance_direct_drift_runs` | referenced by repository code or documentation |

## Retention policy

1. Keep active runs, model weights, frozen input manifests, selected SDFs, exact cohort audits, official PB shards, and final aggregates.
2. Archive completed but superseded runs as intact directories so hashes and relative paths remain meaningful.
3. Delete only unreferenced smoke/dry/debug/failed artifacts after confirming they are not a parent of a retained report.
4. Never delete raw benchmark data or training data; data remain ignored by Git as defined by the project contract.
