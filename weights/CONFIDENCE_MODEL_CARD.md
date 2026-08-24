# EFF-Dock extmatch pose-confidence checkpoint

- File: `effdock_confidence_extmatch_n80_s25_step42500.pt`
- SHA-256: `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`
- Model type: `docking_graph_pose_confidence`
- Training step: 42,500
- Selected metric: `success_selected_lt2`
- Paired docking checkpoint: `effdock_geometry_ft_100k_best.pt`
- Pose-shard tag: `conf_ligonly_extmatch_n80_s25_sig0p5_pc10`
- Seed: 42

## Intended use

Rank multiple poses generated for the same receptor/ligand/explicit-pocket
complex. This checkpoint's training-matched preset was 80 poses, 25 ODE steps,
translation sigma 0.5, and a 10A protein crop. The public `dock` and `evaluate`
commands now use 100 poses and 10 ODE steps by default while retaining sigma
0.5 and the 10A crop. This is an explicit candidate-distribution shift. The
model predicts pose RMSD and success plus per-atom displacement/success heads;
these are ranking signals and are not claimed to be calibrated across datasets
or sampling distributions.

The frozen historical composite selector is
`pair_gate_density_rank_vote_plclash_ambig`. It combines the learned heads with
candidate density/rank voting and a narrow protein-clash fallback. Pure minimum
predicted RMSD is always reported separately so learned-model behavior is not
conflated with the composite selector.

The active implementation was checked against all 393 retained historical
Astex/PoseBusters per-pose records and reproduced the frozen selected index for
393/393 complexes.

## Frozen validation evidence

The retained checkpoint metadata reports, on the frozen 256-complex validation
subset, 49.22% selected RMSD <2A from the RMSD head, 49.61% from the success
head, and 68.75% oracle coverage. Two 500-step hard-pair fine-tunes were tested;
neither improved this baseline, so they were not promoted.

## External evidence boundary

Historical reference-defined redocking reached 81.18% on Astex and 77.60% on
PoseBusters v2 with the frozen composite selector. Those values are diagnostic,
not prospective screening claims. Active frozen-manifest results and their
failure/rescue records live in `docs/BENCHMARK_RESULTS.md`.

The completed active N80 rerun reached composite <2A rates of 78.82% on Astex,
72.73% on PoseBusters, and 68.42% on CASF. Same-candidate pure confidence was
76.47%, 73.05%, and 69.47%, respectively. The checkpoint remains the selected
confidence model; the composite selector is retained for reproducibility but
did not consistently improve the learned RMSD head.

## Experimental S50 successor

A later internal PLINDER study retrained this confidence model on the frozen
S50 N100/S10/sigma-2 bank using symmetry-aware no-alignment RMSD. Its selected
U25k checkpoint reached 58.45% Top-1 `<2A` on 1,035 internal validation
complexes; the terminal U50k checkpoint reached 56.81%. These checkpoints are
preserved under ignored `outputs/` and are not packaged release artifacts, so
they do not change this model card's default checkpoint or intended-use
contract. Exact hashes, slices, and repeated-use external diagnostics are in
`docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`.

## Limitations

- Requires an explicit pocket and a compatible EFF-Dock hidden representation.
- Performance can shift with pose count, sigma, ODE steps, pocket crop, receptor
  source, or candidate generator checkpoint.
- Does not predict binding affinity or binder/non-binder status.
- Reference-defined benchmark pockets do not demonstrate blind pocket finding.
