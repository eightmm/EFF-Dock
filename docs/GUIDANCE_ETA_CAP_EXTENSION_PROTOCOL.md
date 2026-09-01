# EFF-Dock eta cap-saturation extension

## Scope

This paired descriptive extension changes only the normalized direct-drift
guidance coefficient beyond the completed high-eta sweep. It cannot select a
production eta or admit guidance. Astex and PoseBusters outcomes have already
been observed and are report-only.

## Frozen conditions

- New arms: `eta={2.5,3.0}`.
- Reference arms: `eta={0,0.5,1.0,1.5,2.0}` from
  `outputs/benchmarks/guidance_steric_high_eta_confidence_runs/20260807T045916Z`.
- Prior: `sigma=0.5`, 100 poses, base seed 42 plus frozen complex position.
- Cohorts: all 85 Astex Diverse and all 308 PoseBusters v2 complexes.
- Budget: `N100/S10`; late schedule, power 3; reference pocket, 10 Angstrom
  cutoff, no center jitter, no refinement.
- Guidance: `normalized_drift`, start `t=0.5`, ramp power 1, force cap 20,
  translation and angular caps 5, atom-displacement cap 0.25 Angstrom,
  maximum 8 backtracks, protein shell 18 Angstrom, `geometry_only` receptor
  policy. Vina is disabled.
- Frozen docking and confidence checkpoints and input manifests are identical
  to `docs/GUIDANCE_STERIC_HIGH_ETA_CONFIDENCE_PB_PROTOCOL.md`.

## Primary telemetry

For each dataset and eta, report the fraction of direct pose-step evaluations
triggering any cap, multiple caps, translation cap, angular cap, and estimated
atom-displacement cap; mean accepted cap scale; mean applied/model atom-speed
ratio; applied/model path-proxy ratio; and model-guidance cosine. RMSD,
confidence Top-1, internal fast-valid, and their conjunction are secondary
descriptive outcomes. Internal fast-valid is not official PoseBusters.

## Execution gate

Run 4 smoke tasks, audit them, then run 32 full GPU tasks
(`2 datasets x 2 eta arms x 8 shards`), audit the exact 786 complex-arm rows,
and generate one report combining reference and new arms. Every task must have
zero complex failures and zero non-finite guidance counters. CUDA allocated
memory must remain below 48 GiB; reserved allocator cache is record-only.
