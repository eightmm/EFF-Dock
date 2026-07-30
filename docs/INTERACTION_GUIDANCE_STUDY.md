# Interaction Guidance Study

## Status

- Protocol ID: `EFFDOCK-INTERACTION-GUIDANCE-V1`.
- Protocol freeze: pre-registered before screened-formal-charge implementation
  and coordinate scoring.
- Current stage: V4 diagnostic implemented and Stage 1A executable admission
  gates passed on CPU. Stage 1B uses the existing frozen Astex Diverse and
  PoseBusters v2 raw structures for report-only fixed-coordinate
  characterization and is complete. Production activation still requires an
  internal held-out ablation.
- Goal: extend `InteractionEnergy` one independently traceable term at a time,
  while keeping the docking model, confidence model, candidate sets, and
  external benchmarks frozen.
- Runtime boundary: ligand/receptor chemistry may be typed once from the
  sanitized graph and canonical residue tables. Every coordinate-dependent
  energy, gradient, group center, switch, and force is evaluated in Torch.
  External force-field, charge, docking, minimization, and simulation engines
  are forbidden at runtime.

## Decision boundary

Scientific admission and sampler activation are separate decisions.

1. A term is scientifically admitted when its chemical scope is explicit, its
   signs and symmetries are correct, it is finite and smooth over the declared
   domain, unsupported chemistry is reported, and its parameters have
   provenance.
2. An admitted term remains available as a separately traced diagnostic unless
   it is physically wrong or numerically unsafe.
3. A term enters the production ODE corrector only after a frozen internal
   held-out ablation (currently PLINDER validation) improves pose recovery or
   physical validity without violating geometry, clash, finite-gradient, or
   shell gates.
4. Astex Diverse and PoseBusters v2 may be used for frozen, report-only
   characterization, but never select formulas, constants, weights, schedules,
   force caps, term retention, or sampler activation.
5. If any external outcome is later used to make one of those choices, the
   consumed subset must be reclassified as development data and cannot support
   a blind benchmark claim.

## Stage 1 — Screened formal-charge guidance

### Question and hypothesis

Question: does a conservative, screened interaction between declared
protein/ligand formal-charge groups provide useful long-range ionic attraction,
like-charge repulsion, and rigid-body torque that the current hydrophobic and
hydrogen-bond terms cannot provide?

Diagnostic hypothesis: on charged, chemistry-supported complexes, the new term
will often improve the frozen-candidate crystal-preference margin and the
direction of the projected correction force relative to the current
`hydrophobic + hydrogen_bond` baseline. A ligand with no nonzero formal-charge
sites must receive exactly zero energy and gradient from this term. External
benchmark observations test this diagnostic hypothesis but do not activate,
retune, or remove the term.

The conclusion changes if the term violates charge conservation or SE(3)
behavior, produces non-finite/cutoff-discontinuous gradients, systematically
favours frozen wrong poses, or drives new ligand–protein collapse.

### Chemical scope

- Ligand charge is the explicit formal charge and protonation/tautomer state in
  the sanitized input molecule. No pKa, protonation, Gasteiger, MMFF, UFF,
  AM1-BCC, or learned partial-charge inference is performed.
- Common delocalized groups use versioned multi-atom sites so a resonance
  drawing does not duplicate the total charge. Other nonzero formal charges
  remain explicitly atom-centred and are reported.
- Protein sites are canonical side-chain priors only: ARG `+1`, LYS `+1`, ASP
  `-1`, GLU `-1`, and explicitly named HIP `+1`. HID/HIE are neutral; plain HIS,
  termini, modified residues, missing group atoms, cofactors, metals, and
  counterions do not receive inferred charge.
- Every group membership, group charge, exclusion, and total ligand/protein
  site charge is recorded. Charge-group membership must not overlap, and the
  ligand site-charge sum must equal the input molecular formal charge.
- This term replaces the planned generic salt-bridge term. Enabling both would
  double count the same ionic motif.

### Frozen functional form

For ligand charge group `a` and fixed receptor charge group `b`,

```text
R_a = sum_i W_ai x_i,       sum_i W_ai = 1
R_b = sum_j W_bj x_j,       sum_j W_bj = 1
d2_ab = ||R_a - R_b||^2
rho_ab = sqrt(d2_ab + alpha^2)
u_ab = clamp((r_off^2 - d2_ab) / (r_off^2 - r_on^2), 0, 1)
S5(u) = 10u^3 - 15u^4 + 6u^5

E_charge =
  (k_e / epsilon_r) *
  sum_ab Q_a Q_b exp(-kappa rho_ab) S5(u_ab) / rho_ab
```

Frozen diagnostic constants:

| Quantity | Value |
|---|---:|
| `k_e` | `332.06371 kcal mol^-1 A e^-2` |
| `epsilon_r` | `78.5` |
| `kappa` | `0.127 A^-1` |
| `alpha` | `0.75 A` |
| switch-on `r_on` | `8.0 A` |
| cutoff `r_off` | `10.0 A` |

`epsilon_r` is the 25 °C bulk-water value. `kappa` is the rounded
Debye–Hückel value for 0.15 M monovalent ionic strength, 298.15 K, and
`epsilon_r=78.5`. These conservative constants are fixed before any pose
scores are inspected. The soft core and squared-distance C2 switch are
numerical regularizers, not claims about solvation or dielectric
heterogeneity.

Provenance:

- Lee, Fitch, and García-Moreno, *Protein Science* 11 (2002),
  DOI `10.1110/ps.4700102`, for a screened pairwise Coulomb form, the
  Debye-parameter relation, and the 25 °C water dielectric.
- NIST 2022 CODATA recommended fundamental constants for the unit-converted
  Coulomb prefactor.

### Stage 1A — executable admission gates

All gates are mandatory:

- ligand and protein group membership has valid shape, non-overlap, normalized
  weights, and conserved declared charge;
- opposite charge is attractive, like charge is repulsive, and neutral ligand
  energy/gradient is exactly zero;
- energy and gradient remain finite at exact overlap and at large absolute
  coordinate offsets;
- energy and force reach zero smoothly at the cutoff;
- autograd, finite-difference, batch/loop, float32/float64, and CPU behavior
  agree within declared tolerances;
- scalar energy is SE(3)-invariant and force is rotation-equivariant;
- group-center force distributes to member atoms according to the frozen
  weights;
- the trace reports site counts, exclusions, attractive/repulsive sums, and
  strongest signed pairs;
- the active receptor shell covers the 10 A charge cutoff.

Failure of any gate blocks scientific admission and ODE experiments.

Frozen executable tolerances and coordinate domain:

| Gate | Frozen check |
|---|---|
| formal-charge conservation | exact integer site sum; membership row sum absolute tolerance `1e-6` |
| analytic scalar value | float64 relative tolerance `1e-12` |
| finite difference | centered step `1e-5 A`; gradient absolute/relative tolerance `1e-10` / `2e-9` |
| batch versus pose loop | float64 absolute/relative tolerance `1e-13` |
| SE(3) | energy `1e-12`; force `1e-11` absolute/relative tolerance |
| common large offset | float64 translation magnitude up to `3e6 A`; energy/force tolerance `1e-12` |
| float32 versus float64 | pocket-centred coordinates; energy `2e-7` / `2e-6`, force `2e-7` / `2e-5` absolute/relative tolerance |
| switch boundaries | value, first derivative, and second derivative tolerances `1e-14`, `1e-13`, and `1e-12` |

The production float32 domain is pocket-centred before down-casting. Absolute
coordinates near `1e8 A` are explicitly outside that domain because float32
cannot preserve ordinary intermolecular separations there. Diagnostic absolute
PDB coordinates remain float64.

Stage 1A verification evidence:

```text
uv run pytest -q \
  tests/test_formal_charge_typing.py \
  tests/test_interaction_energy.py \
  tests/test_physical_diagnostics.py

64 passed
```

This admits `screened_formal_charge` as a separately traced diagnostic. It does
not activate the term in the production ODE corrector.

### Stage 1B — fixed-coordinate screen

The cheapest falsifying experiment scores identical frozen coordinates without
relaxation. This stage characterizes the already frozen term on external data;
it is not a tuning or production-activation gate.

1. Inventory every frozen Astex Diverse and PoseBusters v2 ligand using the
   exact sanitized raw SDF. Eligibility is **any atom with nonzero formal
   charge**, not nonzero molecular net charge.
2. Before inspecting energy or RMSD outcomes, retain only complexes that pass
   the strict receptor-shell chemistry contract and freeze the sorted IDs.
3. Score the crystal pose and its already frozen/generated candidate poses with
   the baseline and with `baseline + screened_formal_charge`.
4. Record per pose:
   `E_baseline`, `E_charge`, `E_augmented`, attractive/repulsive pair counts,
   maximum/RMS atom force, fragment translation/rotation projections, RMSD,
   physical-validity fields, typing hash, parameter hash, input hash, command,
   seed, and code diff/commit.
5. Primary metric is the paired change in correct-pose selection margin:

```text
margin = min_wrong(E) - min_correct(E)
delta_margin = margin_augmented - margin_baseline
```

Secondary metrics are top-1 RMSD success, charge-force direction on wrong
poses, and physical-validity/clash slices. Existing neutral frozen cases,
including 1KE5, are recorded as `ineligible_no_ligand_formal_charge` and are
not counted as negative evidence.

The external result is reported unchanged. It cannot advance the term, alter
the frozen constants, or select the next interaction. Advancement to
relaxation is decided only by the separate internal held-out gate in Stage 1C.

### Stage 1B coverage inventory

All 308 PoseBusters v2 and 85 Astex Diverse ligand SDFs sanitize successfully.
PoseBusters contains 15 ligands with at least one nonzero formal-charge site;
six of those are net-neutral zwitterions that a net-charge filter would miss.
Five of the 15 pass the strict 10 A receptor-shell chemistry contract. The
other ten fail closed on an active metal, cofactor, ion, or nonprotein residue.
Astex contains five site-bearing ligands, but all five fail the same strict
shell contract because sulfate, sodium, heme/cofactor, or another unsupported
active component is present.

The five outcome-blind PoseBusters IDs are `6Z14_Q4Z`, `7JMV_4NC`,
`7M6K_YRJ`, `7RH3_59O`, and `7V14_ORU`; their newline-list SHA256 is
`9884c9222a37ecda6b27ef655fd814ff5951b8da16d9cfc93e2f7993b40604c6`.
Exact input hashes, exclusions, crystal diagnostics, and frozen 40-pose
characterization are recorded in
`GUIDANCE_FORMAL_CHARGE_BENCHMARK_CHARACTERIZATION_V4.json`.

The earlier PLINDER net-charge-only inventory is preserved in
`GUIDANCE_FORMAL_CHARGE_COVERAGE_V4.json` as a historical proxy. It is
superseded for eligibility because molecular net charge does not detect
zwitterionic formal-charge sites.

### Stage 1B result

The five supported PoseBusters complexes each have 40 candidates frozen before
this term was evaluated. Interaction-energy top pose is unchanged for four of
five complexes. On `7RH3_59O` it changes from RMSD `6.384 A` to `3.840 A`, but
remains incorrect at the `2 A` threshold.

Only three candidate sets contain both a correct and a wrong pose, so the
paired margin is defined for three complexes. One improves and two worsen:
the median `delta_margin` is `-0.010864 kcal/mol`, and the non-loss fraction is
`1/3`. `6Z14_Q4Z` has the largest regression (`-0.857959 kcal/mol`), showing
that the unsaturated screened ionic proxy can favor a wrong highly ionic pose.
All five crystal energies and gradients are finite.

This is a negative descriptive result for using the charge term alone as a
candidate selector. It neither removes the physically interpretable
diagnostic nor admits it to the sampler, and no constant is changed from these
external observations. Exact hashes and per-complex values are frozen in
`GUIDANCE_FORMAL_CHARGE_BENCHMARK_CHARACTERIZATION_V4.json`.

### Stage 1C — operator-split relaxation

Only after a separately frozen internal held-out screen passes:

- reuse identical stored candidates/seeds and the existing trust-region,
  backtracking, force-cap, shell, geometry, and clash contract;
- change only activation of `screened_formal_charge`;
- compare paired final raw/aligned RMSD, top-k success, cut-bond/angle/chiral
  validity, ligand self-clash, protein–ligand clash, and rejection counts;
- require positive median RMSD change or a higher joint-success count with no
  regression in the geometry/finite/shell gates before production ODE
  integration is considered.

The internal gate requires positive median `delta_margin`, at least 60% of
eligible complexes with no margin loss, all force/shell gates passing, and no
new collapse or non-finite case. Astex/PoseBusters outcomes are not substituted
for this gate.

## Later terms

After Stage 1 is interpreted, terms are considered in this order, each with a
new frozen subsection and one-variable ablation:

1. `cation_pi`
2. `pi_stacking`
3. `halogen_bond`
4. `unsatisfied_polar`
5. the existing narrow Zn(II) coordination contract

No later term is combined with an unvalidated new term in its first experiment.
