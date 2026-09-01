# Interaction Guidance Study

## Status

- Protocol ID: `EFFDOCK-INTERACTION-GUIDANCE-V1`.
- Protocol freeze: pre-registered before screened-formal-charge implementation
  and coordinate scoring.
- Current stage: V6 diagnostic implemented and Stage 1A executable admission
  gates passed on CPU. Stage 1B uses the existing frozen Astex Diverse and
  PoseBusters v2 raw structures for report-only fixed-coordinate
  characterization and is complete. By explicit user request, all seven
  implemented interaction energies are enabled in the default diagnostic
  profile. Production activation still requires an internal held-out ablation;
  diagnostic default-on is not sampler admission.
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

## User-requested diagnostic default and later-term admission

The `EFF-Interaction-v1-diagnostic` parameter profile is version `1.6.0`,
formula `effdock-interaction-diagnostic-7`; the combined profile is version
`1.4.0`, formula `physical-v2_plus_interaction-v1.6`. By explicit user request,
the current default `active_terms` contains all seven implemented,
separately traceable interaction energies:

```text
hydrophobic
hydrogen_bond
screened_formal_charge
pi_stacking
cation_pi
halogen_bond
metal_coordination
```

Default-on is an operational diagnostic choice, not a scientific-validation
claim. None of `pi_stacking`, `cation_pi`, `halogen_bond`, or
`metal_coordination` may enter the candidate selector or production sampler
until its own crystal admission and internal held-out one-variable ablation
pass. Astex/PoseBusters observations remain report-only and cannot change
these frozen formulas or constants.

The completed prior-probe V2 remains frozen with its historical three-term
baseline (`hydrophobic + hydrogen_bond + screened_formal_charge`). Its
protocol arms, 96 stored trajectories, and no-admission conclusion are not
reinterpreted or regenerated merely because the current diagnostic default
now enables seven terms.

### Stage 2A — aromatic geometry shared contract

Neutral five- and six-member ligand aromatic rings come from the sanitized
static graph, including neutral N/O/S heteroaromatics, and are grouped into
connected fused systems. Charged rings are excluded. Protein pi-stacking rings
use complete canonical PHE, TYR, the two constituent TRP rings, and neutral
HIS/HID/HIE; HIP is excluded. Cation-pi later uses the narrower neutral
carbocyclic subset and excludes every histidine. A deterministic non-collinear
atom triplet defines each ring normal; the input cross-product magnitude
`A_ref` is stored as static provenance. For current coordinates:

```text
Q5(s) = clamp(s,0,1)^3 * [10 - 15*clamp(s,0,1) + 6*clamp(s,0,1)^2]
S2(z;a,b) = Q5((b^2-z)/(b^2-a^2))

C_ring = mean_i x_i
n_ring = normalize(cross(x_b-x_a, x_c-x_a))
q_collapse =
  Q5((||cross||^2-(0.25*A_ref)^2) /
     ((0.75*A_ref)^2-(0.25*A_ref)^2))
```

A collapsed or nearly collinear ring therefore reaches exactly zero quality
without admitting an unstable unit-normal gradient. Constituent-ring contacts
are soft-OR aggregated once per fused system. Pi stacking is a small
orientation-specific correction on top of the generic carbon-contact
baseline. Their geometric overlap is traced, but one term does not suppress
the other because atom-site and ring-system normalizations are not
interchangeable.

Shared aromatic admission gates:

- deterministic ring/fused-system membership and reference hash;
- ring-order and ligand/protein exchange invariance where chemically
  symmetric;
- finite value and gradient for collapsed, coincident, parallel, and
  perpendicular geometries;
- C2 switch and collapse-quality boundary checks;
- SE(3) energy invariance and force equivariance;
- fused-system and symmetric-site saturation, with no ring-size reward bias;
- explicit hydrophobic-overlap trace and a joint-term sign/scale ablation.

The callable diagnostic currently aggregates fused systems with small Python
loops and scalar system counts. This is accepted only for one-pose tracing.
Production ODE admission additionally requires precomputed static
ring-to-system mappings, vectorized segment reduction, and representative
pocket runtime/peak-memory measurements.

### Stage 2B — pi stacking

For ligand/protein ring displacement `Delta`,

```text
d2    = ||Delta||^2
c2    = dot(n_l,n_p)^2
o_l2  = d2 - dot(Delta,n_l)^2
o_p2  = d2 - dot(Delta,n_p)^2

A_parallel = Q5((c2-0.75)/0.25)
A_T        = Q5((0.25-c2)/0.25)
O = 1 - [1-S2(o_l2;1.5,2.0)] * [1-S2(o_p2;1.5,2.0)]
R = S2(d2;4.5,5.5)

w_pi = R * O * (A_parallel+A_T) * q_l * q_p
E_pi = -0.25 kcal/mol * symmetric_saturated_fused_occupancy(w_pi)
```

The term admits face-to-face and edge-to-face geometry without assigning a
signed normal. `0.25 kcal/mol` is an EFF-Dock diagnostic prior, not a binding
energy. The 5.5 A distance, 2 A offset, and approximately 30 degree parallel
geometry are motivated by PLIP (DOI `10.1093/nar/gkv315`); the distinction
between parallel and edge-to-face protein aromatic geometry is informed by
McGaughey et al. (DOI `10.1074/jbc.273.25.15458`).

The term passes crystal admission only if supported crystal contacts receive
finite nonzero weight, decoy translations/rotations lose weight in the
declared directions, hydrophobic overlap is reported without cross-term
suppression, adding the favorable term cannot increase total energy, and all
shared aromatic gates pass.

### Stage 2C — cation-pi

Only complete existing formal-charge sites with charge exactly `+1 e` are
cations. Ring acceptors are neutral carbocyclic systems; protein acceptors are
PHE, TYR, and TRP. HIS and charged/heteroaromatic acceptor systems are excluded
rather than assigned an untracked quadrupole.

For cation-to-ring displacement `Delta`,

```text
d2 = ||Delta||^2
o2 = d2 - dot(Delta,n_ring)^2
w_cation_pi =
  S2(d2;4.5,6.0) *
  S2(o2;1.5,2.0) *
  q_ring

E_cation_pi =
  -0.50 kcal/mol *
  symmetric_saturated_fused_occupancy(w_cation_pi)
```

Both ligand-cation/protein-ring and protein-cation/ligand-ring directions are
evaluated. There is no partial-charge, pKa, or cation strength inference.
`0.50 kcal/mol` is a diagnostic prior. The 6 A geometric precedent follows
PLIP; the protein structural precedent is Gallivan and Dougherty
(DOI `10.1073/pnas.96.17.9459`).

In addition to the shared aromatic gates, admission requires exact exclusion
of charge states other than `+1`, equal behavior under swapping the two
supported directions, charge-site membership conservation, and separate
cation-pi versus screened-charge traces.

### Stage 3 — halogen bond

The supported direction is ligand `C-X` to protein acceptor, where `X` is
neutral, heavy-degree-one Cl, Br, or I with one single bond to neutral carbon.
F, free/charged/multivalent halogen, X bonded to N/O/P/S, receptor halogen
donors, and halogen-pi are excluded. Protein acceptors reuse the strict N/O
missing-valence cones; MET SD is the only admitted sulfur acceptor and uses
tetrahedral capacity four with expected heavy degree two. Plain HIS, CYS SG,
unsupported variants, cofactors, ions, and incomplete geometry fail closed.

For `C-X...A`:

```text
u_sigma = normalize(x_X-x_C)
v       = normalize(x_A-x_X)
c_sigma = dot(u_sigma,v)
s       = ||x_A-x_X|| / [R_vdw(X)+R_vdw(A)]

R       = S(s;1.00,1.15)
G_sigma = cone_gate(c_sigma; target=1, half_width=40 deg)
G_A     = existing_acceptor_cone_gate(half_width=45 deg)

w_XA =
  R * G_sigma * G_A *
  C-X_bond_quality * acceptor_axis_quality

E_halogen =
  -0.50 kcal/mol * sum_X softOR_A(w_XA)
```

Bondi radii are N `1.55`, O `1.52`, S `1.80`, Cl `1.75`, Br `1.85`, and I
`1.98 A`. The sigma-hole gate is zero at 140 degrees and one at 180 degrees.
There is no radial Gaussian or element-specific epsilon; generic physical LJ
owns short-range repulsion.

Admission requires exact ligand/protein typing exclusions, stronger aligned
than bent geometry, finite collapsed-bond and overlap behavior, C2 checks at
140 degrees and normalized distances 1.00/1.15, donor soft-OR saturation, and
SE(3), finite-difference, batch, dtype, device, hash, and trace gates. The
chemical definition follows IUPAC DOI `10.1351/PAC-REC-12-05-10`, biomolecular
scope follows DOI `10.1073/pnas.0407607101`, and the radii follow Bondi DOI
`10.1021/j100785a001`.

### Stage 4 — profile-dispatched metal coordination V1

The receptor parser auto-detects standalone monatomic
`ZN/MG/CA/MN/FE/CO/NI/CU` records, verifies agreement between residue identity
and element, and dispatches each site by element. Multiple spatially
independent sites are supported and summed. A cofactor-bound metal,
multi-metal cluster, shared/bridging donor, identity mismatch, unsupported
element, ambiguous coordinating water, or incompatible shell fails closed
with site provenance; it never falls back to the ZN profile.

The attractive scope remains deliberately strict:

- ZN: tetrahedral target CN4, exactly three fixed receptor donors, one
  ligand-facing vacancy, and ligand N/O/S donor typing;
- MG: octahedral target CN6, exactly five fixed O donors including at least
  one retained crystallographic water, one ligand-facing vacancy, and ligand
  O donor typing.

Both profiles require the complete retained receptor shell and an unambiguous
geometry before any attraction is constructed. The ZN donor/resonance policy
remains coordinate-resolved HIS/HID/HIE N, monodentate ASP/GLU O, and explicit
CYM S; ambiguous plain-HIS, bidentate carboxylate, or unlabelled CYS
protonation fails closed. The MG profile admits only its strict oxygen shell.
This one-vacancy contract prevents an attractive term from inventing missing
coordination geometry.

CA, MN, FE, CO, NI, and CU are recognized but repulsion-only. Their current
protein structures do not determine the coordination geometry and/or
oxidation state strongly enough for a single safe directional attraction.
Each such site emits an explicit `repulsion_only` trace reason instead of
silently using ZN or MG parameters.

For fixed receptor donors `R_j`, metal `M`, and a permitted ligand donor `x_i`:

```text
v = -normalize(sum_j normalize(R_j-M))
A_i = exp(-(1-dot(normalize(x_i-M),v))/tau_theta)

Delta_r_i = ||x_i-M|| - r0(profile, donor_element_i)

E_pair_i =
  S_pair(r_i) * D *
  [exp(-2*a*Delta_r_i) - 2*A_i*exp(-a*Delta_r_i)]

q_i =
  S_CN(r_i) *
  exp[-0.5*(Delta_r_i/sigma_r)^2] *
  A_i

CN = N_fixed + sum_i q_i
E_over = k_over * relu(CN-CN_target)^2
E_slot = k_slot * sum_{i<j} q_i*q_j
E_under = 0
```

`N_fixed/CN_target` are `3/4` for ZN and `5/6` for MG. Exact
element/donor distances, switches, Morse parameters, soft core, and repulsion
radii are frozen in `interaction_v1.json` version `1.6.0`. Every dispatched
metal also applies a bounded short-range repulsion to ligand atoms not
assigned an attractive donor pair; CA/MN/FE/CO/NI/CU evaluate only this branch.

Dispatched metals are excluded from generic protein LJ attraction,
hydrophobic, hydrogen-bond, formal-charge, and metal-Coulomb masks. RDKit
performs static ligand donor typing once; all coordinate-dependent geometry,
profile energies, occupancies, masking, traces, and gradients are evaluated in
Torch. No external metal, force-field, docking, minimization, or simulation
engine is a runtime dependency.

AutoDock4Zn (DOI `10.1021/ci500209e`) supplies the published ZN N/O/S distance
and tetrahedral precedent. MetalPDB (DOI
`10.1107/S2059798324003152`) and MESPEUS (DOI
`10.1093/nar/gkad1009`) motivate explicit metal/donor/coordination profiles
and fail-closed geometry checks. The Morse/occupancy/penalty and bounded
repulsion equations and coefficients remain frozen EFF-Dock diagnostic priors,
not copied docking-score or binding-energy weights.

Admission requires structured parser/profile/site failures; exact
aligned-minimum and directional-sign checks for ZN/MG; finite overlap and
cutoff derivatives for all profiles; donor-slot and over-coordination
behavior; repulsion-only trace assertions; independent multi-site summation;
generic-term masking; SE(3), autograd, finite-difference, batch, dtype/device,
hash, and trace gates; and a supported crystal inventory frozen before
inspecting energy or RMSD.

### Trace-only polar-unsatisfied proxy

`polar_unsatisfied_proxy` is deliberately not an `InteractionEnergy` term:

```text
B_i = softOR_j S(distance_to_protein_heavy;3.0,4.5)
H_i = softOR(valid_hydrogen_bond_weights,
             admitted_metal_occupancy)
u_i = B_i * (1-H_i)
```

It is dimensionless trace metadata only and contributes no energy, force,
ranking score, confidence feature, or sampler correction. Buried-unsatisfied
polarity is a solvation/desolvation statement, but the current system lacks a
complete differentiable solvent-exposure model, water-mediated satisfaction,
and a fully admitted interaction inventory. Turning this proxy into energy now
could falsely penalize metal-bound or solvent-exposed groups and create a
gradient that ejects a ligand from the pocket.

Promotion requires a separately frozen burial/solvent and water policy,
complete satisfying-interaction inventory, numerical admission, and an
internal held-out one-variable ablation. It is not part of the later-term
sampler queue before those prerequisites exist.

## Later-term execution order

Run independent crystal admission and internal ablation in this order:

1. `pi_stacking`
2. `cation_pi`
3. `halogen_bond`
4. `metal_coordination`

No term is first tested in combination with another newly admitted term.
`polar_unsatisfied_proxy` remains trace-only and is not a fifth energy
experiment.
