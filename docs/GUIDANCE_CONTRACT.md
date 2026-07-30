# EFF-Dock Guidance Contract

## Status

- Protocol: `EFFDOCK-GUIDANCE-DIAGNOSTIC-V4`.
- Implemented: one self-contained
  `GuidanceEnergy = PhysicalEnergy + InteractionEnergy` Torch diagnostic,
  fragment-force projection, crystal perturbation tracing, and
  saved-trajectory tracing. Hydrophobic contact, directional heavy-atom
  hydrogen-bond, and screened formal-charge-group terms are active.
- Not admitted: production ODE coupling and metal coordination.
- Production gates: independent golden fixtures, broader chemistry coverage,
  an operator-split trust-region corrector, and PLINDER-only validation.

The diagnostic CLI remains `eff-dock physical trace` for command
compatibility, but V4 evaluates both physical and interaction layers plus their
combined force. The code lives in the unified `effdock.guidance` package.

## Flat code layout and ownership

Guidance stays in one flat package rather than a hierarchy of small
subpackages:

```text
src/effdock/guidance/
├── physical.py          # scalar physical energies
├── interaction.py       # typed motif and formal-charge interaction energy
├── topology.py          # ligand cut-interface topology and reference targets
├── parameterization.py  # versioned in-repository parameter loading
├── system.py            # receptor shell and tensor system
├── runtime.py           # unified energy and fragment SE(3) projection
├── diagnostics.py       # crystal/trajectory traces
├── errors.py            # structured unsupported-chemistry failures
└── parameters/
    ├── effff_v2.json
    └── interaction_v1.json
```

The one guidance energy has two auditable components:

- `PhysicalEnergy`: generic coordinate energies—ligand cut-interface
  geometry, ligand interfragment sterics/dispersion, and generic
  protein-ligand nonbonded terms.
- `InteractionEnergy`: typed hydrophobic contact, directional heavy-atom
  hydrogen-bond, and screened formal-charge-group terms today; later motifs
  require independent admission.

There is no Vina or `HybridGuidance` component in this contract. Legacy Vina
code and historical reports are preserved outside the active path, but their
equations, typing, coefficients, scores, and gradients must not enter
`GuidanceEnergy`.

## Self-contained runtime rule

The active guidance path may use PyTorch tensor operations and autograd,
versioned constants in this repository, and existing parsing that exposes the
input molecular graph and chemistry.

It may not call or import an external force-field, docking, minimization, or
molecular-simulation engine. This excludes OpenMM, AmberTools, OpenFF force
evaluation, RDKit MMFF/UFF optimization, AutoDock Vina, executables,
subprocesses, web services, and silent external fallbacks.

Published equations and constants may be copied into reviewed in-repository
tables with provenance. External software may be used only offline to produce
immutable validation fixtures that record the program, version, command,
inputs, preparation, units, parameter set, expected values, and license.

Unsupported chemistry fails with a structured reason. A narrower
`geometry-only` mode must be named and reported separately; missing parameters
are never silently zero-filled.

## Physical layer: EFF-FF-v2 diagnostic

The current parameter profile is `EFF-FF-v2-diagnostic`, formula version
`effff-diagnostic-2`. It is not AMBER, GAFF, OpenFF, UFF, or MMFF compatible.
UFF supplies only the cited element-level Lennard-Jones form and constants.
Bonded coefficients and typing rules are EFF-Dock diagnostic choices.

### Why only the cut interface is evaluated

EFF-Dock moves each fragment as a rigid SE(3) body. Bond lengths, angles, and
impropers whose atoms all lie in one fragment therefore cannot change during
the ODE. Evaluating those terms would add constants but no corrective force.

The physical layer evaluates ligand-internal terms that can change when two
fragments move relative to each other:

- covalent bonds cut by fragmentation;
- covalent angles spanning fragments;
- proper torsions around cut bonds;
- chiral or planar impropers spanning fragments;
- nonbonded pairs in different fragments.

Ring bonds, amide-like nonrotatable bonds, and stereodefined double bonds are
not cut by the existing fragmentation policy.

### Receptor admission

Receptor chemistry is admitted by normalized residue identity, never by the
PDB `ATOM` versus `HETATM` record label. Canonical amino acids and explicitly
mapped amino-acid variants may enter the generic element-level physical shell;
unsupported mapped variants fail closed for interaction masks as described
below. Any unmapped cofactor, ion, solvent, carbohydrate, or other nonprotein
residue inside the active shell raises the structured
`active_nonprotein_residue` failure regardless of record type. Outside-shell
nonprotein records are excluded and listed in trace provenance.

This rule prevents depositor formatting from silently turning an active
cofactor such as SAH into generic protein LJ. Metals and cofactors require
their own admitted topology, typing, masking, and parameters; they are not
silently stripped from a full-complex diagnostic.

### Inference-safe reference geometry

For a sanitized ligand input conformer with coordinates `x_ref`,

```text
r_ref     = distance in x_ref for a cut bond
theta_ref = angle in x_ref for a cross-fragment angle
```

The reference is part of the inference input. It is never a hidden target or
benchmark crystal pose. When `physical trace` is deliberately run on a crystal
ligand, that crystal file is the diagnostic input and its reference geometry
must not be reused for sampling, selection, or tuning. Every trace stores
`topology_reference_sha256`.

Bond and angle energies are

```text
E_bond  = sum_b 0.5 * k_b * (r_b - r_ref,b)^2
E_angle = sum_a 0.5 * k_a * (theta_a - theta_ref,a)^2
```

This preserves the supplied local geometry without locking the input conformer
around a rotatable bond.

### Torsion and improper terms

For cut bond `b`, let `M_b` be the number of valid substituent quadruplets.
Each quadruplet receives weight `1/M_b`:

```text
E_proper =
  sum_b sum_{m=1..M_b}
    (k_b / M_b) * [1 + cos(n_b * phi_bm - delta_b)]
```

Normalization prevents a branched bond from receiving a larger barrier merely
because it yields more equivalent quadruplets. The proper term uses a generic
periodic prior; it does not reference the input torsion and therefore does not
freeze one starting conformer.

Cross-fragment chiral and planar impropers are

```text
E_chiral_improper = 0.5 * k_chiral * wrap(phi - phi_ref)^2
E_planar_improper = 0.5 * k_planar * [1 - cos(2 * phi)]
```

The chiral reference comes from declared input stereochemistry and the
sanitized conformer.

### Nonbonded terms

For an admitted pair,

```text
r_eff = sqrt(r^2 + alpha^2)
x_ij  = sqrt(x_i * x_j)
D_ij  = sqrt(D_i * D_j)

E_LJ,ij = S(r) * D_ij * [(x_ij/r_eff)^12 - 2*(x_ij/r_eff)^6]
```

`S(r)` is a quintic switch from 1 at 6 Å to 0 at 8 Å. Ligand 1-2 and
1-3 pairs are excluded, 1-4 pairs use scale 0.5, and only pairs in different
fragments are evaluated. Protein-ligand LJ is evaluated against a fixed
receptor shell. Repulsive and attractive components are traced separately.

The active physical components are:

```text
ligand_intra_bond
ligand_intra_angle
ligand_intra_proper
ligand_intra_improper
ligand_intra_lj_repulsive
ligand_intra_lj_attractive
protein_ligand_lj_repulsive
protein_ligand_lj_attractive
```

Partial-charge electrostatics, solvation, receptor flexibility, covalent
docking, and metal-specific coordination are inactive and absent from the
energy sum. The separately declared screened formal-charge-group proxy belongs
to `InteractionEnergy`.

## Interaction layer

The parameter profile is `EFF-Interaction-v1-diagnostic`, formula version
`effdock-interaction-diagnostic-4`. It activates:

```text
interaction_hydrophobic
interaction_hydrogen_bond
interaction_screened_formal_charge
```

It is a differentiable pose-guidance diagnostic, not an affinity, desolvation,
or binding-free-energy model.

### Strict typing boundary

Ligand donor, acceptor, and carbon-hydrophobe masks come from versioned
single-atom SMARTS in `interaction_v1.json`. Delocalized ligand formal-charge
groups use versioned multi-atom SMARTS. RDKit evaluates those patterns once on
the sanitized input graph; it is not called during energy/force evaluation.
Compilation, overlapping groups, charge non-conservation, or missing active
typing fails explicitly.

Every nonempty charge topology is validated again at construction: site,
charge, and label counts must agree; memberships must have shape `[S,N]`,
finite nonnegative nonoverlapping weights that sum to one; and each site must
carry a nonzero integer formal charge. A legacy/manual topology may use the
explicit `(0,0)` no-charge sentinel only when site charges and labels are also
empty.

Protein masks use standard amino-acid residue/atom tables and canonical
heavy-atom bonds. PDB bond orders are not guessed. Explicit `HID`, `HIE`, and
`HIP` names retain their declared histidine state; plain `HIS` side-chain
nitrogens are excluded and counted as ambiguous. Other explicitly named mapped
states/PTMs, including `ASH`, `GLH`, `CYM`, `CYX`, and `SEP`, fail closed for
all active interaction masks and are listed in the trace. V1 hydrogen bonds
support N/O sites. V1 hydrophobes are carbon-only; sulfur and halogens are
excluded. Ligand hydrophobes use a graph SMARTS exclusion around N/O/F,
whereas protein hydrophobes use a curated canonical residue/atom-name table.
That representation asymmetry is intentional and is not claimed to be a
shared atom-type ontology.

Because explicit hydrogens are removed, the code does not pretend that the
opposite heavy-neighbor direction is a hydrogen or lone-pair coordinate.
Instead, it builds a normalized outward axis from normalized heavy-bond
directions. Each bond first receives a C2 length-quality ramp so a collapsed
bond cannot be hidden by another valid neighbor:

```text
bond_quality(l) = Q5(clamp((l - 0.10 A) / (0.50 A - 0.10 A), 0, 1))
v_i = normalize(sum_{k in bonded heavy neighbors(i)}
                bond_quality(||x_i-x_k||) * normalize(x_i - x_k))
```

RDKit hybridization plus heavy degree chooses an idealized missing-valence cone
for ligand sites. Canonical protein N/O sites use declared trigonal or
tetrahedral geometry and an exact expected heavy degree. A free N terminus,
chain gap, missing required neighbor, or degree mismatch fails closed rather
than being reinterpreted as a different cone. The target cosine between `v_i`
and any missing site is:

```text
linear:      degree 1 -> 1
trigonal:    degree 1 -> 1/2, degree 2 -> 1
tetrahedral: degree 1 -> 1/3, degree 2 -> 1/sqrt(3), degree 3 -> 1
```

A missing, unsupported, or degenerate geometry excludes that site. This is an
idealized heavy-atom cone proxy, not explicit D-H-A or lone-pair geometry.
For ODE stability, the dimensionless weighted-axis norm `rho` receives a C2
quality factor:

```text
axis_quality(rho) = Q5(clamp((rho - 0.10) / (0.40 - 0.10), 0, 1))
site_quality = product_k bond_quality(||x_i-x_k||) * axis_quality(rho)
```

Both donor and acceptor site quality multiply the pair weight. Thus a collapsed
bond or nearly cancelling neighbor axis contributes exactly zero and reaches
full weight only after its declared ramp, without an unstable unit-vector
gradient entering the active energy.

### Smooth contact terms

For `u` in `[0,1]`, define:

```text
Q5(u) = 6*u^5 - 15*u^4 + 10*u^3
```

The decreasing switch `S(r;a,b)` equals 1 for `r <= a`,
`1-Q5((r-a)/(b-a))` for `a < r < b`, and 0 for `r >= b`. Its value, first
derivative, and second derivative are continuous at both boundaries.

For ligand/protein carbon hydrophobes:

```text
c_ij = S(r_ij; 3.5 A, 4.5 A)
```

For donor `D` and acceptor `A`:

```text
u_DA = normalize(x_A - x_D)
R(r) = exp[-0.5*((r-2.9 A)/0.35 A)^2] * S(r; 3.5 A, 4.1 A)

C(c; t, Delta) = two-sided Q5 rise/fall centered at target cosine t,
                  compactly zero outside angles theta(t) +/- Delta

G_D = C(dot(v_D, u_DA);  t_D, 60 deg)
G_A = C(dot(v_A, -u_DA); t_A, 45 deg)

h_DA = R(r) * G_D * G_A * site_quality_D * site_quality_A
```

The two-sided gate is C2 at its lower bound, target, and upper bound. The broad
donor cone covers idealized donor directions up to a 60-degree deviation; the
acceptor cone is narrower. Both widths are diagnostic priors frozen after
formula review and before the final Astex traces, not values fitted on those
crystals.

Raw pair sums over-reward dense receptor atom sets. Each ligand hydrophobe, or
each hydrogen-bond donor, therefore receives a soft-OR occupancy:

```text
q_i = 1 - product_j[1 - (1-delta)*w_ij], delta = 1e-7
E_hydrophobic = -0.25 kcal/mol * sum_i q_i
E_H-bond      = -1.00 kcal/mol * sum_D q_D
```

Every pair weight is clamped to `[0,1]` before the soft-OR so float32
round-off cannot leave the `log1p` probability domain.

The radial, contact, and epsilon coefficients were fixed before the initial
Astex traces. An independent formula audit then rejected the original
opposite-axis-only angular proxy; the cone formula and widths were fixed before
replacement traces were run. A later float32/stability gate added the
probability clamp and bond-quality ramp before the final frozen traces. These
changes were based on formula and numerical-domain review, not replacement
energies. No active value was fitted to crystal energy, RMSD, PoseBusters,
Astex, affinity, or Vina. Rejected provisional formulas are not admitted
results.
PLIP's published geometry motivated the 4.0 Å hydrophobic center criterion and
4.1 Å permissive H-bond outer cutoff
([Salentin et al., 2015](https://doi.org/10.1093/nar/gkv315)). RDKit's
SMARTS/FDef model motivates the local ligand typing representation
([RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html)).

Generic protein-ligand LJ remains physical: its attraction represents a
UFF-style dispersion term, while hydrophobic contact is a small
solvent-mediated pose-guidance surrogate. They are not claimed to be an
orthogonal force-field decomposition. Physical-only, each interaction term,
and their combination must therefore be ablated before production admission.
The screened formal-charge term replaces the planned generic salt-bridge term;
activating both would double count the same ionic motif. Future pi, halogen, or
metal terms require separate typing, double-count masking, parameters, tests,
and provenance.

### Screened formal-charge groups

This term uses declared group charges rather than treating every pharmacophore
flag or resonance-equivalent atom as a full ion. Ligand group coordinates are
weighted sums of current ligand coordinates. Protein group coordinates are
weighted sums over fixed canonical residue atoms. ARG, LYS, ASP, GLU, and
explicit HIP are the admitted receptor-side states; plain HIS, termini,
incomplete groups, mapped variants/PTMs, cofactors, metals, and counterions do
not receive inferred charge.

For group charges `Q_a`, `Q_b` and group centers `R_a`, `R_b`:

```text
d2  = ||R_a - R_b||^2
rho = sqrt(d2 + 0.75^2)
u   = clamp((10.0^2 - d2) / (10.0^2 - 8.0^2), 0, 1)

E_ab =
  (332.06371 / 78.5)
  * Q_a * Q_b
  * exp(-0.127 * rho)
  * Q5(u)
  / rho
```

Units are kcal/mol, Å, and elementary charge. Opposite signs are attractive
and equal signs repulsive. Unlike contact rewards, signed charge pairs are
summed directly and never passed through the soft-OR saturation.

The `0.127 Å^-1` screening constant is the rounded Debye–Hückel value for
0.15 M monovalent ionic strength, 298.15 K, and bulk-water
`epsilon_r=78.5`. Lee, Fitch, and García-Moreno
([DOI 10.1110/ps.4700102](https://doi.org/10.1110/ps.4700102)) provide the
screened pairwise precedent and Debye-parameter relation; the unit-converted
Coulomb prefactor is based on NIST 2022 CODATA constants. The squared-distance
C2 cutoff and soft core are numerical regularizers.

This is a conservative ionic pose-guidance proxy. It does not infer pKa or
protonation, and it omits partial charges, dielectric heterogeneity,
desolvation, polarization, receptor relaxation, and solvent/counterion
reorganization. Its physical admission and its separate sampler-activation
gate are frozen in `docs/INTERACTION_GUIDANCE_STUDY.md`.

## Proposed metal-coordination V0

Metal coordination is not treated as an isotropic Coulomb or LJ attraction.
The first admissible scope is deliberately narrow:

- Zn(II), mononuclear, approximately tetrahedral site;
- exactly three fixed receptor donor atoms;
- at most one ligand donor atom, typed N/O/S;
- explicit protonation/resonance assignment;
- no bridging donor or unresolved coordination water.

For receptor donor unit vectors `u_j` from Zn to donor, the vacant direction is

```text
v = -normalize(sum_j u_j)
```

For a ligand donor at distance `r` and unit direction `u`,

```text
A(u,v) = exp[-(1 - dot(u,v)) / tau_theta]
```

The proposed directional Morse pair energy is

```text
Delta r = r - r0
E_pair =
  S(r) * D * {exp[-2*a*Delta r] - 2*A(u,v)*exp[-a*Delta r]}
```

The repulsive radial and directional attractive pieces are traced separately.
An occupancy proxy controls over-coordination:

```text
q_i = S_CN(r_i)
      * exp[-0.5*((r_i-r0)/sigma_r)^2]
      * A(u_i,v)

CN = N_fixed + sum_i q_i

E_CN =
  k_over  * relu(CN-CN_max)^2
  + k_under * relu(CN_target-CN)^2

E_slot = k_slot * sum_{i<j} q_i*q_j
```

V0 sets `CN_max = CN_target = 4` and `k_under = 0`; the model should not pull a
ligand into an otherwise unsupported site merely to complete coordination.
When this term is admitted, an active Zn-donor pair replaces generic
protein-ligand LJ for that pair. Zn/non-donor pairs retain only short-range
repulsion, and metal Coulomb remains off in V0 to avoid double counting.

The formula is frozen as a contract, but `r0`, `D`, `a`, `tau_theta`,
`sigma_r`, switches, penalties, donor typing, and site detection are not yet
parameterized. Therefore metal coordination contributes no energy today.
Unsupported metals, oxidation states, coordination numbers/geometries,
binuclear sites, bridging donors, ambiguous waters, or donor protonation fail
explicitly rather than falling back to Zn V0.

The design is informed by AutoDock4Zn's zinc-specific energetic/geometric
docking precedent ([Santos-Martins et al., 2014](https://doi.org/10.1021/ci500209e))
and MetalPDB's curated coordination-site definitions
([Putignano et al., 2024](https://doi.org/10.1107/S2059798324003152)).
The EFF-Dock directional Morse/CN expression above is a proposed in-repository
form, not a claim that either reference used this exact equation.

## Force projection and future ODE coupling

For unified guidance energy `E`,

```text
F_atom = -dE/dx
F_fragment = sum_i F_i
tau_fragment = sum_i (x_i - T_fragment) cross F_i
```

Translation uses mass preconditioning and rotation uses the full fragment
inertia pseudo-inverse. Rank-deficient fragments are projected onto observable
rotational axes. These are corrections to a dimensionless generative ODE, not
physical accelerations or molecular dynamics.

Production coupling must use an operator-split trust-region corrector:

1. propose one learned ODE step;
2. apply bounded guidance substeps;
3. verify finite terms/forces and declared descent;
4. shrink or reject a correction that violates its trust region.

Required controls include soft-core annealing, a time ramp, per-term finite
checks, equivariant caps, and maximum translation, rotation, and atom
displacement.

## Diagnostics, leakage boundary, and admission

`eff-dock physical trace` records exact input, combined parameter,
implementation, physical-topology and interaction-typing hashes; pinned
RDKit/Torch runtime versions; coordinate frame; typed/candidate/occupied
counts; physical, interaction and combined component energies; per-term force
norms; fragment corrections; contact statistics; strongest interaction-pair
atom/group identities, distance, radial/cone gates, site quality, nearby radial
candidates rejected by angle/quality gates, every formal-charge site and its
member weights, signed attractive/repulsive charge sums, and structured
unsupported chemistry. The implementation identity includes ligand loading,
fragmentation, and protein typing source, not only the energy kernel. It does
not optimize coordinates.

Saved `results.pt` tracing is strict. Docking-results V2 stores an
order-sensitive ligand identity containing atom attributes, indexed bond
graph/order/stereo, canonical isomeric SMILES, fragment IDs, graph hash, and
exact ligand-source hash. A missing or mismatched identity fails before any
saved coordinate is evaluated; shape equality alone is not accepted.

Crystal traces are diagnostics only. Astex/PoseBusters fixed-coordinate
characterization may report coverage, signs, forces, pose margins, and ranking
changes, but cannot select formulas, constants, scales, schedules, force caps,
term retention, or sampler activation. Guidance settings and production
activation are selected on internal train/validation data (currently
PLINDER), with the external benchmarks kept frozen.

Before sampler coupling, the implementation must retain:

- force/gradient sign and finite-difference checks;
- SE(3) energy invariance and force/projection equivariance;
- topology exclusion and 1-4 scaling tests;
- chirality, planarity, periodicity, cutoff, coincident/collinear, and
  unsupported-chemistry tests;
- immutable independent golden energy/force fixtures;
- PLINDER validity/RMSD guard evaluation;
- operator-split descent/rejection tests.

Frozen pre-formal-charge V3 diagnostic outputs and verification evidence are
recorded in `docs/GUIDANCE_DIAGNOSTIC_RESULTS.json`. The report-only V4
screened-charge characterization is recorded in
`docs/GUIDANCE_FORMAL_CHARGE_BENCHMARK_CHARACTERIZATION_V4.json`. Full
generated traces remain under the ignored `outputs/guidance/` directory.
