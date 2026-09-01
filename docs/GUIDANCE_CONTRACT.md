# EFF-Dock Guidance Contract

## Status

- Protocol: `EFFDOCK-GUIDANCE-DIAGNOSTIC-V7`.
- Implemented: one self-contained
  `GuidanceEnergy = PhysicalEnergy + InteractionEnergy` Torch diagnostic,
  fragment-force projection, crystal perturbation tracing, and
  saved-trajectory tracing. By explicit user request, all seven implemented
  interaction terms—hydrophobic contact, directional heavy-atom hydrogen bond,
  screened formal-charge groups, pi stacking, cation-pi, ligand-to-protein
  halogen bond, and profile-dispatched metal coordination—are enabled by
  default as separately traceable diagnostics.
  The default physical profile also includes the compact, vdW-radius
  `protein_ligand_steric_barrier` as a separately traceable diagnostic guard.
  `polar_unsatisfied_proxy` is implemented as dimensionless trace metadata
  only and contributes no energy or force.
- Experimental sampler couplings: the guarded operator-split corrector and a
  normalized direct ODE drift. Both are explicit evaluation-only modes and
  share the exact same GuidanceEnergy and physical-system provenance.
- Not admitted: production ODE coupling, interaction-energy candidate-selector
  use, or sampler
  activation of the four later interaction terms. User-authorized diagnostic
  default-on is not evidence of production validity.
- Production gates: independent golden fixtures, broader chemistry coverage,
  an operator-split trust-region corrector, internal held-out validation, and
  frozen report-only Astex/PoseBusters characterization.

The diagnostic CLI remains `eff-dock physical trace` for command
compatibility, but V7 evaluates both physical and interaction layers plus their
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
  hydrogen bond, screened formal-charge groups, pi stacking, cation-pi,
  halogen bond, and profile-dispatched metal coordination. All seven are
  user-requested default-on diagnostics, but the latter four remain
  scientifically and sampler-unadmitted. The polar-unsatisfied proxy belongs
  to diagnostics, not this energy sum.

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

Unsupported chemistry fails with a structured reason in the default
`fail_closed` receptor policy. The explicit `geometry_only` coverage policy
is a separate estimand: it preserves resolved terms, converts unresolved
fixed receptor atoms/sites only to declared repulsion-only geometry guards,
and records the downgrade reason. Missing attraction, charge, donor, or bond
typing is never silently zero-filled or inferred from a PDB record label.

## Physical layer: EFF-FF-v2 diagnostic

The current parameter profile is `EFF-FF-v2-diagnostic` version `2.2.0`, formula
version `effff-diagnostic-2.2`. It is not AMBER, GAFF, OpenFF, UFF, or MMFF compatible.
UFF supplies only the cited element-level Lennard-Jones form and constants.
Bonded coefficients and typing rules are EFF-Dock diagnostic choices.

The version-2.2 table covers all 33 elements observed across the frozen S50
confidence train/validation ligand cohort. The original main-group steric
radii remain Bondi/Mantina values; newly admitted cohort elements use copied,
versioned RDKit periodic-table vdW radii. These additions provide finite
diagnostic repulsion and steric terms, not validated transition-metal
coordination chemistry.

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
below. Standalone monatomic ZN/MG/CA/MN/FE/CO/NI/CU records are routed to the
metal profile dispatcher rather than generic protein LJ.

The default `fail_closed` policy raises a structured failure for an active
cofactor, metal cluster, identity-mismatched ion, solvent additive,
carbohydrate, or other unsupported nonprotein residue. The explicit
`geometry_only` policy keeps supported nonmetal heavy atoms as fixed
UFF-style repulsion-only obstacles and uses a bounded generic steric guard for
an element without an admitted element row. These atoms do not enter typed
hydrogen-bond, formal-charge, pi, hydrophobic, halogen, metal-attraction, or
generic LJ-attraction masks. Every admitted/fallback/excluded count and reason
is included in receptor provenance. Water and nucleic-acid filtering remains
an explicit input-preparation policy rather than an inferred chemistry term.

This rule prevents depositor formatting from silently turning an active
cofactor such as SAH into a fully typed protein site. Supported monatomic
metals use their own topology, typing, masking, and parameter profiles.
Cofactors and unsupported metals are neither silently stripped nor granted
unvalidated attraction in a full-cohort diagnostic.

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

The independent protein-ligand steric guard uses versioned in-repository vdW
radii:

```text
d_safe,ij = lambda * (R_vdw,i + R_vdw,j)
h_ij      = tau * softplus((d_safe,ij - r_ij) / tau)
C_ij      = compact quintic switch from 1 at d_safe,ij
            to 0 at d_safe,ij + margin

E_steric,ij = 0.5 * k_steric * h_ij^2 * C_ij
```

The diagnostic defaults are `lambda=0.8`, `tau=0.1 Å`,
`margin=0.5 Å`, and `k_steric=20 kcal mol^-1 Å^-2`. They are initial internal
hypotheses, not PoseBusters-derived fitted values. They must be calibrated and
frozen using PLINDER train/validation before production admission. The compact
switch makes every sufficiently distant pair exactly zero and prevents the
protein-shell size from accumulating a softplus tail. The term is a clash
guard, not a binding affinity or free-energy contribution.

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
protein_ligand_steric_barrier
receptor_geometry_obstacle_uff_repulsive      # geometry_only, typed obstacle atoms
receptor_geometry_obstacle_generic_repulsive  # geometry_only, unparameterized atoms
```

Partial-charge electrostatics, solvation, receptor flexibility, and covalent
docking are inactive and absent from the energy sum. Screened formal-charge
groups and profile-dispatched metal handling belong to `InteractionEnergy`.

## Interaction layer

The parameter profile is `EFF-Interaction-v1-diagnostic` version `1.6.0`,
formula version `effdock-interaction-diagnostic-7`. The combined profile is
version `1.6.0`, formula
`physical-v2.2_plus_interaction-v1.6`. By explicit user request, its default
active terms are all seven implemented interaction energies:

```text
interaction_hydrophobic
interaction_hydrogen_bond
interaction_screened_formal_charge
interaction_pi_stacking
interaction_cation_pi
interaction_halogen_bond
interaction_metal_coordination
```

`polar_unsatisfied_proxy` is a dimensionless contact diagnostic only. It is
not an `InteractionEnergy` component and cannot contribute energy, force,
ranking, confidence, or sampler correction.

These are differentiable pose-guidance diagnostics, not affinity,
desolvation, or binding-free-energy models. Their user-requested default
activation does not admit them to candidate selection or the production ODE
sampler.

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
nitrogens are excluded from hydrogen-bond and charge typing and counted as
ambiguous. A complete neutral HIS ring may still enter the independently typed
pi-stacking term. Other explicitly named mapped
states/PTMs fail closed for the relevant interaction masks and are listed in
the trace. The strict attractive Zn profile separately admits explicit
`CYM:SG`; that exception does not admit CYM to hydrogen-bond, charge, or
hydrophobic typing.
V1 hydrogen bonds support N/O sites. V1 hydrophobes are carbon-only; sulfur
and halogens are excluded from those two masks. Halogen and attractive-metal
profiles use their own narrower, versioned S/Cl/Br/I or N/O/S typing. Ligand
hydrophobes use a
graph SMARTS exclusion around N/O/F,
whereas protein hydrophobes use a curated canonical residue/atom-name table.
That representation asymmetry is intentional and is not claimed to be a
shared atom-type ontology.

RDKit performs this static ligand typing once. Every coordinate-dependent
distance, direction, switch, energy, occupancy, and gradient—including metal
profile evaluation—is computed with Torch and autograd without an external
force-field or docking engine.

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

Here `u` is a dimensionless normalized position inside a switching interval;
it is not an ODE time, an energy, or a molecular coordinate. For example, a
decreasing distance switch over `[a,b]` may use
`u=clamp((b-r)/(b-a),0,1)`: `u=1` at full strength, `u=0` at the cutoff, and
intermediate values only while `a<r<b`. The switch acts as a
coordinate-dependent multiplier,

```text
E_switched(x) = Q5(u(x)) * E_raw(x),
```

so reaching the cutoff turns that energy contribution exactly off; it does not
mean that the coordinates themselves converge to zero. Differentiating the
product also produces a force from the changing switch:

```text
-grad(E_switched) = -Q5(u)*grad(E_raw) - E_raw*grad(Q5(u)).
```

The fifth degree is the minimum polynomial degree that gives a compact `C2`
join to the constant regions on both sides. The desired endpoint constraints
are

```text
Q(0)=0, Q(1)=1,
Q'(0)=Q'(1)=0,
Q''(0)=Q''(1)=0.
```

These are six independent constraints, so a polynomial needs at least six
coefficients and therefore degree five. To see where every term and coefficient
comes from, begin with

```text
Q(u) = a0 + a1*u + a2*u^2 + a3*u^3 + a4*u^4 + a5*u^5.
```

The three conditions at the zero endpoint give

```text
Q(0)   = a0   = 0,
Q'(0)  = a1   = 0,
Q''(0) = 2*a2 = 0.
```

Thus the constant term would leave a nonzero boundary value, the linear term
would leave a nonzero boundary slope and therefore a force jump, and the
quadratic term would leave the constant second derivative `2*a2`. A `C2` join
to the zero-valued constant region forces all three coefficients to vanish; it
is not an arbitrary omission of the lower-order terms. The first power that
can remain is consequently `u^3`:

```text
Q(u) = a3*u^3 + a4*u^4 + a5*u^5.
```

Applying the value, slope, and curvature conditions at `u=1` leaves the linear
system

```text
a3 + a4 + a5       = 1,
3*a3 + 4*a4 + 5*a5 = 0,
6*a3 + 12*a4 + 20*a5 = 0,
```

whose unique solution is `a3=10`, `a4=-15`, and `a5=6`. Therefore

```text
Q5(u) = 10*u^3 - 15*u^4 + 6*u^5.
```

Equivalently, a decreasing ramp written with progress
`t=(r-a)/(b-a)` is

```text
1 - Q5(t) = 1 - 10*t^3 + 15*t^4 - 6*t^5.
```

Its constant term is one because the decreasing switch begins at full
strength; its linear and quadratic terms still vanish for the same slope and
curvature conditions. More generally, making the value and derivatives through
order `m` vanish at an endpoint means the first possible nonzero power is
`u^(m+1)`.

The derivative is `Q5'(u)=30*u^2*(1-u)^2`; `Q5'` and `Q5''` both vanish at
both endpoints. A linear ramp would introduce a force
discontinuity, while the cubic smoothstep removes that discontinuity but still
has a second-derivative jump. Fifth degree is therefore the smallest choice
that keeps energy, force, and the local force derivative continuous for the
autograd/ODE guidance path.
Higher-degree switches could make still higher derivatives continuous, but
that extra smoothness is not required by the current contract and would change
the transition profile without an established benefit. `Q5` is only the
switching/gating function: `GuidanceEnergy` itself is not a fifth-order energy
expansion.

The decreasing switch `S(r;a,b)` equals 1 for `r <= a`,
`1-Q5((r-a)/(b-a))` for `a < r < b`, and 0 for `r >= b`. Its value, first
derivative, and second derivative are continuous at both boundaries.

#### Why the compact quintic is the default

The quintic is a numerical design choice for the current force-guided ODE, not
a claim that fifth-degree polynomials are intrinsically more physical. It is
the smallest function in this family that simultaneously provides:

- exact values of one and zero outside the declared switching interval;
- exact compact support, so thousands of distant atom pairs cannot accumulate
  individually tiny energy or force tails;
- `C2` joins, making energy, force, and the local force derivative continuous;
- a transition width controlled directly by the two physical-distance or
  geometry thresholds;
- inexpensive, branch-light Torch evaluation using only clamping,
  multiplication, and addition; and
- stable float32 behavior without exponential underflow, logarithmic
  singularities, or an arbitrary numerical tail threshold.

Common alternatives are valid for other objectives but have different failure
modes here:

- A hard cutoff or linear ramp is compact and cheap, but introduces a jump in
  energy or force at a boundary.
- The cubic smoothstep and cosine switch make the force continuous, but their
  second derivative does not match the surrounding constant region; the local
  force derivative changes abruptly at the boundary.
- Logistic, `tanh`, and `erf` switches are `C-infinity`, but do not become
  exactly zero or one at any finite coordinate. Their small tails can
  accumulate over a large receptor shell. Truncating those tails at a finite
  cutoff reintroduces a nonsmooth boundary unless another compact switch is
  added.
- `softplus` and `logsumexp` are useful smooth approximations to a hinge,
  minimum, or maximum, but likewise retain tails and do not by themselves
  define a compact two-boundary switch. A raw logarithm is unsuitable because
  `log(u)` is singular as `u` approaches zero; adding an epsilon removes the
  singularity but introduces another scale and still does not give exact
  compact support.
- A compact exponential bump can be made `C-infinity`. For
  `phi(t)=0` when `t<=0` and `phi(t)=exp(-1/t)` otherwise, the ratio
  `phi(1-t)/(phi(t)+phi(1-t))` is an exact decreasing compact switch. It is a
  mathematically sound alternative, but its exponential dynamic range can
  underflow in float32, its all-orders-flat endpoints can make useful gradients
  disappear earlier, and a robust implementation needs explicit boundary
  branches or stable log-domain algebra. The current ODE requires only `C2`, so
  that added complexity has no established validation benefit.
- Seventh- or higher-degree polynomials can enforce still higher derivative
  continuity, but add shape freedom or constraints that the current solver
  does not require and that would need separate train/validation admission.

Accordingly, the quintic is the current minimum-complexity compromise between
compactness, force smoothness, numerical stability, and interpretability. If a
future runtime requires higher-order derivatives, or if an evaluation shows a
material boundary artifact under the frozen quintic, compact exponential or
higher-order polynomial switches may be compared as a preregistered PLINDER
train/validation ablation rather than selected on external test results.

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
activating both would double count the same ionic motif. Pi stacking is a
small orientation-specific correction on top of the generic carbon-contact
baseline. Their overlap is traced, but neither term suppresses the other:
atom-site hydrophobic and ring-system pi normalizations are not
interchangeable, and direct suppression can reverse the sign of adding a
favorable pi term. Aromatic, halogen, and metal diagnostics each retain
separate typing, masking, parameters, tests, and provenance. User-requested
default-on does not mean scientifically validated or sampler-admitted.

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

### Default-on diagnostic aromatic and halogen terms

The implemented motif terms are:

- `interaction_pi_stacking`: neutral five/six-member ligand aromatic rings
  against complete PHE/TYR/TRP and neutral HIS/HID/HIE rings, with
  parallel/edge-to-face gates, fused-system saturation, collapse quality, and
  explicit hydrophobic-overlap tracing;
- `interaction_cation_pi`: declared `+1 e` formal-charge groups against the
  narrower neutral carbocyclic ring set, evaluated in both ligand/protein
  directions;
- `interaction_halogen_bond`: neutral ligand carbon-bound Cl/Br/I directed
  toward strict protein N/O or MET-SD acceptors, with Bondi-radius-normalized
  distance and donor/acceptor angular gates.

Their complete Torch equations, constants, typing exclusions, provenance, and
independent admission gates are frozen in
`docs/INTERACTION_GUIDANCE_STUDY.md` and
`src/effdock/guidance/parameters/interaction_v1.json`. They are included in
the user-requested default diagnostic profile while remaining absent from
candidate selection and the production sampler until their crystal and
internal held-out gates pass.

The current fused-ring aggregation is sized for one-pose diagnostics: it uses
small Python loops and scalar system counts. Before any ODE/sampler admission,
ring-to-system mappings must be precomputed in topology and aggregation must
use a vectorized segment reduction with a representative-pocket
runtime/peak-memory gate.

`polar_unsatisfied_proxy` reports a dimensionless
`burial * (1 - supported_satisfaction)` value. It remains trace-only because
the current boundary does not claim complete solvent exposure, water-mediated
satisfaction, or desolvation. It is never included in `GuidanceEnergy` or its
gradient.

## Profile-dispatched metal coordination V1

Metal coordination is not treated as a universal isotropic Coulomb or LJ
attraction. The receptor parser detects standalone monatomic
`ZN/MG/CA/MN/FE/CO/NI/CU` records, verifies that residue and element identity
agree, and dispatches each site to an element-specific profile. More than one
spatially independent site may be represented and the per-site energies are
summed.

Only two profiles currently permit directional ligand attraction:

| element | strict retained site | ligand donor scope | behavior |
| --- | --- | --- | --- |
| ZN | tetrahedral, target CN4, exactly three fixed receptor donors | N/O/S | one-vacancy directional attraction plus repulsion |
| MG | octahedral, target CN6, exactly five fixed O donors including at least one retained crystallographic water | O | one-vacancy directional attraction plus repulsion |
| CA/MN/FE/CO/NI/CU | geometry or oxidation state not safely determined | none | bounded repulsion only, with an explicit trace reason |

The ZN and MG attractive profiles require a complete retained receptor shell
with exactly one ligand-facing vacancy, an unambiguous standalone ion, and no
bridging donor, unresolved coordinating water, nearby metal cluster, or
cofactor ownership. ZN keeps its strict tetrahedral geometry gate and MG keeps
its strict octahedral geometry gate and requires at least one retained
coordination-water oxygen. A ligand can occupy at most one vacancy per site.
This deliberately excludes underdetermined coordination shells
rather than allowing an attractive potential to invent their geometry.

For fixed receptor donor unit vectors `u_j` from metal `M` to donor, the vacant
direction and directional gate are

```text
v = -normalize(sum_j u_j)
A_i = exp[-(1 - dot(normalize(x_i-M),v)) / tau_theta]
```

For an attractive profile, element/donor-specific constants are loaded from
the versioned in-repository parameter table:

```text
Delta r_i = ||x_i-M|| - r0(profile, donor_element_i)

E_pair_i =
  S_pair(r_i) * D *
  {exp[-2*a*Delta r_i] - 2*A_i*exp[-a*Delta r_i]}

q_i =
  S_CN(r_i) *
  exp[-0.5*(Delta r_i/sigma_r)^2] *
  A_i

CN = N_fixed + sum_i q_i
E_CN =
  k_over  * relu(CN-CN_max)^2
  + k_under * relu(CN_target-CN)^2
E_slot = k_slot * sum_{i<j} q_i*q_j
```

The repulsive radial and directional attractive pieces, occupancy, and slot
penalties are traced separately. `N_fixed/CN_target` are `3/4` for ZN and
`5/6` for MG. `k_under = 0`: the energy does not attract a ligand merely to
complete an unsupported or incomplete site.

All eight dispatched metals use a bounded, soft-core short-range repulsion
for ligand atoms not assigned an attractive donor pair. The
CA/MN/FE/CO/NI/CU profiles evaluate only that repulsion and emit a structured
reason such as unresolved coordination geometry or oxidation state; they
never inherit ZN or MG attraction. Dispatched metals are excluded from generic
protein-ligand LJ, hydrophobic, hydrogen-bond, formal-charge, and metal-Coulomb
masks so the metal interaction is not silently double counted.

Metal sites are auto-detected whenever `metal_coordination` is present in the
default diagnostic profile. Under `fail_closed`, a cofactor-bound metal,
multi-metal cluster, shared/bridging donor, residue/element identity mismatch,
unsupported element, ambiguous water, or incompatible donor shell fails with
site provenance. Under `geometry_only`, the same strict resolution is
attempted first; a chemistry-domain failure downgrades that site to bounded
all-ligand repulsion and records the original failure code/message/details.
It cannot enable directional/radial attraction, under-coordination reward, or
a fabricated vacancy. Malformed parameter tables and non-finite coordinates
remain hard implementation failures in both policies.

RDKit is used only once for static ligand donor typing. Site geometry,
profile dispatch, distance/direction gates, energy, occupancy, repulsion,
trace values, and gradients are evaluated with Torch from versioned local
tables; no external metal, force-field, docking, or minimization engine runs
in this path.

The design is informed by AutoDock4Zn's zinc-specific energetic/geometric
docking precedent ([Santos-Martins et al., 2014](https://doi.org/10.1021/ci500209e)),
MetalPDB's coordination-site definitions
([Bazayeva, Andreini & Rosato, 2024](https://doi.org/10.1107/S2059798324003152)), and
high-resolution structural metal-donor distances
([Dokmanić et al., 2008](https://doi.org/10.1107/S090744490706595X)), and
MESPEUS metal/donor/coordination statistics
([MESPEUS 2023 update](https://doi.org/10.1093/nar/gkad1009)). The directional
Morse, occupancy, repulsion, and penalty equations are EFF-Dock diagnostic
forms, not a claim that those references used this equation or these energy
coefficients.

## Force projection and experimental ODE coupling

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

The separately named `normalized_drift` mode is a report-only comparison, not
the production coupling. For pose `b`, fragment `f`, and atom `i` in `f`, it
maps fragment velocities into a common atom-velocity space:

```text
A_i(v, omega) = v_f + omega_f cross (x_i - T_f)
m_b = RMS_i ||A_i(v_model, omega_model)||
g_b = RMS_i ||A_i(dT_force, domega_force)||
c_b = eta * interval_average(ramp) * m_b / (g_b + eps)

v_total     = v_model     + gamma_b * c_b * dT_force
omega_total = omega_model + gamma_b * c_b * domega_force
```

One positive pose scalar multiplies translation and rotation together so the
Newton--Euler direction is not changed by separate channel normalization.
`gamma_b <= 1` jointly enforces the declared guide-only translation, angular,
and conservative atom-displacement bounds after normalization. The ramp is
integrated over each ODE interval rather than evaluated at only its left or
right endpoint, so its total exposure is invariant to the 10/20/25-step grid
for a fixed continuous trajectory. There is no energy-descent acceptance or
backtracking in this pure direct-drift arm; the learned and guidance fields are
summed and passed once to the ordinary SE(3) integrator. The coefficient is a
dimensionless generative-control strength after velocity normalization, not a
physical timestep, temperature, force conversion, or MD quantity.

## Experimental constraint-only FK and translation SDE

Constraint-only Feynman--Kac (FK) steering is an experimental particle
resampling operator. At declared times it evaluates the versioned local
constraint energy, forms incremental log weights, and performs seeded
systematic resampling. It does not add the guidance gradient to the learned
vector field. Every event records effective sample size, normalized weights,
ancestor indices, the number of unique ancestors, energy components, and the
sampling-dynamics contract. FK remains report-only and is not a production
sampler or candidate selector.

For a linear flow-matching translation path with Gaussian prior variance
`sigma_0^2`, the optional score-corrected translation SDE uses

```text
score_t(x_t) = (t * v_theta(x_t, t) - x_t) / ((1 - t) * sigma_0^2)
dx_t = [v_theta(x_t, t) + 0.5 * g(t)^2 * score_t(x_t)] dt + g(t) dW_t
g(t) = g_0 * (1 - t)
```

and Euler--Maruyama translation updates on the same fixed model-evaluation
grid. A dedicated seeded generator isolates Brownian noise from prior-pool
selection and FK-resampling randomness. The retained model supplies no
compatible SO(3) score, so rotations continue through the deterministic
learned SO(3) flow. This path must be described as a translation
score-corrected SDE with deterministic rotations, not as a complete SE(3)
reverse SDE. It is mutually exclusive with the legacy heuristic stochastic
translation flag and with gradient-guidance couplings.

The implementation follows the Gaussian-prior score identity and
score-corrected SDE construction in
[Feynman-Kac-Flow](https://arxiv.org/abs/2509.01543); the repository records
the exact schedule, seed, discretization, and implementation provenance.
The frozen external Astex comparison is specified in
`docs/FK_SDE_ASTEX_PROTOCOL.md`. Astex is descriptive only: its outcomes may
not select the formula, diffusion scale, FK strength/times, particle budget,
or production admission.

## Diagnostics, leakage boundary, and admission

`eff-dock physical trace` records exact input, combined parameter,
implementation, physical-topology and interaction-typing hashes; pinned
RDKit/Torch runtime versions; coordinate frame; typed/candidate/occupied
counts; physical, interaction and combined component energies; per-term force
norms; fragment corrections; contact statistics; strongest interaction-pair
atom/group identities, distance, radial/cone gates, site quality, nearby radial
candidates rejected by angle/quality gates, every formal-charge site and its
member weights, signed attractive/repulsive charge sums, aromatic
ring/system geometry, halogen donor/acceptor gates, metal element/profile/site
identity, attractive versus repulsion-only disposition, donor occupancy,
masking, and structured profile reason, the trace-only polar-unsatisfied
proxy, and structured unsupported chemistry. The implementation identity
includes ligand loading, fragmentation, and protein typing source, not only
the energy kernel. It does not optimize coordinates.

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

### Confidence and energy-selection boundary

Confidence is evaluated only after model ODE integration, in-loop guidance,
and any explicitly requested post-sampling refinement have produced the final
candidate coordinates. It ranks those completed candidates and never feeds
back into the current ODE.

Raw `InteractionEnergy.total` must not be added directly to confidence. Its
leaf terms mix different signs, opportunity counts, and numerical ranges, and
an attractive total can reward a buried clashing pose. A future combined
selector must first persist final per-candidate leaf energies and availability,
use the vdW steric metric as a hard guard, keep pure predicted RMSD as the base,
and consider opportunity-normalized interaction improvements only inside a
fixed confidence near-tie margin. Normalizers, margins, weights, and switch
thresholds are fitted on PLINDER train and confirmed once on PLINDER
validation; Astex and PoseBusters remain report-only. Until that study is
frozen, pure confidence remains the deployed selector and interaction terms
remain diagnostic selection inputs only.

Before sampler coupling, the implementation must retain:

- force/gradient sign and finite-difference checks;
- SE(3) energy invariance and force/projection equivariance;
- topology exclusion and 1-4 scaling tests;
- chirality, planarity, periodicity, cutoff, coincident/collinear, and
  unsupported-chemistry tests;
- immutable independent golden energy/force fixtures;
- PLINDER validity/RMSD guard evaluation;
- operator-split descent/rejection tests.
- direct-drift scale-zero no-op, interval-ramp, pose-wise normalization,
  guide-only cap, batching, and SE(3) tests.

Frozen pre-formal-charge V3 diagnostic outputs and verification evidence are
preserved as historical provenance in
`docs/GUIDANCE_DIAGNOSTIC_RESULTS.json`; its then-inactive term list is not the
current callable-term inventory. The report-only V4 screened-charge
characterization remains preserved in
`docs/GUIDANCE_FORMAL_CHARGE_BENCHMARK_CHARACTERIZATION_V4.json`. Full
generated traces remain under the ignored `outputs/guidance/` directory.

The completed interaction prior-probe V2 remains historical evidence with its
original three-term baseline (`hydrophobic + hydrogen_bond +
screened_formal_charge`). The later user-authorized seven-term diagnostic
default does not rewrite that protocol, its 96 trajectories, or its
no-admission conclusion.
