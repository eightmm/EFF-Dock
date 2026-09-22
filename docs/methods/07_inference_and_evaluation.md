# Sampling, refinement, selection and evaluation

Implementation: [sampler](../../src/effdock/inference/sampler.py),
[refinement driver](../../scripts/run_guidance_sdf_post_refinement.py),
[rigid optimizer](../../src/effdock/workflows/relax_guidance.py),
[physical energy](../../src/effdock/guidance/physical.py),
[interaction energy](../../src/effdock/guidance/interaction.py), and
[force projection](../../src/effdock/guidance/runtime.py).
The current result conditions are defined by the [figure captions](../paper/FIGURE_CAPTIONS.md).

## 1. Fixed sampler

The released pair is the early-time/t=0 50k docking EMA and U70k confidence
model identified in the [weight manifest](../../weights/MANIFEST.md).
Main evaluation uses 100 candidates, 10 ODE steps, translation-prior sigma
2 Å and a 10 Å receptor crop. Integration uses the late-power-3 grid

$$
t_j=1-(1-j/S)^3,\quad j=0,\ldots,S,\qquad S=10,
$$
$$
T_f^{j+1}=T_f^j+(t_{j+1}-t_j)v_{\theta,f}(t_j),\qquad
q_f^{j+1}=\operatorname{Exp}_q\!\big((t_{j+1}-t_j)\omega_{\theta,f}(t_j)\big)
\otimes q_f^j.
$$

`Exp_q` converts an axis-angle vector to a unit quaternion; it is not the
componentwise exponential of quaternion coordinates. Contact edges are updated
from the evolving coordinates. The public `dock()` API emits raw poses; energy
refinement and chirality-filtered selection are separately executed benchmark
conditions. All candidates retain generation-order indices. No crystal RMSD,
PB label or energy score enters the deployed confidence ranking.

The pocket must be supplied. For these retrospective redocking benchmarks,
frozen pocket centres can be reference-ligand-derived. The receptor can be a
holo structure. This is not a prospective pocket-discovery evaluation.
Reference pose coordinates are used for mapping/evaluation diagnostics, not
for the learned vector field, the minimized energy or candidate ranking.

## 2. Energy used in post-refinement

Starting from each saved pose, keep its intra-fragment local coordinates fixed
and optimize only fragment translations/rotations with a fixed receptor:

$$
E(x)=E_{\rm physical}(x)+E_{\rm interaction}(x),\qquad
F_a=-\nabla_{x_a}E.
$$

The separate chemical-constraint channel has runtime scale zero in this
refinement. The physical **chiral improper** term is active at scale 1; it is
not the zero-weight chemical channel and is not the subsequent chirality mask.
All energies have the code's declared kcal/mol diagnostic scale. They are not
calibrated affinity/free-energy predictions and the descent is not dynamics.

The [EFF-FF parameter table](../../src/effdock/guidance/parameters/effff_v2.json)
(version 2.2.0) specifies equilibrium geometry, atom types, pair exclusions,
1–4 scaling and coefficients. With bond length `d`, angle `theta`, proper
angle `phi`, wrapped improper deviation `Delta phi`, and pair scale `s_ij`,

$$
E_{\rm bond}=\tfrac12\sum_b k_b(d_b-d_b^0)^2,\qquad
E_{\rm angle}=\tfrac12\sum_a k_a(\theta_a-\theta_a^0)^2,
$$
$$
E_{\rm proper}=\sum_p k_p w_p[1+\cos(n_p\phi_p-\delta_p)],
$$
$$
E_{\rm improper}=\tfrac12\sum_{i\in{\rm planar}}k_i[1-\cos(2\phi_i)]
+\tfrac12\sum_{i\in{\rm chiral}}k_i\operatorname{wrap}(\phi_i-\phi_i^0)^2.
$$

Defaults are `k_b=160`, `k_a=35`, `k_p=0.5`, and improper `k_i=12`, with
length/angle units as specified in the table. Reference geometry and improper
targets come from the prepared input ligand, not the benchmark crystal pose.
Bond restraints act on cut bonds, angles/impropers span fragments, and proper
torsions are enumerated around cut bonds. Proper weights sum to one across
the retained quadruplets for each cut bond. Intra-fragment geometry is fixed.
The default 1–4 Lennard-Jones scale is 0.5. For a pair, let
`rho=sqrt(d²+0.75²)`, `u=clip((8-d)/(8-6),0,1)` and
`S(d)=u³(10-15u+6u²)`. The softened pair term is

$$
E_{{\rm LJ},ij}=s_{ij}S(d_{ij})D_{ij}
\left[\left({X_{ij}\over\rho_{ij}}\right)^{12}
-2\left({X_{ij}\over\rho_{ij}}\right)^6\right].
$$

`X_ij` and `D_ij` are geometric means of the atom parameters. Intramolecular
and protein–ligand pairs use the exclusions/scales in the implementation.
The extra protein–ligand overlap barrier uses

$$
d_{ij}^{\rm safe}=0.8(r_i^{\rm vdW}+r_j^{\rm vdW}),\qquad
p_{ij}=0.1\log\left[1+\exp\left({d_{ij}^{\rm safe}-d_{ij}\over0.1}\right)\right],
$$
$$
E_{\rm steric}=\tfrac12(20)\sum_{ij}p_{ij}^2 S_{ij}^{\rm compact},
\qquad
S_{ij}^{\rm compact}=u_{ij}^3(10-15u_{ij}+6u_{ij}^2),\quad
u_{ij}=\operatorname{clip}\left({d_{ij}^{\rm safe}+0.5-d_{ij}\over0.5},0,1\right).
$$

The `geometry_only` receptor policy retains unsupported receptor heavy atoms
as repulsive obstacles. Typed obstacles use the repulsive LJ term; generic
obstacles use `K a/(1+a) S(d)` with `a=(R/sqrt(d²+s²))^12` and the constants
in `PhysicalEnergyConfig`. Unsupported chemistry does not acquire an invented
attractive interaction.

All seven terms in the [interaction parameter table](../../src/effdock/guidance/parameters/interaction_v1.json)
(version 1.6.0) are enabled. Pair weights are bounded geometric/chemical gates;
exact distance, angle, donor-cone, aromatic-system and metal-profile rules are
in the linked implementation and parameter table.

| Term | Aggregation / principal coefficient |
|---|---|
| Hydrophobic | Ligand-site soft occupancy, epsilon 0.25 |
| Hydrogen bond | Both donor–acceptor directions, epsilon 1.0 |
| Screened formal charge | Signed screened Coulomb pair sum; dielectric 78.5 |
| Pi stacking | Symmetric ring-system occupancy, epsilon 0.25 |
| Cation–pi | Symmetric occupancy in both directions, epsilon 0.5 |
| Halogen bond | Ligand donor–receptor acceptor occupancy, epsilon 0.5 |
| Metal coordination | Profile-specific pair, overcoordination, slot and non-donor terms |

For clipped pair weights `w_ij`, the site occupancy is
`O_i=1-product_j[1-(1-1e-7)w_ij]`. Site-saturated energy is `-epsilon sum_i O_i`;
symmetric saturation averages that sum with its transposed counterpart. The
charge term is

$$
E_{\rm charge}=\sum_{ij}{332.06371\over78.5}q_iq_j
{e^{-0.127\rho_{ij}}\over\rho_{ij}}\,S_{q}(d_{ij}^2),
\qquad \rho_{ij}=\sqrt{d_{ij}^2+0.75^2}.
$$

Here `S_q` is the quintic switch applied between squared distances `8²` and
`10²`, not the LJ switch in distance. Charge groups use formal charge, not
learned partial charges. Metal profiles can enable directional attraction only
for admitted coordination geometries; unsupported profiles remain explicitly
repulsion-only. `polar_unsatisfied_proxy` is diagnostic metadata, not energy.

## 3. Rigid-fragment descent and termination

The energy optimizer uses atomic masses, whereas the learned docking head
uses unit-weight aggregation. For fragment mass `M_f`, centre of mass `C_f`,
geometric centroid `T_f`, and `r_a=x_a-C_f`,

$$
C_f={\sum_{a\in f}m_ax_a\over M_f},\qquad
I_f=\sum_{a\in f}m_a(\|r_a\|^2I_3-r_ar_a^\top),
$$
$$
\Omega_f=I_f^+\sum_{a\in f}r_a\times F_a,\qquad
V_f={\sum_{a\in f}F_a\over M_f}+\Omega_f\times(T_f-C_f).
$$

The last term changes the translation direction from centre-of-mass to
centroid coordinates. The inertia inverse retains eigenvalues strictly above
1% of the largest eigenvalue. Directions are clipped per fragment and jointly
rescaled to satisfy the actual reconstructed maximum atom displacement.

| Setting | External saved-pose refinement |
|---|---:|
| Maximum iterations | 100 |
| Base step | 1.0 |
| Maximum translation / rotation / atom displacement per step | 0.10 Å / 5 degrees / 0.10 Å |
| Backtracking | alpha = 0.5^b, b = 0,...,12 |
| Acceptance | E_trial ≤ E_current + 1e-10 max(1, abs(E_current)) |
| Displacement convergence | max atom displacement <1e-5 Å for 20 consecutive accepted steps |
| Energy plateau stopping | From step 25: energy decrease ≤0.02 + 0.001 max(1, abs(E_current)) for five consecutive accepted steps |
| Saved steps | 0, 25, 50, 75, 100; terminal coordinates fill later requested steps |
| Sampling/scoring crop; refinement receptor shell | 10 Å; 18 Å |
| Physical cutoff; largest interaction cutoff | 8 Å; 10 Å |
| Receptor policy | geometry_only |

The full benchmark driver explicitly enables the energy-plateau criterion;
the standalone refiner CLI leaves it disabled unless those arguments are supplied.
Do not confuse CLI defaults with the executed campaign settings.
Each pose has its own line search and stopping state. A finite failed line
search retains its last accepted pose and records failure. Nonfinite states
are explicit failures. The 18 Å shell gives an 8 Å guaranteed envelope around
the supplied centre for the 10 Å maximum interaction cutoff; envelope validity
is recorded, not silently assumed or used as a success selector.

Confidence-bank preparation used a different registered displacement stop
(0.01 Å for five consecutive steps), as implemented in
[refine_s50_confidence_pose_bank.py](../../scripts/refine_s50_confidence_pose_bank.py).
Do not substitute that training-bank stop for the external benchmark stop.

## 4. Guided sampling in the diagnostic figures

Figures 6 and 8 additionally use direct ODE guidance at `eta=2`; the main
unguided condition has `eta=0`. Let `U(v,omega)` denote the induced atom
velocity and `RMS(U)=sqrt(mean_a ||U_a||²)`. With raw projected energy
directions `(V,Omega)`, the added drift before capping is

$$
(\Delta v,\Delta\omega)=\eta\,\bar r_j
{\operatorname{RMS}(U(v_\theta,\omega_\theta))\over
 \operatorname{RMS}(U(V,\Omega))}(V,\Omega),\qquad
\bar r_j={1\over\Delta t_j}\int_{t_j}^{t_{j+1}}\max(0,2t-1)\,dt.
$$

The drift is zero if either RMS is at most `1e-8` or the energy direction is
nonfinite. Atom forces are capped at norm 20 before mass-weighted projection.
The physical LJ softcore decreases linearly from 1.5 to 0.75 Å over the active
half of integration; its value is evaluated at the active interval midpoint.
One nonnegative per-pose scale caps added translation/angular velocity at
5 Å and 5 radians per unit flow time, and the estimated atom displacement
`dt max_a (||Delta v_f|| + ||Delta omega_f|| ||x_a-T_f||)` at 0.25 Å.
The sampler adds this drift to the learned field and integrates it once.
It is distinct from the subsequent backtracking refinement above. The
separate chemical-constraint drift remains zero in these comparisons.

## 5. Confidence and chirality selection

Both raw and refined coordinates are independently scored using U70k with the
paired docking features. Ordinary selection minimizes predicted pose RMSD,
with generation index breaking ties. For chirality-filtered selection, let
`C` be candidates whose tetrahedral stereochemistry agrees with the input
isomeric ligand; select

$$
k^*=\begin{cases}
\operatorname*{arg\,min}_{k\in C}(\widehat r_k,k),&C\ne\varnothing,\\
\operatorname*{arg\,min}_{1\le k\le N}(\widehat r_k,k),&C=\varnothing.
\end{cases}
$$

The main-paper mask is tetrahedral-only: it does not use reference RMSD,
PoseBusters labels or an E/Z filter. Fallback to the unfiltered Top-1 is recorded.
Chirality agreement does not imply official PB validity. The output retains
all original complexes and all candidate indices.

## 6. Metrics, uncertainty and cohort scope

Let `r_ik` be symmetry-aware no-alignment heavy-atom RMSD for candidate `k` of
complex `i`, and `b_ik` its official selected-pose PB validity. For a fixed
cohort of `M` complexes,

$$
\mathrm{SR}=100M^{-1}\sum_i\mathbf1[r_{i,k_i^*}<2\,\mathrm{\AA}],\qquad
\mathrm{SR}_{\rm PB}=100M^{-1}\sum_i\mathbf1[r_{i,k_i^*}<2\,\mathrm{\AA}]b_{i,k_i^*}.
$$

The solid bar is `SR_PB`; the hatched extension is `SR-SR_PB`. Both criteria
must hold on the same selected pose. Each boundary has its own across-seed
sample SD; SD is not a CI or the uncertainty of the difference of boundaries.
RDKit `CalcRMS` handles graph-equivalent atom assignments without alignment.

Top-k cumulative success asks whether any of the best `k` confidence-ranked
poses in the full 100-pose bank has RMSD <2 Å. Generation-prefix cumulative
success asks the same question for saved indices 1,...,N. Prefix selected
Top-1 reranks within the prefix and need not be monotone. Both cumulative
curves equal the full-bank RMSD oracle at 100. Their full-bank PB endpoints
have not been evaluated and must not be inferred from selected-pose PB.

The five main cohorts per seed are Astex Diverse Set (85), PoseBusters v2
(308), PhiBench (206; three reconstructed cases), FoldBench (558), and OpenBind
(925; a dense single-protease auxiliary cohort with two flagged noncovalent
approximations of covalent systems). They total 2,082 complexes. Validation
(1,076) is additionally included in relatedness composition, but has no matched
three-repeat performance bank. Input/evaluation failures retain denominators.
The PoseBusters **v2 dataset** label is distinct from the evaluator's software
version. Literature baselines retain their reported evaluation versions.

Main figures show means and sample SD over three seeds. For Figure S1, average
paired per-complex success differences across seeds, then take 2,000 bootstrap
resamples, either individual complexes or complete exact-PDB groups; the latter
uses the complex-weighted mean. Report percentile 95% CIs in percentage points,
without implying protein-family clustering or multiple-comparison adjustment.
Sequence relatedness and same-training-sample ligand–sequence intersection
are defined in [RELATEDNESS.md](../paper/RELATEDNESS.md); they are not certified
pocket identity or a validated binary leakage definition.

Opened external cohorts are descriptive. U70k was selected on the fixed
1,035-complex internal bank. Guidance/budget and pocket/prior figures retain
their separately captioned guided conditions and cannot be pooled with the
unguided main condition. PoseX follows a separate protocol and is outside this
14-figure package.
