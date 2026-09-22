# Docking vector-field head and flow objective

Implementation sources: `src/effdock/models/effdock.py`,
`src/effdock/training/losses.py`, and `configs/train.yaml`.

## 1. Atom field to fragment twist

After six equivariant layers, the docking head emits an odd vector field
`f_a in R^3` for every ligand atom. The fragment twist is obtained by
Newton--Euler aggregation, not an unconstrained fragment MLP:

$$
v_f={1\over |f|}\sum_{a:f(a)=f}f_a,
\qquad
\tau_f=\sum_{a:f(a)=f}(x_a-T_f)\times f_a,
$$
$$
I_f=\sum_{a:f(a)=f}\left(\|r_a\|^2I_3-r_ar_a^\top\right),
\qquad \omega_f=I_f^+\tau_f,
\quad r_a=x_a-T_f.
$$

The head's pre-projection irreps use 192 scalar channels and retain the
non-scalar widths `32,32,16,16`, for flattened width `544`; its final atom
output is one `1o` vector (`[A_total,3]`). A zero-initialized self tensor
product augments the linear equivariant head without changing the initial
function.

`I_f` is diagonalized in float64. Eigenvalues below 1% of the fragment's
largest eigenvalue are unobservable and removed from the pseudo-inverse. The
same projection matrix `P_f [3,3]` is applied to the angular target; this
prevents a fictitious loss for rotations of collinear or otherwise
rank-deficient geometry.

## 2. Primary conditional-flow loss

For batch fragment targets `v*` and `omega*` defined in
`02_fragment_se3_flow.md`, the base loss is

$$
\mathcal L_v={1\over 3F}\sum_f\|v_f-v_f^*\|_2^2,
\qquad
\mathcal L_\omega={1\over 3F}\sum_f\|\omega_f-P_f\omega_f^*\|_2^2,
$$
$$
\mathcal L_{\mathrm{flow}}=\mathcal L_v+8\mathcal L_\omega.
$$

In the angular equation the sum and denominator include only fragments with
more than one atom; replace `F` by their count `F_omega` if singleton fragments
are present. `P_f` projects the target into the observable rotational subspace;
the denominator is three times the number of eligible fragments, not the number
of observable eigenvectors. The released fragmenter avoids singletons.
Fragment means weight complexes with more fragments more heavily. DDP scales
each rank's mean to reproduce the global valid-fragment mean. The optional
per-time flow weighting is disabled in `train_early_time_ft_50k.yaml`.

## 3. Atom-velocity and distance-geometry losses

The atom auxiliary term compares **instantaneous atom velocities**, not endpoint
positions. For `r_a=x_a^t-T_{f(a)}^t`, rigid-body kinematics gives

$$
u_a=v_{f(a)}+\omega_{f(a)}\times r_a,\qquad
u_a^*=v_{f(a)}^*+\omega_{f(a)}^*\times r_a,
\qquad L_{\rm atom}={1\over3A}\sum_a\|u_a-u_a^*\|^2.
$$

This reduction is over atoms, so larger ligands have greater weight. The
auxiliary implementation uses the unprojected angular target in the induced
atom field; an exactly unobservable axial rotation contributes zero motion.

The separate distance-geometry term reconstructs a one-step endpoint. With
`Delta t=1-t`, the predicted endpoint is

$$
\widehat T_f^1=T_f^t+\Delta t\,v_f,
\qquad
\widehat q_f^1=\exp(\Delta t\,\omega_f)\otimes q_f^t.
$$

The endpoint atoms follow the rigid reconstruction equation from
`02_fragment_se3_flow.md`. For complex `b`, let `P_b` contain all unordered atom
pairs in different fragments, and let `B_+` contain complexes with at least one
such pair. The implemented term is

$$
D_b={1\over|P_b|}\sum_{(a,c)\in P_b}
\left(\|\widehat x_a^1-\widehat x_c^1\|-\|x_a^1-x_c^1\|\right)^2,
\qquad
L_{\rm DG}={\sum_{b\in B_+}t_b^2D_b\over\sum_{b\in B_+}t_b^2}.
$$

The term is zero when the denominator is zero. Thus it is a `t²`-weighted
mean of per-complex pair means, not an unweighted mean over all atom pairs.
DDP uses the global sum of `t²` as the normalizer. Intra-fragment pairs are
excluded explicitly because their distances are invariant under rigid motion.

The complete active objective is

$$
L=L_v+8L_\omega+0.3L_{\rm atom}+3L_{\rm DG}.
$$

For the released architecture, `omega_weight=8.0`, `atom_aux_weight=0.3`, and
`dg_weight=3.0`. The exact active loss composition is recorded with the run;
these scalar values are the architecture configuration, not an invitation to
retune on opened external sets.

## 4. Numerical contract

The inertia eigensolve validates finite inputs, symmetrizes `I_f`, and retries
a CUDA convergence failure on CPU in float64. It does not silently replace an
invalid tensor. This keeps the geometry of the rigid-fragment output defined
for small and nearly collinear fragments while preserving explicit failures
for non-finite states.
