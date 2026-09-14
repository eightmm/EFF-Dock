# Docking vector-field head and flow objective

Implementation sources: `src/effdock/models/effdock.py`,
`src/effdock/training/losses.py`, and `configs/train.yaml`.

## 1. Atom field to fragment twist

After six equivariant layers, the docking head emits an odd vector field
`f_a in R^3` for every ligand atom. The fragment twist is obtained by
Newton--Euler aggregation, not an unconstrained fragment MLP:

\[
v_f={1\over |f|}\sum_{a:f(a)=f}f_a,
\qquad
\tau_f=\sum_{a:f(a)=f}(x_a-T_f)\times f_a,
\]
\[
I_f=\sum_{a:f(a)=f}\left(\|r_a\|^2I_3-r_ar_a^\top\right),
\qquad \omega_f=I_f^+\tau_f,
\quad r_a=x_a-T_f.
\]

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

\[
\mathcal L_v={1\over 3F}\sum_f\|v_f-v_f^*\|_2^2,
\qquad
\mathcal L_\omega={1\over 3F}\sum_f\|\omega_f-P_f\omega_f^*\|_2^2,
\]
\[
\mathcal L_{\mathrm{flow}}=\mathcal L_v+8\mathcal L_\omega.
\]

The angular term is additionally masked by observable axes and valid
fragment-size conditions. All reductions are means over the valid collated
fragments, so a large ligand does not scale the loss solely by atom count.

## 3. Auxiliary coordinate losses

Training also uses a rigidly reconstructed atom auxiliary loss and a one-step
distance-geometry loss. With `Delta t=1-t`, the predicted endpoint is

\[
\widehat T_f^1=T_f^t+\Delta t\,v_f,
\qquad
\widehat q_f^1=\exp(\Delta t\,\omega_f)\otimes q_f^t.
\]

The predicted endpoint atoms follow the rigid reconstruction equation from
`02_fragment_se3_flow.md`. The distance-geometry term compares pair distances
at that endpoint to the target. Within-fragment distances are rigid-body
invariant, so its useful signal is cross-fragment spacing, including cut-bond
geometry and global molecular shape.

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
