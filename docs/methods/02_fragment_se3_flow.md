# Fragment representation and conditional SE(3) flow

Implementation sources: `src/effdock/preprocess/fragments.py`,
`src/effdock/geometry/flow_matching.py`, and the training dataset/collator.

## 1. Rigid-fragment decomposition

For a sanitized heavy-atom ligand graph, EFF-Dock cuts a bond only when it is
single, non-ring, not planar-conjugated (amide/ester/urea/carbamate/
sulfonamide), and both endpoints retain heavy-atom degree greater than one.
A deterministic greedy pass accepts a cut only if both endpoints retain one
uncut heavy-atom neighbour. Singleton fragments are then merged into their
largest adjacent fragment. Thus every fragment has at least two heavy atoms.

Let `f(a)` map ligand atom `a` to one of `F` fragments. Crystal coordinates
are decomposed into centroid `T_f in R^3` and local coordinates `ell_a`:

$$
T_f = |f|^{-1}\sum_{a:f(a)=f}x_a, \qquad
\ell_a=x_a-T_f.
$$

At any state, the reconstructed atom coordinate is

$$
x_a(T,q)=R(q_{f(a)})\ell_a+T_{f(a)},
$$

where `q_f` is a unit quaternion and `R(q_f)` its `3 x 3` rotation matrix.
The state tensors per collated batch are `T_frag [F_total,3]`,
`q_frag [F_total,4]`, `frag_sizes [F_total]`, and `frag_id_for_atoms
[A_total]`.

Cut-bond, fragment-adjacency, and local triangulation edges preserve the
cross-fragment topology; the triangulation is formed from endpoints around a
cut bond and stores the reference distance of each directed pair.

## 2. Conditional interpolation and targets

The receptor pocket centre is `c`. Translation priors use the conditioned
Gaussian `T_f^0 = c + sigma epsilon_f`, with `epsilon_f ~ N(0,I_3)`.
Initial rotations are uniform on `SO(3)`; a single-atom fragment would use the
identity rotation, although the released fragmenter avoids such fragments.

For target pose `(T^1,q^1)` and sampled `t in [0,1]`, translation is linearly
interpolated and rotation is interpolated by SLERP:

$$
T_f^t=(1-t)T_f^0+tT_f^1, \qquad q_f^t=\operatorname{SLERP}(q_f^0,q_f^1;t).
$$

The translation target and global-frame angular target are

$$
v_f^*=T_f^1-T_f^0,\qquad
\omega_f^*=\operatorname{axisAngle}(q_f^1\otimes(q_f^0)^{-1}).
$$

The angular convention is left multiplication,
`q_{t+dt}=exp(dt omega) tensor-product q_t`. The implied atom field is
`v_f + omega_f cross (x_a-T_f)`. Ligand-wide uniform rotational augmentation
is applied consistently to the target geometry and local frames, preventing a
fixed laboratory-frame orientation shortcut.

## 3. Time and prior conditioning

The base configuration uses pocket crops sampled uniformly from 6--12 A and
translation-prior sigma values `{0.5, 1, 2, 3, 4}` A with probabilities
`{.10, .25, .30, .25, .10}`. These are training augmentations, not the
released endpoint.

The released docking fine-tune is the early-time/t=0 50k run. Its exact time
law is

$$
p(t)=0.80p_{\mathrm{SimpleFold}}(t)+0.10\,U(0,0.3)+0.10\,\delta_0.
$$

The retained component is implemented by drawing `U(0,1)` with probability
0.02 and otherwise setting `t=sigmoid(0.8+1.7Z)`, `Z~N(0,1)`:

$$
p_{\mathrm{SimpleFold}}=0.02\,U(0,1)
+0.98\,\operatorname{Law}\!\left[\operatorname{sigmoid}(0.8+1.7Z)\right].
$$

This specifies the implemented sampling law; the exact-zero replay is a
point mass and is not approximated by a small positive time.

At deployment, sampling fixes `sigma=2.0 A`; time integration uses the
late-power-3 grid specified in `07_inference_and_evaluation.md`. The model
receives both a sinusoidal time condition and a log-sigma condition, each
mapped to 128 dimensions; a zero-initialized final sigma projection makes the
sigma path an explicit learned correction rather than a change in the initial
architecture.

## 4. Coordinate and unit conventions

Coordinates, distances, crop radii, and RMSD labels are in Angstroms. `t` is
dimensionless. `v` is displacement over the normalized unit integration
interval and `omega` is radians over that interval. The model never predicts
independent atom coordinates: it predicts a rigid transform per fragment, so
intra-fragment distances are exactly preserved by the flow state.
