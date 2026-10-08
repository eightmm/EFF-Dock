# Fragment orientation and coordinate-only confidence scoring

The evaluated method maps a learned fragment-local vector into world coordinates with **Rw**. Released docking and confidence checkpoints were trained with the historical **Rᵀw** operator. Executing Rw with those weights changes their computation, not their tensor shapes or training history. The manuscript specifies this frozen-weight evaluation; it does not claim a retrained model or a universal accuracy improvement.

## Coordinate convention

Atom coordinates satisfy `x = R local + T`, with R an active local-to-world rotation. Under a joint world rotation Q, the fragment frame becomes QR. Therefore

$$F(R,w)=Rw,\qquad F(QR,w)=QRw=QF(R,w).$$

The historical contraction gives

$$G(R,w)=R^\mathsf{T}w,\qquad G(QR,w)=R^\mathsf{T}Q^\mathsf{T}w,$$

which generally differs from `Q G(R,w)`. For R=I, w=(1,0,0) and a 90-degree rotation about z, Rw transforms to (0,1,0), whereas the historical feature becomes (0,-1,0). This counterexample concerns the feature map, not a measured docking RMSD.

The corrected einsum is `nik,ck->nci`; the historical one is `nki,ck->nci`. The interaction layer's separate **Rᵀ(x_j−x_i)** operation maps a world displacement to local coordinates and remains correct. The force/torque projection, geometric-centroid correction and left-multiplied rotation update in refinement are unchanged.

## Checkpoints and compatibility

| Entry point | Operator rule |
|---|---|
| Fresh training with `configs/train_rw.yaml` | Explicit `model.orientation_injection: rw` |
| Historical configurations omitting the field | `legacy_rt_w`; new checkpoints record the effective value |
| Checkpoint loading and inference without an override | Follow checkpoint metadata; a missing field means `legacy_rt_w` |
| Explicit `--orientation-injection rw` | Execute Rw with the supplied weights |
| Explicit `--orientation-injection legacy_rt_w` | Execute the historical operator |
| Exact training resume | Reject an operator change, including with relaxed config checking |
| Weights-only initialization | Permit an explicit change while recording source checksum, step and operator |

Checkpoint metadata owns the operator unless explicitly overridden. Loading reports the trained and effective choices; inference artifacts preserve them. Historical measurements must retain their operator labels and must not be pooled with a different convention.

## Reproducing the coordinate-only scoring convention

Use `--orientation-injection rw` together with `--confidence-frame-policy contextual_v1` for the manuscript's coordinate-only scoring convention. Refinement and its subsequent scorer use `--reference-conformer-policy generation`; both use the same ligand template rebuilt from the recorded generation preparation and conformer seed. Pass the source translation-prior scale explicitly when rescoring; the primary value is 2 Angstrom. This scale enters the recomputed terminal docking features, rather than a separate conditioning input to the confidence interaction layers.

`contextual_v1` retains proper Kabsch fitting for templates of rank at least two. Lower-rank fragments use indexed ligand and receptor context to define a frame, failing explicitly when the required context is degenerate. The compatibility default remains `historical_kabsch`, as used for confidence training. A changed scoring convention requires fresh ranking of the complete saved bank; old selected-pose labels cannot be reused as new outcomes.

These statements concern joint rigid transformations with a fixed prepared fragment-local template. Scene-defined recovery does not make the network invariant to every unobservable fragment twist or to reorienting the local template itself. Finite rotation and backend checks do not prove exact equivariance of the entire preparation, sampling, refinement and selection pipeline.

The canonical [manuscript](../papers/main.tex) and [Supplementary Information](../papers/SI.tex) separate orientation-map controls, preparation changes and frame-recovery checks. Historical diagnostic outcomes remain in Git history and the retained numerical provenance, rather than serving as the current primary benchmark table.
