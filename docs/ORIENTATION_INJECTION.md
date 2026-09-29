# Fragment orientation: corrected Rw and historical Rᵀw

The corrected fragment-to-world feature is **Rw**. The implementation now supports
it explicitly without changing learned tensor shapes or overwriting released
weights. Historical checkpoints and manuscript benchmark figures used **Rᵀw**.
The correction follows the coordinate convention; the small external comparison
below does not select a new production model or establish an accuracy gain.

## Why the operators differ

The model represents atom coordinates as `x = R local + T`, where R is an active
local-to-world rotation. Under a joint world rotation Q, the fragment frame
becomes QR. For a learned local vector w,

$$F(R,w)=Rw,\qquad F(QR,w)=QRw=QF(R,w).$$

The historical contraction instead gives

$$G(R,w)=R^\mathsf{T}w,\qquad G(QR,w)=R^\mathsf{T}Q^\mathsf{T}w,$$

which generally differs from `Q G(R,w)`. With R=I, w=(1,0,0), and a 90-degree
rotation about z, Rw rotates to (0,1,0), whereas the historical map gives
(0,-1,0). This counterexample concerns the feature map, not a two-Angstrom
measured error in a predicted ligand pose.

The corrected einsum is `nik,ck->nci`; the historical one is `nki,ck->nci`.
This is a change of model computation under fixed weights, not merely a new
notation. Conversely, the interaction layer's **Rᵀ(x_j-x_i)** maps a world
relative vector into a local frame and is already correct. It stays unchanged.
No evidence from this audit warrants changing the force/torque projection,
centroid correction, or left-multiplied quaternion update in refinement.

## Runtime and checkpoint contract

| Entry point | Operator rule |
|---|---|
| Fresh corrected training with `configs/train_rw.yaml` | Explicit `model.orientation_injection: rw` |
| Historical training configurations omitting the field | `legacy_rt_w`; the effective value is saved in new checkpoints |
| `load_model`, `eff-dock dock`, `eff-dock evaluate` | Follow checkpoint metadata; missing field means `legacy_rt_w` |
| Explicit `--orientation-injection rw` | Run corrected Rw with the supplied weights |
| Explicit `--orientation-injection legacy_rt_w` | Run the historical operator |
| Exact training resume | Reject a change of operator, even if strict-config checking is disabled |
| Weights-only `--init-from` | Permit migration; record source checksum, step and source operator |

For checkpoint loading, YAML continues to supply architecture dimensions, but
**checkpoint metadata owns this operator unless explicitly overridden**. The
returned config reflects the effective choice. The loader prints the checkpoint
and effective choices. SDF/PT results and evaluation summaries record the
requested, effective and checkpoint operators. Benchmark aggregation rejects
mixed-operator shards; absent historical shard metadata means `legacy_rt_w`.
No learned state_dict keys or checkpoint tensor dimensions were added.

To evaluate the corrected operator, add this option to an otherwise unchanged
command, using a new output directory:

```bash
uv run eff-dock dock --protein receptor.pdb --ligand ligand.sdf \
  --pocket-center 0,0,0 --orientation-injection rw --out-dir outputs/rw
# The evaluate subcommand accepts the same --orientation-injection option.
```

`configs/train.yaml` remains byte-identical to the historical release, including
its config hash. Use `configs/train_rw.yaml` for fresh corrected training; all
other settings are identical.

Do not simply edit a checkpoint's metadata to claim it was trained with Rw.
Source-hash-pinned historical confidence-bank/audit manifests remain unchanged;
they may correctly refuse the edited source. Use their recorded source revision
for exact historical reproduction instead of silently re-pinning those records.

## Confidence features and retraining

Confidence consumes freshly extracted docking hidden states at t=1. Changing
the docking operator changes those features as well as the generated poses.
The frozen released confidence checkpoint was trained on `legacy_rt_w` features;
its combination with corrected Rw is a **cross-operator frozen-weight arm**.
Scoring emits a warning and user-facing result metadata records this combination.
The same docking instance supplies generation and confidence feature extraction.
There is no new selector, threshold, recalibration, or energy tie-break.

New confidence shards and checkpoints record `docking_orientation_injection`.
Confidence training checks every consumed shard against the declared operator,
including both banks in paired training and validation. The training option is
`--docking-orientation-injection`; missing historical metadata means legacy.
A confidence resume cannot change this feature convention. Use distinct bank
names and output directories for newly generated data. The historical S50 bank
preparation/materialization scripts are legacy-only reproduction tools and omit
this metadata; do not use them with corrected checkpoints. Use
`eff-dock confidence prepare` with an operator-versioned checkpoint instead.

Retraining is **not mathematically required to execute the corrected operator**.
It may still improve adaptation or confidence calibration; that remains an
empirical question. No retraining or new full benchmark sweep accompanies this
code correction.

## Frozen-weight diagnostic

[Comparison PDF](paper/diagnostics/Rw_comparison.pdf) ·
[Preview](paper/diagnostics/Rw_comparison.png) ·
[Aggregate numerical data](../benchmarks/results/paper/orientation/summary.json)

One paired seed 42, 100 identical initial candidates per arm, 10 ODE steps,
supplied 10 Å pocket, fixed 50k EMA docking/U70k confidence weights. Each arm uses
the same existing physical/interaction refinement and input-chirality-filtered
minimum-pRMSD selection. The table is **refined** performance:

| Dataset | Metric | Historical Rᵀw | Corrected Rw |
|---|---|---:|---:|
| Astex Diverse Set (85/85) | RMSD<2Å | 71/85 (83.53%) | 69/85 (81.18%) |
| Astex Diverse Set (85/85) | RMSD<2Å and PB-valid | 69/85 (81.18%) | 67/85 (78.82%) |
| PoseBusters v2 (308/308) | RMSD<2Å | 256/308 (83.12%) | 258/308 (83.77%) |
| PoseBusters v2 (308/308) | RMSD<2Å and PB-valid | 247/308 (80.19%) | 248/308 (80.52%) |

PB uses all 27 non-RMSD PoseBusters 0.6.5 validity checks. RMSD uses all-heavy-atom
receptor-frame comparison, symmetry-aware where representation permits. Two PB
cases use the established calculator's full atom-map fallback for input/reference
representation mismatch; there is no subset RMSD. Raw, unfiltered, oracle and
paired complex-bootstrap results are retained in the numerical data.

**PB now includes all 308 cases: 307 original cases plus one disclosed supplement.**
7UJ5_DGL originally stopped before scoring/refinement and the corrected arm at
a repeated-generation difference guard of 0.01 Å. A bounded replay also failed
(maximum difference 0.0218446 Å); both failed attempts are preserved. These were
repeatability failures, not measured unsuccessful docking outcomes.

A separately authorized, predeclared supplementary run retained the same weights,
seed, explicit priors, N100/S10 generation, refinement and confidence selector,
but made the repeat check diagnostic-only. Its first realization was used
regardless of outcome. This run happened to pass the original repeat threshold
(maximum difference 0.00000705 Å); that does not establish numerical stability.
Both arms selected candidate 12 after refinement (zero-based):

| 7UJ5_DGL refined selection | pRMSD (Å) | RMSD (Å) | PB-valid |
|---|---:|---:|---|
| Historical Rᵀw | 0.5751 | 0.4025 | Yes |
| Corrected Rw | 0.5761 | 0.4054 | Yes |

The 308-case table and figure include this measured supplement, without imputing
an outcome. They do not represent an uninterrupted execution under the original
strict repeatability gate. The previous 307-case report remains in Git history
and the local audit archive. PB-valid success improves in 3 cases and worsens in
2, yielding **+0.325 percentage points**. There is no longer a missing outcome;
paired case-bootstrap intervals were recomputed on all 308 cases. The 95% interval
for the PB-valid-success difference is −0.974 to +1.623 percentage points; it
includes zero and excludes seed/execution variability.

The original Astex 85 run is retained. One of its two corrected-arm losses
recovered in a same-seed replay. Both this observation and the PB repeat guard
limit attribution of small differences to the operator. Paired case-bootstrap
intervals exclude seed/execution variability. These repeatedly inspected external
sets do not provide independent model-selection or generalization evidence.

## Are three inference seeds needed?

The correction is justified by the active-frame algebra and numerical checks;
three seeds are not needed to decide which contraction is mathematically correct.
They are recommended before replacing the manuscript's primary performance
results with corrected-Rw results. The present PB advantage is one complex, and
the Astex and same-seed replay observations show that execution variation matters.
Three inference seeds do not mean three model-training runs, and do not by
themselves establish statistical significance.

For the final comparison, first run Astex and PB with both operators, three
predeclared sampling seeds, and identical explicit priors for each paired
complex/seed. Freeze source, weights, hardware policy, refinement and selection;
record numerical repeat controls separately from docking success. Report each
seed and paired differences, plus mean and sample SD. Do not select the best seed
or mix old-operator values into corrected-Rw averages. Retain the present seed-42
comparison as a disclosed diagnostic; a fresh uniform three-seed run is the
preferred final estimate. Historical baselines can be reused only when their
inputs, priors and full protocol match the paired comparison.

If the paper adopts Rw as its main method, every benchmark table and analysis
presented as Rw must use corrected outputs; Astex/PB alone cannot relabel results
for PhiBench, FoldBench or OpenBind. No new GPU inference or training was launched
for this 308-case report update.

## Discussion and decision

A two-round read-only review with Claude Opus 5.5 independently confirmed the
active-frame algebra and the distinct purpose of local-frame RᵀΔx. We rejected
silently flipping all legacy inference/resume paths, and agreed on checkpoint-owned
semantics with an explicit inference override and corrected fresh-training config.
The review also identified confidence feature provenance, unconditional resume
checks, and warm-start provenance as required compatibility work. These were
implemented. A final code review also identified the pinned training-config hash;
the corrected config was separated to preserve historical replays. No reviewer confidence was treated as experimental evidence.

The corrected map is analytically rotation-equivariant. Existing building-block
tests or a finite full-model rotation check do not prove that the entire sampler,
frame reconstruction, confidence selector and refinement pipeline is exactly
SE(3)-equivariant. SO(3) fragment frames also do not establish reflection invariance
of stereochemical inputs. Existing manuscript numbers remain labelled as legacy;
they must be reevaluated before being presented as corrected-Rw results.

## Verification

`tests/test_orientation_injection.py` tests nonzero-weight covariance, a legacy
counterexample, the exact historical contraction, gradients, checkpoint/override
precedence, resume protection, confidence-bank consistency and mixed-shard
rejection. EMA export preserves operator and initialization metadata. Tests do
not rely on zero initialization, which would hide the indexing error.

The earlier frozen-weight core probe covered five prespecified complexes, three
times and four proper rigid transforms. Maximum angular-velocity relative error
changed from 0.0270 to 2.53e-5; this is a finite numerical diagnostic. Source tests
and the actual-implementation recheck are reported in the release verification
record beside the aggregate numerical data. The actual corrected implementation subsequently
passed all 60 tested rigid-transform conditions for both velocity outputs, with
maximum relative angular error 1.81e-5. The affected CPU suite passed 90 tests;
three CUDA-only layer tests were skipped on CPU. See the
[verification record](../benchmarks/results/paper/orientation/verification.json).

**Figure caption.** Frozen-weight orientation-injection comparison on (A) Astex
Diverse Set (n=85) and (B) PoseBusters v2 (n=308).
Bar totals show chirality-filtered confidence Top-1 RMSD <2 Å success before and
after fixed refinement. Solid regions additionally pass all 27 non-RMSD
PoseBusters checks; hatched regions are RMSD-successful but PB-invalid. The
308-case PB cohort contains 307 original cases and one predeclared supplementary
run with diagnostic-only repeat control, unchanged selection and first-realization
outputs. Results use one paired seed and are separate from the three-repeat
legacy manuscript figures.
