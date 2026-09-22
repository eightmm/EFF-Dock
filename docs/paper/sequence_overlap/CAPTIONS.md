# Binding-chain sequence relatedness

Reference: 47,277 executed docking-training samples; not the 43k confidence subset.
Six cohorts: Astex Diverse Set (85), PoseBusters v2 (308), PhiBench (206),
FoldBench (558), OpenBind (925), Validation (1,076).

For each complex, query-normalized sequence identity is the maximum sum of
identical known residues across one-to-one assignments of ligand-binding chains
within a single training sample, divided by the total observed query binding-chain
sequence length. Unaligned query residues and unmatched query chains remain in
the denominator. Thus >=70% also requires >=70% of query residues to have known
identical aligned counterparts; ordinary alignment identity alone is not used.
Binding chains use the existing 6 Å heavy-atom criterion and frozen annotated
training candidate set with six previously audited ineligible references removed.
Search reuses frozen exhaustive local MMseqs alignments, including documented
malformed-alignment recovery; source-sequence and eligibility limitations remain. Two reference chains contain
only UNK residues. Five chains in three PDB entries retain differences from the
historical PLINDER length metadata; current and recovered sequences do not
constitute bit-exact reconstruction of every historical reference sequence.
This is a maximum over the declared candidate set, not all possible biological
assemblies. Unknown residues provide no identity evidence; absence of observed
matches is not a guarantee of biological novelty.

Composition and cumulative distributions include all 3,158 complexes. Composition
bins <30, 30–<70, 70–<90, 90–100% and overlap cutoff >=70% were inherited before
joining outcomes; these are descriptive conventions, not calibrated leakage
thresholds. Performance includes five external benchmarks with three matched
refined/chirality-filtered repeat banks. Error bars are sample SD, not confidence
intervals; total bar height is RMSD <2 Å success, solid height additionally
requires PB validity; only the RMSD-passing but PB-invalid extension is hatched. Validation has no equivalent three-repeat bank.

Ligand–protein overlap requires exact canonical heavy-isomeric SMILES AND sequence
score >=70% within the SAME training sample. All exact-ligand training samples
are searched, not just the unconstrained best sequence hit. Separate matches
means both conditions have witnesses somewhere in training but no single sample
satisfies both. Ligand only/sequence only mean only that respective condition has
a match; neither observed means neither has a match under the stated evidence.
One unparseable training ligand was excluded for every query by differing heavy
atom element inventories. Per-complex constrained maxima and witness IDs are
in per_complex.csv; all 3,158 queries have numerical sequence scores.

This analysis describes ligand–protein relatedness, not confirmed pocket overlap
or proven data leakage. The original PLINDER community-based train exclusion
procedure remains unchanged and is described separately. Direct sequence scores
do not retroactively resolve historical community labels.

Rendering: dataset name and sample count occupy separate lines. Sequence-bin colors
use peach, blue-gray, blue, and lavender consistently in composition and performance.
PB-invalid extensions remain hatched; PB-valid portions remain solid.

Figure labels shorten the metric to "Sequence identity (%)". Throughout these
figures this means query-normalized identity as defined above, not identity
normalized only by the aligned region.
