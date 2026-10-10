Dear Editor,

We are pleased to submit our manuscript, "EFF-Dock: Separating Candidate Generation, Physical Validity and Selection in Fragment-Based Flow-Matching Docking", for consideration as an Article in *Nature Computational Science*.

A selected docking pose alone cannot reveal whether a failure arose from candidate generation, geometric validity or selection. Distinguishing these limitations shows where improvements to a generative pipeline could increase accuracy.

We present EFF-Dock, a supplied-pocket docking pipeline built so that each stage can be measured on the same candidates. A rigid-fragment SE(3)-equivariant flow-matching model generates pose banks, a bounded geometric-energy refinement improves candidate validity, and a separate confidence network ranks the candidates. Across five redocking cohorts (Astex Diverse Set, PoseBusters v2, PhiBench-derived, FoldBench and OpenBind), we find the following:

- Refined 100-pose banks contained a candidate within 2 Å RMSD of the reference for 87.9–96.5% of complexes, whereas selected poses met this threshold and passed all PoseBusters validity checks for 53–80%.
- On Astex and PoseBusters v2, where we applied the official PoseBusters checks to every candidate, refinement increased the fraction of valid candidates from 24–27% to 88%, while relatively few candidates crossed the RMSD-success threshold in either direction.
- On these two cohorts, valid near-native candidates went unselected for approximately 14% of complexes, accounting for 66–71% of remaining failures. Selection was therefore the largest remaining source of failure within the evaluated candidate banks.
- A matched, reduced-budget ablation found no clear advantage of a Newton–Euler readout over direct fragment-motion prediction under the tested conditions.

The computational contribution is a controlled evaluation of candidate availability, geometric validity and selection within one fragment-based generative pipeline. The analysis identifies which stage limits docking accuracy and could inform other scientific workflows that generate multiple candidates for subsequent ranking. The joint oracle defined in the manuscript gives an upper bound on selection gains within a fixed candidate bank. The manuscript documents baseline protocols, uncertainty estimates, candidate-budget controls, training-set relatedness, development-set use and a training–evaluation difference in one feature map.

The confidence network follows the separation of pose-quality and binding-affinity prediction introduced in our earlier work (Sim and Lee, "BA-Pred and RMSD-Pred", J. Chem. Inf. Model. 66, 3480–3495 (2026)), which is cited in the manuscript. The present work is distinct in addressing pose generation, refinement and stage-resolved evaluation. [Authors to confirm: any preprint of this manuscript, other related manuscripts under consideration or in press, and any prior discussions with an NCS editor, or state that there are none.]

All authors have approved this submission, and the manuscript is not under consideration elsewhere. [Authors to confirm.]

Source code, released weights, training-membership lists, per-complex outcomes and the numerical source data for every figure are available at https://github.com/eightmm/EFF-Dock. The version used in the study will be archived with a persistent identifier upon publication.

Competing interests: The authors declare no competing financial interest. J.L. is affiliated with Arontier Co., as listed in the author information.

Thank you for considering our manuscript.

Sincerely,

Juyong Lee, on behalf of all authors
College of Pharmacy and Graduate School of Convergence Science and Technology, Seoul National University
nicole23@snu.ac.kr
