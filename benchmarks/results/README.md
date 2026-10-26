# Public benchmark results

This directory contains compact, claim-bearing benchmark artifacts intended
for the public EFF-Dock repository. Raw inference outputs remain under ignored
`outputs/` trees.

Each admitted benchmark result should provide:

1. a machine-readable JSON or CSV summary with schema/version information;
2. dataset identity, denominator, completed/missing target counts, and receptor
   policy;
3. model checkpoint/source revision, inference settings, seeds, candidate
   count, and selector/refinement definition;
4. per-seed metrics plus mean and sample standard deviation when repeated;
5. generation, reranking, refinement, and evaluation runtimes kept separate;
6. hashes or stable references for the protocol, input manifest, and producing
   command; and
7. the corresponding paper-ready figure when one is reported.

Do not copy raw structures, complete pose ensembles, scheduler logs, private
paths, or unredistributable model/data artifacts into this directory.
