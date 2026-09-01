# Slurm launchers

These launchers record the cluster workflows used for the paper. They are
reproducibility aids, not a portable scheduler abstraction. Partition, QoS,
GPU, account, data-root, and output-root defaults are site-specific and should
be overridden or edited for another cluster.

The reusable scientific implementation belongs in `src/effdock/` or a Python
script under `scripts/`; Slurm files should remain thin resource and argument
wrappers. Raw scheduler logs and machine-generated submission ledgers are not
published.
