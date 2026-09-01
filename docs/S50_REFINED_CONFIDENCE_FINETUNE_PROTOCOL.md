# S50 refined confidence fine-tune protocol

The refined-confidence branch starts only after the current symmetry-label
confidence U50,000 job, the complete refined confidence bank, and the complete
refined train symmetry-RMSD sidecar succeed. It warm-starts the symmetry-label
U50,000 `latest.pt` weights with a fresh optimizer and scheduler in a separate
no-overwrite output directory.

The first bounded continuation is 10,000 updates with stratified N64 training
and N100 validation. Both learning rates are tenfold below the raw adaptation
(`Muon=2e-4`, `AdamW=3e-6`); the confidence architecture, loss, global complex
batch, weight decay, and gradient clipping remain fixed. Atomic `latest.pt` is
published every 500 updates. `best.pt` is selected only at the U0, U2000, ...,
U10000 full-validation looks using symmetry-aware Top-1 success below 2 A.

A same-four-GPU one-update smoke over the sealed refined bank gates the full
continuation. Missing or non-U50,000 raw latest weights, bank/code/config hash
drift, non-finite values, OOM, or DDP failures block the continuation rather
than silently changing pose count or learning rate.

Every pose-level objective uses `pose_rmsd_symmetry_no_align` from the sealed
refined sidecar. The atom displacement regression/BCE targets remain the
fixed-map `atom_disp`, because a unique per-atom displacement is undefined
across symmetry-equivalent atom mappings.
