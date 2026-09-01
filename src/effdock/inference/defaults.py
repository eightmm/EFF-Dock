"""Deployment defaults for the promoted EFF-Dock inference stack."""

from pathlib import Path

DEFAULT_CONFIG = Path("configs/train.yaml")
DEFAULT_DOCKING_CHECKPOINT = Path(
    "weights/effdock_docking_early_time_t0p10_50k.pt"
)
DEFAULT_CONFIDENCE_CHECKPOINT = Path(
    "weights/effdock_confidence_s50_raw_refined_u70k.pt"
)
DEFAULT_NUM_SAMPLES = 100
DEFAULT_NUM_STEPS = 10
DEFAULT_SIGMA = 2.0
DEFAULT_POCKET_CUTOFF = 10.0
DEFAULT_TIME_SCHEDULE = "late"
DEFAULT_SCHEDULE_POWER = 3.0
