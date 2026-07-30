"""Frozen defaults for the promoted EFF-Dock inference stack."""

from pathlib import Path

DEFAULT_CONFIG = Path("configs/train.yaml")
DEFAULT_DOCKING_CHECKPOINT = Path("weights/effdock_geometry_ft_100k_best.pt")
DEFAULT_CONFIDENCE_CHECKPOINT = Path(
    "weights/effdock_confidence_extmatch_n80_s25_step42500.pt"
)
DEFAULT_NUM_SAMPLES = 80
DEFAULT_NUM_STEPS = 25
DEFAULT_SIGMA = 0.5
DEFAULT_POCKET_CUTOFF = 10.0
DEFAULT_TIME_SCHEDULE = "late"
DEFAULT_SCHEDULE_POWER = 3.0
