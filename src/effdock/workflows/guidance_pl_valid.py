"""Scoped protein-ligand validity view over official PoseBusters redock checks."""

from __future__ import annotations

from effdock.workflows.posebusters_report import VALIDITY_CHECKS

COFACTOR_AND_WATER_CHECKS = (
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
)
PL_VALIDITY_CHECKS = tuple(
    check for check in VALIDITY_CHECKS if check not in COFACTOR_AND_WATER_CHECKS
)

if len(VALIDITY_CHECKS) != 27 or len(PL_VALIDITY_CHECKS) != 21:
    raise RuntimeError("unexpected PoseBusters validity schema")


def is_pl_valid(checks: dict[str, bool]) -> bool:
    return all(bool(checks[check]) for check in PL_VALIDITY_CHECKS)


__all__ = ["COFACTOR_AND_WATER_CHECKS", "PL_VALIDITY_CHECKS", "is_pl_valid"]
