"""Structured failures for unsupported guidance parameterization."""

from __future__ import annotations


class UnsupportedPhysicalChemistryError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


__all__ = ["UnsupportedPhysicalChemistryError"]
