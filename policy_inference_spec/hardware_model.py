from __future__ import annotations

from enum import StrEnum


class HardwareModel(StrEnum):
    GEN1 = "gen1"
    GEN2 = "gen2"

    @classmethod
    def _missing_(cls, value: object) -> HardwareModel | None:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized == value:
                return None
        else:
            normalized = str(value).strip()
        try:
            return cls(normalized)
        except ValueError:
            return None


__all__ = ["HardwareModel"]
