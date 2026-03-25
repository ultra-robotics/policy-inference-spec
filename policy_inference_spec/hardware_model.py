from __future__ import annotations

from enum import StrEnum


class HardwareModel(StrEnum):
    GEN1 = "gen1"
    GEN2 = "gen2"


def as_hardware_model(value: str | HardwareModel) -> HardwareModel:
    if isinstance(value, HardwareModel):
        return value
    try:
        return HardwareModel(str(value).strip())
    except ValueError as e:
        raise AssertionError(f"hardware_model must be gen1 or gen2, got {value!r}") from e


__all__ = ["HardwareModel", "as_hardware_model"]
