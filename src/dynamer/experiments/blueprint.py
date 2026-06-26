from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentBlueprint:
    dataset: str
    task: str
    protocol: str
    modalities: tuple[str, ...]
    model_name: str
    ablation_name: str | None = None
