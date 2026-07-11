from __future__ import annotations

from random import Random
from typing import ClassVar

from ..models import TaskRecord, ViewGraph


class TaskModifier:
    name: ClassVar[str] = "base"

    def apply(self, task: TaskRecord, graph: ViewGraph, rng: Random) -> TaskRecord | None:
        raise NotImplementedError


MODIFIER_REGISTRY: dict[str, type[TaskModifier]] = {}


def register_modifier(cls: type[TaskModifier]) -> type[TaskModifier]:
    MODIFIER_REGISTRY[cls.name] = cls
    return cls


def available_modifiers() -> tuple[str, ...]:
    return tuple(sorted(MODIFIER_REGISTRY))


def build_modifiers(names: tuple[str, ...] | list[str]) -> list[TaskModifier]:
    if not names:
        return []
    expanded = tuple(MODIFIER_REGISTRY) if "all" in names else tuple(names)
    modifiers = []
    for name in expanded:
        if name not in MODIFIER_REGISTRY:
            known = ", ".join(available_modifiers())
            raise ValueError(f"Unknown setting {name!r}. Available settings: {known}")
        modifiers.append(MODIFIER_REGISTRY[name]())
    return modifiers
