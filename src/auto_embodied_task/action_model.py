from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SymbolicCondition:
    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args), "negated": self.negated}


@dataclass(frozen=True)
class SymbolicEffect:
    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "args": list(self.args), "negated": self.negated}


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: tuple[str, ...]
    preconditions: tuple[SymbolicCondition, ...] = ()
    effects: tuple[SymbolicEffect, ...] = ()

    def bind(self, node_ids: list[str]) -> dict[str, str]:
        return {param: node_ids[index] for index, param in enumerate(self.parameters) if index < len(node_ids)}

    def to_json(self, node_ids: list[str] | None = None) -> dict[str, Any]:
        binding = self.bind(node_ids or [])
        return {
            "name": self.name,
            "parameters": list(self.parameters),
            "binding": binding,
            "preconditions": [_ground_condition(item, binding).to_json() for item in self.preconditions],
            "effects": [_ground_effect(item, binding).to_json() for item in self.effects],
        }


ACTION_SCHEMAS: dict[str, ActionSchema] = {
    "look": ActionSchema(
        name="look",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("VISIBLE", ("object",)),
        ),
        effects=(SymbolicEffect("INSPECTED", ("object",)), SymbolicEffect("FOCUS", ("object",))),
    ),
    "inspect": ActionSchema(
        name="inspect",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("VISIBLE", ("object",)),
        ),
        effects=(SymbolicEffect("INSPECTED", ("object",)), SymbolicEffect("FOCUS", ("object",))),
    ),
    "reach": ActionSchema(
        name="reach",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("VISIBLE", ("object",)),
        ),
        effects=(SymbolicEffect("FOCUS", ("object",)),),
    ),
    "walk": ActionSchema(
        name="walk",
        parameters=("target",),
        preconditions=(SymbolicCondition("KNOWN", ("target",)),),
        effects=(SymbolicEffect("FOCUS", ("target",)),),
    ),
    "open": ActionSchema(
        name="open",
        parameters=("container",),
        preconditions=(
            SymbolicCondition("KNOWN", ("container",)),
            SymbolicCondition("OPENABLE", ("container",)),
            SymbolicCondition("VISIBLE", ("container",)),
        ),
        effects=(SymbolicEffect("OPEN", ("container",)), SymbolicEffect("INSPECTED", ("container",))),
    ),
    "close": ActionSchema(
        name="close",
        parameters=("container",),
        preconditions=(
            SymbolicCondition("KNOWN", ("container",)),
            SymbolicCondition("OPENABLE", ("container",)),
        ),
        effects=(SymbolicEffect("OPEN", ("container",), negated=True),),
    ),
    "press": ActionSchema(
        name="press",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("PRESSABLE", ("object",)),
            SymbolicCondition("VISIBLE", ("object",)),
            SymbolicCondition("REACHABLE", ("object",)),
        ),
        effects=(SymbolicEffect("PRESSED", ("object",)), SymbolicEffect("FOCUS", ("object",))),
    ),
    "grab": ActionSchema(
        name="grab",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("GRABBABLE", ("object",)),
            SymbolicCondition("MOVABLE", ("object",)),
            SymbolicCondition("REACHABLE", ("object",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",)),
            SymbolicEffect("LOCATION", ("object",), negated=True),
            SymbolicEffect("FOCUS", ("object",)),
        ),
    ),
    "pick": ActionSchema(
        name="pick",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("GRABBABLE", ("object",)),
            SymbolicCondition("MOVABLE", ("object",)),
            SymbolicCondition("REACHABLE", ("object",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",)),
            SymbolicEffect("LOCATION", ("object",), negated=True),
            SymbolicEffect("FOCUS", ("object",)),
        ),
    ),
    "attach": ActionSchema(
        name="attach",
        parameters=("part", "target"),
        preconditions=(
            SymbolicCondition("KNOWN", ("part",)),
            SymbolicCondition("KNOWN", ("target",)),
            SymbolicCondition("REACHABLE", ("part",)),
            SymbolicCondition("REACHABLE", ("target",)),
            SymbolicCondition("PART_RELATED", ("part", "target")),
        ),
        effects=(SymbolicEffect("ATTACHED", ("part", "target")),),
    ),
    "assemble": ActionSchema(
        name="assemble",
        parameters=("object",),
        preconditions=(
            SymbolicCondition("KNOWN", ("object",)),
            SymbolicCondition("HAS_PARTS", ("object",)),
            SymbolicCondition("PARTS_REACHABLE", ("object",)),
        ),
        effects=(SymbolicEffect("ASSEMBLED", ("object",)),),
    ),
    "putin": ActionSchema(
        name="putin",
        parameters=("object", "container"),
        preconditions=(
            SymbolicCondition("HELD", ("object",)),
            SymbolicCondition("CONTAINER", ("container",)),
            SymbolicCondition("OPEN_IF_OPENABLE", ("container",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",), negated=True),
            SymbolicEffect("INSIDE", ("object", "container")),
        ),
    ),
    "puton": ActionSchema(
        name="puton",
        parameters=("object", "target"),
        preconditions=(
            SymbolicCondition("HELD", ("object",)),
            SymbolicCondition("SURFACE", ("target",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",), negated=True),
            SymbolicEffect("ON", ("object", "target")),
        ),
    ),
    "place_in": ActionSchema(
        name="place_in",
        parameters=("object", "container"),
        preconditions=(
            SymbolicCondition("HELD", ("object",)),
            SymbolicCondition("CONTAINER", ("container",)),
            SymbolicCondition("OPEN_IF_OPENABLE", ("container",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",), negated=True),
            SymbolicEffect("INSIDE", ("object", "container")),
        ),
    ),
    "place_on": ActionSchema(
        name="place_on",
        parameters=("object", "target"),
        preconditions=(
            SymbolicCondition("HELD", ("object",)),
            SymbolicCondition("SURFACE", ("target",)),
        ),
        effects=(
            SymbolicEffect("HELD", ("object",), negated=True),
            SymbolicEffect("ON", ("object", "target")),
        ),
    ),
    "move_aside": ActionSchema(
        name="move_aside",
        parameters=("object",),
        preconditions=(SymbolicCondition("KNOWN", ("object",)), SymbolicCondition("VISIBLE", ("object",))),
        effects=(SymbolicEffect("MOVED_ASIDE", ("object",)), SymbolicEffect("FOCUS", ("object",))),
    ),
    "recover": ActionSchema(
        name="recover",
        parameters=(),
        effects=(SymbolicEffect("RECOVERED"),),
    ),
    "stop": ActionSchema(
        name="stop",
        parameters=(),
        effects=(SymbolicEffect("STOPPED"),),
    ),
}


def schema_for(action_name: str) -> ActionSchema | None:
    base = action_name.lower().removeprefix("failed_")
    return ACTION_SCHEMAS.get(base)


def action_model_trace(action_name: str, node_ids: list[str]) -> dict[str, Any] | None:
    schema = schema_for(action_name)
    if schema is None:
        return None
    return schema.to_json(node_ids)


def _ground_condition(condition: SymbolicCondition, binding: dict[str, str]) -> SymbolicCondition:
    return SymbolicCondition(
        predicate=condition.predicate,
        args=tuple(binding.get(arg, arg) for arg in condition.args),
        negated=condition.negated,
    )


def _ground_effect(effect: SymbolicEffect, binding: dict[str, str]) -> SymbolicEffect:
    return SymbolicEffect(
        predicate=effect.predicate,
        args=tuple(binding.get(arg, arg) for arg in effect.args),
        negated=effect.negated,
    )
