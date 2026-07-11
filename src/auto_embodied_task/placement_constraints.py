from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .models import normalize_relation


PLACEMENT_ACTION_RELATIONS = {
    "putin": "INSIDE",
    "place_in": "INSIDE",
    "puton": "ON",
    "place_on": "ON",
}


@dataclass(frozen=True)
class PlacementEdgeRule:
    source: str
    target: str
    relation: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PlacementEdgeRule":
        source = payload.get("from", payload.get("source", payload.get("object")))
        target = payload.get("to", payload.get("target", payload.get("container", payload.get("surface"))))
        relation = payload.get("relation")
        action = payload.get("action")
        if relation is None and action is not None:
            relation = PLACEMENT_ACTION_RELATIONS.get(str(action).strip().lower())
        if source is None or target is None or relation is None:
            raise ValueError("placement edge rule needs source/from/object, target/to, and relation/action")
        return cls(
            source=str(source).strip(),
            target=str(target).strip(),
            relation=_canonical_constraint_relation(relation),
        )

    def matches(
        self,
        *,
        source_id: str,
        target_id: str,
        relation: str,
        source_name: str | None = None,
        target_name: str | None = None,
    ) -> bool:
        return (
            self.relation == _canonical_constraint_relation(relation)
            and _constraint_ref_matches(self.source, source_id, source_name)
            and _constraint_ref_matches(self.target, target_id, target_name)
        )

    def to_json(self) -> dict[str, str]:
        return {"from": self.source, "to": self.target, "relation": self.relation}


@dataclass(frozen=True)
class PlacementEdgeConstraints:
    forbidden_edges: tuple[PlacementEdgeRule, ...] = ()
    allowed_edges: tuple[PlacementEdgeRule, ...] = ()

    @classmethod
    def from_json(cls, payload: Any) -> "PlacementEdgeConstraints":
        if payload is None:
            return cls()
        if isinstance(payload, list):
            return cls(forbidden_edges=tuple(_placement_edge_rules(payload)))
        if not isinstance(payload, dict):
            raise ValueError("placement edge constraints must be a JSON object or list")
        forbidden_payload = _first_present(
            payload,
            "forbidden_edges",
            "invalid_edges",
            "blocked_edges",
            "nonexistent_edges",
            "edges",
        )
        allowed_payload = _first_present(payload, "allowed_edges", "valid_edges")
        return cls(
            forbidden_edges=tuple(_placement_edge_rules(forbidden_payload or [])),
            allowed_edges=tuple(_placement_edge_rules(allowed_payload or [])),
        )

    def allows(
        self,
        *,
        source_id: str,
        target_id: str,
        relation: str,
        source_name: str | None = None,
        target_name: str | None = None,
    ) -> bool:
        if self.allowed_edges and not any(
            rule.matches(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                source_name=source_name,
                target_name=target_name,
            )
            for rule in self.allowed_edges
        ):
            return False
        return not any(
            rule.matches(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                source_name=source_name,
                target_name=target_name,
            )
            for rule in self.forbidden_edges
        )

    def to_json(self) -> dict[str, list[dict[str, str]]]:
        return {
            "forbidden_edges": [rule.to_json() for rule in self.forbidden_edges],
            "allowed_edges": [rule.to_json() for rule in self.allowed_edges],
        }

    def is_empty(self) -> bool:
        return not self.forbidden_edges and not self.allowed_edges


def load_placement_edge_constraints(path: str | Path) -> PlacementEdgeConstraints:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return PlacementEdgeConstraints.from_json(payload)
    except ValueError as exc:
        raise ValueError(f"{source}: invalid placement edge constraints: {exc}") from exc


def _placement_edge_rules(payload: Any) -> list[PlacementEdgeRule]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("placement edge rules must be a list")
    rules: list[PlacementEdgeRule] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("placement edge rule must be a JSON object")
        rules.append(PlacementEdgeRule.from_json(item))
    return rules


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _canonical_constraint_relation(value: Any) -> str:
    relation = normalize_relation(str(value))
    if relation == "IN":
        return "INSIDE"
    return relation


def _constraint_ref_matches(rule_ref: str, node_id: str, node_name: str | None) -> bool:
    if rule_ref == "*":
        return True
    candidates = {node_id, node_id.lower()}
    if node_name is not None:
        candidates.add(node_name)
        candidates.add(node_name.lower())
    return rule_ref in candidates or rule_ref.lower() in candidates
