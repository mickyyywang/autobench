#!/usr/bin/env python3
"""Generate semantically tailored intervention manifests.

Candidates come from each task's goal and initial view graph. Aligned episodes use
their successful teacher trajectory; direct view-graph/task sources use a design-only
goal reference that is excluded from teacher-efficiency reporting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
TASK_SPECS_DIR = PROJECT_DIR / "task_specs_cn"
COPY_OBJECTS_PATH = TASK_SPECS_DIR / "copy_objects.json"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from auto_embodied_task.episode_sources import (  # noqa: E402
    direct_manifest_source,
    load_direct_episode,
)


INJECTABLE_ACTIONS = {
    "reach",
    "walk",
    "open",
    "close",
    "press",
    "grab",
    "pick",
    "attach",
    "puton",
    "putin",
    "move_aside",
}

PLACEMENT_RELATIONS = {"putin": "INSIDE", "puton": "ON"}

STATE_REGRESSIONS = {
    "open": ("open", False),
    "close": ("open", True),
    "press": ("pressed", False),
    "move_aside": ("moved_aside", False),
}

ADD_OBJECT_STAGING_SURFACES = {"桌面", "餐桌"}

# Direct-source task families can opt into an equally explicit canonical work
# surface without weakening the no-fallback policy used by the existing tasks.
ADD_OBJECT_STAGING_SURFACES_BY_TASK_GROUP = {
    "LoadCondimentsInFridge": {"厨房台面"},
}

# Buffet additions originate from the semantically matching storage section.
# The OPEN predicate is added to the runtime trigger below, so the new object is
# introduced while it is observable rather than silently appearing in a closed
# container.
ADD_OBJECT_SOURCE_CONTAINERS_BY_TASK_GROUP = {
    "DivideBuffetTrays": {
        "蔬菜托盘": "冰箱冷藏层",
        "肉类托盘": "冰箱冷冻层",
    },
}


def _material_spec_group(task_group: str) -> str:
    direct = TASK_SPECS_DIR / f"{task_group}_material_properties.json"
    if direct.is_file():
        return task_group
    candidates = [
        source.name.removesuffix("_material_properties.json")
        for source in TASK_SPECS_DIR.glob("*_material_properties.json")
        if task_group.startswith(source.name.removesuffix("_material_properties.json"))
    ]
    return max(candidates, key=len) if candidates else task_group


def _load_copy_object_registry() -> tuple[dict[str, Any], set[str]]:
    """Load copy templates and task groups that explicitly disable inheritance."""
    if not COPY_OBJECTS_PATH.is_file():
        return {}, set()
    registry = json.loads(COPY_OBJECTS_PATH.read_text(encoding="utf-8"))
    task_groups = registry.get("task_groups", {})
    if not isinstance(task_groups, dict):
        raise ValueError(f"{COPY_OBJECTS_PATH}: task_groups must be an object")
    raw_disabled = registry.get("inherit_copy_disabled_task_groups", [])
    if not isinstance(raw_disabled, list) or any(
        not isinstance(value, str) or not value.strip() for value in raw_disabled
    ):
        raise ValueError(
            f"{COPY_OBJECTS_PATH}: inherit_copy_disabled_task_groups must be "
            "an array of non-empty strings"
        )
    disabled = {value.strip() for value in raw_disabled}
    registered_disabled = disabled & set(task_groups)
    if registered_disabled:
        raise ValueError(
            f"{COPY_OBJECTS_PATH}: inherit-copy-disabled task groups must not define "
            f"copy templates: {sorted(registered_disabled)}"
        )
    return task_groups, disabled


def _inherit_copy_enabled(task_group: str) -> bool:
    _, disabled = _load_copy_object_registry()
    return _material_spec_group(task_group) not in disabled


def _load_add_object_templates(task_group: str) -> dict[str, dict[str, Any]]:
    """Load task-local add-object templates without making them initial materials."""
    task_groups, disabled = _load_copy_object_registry()
    spec_group = _material_spec_group(task_group)
    if spec_group in disabled:
        return {}
    raw_templates = task_groups.get(spec_group, {})
    if not isinstance(raw_templates, dict):
        raise ValueError(
            f"{COPY_OBJECTS_PATH}: task group {spec_group!r} must be an object"
        )

    material_source = TASK_SPECS_DIR / f"{spec_group}_material_properties.json"
    if not material_source.is_file():
        if raw_templates:
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: missing material properties for {spec_group!r}"
            )
        return {}
    material_payload = json.loads(material_source.read_text(encoding="utf-8"))
    materials = material_payload.get("materials", {})
    if not isinstance(materials, dict):
        raise ValueError(f"{material_source}: materials must be an object")
    declared_sources = dict(materials)
    for material in materials.values():
        if not isinstance(material, dict):
            continue
        for part in material.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            part_id = str(part.get("id") or part.get("name") or "").strip()
            if part_id:
                declared_sources[part_id] = part
    initial_ids = {str(material_id) for material_id in declared_sources}
    initial_names = {
        str(item.get("name") or material_id)
        for material_id, item in declared_sources.items()
        if isinstance(item, dict)
    }
    copyable_sources = {
        str(material_id)
        for material_id, item in declared_sources.items()
        if isinstance(item, dict)
        and "COPYABLE"
        in {str(value).strip().upper() for value in item.get("properties", [])}
    }
    missing_templates = copyable_sources - set(raw_templates)
    extra_templates = set(raw_templates) - copyable_sources
    if missing_templates or extra_templates:
        raise ValueError(
            f"{COPY_OBJECTS_PATH}: task group {spec_group!r} must define exactly its "
            f"COPYABLE materials; missing={sorted(missing_templates)}, "
            f"extra={sorted(extra_templates)}"
        )

    templates: dict[str, dict[str, Any]] = {}
    added_ids: set[str] = set()
    added_names: set[str] = set()
    for raw_source_id, raw_template in raw_templates.items():
        source_id = str(raw_source_id).strip()
        if not source_id or not isinstance(raw_template, dict):
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: every copy-object entry needs a source id and object"
            )
        source_material = declared_sources.get(source_id)
        if not isinstance(source_material, dict):
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: source {source_id!r} is not a declared "
                f"{spec_group!r} material or part"
            )
        source_properties = {
            str(value).strip().upper() for value in source_material.get("properties", [])
        }
        if "COPYABLE" not in source_properties:
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: source {source_id!r} needs COPYABLE"
            )
        object_spec = raw_template.get("object", raw_template)
        if not isinstance(object_spec, dict):
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: template {source_id!r} must be an object"
            )
        object_id = str(object_spec.get("id") or "").strip()
        object_name = str(object_spec.get("name") or "").strip()
        if not object_id or not object_name:
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: template {source_id!r} needs object.id and object.name"
            )
        if object_id in initial_ids or object_name in initial_names:
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: template {source_id!r} must be absent from initial materials"
            )
        if object_id in added_ids or object_name in added_names:
            raise ValueError(
                f"{COPY_OBJECTS_PATH}: template {source_id!r} reuses an add-object id/name"
            )
        added_ids.add(object_id)
        added_names.add(object_name)
        templates[source_id] = copy.deepcopy(object_spec)
    return templates


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_relation(value: Any) -> str:
    relation = str(value or "").strip().upper()
    return "INSIDE" if relation == "IN" else relation


def _successful_actions(
    episode: dict[str, Any], *, include_aligned_actions: bool = False
) -> list[dict[str, Any]]:
    """Return successful actions, optionally including alignment-only action rows."""
    actions: list[dict[str, Any]] = []
    for step in episode.get("trajectory") or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        event = step.get("event")
        if not isinstance(action, dict):
            continue
        status = event.get("status") if isinstance(event, dict) else None
        if status != "success" and not (include_aligned_actions and status is None):
            continue
        name = str(action.get("base_name") or action.get("name") or "").lower()
        if not name or name in {"recover", "stop"}:
            continue
        actions.append(
            {
                "step": int(step.get("step") or 0),
                "name": name,
                "node_ids": [str(node_id) for node_id in action.get("node_ids") or []],
                "execution_evidence": (
                    "event_success" if status == "success" else "aligned_action"
                ),
            }
        )
    return actions


def _action_key(action: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(action["name"]), tuple(str(value) for value in action["node_ids"])


def _unique_actions(actions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for action in actions:
        key = _action_key(action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def _visible_node_ids(episode: dict[str, Any]) -> set[str]:
    return {
        str(node["id"])
        for node in (episode.get("initial_observation") or {}).get("visible_nodes") or []
        if isinstance(node, dict) and node.get("id") is not None
    }


def _node_map(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): node
        for node in (episode.get("initial_view_graph") or {}).get("nodes") or []
        if isinstance(node, dict) and node.get("id") is not None
    }


def _graph_edges(episode: dict[str, Any]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for raw in (episode.get("initial_view_graph") or {}).get("edges") or []:
        if not isinstance(raw, dict):
            continue
        source = raw.get("from", raw.get("source"))
        target = raw.get("to", raw.get("target"))
        relation = _normalize_relation(raw.get("relation"))
        if source is None or target is None or not relation:
            continue
        edges.append({"from": str(source), "to": str(target), "relation": relation})
    return edges


def _node_properties(node: dict[str, Any]) -> set[str]:
    return {str(value).upper() for value in node.get("properties") or []}


def _node_states(node: dict[str, Any]) -> set[str]:
    return {str(value).upper() for value in node.get("states") or []}


def _is_container(node: dict[str, Any]) -> bool:
    properties = _node_properties(node)
    if properties:
        return bool(properties & {"CONTAINER", "CONTAINERS"})
    return str(node.get("category") or "").lower() in {"container", "receptacle"}


def _is_surface(node: dict[str, Any]) -> bool:
    properties = _node_properties(node)
    if properties:
        return bool(properties & {"SURFACE", "SURFACES"})
    return str(node.get("category") or "").lower() == "surface"


def _is_openable(node: dict[str, Any]) -> bool:
    return "CAN_OPEN" in _node_properties(node)


def _is_movable(node: dict[str, Any]) -> bool:
    return bool(_node_properties(node) & {"MOVABLE", "GRABBABLE"})


def _goal_facts(criterion: Any) -> list[dict[str, Any]]:
    """Flatten a goal while recording whether a fact belongs to an OR branch."""
    facts: list[dict[str, Any]] = []

    def visit(value: Any, *, optional: bool, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            if "final" in value:
                visit(value["final"], optional=optional, path=(*path, "final"))
                return
            if "predicate" in value:
                args = value.get("args", value.get("arguments", []))
                facts.append(
                    {
                        "predicate": _normalize_relation(value["predicate"]),
                        "args": [str(item) for item in (args if isinstance(args, list) else [args])],
                        "optional": optional,
                        "path": list(path),
                    }
                )
                return
            for key in ("and", "all"):
                if key in value:
                    children = value[key] if isinstance(value[key], list) else [value[key]]
                    for index, child in enumerate(children):
                        visit(child, optional=optional, path=(*path, key, str(index)))
                    return
            for key in ("or", "any"):
                if key in value:
                    children = value[key] if isinstance(value[key], list) else [value[key]]
                    for index, child in enumerate(children):
                        visit(child, optional=True, path=(*path, key, str(index)))
                    return
            return
        if not isinstance(value, list) or not value:
            return
        head = value[0]
        if isinstance(head, str) and _normalize_relation(head) in {"AND", "OR"}:
            branch_optional = optional or _normalize_relation(head) == "OR"
            for index, child in enumerate(value[1:]):
                visit(child, optional=branch_optional, path=(*path, str(head).lower(), str(index)))
            return
        if isinstance(head, str) and _normalize_relation(head) != "NOT":
            facts.append(
                {
                    "predicate": _normalize_relation(head),
                    "args": [str(item) for item in value[1:]],
                    "optional": optional,
                    "path": list(path),
                }
            )

    visit(criterion, optional=False, path=())
    return facts


def _episode_selection_key(
    episode: dict[str, Any],
    actions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> str:
    return _sha256_json(
        {
            "episode_id": episode.get("episode_id"),
            "actions": actions,
            "goal_facts": facts,
            "edges": _graph_edges(episode),
        }
    )


def _ranked_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    selection_key: str,
    role: str,
) -> list[dict[str, Any]]:
    def rank(candidate: dict[str, Any]) -> tuple[str, str]:
        canonical = _canonical_json(candidate)
        digest = hashlib.sha256(
            f"{selection_key}:{role}:{canonical}".encode("utf-8")
        ).hexdigest()
        return digest, canonical

    return sorted(candidates, key=rank)


def _placement_candidates(
    actions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    *,
    prefer_strict: bool = True,
) -> list[dict[str, Any]]:
    placements: list[dict[str, Any]] = []
    for action in _unique_actions(actions):
        relation = PLACEMENT_RELATIONS.get(action["name"])
        if relation is None or len(action["node_ids"]) < 2:
            continue
        object_id, target_id = action["node_ids"][:2]
        matching = [
            fact
            for fact in facts
            if fact["predicate"] == relation
            and len(fact["args"]) >= 2
            and fact["args"][:2] == [object_id, target_id]
        ]
        if not matching:
            continue
        strict = any(not fact["optional"] for fact in matching)
        placements.append(
            {
                **action,
                "goal_relation": relation,
                "goal_optional": not strict,
            }
        )
    strict = [candidate for candidate in placements if not candidate["goal_optional"]]
    return (strict or placements) if prefer_strict else placements


def _state_candidates(
    actions: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for action in _unique_actions(actions):
        regression = STATE_REGRESSIONS.get(action["name"])
        if regression is None or not action["node_ids"]:
            continue
        node_id = action["node_ids"][0]
        node = nodes.get(node_id)
        if node is None:
            continue
        if action["name"] in {"open", "close"} and not _is_openable(node):
            continue
        if action["name"] == "move_aside" and not _is_movable(node):
            continue
        field, value = regression
        candidates.append({**action, "state_field": field, "regressed_value": value})
    return candidates


def _root_surface_candidates(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    incoming_on: dict[str, int] = {}
    for edge in edges:
        if edge["relation"] == "ON":
            incoming_on[edge["to"]] = incoming_on.get(edge["to"], 0) + 1
    candidates = [
        {
            "node_id": node_id,
            "incoming_on_count": incoming_on.get(node_id, 0),
            "static": "STATIC" in _node_properties(node),
        }
        for node_id, node in nodes.items()
        if _is_surface(node)
    ]
    return sorted(
        candidates,
        key=lambda item: (-item["incoming_on_count"], not item["static"], item["node_id"]),
    )


def _resolve_initial_node_ref(
    nodes: dict[str, dict[str, Any]], reference: Any
) -> str | None:
    wanted = str(reference)
    if wanted in nodes:
        return wanted
    matches = [
        node_id
        for node_id, node in nodes.items()
        if str(node.get("name") or "") == wanted
    ]
    return matches[0] if len(matches) == 1 else None


def _copy_template_key(
    source_id: str,
    source: dict[str, Any],
    object_templates: dict[str, dict[str, Any]],
) -> str | None:
    candidates = [source_id, str(source.get("name") or "")]
    candidates.extend(re.sub(r"_\d+$", "", value) for value in list(candidates))
    return next((value for value in candidates if value in object_templates), None)


def _copy_identity_spec(
    template: dict[str, Any],
    *,
    source_node_id: str,
    part_of: str | None = None,
) -> dict[str, Any]:
    """Keep natural identity fields while inheriting behavior from copy_from.

    Runtime copies properties, states, capacity, and other behavioral metadata
    from the actual source node.  Registry values must not override those
    attributes because the profiled view graph is the source of truth.
    """
    result = {
        field: copy.deepcopy(template[field])
        for field in ("id", "name")
        if template.get(field) is not None
    }
    result["copy_from"] = source_node_id
    if part_of is not None:
        result["part_of"] = part_of
    return result


def _direct_part_ids(
    parent_id: str,
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
) -> list[str]:
    part_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("part_of") or "") == parent_id
    }
    part_ids.update(
        edge["from"]
        for edge in edges
        if edge["relation"] == "PART_OF" and edge["to"] == parent_id
    )
    return sorted(part_id for part_id in part_ids if part_id in nodes)


def _strict_required_inside_count(
    target_id: str,
    *,
    facts: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
) -> int:
    subjects = {
        resolved_subject
        for fact in facts
        if fact["predicate"] == "INSIDE"
        and not fact["optional"]
        and len(fact["args"]) >= 2
        and _resolve_initial_node_ref(nodes, fact["args"][1]) == target_id
        for resolved_subject in [_resolve_initial_node_ref(nodes, fact["args"][0])]
        if resolved_subject is not None
    }
    return len(subjects)


def _visible_staging_surfaces(
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    visible: set[str],
    excluded_node_ids: set[str],
    allowed_surface_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return visible canonical work surfaces for spawning added objects.

    Add-object interventions must not introduce a second spatial challenge by
    spawning an object on a movable plate, tray, or other incidental surface.
    There is intentionally no fallback when the task family's explicit
    canonical surface is unavailable.
    """
    allowed_surface_ids = allowed_surface_ids or ADD_OBJECT_STAGING_SURFACES
    canonical_surface_ids = {
        node_id
        for node_id, node in nodes.items()
        if node_id in allowed_surface_ids
        or str(node.get("name") or "") in allowed_surface_ids
    }
    return [
        candidate
        for candidate in _root_surface_candidates(nodes, edges)
        if candidate["node_id"] in canonical_surface_ids
        and candidate["node_id"] in visible
        and candidate["node_id"] not in excluded_node_ids
    ]


def _add_object_inherit_design(
    *,
    placements: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    visible: set[str],
    selection_key: str,
    object_templates: dict[str, dict[str, Any]],
    task_group: str,
) -> dict[str, Any]:
    """Design the executable copy/same-class add-object condition."""
    supported_inherited_predicates = {
        "ON",
        "INSIDE",
        "OPEN",
        "CLOSED",
        "AT_MOST_INSIDE",
        "ASSEMBLED",
    }
    inheritable_placements = []
    for candidate in placements:
        candidate_source_id = candidate["node_ids"][0]
        if "COPYABLE" not in _node_properties(nodes[candidate_source_id]):
            continue
        source_predicates = {
            fact["predicate"]
            for fact in facts
            if fact["args"]
            and _resolve_initial_node_ref(nodes, fact["args"][0]) == candidate_source_id
        }
        if not source_predicates or not source_predicates <= supported_inherited_predicates:
            continue
        source_parts = _direct_part_ids(
            candidate_source_id,
            nodes=nodes,
            edges=edges,
        )
        if "ASSEMBLED" in source_predicates and (
            not source_parts
            or any(
                _copy_template_key(part_id, nodes[part_id], object_templates) is None
                for part_id in source_parts
            )
        ):
            continue
        relation = candidate["goal_relation"]
        target_id = candidate["node_ids"][1]
        maximum = _node_max_items(nodes[target_id]) if target_id in nodes else None
        if (
            relation == "INSIDE"
            and maximum is not None
            and _strict_required_inside_count(
                target_id,
                facts=facts,
                nodes=nodes,
            )
            >= maximum
        ):
            continue
        inheritable_placements.append(candidate)
    if not inheritable_placements:
        return {
            "eligible": False,
            "ineligible_reason": (
                "No COPYABLE teacher placement has a locally inheritable goal, "
                "registered assembly family, and one-item destination capacity."
            ),
            "candidate_count": 0,
        }
    templated_placements = [
        candidate
        for candidate in inheritable_placements
        if _copy_template_key(
            candidate["node_ids"][0],
            nodes[candidate["node_ids"][0]],
            object_templates,
        )
        is not None
    ]
    if not templated_placements:
        return {
            "eligible": False,
            "ineligible_reason": (
                "COPYABLE teacher placements exist, but none has a registered "
                "copy_objects.json template."
            ),
            "candidate_count": 0,
        }
    source_or_group_targets: dict[
        tuple[str, tuple[str, ...]], set[tuple[str, str]]
    ] = {}
    for fact in facts:
        if (
            fact["predicate"] not in {"INSIDE", "ON"}
            or len(fact["args"]) < 2
        ):
            continue
        path = tuple(str(value) for value in fact.get("path") or [])
        or_indexes = [
            index for index, value in enumerate(path) if value in {"or", "any"}
        ]
        if not or_indexes:
            continue
        source_ref = _resolve_initial_node_ref(nodes, fact["args"][0])
        target_ref = _resolve_initial_node_ref(nodes, fact["args"][1])
        if source_ref is None or target_ref is None:
            continue
        or_group = path[: or_indexes[-1] + 1]
        source_or_group_targets.setdefault((source_ref, or_group), set()).add(
            (fact["predicate"], target_ref)
        )
    multi_position_or_sources = {
        source_id
        for (source_id, _), targets in source_or_group_targets.items()
        if len(targets) >= 2
    }
    or_source_candidates = [
        candidate
        for candidate in templated_placements
        if candidate["node_ids"][0] in multi_position_or_sources
    ]
    source_candidates = or_source_candidates or templated_placements
    placement = _ranked_candidates(
        source_candidates,
        selection_key=selection_key,
        role="add_object_inherit_source",
    )[0]
    source_id = placement["node_ids"][0]
    source = nodes[source_id]
    goal_targets = {
        resolved
        for fact in facts
        if fact["args"]
        and _resolve_initial_node_ref(nodes, fact["args"][0]) == source_id
        for resolved in (
            [_resolve_initial_node_ref(nodes, fact["args"][1])]
            if len(fact["args"]) >= 2
            else []
        )
        if resolved is not None
    }
    spec_group = _material_spec_group(task_group)
    source_container_by_goal = ADD_OBJECT_SOURCE_CONTAINERS_BY_TASK_GROUP.get(
        spec_group, {}
    )
    source_container_id = source_container_by_goal.get(placement["node_ids"][1])
    required_predicates: list[dict[str, Any]] = []
    if source_container_id is not None:
        source_container_id = _resolve_initial_node_ref(nodes, source_container_id)
        if source_container_id is None:
            return {
                "eligible": False,
                "ineligible_reason": (
                    "Configured add-object source container is absent from the "
                    "initial view graph."
                ),
                "candidate_count": len(inheritable_placements),
            }
        source_container = nodes[source_container_id]
        if not _is_container(source_container) or source_container_id not in visible:
            return {
                "eligible": False,
                "ineligible_reason": (
                    f"Configured add-object source {source_container_id!r} must be a "
                    "visible container."
                ),
                "candidate_count": len(inheritable_placements),
            }
        staging_relation = "INSIDE"
        staging_target = source_container_id
        staging_candidate_count = 1
        if _is_openable(source_container):
            required_predicates.append(
                {"predicate": "OPEN", "args": [source_container_id]}
            )
    else:
        allowed_surfaces = ADD_OBJECT_STAGING_SURFACES_BY_TASK_GROUP.get(
            spec_group, ADD_OBJECT_STAGING_SURFACES
        )
        staging_candidates = _visible_staging_surfaces(
            nodes=nodes,
            edges=edges,
            visible=visible,
            excluded_node_ids={source_id, *goal_targets},
            allowed_surface_ids=allowed_surfaces,
        )
        if not staging_candidates:
            expected = " or ".join(sorted(allowed_surfaces))
            return {
                "eligible": False,
                "ineligible_reason": (
                    f"No visible canonical add-object surface ({expected}) exists "
                    "outside the source object's success targets; staging has no "
                    "fallback surface."
                ),
                "candidate_count": len(inheritable_placements),
            }
        staging = _ranked_candidates(
            staging_candidates,
            selection_key=selection_key,
            role="add_object_inherit_staging_surface",
        )[0]
        staging_relation = "ON"
        staging_target = staging["node_id"]
        staging_candidate_count = len(staging_candidates)
    template_key = _copy_template_key(source_id, source, object_templates)
    if template_key is None:
        raise ValueError(f"COPYABLE source {source_id!r} has no copy-object template")
    template = object_templates[template_key]
    added_object = _copy_identity_spec(template, source_node_id=source_id)
    if str(added_object["id"]) in nodes:
        raise ValueError(
            f"add_object template for {source_id!r} uses initial node id "
            f"{added_object['id']!r}"
        )
    added_id = str(added_object["id"])
    source_placement_facts = [
        (fact["predicate"], resolved_target)
        for fact in facts
        if fact["predicate"] in {"INSIDE", "ON"}
        and len(fact["args"]) >= 2
        and _resolve_initial_node_ref(nodes, fact["args"][0]) == source_id
        for resolved_target in [_resolve_initial_node_ref(nodes, fact["args"][1])]
        if resolved_target is not None
    ]
    source_placement_facts = list(dict.fromkeys(source_placement_facts))
    placement_targets = list(
        dict.fromkeys(target_id for _, target_id in source_placement_facts)
    )
    recovery_actions = [
        {
            "name": "putin" if relation == "INSIDE" else "puton",
            "node_ids": [added_id, target_id],
        }
        for relation, target_id in source_placement_facts
    ]
    design = {
        "eligible": True,
        "candidate_count": len(source_candidates) * staging_candidate_count,
        "source_candidate_count": len(source_candidates),
        "or_source_candidate_count": len(or_source_candidates),
        "or_source_priority_applied": bool(or_source_candidates),
        "staging_candidate_count": staging_candidate_count,
        "source_node_id": source_id,
        "object": added_object,
        "relation": staging_relation,
        "target": staging_target,
        "required_predicates": required_predicates,
        "recovery_action": recovery_actions[0],
        "recovery_actions": recovery_actions,
        "placement_alternatives": placement_targets,
    }
    source_parts = _direct_part_ids(source_id, nodes=nodes, edges=edges)
    if source_parts and any(
        fact["predicate"] == "ASSEMBLED"
        and fact["args"]
        and _resolve_initial_node_ref(nodes, fact["args"][0]) == source_id
        for fact in facts
    ):
        component_objects = []
        for source_part_id in source_parts:
            part_template_key = _copy_template_key(
                source_part_id,
                nodes[source_part_id],
                object_templates,
            )
            if part_template_key is None:
                raise ValueError(
                    f"COPYABLE assembly part {source_part_id!r} has no copy-object template"
                )
            part_object = _copy_identity_spec(
                object_templates[part_template_key],
                source_node_id=source_part_id,
                part_of=added_id,
            )
            component_objects.append(
                {
                    "object": part_object,
                    "relation": "ON",
                    "target": staging["node_id"],
                }
            )
        design["component_objects"] = component_objects
        design["component_node_ids"] = [
            str(item["object"]["id"]) for item in component_objects
        ]
        first_component_id = design["component_node_ids"][0]
        design["assembly_actions"] = [
            {
                "name": "attach",
                "node_ids": [component_id, first_component_id],
            }
            for component_id in design["component_node_ids"][1:]
        ]
        design["source_requires_assembly"] = True
    design["used_predefined_object_template"] = True
    design["copy_template_key"] = template_key
    return design


def _initial_container_counts(
    nodes: dict[str, dict[str, Any]], edges: list[dict[str, str]]
) -> dict[str, int]:
    counts = {node_id: 0 for node_id, node in nodes.items() if _is_container(node)}
    for edge in edges:
        if edge["relation"] == "INSIDE" and edge["to"] in counts:
            counts[edge["to"]] += 1
    return counts


def _node_max_items(node: dict[str, Any]) -> int | None:
    for field in ("max_items", "item_capacity", "max_capacity", "capacity"):
        raw_value = node.get(field)
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("max_items", raw_value.get("items", raw_value.get("max")))
        if isinstance(raw_value, bool):
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _capacity_trigger_candidates(
    *,
    actions: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Find containers that go from below capacity to max_items in teacher actions."""
    counts = _initial_container_counts(nodes, edges)
    locations = {
        edge["from"]: (edge["relation"], edge["to"])
        for edge in edges
        if edge["relation"] in {"ON", "INSIDE"}
    }
    maxima = {
        node_id: maximum
        for node_id, node in nodes.items()
        if _is_container(node)
        for maximum in [_node_max_items(node)]
        if maximum is not None and counts.get(node_id, 0) < maximum
    }
    reached: list[dict[str, Any]] = []
    reached_ids: set[str] = set()
    for action in sorted(actions, key=lambda item: item["step"]):
        if not action["node_ids"]:
            continue
        object_id = action["node_ids"][0]
        if action["name"] in {"grab", "pick"}:
            previous = locations.pop(object_id, None)
            if previous and previous[0] == "INSIDE" and previous[1] in counts:
                counts[previous[1]] = max(0, counts[previous[1]] - 1)
        elif action["name"] in PLACEMENT_RELATIONS and len(action["node_ids"]) >= 2:
            previous = locations.get(object_id)
            if previous and previous[0] == "INSIDE" and previous[1] in counts:
                counts[previous[1]] = max(0, counts[previous[1]] - 1)
            relation = PLACEMENT_RELATIONS[action["name"]]
            target_id = action["node_ids"][1]
            locations[object_id] = (relation, target_id)
            if relation == "INSIDE" and target_id in counts:
                counts[target_id] += 1
        for node_id, maximum in maxima.items():
            if node_id not in reached_ids and counts.get(node_id, 0) >= maximum:
                reached.append(
                    {
                        "node_id": node_id,
                        "max_items": maximum,
                        "teacher_step": action["step"],
                    }
                )
                reached_ids.add(node_id)
    return reached


def _add_object_existing_goal_design(
    *,
    actions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
    visible: set[str],
    selection_key: str,
    task_group: str,
) -> dict[str, Any]:
    """Design max-items add-object, or record why an old episode is ineligible."""
    absent_goal_facts = [
        fact
        for fact in facts
        if fact["predicate"] in {"ON", "INSIDE"}
        and len(fact["args"]) >= 2
        and _resolve_initial_node_ref(nodes, fact["args"][0]) is None
        and _resolve_initial_node_ref(nodes, fact["args"][1]) is not None
    ]
    capacity_triggers = _capacity_trigger_candidates(
        actions=actions,
        nodes=nodes,
        edges=edges,
    )
    candidate_counts = {
        "absent_task_goal_object": len(absent_goal_facts),
        "container_reaches_max_items": len(capacity_triggers),
        "safe_visible_staging_surface": 0,
    }
    missing_requirements = []
    if not absent_goal_facts:
        missing_requirements.append(
            "task completion has no object that is absent from the initial view graph"
        )
    if not capacity_triggers:
        missing_requirements.append(
            "no container goes from below capacity to max_items in the teacher trajectory"
        )
    if missing_requirements:
        return {
            "eligible": False,
            "ineligible_reason": "; ".join(missing_requirements),
            "candidate_counts": candidate_counts,
        }

    goal_fact = _ranked_candidates(
        absent_goal_facts,
        selection_key=selection_key,
        role="add_object_existing_task_goal",
    )[0]
    goal_target_id = _resolve_initial_node_ref(nodes, goal_fact["args"][1])
    allowed_surfaces = ADD_OBJECT_STAGING_SURFACES_BY_TASK_GROUP.get(
        _material_spec_group(task_group), ADD_OBJECT_STAGING_SURFACES
    )
    staging_candidates = _visible_staging_surfaces(
        nodes=nodes,
        edges=edges,
        visible=visible,
        excluded_node_ids={goal_target_id} if goal_target_id is not None else set(),
        allowed_surface_ids=allowed_surfaces,
    )
    candidate_counts["safe_visible_staging_surface"] = len(staging_candidates)
    if not staging_candidates:
        return {
            "eligible": False,
            "ineligible_reason": (
                "no visible canonical add-object staging surface is available for "
                f"{_material_spec_group(task_group)!r}; fallback surfaces are disabled"
            ),
            "candidate_counts": candidate_counts,
        }
    staging = _ranked_candidates(
        staging_candidates,
        selection_key=selection_key,
        role="add_object_existing_staging_surface",
    )[0]
    capacity_trigger = _ranked_candidates(
        capacity_triggers,
        selection_key=selection_key,
        role="add_object_existing_capacity_trigger",
    )[0]
    object_id = goal_fact["args"][0]
    return {
        "eligible": True,
        "candidate_counts": candidate_counts,
        "trigger": capacity_trigger,
        "object": {
            "id": object_id,
            "name": object_id,
            "category": "object",
            "properties": ["GRABBABLE", "MOVABLE"],
        },
        "relation": "ON",
        "target": staging["node_id"],
        "goal_fact": goal_fact,
        "recovery_action": {
            "name": "putin" if goal_fact["predicate"] == "INSIDE" else "puton",
            "node_ids": [object_id, goal_fact["args"][1]],
        },
    }


def _allowed_destinations(
    facts: list[dict[str, Any]], object_id: str, relation: str
) -> set[str]:
    return {
        fact["args"][1]
        for fact in facts
        if fact["predicate"] == relation
        and len(fact["args"]) >= 2
        and fact["args"][0] == object_id
    }


def _wrong_destination_candidates(
    *,
    placement: dict[str, Any],
    actions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    visible: set[str],
    best_only: bool = True,
) -> list[dict[str, Any]]:
    object_id, correct_target = placement["node_ids"][:2]
    relation = placement["goal_relation"]
    allowed = _allowed_destinations(facts, object_id, relation)
    observed_placement_targets = {
        action["node_ids"][1]
        for action in actions
        if action["name"] in PLACEMENT_RELATIONS and len(action["node_ids"]) >= 2
    }
    candidates: list[dict[str, Any]] = []
    for node_id, node in nodes.items():
        if node_id in allowed or node_id in {object_id, correct_target}:
            continue
        if relation == "INSIDE" and not _is_container(node):
            continue
        if relation == "ON" and not _is_surface(node):
            continue
        if node_id not in visible:
            continue
        # OPEN/CLOSED is deliberately not filtered here.  This function builds
        # the structural candidate pool, while the closed-loop runtime waits
        # until a candidate container is live, visible, open, and non-full.
        # Using a teacher or goal-derived reference state here would discard
        # valid opportunities before the runtime semantic trigger can see them.
        candidates.append(
            {
                "node_id": node_id,
                "relation": relation,
                "category": str(node.get("category") or ""),
                "observed_as_placement_target": node_id in observed_placement_targets,
                "static": "STATIC" in _node_properties(node),
            }
        )
    if not candidates or not best_only:
        return candidates
    best_plausibility = max(
        (
            int(candidate["observed_as_placement_target"]),
            int(candidate["static"]),
        )
        for candidate in candidates
    )
    return [
        candidate
        for candidate in candidates
        if (
            int(candidate["observed_as_placement_target"]),
            int(candidate["static"]),
        )
        == best_plausibility
    ]


def _occlusion_candidates(
    *,
    actions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sources: list[tuple[str, list[str]]] = []
    for node_id, node in nodes.items():
        if "OCCLUDER" not in _node_properties(node):
            continue
        supported_resolution_actions = []
        if _is_openable(node):
            supported_resolution_actions.append("open")
        if _is_movable(node):
            supported_resolution_actions.append("move_aside")
        if supported_resolution_actions:
            sources.append((node_id, supported_resolution_actions))

    strict_goal_objects = {
        fact["args"][0]
        for fact in facts
        if fact["predicate"] in {"INSIDE", "ON"}
        and fact["args"]
        and not fact["optional"]
    }
    goal_objects = strict_goal_objects or {
        fact["args"][0]
        for fact in facts
        if fact["predicate"] in {"INSIDE", "ON"}
        and fact["args"]
    }
    teacher_objects = {
        action["node_ids"][0]
        for action in actions
        if action["node_ids"] and action["name"] in {"putin", "puton"}
    }
    preferred_targets = goal_objects & teacher_objects
    if not preferred_targets:
        preferred_targets = teacher_objects
    active_pairs = {
        (edge["from"], edge["to"])
        for edge in edges
        if edge["relation"] in {"OCCLUDES", "BLOCKS", "COVERS"}
    }
    candidates: list[dict[str, Any]] = []
    for source_id, supported_resolution_actions in sources:
        for target_id in preferred_targets:
            if source_id == target_id or (source_id, target_id) in active_pairs:
                continue
            target = nodes.get(target_id)
            if target is None or "PART_OF" in _node_properties(target):
                continue
            candidates.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "relation": "OCCLUDES",
                    "supported_resolution_actions": supported_resolution_actions,
                }
            )
    return candidates


def _semantic_trigger(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "after_successful_action",
        "action": {"name": action["name"], "node_ids": action["node_ids"]},
        "apply": "before_next_observation",
        "teacher_reference_step": action["step"] + 1,
    }


def _failure_none() -> dict[str, Any]:
    return {"mode": "none"}


def _task_group(episode_id: str) -> str:
    return episode_id.rsplit("_", 1)[0] if "_" in episode_id else episode_id


def _with_goal_derived_reference(episode: dict[str, Any]) -> dict[str, Any]:
    """Add design-only successful actions when no aligned teacher path exists.

    These rows select semantic intervention candidates.  They are never exposed
    as a teacher rollout and ``teacher_reference_available`` remains false, so
    closed-loop efficiency is reported as unavailable rather than fabricated.
    """

    if _successful_actions(episode):
        return episode
    result = copy.deepcopy(episode)
    facts = _goal_facts(result.get("task_completion_criterion"))
    nodes = _node_map(result)
    actions: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    for fact in facts:
        if fact["predicate"] not in {"INSIDE", "ON"} or len(fact["args"]) < 2:
            continue
        object_id, target_id = fact["args"][:2]
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        actions.extend(
            [
                {"name": "grab", "node_ids": [object_id]},
                {
                    "name": "putin" if fact["predicate"] == "INSIDE" else "puton",
                    "node_ids": [object_id, target_id],
                },
            ]
        )

    state_action: dict[str, Any] | None = None
    for predicate, action_name in (("CLOSED", "close"), ("OPEN", "open")):
        state_fact = next(
            (
                fact
                for fact in facts
                if fact["predicate"] == predicate and fact["args"]
            ),
            None,
        )
        if state_fact is not None:
            state_action = {"name": action_name, "node_ids": [state_fact["args"][0]]}
            break
    if state_action is None:
        openable = next(
            (node_id for node_id, node in nodes.items() if _is_openable(node)),
            None,
        )
        if openable is not None:
            state_action = {"name": "open", "node_ids": [openable]}
    if state_action is None:
        raise ValueError(f"{result['episode_id']}: no regressible state action in direct source")
    actions.append(state_action)
    result["trajectory"] = [
        {
            "step": step,
            "action": action,
            "event": {"status": "success", "source": "goal_derived_design_reference"},
        }
        for step, action in enumerate(actions, start=1)
    ]
    result["teacher_reference_available"] = False
    result["teacher_reference_kind"] = "goal_derived_design_only"
    return result


def _task_collection_add_object_design(
    episode: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = episode.get("task_metadata")
    collection = metadata.get("collection_conditions") if isinstance(metadata, dict) else None
    raw_conditions = collection.get("conditions") if isinstance(collection, dict) else None
    candidates = [
        item
        for item in (raw_conditions or [])
        if isinstance(item, dict)
        and item.get("intervention_type") == "add_object"
        and isinstance(item.get("graph_disturbance"), dict)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{episode['episode_id']}: expected exactly one task metadata add-object "
            f"condition, got {len(candidates)}"
        )
    source_condition = copy.deepcopy(candidates[0])
    trigger = source_condition.get("trigger")
    disturbance = source_condition.get("graph_disturbance")
    if not isinstance(trigger, dict) or trigger.get("type") != "on_any_container_max_items_reached":
        raise ValueError(f"{episode['episode_id']}: unsupported collection trigger")
    node_ids = [str(value) for value in trigger.get("node_ids") or []]
    if not node_ids:
        raise ValueError(f"{episode['episode_id']}: collection trigger has no containers")
    if not isinstance(disturbance, dict):
        raise ValueError(f"{episode['episode_id']}: collection disturbance is missing")
    success_policy = disturbance.get("success_policy")
    if not isinstance(success_policy, dict) or success_policy.get("type") != "trigger_container_goal":
        raise ValueError(f"{episode['episode_id']}: collection policy must be trigger_container_goal")
    object_spec = disturbance.get("object")
    if not isinstance(object_spec, dict) or not object_spec.get("id"):
        raise ValueError(f"{episode['episode_id']}: collection object is invalid")
    selected_container = node_ids[0]
    design = {
        "eligible": True,
        "source": "task_metadata.collection_conditions",
        "candidate_count": 1,
        "candidate_counts": {"task_metadata_conditions": 1},
        "trigger": {
            "type": "on_any_container_max_items_reached",
            "node_ids": node_ids,
        },
        "object": copy.deepcopy(object_spec),
        "relation": str(disturbance.get("relation") or "ON"),
        "target": str(disturbance.get("target") or "桌面"),
        "recovery_action": {
            "name": "putin",
            "node_ids": [str(object_spec["id"]), selected_container],
        },
    }
    condition = {
        "condition_id": "add_object_existing_task_goal_at_capacity",
        "intervention_type": "add_object",
        "eligible": True,
        "failure_injection": _failure_none(),
        "success_policy_type": "trigger_container_goal",
        "eligibility_candidate_counts": design["candidate_counts"],
        "trigger": {
            **copy.deepcopy(trigger),
            "apply": "before_current_observation",
        },
        "graph_disturbance": copy.deepcopy(disturbance),
        "expected_effect": (
            f"When either {' or '.join(node_ids)} reaches max_items, "
            f"{object_spec['id']} is introduced and its INSIDE goal is bound to "
            "the triggering container."
        ),
        "expected_recovery": str(
            source_condition.get("expected_recovery")
            or "Make room if needed and place the introduced object inside the triggering container."
        ),
        "solvability_preserved": True,
        "source_condition_id": source_condition.get("condition_id"),
    }
    return design, condition


def _episode_design(
    episode: dict[str, Any],
    actions: list[dict[str, Any]],
    timeline_actions: list[dict[str, Any]],
    object_templates: dict[str, dict[str, Any]],
    *,
    inherit_copy_enabled: bool,
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    task_group = _task_group(episode_id)
    nodes = _node_map(episode)
    edges = _graph_edges(episode)
    visible = _visible_node_ids(episode)
    facts = _goal_facts(episode.get("task_completion_criterion"))
    selection_key = _episode_selection_key(episode, actions, facts)

    placements = _placement_candidates(actions, facts)
    if not placements:
        raise ValueError(f"{episode_id}: no successful goal-achieving putin/puton action")
    ranked_placements = _ranked_candidates(
        placements, selection_key=selection_key, role="completed_subgoal_rollback"
    )
    placement = ranked_placements[0]

    states = _state_candidates(actions, nodes)
    if not states:
        raise ValueError(f"{episode_id}: no successful regressible state-changing action")
    ranked_states = _ranked_candidates(
        states, selection_key=selection_key, role="state_regression"
    )
    state_action = ranked_states[0]

    surfaces = _root_surface_candidates(nodes, edges)
    if not surfaces:
        raise ValueError(f"{episode_id}: no surface available for subgoal rollback")
    correct_target = placement["node_ids"][1]
    rollback_surface = next(
        (item["node_id"] for item in surfaces if item["node_id"] != correct_target),
        None,
    )
    if rollback_surface is None:
        raise ValueError(f"{episode_id}: no distinct rollback surface")

    rollback_candidates: list[dict[str, Any]] = []
    seen_rollback_objects: set[str] = set()
    for candidate in ranked_placements:
        candidate_object, candidate_target = candidate["node_ids"][:2]
        if candidate_object in seen_rollback_objects:
            continue
        candidate_surfaces = [
            surface for surface in surfaces if surface["node_id"] != candidate_target
        ]
        if not candidate_surfaces:
            continue
        selected_surface = _ranked_candidates(
            candidate_surfaces,
            selection_key=selection_key,
            role=f"rollback_surface:{candidate_object}",
        )[0]
        rollback_candidates.append(
            {
                "node_id": candidate_object,
                "relation": "ON",
                "target": selected_surface["node_id"],
                "goal_action": {
                    "name": candidate["name"],
                    "node_ids": candidate["node_ids"],
                },
            }
        )
        seen_rollback_objects.add(candidate_object)
    if not rollback_candidates:
        raise ValueError(f"{episode_id}: no runtime rollback candidate")

    wrong_destinations = _wrong_destination_candidates(
        placement=placement,
        actions=timeline_actions,
        facts=facts,
        nodes=nodes,
        visible=visible,
    )
    if not wrong_destinations:
        raise ValueError(
            f"{episode_id}: no visible wrong destination outside the object's allowed goals"
        )
    wrong_destination = _ranked_candidates(
        wrong_destinations, selection_key=selection_key, role="wrong_destination"
    )[0]
    wrong_relocation_candidates: list[dict[str, Any]] = []
    seen_wrong_specs: set[tuple[str, str, str]] = set()
    for candidate in ranked_placements:
        candidate_destinations = _wrong_destination_candidates(
            placement=candidate,
            actions=timeline_actions,
            facts=facts,
            nodes=nodes,
            visible=visible,
            best_only=False,
        )
        ranked_destinations = _ranked_candidates(
            candidate_destinations,
            selection_key=selection_key,
            role=f"wrong_destination:{candidate['node_ids'][0]}",
        )
        for destination in ranked_destinations:
            spec_key = (
                candidate["node_ids"][0],
                destination["relation"],
                destination["node_id"],
            )
            if spec_key in seen_wrong_specs:
                continue
            seen_wrong_specs.add(spec_key)
            wrong_relocation_candidates.append(
                {
                    "node_id": candidate["node_ids"][0],
                    "relation": destination["relation"],
                    "target": destination["node_id"],
                    "goal_action": {
                        "name": candidate["name"],
                        "node_ids": candidate["node_ids"],
                    },
                }
            )
    if not wrong_relocation_candidates:
        raise ValueError(f"{episode_id}: no runtime wrong-relocation candidate")

    occlusions = _occlusion_candidates(
        actions=actions,
        facts=facts,
        nodes=nodes,
        edges=edges,
    )
    if not occlusions:
        raise ValueError(f"{episode_id}: no resolvable new occlusion pair")
    ranked_occlusions = _ranked_candidates(
        occlusions, selection_key=selection_key, role="add_occlusion"
    )
    occlusion = ranked_occlusions[0]

    if inherit_copy_enabled:
        add_object_placements = _placement_candidates(
            actions,
            facts,
            prefer_strict=False,
        )
        add_object_inherit = _add_object_inherit_design(
            placements=add_object_placements,
            facts=facts,
            nodes=nodes,
            edges=edges,
            visible=visible,
            selection_key=selection_key,
            object_templates=object_templates,
            task_group=task_group,
        )
    else:
        add_object_inherit = {
            "eligible": False,
            "disabled": True,
            "ineligible_reason": (
                "Task group explicitly disables add_object_inherit_source_goal."
            ),
            "candidate_count": 0,
        }
    add_object_existing_goal = _add_object_existing_goal_design(
        actions=timeline_actions,
        facts=facts,
        nodes=nodes,
        edges=edges,
        visible=visible,
        selection_key=selection_key,
        task_group=task_group,
    )

    return {
        "selection_key": selection_key,
        "facts": facts,
        "placement": placement,
        "placement_candidates": ranked_placements,
        "state_action": state_action,
        "state_candidates": ranked_states,
        "rollback_surface": rollback_surface,
        "rollback_candidates": rollback_candidates,
        "wrong_destination": wrong_destination,
        "wrong_relocation_candidates": wrong_relocation_candidates,
        "occlusion": occlusion,
        "occlusion_candidates": ranked_occlusions,
        "add_object_inherit": add_object_inherit,
        "add_object_existing_goal": add_object_existing_goal,
        "candidate_counts": {
            "placement": len(placements),
            "state_regression": len(states),
            "wrong_destination": len(wrong_destinations),
            "add_occlusion": len(occlusions),
            "add_object_inherit": add_object_inherit["candidate_count"],
        },
    }


def build_manifest(
    source: Path | None,
    episode: dict[str, Any],
    *,
    manifest_source: dict[str, Any] | None = None,
    add_object_mode: str = "auto",
) -> dict[str, Any]:
    if add_object_mode not in {"auto", "inherit", "task_collection"}:
        raise ValueError(f"unsupported add_object_mode: {add_object_mode!r}")
    episode = _with_goal_derived_reference(episode)
    episode_id = str(episode["episode_id"])
    actions = _successful_actions(episode)
    if not actions:
        raise ValueError(f"{episode_id}: no successful teacher actions")
    timeline_actions = _successful_actions(episode, include_aligned_actions=True)
    task_group = _task_group(episode_id)
    inherit_copy_enabled = _inherit_copy_enabled(task_group)
    object_templates = _load_add_object_templates(task_group)
    design = _episode_design(
        episode,
        actions,
        timeline_actions,
        object_templates,
        inherit_copy_enabled=inherit_copy_enabled,
    )
    placement = design["placement"]
    object_id, correct_target = placement["node_ids"][:2]
    correct_action = {"name": placement["name"], "node_ids": placement["node_ids"]}
    state_action = design["state_action"]
    state_node = state_action["node_ids"][0]
    state_field = state_action["state_field"]
    state_value = state_action["regressed_value"]
    state_regression_candidates = [
        {
            "node_id": candidate["node_ids"][0],
            "values": {candidate["state_field"]: candidate["regressed_value"]},
            "achieved_values": {
                candidate["state_field"]: not candidate["regressed_value"]
            },
            "recovery_action": {
                "name": candidate["name"],
                "node_ids": candidate["node_ids"],
            },
        }
        for candidate in design["state_candidates"]
    ]
    wrong = design["wrong_destination"]
    occlusion = design["occlusion"]
    occlusion_candidates = design["occlusion_candidates"]
    add_object_inherit = design["add_object_inherit"]
    add_object_existing_goal = design["add_object_existing_goal"]

    failure_seed = int(design["selection_key"][:8], 16)
    inherit_condition: dict[str, Any] = {
        "condition_id": "add_object_inherit_source_goal",
        "intervention_type": "add_object",
        "eligible": bool(add_object_inherit["eligible"]),
        "failure_injection": _failure_none(),
        "success_policy_type": "inherit_from",
    }
    if add_object_inherit["eligible"]:
        inherited_object = add_object_inherit["object"]
        inherited_recovery = add_object_inherit["recovery_action"]
        inherit_trigger: dict[str, Any] = {
            "type": "on_object_goal_satisfied",
            "node_id": add_object_inherit["source_node_id"],
            "apply": "before_current_observation",
        }
        if add_object_inherit.get("required_predicates"):
            inherit_trigger["required_predicates"] = copy.deepcopy(
                add_object_inherit["required_predicates"]
            )
        inherit_disturbance: dict[str, Any] = {
            "operation": "add_object",
            "object": inherited_object,
            "relation": add_object_inherit["relation"],
            "target": add_object_inherit["target"],
            "success_policy": {
                "type": "inherit_from",
                "source_node_id": add_object_inherit["source_node_id"],
                "placement_alternatives": copy.deepcopy(
                    add_object_inherit.get("placement_alternatives", [])
                ),
            },
        }
        if add_object_inherit.get("component_objects"):
            inherit_disturbance["component_objects"] = copy.deepcopy(
                add_object_inherit["component_objects"]
            )
        assembly_text = (
            "assemble the added parts, "
            if add_object_inherit.get("assembly_actions")
            else ""
        )
        spawn_preposition = (
            "inside" if add_object_inherit["relation"] == "INSIDE" else "on"
        )
        inherit_condition.update(
            {
                "trigger": inherit_trigger,
                "graph_disturbance": inherit_disturbance,
                "expected_effect": (
                    f"After {add_object_inherit['source_node_id']} fully satisfies its "
                    "object-local success criterion, "
                    f"a new same-class object {inherited_object['id']} appears "
                    f"{spawn_preposition} "
                    f"{add_object_inherit['target']} and strictly inherits that same "
                    "local success expression."
                ),
                "expected_recovery": (
                    f"Open the destination if needed, {assembly_text}grab "
                    f"{inherited_object['id']}, and place it in any non-full inherited "
                    f"destination: "
                    + " or ".join(
                        f"{action['name']}({', '.join(action['node_ids'])})"
                        for action in add_object_inherit["recovery_actions"]
                    )
                    + "."
                ),
                "solvability_preserved": True,
            }
        )
    else:
        inherit_condition.update(
            {
                "graph_disturbance": None,
                "ineligible_reason": add_object_inherit["ineligible_reason"],
            }
        )

    existing_goal_condition: dict[str, Any] = {
        "condition_id": "add_object_existing_task_goal_at_capacity",
        "intervention_type": "add_object",
        "eligible": bool(add_object_existing_goal["eligible"]),
        "failure_injection": _failure_none(),
        "success_policy_type": "existing_task_goal",
        "eligibility_candidate_counts": add_object_existing_goal["candidate_counts"],
    }
    if add_object_existing_goal["eligible"]:
        capacity_trigger = add_object_existing_goal["trigger"]
        existing_object = add_object_existing_goal["object"]
        existing_goal_condition.update(
            {
                "trigger": {
                    "type": "on_container_max_items_reached",
                    "node_id": capacity_trigger["node_id"],
                    "apply": "before_current_observation",
                    "teacher_reference_step": capacity_trigger["teacher_step"] + 1,
                },
                "graph_disturbance": {
                    "operation": "add_object",
                    "object": existing_object,
                    "relation": add_object_existing_goal["relation"],
                    "target": add_object_existing_goal["target"],
                    "success_policy": {"type": "existing_task_goal"},
                },
                "expected_effect": (
                    f"When {capacity_trigger['node_id']} reaches max_items="
                    f"{capacity_trigger['max_items']}, task object {existing_object['id']} "
                    "is introduced from outside the initial view graph."
                ),
                "expected_recovery": (
                    "Place the introduced object according to its success rule already "
                    "present in task_completion_criterion."
                ),
                "solvability_preserved": True,
            }
        )
    else:
        existing_goal_condition.update(
            {
                "graph_disturbance": None,
                "ineligible_reason": add_object_existing_goal["ineligible_reason"],
            }
        )

    if add_object_mode == "inherit":
        if inherit_condition.get("eligible") is not True:
            raise ValueError(
                f"{episode_id}: add_object_inherit_source_goal must be eligible"
            )
        add_object_existing_goal = {
            **copy.deepcopy(add_object_existing_goal),
            "eligible": False,
            "disabled": True,
            "ineligible_reason": (
                "Batch policy enables add_object_inherit_source_goal for this task group."
            ),
        }
        existing_goal_condition = {
            "condition_id": "add_object_existing_task_goal_at_capacity",
            "intervention_type": "add_object",
            "eligible": False,
            "failure_injection": _failure_none(),
            "success_policy_type": "existing_task_goal",
            "eligibility_candidate_counts": add_object_existing_goal.get(
                "candidate_counts", {}
            ),
            "graph_disturbance": None,
            "ineligible_reason": add_object_existing_goal["ineligible_reason"],
        }
    elif add_object_mode == "task_collection":
        add_object_existing_goal, existing_goal_condition = (
            _task_collection_add_object_design(episode)
        )
        if inherit_condition.get("eligible") is not False:
            raise ValueError(
                f"{episode_id}: toy task must disable add_object_inherit_source_goal"
            )

    conditions = [
        {
            "condition_id": "baseline",
            "intervention_type": "baseline",
            "eligible": True,
            "failure_injection": _failure_none(),
            "graph_disturbance": None,
            "expected_challenge": "none",
        },
        {
            "condition_id": "action_failure_once_per_action_type",
            "intervention_type": "action_failure",
            "eligible": True,
            "failure_injection": {
                "mode": "all",
                "actions": ["all"],
                "probability": 0.0,
                "max_failures_per_episode": len(INJECTABLE_ACTIONS),
                "seed": failure_seed,
                "only_normally_successful_actions": True,
                "deduplication_scope": "action_name",
            },
            "graph_disturbance": None,
            "trigger": {"type": "first_normally_successful_action_of_each_type"},
            "expected_challenge": (
                "Each normally successful action type fails at most once; detect every "
                "failure, ground its objects, then retry or replan."
            ),
        },
        {
            "condition_id": "state_regression",
            "intervention_type": "set_state",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": {
                "type": "first_eligible_state_regression_opportunity",
                "minimum_step": 2,
                "apply": "before_current_observation",
            },
            "graph_disturbance": {
                "operation": "set_state",
                "selection": "runtime_first_eligible_state_regression",
                "node_id": state_node,
                "values": {state_field: state_value},
                "candidate_regressions": state_regression_candidates,
            },
            "expected_effect": (
                "The first achieved regressible state among the episode-specific "
                "candidates is externally reset exactly once."
            ),
            "expected_recovery": (
                "Recognize the live regressed node and repeat the corresponding state action."
            ),
            "solvability_preserved": True,
        },
        {
            "condition_id": "completed_subgoal_rollback",
            "intervention_type": "relocate",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": {
                "type": "first_satisfied_goal_placement_opportunity",
                "minimum_step": 2,
                "apply": "before_current_observation",
            },
            "graph_disturbance": {
                "operation": "relocate",
                "selection": "runtime_first_satisfied_goal_placement",
                "node_id": object_id,
                "relation": "ON",
                "target": design["rollback_surface"],
                "candidate_relocations": copy.deepcopy(
                    design["rollback_candidates"]
                ),
            },
            "expected_effect": (
                "The first currently satisfied object-placement subgoal is reverted "
                "onto an episode-specific staging surface."
            ),
            "expected_recovery": (
                "Use the live previous_location recorded by the intervention, grab the "
                "relocated object, and restore that ON/INSIDE goal branch."
            ),
            "solvability_preserved": True,
        },
        {
            "condition_id": "wrong_container_relocation",
            "intervention_type": "relocate",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": {
                "type": "first_satisfied_goal_wrong_destination_opportunity",
                "minimum_step": 2,
                "apply": "before_current_observation",
            },
            "graph_disturbance": {
                "operation": "relocate",
                "selection": "runtime_first_satisfied_goal_wrong_destination",
                "node_id": object_id,
                "relation": wrong["relation"],
                "target": wrong["node_id"],
                "candidate_relocations": copy.deepcopy(
                    design["wrong_relocation_candidates"]
                ),
            },
            "expected_effect": (
                "The first currently satisfied placement with an eligible alternative is "
                "moved to a wrong but visible and plausible destination."
            ),
            "expected_recovery": (
                "Detect the live wrong destination, grab the object, and restore the "
                "recorded previous ON/INSIDE goal branch."
            ),
            "solvability_preserved": True,
        },
        {
            "condition_id": "add_occlusion",
            "intervention_type": "add_occlusion",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": {
                "type": "first_eligible_occlusion_opportunity",
                "minimum_step": 8,
                "min_goal_progress": 0.4,
                "max_goal_progress": 0.95,
                "apply": "before_current_observation",
            },
            "graph_disturbance": {
                "operation": "add_occlusion",
                "selection": "runtime_prefer_open_then_first_eligible",
                "candidate_pairs": occlusion_candidates,
            },
            "expected_effect": (
                "At the first eligible mid-to-late episode opportunity, one completed placement "
                "is moved onto the occluder's support surface and hidden by that occluder."
            ),
            "expected_recovery": (
                "Infer the runtime-selected blocker from the visible graph, execute its "
                "available open or move_aside resolution, then restore the object's "
                "recorded ON/INSIDE goal placement. If opening the blocker changed its "
                "required final state, close it again afterward."
            ),
            "solvability_preserved": True,
        },
    ]
    # Keep a stable condition matrix across task groups. Disabled copy tasks
    # expose the condition as eligible=false with no graph disturbance.
    conditions.append(inherit_condition)
    conditions.append(existing_goal_condition)
    design_summary = {
        "generation_algorithm": "episode_semantic_v7_or_inheritance_priority",
        "selection_key": design["selection_key"],
        "task_group": task_group,
        "state_regression_action": {
            "step": state_action["step"],
            "name": state_action["name"],
            "node_ids": state_action["node_ids"],
        },
        "placement_action": correct_action,
        "placement_teacher_step": placement["step"],
        "placement_is_strict_goal": not placement["goal_optional"],
        "rollback_surface": design["rollback_surface"],
        "wrong_destination": wrong,
        "occlusion": occlusion,
        "occlusion_candidates": occlusion_candidates,
        "occlusion_trigger": {
            "minimum_step": 8,
            "min_goal_progress": 0.4,
            "max_goal_progress": 0.95,
        },
        "add_object_inherit": add_object_inherit,
        "add_object_existing_goal": add_object_existing_goal,
        "candidate_counts": design["candidate_counts"],
    }
    design_summary["design_fingerprint"] = _sha256_json(
        {
            "state": design_summary["state_regression_action"],
            "placement": correct_action,
            "rollback_surface": design_summary["rollback_surface"],
            "wrong_destination": wrong,
            "occlusion_candidates": occlusion_candidates,
            "occlusion_trigger": design_summary["occlusion_trigger"],
            "add_object_inherit": add_object_inherit,
            "add_object_existing_goal": add_object_existing_goal,
        }
    )
    if manifest_source is None:
        if source is None:
            raise ValueError("source is required when manifest_source is not provided")
        manifest_source = {
            "source_type": "aligned_episode",
            "aligned_episode": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "episode_id": episode_id,
            "scene_id": str(episode.get("scene_id") or episode_id),
            "env_id": str(episode.get("env_id") or episode_id),
            "teacher_reference_available": True,
        }
    return {
        "manifest_version": 5,
        "manifest_type": "closed_loop_intervention_suite",
        "suite_id": f"{episode_id}_interventions_v5",
        "source": copy.deepcopy(manifest_source),
        "execution_policy": {
            "reset_from_initial_view_graph_for_each_condition": True,
            "conditions_are_mutually_exclusive": True,
            "max_graph_interventions_per_rollout": 1,
            "failure_injection_policy": "once_per_action_name",
            "include_baseline": True,
            "graph_disturbance_timing": "before_current_step_observation",
            "primary_trigger_policy": (
                "semantic_with_runtime_occlusion_and_dynamic_add_object"
            ),
        },
        "design": design_summary,
        "conditions": conditions,
        "reporting": {
            "group_by": ["condition_id", "includes_valid_actions", "model_name"],
            "primary_metrics": [
                "task_success_rate",
                "normalized_goal_progress",
                "teacher_normalized_efficiency",
            ],
            "intervention_metrics": [
                "intervention_applied_rate",
                "average_intervention_count",
            ],
        },
    }


def validate_manifest_semantics(episode: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Replay teacher successes and verify every generated graph challenge is recoverable."""
    from auto_embodied_task.goal import evaluate_goal_expression
    from auto_embodied_task.harness import SymbolicBackend
    from auto_embodied_task.real_observation_eval import _parsed_action, _replay_inputs
    from auto_embodied_task.view_graph_rollout_eval import (
        _ManifestInterventionRuntime,
        _manifest_goal_for_node,
        _relaxed_completion_cost,
    )

    episode = _with_goal_derived_reference(episode)
    task, graph, constraints = _replay_inputs(episode)
    graph_conditions = [
        condition
        for condition in manifest["conditions"]
        if condition.get("eligible") is not False
        and condition.get("graph_disturbance") is not None
    ]
    design = manifest["design"]
    validation_goal_facts = _goal_facts(episode.get("task_completion_criterion"))
    direct_design = episode.get("teacher_reference_available") is False
    for condition in graph_conditions:
        backend = SymbolicBackend(graph, task, constraints)
        initial_completion_cost = _relaxed_completion_cost(backend)
        runtime = _ManifestInterventionRuntime(
            condition,
            initial_completion_cost=initial_completion_cost,
        )
        reports = runtime.before_step(backend, step_number=1, history=[])
        last_success: dict[str, Any] | None = None
        condition_id = condition["condition_id"]
        if direct_design and not reports and condition_id == "state_regression":
            candidate_regressions = condition["graph_disturbance"].get(
                "candidate_regressions", []
            )
            for candidate in candidate_regressions:
                if not isinstance(candidate, dict):
                    continue
                trial_backend = copy.deepcopy(backend)
                state_node = trial_backend.world.resolve_node_id(
                    candidate.get("node_id")
                )
                achieved_values = candidate.get("achieved_values")
                if state_node is None or not isinstance(achieved_values, dict):
                    continue
                state = trial_backend.world.states[state_node]
                if any(not hasattr(state, field) for field in achieved_values):
                    continue
                for field, value in achieved_values.items():
                    setattr(state, field, value)
                trial_condition = copy.deepcopy(condition)
                trial_condition["graph_disturbance"]["candidate_regressions"] = [
                    copy.deepcopy(candidate)
                ]
                trial_runtime = _ManifestInterventionRuntime(
                    trial_condition,
                    initial_completion_cost=initial_completion_cost,
                )
                trial_reports = trial_runtime.before_step(
                    trial_backend,
                    step_number=2,
                    history=[],
                )
                if trial_reports:
                    backend = trial_backend
                    runtime = trial_runtime
                    reports = trial_reports
                    break
        elif (
            direct_design
            and not reports
            and condition_id in {"completed_subgoal_rollback", "wrong_container_relocation"}
        ):
            candidate_relocations = condition["graph_disturbance"].get(
                "candidate_relocations", []
            )
            for candidate in candidate_relocations:
                if not isinstance(candidate, dict):
                    continue
                placement_action = candidate.get("goal_action")
                if not isinstance(placement_action, dict):
                    continue
                placement_nodes = [
                    str(value) for value in placement_action.get("node_ids") or []
                ]
                placement_name = str(placement_action.get("name") or "").lower()
                if placement_name not in PLACEMENT_RELATIONS or len(placement_nodes) < 2:
                    continue
                object_id, target_id = placement_nodes[:2]
                trial_backend = copy.deepcopy(backend)
                if (
                    object_id not in trial_backend.world.states
                    or target_id not in trial_backend.world.states
                ):
                    continue
                object_state = trial_backend.world.states[object_id]
                trial_backend.world._remove_occlusions_for_location_change(object_id)
                trial_backend.world.memory_hidden.pop(object_id, None)
                object_state.location_relation = PLACEMENT_RELATIONS[placement_name]
                object_state.location_target = target_id
                object_state.held = False
                if any(
                    fact["predicate"] == "ASSEMBLED"
                    and fact["args"][:1] == [object_id]
                    for fact in validation_goal_facts
                ):
                    object_state.assembled = True
                    for part_id in trial_backend.world._direct_part_ids(object_id):
                        trial_backend.world._clear_part_spatial_state(part_id)
                target_state = trial_backend.world.states[target_id]
                if target_state.node.is_openable:
                    target_state.open = True
                wrong_target = trial_backend.world.resolve_node_id(candidate.get("target"))
                if wrong_target is not None:
                    wrong_target_state = trial_backend.world.states[wrong_target]
                    if wrong_target_state.node.is_openable:
                        wrong_target_state.open = True
                trial_condition = copy.deepcopy(condition)
                trial_condition["graph_disturbance"]["candidate_relocations"] = [
                    copy.deepcopy(candidate)
                ]
                trial_runtime = _ManifestInterventionRuntime(
                    trial_condition,
                    initial_completion_cost=initial_completion_cost,
                )
                trial_reports = trial_runtime.before_step(
                    trial_backend,
                    step_number=2,
                    history=[],
                )
                if trial_reports:
                    backend = trial_backend
                    runtime = trial_runtime
                    reports = trial_reports
                    break
        elif direct_design and not reports and condition_id == "add_occlusion":
            candidates = condition["graph_disturbance"].get("candidate_pairs") or []
            preferred_target = str(candidates[0]["target"]) if candidates else ""
            placement_facts = [
                fact
                for fact in validation_goal_facts
                if fact["predicate"] in {"INSIDE", "ON"}
                and len(fact["args"]) >= 2
                and fact["args"][0] in backend.world.states
                and fact["args"][1] in backend.world.states
            ]
            placement_facts.sort(
                key=lambda fact: (fact["args"][0] != preferred_target, fact["optional"])
            )
            minimum_progress = float(condition["trigger"].get("min_goal_progress", 0.1))
            maximum_progress = float(condition["trigger"].get("max_goal_progress", 0.8))
            for fact in placement_facts:
                object_id, target_id = fact["args"][:2]
                object_state = backend.world.states[object_id]
                backend.world._remove_occlusions_for_location_change(object_id)
                backend.world.memory_hidden.pop(object_id, None)
                object_state.location_relation = fact["predicate"]
                object_state.location_target = target_id
                object_state.held = False
                if any(
                    state_fact["predicate"] == "ASSEMBLED"
                    and state_fact["args"][:1] == [object_id]
                    for state_fact in validation_goal_facts
                ):
                    object_state.assembled = True
                    for part_id in backend.world._direct_part_ids(object_id):
                        backend.world._clear_part_spatial_state(part_id)
                target_state = backend.world.states[target_id]
                if target_state.node.is_openable:
                    target_state.open = True
                current_cost = _relaxed_completion_cost(backend)
                progress = (
                    (initial_completion_cost - current_cost) / initial_completion_cost
                    if initial_completion_cost > 0
                    else 1.0
                )
                if minimum_progress <= progress <= maximum_progress:
                    reports = runtime.before_step(
                        backend,
                        step_number=max(3, int(condition["trigger"].get("minimum_step", 3))),
                        history=[],
                    )
                    if reports:
                        break
        if (
            direct_design
            and not reports
            and condition_id == "add_object_inherit_source_goal"
        ):
            add_design = design["add_object_inherit"]
            recovery = add_design["recovery_action"]
            source_id = add_design["source_node_id"]
            target_id = recovery["node_ids"][1]
            source_state = backend.world.states[source_id]
            source_state.location_relation = PLACEMENT_RELATIONS[recovery["name"]]
            source_state.location_target = target_id
            source_state.held = False
            if add_design.get("source_requires_assembly"):
                source_state.assembled = True
                for part_id in backend.world._direct_part_ids(source_id):
                    backend.world._clear_part_spatial_state(part_id)
            reports = runtime.before_step(
                backend,
                step_number=int(design["placement_teacher_step"]) + 1,
                history=[],
            )
            required_predicates = condition.get("trigger", {}).get(
                "required_predicates", []
            )
            for required in required_predicates if not reports else []:
                predicate = _normalize_relation(required.get("predicate"))
                args = required.get("args") or []
                required_node_id = (
                    backend.world.resolve_node_id(args[0]) if args else None
                )
                if required_node_id is None:
                    raise ValueError(
                        f"{episode['episode_id']}/{condition_id}: unknown required "
                        f"predicate node {args!r}"
                    )
                if predicate not in {"OPEN", "CLOSED"}:
                    raise ValueError(
                        f"{episode['episode_id']}/{condition_id}: semantic validation "
                        f"does not support required predicate {predicate!r}"
                    )
                backend.world.states[required_node_id].open = predicate == "OPEN"
            if required_predicates and not reports:
                reports = runtime.before_step(
                    backend,
                    step_number=int(design["placement_teacher_step"]) + 2,
                    history=[],
                )
        elif (
            not reports
            and condition_id == "add_object_existing_task_goal_at_capacity"
        ):
            trigger_design = design["add_object_existing_goal"]["trigger"]
            trigger_ids = trigger_design.get("node_ids") or [trigger_design.get("node_id")]
            trigger_id = str(next(value for value in trigger_ids if value))
            trigger_state = backend.world.states[trigger_id]
            maximum = trigger_state.node.max_items
            if maximum is not None:
                for candidate_id, candidate_state in backend.world.states.items():
                    if backend.world._container_item_count(trigger_id) >= maximum:
                        break
                    if (
                        candidate_id == trigger_id
                        or not candidate_state.node.is_movable
                        or candidate_state.held
                    ):
                        continue
                    candidate_state.location_relation = "INSIDE"
                    candidate_state.location_target = trigger_id
                reports = runtime.before_step(
                    backend,
                    step_number=int(trigger_design.get("teacher_step", 0)) + 1,
                    history=[],
                )
        if not reports:
            for step in episode.get("trajectory") or []:
                action = step.get("action") if isinstance(step, dict) else None
                event = step.get("event") if isinstance(step, dict) else None
                if not isinstance(action, dict):
                    continue
                name = str(action.get("base_name") or action.get("name") or "").lower()
                status = event.get("status") if isinstance(event, dict) else None
                if status not in {None, "success"} or name in {"recover", "stop"}:
                    continue
                replay_event = backend.step(_parsed_action(action))
                if replay_event.get("status") != "success":
                    # Real-alignment rows can be physically successful while the
                    # symbolic view graph cannot reproduce an unrelated intermediate
                    # action exactly (for example, a part remains occluded here but
                    # was reachable on the robot). Keep replaying; the validation
                    # still fails below unless the condition's own semantic trigger
                    # is reached and its recovery succeeds.
                    continue
                last_success = {"step": step.get("step"), "action": action, "event": replay_event}
                reports = runtime.before_step(
                    backend,
                    step_number=int(step.get("step") or 0) + 1,
                    history=[last_success],
                )
                if reports:
                    break
        if (
            not direct_design
            and not reports
            and condition_id == "add_object_inherit_source_goal"
        ):
            # A real-alignment action can be physically successful while the
            # symbolic replay still considers one assembly part occluded.  Do
            # not mutate the source directly: finish its local goal through the
            # same executable actions available to a closed-loop model.
            add_design = design["add_object_inherit"]
            source_id = str(add_design["source_node_id"])
            recovery = add_design["recovery_action"]
            target_id = str(recovery["node_ids"][1])
            source_recovery = {
                "name": recovery["name"],
                "node_ids": [source_id, target_id],
            }

            def resolve_blockers(node_id: str) -> bool:
                for _ in range(len(backend.world.states)):
                    blockers = backend.world._active_blockers(node_id)
                    if not blockers:
                        return True
                    made_progress = False
                    for blocker_id in blockers:
                        active_edge = next(
                            (
                                edge
                                for edge in backend.world.active_occlusion_edges
                                if edge[0] == blocker_id and edge[1] == node_id
                            ),
                            None,
                        )
                        if active_edge is None:
                            continue
                        resolution_action = (
                            backend.world._occlusion_edge_resolution_action(active_edge)
                        )
                        if resolution_action not in {"open", "close", "move_aside"}:
                            continue
                        event = backend.step(
                            _parsed_action(
                                {
                                    "name": resolution_action,
                                    "node_ids": [blocker_id],
                                }
                            )
                        )
                        made_progress = event.get("status") == "success" or made_progress
                    if not made_progress:
                        return False
                return not backend.world._active_blockers(node_id)

            source_state = backend.world.states[source_id]
            if add_design.get("source_requires_assembly") and not source_state.assembled:
                source_parts = backend.world._direct_part_ids(source_id)
                for part_id in source_parts:
                    resolve_blockers(part_id)
                if source_parts:
                    anchor_id = source_parts[0]
                    for part_id in source_parts[1:]:
                        attach_event = backend.step(
                            _parsed_action(
                                {
                                    "name": "attach",
                                    "node_ids": [part_id, anchor_id],
                                }
                            )
                        )
                        if attach_event.get("status") != "success":
                            break

            target_state = backend.world.states[target_id]
            if target_state.node.is_openable and not target_state.open:
                resolve_blockers(target_id)
                backend.step(_parsed_action({"name": "open", "node_ids": [target_id]}))
            resolve_blockers(source_id)
            if not backend.world.states[source_id].held:
                backend.step(
                    _parsed_action({"name": "grab", "node_ids": [source_id]})
                )
            if backend.world.states[source_id].held:
                backend.step(_parsed_action(source_recovery))
            reports = runtime.before_step(
                backend,
                step_number=int(design["placement_teacher_step"]) + 1,
                history=[],
            )
        if not reports:
            raise ValueError(
                f"{episode['episode_id']}/{condition['condition_id']}: trigger was not reached"
            )
        report = reports[0]
        if condition_id == "state_regression":
            runtime_selection = report.get("runtime_selection") or {}
            recovery = runtime_selection.get("recovery_action")
            if not isinstance(recovery, dict):
                recovery = design["state_regression_action"]
            recovery_event = backend.step(_parsed_action(recovery))
            if recovery_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery failed: {recovery_event}"
                )
        elif condition_id in {"completed_subgoal_rollback", "wrong_container_relocation"}:
            runtime_selection = report.get("runtime_selection") or {}
            placement = runtime_selection.get("recovery_action")
            if not isinstance(placement, dict):
                placement = design["placement_action"]
            object_id = placement["node_ids"][0]
            if not backend.world.is_visible(object_id):
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: relocated object {object_id} "
                    "is not visible"
                )
            grab_event = backend.step(_parsed_action({"name": "grab", "node_ids": [object_id]}))
            if grab_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery grab failed: {grab_event}"
                )
            placement_event = backend.step(_parsed_action(placement))
            if placement_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery placement failed: "
                    f"{placement_event}"
                )
        elif condition_id == "add_occlusion":
            disturbance = report.get("spec") if isinstance(report.get("spec"), dict) else {}
            target = disturbance.get("target")
            details = report.get("details") if isinstance(report.get("details"), dict) else {}
            occlusion = details.get("occlusion") if isinstance(details.get("occlusion"), dict) else {}
            if (
                report.get("step") == 1
                or not target
                or report.get("operation") != "relocate_and_add_occlusion"
                or occlusion.get("active_after") is not True
                or backend.world.is_visible(target)
                or report.get("goal_completion_cost_after", 0)
                <= report.get("goal_completion_cost_before", 0)
                or report.get("planning_cost_after", 0)
                <= report.get("planning_cost_before", 0)
            ):
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: completed placement was not "
                    "relocated and actively hidden"
                )
            recovery_event = backend.step(
                _parsed_action(
                    {
                        "name": disturbance["resolution_action"],
                        "node_ids": [disturbance["source"]],
                    }
                )
            )
            if recovery_event.get("status") != "success" or not backend.world.is_visible(target):
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: occlusion recovery failed: "
                    f"{recovery_event}"
                )
            grab_event = backend.step(
                _parsed_action({"name": "grab", "node_ids": [target]})
            )
            if grab_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery grab failed: "
                    f"{grab_event}"
                )
            previous_location = disturbance.get("previous_location") or {}
            placement_action = (
                "putin" if previous_location.get("relation") == "INSIDE" else "puton"
            )
            placement_event = backend.step(
                _parsed_action(
                    {
                        "name": placement_action,
                        "node_ids": [target, previous_location.get("target")],
                    }
                )
            )
            if placement_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery placement failed: "
                    f"{placement_event}"
                )
        elif condition_id in {
            "add_object_inherit_source_goal",
            "add_object_existing_task_goal_at_capacity",
        }:
            details = report.get("details") if isinstance(report.get("details"), dict) else {}
            object_id = details.get("node_id")
            if not object_id or object_id not in backend.world.states:
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: added object is missing"
                )
            expression, _ = _manifest_goal_for_node(
                backend,
                object_id,
                field="validation.object_id",
            )
            if evaluate_goal_expression(
                expression,
                backend.evaluator._predicate_met,
            ).success:
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: added object already satisfies "
                    "its success criterion at the staging location"
                )
            recovery_design = (
                design["add_object_inherit"]
                if condition_id == "add_object_inherit_source_goal"
                else design["add_object_existing_goal"]
            )
            recovery_actions = recovery_design.get("recovery_actions") or [
                recovery_design["recovery_action"]
            ]
            recovery_action = next(
                (
                    action
                    for action in recovery_actions
                    if action["name"] != "putin"
                    or not backend.world._container_is_full(action["node_ids"][1])
                ),
                recovery_actions[0],
            )
            if recovery_action["name"] == "putin":
                target_id = recovery_action["node_ids"][1]
                target_state = backend.world.states[target_id]
                if backend.world._container_is_full(target_id):
                    if (
                        direct_design
                        and condition.get("success_policy_type")
                        == "trigger_container_goal"
                    ):
                        alternate_targets = [
                            str(value)
                            for value in condition.get("trigger", {}).get("node_ids") or []
                            if str(value) != target_id
                            and str(value) in backend.world.states
                            and not backend.world._container_is_full(str(value))
                        ]
                        movable_blocker = next(
                            (
                                node_id
                                for node_id, state in backend.world.states.items()
                                if state.location_relation == "INSIDE"
                                and state.location_target == target_id
                                and node_id != object_id
                                and state.node.is_movable
                                and backend.world.is_reachable(node_id)
                                and (
                                    not alternate_targets
                                    or alternate_targets[0]
                                    in _allowed_destinations(
                                        validation_goal_facts,
                                        node_id,
                                        "INSIDE",
                                    )
                                )
                            ),
                            None,
                        )
                        if movable_blocker is not None and alternate_targets:
                            if target_state.node.is_openable:
                                target_state.open = True
                            alternate_id = alternate_targets[0]
                            alternate_state = backend.world.states[alternate_id]
                            if alternate_state.node.is_openable:
                                alternate_state.open = True
                            move_grab = backend.step(
                                _parsed_action(
                                    {"name": "grab", "node_ids": [movable_blocker]}
                                )
                            )
                            if move_grab.get("status") == "success":
                                backend.step(
                                    _parsed_action(
                                        {
                                            "name": "putin",
                                            "node_ids": [movable_blocker, alternate_id],
                                        }
                                    )
                                )
                    # The semantic trigger can be reached before another task
                    # object has been moved out of the destination. Follow that
                    # object's successful teacher placement first so validation
                    # checks a real capacity-clearing plan instead of reporting a
                    # false container_full failure (for example B_14's red pen).
                    direct_blockers = {
                        node_id
                        for node_id, state in backend.world.states.items()
                        if state.location_relation == "INSIDE"
                        and state.location_target == target_id
                        and node_id != recovery_design.get("source_node_id")
                    }
                    occluded_blockers = {
                        hidden_id
                        for source_id, hidden_id, relation in backend.world.active_occlusion_edges
                        if source_id == target_id
                        and relation in {"OCCLUDES", "BLOCKS_VIEW"}
                        and hidden_id != recovery_design.get("source_node_id")
                    }
                    blockers = direct_blockers | occluded_blockers
                    for blocker_id in sorted(blockers):
                        if not backend.world._container_is_full(target_id):
                            break
                        teacher_placement = next(
                            (
                                action
                                for step in episode.get("trajectory") or []
                                for action in [
                                    step.get("action") if isinstance(step, dict) else None
                                ]
                                if isinstance(action, dict)
                                and (
                                    not isinstance(step.get("event"), dict)
                                    or step["event"].get("status") in {None, "success"}
                                )
                                and str(
                                    action.get("base_name") or action.get("name") or ""
                                ).lower()
                                in PLACEMENT_RELATIONS
                                and [str(value) for value in action.get("node_ids") or []][
                                    :1
                                ]
                                == [blocker_id]
                                and len(action.get("node_ids") or []) >= 2
                                and str((action.get("node_ids") or [None, None])[1])
                                != target_id
                            ),
                            None,
                        )
                        if teacher_placement is None:
                            continue
                        if target_state.node.is_openable and not target_state.open:
                            open_event = backend.step(
                                _parsed_action({"name": "open", "node_ids": [target_id]})
                            )
                            if open_event.get("status") != "success":
                                continue
                        for _ in range(len(backend.world.states)):
                            active_blockers = backend.world._active_blockers(blocker_id)
                            if not active_blockers:
                                break
                            blocker_resolved = False
                            for occluder_id in active_blockers:
                                active_edge = next(
                                    (
                                        edge
                                        for edge in backend.world.active_occlusion_edges
                                        if edge[0] == occluder_id
                                        and edge[1] == blocker_id
                                    ),
                                    None,
                                )
                                if active_edge is None:
                                    continue
                                resolution_action = (
                                    backend.world._occlusion_edge_resolution_action(
                                        active_edge
                                    )
                                )
                                if resolution_action not in {"open", "move_aside", "close"}:
                                    continue
                                resolution_event = backend.step(
                                    _parsed_action(
                                        {
                                            "name": resolution_action,
                                            "node_ids": [occluder_id],
                                        }
                                    )
                                )
                                blocker_resolved = (
                                    resolution_event.get("status") == "success"
                                ) or blocker_resolved
                            if not blocker_resolved:
                                break
                        grab_blocker_event = backend.step(
                            _parsed_action({"name": "grab", "node_ids": [blocker_id]})
                        )
                        if grab_blocker_event.get("status") != "success":
                            continue
                        destination_id = str(teacher_placement["node_ids"][1])
                        destination_state = backend.world.states[destination_id]
                        if (
                            teacher_placement.get("base_name", teacher_placement.get("name"))
                            == "putin"
                            and destination_state.node.is_openable
                            and not destination_state.open
                        ):
                            open_destination_event = backend.step(
                                _parsed_action(
                                    {"name": "open", "node_ids": [destination_id]}
                                )
                            )
                            if open_destination_event.get("status") != "success":
                                continue
                        placement_event = backend.step(
                            _parsed_action(
                                {
                                    "name": str(
                                        teacher_placement.get("base_name")
                                        or teacher_placement.get("name")
                                    ).lower(),
                                    "node_ids": teacher_placement["node_ids"],
                                }
                            )
                        )
                        if placement_event.get("status") != "success":
                            continue
            for assembly_action in recovery_design.get("assembly_actions", []):
                assembly_event = backend.step(_parsed_action(assembly_action))
                if assembly_event.get("status") != "success":
                    raise ValueError(
                        f"{episode['episode_id']}/{condition_id}: recovery assembly "
                        f"failed: {assembly_event}"
                    )
            if recovery_action["name"] == "putin":
                target_id = recovery_action["node_ids"][1]
                target_state = backend.world.states[target_id]
                if target_state.node.is_openable and not target_state.open:
                    open_event = backend.step(
                        _parsed_action({"name": "open", "node_ids": [target_id]})
                    )
                    if open_event.get("status") != "success":
                        raise ValueError(
                            f"{episode['episode_id']}/{condition_id}: recovery open failed: "
                            f"{open_event}"
                        )
            grab_event = backend.step(
                _parsed_action({"name": "grab", "node_ids": [object_id]})
            )
            if grab_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery grab failed: "
                    f"{grab_event}"
                )
            recovery_event = backend.step(
                _parsed_action(recovery_action)
            )
            if recovery_event.get("status") != "success" or not evaluate_goal_expression(
                expression,
                backend.evaluator._predicate_met,
            ).success:
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery placement failed: "
                    f"{recovery_event}"
                )


def _load_episode(source: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"{source}: expected exactly one episode JSON object")
    episode = records[0]
    if not str(episode.get("episode_id") or "").strip():
        raise ValueError(f"{source}: missing episode_id")
    return episode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate episode-specific closed-loop intervention manifests."
    )
    parser.add_argument("--saved-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Design and validate without writing.")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip symbolic teacher replay validation (not recommended).",
    )
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated episode ids to generate.",
    )
    args = parser.parse_args()

    selected_episode_ids = {
        value.strip() for value in args.episode_ids.split(",") if value.strip()
    }
    if not args.saved_dir.is_dir():
        raise ValueError(f"saved directory does not exist: {args.saved_dir}")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    designed = 0
    validated = 0
    skipped = 0
    seen_episode_ids: set[str] = set()
    for source in sorted(args.saved_dir.glob("*__aligned_*.jsonl")):
        episode = _load_episode(source)
        episode_id = str(episode["episode_id"])
        if selected_episode_ids and episode_id not in selected_episode_ids:
            continue
        if episode_id in seen_episode_ids:
            raise ValueError(f"duplicate saved episode_id: {episode_id}")
        seen_episode_ids.add(episode_id)
        target = args.output_dir / f"{episode_id}_intervention_manifest.json"
        if target.exists() and not args.overwrite and not args.dry_run:
            print(f"SKIP {target} (already exists)")
            skipped += 1
            continue
        manifest = build_manifest(source, episode)
        designed += 1
        if not args.no_validate:
            validate_manifest_semantics(episode, manifest)
            validated += 1
        design = manifest["design"]
        summary = (
            f"state={design['state_regression_action']['name']}"
            f"({','.join(design['state_regression_action']['node_ids'])}), "
            f"rollback={design['placement_action']['name']}"
            f"({','.join(design['placement_action']['node_ids'])}), "
            f"wrong={design['wrong_destination']['node_id']}, "
            f"occlusion_candidates={design['candidate_counts']['add_occlusion']}"
            f"(first={design['occlusion']['source']}->{design['occlusion']['target']}), "
            f"add_copy={design['add_object_inherit']['eligible']}, "
            f"add_existing_goal={design['add_object_existing_goal']['eligible']}"
        )
        if args.dry_run:
            print(f"VALID {episode_id}: {summary}")
            continue
        target.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {target}: {summary}")
        written += 1
    missing = selected_episode_ids - seen_episode_ids
    if missing:
        raise ValueError(f"requested episode ids not found: {sorted(missing)}")
    verb = "Checked" if args.dry_run else "Generated"
    count = designed if args.dry_run else written
    print(f"{verb} {count} manifest(s); validated {validated}; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
