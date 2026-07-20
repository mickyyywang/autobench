#!/usr/bin/env python3
"""Generate one semantically tailored intervention manifest per aligned episode.

The generator deliberately does not contain task-family object templates.  It derives
the selected actions, objects, destinations, state transitions, and occlusion pair
from each episode's own goal, initial view graph, and successful teacher trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


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
    actions: list[dict[str, Any]], facts: list[dict[str, Any]]
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
    return strict or placements


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
        # A relocation into a container that is closed at the semantic trigger
        # would be invisible immediately and confound wrong-destination recovery
        # with a second exploration challenge.
        if relation == "INSIDE" and _is_openable(node):
            is_open = "OPEN" in _node_states(node)
            for action in actions:
                if action["step"] > placement["step"] or action["node_ids"] != [node_id]:
                    continue
                if action["name"] == "open":
                    is_open = True
                elif action["name"] == "close":
                    is_open = False
            if not is_open:
                continue
        candidates.append(
            {
                "node_id": node_id,
                "relation": relation,
                "category": str(node.get("category") or ""),
                "observed_as_placement_target": node_id in observed_placement_targets,
                "static": "STATIC" in _node_properties(node),
            }
        )
    if not candidates:
        return []
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
        if fact["predicate"] in {"INSIDE", "ON", "ASSEMBLED", "ATTACHED"}
        and fact["args"]
        and not fact["optional"]
    }
    goal_objects = strict_goal_objects or {
        fact["args"][0]
        for fact in facts
        if fact["predicate"] in {"INSIDE", "ON", "ASSEMBLED", "ATTACHED"}
        and fact["args"]
    }
    teacher_objects = {
        action["node_ids"][0]
        for action in actions
        if action["node_ids"] and action["name"] in {"grab", "pick", "putin", "puton", "attach"}
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


def _episode_design(
    episode: dict[str, Any],
    actions: list[dict[str, Any]],
    timeline_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    nodes = _node_map(episode)
    edges = _graph_edges(episode)
    visible = _visible_node_ids(episode)
    facts = _goal_facts(episode.get("task_completion_criterion"))
    selection_key = _episode_selection_key(episode, actions, facts)

    placements = _placement_candidates(actions, facts)
    if not placements:
        raise ValueError(f"{episode_id}: no successful goal-achieving putin/puton action")
    placement = _ranked_candidates(
        placements, selection_key=selection_key, role="completed_subgoal_rollback"
    )[0]

    states = _state_candidates(actions, nodes)
    if not states:
        raise ValueError(f"{episode_id}: no successful regressible state-changing action")
    state_action = _ranked_candidates(
        states, selection_key=selection_key, role="state_regression"
    )[0]

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

    return {
        "selection_key": selection_key,
        "facts": facts,
        "placement": placement,
        "state_action": state_action,
        "rollback_surface": rollback_surface,
        "wrong_destination": wrong_destination,
        "occlusion": occlusion,
        "occlusion_candidates": ranked_occlusions,
        "candidate_counts": {
            "placement": len(placements),
            "state_regression": len(states),
            "wrong_destination": len(wrong_destinations),
            "add_occlusion": len(occlusions),
        },
    }


def build_manifest(source: Path, episode: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    actions = _successful_actions(episode)
    if not actions:
        raise ValueError(f"{episode_id}: no successful teacher actions")
    timeline_actions = _successful_actions(episode, include_aligned_actions=True)
    design = _episode_design(episode, actions, timeline_actions)
    placement = design["placement"]
    object_id, correct_target = placement["node_ids"][:2]
    correct_action = {"name": placement["name"], "node_ids": placement["node_ids"]}
    state_action = design["state_action"]
    state_node = state_action["node_ids"][0]
    state_field = state_action["state_field"]
    state_value = state_action["regressed_value"]
    wrong = design["wrong_destination"]
    occlusion = design["occlusion"]
    occlusion_candidates = design["occlusion_candidates"]

    failure_seed = int(design["selection_key"][:8], 16)
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
            "trigger": _semantic_trigger(state_action),
            "graph_disturbance": {
                "operation": "set_state",
                "node_id": state_node,
                "values": {state_field: state_value},
            },
            "expected_effect": f"{state_node}.{state_field} is externally reset to {state_value}.",
            "expected_recovery": (
                f"Execute {state_action['name']}({state_node}) again when it is still needed."
            ),
            "solvability_preserved": True,
        },
        {
            "condition_id": "completed_subgoal_rollback",
            "intervention_type": "relocate",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": _semantic_trigger(placement),
            "graph_disturbance": {
                "operation": "relocate",
                "node_id": object_id,
                "relation": "ON",
                "target": design["rollback_surface"],
            },
            "expected_effect": (
                f"The completed placement of {object_id} is reverted onto "
                f"{design['rollback_surface']}."
            ),
            "expected_recovery": (
                f"Grab {object_id} and repeat {correct_action['name']}"
                f"({', '.join(correct_action['node_ids'])})."
            ),
            "solvability_preserved": True,
        },
        {
            "condition_id": "wrong_container_relocation",
            "intervention_type": "relocate",
            "eligible": True,
            "failure_injection": _failure_none(),
            "trigger": _semantic_trigger(placement),
            "graph_disturbance": {
                "operation": "relocate",
                "node_id": object_id,
                "relation": wrong["relation"],
                "target": wrong["node_id"],
            },
            "expected_effect": (
                f"{object_id} is moved from {correct_target} to the wrong but plausible "
                f"destination {wrong['node_id']}."
            ),
            "expected_recovery": (
                f"Grab {object_id} from {wrong['node_id']} and repeat "
                f"{correct_action['name']}({', '.join(correct_action['node_ids'])})."
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
                "minimum_step": 3,
                "min_goal_progress": 0.1,
                "max_goal_progress": 0.8,
                "apply": "before_current_observation",
            },
            "graph_disturbance": {
                "operation": "add_occlusion",
                "selection": "runtime_first_eligible",
                "candidate_pairs": occlusion_candidates,
            },
            "expected_effect": (
                "At the first eligible mid-episode opportunity, one currently visible "
                "unfinished goal object becomes hidden by a visible resolvable occluder."
            ),
            "expected_recovery": (
                "Infer the runtime-selected blocker from the visible graph, execute its "
                "available open or move_aside resolution, then continue the task."
            ),
            "solvability_preserved": True,
        },
    ]
    design_summary = {
        "generation_algorithm": "episode_semantic_v3_runtime_occlusion",
        "selection_key": design["selection_key"],
        "task_group": _task_group(episode_id),
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
            "minimum_step": 3,
            "min_goal_progress": 0.1,
            "max_goal_progress": 0.8,
        },
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
        }
    )
    return {
        "manifest_version": 3,
        "manifest_type": "closed_loop_intervention_suite",
        "suite_id": f"{episode_id}_interventions_v3",
        "source": {
            "aligned_episode": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "episode_id": episode_id,
            "scene_id": str(episode.get("scene_id") or episode_id),
            "env_id": str(episode.get("env_id") or episode_id),
        },
        "execution_policy": {
            "reset_from_initial_view_graph_for_each_condition": True,
            "conditions_are_mutually_exclusive": True,
            "max_graph_interventions_per_rollout": 1,
            "failure_injection_policy": "once_per_action_name",
            "include_baseline": True,
            "graph_disturbance_timing": "before_current_step_observation",
            "primary_trigger_policy": "semantic_with_runtime_mid_episode_occlusion",
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
    from auto_embodied_task.harness import SymbolicBackend
    from auto_embodied_task.real_observation_eval import _parsed_action, _replay_inputs
    from auto_embodied_task.view_graph_rollout_eval import _ManifestInterventionRuntime

    task, graph, constraints = _replay_inputs(episode)
    graph_conditions = [
        condition
        for condition in manifest["conditions"]
        if condition.get("graph_disturbance") is not None
    ]
    design = manifest["design"]
    for condition in graph_conditions:
        backend = SymbolicBackend(graph, task, constraints)
        runtime = _ManifestInterventionRuntime(condition)
        reports = runtime.before_step(backend, step_number=1, history=[])
        last_success: dict[str, Any] | None = None
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
                    if status is None:
                        # Alignment-only rows describe physical actions and can be
                        # intentionally redundant or outside the symbolic action
                        # model. Apply them when possible, otherwise keep replaying
                        # from the last valid symbolic state.
                        continue
                    raise ValueError(
                        f"{episode['episode_id']}/{condition['condition_id']}: teacher replay "
                        f"failed at step {step.get('step')}: {action} -> {replay_event}"
                    )
                last_success = {"step": step.get("step"), "action": action, "event": replay_event}
                reports = runtime.before_step(
                    backend,
                    step_number=int(step.get("step") or 0) + 1,
                    history=[last_success],
                )
                if reports:
                    break
        if not reports:
            raise ValueError(
                f"{episode['episode_id']}/{condition['condition_id']}: trigger was not reached"
            )
        report = reports[0]
        condition_id = condition["condition_id"]
        if condition_id == "state_regression":
            recovery = design["state_regression_action"]
            recovery_event = backend.step(_parsed_action(recovery))
            if recovery_event.get("status") != "success":
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: recovery failed: {recovery_event}"
                )
        elif condition_id in {"completed_subgoal_rollback", "wrong_container_relocation"}:
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
            if (
                report.get("step") == 1
                or not target
                or details.get("active_after") is not True
                or backend.world.is_visible(target)
            ):
                raise ValueError(
                    f"{episode['episode_id']}/{condition_id}: target was not actively hidden"
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
            f"(first={design['occlusion']['source']}->{design['occlusion']['target']})"
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
