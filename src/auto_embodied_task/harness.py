from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
from datetime import datetime
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from openai import OpenAI

from .action_model import ACTION_SCHEMAS, action_model_trace
from .brain import BrainHarness, BrainPolicy, BrainPolicyConfig, BrainRequest
from .graph_io import load_view_graphs_jsonl
from .goal import evaluate_goal_expression, extract_goal_predicates, normalize_goal_expression
from .models import Node, TaskRecord, ViewGraph, normalize_relation
from .placement_constraints import PlacementEdgeConstraints, load_placement_edge_constraints


LOCATION_RELATIONS = {"ON", "INSIDE", "IN", "BENEATH"}
BLOCKING_RELATIONS = {
    "OCCLUDES",
    "PARTIALLY_OCCLUDES",
    "BLOCKS",
    "HIDES",
    "COVERS",
}
OCCLUSION_RELATIONS = {"OCCLUDES", "PARTIALLY_OCCLUDES", "BLOCKS", "HIDES", "COVERS"}
DECOMPOSED_STATE = "DECOMPOSED"
INSPECT_REVEAL_PROPERTIES = {"HIDDEN", "INSPECT_REVEALS_OCCLUDED", "INSPECT_REVEALS_HIDDEN"}


@dataclass
class ParsedAction:
    name: str
    node_ids: list[str] = field(default_factory=list)
    raw: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def base_name(self) -> str:
        if self.name.startswith("failed_"):
            return self.name.removeprefix("failed_")
        return self.name

    def to_json(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "base_name": self.base_name,
            "node_ids": list(self.node_ids),
            "raw": self.raw,
        }
        if self.arguments:
            payload["arguments"] = copy.deepcopy(self.arguments)
        return payload


@dataclass
class NodeState:
    node: Node
    location_relation: str | None = None
    location_target: str | None = None
    open: bool = False
    held: bool = False
    assembled: bool = False
    pressed: bool = False
    attached_to: str | None = None
    moved_aside: bool = False
    cleared: bool = False
    inspected: bool = False

    def to_json(self, visible: bool, reachable: bool) -> dict[str, Any]:
        return {
            "id": self.node.id,
            "name": self.node.name,
            "category": self.node.category,
            "location": {
                "relation": self.location_relation,
                "target": self.location_target,
            },
            "open": self.open,
            "held": self.held,
            "assembled": self.assembled,
            "pressed": self.pressed,
            "attached_to": self.attached_to,
            "moved_aside": self.moved_aside,
            "cleared": self.cleared,
            "inspected": self.inspected,
            "visible": visible,
            "reachable": reachable,
        }


@dataclass
class TeacherDecision:
    action: ParsedAction
    raw_response: str
    parsed_response: dict[str, Any] | None = None
    reason: str = ""
    parse_error: str | None = None
    parse_repair: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "reason": self.reason,
            "parse_error": self.parse_error,
            "parse_repair": self.parse_repair,
        }


@dataclass
class TeacherPolicyConfig:
    provider: str = "qwen"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    timeout_seconds: int = 60
    temperature: float = 0.0
    max_attempts: int = 1
    retry_backoff_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    api_style: str = "chat_completions"


class TeacherPolicyProtocol(Protocol):
    def act(
        self,
        *,
        task: TaskRecord,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> TeacherDecision:
        ...


@dataclass(frozen=True)
class TrajectoryCollectionResult:
    count: int
    output_path: Path


AVAILABLE_ACTIONS = (
    "look",
    "observe",
    "inspect",
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
    "recover",
    "stop",
)

NON_FAILURE_INJECTABLE_ACTIONS = {"look", "observe", "inspect", "recover", "stop"}
FAILURE_INJECTABLE_ACTIONS = tuple(
    action
    for action in dict.fromkeys((*AVAILABLE_ACTIONS, "place_on", "place_in"))
    if action not in NON_FAILURE_INJECTABLE_ACTIONS
)


@dataclass
class FailureInjectionConfig:
    mode: str = "none"
    actions: tuple[str, ...] = ("all",)
    probability: float = 0.0
    max_failures_per_episode: int = 1
    seed: int | None = None
    deduplication_scope: str = "signature"

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in {"none", "once", "probability", "all"}:
            raise ValueError("failure injection mode must be none, once, probability, or all")
        self.actions = tuple(str(action).strip().lower() for action in self.actions if str(action).strip())
        if not self.actions:
            self.actions = ("all",)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("failure injection probability must be between 0 and 1")
        if self.max_failures_per_episode < 0:
            raise ValueError("max_failures_per_episode must be non-negative")
        self.deduplication_scope = str(self.deduplication_scope).strip().lower()
        if self.deduplication_scope not in {"signature", "action_name"}:
            raise ValueError(
                "failure injection deduplication_scope must be signature or action_name"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.max_failures_per_episode != 0

    def allows(self, action_name: str) -> bool:
        action_name = action_name.lower().removeprefix("failed_")
        if action_name not in FAILURE_INJECTABLE_ACTIONS:
            return False
        if "all" in self.actions:
            return True
        return action_name in self.actions

    def deduplication_key(
        self,
        action_name: str,
        node_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_name = str(action_name).lower().removeprefix("failed_")
        if self.deduplication_scope == "action_name":
            return (normalized_name,)
        return (normalized_name, *(str(node_id) for node_id in node_ids))

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "actions": list(self.actions),
            "probability": self.probability,
            "max_failures_per_episode": self.max_failures_per_episode,
            "seed": self.seed,
            "deduplication_scope": self.deduplication_scope,
        }


ACTION_DESCRIPTIONS = {
    "observe": "Refresh the current observation without changing object state.",
    "look": "Focus on a visible object and mark it inspected.",
    "inspect": "Inspect a visible object; special hidden occluders may reveal blocked nodes.",
    "reach": "Focus on a visible reachable object.",
    "walk": "Focus on a known visible location or target.",
    "open": "Open a visible closed openable container or reveal-capable object.",
    "close": "Close a visible open openable container.",
    "press": "Press a visible reachable object with PRESSABLE affordance.",
    "grab": "Try to hold a visible reachable grabbable movable object; loaded carriers may reject the attempt.",
    "pick": "Alias of grab; loaded carriers may reject the attempt.",
    "attach": "Attach two visible or held compatible part nodes; use this action to satisfy ASSEMBLED goals.",
    "puton": "Place a held object on a visible surface.",
    "putin": "Place a held object inside a visible container that is open if openable.",
    "move_aside": "Move a visible movable blocker aside so hidden nodes may become visible.",
    "recover": "Recover after a previous action failure.",
    "stop": "End the episode and submit the current state for success evaluation.",
}

ACTION_HAND_USAGE = {
    "observe": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "look": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "inspect": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "reach": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "walk": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "open": {"required_free_hands": 2, "held_object_required": False, "result": "hands_unchanged"},
    "close": {"required_free_hands": 2, "held_object_required": False, "result": "hands_unchanged"},
    "press": {"required_free_hands": 1, "held_object_required": False, "result": "hands_unchanged"},
    "grab": {"required_free_hands": 1, "held_object_required": False, "result": "occupies_one_hand"},
    "pick": {"required_free_hands": 1, "held_object_required": False, "result": "occupies_one_hand"},
    "attach": {"required_free_hands": 2, "held_object_required": False, "result": "two_hand_coordination"},
    "puton": {"required_free_hands": 0, "held_object_required": True, "result": "frees_one_hand"},
    "putin": {"required_free_hands": 0, "held_object_required": True, "result": "frees_one_hand"},
    "move_aside": {"required_free_hands": 1, "held_object_required": False, "result": "hands_unchanged"},
    "recover": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
    "stop": {"required_free_hands": 0, "held_object_required": False, "result": "hands_unchanged"},
}


TEACHER_SYSTEM_PROMPT = """You are a teacher policy for high-level embodied task collection.
Choose exactly one semantic action from valid_actions.
Output strict JSON only."""


def _teacher_user_prompt(task: TaskRecord, observation: dict[str, Any], history: list[dict[str, Any]]) -> str:
    history_tail = history
    teacher_observation = _teacher_observation(observation)
    allowed_node_ids, allowed_node_names = _allowed_teacher_nodes(teacher_observation)
    valid_actions = _valid_teacher_actions(observation, history_tail)
    payload = {
        "task": task.task,
        "task_type": task.task_type,
        "settings": task.settings,
        "success_criterion": task.task_completion_criterion,
        "objects": task.objects,
        "current_observation": teacher_observation,
        "allowed_node_ids": allowed_node_ids,
        "allowed_node_names": allowed_node_names,
        "action_catalog": _teacher_action_catalog(),
        "valid_actions": valid_actions,
        "action_constraints": [
            "Choose one action object exactly from valid_actions; copy its name, object, target, and node_ids.",
            "valid_actions are actions that may be attempted, not a guarantee of execution success; use a non-injected failure to replan.",
            "Before choosing any valid action, always check whether success_criterion is already satisfied; if it is satisfied, choose stop.",
            "Use current_observation.robot.hands and action_catalog hand_usage to reason about hand availability; do not ignore held objects.",
            "After an injected failure, valid_actions contains recover and stop; choose recover to continue or stop to submit the current state.",
            "After recover succeeds for an injected failure, valid_actions contains only the previous original failed action so it can be retried cleanly.",
            "Do not invent actions or nodes from task text, objects, success_criterion, memory, or recent_history.",
            "recent_history.new_visible_nodes lists nodes revealed by the previous action.",
        ],
        "recent_history": history_tail,
        "output_schema": {
            "reason": "short rationale",
            "action": {
                "name": "copied from one valid_actions item",
                "object": "copied from the selected valid_actions item when present",
                "target": "copied from the selected valid_actions item when present",
                "node_ids": "copied from the selected valid_actions item",
            },
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _teacher_observation(observation: dict[str, Any]) -> dict[str, Any]:
    visible_observation = copy.deepcopy(observation)
    visible_observation.pop("map_layout", None)
    for node in visible_observation.get("visible_nodes", []):
        if isinstance(node, dict):
            node.pop("occludes_hidden_count", None)
    return visible_observation


def _allowed_teacher_nodes(observation: dict[str, Any]) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    names: list[str] = []
    for node in observation.get("visible_nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is not None and str(node_id) not in ids:
            ids.append(str(node_id))
        name = node.get("name")
        if name is not None and str(name) not in names:
            names.append(str(name))
    return ids, names


def _teacher_action_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for action in AVAILABLE_ACTIONS:
        if action == "pick":
            continue
        schema = ACTION_SCHEMAS.get(action)
        catalog[action] = {
            "description": ACTION_DESCRIPTIONS[action],
            "parameters": _teacher_action_parameters(action, schema),
            "hand_usage": ACTION_HAND_USAGE[action],
            "failure_injectable": action in FAILURE_INJECTABLE_ACTIONS,
        }
    return catalog


def _teacher_action_parameters(action: str, schema: Any) -> list[str]:
    return list(schema.parameters) if schema is not None else []


_GRAB_LIKE_TEACHER_ACTIONS = {"grab", "pick"}


def _prioritize_open_before_grab(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_grab_index = next(
        (index for index, action in enumerate(actions) if action.get("name") in _GRAB_LIKE_TEACHER_ACTIONS),
        None,
    )
    if first_grab_index is None:
        return actions
    delayed_open_actions = [
        action
        for index, action in enumerate(actions)
        if index > first_grab_index and action.get("name") == "open"
    ]
    if not delayed_open_actions:
        return actions
    prioritized: list[dict[str, Any]] = []
    inserted_delayed_opens = False
    for index, action in enumerate(actions):
        if index == first_grab_index and not inserted_delayed_opens:
            prioritized.extend(delayed_open_actions)
            inserted_delayed_opens = True
        if index <= first_grab_index or action.get("name") != "open":
            prioritized.append(action)
    return prioritized


def _valid_teacher_actions(observation: dict[str, Any], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible_nodes = [node for node in observation.get("visible_nodes", []) if isinstance(node, dict)]
    visible_by_id = {str(node["id"]): node for node in visible_nodes if node.get("id") is not None}
    held_ids = [
        str(item["id"])
        for item in observation.get("held_objects", [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    open_priority_node_ids: set[str] = set()

    def add(name: str, node_ids: list[str] | None = None, **extra: Any) -> None:
        node_ids = [str(node_id) for node_id in (node_ids or [])]
        key = (name, tuple(node_ids))
        if key in seen:
            return
        seen.add(key)
        item: dict[str, Any] = {"name": name, "node_ids": node_ids}
        if node_ids:
            item["object"] = node_ids[0]
        if len(node_ids) >= 2:
            item["target"] = node_ids[1]
        item.update(extra)
        actions.append(item)

    recovered_failure_item = _last_recovered_failure_item(history)
    if recovered_failure_item is not None:
        failed_action = _failed_history_action_name(recovered_failure_item)
        if failed_action:
            add(failed_action, _failed_history_node_ids(recovered_failure_item))
        else:
            add("observe")
        return actions

    failed_history_item = _last_injected_failed_history_item(history)
    if failed_history_item is not None:
        add("recover")
        add("stop")
        return actions

    for node in visible_nodes:
        node_id = str(node.get("id"))
        if not node_id:
            continue
        can_open = bool(
            node.get("openable")
            and not node.get("open")
            and not held_ids
            and (
                _teacher_node_on_surface(node, visible_by_id)
                or _teacher_node_is_reachable_static_part(node)
            )
        )
        if can_open:
            extra = _reveal_action_extra(node) if node.get("container") else {}
            add("open", [node_id], **extra)
            if not node.get("inspected"):
                open_priority_node_ids.add(node_id)
        if node.get("openable") and node.get("open") and not held_ids:
            extra = _reveal_action_extra(node) if node.get("container") else {}
            add("close", [node_id], **extra)
        if node.get("pressable") and node.get("reachable"):
            add("press", [node_id])
        for reveal_action in _reveal_valid_actions(node):
            add(reveal_action, [node_id], **_reveal_action_extra(node))
        if node_id not in open_priority_node_ids and node.get("grabbable") and node.get("movable") and node.get("reachable"):
            add("grab", [node_id])

    part_nodes = [
        node
        for node in visible_nodes
        if _is_attachable_teacher_part(node)
    ]
    for index, left in enumerate(part_nodes):
        left_id = str(left.get("id"))
        for right in part_nodes[index + 1 :]:
            right_id = str(right.get("id"))
            if left.get("part_of") == right.get("part_of"):
                add("attach", [left_id, right_id])

    placement_targets = [
        node
        for node in visible_nodes
        if node.get("id") is not None and (node.get("container") or node.get("surface"))
    ]
    for held_id in held_ids:
        if held_id not in visible_by_id:
            continue
        for target in placement_targets:
            target_id = str(target["id"])
            if target_id == held_id:
                continue
            if target.get("container") and (not target.get("openable") or target.get("open")):
                add("putin", [held_id, target_id])
            if target.get("surface"):
                add("puton", [held_id, target_id])

    if not actions:
        add("observe")
    actions[:] = _prioritize_open_before_grab(actions)
    add("stop")
    return actions


def _teacher_node_on_surface(node: dict[str, Any], visible_by_id: dict[str, dict[str, Any]]) -> bool:
    location = node.get("location")
    if not isinstance(location, dict):
        return False
    if str(location.get("relation") or "").upper() not in {"ON", "BENEATH"}:
        return False
    target_id = location.get("target")
    if target_id is None:
        return False
    target = visible_by_id.get(str(target_id))
    return bool(target and target.get("surface"))


def _teacher_node_is_reachable_static_part(node: dict[str, Any]) -> bool:
    properties = {
        str(prop).strip().upper().replace(" ", "_")
        for prop in node.get("properties", [])
    }
    return bool(node.get("reachable") and "STATIC" in properties and node.get("part_of") is not None)


def _reveal_valid_actions(node: dict[str, Any]) -> list[str]:
    if int(node.get("occludes_hidden_count") or 0) <= 0:
        return []
    if node.get("container"):
        if (not node.get("openable") or node.get("open")) and node.get("movable"):
            return ["move_aside"]
        return []
    if node.get("movable"):
        return ["move_aside"]
    return []


def _reveal_action_extra(node: dict[str, Any]) -> dict[str, str]:
    if int(node.get("occludes_hidden_count") or 0) > 0:
        return {"effect_hint": "may_reveal_hidden"}
    return {}


def _is_attachable_teacher_part(node: dict[str, Any]) -> bool:
    return bool(
        node.get("part_of") is not None
        and node.get("reachable")
        and node.get("grabbable")
        and node.get("movable")
    )


def _last_history_failed(history: list[dict[str, Any]]) -> bool:
    return _last_failed_history_item(history) is not None


def _last_failed_history_item(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    item = history[-1]
    if not isinstance(item, dict):
        return None
    event = item.get("event")
    if isinstance(event, dict) and event.get("status") == "failure":
        return item
    return None


def _last_injected_failed_history_item(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    item = _last_failed_history_item(history)
    if item is None:
        return None
    return item if _is_injected_failure_item(item) else None


def _is_injected_failure_item(item: dict[str, Any]) -> bool:
    event = item.get("event")
    return isinstance(event, dict) and (
        event.get("injected") is True or event.get("failure_type") == "injected"
    )


def _last_recovered_failure_item(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(history) < 2:
        return None
    recovered_item = history[-1]
    if not isinstance(recovered_item, dict):
        return None
    event = recovered_item.get("event")
    action = recovered_item.get("action")
    if not (
        isinstance(event, dict)
        and event.get("status") == "success"
        and isinstance(action, dict)
        and str(action.get("base_name") or action.get("name")).lower() == "recover"
    ):
        return None
    return _last_injected_failed_history_item(history[:-1])


def _failed_history_action_name(history_item: dict[str, Any]) -> str | None:
    action = _failed_history_original_action(history_item)
    if isinstance(action, dict):
        failed_action = action.get("base_name") or action.get("name")
        if failed_action:
            return str(failed_action).lower().removeprefix("failed_")
    event = history_item.get("event")
    if isinstance(event, dict):
        failed_action = event.get("failed_action") or event.get("action")
        if failed_action:
            return str(failed_action).lower().removeprefix("failed_")
    return None


def _failed_history_node_ids(history_item: dict[str, Any]) -> list[str]:
    action = _failed_history_original_action(history_item)
    if not isinstance(action, dict):
        return []
    raw_ids = action.get("node_ids")
    if not isinstance(raw_ids, list):
        return []
    return [str(node_id) for node_id in raw_ids if str(node_id)]


def _failed_history_original_action(history_item: dict[str, Any]) -> dict[str, Any] | None:
    requested_action = history_item.get("requested_action")
    if isinstance(requested_action, dict):
        return requested_action
    action = history_item.get("action")
    if isinstance(action, dict):
        return action
    return None


def _new_visible_nodes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_ids = {
        str(node.get("id"))
        for node in before.get("visible_nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    new_nodes: list[dict[str, Any]] = []
    for node in after.get("visible_nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None or str(node_id) in before_ids:
            continue
        new_nodes.append(
            {
                key: copy.deepcopy(node[key])
                for key in (
                    "id",
                    "name",
                    "category",
                    "properties",
                    "states",
                    "openable",
                    "container",
                    "surface",
                    "grabbable",
                    "movable",
                    "part_of",
                    "reachable",
                )
                if key in node
            }
        )
    return new_nodes


class SymbolicObservationAdapter:
    def build_request(
        self,
        *,
        task: TaskRecord,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> BrainRequest:
        return BrainRequest(
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": _teacher_user_prompt(task, observation, history)},
            ],
            summary={"adapter": "symbolic", "has_view_graph": True, "frame_count": 0},
        )

    def parse_response(self, text: str) -> TeacherDecision:
        return parse_teacher_decision(text)


class TeacherPolicy:
    def __init__(self, config: TeacherPolicyConfig) -> None:
        self.config = config
        self.model = config.model or os.environ.get("AUTO_EMBODIED_TEACHER_MODEL") or _default_teacher_model(
            config.provider
        )
        self.brain_harness = BrainHarness(
            BrainPolicy(
                BrainPolicyConfig(
                    provider=config.provider,
                    model=self.model,
                    api_key=config.api_key,
                    api_key_env=config.api_key_env,
                    api_base_url=config.api_base_url,
                    timeout_seconds=config.timeout_seconds,
                    temperature=config.temperature,
                    max_attempts=config.max_attempts,
                    retry_backoff_seconds=config.retry_backoff_seconds,
                    retry_max_seconds=config.retry_max_seconds,
                    api_style=config.api_style,
                ),
                client_factory=OpenAI,
            ),
            SymbolicObservationAdapter(),
        )

    def act(
        self,
        *,
        task: TaskRecord,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> TeacherDecision:
        return self.brain_harness.decide(task=task, observation=observation, history=history)


class ScriptedTeacherPolicy:
    def __init__(self, actions: Iterable[ParsedAction | dict[str, Any] | str]) -> None:
        self.actions = list(actions)
        self.index = 0

    def act(
        self,
        *,
        task: TaskRecord,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> TeacherDecision:
        if self.index >= len(self.actions):
            action = ParsedAction(name="stop", raw="script exhausted")
            return TeacherDecision(action=action, raw_response='{"action":{"name":"stop"}}')
        item = self.actions[self.index]
        self.index += 1
        if isinstance(item, ParsedAction):
            return TeacherDecision(action=item, raw_response=json.dumps({"action": item.to_json()}))
        if isinstance(item, dict):
            return parse_teacher_decision(json.dumps(item, ensure_ascii=False))
        return parse_teacher_decision(item)


def parse_teacher_decision(text: str) -> TeacherDecision:
    raw = text.strip()
    parse_repair: str | None = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        parsed = None
        if exc.msg == "Extra data":
            try:
                candidate, end_index = json.JSONDecoder().raw_decode(raw)
            except json.JSONDecodeError:
                candidate = None
                end_index = 0
            suffix = raw[end_index:]
            if end_index > 0 and suffix and all(
                character == "}" or character.isspace() for character in suffix
            ):
                parsed = candidate
                parse_repair = "trailing_closing_braces"
        if parsed is None:
            return TeacherDecision(
                action=ParsedAction(name="invalid_teacher_action", raw=raw),
                raw_response=raw,
                parse_error=str(exc),
            )
    # Some JSON-mode providers (notably Gemini through ModelRouter) wrap the
    # requested object in a singleton array even though the prompt asks for an
    # object.  This is unambiguous to normalize; larger or non-object arrays
    # remain invalid responses.
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return TeacherDecision(
            action=ParsedAction(name="invalid_teacher_action", raw=raw),
            raw_response=raw,
            parsed_response=None,
            parse_error="teacher response must be a JSON object",
            parse_repair=parse_repair,
        )
    try:
        action = _parsed_response_to_action(parsed, raw)
    except ValueError as exc:
        return TeacherDecision(
            action=ParsedAction(name="invalid_teacher_action", raw=raw),
            raw_response=raw,
            parsed_response=parsed,
            reason=str(parsed.get("reason", "")),
            parse_error=str(exc),
            parse_repair=parse_repair,
        )
    return TeacherDecision(
        action=action,
        raw_response=raw,
        parsed_response=parsed,
        reason=str(parsed.get("reason", "")),
        parse_repair=parse_repair,
    )


def _parsed_response_to_action(parsed: dict[str, Any], raw: str) -> ParsedAction:
    action_obj = parsed.get("action", parsed)
    if not isinstance(action_obj, dict):
        raise ValueError("action must be a JSON object")
    name = action_obj.get("name") or action_obj.get("action")
    if not name:
        raise ValueError("action.name is required")
    action_name = str(name).strip().lower()
    node_ids = _node_ids_from_action(action_obj)
    arguments = _action_arguments_from_action(action_obj)
    return ParsedAction(name=action_name, node_ids=node_ids, raw=raw, arguments=arguments)


def _action_arguments_from_action(action_obj: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = action_obj.get("arguments")
    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
    failed_action = (
        action_obj.get("failed_action")
        or action_obj.get("recover_action")
        or arguments.get("failed_action")
        or arguments.get("recover_action")
    )
    if failed_action is None:
        return {}
    failed_action_name = str(failed_action).strip().lower().removeprefix("failed_")
    return {"failed_action": failed_action_name} if failed_action_name else {}


def _node_ids_from_action(action_obj: dict[str, Any]) -> list[str]:
    raw_ids = action_obj.get("node_ids")
    if raw_ids is None:
        raw_ids = action_obj.get("nodes")
    if isinstance(raw_ids, list):
        return [str(item).strip() for item in raw_ids if str(item).strip()]
    if isinstance(raw_ids, str) and raw_ids.strip():
        return [raw_ids.strip()]

    arguments = action_obj.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    merged = {**arguments, **{k: v for k, v in action_obj.items() if k not in {"arguments", "action", "name"}}}
    ordered_keys = (
        "object",
        "target",
        "container",
        "surface",
        "anchor",
        "source",
        "destination",
    )
    node_ids = []
    for key in ordered_keys:
        value = merged.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            node_ids.extend(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value).strip()
            if text:
                node_ids.append(text)
    return node_ids


def _default_teacher_model(provider: str) -> str:
    if provider == "qwen":
        return "qwen3.6-plus"
    if provider == "openai":
        return "gpt-4o-mini"
    return "model"


def parse_plan_action(plan_step: str) -> ParsedAction:
    action_match = re.search(r"\[([^\]]+)\]", plan_step)
    if not action_match:
        return ParsedAction(name="unknown", raw=plan_step)
    name = action_match.group(1).strip().lower()
    node_ids = [match.group(1).strip() for match in re.finditer(r"<[^>]*>\s*\(([^)]+)\)", plan_step)]
    return ParsedAction(name=name, node_ids=node_ids, raw=plan_step)


def _canonical_location(relation: str | None) -> str | None:
    if relation is None:
        return None
    normalized = normalize_relation(relation)
    if normalized == "IN":
        return "INSIDE"
    return normalized


def _load_tasks_jsonl(path: str | Path) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_no}: expected a JSON object")
            tasks.append(TaskRecord(**payload))
    return tasks


def _state_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _parse_required_count(value: Any) -> int | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _view_graph_to_json(graph: ViewGraph) -> dict[str, Any]:
    nodes = []
    for node in sorted(graph.nodes.values(), key=lambda item: (item.name, item.id)):
        payload = {
            "id": node.id,
            "name": node.name,
            "category": node.category,
            "properties": list(node.properties),
            "states": list(node.states),
        }
        if node.room is not None:
            payload["room"] = node.room
        if node.parent is not None:
            payload["parent"] = node.parent
        payload.update(copy.deepcopy(node.metadata))
        nodes.append(payload)

    edges = []
    for edge in graph.edges:
        payload = {
            "from": edge.source,
            "to": edge.target,
            "relation": edge.relation,
        }
        payload.update(copy.deepcopy(edge.metadata))
        edges.append(payload)

    return {
        "scene_id": graph.scene_id,
        "env_id": graph.env_id,
        "layout": graph.layout,
        "robot": copy.deepcopy(graph.robot),
        "metadata": copy.deepcopy(graph.metadata),
        "nodes": nodes,
        "edges": edges,
    }


class SemanticWorld:
    """Small symbolic backend driven by a view graph.

    It deliberately models only high-level embodied state: object locations,
    open/closed containers, visibility, reachability, held objects, and
    recoverable action failures.
    """

    def __init__(
        self,
        graph: ViewGraph,
        task: TaskRecord,
        placement_edge_constraints: PlacementEdgeConstraints | None = None,
    ) -> None:
        self.graph = graph
        self.task = task
        self.placement_edge_constraints = placement_edge_constraints or PlacementEdgeConstraints()
        self.states: dict[str, NodeState] = {}
        self.focus: str | None = None
        self.visited: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.active_occlusion_edges: set[tuple[str, str, str]] = set()
        self.occlusion_edge_resolution_actions: dict[tuple[str, str, str], str] = {}
        self._name_to_id = self._build_name_lookup(graph)
        self.memory_hidden: dict[str, str] = self._memory_hidden_targets(task)
        self._initialize_states()
        self._initialize_occlusion_edges()

    @staticmethod
    def _build_name_lookup(graph: ViewGraph) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for node in graph.nodes.values():
            lookup.setdefault(node.id, node.id)
            lookup.setdefault(node.name, node.id)
            lookup.setdefault(node.name.lower(), node.id)
        return lookup

    def _memory_hidden_targets(self, task: TaskRecord) -> dict[str, str]:
        constraint = task.metadata.get("memory_constraint")
        if not isinstance(constraint, dict) or not constraint.get("not_initial_state"):
            return {}
        object_id = self.resolve_node_id(constraint.get("remember_object"))
        anchor_id = self.resolve_node_id(constraint.get("remember_anchor"))
        if object_id is None or anchor_id is None:
            return {}
        return {object_id: anchor_id}

    def _initialize_states(self) -> None:
        for node in self.graph.nodes.values():
            relation, target = self._initial_location(node)
            is_open = "OPEN" in node.states
            if node.is_openable and "OPEN" not in node.states:
                is_open = False
            is_pressed = "PRESSED" in node.states
            self.states[node.id] = NodeState(
                node=node,
                location_relation=relation,
                location_target=target,
                open=is_open,
                pressed=is_pressed,
            )

    def _initialize_occlusion_edges(self) -> None:
        for edge in self.graph.edges:
            if edge.relation not in OCCLUSION_RELATIONS:
                continue
            key = (edge.source, edge.target, edge.relation)
            self.active_occlusion_edges.add(key)
            explicit_resolution = edge.metadata.get("resolution_action")
            if explicit_resolution is None:
                resolution_action = self._default_occlusion_resolution_action(edge.source)
            else:
                resolution_action = str(explicit_resolution).strip().lower().replace("-", "_").replace(" ", "_")
                if resolution_action == "moveaside":
                    resolution_action = "move_aside"
                if resolution_action == "shut":
                    resolution_action = "close"
                if resolution_action not in {"open", "close", "move_aside"}:
                    raise ValueError(
                        f"Unsupported occlusion resolution_action {explicit_resolution!r} "
                        f"for edge {edge.source!r} -> {edge.target!r}"
                    )
            self.occlusion_edge_resolution_actions[key] = resolution_action

    def _default_occlusion_resolution_action(self, source_id: str) -> str:
        source = self.states.get(source_id)
        if source is None:
            return "move_aside"
        if source.node.is_container:
            if source.node.is_openable and not source.open:
                return "open"
            if source.node.is_movable:
                return "move_aside"
            return "open"
        return "move_aside"

    def _initial_location(self, node: Node) -> tuple[str | None, str | None]:
        for edge in self.graph.outgoing(node.id):
            if edge.relation in LOCATION_RELATIONS:
                return _canonical_location(edge.relation), edge.target
        if node.parent and node.parent in self.graph.nodes:
            parent = self.graph.get(node.parent)
            relation = "INSIDE" if parent.is_container else "ON"
            return relation, parent.id
        if node.room and node.room in self.graph.nodes:
            return "INSIDE", node.room
        return None, None

    def resolve_node_id(self, ref: Any) -> str | None:
        if ref is None:
            return None
        text = str(ref)
        if text in self.graph.nodes:
            return text
        return self._name_to_id.get(text) or self._name_to_id.get(text.lower())

    def snapshot(self) -> dict[str, Any]:
        return {
            "robot": {
                "focus": self.focus,
                "visited": list(self.visited),
            },
            "nodes": {
                node_id: self._node_state_json(node_id, state)
                for node_id, state in sorted(self.states.items())
            },
            "active_occlusion_edges": [
                {"from": source, "to": target, "relation": relation}
                for source, target, relation in sorted(self.active_occlusion_edges)
            ],
        }

    def _node_state_json(self, node_id: str, state: NodeState) -> dict[str, Any]:
        payload = state.to_json(
            visible=self.is_visible(node_id),
            reachable=self.is_reachable(node_id),
        )
        payload.update(self._container_capacity_info(node_id))
        return payload

    def observe(self) -> dict[str, Any]:
        visible_nodes = []
        for node_id, state in sorted(self.states.items(), key=lambda item: (item[1].node.name, item[0])):
            if self.is_visible(node_id):
                hidden_targets = self._hidden_targets_blocked_by(node_id)
                payload = {
                    "id": node_id,
                    "name": state.node.name,
                    "category": state.node.category,
                    "properties": list(state.node.properties),
                    "states": self._runtime_node_states(state),
                    "openable": state.node.is_openable,
                    "container": state.node.is_container,
                    "surface": state.node.is_surface,
                    "pressable": state.node.is_pressable,
                    "grabbable": state.node.is_grabbable,
                    "movable": state.node.is_movable,
                    "location": {
                        "relation": state.location_relation,
                        "target": state.location_target,
                    },
                    "open": state.open,
                    "assembled": state.assembled,
                    "pressed": state.pressed,
                    "attached_to": state.attached_to,
                    "inspected": state.inspected,
                    "part_of": state.node.metadata.get("part_of"),
                    "reachable": self.is_reachable(node_id),
                    "occludes_hidden_count": len(hidden_targets),
                }
                payload.update(self._container_capacity_info(node_id))
                visible_nodes.append(payload)
        visible_node_ids = {item["id"] for item in visible_nodes}
        visible_edges = self._current_visible_edges(visible_node_ids)
        held_objects = [
            {"id": node_id, "name": state.node.name}
            for node_id, state in sorted(self.states.items())
            if state.held
        ]
        return {
            "task": self.task.task,
            "settings": list(self.task.settings),
            "visible_nodes": visible_nodes,
            "visible_edges": visible_edges,
            "held_objects": held_objects,
            "robot": {
                "focus": self.focus,
                "visited": list(self.visited),
                "hands": self._hand_state(held_objects),
            },
            "memory": copy.deepcopy(self.task.metadata.get("memory_episode")),
            "map_layout": self.map_layout(visible_node_ids),
        }

    def _runtime_node_states(self, state: NodeState) -> list[str]:
        states = [item for item in state.node.states if item not in {"OPEN", "CLOSED"}]
        if state.node.is_openable:
            states.append("OPEN" if state.open else "CLOSED")
        return states

    def _hand_state(self, held_objects: list[dict[str, str]]) -> dict[str, Any]:
        capacity = self._hand_capacity()
        slots = []
        for index, name in enumerate(("left", "right")):
            available = index < capacity
            holding = held_objects[index] if available and index < len(held_objects) else None
            slots.append({"name": name, "available": available, "holding": copy.deepcopy(holding)})
        return {
            "capacity": capacity,
            "slots": slots,
            "held_object_ids": [item["id"] for item in held_objects],
            "occupied_count": min(len(held_objects), capacity),
            "free_count": max(0, capacity - len(held_objects)),
            "overflow_held_objects": copy.deepcopy(held_objects[capacity:]),
        }

    def _hand_capacity(self) -> int:
        arms = str(self.task.arms or self.graph.robot.get("arms") or "").lower()
        return 2 if arms == "double" else 1

    def _container_capacity_info(self, container_id: str) -> dict[str, Any]:
        state = self.states.get(container_id)
        if state is None or not state.node.is_container:
            return {}
        max_items = state.node.max_items
        if max_items is None:
            return {}
        item_count = self._container_item_count(container_id)
        return {
            "max_items": max_items,
            "item_count": item_count,
            "is_full": item_count >= max_items,
        }

    def _container_item_count(self, container_id: str, exclude: set[str] | None = None) -> int:
        exclude = exclude or set()
        items: set[str] = set()
        for node_id, state in self.states.items():
            if node_id == container_id or node_id in exclude or state.held:
                continue
            if state.location_relation == "INSIDE" and state.location_target == container_id:
                items.add(node_id)
        for source_id, target_id, relation in self.active_occlusion_edges:
            if source_id != container_id or relation not in OCCLUSION_RELATIONS:
                continue
            if self._occlusion_edge_resolution_action((source_id, target_id, relation)) != "open":
                continue
            if target_id == container_id or target_id in exclude:
                continue
            target = self.states.get(target_id)
            if target is None or target.held:
                continue
            if target.location_relation is None:
                items.add(target_id)
        return sum(self._capacity_item_units(node_id) for node_id in items)

    def _capacity_item_units(
        self,
        node_id: str,
        visiting: set[str] | None = None,
    ) -> int:
        """Return how many parent-container capacity slots a direct item uses.

        Ordinary objects and containers use one slot. An explicitly annotated
        carrier can instead use the number of items it currently carries; the
        minimum remains one so an empty carrier still occupies physical space.
        """

        state = self.states.get(node_id)
        if state is None:
            return 1
        if not state.node.has_property("CAPACITY_COUNTS_CONTENTS"):
            return 1
        visiting = set(visiting or ())
        if node_id in visiting:
            return 1
        visiting.add(node_id)
        payload_ids = self._carrier_payload_ids(node_id)
        if not payload_ids:
            return 1
        return max(
            1,
            sum(
                self._capacity_item_units(payload_id, visiting)
                for payload_id in payload_ids
            ),
        )

    def _container_projected_item_count(
        self,
        container_id: str,
        object_id: str,
    ) -> int:
        return self._container_item_count(
            container_id,
            exclude={object_id},
        ) + self._capacity_item_units(object_id)

    def _container_would_exceed_capacity(
        self,
        container_id: str,
        object_id: str,
    ) -> bool:
        state = self.states.get(container_id)
        if state is None or not state.node.is_container or state.node.max_items is None:
            return False
        return (
            self._container_projected_item_count(container_id, object_id)
            > state.node.max_items
        )

    def _container_is_full(self, container_id: str, exclude: set[str] | None = None) -> bool:
        state = self.states.get(container_id)
        if state is None or not state.node.is_container or state.node.max_items is None:
            return False
        return self._container_item_count(container_id, exclude=exclude) >= state.node.max_items

    def _carrier_payload_ids(self, carrier_id: str) -> list[str]:
        payload_ids = []
        for node_id, state in self.states.items():
            if node_id == carrier_id or state.held:
                continue
            if state.location_target != carrier_id:
                continue
            if state.location_relation not in {"INSIDE", "ON"}:
                continue
            payload_ids.append(node_id)
        return sorted(payload_ids)

    def _is_on_surface(self, node_id: str) -> bool:
        state = self.states.get(node_id)
        if state is None or state.location_relation != "ON" or state.location_target is None:
            return False
        target = self.states.get(state.location_target)
        return bool(target and target.node.is_surface)

    def _is_reachable_static_part(self, node_id: str) -> bool:
        state = self.states.get(node_id)
        return bool(
            state
            and self.is_reachable(node_id)
            and state.node.has_property("STATIC")
            and self._part_parent(node_id) is not None
        )

    def _current_visible_edges(self, visible_node_ids: set[str]) -> list[dict[str, str]]:
        edges: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        def add_edge(source: str | None, target: str | None, relation: str | None) -> None:
            if source is None or target is None or relation is None:
                return
            source = str(source)
            target = str(target)
            relation = normalize_relation(str(relation))
            if source not in visible_node_ids or target not in visible_node_ids:
                return
            key = (source, target, relation)
            if key in seen:
                return
            seen.add(key)
            edges.append({"from": source, "to": target, "relation": relation})

        for node_id, state in sorted(self.states.items()):
            if state.held:
                continue
            add_edge(node_id, state.location_target, state.location_relation)

        for edge in self.graph.edges:
            if edge.relation in LOCATION_RELATIONS:
                continue
            if edge.relation in BLOCKING_RELATIONS and edge.source not in self._active_blockers(
                edge.target,
                relations={edge.relation},
            ):
                continue
            add_edge(edge.source, edge.target, edge.relation)

        for node_id, state in sorted(self.states.items()):
            add_edge(node_id, state.attached_to, "ATTACHED")

        return edges

    def map_layout(self, visible_node_ids: set[str] | None = None) -> dict[str, Any]:
        visible_node_ids = visible_node_ids or set(self.states)
        zones = []
        zone_ids: list[str] = []
        for node_id, state in sorted(self.states.items(), key=lambda item: (item[1].node.name, item[0])):
            if node_id not in visible_node_ids:
                continue
            if not (state.node.is_room or state.node.is_surface or state.node.is_container):
                continue
            zone_ids.append(node_id)
            zones.append(
                {
                    "id": node_id,
                    "name": state.node.name,
                    "kind": "room" if state.node.is_room else "container" if state.node.is_container else "surface",
                    "open": state.open,
                }
            )
        if not zones:
            zones.append({"id": "workspace", "name": "workspace", "kind": "workspace", "open": True})
            zone_ids.append("workspace")

        zone_order = {zone_id: index for index, zone_id in enumerate(zone_ids)}
        objects = []
        for node_id, state in sorted(self.states.items(), key=lambda item: (item[1].node.name, item[0])):
            if node_id not in visible_node_ids or state.node.is_room:
                continue
            target = state.location_target if state.location_target in zone_order else zone_ids[0]
            index = len(objects)
            objects.append(
                {
                    "id": node_id,
                    "name": state.node.name,
                    "zone": target,
                    "x": 0.16 + (index % 5) * 0.17,
                    "y": 0.18 + ((index // 5) % 4) * 0.2,
                }
            )
        return {"zones": zones, "objects": objects}

    def is_visible(self, node_id: str) -> bool:
        return self._is_visible(node_id, set())

    def _is_visible(self, node_id: str, visiting: set[str]) -> bool:
        if node_id not in self.states:
            return False
        if node_id in visiting:
            return False
        visiting = visiting | {node_id}
        state = self.states[node_id]
        if state.node.is_room or state.held:
            return True
        if self._is_decomposed_parent(node_id) and not state.assembled:
            return False
        assembled_parent_id = self._assembled_part_parent(node_id)
        if assembled_parent_id is not None and not self._is_visible(assembled_parent_id, visiting):
            return False
        memory_anchor = self.memory_hidden.get(node_id)
        if memory_anchor is not None and not self._memory_target_revealed(memory_anchor):
            return False
        if self._inside_closed_container(node_id):
            return False
        if (
            state.location_relation == "INSIDE"
            and state.location_target in self.states
            and not self._is_visible(state.location_target, visiting)
        ):
            return False
        return not self._active_blockers(node_id)

    def is_reachable(self, node_id: str) -> bool:
        if not self.is_visible(node_id):
            return False
        if node_id not in self.states:
            return False
        if self.states[node_id].node.is_room:
            return False
        return not self._inside_closed_container(node_id)

    def _memory_target_revealed(self, anchor_id: str) -> bool:
        if anchor_id not in self.states:
            return True
        anchor = self.states[anchor_id]
        if anchor.node.is_openable and not anchor.open:
            return False
        return anchor.inspected or anchor.open

    def _inside_closed_container(self, node_id: str) -> bool:
        seen: set[str] = set()
        current = node_id
        while current in self.states and current not in seen:
            seen.add(current)
            state = self.states[current]
            parent_id = state.location_target
            if parent_id is None or parent_id not in self.states:
                return False
            parent_state = self.states[parent_id]
            if state.location_relation == "INSIDE" and parent_state.node.is_openable and not parent_state.open:
                return True
            current = parent_id
        return False

    def _remove_occlusions_for_location_change(self, node_id: str) -> None:
        self.active_occlusion_edges = {
            edge
            for edge in self.active_occlusion_edges
            if edge[0] != node_id and edge[1] != node_id
        }
        self.occlusion_edge_resolution_actions = {
            edge: resolution_action
            for edge, resolution_action in self.occlusion_edge_resolution_actions.items()
            if edge in self.active_occlusion_edges
        }

    def _active_blockers(self, node_id: str, relations: set[str] | None = None) -> list[str]:
        active = []
        wanted = relations or BLOCKING_RELATIONS
        for source_id, target_id, relation in sorted(self.active_occlusion_edges):
            if target_id != node_id or relation not in wanted:
                continue
            source = self.states.get(source_id)
            if source is None:
                continue
            edge = (source_id, target_id, relation)
            if relation in OCCLUSION_RELATIONS and not self._occlusion_edge_active(edge):
                continue
            active.append(source.node.id)
        return active

    def _occlusion_edge_resolution_action(self, edge: tuple[str, str, str]) -> str:
        return self.occlusion_edge_resolution_actions.get(edge) or self._default_occlusion_resolution_action(edge[0])

    def _occlusion_edge_active(self, edge: tuple[str, str, str]) -> bool:
        source = self.states.get(edge[0])
        if source is None:
            return False
        resolution_action = self._occlusion_edge_resolution_action(edge)
        if resolution_action == "open":
            return not (source.node.is_openable and source.open)
        if resolution_action == "close":
            return source.node.is_openable and source.open
        if resolution_action == "move_aside":
            return not source.moved_aside
        return False

    def _inspect_reveals_occlusion(self, state: NodeState) -> bool:
        return state.node.has_property(*INSPECT_REVEAL_PROPERTIES)

    def _hidden_targets_blocked_by(self, source_id: str) -> list[str]:
        hidden_targets = []
        for edge_source, edge_target, relation in sorted(self.active_occlusion_edges):
            if edge_source != source_id or relation not in OCCLUSION_RELATIONS:
                continue
            if not self._occlusion_edge_active((edge_source, edge_target, relation)):
                continue
            if edge_target in self.states and not self.is_visible(edge_target):
                hidden_targets.append(edge_target)
        return hidden_targets

    def _hidden_targets_blocked_by_resolution(self, source_id: str, resolution_action: str) -> list[str]:
        hidden_targets = []
        for edge_source, edge_target, relation in sorted(self.active_occlusion_edges):
            edge = (edge_source, edge_target, relation)
            if edge_source != source_id or relation not in OCCLUSION_RELATIONS:
                continue
            if self._occlusion_edge_resolution_action(edge) != resolution_action:
                continue
            if not self._occlusion_edge_active(edge):
                continue
            if edge_target in self.states and not self.is_visible(edge_target):
                hidden_targets.append(edge_target)
        return hidden_targets

    def _is_decomposed_parent(self, node_id: str) -> bool:
        state = self.states.get(node_id)
        if state is None:
            return False
        parts = self._direct_part_ids(node_id)
        if not parts:
            return False
        return DECOMPOSED_STATE in state.node.states or state.node.has_property(DECOMPOSED_STATE)

    def _assembled_part_parent(self, node_id: str) -> str | None:
        parent_id = self._part_parent(node_id)
        if parent_id is None or parent_id == node_id:
            return None
        parent_state = self.states.get(parent_id)
        if parent_state is not None and parent_state.assembled:
            return parent_id
        return None

    def step(self, action: ParsedAction) -> dict[str, Any]:
        handler_name = f"_step_{action.base_name}"
        if action.name.startswith("failed_"):
            event = {
                "status": "failure",
                "failure_type": "injected",
                "failed_action": action.base_name,
                "node_ids": list(action.node_ids),
                "message": f"injected {action.base_name} failure",
                "injected": True,
            }
            schema = action_model_trace(action.base_name, list(action.node_ids))
            if schema is not None:
                event["action_model"] = schema
            self.events.append({"action": action.to_json(), "event": event})
            return event
        handler = getattr(self, handler_name, None)
        if handler is None:
            event = self._failure(action, "unknown_action", f"unknown action {action.name!r}")
        else:
            event = handler(action)
        self.events.append({"action": action.to_json(), "event": event})
        return event

    def _success(self, action: ParsedAction, **extra: Any) -> dict[str, Any]:
        event = {
            "status": "success",
            "action": action.base_name,
            "node_ids": list(action.node_ids),
        }
        schema = action_model_trace(action.base_name, list(action.node_ids))
        if schema is not None:
            event["action_model"] = schema
        event.update(extra)
        return event

    def _failure(self, action: ParsedAction, failure_type: str, message: str, **extra: Any) -> dict[str, Any]:
        event = {
            "status": "failure",
            "failure_type": failure_type,
            "action": action.base_name,
            "node_ids": list(action.node_ids),
            "message": message,
            "injected": False,
        }
        schema = action_model_trace(action.base_name, list(action.node_ids))
        if schema is not None:
            event["action_model"] = schema
        event.update(extra)
        return event

    def _target_state(self, action: ParsedAction, index: int = 0) -> NodeState | None:
        if len(action.node_ids) <= index:
            return None
        node_id = self.resolve_node_id(action.node_ids[index])
        if node_id is None:
            return None
        return self.states.get(node_id)

    def _part_parent(self, node_id: str) -> str | None:
        state = self.states.get(node_id)
        if state is None:
            return None
        raw_parent = state.node.metadata.get("part_of")
        if raw_parent is not None:
            resolved = self.resolve_node_id(raw_parent)
            if resolved is not None:
                return resolved
        for edge in self.graph.outgoing(node_id):
            if edge.relation == "PART_OF":
                return edge.target
        return None

    def _direct_part_ids(self, parent_id: str) -> list[str]:
        return sorted(node_id for node_id in self.states if self._part_parent(node_id) == parent_id)

    def _is_attachable_part(self, node_id: str) -> bool:
        state = self.states.get(node_id)
        return bool(
            state
            and self._part_parent(node_id) is not None
            and state.node.is_grabbable
            and state.node.is_movable
        )

    def _is_assemblable_parent(self, node_id: str) -> bool:
        state = self.states.get(node_id)
        return bool(state and state.node.is_grabbable and state.node.is_movable)

    def _set_parent_assembled(self, parent_id: str, part_ids: list[str]) -> None:
        parent_state = self.states.get(parent_id)
        if parent_state is None:
            return
        parent_state.assembled = True
        for part_id in part_ids:
            self._clear_part_spatial_state(part_id)

    def _clear_part_spatial_state(self, part_id: str) -> None:
        part_state = self.states.get(part_id)
        if part_state is None:
            return
        part_state.held = False
        part_state.location_relation = None
        part_state.location_target = None
        self.active_occlusion_edges = {
            edge
            for edge in self.active_occlusion_edges
            if edge[0] != part_id and edge[1] != part_id
        }
        self.occlusion_edge_resolution_actions = {
            edge: resolution_action
            for edge, resolution_action in self.occlusion_edge_resolution_actions.items()
            if edge in self.active_occlusion_edges
        }

    def attachment_matches(self, left_id: str, right_id: str) -> bool:
        left = self.states.get(left_id)
        right = self.states.get(right_id)
        return bool(
            left
            and right
            and (
                left.attached_to == right_id
                or right.attached_to == left_id
            )
        )

    def _step_look(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "look needs a known target")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        self._mark_focus(target.node.id)
        target.inspected = True
        return self._success(action, focus=target.node.id)

    def _step_observe(self, action: ParsedAction) -> dict[str, Any]:
        for node_id in action.node_ids:
            resolved = self.resolve_node_id(node_id)
            if resolved in self.states and self.is_visible(resolved):
                self.states[resolved].inspected = True
        return self._success(action)

    def _step_inspect(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "inspect needs a known target")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        target.inspected = True
        self._mark_focus(target.node.id)
        return self._success(action, inspected=target.node.id)

    def _step_reach(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "reach needs a known target")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        self._mark_focus(target.node.id)
        return self._success(action, focus=target.node.id)

    def _step_walk(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "walk needs a known target")
        self._mark_focus(target.node.id)
        return self._success(action, focus=target.node.id)

    def _step_open(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "open needs a known target")
        if not target.node.is_openable:
            return self._failure(action, "not_openable", f"{target.node.id} is not openable")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        if not (
            self._is_on_surface(target.node.id)
            or self._is_reachable_static_part(target.node.id)
        ):
            return self._failure(action, "not_on_surface", f"{target.node.id} must be on a surface before opening")
        if any(state.held for state in self.states.values()):
            return self._failure(action, "hands_occupied", f"{target.node.id} needs both hands free to open")
        target.open = True
        target.inspected = True
        self._mark_focus(target.node.id)
        return self._success(action, open=True)

    def _step_close(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "close needs a known target")
        if not target.node.is_openable:
            return self._failure(action, "not_openable", f"{target.node.id} is not openable")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        if any(state.held for state in self.states.values()):
            return self._failure(action, "hands_occupied", f"{target.node.id} needs both hands free to close")
        target.open = False
        return self._success(action, open=False)

    def _step_press(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "press needs a known target")
        if not target.node.is_pressable:
            return self._failure(action, "not_pressable", f"{target.node.id} is not pressable")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        if not self.is_reachable(target.node.id):
            return self._failure(action, "not_reachable", f"{target.node.id} is not reachable")
        target.pressed = True
        target.inspected = True
        self._mark_focus(target.node.id)
        return self._success(action, pressed=target.node.id)

    def _step_grab(self, action: ParsedAction) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "grab needs a known target")
        if not target.node.is_grabbable or not target.node.is_movable:
            return self._failure(action, "not_grabbable", f"{target.node.id} is not grabbable and movable")
        if not self.is_reachable(target.node.id):
            return self._failure(action, "not_reachable", f"{target.node.id} is not reachable")
        payload_ids = self._carrier_payload_ids(target.node.id)
        can_carry_contents = target.node.has_property("CARRY_CONTENTS", "STABLE_TRANSPORT")
        if payload_ids and not can_carry_contents:
            return self._failure(
                action,
                "non_empty_payload",
                f"{target.node.id} cannot be grabbed while carrying other objects",
                payload_ids=payload_ids,
                payload_count=len(payload_ids),
            )
        if target.location_relation is not None or target.location_target is not None:
            self._remove_occlusions_for_location_change(target.node.id)
        target.held = True
        target.location_relation = None
        target.location_target = None
        self._mark_focus(target.node.id)
        return self._success(action, held=target.node.id)

    def _step_pick(self, action: ParsedAction) -> dict[str, Any]:
        return self._step_grab(action)

    def _step_attach(self, action: ParsedAction) -> dict[str, Any]:
        left = self._target_state(action, 0)
        right = self._target_state(action, 1)
        if left is None or right is None:
            return self._failure(action, "missing_target", "attach needs two known parts or a part and parent")
        left_id = left.node.id
        right_id = right.node.id
        if not self.is_reachable(left_id):
            return self._failure(action, "not_reachable", f"{left_id} is not reachable")
        if not self.is_reachable(right_id):
            return self._failure(action, "not_reachable", f"{right_id} is not reachable")

        left_parent = self._part_parent(left_id)
        right_parent = self._part_parent(right_id)
        parent_id = None
        attached_parts: list[str] = []
        if left_parent and left_parent == right_parent:
            if not (self._is_attachable_part(left_id) and self._is_attachable_part(right_id)):
                return self._failure(action, "not_attachable", f"{left_id} and {right_id} are not attachable assembly parts")
            parent_id = left_parent
            attached_parts = [left_id, right_id]
            left.attached_to = right_id
            right.attached_to = left_id
        elif left_parent == right_id:
            if not self._is_attachable_part(left_id) or not self._is_assemblable_parent(right_id):
                return self._failure(action, "not_attachable", f"{left_id} cannot be attached to {right_id}")
            parent_id = right_id
            attached_parts = [left_id]
            left.attached_to = right_id
        elif right_parent == left_id:
            if not self._is_attachable_part(right_id) or not self._is_assemblable_parent(left_id):
                return self._failure(action, "not_attachable", f"{right_id} cannot be attached to {left_id}")
            parent_id = left_id
            attached_parts = [right_id]
            right.attached_to = left_id
        else:
            return self._failure(action, "not_part_related", f"{left_id} and {right_id} are not compatible parts")

        all_parts = self._direct_part_ids(parent_id)
        if all_parts and all(self.states[part_id].attached_to is not None for part_id in all_parts):
            self._set_parent_assembled(parent_id, all_parts)
        else:
            for part_id in attached_parts:
                self._clear_part_spatial_state(part_id)
        self._mark_focus(parent_id)
        return self._success(
            action,
            attached={"parts": attached_parts, "target": right_id, "parent": parent_id},
            assembled=self.states[parent_id].assembled,
        )

    def _step_assemble(self, action: ParsedAction) -> dict[str, Any]:
        if len(action.node_ids) >= 2:
            return self._step_attach(action)
        parent = self._target_state(action)
        if parent is None:
            return self._failure(action, "missing_target", "assemble needs a known parent object")
        parent_id = parent.node.id
        part_ids = self._direct_part_ids(parent_id)
        if not part_ids:
            return self._failure(action, "no_parts", f"{parent_id} has no direct parts to assemble")
        not_attachable = [part_id for part_id in part_ids if not self._is_attachable_part(part_id)]
        if not_attachable:
            return self._failure(
                action,
                "not_attachable_parts",
                f"{parent_id} has non-attachable parts",
                not_attachable=not_attachable,
            )
        unreachable = [part_id for part_id in part_ids if not self.is_reachable(part_id)]
        if unreachable:
            return self._failure(action, "parts_not_reachable", "not all parts are reachable", unreachable=unreachable)
        for part_id in part_ids:
            self.states[part_id].attached_to = parent_id
        self._set_parent_assembled(parent_id, part_ids)
        self._mark_focus(parent_id)
        return self._success(action, assembled=parent_id, parts=part_ids)

    def _step_puton(self, action: ParsedAction) -> dict[str, Any]:
        return self._place(action, relation="ON")

    def _step_putin(self, action: ParsedAction) -> dict[str, Any]:
        return self._place(action, relation="INSIDE")

    def _step_place_on(self, action: ParsedAction) -> dict[str, Any]:
        return self._place(action, relation="ON")

    def _step_place_in(self, action: ParsedAction) -> dict[str, Any]:
        return self._place(action, relation="INSIDE")

    def _place(self, action: ParsedAction, relation: str) -> dict[str, Any]:
        obj = self._target_state(action, 0)
        target = self._target_state(action, 1)
        if obj is None or target is None:
            return self._failure(action, "missing_target", f"{relation.lower()} needs object and target")
        if not obj.held:
            return self._failure(action, "not_held", f"{obj.node.id} is not held")
        if relation == "INSIDE":
            if not target.node.is_container:
                return self._failure(action, "not_container", f"{target.node.id} is not a container")
            if target.node.is_openable and not target.open:
                return self._failure(action, "closed_target", f"{target.node.id} is closed")
            if self._container_would_exceed_capacity(target.node.id, obj.node.id):
                item_count = self._container_item_count(
                    target.node.id,
                    exclude={obj.node.id},
                )
                incoming_item_count = self._capacity_item_units(obj.node.id)
                return self._failure(
                    action,
                    "container_full",
                    f"{target.node.id} would exceed max_items={target.node.max_items}",
                    max_items=target.node.max_items,
                    item_count=item_count,
                    incoming_item_count=incoming_item_count,
                    projected_item_count=item_count + incoming_item_count,
                )
        if relation == "ON" and not target.node.is_surface:
            return self._failure(action, "not_surface", f"{target.node.id} is not a surface")
        if not self.placement_edge_constraints.allows(
            source_id=obj.node.id,
            target_id=target.node.id,
            relation=relation,
            source_name=obj.node.name,
            target_name=target.node.name,
        ):
            return self._failure(
                action,
                "disallowed_placement_edge",
                f"{relation} from {obj.node.id} to {target.node.id} is disallowed by placement edge constraints",
            )
        old_location = (obj.location_relation, obj.location_target)
        new_location = (relation, target.node.id)
        if old_location != new_location:
            self._remove_occlusions_for_location_change(obj.node.id)
        obj.held = False
        obj.location_relation = relation
        obj.location_target = target.node.id
        self._mark_focus(target.node.id)
        return self._success(action, placed={"object": obj.node.id, "relation": relation, "target": target.node.id})

    def _step_move_aside(self, action: ParsedAction) -> dict[str, Any]:
        return self._resolve_blocker(action, moved_aside=True)

    def _step_clear(self, action: ParsedAction) -> dict[str, Any]:
        return self._resolve_blocker(action, cleared=True)

    def _resolve_blocker(self, action: ParsedAction, moved_aside: bool = False, cleared: bool = False) -> dict[str, Any]:
        target = self._target_state(action)
        if target is None:
            return self._failure(action, "missing_target", "blocker resolution needs a known target")
        if not self.is_visible(target.node.id):
            return self._failure(action, "not_visible", f"{target.node.id} is not visible")
        if moved_aside and self._hidden_targets_blocked_by_resolution(target.node.id, "open"):
            return self._failure(action, "requires_open", f"{target.node.id} is a container occluder and must be opened")
        if moved_aside and not target.node.is_movable:
            return self._failure(action, "not_movable", f"{target.node.id} is not movable")
        if cleared and self._hidden_targets_blocked_by(target.node.id):
            return self._failure(action, "unsupported_resolution", "clear does not resolve occlusion")
        if moved_aside:
            target.moved_aside = True
        if cleared:
            target.cleared = True
        self._mark_focus(target.node.id)
        return self._success(action, resolved_blocker=target.node.id)

    def _step_recover(self, action: ParsedAction) -> dict[str, Any]:
        extra: dict[str, Any] = {"recovered": True}
        recovered_action = action.arguments.get("failed_action")
        if recovered_action:
            extra["recovered_action"] = recovered_action
        if action.node_ids:
            extra["recovered_node_ids"] = list(action.node_ids)
        return self._success(action, **extra)

    def _step_stop(self, action: ParsedAction) -> dict[str, Any]:
        return self._success(action, stopped=True)

    def _mark_focus(self, node_id: str) -> None:
        self.focus = node_id
        if node_id not in self.visited:
            self.visited.append(node_id)

    def location_matches(self, object_id: str, relation: str, target_id: str) -> bool:
        if object_id not in self.states or target_id not in self.states:
            return False
        state = self.states[object_id]
        return state.location_relation == _canonical_location(relation) and state.location_target == target_id

    def graph_relation_matches(self, source_id: str, relation: str, target_id: str) -> bool:
        relation = normalize_relation(relation)
        for edge in self.graph.edges:
            if edge.source == source_id and edge.target == target_id and edge.relation == relation:
                return True
        return False


class WorldBackend(Protocol):
    """Backend contract used by trajectory collection.

    Implementations can be symbolic, simulator-backed, or robot-backed. The
    harness only needs observations, action execution events, snapshots, and a
    success signal.
    """

    @property
    def name(self) -> str:
        ...

    def observe(self) -> dict[str, Any]:
        ...

    def step(self, action: ParsedAction) -> dict[str, Any]:
        ...

    def snapshot(self) -> dict[str, Any]:
        ...

    def success(self) -> bool:
        ...

    def metrics(self, initial_observation: dict[str, Any]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class SymbolicBackend:
    name = "symbolic"

    def __init__(
        self,
        graph: ViewGraph,
        task: TaskRecord,
        placement_edge_constraints: PlacementEdgeConstraints | None = None,
    ) -> None:
        self.world = SemanticWorld(graph, task, placement_edge_constraints)
        self.evaluator = TrajectoryEvaluator(task, self.world)

    def observe(self) -> dict[str, Any]:
        return self.world.observe()

    def step(self, action: ParsedAction) -> dict[str, Any]:
        return self.world.step(action)

    def snapshot(self) -> dict[str, Any]:
        return self.world.snapshot()

    def success(self) -> bool:
        return self.evaluator.success()

    def metrics(self, initial_observation: dict[str, Any]) -> dict[str, Any]:
        return self.evaluator.metrics(initial_observation)

    def close(self) -> None:
        return None


class TrajectoryEvaluator:
    def __init__(self, task: TaskRecord, world: SemanticWorld) -> None:
        self.task = task
        self.world = world

    def success(self) -> bool:
        if self.task.task_type == "navigation":
            target_id = self.world.resolve_node_id(self.task.objects.get("target"))
            return bool(target_id and target_id in self.world.visited)
        if self._has_structured_goal():
            return self._evaluate_goal().success
        placements = self._expected_placements()
        if placements:
            return all(
                self.world.location_matches(object_id, relation, target_id)
                for object_id, target_id, relation in placements
            )
        return self._evaluate_goal().success

    def metrics(self, initial_observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": self.success(),
            "goal": self._goal_metrics(),
            "spatial": self._spatial_metrics(),
            "temporal": self._temporal_metrics(),
            "memory": self._memory_metrics(initial_observation),
            "failure_recovery": self._failure_metrics(),
        }

    def _has_structured_goal(self) -> bool:
        return isinstance(self.task.task_completion_criterion, (dict, list))

    def _evaluate_goal(self):
        expression = normalize_goal_expression(self.task.task_completion_criterion)
        return evaluate_goal_expression(expression, self._predicate_met)

    def _goal_metrics(self) -> dict[str, Any]:
        expression = normalize_goal_expression(self.task.task_completion_criterion)
        try:
            evaluation = evaluate_goal_expression(expression, self._predicate_met)
            return {
                "success": evaluation.success,
                "predicates": extract_goal_predicates(expression),
                "checks": evaluation.checks,
            }
        except Exception as exc:  # noqa: BLE001 - metrics should record malformed goals without crashing.
            return {
                "success": False,
                "predicates": extract_goal_predicates(expression),
                "checks": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _expected_placements(self) -> list[tuple[str, str, str]]:
        objects = self.task.objects
        placements = []
        for placement in objects.get("placements", []) or []:
            object_id = self.world.resolve_node_id(placement.get("object"))
            target_id = self.world.resolve_node_id(placement.get("target"))
            relation = _canonical_location(str(placement.get("relation", "ON"))) or "ON"
            if object_id and target_id:
                placements.append((object_id, target_id, relation))
        if placements:
            return placements
        if "object" in objects and "target" in objects:
            object_id = self.world.resolve_node_id(objects["object"])
            target_id = self.world.resolve_node_id(objects["target"])
            relation = _canonical_location(str(objects.get("relation", "ON"))) or "ON"
            if object_id and target_id:
                return [(object_id, target_id, relation)]
        if objects.get("objects") and "target" in objects:
            target_id = self.world.resolve_node_id(objects["target"])
            relation = _canonical_location(str(objects.get("relation", "ON"))) or "ON"
            if target_id:
                return [
                    (object_id, target_id, relation)
                    for raw_id in objects["objects"]
                    if (object_id := self.world.resolve_node_id(raw_id))
                ]
        return []

    def _criterion_relations(self) -> list[tuple[str, str, str]]:
        if not isinstance(self.task.task_completion_criterion, str):
            return []
        relations = []
        for match in re.finditer(r"\(([A-Za-z_]+),\s*([^,]+),\s*([^)]+)\)", self.task.task_completion_criterion):
            relation = normalize_relation(match.group(1))
            left = match.group(2).strip()
            right = match.group(3).strip()
            if relation == "CLOSE":
                continue
            relations.append((relation, left, right))
        return relations

    def _criterion_relation_met(self, item: tuple[str, str, str]) -> bool:
        relation, left, right = item
        return self._predicate_met(relation, [left, right])

    def _predicate_met(self, predicate: str, args: list[Any]) -> bool:
        predicate = normalize_relation(predicate)
        if predicate == "CLOSE":
            if len(args) < 2:
                return False
            target_id = self.world.resolve_node_id(args[1])
            return bool(target_id and target_id in self.world.visited)
        if predicate == "PRESSED_TIMES":
            if len(args) < 2:
                return False
            target_id = self.world.resolve_node_id(args[0])
            required_count = _parse_required_count(args[1])
            return bool(
                target_id
                and required_count is not None
                and self._successful_action_count("press", target_id) >= required_count
            )
        if predicate == "PRESSED_SEQUENCE":
            raw_sequence = args[0] if len(args) == 1 and isinstance(args[0], list) else args
            expected_sequence = []
            for raw_target in raw_sequence:
                target_id = self.world.resolve_node_id(raw_target)
                if target_id is None:
                    return False
                expected_sequence.append(target_id)
            return self._successful_press_sequence() == expected_sequence
        if predicate == "ACTION_COUNT":
            if len(args) < 3:
                return False
            action_name = str(args[0]).strip().lower().removeprefix("failed_")
            target_id = self.world.resolve_node_id(args[1])
            required_count = _parse_required_count(args[2])
            return bool(
                action_name
                and target_id
                and required_count is not None
                and self._successful_action_count(action_name, target_id) >= required_count
            )
        if predicate in {"AT_MOST_INSIDE", "CONTAINS_AT_MOST", "MAX_ITEMS"}:
            if len(args) < 2:
                return False
            target_id = self.world.resolve_node_id(args[0])
            required_count = _parse_required_count(args[1])
            return bool(
                target_id
                and target_id in self.world.states
                and required_count is not None
                and self.world._container_item_count(target_id) <= required_count
            )
        if predicate in {
            "OPEN",
            "CLOSED",
            "HELD",
            "VISIBLE",
            "REACHABLE",
            "MOVED_ASIDE",
            "CLEARED",
            "INSPECTED",
            "ASSEMBLED",
            "PRESSED",
        }:
            if not args:
                return False
            node_id = self.world.resolve_node_id(args[0])
            if node_id is None or node_id not in self.world.states:
                return False
            state = self.world.states[node_id]
            if predicate == "OPEN":
                return state.open
            if predicate == "CLOSED":
                return state.node.is_openable and not state.open
            if predicate == "HELD":
                return state.held
            if predicate == "VISIBLE":
                return self.world.is_visible(node_id)
            if predicate == "REACHABLE":
                return self.world.is_reachable(node_id)
            if predicate == "MOVED_ASIDE":
                return state.moved_aside
            if predicate == "CLEARED":
                return state.cleared
            if predicate == "INSPECTED":
                return state.inspected
            if predicate == "ASSEMBLED":
                parts = self.world._direct_part_ids(node_id)
                return state.assembled or bool(
                    parts and all(self.world.states[part_id].attached_to is not None for part_id in parts)
                )
            if predicate == "PRESSED":
                return state.pressed
        if predicate == "ATTACHED" and len(args) == 1:
            node_id = self.world.resolve_node_id(args[0])
            return bool(node_id and self.world.states[node_id].attached_to)
        if len(args) < 2:
            return False
        left_id = self.world.resolve_node_id(args[0])
        right_id = self.world.resolve_node_id(args[1])
        if left_id is None or right_id is None:
            return False
        if predicate == "ATTACHED":
            return self.world.attachment_matches(left_id, right_id)
        if predicate in {"ON", "INSIDE", "IN"}:
            return self.world.location_matches(left_id, predicate, right_id)
        if predicate == "OCCLUDED_BY":
            return right_id in self.world._active_blockers(left_id, relations=OCCLUSION_RELATIONS)
        if predicate in OCCLUSION_RELATIONS:
            return left_id in self.world._active_blockers(right_id, relations={predicate})
        if predicate == "PART_OF":
            state = self.world.states.get(left_id)
            return bool(
                state
                and (
                    state.node.metadata.get("part_of") == right_id
                    or self.world.graph_relation_matches(left_id, "PART_OF", right_id)
                )
            )
        return self.world.graph_relation_matches(left_id, predicate, right_id)

    def _successful_action_count(self, action_name: str, target_id: str) -> int:
        count = 0
        for item in self.world.events:
            action = item.get("action", {})
            event = item.get("event", {})
            if not isinstance(action, dict) or not isinstance(event, dict):
                continue
            if event.get("status") != "success":
                continue
            if str(action.get("base_name") or action.get("name") or "").lower() != action_name:
                continue
            node_ids = [str(node_id) for node_id in action.get("node_ids", [])]
            event_node_ids = [str(node_id) for node_id in event.get("node_ids", [])]
            if target_id in node_ids or target_id in event_node_ids or event.get("pressed") == target_id:
                count += 1
        return count

    def _successful_press_sequence(self) -> list[str]:
        sequence = []
        for item in self.world.events:
            action = item.get("action", {})
            event = item.get("event", {})
            if not isinstance(action, dict) or not isinstance(event, dict):
                continue
            if event.get("status") != "success":
                continue
            if str(action.get("base_name") or action.get("name") or "").lower() != "press":
                continue
            target_id = event.get("pressed")
            if target_id is None:
                node_ids = event.get("node_ids") or action.get("node_ids") or []
                target_id = node_ids[0] if node_ids else None
            resolved = self.world.resolve_node_id(target_id)
            if resolved is not None:
                sequence.append(resolved)
        return sequence

    def _spatial_metrics(self) -> dict[str, Any]:
        subtasks = [
            item
            for item in self.task.metadata.get("constraint_subtasks", [])
            if item.get("setting") in {"spatial", "long_horizon"} or item.get("type") == "resolve_access_constraint"
        ]
        resolved_events = [
            item
            for item in self.world.events
            if item["event"].get("status") == "success"
            and item["action"]["base_name"] in {"open", "move_aside"}
        ]
        return {
            "num_constraint_subtasks": len(subtasks),
            "num_resolution_actions": len(resolved_events),
            "precondition_failures": sum(
                1
                for item in self.world.events
                if item["event"].get("status") == "failure" and not item["event"].get("injected")
            ),
        }

    def _temporal_metrics(self) -> dict[str, Any]:
        expected = []
        seen_keys: set[tuple[int, str, str, str]] = set()
        for item in self.task.metadata.get("constraint_subtasks", []):
            if item.get("type") != "ordered_placement":
                continue
            step = int(item.get("step", len(expected) + 1))
            object_id = self.world.resolve_node_id(item.get("object"))
            target_id = self.world.resolve_node_id(item.get("target"))
            relation = _canonical_location(str(item.get("relation", "ON"))) or "ON"
            if object_id is None or target_id is None:
                continue
            key = (step, object_id, target_id, relation)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            expected.append(key)
        expected.sort()
        actual = []
        for index, item in enumerate(self.world.events):
            placed = item["event"].get("placed")
            if item["event"].get("status") == "success" and isinstance(placed, dict):
                actual.append((index, placed["object"], placed["target"], placed["relation"]))
        cursor = -1
        ordered = True
        matched = 0
        for _, object_id, target_id, relation in expected:
            next_index = next(
                (
                    index
                    for index, actual_object, actual_target, actual_relation in actual
                    if index > cursor
                    and actual_object == object_id
                    and actual_target == target_id
                    and actual_relation == relation
                ),
                None,
            )
            if next_index is None:
                ordered = False
                continue
            cursor = next_index
            matched += 1
        return {
            "expected_steps": len(expected),
            "matched_steps": matched,
            "ordered": ordered if expected else None,
        }

    def _memory_metrics(self, initial_observation: dict[str, Any]) -> dict[str, Any]:
        constraint = self.task.metadata.get("memory_constraint")
        if not isinstance(constraint, dict):
            return {
                "prior_observation_available": False,
                "object_hidden_initially": None,
                "used_remembered_anchor": None,
            }
        object_id = self.world.resolve_node_id(constraint.get("remember_object"))
        anchor_id = self.world.resolve_node_id(constraint.get("remember_anchor"))
        visible_initial = {item["id"] for item in initial_observation.get("visible_nodes", [])}
        object_hidden = object_id not in visible_initial if object_id is not None else None
        first_object_action = None
        first_anchor_action = None
        for index, item in enumerate(self.world.events):
            node_ids = {
                resolved
                for raw_id in item["action"].get("node_ids", [])
                if (resolved := self.world.resolve_node_id(raw_id)) is not None
            }
            if object_id in node_ids and first_object_action is None:
                first_object_action = index
            if anchor_id in node_ids and first_anchor_action is None:
                first_anchor_action = index
        used_anchor = None
        if anchor_id is not None:
            used_anchor = first_anchor_action is not None and (
                first_object_action is None or first_anchor_action <= first_object_action
            )
        return {
            "prior_observation_available": True,
            "remembered_object": object_id,
            "remembered_anchor": anchor_id,
            "object_hidden_initially": object_hidden,
            "used_remembered_anchor": used_anchor,
        }

    def _failure_metrics(self) -> dict[str, Any]:
        failures = [
            (index, item)
            for index, item in enumerate(self.world.events)
            if item["event"].get("status") == "failure"
        ]
        execution_failures = [
            (index, item)
            for index, item in failures
            if item["event"].get("injected") is True or item["event"].get("failure_type") == "injected"
        ]
        semantic_failures = [
            (index, item)
            for index, item in failures
            if not (item["event"].get("injected") is True or item["event"].get("failure_type") == "injected")
        ]

        execution_recovered = False
        execution_retried = False
        if execution_failures:
            first_execution_index, first_execution_failure = execution_failures[0]
            failed_action = first_execution_failure["event"].get("failed_action") or first_execution_failure[
                "action"
            ].get("base_name")
            failed_nodes = list(first_execution_failure["action"].get("node_ids", []))
            for item in self.world.events[first_execution_index + 1 :]:
                if item["event"].get("status") == "success" and item["action"].get("base_name") == "recover":
                    execution_recovered = True
                if (
                    item["event"].get("status") == "success"
                    and failed_action
                    and item["action"].get("base_name") == failed_action
                    and item["action"].get("node_ids") == failed_nodes
                ):
                    execution_retried = True
                    execution_recovered = True

        semantic_replanned = False
        if semantic_failures:
            first_semantic_index, first_semantic_failure = semantic_failures[0]
            failed_signature = (
                first_semantic_failure["action"].get("base_name"),
                tuple(first_semantic_failure["action"].get("node_ids", [])),
            )
            semantic_replanned = any(
                item["event"].get("status") == "success"
                and item["action"].get("base_name") not in {"recover", "stop"}
                and (
                    item["action"].get("base_name"),
                    tuple(item["action"].get("node_ids", [])),
                )
                != failed_signature
                for item in self.world.events[first_semantic_index + 1 :]
            )

        failure_types = [
            str(item["event"].get("failure_type"))
            for _, item in semantic_failures
            if item["event"].get("failure_type") is not None
        ]
        return {
            "failure_observed": bool(failures),
            "recovered_after_failure": execution_recovered or semantic_replanned,
            "retried_failed_action": execution_retried,
            "execution": {
                "failure_observed": bool(execution_failures),
                "failure_count": len(execution_failures),
                "recovered_after_failure": execution_recovered,
                "retried_failed_action": execution_retried,
            },
            "semantic": {
                "failure_observed": bool(semantic_failures),
                "failure_count": len(semantic_failures),
                "failure_types": list(dict.fromkeys(failure_types)),
                "replanned_after_failure": semantic_replanned,
            },
        }


def load_collection_condition(
    path: str | Path,
    condition_id: str | None = None,
) -> dict[str, Any]:
    """Load one condition for ``collect-trajectories``.

    The input may be a task-design JSON containing
    ``conditional_interventions.conditions``, an object with a top-level
    ``conditions`` array, or one condition object.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"collection condition file does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: collection condition file must be a JSON object")

    condition_container = payload.get("conditional_interventions", payload)
    if not isinstance(condition_container, dict):
        raise ValueError(f"{source}: conditional_interventions must be a JSON object")
    raw_conditions = condition_container.get("conditions")
    if raw_conditions is None and condition_container.get("condition_id"):
        raw_conditions = [condition_container]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError(f"{source}: no collection conditions were found")

    conditions: dict[str, dict[str, Any]] = {}
    for index, raw_condition in enumerate(raw_conditions, start=1):
        if not isinstance(raw_condition, dict):
            raise ValueError(f"{source}: condition {index} must be a JSON object")
        normalized = _normalize_collection_condition(raw_condition, source=source, index=index)
        normalized_id = str(normalized["condition_id"])
        if normalized_id in conditions:
            raise ValueError(f"{source}: duplicate condition_id {normalized_id!r}")
        conditions[normalized_id] = normalized

    selected_id = str(condition_id or "").strip()
    if not selected_id:
        if len(conditions) != 1:
            raise ValueError(
                f"{source}: choose --condition-id from {sorted(conditions)}"
            )
        selected_id = next(iter(conditions))
    if selected_id not in conditions:
        raise ValueError(
            f"{source}: unknown condition_id {selected_id!r}; available ids: {sorted(conditions)}"
        )
    return copy.deepcopy(conditions[selected_id])


def load_task_collection_condition(
    task: TaskRecord,
    condition_id: str,
) -> dict[str, Any]:
    """Load one collection condition embedded in a TaskRecord's metadata."""

    condition_container = task.metadata.get("collection_conditions")
    if condition_container is None:
        condition_container = task.metadata.get("conditional_interventions")
    source = Path(f"outputs/<task:{task.task_id}>")
    if not isinstance(condition_container, dict):
        raise ValueError(
            f"task {task.task_id!r} has no metadata.collection_conditions"
        )
    raw_conditions = condition_container.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError(
            f"task {task.task_id!r} metadata.collection_conditions.conditions "
            "must be a non-empty array"
        )
    selected_id = str(condition_id or "").strip()
    for index, raw_condition in enumerate(raw_conditions, start=1):
        if not isinstance(raw_condition, dict):
            raise ValueError(
                f"task {task.task_id!r} collection condition {index} must be an object"
            )
        normalized = _normalize_collection_condition(
            raw_condition,
            source=source,
            index=index,
        )
        if normalized["condition_id"] == selected_id:
            return normalized
    available_ids = [
        str(item.get("condition_id"))
        for item in raw_conditions
        if isinstance(item, dict) and item.get("condition_id")
    ]
    raise ValueError(
        f"task {task.task_id!r} has no collection condition {selected_id!r}; "
        f"available ids: {available_ids}"
    )


def _normalize_collection_condition(
    condition: dict[str, Any],
    *,
    source: Path,
    index: int,
) -> dict[str, Any]:
    normalized = copy.deepcopy(condition)
    condition_id = str(normalized.get("condition_id") or "").strip()
    if not condition_id:
        raise ValueError(f"{source}: condition {index} needs condition_id")
    trigger = normalized.get("trigger")
    disturbance = normalized.get("graph_disturbance")
    if not isinstance(trigger, dict):
        raise ValueError(f"{source}: condition {condition_id} needs a trigger object")
    if not isinstance(disturbance, dict):
        raise ValueError(
            f"{source}: condition {condition_id} needs a graph_disturbance object"
        )

    trigger_type = str(trigger.get("type") or "").strip().lower()
    supported_trigger_types = {
        "on_container_max_items_reached",
        "on_any_container_max_items_reached",
    }
    if trigger_type not in supported_trigger_types:
        raise ValueError(
            f"{source}: collect-trajectories condition {condition_id} only supports "
            "trigger types on_container_max_items_reached or "
            "on_any_container_max_items_reached"
        )
    trigger["type"] = trigger_type
    if trigger_type == "on_container_max_items_reached":
        if not trigger.get("node_id"):
            raise ValueError(f"{source}: condition {condition_id} trigger needs node_id")
    else:
        node_ids = trigger.get("node_ids")
        if not isinstance(node_ids, list) or not node_ids:
            raise ValueError(
                f"{source}: condition {condition_id} trigger needs a non-empty node_ids array"
            )
        trigger["node_ids"] = [str(node_id) for node_id in node_ids]

    operation = str(disturbance.get("operation") or "").strip().lower()
    if operation != "add_object":
        raise ValueError(
            f"{source}: collect-trajectories condition {condition_id} only supports "
            f"operation add_object, got {operation!r}"
        )
    object_spec = disturbance.get("object")
    if not isinstance(object_spec, dict):
        raise ValueError(
            f"{source}: condition {condition_id} add_object needs an object definition"
        )
    if not object_spec.get("id") or not object_spec.get("name"):
        raise ValueError(
            f"{source}: condition {condition_id} add_object object needs id and name"
        )
    if object_spec.get("copy_from") is not None:
        raise ValueError(
            f"{source}: collect-trajectories condition {condition_id} does not support "
            "object.copy_from; use the closed-loop intervention runtime"
        )
    success_policy = disturbance.get("success_policy") or {
        "type": "existing_task_goal"
    }
    if not isinstance(success_policy, dict):
        raise ValueError(
            f"{source}: condition {condition_id} success_policy must be an object"
        )
    policy_type = str(success_policy.get("type") or "").strip().lower()
    supported_policy_types = {"existing_task_goal", "trigger_container_goal"}
    if policy_type not in supported_policy_types:
        raise ValueError(
            f"{source}: collect-trajectories condition {condition_id} only supports "
            "success_policy.type existing_task_goal or trigger_container_goal"
        )
    success_policy["type"] = policy_type
    if policy_type == "trigger_container_goal":
        predicate = normalize_relation(str(success_policy.get("predicate") or "INSIDE"))
        if predicate == "IN":
            predicate = "INSIDE"
        if predicate != "INSIDE":
            raise ValueError(
                f"{source}: condition {condition_id} trigger_container_goal only supports "
                "predicate INSIDE"
            )
        if trigger_type != "on_any_container_max_items_reached":
            raise ValueError(
                f"{source}: condition {condition_id} trigger_container_goal requires "
                "trigger type on_any_container_max_items_reached"
            )
        success_policy["predicate"] = predicate
    disturbance["success_policy"] = success_policy
    relation = disturbance.get("relation")
    if relation is None:
        raise ValueError(
            f"{source}: condition {condition_id} add_object needs a spawn relation"
        )
    if relation is not None:
        relation = normalize_relation(str(relation))
        relation = "INSIDE" if relation == "IN" else relation
        if relation not in {"ON", "INSIDE", "BENEATH"}:
            raise ValueError(
                f"{source}: condition {condition_id} add_object relation must be "
                "ON, INSIDE/IN, or BENEATH"
            )
        if not disturbance.get("target"):
            raise ValueError(f"{source}: condition {condition_id} disturbance needs target")
    disturbance["operation"] = operation
    disturbance["relation"] = relation
    normalized["condition_id"] = condition_id
    normalized["intervention_type"] = str(
        normalized.get("intervention_type") or operation
    ).strip().lower()
    normalized["trigger"] = trigger
    normalized["graph_disturbance"] = disturbance
    return normalized


def _goal_subject_argument_indexes(predicate: str, args: list[Any]) -> tuple[int, ...]:
    predicate = normalize_relation(predicate)
    if predicate == "CLOSE":
        return (1,) if len(args) > 1 else ()
    if predicate == "PRESSED_SEQUENCE":
        return ()
    return (0,) if args else ()


def _project_goal_expression(
    expression: Any,
    subject_matches: Callable[[Any], bool],
) -> Any | None:
    """Keep only goal atoms whose subject matches a selected object."""

    expression = normalize_goal_expression(expression)
    if isinstance(expression, dict):
        if "predicate" in expression:
            predicate = normalize_relation(str(expression["predicate"]))
            raw_args = expression.get("args", expression.get("arguments", []))
            args = list(raw_args) if isinstance(raw_args, (list, tuple)) else [raw_args]
            indexes = _goal_subject_argument_indexes(predicate, args)
            return (
                copy.deepcopy(expression)
                if any(index < len(args) and subject_matches(args[index]) for index in indexes)
                else None
            )
        for operator in ("and", "or"):
            if operator not in expression:
                continue
            raw_children = expression[operator]
            children = (
                list(raw_children)
                if isinstance(raw_children, (list, tuple))
                else [raw_children]
            )
            projected = [
                child
                for raw_child in children
                if (child := _project_goal_expression(raw_child, subject_matches)) is not None
            ]
            return {operator: projected} if projected else None
        if "not" in expression:
            projected = _project_goal_expression(expression["not"], subject_matches)
            return {"not": projected} if projected is not None else None
        if "final" in expression:
            return _project_goal_expression(expression["final"], subject_matches)
        return None
    if isinstance(expression, list):
        if not expression:
            return None
        head = expression[0]
        normalized_head = normalize_relation(head) if isinstance(head, str) else ""
        if normalized_head in {"AND", "OR"}:
            projected = [
                child
                for raw_child in expression[1:]
                if (child := _project_goal_expression(raw_child, subject_matches)) is not None
            ]
            return [normalized_head, *projected] if projected else None
        if normalized_head == "NOT":
            if len(expression) != 2:
                return None
            projected = _project_goal_expression(expression[1], subject_matches)
            return ["NOT", projected] if projected is not None else None
        if isinstance(head, str):
            args = list(expression[1:])
            indexes = _goal_subject_argument_indexes(normalized_head, args)
            return (
                copy.deepcopy(expression)
                if any(index < len(args) and subject_matches(args[index]) for index in indexes)
                else None
            )
    return None


def _replace_goal_subject(
    expression: Any,
    subject_matches: Callable[[Any], bool],
    replacement: str,
) -> Any:
    expression = copy.deepcopy(expression)
    if isinstance(expression, dict):
        if "predicate" in expression:
            predicate = normalize_relation(str(expression["predicate"]))
            args_key = "args" if "args" in expression else "arguments"
            raw_args = expression.get(args_key, [])
            args = list(raw_args) if isinstance(raw_args, (list, tuple)) else [raw_args]
            for index in _goal_subject_argument_indexes(predicate, args):
                if index < len(args) and subject_matches(args[index]):
                    args[index] = replacement
            expression[args_key] = args
            return expression
        for key in ("and", "or", "not", "final"):
            if key not in expression:
                continue
            value = expression[key]
            if key in {"and", "or"} and isinstance(value, (list, tuple)):
                expression[key] = [
                    _replace_goal_subject(item, subject_matches, replacement)
                    for item in value
                ]
            else:
                expression[key] = _replace_goal_subject(value, subject_matches, replacement)
        return expression
    if isinstance(expression, list):
        if not expression:
            return expression
        head = expression[0]
        normalized_head = normalize_relation(head) if isinstance(head, str) else ""
        if normalized_head in {"AND", "OR", "NOT"}:
            return [
                expression[0],
                *[
                    _replace_goal_subject(item, subject_matches, replacement)
                    for item in expression[1:]
                ],
            ]
        args = list(expression[1:])
        for index in _goal_subject_argument_indexes(normalized_head, args):
            if index < len(args) and subject_matches(args[index]):
                args[index] = replacement
        return [expression[0], *args]
    return expression


def _append_goal_conjunct(expression: Any, addition: Any) -> Any:
    """Append one runtime goal without rewriting the existing expression."""

    current = copy.deepcopy(expression)
    added = copy.deepcopy(addition)
    if current is None or current == "":
        return added
    if isinstance(current, dict) and set(current) == {"and"}:
        raw_children = current["and"]
        children = list(raw_children) if isinstance(raw_children, (list, tuple)) else [raw_children]
        if added not in children:
            children.append(added)
        return {"and": children}
    if current == added:
        return current
    return {"and": [current, added]}


def _node_to_condition_object_spec(node: Node) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": node.id,
        "name": node.name,
        "category": node.category,
        "properties": list(node.properties),
        "states": list(node.states),
    }
    if node.room is not None:
        payload["room"] = node.room
    if node.parent is not None:
        payload["parent"] = node.parent
    payload.update(copy.deepcopy(node.metadata))
    return payload


class CollectionConditionRuntime:
    """Add one absent task object when a container reaches max_items."""

    def __init__(self, condition: dict[str, Any] | None) -> None:
        self.condition = copy.deepcopy(condition) if condition is not None else None
        self.applied = False

    def validate_initial_state(
        self,
        backend: WorldBackend,
        observation: dict[str, Any],
    ) -> None:
        if self.condition is None:
            return
        symbolic = self._symbolic_backend(backend)
        disturbance = self.condition["graph_disturbance"]
        self._validate_add_object_initial_state(symbolic, disturbance)

    def before_step(
        self,
        backend: WorldBackend,
        *,
        step_number: int,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.condition is None or self.applied:
            return []
        symbolic = self._symbolic_backend(backend)
        trigger = self.condition["trigger"]
        trigger_container_id = self._matching_trigger_container(
            symbolic,
            trigger=trigger,
        )
        if trigger_container_id is None:
            return []

        disturbance = self.condition["graph_disturbance"]
        before_observation = symbolic.observe()
        before_snapshot = symbolic.snapshot()
        details = self._apply_add_object(
            symbolic,
            disturbance,
            trigger_container_id=trigger_container_id,
        )
        after_observation = symbolic.observe()
        after_snapshot = symbolic.snapshot()
        before_visible = {
            str(node.get("id"))
            for node in before_observation.get("visible_nodes", [])
            if isinstance(node, dict) and node.get("id") is not None
        }
        after_visible = {
            str(node.get("id"))
            for node in after_observation.get("visible_nodes", [])
            if isinstance(node, dict) and node.get("id") is not None
        }
        report = {
            "condition_id": self.condition["condition_id"],
            "intervention_type": self.condition["intervention_type"],
            "step": step_number,
            "phase": "before_observation",
            "trigger": copy.deepcopy(trigger),
            "trigger_container_id": trigger_container_id,
            "trigger_container_name": symbolic.world.states[trigger_container_id].node.name,
            "operation": "add_object",
            "spec": copy.deepcopy(disturbance),
            "details": details,
            "pre_state_hash": _state_hash(before_snapshot),
            "post_state_hash": _state_hash(after_snapshot),
            "state_changed": before_snapshot != after_snapshot,
            "new_visible_nodes": sorted(after_visible - before_visible),
            "new_hidden_nodes": sorted(before_visible - after_visible),
        }
        self.applied = True
        return [report]

    @staticmethod
    def _symbolic_backend(backend: WorldBackend) -> SymbolicBackend:
        if not isinstance(backend, SymbolicBackend):
            raise ValueError("collection conditions currently require the symbolic backend")
        return backend

    def _matching_trigger_container(
        self,
        backend: SymbolicBackend,
        *,
        trigger: dict[str, Any],
    ) -> str | None:
        trigger_type = trigger.get("type")
        if trigger_type == "on_container_max_items_reached":
            container_refs = [trigger.get("node_id")]
        elif trigger_type == "on_any_container_max_items_reached":
            container_refs = list(trigger.get("node_ids") or [])
        else:
            raise ValueError(
                "collect-trajectories only supports on_container_max_items_reached "
                "or on_any_container_max_items_reached"
            )
        for container_ref in container_refs:
            container_id = backend.world.resolve_node_id(container_ref)
            if container_id is None:
                raise ValueError(
                    f"collection condition {self.condition['condition_id']} has unknown "
                    f"trigger container {container_ref!r}"
                )
            state = backend.world.states[container_id]
            if not state.node.is_container:
                raise ValueError(
                    f"collection condition trigger node is not a container: {container_id}"
                )
            max_items = state.node.max_items
            if max_items is None:
                raise ValueError(
                    f"collection condition trigger container has no max_items: {container_id}"
                )
            if backend.world._container_item_count(container_id) >= max_items:
                return container_id
        return None

    def _validate_add_object_initial_state(
        self,
        backend: SymbolicBackend,
        disturbance: dict[str, Any],
    ) -> None:
        world = backend.world
        object_spec = disturbance["object"]
        object_id = str(object_spec["id"])
        if object_id in world.states:
            raise ValueError(
                f"collection condition {self.condition['condition_id']} add_object node "
                f"{object_id!r} must be absent from the initial view graph"
            )
        self._resolve_add_object_target(backend, disturbance)
        object_name = str(object_spec.get("name") or "")
        references = {object_id, object_name} - {""}
        projected = _project_goal_expression(
            backend.evaluator.task.task_completion_criterion,
            lambda raw: str(raw) in references,
        )
        policy_type = disturbance["success_policy"]["type"]
        if policy_type == "existing_task_goal" and projected is None:
            raise ValueError(
                f"collection condition {self.condition['condition_id']} requires the "
                f"initial success criterion to reference add_object node "
                f"{object_id!r}/{object_name!r}"
            )
        if policy_type == "trigger_container_goal" and projected is not None:
            raise ValueError(
                f"collection condition {self.condition['condition_id']} requires the "
                f"initial success criterion not to reference add_object node "
                f"{object_id!r}/{object_name!r}; its goal is added at runtime"
            )

    def _resolve_add_object_target(
        self,
        backend: SymbolicBackend,
        disturbance: dict[str, Any],
    ) -> str:
        relation = disturbance["relation"]
        target_ref = disturbance.get("target")
        target_id = backend.world.resolve_node_id(target_ref)
        if target_id is None:
            raise ValueError(
                f"collection condition {self.condition['condition_id']} has unknown "
                f"add_object target {target_ref!r}"
            )
        target = backend.world.states[target_id]
        if relation == "ON" and not target.node.is_surface:
            raise ValueError(f"collection condition ON target is not a surface: {target_id}")
        if relation == "INSIDE" and not target.node.is_container:
            raise ValueError(
                f"collection condition INSIDE target is not a container: {target_id}"
            )
        return target_id

    def _apply_add_object(
        self,
        backend: SymbolicBackend,
        disturbance: dict[str, Any],
        *,
        trigger_container_id: str,
    ) -> dict[str, Any]:
        world = backend.world
        object_spec = copy.deepcopy(disturbance["object"])
        node = Node.from_dict(object_spec)
        if node.id in world.states:
            raise ValueError(f"add_object node id already exists: {node.id!r}")
        target_id = self._resolve_add_object_target(backend, disturbance)

        state = NodeState(
            node=node,
            location_relation=disturbance["relation"],
            location_target=target_id,
            open="OPEN" in node.states,
            pressed="PRESSED" in node.states,
        )
        world.states[node.id] = state
        world._name_to_id.setdefault(node.id, node.id)
        world._name_to_id.setdefault(node.name, node.id)
        world._name_to_id.setdefault(node.name.lower(), node.id)
        success_policy = disturbance["success_policy"]
        policy_type = success_policy["type"]
        goal_update: dict[str, Any] = {"type": policy_type, "changed": False}
        if policy_type == "trigger_container_goal":
            trigger_container = world.states[trigger_container_id].node
            added_expression = [
                success_policy.get("predicate", "INSIDE"),
                node.name,
                trigger_container.name,
            ]
            previous_criterion = copy.deepcopy(
                backend.evaluator.task.task_completion_criterion
            )
            effective_criterion = _append_goal_conjunct(
                previous_criterion,
                added_expression,
            )
            backend.evaluator.task.task_completion_criterion = effective_criterion
            goal_update.update(
                {
                    "changed": effective_criterion != previous_criterion,
                    "trigger_container_id": trigger_container_id,
                    "trigger_container_name": trigger_container.name,
                    "added_expression": added_expression,
                    "previous_criterion": previous_criterion,
                    "effective_criterion": copy.deepcopy(effective_criterion),
                }
            )
        return {
            "node_id": node.id,
            "node": _node_to_condition_object_spec(node),
            "new_location": {
                "relation": state.location_relation,
                "target": state.location_target,
                "held": state.held,
            },
            "goal_update": goal_update,
        }


class SymbolicHarness:
    def __init__(
        self,
        graph: ViewGraph,
        task: TaskRecord,
        mode: str = "replay",
        max_steps: int | None = None,
        teacher_policy: TeacherPolicyProtocol | None = None,
        failure_injection: FailureInjectionConfig | None = None,
        placement_edge_constraints: PlacementEdgeConstraints | None = None,
        collection_condition: dict[str, Any] | None = None,
        backend: WorldBackend | None = None,
    ) -> None:
        if mode not in {"replay", "teacher"}:
            raise ValueError(f"Unsupported collection mode {mode!r}; use replay or teacher")
        if mode == "teacher" and teacher_policy is None:
            raise ValueError("teacher mode requires a teacher_policy")
        self.graph = graph
        self.task = task
        self.mode = mode
        self.max_steps = max_steps
        self.placement_edge_constraints = placement_edge_constraints or PlacementEdgeConstraints()
        self.backend: WorldBackend = backend or SymbolicBackend(graph, task, self.placement_edge_constraints)
        if collection_condition is not None and not isinstance(self.backend, SymbolicBackend):
            raise ValueError("collection conditions currently require the symbolic backend")
        self.condition_runtime = CollectionConditionRuntime(collection_condition)
        self.teacher_policy = teacher_policy
        self.failure_injection = failure_injection or FailureInjectionConfig()
        self.failure_rng = random.Random(self.failure_injection.seed)
        self.injected_failure_count = 0
        self.failed_action_keys: set[tuple[str, ...]] = set()

    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        finally:
            self.backend.close()

    def _run(self) -> dict[str, Any]:
        backend = self.backend
        initial_task_completion_criterion = copy.deepcopy(
            self.task.task_completion_criterion
        )
        initial_observation = backend.observe()
        initial_snapshot = backend.snapshot()
        self.condition_runtime.validate_initial_state(backend, initial_observation)
        trajectory = []
        max_steps = self._max_steps()
        history: list[dict[str, Any]] = []
        condition_events: list[dict[str, Any]] = []

        for step_index in range(1, max_steps + 1):
            before_condition_observation = backend.observe()
            applied_conditions = self.condition_runtime.before_step(
                backend,
                step_number=step_index,
                history=history,
            )
            observation = backend.observe()
            condition_new_visible_nodes = _new_visible_nodes(
                before_condition_observation,
                observation,
            )
            condition_events.extend(copy.deepcopy(applied_conditions))
            pre_snapshot = backend.snapshot()
            decision = self._next_action(observation, history, step_index)
            requested_action = decision.action
            action, injection_record = self._maybe_inject_failure(requested_action)
            event = backend.step(action)
            post_observation = backend.observe()
            new_visible_nodes = _new_visible_nodes(observation, post_observation)
            post_snapshot = backend.snapshot()
            record = {
                "step": step_index,
                "mode": self.mode,
                "task_completion_criterion": copy.deepcopy(
                    self.task.task_completion_criterion
                ),
                "observation": observation,
                "post_observation": post_observation,
                "new_visible_nodes": new_visible_nodes,
                "action": action.to_json(),
                "event": event,
                "pre_state_hash": _state_hash(pre_snapshot),
                "post_state_hash": _state_hash(post_snapshot),
                "success_after_step": backend.success(),
            }
            if applied_conditions:
                record["conditions_applied"] = copy.deepcopy(applied_conditions)
                record["condition_new_visible_nodes"] = condition_new_visible_nodes
            if injection_record is not None:
                record["requested_action"] = requested_action.to_json()
                record["failure_injection"] = injection_record
            if decision.raw_response:
                record["teacher_response"] = decision.to_json()
            if self.mode == "replay":
                record["source_plan_step"] = action.raw
            trajectory.append(record)
            history.append(
                {
                    "step": step_index,
                    "action": action.to_json(),
                    "requested_action": requested_action.to_json(),
                    "event": event,
                    "new_visible_nodes": new_visible_nodes,
                    "conditions_applied": copy.deepcopy(applied_conditions),
                    "condition_new_visible_nodes": condition_new_visible_nodes,
                    "success_after_step": record["success_after_step"],
                }
            )
            if self.mode == "teacher" and action.base_name == "stop":
                break

        final_snapshot = backend.snapshot()
        metrics = backend.metrics(initial_observation)
        return {
            "episode_id": self.task.task_id,
            "scene_id": self.task.scene_id,
            "env_id": self.task.env_id,
            "backend": backend.name,
            "mode": self.mode,
            "task_type": self.task.task_type,
            "settings": list(self.task.settings),
            "task": self.task.task,
            "initial_task_completion_criterion": initial_task_completion_criterion,
            "task_completion_criterion": self.task.task_completion_criterion,
            "failure_injection": self.failure_injection.to_json(),
            "collection_condition": self.condition_runtime.condition,
            "condition_applied": bool(condition_events),
            "condition_events": condition_events,
            "placement_edge_constraints": self.placement_edge_constraints.to_json(),
            "initial_view_graph": _view_graph_to_json(self.graph),
            "initial_observation": initial_observation,
            "initial_state": initial_snapshot,
            "trajectory": trajectory,
            "final_state": final_snapshot,
            "metrics": metrics,
            "success": metrics["success"],
        }

    def _max_steps(self) -> int:
        if self.mode == "replay":
            if self.max_steps is None:
                return len(self.task.ground_truth_plan)
            return min(self.max_steps, len(self.task.ground_truth_plan))
        if self.max_steps is not None:
            return self.max_steps
        return 20

    def _next_action(
        self,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
        step_index: int,
    ) -> TeacherDecision:
        if self.mode == "replay":
            plan_step = self.task.ground_truth_plan[step_index - 1]
            action = parse_plan_action(plan_step)
            return TeacherDecision(action=action, raw_response="")
        assert self.teacher_policy is not None
        return self.teacher_policy.act(task=self.task, observation=observation, history=history)

    def _maybe_inject_failure(self, action: ParsedAction) -> tuple[ParsedAction, dict[str, Any] | None]:
        if not self._should_inject_failure(action):
            return action, None
        failure_key = self._failure_key(action)
        self.injected_failure_count += 1
        self.failed_action_keys.add(failure_key)
        failed_action = ParsedAction(
            name=f"failed_{action.base_name}",
            node_ids=list(action.node_ids),
            raw=action.raw,
            arguments=copy.deepcopy(action.arguments),
        )
        return failed_action, {
            "mode": self.failure_injection.mode,
            "original_action": action.to_json(),
            "failed_action": failed_action.to_json(),
            "failure_index": self.injected_failure_count,
            "deduplication_scope": self.failure_injection.deduplication_scope,
            "deduplication_key": list(failure_key),
        }

    def _should_inject_failure(self, action: ParsedAction) -> bool:
        config = self.failure_injection
        if not config.enabled:
            return False
        if action.name.startswith("failed_"):
            return False
        if not config.allows(action.base_name):
            return False
        if self.injected_failure_count >= config.max_failures_per_episode:
            return False
        failure_key = self._failure_key(action)
        if failure_key in self.failed_action_keys:
            return False
        if config.mode == "once":
            return self.injected_failure_count == 0
        if config.mode == "all":
            return True
        if config.mode == "probability":
            return self.failure_rng.random() < config.probability
        return False

    def _failure_key(self, action: ParsedAction) -> tuple[str, ...]:
        return self.failure_injection.deduplication_key(action.base_name, action.node_ids)


def collect_symbolic_trajectories(
    *,
    view_graph_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path,
    mode: str = "replay",
    max_episodes: int | None = None,
    max_steps: int | None = None,
    teacher_config: TeacherPolicyConfig | None = None,
    teacher_policy: TeacherPolicyProtocol | None = None,
    failure_injection: FailureInjectionConfig | None = None,
    placement_edge_constraints_path: str | Path | None = None,
    placement_edge_constraints: PlacementEdgeConstraints | None = None,
    collection_condition: dict[str, Any] | None = None,
    collection_condition_id: str | None = None,
    backend_factory: Callable[[ViewGraph, TaskRecord, PlacementEdgeConstraints], WorldBackend] | None = None,
) -> TrajectoryCollectionResult:
    graphs = {graph.scene_id: graph for graph in load_view_graphs_jsonl(view_graph_path)}
    tasks = _load_tasks_jsonl(tasks_path)
    if placement_edge_constraints_path is not None:
        placement_edge_constraints = load_placement_edge_constraints(placement_edge_constraints_path)
    if max_episodes is not None:
        tasks = tasks[:max_episodes]
    if mode == "teacher" and teacher_policy is None:
        teacher_policy = TeacherPolicy(teacher_config or TeacherPolicyConfig())
    failure_seed_source = None
    if failure_injection is not None and failure_injection.seed is not None:
        failure_seed_source = random.Random(failure_injection.seed)

    target = _timestamped_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for task in tasks:
            graph = graphs.get(task.scene_id)
            if graph is None:
                raise ValueError(f"No view graph found for task scene_id={task.scene_id!r}")
            episode_failure_injection = failure_injection
            if failure_injection is not None and failure_seed_source is not None:
                episode_failure_injection = replace(
                    failure_injection,
                    seed=failure_seed_source.randrange(0, 2**63),
                )
            backend = (
                backend_factory(graph, task, placement_edge_constraints or PlacementEdgeConstraints())
                if backend_factory is not None
                else None
            )
            episode_collection_condition = collection_condition
            if collection_condition_id is not None:
                episode_collection_condition = load_task_collection_condition(
                    task,
                    collection_condition_id,
                )
            episode = SymbolicHarness(
                graph,
                task,
                mode=mode,
                max_steps=max_steps,
                teacher_policy=teacher_policy,
                failure_injection=episode_failure_injection,
                placement_edge_constraints=placement_edge_constraints,
                collection_condition=episode_collection_condition,
                backend=backend,
            ).run()
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            count += 1
    return TrajectoryCollectionResult(count=count, output_path=target)


def _timestamped_output_path(path: str | Path) -> Path:
    target = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")
    index = 2
    while candidate.exists():
        candidate = target.with_name(f"{target.stem}_{timestamp}_{index}{target.suffix}")
        index += 1
    return candidate


def iter_trajectory_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
