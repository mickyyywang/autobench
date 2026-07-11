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
from typing import Any, Iterable, Protocol

from openai import APITimeoutError, OpenAI

from .action_model import ACTION_SCHEMAS, action_model_trace
from .graph_io import load_view_graphs_jsonl
from .goal import evaluate_goal_expression, extract_goal_predicates, normalize_goal_expression
from .models import Node, TaskRecord, ViewGraph, normalize_relation
from .placement_constraints import PlacementEdgeConstraints, load_placement_edge_constraints


LOCATION_RELATIONS = {"ON", "INSIDE", "IN"}
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

    def to_json(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "reason": self.reason,
            "parse_error": self.parse_error,
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
    "assemble",
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

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "actions": list(self.actions),
            "probability": self.probability,
            "max_failures_per_episode": self.max_failures_per_episode,
            "seed": self.seed,
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
    "grab": "Hold a visible reachable grabbable movable object.",
    "pick": "Alias of grab.",
    "attach": "Attach two visible or held compatible part nodes.",
    "assemble": "Assemble a visible object when all parts are reachable.",
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
    "assemble": {"required_free_hands": 2, "held_object_required": False, "result": "two_hand_coordination"},
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
        if node.get("openable") and not node.get("open") and not held_ids:
            extra = _reveal_action_extra(node) if node.get("container") else {}
            add("open", [node_id], **extra)
        if node.get("openable") and node.get("open") and not held_ids:
            add("close", [node_id])
        if node.get("pressable") and node.get("reachable"):
            add("press", [node_id])
        for reveal_action in _reveal_valid_actions(node):
            add(reveal_action, [node_id], **_reveal_action_extra(node))
        if node.get("grabbable") and node.get("movable") and node.get("reachable"):
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
            if target.get("container") and (not target.get("openable") or target.get("open")) and not target.get("is_full"):
                add("putin", [held_id, target_id])
            if target.get("surface"):
                add("puton", [held_id, target_id])

    if not actions:
        add("observe")
    actions[:] = _prioritize_open_before_grab(actions)
    add("stop")
    return actions


def _reveal_valid_actions(node: dict[str, Any]) -> list[str]:
    if int(node.get("occludes_hidden_count") or 0) <= 0:
        return []
    if node.get("container"):
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


class TeacherPolicy:
    def __init__(self, config: TeacherPolicyConfig) -> None:
        if config.provider not in {"openai", "qwen", "compatible"}:
            raise ValueError(f"Unknown teacher provider {config.provider!r}; use openai, qwen, or compatible")
        self.config = config
        self.model = config.model or os.environ.get("AUTO_EMBODIED_TEACHER_MODEL") or _default_teacher_model(
            config.provider
        )
        api_key_env = _teacher_api_key_env(config.provider, config.api_key_env)
        key = config.api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key. Set {api_key_env} or pass api_key.")
        self.client = OpenAI(
            api_key=key,
            base_url=_teacher_api_base_url(config.provider, config.api_base_url),
            timeout=config.timeout_seconds,
        )

    def act(
        self,
        *,
        task: TaskRecord,
        observation: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> TeacherDecision:
        messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": _teacher_user_prompt(task, observation, history)},
        ]
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.config.provider == "qwen":
            create_kwargs["extra_body"] = {"enable_thinking": False}
        try:
            completion = self.client.chat.completions.create(**create_kwargs)
        except (TimeoutError, APITimeoutError) as exc:
            raise RuntimeError(
                f"{self.config.provider} teacher API request timed out after "
                f"{self.config.timeout_seconds} seconds."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"{self.config.provider} teacher API request failed: {exc}") from exc
        if not completion.choices:
            raise RuntimeError(f"{self.config.provider} teacher API returned no choices")
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError(f"{self.config.provider} teacher API returned empty content")
        return parse_teacher_decision(content)


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
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return TeacherDecision(
            action=ParsedAction(name="invalid_teacher_action", raw=raw),
            raw_response=raw,
            parse_error=str(exc),
        )
    if not isinstance(parsed, dict):
        return TeacherDecision(
            action=ParsedAction(name="invalid_teacher_action", raw=raw),
            raw_response=raw,
            parsed_response=None,
            parse_error="teacher response must be a JSON object",
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
        )
    return TeacherDecision(
        action=action,
        raw_response=raw,
        parsed_response=parsed,
        reason=str(parsed.get("reason", "")),
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


def _teacher_api_key_env(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "qwen":
        return "DASHSCOPE_API_KEY"
    return "OPENAI_API_KEY"


def _teacher_api_base_url(provider: str, override: str | None) -> str | None:
    if override:
        return override
    if provider == "qwen":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if provider == "compatible":
        return os.environ.get("OPENAI_BASE_URL")
    return None


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
        self.active_occlusion_edges: set[tuple[str, str, str]] = {
            (edge.source, edge.target, edge.relation)
            for edge in graph.edges
            if edge.relation in OCCLUSION_RELATIONS
        }
        self._name_to_id = self._build_name_lookup(graph)
        self.memory_hidden: dict[str, str] = self._memory_hidden_targets(task)
        self._initialize_states()

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
                    "states": list(state.node.states),
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
            if target_id == container_id or target_id in exclude:
                continue
            target = self.states.get(target_id)
            if target is None or target.held:
                continue
            if target.location_relation is None:
                items.add(target_id)
        return len(items)

    def _container_is_full(self, container_id: str, exclude: set[str] | None = None) -> bool:
        state = self.states.get(container_id)
        if state is None or not state.node.is_container or state.node.max_items is None:
            return False
        return self._container_item_count(container_id, exclude=exclude) >= state.node.max_items

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
        if node_id not in self.states:
            return False
        state = self.states[node_id]
        if state.node.is_room or state.held:
            return True
        if self._is_decomposed_parent(node_id) and not state.assembled:
            return False
        assembled_parent_id = self._assembled_part_parent(node_id)
        if assembled_parent_id is not None and not self.is_visible(assembled_parent_id):
            return False
        memory_anchor = self.memory_hidden.get(node_id)
        if memory_anchor is not None and not self._memory_target_revealed(memory_anchor):
            return False
        if self._inside_closed_container(node_id):
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
            if edge[1] != node_id
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
            if relation in OCCLUSION_RELATIONS and not self._occlusion_source_active(source):
                continue
            active.append(source.node.id)
        return active

    def _occlusion_source_active(self, source: NodeState) -> bool:
        if source.node.is_container:
            return not (source.node.is_openable and source.open)
        return not source.moved_aside

    def _inspect_reveals_occlusion(self, state: NodeState) -> bool:
        return state.node.has_property(*INSPECT_REVEAL_PROPERTIES)

    def _hidden_targets_blocked_by(self, source_id: str) -> list[str]:
        source = self.states.get(source_id)
        if source is None:
            return []
        if not self._occlusion_source_active(source):
            return []
        hidden_targets = []
        for edge_source, edge_target, relation in sorted(self.active_occlusion_edges):
            if edge_source != source_id or relation not in OCCLUSION_RELATIONS:
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

    def _common_part_location(self, part_ids: list[str]) -> tuple[str | None, str | None]:
        locations = {
            (self.states[part_id].location_relation, self.states[part_id].location_target)
            for part_id in part_ids
            if self.states[part_id].location_relation is not None
        }
        if len(locations) == 1:
            return next(iter(locations))
        return None, None

    def _set_parent_assembled(self, parent_id: str, part_ids: list[str]) -> None:
        parent_state = self.states.get(parent_id)
        if parent_state is None:
            return
        parent_state.assembled = True
        relation, target = self._common_part_location(part_ids)
        if relation is not None:
            parent_state.location_relation = relation
            parent_state.location_target = target
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
            if self._container_is_full(target.node.id, exclude={obj.node.id}):
                return self._failure(
                    action,
                    "container_full",
                    f"{target.node.id} has reached max_items={target.node.max_items}",
                    max_items=target.node.max_items,
                    item_count=self._container_item_count(target.node.id, exclude={obj.node.id}),
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
        if moved_aside and target.node.is_container and self._hidden_targets_blocked_by(target.node.id):
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
        first_failure_index = None
        failed_action = None
        failed_nodes: list[str] = []
        for index, item in enumerate(self.world.events):
            if item["event"].get("status") != "failure":
                continue
            first_failure_index = index
            failed_action = item["event"].get("failed_action") or item["action"].get("base_name")
            failed_nodes = list(item["action"].get("node_ids", []))
            break
        recovered = False
        retried = False
        if first_failure_index is not None:
            for item in self.world.events[first_failure_index + 1 :]:
                if item["event"].get("status") == "success" and item["action"].get("base_name") == "recover":
                    recovered = True
                if (
                    item["event"].get("status") == "success"
                    and failed_action
                    and item["action"].get("base_name") == failed_action
                    and item["action"].get("node_ids") == failed_nodes
                ):
                    retried = True
                    recovered = True
        return {
            "failure_observed": first_failure_index is not None,
            "recovered_after_failure": recovered,
            "retried_failed_action": retried,
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
        self.backend: WorldBackend = SymbolicBackend(graph, task, self.placement_edge_constraints)
        self.teacher_policy = teacher_policy
        self.failure_injection = failure_injection or FailureInjectionConfig()
        self.failure_rng = random.Random(self.failure_injection.seed)
        self.injected_failure_count = 0
        self.failed_action_signatures: set[tuple[str, tuple[str, ...]]] = set()

    def run(self) -> dict[str, Any]:
        backend = self.backend
        initial_observation = backend.observe()
        initial_snapshot = backend.snapshot()
        trajectory = []
        max_steps = self._max_steps()
        history: list[dict[str, Any]] = []

        for step_index in range(1, max_steps + 1):
            observation = backend.observe()
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
                "observation": observation,
                "post_observation": post_observation,
                "new_visible_nodes": new_visible_nodes,
                "action": action.to_json(),
                "event": event,
                "pre_state_hash": _state_hash(pre_snapshot),
                "post_state_hash": _state_hash(post_snapshot),
                "success_after_step": backend.success(),
            }
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
            "task_completion_criterion": self.task.task_completion_criterion,
            "failure_injection": self.failure_injection.to_json(),
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
        signature = self._action_signature(action)
        self.injected_failure_count += 1
        self.failed_action_signatures.add(signature)
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
        signature = self._action_signature(action)
        if signature in self.failed_action_signatures:
            return False
        if config.mode == "once":
            return self.injected_failure_count == 0
        if config.mode == "all":
            return True
        if config.mode == "probability":
            return self.failure_rng.random() < config.probability
        return False

    @staticmethod
    def _action_signature(action: ParsedAction) -> tuple[str, tuple[str, ...]]:
        return action.base_name, tuple(action.node_ids)


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
            episode = SymbolicHarness(
                graph,
                task,
                mode=mode,
                max_steps=max_steps,
                teacher_policy=teacher_policy,
                failure_injection=episode_failure_injection,
                placement_edge_constraints=placement_edge_constraints,
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
