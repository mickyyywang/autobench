from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

from .brain import BrainHarness, BrainPolicy, BrainPolicyConfig, BrainRequest
from .episode_sources import episode_from_manifest_source
from .harness import (
    OCCLUSION_RELATIONS,
    FailureInjectionConfig,
    NodeState,
    ParsedAction,
    SymbolicBackend,
    _append_goal_conjunct,
    _new_visible_nodes,
    _node_to_condition_object_spec,
    _project_goal_expression,
    _replace_goal_subject,
    _state_hash,
    _timestamped_output_path,
    _valid_teacher_actions,
)
from .goal import evaluate_goal_expression, normalize_goal_expression
from .models import Edge, Node, normalize_relation
from .real_observation_eval import (
    CAPABILITY_METRIC_VERSION,
    RealObservationAdapter,
    ReplayStepContext,
    _aggregate_capability_scores,
    _canonical_action_key,
    _exploration_obligations,
    _goal_atoms,
    _parsed_action,
    _planning_completion_cost,
    _predicted_recovery,
    _relaxed_completion_cost,
    _replay_inputs,
    _resolved_api_style,
    _safe_backend_success,
    _score_recovery,
    _soft_optimal_action_score,
    _structured_action,
    _visible_graph_observation,
)


CLOSED_LOOP_EVALUATION_TYPE = "closed_loop_visible_graph"
CLOSED_LOOP_METRIC_VERSION = 4
CLOSED_LOOP_MODE = "visible_graph_only"

# Visible-graph state is refreshed automatically before every model action.  Actions
# that only refresh the observation or mutate hidden focus/inspection bookkeeping
# therefore cannot reveal a node, change task state, or reduce planning cost in this
# evaluator. Keep one explicit `observe` action as a deliberate wait/refresh option,
# while removing redundant focus-only variants that encouraged long no-progress
# loops.
CLOSED_LOOP_EXECUTABLE_ACTIONS = frozenset(
    {
        "observe",
        "open",
        "close",
        "press",
        "grab",
        "attach",
        "puton",
        "putin",
        "move_aside",
        "stop",
    }
)
RUNTIME_OCCLUSION_SELECTIONS = frozenset(
    {
        # Legacy manifests remain executable with the corrected selection policy.
        "runtime_first_eligible",
        "runtime_prefer_open_then_first_eligible",
    }
)
RUNTIME_STATE_REGRESSION_SELECTION = "runtime_first_eligible_state_regression"
RUNTIME_COMPLETED_ROLLBACK_SELECTION = "runtime_first_satisfied_goal_placement"
RUNTIME_WRONG_RELOCATION_SELECTION = "runtime_first_satisfied_goal_wrong_destination"

GRAPH_DISTURBANCE_OPERATIONS = {
    "add_object",
    "set_state",
    "relocate",
    "set_capacity",
    "add_occlusion",
    "relocate_and_add_occlusion",
    "remove_occlusion",
}
GRAPH_DISTURBANCE_STATE_FIELDS = {
    "open",
    "held",
    "assembled",
    "pressed",
    "moved_aside",
    "cleared",
    "inspected",
    "attached_to",
}


@dataclass(frozen=True)
class ViewGraphRolloutEvalConfig:
    provider: str = "qwen"
    model: str | None = None
    model_name: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    api_style: str = "auto"
    json_response_format: bool = True
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: int = 120
    include_valid_actions: bool = True
    max_api_attempts: int = 1
    retry_backoff_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    max_steps: int = 100
    history_window: int = 8
    max_consecutive_model_errors: int = 3
    failure_injection: str = "none"
    failure_actions: tuple[str, ...] = ("all",)
    failure_probability: float = 0.0
    max_failures_per_episode: int = 1
    failure_seed: int | None = None
    failure_deduplication_scope: str = "signature"
    graph_disturbance_file: str | None = None
    soft_optimal_beta: float = 1.0
    fail_fast: bool = False

    def __post_init__(self) -> None:
        providers = {"openai", "qwen", "compatible", "mr_openai", "mr_anthropic", "mr_google"}
        if self.provider not in providers:
            raise ValueError(f"provider must be one of {sorted(providers)}")
        if self.api_style not in {
            "auto",
            "chat_completions",
            "responses",
            "anthropic_messages",
            "gemini_generate_content",
        }:
            raise ValueError("unsupported api_style")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.history_window <= 0:
            raise ValueError("history_window must be positive")
        if self.max_consecutive_model_errors <= 0:
            raise ValueError("max_consecutive_model_errors must be positive")
        raw_failure_actions = (
            (self.failure_actions,)
            if isinstance(self.failure_actions, str)
            else self.failure_actions
        )
        normalized_failure_actions = tuple(
            str(action).strip().lower()
            for action in raw_failure_actions
            if str(action).strip()
        ) or ("all",)
        object.__setattr__(self, "failure_actions", normalized_failure_actions)
        FailureInjectionConfig(
            mode=self.failure_injection,
            actions=normalized_failure_actions,
            probability=self.failure_probability,
            max_failures_per_episode=self.max_failures_per_episode,
            seed=self.failure_seed,
            deduplication_scope=self.failure_deduplication_scope,
        )
        if self.max_api_attempts <= 0:
            raise ValueError("max_api_attempts must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.soft_optimal_beta <= 0:
            raise ValueError("soft_optimal_beta must be positive")


def load_graph_disturbances(path: str | Path | None) -> tuple[dict[str, Any], ...]:
    """Load step-level external graph changes from JSON or JSONL."""
    if path is None or not str(path).strip():
        return ()
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"graph disturbance file does not exist: {source}")
    if source.suffix.lower() == ".jsonl":
        raw_items: list[Any] = []
        for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
    else:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc}") from exc
        if isinstance(payload, dict) and "disturbances" in payload:
            raw_items = payload["disturbances"]
        elif isinstance(payload, list):
            raw_items = payload
        else:
            raw_items = [payload]
    if not isinstance(raw_items, list):
        raise ValueError(f"{source}: disturbances must be a JSON array")
    return tuple(
        _normalize_graph_disturbance(item, source=source, index=index)
        for index, item in enumerate(raw_items, start=1)
    )


def _normalize_graph_disturbance(
    item: Any,
    *,
    source: Path,
    index: int,
    require_step: bool = True,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{source}: disturbance {index} must be a JSON object")
    normalized = copy.deepcopy(item)
    step = normalized.get("step")
    if require_step and (isinstance(step, bool) or not isinstance(step, int) or step <= 0):
        raise ValueError(f"{source}: disturbance {index} needs a positive integer step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step <= 0):
        raise ValueError(f"{source}: disturbance {index} has an invalid step")
    operation = str(normalized.get("operation") or normalized.get("op") or "").strip().lower()
    if operation not in GRAPH_DISTURBANCE_OPERATIONS:
        raise ValueError(
            f"{source}: disturbance {index} operation must be one of "
            f"{sorted(GRAPH_DISTURBANCE_OPERATIONS)}"
        )
    normalized["operation"] = operation
    normalized.pop("op", None)
    if operation in {"set_state", "relocate", "set_capacity"} and not normalized.get("node_id"):
        raise ValueError(f"{source}: disturbance {index} {operation} needs node_id")
    if operation == "set_state":
        values = normalized.get("values")
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{source}: disturbance {index} set_state needs non-empty values")
        unsupported = set(values) - GRAPH_DISTURBANCE_STATE_FIELDS
        if unsupported:
            raise ValueError(
                f"{source}: disturbance {index} unsupported state fields: {sorted(unsupported)}"
            )
        for key, value in values.items():
            if key != "attached_to" and not isinstance(value, bool):
                raise ValueError(f"{source}: disturbance {index} values.{key} must be boolean")
            if key == "attached_to" and value is not None and not str(value).strip():
                raise ValueError(f"{source}: disturbance {index} values.attached_to is invalid")
    if operation == "relocate":
        relation = normalized.get("relation")
        target = normalized.get("target")
        if (relation is None) != (target is None):
            raise ValueError(
                f"{source}: disturbance {index} relocate relation and target must both be set or null"
            )
    if operation == "add_object":
        object_spec = normalized.get("object")
        if not isinstance(object_spec, dict):
            raise ValueError(
                f"{source}: disturbance {index} add_object needs an object definition"
            )
        if not object_spec.get("id"):
            raise ValueError(
                f"{source}: disturbance {index} add_object object needs a unique id"
            )
        if not object_spec.get("name") and not object_spec.get("copy_from"):
            raise ValueError(
                f"{source}: disturbance {index} add_object object needs name or copy_from"
            )
        relation = normalize_relation(str(normalized.get("relation") or ""))
        relation = "INSIDE" if relation == "IN" else relation
        if relation not in {"ON", "INSIDE", "BENEATH"} or not normalized.get("target"):
            raise ValueError(
                f"{source}: disturbance {index} add_object needs relation "
                "ON/INSIDE/BENEATH and target"
            )
        normalized["relation"] = relation
        component_ids = {str(object_spec["id"])}
        raw_components = normalized.get("component_objects", [])
        if not isinstance(raw_components, list):
            raise ValueError(
                f"{source}: disturbance {index} add_object component_objects must be a list"
            )
        components: list[dict[str, Any]] = []
        for component_index, raw_component in enumerate(raw_components, start=1):
            if not isinstance(raw_component, dict):
                raise ValueError(
                    f"{source}: disturbance {index} component {component_index} "
                    "must be an object"
                )
            component = copy.deepcopy(raw_component)
            component_spec = component.get("object")
            if not isinstance(component_spec, dict) or not component_spec.get("id"):
                raise ValueError(
                    f"{source}: disturbance {index} component {component_index} "
                    "needs object.id"
                )
            if not component_spec.get("name") and not component_spec.get("copy_from"):
                raise ValueError(
                    f"{source}: disturbance {index} component {component_index} "
                    "needs object.name or object.copy_from"
                )
            component_id = str(component_spec["id"])
            if component_id in component_ids:
                raise ValueError(
                    f"{source}: disturbance {index} reuses add_object id {component_id!r}"
                )
            component_ids.add(component_id)
            component_relation = normalize_relation(
                str(component.get("relation") or "")
            )
            component_relation = (
                "INSIDE" if component_relation == "IN" else component_relation
            )
            if (
                component_relation not in {"ON", "INSIDE", "BENEATH"}
                or not component.get("target")
            ):
                raise ValueError(
                    f"{source}: disturbance {index} component {component_index} "
                    "needs relation ON/INSIDE/BENEATH and target"
                )
            component["relation"] = component_relation
            components.append(component)
        normalized["component_objects"] = components
        success_policy = normalized.get("success_policy")
        if not isinstance(success_policy, dict):
            raise ValueError(
                f"{source}: disturbance {index} add_object needs success_policy"
            )
        policy_type = str(success_policy.get("type") or "").strip().lower()
        if policy_type not in {
            "inherit_from",
            "existing_task_goal",
            "trigger_container_goal",
        }:
            raise ValueError(
                f"{source}: disturbance {index} add_object success_policy.type must be "
                "inherit_from, existing_task_goal, or trigger_container_goal"
            )
        if policy_type == "inherit_from":
            source_ref = success_policy.get("source_node_id") or object_spec.get("copy_from")
            if not source_ref:
                raise ValueError(
                    f"{source}: disturbance {index} inherit_from needs source_node_id"
                )
            success_policy["source_node_id"] = source_ref
            placement_alternatives = success_policy.get("placement_alternatives", [])
            if not isinstance(placement_alternatives, list) or any(
                not str(value).strip() for value in placement_alternatives
            ):
                raise ValueError(
                    f"{source}: disturbance {index} inherit_from "
                    "placement_alternatives must be an array of node ids"
                )
            success_policy["placement_alternatives"] = [
                str(value) for value in placement_alternatives
            ]
        if policy_type == "trigger_container_goal":
            predicate = normalize_relation(
                str(success_policy.get("predicate") or "INSIDE")
            )
            if predicate == "IN":
                predicate = "INSIDE"
            if predicate != "INSIDE":
                raise ValueError(
                    f"{source}: disturbance {index} trigger_container_goal only "
                    "supports predicate INSIDE"
                )
            success_policy["predicate"] = predicate
        success_policy["type"] = policy_type
        normalized["success_policy"] = success_policy
    if operation == "set_capacity":
        max_items = normalized.get("max_items")
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0:
            raise ValueError(f"{source}: disturbance {index} max_items must be a non-negative integer")
    if (
        operation == "add_occlusion"
        and normalized.get("selection") in RUNTIME_OCCLUSION_SELECTIONS
    ):
        raw_candidates = normalized.get("candidate_pairs")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(
                f"{source}: disturbance {index} runtime add_occlusion needs candidate_pairs"
            )
        candidates: list[dict[str, Any]] = []
        for candidate_index, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                raise ValueError(
                    f"{source}: disturbance {index} candidate {candidate_index} "
                    "must be an object"
                )
            candidate = copy.deepcopy(raw_candidate)
            if not candidate.get("source") or not candidate.get("target"):
                raise ValueError(
                    f"{source}: disturbance {index} candidate {candidate_index} "
                    "needs source and target"
                )
            if str(candidate["source"]) == str(candidate["target"]):
                raise ValueError(
                    f"{source}: disturbance {index} candidate {candidate_index} "
                    "cannot occlude itself"
                )
            relation = normalize_relation(str(candidate.get("relation") or "OCCLUDES"))
            if relation not in OCCLUSION_RELATIONS:
                raise ValueError(
                    f"{source}: disturbance {index} candidate {candidate_index} "
                    f"relation must be one of {sorted(OCCLUSION_RELATIONS)}"
                )
            candidate["relation"] = relation
            raw_actions = candidate.get("supported_resolution_actions") or ("open", "move_aside")
            if isinstance(raw_actions, str):
                raw_actions = [raw_actions]
            actions = [str(action).strip().lower() for action in raw_actions]
            if not actions or any(action not in {"open", "move_aside"} for action in actions):
                raise ValueError(
                    f"{source}: disturbance {index} candidate {candidate_index} has "
                    "invalid supported_resolution_actions"
                )
            candidate["supported_resolution_actions"] = list(dict.fromkeys(actions))
            candidates.append(candidate)
        normalized["candidate_pairs"] = candidates
    elif operation == "relocate_and_add_occlusion":
        required = ("source", "target", "staging_relation", "staging_target", "previous_location")
        missing = [field for field in required if normalized.get(field) is None]
        if missing:
            raise ValueError(
                f"{source}: disturbance {index} relocate_and_add_occlusion needs "
                f"{', '.join(missing)}"
            )
        staging_relation = normalize_relation(str(normalized["staging_relation"]))
        if staging_relation != "ON":
            raise ValueError(
                f"{source}: disturbance {index} relocate_and_add_occlusion "
                "staging_relation must be ON"
            )
        normalized["staging_relation"] = staging_relation
        if not isinstance(normalized["previous_location"], dict):
            raise ValueError(
                f"{source}: disturbance {index} relocate_and_add_occlusion "
                "previous_location must be an object"
            )
    elif operation in {"add_occlusion", "remove_occlusion"}:
        if not normalized.get("source") or not normalized.get("target"):
            raise ValueError(f"{source}: disturbance {index} {operation} needs source and target")
    return normalized


def load_intervention_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"intervention manifest does not exist: {source}")
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{source}: intervention manifest must be a JSON object")
    if manifest.get("manifest_type") != "closed_loop_intervention_suite":
        raise ValueError(f"{source}: unsupported manifest_type")
    source_config = manifest.get("source")
    if not isinstance(source_config, dict):
        raise ValueError(f"{source}: manifest source must be an object")
    source_type = str(
        source_config.get("source_type")
        or ("aligned_episode" if source_config.get("aligned_episode") else "")
    )
    if source_type == "aligned_episode":
        if not source_config.get("aligned_episode"):
            raise ValueError(f"{source}: manifest source.aligned_episode is required")
        aligned_episode = Path(str(source_config["aligned_episode"]))
        if not aligned_episode.is_file():
            raise ValueError(f"{source}: aligned episode does not exist: {aligned_episode}")
        _validate_source_hash(
            source=source,
            data_path=aligned_episode,
            expected=source_config.get("sha256"),
            field="aligned episode",
        )
        source_config["source_type"] = "aligned_episode"
        source_config["aligned_episode"] = str(aligned_episode.resolve())
    elif source_type == "view_graph_and_task":
        for field, hash_field, label in (
            ("view_graph", "view_graph_sha256", "view graph"),
            ("tasks", "tasks_sha256", "tasks"),
        ):
            data_path = Path(str(source_config.get(field) or ""))
            if not data_path.is_file():
                raise ValueError(f"{source}: {label} does not exist: {data_path}")
            _validate_source_hash(
                source=source,
                data_path=data_path,
                expected=source_config.get(hash_field),
                field=label,
            )
            source_config[field] = str(data_path.resolve())
    else:
        raise ValueError(
            f"{source}: source_type must be aligned_episode or view_graph_and_task"
        )
    raw_conditions = manifest.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError(f"{source}: manifest conditions must be a non-empty array")
    condition_ids: set[str] = set()
    conditions: list[dict[str, Any]] = []
    for index, raw_condition in enumerate(raw_conditions, start=1):
        if not isinstance(raw_condition, dict):
            raise ValueError(f"{source}: condition {index} must be a JSON object")
        condition = copy.deepcopy(raw_condition)
        condition_id = str(condition.get("condition_id") or "").strip()
        intervention_type = str(condition.get("intervention_type") or "").strip().lower()
        if not condition_id or not intervention_type:
            raise ValueError(f"{source}: condition {index} needs condition_id and intervention_type")
        if condition_id in condition_ids:
            raise ValueError(f"{source}: duplicate condition_id {condition_id!r}")
        condition_ids.add(condition_id)
        condition["condition_id"] = condition_id
        condition["intervention_type"] = intervention_type
        failure = condition.get("failure_injection") or {"mode": "none"}
        if not isinstance(failure, dict):
            raise ValueError(f"{source}: condition {condition_id} failure_injection must be an object")
        FailureInjectionConfig(
            mode=str(failure.get("mode") or "none"),
            actions=tuple(failure.get("actions") or ("all",)),
            probability=float(failure.get("probability", 0.0)),
            max_failures_per_episode=int(failure.get("max_failures_per_episode", 1)),
            seed=int(failure["seed"]) if failure.get("seed") is not None else None,
            deduplication_scope=str(failure.get("deduplication_scope") or "signature"),
        )
        disturbance = condition.get("graph_disturbance")
        if disturbance is not None:
            condition["graph_disturbance"] = _normalize_graph_disturbance(
                disturbance,
                source=source,
                index=index,
                require_step=False,
            )
            if not isinstance(condition.get("trigger"), dict):
                raise ValueError(f"{source}: condition {condition_id} needs a trigger")
            trigger_type = str(condition["trigger"].get("type") or "").strip().lower()
            if condition["graph_disturbance"].get("operation") == "add_object":
                policy_type = condition["graph_disturbance"]["success_policy"]["type"]
                expected_triggers = (
                    {"on_object_goal_satisfied", "first_goal_progress_opportunity"}
                    if policy_type == "inherit_from"
                    else {
                        "on_any_container_max_items_reached"
                        if policy_type == "trigger_container_goal"
                        else "on_container_max_items_reached"
                    }
                )
                if trigger_type not in expected_triggers:
                    raise ValueError(
                        f"{source}: condition {condition_id} add_object policy "
                        f"{policy_type} requires trigger type in "
                        f"{sorted(expected_triggers)}"
                    )
            if trigger_type == "on_any_container_max_items_reached":
                node_ids = condition["trigger"].get("node_ids")
                if not isinstance(node_ids, list) or not node_ids:
                    raise ValueError(
                        f"{source}: condition {condition_id} trigger needs a "
                        "non-empty node_ids array"
                    )
                condition["trigger"]["node_ids"] = [
                    str(node_id) for node_id in node_ids
                ]
            required_predicates = condition["trigger"].get(
                "required_predicates", []
            )
            if not isinstance(required_predicates, list):
                raise ValueError(
                    f"{source}: condition {condition_id} trigger.required_predicates "
                    "must be a list"
                )
            if any(not isinstance(item, (dict, list)) for item in required_predicates):
                raise ValueError(
                    f"{source}: condition {condition_id} trigger.required_predicates "
                    "entries must be goal expressions"
                )
        cleanup = condition.get("cleanup")
        if cleanup is not None:
            condition["cleanup"] = _normalize_graph_disturbance(
                cleanup,
                source=source,
                index=index,
                require_step=False,
            )
            if not isinstance(cleanup.get("trigger"), dict):
                raise ValueError(f"{source}: condition {condition_id} cleanup needs a trigger")
        conditions.append(condition)
    normalized = copy.deepcopy(manifest)
    normalized["_manifest_path"] = str(source.resolve())
    normalized["source"] = copy.deepcopy(source_config)
    normalized["conditions"] = conditions
    return normalized


def _validate_source_hash(
    *,
    source: Path,
    data_path: Path,
    expected: Any,
    field: str,
) -> None:
    if not expected:
        return
    actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if str(expected) != actual:
        raise ValueError(
            f"{source}: {field} sha256 mismatch: expected {expected}, got {actual}"
        )


class ClosedLoopVisibleGraphAdapter(RealObservationAdapter):
    def __init__(self, *, history_window: int = 8) -> None:
        self.history_window = history_window

    def build_request(self, **kwargs: Any) -> BrainRequest:
        request = super().build_request(**kwargs)
        content = request.messages[1]["content"]
        payload = json.loads(content[0]["text"])
        action_catalog = payload.get("action_catalog")
        if isinstance(action_catalog, dict):
            for action_name in list(action_catalog):
                if action_name not in CLOSED_LOOP_EXECUTABLE_ACTIONS:
                    action_catalog.pop(action_name, None)
        payload["recent_history"] = copy.deepcopy(list(kwargs.get("history") or [])[-self.history_window :])
        constraints = payload.get("action_constraints")
        if isinstance(constraints, list):
            payload["action_constraints"] = [
                item for item in constraints if item != "Use real images whenever images are present."
            ]
        content[0]["text"] = json.dumps(payload, ensure_ascii=False, indent=2)
        request.summary.update(
            {
                "adapter": "closed_loop_visible_graph",
                "evaluation_type": CLOSED_LOOP_EVALUATION_TYPE,
                "history_source": "inference",
                "history_window": self.history_window,
            }
        )
        return request


class _ManifestInterventionRuntime:
    def __init__(
        self,
        condition: dict[str, Any] | None,
        *,
        initial_completion_cost: float | None = None,
    ) -> None:
        self.condition = copy.deepcopy(condition) if condition is not None else None
        self.initial_completion_cost = initial_completion_cost
        self.primary_applied = False
        self.cleanup_applied = False
        self.cleanup_pending = False
        self.model_actions_since_primary = 0
        self.added_object_goal_expression: Any | None = None
        self.added_object_goal_activation_step: int | None = None

    @property
    def condition_id(self) -> str | None:
        if self.condition is None:
            return None
        return str(self.condition.get("condition_id") or "") or None

    @property
    def intervention_type(self) -> str | None:
        if self.condition is None:
            return None
        return str(self.condition.get("intervention_type") or "") or None

    def before_step(
        self,
        backend: SymbolicBackend,
        *,
        step_number: int,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.condition is None:
            return []
        if self.initial_completion_cost is None:
            self.initial_completion_cost = _relaxed_completion_cost(backend)
        reports: list[dict[str, Any]] = []
        cleanup = self.condition.get("cleanup")
        if (
            self.cleanup_pending
            and not self.cleanup_applied
            and isinstance(cleanup, dict)
            and _manifest_cleanup_due(cleanup, self.model_actions_since_primary)
        ):
            report = _graph_disturbance_report(
                backend,
                cleanup,
                step_number=step_number,
                phase="cleanup",
                counts_as_intervention=False,
            )
            report["condition_id"] = self.condition_id
            report["intervention_type"] = self.intervention_type
            reports.append(report)
            self.cleanup_applied = True
            self.cleanup_pending = False

        disturbance = self.condition.get("graph_disturbance")
        trigger = self.condition.get("trigger")
        trigger_matches = bool(
            not self.primary_applied
            and isinstance(disturbance, dict)
            and isinstance(trigger, dict)
            and _manifest_trigger_matches(
                trigger,
                backend=backend,
                step_number=step_number,
                history=history,
                initial_completion_cost=self.initial_completion_cost,
            )
        )
        if (
            not self.primary_applied
            and isinstance(disturbance, dict)
            and isinstance(trigger, dict)
            and trigger_matches
        ):
            disturbance_for_apply = copy.deepcopy(disturbance)
            success_policy = disturbance_for_apply.get("success_policy")
            if (
                disturbance_for_apply.get("operation") == "add_object"
                and isinstance(success_policy, dict)
                and success_policy.get("type") == "trigger_container_goal"
            ):
                trigger_container_id = _matching_manifest_trigger_container(
                    trigger,
                    backend=backend,
                )
                if trigger_container_id is None:
                    return reports
                success_policy["trigger_container_id"] = trigger_container_id
            materialized = _materialize_manifest_disturbance(
                backend,
                disturbance_for_apply,
            )
            if materialized is None:
                return reports
            applied_disturbance, runtime_selection = materialized
            report = _graph_disturbance_report(
                backend,
                applied_disturbance,
                step_number=step_number,
                phase="intervention",
                counts_as_intervention=True,
            )
            if runtime_selection is not None:
                report["runtime_selection"] = runtime_selection
            report["condition_id"] = self.condition_id
            report["intervention_type"] = self.intervention_type
            reports.append(report)
            if report.get("operation") == "add_object":
                details = report.get("details")
                goal_update = (
                    details.get("goal_update") if isinstance(details, dict) else None
                )
                if isinstance(goal_update, dict):
                    expression = (
                        goal_update["added_expression"]
                        if "added_expression" in goal_update
                        else goal_update.get("existing_expression")
                    )
                    if expression is not None:
                        self.added_object_goal_expression = copy.deepcopy(expression)
                        self.added_object_goal_activation_step = step_number
            self.primary_applied = True
            if isinstance(cleanup, dict):
                self.cleanup_pending = True
                self.model_actions_since_primary = 0
        return reports

    def after_model_action(self) -> None:
        if self.cleanup_pending:
            self.model_actions_since_primary += 1


def _goal_expression_satisfied(
    backend: SymbolicBackend,
    expression: Any | None,
) -> bool | None:
    if expression is None:
        return None
    return bool(
        evaluate_goal_expression(
            normalize_goal_expression(expression),
            backend.evaluator._predicate_met,
        ).success
    )


def _manifest_goal_for_node(
    backend: SymbolicBackend,
    node_ref: Any,
    *,
    field: str,
) -> tuple[Any, str]:
    node_id = backend.world.resolve_node_id(node_ref)
    if node_id is None:
        raise ValueError(f"manifest add_object has unknown {field}: {node_ref!r}")
    projected = _project_goal_expression(
        backend.evaluator.task.task_completion_criterion,
        lambda raw: backend.world.resolve_node_id(raw) == node_id,
    )
    if projected is None:
        raise ValueError(
            f"manifest add_object cannot find a success criterion whose subject is {node_id!r}"
        )
    return projected, node_id


def _manifest_trigger_matches(
    trigger: dict[str, Any],
    *,
    backend: SymbolicBackend,
    step_number: int,
    history: list[dict[str, Any]],
    initial_completion_cost: float | None = None,
) -> bool:
    trigger_type = str(trigger.get("type") or "").strip().lower()
    if trigger_type == "at_step":
        return int(trigger.get("step", -1)) == step_number
    if trigger_type == "after_successful_action":
        if not history:
            return False
        previous = history[-1]
        event = previous.get("event")
        action = previous.get("action")
        if not isinstance(event, dict) or event.get("status") != "success":
            return False
        if not isinstance(action, dict):
            return False
        expected = trigger.get("action")
        if not isinstance(expected, dict):
            return False
        actual_name = str(action.get("base_name") or action.get("name") or "").lower()
        expected_name = str(expected.get("base_name") or expected.get("name") or "").lower()
        if actual_name.removeprefix("failed_") != expected_name.removeprefix("failed_"):
            return False
        expected_nodes = [str(node_id) for node_id in expected.get("node_ids") or []]
        actual_nodes = [str(node_id) for node_id in action.get("node_ids") or []]
        return not expected_nodes or actual_nodes == expected_nodes
    if trigger_type == "on_container_item_count_reached":
        node_id = backend.world.resolve_node_id(trigger.get("node_id"))
        if node_id is None:
            raise ValueError(f"manifest trigger has unknown container: {trigger.get('node_id')!r}")
        wanted_count = int(trigger.get("item_count", -1))
        return backend.world._container_item_count(node_id) == wanted_count
    if trigger_type == "on_container_max_items_reached":
        node_id = backend.world.resolve_node_id(trigger.get("node_id"))
        if node_id is None:
            raise ValueError(f"manifest trigger has unknown container: {trigger.get('node_id')!r}")
        state = backend.world.states[node_id]
        if not state.node.is_container:
            raise ValueError(f"manifest trigger node is not a container: {node_id}")
        max_items = state.node.max_items
        if max_items is None:
            raise ValueError(f"manifest trigger container has no max_items: {node_id}")
        return backend.world._container_item_count(node_id) >= max_items
    if trigger_type == "on_any_container_max_items_reached":
        return _matching_manifest_trigger_container(trigger, backend=backend) is not None
    if trigger_type == "on_object_goal_satisfied":
        expression, _ = _manifest_goal_for_node(
            backend,
            trigger.get("node_id"),
            field="trigger.node_id",
        )
        if not evaluate_goal_expression(
            expression,
            backend.evaluator._predicate_met,
        ).success:
            return False
        required_predicates = trigger.get("required_predicates", [])
        if not isinstance(required_predicates, list):
            raise ValueError("manifest trigger required_predicates must be a list")
        return all(
            evaluate_goal_expression(
                required,
                backend.evaluator._predicate_met,
            ).success
            for required in required_predicates
        )
    if trigger_type in {
        "first_eligible_state_regression_opportunity",
        "first_satisfied_goal_placement_opportunity",
        "first_satisfied_goal_wrong_destination_opportunity",
    }:
        # Candidate eligibility is evaluated against the live world by
        # _materialize_manifest_disturbance.  Returning true here makes the
        # runtime retry on every later step until one semantic opportunity
        # exists, instead of tying the condition to a teacher action/step.
        return step_number >= int(trigger.get("minimum_step", 2))
    if trigger_type == "first_goal_progress_opportunity":
        if step_number < int(trigger.get("minimum_step", 2)):
            return False
        if initial_completion_cost is None or initial_completion_cost <= 0:
            return False
        current_cost = _relaxed_completion_cost(backend)
        progress = max(
            0.0,
            min(1.0, (initial_completion_cost - current_cost) / initial_completion_cost),
        )
        return progress >= float(trigger.get("min_goal_progress", 0.01))
    if trigger_type == "first_eligible_occlusion_opportunity":
        minimum_step = int(trigger.get("minimum_step", 3))
        if step_number < minimum_step or bool(_safe_backend_success(backend)):
            return False
        if initial_completion_cost is None or initial_completion_cost <= 0:
            return False
        minimum_progress = float(trigger.get("min_goal_progress", 0.1))
        maximum_progress = float(trigger.get("max_goal_progress", 0.8))
        if not 0.0 <= minimum_progress <= maximum_progress <= 1.0:
            raise ValueError(
                "manifest occlusion progress window must satisfy "
                "0 <= min_goal_progress <= max_goal_progress <= 1"
            )
        current_cost = _relaxed_completion_cost(backend)
        progress = max(
            0.0,
            min(1.0, (initial_completion_cost - current_cost) / initial_completion_cost),
        )
        return minimum_progress <= progress <= maximum_progress
    raise ValueError(f"unsupported manifest trigger type: {trigger_type!r}")


def _matching_manifest_trigger_container(
    trigger: dict[str, Any],
    *,
    backend: SymbolicBackend,
) -> str | None:
    node_refs = trigger.get("node_ids")
    if not isinstance(node_refs, list) or not node_refs:
        raise ValueError("manifest any-container trigger needs a non-empty node_ids array")
    for node_ref in node_refs:
        node_id = backend.world.resolve_node_id(node_ref)
        if node_id is None:
            raise ValueError(f"manifest trigger has unknown container: {node_ref!r}")
        state = backend.world.states[node_id]
        if not state.node.is_container:
            raise ValueError(f"manifest trigger node is not a container: {node_id}")
        max_items = state.node.max_items
        if max_items is None:
            raise ValueError(f"manifest trigger container has no max_items: {node_id}")
        if backend.world._container_item_count(node_id) >= max_items:
            return node_id
    return None


def _materialize_manifest_disturbance(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    selection = disturbance.get("selection")
    if selection == RUNTIME_STATE_REGRESSION_SELECTION:
        return _materialize_runtime_state_regression(backend, disturbance)
    if selection == RUNTIME_COMPLETED_ROLLBACK_SELECTION:
        return _materialize_runtime_completed_rollback(backend, disturbance)
    if selection == RUNTIME_WRONG_RELOCATION_SELECTION:
        return _materialize_runtime_wrong_relocation(backend, disturbance)
    if not (
        disturbance.get("operation") == "add_occlusion"
        and selection in RUNTIME_OCCLUSION_SELECTIONS
    ):
        return copy.deepcopy(disturbance), None

    candidates = disturbance.get("candidate_pairs")
    if not isinstance(candidates, list):
        return None
    eligible: list[tuple[int, dict[str, Any]]] = []
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        spec = _eligible_runtime_occlusion_spec(backend, candidate)
        if spec is None:
            continue
        eligible.append((candidate_index, spec))
    if not eligible:
        return None

    # Closed openable occluders exercise a distinct and more constrained recovery
    # than a movable bag.  Candidate order previously made a movable bag win even
    # when several closed drawers were eligible later in the list.  Prefer an
    # executable `open` resolution, retaining manifest order within each class and
    # falling back to `move_aside` when no closed openable source is available.
    candidate_index, spec = min(
        eligible,
        key=lambda item: (
            0 if item[1]["resolution_action"] == "open" else 1,
            item[0],
        ),
    )
    return spec, {
        "strategy": "runtime_prefer_open_then_first_eligible",
        "selection_priority": "open_before_move_aside",
        "candidate_index": candidate_index,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(candidates),
        "source": spec["source"],
        "target": spec["target"],
        "resolution_action": spec["resolution_action"],
        "restore_source_action": spec.get("restore_source_action"),
        "previous_location": copy.deepcopy(spec["previous_location"]),
        "staging_location": {
            "relation": spec["staging_relation"],
            "target": spec["staging_target"],
        },
    }


def _materialize_runtime_state_regression(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = disturbance.get("candidate_regressions")
    if not isinstance(candidates, list):
        return None
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        node_id = backend.world.resolve_node_id(candidate.get("node_id"))
        values = candidate.get("values")
        achieved_values = candidate.get("achieved_values")
        if (
            node_id is None
            or not isinstance(values, dict)
            or not values
            or not isinstance(achieved_values, dict)
        ):
            continue
        state = backend.world.states[node_id]
        if not all(
            hasattr(state, field) and getattr(state, field) == wanted
            for field, wanted in achieved_values.items()
        ) or not backend.world.is_visible(node_id):
            continue
        spec = {
            "operation": "set_state",
            "node_id": node_id,
            "values": copy.deepcopy(values),
        }
        trial = copy.deepcopy(backend)
        cost_before = _relaxed_completion_cost(trial)
        planning_cost_before = _planning_completion_cost(trial)
        try:
            _apply_graph_disturbance(trial, spec)
            cost_after_disturbance = _relaxed_completion_cost(trial)
            planning_cost_after_disturbance = _planning_completion_cost(trial)
            recovery_action = candidate.get("recovery_action")
            recovery_event = (
                trial.step(_parsed_action(recovery_action))
                if isinstance(recovery_action, dict)
                else {"status": "failure"}
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            (
                cost_after_disturbance <= cost_before
                and planning_cost_after_disturbance <= planning_cost_before
            )
            or
            recovery_event.get("status") != "success"
            or _relaxed_completion_cost(trial) > cost_before
            or _planning_completion_cost(trial) > planning_cost_before
        ):
            continue
        return spec, {
            "strategy": RUNTIME_STATE_REGRESSION_SELECTION,
            "candidate_index": candidate_index,
            "candidate_count": len(candidates),
            "node_id": node_id,
            "recovery_action": copy.deepcopy(candidate.get("recovery_action")),
        }
    return None


def _materialize_runtime_completed_rollback(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = disturbance.get("candidate_relocations")
    if not isinstance(candidates, list):
        return None
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        node_id = backend.world.resolve_node_id(candidate.get("node_id"))
        target_id = backend.world.resolve_node_id(candidate.get("target"))
        if node_id is None or target_id is None:
            continue
        previous_location = _satisfied_goal_placement(backend, node_id)
        state = backend.world.states[node_id]
        if previous_location is None or state.held or not backend.world.is_visible(node_id):
            continue
        if previous_location["relation"] == "ON" and previous_location["target"] == target_id:
            continue
        spec = {
            "operation": "relocate",
            "node_id": node_id,
            "relation": "ON",
            "target": target_id,
        }
        trial = copy.deepcopy(backend)
        cost_before = _relaxed_completion_cost(trial)
        try:
            _apply_graph_disturbance(trial, spec)
        except (KeyError, TypeError, ValueError):
            continue
        if _relaxed_completion_cost(trial) <= cost_before or not trial.world.is_visible(node_id):
            continue
        return spec, {
            "strategy": RUNTIME_COMPLETED_ROLLBACK_SELECTION,
            "candidate_index": candidate_index,
            "candidate_count": len(candidates),
            "node_id": node_id,
            "previous_location": previous_location,
            "recovery_action": {
                "name": "putin" if previous_location["relation"] == "INSIDE" else "puton",
                "node_ids": [node_id, previous_location["target"]],
            },
        }
    return None


def _materialize_runtime_wrong_relocation(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    candidates = disturbance.get("candidate_relocations")
    if not isinstance(candidates, list):
        return None
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        node_id = backend.world.resolve_node_id(candidate.get("node_id"))
        target_id = backend.world.resolve_node_id(candidate.get("target"))
        relation = normalize_relation(str(candidate.get("relation") or ""))
        relation = "INSIDE" if relation == "IN" else relation
        if node_id is None or target_id is None or relation not in {"ON", "INSIDE"}:
            continue
        previous_location = _satisfied_goal_placement(backend, node_id)
        state = backend.world.states[node_id]
        target_state = backend.world.states[target_id]
        if previous_location is None or state.held or target_id == previous_location["target"]:
            continue
        if relation == "INSIDE" and (
            (target_state.node.is_openable and not target_state.open)
            or backend.world._container_is_full(target_id)
        ):
            continue
        if not backend.world.is_visible(target_id):
            continue
        spec = {
            "operation": "relocate",
            "node_id": node_id,
            "relation": relation,
            "target": target_id,
        }
        trial = copy.deepcopy(backend)
        cost_before = _relaxed_completion_cost(trial)
        try:
            _apply_graph_disturbance(trial, spec)
        except (KeyError, TypeError, ValueError):
            continue
        if _relaxed_completion_cost(trial) <= cost_before or not trial.world.is_visible(node_id):
            continue
        return spec, {
            "strategy": RUNTIME_WRONG_RELOCATION_SELECTION,
            "candidate_index": candidate_index,
            "candidate_count": len(candidates),
            "node_id": node_id,
            "previous_location": previous_location,
            "wrong_location": {"relation": relation, "target": target_id},
            "recovery_action": {
                "name": "putin" if previous_location["relation"] == "INSIDE" else "puton",
                "node_ids": [node_id, previous_location["target"]],
            },
        }
    return None


def _eligible_runtime_occlusion_spec(
    backend: SymbolicBackend,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    world = backend.world
    source_id = world.resolve_node_id(candidate.get("source"))
    target_id = world.resolve_node_id(candidate.get("target"))
    if (
        source_id is None
        or target_id is None
        or source_id == target_id
        or not world.is_visible(source_id)
        or not world.is_visible(target_id)
    ):
        return None
    source_state = world.states[source_id]
    target_state = world.states[target_id]
    if source_state.held or target_state.held:
        return None
    previous_location = _satisfied_goal_placement(backend, target_id)
    if previous_location is None:
        return None
    source_surface = _root_support_surface(world, source_id)
    target_surface = _root_support_surface(world, target_id)
    if source_surface is None or source_surface != target_surface:
        return None
    if (
        previous_location["relation"] == "ON"
        and previous_location["target"] == source_surface
    ):
        return None

    relation = normalize_relation(str(candidate.get("relation") or "OCCLUDES"))
    if any(
        edge[0] == source_id and edge[1] == target_id and edge[2] == relation
        for edge in world.active_occlusion_edges
    ):
        return None
    raw_actions = candidate.get("supported_resolution_actions") or ("open", "move_aside")
    supported_actions = {
        str(action).strip().lower()
        for action in ([raw_actions] if isinstance(raw_actions, str) else raw_actions)
    }
    resolution_action: str | None = None
    if (
        "open" in supported_actions
        and source_state.node.is_openable
        and not source_state.open
    ):
        resolution_action = "open"
    elif (
        "move_aside" in supported_actions
        and source_state.node.is_movable
    ):
        # A new external occlusion can move a previously cleared occluder back
        # into the way. add_occlusion(activate=True) resets moved_aside=False,
        # so the model must resolve this newly introduced obstruction again.
        resolution_action = "move_aside"
    if resolution_action is None:
        return None

    spec = {
        "operation": "relocate_and_add_occlusion",
        "source": source_id,
        "target": target_id,
        "relation": relation,
        "resolution_action": resolution_action,
        "restore_source_action": (
            "close" if resolution_action == "open" and not source_state.open else None
        ),
        "activate": True,
        "previous_location": previous_location,
        "staging_relation": "ON",
        "staging_target": source_surface,
    }
    trial = copy.deepcopy(backend)
    cost_before = _relaxed_completion_cost(trial)
    planning_cost_before = _planning_completion_cost(trial)
    try:
        details = _apply_graph_disturbance(trial, spec)
        occlusion_details = details.get("occlusion") or {}
        if (
            occlusion_details.get("active_after") is not True
            or trial.world.is_visible(target_id)
            or _relaxed_completion_cost(trial) <= cost_before
            or _planning_completion_cost(trial) <= planning_cost_before
        ):
            return None
        recovery_event = trial.step(
            _parsed_action({"name": resolution_action, "node_ids": [source_id]})
        )
        if recovery_event.get("status") != "success" or not trial.world.is_visible(target_id):
            return None
        grab_event = trial.step(_parsed_action({"name": "grab", "node_ids": [target_id]}))
        if grab_event.get("status") != "success":
            return None
        placement_target = str(previous_location["target"])
        placement_target_state = trial.world.states[placement_target]
        if (
            previous_location["relation"] == "INSIDE"
            and placement_target_state.node.is_openable
            and not placement_target_state.open
        ):
            # Opening while holding is disallowed, so a target that unexpectedly
            # became closed is not a recoverable candidate for this intervention.
            return None
        placement_action = "putin" if previous_location["relation"] == "INSIDE" else "puton"
        placement_event = trial.step(
            _parsed_action(
                {
                    "name": placement_action,
                    "node_ids": [target_id, placement_target],
                }
            )
        )
        restore_source_event: dict[str, Any] | None = None
        if spec["restore_source_action"] is not None:
            restore_source_event = trial.step(
                _parsed_action(
                    {
                        "name": spec["restore_source_action"],
                        "node_ids": [source_id],
                    }
                )
            )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        placement_event.get("status") != "success"
        or (
            restore_source_event is not None
            and restore_source_event.get("status") != "success"
        )
        or _relaxed_completion_cost(trial) > cost_before
        or _planning_completion_cost(trial) > planning_cost_before
    ):
        return None
    return spec


def _satisfied_goal_placement(
    backend: SymbolicBackend,
    node_id: str,
) -> dict[str, str] | None:
    """Return the exact currently satisfied ON/INSIDE goal leaf for a node.

    Goal leaves are checked independently so a satisfied branch of an OR is not
    rejected merely because its alternative destinations are false.
    """
    state = backend.world.states[node_id]
    criterion = backend.evaluator.task.task_completion_criterion
    for predicate, args in _goal_atoms(criterion):
        relation = "INSIDE" if predicate == "IN" else predicate
        if relation not in {"ON", "INSIDE"} or len(args) < 2:
            continue
        object_id = backend.world.resolve_node_id(args[0])
        target_id = backend.world.resolve_node_id(args[1])
        if object_id != node_id or target_id is None:
            continue
        if state.location_relation != relation or state.location_target != target_id:
            continue
        try:
            if backend.evaluator._predicate_met(predicate, args):
                return {"relation": relation, "target": target_id}
        except Exception:  # noqa: BLE001 - unsupported leaves are not eligible.
            continue
    return None


def _root_support_surface(world: Any, node_id: str) -> str | None:
    """Find the outermost surface supporting a node through ON/INSIDE ancestry."""
    current = node_id
    seen: set[str] = set()
    outermost_surface: str | None = None
    while current in world.states and current not in seen:
        seen.add(current)
        parent_id = world.states[current].location_target
        if parent_id is None or parent_id not in world.states:
            break
        parent = world.states[parent_id]
        if parent.node.is_surface:
            outermost_surface = parent_id
        current = parent_id
    return outermost_surface


def _manifest_cleanup_due(cleanup: dict[str, Any], model_action_count: int) -> bool:
    trigger = cleanup.get("trigger")
    if not isinstance(trigger, dict):
        return False
    trigger_type = str(trigger.get("type") or "").strip().lower()
    if trigger_type != "after_model_actions":
        raise ValueError(f"unsupported manifest cleanup trigger type: {trigger_type!r}")
    return model_action_count >= int(trigger.get("count", 1))


class ClosedLoopViewGraphHarness:
    def __init__(
        self,
        *,
        config: ViewGraphRolloutEvalConfig,
        brain_harness: BrainHarness,
        intervention_condition: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.brain_harness = brain_harness
        self.adapter = brain_harness.adapter
        self.failure_injection = FailureInjectionConfig(
            mode=config.failure_injection,
            actions=config.failure_actions,
            probability=config.failure_probability,
            max_failures_per_episode=config.max_failures_per_episode,
            seed=config.failure_seed,
            deduplication_scope=config.failure_deduplication_scope,
        )
        self.failure_seed_source = (
            random.Random(config.failure_seed) if config.failure_seed is not None else None
        )
        self.graph_disturbances = load_graph_disturbances(config.graph_disturbance_file)
        self.intervention_condition = copy.deepcopy(intervention_condition)

    def run_episode(
        self,
        episode: dict[str, Any],
        source_file: Path,
        record_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        task, graph, constraints = _replay_inputs(episode)
        backend = SymbolicBackend(graph, task, constraints)
        initial_observation = backend.observe()
        initial_cost = _relaxed_completion_cost(backend)
        initial_planning_cost = _planning_completion_cost(backend)
        initial_goal_satisfied = bool(_safe_backend_success(backend))
        teacher_step_count = _teacher_non_stop_step_count(episode)

        failure_episode_seed = (
            self.failure_seed_source.randrange(0, 2**63)
            if self.failure_injection.enabled and self.failure_seed_source is not None
            else random.SystemRandom().randrange(0, 2**63)
            if self.failure_injection.enabled
            else None
        )
        failure_rng = random.Random(failure_episode_seed)
        injected_failure_count = 0
        failed_action_keys: set[tuple[str, ...]] = set()
        disturbance_count = 0
        disturbance_cleanup_count = 0
        intervention_runtime = _ManifestInterventionRuntime(
            self.intervention_condition,
            initial_completion_cost=initial_cost,
        )

        records: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        first_goal_satisfied_step: int | None = 0 if initial_goal_satisfied else None
        first_added_object_goal_satisfied_step: int | None = None
        added_object_goal_model_steps_to_satisfaction: int | None = None
        goal_damaged_after_satisfaction = False
        consecutive_model_errors = 0
        termination_reason = "max_steps"

        for step_number in range(1, self.config.max_steps + 1):
            scheduled_disturbances = _apply_scheduled_graph_disturbances(
                backend,
                self.graph_disturbances,
                episode=episode,
                step_number=step_number,
            )
            manifest_disturbances = intervention_runtime.before_step(
                backend,
                step_number=step_number,
                history=history,
            )
            disturbances_applied = [*scheduled_disturbances, *manifest_disturbances]
            disturbance_count += sum(
                report.get("counts_as_intervention", True) is True
                for report in disturbances_applied
            )
            disturbance_cleanup_count += sum(
                report.get("phase") == "cleanup" for report in disturbances_applied
            )
            observation = backend.observe()
            projected_observation = _visible_graph_observation(observation)
            pre_snapshot = backend.snapshot()
            goal_before = bool(_safe_backend_success(backend))
            added_object_goal_expression = (
                intervention_runtime.added_object_goal_expression
            )
            added_object_goal_before = _goal_expression_satisfied(
                backend,
                added_object_goal_expression,
            )
            if (
                added_object_goal_before is True
                and first_added_object_goal_satisfied_step is None
            ):
                first_added_object_goal_satisfied_step = step_number
                activation_step = intervention_runtime.added_object_goal_activation_step
                added_object_goal_model_steps_to_satisfaction = (
                    step_number - activation_step
                    if activation_step is not None
                    else None
                )
            cost_before = _relaxed_completion_cost(backend)
            planning_cost_before = _planning_completion_cost(backend)
            if first_goal_satisfied_step is not None and not goal_before:
                goal_damaged_after_satisfaction = True
            current_opportunities, current_hidden = _exploration_obligations(backend)
            generated_valid_actions = _normal_valid_actions(observation)
            provided_valid_actions = generated_valid_actions if self.config.include_valid_actions else []
            expected_recovery = _expected_recovery_from_history(history)
            request = self.adapter.build_request(
                task=task,
                step={"step": step_number},
                history=history,
                valid_actions=provided_valid_actions,
                graph_observation=observation,
                wrong_graph_observation=None,
                frame_files=[],
                mode=CLOSED_LOOP_MODE,
                history_source="inference",
                frame_observation={},
                include_valid_actions=self.config.include_valid_actions,
            )
            record: dict[str, Any] = {
                "evaluation_type": CLOSED_LOOP_EVALUATION_TYPE,
                "source_file": str(source_file),
                "model_name": _rollout_model_name(self.config),
                "episode_id": episode.get("episode_id"),
                "scene_id": episode.get("scene_id"),
                "env_id": episode.get("env_id"),
                "step": step_number,
                "mode": CLOSED_LOOP_MODE,
                "condition_id": intervention_runtime.condition_id,
                "intervention_type": intervention_runtime.intervention_type,
                "includes_valid_actions": self.config.include_valid_actions,
                "failure_injection": self.config.failure_injection,
                "failure_injection_config": self.failure_injection.to_json(),
                "failure_episode_seed": failure_episode_seed,
                "current_observation": projected_observation,
                "generated_valid_actions": generated_valid_actions,
                "valid_actions": provided_valid_actions,
                "input_history": copy.deepcopy(history[-self.config.history_window :]),
                "expected_recovery": expected_recovery,
                "request_summary": request.summary,
                "raw_response": None,
                "parsed_response": None,
                "predicted_action": None,
                "predicted_recovery": None,
                "reason": None,
                "parse_error": None,
                "parse_repair": None,
                "response_metadata": None,
                "model_error": None,
                "event": None,
                "injection_applied": False,
                "failure_injection_record": None,
                "disturbance_applied": bool(disturbances_applied),
                "disturbances_applied": disturbances_applied,
                "normally_executable": None,
                "pre_state_hash": _state_hash(pre_snapshot),
                "post_state_hash": None,
                "goal_satisfied_before_action": goal_before,
                "goal_satisfied_after_action": None,
                "added_object_goal_eligible": added_object_goal_expression is not None,
                "added_object_goal_expression": copy.deepcopy(
                    added_object_goal_expression
                ),
                "added_object_goal_satisfied_before_action": added_object_goal_before,
                "added_object_goal_satisfied_after_action": None,
                "relaxed_completion_cost_before": cost_before,
                "goal_completion_cost_before": cost_before,
                "planning_cost_before": planning_cost_before,
                "relaxed_completion_cost_after": None,
                "goal_completion_cost_after": None,
                "planning_cost_after": None,
                "new_visible_nodes": [],
                "recovery_score": None,
                "capability_scores": None,
            }

            predicted: dict[str, Any] = {}
            predicted_recovery: dict[str, Any] | None = None
            decision_parse_error: str | None = None
            model_error: str | None = None
            try:
                decision = self.brain_harness.decide_request(request)
                predicted = _structured_action(decision.action)
                predicted_recovery = _predicted_recovery(decision.parsed_response)
                decision_parse_error = decision.parse_error
                record.update(
                    {
                        "raw_response": decision.raw_response,
                        "parsed_response": decision.parsed_response,
                        "predicted_action": predicted,
                        "predicted_recovery": predicted_recovery,
                        "reason": decision.reason,
                        "parse_error": decision.parse_error,
                        "parse_repair": decision.parse_repair,
                        "response_metadata": copy.deepcopy(
                            self.brain_harness.last_response_metadata
                        ),
                    }
                )
                consecutive_model_errors = 0
            except Exception as exc:  # noqa: BLE001 - rollout records provider failures.
                if self.config.fail_fast:
                    raise
                model_error = str(exc)
                decision_parse_error = f"model request failed: {exc}"
                record["model_error"] = model_error
                record["response_metadata"] = copy.deepcopy(
                    self.brain_harness.last_response_metadata
                )
                consecutive_model_errors += 1

            context = ReplayStepContext(
                step_index=step_number - 1,
                step={"step": step_number},
                graph_observation=observation,
                valid_actions=generated_valid_actions,
                history=copy.deepcopy(history),
                expected_action={},
                expected_recovery=expected_recovery,
                generated_valid_actions=generated_valid_actions,
                counterfactual_backend=copy.deepcopy(backend),
            )
            parsed = bool(
                model_error is None
                and decision_parse_error is None
                and predicted.get("name")
                and predicted.get("name") != "invalid_teacher_action"
            )
            if parsed and str(predicted.get("base_name") or predicted.get("name")) == "recover":
                parsed = False
                decision_parse_error = "recover is an internal recovery flag, not an executable action"
                record["parse_error"] = decision_parse_error
            unordered_attach = not self.config.include_valid_actions
            predicted_key = (
                _canonical_action_key(predicted, backend, unordered_attach=unordered_attach)
                if parsed
                else None
            )
            candidate_keys = {
                _canonical_action_key(action, backend, unordered_attach=unordered_attach)
                for action in generated_valid_actions
            }
            soft_score = _soft_optimal_action_score(
                context,
                predicted if parsed else {},
                beta=self.config.soft_optimal_beta,
                unordered_attach=unordered_attach,
            )
            recovery_score = _score_recovery(
                expected_recovery,
                predicted_recovery,
                decision_parse_error,
            )

            normal_event: dict[str, Any] | None = None
            if parsed and predicted_key is not None and predicted_key[0] != "stop":
                trial = copy.deepcopy(backend)
                normal_event = trial.step(_parsed_action(predicted))
                record["normally_executable"] = normal_event.get("status") == "success"

            if model_error is not None:
                event = {
                    "status": "failure",
                    "failure_type": "model_error",
                    "message": model_error,
                    "attempted": False,
                }
            elif not parsed:
                event = {
                    "status": "failure",
                    "failure_type": "parse_error",
                    "message": decision_parse_error or "invalid model action",
                    "attempted": False,
                }
            elif predicted_key is not None and predicted_key[0] == "stop":
                event = backend.step(_parsed_action(predicted))
                event["attempted"] = True
                termination_reason = "correct_stop" if goal_before else "premature_stop"
            elif _should_inject_closed_loop_failure(
                config=self.failure_injection,
                action=predicted,
                normal_event=normal_event,
                injected_failure_count=injected_failure_count,
                failed_action_keys=failed_action_keys,
                rng=failure_rng,
            ):
                failed_action_name = str(
                    predicted.get("base_name") or predicted.get("name")
                ).lower().removeprefix("failed_")
                failure_key = self.failure_injection.deduplication_key(
                    failed_action_name,
                    [str(node_id) for node_id in predicted.get("node_ids") or []],
                )
                injected_failure_count += 1
                failed_action_keys.add(failure_key)
                failed_action = ParsedAction(
                    name=f"failed_{failed_action_name}",
                    node_ids=list(predicted.get("node_ids") or []),
                    raw=json.dumps(predicted, ensure_ascii=False),
                )
                event = backend.step(failed_action)
                event["attempted"] = True
                record["injection_applied"] = True
                record["failure_injection_record"] = {
                    "mode": self.failure_injection.mode,
                    "original_action": copy.deepcopy(predicted),
                    "failed_action": failed_action.to_json(),
                    "failure_index": injected_failure_count,
                    "episode_seed": failure_episode_seed,
                    "deduplication_scope": self.failure_injection.deduplication_scope,
                    "deduplication_key": list(failure_key),
                }
            else:
                event = backend.step(_parsed_action(predicted))
                event["attempted"] = True

            post_observation = backend.observe()
            post_snapshot = backend.snapshot()
            goal_after = bool(_safe_backend_success(backend))
            added_object_goal_after = _goal_expression_satisfied(
                backend,
                added_object_goal_expression,
            )
            if (
                added_object_goal_after is True
                and first_added_object_goal_satisfied_step is None
            ):
                first_added_object_goal_satisfied_step = step_number
                activation_step = intervention_runtime.added_object_goal_activation_step
                added_object_goal_model_steps_to_satisfaction = (
                    step_number - activation_step + 1
                    if activation_step is not None
                    else None
                )
            cost_after = _relaxed_completion_cost(backend)
            planning_cost_after = _planning_completion_cost(backend)
            new_visible_nodes = _new_visible_nodes(observation, post_observation)
            if goal_after and first_goal_satisfied_step is None:
                first_goal_satisfied_step = step_number
            if first_goal_satisfied_step is not None and not goal_after:
                goal_damaged_after_satisfaction = True

            handled_opportunities: list[str] = []
            action_name, action_nodes = predicted_key or ("", ())
            if event.get("status") == "success" and action_name in {"open", "close", "move_aside"} and action_nodes:
                opportunity_id = f"{action_name}:{action_nodes[0]}"
                if opportunity_id in current_opportunities:
                    handled_opportunities.append(opportunity_id)
            revealed_hidden = sorted(
                node_id for node_id in current_hidden if backend.world.is_visible(node_id)
            )
            predicted_stop = bool(parsed and action_name == "stop")
            capability_scores = {
                "action_selection": {
                    "action_admissibility_rate": {
                        "eligible": True,
                        "value": bool(predicted_key in candidate_keys) if predicted_key is not None else False,
                        "predicted_key": _action_key_payload(predicted_key),
                        "candidate_count": len(candidate_keys),
                    },
                    "soft_optimal_action_score": soft_score,
                },
                "failure_recovery": {
                    "recovery_detection_f1": {
                        "eligible": True,
                        "expected_positive": bool(expected_recovery["required"]),
                        "predicted_positive": bool(
                            predicted_recovery is not None and predicted_recovery.get("required") is True
                        ),
                    },
                    "recovery_grounding_accuracy": {
                        "eligible": bool(expected_recovery["required"]),
                        "value": bool(recovery_score["grounding_exact"])
                        if expected_recovery["required"]
                        else None,
                        "expected_failed_action": expected_recovery.get("failed_action"),
                        "expected_failed_node_ids": list(expected_recovery.get("failed_node_ids") or []),
                    },
                },
                "active_exploration": {
                    "exploration_opportunity_recall": {
                        "eligible": bool(current_opportunities),
                        "opportunity_ids": sorted(current_opportunities),
                        "handled_opportunity_ids": handled_opportunities,
                    },
                    "normalized_goal_information_gain": {
                        "eligible": bool(current_hidden),
                        "hidden_goal_node_ids": sorted(current_hidden),
                        "revealed_goal_node_ids": revealed_hidden,
                    },
                },
                "completion_judgment": {
                    "premature_stop_rate": {
                        "eligible": not goal_before,
                        "value": predicted_stop if not goal_before else None,
                    },
                    "completion_stop_recall": {
                        "eligible": goal_before,
                        "value": predicted_stop if goal_before else None,
                    },
                    "goal_satisfied_before_action": goal_before,
                },
            }
            record.update(
                {
                    "event": event,
                    "post_state_hash": _state_hash(post_snapshot),
                    "goal_satisfied_after_action": goal_after,
                    "added_object_goal_satisfied_after_action": added_object_goal_after,
                    "relaxed_completion_cost_after": cost_after,
                    "goal_completion_cost_after": cost_after,
                    "planning_cost_after": planning_cost_after,
                    "new_visible_nodes": new_visible_nodes,
                    "recovery_score": recovery_score,
                    "capability_scores": capability_scores,
                }
            )

            history.append(
                {
                    "step": step_number,
                    "action": copy.deepcopy(predicted) if predicted else None,
                    "event": copy.deepcopy(event),
                    "new_visible_nodes": copy.deepcopy(new_visible_nodes),
                    "success_after_step": goal_after,
                    "parse_error": decision_parse_error if model_error is None else None,
                    "model_error": model_error,
                }
            )
            intervention_runtime.after_model_action()

            terminal = predicted_stop or consecutive_model_errors >= self.config.max_consecutive_model_errors
            if consecutive_model_errors >= self.config.max_consecutive_model_errors:
                termination_reason = "model_error_limit"
            if step_number == self.config.max_steps and not terminal:
                terminal = True
                termination_reason = "max_steps"
            if terminal:
                outcome = _episode_outcome(
                    backend=backend,
                    records=[*records, record],
                    termination_reason=termination_reason,
                    first_goal_satisfied_step=first_goal_satisfied_step,
                    goal_damaged_after_satisfaction=goal_damaged_after_satisfaction,
                    initial_cost=initial_cost,
                    initial_planning_cost=initial_planning_cost,
                    teacher_step_count=teacher_step_count,
                    injected_failure_count=injected_failure_count,
                    disturbance_count=disturbance_count,
                    disturbance_cleanup_count=disturbance_cleanup_count,
                    condition_id=intervention_runtime.condition_id,
                    intervention_type=intervention_runtime.intervention_type,
                    added_object_goal_expression=(
                        intervention_runtime.added_object_goal_expression
                    ),
                    added_object_goal_activation_step=(
                        intervention_runtime.added_object_goal_activation_step
                    ),
                    first_added_object_goal_satisfied_step=(
                        first_added_object_goal_satisfied_step
                    ),
                    added_object_goal_model_steps_to_satisfaction=(
                        added_object_goal_model_steps_to_satisfaction
                    ),
                )
                record["rollout_outcome"] = outcome

            records.append(record)
            if record_callback is not None:
                record_callback(record)
            if terminal:
                break

        return records, records[-1]["rollout_outcome"]


def evaluate_view_graph_rollouts(
    *,
    input_path: str | Path,
    output_path: str | Path,
    config: ViewGraphRolloutEvalConfig,
    brain_harness: BrainHarness | None = None,
    intervention_condition: dict[str, Any] | None = None,
    episode_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    target = _timestamped_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    adapter = ClosedLoopVisibleGraphAdapter(history_window=config.history_window)
    if brain_harness is None:
        policy = BrainPolicy(
            BrainPolicyConfig(
                provider=config.provider,
                model=config.model or _default_model(config.provider),
                api_key_env=config.api_key_env,
                api_base_url=config.api_base_url,
                timeout_seconds=config.timeout_seconds,
                temperature=config.temperature,
                max_attempts=config.max_api_attempts,
                retry_backoff_seconds=config.retry_backoff_seconds,
                retry_max_seconds=config.retry_max_seconds,
                api_style=_resolved_api_style(config.provider, config.api_style),
                max_output_tokens=config.max_output_tokens,
                json_response_format=config.json_response_format,
            )
        )
        brain_harness = BrainHarness(policy, adapter)
    harness = ClosedLoopViewGraphHarness(
        config=config,
        brain_harness=brain_harness,
        intervention_condition=intervention_condition,
    )

    records: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    completed_records = 0
    with target.open("w", encoding="utf-8") as out:
        def write_record(record: dict[str, Any]) -> None:
            nonlocal completed_records
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            completed_records += 1
            print(
                f"Completed closed-loop record {completed_records}: "
                f"episode={record['episode_id']} step={record['step']}",
                flush=True,
            )

        if episode_override is not None:
            episodes = [copy.deepcopy(episode_override)]
        else:
            episodes = []
            with source.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    episode = json.loads(stripped)
                    if not isinstance(episode, dict):
                        raise ValueError(f"{source}:{line_no}: expected JSON object")
                    episodes.append(episode)
        for episode in episodes:
            episode_records, outcome = harness.run_episode(
                episode,
                source,
                record_callback=write_record,
            )
            records.extend(episode_records)
            outcomes.append(outcome)

    summary = _closed_loop_summary(
        records,
        outcomes,
        config,
        intervention_condition=intervention_condition,
    )
    summary_path = target.with_name(f"{target.stem}__summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "count": len(records),
        "episode_count": len(outcomes),
        "output_path": str(target),
        "summary_path": str(summary_path),
    }


def evaluate_view_graph_intervention_manifest(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    config: ViewGraphRolloutEvalConfig,
    condition_ids: tuple[str, ...] = (),
    brain_harness_factory: Callable[[dict[str, Any]], BrainHarness] | None = None,
) -> dict[str, Any]:
    """Run each selected manifest condition as an isolated rollout."""
    manifest = load_intervention_manifest(manifest_path)
    selected_ids = {condition_id for condition_id in condition_ids if condition_id and condition_id != "all"}
    all_condition_ids = {str(item["condition_id"]) for item in manifest["conditions"]}
    unknown_ids = selected_ids - all_condition_ids
    if unknown_ids:
        raise ValueError(f"unknown manifest condition ids: {sorted(unknown_ids)}")
    conditions = [
        condition
        for condition in manifest["conditions"]
        if not selected_ids or condition["condition_id"] in selected_ids
    ]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    episode, source_file = episode_from_manifest_source(manifest["source"])
    episode_id = str(manifest["source"].get("episode_id") or episode["episode_id"])
    variant = "valid_action" if config.include_valid_actions else "no_valid_action"
    model_slug = _filename_slug(config.model or _default_model(config.provider))

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        if condition.get("eligible") is False:
            skipped.append({"condition_id": condition_id, "reason": "ineligible"})
            continue
        failure = condition.get("failure_injection") or {"mode": "none"}
        raw_actions = failure.get("actions") or ("all",)
        failure_actions = (
            (str(raw_actions),)
            if isinstance(raw_actions, str)
            else tuple(str(action) for action in raw_actions)
        )
        condition_config = replace(
            config,
            failure_injection=str(failure.get("mode") or "none"),
            failure_actions=failure_actions,
            failure_probability=float(failure.get("probability", 0.0)),
            max_failures_per_episode=int(failure.get("max_failures_per_episode", 1)),
            failure_seed=int(failure["seed"]) if failure.get("seed") is not None else None,
            failure_deduplication_scope=str(
                failure.get("deduplication_scope") or "signature"
            ),
            graph_disturbance_file=None,
        )
        output_base = target_dir / (
            f"closed_loop_eval_{_filename_slug(episode_id)}_{model_slug}_{variant}_"
            f"{_filename_slug(condition_id)}.jsonl"
        )
        brain_harness = (
            brain_harness_factory(condition)
            if brain_harness_factory is not None
            else None
        )
        result = evaluate_view_graph_rollouts(
            input_path=source_file,
            output_path=output_base,
            config=condition_config,
            brain_harness=brain_harness,
            intervention_condition=condition,
            episode_override=episode,
        )
        condition_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        results.append(
            {
                "condition_id": condition_id,
                "intervention_type": condition["intervention_type"],
                **result,
                "outcomes": condition_summary.get("outcomes"),
            }
        )

    suite_summary = {
        "evaluation_type": CLOSED_LOOP_EVALUATION_TYPE,
        "closed_loop_metric_version": CLOSED_LOOP_METRIC_VERSION,
        "capability_metric_version": CAPABILITY_METRIC_VERSION,
        "manifest_type": manifest["manifest_type"],
        "manifest_path": manifest["_manifest_path"],
        "suite_id": manifest.get("suite_id"),
        "episode_id": episode_id,
        "model_name": _rollout_model_name(config),
        "provider": config.provider,
        "includes_valid_actions": config.include_valid_actions,
        "selected_condition_ids": [condition["condition_id"] for condition in conditions],
        "completed_condition_count": len(results),
        "skipped_condition_count": len(skipped),
        "results": results,
        "skipped": skipped,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    suite_summary_path = _timestamped_output_path(
        target_dir
        / (
            f"closed_loop_intervention_suite_{_filename_slug(episode_id)}_"
            f"{model_slug}_{variant}.json"
        )
    )
    suite_summary_path.write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "condition_count": len(results),
        "skipped_condition_count": len(skipped),
        "suite_summary_path": str(suite_summary_path),
        "results": results,
    }


def _filename_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in str(value))
    return slug.strip("_") or "unnamed"


def _normal_valid_actions(observation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        action
        for action in _valid_teacher_actions(observation, [])
        if action.get("name") in CLOSED_LOOP_EXECUTABLE_ACTIONS
    ]


def _should_inject_closed_loop_failure(
    *,
    config: FailureInjectionConfig,
    action: dict[str, Any],
    normal_event: dict[str, Any] | None,
    injected_failure_count: int,
    failed_action_keys: set[tuple[str, ...]],
    rng: random.Random,
) -> bool:
    if not config.enabled or normal_event is None or normal_event.get("status") != "success":
        return False
    action_name = str(action.get("base_name") or action.get("name") or "").lower()
    action_name = action_name.removeprefix("failed_")
    if not action_name or not config.allows(action_name):
        return False
    if injected_failure_count >= config.max_failures_per_episode:
        return False
    failure_key = config.deduplication_key(
        action_name,
        [str(node_id) for node_id in action.get("node_ids") or []],
    )
    if failure_key in failed_action_keys:
        return False
    if config.mode == "once":
        return injected_failure_count == 0
    if config.mode == "all":
        return True
    if config.mode == "probability":
        return rng.random() < config.probability
    return False


def _apply_scheduled_graph_disturbances(
    backend: SymbolicBackend,
    disturbances: tuple[dict[str, Any], ...],
    *,
    episode: dict[str, Any],
    step_number: int,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for schedule_index, disturbance in enumerate(disturbances, start=1):
        if not _graph_disturbance_matches(
            disturbance,
            episode=episode,
            step_number=step_number,
        ):
            continue
        reports.append(
            _graph_disturbance_report(
                backend,
                disturbance,
                step_number=step_number,
                phase="scheduled",
                counts_as_intervention=True,
                schedule_index=schedule_index,
            )
        )
    return reports


def _graph_disturbance_report(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
    *,
    step_number: int,
    phase: str,
    counts_as_intervention: bool,
    schedule_index: int | None = None,
) -> dict[str, Any]:
    before_observation = backend.observe()
    before_snapshot = backend.snapshot()
    goal_before = bool(_safe_backend_success(backend))
    cost_before = _relaxed_completion_cost(backend)
    planning_cost_before = _planning_completion_cost(backend)
    details = _apply_graph_disturbance(backend, disturbance)
    after_observation = backend.observe()
    after_snapshot = backend.snapshot()
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
        "step": step_number,
        "phase": phase,
        "counts_as_intervention": counts_as_intervention,
        "operation": disturbance["operation"],
        "spec": copy.deepcopy(disturbance),
        "details": details,
        "pre_state_hash": _state_hash(before_snapshot),
        "post_state_hash": _state_hash(after_snapshot),
        "state_changed": before_snapshot != after_snapshot,
        "goal_satisfied_before": goal_before,
        "goal_satisfied_after": bool(_safe_backend_success(backend)),
        "relaxed_completion_cost_before": cost_before,
        "relaxed_completion_cost_after": _relaxed_completion_cost(backend),
        "goal_completion_cost_before": cost_before,
        "goal_completion_cost_after": _relaxed_completion_cost(backend),
        "planning_cost_before": planning_cost_before,
        "planning_cost_after": _planning_completion_cost(backend),
        "new_visible_nodes": sorted(after_visible - before_visible),
        "new_hidden_nodes": sorted(before_visible - after_visible),
    }
    if schedule_index is not None:
        report["schedule_index"] = schedule_index
    return report


def _graph_disturbance_matches(
    disturbance: dict[str, Any],
    *,
    episode: dict[str, Any],
    step_number: int,
) -> bool:
    if disturbance.get("step") != step_number:
        return False
    for field in ("episode_id", "scene_id", "env_id"):
        wanted = disturbance.get(field)
        if wanted is not None and str(wanted) != str(episode.get(field)):
            return False
    return True


def _resolve_disturbance_node_id(backend: SymbolicBackend, reference: Any, field: str) -> str:
    node_id = backend.world.resolve_node_id(reference)
    if node_id is None:
        raise ValueError(f"graph disturbance has unknown {field}: {reference!r}")
    return node_id


def _copy_from_node_payload(
    source_node: Node,
    identity_spec: dict[str, Any],
) -> dict[str, Any]:
    """Copy every source attribute while allowing only identity/part ownership changes."""
    payload = _node_to_condition_object_spec(source_node)
    for field in ("id", "name"):
        if identity_spec.get(field) is not None:
            payload[field] = copy.deepcopy(identity_spec[field])
    if identity_spec.get("part_of") is not None:
        payload["part_of"] = copy.deepcopy(identity_spec["part_of"])
    return payload


def _apply_manifest_add_object(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> dict[str, Any]:
    world = backend.world
    raw_spawns = [
        {
            "object": copy.deepcopy(disturbance["object"]),
            "relation": disturbance["relation"],
            "target": disturbance["target"],
        },
        *copy.deepcopy(disturbance.get("component_objects", [])),
    ]
    spawned: list[tuple[Node, NodeState, str | None]] = []
    new_ids: set[str] = set()
    has_components = len(raw_spawns) > 1

    for spawn_index, spawn in enumerate(raw_spawns):
        object_spec = spawn["object"]
        object_id = str(object_spec["id"])
        if object_id in world.states or object_id in new_ids:
            raise ValueError(
                f"manifest add_object node {object_id!r} must be absent and unique"
            )
        new_ids.add(object_id)

        copy_from = object_spec.pop("copy_from", None)
        copied_from_id: str | None = None
        if copy_from is not None:
            copied_from_id = _resolve_disturbance_node_id(
                backend,
                copy_from,
                f"component_objects[{spawn_index}].object.copy_from",
            )
            node_payload = _copy_from_node_payload(
                world.states[copied_from_id].node,
                object_spec,
            )
            # The registry supplies a natural identity. Behavioral attributes
            # always come from the actual copy_from node in the profiled graph.
        else:
            node_payload = object_spec
        node = Node.from_dict(node_payload)

        relation = spawn["relation"]
        target_id = _resolve_disturbance_node_id(
            backend,
            spawn["target"],
            f"component_objects[{spawn_index}].target",
        )
        if target_id == node.id:
            raise ValueError("manifest add_object cannot place a node onto/inside itself")
        target = world.states[target_id]
        if relation == "ON" and not target.node.is_surface:
            raise ValueError(f"manifest add_object ON target is not a surface: {target_id}")
        if relation == "INSIDE" and not target.node.is_container:
            raise ValueError(
                f"manifest add_object INSIDE target is not a container: {target_id}"
            )
        state = NodeState(
            node=node,
            location_relation=relation,
            location_target=target_id,
            open="OPEN" in node.states,
            assembled=("ASSEMBLED" in node.states and not has_components),
            pressed="PRESSED" in node.states,
        )
        spawned.append((node, state, copied_from_id))

    node, state, copied_from_id = spawned[0]

    success_policy = disturbance["success_policy"]
    policy_type = success_policy["type"]
    goal_update: dict[str, Any] = {"type": policy_type, "changed": False}
    inherited_expression: Any | None = None
    if policy_type == "inherit_from":
        source_expression, source_id = _manifest_goal_for_node(
            backend,
            success_policy.get("source_node_id"),
            field="success_policy.source_node_id",
        )
        placement_alternatives = [
            str(value)
            for value in success_policy.get("placement_alternatives") or []
        ]
        source_placement_targets: list[str] = []
        for predicate, args in _goal_atoms(source_expression):
            relation = normalize_relation(predicate)
            relation = "INSIDE" if relation == "IN" else relation
            if relation not in {"INSIDE", "ON"} or len(args) < 2:
                continue
            if world.resolve_node_id(args[0]) != source_id:
                continue
            target_id = world.resolve_node_id(args[1])
            if target_id is not None and target_id not in source_placement_targets:
                source_placement_targets.append(target_id)
        if placement_alternatives and set(placement_alternatives) != set(
            source_placement_targets
        ):
            raise ValueError(
                "manifest inherit_from placement_alternatives must exactly match "
                f"the source goal targets: expected {source_placement_targets}, "
                f"got {placement_alternatives}"
            )
        inherited_expression = _replace_goal_subject(
            source_expression,
            lambda raw: world.resolve_node_id(raw) == source_id,
            node.id,
        )
        goal_update.update(
            {
                "source_node_id": source_id,
                "source_expression": source_expression,
                "placement_alternatives": placement_alternatives,
                "added_expression": inherited_expression,
            }
        )
    elif policy_type == "existing_task_goal":
        references = {node.id, node.name}
        existing_expression = _project_goal_expression(
            backend.evaluator.task.task_completion_criterion,
            lambda raw: str(raw) in references,
        )
        if existing_expression is None:
            raise ValueError(
                "manifest add_object existing_task_goal requires the initial task "
                f"criterion to reference {node.id!r}/{node.name!r}"
            )
        goal_update["existing_expression"] = existing_expression
    else:
        trigger_container_id = _resolve_disturbance_node_id(
            backend,
            success_policy.get("trigger_container_id"),
            "success_policy.trigger_container_id",
        )
        trigger_container = world.states[trigger_container_id].node
        references = {node.id, node.name}
        existing_expression = _project_goal_expression(
            backend.evaluator.task.task_completion_criterion,
            lambda raw: str(raw) in references,
        )
        if existing_expression is not None:
            raise ValueError(
                "manifest add_object trigger_container_goal requires the initial task "
                f"criterion not to reference {node.id!r}/{node.name!r}"
            )
        inherited_expression = [
            success_policy.get("predicate", "INSIDE"),
            node.name,
            trigger_container.name,
        ]
        goal_update.update(
            {
                "trigger_container_id": trigger_container_id,
                "trigger_container_name": trigger_container.name,
                "added_expression": inherited_expression,
            }
        )

    for spawned_node, spawned_state, _ in spawned:
        world.states[spawned_node.id] = spawned_state
        world._name_to_id.setdefault(spawned_node.id, spawned_node.id)
        world._name_to_id.setdefault(spawned_node.name, spawned_node.id)
        world._name_to_id.setdefault(spawned_node.name.lower(), spawned_node.id)

    if inherited_expression is not None:
        previous_criterion = copy.deepcopy(
            backend.evaluator.task.task_completion_criterion
        )
        if policy_type == "inherit_from":
            backend.evaluator.task.task_completion_criterion = {
                "and": [
                    normalize_goal_expression(previous_criterion),
                    inherited_expression,
                ]
            }
        else:
            backend.evaluator.task.task_completion_criterion = _append_goal_conjunct(
                previous_criterion,
                inherited_expression,
            )
        goal_update.update(
            {
                "changed": True,
                "previous_criterion": previous_criterion,
                "effective_criterion": copy.deepcopy(
                    backend.evaluator.task.task_completion_criterion
                ),
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
        "copied_from": copied_from_id,
        "component_node_ids": [item[0].id for item in spawned[1:]],
        "components": [
            {
                "node": _node_to_condition_object_spec(component_node),
                "copied_from": component_source,
                "new_location": {
                    "relation": component_state.location_relation,
                    "target": component_state.location_target,
                    "held": component_state.held,
                },
            }
            for component_node, component_state, component_source in spawned[1:]
        ],
        "goal_update": goal_update,
    }


def _apply_graph_disturbance(
    backend: SymbolicBackend,
    disturbance: dict[str, Any],
) -> dict[str, Any]:
    operation = disturbance["operation"]
    world = backend.world
    if operation == "add_object":
        return _apply_manifest_add_object(backend, disturbance)
    if operation == "relocate_and_add_occlusion":
        target_id = _resolve_disturbance_node_id(backend, disturbance["target"], "target")
        expected_location = disturbance.get("previous_location")
        if not isinstance(expected_location, dict):
            raise ValueError(
                "graph disturbance relocate_and_add_occlusion needs previous_location"
            )
        target_state = world.states[target_id]
        expected_relation = normalize_relation(str(expected_location.get("relation") or ""))
        expected_relation = "INSIDE" if expected_relation == "IN" else expected_relation
        expected_target = _resolve_disturbance_node_id(
            backend, expected_location.get("target"), "previous_location.target"
        )
        if (
            target_state.location_relation != expected_relation
            or target_state.location_target != expected_target
            or target_state.held
        ):
            raise ValueError(
                "graph disturbance target no longer matches its recorded completed placement"
            )
        relocation = _apply_graph_disturbance(
            backend,
            {
                "operation": "relocate",
                "node_id": target_id,
                "relation": disturbance["staging_relation"],
                "target": disturbance["staging_target"],
            },
        )
        occlusion = _apply_graph_disturbance(
            backend,
            {
                "operation": "add_occlusion",
                "source": disturbance["source"],
                "target": target_id,
                "relation": disturbance.get("relation", "OCCLUDES"),
                "resolution_action": disturbance.get("resolution_action"),
                "activate": disturbance.get("activate", True),
            },
        )
        return {
            "target": target_id,
            "previous_location": copy.deepcopy(expected_location),
            "staging_location": copy.deepcopy(relocation["new_location"]),
            "relocation": relocation,
            "occlusion": occlusion,
        }

    if operation == "set_state":
        node_id = _resolve_disturbance_node_id(backend, disturbance["node_id"], "node_id")
        state = world.states[node_id]
        requested_values = disturbance["values"]
        previous_values = {
            key: copy.deepcopy(getattr(state, key))
            for key in requested_values
        }
        for key, value in requested_values.items():
            if key == "attached_to" and value is not None:
                value = _resolve_disturbance_node_id(backend, value, "values.attached_to")
            setattr(state, key, copy.deepcopy(value))
        if requested_values.get("held") is True:
            world._remove_occlusions_for_location_change(node_id)
            state.location_relation = None
            state.location_target = None
        return {
            "node_id": node_id,
            "previous_values": previous_values,
            "new_values": {
                key: copy.deepcopy(getattr(state, key))
                for key in requested_values
            },
        }

    if operation == "relocate":
        node_id = _resolve_disturbance_node_id(backend, disturbance["node_id"], "node_id")
        state = world.states[node_id]
        previous_location = {
            "relation": state.location_relation,
            "target": state.location_target,
            "held": state.held,
        }
        relation = disturbance.get("relation")
        target_ref = disturbance.get("target")
        target_id: str | None = None
        if relation is not None:
            relation = normalize_relation(str(relation))
            relation = "INSIDE" if relation == "IN" else relation
            if relation not in {"ON", "INSIDE", "BENEATH"}:
                raise ValueError(
                    "graph disturbance relocate relation must be ON, INSIDE/IN, BENEATH, or null"
                )
            target_id = _resolve_disturbance_node_id(backend, target_ref, "target")
            if target_id == node_id:
                raise ValueError("graph disturbance cannot relocate a node onto/inside itself")
            target = world.states[target_id]
            if relation == "ON" and not target.node.is_surface:
                raise ValueError(f"graph disturbance ON target is not a surface: {target_id}")
            if relation == "INSIDE" and not target.node.is_container:
                raise ValueError(f"graph disturbance INSIDE target is not a container: {target_id}")
        world._remove_occlusions_for_location_change(node_id)
        # An external relocation invalidates the object's old memory-only hiding
        # anchor. Visibility at the new location is still governed by closed
        # containers and active occlusion edges below.
        world.memory_hidden.pop(node_id, None)
        state.held = False
        state.location_relation = relation
        state.location_target = target_id
        return {
            "node_id": node_id,
            "previous_location": previous_location,
            "new_location": {
                "relation": state.location_relation,
                "target": state.location_target,
                "held": state.held,
            },
        }

    if operation == "set_capacity":
        node_id = _resolve_disturbance_node_id(backend, disturbance["node_id"], "node_id")
        state = world.states[node_id]
        if not state.node.is_container:
            raise ValueError(f"graph disturbance capacity target is not a container: {node_id}")
        previous_max_items = state.node.max_items
        state.node.metadata["max_items"] = int(disturbance["max_items"])
        return {
            "node_id": node_id,
            "previous_max_items": previous_max_items,
            "new_max_items": state.node.max_items,
        }

    source_id = _resolve_disturbance_node_id(backend, disturbance["source"], "source")
    target_id = _resolve_disturbance_node_id(backend, disturbance["target"], "target")
    relation_value = disturbance.get("relation")
    relation = normalize_relation(str(relation_value or "OCCLUDES"))
    if relation not in OCCLUSION_RELATIONS:
        raise ValueError(
            f"graph disturbance occlusion relation must be one of {sorted(OCCLUSION_RELATIONS)}"
        )

    if operation == "add_occlusion":
        resolution_action = str(
            disturbance.get("resolution_action")
            or world._default_occlusion_resolution_action(source_id)
        ).strip().lower()
        if resolution_action not in {"open", "move_aside"}:
            raise ValueError("graph disturbance resolution_action must be open or move_aside")
        source_state = world.states[source_id]
        if resolution_action == "open" and not source_state.node.is_openable:
            raise ValueError(f"graph disturbance open occluder is not openable: {source_id}")
        if resolution_action == "move_aside" and not source_state.node.is_movable:
            raise ValueError(f"graph disturbance move_aside occluder is not movable: {source_id}")
        edge = (source_id, target_id, relation)
        already_active = edge in world.active_occlusion_edges
        world.active_occlusion_edges.add(edge)
        world.occlusion_edge_resolution_actions[edge] = resolution_action
        if not any(
            item.source == source_id and item.target == target_id and item.relation == relation
            for item in world.graph.edges
        ):
            world.graph.edges.append(
                Edge(
                    source=source_id,
                    target=target_id,
                    relation=relation,
                    metadata={"runtime_disturbance": True},
                )
            )
        activate = disturbance.get("activate", True)
        if not isinstance(activate, bool):
            raise ValueError("graph disturbance add_occlusion activate must be boolean")
        if activate and resolution_action == "open":
            source_state.open = False
        if activate and resolution_action == "move_aside":
            source_state.moved_aside = False
        return {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "resolution_action": resolution_action,
            "already_active": already_active,
            "active_after": world._occlusion_edge_active(edge),
        }

    matching_edges = {
        edge
        for edge in world.active_occlusion_edges
        if edge[0] == source_id
        and edge[1] == target_id
        and (relation_value is None or edge[2] == relation)
    }
    if not matching_edges:
        raise ValueError(
            f"graph disturbance found no active occlusion from {source_id} to {target_id}"
        )
    world.active_occlusion_edges.difference_update(matching_edges)
    for edge in matching_edges:
        world.occlusion_edge_resolution_actions.pop(edge, None)
    return {
        "source": source_id,
        "target": target_id,
        "removed_edges": [
            {"source": edge[0], "target": edge[1], "relation": edge[2]}
            for edge in sorted(matching_edges)
        ],
    }


def _expected_recovery_from_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    default = {"required": False, "failed_action": None, "failed_node_ids": []}
    if not history:
        return default
    previous = history[-1]
    event = previous.get("event")
    action = previous.get("action")
    if (
        not isinstance(event, dict)
        or event.get("status") != "failure"
        or event.get("attempted") is not True
        or not isinstance(action, dict)
    ):
        return default
    name = str(action.get("base_name") or action.get("name") or "").removeprefix("failed_")
    if not name or name == "stop":
        return default
    return {
        "required": True,
        "failed_action": name,
        "failed_node_ids": [str(item) for item in action.get("node_ids", [])],
    }


def _teacher_non_stop_step_count(episode: dict[str, Any]) -> int | None:
    if episode.get("teacher_reference_available") is False:
        return None
    count = 0
    for step in episode.get("trajectory", []) or []:
        if not isinstance(step, dict):
            continue
        action = step.get("requested_action") if isinstance(step.get("requested_action"), dict) else step.get("action")
        action = action if isinstance(action, dict) else {}
        name = str(action.get("base_name") or action.get("name") or step.get("manual_name") or "")
        name = name.lower().removeprefix("failed_")
        if name and name not in {"stop", "recover"}:
            count += 1
    return count


def _episode_outcome(
    *,
    backend: SymbolicBackend,
    records: list[dict[str, Any]],
    termination_reason: str,
    first_goal_satisfied_step: int | None,
    goal_damaged_after_satisfaction: bool,
    initial_cost: float,
    initial_planning_cost: float,
    teacher_step_count: int | None,
    injected_failure_count: int,
    disturbance_count: int,
    disturbance_cleanup_count: int,
    condition_id: str | None,
    intervention_type: str | None,
    added_object_goal_expression: Any | None,
    added_object_goal_activation_step: int | None,
    first_added_object_goal_satisfied_step: int | None,
    added_object_goal_model_steps_to_satisfaction: int | None,
) -> dict[str, Any]:
    final_goal_satisfied = bool(_safe_backend_success(backend))
    final_cost = _relaxed_completion_cost(backend)
    final_planning_cost = _planning_completion_cost(backend)
    normalized_goal_progress = 1.0 if initial_cost <= 0 else max(0.0, min(1.0, (initial_cost - final_cost) / initial_cost))
    success = termination_reason == "correct_stop" and final_goal_satisfied
    model_non_stop_steps = sum(
        1
        for record in records
        if str((record.get("predicted_action") or {}).get("base_name") or (record.get("predicted_action") or {}).get("name") or "") != "stop"
    )
    efficiency = (
        min(1.0, teacher_step_count / max(1, model_non_stop_steps))
        if success and teacher_step_count is not None
        else (0.0 if teacher_step_count is not None else None)
    )
    intervention_count = injected_failure_count + disturbance_count
    intervention_steps = [
        int(record.get("step") or 0)
        for record in records
        if record.get("injection_applied") is True
        or any(
            report.get("counts_as_intervention", True) is True
            for report in record.get("disturbances_applied") or []
            if isinstance(report, dict)
        )
    ]
    added_object_goal_eligible = added_object_goal_expression is not None
    final_added_object_goal_satisfied = _goal_expression_satisfied(
        backend,
        added_object_goal_expression,
    )
    return {
        "condition_id": condition_id,
        "intervention_type": intervention_type,
        "success": success,
        "termination_reason": termination_reason,
        "step_count": len(records),
        "model_non_stop_step_count": model_non_stop_steps,
        "teacher_non_stop_step_count": teacher_step_count,
        "teacher_normalized_efficiency": efficiency,
        "goal_ever_satisfied": first_goal_satisfied_step is not None,
        "first_goal_satisfied_step": first_goal_satisfied_step,
        "final_goal_satisfied": final_goal_satisfied,
        "goal_damaged_after_satisfaction": goal_damaged_after_satisfaction,
        "initial_relaxed_completion_cost": initial_cost,
        "final_relaxed_completion_cost": final_cost,
        "initial_goal_completion_cost": initial_cost,
        "final_goal_completion_cost": final_cost,
        "initial_planning_cost": initial_planning_cost,
        "final_planning_cost": final_planning_cost,
        "normalized_goal_progress": normalized_goal_progress,
        "premature_stop": termination_reason == "premature_stop",
        "completion_stop_success": success,
        "failure_injected": injected_failure_count > 0,
        "injected_failure_count": injected_failure_count,
        "disturbance_applied": disturbance_count > 0,
        "disturbance_count": disturbance_count,
        "disturbance_cleanup_count": disturbance_cleanup_count,
        "intervention_applied": intervention_count > 0,
        "intervention_count": intervention_count,
        "first_intervention_step": min(intervention_steps) if intervention_steps else None,
        "added_object_goal": {
            "eligible": added_object_goal_eligible,
            "expression": copy.deepcopy(added_object_goal_expression),
            "activation_step": added_object_goal_activation_step,
            "ever_satisfied": (
                first_added_object_goal_satisfied_step is not None
                if added_object_goal_eligible
                else None
            ),
            "first_satisfied_step": first_added_object_goal_satisfied_step,
            "final_satisfied": final_added_object_goal_satisfied,
            "model_steps_to_satisfaction": (
                added_object_goal_model_steps_to_satisfaction
                if added_object_goal_eligible
                else None
            ),
        },
    }


def _closed_loop_summary(
    records: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    config: ViewGraphRolloutEvalConfig,
    *,
    intervention_condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    capabilities = _aggregate_capability_scores(records)
    capabilities["completion_judgment"] = {
        "premature_stop_rate": mean([float(item["premature_stop"]) for item in outcomes]),
        "completion_stop_recall": mean(
            [float(item["completion_stop_success"]) for item in outcomes if item["goal_ever_satisfied"]]
        ),
    }
    normal_executability = [
        float(record["normally_executable"])
        for record in records
        if record.get("normally_executable") is not None
    ]
    added_object_goal_outcomes = [
        item["added_object_goal"]
        for item in outcomes
        if isinstance(item.get("added_object_goal"), dict)
        and item["added_object_goal"].get("eligible") is True
    ]
    outcome_metrics = {
        "episode_count": len(outcomes),
        "task_success_rate": mean([float(item["success"]) for item in outcomes]),
        "goal_ever_satisfied_rate": mean([float(item["goal_ever_satisfied"]) for item in outcomes]),
        "final_goal_satisfied_rate": mean([float(item["final_goal_satisfied"]) for item in outcomes]),
        "normalized_goal_progress": mean([float(item["normalized_goal_progress"]) for item in outcomes]),
        "teacher_normalized_efficiency": mean(
            [
                float(item["teacher_normalized_efficiency"])
                for item in outcomes
                if item.get("teacher_normalized_efficiency") is not None
            ]
        ),
        "average_step_count": mean([float(item["step_count"]) for item in outcomes]),
        "action_executability_rate": mean(normal_executability),
        "episodes_with_injected_failure_rate": mean(
            [float(item["failure_injected"]) for item in outcomes]
        ),
        "average_injected_failure_count": mean(
            [float(item["injected_failure_count"]) for item in outcomes]
        ),
        "episodes_with_disturbance_rate": mean(
            [float(item["disturbance_applied"]) for item in outcomes]
        ),
        "average_disturbance_count": mean(
            [float(item["disturbance_count"]) for item in outcomes]
        ),
        "intervention_applied_rate": mean(
            [float(item["intervention_applied"]) for item in outcomes]
        ),
        "average_intervention_count": mean(
            [float(item["intervention_count"]) for item in outcomes]
        ),
        "added_object_goal_eligible_episode_count": len(
            added_object_goal_outcomes
        ),
        "added_object_goal_ever_satisfied_rate": mean(
            [
                float(item["ever_satisfied"])
                for item in added_object_goal_outcomes
            ]
        ),
        "added_object_goal_final_satisfied_rate": mean(
            [
                float(item["final_satisfied"])
                for item in added_object_goal_outcomes
            ]
        ),
        "average_added_object_goal_model_steps_to_satisfaction": mean(
            [
                float(item["model_steps_to_satisfaction"])
                for item in added_object_goal_outcomes
                if item.get("model_steps_to_satisfaction") is not None
            ]
        ),
        "parse_success_rate": mean(
            [
                float(
                    record.get("parse_error") is None
                    and record.get("model_error") is None
                    and isinstance(record.get("predicted_action"), dict)
                )
                for record in records
            ]
        ),
        "model_error_rate": mean([float(record.get("model_error") is not None) for record in records]),
        "termination_reasons": {
            reason: sum(item["termination_reason"] == reason for item in outcomes)
            for reason in ("correct_stop", "premature_stop", "max_steps", "model_error_limit")
        },
    }
    by_mode = {
        CLOSED_LOOP_MODE: {
            **outcome_metrics,
            **{
                metric: value
                for dimension in (
                    "action_selection",
                    "failure_recovery",
                    "active_exploration",
                    "completion_judgment",
                )
                for metric, value in capabilities[dimension].items()
            },
            "outcomes": outcome_metrics,
            "capabilities": capabilities,
        }
    }
    return {
        "evaluation_type": CLOSED_LOOP_EVALUATION_TYPE,
        "closed_loop_metric_version": CLOSED_LOOP_METRIC_VERSION,
        "capability_metric_version": CAPABILITY_METRIC_VERSION,
        "cost_semantics": {
            "goal_completion_cost": "direct_goal_fact_deficit_without_access_prerequisites",
            "planning_cost": (
                "goal_actions_plus_deduplicated_open_move_aside_and_access_prerequisites"
            ),
            "soft_optimal_cost": "planning_cost",
            "normalized_goal_progress_cost": "goal_completion_cost",
        },
        "counterfactual_scope": "one_step_from_closed_loop_state",
        "model_name": _rollout_model_name(config),
        "condition_id": (
            str(intervention_condition.get("condition_id"))
            if isinstance(intervention_condition, dict)
            else None
        ),
        "intervention_type": (
            str(intervention_condition.get("intervention_type"))
            if isinstance(intervention_condition, dict)
            else None
        ),
        "intervention_condition": copy.deepcopy(intervention_condition),
        "provider": config.provider,
        "api_style": _resolved_api_style(config.provider, config.api_style),
        "modes": [CLOSED_LOOP_MODE],
        "history_source": "inference",
        "history_window": config.history_window,
        "includes_valid_actions": config.include_valid_actions,
        "failure_injection": config.failure_injection,
        "failure_injection_config": {
            "mode": config.failure_injection,
            "actions": list(config.failure_actions),
            "probability": config.failure_probability,
            "max_failures_per_episode": config.max_failures_per_episode,
            "seed": config.failure_seed,
            "deduplication_scope": config.failure_deduplication_scope,
            "only_normally_successful_actions": True,
            "unique_by_configured_scope": True,
        },
        "graph_disturbance_file": config.graph_disturbance_file,
        "graph_disturbance_count": len(load_graph_disturbances(config.graph_disturbance_file)),
        "graph_disturbance_timing": "before_current_step_observation",
        "max_steps": config.max_steps,
        "soft_optimal_beta": config.soft_optimal_beta,
        "record_count": len(records),
        "episode_count": len(outcomes),
        "outcomes": outcome_metrics,
        "capabilities": {CLOSED_LOOP_MODE: capabilities},
        "by_mode": by_mode,
    }


def _action_key_payload(key: tuple[str, tuple[str, ...]] | None) -> dict[str, Any] | None:
    if key is None:
        return None
    return {"name": key[0], "node_ids": list(key[1])}


def _rollout_model_name(config: ViewGraphRolloutEvalConfig) -> str:
    if config.model_name and config.model_name.strip():
        return config.model_name.strip()
    suffix = "valid_action" if config.include_valid_actions else "no_valid_action"
    return f"{config.model or _default_model(config.provider)}_{suffix}"


def _default_model(provider: str) -> str:
    if provider == "qwen":
        return "qwen-vl-plus"
    if provider == "mr_openai":
        return "gpt-5.5"
    if provider == "mr_anthropic":
        return "claude-opus-4-7"
    if provider == "mr_google":
        return "gemini-3.1-pro-preview"
    return "gpt-4o-mini"
