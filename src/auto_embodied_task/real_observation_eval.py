from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

from . import real_alignment as _alignment
from .brain import BrainHarness, BrainPolicy, BrainPolicyConfig, BrainRequest
from .harness import (
    TEACHER_SYSTEM_PROMPT,
    ParsedAction,
    SymbolicBackend,
    TeacherDecision,
    _new_visible_nodes,
    _teacher_action_catalog,
    _timestamped_output_path,
    _valid_teacher_actions,
    parse_teacher_decision,
)
from .manual_actions import parse_manual_action
from .models import TaskRecord, ViewGraph
from .placement_constraints import PlacementEdgeConstraints


EVAL_MODES = ("obs_only", "graph_only", "obs_plus_graph", "wrong_graph_plus_obs")
IMAGE_MODES = {"obs_only", "obs_plus_graph", "wrong_graph_plus_obs"}
GRAPH_MODES = {"graph_only", "obs_plus_graph", "wrong_graph_plus_obs"}
HISTORY_SOURCES = ("teacher", "inference")
OPEN_ACTION_SYSTEM_PROMPT = """You are a policy for high-level embodied task evaluation.
Choose exactly one semantic action using the action catalog and the provided observation.
Output strict JSON only."""


@dataclass(frozen=True)
class RealObservationEvalConfig:
    provider: str = "qwen"
    model: str | None = None
    model_name: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    api_style: str = "auto"
    temperature: float = 0.0
    max_output_tokens: int = 2048
    timeout_seconds: int = 120
    modes: tuple[str, ...] = EVAL_MODES
    history_source: str = "teacher"
    include_valid_actions: bool = True
    max_api_attempts: int = 1
    retry_backoff_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    frame_count: int = 2
    cameras: tuple[str, ...] = _alignment.CAMERA_KEYS
    observation_window_seconds: float = 0.5
    frame_sampling: str = "head"
    max_steps: int | None = None
    dry_run: bool = False
    fail_fast: bool = False
    oss_region: str = "cn-shanghai"
    oss_endpoint: str | None = None
    cache_dir: str | Path | None = None

    def __post_init__(self) -> None:
        providers = {
            "openai",
            "qwen",
            "compatible",
            "mr_openai",
            "mr_anthropic",
            "mr_google",
        }
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
        unknown = [mode for mode in self.modes if mode not in EVAL_MODES]
        if unknown:
            raise ValueError(f"unknown eval modes: {unknown}")
        if self.history_source not in HISTORY_SOURCES:
            raise ValueError(f"history_source must be one of {HISTORY_SOURCES}")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.observation_window_seconds <= 0:
            raise ValueError("observation_window_seconds must be positive")
        if self.max_api_attempts <= 0:
            raise ValueError("max_api_attempts must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.frame_sampling not in {"head", "previous_tail"}:
            raise ValueError("frame_sampling must be head or previous_tail")


@dataclass(frozen=True)
class ReplayStepContext:
    step_index: int
    step: dict[str, Any]
    graph_observation: dict[str, Any]
    valid_actions: list[dict[str, Any]]
    history: list[dict[str, Any]]
    expected_action: dict[str, Any]
    expected_recovery: dict[str, Any]
    generated_valid_actions: list[dict[str, Any]] = field(default_factory=list)
    valid_actions_added: list[dict[str, Any]] = field(default_factory=list)


class RealObservationAdapter:
    def build_request(
        self,
        *,
        task: TaskRecord,
        step: dict[str, Any],
        history: list[dict[str, Any]],
        valid_actions: list[dict[str, Any]],
        graph_observation: dict[str, Any],
        wrong_graph_observation: dict[str, Any] | None,
        frame_files: list[str],
        mode: str,
        history_source: str = "teacher",
        frame_observation: dict[str, Any] | None = None,
        include_valid_actions: bool = True,
    ) -> BrainRequest:
        if include_valid_actions:
            action_constraints = [
                "Choose exactly one action from valid_actions.",
                "Copy its name, object, target, and node_ids without inventing nodes.",
            ]
            action_schema = {
                "name": "copied from one valid_actions item",
                "object": "copied when present",
                "target": "copied when present",
                "node_ids": "copied from the selected valid_actions item",
            }
        else:
            action_constraints = [
                "Choose exactly one action using action_catalog and the provided observation.",
                "Ground object, target, and node_ids in the visual observation and task context.",
                "Do not assume that a candidate action list is available.",
            ]
            action_schema = {
                "name": "one semantic action name from action_catalog",
                "object": "observed object identifier when applicable",
                "target": "observed target identifier when applicable",
                "node_ids": "ordered observed object and target identifiers when applicable",
            }
        action_constraints.extend(
            [
                "Treat recover as an internal transition, not as a second physical action.",
                "The current observation was captured after the previous task action and before the current action.",
                "Use the current observation to assess the most recent task action in recent_history; do not use that decision's prior recovery value as its outcome.",
                "Set recovery.required=true when the current observation shows that the most recent task action failed and the selected current action corrects that failure's consequences or retries the failed action.",
                "Set recovery.required=false when the current observation shows that it succeeded.",
                "Use real images whenever images are present.",
            ]
        )
        payload: dict[str, Any] = {
            "task": task.task,
            "task_type": task.task_type,
            "settings": task.settings,
            "success_criterion": task.task_completion_criterion,
            "step": step.get("step"),
            "action_catalog": _teacher_action_catalog(),
            "action_constraints": action_constraints,
            "observation_timing": {
                "current_observation": "after the previous task action, before the selected current action",
                "recovery_assessment": "judge the previous task action from the current observation",
            },
            "recent_history": history[-8:],
            "output_schema": {
                "reason": "short rationale that assesses the previous action outcome from the current observation, then explains the selected current action",
                "recovery": {
                    "required": "true when the current observation shows the previous task action failed and this action corrects its consequences or retries it",
                    "failed_action": "previous task action name when required, otherwise null",
                },
                "action": action_schema,
            },
        }
        if include_valid_actions:
            payload["valid_actions"] = valid_actions
        if mode in {"graph_only", "obs_plus_graph"}:
            payload["current_observation"] = graph_observation
        elif mode == "wrong_graph_plus_obs":
            payload["current_observation"] = wrong_graph_observation
            payload["graph_warning"] = "The graph observation may conflict with the real images."

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if mode in IMAGE_MODES:
            content.extend(
                {"type": "image_url", "image_url": {"url": _image_data_url(Path(frame_file))}}
                for frame_file in frame_files
            )
        return BrainRequest(
            messages=[
                {
                    "role": "system",
                    "content": TEACHER_SYSTEM_PROMPT if include_valid_actions else OPEN_ACTION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": content},
            ],
            summary={
                "adapter": "real",
                "mode": mode,
                "frame_count": len(frame_files) if mode in IMAGE_MODES else 0,
                "frame_files": frame_files if mode in IMAGE_MODES else [],
                "camera_count": (
                    len({Path(path).parent.name for path in frame_files}) if mode in IMAGE_MODES else 0
                ),
                "has_view_graph": mode in GRAPH_MODES,
                "includes_valid_actions": include_valid_actions,
                "valid_action_count": len(valid_actions) if include_valid_actions else 0,
                "history_source": history_source,
                "history_count": len(history),
                "frame_observation": frame_observation or {},
            },
        )

    def parse_response(self, text: str) -> TeacherDecision:
        return parse_teacher_decision(text)


class RealTrajectoryHarness:
    def __init__(
        self,
        *,
        config: RealObservationEvalConfig,
        brain_harness: BrainHarness | None,
        oss_client: _alignment.OssClientProtocol,
    ) -> None:
        self.config = config
        self.brain_harness = brain_harness
        self.adapter = RealObservationAdapter()
        self.oss_client = oss_client

    def run_episode(
        self,
        episode: dict[str, Any],
        source_file: Path,
        record_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        task, graph, constraints = _replay_inputs(episode)
        contexts = _build_replay_contexts(episode, task, graph, constraints)
        evaluable_contexts = _evaluable_contexts(
            contexts,
            frame_sampling=self.config.frame_sampling,
        )
        if self.config.max_steps is not None:
            evaluable_contexts = evaluable_contexts[: self.config.max_steps]

        records: list[dict[str, Any]] = []
        inference_histories: dict[str, list[dict[str, Any]]] = {
            mode: [] for mode in self.config.modes
        }
        context_positions = {id(context): index for index, context in enumerate(contexts)}
        for context in evaluable_contexts:
            frames = []
            frame_observation: dict[str, Any] = {}
            if any(mode in IMAGE_MODES for mode in self.config.modes):
                frames, frame_observation = _frames_for_context(
                    contexts,
                    context_positions[id(context)],
                    config=self.config,
                    oss_client=self.oss_client,
                )
            wrong_graph = _wrong_replay_observation(contexts, context.step_index)
            for mode in self.config.modes:
                input_history = (
                    context.history
                    if self.config.history_source == "teacher"
                    else inference_histories[mode]
                )
                request = self.adapter.build_request(
                    task=task,
                    step=context.step,
                    history=input_history,
                    valid_actions=context.valid_actions,
                    graph_observation=context.graph_observation,
                    wrong_graph_observation=wrong_graph,
                    frame_files=frames,
                    mode=mode,
                    history_source=self.config.history_source,
                    frame_observation=frame_observation,
                    include_valid_actions=self.config.include_valid_actions,
                )
                record = {
                    "source_file": str(source_file),
                    "model_name": _evaluation_model_name(self.config),
                    "episode_id": episode.get("episode_id"),
                    "step": context.step.get("step", context.step_index + 1),
                    "mode": mode,
                    "expected_action": context.expected_action,
                    "expected_recovery": context.expected_recovery,
                    "real_reply": context.step.get("real_reply") or {},
                    "valid_actions": context.valid_actions,
                    "generated_valid_actions": context.generated_valid_actions,
                    "valid_actions_added": context.valid_actions_added,
                    "history_source": self.config.history_source,
                    "input_history": json.loads(json.dumps(input_history, ensure_ascii=False)),
                    "frame_observation": frame_observation,
                    "request_summary": request.summary,
                    "raw_response": None,
                    "parsed_response": None,
                    "predicted_action": None,
                    "predicted_recovery": None,
                    "reason": None,
                    "parse_error": None,
                    "model_error": None,
                    "score": None,
                    "recovery_score": None,
                }
                if not self.config.dry_run:
                    try:
                        if self.brain_harness is None:
                            raise RuntimeError("brain harness is not configured")
                        decision = self.brain_harness.decide_request(request)
                        predicted = _structured_action(decision.action)
                        predicted_recovery = _predicted_recovery(decision.parsed_response)
                        record.update(
                            {
                                "raw_response": decision.raw_response,
                                "parsed_response": decision.parsed_response,
                                "predicted_action": predicted,
                                "predicted_recovery": predicted_recovery,
                                "reason": decision.reason,
                                "parse_error": decision.parse_error,
                                "score": _score_action(context.expected_action, predicted, decision.parse_error),
                                "recovery_score": _score_recovery(
                                    context.expected_recovery,
                                    predicted_recovery,
                                    decision.parse_error,
                                ),
                            }
                        )
                        if self.config.history_source == "inference":
                            inference_histories[mode].append(
                                _inference_history_item(
                                    step=context.step.get("step", context.step_index + 1),
                                    decision=decision,
                                    predicted_action=predicted,
                                )
                            )
                    except Exception as exc:
                        if self.config.fail_fast:
                            raise
                        record["model_error"] = str(exc)
                        record["score"] = _score_action(
                            context.expected_action,
                            {},
                            f"model request failed: {exc}",
                        )
                        record["recovery_score"] = _score_recovery(
                            context.expected_recovery,
                            None,
                            f"model request failed: {exc}",
                        )
                records.append(record)
                if record_callback is not None:
                    record_callback(record)
        return records, len(evaluable_contexts)


def evaluate_real_trajectories(
    *,
    input_path: str | Path,
    output_path: str | Path,
    config: RealObservationEvalConfig,
) -> dict[str, Any]:
    source = Path(input_path)
    target = _timestamped_output_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    adapter = RealObservationAdapter()
    brain_harness = None
    if not config.dry_run:
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
            )
        )
        brain_harness = BrainHarness(policy, adapter)
    oss_client = _alignment.OssUtilClient(region=config.oss_region, endpoint=config.oss_endpoint)
    harness = RealTrajectoryHarness(config=config, brain_harness=brain_harness, oss_client=oss_client)

    records: list[dict[str, Any]] = []
    logical_steps = 0
    completed_records = 0
    with target.open("w", encoding="utf-8") as out:
        def write_record(record: dict[str, Any]) -> None:
            nonlocal completed_records
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            completed_records += 1
            print(
                f"Completed record {completed_records}: step={record['step']} mode={record['mode']}",
                flush=True,
            )

        with source.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                episode = json.loads(stripped)
                if not isinstance(episode, dict):
                    raise ValueError(f"{source}:{line_no}: expected JSON object")
                episode_records, episode_steps = harness.run_episode(
                    episode,
                    source,
                    record_callback=write_record,
                )
                records.extend(episode_records)
                logical_steps += episode_steps
    summary = _evaluation_summary(records, config=config, logical_steps=logical_steps)
    summary_path = target.with_name(f"{target.stem}__summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "count": len(records),
        "logical_step_count": logical_steps,
        "output_path": str(target),
        "summary_path": str(summary_path),
    }


def _replay_inputs(
    episode: dict[str, Any],
) -> tuple[TaskRecord, ViewGraph, PlacementEdgeConstraints]:
    graph_payload = episode.get("initial_view_graph")
    if not isinstance(graph_payload, dict):
        raise ValueError("aligned episode is missing initial_view_graph")
    graph = ViewGraph.from_dict(graph_payload, fallback_scene_id=str(episode.get("scene_id") or "scene"))
    robot = graph_payload.get("robot") if isinstance(graph_payload.get("robot"), dict) else {}
    task = TaskRecord(
        task_id=str(episode.get("episode_id") or "real_episode"),
        scene_id=str(episode.get("scene_id") or graph_payload.get("scene_id") or "scene"),
        env_id=episode.get("env_id", graph_payload.get("env_id", "real")),
        layout=str(graph_payload.get("layout") or "tabletop"),
        arms=str(robot.get("arms") or "double"),
        task_type=str(episode.get("task_type") or "manipulation"),
        task=str(episode.get("task") or ""),
        task_completion_criterion=episode.get("task_completion_criterion"),
        ground_truth_plan=[],
        objects={},
        settings=list(episode.get("settings") or []),
        metadata={"source": "real_aligned_trajectory"},
    )
    constraints = PlacementEdgeConstraints.from_json(episode.get("placement_edge_constraints"))
    return task, graph, constraints


def _build_replay_contexts(
    episode: dict[str, Any],
    task: TaskRecord,
    graph: ViewGraph,
    constraints: PlacementEdgeConstraints,
) -> list[ReplayStepContext]:
    backend = SymbolicBackend(graph, task, constraints)
    history: list[dict[str, Any]] = []
    contexts: list[ReplayStepContext] = []
    steps = [step for step in episode.get("trajectory", []) or [] if isinstance(step, dict)]
    for step_index, step in enumerate(steps):
        observation = backend.observe()
        expected_action = _expected_action(step)
        expected_recovery = _expected_recovery(history, expected_action)
        generated_valid_actions = _valid_teacher_actions(observation, history)
        valid_actions_added = _manual_valid_action_additions(
            step,
            expected_action,
            generated_valid_actions,
        )
        valid_actions = [*generated_valid_actions, *valid_actions_added]
        contexts.append(
            ReplayStepContext(
                step_index=step_index,
                step=step,
                graph_observation=observation,
                valid_actions=valid_actions,
                history=json.loads(json.dumps(history, ensure_ascii=False)),
                expected_action=expected_action,
                expected_recovery=expected_recovery,
                generated_valid_actions=generated_valid_actions,
                valid_actions_added=valid_actions_added,
            )
        )
        executed_action = _executed_action(step)
        replay_event = backend.step(_parsed_action(executed_action))
        post_observation = backend.observe()
        history.append(
            {
                "step": step.get("step", step_index + 1),
                "action": executed_action,
                "requested_action": expected_action,
                "event": replay_event,
                "new_visible_nodes": _new_visible_nodes(observation, post_observation),
                "success_after_step": backend.success(),
            }
        )
    return contexts


def _manual_valid_action_additions(
    step: dict[str, Any],
    expected_action: dict[str, Any],
    generated_valid_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if step.get("manual_inserted") is not True:
        return []
    candidate = {
        "name": str(expected_action.get("base_name") or expected_action.get("name") or ""),
        "node_ids": list(expected_action.get("node_ids") or []),
    }
    if not candidate["name"]:
        return []
    if candidate["node_ids"]:
        candidate["object"] = candidate["node_ids"][0]
    if len(candidate["node_ids"]) > 1:
        candidate["target"] = candidate["node_ids"][1]
    key = (candidate["name"], tuple(candidate["node_ids"]))
    generated_keys = {
        (str(action.get("name") or ""), tuple(action.get("node_ids") or []))
        for action in generated_valid_actions
        if isinstance(action, dict)
    }
    return [] if key in generated_keys else [candidate]


def _expected_action(step: dict[str, Any]) -> dict[str, Any]:
    action = step.get("requested_action") if isinstance(step.get("requested_action"), dict) else step.get("action")
    normalized = _normalize_action(action, manual_name=step.get("manual_name"))
    normalized["name"] = normalized["base_name"]
    return normalized


def _executed_action(step: dict[str, Any]) -> dict[str, Any]:
    return _normalize_action(step.get("action"), manual_name=step.get("manual_name"))


def _expected_recovery(
    history: list[dict[str, Any]],
    expected_action: dict[str, Any],
) -> dict[str, Any]:
    default = {"required": False, "failed_action": None}
    if len(history) < 2:
        return default
    recovery_item = history[-1]
    failed_item = history[-2]
    recovery_action = recovery_item.get("action") if isinstance(recovery_item, dict) else None
    recovery_event = recovery_item.get("event") if isinstance(recovery_item, dict) else None
    failed_event = failed_item.get("event") if isinstance(failed_item, dict) else None
    failed_action = failed_item.get("requested_action") if isinstance(failed_item, dict) else None
    if not (
        isinstance(recovery_action, dict)
        and recovery_action.get("base_name") == "recover"
        and isinstance(recovery_event, dict)
        and recovery_event.get("status") == "success"
        and isinstance(failed_event, dict)
        and failed_event.get("status") == "failure"
        and isinstance(failed_action, dict)
    ):
        return default
    failed_name = str(failed_action.get("base_name") or failed_action.get("name") or "").removeprefix("failed_")
    current_name = str(expected_action.get("base_name") or expected_action.get("name") or "")
    if not failed_name or current_name in {"", "recover", "stop"}:
        return default
    return {
        "required": True,
        "failed_action": failed_name,
    }


def _normalize_action(action: Any, *, manual_name: Any = None) -> dict[str, Any]:
    payload = action if isinstance(action, dict) else {}
    name = str(payload.get("name") or payload.get("base_name") or manual_name or "").strip()
    if payload.get("base_name") == "manual_inserted" or manual_name:
        manual_action = parse_manual_action(str(manual_name or name))
        name = str(manual_action["name"])
        node_ids = list(manual_action["node_ids"])
    else:
        node_ids = [str(value).strip() for value in payload.get("node_ids", []) if str(value).strip()]
    if not name:
        raise ValueError("trajectory action is missing name")
    normalized = {
        "name": name.lower(),
        "base_name": name.lower().removeprefix("failed_"),
        "node_ids": node_ids,
    }
    if node_ids:
        normalized["object"] = node_ids[0]
    if len(node_ids) > 1:
        normalized["target"] = node_ids[1]
    arguments = payload.get("arguments")
    if isinstance(arguments, dict) and arguments:
        normalized["arguments"] = arguments
    return normalized


def _parsed_action(action: dict[str, Any]) -> ParsedAction:
    return ParsedAction(
        name=str(action["name"]),
        node_ids=list(action.get("node_ids") or []),
        arguments=dict(action.get("arguments") or {}),
        raw=json.dumps(action, ensure_ascii=False),
    )


def _structured_action(action: ParsedAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": action.name,
        "base_name": action.base_name,
        "node_ids": list(action.node_ids),
    }
    if action.node_ids:
        payload["object"] = action.node_ids[0]
    if len(action.node_ids) > 1:
        payload["target"] = action.node_ids[1]
    return payload


def _inference_history_item(
    *,
    step: Any,
    decision: TeacherDecision,
    predicted_action: dict[str, Any],
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "step": step,
        "action": predicted_action,
    }
    if decision.parse_error is not None:
        item["parse_error"] = decision.parse_error
    return item


def _predicted_recovery(parsed_response: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(parsed_response, dict) or "recovery" not in parsed_response:
        return None
    raw = parsed_response.get("recovery")
    if isinstance(raw, bool):
        return {"required": raw, "failed_action": None}
    if not isinstance(raw, dict) or not isinstance(raw.get("required"), bool):
        return None
    failed_action = raw.get("failed_action")
    if isinstance(failed_action, dict):
        failed_action = failed_action.get("base_name") or failed_action.get("name")
    normalized_failed_action = (
        str(failed_action).strip().lower().removeprefix("failed_")
        if failed_action is not None and str(failed_action).strip()
        else None
    )
    return {
        "required": raw["required"],
        "failed_action": normalized_failed_action,
    }


def _score_action(
    expected: dict[str, Any],
    predicted: dict[str, Any],
    parse_error: str | None,
) -> dict[str, Any]:
    parsed = parse_error is None and predicted.get("name") != "invalid_teacher_action"
    fields = {
        "name": parsed and predicted.get("base_name") == expected.get("base_name"),
        "object": parsed and predicted.get("object") == expected.get("object"),
        "target": parsed and predicted.get("target") == expected.get("target"),
        "node_ids": parsed and predicted.get("node_ids", []) == expected.get("node_ids", []),
    }
    return {
        "parsed": parsed,
        **fields,
        "full_exact": all(fields.values()),
        "object_applicable": expected.get("object") is not None,
        "target_applicable": expected.get("target") is not None,
    }


def _score_recovery(
    expected: dict[str, Any],
    predicted: dict[str, Any] | None,
    parse_error: str | None,
) -> dict[str, Any]:
    parsed = parse_error is None and predicted is not None
    required_match = parsed and predicted.get("required") == expected.get("required")
    failed_action_applicable = bool(expected.get("required"))
    failed_action_match = (
        parsed and predicted.get("failed_action") == expected.get("failed_action")
        if failed_action_applicable
        else True
    )
    return {
        "parsed": parsed,
        "required": required_match,
        "failed_action": failed_action_match,
        "full_exact": bool(required_match and failed_action_match),
        "failed_action_applicable": failed_action_applicable,
    }


def _evaluation_summary(
    records: list[dict[str, Any]],
    *,
    config: RealObservationEvalConfig,
    logical_steps: int,
) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for mode in config.modes:
        mode_records = [record for record in records if record.get("mode") == mode]
        scored = [record["score"] for record in mode_records if isinstance(record.get("score"), dict)]
        recovery_scored = [
            record["recovery_score"]
            for record in mode_records
            if isinstance(record.get("recovery_score"), dict)
        ]
        by_mode[mode] = _aggregate_scores(
            scored,
            recovery_scores=recovery_scored,
            record_count=len(mode_records),
            model_error_count=sum(record.get("model_error") is not None for record in mode_records),
        )
    return {
        "dry_run": config.dry_run,
        "model_name": _evaluation_model_name(config),
        "provider": config.provider,
        "api_style": _resolved_api_style(config.provider, config.api_style),
        "logical_step_count": logical_steps,
        "record_count": len(records),
        "modes": list(config.modes),
        "history_source": config.history_source,
        "includes_valid_actions": config.include_valid_actions,
        "frame_sampling": config.frame_sampling,
        "observation_window_seconds": config.observation_window_seconds,
        "frames_per_camera": config.frame_count,
        "by_mode": by_mode,
    }


def _evaluation_model_name(config: RealObservationEvalConfig) -> str:
    if config.model_name and config.model_name.strip():
        return config.model_name.strip()
    name = config.model or _default_model(config.provider)
    if not config.include_valid_actions:
        return f"{name}_no_valid_action"
    return name


def _aggregate_scores(
    scores: list[dict[str, Any]],
    *,
    recovery_scores: list[dict[str, Any]] | None = None,
    record_count: int,
    model_error_count: int = 0,
) -> dict[str, Any]:
    recovery_scores = recovery_scores or []
    parsed_count = sum(bool(score["parsed"]) for score in scores)
    object_scores = [score for score in scores if score["object_applicable"]]
    target_scores = [score for score in scores if score["target_applicable"]]
    recovery_failed_action_scores = [
        score for score in recovery_scores if score["failed_action_applicable"]
    ]
    paired_scores = list(zip(scores, recovery_scores))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "record_count": record_count,
        "scored_count": len(scores),
        "model_error_count": model_error_count,
        "parse_success_rate": ratio(parsed_count, len(scores)),
        "action_name_accuracy": ratio(sum(bool(score["name"]) for score in scores), len(scores)),
        "object_accuracy": ratio(sum(bool(score["object"]) for score in object_scores), len(object_scores)),
        "target_accuracy": ratio(sum(bool(score["target"]) for score in target_scores), len(target_scores)),
        "node_ids_exact_accuracy": ratio(sum(bool(score["node_ids"]) for score in scores), len(scores)),
        "full_action_exact_accuracy": ratio(sum(bool(score["full_exact"]) for score in scores), len(scores)),
        "recovery_scored_count": len(recovery_scores),
        "recovery_parse_success_rate": ratio(
            sum(bool(score["parsed"]) for score in recovery_scores),
            len(recovery_scores),
        ),
        "recovery_required_accuracy": ratio(
            sum(bool(score["required"]) for score in recovery_scores),
            len(recovery_scores),
        ),
        "recovery_failed_action_accuracy": ratio(
            sum(bool(score["failed_action"]) for score in recovery_failed_action_scores),
            len(recovery_failed_action_scores),
        ),
        "recovery_exact_accuracy": ratio(
            sum(bool(score["full_exact"]) for score in recovery_scores),
            len(recovery_scores),
        ),
        "recovery_and_action_exact_accuracy": ratio(
            sum(bool(action["full_exact"] and recovery["full_exact"]) for action, recovery in paired_scores),
            len(paired_scores),
        ),
    }


def _is_matched(step: dict[str, Any]) -> bool:
    real_reply = step.get("real_reply") if isinstance(step.get("real_reply"), dict) else {}
    return real_reply.get("status") == "matched" and real_reply.get("episode_index") is not None


def _is_evaluable_context(context: ReplayStepContext) -> bool:
    return _is_matched(context.step) and context.expected_action.get("base_name") != "recover"


def _evaluable_contexts(
    contexts: list[ReplayStepContext],
    *,
    frame_sampling: str,
) -> list[ReplayStepContext]:
    evaluable: list[ReplayStepContext] = []
    has_previous_observation_source = False
    for context in contexts:
        action_name = context.expected_action.get("base_name")
        matched = _is_matched(context.step)
        can_use_previous_tail = (
            frame_sampling == "previous_tail"
            and has_previous_observation_source
            and (action_name == "stop" or context.step.get("manual_inserted") is True)
        )
        if action_name != "recover" and (
            matched
            or can_use_previous_tail
        ):
            evaluable.append(context)
        if matched or _observation_tail_payload(context.step) is not None:
            has_previous_observation_source = True
    return evaluable


def _wrong_replay_observation(
    contexts: list[ReplayStepContext],
    step_index: int,
) -> dict[str, Any] | None:
    if len(contexts) < 2:
        return None
    if step_index + 1 < len(contexts):
        return contexts[step_index + 1].graph_observation
    return contexts[step_index - 1].graph_observation


def _frames_for_context(
    contexts: list[ReplayStepContext],
    context_index: int,
    *,
    config: RealObservationEvalConfig,
    oss_client: _alignment.OssClientProtocol | None,
) -> tuple[list[str], dict[str, Any]]:
    current = contexts[context_index]
    source = current
    segment_position = "first"
    window_position = "head"
    applied_sampling = "head"
    if config.frame_sampling == "previous_tail" and context_index > 0:
        previous_source = next(
            (
                candidate
                for candidate in reversed(contexts[:context_index])
                if _has_observation_tail_source(candidate.step)
            ),
            None,
        )
        if previous_source is not None:
            source = previous_source
            segment_position = "last"
            window_position = "tail"
            applied_sampling = (
                "custom_observation_tail"
                if _observation_tail_payload(source.step) is not None
                else "previous_tail"
            )

    real_reply = source.step.get("real_reply") or {}
    frames = _frames_for_real_reply(
        real_reply,
        config=config,
        oss_client=oss_client,
        segment_position=segment_position,
        window_position=window_position,
    )
    return frames, {
        "requested_sampling": config.frame_sampling,
        "applied_sampling": applied_sampling,
        "source_step": source.step.get("step", source.step_index + 1),
        "source_episode_indices": _real_reply_episode_indices(real_reply),
        "custom_timestamps": _observation_tail_timestamps(real_reply),
        "custom_camera": _observation_tail_camera(real_reply),
        "segment_position": segment_position,
        "window_position": window_position,
    }


def _has_observation_tail_source(step: dict[str, Any]) -> bool:
    return _is_matched(step) or _observation_tail_payload(step) is not None


def _observation_tail_payload(step: dict[str, Any]) -> dict[str, Any] | None:
    real_reply = step.get("real_reply") if isinstance(step.get("real_reply"), dict) else {}
    tail = real_reply.get("observation_tail")
    if not isinstance(tail, dict):
        return None
    timestamps = tail.get("timestamps")
    if tail.get("episode_index") is None or not isinstance(timestamps, list) or len(timestamps) != 2:
        return None
    return tail


def _observation_tail_timestamps(real_reply: dict[str, Any]) -> list[float]:
    tail = real_reply.get("observation_tail")
    timestamps = tail.get("timestamps") if isinstance(tail, dict) else None
    if not isinstance(timestamps, list):
        return []
    try:
        return [float(value) for value in timestamps]
    except (TypeError, ValueError):
        return []


def _observation_tail_camera(real_reply: dict[str, Any]) -> str | None:
    tail = real_reply.get("observation_tail")
    camera = tail.get("camera") if isinstance(tail, dict) else None
    return str(camera) if camera else None


def _real_reply_episode_indices(real_reply: dict[str, Any]) -> list[int]:
    values = real_reply.get("episode_indices")
    if isinstance(values, list):
        return [int(value) for value in values if value is not None]
    episode_index = real_reply.get("episode_index")
    if episode_index is not None:
        return [int(episode_index)]
    tail = real_reply.get("observation_tail")
    tail_index = tail.get("episode_index") if isinstance(tail, dict) else None
    return [int(tail_index)] if tail_index is not None else []


def _frames_for_real_reply(
    real_reply: dict[str, Any],
    *,
    config: RealObservationEvalConfig,
    oss_client: _alignment.OssClientProtocol | None,
    segment_position: str = "first",
    window_position: str = "head",
) -> list[str]:
    observation_tail = real_reply.get("observation_tail")
    if window_position == "tail" and isinstance(observation_tail, dict):
        return _frames_for_observation_tail(
            observation_tail,
            config=config,
            oss_client=oss_client,
        )
    segments = real_reply.get("segments")
    replies = [segment for segment in segments if isinstance(segment, dict)] if isinstance(segments, list) else []
    reply = real_reply
    if replies:
        reply = replies[-1] if segment_position == "last" else replies[0]
    return _frames_for_single_real_reply(
        reply,
        config=config,
        oss_client=oss_client,
        window_position=window_position,
    )


def _frames_for_observation_tail(
    observation_tail: dict[str, Any],
    *,
    config: RealObservationEvalConfig,
    oss_client: _alignment.OssClientProtocol | None,
) -> list[str]:
    oss_root = observation_tail.get("oss_root")
    episode_index = observation_tail.get("episode_index")
    timestamps = observation_tail.get("timestamps")
    if oss_root is None or episode_index is None or not isinstance(timestamps, list):
        return []
    selected_timestamps = [float(value) for value in timestamps]
    cached = _alignment.cache_episode_assets(
        str(oss_root),
        int(episode_index),
        client=oss_client,
        region=config.oss_region,
        endpoint=config.oss_endpoint,
        cache_root=config.cache_dir,
    )
    episode_dir = Path(cached["cache_dir"])
    selection = "_".join(f"{value:.6f}".replace(".", "p") for value in selected_timestamps)
    frames_dir = episode_dir / "eval_frames" / f"custom_tail_{selection}"
    frames: list[str] = []
    selected_camera = observation_tail.get("camera")
    cameras = (str(selected_camera),) if selected_camera else config.cameras
    for camera in cameras:
        video = cached["videos"].get(camera)
        if not video:
            continue
        frames.extend(
            _extract_video_frames_at_timestamps(
                Path(video),
                frames_dir / _safe_camera(camera),
                selected_timestamps,
            )
        )
    return frames


def _frames_for_single_real_reply(
    real_reply: dict[str, Any],
    *,
    config: RealObservationEvalConfig,
    oss_client: _alignment.OssClientProtocol | None,
    window_position: str = "head",
) -> list[str]:
    oss_root = real_reply.get("oss_root")
    episode_index = real_reply.get("episode_index")
    if oss_root is None or episode_index is None:
        return []
    cached = _alignment.cache_episode_assets(
        str(oss_root),
        int(episode_index),
        client=oss_client,
        region=config.oss_region,
        endpoint=config.oss_endpoint,
        cache_root=config.cache_dir,
    )
    episode_dir = Path(cached["cache_dir"])
    strategy = f"{window_position}_{config.observation_window_seconds:.3f}s"
    frames_dir = episode_dir / "eval_frames" / strategy / str(config.frame_count)
    frames: list[str] = []
    for camera in config.cameras:
        video = cached["videos"].get(camera)
        if not video:
            continue
        frames.extend(
            _extract_video_frames(
                Path(video),
                frames_dir / _safe_camera(camera),
                config.frame_count,
                window_seconds=config.observation_window_seconds,
                window_position=window_position,
            )
        )
    return frames


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    count: int,
    *,
    window_seconds: float,
    window_position: str = "head",
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) >= count:
        return [str(path) for path in existing[:count]]
    for path in existing:
        path.unlink()
    duration = _video_duration(video_path)
    window = min(window_seconds, duration) if duration > 0 else window_seconds
    window_start = max(duration - window, 0.0) if window_position == "tail" and duration > 0 else 0.0
    frames = []
    for index in range(count):
        if window_position == "tail":
            timestamp = window_start + (index + 0.5) * window / count
        else:
            timestamp = index * window / count
        target = output_dir / f"frame_{index:03d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=512:-1",
                str(target),
            ],
            check=True,
        )
        frames.append(str(target))
    return frames


def _extract_video_frames_at_timestamps(
    video_path: Path,
    output_dir: Path,
    timestamps: list[float],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if len(existing) == len(timestamps):
        return [str(path) for path in existing]
    for path in existing:
        path.unlink()
    frames: list[str] = []
    for index, timestamp in enumerate(timestamps):
        target = output_dir / f"frame_{index:03d}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(target),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames.append(str(target))
    return frames


def _video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return max(float(result.stdout.strip()), 0.0)
    except Exception:
        return 0.0


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _default_model(provider: str) -> str:
    if provider == "qwen":
        return "qwen-vl-plus"
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "mr_openai":
        return "gpt-5.5"
    if provider == "mr_anthropic":
        return "claude-opus-4-8"
    if provider == "mr_google":
        return "gemini-3.1-pro-preview"
    return "model"


def _resolved_api_style(provider: str, api_style: str) -> str:
    if api_style != "auto":
        return api_style
    if provider == "mr_openai":
        return "responses"
    if provider == "mr_anthropic":
        return "anthropic_messages"
    if provider == "mr_google":
        return "gemini_generate_content"
    return "chat_completions"


def _safe_camera(camera: str) -> str:
    return camera.replace(".", "_").replace("/", "_")
