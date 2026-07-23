from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


from .harness import SymbolicBackend
from .models import TaskRecord, ViewGraph
from .placement_constraints import PlacementEdgeConstraints
from .placement_constraints import load_placement_edge_constraints


PROJECT_DIR = Path(__file__).resolve().parents[2]


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def load_direct_episode(
    *,
    view_graph_path: str | Path,
    tasks_path: str | Path,
    episode_id: str | None = None,
    project_dir: str | Path = PROJECT_DIR,
) -> dict[str, Any]:
    """Build the closed-loop episode shape from a view graph and task JSONL.

    This is the no-trajectory counterpart of an aligned ``saved`` episode.  The
    initial graph and completion criterion stay byte-sourceable from their two
    canonical files; no teacher trajectory is invented for evaluation metrics.
    """

    graph_source = Path(view_graph_path).resolve()
    task_source = Path(tasks_path).resolve()
    if not graph_source.is_file():
        raise ValueError(f"view graph does not exist: {graph_source}")
    if not task_source.is_file():
        raise ValueError(f"tasks file does not exist: {task_source}")

    task_rows = _read_jsonl_objects(task_source)
    if episode_id:
        task_rows = [
            row
            for row in task_rows
            if str(row.get("task_id") or row.get("episode_id") or "") == episode_id
        ]
    if len(task_rows) != 1:
        qualifier = f" for {episode_id!r}" if episode_id else ""
        raise ValueError(
            f"{task_source}: expected exactly one task{qualifier}, got {len(task_rows)}"
        )
    task = task_rows[0]
    resolved_episode_id = str(
        task.get("task_id") or task.get("episode_id") or episode_id or ""
    ).strip()
    if not resolved_episode_id:
        raise ValueError(f"{task_source}: task is missing task_id")

    scene_id = str(task.get("scene_id") or resolved_episode_id)
    graph_rows = _read_jsonl_objects(graph_source)
    matching_graphs = [
        row
        for row in graph_rows
        if str(row.get("scene_id") or row.get("id") or graph_source.stem) == scene_id
    ]
    if not matching_graphs and len(graph_rows) == 1:
        matching_graphs = graph_rows
    if len(matching_graphs) != 1:
        raise ValueError(
            f"{graph_source}: expected exactly one graph for scene {scene_id!r}, "
            f"got {len(matching_graphs)}"
        )
    graph = matching_graphs[0]

    criterion = copy.deepcopy(task.get("task_completion_criterion"))
    if criterion is None:
        raise ValueError(f"{task_source}: task {resolved_episode_id!r} has no completion criterion")
    task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    constraints = _load_task_constraints(
        task_metadata,
        project_dir=Path(project_dir).resolve(),
    )
    episode = {
        "episode_id": resolved_episode_id,
        "scene_id": scene_id,
        "env_id": task.get("env_id", graph.get("env_id", scene_id)),
        "task_type": str(task.get("task_type") or "manipulation"),
        "task": str(task.get("task") or ""),
        "settings": copy.deepcopy(task.get("settings") or []),
        "initial_task_completion_criterion": criterion,
        "task_completion_criterion": copy.deepcopy(criterion),
        "initial_view_graph": copy.deepcopy(graph),
        "placement_edge_constraints": constraints,
        "trajectory": [],
        "teacher_reference_available": False,
        "episode_source_type": "view_graph_and_task",
        "task_metadata": copy.deepcopy(task_metadata),
    }
    graph_model = ViewGraph.from_dict(graph, fallback_scene_id=scene_id)
    task_model = TaskRecord(
        task_id=resolved_episode_id,
        scene_id=scene_id,
        env_id=episode["env_id"],
        layout=str(task.get("layout") or graph.get("layout") or "tabletop"),
        arms=str(task.get("arms") or "double"),
        task_type=episode["task_type"],
        task=episode["task"],
        task_completion_criterion=copy.deepcopy(criterion),
        ground_truth_plan=[],
        objects=copy.deepcopy(task.get("objects") or {}),
        settings=copy.deepcopy(episode["settings"]),
        metadata=copy.deepcopy(task_metadata),
    )
    constraint_model = PlacementEdgeConstraints.from_json(constraints)
    episode["initial_observation"] = SymbolicBackend(
        graph_model,
        task_model,
        constraint_model,
    ).observe()
    return episode


def direct_manifest_source(
    *,
    view_graph_path: str | Path,
    tasks_path: str | Path,
    episode: dict[str, Any],
) -> dict[str, Any]:
    graph_source = Path(view_graph_path).resolve()
    task_source = Path(tasks_path).resolve()
    episode_id = str(episode["episode_id"])
    return {
        "source_type": "view_graph_and_task",
        "view_graph": str(graph_source),
        "view_graph_sha256": file_sha256(graph_source),
        "tasks": str(task_source),
        "tasks_sha256": file_sha256(task_source),
        "episode_id": episode_id,
        "scene_id": str(episode.get("scene_id") or episode_id),
        "env_id": str(episode.get("env_id") or episode_id),
        "teacher_reference_available": False,
    }


def episode_from_manifest_source(source: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    source_type = str(source.get("source_type") or "aligned_episode")
    if source_type == "aligned_episode":
        aligned_path = Path(str(source["aligned_episode"]))
        rows = _read_jsonl_objects(aligned_path)
        if len(rows) != 1:
            raise ValueError(f"{aligned_path}: expected exactly one episode JSON object")
        return rows[0], aligned_path
    if source_type != "view_graph_and_task":
        raise ValueError(f"unsupported manifest source_type: {source_type!r}")
    graph_path = Path(str(source["view_graph"]))
    tasks_path = Path(str(source["tasks"]))
    return (
        load_direct_episode(
            view_graph_path=graph_path,
            tasks_path=tasks_path,
            episode_id=str(source.get("episode_id") or "") or None,
        ),
        graph_path,
    )


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _load_task_constraints(
    metadata: dict[str, Any],
    *,
    project_dir: Path,
) -> dict[str, Any] | None:
    raw = metadata.get("placement_edge_constraints")
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return PlacementEdgeConstraints.from_json(raw).to_json()
    source = Path(str(raw))
    if not source.is_absolute():
        source = project_dir / source
    if not source.is_file():
        raise ValueError(f"placement edge constraints do not exist: {source}")
    return load_placement_edge_constraints(source).to_json()
