from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import TaskRecord, ViewGraph


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: every JSONL line must be an object")
            yield line_no, value


def load_view_graphs_jsonl(path: str | Path) -> list[ViewGraph]:
    """Load view graphs from a JSONL file.

    The primary format is one complete scene per JSONL line. A lightweight
    streaming format is also accepted with records containing `record_type`
    values `scene`, `node`, and `edge`.
    """

    source = Path(path)
    records = list(_iter_jsonl(source))
    if not records:
        return []

    if any("nodes" in record for _, record in records):
        graphs = []
        for index, (_, record) in enumerate(records):
            fallback = f"{source.stem}_{index}"
            graphs.append(ViewGraph.from_dict(record, fallback_scene_id=fallback))
        return graphs

    scenes: dict[str, dict] = {}
    for line_no, record in records:
        record_type = str(record.get("record_type", record.get("type", ""))).lower()
        scene_id = str(record.get("scene_id", "default"))
        scene = scenes.setdefault(scene_id, {"scene_id": scene_id, "nodes": [], "edges": []})
        if record_type == "scene":
            scene.update({k: v for k, v in record.items() if k not in {"record_type", "type"}})
        elif record_type == "node":
            node = {k: v for k, v in record.items() if k not in {"record_type", "type", "scene_id"}}
            scene["nodes"].append(node)
        elif record_type == "edge":
            edge = {k: v for k, v in record.items() if k not in {"record_type", "type", "scene_id"}}
            scene["edges"].append(edge)
        else:
            raise ValueError(f"{source}:{line_no}: expected nodes or record_type scene/node/edge")

    return [
        ViewGraph.from_dict(scene, fallback_scene_id=scene_id)
        for scene_id, scene in sorted(scenes.items(), key=lambda item: item[0])
    ]


def write_tasks_jsonl(tasks: Iterable[TaskRecord], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count
