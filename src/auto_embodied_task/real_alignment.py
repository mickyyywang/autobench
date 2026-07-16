from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Protocol

from .manual_actions import parse_manual_action


CAMERA_KEYS = (
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
)


class OssClientProtocol(Protocol):
    def cat_text(self, uri: str) -> str:
        ...

    def copy(self, source_uri: str, target_path: str | Path, *, force: bool = False) -> None:
        ...

    def exists(self, uri: str) -> bool:
        ...


@dataclass(frozen=True)
class OssUtilClient:
    region: str = "cn-shanghai"
    endpoint: str | None = None
    ossutil_bin: str = "ossutil"
    config_file: str | None = None

    def _cmd(self, command: str, *args: str) -> list[str]:
        cmd = [self.ossutil_bin, "-q", command]
        if self.region:
            cmd.extend(["--region", self.region])
        if self.endpoint:
            cmd.extend(["--endpoint", self.endpoint])
        if self.config_file:
            cmd.extend(["--config-file", self.config_file])
        cmd.extend(args)
        return cmd

    def cat_text(self, uri: str) -> str:
        result = subprocess.run(
            self._cmd("cat", uri),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout

    def copy(self, source_uri: str, target_path: str | Path, *, force: bool = False) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        args = [source_uri, str(target)]
        if force:
            args.append("-f")
        subprocess.run(
            self._cmd("cp", *args),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def exists(self, uri: str) -> bool:
        result = subprocess.run(
            self._cmd("stat", uri),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0


def default_outputs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs"


def default_saved_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "saved"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "auto_embodied_task" / "real_alignment"


def trajectory_files(root: str | Path) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"trajectory directory is not a directory: {directory}")
    files = []
    for path in directory.glob("*.jsonl"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if not _looks_like_trajectory_jsonl(path):
            continue
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    files.sort(key=lambda item: (-float(item["mtime"]), str(item["name"])))
    return files


def load_trajectory_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    episodes: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                episode = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(episode, dict):
                raise ValueError(f"{source}:{line_no}: trajectory line must be a JSON object")
            episodes.append(episode)
    return episodes


def trajectory_summary(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    episodes = load_trajectory_jsonl(source)
    return {
        "path": str(source),
        "name": source.name,
        "episode_count": len(episodes),
        "episodes": [_episode_summary(episode, index) for index, episode in enumerate(episodes)],
    }


def saved_alignment_summary(
    path: str | Path,
    *,
    trajectory_root: str | Path | None = None,
    episode_index: int = 0,
) -> dict[str, Any]:
    source = Path(path)
    episodes = load_trajectory_jsonl(source)
    if episode_index < 0 or episode_index >= len(episodes):
        raise ValueError(f"episode_index out of range: {episode_index}")
    aligned_episode = episodes[episode_index]
    alignment = aligned_episode.get("real_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"file is not a saved alignment: {source}")

    original_path: Path | None = None
    original_summary: dict[str, Any] | None = None
    source_episode_index = _optional_int(alignment.get("source_episode_index")) or 0
    raw_original_path = alignment.get("source_trajectory")
    if raw_original_path and trajectory_root is not None:
        try:
            original_path = resolve_trajectory_path(str(raw_original_path), trajectory_root)
            original_summary = trajectory_summary(original_path)
            if source_episode_index >= original_summary["episode_count"]:
                raise ValueError("saved source_episode_index is outside the source trajectory")
        except (OSError, ValueError):
            original_path = None
            original_summary = None
    if original_summary is None:
        original_summary = _fallback_source_summary(aligned_episode, source)
        source_episode_index = 0

    rows = _alignment_rows_from_saved_episode(aligned_episode)
    real_episodes = _real_episodes_from_saved_episode(aligned_episode)
    oss_root = str(alignment.get("oss_root") or _saved_episode_oss_root(aligned_episode) or "")
    return {
        "path": str(source.resolve()),
        "name": source.name,
        "saved_episode_index": episode_index,
        "saved_episode_count": len(episodes),
        "source_file": str(original_path) if original_path is not None else None,
        "source_available": original_path is not None,
        "source_episode_index": source_episode_index,
        "trajectory": original_summary,
        "oss_root": oss_root,
        "oss": {
            "oss_root": oss_root,
            "dataset_name": alignment.get("dataset_name") or (dataset_name_from_oss_root(oss_root) if oss_root else ""),
            "episode_count": len(real_episodes),
            "episodes": real_episodes,
            "partial": True,
        },
        "rows": rows,
        "real_alignment": copy.deepcopy(alignment),
    }


def resolve_trajectory_path(raw_path: str, root: str | Path) -> Path:
    directory = Path(root).resolve()
    requested = Path(raw_path)
    candidate = requested.resolve() if requested.is_absolute() else (directory / requested).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise ValueError(f"trajectory file must be inside {directory}") from exc
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"trajectory file does not exist: {candidate}")
    if candidate.suffix != ".jsonl":
        raise ValueError(f"trajectory file must be .jsonl: {candidate}")
    return candidate


def _alignment_rows_from_saved_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps = [step for step in episode.get("trajectory", []) or [] if isinstance(step, dict)]
    for index, step in enumerate(steps):
        source_indices = _saved_source_indices(step)
        real_reply = step.get("real_reply") if isinstance(step.get("real_reply"), dict) else {}
        real_indices = _saved_real_indices(step, real_reply)
        status = str(real_reply.get("status") or ("matched" if real_indices else "unmatched"))
        rows.append(
            {
                "id": f"saved_row_{index + 1}",
                "traj_step_index": source_indices[0] if source_indices else None,
                "traj_step_indices": source_indices,
                "source_step": step.get("source_step"),
                "action": copy.deepcopy(step.get("action") if isinstance(step.get("action"), dict) else {}),
                "manual_name": step.get("manual_name"),
                "real_episode_index": real_indices[0] if real_indices else None,
                "real_episode_indices": real_indices,
                "status": status,
                "notes": str(real_reply.get("notes") or ""),
                "observation_tail": _saved_observation_tail(real_reply),
                "include": True,
            }
        )
    return rows


def _saved_source_indices(step: dict[str, Any]) -> list[int]:
    raw = step.get("source_traj_step_indices")
    if isinstance(raw, list):
        return _unique_ints(raw)
    source_index = _optional_int(step.get("source_traj_step_index"))
    if source_index is not None:
        return [source_index]
    source_step = _optional_int(step.get("step"))
    return [] if step.get("manual_inserted") or source_step is None else [source_step - 1]


def _saved_real_indices(step: dict[str, Any], real_reply: dict[str, Any]) -> list[int]:
    raw = real_reply.get("episode_indices")
    if isinstance(raw, list):
        return _unique_ints(raw)
    replies = step.get("real_replies")
    if isinstance(replies, list):
        indices = [item.get("episode_index") for item in replies if isinstance(item, dict)]
        if indices:
            return _unique_ints(indices)
    episode_index = _optional_int(real_reply.get("episode_index"))
    return [] if episode_index is None else [episode_index]


def _saved_observation_tail(real_reply: dict[str, Any]) -> dict[str, Any] | None:
    raw = real_reply.get("observation_tail")
    if not isinstance(raw, dict):
        return None
    episode_index = _optional_int(raw.get("episode_index"))
    timestamps = raw.get("timestamps")
    if episode_index is None or not isinstance(timestamps, list):
        return None
    return {
        "episode_index": episode_index,
        "timestamps": copy.deepcopy(timestamps),
        "camera": str(raw.get("camera") or CAMERA_KEYS[0]),
        "source": str(raw.get("source") or "manual_frame_selection"),
    }


def _real_episodes_from_saved_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for step in episode.get("trajectory", []) or []:
        if not isinstance(step, dict):
            continue
        real_reply = step.get("real_reply") if isinstance(step.get("real_reply"), dict) else {}
        replies = step.get("real_replies")
        if not isinstance(replies, list) or not replies:
            replies = real_reply.get("segments")
        if not isinstance(replies, list) or not replies:
            replies = [real_reply]
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            real_index = _optional_int(reply.get("episode_index"))
            if real_index is not None:
                by_index[real_index] = copy.deepcopy(reply)
        observation_tail = real_reply.get("observation_tail")
        if isinstance(observation_tail, dict):
            tail_index = _optional_int(observation_tail.get("episode_index"))
            if tail_index is not None:
                by_index[tail_index] = copy.deepcopy(observation_tail)
    return [by_index[index] for index in sorted(by_index)]


def _saved_episode_oss_root(episode: dict[str, Any]) -> str | None:
    for step in episode.get("trajectory", []) or []:
        if not isinstance(step, dict):
            continue
        real_reply = step.get("real_reply")
        if isinstance(real_reply, dict) and real_reply.get("oss_root"):
            return str(real_reply["oss_root"])
    return None


def _fallback_source_summary(episode: dict[str, Any], source: Path) -> dict[str, Any]:
    aligned_steps = [step for step in episode.get("trajectory", []) or [] if isinstance(step, dict)]
    deleted = _nested(episode, ("real_alignment", "deleted_source_steps"))
    max_index = max(
        [
            *[value for step in aligned_steps for value in _saved_source_indices(step)],
            *[
                value
                for item in (deleted if isinstance(deleted, list) else [])
                if isinstance(item, dict)
                for value in [_optional_int(item.get("traj_step_index"))]
                if value is not None
            ],
        ],
        default=-1,
    )
    source_steps: list[dict[str, Any] | None] = [None] * (max_index + 1)
    for step in aligned_steps:
        indices = _saved_source_indices(step)
        merged = step.get("merged_source_steps")
        if isinstance(merged, list) and len(merged) == len(indices):
            candidates = merged
        elif len(indices) == 1:
            candidates = [step]
        else:
            actions = step.get("actions") if isinstance(step.get("actions"), list) else []
            candidates = [
                {"step": index + 1, "action": actions[offset] if offset < len(actions) else {}}
                for offset, index in enumerate(indices)
            ]
        for source_index, candidate in zip(indices, candidates):
            if 0 <= source_index < len(source_steps) and isinstance(candidate, dict):
                source_steps[source_index] = _step_summary(candidate, source_index)
    for index, step in enumerate(source_steps):
        if step is None:
            source_steps[index] = {
                "index": index,
                "step": index + 1,
                "action": {"name": "source_step_unavailable", "node_ids": []},
                "requested_action": None,
                "event": None,
                "success_after_step": None,
                "teacher_reason": None,
            }
    episode_summary = _episode_summary(episode, 0)
    episode_summary["steps"] = source_steps
    episode_summary["step_count"] = len(source_steps)
    return {
        "path": str(source),
        "name": source.name,
        "episode_count": 1,
        "episodes": [episode_summary],
        "fallback": True,
    }


def _unique_ints(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and parsed not in result:
            result.append(parsed)
    return result


def load_lerobot_dataset(
    oss_root: str,
    *,
    client: OssClientProtocol | None = None,
    region: str = "cn-shanghai",
    endpoint: str | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    root = normalize_oss_root(oss_root)
    client = client or OssUtilClient(region=region, endpoint=endpoint)
    info = _parse_json_object(client.cat_text(oss_join(root, "meta/info.json")))
    if not isinstance(info, dict):
        raise ValueError(f"{root}/meta/info.json must be a JSON object")
    total = int(info.get("total_episodes") or 0)
    if max_episodes is not None:
        total = min(total, max_episodes)

    episodes = []
    for episode_index in range(total):
        meta_uri = oss_join(root, f"meta/episodes/{episode_index:06d}.json")
        meta: dict[str, Any] = {}
        try:
            loaded_meta = _parse_json_object(client.cat_text(meta_uri))
            if isinstance(loaded_meta, dict):
                meta = loaded_meta
        except Exception:
            meta = {}
        episodes.append(real_episode_descriptor(root, episode_index, info, meta))
    return {
        "oss_root": root,
        "dataset_name": dataset_name_from_oss_root(root),
        "info": info,
        "episode_count": total,
        "episodes": episodes,
        "camera_keys": list(CAMERA_KEYS),
    }


def real_episode_descriptor(
    oss_root: str,
    episode_index: int,
    info: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = normalize_oss_root(oss_root)
    info = info or {}
    meta = meta or {}
    data_parquet = oss_join(root, f"data/chunk-000/file-{episode_index:03d}.parquet")
    raw_parquet = oss_join(root, f"raw/episode_{episode_index:06d}.parquet")
    videos = {
        camera: oss_join(root, f"videos/{camera}/chunk-000/file-{episode_index:03d}.mp4")
        for camera in CAMERA_KEYS
    }
    frame_count = meta.get("frame_count")
    if frame_count is None:
        frame_count = _feature_frame_count_hint(info, episode_index)
    return {
        "episode_index": episode_index,
        "episode_id": f"episode_{episode_index:06d}",
        "task_name": _nested(meta, ("params", "task_name")),
        "collector_name": meta.get("collector_name") or _nested(meta, ("params", "collector_name")),
        "recorded_at": meta.get("recorded_at"),
        "frame_count": frame_count,
        "fps": meta.get("fps") or _nested(meta, ("params", "fps")) or info.get("fps"),
        "meta_episode": oss_join(root, f"meta/episodes/{episode_index:06d}.json"),
        "data_parquet": data_parquet,
        "raw_parquet": raw_parquet,
        "videos": videos,
        "action_columns": _action_columns(info),
        "meta": meta,
    }


def cache_episode_assets(
    oss_root: str,
    episode_index: int,
    *,
    client: OssClientProtocol | None = None,
    region: str = "cn-shanghai",
    endpoint: str | None = None,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = normalize_oss_root(oss_root)
    client = client or OssUtilClient(region=region, endpoint=endpoint)
    base = episode_cache_dir(root, cache_root=cache_root)
    episode_dir = base / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    info = _read_dataset_info_best_effort(root, client)
    meta = _read_episode_meta_best_effort(root, episode_index, client)
    descriptor = real_episode_descriptor(root, episode_index, info, meta)

    local_videos: dict[str, str] = {}
    for camera, uri in descriptor["videos"].items():
        target = episode_dir / f"{camera.replace('.', '_')}.mp4"
        if force or not target.exists():
            client.copy(uri, target, force=True)
        local_videos[camera] = str(target)

    data_target = episode_dir / "data.parquet"
    if force or not data_target.exists():
        client.copy(descriptor["data_parquet"], data_target, force=True)

    meta_target = episode_dir / "meta_episode.json"
    meta_target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    parquet_summary = parquet_action_summary(data_target)
    return {
        "oss_root": root,
        "episode_index": episode_index,
        "cache_dir": str(episode_dir),
        "videos": local_videos,
        "data_parquet": str(data_target),
        "meta_episode": str(meta_target),
        "parquet_summary": parquet_summary,
        "descriptor": descriptor,
    }


def episode_cache_dir(oss_root: str, *, cache_root: str | Path | None = None) -> Path:
    digest = hashlib.sha1(normalize_oss_root(oss_root).encode("utf-8")).hexdigest()[:16]
    dataset = _safe_filename(dataset_name_from_oss_root(oss_root))
    return Path(cache_root or default_cache_dir()) / f"{dataset}_{digest}"


def parquet_action_summary(path: str | Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - depends on optional env
        return {"ok": False, "error": f"pyarrow unavailable: {exc}"}
    try:
        parquet_file = pq.ParquetFile(path)
        columns = list(parquet_file.schema.names)
        row_count = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        action_columns = [column for column in columns if column.startswith("action.")]
        summary: dict[str, Any] = {
            "ok": True,
            "row_count": row_count,
            "columns": columns,
            "action_columns": action_columns,
        }
        if row_count:
            sample_columns = [column for column in ("timestamp", "frame_index", *action_columns[:4]) if column in columns]
            if sample_columns:
                table = parquet_file.read(columns=sample_columns)
                first = table.slice(0, 1).to_pylist()[0]
                last = table.slice(max(row_count - 1, 0), 1).to_pylist()[0]
                summary["first_row"] = first
                summary["last_row"] = last
        return summary
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_initial_alignment(episode: dict[str, Any], real_episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = [step for step in episode.get("trajectory", []) or [] if isinstance(step, dict)]
    rows = []
    for index, step in enumerate(steps):
        real_episode = real_episodes[index] if index < len(real_episodes) else None
        rows.append(
            {
                "id": f"row_{index + 1}",
                "traj_step_index": index,
                "traj_step_indices": [index],
                "source_step": step.get("step", index + 1),
                "action": step.get("action") if isinstance(step.get("action"), dict) else {},
                "real_episode_index": real_episode.get("episode_index") if real_episode else None,
                "real_episode_indices": [real_episode.get("episode_index")] if real_episode else [],
                "status": "matched" if real_episode else "unmatched",
                "notes": "",
                "include": True,
            }
        )
    return rows


def save_aligned_episode(
    *,
    trajectory_path: str | Path,
    episode_index: int,
    oss_root: str,
    rows: list[dict[str, Any]],
    real_episodes: list[dict[str, Any]],
    saved_dir: str | Path | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    source = Path(trajectory_path)
    episodes = load_trajectory_jsonl(source)
    if episode_index < 0 or episode_index >= len(episodes):
        raise ValueError(f"episode_index out of range: {episode_index}")
    original_episode = episodes[episode_index]
    real_by_index = {
        int(item["episode_index"]): item for item in real_episodes if item.get("episode_index") is not None
    }
    aligned_episode = _aligned_episode_payload(
        original_episode=original_episode,
        source_path=source,
        episode_index=episode_index,
        oss_root=normalize_oss_root(oss_root),
        rows=rows,
        real_by_index=real_by_index,
    )
    target_dir = Path(saved_dir or default_saved_dir())
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (output_name or _default_saved_name(source, oss_root))
    with target.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(aligned_episode, ensure_ascii=False) + "\n")
    return {
        "path": str(target),
        "episode_id": aligned_episode.get("episode_id"),
        "step_count": len(aligned_episode.get("trajectory", []) or []),
        "matched_count": aligned_episode.get("real_alignment", {}).get("matched_count", 0),
        "skipped_real_episodes": aligned_episode.get("real_alignment", {}).get("skipped_real_episodes", []),
    }


def normalize_oss_root(value: str) -> str:
    text = str(value).strip()
    if not text.startswith("oss://"):
        raise ValueError(f"OSS root must start with oss://: {value}")
    return text.rstrip("/")


def oss_join(root: str, suffix: str) -> str:
    return f"{normalize_oss_root(root)}/{suffix.strip('/')}"


def dataset_name_from_oss_root(root: str) -> str:
    return normalize_oss_root(root).rstrip("/").split("/")[-1]


def _aligned_episode_payload(
    *,
    original_episode: dict[str, Any],
    source_path: Path,
    episode_index: int,
    oss_root: str,
    rows: list[dict[str, Any]],
    real_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    source_steps = [step for step in original_episode.get("trajectory", []) or [] if isinstance(step, dict)]
    used_real_indices: set[int] = set()
    aligned_steps = []
    deleted_source_steps = []

    for new_step_number, row in enumerate((row for row in rows if row.get("include", True)), start=1):
        source_step_indices = _row_source_indices(row)
        grouped_source_steps = [
            (index, source_steps[index])
            for index in source_step_indices
            if 0 <= index < len(source_steps)
        ]
        if not grouped_source_steps:
            manual_name = str(row.get("manual_name") or "manual_inserted").strip() or "manual_inserted"
            try:
                manual_action = parse_manual_action(manual_name)
            except ValueError:
                manual_action = {"name": manual_name, "base_name": "manual_inserted", "node_ids": []}
            step_payload = {
                "step": new_step_number,
                "mode": "manual_alignment",
                "manual_inserted": True,
                "manual_name": manual_name,
                "action": manual_action,
            }
        else:
            first_index, first_step = grouped_source_steps[0]
            step_payload = copy.deepcopy(first_step)
            step_payload["source_step"] = first_step.get("step", first_index + 1)
            step_payload["source_traj_step_index"] = first_index
            step_payload["source_steps"] = [
                step.get("step", index + 1) for index, step in grouped_source_steps
            ]
            step_payload["source_traj_step_indices"] = [index for index, _ in grouped_source_steps]
            step_payload["actions"] = [
                copy.deepcopy(step.get("action") if isinstance(step.get("action"), dict) else {})
                for _, step in grouped_source_steps
            ]
            if len(grouped_source_steps) > 1:
                step_payload["merged_source_steps"] = [
                    copy.deepcopy(step) for _, step in grouped_source_steps
                ]
            step_payload["step"] = new_step_number

        real_indices = _row_real_indices(row)
        real_episodes = [real_by_index[index] for index in real_indices if index in real_by_index]
        for real_index in real_indices:
            if real_index not in real_by_index:
                continue
            if real_index in used_real_indices:
                raise ValueError(
                    f"real episode {real_index} is assigned to multiple alignment rows; "
                    "merge the traj steps or real episodes into one row instead"
                )
            used_real_indices.add(real_index)
        status = str(row.get("status") or ("matched" if real_episodes else "unmatched"))
        real_replies = [
            _real_reply_payload(
                status=status,
                oss_root=oss_root,
                real_episode=real_episode,
                notes=str(row.get("notes") or ""),
            )
            for real_episode in real_episodes
        ]
        step_payload["real_reply"] = _merged_real_reply_payload(
            status=status,
            oss_root=oss_root,
            real_replies=real_replies,
            notes=str(row.get("notes") or ""),
        )
        observation_tail = _row_observation_tail(
            row,
            oss_root=oss_root,
            real_by_index=real_by_index,
        )
        if observation_tail is not None:
            if real_episodes:
                raise ValueError(
                    "observation_tail is only valid for a row without a paired real episode"
                )
            step_payload["real_reply"]["observation_tail"] = observation_tail
        step_payload["real_replies"] = copy.deepcopy(real_replies)
        aligned_steps.append(step_payload)

    included_source_indices = {
        index
        for row in rows
        if row.get("include", True)
        for index in _row_source_indices(row)
    }
    for index, step in enumerate(source_steps):
        if index not in included_source_indices:
            deleted_source_steps.append({"traj_step_index": index, "source_step": step.get("step", index + 1)})

    skipped_real = [
        index
        for index in sorted(real_by_index)
        if index not in used_real_indices
    ]
    episode = copy.deepcopy(original_episode)
    episode["trajectory"] = aligned_steps
    episode["real_alignment"] = {
        "version": 4,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source_trajectory": str(source_path),
        "source_episode_index": episode_index,
        "oss_root": oss_root,
        "dataset_name": dataset_name_from_oss_root(oss_root),
        "row_count": len(rows),
        "output_step_count": len(aligned_steps),
        "source_step_count": len(included_source_indices),
        "merged_group_count": sum(
            1 for row in rows if row.get("include", True) and len(_row_source_indices(row)) > 1
        ),
        "merged_real_episode_group_count": sum(
            1 for row in rows if row.get("include", True) and len(_row_real_indices(row)) > 1
        ),
        "matched_count": sum(
            1
            for step in aligned_steps
            if step.get("real_reply", {}).get("status") == "matched"
            and step.get("real_reply", {}).get("episode_index") is not None
        ),
        "custom_observation_tail_count": sum(
            1
            for step in aligned_steps
            if isinstance(step.get("real_reply", {}).get("observation_tail"), dict)
        ),
        "skipped_real_episodes": skipped_real,
        "deleted_source_steps": deleted_source_steps,
    }
    return episode


def _row_source_indices(row: dict[str, Any]) -> list[int]:
    raw_indices = row.get("traj_step_indices")
    if not isinstance(raw_indices, list):
        raw_index = row.get("traj_step_index")
        raw_indices = [] if raw_index is None else [raw_index]
    indices: list[int] = []
    for value in raw_indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in indices:
            indices.append(index)
    return indices


def _row_real_indices(row: dict[str, Any]) -> list[int]:
    raw_indices = row.get("real_episode_indices")
    if not isinstance(raw_indices, list):
        raw_index = row.get("real_episode_index")
        raw_indices = [] if raw_index is None else [raw_index]
    indices: list[int] = []
    for value in raw_indices:
        index = _optional_int(value)
        if index is not None and index not in indices:
            indices.append(index)
    return indices


def _row_observation_tail(
    row: dict[str, Any],
    *,
    oss_root: str,
    real_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    raw = row.get("observation_tail")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("observation_tail must be an object")
    episode_index = _optional_int(raw.get("episode_index"))
    if episode_index is None or episode_index not in real_by_index:
        raise ValueError("observation_tail must reference an available real episode")
    raw_timestamps = raw.get("timestamps")
    if not isinstance(raw_timestamps, list) or len(raw_timestamps) != 2:
        raise ValueError("observation_tail must contain exactly two timestamps")
    try:
        timestamps = [float(value) for value in raw_timestamps]
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_tail timestamps must be numbers") from exc
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError("observation_tail timestamps must be finite and non-negative")
    timestamps.sort()
    if timestamps[0] == timestamps[1]:
        raise ValueError("observation_tail timestamps must select two distinct frames")
    camera = str(raw.get("camera") or CAMERA_KEYS[0])
    if camera not in CAMERA_KEYS:
        raise ValueError(f"observation_tail camera must be one of {CAMERA_KEYS}")

    payload = _real_reply_payload(
        status="custom_observation_tail",
        oss_root=oss_root,
        real_episode=real_by_index[episode_index],
        notes="",
    )
    payload["source"] = "manual_frame_selection"
    payload["camera"] = camera
    payload["timestamps"] = timestamps
    return payload


def _merged_real_reply_payload(
    *,
    status: str,
    oss_root: str,
    real_replies: list[dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    if not real_replies:
        return _real_reply_payload(
            status=status,
            oss_root=oss_root,
            real_episode=None,
            notes=notes,
        )
    payload = copy.deepcopy(real_replies[0])
    payload["episode_indices"] = [reply.get("episode_index") for reply in real_replies]
    payload["episode_ids"] = [reply.get("episode_id") for reply in real_replies]
    payload["segment_count"] = len(real_replies)
    payload["segments"] = copy.deepcopy(real_replies)
    payload["frame_count"] = sum(int(reply.get("frame_count") or 0) for reply in real_replies)
    return payload


def _real_reply_payload(
    *,
    status: str,
    oss_root: str,
    real_episode: dict[str, Any] | None,
    notes: str,
) -> dict[str, Any]:
    if status == "matched" and real_episode is None:
        status = "unmatched"
    payload: dict[str, Any] = {
        "status": status,
        "oss_root": oss_root,
        "notes": notes,
    }
    if real_episode is None:
        payload.update(
            {
                "episode_index": None,
                "episode_id": None,
                "videos": {},
                "data_parquet": None,
                "raw_parquet": None,
                "meta_episode": None,
                "frame_count": None,
                "fps": None,
                "action_columns": [],
            }
        )
        return payload
    payload.update(
        {
            "episode_index": real_episode.get("episode_index"),
            "episode_id": real_episode.get("episode_id"),
            "videos": copy.deepcopy(real_episode.get("videos", {})),
            "data_parquet": real_episode.get("data_parquet"),
            "raw_parquet": real_episode.get("raw_parquet"),
            "meta_episode": real_episode.get("meta_episode"),
            "frame_count": real_episode.get("frame_count"),
            "fps": real_episode.get("fps"),
            "action_columns": list(real_episode.get("action_columns", []) or []),
            "task_name": real_episode.get("task_name"),
            "collector_name": real_episode.get("collector_name"),
            "recorded_at": real_episode.get("recorded_at"),
        }
    )
    return payload


def _episode_summary(episode: dict[str, Any], index: int) -> dict[str, Any]:
    steps = [step for step in episode.get("trajectory", []) or [] if isinstance(step, dict)]
    return {
        "index": index,
        "episode_id": str(episode.get("episode_id") or episode.get("task_id") or f"episode_{index + 1}"),
        "task": episode.get("task"),
        "task_type": episode.get("task_type"),
        "scene_id": episode.get("scene_id"),
        "success": episode.get("success"),
        "step_count": len(steps),
        "steps": [_step_summary(step, step_index) for step_index, step in enumerate(steps)],
    }


def _step_summary(step: dict[str, Any], index: int) -> dict[str, Any]:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    requested = step.get("requested_action") if isinstance(step.get("requested_action"), dict) else None
    return {
        "index": index,
        "step": step.get("step", index + 1),
        "action": action,
        "requested_action": requested,
        "event": step.get("event"),
        "success_after_step": step.get("success_after_step"),
        "teacher_reason": _nested(step, ("teacher_response", "reason")),
    }


def _looks_like_trajectory_jsonl(path: Path) -> bool:
    name = path.name.lower()
    if "trajectory" in name or "trajectories" in name:
        return True
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                return isinstance(payload, dict) and (
                    "trajectory" in payload or "initial_observation" in payload
                )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def _parse_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty JSON payload")
    decoder = json.JSONDecoder()
    for start in [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found")


def _read_dataset_info_best_effort(root: str, client: OssClientProtocol) -> dict[str, Any]:
    try:
        info = _parse_json_object(client.cat_text(oss_join(root, "meta/info.json")))
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def _read_episode_meta_best_effort(root: str, episode_index: int, client: OssClientProtocol) -> dict[str, Any]:
    try:
        meta = _parse_json_object(client.cat_text(oss_join(root, f"meta/episodes/{episode_index:06d}.json")))
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def _feature_frame_count_hint(info: dict[str, Any], episode_index: int) -> int | None:
    episodes = info.get("episodes")
    if isinstance(episodes, list) and episode_index < len(episodes) and isinstance(episodes[episode_index], dict):
        frame_count = episodes[episode_index].get("frame_count")
        return _optional_int(frame_count)
    return None


def _action_columns(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if isinstance(features, dict):
        names = list(features)
    elif isinstance(features, list):
        names = [str(item) for item in features]
    else:
        names = []
    return [name for name in names if name.startswith("action.")]


def _nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_saved_name(source: Path, oss_root: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = _safe_filename(dataset_name_from_oss_root(oss_root))
    return f"{source.stem}__{dataset}__aligned_{timestamp}.jsonl"


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._")
    return text or "dataset"


def ensure_command_available(command: str) -> bool:
    return shutil.which(command) is not None
