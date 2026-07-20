from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
import webbrowser


def default_evaluation_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "evaluations"


def normalize_base_path(value: str | None) -> str:
    stripped = str(value or "").strip().strip("/")
    return f"/{stripped}" if stripped else ""


def infer_model_name(
    path: str | Path,
    *,
    summary: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> str:
    for record in records or []:
        value = record.get("model_name")
        if value is not None and str(value).strip():
            return str(value).strip()
    if summary:
        value = summary.get("model_name")
        if value is not None and str(value).strip():
            return str(value).strip()

    stem = Path(path).stem
    stem = re.sub(r"^(real_)?eval_", "", stem)
    no_valid_actions = bool(re.search(r"_no_valid_actions?$", stem))
    stem = re.sub(r"_no_valid_actions?$", "", stem)
    match = re.fullmatch(r"qwen(\d+)_(\d+)_(plus|max)", stem)
    if match:
        name = f"qwen{match.group(1)}.{match.group(2)}-{match.group(3)}"
    else:
        name = stem
    return f"{name}_no_valid_action" if no_valid_actions else name


def summary_path_for(result_path: str | Path) -> Path:
    source = Path(result_path)
    return source.with_name(f"{source.stem}__summary.json")


def load_evaluation_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_no}: expected a JSON object")
            records.append(record)
    return records


def load_evaluation_summary(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.exists():
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: summary must be a JSON object")
    return payload


def evaluation_sources(root: str | Path) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.exists():
        return []
    sources: list[dict[str, Any]] = []
    paths = [*directory.glob("*.jsonl"), *(directory / "closed_loop").glob("*.jsonl")]
    for path in sorted(paths):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            records = load_evaluation_records(path)
            if not records or not any("mode" in record and "step" in record for record in records):
                continue
            summary_path = summary_path_for(path)
            summary = load_evaluation_summary(summary_path)
            model_name = infer_model_name(path, summary=summary, records=records)
            evaluation_type = str(
                (summary or {}).get("evaluation_type")
                or records[0].get("evaluation_type")
                or "open_loop_real"
            )
            episode_ids = sorted({str(record.get("episode_id") or "") for record in records})
            modes = sorted({str(record.get("mode") or "") for record in records})
            condition_ids = sorted(
                {
                    str(record.get("condition_id") or "")
                    for record in records
                    if record.get("condition_id")
                }
            )
            sources.append(
                {
                    "path": path,
                    "name": str(path.relative_to(directory)),
                    "model_name": model_name,
                    "evaluation_type": evaluation_type,
                    "record_count": len(records),
                    "episode_ids": [value for value in episode_ids if value],
                    "modes": [value for value in modes if value],
                    "condition_ids": condition_ids,
                    "summary_path": summary_path,
                    "has_summary": summary is not None,
                    "mtime": path.stat().st_mtime,
                }
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    sources.sort(key=lambda item: (str(item["model_name"]), str(item["name"])))
    return sources


def evaluation_catalog(root: str | Path) -> dict[str, Any]:
    sources = evaluation_sources(root)
    return {
        "evaluation_dir": str(Path(root).resolve()),
        "source_count": len(sources),
        "record_count": sum(int(source["record_count"]) for source in sources),
        "model_names": sorted({str(source["model_name"]) for source in sources}),
        "evaluation_types": sorted({str(source["evaluation_type"]) for source in sources}),
        "episode_ids": sorted({value for source in sources for value in source["episode_ids"]}),
        "modes": sorted({value for source in sources for value in source["modes"]}),
        "condition_ids": sorted(
            {value for source in sources for value in source["condition_ids"]}
        ),
        "sources": [
            {
                key: value
                for key, value in source.items()
                if key not in {"path", "summary_path", "mtime"}
            }
            for source in sources
        ],
    }


def evaluation_image_paths(root: str | Path) -> set[Path]:
    return {
        Path(frame).resolve()
        for source in evaluation_sources(root)
        for record in load_evaluation_records(source["path"])
        for frame in list((record.get("request_summary") or {}).get("frame_files") or [])
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if any(record.get("evaluation_type") == "closed_loop_visible_graph" for record in records):
        return _aggregate_closed_loop_record_metrics(records)
    scores = [record["score"] for record in records if isinstance(record.get("score"), dict)]
    recovery_scores = [
        record["recovery_score"]
        for record in records
        if isinstance(record.get("recovery_score"), dict)
    ]
    object_scores = [score for score in scores if score.get("object_applicable")]
    target_scores = [score for score in scores if score.get("target_applicable")]
    failed_action_scores = [
        score for score in recovery_scores if score.get("failed_action_applicable")
    ]
    grounding_scores = [
        score for score in recovery_scores if score.get("failed_node_ids_applicable")
    ]
    paired_scores = list(zip(scores, recovery_scores))
    teacher_imitation = {
        "record_count": len(records),
        "scored_count": len(scores),
        "model_error_count": sum(record.get("model_error") is not None for record in records),
        "parse_success_rate": _ratio(sum(bool(score.get("parsed")) for score in scores), len(scores)),
        "action_name_accuracy": _ratio(sum(bool(score.get("name")) for score in scores), len(scores)),
        "object_accuracy": _ratio(
            sum(bool(score.get("object")) for score in object_scores), len(object_scores)
        ),
        "target_accuracy": _ratio(
            sum(bool(score.get("target")) for score in target_scores), len(target_scores)
        ),
        "node_ids_exact_accuracy": _ratio(
            sum(bool(score.get("node_ids")) for score in scores), len(scores)
        ),
        "full_action_exact_accuracy": _ratio(
            sum(bool(score.get("full_exact")) for score in scores), len(scores)
        ),
        "recovery_scored_count": len(recovery_scores),
        "recovery_parse_success_rate": _ratio(
            sum(bool(score.get("parsed")) for score in recovery_scores), len(recovery_scores)
        ),
        "recovery_required_accuracy": _ratio(
            sum(bool(score.get("required")) for score in recovery_scores), len(recovery_scores)
        ),
        "recovery_failed_action_accuracy": _ratio(
            sum(bool(score.get("failed_action")) for score in failed_action_scores),
            len(failed_action_scores),
        ),
        "recovery_grounding_accuracy": _ratio(
            sum(bool(score.get("grounding_exact")) for score in grounding_scores),
            len(grounding_scores),
        ),
        "recovery_exact_accuracy": _ratio(
            sum(bool(score.get("full_exact")) for score in recovery_scores), len(recovery_scores)
        ),
        "recovery_and_action_exact_accuracy": _ratio(
            sum(
                bool(action.get("full_exact") and recovery.get("full_exact"))
                for action, recovery in paired_scores
            ),
            len(paired_scores),
        ),
    }
    from .real_observation_eval import _aggregate_capability_scores

    capabilities = _aggregate_capability_scores(records)
    primary_metrics = {
        metric: value
        for dimension in (
            "action_selection",
            "failure_recovery",
            "active_exploration",
            "completion_judgment",
        )
        for metric, value in capabilities[dimension].items()
    }
    return {
        **teacher_imitation,
        **primary_metrics,
        "teacher_imitation": teacher_imitation,
        "capabilities": capabilities,
    }


def _aggregate_closed_loop_record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    from .real_observation_eval import _aggregate_capability_scores

    capabilities = _aggregate_capability_scores(records)
    outcomes = [
        record["rollout_outcome"]
        for record in records
        if isinstance(record.get("rollout_outcome"), dict)
    ]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    capabilities["completion_judgment"] = {
        "premature_stop_rate": mean([float(item.get("premature_stop", False)) for item in outcomes]),
        "completion_stop_recall": mean(
            [
                float(item.get("completion_stop_success", False))
                for item in outcomes
                if item.get("goal_ever_satisfied") is True
            ]
        ),
    }
    executable = [
        float(record["normally_executable"])
        for record in records
        if record.get("normally_executable") is not None
    ]
    outcome_metrics = {
        "task_success_rate": mean([float(item.get("success", False)) for item in outcomes]),
        "goal_ever_satisfied_rate": mean(
            [float(item.get("goal_ever_satisfied", False)) for item in outcomes]
        ),
        "final_goal_satisfied_rate": mean(
            [float(item.get("final_goal_satisfied", False)) for item in outcomes]
        ),
        "normalized_goal_progress": mean(
            [float(item.get("normalized_goal_progress", 0.0)) for item in outcomes]
        ),
        "teacher_normalized_efficiency": mean(
            [float(item.get("teacher_normalized_efficiency", 0.0)) for item in outcomes]
        ),
        "action_executability_rate": mean(executable),
        "average_step_count": mean([float(item.get("step_count", 0)) for item in outcomes]),
        "episodes_with_injected_failure_rate": mean(
            [float(item.get("failure_injected", False)) for item in outcomes]
        ),
        "average_injected_failure_count": mean(
            [float(item.get("injected_failure_count", 0)) for item in outcomes]
        ),
        "episodes_with_disturbance_rate": mean(
            [float(item.get("disturbance_applied", False)) for item in outcomes]
        ),
        "average_disturbance_count": mean(
            [float(item.get("disturbance_count", 0)) for item in outcomes]
        ),
    }
    primary_metrics = {
        metric: value
        for dimension in (
            "action_selection",
            "failure_recovery",
            "active_exploration",
            "completion_judgment",
        )
        for metric, value in capabilities[dimension].items()
    }
    return {
        **outcome_metrics,
        **primary_metrics,
        "record_count": len(records),
        "model_error_count": sum(record.get("model_error") is not None for record in records),
        "outcomes": outcome_metrics,
        "capabilities": capabilities,
    }


def evaluation_reply_payload(
    root: str | Path,
    *,
    model_name: str | None = None,
    model_names: list[str] | tuple[str, ...] | None = None,
    episode_id: str | None = None,
    mode: str | None = None,
    evaluation_type: str | None = None,
    condition_id: str | None = None,
    base_path: str = "",
) -> dict[str, Any]:
    base_path = normalize_base_path(base_path)
    selected_models = {value for value in (model_names or []) if value}
    if model_name:
        selected_models.add(model_name)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    metric_groups: list[dict[str, Any]] = []
    for source in evaluation_sources(root):
        source_evaluation_type = str(source.get("evaluation_type") or "open_loop_real")
        if evaluation_type and source_evaluation_type != evaluation_type:
            continue
        source_model = str(source["model_name"])
        if selected_models and source_model not in selected_models:
            continue
        source_records = load_evaluation_records(source["path"])
        filtered_source_records: list[dict[str, Any]] = []
        for record in source_records:
            record_model = str(record.get("model_name") or source_model)
            if selected_models and record_model not in selected_models:
                continue
            if episode_id and str(record.get("episode_id") or "") != episode_id:
                continue
            if mode and str(record.get("mode") or "") != mode:
                continue
            if condition_id and str(record.get("condition_id") or "") != condition_id:
                continue
            item = dict(record)
            item["model_name"] = record_model
            item["evaluation_type"] = str(
                item.get("evaluation_type") or source_evaluation_type
            )
            item["result_file"] = source["name"]
            frame_files = list((item.get("request_summary") or {}).get("frame_files") or [])
            item["image_urls"] = [
                f"{base_path}/api/image?path={quote(str(path))}" for path in frame_files
            ]
            records.append(item)
            filtered_source_records.append(item)

        summary = load_evaluation_summary(source["summary_path"])
        if summary is not None:
            summary_item = dict(summary)
            summary_item["model_name"] = str(summary_item.get("model_name") or source_model)
            summaries.append(
                {
                    "result_file": source["name"],
                    "summary_file": Path(source["summary_path"]).name,
                    "model_name": summary_item["model_name"],
                    "summary": summary_item,
                }
            )

        metric_keys = sorted(
            {
                (
                    str(record.get("episode_id") or ""),
                    str(record.get("model_name") or source_model),
                    str(record.get("mode") or ""),
                    str(record.get("evaluation_type") or source_evaluation_type),
                    str(record.get("condition_id") or ""),
                )
                for record in filtered_source_records
            }
        )
        for (
            group_episode,
            group_model,
            group_mode,
            group_evaluation_type,
            group_condition,
        ) in metric_keys:
            group_records = [
                record
                for record in filtered_source_records
                if str(record.get("episode_id") or "") == group_episode
                and str(record.get("model_name") or source_model) == group_model
                and str(record.get("mode") or "") == group_mode
                and str(record.get("evaluation_type") or source_evaluation_type)
                == group_evaluation_type
                and str(record.get("condition_id") or "") == group_condition
            ]
            metric_groups.append(
                {
                    "episode_id": group_episode,
                    "model_name": group_model,
                    "mode": group_mode,
                    "evaluation_type": group_evaluation_type,
                    "condition_id": group_condition or None,
                    "result_file": source["name"],
                    "metrics": _aggregate_record_metrics(group_records),
                    "config": {
                        key: summary.get(key) if summary else None
                        for key in (
                            "history_source",
                            "frames_per_camera",
                            "frame_sampling",
                            "includes_valid_actions",
                            "failure_injection",
                            "failure_injection_config",
                            "graph_disturbance_file",
                            "graph_disturbance_count",
                            "max_steps",
                            "condition_id",
                            "intervention_type",
                        )
                    },
                }
            )

    records.sort(
        key=lambda record: (
            str(record.get("episode_id") or ""),
            int(record.get("step") or 0),
            str(record.get("model_name") or ""),
            str(record.get("mode") or ""),
            str(record.get("condition_id") or ""),
        )
    )
    return {
        "filters": {
            "model_names": sorted(selected_models),
            "episode_id": episode_id,
            "mode": mode,
            "evaluation_type": evaluation_type,
            "condition_id": condition_id,
        },
        "record_count": len(records),
        "records": records,
        "summaries": summaries,
        "metric_groups": sorted(
            metric_groups,
            key=lambda item: (
                str(item["episode_id"]),
                str(item["model_name"]),
                str(item["mode"]),
                str(item.get("evaluation_type") or ""),
                str(item.get("condition_id") or ""),
                str(item["result_file"]),
            ),
        ),
    }


def import_evaluation_files(
    root: str | Path,
    *,
    jsonl_name: str,
    jsonl_content: str,
    summary_name: str,
    summary_content: str,
) -> dict[str, Any]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    result_name = Path(jsonl_name).name
    imported_summary_name = Path(summary_name).name
    if not result_name.endswith(".jsonl"):
        raise ValueError("evaluation result must be a .jsonl file")
    expected_summary_name = f"{Path(result_name).stem}__summary.json"
    if imported_summary_name != expected_summary_name:
        raise ValueError(f"summary filename must be {expected_summary_name}")

    result_path = directory / result_name
    summary_path = directory / imported_summary_name
    result_temp = directory / f".{result_name}.tmp"
    summary_temp = directory / f".{imported_summary_name}.tmp"
    result_temp.write_text(jsonl_content, encoding="utf-8")
    summary_temp.write_text(summary_content, encoding="utf-8")
    try:
        records = load_evaluation_records(result_temp)
        if not records:
            raise ValueError("evaluation JSONL is empty")
        summary = load_evaluation_summary(summary_temp)
        if summary is None:
            raise ValueError("summary is empty")
        model_name = infer_model_name(result_path, summary=summary, records=records)
        result_temp.replace(result_path)
        summary_temp.replace(summary_path)
    except Exception:
        result_temp.unlink(missing_ok=True)
        summary_temp.unlink(missing_ok=True)
        raise
    return {
        "result_file": result_name,
        "summary_file": imported_summary_name,
        "record_count": len(records),
        "model_name": model_name,
    }


def create_evaluation_reply_app(
    *,
    evaluation_dir: str | Path | None = None,
    base_path: str = "",
):
    try:
        from fastapi import APIRouter, FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install fastapi and uvicorn to use the evaluation reply UI") from exc
    globals()["Request"] = Request

    base_path = normalize_base_path(base_path)
    app = FastAPI(title="Auto Embodied Task Evaluation Reply")
    router = APIRouter()
    app.state.evaluation_dir = Path(evaluation_dir or default_evaluation_dir()).resolve()
    app.state.allowed_images = evaluation_image_paths(app.state.evaluation_dir)
    app.state.base_path = base_path

    @router.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_render_evaluation_reply_html(base_path=base_path))

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "evaluation_dir": str(app.state.evaluation_dir)}

    @router.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        return evaluation_catalog(app.state.evaluation_dir)

    @router.get("/api/replies")
    def replies(
        model_name: str | None = None,
        model_names: str | None = None,
        episode_id: str | None = None,
        mode: str | None = None,
        evaluation_type: str | None = None,
        condition_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return evaluation_reply_payload(
                app.state.evaluation_dir,
                model_name=model_name or None,
                model_names=[value for value in (model_names or "").split(",") if value],
                episode_id=episode_id or None,
                mode=mode or None,
                evaluation_type=evaluation_type or None,
                condition_id=condition_id or None,
                base_path=base_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/image")
    def image(path: str):
        target = Path(path).resolve()
        if target not in app.state.allowed_images or not target.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(target)

    @router.post("/api/import")
    async def import_files(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            result = import_evaluation_files(
                app.state.evaluation_dir,
                jsonl_name=str(body["jsonl_name"]),
                jsonl_content=str(body["jsonl_content"]),
                summary_name=str(body["summary_name"]),
                summary_content=str(body["summary_content"]),
            )
            app.state.allowed_images = evaluation_image_paths(app.state.evaluation_dir)
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    app.include_router(router, prefix=base_path)
    if base_path:
        # Some ingress configurations strip the public prefix before proxying,
        # and many health checks probe the upstream root path directly.
        app.include_router(router)
    return app


def serve_evaluation_reply_app(
    *,
    evaluation_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8771,
    base_path: str = "",
    open_browser: bool = False,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install fastapi and uvicorn to use the evaluation reply UI") from exc
    base_path = normalize_base_path(base_path)
    app = create_evaluation_reply_app(evaluation_dir=evaluation_dir, base_path=base_path)
    url = f"http://{host}:{port}{base_path}/"
    print(f"Evaluation reply UI running at {url}")
    print(f"Evaluation directory: {app.state.evaluation_dir}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port)


def _render_evaluation_reply_html(*, base_path: str = "") -> str:
    html = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Real Brain Evaluation</title>
<style>
:root{--bg:#f6f7fa;--surface:#fff;--surface-2:#f0f2f6;--border:#dfe3ea;--border-strong:#c5cbd5;--text:#20242c;--muted:#687181;--accent:#3f5fbd;--green:#16794a;--green-bg:#e8f6ef;--red:#b4232f;--red-bg:#fdecee;--amber:#9a5a06;--amber-bg:#fff3d9;--radius:6px;--mono:"SFMono-Regular",Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select,input{font:inherit}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #8298dc;outline-offset:1px}
.topbar{height:54px;background:var(--surface);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 22px;position:sticky;top:0;z-index:20}.brand{font-weight:720;font-size:17px}.brand small{font-weight:500;color:var(--muted);margin-left:10px}.top-actions{display:flex;gap:8px;align-items:center}.count{font:12px var(--mono);color:var(--muted)}
.btn{height:34px;border:1px solid var(--border-strong);background:var(--surface);color:var(--text);padding:0 13px;border-radius:5px;cursor:pointer}.btn:hover{border-color:var(--accent);color:var(--accent)}.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}.btn:disabled{opacity:.5;cursor:not-allowed}
.filters{background:var(--surface);border-bottom:1px solid var(--border);padding:14px 22px;display:grid;grid-template-columns:minmax(220px,1.2fr) minmax(170px,1fr) minmax(160px,.9fr) minmax(150px,.8fr) minmax(190px,1fr) auto;gap:12px;align-items:end;position:sticky;top:54px;z-index:19}.field label{display:block;font-size:12px;font-weight:650;color:var(--muted);margin-bottom:5px}.field select{height:36px;width:100%;border:1px solid var(--border-strong);border-radius:5px;background:#fff;padding:0 32px 0 10px;color:var(--text)}.filter-status{height:36px;display:flex;align-items:center;white-space:nowrap;color:var(--muted);font:12px var(--mono)}
.model-picker{position:relative}.model-trigger{height:36px;width:100%;border:1px solid var(--border-strong);border-radius:5px;background:#fff;padding:0 10px;display:flex;align-items:center;justify-content:space-between;gap:10px;cursor:pointer;text-align:left}.model-trigger::after{content:"▾";color:var(--muted)}.model-menu{display:none;position:absolute;left:0;right:0;top:41px;background:var(--surface);border:1px solid var(--border-strong);border-radius:5px;box-shadow:0 12px 28px rgba(29,37,52,.16);z-index:40;max-height:280px;overflow:auto}.model-menu.open{display:block}.model-menu-actions{display:flex;gap:6px;padding:8px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--surface)}.model-menu-actions button{border:0;background:transparent;color:var(--accent);font-size:12px;cursor:pointer;padding:3px 5px}.model-option{display:flex;align-items:center;gap:8px;padding:8px 10px;cursor:pointer;font:12px var(--mono)}.model-option:hover{background:var(--surface-2)}.model-option input{width:15px;height:15px;accent-color:var(--accent)}
.main{max-width:1560px;margin:0 auto;padding:18px 22px 40px}.notice{display:none;border:1px solid var(--border);background:var(--surface);padding:12px 14px;margin-bottom:14px;border-radius:var(--radius)}.notice.error{display:block;color:var(--red);background:var(--red-bg);border-color:#f4b8bd}.empty{padding:64px 20px;text-align:center;color:var(--muted);background:var(--surface);border:1px solid var(--border);border-radius:var(--radius)}
.records{display:flex;flex-direction:column;gap:10px}.record{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}.record.has-error{border-left:3px solid var(--red)}.record-head{min-height:42px;background:#fafbfc;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;padding:7px 11px}.step{font:700 13px var(--mono);min-width:62px}.tag{font:11px var(--mono);padding:2px 6px;border-radius:4px;background:var(--surface-2);color:var(--muted);border:1px solid var(--border)}.tag.good{color:var(--green);background:var(--green-bg);border-color:#b8dfca}.tag.bad{color:var(--red);background:var(--red-bg);border-color:#f1bdc1}.record-file{color:var(--muted);font:10px/1.4 var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.record-body{display:grid;grid-template-columns:minmax(270px,350px) minmax(0,1fr);min-height:250px}.visuals{padding:12px}.results-pane{min-width:0;border-left:1px solid var(--border);background:#fafbfc}.model-results-scroll{width:100%;overflow-x:scroll;overflow-y:hidden;padding:12px 12px 14px;scrollbar-gutter:stable;scrollbar-color:#8793a7 #e4e7ed;scrollbar-width:auto}.model-results-scroll::-webkit-scrollbar{height:12px}.model-results-scroll::-webkit-scrollbar-track{background:#e4e7ed;border-radius:8px}.model-results-scroll::-webkit-scrollbar-thumb{background:#8793a7;border:2px solid #e4e7ed;border-radius:8px}.model-results-scroll::-webkit-scrollbar-thumb:hover{background:#66758d}.model-results{display:flex;align-items:stretch;gap:10px;width:max-content;min-width:100%;padding-bottom:2px}.model-result{width:360px;min-width:360px;background:var(--surface);border:1px solid var(--border);border-radius:5px;overflow:hidden;display:flex;flex-direction:column}.model-result.error{border-color:#efb5ba}.model-result-head{min-height:44px;padding:8px 10px;background:var(--surface-2);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:6px;flex-wrap:wrap}.model-name{font:700 12px var(--mono);margin-right:auto;overflow-wrap:anywhere}.model-result-body{display:flex;flex-direction:column;flex:1}.reason,.decision{padding:11px}.decision{border-top:1px solid var(--border);margin-top:auto}.section-label{font-size:11px;text-transform:uppercase;font-weight:750;color:var(--muted);margin-bottom:8px}.image-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.frame{margin:0;background:#111;aspect-ratio:16/9;overflow:hidden;border-radius:4px;border:1px solid #cfd4dc;cursor:zoom-in}.frame img{width:100%;height:100%;object-fit:contain;display:block}.frame.broken{background:var(--surface-2);display:grid;place-items:center;color:var(--muted);font-size:12px}.frame-meta{margin-top:8px;color:var(--muted);font:11px/1.5 var(--mono)}.reason-text{white-space:pre-wrap;overflow-wrap:anywhere;color:#303641}.error-text{margin-top:10px;padding:8px;background:var(--red-bg);color:var(--red);border-radius:4px;font:12px var(--mono)}
.compare{display:grid;grid-template-columns:74px minmax(0,1fr);gap:7px 9px;align-items:start;margin-bottom:13px}.compare dt{color:var(--muted);font-size:12px}.compare dd{margin:0;font:12px/1.55 var(--mono);overflow-wrap:anywhere}.action-match{color:var(--green)}.action-miss{color:var(--red)}.recovery-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.recovery-pill{padding:2px 6px;border-radius:4px;font:11px var(--mono);background:var(--surface-2);border:1px solid var(--border)}.recovery-pill.yes{background:var(--amber-bg);color:var(--amber);border-color:#edcf98}.recovery-pill.no{background:var(--green-bg);color:var(--green);border-color:#b8dfca}
.metrics{margin-top:22px;padding-top:18px;border-top:2px solid var(--border-strong)}.metrics-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:10px}.metrics h2{font-size:16px;margin:0}.metrics-context{color:var(--muted);font:12px var(--mono)}.metric-facet{margin-top:20px}.metric-facet:first-child{margin-top:0}.metric-facet-title{font:700 14px var(--mono);margin:0 0 10px}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:10px}.metric-chart{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;min-width:0}.metric-chart h3{font-size:13px;margin:0 0 11px}.metric-bars{display:flex;flex-direction:column;gap:10px}.metric-episode{display:flex;flex-direction:column;gap:6px;padding-top:9px;border-top:1px solid var(--border)}.metric-episode:first-child{padding-top:0;border-top:0}.metric-episode-name{font:700 11px var(--mono);color:var(--text);overflow-wrap:anywhere}.metric-bar-row{display:grid;grid-template-columns:minmax(120px,170px) minmax(90px,1fr) 54px;gap:8px;align-items:center}.metric-series{font:11px var(--mono);color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.bar-track{height:10px;background:var(--surface-2);border-radius:3px;overflow:hidden}.bar-fill{height:100%;width:var(--bar-width);background:var(--bar-color);border-radius:3px;min-width:0}.metric-value{text-align:right;font:11px var(--mono)}.metric-config{margin-top:10px;color:var(--muted);font:11px/1.6 var(--mono)}
dialog{border:0;border-radius:7px;padding:0;box-shadow:0 20px 60px rgba(20,28,45,.25);max-width:min(560px,calc(100vw - 28px));width:100%}dialog::backdrop{background:rgba(24,30,42,.46)}.dialog-head{padding:15px 18px;border-bottom:1px solid var(--border);font-weight:700;display:flex;justify-content:space-between}.dialog-body{padding:18px}.dialog-field{margin-bottom:14px}.dialog-field label{display:block;font-size:12px;font-weight:650;margin-bottom:5px}.dialog-field input{width:100%;border:1px solid var(--border-strong);padding:8px;border-radius:5px}.dialog-actions{display:flex;justify-content:flex-end;gap:8px;padding:12px 18px;border-top:1px solid var(--border)}.image-dialog{max-width:min(1100px,calc(100vw - 30px));background:#111}.image-dialog img{display:block;width:100%;max-height:88vh;object-fit:contain}.image-close{position:absolute;right:10px;top:10px;background:rgba(0,0,0,.62);color:#fff;border-color:#777}
@media(max-width:980px){.record-body{grid-template-columns:minmax(240px,300px) minmax(0,1fr)}.model-result{width:330px;min-width:330px}.filters{grid-template-columns:1fr 1fr}.filter-status{display:none}}
@media(max-width:680px){.topbar{padding:0 12px}.brand small,.count{display:none}.filters{position:static;grid-template-columns:1fr;padding:12px}.main{padding:12px}.record-body{display:block}.results-pane{border-left:0;border-top:1px solid var(--border)}.model-result{width:min(360px,calc(100vw - 64px));min-width:min(360px,calc(100vw - 64px))}.record-head{flex-wrap:wrap}.image-grid{grid-template-columns:1fr}.metrics-head{display:block}.metrics-context{margin-top:4px}}
</style>
</head>
<body>
<header class="topbar"><div class="brand">Real Brain Evaluation <small>trajectory replies</small></div><div class="top-actions"><span id="sourceCount" class="count"></span><button id="openImport" class="btn">导入结果</button></div></header>
<section class="filters">
  <div class="field"><label for="modelTrigger">模型</label><div class="model-picker"><button id="modelTrigger" class="model-trigger" type="button">全部模型</button><div id="modelMenu" class="model-menu"><div class="model-menu-actions"><button id="selectAllModels" type="button">全选</button><button id="clearModels" type="button">清空</button></div><div id="modelOptions"></div></div></div></div>
  <div class="field"><label for="episodeFilter">Episode</label><select id="episodeFilter"><option value="">全部 Episode</option></select></div>
  <div class="field"><label for="evaluationTypeFilter">Evaluation</label><select id="evaluationTypeFilter"><option value="">全部 Evaluation</option></select></div>
  <div class="field"><label for="modeFilter">Mode</label><select id="modeFilter"><option value="">全部 Mode</option></select></div>
  <div class="field"><label for="conditionFilter">Condition</label><select id="conditionFilter"><option value="">全部 Condition</option></select></div>
  <div id="filterStatus" class="filter-status"></div>
</section>
<main class="main">
  <div id="notice" class="notice"></div>
  <section id="records" class="records"></section>
  <section id="metrics" class="metrics"></section>
</main>
<dialog id="importDialog">
  <div class="dialog-head"><span>导入评测结果</span><button class="btn" data-close="importDialog">关闭</button></div>
  <div class="dialog-body">
    <div class="dialog-field"><label for="jsonlFile">Evaluation JSONL</label><input id="jsonlFile" type="file" accept=".jsonl,application/json"></div>
    <div class="dialog-field"><label for="summaryFile">Summary JSON</label><input id="summaryFile" type="file" accept=".json,application/json"></div>
    <div id="importError" class="error-text" style="display:none"></div>
  </div>
  <div class="dialog-actions"><button class="btn" data-close="importDialog">取消</button><button id="importFiles" class="btn primary">导入</button></div>
</dialog>
<dialog id="imageDialog" class="image-dialog"><button class="btn image-close" data-close="imageDialog">关闭</button><img id="largeImage" alt="Observation frame"></dialog>
<script>
const BASE_PATH=__BASE_PATH_JSON__;
const apiPath=path=>`${BASE_PATH}${path}`;
const state={catalog:null,payload:null,selectedModels:new Set(),modelsInitialized:false};
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>'"]/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch]));
const pct=value=>value==null?"—":`${(Number(value)*100).toFixed(1)}%`;
const fetchJSON=async(url,options)=>{const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(data.detail||data.error||response.statusText);return data};
function fillSelect(select,values,label){const current=select.value;select.innerHTML=`<option value="">全部 ${esc(label)}</option>`+values.map(value=>`<option value="${esc(value)}">${esc(value)}</option>`).join("");if(values.includes(current))select.value=current}
function updateModelTrigger(){const selected=[...state.selectedModels];const total=state.catalog?.model_names.length||0;$("modelTrigger").textContent=selected.length===total&&total?`全部模型 (${total})`:selected.length===0?"未选择模型":selected.length===1?selected[0]:`已选 ${selected.length} 个模型`}
function renderModelPicker(selectAll=false){const models=state.catalog.model_names||[];if(selectAll||!state.modelsInitialized){state.selectedModels=new Set(models);state.modelsInitialized=true}else{state.selectedModels=new Set([...state.selectedModels].filter(model=>models.includes(model)))}$("modelOptions").innerHTML=models.map(model=>`<label class="model-option"><input type="checkbox" value="${esc(model)}" ${state.selectedModels.has(model)?"checked":""}><span>${esc(model)}</span></label>`).join("");$("modelOptions").querySelectorAll("input").forEach(input=>input.addEventListener("change",()=>{input.checked?state.selectedModels.add(input.value):state.selectedModels.delete(input.value);updateModelTrigger();loadReplies().catch(showError)}));updateModelTrigger()}
async function loadCatalog(selectAll=false){state.catalog=await fetchJSON(apiPath("/api/catalog"));renderModelPicker(selectAll);fillSelect($("episodeFilter"),state.catalog.episode_ids,"Episode");fillSelect($("evaluationTypeFilter"),state.catalog.evaluation_types||[],"Evaluation");fillSelect($("modeFilter"),state.catalog.modes,"Mode");fillSelect($("conditionFilter"),state.catalog.condition_ids||[],"Condition");$("sourceCount").textContent=`${state.catalog.source_count} files · ${state.catalog.record_count} records`;await loadReplies()}
async function loadReplies(){if(state.selectedModels.size===0){state.payload={record_count:0,records:[],summaries:[],metric_groups:[]};$("filterStatus").textContent="0 steps";renderRecords();renderMetrics();return}const params=new URLSearchParams();params.set("model_names",[...state.selectedModels].join(","));[["episode_id",$("episodeFilter").value],["evaluation_type",$("evaluationTypeFilter").value],["mode",$("modeFilter").value],["condition_id",$("conditionFilter").value]].forEach(([key,value])=>{if(value)params.set(key,value)});$("filterStatus").textContent="加载中";state.payload=await fetchJSON(apiPath(`/api/replies?${params}`));const stepCount=new Set((state.payload.records||[]).map(recordGroupKey)).size;$("filterStatus").textContent=`${stepCount} steps · ${state.payload.record_count} results`;renderRecords();renderMetrics()}
function actionText(action){if(!action)return"—";const nodes=Array.isArray(action.node_ids)?action.node_ids.join(" → "):"";return `${action.name||action.base_name||"unknown"}${nodes?`(${nodes})`:""}`}
function recoveryHTML(value){if(!value)return'<span class="recovery-pill">未输出</span>';const required=value.required===true;const nodes=Array.isArray(value.failed_node_ids)&&value.failed_node_ids.length?`<span class="tag">${esc(value.failed_node_ids.join(" → "))}</span>`:"";return `<span class="recovery-pill ${required?"yes":"no"}">${required?"需要恢复":"无需恢复"}</span>${value.failed_action?`<span class="tag">${esc(value.failed_action)}</span>`:""}${nodes}`}
function recordGroupKey(record){return [record.evaluation_type||"open_loop_real",record.source_file||"",record.episode_id||"",record.step??"",record.mode||"",record.condition_id||""].join("\u001f")}
function groupRecords(records){const groups=new Map();records.forEach(record=>{const key=recordGroupKey(record);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(record)});const modelOrder=new Map([...state.selectedModels].map((model,index)=>[model,index]));return [...groups.values()].map(group=>group.sort((left,right)=>(modelOrder.get(left.model_name)??9999)-(modelOrder.get(right.model_name)??9999)||String(left.result_file||"").localeCompare(String(right.result_file||""))))}
function renderModelResult(record){if(record.evaluation_type==="closed_loop_visible_graph"){const eventGood=record.event?.status==="success";const goal=record.goal_satisfied_after_action===true;const eventText=record.event?`${record.event.status||"unknown"}${record.event.failure_type?` · ${record.event.failure_type}`:""}`:"—";const disturbances=record.disturbances_applied||[];const disturbanceTag=disturbances.length?`<span class="tag">干扰 ${disturbances.length}</span>`:"";const disturbanceBlock=disturbances.length?`<div class="section-label">External graph disturbance</div><pre class="reason-text">${esc(JSON.stringify(disturbances,null,2))}</pre>`:"";return `<article class="model-result ${record.model_error?"error":""}"><header class="model-result-head"><span class="model-name">${esc(record.model_name)}</span><span class="tag ${eventGood?"good":"bad"}">Event ${eventGood?"成功":"失败"}</span><span class="tag ${goal?"good":""}">Goal ${goal?"满足":"未满足"}</span>${record.injection_applied?'<span class="tag bad">Injected failure</span>':""}${disturbanceTag}<div class="record-file" title="${esc(record.result_file)}">${esc(record.result_file)}</div></header><div class="model-result-body"><section class="reason"><div class="section-label">Reason</div><div class="reason-text">${esc(record.reason||"未输出 reason")}</div>${record.model_error?`<div class="error-text">${esc(record.model_error)}</div>`:""}${record.parse_error?`<div class="error-text">${esc(record.parse_error)}</div>`:""}${disturbanceBlock}</section><section class="decision"><div class="section-label">Closed-loop transition</div><dl class="compare"><dt>Action</dt><dd>${esc(actionText(record.predicted_action))}</dd><dt>Event</dt><dd class="${eventGood?"action-match":"action-miss"}">${esc(eventText)}</dd><dt>Goal cost</dt><dd>${esc(record.relaxed_completion_cost_before??"—")} → ${esc(record.relaxed_completion_cost_after??"—")}</dd></dl><div class="section-label">Failure recovery</div><dl class="compare"><dt>Expected</dt><dd class="recovery-row">${recoveryHTML(record.expected_recovery)}</dd><dt>Predicted</dt><dd class="recovery-row">${recoveryHTML(record.predicted_recovery)}</dd></dl></section></div></article>`}const exact=record.score?.full_exact===true;const groundingExact=record.recovery_score?.grounding_exact;const recoveryExact=groundingExact===true||(groundingExact==null&&record.recovery_score?.full_exact===true);return `<article class="model-result ${record.model_error?"error":""}"><header class="model-result-head"><span class="model-name">${esc(record.model_name)}</span><span class="tag ${exact?"good":"bad"}">Action ${exact?"正确":"错误"}</span><span class="tag ${recoveryExact?"good":"bad"}">Recovery ${recoveryExact?"正确":"错误"}</span><div class="record-file" title="${esc(record.result_file)}">${esc(record.result_file)}</div></header><div class="model-result-body"><section class="reason"><div class="section-label">Reason</div><div class="reason-text">${esc(record.reason||"未输出 reason")}</div>${record.model_error?`<div class="error-text">${esc(record.model_error)}</div>`:""}${record.parse_error?`<div class="error-text">${esc(record.parse_error)}</div>`:""}</section><section class="decision"><div class="section-label">Action</div><dl class="compare"><dt>Expected</dt><dd>${esc(actionText(record.expected_action))}</dd><dt>Predicted</dt><dd class="${exact?"action-match":"action-miss"}">${esc(actionText(record.predicted_action))}</dd></dl><div class="section-label">Failure recovery</div><dl class="compare"><dt>Expected</dt><dd class="recovery-row">${recoveryHTML(record.expected_recovery)}</dd><dt>Predicted</dt><dd class="recovery-row">${recoveryHTML(record.predicted_recovery)}</dd></dl></section></div></article>`}
function renderRecords(){const root=$("records");const records=state.payload.records||[];if(!records.length){root.innerHTML='<div class="empty">没有符合当前筛选条件的记录</div>';return}const groups=groupRecords(records);root.innerHTML=groups.map(group=>{const shared=group.find(record=>(record.image_urls||[]).length)||group[0];const images=(shared.image_urls||[]).map((url,index)=>`<figure class="frame" data-image="${esc(url)}"><img src="${esc(url)}" alt="Observation ${index+1}" loading="lazy" onerror="this.parentElement.classList.add('broken');this.remove();this.parentElement.textContent='图片不可用'"></figure>`).join("");const closed=shared.evaluation_type==="closed_loop_visible_graph";const graph=closed?`<pre class="reason-text">${esc(JSON.stringify(shared.current_observation||{},null,2))}</pre>`:"";const frame=shared.frame_observation||shared.request_summary?.frame_observation||{};const hasError=group.some(record=>record.model_error);const modelCount=new Set(group.map(record=>record.model_name)).size;const conditionTag=shared.condition_id?`<span class="tag">${esc(shared.condition_id)}</span>`:"";const observation=closed?graph:`<div class="image-grid">${images||'<div class="frame broken">无图片</div>'}</div><div class="frame-meta">source step ${esc(frame.source_step??"—")} · episode ${esc((frame.source_episode_indices||[]).join(",")||"—")} · ${esc(frame.applied_sampling||"—")}</div>`;return `<article class="record ${hasError?"has-error":""}"><header class="record-head"><span class="step">STEP ${esc(shared.step)}</span><span class="tag">${esc(shared.episode_id)}</span><span class="tag">${esc(shared.evaluation_type||"open_loop_real")}</span><span class="tag">${esc(shared.mode)}</span>${conditionTag}<span class="tag">${modelCount} models · ${group.length} results</span></header><div class="record-body"><section class="visuals"><div class="section-label">Shared observation</div>${observation}</section><section class="results-pane"><div class="model-results-scroll" aria-label="模型结果横向对比"><div class="model-results">${group.map(renderModelResult).join("")}</div></div></section></div></article>`}).join("");root.querySelectorAll("[data-image]").forEach(node=>node.addEventListener("click",()=>{$("largeImage").src=node.dataset.image;$("imageDialog").showModal()}))}
const openLoopMetrics=["action_admissibility_rate","soft_optimal_action_score","recovery_detection_f1","recovery_grounding_accuracy","exploration_opportunity_recall","normalized_goal_information_gain","premature_stop_rate","completion_stop_recall"];
const closedLoopMetrics=["task_success_rate","goal_ever_satisfied_rate","final_goal_satisfied_rate","normalized_goal_progress","teacher_normalized_efficiency","action_executability_rate","average_step_count","episodes_with_injected_failure_rate","average_injected_failure_count","episodes_with_disturbance_rate","average_disturbance_count",...openLoopMetrics];
const metricLabels={task_success_rate:"闭环结果 · 正确 Stop 成功率",goal_ever_satisfied_rate:"闭环结果 · 曾达成目标率",final_goal_satisfied_rate:"闭环结果 · 最终目标满足率",normalized_goal_progress:"闭环结果 · 归一化目标进展",teacher_normalized_efficiency:"闭环结果 · Teacher 归一化效率",action_executability_rate:"闭环诊断 · 动作可执行率",average_step_count:"闭环诊断 · 平均步数",episodes_with_injected_failure_rate:"干预诊断 · Failure episode 覆盖率",average_injected_failure_count:"干预诊断 · 平均注入 Failure 数",episodes_with_disturbance_rate:"干预诊断 · Graph 干扰 episode 覆盖率",average_disturbance_count:"干预诊断 · 平均 Graph 干扰数",action_admissibility_rate:"动作合理性 · 可接受动作率",soft_optimal_action_score:"动作合理性 · Soft 最优动作分",recovery_detection_f1:"失败恢复 · Recovery 检测 F1",recovery_grounding_accuracy:"失败恢复 · 对象 Grounding 准确率",exploration_opportunity_recall:"主动探索 · 机会召回率",normalized_goal_information_gain:"主动探索 · 归一化目标信息增益",premature_stop_rate:"完成判断 · 提前停止率（越低越好）",completion_stop_recall:"完成判断 · 完成停止召回率"};
function renderMetricFacet(type,columns,selectedEpisode,modelColor){const episodes=[...new Set(columns.map(column=>column.episode_id))];const keys=type==="closed_loop_visible_graph"?closedLoopMetrics:openLoopMetrics;const charts=keys.map(key=>{const isCount=key.endsWith("count");const maximum=isCount?Math.max(1,...columns.map(column=>Number(column.metrics[key]||0))):1;const episodeGroups=episodes.map(episode=>{const episodeColumns=columns.filter(column=>column.episode_id===episode);const rows=episodeColumns.map(column=>{const raw=column.metrics[key];const value=raw==null?0:Number(raw);const width=Math.max(0,Math.min(100,value/maximum*100));const display=isCount?(raw??"—"):pct(raw);const failure=column.config?.failure_injection;const condition=column.condition_id;const name=`${column.model_name} · ${column.mode}${condition?` · ${condition}`:""}${failure&&failure!=="none"?` · ${failure}`:""}`;return `<div class="metric-bar-row" title="${esc(episode)} · ${esc(name)}: ${esc(display)}"><div class="metric-series">${esc(name)}</div><div class="bar-track"><div class="bar-fill" style="--bar-width:${width}%;--bar-color:${modelColor(column.model_name)}"></div></div><div class="metric-value">${esc(display)}</div></div>`}).join("");return `<section class="metric-episode">${selectedEpisode?"":`<div class="metric-episode-name">${esc(episode)}</div>`}${rows}</section>`}).join("");return `<article class="metric-chart"><h3>${esc(metricLabels[key]||key)}</h3><div class="metric-bars">${episodeGroups}</div></article>`}).join("");const title=type==="closed_loop_visible_graph"?"Closed-loop · Visible Graph":"Open-loop · Real Observation";return `<section class="metric-facet"><h3 class="metric-facet-title">${esc(title)}</h3><div class="metric-grid">${charts}</div></section>`}
function renderMetrics(){const root=$("metrics");const columns=state.payload.metric_groups||[];if(!columns.length){root.innerHTML='<div class="metrics-head"><h2>Episode metrics</h2></div><div class="empty">当前筛选范围没有可计算的指标</div>';return}const selectedEpisode=$("episodeFilter").value;const modelOrder=new Map([...state.selectedModels].map((model,index)=>[model,index]));const colors=["#3f5fbd","#16794a","#c27412","#b4232f","#087f8c","#7651a8"];const modelColor=model=>colors[(modelOrder.get(model)??0)%colors.length];const types=[...new Set(columns.map(column=>column.evaluation_type||"open_loop_real"))];const facets=types.map(type=>renderMetricFacet(type,columns.filter(column=>(column.evaluation_type||"open_loop_real")===type),selectedEpisode,modelColor)).join("");const configs=columns.map(column=>{const config=column.config||{};return `${column.episode_id} · ${column.model_name} · ${column.mode}: type=${column.evaluation_type||"open_loop_real"}, history=${config.history_source||"—"}, valid_actions=${config.includes_valid_actions!==false}, failure=${config.failure_injection||"none"}, disturbance=${config.graph_disturbance_file||"none"}`}).join(" · ");const episodeCount=new Set(columns.map(column=>column.episode_id)).size;const scope=selectedEpisode?selectedEpisode:`${episodeCount} episodes`;root.innerHTML=`<div class="metrics-head"><h2>Episode metrics</h2><div class="metrics-context">${esc(scope)} · ${columns.length} result groups</div></div>${facets}<div class="metric-config">${esc(configs)}</div>`}
[$("episodeFilter"),$("evaluationTypeFilter"),$("modeFilter"),$("conditionFilter")].forEach(select=>select.addEventListener("change",()=>loadReplies().catch(showError)));$("modelTrigger").addEventListener("click",event=>{event.stopPropagation();$("modelMenu").classList.toggle("open")});$("modelMenu").addEventListener("click",event=>event.stopPropagation());document.addEventListener("click",()=>$("modelMenu").classList.remove("open"));$("selectAllModels").addEventListener("click",()=>{state.selectedModels=new Set(state.catalog.model_names);renderModelPicker();loadReplies().catch(showError)});$("clearModels").addEventListener("click",()=>{state.selectedModels.clear();renderModelPicker();loadReplies().catch(showError)});
function showError(error){const notice=$("notice");notice.textContent=error.message;notice.className="notice error"}
$("openImport").addEventListener("click",()=>$("importDialog").showModal());document.querySelectorAll("[data-close]").forEach(button=>button.addEventListener("click",()=>$(button.dataset.close).close()));
$("importFiles").addEventListener("click",async()=>{const jsonl=$("jsonlFile").files[0],summary=$("summaryFile").files[0],error=$("importError");error.style.display="none";if(!jsonl||!summary){error.textContent="请选择 JSONL 和 summary 文件";error.style.display="block";return}const button=$("importFiles");button.disabled=true;try{await fetchJSON(apiPath("/api/import"),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({jsonl_name:jsonl.name,jsonl_content:await jsonl.text(),summary_name:summary.name,summary_content:await summary.text()})});$("importDialog").close();$("jsonlFile").value="";$("summaryFile").value="";await loadCatalog(true)}catch(err){error.textContent=err.message;error.style.display="block"}finally{button.disabled=false}});
loadCatalog(true).catch(showError);
</script>
</body>
</html>"""
    return html.replace("__BASE_PATH_JSON__", json.dumps(normalize_base_path(base_path)))
