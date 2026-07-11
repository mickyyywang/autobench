from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser


def serve_trajectory_app(
    trajectory_path: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8766,
    *,
    base_path: str = "",
    open_browser: bool = False,
) -> None:
    server = ThreadingHTTPServer((host, port), _TrajectoryAppHandler)
    root = _trajectory_root(trajectory_path, trajectory_dir)
    base_path = _normalize_base_path(base_path)
    server.trajectory_dir = root  # type: ignore[attr-defined]
    server.trajectory_path = _initial_trajectory_path(trajectory_path, root)  # type: ignore[attr-defined]
    server.base_path = base_path  # type: ignore[attr-defined]
    url_path = f"{base_path}/" if base_path else "/"
    url = f"http://{host}:{server.server_port}{url_path}"
    print(f"Trajectory replay UI running at {url}")
    print(f"Trajectory directory: {root}")
    if server.trajectory_path is not None:  # type: ignore[attr-defined]
        print(f"Initial trajectory file: {server.trajectory_path}")  # type: ignore[attr-defined]
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping trajectory replay UI")
    finally:
        server.server_close()


class _TrajectoryAppHandler(BaseHTTPRequestHandler):
    server_version = "AutoEmbodiedTaskTrajectoryUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        base_path = getattr(self.server, "base_path", "")
        if base_path and parsed.path == base_path:
            suffix = f"?{parsed.query}" if parsed.query else ""
            self._redirect(f"{base_path}/{suffix}")
            return
        route_path = _route_path(parsed.path, base_path)
        if route_path is None:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if route_path in {"/", "/index.html"}:
            self._send_text(_render_trajectory_html(base_path), content_type="text/html; charset=utf-8")
            return
        if route_path == "/api/trajectory-files":
            self._handle_trajectory_files()
            return
        if route_path == "/api/trajectories":
            self._handle_trajectories(parsed.query)
            return
        if route_path == "/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_trajectory_files(self) -> None:
        root = getattr(self.server, "trajectory_dir", default_trajectory_dir())
        selected = getattr(self.server, "trajectory_path", None)
        try:
            payload = trajectory_file_listing(root, selected)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload)

    def _handle_trajectories(self, query: str = "") -> None:
        root = getattr(self.server, "trajectory_dir", default_trajectory_dir())
        params = parse_qs(query)
        requested_file = params.get("file", [None])[0]
        path = _resolve_requested_trajectory_path(requested_file, root) if requested_file else getattr(
            self.server,
            "trajectory_path",
            None,
        )
        if path is None:
            files = trajectory_files(root)
            path = files[0]["path"] if files else None
        if path is None:
            self._send_json({"error": f"no trajectory JSONL files found in {root}"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = trajectory_replay_payload(path)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload)

    def _send_text(self, body: str, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, body: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _redirect(self, location: str, status: HTTPStatus = HTTPStatus.FOUND) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _normalize_base_path(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text == "/":
        return ""
    return "/" + text.strip("/")


def _route_path(path: str, base_path: str) -> str | None:
    if not path:
        path = "/"
    if not base_path:
        return path
    if path == base_path:
        return "/"
    if path.startswith(f"{base_path}/"):
        return path[len(base_path) :] or "/"
    return None


def trajectory_replay_payload(path: str | Path) -> dict[str, Any]:
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
            episodes.append(_episode_to_replay(episode, len(episodes)))
    return {
        "trajectory_path": str(source),
        "trajectory_file": source.name,
        "episode_count": len(episodes),
        "episodes": episodes,
    }


def default_trajectory_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs"


def _trajectory_root(trajectory_path: str | Path | None, trajectory_dir: str | Path | None) -> Path:
    if trajectory_dir is not None:
        return Path(trajectory_dir)
    if trajectory_path is not None:
        return Path(trajectory_path).parent
    return default_trajectory_dir()


def trajectory_files(root: str | Path) -> list[dict[str, Any]]:
    directory = Path(root)
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError(f"trajectory directory is not a directory: {directory}")
    files = []
    for path in directory.glob("*.jsonl"):
        if not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size == 0:
            continue
        if not _looks_like_trajectory_jsonl(path):
            continue
        files.append(
            {
                "name": path.name,
                "path": path,
                "display_path": str(path),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    files.sort(key=lambda item: (-float(item["mtime"]), str(item["name"])))
    return files


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


def trajectory_file_listing(root: str | Path, selected: str | Path | None = None) -> dict[str, Any]:
    directory = Path(root)
    selected_path = Path(selected).resolve() if selected is not None else None
    files = []
    for item in trajectory_files(directory):
        path = Path(item["path"])
        files.append(
            {
                "name": item["name"],
                "path": item["display_path"],
                "size_bytes": item["size_bytes"],
                "mtime": item["mtime"],
                "selected": bool(selected_path is not None and path.resolve() == selected_path),
            }
        )
    selected_name = next((item["name"] for item in files if item["selected"]), None)
    if selected_name is None and files:
        selected_name = str(files[0]["name"])
        files[0]["selected"] = True
    return {
        "trajectory_dir": str(directory),
        "selected": selected_name,
        "files": files,
    }


def _initial_trajectory_path(path: str | Path | None, root: Path) -> Path | None:
    if path is not None:
        requested = Path(path)
        candidate = requested.resolve() if requested.is_absolute() else requested.resolve()
        directory = Path(root).resolve()
        if candidate.exists():
            try:
                candidate.relative_to(directory)
            except ValueError as exc:
                raise ValueError(f"trajectory file must be inside {directory}") from exc
            return candidate
        return _resolve_requested_trajectory_path(str(requested.name), root)
    files = trajectory_files(root)
    return Path(files[0]["path"]) if files else None


def _resolve_requested_trajectory_path(raw_path: str | None, root: str | Path) -> Path:
    if raw_path is None or not str(raw_path).strip():
        raise ValueError("trajectory file is required")
    directory = Path(root).resolve()
    requested = Path(str(raw_path))
    candidate = requested.resolve() if requested.is_absolute() else (directory / requested).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise ValueError(f"trajectory file must be inside {directory}") from exc
    if not candidate.exists():
        raise ValueError(f"trajectory file does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"trajectory path is not a file: {candidate}")
    if candidate.suffix != ".jsonl":
        raise ValueError(f"trajectory file must be .jsonl: {candidate}")
    return candidate


def _episode_to_replay(episode: dict[str, Any], index: int) -> dict[str, Any]:
    episode_id = str(episode.get("episode_id") or episode.get("task_id") or f"episode_{index + 1}")
    frames = []
    initial_observation = episode.get("initial_observation")
    if isinstance(initial_observation, dict):
        frames.append(
            _frame_from_observation(
                episode=episode,
                observation=initial_observation,
                frame_id=f"{episode_id}_initial",
                label="Initial observation",
                kind="initial",
            )
        )
    for step_index, step in enumerate(episode.get("trajectory", []) or [], start=1):
        if not isinstance(step, dict):
            continue
        observation = step.get("post_observation") if isinstance(step.get("post_observation"), dict) else step.get("observation")
        if not isinstance(observation, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        action_name = str(action.get("name") or action.get("base_name") or "action")
        frame = _frame_from_observation(
            episode=episode,
            observation=observation,
            frame_id=f"{episode_id}_step_{step.get('step', step_index)}",
            label=f"Step {step.get('step', step_index)} after {action_name}",
            kind="post_action",
        )
        frame.update(
            {
                "step": step.get("step", step_index),
                "action": action,
                "requested_action": step.get("requested_action"),
                "event": step.get("event"),
                "teacher_response": step.get("teacher_response"),
                "failure_injection": step.get("failure_injection"),
                "new_visible_nodes": step.get("new_visible_nodes", []),
                "success_after_step": step.get("success_after_step"),
            }
        )
        frames.append(frame)
    return {
        "index": index,
        "episode_id": episode_id,
        "scene_id": episode.get("scene_id"),
        "env_id": episode.get("env_id"),
        "mode": episode.get("mode"),
        "task_type": episode.get("task_type"),
        "task": episode.get("task"),
        "success": episode.get("success"),
        "initial_view_graph": _initial_view_graph_for_episode(episode, episode_id),
        "frame_count": len(frames),
        "frames": frames,
    }


def _initial_view_graph_for_episode(episode: dict[str, Any], episode_id: str) -> dict[str, Any]:
    explicit_graph = episode.get("initial_view_graph")
    if isinstance(explicit_graph, dict):
        return _normalize_view_graph(
            explicit_graph,
            default_scene_id=str(episode.get("scene_id") or episode_id),
            default_env_id=episode.get("env_id"),
            default_layout=str(episode.get("layout", "tabletop")),
            source="initial_view_graph",
            limited=False,
        )

    initial_state = episode.get("initial_state")
    if isinstance(initial_state, dict):
        return _state_to_view_graph(
            episode=episode,
            state=initial_state,
            scene_id=str(episode.get("scene_id") or episode_id),
            source="initial_state",
        )

    initial_observation = episode.get("initial_observation")
    if isinstance(initial_observation, dict):
        graph = _observation_to_view_graph(episode, initial_observation, f"{episode_id}_initial")
        graph["source"] = "initial_observation"
        graph["limited"] = True
        return graph

    return {
        "scene_id": str(episode.get("scene_id") or episode_id),
        "env_id": episode.get("env_id"),
        "layout": episode.get("layout", "tabletop"),
        "robot": {"arms": episode.get("arms", "single")},
        "nodes": [],
        "edges": [],
        "source": "missing",
        "limited": True,
    }


def _frame_from_observation(
    *,
    episode: dict[str, Any],
    observation: dict[str, Any],
    frame_id: str,
    label: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "id": frame_id,
        "label": label,
        "kind": kind,
        "held_objects": observation.get("held_objects", []),
        "robot": observation.get("robot", {}),
        "view_graph": _observation_to_view_graph(episode, observation, frame_id),
    }


def _normalize_view_graph(
    graph: dict[str, Any],
    *,
    default_scene_id: str,
    default_env_id: Any,
    default_layout: str,
    source: str,
    limited: bool,
) -> dict[str, Any]:
    raw_nodes = graph.get("nodes", [])
    if isinstance(raw_nodes, dict):
        node_items = raw_nodes.items()
    elif isinstance(raw_nodes, list):
        node_items = ((None, item) for item in raw_nodes)
    else:
        node_items = ()

    nodes = []
    for fallback_id, node in node_items:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id", fallback_id)
        if node_id is None:
            continue
        nodes.append(_node_payload(node, node_id))

    raw_edges = graph.get("edges", [])
    edges = []
    if isinstance(raw_edges, list):
        for edge in raw_edges:
            converted = _edge_payload(edge)
            if converted is not None:
                edges.append(converted)

    return {
        "scene_id": graph.get("scene_id", default_scene_id),
        "env_id": graph.get("env_id", default_env_id),
        "layout": graph.get("layout", default_layout),
        "robot": graph.get("robot", {"arms": "single"}),
        "nodes": nodes,
        "edges": edges,
        "source": source,
        "limited": limited,
    }


def _node_payload(node: dict[str, Any], node_id: Any) -> dict[str, Any]:
    payload = {
        "id": str(node_id),
        "name": str(node.get("name") or node.get("class_name") or node_id),
        "category": str(node.get("category", "object")),
        "properties": list(node.get("properties", []) or []),
        "states": list(node.get("states", []) or []),
    }
    for key in (
        "open",
        "reachable",
        "visible",
        "held",
        "assembled",
        "pressed",
        "pressable",
        "attached_to",
        "location",
        "part_of",
        "room",
        "parent",
    ):
        if key in node:
            payload[key] = node[key]
    return payload


def _edge_payload(edge: Any) -> dict[str, str] | None:
    if not isinstance(edge, dict):
        return None
    source = edge.get("from", edge.get("from_id", edge.get("source")))
    target = edge.get("to", edge.get("to_id", edge.get("target")))
    relation = edge.get("relation", edge.get("relation_type"))
    if source is None or target is None or relation is None:
        return None
    return {"from": str(source), "to": str(target), "relation": str(relation)}


def _state_to_view_graph(
    *,
    episode: dict[str, Any],
    state: dict[str, Any],
    scene_id: str,
    source: str,
) -> dict[str, Any]:
    raw_nodes = state.get("nodes", {})
    if isinstance(raw_nodes, dict):
        node_items = raw_nodes.items()
    elif isinstance(raw_nodes, list):
        node_items = ((None, item) for item in raw_nodes)
    else:
        node_items = ()

    nodes = []
    node_ids: set[str] = set()
    for fallback_id, node in node_items:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id", fallback_id)
        if node_id is None:
            continue
        text_id = str(node_id)
        node_ids.add(text_id)
        nodes.append(_node_payload(node, text_id))

    edges = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(source_id: Any, target_id: Any, relation: Any) -> None:
        if source_id is None or target_id is None or relation is None:
            return
        source_text = str(source_id)
        target_text = str(target_id)
        relation_text = str(relation)
        if source_text not in node_ids or target_text not in node_ids:
            return
        signature = (source_text, target_text, relation_text)
        if signature in seen_edges:
            return
        seen_edges.add(signature)
        edges.append({"from": source_text, "to": target_text, "relation": relation_text})

    for node in nodes:
        location = node.get("location")
        if isinstance(location, dict):
            add_edge(node["id"], location.get("target"), location.get("relation"))
        add_edge(node["id"], node.get("attached_to"), "ATTACHED")
        add_edge(node["id"], node.get("part_of"), "PART_OF")
        add_edge(node["id"], node.get("parent"), "ON")

    return {
        "scene_id": scene_id,
        "env_id": episode.get("env_id"),
        "layout": episode.get("layout", "tabletop"),
        "robot": {"arms": episode.get("arms", "single")},
        "nodes": nodes,
        "edges": edges,
        "source": source,
        "limited": False,
    }


def _observation_to_view_graph(episode: dict[str, Any], observation: dict[str, Any], scene_id: str) -> dict[str, Any]:
    nodes = []
    for node in observation.get("visible_nodes", []) or []:
        if not isinstance(node, dict) or node.get("id") is None:
            continue
        converted = {
            "id": str(node["id"]),
            "name": str(node.get("name") or node["id"]),
            "category": str(node.get("category", "object")),
            "properties": list(node.get("properties", []) or []),
            "states": list(node.get("states", []) or []),
            "open": node.get("open"),
            "reachable": node.get("reachable"),
            "assembled": node.get("assembled"),
            "pressed": node.get("pressed"),
            "pressable": node.get("pressable"),
            "attached_to": node.get("attached_to"),
            "location": node.get("location"),
        }
        if node.get("part_of") is not None:
            converted["part_of"] = node.get("part_of")
        nodes.append(converted)
    edges = []
    for edge in observation.get("visible_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        source = edge.get("from")
        target = edge.get("to")
        relation = edge.get("relation")
        if source is None or target is None or relation is None:
            continue
        edges.append({"from": str(source), "to": str(target), "relation": str(relation)})
    return {
        "scene_id": scene_id,
        "env_id": episode.get("env_id"),
        "layout": episode.get("layout", "tabletop"),
        "robot": {"arms": episode.get("arms", "single")},
        "nodes": nodes,
        "edges": edges,
    }


def _render_trajectory_html(base_path: str = "") -> str:
    body = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trajectory Replay</title>
  <style>
    :root {
      --bg: #f6f7f3;
      --panel: #ffffff;
      --line: #d8ddd2;
      --ink: #20251f;
      --muted: #667063;
      --green: #2f7d4f;
      --blue: #2f6f9f;
      --amber: #ad6b18;
      --red: #b64242;
      --surface: #d9eee8;
      --container: #dce8f5;
      --room: #e5e5e0;
      --object: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .meta {
      min-width: 0;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .app {
      height: calc(100vh - 56px);
      display: grid;
      grid-template-columns: 280px minmax(420px, 1fr) 340px;
      min-height: 540px;
    }
    aside, main {
      min-height: 0;
    }
    .left, .right {
      background: var(--panel);
      border-right: 1px solid var(--line);
      overflow: auto;
    }
    .right {
      border-right: 0;
      border-left: 1px solid var(--line);
    }
    .section {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .section-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 10px;
    }
    select, input[type="range"] {
      width: 100%;
    }
    select {
      height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 0 8px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-top: 10px;
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      cursor: pointer;
      font-weight: 700;
    }
    button:hover { border-color: var(--blue); }
    button:disabled {
      color: #9aa197;
      cursor: default;
      background: #f1f2ef;
    }
    .timeline {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .frame-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .frame-row {
      width: 100%;
      height: auto;
      min-height: 44px;
      display: block;
      text-align: left;
      border-radius: 6px;
      padding: 7px 8px;
      font-weight: 500;
    }
    .frame-row.active {
      border-color: var(--green);
      box-shadow: inset 3px 0 0 var(--green);
    }
    .frame-row.failed {
      border-color: #e4b6b6;
      box-shadow: inset 3px 0 0 var(--red);
    }
    .frame-name {
      display: block;
      overflow-wrap: anywhere;
    }
    .frame-sub {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    main {
      display: grid;
      grid-template-rows: 1fr auto;
      background: #fbfbf8;
    }
    .graph-wrap {
      min-height: 0;
      position: relative;
    }
    svg {
      display: block;
      width: 100%;
      height: 100%;
      min-height: 420px;
      background: #fbfbf8;
    }
    .node circle {
      stroke: #59635a;
      stroke-width: 1.5;
    }
    .node.action circle {
      stroke: var(--amber);
      stroke-width: 3;
    }
    .node.new circle {
      stroke: var(--green);
      stroke-width: 3;
    }
    .node text {
      pointer-events: none;
      font-size: 12px;
      fill: var(--ink);
      text-anchor: middle;
    }
    .edge {
      stroke: #8f978e;
      stroke-width: 1.3;
    }
    .edge-label {
      font-size: 10px;
      fill: #59635a;
      paint-order: stroke;
      stroke: #fbfbf8;
      stroke-width: 3px;
      text-anchor: middle;
    }
    .statusbar {
      min-height: 46px;
      border-top: 1px solid var(--line);
      padding: 10px 14px;
      color: var(--muted);
      background: var(--panel);
      overflow-wrap: anywhere;
    }
    .mini-graph-wrap {
      height: 230px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fbfbf8;
    }
    .mini-graph-wrap svg {
      min-height: 0;
      height: 100%;
      background: #fbfbf8;
    }
    .mini-graph-wrap .node text {
      font-size: 10px;
    }
    .mini-graph-wrap .edge {
      stroke-width: 1;
    }
    .mini-graph-wrap .edge-label {
      display: none;
    }
    .mini-graph-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .kv {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      gap: 6px 10px;
      margin-bottom: 10px;
    }
    .key {
      color: var(--muted);
    }
    .value {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #f3f5f0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      margin: 0 4px 4px 0;
      color: var(--muted);
      background: #fff;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      header { height: auto; min-height: 56px; flex-wrap: wrap; padding: 10px 14px; }
      .app {
        height: auto;
        min-height: calc(100vh - 56px);
        grid-template-columns: 1fr;
      }
      .left, .right {
        border: 0;
        border-bottom: 1px solid var(--line);
        max-height: 42vh;
      }
      main { min-height: 520px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Trajectory Replay</h1>
    <div class="meta" id="file-meta">Loading</div>
  </header>
  <div class="app">
    <aside class="left">
      <div class="section">
        <div class="section-title">Trajectory File</div>
        <select id="file-select"></select>
        <button id="refresh-files-btn" type="button" style="width:100%;margin-top:8px;">Refresh Files</button>
      </div>
      <div class="section">
        <div class="section-title">Episode</div>
        <select id="episode-select"></select>
      </div>
      <div class="section">
        <div class="section-title">Playback</div>
        <input id="frame-range" type="range" min="0" max="0" value="0">
        <div class="timeline">
          <span id="frame-index">0 / 0</span>
          <span id="play-state">Paused</span>
        </div>
        <div class="controls">
          <button id="prev-btn" title="Previous frame" aria-label="Previous frame">&lt;</button>
          <button id="play-btn" title="Play" aria-label="Play">&gt;</button>
          <button id="pause-btn" title="Pause" aria-label="Pause">||</button>
          <button id="next-btn" title="Next frame" aria-label="Next frame">&gt;&gt;</button>
        </div>
      </div>
      <div class="section">
        <div class="section-title">Frames</div>
        <div class="frame-list" id="frame-list"></div>
      </div>
    </aside>
    <main>
      <div class="graph-wrap">
        <svg id="graph" role="img" aria-label="Visible view graph"></svg>
      </div>
      <div class="statusbar" id="statusbar">Loading trajectory data</div>
    </main>
    <aside class="right">
      <div class="section">
        <div class="section-title">Initial View Graph</div>
        <div class="mini-graph-wrap">
          <svg id="initial-graph" role="img" aria-label="Initial full view graph"></svg>
        </div>
        <div class="mini-graph-meta" id="initial-graph-meta"></div>
      </div>
      <div class="section">
        <div class="section-title">Step</div>
        <div class="kv" id="step-meta"></div>
      </div>
      <div class="section">
        <div class="section-title">New Visible</div>
        <div id="new-visible"></div>
      </div>
      <div class="section">
        <div class="section-title">Action</div>
        <pre id="action-json">{}</pre>
      </div>
      <div class="section">
        <div class="section-title">Event</div>
        <pre id="event-json">{}</pre>
      </div>
      <div class="section">
        <div class="section-title">Teacher</div>
        <pre id="teacher-json">{}</pre>
      </div>
    </aside>
  </div>
  <script>
    const BASE_PATH = __BASE_PATH_JSON__;
    function apiPath(path) {
      const base = BASE_PATH.endsWith("/") ? BASE_PATH.slice(0, -1) : BASE_PATH;
      const suffix = path.startsWith("/") ? path : `/${path}`;
      return `${base}${suffix}`;
    }

    const state = {
      payload: null,
      files: [],
      selectedFile: null,
      episodeIndex: 0,
      frameIndex: 0,
      timer: null,
      positions: {},
      initialPositions: {},
    };

    const els = {
      fileMeta: document.getElementById("file-meta"),
      fileSelect: document.getElementById("file-select"),
      refreshFilesBtn: document.getElementById("refresh-files-btn"),
      episodeSelect: document.getElementById("episode-select"),
      frameRange: document.getElementById("frame-range"),
      frameIndex: document.getElementById("frame-index"),
      playState: document.getElementById("play-state"),
      frameList: document.getElementById("frame-list"),
      graph: document.getElementById("graph"),
      initialGraph: document.getElementById("initial-graph"),
      initialGraphMeta: document.getElementById("initial-graph-meta"),
      statusbar: document.getElementById("statusbar"),
      stepMeta: document.getElementById("step-meta"),
      newVisible: document.getElementById("new-visible"),
      actionJson: document.getElementById("action-json"),
      eventJson: document.getElementById("event-json"),
      teacherJson: document.getElementById("teacher-json"),
    };

    function episode() {
      return state.payload?.episodes?.[state.episodeIndex] || null;
    }

    function frame() {
      return episode()?.frames?.[state.frameIndex] || null;
    }

    function hashText(value) {
      let hash = 2166136261;
      const text = String(value);
      for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
      }
      return hash >>> 0;
    }

    function nodeFill(node) {
      const category = String(node.category || "").toLowerCase();
      const props = new Set((Array.isArray(node.properties) ? node.properties : []).map(v => String(v).toUpperCase()));
      if (category.includes("room")) return "var(--room)";
      if (props.has("SURFACES") || props.has("SURFACE") || category.includes("surface")) return "var(--surface)";
      if (props.has("CONTAINERS") || category.includes("container")) return "var(--container)";
      return "var(--object)";
    }

    function actionNodeIds(currentFrame) {
      const action = currentFrame?.action || {};
      return new Set(Array.isArray(action.node_ids) ? action.node_ids.map(String) : []);
    }

    function newNodeIds(currentFrame) {
      return new Set((currentFrame?.new_visible_nodes || [])
        .filter(item => item && item.id !== undefined)
        .map(item => String(item.id)));
    }

    function visualNodeRadius(nodes, width, height) {
      const count = Math.max(1, nodes.length);
      const base = count > 32 ? 12 : count > 22 ? 14 : count > 14 ? 16 : 18;
      return Math.max(10, Math.min(base, Math.min(width, height) / 30));
    }

    function computePositions(nodes, edges, width, height, nodeRadius, previousPositions, persistCallback) {
      const positions = {};
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.max(110, Math.min(width, height) * 0.42);
      nodes.forEach((node, index) => {
        const id = String(node.id);
        const previous = previousPositions?.[id];
        if (previous) {
          positions[id] = { x: previous.x, y: previous.y };
          return;
        }
        const seed = hashText(id);
        const angle = (index / Math.max(1, nodes.length)) * Math.PI * 2 + (seed % 100) / 100;
        positions[id] = {
          x: cx + Math.cos(angle) * radius * (0.72 + ((seed % 17) / 80)),
          y: cy + Math.sin(angle) * radius * (0.72 + (((seed >> 4) % 17) / 80)),
        };
      });
      const nodeIds = new Set(nodes.map(node => String(node.id)));
      const links = edges
        .map(edge => ({ source: String(edge.from), target: String(edge.to) }))
        .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target));
      const desiredGap = nodeRadius * 3.4;
      const repelStrength = Math.max(2600, nodes.length * desiredGap * 42);
      const edgeLength = Math.max(120, 190 - nodes.length * 2.5);
      for (let iteration = 0; iteration < 120; iteration += 1) {
        for (let i = 0; i < nodes.length; i += 1) {
          for (let j = i + 1; j < nodes.length; j += 1) {
            const a = positions[String(nodes[i].id)];
            const b = positions[String(nodes[j].id)];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let dist = Math.sqrt(dx * dx + dy * dy) || 1;
            const push = Math.min(7, repelStrength / (dist * dist));
            const overlapPush = dist < desiredGap ? (desiredGap - dist) * 0.08 : 0;
            dx /= dist;
            dy /= dist;
            a.x -= dx * (push + overlapPush);
            a.y -= dy * (push + overlapPush);
            b.x += dx * (push + overlapPush);
            b.y += dy * (push + overlapPush);
          }
        }
        links.forEach(link => {
          const a = positions[link.source];
          const b = positions[link.target];
          if (!a || !b) return;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const pull = (dist - edgeLength) * 0.014;
          const ux = dx / dist;
          const uy = dy / dist;
          a.x += ux * pull;
          a.y += uy * pull;
          b.x -= ux * pull;
          b.y -= uy * pull;
        });
        nodes.forEach(node => {
          const p = positions[String(node.id)];
          p.x += (cx - p.x) * 0.012;
          p.y += (cy - p.y) * 0.012;
          p.x = Math.max(42, Math.min(width - 42, p.x));
          p.y = Math.max(42, Math.min(height - 42, p.y));
        });
      }
      if (persistCallback) persistCallback(positions);
      return positions;
    }

    function renderGraphSvg(svg, graph, options = {}) {
      svg.replaceChildren();
      const rect = svg.getBoundingClientRect();
      const width = Math.max(options.minWidth || 420, rect.width || options.defaultWidth || 900);
      const height = Math.max(options.minHeight || 420, rect.height || options.defaultHeight || 620);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      if (!graph) return;
      const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
      const edges = Array.isArray(graph.edges) ? graph.edges : [];
      const circleRadius = visualNodeRadius(nodes, width, height);
      const positions = computePositions(
        nodes,
        edges,
        width,
        height,
        circleRadius,
        options.positions || {},
        options.persistPositions || null,
      );
      const currentFrame = options.currentFrame || null;
      const actionIds = currentFrame ? actionNodeIds(currentFrame) : new Set();
      const visibleIds = currentFrame ? newNodeIds(currentFrame) : new Set();
      const markerId = options.markerId || "arrow";
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML = `<marker id="${markerId}" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#8f978e"></path></marker>`;
      svg.appendChild(defs);
      edges.forEach(edge => {
        const source = positions[String(edge.from)];
        const target = positions[String(edge.to)];
        if (!source || !target) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("class", "edge");
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x);
        line.setAttribute("y2", target.y);
        line.setAttribute("marker-end", `url(#${markerId})`);
        svg.appendChild(line);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("class", "edge-label");
        label.setAttribute("x", (source.x + target.x) / 2);
        label.setAttribute("y", (source.y + target.y) / 2 - 6);
        label.textContent = String(edge.relation || "");
        svg.appendChild(label);
      });
      nodes.forEach(node => {
        const id = String(node.id);
        const p = positions[id];
        if (!p) return;
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const classes = ["node"];
        if (actionIds.has(id)) classes.push("action");
        if (visibleIds.has(id)) classes.push("new");
        group.setAttribute("class", classes.join(" "));
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", p.x);
        circle.setAttribute("cy", p.y);
        circle.setAttribute("r", String(circleRadius));
        circle.setAttribute("fill", nodeFill(node));
        group.appendChild(circle);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", p.x);
        text.setAttribute("y", p.y + circleRadius + 17);
        text.textContent = String(node.name || node.id);
        group.appendChild(text);
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${node.id}\n${node.category || "object"}`;
        group.appendChild(title);
        svg.appendChild(group);
      });
    }

    function renderGraph() {
      const currentFrame = frame();
      renderGraphSvg(els.graph, currentFrame?.view_graph || null, {
        currentFrame,
        positions: state.positions,
        persistPositions: positions => { state.positions = positions; },
        markerId: "arrow-main",
        minWidth: 420,
        minHeight: 420,
        defaultWidth: 900,
        defaultHeight: 620,
      });
    }

    function renderInitialGraph() {
      const currentEpisode = episode();
      const graph = currentEpisode?.initial_view_graph || null;
      renderGraphSvg(els.initialGraph, graph, {
        positions: state.initialPositions,
        persistPositions: positions => { state.initialPositions = positions; },
        markerId: "arrow-initial",
        minWidth: 260,
        minHeight: 180,
        defaultWidth: 300,
        defaultHeight: 230,
      });
      const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
      const edges = Array.isArray(graph?.edges) ? graph.edges : [];
      const suffix = graph?.limited ? " (initial visible fallback)" : "";
      els.initialGraphMeta.textContent = graph
        ? `${nodes.length} nodes, ${edges.length} edges${suffix}`
        : "No initial view graph";
    }

    function renderFrameList() {
      const currentEpisode = episode();
      els.frameList.replaceChildren();
      (currentEpisode?.frames || []).forEach((item, index) => {
        const button = document.createElement("button");
        const event = item.event || {};
        const action = item.action || {};
        button.className = `frame-row ${index === state.frameIndex ? "active" : ""} ${event.status === "failure" ? "failed" : ""}`;
        button.innerHTML = `<span class="frame-name">${item.label}</span><span class="frame-sub">${action.name || item.kind || ""} ${event.status || ""}</span>`;
        button.addEventListener("click", () => setFrame(index));
        els.frameList.appendChild(button);
      });
    }

    function renderDetails() {
      const currentEpisode = episode();
      const currentFrame = frame();
      const frames = currentEpisode?.frames || [];
      els.frameRange.max = Math.max(0, frames.length - 1);
      els.frameRange.value = state.frameIndex;
      els.frameIndex.textContent = frames.length ? `${state.frameIndex + 1} / ${frames.length}` : "0 / 0";
      if (!currentFrame || !currentEpisode) {
        els.statusbar.textContent = "No frame selected";
        return;
      }
      const graph = currentFrame.view_graph || {};
      const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
      const edges = Array.isArray(graph.edges) ? graph.edges : [];
      els.statusbar.textContent = `${currentFrame.label}: ${nodes.length} visible nodes, ${edges.length} visible edges`;
      const rows = [
        ["episode", currentEpisode.episode_id],
        ["task", currentEpisode.task],
        ["mode", currentEpisode.mode],
        ["success", String(currentEpisode.success)],
        ["frame", currentFrame.label],
        ["held", (currentFrame.held_objects || []).map(item => item.name || item.id).join(", ")],
      ];
      els.stepMeta.replaceChildren();
      rows.forEach(([key, value]) => {
        const keyEl = document.createElement("div");
        keyEl.className = "key";
        keyEl.textContent = key;
        const valueEl = document.createElement("div");
        valueEl.className = "value";
        valueEl.textContent = value || "";
        els.stepMeta.append(keyEl, valueEl);
      });
      els.newVisible.replaceChildren();
      const newVisible = currentFrame.new_visible_nodes || [];
      if (!newVisible.length) {
        const empty = document.createElement("span");
        empty.className = "pill";
        empty.textContent = "none";
        els.newVisible.appendChild(empty);
      } else {
        newVisible.forEach(node => {
          const pill = document.createElement("span");
          pill.className = "pill";
          pill.textContent = String(node.name || node.id);
          els.newVisible.appendChild(pill);
        });
      }
      els.actionJson.textContent = JSON.stringify(currentFrame.action || {}, null, 2);
      els.eventJson.textContent = JSON.stringify(currentFrame.event || {}, null, 2);
      els.teacherJson.textContent = JSON.stringify(currentFrame.teacher_response || {}, null, 2);
    }

    function render() {
      renderFrameList();
      renderDetails();
      renderGraph();
      renderInitialGraph();
    }

    function setFrame(index) {
      const frames = episode()?.frames || [];
      state.frameIndex = Math.max(0, Math.min(Math.max(0, frames.length - 1), index));
      render();
    }

    function setEpisode(index) {
      state.episodeIndex = index;
      state.frameIndex = 0;
      state.positions = {};
      state.initialPositions = {};
      render();
    }

    function play() {
      if (state.timer) return;
      els.playState.textContent = "Playing";
      state.timer = window.setInterval(() => {
        const frames = episode()?.frames || [];
        if (!frames.length || state.frameIndex >= frames.length - 1) {
          pause();
          return;
        }
        setFrame(state.frameIndex + 1);
      }, 900);
    }

    function pause() {
      if (state.timer) {
        window.clearInterval(state.timer);
        state.timer = null;
      }
      els.playState.textContent = "Paused";
    }

    async function loadFiles(keepSelection=false) {
      const response = await fetch(apiPath("/api/trajectory-files"));
      const listing = await response.json();
      if (!response.ok) throw new Error(listing.error || "failed to list trajectory files");
      state.files = Array.isArray(listing.files) ? listing.files : [];
      const current = keepSelection ? state.selectedFile : null;
      const selected = (
        current && state.files.some(item => item.name === current)
          ? current
          : listing.selected || state.files[0]?.name || null
      );
      els.fileSelect.replaceChildren();
      if (!state.files.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No .jsonl files";
        els.fileSelect.appendChild(option);
        els.fileSelect.disabled = true;
        state.payload = { episodes: [] };
        state.selectedFile = null;
        els.fileMeta.textContent = `${listing.trajectory_dir} (0 files)`;
        render();
        els.statusbar.textContent = `No trajectory JSONL files found in ${listing.trajectory_dir}`;
        return;
      }
      els.fileSelect.disabled = false;
      state.files.forEach(item => {
        const option = document.createElement("option");
        option.value = item.name;
        option.textContent = `${item.name} (${Math.max(1, Math.round((item.size_bytes || 0) / 1024))} KB)`;
        if (item.name === selected) option.selected = true;
        els.fileSelect.appendChild(option);
      });
      await loadTrajectory(selected);
    }

    async function loadTrajectory(fileName) {
      pause();
      if (!fileName) return;
      state.selectedFile = fileName;
      const response = await fetch(apiPath(`/api/trajectories?file=${encodeURIComponent(fileName)}`));
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "failed to load trajectories");
      state.payload = payload;
      els.fileMeta.textContent = `${payload.trajectory_file || payload.trajectory_path} (${payload.episode_count} episodes)`;
      els.episodeSelect.replaceChildren();
      payload.episodes.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${index + 1}. ${item.episode_id}`;
        els.episodeSelect.appendChild(option);
      });
      setEpisode(0);
      if (!payload.episodes.length) {
        els.statusbar.textContent = "No trajectories found in file";
      }
    }

    async function load() {
      await loadFiles(false);
    }

    document.getElementById("prev-btn").addEventListener("click", () => setFrame(state.frameIndex - 1));
    document.getElementById("next-btn").addEventListener("click", () => setFrame(state.frameIndex + 1));
    document.getElementById("play-btn").addEventListener("click", play);
    document.getElementById("pause-btn").addEventListener("click", pause);
    els.frameRange.addEventListener("input", event => setFrame(Number(event.target.value)));
    els.episodeSelect.addEventListener("change", event => setEpisode(Number(event.target.value)));
    els.fileSelect.addEventListener("change", event => loadTrajectory(event.target.value).catch(error => {
      els.fileMeta.textContent = "Load failed";
      els.statusbar.textContent = error.message;
    }));
    els.refreshFilesBtn.addEventListener("click", () => loadFiles(true).catch(error => {
      els.fileMeta.textContent = "Load failed";
      els.statusbar.textContent = error.message;
    }));
    window.addEventListener("resize", () => {
      renderGraph();
      renderInitialGraph();
    });
    load().catch(error => {
      els.fileMeta.textContent = "Load failed";
      els.statusbar.textContent = error.message;
    });
  </script>
</body>
</html>
"""
    return body.replace("__BASE_PATH_JSON__", json.dumps(_normalize_base_path(base_path)))
