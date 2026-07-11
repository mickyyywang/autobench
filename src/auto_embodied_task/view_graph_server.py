from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import random
import re
from typing import Any
import webbrowser

from .layout_synthesis import TaskViewGraphSynthesisConfig, synthesize_task_view_graph
from .placement_constraints import PlacementEdgeConstraints
from .profile_editor import edit_view_graph_with_profile


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def serve_view_graph_app(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> None:
    server = ThreadingHTTPServer((host, port), _ViewGraphAppHandler)
    url = f"http://{host}:{server.server_port}/"
    print(f"View graph UI running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping view graph UI")
    finally:
        server.server_close()


class _ViewGraphAppHandler(BaseHTTPRequestHandler):
    server_version = "AutoEmbodiedTaskViewGraphUI/0.1"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send_text(_render_app_html(), content_type="text/html; charset=utf-8")
            return
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path == "/api/create-view-graph":
            self._handle_create_view_graph()
            return
        if self.path == "/api/edit-view-graph":
            self._handle_edit_view_graph()
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_create_view_graph(self) -> None:
        try:
            payload = self._read_json()
            config = _config_from_payload(payload)
            package = synthesize_task_view_graph(config)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(package)

    def _handle_edit_view_graph(self) -> None:
        try:
            payload = self._read_json()
            package = _edit_view_graphs_from_payload(payload)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(package)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

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


def _config_from_payload(payload: dict[str, Any]) -> TaskViewGraphSynthesisConfig:
    materials = _materials_from_text(str(payload.get("materials_text", "")))
    material_properties = _material_properties_from_text(str(payload.get("material_properties_text", "")).strip())
    scene = str(payload.get("scene", "")).strip()
    layout = str(payload.get("layout", "tabletop")).strip()
    arms = str(payload.get("arms", "single")).strip()
    api_key, api_key_env = _api_key_fields_from_payload(payload)
    if not scene:
        raise ValueError("scene is required")
    if not materials:
        raise ValueError("at least one material is required")
    timeout_raw = payload.get("timeout_seconds", 60)
    try:
        timeout_seconds = int(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be an integer") from exc
    return TaskViewGraphSynthesisConfig(
        materials=materials,
        scene=scene,
        layout=layout,
        arms=arms,
        material_properties=material_properties,
        task_hint=_optional_str(payload.get("task_hint")),
        scene_id=_optional_str(payload.get("scene_id")),
        env_id=_optional_str(payload.get("env_id")),
        provider=str(payload.get("provider", "qwen")).strip() or "qwen",
        model=_optional_str(payload.get("model")),
        api_key=api_key,
        api_key_env=api_key_env,
        api_base_url=_optional_str(payload.get("api_base_url")),
        timeout_seconds=timeout_seconds,
        enable_thinking=_bool_from_payload(payload.get("enable_thinking", False)),
    )


def _api_key_fields_from_payload(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    api_key = _optional_str(payload.get("api_key"))
    api_key_env = _optional_str(payload.get("api_key_env"))
    if api_key_env and not _ENV_NAME_RE.fullmatch(api_key_env):
        if api_key is None:
            api_key = api_key_env
        api_key_env = None
    return api_key, api_key_env


def _edit_view_graphs_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload.get("view_graph", payload.get("graph"))
    if not isinstance(graph, dict) or isinstance(graph, list):
        raise ValueError("view_graph is required")
    profile = _profile_from_edit_payload(payload)
    placement_edge_constraints = _placement_edge_constraints_from_payload(payload)
    num_samples = _int_from_payload(payload.get("num_samples", 1), "num_samples", default=1, minimum=1)
    seed = _optional_int_from_payload(payload.get("seed"), "seed")
    seed_source = random.Random(seed)
    results = []
    for sample_index in range(num_samples):
        sample_rng = random.Random(seed_source.randrange(0, 2**63))
        results.append(
            edit_view_graph_with_profile(
                graph,
                profile,
                rng=sample_rng,
                sample_index=sample_index,
                num_samples=num_samples,
                placement_edge_constraints=placement_edge_constraints,
            )
        )
    graphs = [result.graph for result in results]
    return {
        "view_graph": graphs[0] if graphs else None,
        "view_graphs": graphs,
        "requested_profile": profile,
        "achieved_profiles": [result.achieved_profile for result in results],
        "profile_constraints": [result.constraints for result in results],
        "graph_edits": [result.graph_edits for result in results],
        "num_samples": len(results),
        "placement_edge_constraints": placement_edge_constraints.to_json()
        if placement_edge_constraints is not None and not placement_edge_constraints.is_empty()
        else None,
    }


def _profile_from_edit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_profile = payload.get("profile")
    if raw_profile is not None:
        if not isinstance(raw_profile, dict):
            raise ValueError("profile must be a JSON object")
        return raw_profile
    return {
        "profile_id": _optional_str(payload.get("profile_id")) or "ui_profile",
        "spatial": {
            "enabled": _bool_from_payload(payload.get("spatial_enabled", True)),
            "num_occluded_objects": _int_from_payload(
                payload.get("num_occluded_objects", 1),
                "num_occluded_objects",
                default=1,
            ),
            "occlusion_depth": _int_from_payload(
                payload.get("occlusion_depth", 1),
                "occlusion_depth",
                default=1,
            ),
            "num_decomposed_parents": _int_from_payload(
                payload.get("num_decomposed_parents", 0),
                "num_decomposed_parents",
            ),
        },
    }


def _placement_edge_constraints_from_payload(payload: dict[str, Any]) -> PlacementEdgeConstraints | None:
    raw_constraints = payload.get("placement_edge_constraints", payload.get("placement_constraints"))
    if raw_constraints is None:
        return None
    if isinstance(raw_constraints, str):
        text = raw_constraints.strip()
        if not text:
            return None
        try:
            raw_constraints = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"placement_edge_constraints must be valid JSON: {exc}") from exc
    try:
        return PlacementEdgeConstraints.from_json(raw_constraints)
    except ValueError as exc:
        raise ValueError(f"invalid placement_edge_constraints: {exc}") from exc


def _int_from_payload(value: Any, name: str, *, default: int = 0, minimum: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    if isinstance(value, bool):
        parsed = int(value)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _optional_int_from_payload(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _int_from_payload(value, name, minimum=0)


def _materials_from_text(value: str) -> tuple[str, ...]:
    materials = []
    for line in value.splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        materials.append(item)
    return tuple(materials)


def _material_properties_from_text(value: str) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("material properties must be a JSON object")
    return data


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_from_payload(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(text)


def _render_app_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>View Graph Workbench</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #65758b;
      --line: #d7dee7;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --danger: #b42318;
      --soft: #eef6f5;
      --shadow: 0 1px 2px rgba(15, 23, 42, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select, textarea { font: inherit; }
    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: 54px 1fr;
      overflow: hidden;
    }
    .topbar {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .brand {
      min-width: 260px;
      display: grid;
      gap: 1px;
    }
    .brand strong {
      font-size: 15px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .brand span {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .toolbar {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      cursor: pointer;
    }
    .btn.active {
      background: var(--soft);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 650;
    }
    .btn.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .btn.danger {
      color: var(--danger);
      border-color: #efc6c1;
    }
    .main {
      min-height: 0;
      display: grid;
      grid-template-columns: 360px minmax(460px, 1fr) 360px;
      gap: 1px;
      background: var(--line);
    }
    .panel {
      min-height: 0;
      background: var(--panel);
      display: flex;
      flex-direction: column;
    }
    .panel header {
      height: 42px;
      padding: 0 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }
    .scroll {
      min-height: 0;
      overflow: auto;
      padding: 12px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .section {
      display: grid;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }
    .section:last-child { border-bottom: 0; }
    .section-title {
      color: #243447;
      font-weight: 700;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 8px;
      min-height: 34px;
    }
    .checkbox {
      min-height: 34px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .checkbox input {
      width: auto;
      min-height: 0;
      margin: 0;
    }
    input[type="range"] {
      padding: 0;
      min-height: 26px;
      accent-color: var(--accent);
    }
    input[type="number"] {
      min-width: 0;
    }
    .slider-field {
      display: grid;
      gap: 5px;
    }
    .slider-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .slider-label strong {
      color: var(--ink);
      font-weight: 700;
      min-width: 24px;
      text-align: right;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    textarea {
      resize: vertical;
      min-height: 92px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .row3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .canvas-wrap {
      position: relative;
      background: #fbfcfd;
      min-height: 0;
    }
    svg {
      display: block;
      width: 100%;
      height: 100%;
      touch-action: none;
      user-select: none;
    }
    .edge {
      stroke: #6b7a89;
      stroke-width: 1.8;
      marker-end: url(#arrow);
    }
    .edge.active {
      stroke: var(--accent-2);
      stroke-width: 3;
    }
    .edge-label {
      font-size: 11px;
      fill: #334155;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .node { cursor: grab; }
    .node circle {
      fill: #fff;
      stroke: #435466;
      stroke-width: 2;
    }
    .node.active circle {
      fill: var(--soft);
      stroke: var(--accent);
      stroke-width: 3;
    }
    .node text {
      font-size: 12px;
      fill: #17212b;
      text-anchor: middle;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .map-zone rect {
      fill: #f7f9fb;
      stroke: #48566a;
      stroke-width: 2.6;
    }
    .map-zone.active rect {
      stroke: var(--accent);
      stroke-width: 3;
    }
    .map-zone text {
      fill: #39485c;
      font-size: 14px;
      font-weight: 700;
      pointer-events: none;
    }
    .map-object {
      cursor: pointer;
    }
    .map-object rect {
      stroke-width: 2;
    }
    .map-object.active rect {
      stroke: var(--accent);
      stroke-width: 3;
    }
    .map-object.nested rect {
      stroke-width: 1.5;
    }
    .map-object .occlusion-box {
      fill: none;
      stroke: #dc2626;
      stroke-width: 2.6;
      stroke-dasharray: 7 5;
    }
    .map-object .node-name {
      fill: #172033;
      font-size: 12px;
      font-weight: 700;
      text-anchor: middle;
      pointer-events: none;
    }
    .map-object .node-meta {
      fill: #536070;
      font-size: 10px;
      text-anchor: middle;
      pointer-events: none;
    }
    .map-object .relation-badge {
      fill: #0f766e;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: 0;
      text-anchor: middle;
      text-transform: lowercase;
      pointer-events: none;
    }
    .map-object.nested .node-name { font-size: 10px; }
    .map-object.nested .node-meta { display: none; }
    .map-legend .legend-swatch {
      stroke-width: 2;
    }
    .map-legend-bg {
      fill: rgba(255,255,255,.94);
      stroke: #d7dee7;
    }
    .map-legend text {
      fill: #465468;
      font-size: 11px;
      pointer-events: none;
    }
    .list {
      display: grid;
      gap: 6px;
      align-content: start;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      cursor: pointer;
      text-align: left;
      min-width: 0;
    }
    .item.active {
      border-color: var(--accent);
      background: var(--soft);
    }
    .item-title {
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .item-meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .status {
      position: absolute;
      left: 12px;
      bottom: 12px;
      max-width: calc(100% - 24px);
      background: rgba(255,255,255,.94);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--muted);
      box-shadow: var(--shadow);
      pointer-events: none;
    }
    .message {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--muted);
      white-space: pre-wrap;
    }
    .message.error {
      color: var(--danger);
      border-color: #efc6c1;
      background: #fff7f6;
    }
    .empty {
      color: var(--muted);
      padding: 8px;
    }
    @media (max-width: 1120px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100vh; }
      .topbar { height: auto; min-height: 54px; flex-wrap: wrap; padding: 10px 12px; }
      .toolbar { margin-left: 0; flex-wrap: wrap; }
      .main { grid-template-columns: 1fr; grid-template-rows: auto 560px auto; }
      .panel { min-height: 240px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="brand">
        <strong id="scene-title">View Graph Workbench</strong>
        <span id="scene-subtitle">create -> inspect/edit -> download JSONL -> generate</span>
      </div>
      <div class="toolbar">
        <button class="btn active" id="graph-view-btn" type="button">Graph</button>
        <button class="btn" id="map-view-btn" type="button">Map</button>
        <button class="btn" id="fit-btn" type="button">Fit</button>
        <button class="btn" id="add-node-btn" type="button">Add Node</button>
        <button class="btn" id="add-edge-btn" type="button">Add Edge</button>
        <button class="btn primary" id="download-btn" type="button">Download JSONL</button>
      </div>
    </div>
    <div class="main">
      <section class="panel">
        <header>Create View Graph</header>
        <div class="scroll">
          <form id="create-form" class="section">
            <div class="section-title">Task Spec</div>
            <label>Materials, one per line
              <textarea id="materials">桌面
抽屉
书
蓝色笔
红色笔
黑色笔
纸
铅笔盒
文件夹
收纳盒</textarea>
            </label>
            <label>Load materials file (.txt)
              <input id="materials-file" type="file" accept=".txt,text/plain">
            </label>
            <label>Material properties JSON
              <textarea id="material-properties">{}</textarea>
            </label>
            <label>Load material properties file (.json)
              <input id="material-properties-file" type="file" accept=".json,application/json">
            </label>
            <div class="row">
              <label>Scene
                <input id="scene" value="办公桌面">
              </label>
              <label>Task / Activity
                <input id="task-hint" value="整理办公桌">
              </label>
            </div>
            <div class="row3">
              <label>Layout
                <select id="layout">
                  <option value="tabletop" selected>tabletop</option>
                  <option value="indoor">indoor</option>
                </select>
              </label>
              <label>Arms
                <select id="arms">
                  <option value="double" selected>double</option>
                  <option value="single">single</option>
                </select>
              </label>
              <label>Timeout
                <input id="timeout-seconds" value="60">
              </label>
            </div>
            <div class="row">
              <label>Provider
                <select id="provider">
                  <option value="qwen" selected>qwen</option>
                  <option value="openai">openai</option>
                  <option value="compatible">compatible</option>
                </select>
              </label>
              <label>Model
                <input id="model" value="qwen3.6-plus">
              </label>
            </div>
            <div class="row3">
              <label>API key
                <input id="api-key" type="password" placeholder="sk-...">
              </label>
              <label>API key env
                <input id="api-key-env" placeholder="DASHSCOPE_API_KEY">
              </label>
              <label>API base URL
                <input id="api-base-url" placeholder="provider default">
              </label>
            </div>
            <div class="row">
              <label>Scene ID
                <input id="scene-id" placeholder="optional">
              </label>
              <label>Env ID
                <input id="env-id" placeholder="optional">
              </label>
            </div>
            <label class="checkbox">
              <input id="enable-thinking" type="checkbox">
              Enable thinking for Qwen
            </label>
            <button class="btn primary" id="create-btn" type="submit">Create View Graph</button>
          </form>
          <div class="section">
            <div class="section-title">Backend</div>
            <div id="create-message" class="message">Ready. Load materials/properties files or edit the text directly. Use API key for a raw key, or API key env for an environment variable name.</div>
          </div>
          <div class="section">
            <div class="section-title">Import Existing View Graph</div>
            <label>View graph JSON / JSONL
              <textarea id="import-view-graph" placeholder='Paste {"view_graph": {...}} package JSON, direct view graph JSON, or one JSONL line'></textarea>
            </label>
            <label>Load view graph file (.json/.jsonl)
              <input id="view-graph-file" type="file" accept=".json,.jsonl,application/json,application/jsonl">
            </label>
            <button class="btn" id="import-view-graph-btn" type="button">Import View Graph</button>
          </div>
          <div class="section">
            <div class="section-title">Profile Edit</div>
            <div class="row">
              <label>Profile ID
                <input id="profile-id" value="ui_profile">
              </label>
              <label>Num samples
                <input id="profile-num-samples" type="number" min="1" step="1" value="1">
              </label>
            </div>
            <label>Seed
              <input id="profile-seed" type="number" step="1" placeholder="random">
            </label>
            <label class="checkbox">
              <input id="spatial-enabled" type="checkbox" checked>
              Spatial
            </label>
            <div class="slider-field">
              <div class="slider-label"><span>Occluded objects</span><strong id="spatial-num-occluded-value">1</strong></div>
              <input id="spatial-num-occluded" type="range" min="0" max="8" step="1" value="1">
            </div>
            <div class="slider-field">
              <div class="slider-label"><span>Occlusion depth</span><strong id="spatial-occlusion-depth-value">1</strong></div>
              <input id="spatial-occlusion-depth" type="range" min="0" max="4" step="1" value="1">
            </div>
            <div class="slider-field">
              <div class="slider-label"><span>Decomposed parents</span><strong id="spatial-decomposed-parents-value">0</strong></div>
              <input id="spatial-decomposed-parents" type="range" min="0" max="8" step="1" value="0">
            </div>
            <label>Placement constraints JSON
              <textarea id="profile-placement-constraints" placeholder='{"nonexistent_edges":[{"from":"drawer","to":"book","relation":"OCCLUDES"}]}'></textarea>
            </label>
            <label>Load placement constraints file (.json)
              <input id="placement-constraints-file" type="file" accept=".json,application/json">
            </label>
            <div class="actions">
              <button class="btn primary" id="apply-profile-btn" type="button">Apply Profile</button>
              <button class="btn" id="download-profiled-btn" type="button">Download Samples JSONL</button>
            </div>
          </div>
          <div class="section">
            <div class="section-title">Nodes</div>
            <div class="list" id="node-list"></div>
          </div>
        </div>
      </section>
      <section class="canvas-wrap">
        <svg id="graph-svg" aria-label="view graph editor canvas">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7a89"></path>
            </marker>
          </defs>
        </svg>
        <div class="status" id="status">No graph loaded.</div>
      </section>
      <section class="panel">
        <header>Edit Graph</header>
        <div class="scroll">
          <div class="section" id="edge-editor"></div>
          <div class="section" id="node-editor"></div>
          <div class="section">
            <div class="section-title">Edge List</div>
            <div class="list" id="edge-list"></div>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    const RELATIONS = ["ON", "BENEATH", "INSIDE", "PART_OF", "OCCLUDES", "LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "CONNECTED", "CLOSE", "NEAR"];
    let graph = { scene_id: null, env_id: null, layout: "tabletop", robot: { arms: "double" }, nodes: [], edges: [] };
    let profiledSamples = [];
    let profileBaseGraph = null;
    const state = { selectedNodeId: null, selectedEdgeIndex: null, positions: {}, dragging: null, viewMode: "graph" };

    const svg = document.getElementById("graph-svg");
    const nodeList = document.getElementById("node-list");
    const edgeList = document.getElementById("edge-list");
    const edgeEditor = document.getElementById("edge-editor");
    const nodeEditor = document.getElementById("node-editor");
    const statusBox = document.getElementById("status");
    const messageBox = document.getElementById("create-message");

    function nodeId(node) { return String(node.id ?? ""); }
    function nodeName(node) { return String(node.name ?? node.id ?? ""); }
    function edgeFrom(edge) { return String(edge.from ?? edge.from_id ?? edge.source ?? ""); }
    function edgeTo(edge) { return String(edge.to ?? edge.to_id ?? edge.target ?? ""); }
    function edgeRelation(edge) { return String(edge.relation ?? edge.relation_type ?? ""); }
    function setEdgeFrom(edge, value) { delete edge.from_id; delete edge.source; edge.from = value; }
    function setEdgeTo(edge, value) { delete edge.to_id; delete edge.target; edge.to = value; }
    function setEdgeRelation(edge, value) { delete edge.relation_type; edge.relation = value; }
    function getNode(id) { return graph.nodes.find(node => nodeId(node) === String(id)); }

    function setMessage(text, isError=false) {
      messageBox.textContent = text;
      messageBox.className = "message" + (isError ? " error" : "");
    }

    function initPositions(force=false) {
      const rect = svg.getBoundingClientRect();
      const width = Math.max(rect.width, 720);
      const height = Math.max(rect.height, 520);
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.max(120, Math.min(width, height) * 0.36);
      graph.nodes.forEach((node, index) => {
        const id = nodeId(node);
        if (!force && state.positions[id]) return;
        const angle = (Math.PI * 2 * index) / Math.max(graph.nodes.length, 1) - Math.PI / 2;
        state.positions[id] = { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius };
      });
    }

    function render() {
      graph.nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
      graph.edges = Array.isArray(graph.edges) ? graph.edges : [];
      initPositions();
      renderViewButtons();
      const hasGraph = graph.nodes.length > 0 || graph.edges.length > 0;
      const difficulty = difficultyTagText(graph);
      document.getElementById("scene-title").textContent = hasGraph ? String(graph.scene_id ?? graph.id ?? "view_graph") : "View Graph Workbench";
      document.getElementById("scene-subtitle").textContent = hasGraph ? `${graph.layout ?? "layout"} · ${graph.nodes.length} nodes · ${graph.edges.length} edges${difficulty ? " · " + difficulty : ""}` : "No graph loaded. Create a graph first.";
      renderLists();
      renderEditors();
      renderSvg();
      validateGraph();
    }

    function renderViewButtons() {
      document.getElementById("graph-view-btn").classList.toggle("active", state.viewMode === "graph");
      document.getElementById("map-view-btn").classList.toggle("active", state.viewMode === "map");
    }

    function validateGraph() {
      if (!graph.nodes.length && !graph.edges.length) {
        statusBox.textContent = "No graph loaded.";
        return;
      }
      const ids = new Set();
      const duplicateIds = [];
      const names = new Set();
      const duplicateNames = [];
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        if (ids.has(id)) duplicateIds.push(id);
        ids.add(id);
        const name = nodeName(node).trim();
        if (name) {
          if (names.has(name)) duplicateNames.push(name);
          names.add(name);
        }
      });
      const badEdges = [];
      graph.edges.forEach((edge, index) => {
        if (!ids.has(edgeFrom(edge)) || !ids.has(edgeTo(edge)) || !edgeRelation(edge)) badEdges.push(index + 1);
      });
      const parts = [`${graph.nodes.length} nodes`, `${graph.edges.length} edges`];
      if (duplicateIds.length) parts.push(`duplicate ids: ${duplicateIds.join(", ")}`);
      if (duplicateNames.length) parts.push(`duplicate names: ${duplicateNames.join(", ")}`);
      if (badEdges.length) parts.push(`invalid edges: ${badEdges.join(", ")}`);
      statusBox.textContent = (state.viewMode === "map" ? "Map layout inferred | " : "") + parts.join(" | ");
    }

    function renderLists() {
      nodeList.innerHTML = "";
      if (!graph.nodes.length) nodeList.innerHTML = '<div class="empty">Create a graph first.</div>';
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        const item = document.createElement("button");
        item.type = "button";
        item.className = "item" + (state.selectedNodeId === id ? " active" : "");
        item.innerHTML = `<div class="item-title">${escapeHtml(nodeName(node))}</div><div class="item-meta">${escapeHtml(id)} · ${escapeHtml(node.category ?? "object")}</div>`;
        item.addEventListener("click", () => { state.selectedNodeId = id; state.selectedEdgeIndex = null; render(); });
        nodeList.appendChild(item);
      });

      edgeList.innerHTML = "";
      if (!graph.edges.length) edgeList.innerHTML = '<div class="empty">No edges.</div>';
      graph.edges.forEach((edge, index) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "item" + (state.selectedEdgeIndex === index ? " active" : "");
        item.innerHTML = `<div class="item-title">${escapeHtml(edgeRelation(edge))}</div><div class="item-meta">${escapeHtml(edgeFrom(edge))} -> ${escapeHtml(edgeTo(edge))}</div>`;
        item.addEventListener("click", () => { state.selectedEdgeIndex = index; state.selectedNodeId = null; render(); });
        edgeList.appendChild(item);
      });
    }

    function renderEditors() {
      renderEdgeEditor();
      renderNodeEditor();
    }

    function renderEdgeEditor() {
      edgeEditor.innerHTML = '<div class="section-title">Selected Edge</div>';
      const edge = graph.edges[state.selectedEdgeIndex];
      if (!edge) {
        edgeEditor.insertAdjacentHTML("beforeend", '<div class="empty">Select an edge or add a new one.</div>');
        return;
      }
      edgeEditor.appendChild(selectField("From", edgeFrom(edge), graph.nodes.map(node => nodeId(node)), value => { setEdgeFrom(edge, value); render(); }));
      edgeEditor.appendChild(selectField("To", edgeTo(edge), graph.nodes.map(node => nodeId(node)), value => { setEdgeTo(edge, value); render(); }));
      edgeEditor.appendChild(selectField("Relation", edgeRelation(edge), RELATIONS, value => { setEdgeRelation(edge, value); render(); }));
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "btn danger";
      deleteBtn.textContent = "Delete Edge";
      deleteBtn.addEventListener("click", () => {
        graph.edges.splice(state.selectedEdgeIndex, 1);
        state.selectedEdgeIndex = null;
        render();
      });
      edgeEditor.appendChild(deleteBtn);
    }

    function renderNodeEditor() {
      nodeEditor.innerHTML = '<div class="section-title">Selected Node</div>';
      const node = getNode(state.selectedNodeId);
      if (!node) {
        nodeEditor.insertAdjacentHTML("beforeend", '<div class="empty">Select a node to edit metadata.</div>');
        return;
      }
      nodeEditor.appendChild(textField("ID", nodeId(node), value => {
        const oldId = nodeId(node);
        if (!value || oldId === value) return;
        if (getNode(value)) {
          setMessage(`Duplicate node id "${value}" is not allowed.`, true);
          render();
          return;
        }
        node.id = value;
        if (state.positions[oldId]) {
          state.positions[value] = state.positions[oldId];
          delete state.positions[oldId];
        }
        graph.edges.forEach(edge => {
          if (edgeFrom(edge) === oldId) setEdgeFrom(edge, value);
          if (edgeTo(edge) === oldId) setEdgeTo(edge, value);
        });
        state.selectedNodeId = value;
        render();
      }));
      nodeEditor.appendChild(textField("Name", node.name ?? "", value => {
        const name = value.trim();
        if (!name) {
          setMessage("Node name is required.", true);
          render();
          return;
        }
        if (isNodeNameTaken(name, nodeId(node))) {
          setMessage(`Duplicate node name "${name}" is not allowed.`, true);
          render();
          return;
        }
        node.name = name;
        render();
      }));
      nodeEditor.appendChild(textField("Category", node.category ?? "", value => { node.category = value; render(); }));
      nodeEditor.appendChild(textField("Parent", node.parent ?? "", value => { if (value) node.parent = value; else delete node.parent; render(); }));
      nodeEditor.appendChild(textAreaField("Properties JSON", JSON.stringify(node.properties ?? [], null, 2), value => {
        try { node.properties = JSON.parse(value || "[]"); render(); } catch (err) { statusBox.textContent = "Invalid properties JSON"; }
      }));
      nodeEditor.appendChild(textAreaField("States JSON", JSON.stringify(node.states ?? [], null, 2), value => {
        try { node.states = JSON.parse(value || "[]"); render(); } catch (err) { statusBox.textContent = "Invalid states JSON"; }
      }));
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "btn danger";
      deleteBtn.textContent = "Delete Node";
      deleteBtn.addEventListener("click", () => {
        const id = nodeId(node);
        graph.nodes = graph.nodes.filter(item => nodeId(item) !== id);
        graph.edges = graph.edges.filter(edge => edgeFrom(edge) !== id && edgeTo(edge) !== id);
        delete state.positions[id];
        state.selectedNodeId = null;
        state.selectedEdgeIndex = null;
        render();
      });
      nodeEditor.appendChild(deleteBtn);
    }

    function renderSvg() {
      const defs = svg.querySelector("defs");
      svg.innerHTML = "";
      if (defs) svg.appendChild(defs);
      const rect = svg.getBoundingClientRect();
      const width = Math.max(rect.width, 720);
      const height = Math.max(rect.height, 520);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      if (state.viewMode === "map") {
        renderMapSvg(width, height);
        return;
      }
      renderGraphSvg();
    }

    function renderGraphSvg() {
      graph.edges.forEach((edge, index) => {
        const from = state.positions[edgeFrom(edge)];
        const to = state.positions[edgeTo(edge)];
        if (!from || !to) return;
        const line = svgEl("line", { x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: "edge" + (state.selectedEdgeIndex === index ? " active" : "") });
        line.addEventListener("click", event => { event.stopPropagation(); state.selectedEdgeIndex = index; state.selectedNodeId = null; render(); });
        svg.appendChild(line);
        const label = svgEl("text", { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 - 6, class: "edge-label" });
        label.textContent = edgeRelation(edge);
        svg.appendChild(label);
      });
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        const pos = state.positions[id];
        const group = svgEl("g", { class: "node" + (state.selectedNodeId === id ? " active" : ""), transform: `translate(${pos.x},${pos.y})` });
        group.appendChild(svgEl("circle", { r: 24 }));
        const text = svgEl("text", { y: 42 });
        text.textContent = nodeName(node);
        group.appendChild(text);
        group.addEventListener("mousedown", event => {
          state.dragging = { id, dx: event.clientX - pos.x, dy: event.clientY - pos.y };
          state.selectedNodeId = id;
          state.selectedEdgeIndex = null;
          render();
        });
        group.addEventListener("click", event => {
          event.stopPropagation();
          state.selectedNodeId = id;
          state.selectedEdgeIndex = null;
          render();
        });
        svg.appendChild(group);
      });
    }

    function renderMapSvg(width, height) {
      const layout = buildMapLayout(width, height);
      layout.zones.forEach(zone => {
        const group = svgEl("g", { class: "map-zone" + (state.selectedNodeId === zone.id ? " active" : "") });
        group.appendChild(svgEl("rect", { x: zone.x, y: zone.y, width: zone.w, height: zone.h, rx: graph.layout === "tabletop" ? 22 : 12 }));
        const label = svgEl("text", { x: zone.x + zone.w / 2, y: zone.y + 34, "text-anchor": "middle" });
        label.textContent = zone.label;
        group.appendChild(label);
        if (!zone.synthetic) {
          group.addEventListener("click", event => {
            event.stopPropagation();
            state.selectedNodeId = zone.id;
            state.selectedEdgeIndex = null;
            render();
          });
        }
        svg.appendChild(group);
      });

      layout.items.forEach(item => {
        const node = getNode(item.id);
        const style = mapNodeStyle(node);
        const classes = [
          "map-object",
          item.nested ? "nested" : "",
          state.selectedNodeId === item.id ? "active" : "",
          isOccludedTarget(item.id) ? "occluded" : ""
        ].filter(Boolean).join(" ");
        const group = svgEl("g", { class: classes, transform: `translate(${item.x},${item.y})` });
        group.appendChild(svgEl("rect", { x: -item.w / 2, y: -item.h / 2, width: item.w, height: item.h, rx: item.nested ? 5 : 8, fill: style.fill, stroke: style.stroke }));
        if (isOccludedTarget(item.id)) {
          group.appendChild(svgEl("rect", { x: -item.w / 2 - 5, y: -item.h / 2 - 5, width: item.w + 10, height: item.h + 10, rx: item.nested ? 7 : 10, class: "occlusion-box" }));
        }
        const labelY = item.stackLabel ? (item.nested ? 9 : 5) : (item.nested ? 4 : -5);
        const label = svgEl("text", { y: labelY, class: "node-name" });
        label.textContent = item.label;
        group.appendChild(label);
        if (item.stackLabel) {
          const badge = svgEl("text", { y: -item.h / 2 + 10, class: "relation-badge" });
          badge.textContent = item.stackLabel;
          group.appendChild(badge);
        }
        if (!item.nested && !item.stackLabel) {
          const meta = svgEl("text", { y: 15, class: "node-meta" });
          meta.textContent = nodeCategory(node);
          group.appendChild(meta);
        }
        group.addEventListener("click", event => {
          event.stopPropagation();
          state.selectedNodeId = item.id;
          state.selectedEdgeIndex = null;
          render();
        });
        svg.appendChild(group);
      });
      renderMapLegend(width, height);
    }

    function buildMapLayout(width, height) {
      const padding = 36;
      const legendHeight = 48;
      const roomNodes = graph.nodes.filter(isRoomNode);
      const surfaceNodes = graph.nodes.filter(isSurfaceNode);
      const topSurfaceNodes = surfaceNodes.filter(node => !parentIdFor(node));
      const zoneNodes = graph.layout === "indoor" && roomNodes.length
        ? roomNodes
        : (topSurfaceNodes.length ? topSurfaceNodes : (surfaceNodes.length ? surfaceNodes.slice(0, 1) : roomNodes));
      const zones = zoneNodes.length ? zoneNodes.map(node => ({ id: nodeId(node), label: nodeName(node), synthetic: false })) : [{ id: "__scene__", label: String(graph.scene_id ?? "scene"), synthetic: true }];
      const cols = Math.ceil(Math.sqrt(zones.length));
      const rows = Math.ceil(zones.length / cols);
      const zoneW = (width - padding * 2 - (cols - 1) * 18) / cols;
      const zoneH = (height - padding * 2 - legendHeight - (rows - 1) * 18) / rows;
      zones.forEach((zone, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        zone.x = padding + col * (zoneW + 18);
        zone.y = padding + row * (zoneH + 18);
        zone.w = Math.max(180, zoneW);
        zone.h = Math.max(150, zoneH);
      });

      const zoneIds = new Set(zones.filter(zone => !zone.synthetic).map(zone => zone.id));
      const defaultZoneId = zones[0].id;
      const parentById = buildParentById(zoneIds);
      const childrenByParent = buildChildrenByParent(parentById, zoneIds);
      const dimensions = buildMapItemDimensions(parentById, childrenByParent, zoneIds);
      const buckets = new Map(zones.map(zone => [zone.id, []]));
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        if (zoneIds.has(id)) return;
        const zoneId = containingZoneId(node, zoneIds) || defaultZoneId;
        if (parentById[id] && !zoneIds.has(parentById[id])) return;
        if (!buckets.has(zoneId)) buckets.set(zoneId, []);
        buckets.get(zoneId).push(node);
      });

      const positions = {};
      zones.forEach(zone => {
        if (!zone.synthetic) positions[zone.id] = { x: zone.x + zone.w / 2, y: zone.y + 22 };
        const children = orderNodesByRelations(buckets.get(zone.id) || []);
        placeNodesInGrid(children, {
          minX: zone.x + 28,
          maxX: zone.x + zone.w - 28,
          minY: zone.y + 58,
          maxY: zone.y + zone.h - 28
        }, positions, dimensions);
      });
      applyNestedPositions(positions, parentById, childrenByParent, dimensions, zoneIds);

      const items = graph.nodes
        .filter(node => !zoneIds.has(nodeId(node)))
        .map(node => {
          const id = nodeId(node);
          const size = dimensions[id] || defaultMapItemSize(false, 0);
          return {
            id,
            label: nodeName(node),
            nested: Boolean(parentById[id] && !zoneIds.has(parentById[id])),
            stackLabel: relationLabelForParent(id, parentById[id]),
            depth: nodeDepth(id, parentById),
            w: size.w,
            h: size.h,
            ...(positions[id] || { x: width / 2, y: height / 2 })
          };
        });
      items.sort((left, right) => left.depth - right.depth);
      return { zones, items, positions };
    }

    function containingZoneId(node, zoneIds) {
      if (node.room && zoneIds.has(String(node.room))) return String(node.room);
      const seen = new Set();
      let parent = parentIdFor(node);
      while (parent && !seen.has(parent)) {
        if (zoneIds.has(parent)) return parent;
        seen.add(parent);
        const parentNode = getNode(parent);
        parent = parentNode ? parentIdFor(parentNode) : null;
      }
      return null;
    }

    function parentIdFor(node) {
      const id = nodeId(node);
      if (node.parent) return String(node.parent);
      if (node.part_of) return String(node.part_of);
      for (const edge of graph.edges) {
        if (edgeFrom(edge) === id && ["ON", "BENEATH", "UNDER", "BELOW", "INSIDE", "IN", "PART_OF"].includes(edgeRelation(edge))) return edgeTo(edge);
        if (edgeTo(edge) === id && ["CONTAINS", "HAS_INSIDE", "SUPPORTS", "HAS_ON"].includes(edgeRelation(edge))) return edgeFrom(edge);
      }
      return null;
    }

    function buildParentById(zoneIds) {
      const parentById = {};
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        if (zoneIds.has(id)) return;
        const parent = parentIdFor(node);
        if (parent && getNode(parent)) parentById[id] = parent;
      });
      applyOcclusionParents(parentById, zoneIds);
      return parentById;
    }

    function applyOcclusionParents(parentById, zoneIds) {
      graph.edges.forEach(edge => {
        if (edgeRelation(edge) !== "OCCLUDES") return;
        const occluderId = edgeFrom(edge);
        const occludedId = edgeTo(edge);
        if (!occluderId || !occludedId || occluderId === occludedId) return;
        if (zoneIds.has(occludedId)) return;
        if (!getNode(occluderId) || !getNode(occludedId)) return;
        if (wouldCreateParentCycle(parentById, occludedId, occluderId)) return;
        parentById[occludedId] = occluderId;
      });
    }

    function wouldCreateParentCycle(parentById, childId, parentId) {
      const seen = new Set([childId]);
      let current = parentId;
      while (current) {
        if (current === childId) return true;
        if (seen.has(current)) return true;
        seen.add(current);
        current = parentById[current];
      }
      return false;
    }

    function buildChildrenByParent(parentById, zoneIds) {
      const childrenByParent = new Map();
      Object.entries(parentById).forEach(([childId, parentId]) => {
        if (zoneIds.has(parentId)) return;
        if (!childrenByParent.has(parentId)) childrenByParent.set(parentId, []);
        childrenByParent.get(parentId).push(childId);
      });
      return childrenByParent;
    }

    function buildMapItemDimensions(parentById, childrenByParent, zoneIds) {
      const dimensions = {};
      graph.nodes.forEach(node => {
        const id = nodeId(node);
        const childCount = (childrenByParent.get(id) || []).length;
        const nested = Boolean(parentById[id] && !zoneIds.has(parentById[id]));
        dimensions[id] = defaultMapItemSize(nested, childCount);
      });
      return dimensions;
    }

    function defaultMapItemSize(nested, childCount) {
      if (childCount > 0) {
        const rows = Math.ceil(childCount / 2);
        return { w: nested ? 138 : 174, h: nested ? 72 + rows * 34 : 86 + rows * 38 };
      }
      return nested ? { w: 86, h: 30 } : { w: 116, h: 56 };
    }

    function placeNodesInGrid(nodes, bounds, positions, dimensions) {
      const children = orderNodesByRelations(nodes);
      if (!children.length) return;
      const gap = 14;
      const orientation = relationOrientation(children);
      const width = Math.max(1, bounds.maxX - bounds.minX);
      const height = Math.max(1, bounds.maxY - bounds.minY);
      const maxW = Math.max(...children.map(node => (dimensions[nodeId(node)] || defaultMapItemSize(false, 0)).w));
      const maxH = Math.max(...children.map(node => (dimensions[nodeId(node)] || defaultMapItemSize(false, 0)).h));
      let best = null;
      for (let cols = 1; cols <= children.length; cols += 1) {
        const rows = Math.ceil(children.length / cols);
        const scale = Math.min(1, (width / cols - gap) / maxW, (height / rows - gap) / maxH);
        let score = Math.min(scale, 1) - Math.abs(cols - rows) * 0.015;
        if (orientation === "horizontal") score += cols >= rows ? 0.08 : -0.08;
        if (orientation === "vertical") score += rows >= cols ? 0.08 : -0.08;
        if (!best || score > best.score) best = { cols, rows, scale: Math.max(0.35, Math.min(1, scale)), score };
      }
      const cols = best.cols;
      const cellW = width / cols;
      const cellH = height / best.rows;
      children.forEach((node, index) => {
        const id = nodeId(node);
        const size = dimensions[id] || defaultMapItemSize(false, 0);
        if (best.scale < 1) {
          size.w = Math.max(38, Math.floor(size.w * best.scale));
          size.h = Math.max(22, Math.floor(size.h * best.scale));
          dimensions[id] = size;
        }
        const col = index % cols;
        const row = Math.floor(index / cols);
        const x = bounds.minX + cellW * col + cellW / 2;
        const y = bounds.minY + cellH * row + cellH / 2;
        positions[id] = {
          x: clamp(x, bounds.minX + size.w / 2, bounds.maxX - size.w / 2),
          y: clamp(y, bounds.minY + size.h / 2, bounds.maxY - size.h / 2)
        };
      });
    }

    function relationOrientation(nodes) {
      const ids = new Set(nodes.map(node => nodeId(node)));
      let horizontal = 0;
      let vertical = 0;
      graph.edges.forEach(edge => {
        if (!ids.has(edgeFrom(edge)) || !ids.has(edgeTo(edge))) return;
        const relation = edgeRelation(edge);
        if (["LEFT_OF", "RIGHT_OF"].includes(relation)) horizontal += 1;
        if (["FRONT_OF", "BEHIND"].includes(relation)) vertical += 1;
      });
      if (horizontal > vertical) return "horizontal";
      if (vertical > horizontal) return "vertical";
      return "balanced";
    }

    function applyNestedPositions(positions, parentById, childrenByParent, dimensions, zoneIds) {
      const visited = new Set();
      const placeChildren = parentId => {
        if (visited.has(parentId)) return;
        visited.add(parentId);
        const childIds = childrenByParent.get(parentId) || [];
        if (!childIds.length || !positions[parentId]) return;
        const parentSize = dimensions[parentId] || defaultMapItemSize(false, childIds.length);
        const children = orderNodesByRelations(childIds.map(getNode).filter(Boolean));
        placeNodesInGrid(children, {
          minX: positions[parentId].x - parentSize.w / 2 + 10,
          maxX: positions[parentId].x + parentSize.w / 2 - 10,
          minY: positions[parentId].y - parentSize.h / 2 + 28,
          maxY: positions[parentId].y + parentSize.h / 2 - 10
        }, positions, dimensions);
        children.forEach(node => placeChildren(nodeId(node)));
      };

      graph.nodes.forEach(node => {
        const id = nodeId(node);
        if (zoneIds.has(id)) return;
        const parentId = parentById[id];
        if (!parentId || zoneIds.has(parentId)) placeChildren(id);
      });
    }

    function orderNodesByRelations(nodes) {
      const order = new Map(nodes.map((node, index) => [nodeId(node), index]));
      return [...nodes].sort((left, right) => {
        const leftScore = relationScore(nodeId(left));
        const rightScore = relationScore(nodeId(right));
        if (leftScore.y !== rightScore.y) return leftScore.y - rightScore.y;
        if (leftScore.x !== rightScore.x) return leftScore.x - rightScore.x;
        return order.get(nodeId(left)) - order.get(nodeId(right));
      });
    }

    function relationScore(id) {
      const score = { x: 0, y: 0 };
      graph.edges.forEach(edge => {
        const relation = edgeRelation(edge);
        if (edgeFrom(edge) === id) {
          if (relation === "LEFT_OF") score.x -= 1;
          if (relation === "RIGHT_OF") score.x += 1;
          if (relation === "FRONT_OF") score.y += 1;
          if (relation === "BEHIND") score.y -= 1;
        }
        if (edgeTo(edge) === id) {
          if (relation === "LEFT_OF") score.x += 1;
          if (relation === "RIGHT_OF") score.x -= 1;
          if (relation === "FRONT_OF") score.y -= 1;
          if (relation === "BEHIND") score.y += 1;
        }
      });
      return score;
    }

    function isMapRelation(relation) {
      return ["ON", "BENEATH", "UNDER", "BELOW", "INSIDE", "IN", "PART_OF", "LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "OCCLUDES", "CLOSE", "NEAR", "CONNECTED"].includes(relation);
    }

    function nodeDepth(id, parentById) {
      let depth = 0;
      const seen = new Set();
      let parent = parentById[id];
      while (parent && !seen.has(parent)) {
        seen.add(parent);
        depth += 1;
        parent = parentById[parent];
      }
      return depth;
    }

    function relationLabelForParent(id, parentId) {
      if (!parentId) return "";
      for (const edge of graph.edges) {
        if (edgeFrom(edge) !== id || edgeTo(edge) !== parentId) continue;
        const relation = edgeRelation(edge);
        if (relation === "ON") return "on";
        if (["BENEATH", "UNDER", "BELOW"].includes(relation)) return "beneath";
      }
      return "";
    }

    function isRoomNode(node) {
      return String(node.category ?? "").toLowerCase() === "room";
    }

    function isSurfaceNode(node) {
      const category = String(node.category ?? "").toLowerCase();
      const props = normalisedProps(node);
      return props.has("SURFACES") || ["surface", "furniture", "table", "counter", "workspace"].includes(category);
    }

    function isContainerNode(node) {
      if (!node) return false;
      const category = String(node.category ?? "").toLowerCase();
      const props = normalisedProps(node);
      return props.has("CONTAINERS") || ["container", "receptacle"].includes(category);
    }

    function nodeCategory(node) {
      return String(node?.category ?? "object");
    }

    function mapNodeStyle(node) {
      const category = nodeCategory(node).toLowerCase();
      const props = normalisedProps(node || {});
      if (props.has("SURFACES") || category === "surface") return { fill: "#e9eef5", stroke: "#48566a" };
      if (props.has("CONTAINERS") || ["container", "receptacle"].includes(category)) return { fill: "#fff4d6", stroke: "#a56b18" };
      if (category === "food") return { fill: "#ffe2dc", stroke: "#a34232" };
      if (category === "tool") return { fill: "#e1f4ee", stroke: "#27755f" };
      if (props.has("GRABBABLE") || props.has("MOVABLE")) return { fill: "#e8e4ff", stroke: "#5b4ab3" };
      return { fill: "#ffffff", stroke: "#52616f" };
    }

    function isOccludedTarget(id) {
      return graph.edges.some(edge => edgeTo(edge) === id && edgeRelation(edge) === "OCCLUDES");
    }

    function isOccluderSource(id) {
      return graph.edges.some(edge => edgeFrom(edge) === id && edgeRelation(edge) === "OCCLUDES");
    }

    function normalisedProps(node) {
      return new Set((Array.isArray(node.properties) ? node.properties : []).map(value => String(value).trim().toUpperCase().replaceAll(" ", "_")));
    }

    function renderMapLegend(width, height) {
      const items = [
        { label: "parent contains child", fill: "#fff4d6", stroke: "#a56b18" },
        { label: "occluded target", fill: "none", stroke: "#dc2626", dash: "7 5" },
        { label: "top-down placement", fill: "#ffffff", stroke: "#52616f" }
      ];
      const legendW = 210;
      const legendH = 86;
      const startX = Math.max(16, width - legendW - 18);
      const startY = 16;
      const group = svgEl("g", { class: "map-legend" });
      group.appendChild(svgEl("rect", { x: startX, y: startY, width: legendW, height: legendH, rx: 8, class: "map-legend-bg" }));
      items.forEach((item, index) => {
        const x = startX + 14;
        const y = startY + 23 + index * 22;
        group.appendChild(svgEl("rect", { x, y: y - 11, width: 24, height: 18, rx: 4, fill: item.fill, stroke: item.stroke, "stroke-dasharray": item.dash || "", class: "legend-swatch" }));
        const text = svgEl("text", { x: x + 34, y: y + 3 });
        text.textContent = item.label;
        group.appendChild(text);
      });
      svg.appendChild(group);
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function textField(labelText, value, onChange) {
      const label = document.createElement("label");
      label.textContent = labelText;
      const input = document.createElement("input");
      input.value = value;
      input.addEventListener("change", () => onChange(input.value.trim()));
      label.appendChild(input);
      return label;
    }

    function textAreaField(labelText, value, onChange) {
      const label = document.createElement("label");
      label.textContent = labelText;
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.addEventListener("change", () => onChange(textarea.value));
      label.appendChild(textarea);
      return label;
    }

    function selectField(labelText, value, options, onChange) {
      const wrap = document.createElement("label");
      wrap.textContent = labelText;
      const select = document.createElement("select");
      [...new Set([...options, value].filter(Boolean))].forEach(optionValue => {
        const option = document.createElement("option");
        option.value = optionValue;
        option.textContent = optionValue;
        option.selected = optionValue === value;
        select.appendChild(option);
      });
      select.addEventListener("change", () => onChange(select.value));
      wrap.appendChild(select);
      return wrap;
    }

    function bindFileToTextarea(fileInputId, textareaId) {
      const input = document.getElementById(fileInputId);
      const textarea = document.getElementById(textareaId);
      input.addEventListener("change", async () => {
        const file = input.files?.[0];
        if (!file) return;
        try {
          textarea.value = await file.text();
          setMessage(`Loaded ${file.name}.`);
        } catch (err) {
          setMessage(`Failed to load ${file.name}: ${String(err.message || err)}`, true);
        }
      });
    }

    function parseExistingViewGraphText(text) {
      const raw = String(text ?? "").trim();
      if (!raw) throw new Error("View graph JSON/JSONL is empty.");
      let payload;
      try {
        payload = JSON.parse(raw);
      } catch (jsonError) {
        const line = raw.split(/\\r?\\n/).map(item => item.trim()).find(item => item && !item.startsWith("#"));
        if (!line) throw new Error("View graph JSONL has no non-empty JSON line.");
        payload = JSON.parse(line);
      }
      if (Array.isArray(payload)) {
        if (!payload.length) throw new Error("View graph array is empty.");
        payload = payload[0];
      }
      const graphPayload = payload?.view_graph ?? payload?.viewGraph ?? payload?.graph ?? payload;
      if (!graphPayload || typeof graphPayload !== "object" || Array.isArray(graphPayload)) {
        throw new Error("Imported data must be a view graph object.");
      }
      if (!Array.isArray(graphPayload.nodes)) {
        throw new Error("Imported view graph must contain a nodes array.");
      }
      const imported = JSON.parse(JSON.stringify(graphPayload));
      imported.edges = Array.isArray(imported.edges) ? imported.edges : [];
      imported.layout = imported.layout || "tabletop";
      imported.robot = imported.robot || { arms: "double" };
      return imported;
    }

    function setLoadedGraph(nextGraph, message, options={}) {
      graph = nextGraph;
      if (!options.preserveSamples) {
        profiledSamples = [];
        profileBaseGraph = cloneGraph(nextGraph);
      }
      state.positions = {};
      state.selectedNodeId = graph.nodes?.[0]?.id ? String(graph.nodes[0].id) : null;
      state.selectedEdgeIndex = null;
      setMessage(message);
      render();
    }

    function importViewGraph() {
      try {
        const imported = parseExistingViewGraphText(document.getElementById("import-view-graph").value);
        setLoadedGraph(imported, `Imported ${imported.nodes?.length || 0} nodes and ${imported.edges?.length || 0} edges.`);
      } catch (err) {
        setMessage(String(err.message || err), true);
      }
    }

    function buildProfileFromControls() {
      return {
        profile_id: document.getElementById("profile-id").value.trim() || "ui_profile",
        spatial: {
          enabled: document.getElementById("spatial-enabled").checked,
          num_occluded_objects: integerValue("spatial-num-occluded", 1, 0),
          occlusion_depth: integerValue("spatial-occlusion-depth", 1, 0),
          num_decomposed_parents: integerValue("spatial-decomposed-parents", 0, 0)
        }
      };
    }

    function parseOptionalJsonTextarea(textareaId, fieldName) {
      const raw = document.getElementById(textareaId).value.trim();
      if (!raw) return null;
      try {
        return JSON.parse(raw);
      } catch (err) {
        throw new Error(`${fieldName} must be valid JSON.`);
      }
    }

    async function applyProfileEdit() {
      if (!Array.isArray(graph.nodes) || !graph.nodes.length) {
        setMessage("No graph loaded. Create or import a graph first.", true);
        return;
      }
      const button = document.getElementById("apply-profile-btn");
      const numSamples = integerValue("profile-num-samples", 1, 1);
      document.getElementById("profile-num-samples").value = String(numSamples);
      const seed = document.getElementById("profile-seed").value.trim();
      const sourceGraph = profileSourceGraph();
      button.disabled = true;
      setMessage(`Applying profile to ${numSamples} sample${numSamples === 1 ? "" : "s"} from base graph...`);
      try {
        const placementEdgeConstraints = parseOptionalJsonTextarea("profile-placement-constraints", "Placement constraints");
        const requestPayload = {
          view_graph: sourceGraph,
          profile: buildProfileFromControls(),
          num_samples: numSamples,
          seed
        };
        if (placementEdgeConstraints) {
          requestPayload.placement_edge_constraints = placementEdgeConstraints;
        }
        const response = await fetch("/api/edit-view-graph", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestPayload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "profile edit failed");
        const samples = Array.isArray(data.view_graphs) ? data.view_graphs : (data.view_graph ? [data.view_graph] : []);
        if (!samples.length) throw new Error("profile edit returned no graphs");
        profiledSamples = samples;
        setLoadedGraph(samples[0], profileApplyMessage(samples.length, samples[0]), { preserveSamples: true });
      } catch (err) {
        setMessage(String(err.message || err), true);
      } finally {
        button.disabled = false;
      }
    }

    function profileSourceGraph() {
      if (isProfiledGraph(graph) && profileBaseGraph) {
        return cloneGraph(profileBaseGraph);
      }
      profileBaseGraph = cloneGraph(graph);
      return cloneGraph(graph);
    }

    function isProfiledGraph(targetGraph) {
      const metadata = targetGraph?.metadata || {};
      return Boolean(
        metadata.requested_constraint_profile ||
        metadata.achieved_constraint_profile ||
        metadata.profile_constraints ||
        metadata.difficulty_tags
      );
    }

    function cloneGraph(targetGraph) {
      return JSON.parse(JSON.stringify(targetGraph));
    }

    function profileApplyMessage(sampleCount, targetGraph) {
      const tags = difficultyTagText(targetGraph);
      return `Applied profile to ${sampleCount} sample${sampleCount === 1 ? "" : "s"} from base graph.${tags ? "\\n" + tags : ""}`;
    }

    function difficultyTagText(targetGraph) {
      const tags = targetGraph?.metadata?.difficulty_tags;
      if (!tags || typeof tags !== "object" || Array.isArray(tags)) return "";
      const values = [];
      Object.entries(tags).forEach(([dimension, items]) => {
        if (Array.isArray(items)) {
          items.forEach(item => values.push(String(item)));
        } else if (items && typeof items === "object") {
          Object.entries(items).forEach(([key, value]) => values.push(`${dimension}.${key}=${value}`));
        } else if (items != null) {
          values.push(`${dimension}=${items}`);
        }
      });
      return values.join(" · ");
    }

    function integerValue(id, fallback=0, minimum=0) {
      const input = document.getElementById(id);
      const parsed = Number.parseInt(input?.value ?? "", 10);
      if (!Number.isFinite(parsed)) return fallback;
      return Math.max(parsed, minimum);
    }

    function bindSliderValue(inputId, valueId) {
      const input = document.getElementById(inputId);
      const value = document.getElementById(valueId);
      const update = () => { value.textContent = input.value; };
      input.addEventListener("input", update);
      update();
    }

    async function createViewGraph(event) {
      event.preventDefault();
      const button = document.getElementById("create-btn");
      button.disabled = true;
      setMessage("Creating view graph...");
      const payload = {
        materials_text: document.getElementById("materials").value,
        material_properties_text: document.getElementById("material-properties").value,
        scene: document.getElementById("scene").value,
        task_hint: document.getElementById("task-hint").value,
        layout: document.getElementById("layout").value,
        arms: document.getElementById("arms").value,
        provider: document.getElementById("provider").value,
        model: document.getElementById("model").value,
        api_key: document.getElementById("api-key").value,
        api_key_env: document.getElementById("api-key-env").value,
        api_base_url: document.getElementById("api-base-url").value,
        scene_id: document.getElementById("scene-id").value,
        env_id: document.getElementById("env-id").value,
        timeout_seconds: document.getElementById("timeout-seconds").value,
        enable_thinking: document.getElementById("enable-thinking").checked
      };
      try {
        const response = await fetch("/api/create-view-graph", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "create failed");
        const created = data.view_graph || data;
        setLoadedGraph(created, `Created ${created.nodes?.length || 0} nodes and ${created.edges?.length || 0} edges.`);
      } catch (err) {
        setMessage(String(err.message || err), true);
      } finally {
        button.disabled = false;
      }
    }

    function addEdge() {
      const from = state.selectedNodeId || nodeId(graph.nodes[0] || {});
      const to = nodeId(graph.nodes.find(node => nodeId(node) !== from) || graph.nodes[0] || {});
      if (!from || !to) return;
      graph.edges.push({ from, to, relation: "ON" });
      state.selectedEdgeIndex = graph.edges.length - 1;
      state.selectedNodeId = null;
      render();
    }

    function nextNodeId() {
      const ids = new Set(graph.nodes.map(node => nodeId(node)));
      const names = new Set(graph.nodes.map(node => nodeName(node).trim()).filter(Boolean));
      let index = graph.nodes.length + 1;
      let id = `node_${index}`;
      while (ids.has(id) || names.has(id)) {
        index += 1;
        id = `node_${index}`;
      }
      return id;
    }

    function addNode() {
      const id = nextNodeId();
      if (isNodeNameTaken(id)) {
        setMessage(`Duplicate node name "${id}" is not allowed.`, true);
        return;
      }
      graph.nodes.push({ id, name: id, category: "object", properties: [] });
      const rect = svg.getBoundingClientRect();
      state.positions[id] = { x: Math.max(rect.width, 720) / 2, y: Math.max(rect.height, 520) / 2 };
      state.selectedNodeId = id;
      state.selectedEdgeIndex = null;
      render();
    }

    function isNodeNameTaken(name, exceptId=null) {
      const target = String(name ?? "").trim();
      if (!target) return false;
      return graph.nodes.some(node => nodeId(node) !== exceptId && nodeName(node).trim() === target);
    }

    function downloadJsonl() {
      if (!Array.isArray(graph.nodes) || !graph.nodes.length) {
        setMessage("No graph loaded. Create or import a graph first.", true);
        return;
      }
      downloadGraphsJsonl([graph], "edited");
    }

    function downloadProfiledSamples() {
      const samples = profiledSamples.length ? profiledSamples : (Array.isArray(graph.nodes) && graph.nodes.length ? [graph] : []);
      if (!samples.length) {
        setMessage("No graph loaded. Apply a profile first.", true);
        return;
      }
      downloadGraphsJsonl(samples, "profiled");
    }

    function downloadGraphsJsonl(graphs, suffix) {
      const jsonl = graphs.map(item => JSON.stringify(item)).join("\\n") + "\\n";
      const blob = new Blob([jsonl], { type: "application/jsonl;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const scene = String(graphs[0]?.scene_id ?? "view_graph").replace(/[^\\w\\u4e00-\\u9fff-]+/g, "_");
      link.href = url;
      link.download = `${scene}_${suffix}.jsonl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
    }

    function svgEl(name, attrs) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }

    window.addEventListener("mousemove", event => {
      if (!state.dragging) return;
      state.positions[state.dragging.id] = { x: event.clientX - state.dragging.dx, y: event.clientY - state.dragging.dy };
      renderSvg();
    });
    window.addEventListener("mouseup", () => { state.dragging = null; });
    window.addEventListener("resize", () => renderSvg());
    svg.addEventListener("click", () => { state.selectedEdgeIndex = null; render(); });
    document.getElementById("create-form").addEventListener("submit", createViewGraph);
    bindFileToTextarea("materials-file", "materials");
    bindFileToTextarea("material-properties-file", "material-properties");
    bindFileToTextarea("view-graph-file", "import-view-graph");
    bindFileToTextarea("placement-constraints-file", "profile-placement-constraints");
    document.getElementById("import-view-graph-btn").addEventListener("click", importViewGraph);
    document.getElementById("apply-profile-btn").addEventListener("click", applyProfileEdit);
    document.getElementById("download-profiled-btn").addEventListener("click", downloadProfiledSamples);
    document.getElementById("add-node-btn").addEventListener("click", addNode);
    document.getElementById("add-edge-btn").addEventListener("click", addEdge);
    document.getElementById("download-btn").addEventListener("click", downloadJsonl);
    document.getElementById("fit-btn").addEventListener("click", () => { initPositions(true); render(); });
    document.getElementById("graph-view-btn").addEventListener("click", () => { state.viewMode = "graph"; render(); });
    document.getElementById("map-view-btn").addEventListener("click", () => { state.viewMode = "map"; state.dragging = null; render(); });
    bindSliderValue("spatial-num-occluded", "spatial-num-occluded-value");
    bindSliderValue("spatial-occlusion-depth", "spatial-occlusion-depth-value");
    bindSliderValue("spatial-decomposed-parents", "spatial-decomposed-parents-value");
    render();
  </script>
</body>
</html>
"""
