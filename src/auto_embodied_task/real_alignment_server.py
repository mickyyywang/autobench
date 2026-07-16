from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote
import webbrowser

from . import real_alignment as _alignment


def create_real_alignment_app(
    *,
    trajectory_dir: str | Path | None = None,
    saved_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    alignment_path: str | Path | None = None,
    oss_region: str = "cn-shanghai",
    oss_endpoint: str | None = None,
    ossutil_bin: str = "ossutil",
):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised by CLI users without dependency installed
        raise RuntimeError(
            "FastAPI UI dependencies are missing. Install the project with web extras or run: "
            "pip install fastapi uvicorn"
        ) from exc
    globals()["Request"] = Request

    app = FastAPI(title="Auto Embodied Task Real Alignment")
    app.state.trajectory_dir = Path(trajectory_dir or _alignment.default_outputs_dir()).resolve()
    app.state.saved_dir = Path(saved_dir or _alignment.default_saved_dir()).resolve()
    app.state.cache_dir = Path(cache_dir or _alignment.default_cache_dir()).resolve()
    app.state.initial_alignment_path = str(Path(alignment_path).resolve()) if alignment_path else None
    app.state.oss_region = oss_region
    app.state.oss_endpoint = oss_endpoint
    app.state.ossutil_bin = ossutil_bin

    def client() -> _alignment.OssUtilClient:
        return _alignment.OssUtilClient(
            region=app.state.oss_region,
            endpoint=app.state.oss_endpoint,
            ossutil_bin=app.state.ossutil_bin,
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(_render_alignment_html())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "trajectory_dir": str(app.state.trajectory_dir),
            "saved_dir": str(app.state.saved_dir),
            "cache_dir": str(app.state.cache_dir),
            "initial_alignment_path": app.state.initial_alignment_path,
            "oss_region": app.state.oss_region,
        }

    @app.get("/api/trajectory-files")
    def trajectory_files() -> dict[str, Any]:
        try:
            files = _alignment.trajectory_files(app.state.trajectory_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"trajectory_dir": str(app.state.trajectory_dir), "files": files}

    @app.get("/api/trajectory")
    def trajectory(file: str) -> dict[str, Any]:
        try:
            path = _alignment.resolve_trajectory_path(file, app.state.trajectory_dir)
            return _alignment.trajectory_summary(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/saved-alignment-files")
    def saved_alignment_files() -> dict[str, Any]:
        try:
            files = _alignment.trajectory_files(app.state.saved_dir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "saved_dir": str(app.state.saved_dir),
            "initial_alignment_path": app.state.initial_alignment_path,
            "files": files,
        }

    @app.get("/api/saved-alignment")
    def saved_alignment(file: str, episode_index: int = 0) -> dict[str, Any]:
        try:
            path = _alignment.resolve_trajectory_path(file, app.state.saved_dir)
            return _alignment.saved_alignment_summary(
                path,
                trajectory_root=app.state.trajectory_dir,
                episode_index=episode_index,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/oss-dataset")
    def oss_dataset(oss_root: str, max_episodes: int | None = None) -> dict[str, Any]:
        try:
            return _alignment.load_lerobot_dataset(
                oss_root,
                client=client(),
                region=app.state.oss_region,
                endpoint=app.state.oss_endpoint,
                max_episodes=max_episodes,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/initial-alignment")
    async def initial_alignment(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            path = _alignment.resolve_trajectory_path(body["trajectory_file"], app.state.trajectory_dir)
            episode_index = int(body.get("episode_index", 0))
            episodes = _alignment.load_trajectory_jsonl(path)
            real_episodes = list(body.get("real_episodes") or [])
            return {
                "rows": _alignment.build_initial_alignment(episodes[episode_index], real_episodes),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cache-episode")
    async def cache_episode(request: Request) -> dict[str, Any]:
        body = await request.json()
        try:
            cached = _alignment.cache_episode_assets(
                body["oss_root"],
                int(body["episode_index"]),
                client=client(),
                region=app.state.oss_region,
                endpoint=app.state.oss_endpoint,
                cache_root=app.state.cache_dir,
                force=bool(body.get("force", False)),
            )
            cached["video_urls"] = {
                camera: _cache_url(Path(path), app.state.cache_dir) for camera, path in cached["videos"].items()
            }
            return cached
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/cache/{relative_path:path}")
    def cached_file(relative_path: str):
        root = Path(app.state.cache_dir).resolve()
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target)

    @app.post("/api/save-alignment")
    async def save_alignment(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            path = _alignment.resolve_trajectory_path(body["trajectory_file"], app.state.trajectory_dir)
            result = _alignment.save_aligned_episode(
                trajectory_path=path,
                episode_index=int(body.get("episode_index", 0)),
                oss_root=body["oss_root"],
                rows=list(body.get("rows") or []),
                real_episodes=list(body.get("real_episodes") or []),
                saved_dir=app.state.saved_dir,
                output_name=body.get("output_name") or None,
            )
            return JSONResponse(result)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve_real_alignment_app(
    *,
    trajectory_dir: str | Path | None = None,
    saved_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    alignment_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8767,
    oss_region: str = "cn-shanghai",
    oss_endpoint: str | None = None,
    ossutil_bin: str = "ossutil",
    open_browser: bool = False,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised by CLI users without dependency installed
        raise RuntimeError(
            "uvicorn is missing. Install the project with web extras or run: pip install fastapi uvicorn"
        ) from exc

    app = create_real_alignment_app(
        trajectory_dir=trajectory_dir,
        saved_dir=saved_dir,
        cache_dir=cache_dir,
        alignment_path=alignment_path,
        oss_region=oss_region,
        oss_endpoint=oss_endpoint,
        ossutil_bin=ossutil_bin,
    )
    url = f"http://{host}:{port}/"
    print(f"Real alignment UI running at {url}")
    print(f"Trajectory directory: {app.state.trajectory_dir}")
    print(f"Saved directory: {app.state.saved_dir}")
    print(f"Cache directory: {app.state.cache_dir}")
    print(f"OSS region: {oss_region}")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port)


def _cache_url(path: Path, cache_root: Path) -> str:
    relative = path.resolve().relative_to(Path(cache_root).resolve())
    return "/cache/" + quote(relative.as_posix())


def _render_alignment_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real Trajectory Alignment</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-2: #f9faf7;
      --line: #d6dde3;
      --ink: #1f272e;
      --muted: #687581;
      --accent: #236d86;
      --accent-2: #586b2f;
      --warn: #a36518;
      --bad: #a43d46;
      --good: #24754d;
      --shadow: 0 1px 2px rgba(31, 39, 46, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select, textarea {
      font: inherit;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    header {
      display: grid;
      grid-template-columns: minmax(220px, 0.9fr) minmax(300px, 1.25fr) minmax(340px, 1.4fr) auto;
      gap: 12px;
      align-items: end;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    label {
      display: grid;
      gap: 5px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    input, select, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      min-height: 34px;
      padding: 7px 9px;
      outline: none;
      min-width: 0;
      max-width: 100%;
    }
    textarea {
      min-height: 64px;
      resize: vertical;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(35, 109, 134, 0.14);
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      padding: 7px 10px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button.good {
      background: var(--accent-2);
      color: #fff;
      border-color: var(--accent-2);
    }
    button.ghost {
      background: transparent;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    main {
      display: grid;
      grid-template-columns: 250px minmax(620px, 1.15fr) minmax(480px, 0.85fr);
      gap: 12px;
      padding: 12px;
      min-height: 0;
      overflow: hidden;
    }
    section {
      min-height: 0;
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .section-head {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    h1, h2, h3 {
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
    }
    .scroll {
      overflow: auto;
      min-height: 0;
    }
    .stack {
      display: grid;
      gap: 10px;
      padding: 12px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .episode-list {
      display: grid;
      gap: 6px;
    }
    .list-item {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      cursor: pointer;
    }
    .list-item.selected {
      border-color: var(--accent);
      background: #eef7fa;
    }
    .list-item.used:not(.selected) {
      background: #f3f6f4;
      color: var(--muted);
    }
    .list-item strong {
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
    }
    thead {
      position: sticky;
      top: 0;
      background: var(--panel-2);
      z-index: 5;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
    }
    th {
      text-align: left;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    tr.selected {
      background: #edf7fb;
    }
    tbody tr { cursor: grab; }
    tbody tr.dragging { opacity: 0.45; }
    tbody tr.drop-target { box-shadow: inset 0 3px 0 var(--accent); }
    .tiny {
      font-size: 11px;
      color: var(--muted);
    }
    .action-name {
      font-weight: 750;
      color: var(--accent);
    }
    .step-stack {
      display: grid;
      gap: 5px;
    }
    .step-line {
      border-left: 3px solid var(--line);
      padding-left: 7px;
      overflow-wrap: anywhere;
    }
    .step-line:first-child { border-left-color: var(--accent); }
    .manual-name {
      width: 100%;
      min-width: 0;
      font-weight: 700;
    }
    .episode-chips {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 5px;
    }
    .episode-chip {
      min-height: 24px;
      padding: 2px 6px;
      color: var(--accent);
      border-color: #9fc5d1;
      background: #eef7fa;
      font-size: 11px;
    }
    .list-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      margin-top: 6px;
    }
    .list-actions button {
      min-height: 27px;
      padding: 3px 5px;
      font-size: 11px;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .summary-cell {
      min-width: 0;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
    }
    .summary-cell strong {
      display: block;
      font-size: 18px;
    }
    .current-source {
      display: grid;
      gap: 5px;
      padding: 9px;
      border-left: 3px solid var(--accent);
      background: #f3f8fa;
      overflow-wrap: anywhere;
    }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 11px;
      border: 1px solid var(--line);
      white-space: nowrap;
    }
    .status.matched { color: var(--good); background: #edf8f1; border-color: #b8d9c4; }
    .status.skipped { color: var(--warn); background: #fff6e7; border-color: #e5c58f; }
    .status.unmatched { color: var(--bad); background: #fff0f1; border-color: #e2b8bd; }
    .ops {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px;
    }
    .ops button {
      min-height: 28px;
      padding: 4px 6px;
      font-size: 12px;
    }
    .details {
      display: grid;
      gap: 12px;
      padding: 12px;
      min-height: 0;
      align-content: start;
    }
    .video-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .video-grid > div { min-width: 0; }
    .camera-head { grid-column: 1 / -1; }
    .camera-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111;
      border-radius: 6px;
      border: 1px solid var(--line);
    }
    .camera-head video { max-height: 46vh; }
    .tail-picker {
      display: grid;
      gap: 9px;
      padding: 10px;
      border: 1px solid #b7ccd4;
      border-radius: 6px;
      background: #f3f8fa;
    }
    .tail-picker[hidden] { display: none; }
    .tail-frame-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .tail-frame {
      display: grid;
      gap: 5px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      padding: 8px 10px;
      font-size: 13px;
      font-weight: 700;
      background: var(--panel-2);
    }
    details pre { border-radius: 0; }
    pre {
      margin: 0;
      padding: 9px;
      overflow: auto;
      max-height: 220px;
      background: #111820;
      color: #e5edf5;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
    }
    .notice {
      min-height: 28px;
      color: var(--muted);
      font-size: 12px;
      padding-top: 4px;
    }
    .notice.error { color: var(--bad); }
    .notice.ok { color: var(--good); }
    @media (max-width: 1450px) {
      main { grid-template-columns: 190px minmax(620px, 1fr) minmax(410px, 0.75fr); }
      th:nth-child(1), td:nth-child(1) { width: 38px !important; }
      th:nth-child(3), td:nth-child(3) { width: 112px !important; }
      th:nth-child(4), td:nth-child(4) { width: 90px !important; }
      th:nth-child(5), td:nth-child(5) { display: none; }
      th:nth-child(6), td:nth-child(6) { width: 136px !important; }
    }
    @media (max-width: 1100px) {
      header { grid-template-columns: 1fr; }
      .app { height: auto; min-height: 100vh; overflow: visible; }
      main { grid-template-columns: 1fr; }
      section { height: min(720px, calc(100vh - 40px)); min-height: 420px; }
    }
    @media (max-width: 680px) {
      .video-grid { grid-template-columns: 1fr; }
      .camera-head { grid-column: auto; }
    }
  </style>
</head>
<body>
<div class="app">
  <header>
    <label>Traj 文件
      <select id="trajectorySelect"></select>
    </label>
    <label>已保存 Alignment 路径
      <input id="savedAlignmentPath" list="savedAlignmentFiles" spellcheck="false" placeholder="输入 saved/*.jsonl 的完整路径">
      <datalist id="savedAlignmentFiles"></datalist>
    </label>
    <label>真机数据 OSS 目录
      <input id="ossRoot" spellcheck="false" value="oss://brain-imagegen-sh/users/wuzhao/data/robot/telep/galaxea_r1lite/data/raw/galaxea_r1lite_20260713_165639_192.168.31.142/">
    </label>
    <div class="row">
      <button id="loadTrajectory">加载 Traj</button>
      <button id="loadSavedAlignment">加载结果</button>
      <button id="loadOss">加载 OSS</button>
      <button id="save" class="primary">保存配对</button>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>当前配对</h2>
        <span id="sourceMeta" class="meta"></span>
      </div>
      <div class="scroll stack">
        <div id="currentSource" class="current-source"></div>
        <label>Traj 内的任务 Episode
          <select id="episodeSelect"></select>
        </label>
        <div id="pairingSummary" class="summary-grid"></div>
        <div>
          <div class="row" style="justify-content:space-between; margin-bottom:6px;">
            <h3>真机 Episodes</h3>
            <span class="tiny">替换或并入选中行</span>
          </div>
          <input id="realEpisodeSearch" placeholder="筛选 episode" style="width:100%; margin-bottom:7px;">
          <div id="realEpisodeList" class="episode-list"></div>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>Step 与真机 Episode 配对</h2>
          <div id="alignmentMeta" class="meta"></div>
        </div>
        <div class="row">
          <button id="undo" disabled title="撤销上一步操作（Ctrl/Cmd+Z）">↶ 撤销</button>
          <button id="autoAlign">按顺序重置</button>
          <button id="addRow">增加空行</button>
        </div>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th style="width:52px;">拖动</th>
              <th>Traj step 组</th>
              <th style="width:148px;">真机 Episode</th>
              <th style="width:108px;">状态</th>
              <th style="width:130px;">备注</th>
              <th style="width:150px;">操作</th>
            </tr>
          </thead>
          <tbody id="alignmentBody"></tbody>
        </table>
      </div>
    </section>

    <section>
      <div class="section-head">
        <div>
          <h2>真机视频预览</h2>
          <div id="previewMeta" class="meta"></div>
        </div>
        <div class="row">
          <select id="previewEpisodeSelect" title="选择合并组中的视频段" style="max-width:130px;"></select>
          <button id="preview" class="good">加载视频</button>
        </div>
      </div>
      <div class="scroll details">
        <div class="video-grid">
          <div class="camera-head">
            <div class="camera-title"><strong>Head camera</strong><span class="tiny">完整画面</span></div>
            <video id="videoHead" controls muted></video>
          </div>
          <div>
            <div class="camera-title"><strong>Left wrist</strong></div>
            <video id="videoLeft" controls muted></video>
          </div>
          <div>
            <div class="camera-title"><strong>Right wrist</strong></div>
            <video id="videoRight" controls muted></video>
          </div>
        </div>
        <div id="tailPicker" class="tail-picker" hidden>
          <div>
            <h3>新增 step 的 Observation Tail</h3>
            <div class="tiny">选择一个 episode 和相机视频，并从当前播放位置记录两帧。保存后仅作为下一步 observation，不占用或配对该 episode。</div>
          </div>
          <div class="row">
            <label style="flex:1;">来源 episode
              <select id="tailEpisodeSelect"></select>
            </label>
            <label style="flex:1;">来源相机视频
              <select id="tailCameraSelect">
                <option value="observation.images.head_rgb">Head camera</option>
                <option value="observation.images.left_wrist_rgb">Left wrist</option>
                <option value="observation.images.right_wrist_rgb">Right wrist</option>
              </select>
            </label>
            <button id="loadTailEpisode">加载来源视频</button>
          </div>
          <div class="tail-frame-grid">
            <div class="tail-frame">
              <label>帧 1 时间（秒）
                <input id="tailFrame1" type="number" min="0" step="0.001" placeholder="播放后点击记录">
              </label>
              <div class="row">
                <button data-tail-capture="0">记录当前时间</button>
                <button data-tail-seek="0">预览此帧</button>
              </div>
            </div>
            <div class="tail-frame">
              <label>帧 2 时间（秒）
                <input id="tailFrame2" type="number" min="0" step="0.001" placeholder="播放后点击记录">
              </label>
              <div class="row">
                <button data-tail-capture="1">记录当前时间</button>
                <button data-tail-seek="1">预览此帧</button>
              </div>
            </div>
          </div>
          <div class="row">
            <button id="saveTailFrames" class="good">保存两帧</button>
            <button id="clearTailFrames">清除自定义 Tail</button>
            <span id="tailSelectionMeta" class="tiny"></span>
          </div>
        </div>
        <details>
          <summary>选中行与真机数据详情</summary>
          <pre id="selectedJson">{}</pre>
        </details>
        <details>
          <summary>View graph 辅助信息</summary>
          <pre id="viewGraphJson">{}</pre>
        </details>
        <div id="notice" class="notice"></div>
      </div>
    </section>
  </main>
</div>

<script>
const state = {
  files: [],
  trajectory: null,
  selectedFile: "",
  selectedEpisodeIndex: 0,
  oss: null,
  rows: [],
  selectedRowIndex: 0,
  history: [],
  dragIndex: null,
  savedFiles: [],
  loadedSavedPath: null,
  displaySourceName: null,
  canSave: true,
  loadedPreviewEpisodeIndex: null,
};

const el = (id) => document.getElementById(id);

function cloneRows(rows) {
  return JSON.parse(JSON.stringify(rows));
}

function pushHistory(label) {
  state.history.push({
    label,
    rows: cloneRows(state.rows),
    selectedRowIndex: state.selectedRowIndex,
  });
  if (state.history.length > 50) state.history.shift();
  updateUndoButton();
}

function clearHistory() {
  state.history = [];
  updateUndoButton();
}

function undoLastOperation() {
  const snapshot = state.history.pop();
  if (!snapshot) return;
  state.rows = snapshot.rows;
  state.selectedRowIndex = Math.max(0, Math.min(snapshot.selectedRowIndex, state.rows.length - 1));
  renderWorkspace();
  updateUndoButton();
  setNotice(`已撤销：${snapshot.label}`, "ok");
}

function updateUndoButton() {
  const button = el("undo");
  if (!button) return;
  button.disabled = state.history.length === 0;
  button.textContent = state.history.length ? `↶ 撤销 (${state.history.length})` : "↶ 撤销";
}

function renderWorkspace() {
  renderAlignment();
  renderRealEpisodes();
  renderSourceContext();
  renderSelected();
}

function setNotice(message, kind = "") {
  const node = el("notice");
  node.textContent = message || "";
  node.className = "notice" + (kind ? " " + kind : "");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || payload.error || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return await response.json();
}

function actionLabel(action) {
  if (!action || typeof action !== "object") return "manual";
  const name = action.name || action.base_name || "action";
  const nodes = Array.isArray(action.node_ids) ? action.node_ids.join(", ") : "";
  return nodes ? `${name}(${nodes})` : name;
}

function currentEpisode() {
  if (!state.trajectory) return null;
  return state.trajectory.episodes[state.selectedEpisodeIndex] || null;
}

function currentRow() {
  return state.rows[state.selectedRowIndex] || null;
}

function realByIndex(index) {
  if (!state.oss || index === null || index === undefined || index === "") return null;
  return state.oss.episodes.find((item) => Number(item.episode_index) === Number(index)) || null;
}

async function loadFiles() {
  const payload = await api("/api/trajectory-files");
  state.files = payload.files || [];
  const select = el("trajectorySelect");
  select.innerHTML = "";
  state.files.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.path;
    option.textContent = file.name;
    select.appendChild(option);
  });
  if (state.files.length && !state.selectedFile) {
    state.selectedFile = state.files[0].path;
    select.value = state.selectedFile;
  }
  el("sourceMeta").textContent = `${state.files.length} 个 traj 文件`;
}

async function loadSavedFiles() {
  const payload = await api("/api/saved-alignment-files");
  state.savedFiles = payload.files || [];
  const datalist = el("savedAlignmentFiles");
  datalist.innerHTML = "";
  state.savedFiles.forEach((file) => {
    const option = document.createElement("option");
    option.value = file.path;
    option.label = file.name;
    datalist.appendChild(option);
  });
  if (payload.initial_alignment_path && !el("savedAlignmentPath").value) {
    el("savedAlignmentPath").value = payload.initial_alignment_path;
  }
  return payload.initial_alignment_path || null;
}

function selectedFileName() {
  if (state.displaySourceName) return state.displaySourceName;
  const file = state.files.find((item) => item.path === state.selectedFile);
  return file?.name || state.selectedFile.split("/").pop() || "未选择";
}

function sourceStepIndices(row) {
  if (Array.isArray(row?.traj_step_indices)) return row.traj_step_indices;
  return row?.traj_step_index === null || row?.traj_step_index === undefined
    ? []
    : [row.traj_step_index];
}

function realEpisodeIndices(row) {
  if (Array.isArray(row?.real_episode_indices)) return row.real_episode_indices.map(Number);
  return row?.real_episode_index === null || row?.real_episode_index === undefined
    ? []
    : [Number(row.real_episode_index)];
}

function syncPrimaryRealEpisode(row) {
  row.real_episode_index = realEpisodeIndices(row)[0] ?? null;
}

function observationTail(row) {
  const tail = row?.observation_tail;
  if (!tail || typeof tail !== "object") return null;
  const timestamps = Array.isArray(tail.timestamps) ? tail.timestamps.map(Number) : [];
  if (tail.episode_index === null || tail.episode_index === undefined || timestamps.length !== 2) return null;
  return {
    episode_index: Number(tail.episode_index),
    timestamps,
    camera: tail.camera || "observation.images.head_rgb",
    source: tail.source || "manual_frame_selection",
  };
}

function sourceStep(index) {
  return currentEpisode()?.steps?.[Number(index)] || null;
}

function renderSourceContext() {
  const episode = currentEpisode();
  el("currentSource").innerHTML = `
    <strong>${escapeHtml(selectedFileName())}</strong>
    <span class="tiny">${escapeHtml(episode?.episode_id || "未加载 episode")}</span>
    <span>${escapeHtml(episode?.task || episode?.task_type || "无任务描述")}</span>`;
  const sourceCount = episode?.step_count || 0;
  const matched = state.rows.filter((row) => row.status === "matched" && realEpisodeIndices(row).length).length;
  const merged = state.rows.filter((row) => sourceStepIndices(row).length > 1).length;
  const used = new Set(state.rows.flatMap(realEpisodeIndices));
  const mergedReal = state.rows.filter((row) => realEpisodeIndices(row).length > 1).length;
  el("pairingSummary").innerHTML = `
    <div class="summary-cell"><strong>${sourceCount}</strong><span class="tiny">原始 steps</span></div>
    <div class="summary-cell"><strong>${state.rows.length}</strong><span class="tiny">配对组</span></div>
    <div class="summary-cell"><strong>${matched}</strong><span class="tiny">已配对</span></div>
    <div class="summary-cell"><strong>${merged}/${mergedReal}</strong><span class="tiny">traj/real 合并组 · 已用 ${used.size} episodes</span></div>`;
}

async function loadTrajectory() {
  state.selectedFile = el("trajectorySelect").value;
  if (!state.selectedFile) return;
  setNotice("Loading trajectory...");
  state.trajectory = await api(`/api/trajectory?file=${encodeURIComponent(state.selectedFile)}`);
  state.loadedSavedPath = null;
  state.displaySourceName = null;
  state.canSave = true;
  el("save").disabled = false;
  state.selectedEpisodeIndex = 0;
  state.loadedPreviewEpisodeIndex = null;
  renderEpisodeSelect();
  resetAlignment(false);
  setNotice("Trajectory loaded.", "ok");
}

function ensureSourceFileOption(path) {
  if (!path) return;
  const select = el("trajectorySelect");
  if (![...select.options].some((option) => option.value === path)) {
    const option = document.createElement("option");
    option.value = path;
    option.textContent = path.split("/").pop();
    select.appendChild(option);
  }
  select.value = path;
}

async function loadSavedAlignment(path = el("savedAlignmentPath").value.trim()) {
  if (!path) {
    setNotice("请输入保存好的 alignment JSONL 路径。", "error");
    return;
  }
  setNotice("Loading saved alignment...");
  const payload = await api(`/api/saved-alignment?file=${encodeURIComponent(path)}`);
  state.loadedSavedPath = payload.path;
  state.displaySourceName = `${payload.name} · 已保存结果`;
  state.canSave = Boolean(payload.source_available);
  state.trajectory = payload.trajectory;
  state.selectedFile = payload.source_file || "";
  state.selectedEpisodeIndex = Number(payload.source_episode_index || 0);
  state.oss = payload.oss;
  state.rows = cloneRows(payload.rows || []);
  state.selectedRowIndex = 0;
  clearHistory();
  el("savedAlignmentPath").value = payload.path;
  el("ossRoot").value = payload.oss_root || "";
  ensureSourceFileOption(payload.source_file);
  renderEpisodeSelect();
  renderWorkspace();
  el("sourceMeta").textContent = payload.source_available
    ? `已恢复 ${state.rows.length} 个配对组 · 可继续编辑保存`
    : `已恢复 ${state.rows.length} 个配对组 · 原始 traj 不可用，只读展示`;
  el("save").disabled = !state.canSave;
  setNotice(
    payload.source_available
      ? `已加载保存结果：${payload.name}`
      : `已加载保存结果；找不到原始 traj，当前仅支持展示。`,
    payload.source_available ? "ok" : ""
  );
}

function renderEpisodeSelect() {
  const select = el("episodeSelect");
  select.innerHTML = "";
  (state.trajectory?.episodes || []).forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.index;
    option.textContent = `${episode.episode_id} (${episode.step_count})`;
    select.appendChild(option);
  });
  select.value = String(state.selectedEpisodeIndex);
  renderSourceContext();
}

async function loadOss() {
  const ossRoot = el("ossRoot").value.trim();
  if (!ossRoot) return;
  setNotice("Loading OSS dataset...");
  const preserveSavedRows = Boolean(state.loadedSavedPath && state.rows.length);
  state.oss = await api(`/api/oss-dataset?oss_root=${encodeURIComponent(ossRoot)}`);
  if (preserveSavedRows) renderWorkspace();
  else resetAlignment(false);
  setNotice(`OSS loaded: ${state.oss.episode_count} episodes`, "ok");
}

function renderRealEpisodes() {
  const list = el("realEpisodeList");
  list.innerHTML = "";
  const selectedHasRealEpisode = realEpisodeIndices(currentRow()).length > 0;
  const query = el("realEpisodeSearch").value.trim().toLowerCase();
  const usedBy = new Map();
  state.rows.forEach((row, index) => {
    realEpisodeIndices(row).forEach((realIndex) => usedBy.set(realIndex, index));
  });
  (state.oss?.episodes || []).filter((episode) => {
    const text = `${episode.episode_index} ${episode.task_name || ""} ${episode.episode_id || ""}`.toLowerCase();
    return !query || text.includes(query);
  }).forEach((episode) => {
    const rowIndex = usedBy.get(Number(episode.episode_index));
    const isSelected = rowIndex === state.selectedRowIndex;
    const item = document.createElement("div");
    item.className = "list-item" + (isSelected ? " selected" : rowIndex !== undefined ? " used" : "");
    item.innerHTML = `
      <strong>#${episode.episode_index} ${escapeHtml(episode.task_name || episode.episode_id)}</strong>
      <span class="tiny">${episode.frame_count || "?"} frames · ${episode.fps || "?"} fps${rowIndex !== undefined ? ` · 配对组 ${rowIndex + 1}` : " · 未使用"}</span>
      <div class="list-actions">
        <button data-assign="replace">${selectedHasRealEpisode ? "替换当前" : "分配当前"}</button>
        <button data-assign="merge">+ 并入当前</button>
      </div>`;
    item.querySelector('[data-assign="replace"]').addEventListener("click", () => {
      assignRealEpisode(episode.episode_index, state.selectedRowIndex, "replace");
    });
    item.querySelector('[data-assign="merge"]').addEventListener("click", () => {
      assignRealEpisode(episode.episode_index, state.selectedRowIndex, "merge");
    });
    list.appendChild(item);
  });
}

function assignRealEpisode(realIndex, targetRowIndex, mode = "replace") {
  const target = state.rows[targetRowIndex];
  if (!target) return;
  pushHistory(mode === "merge" ? "并入真机 episode" : "替换真机 episode");
  state.rows.forEach((row, index) => {
    if (
      realIndex !== null &&
      index !== targetRowIndex &&
      realEpisodeIndices(row).includes(Number(realIndex))
    ) {
      row.real_episode_indices = realEpisodeIndices(row).filter((value) => value !== Number(realIndex));
      syncPrimaryRealEpisode(row);
      if (!row.real_episode_indices.length && row.status === "matched") row.status = "unmatched";
    }
  });
  if (realIndex === null) {
    target.real_episode_indices = [];
  } else if (mode === "merge") {
    target.real_episode_indices = [...realEpisodeIndices(target)];
    if (!target.real_episode_indices.includes(Number(realIndex))) {
      target.real_episode_indices.push(Number(realIndex));
    }
  } else {
    target.real_episode_indices = [Number(realIndex)];
  }
  if (target.real_episode_indices.length) target.observation_tail = null;
  syncPrimaryRealEpisode(target);
  target.status = target.real_episode_indices.length ? "matched" : "unmatched";
  state.selectedRowIndex = targetRowIndex;
  renderWorkspace();
}

function resetAlignment(recordHistory = false) {
  if (recordHistory) pushHistory("按顺序重置");
  else clearHistory();
  const episode = currentEpisode();
  if (!episode) {
    state.rows = [];
  } else {
    const real = state.oss?.episodes || [];
    state.rows = episode.steps.map((step, index) => {
      const realEpisode = real[index] || null;
      return {
        id: `row_${Date.now()}_${index}`,
        traj_step_index: index,
        traj_step_indices: [index],
        source_step: step.step,
        action: step.action || {},
        real_episode_index: realEpisode ? realEpisode.episode_index : null,
        real_episode_indices: realEpisode ? [realEpisode.episode_index] : [],
        status: realEpisode ? "matched" : "unmatched",
        notes: "",
        observation_tail: null,
        include: true,
      };
    });
  }
  state.selectedRowIndex = 0;
  renderWorkspace();
}

function renderAlignment() {
  const body = el("alignmentBody");
  body.innerHTML = "";
  state.rows.forEach((row, index) => {
    const realIndices = realEpisodeIndices(row);
    const real = realByIndex(realIndices[0]);
    const realOptions = [`<option value="">${realIndices.length ? "替换..." : "选择..."}</option>`].concat((state.oss?.episodes || []).map((episode) => {
      const label = `#${episode.episode_index} · ${episode.frame_count || "?"}f`;
      return `<option value="${episode.episode_index}">${escapeHtml(label)}</option>`;
    })).join("");
    const stepIndices = sourceStepIndices(row);
    const tail = observationTail(row);
    const sourceStepHtml = stepIndices.length === 0
      ? `<div class="step-stack"><span class="tiny">手动空行名称</span><input class="manual-name" data-field="manual-name" data-index="${index}" value="${escapeAttr(row.manual_name || "manual_inserted")}"></div>`
      : `<div class="step-stack">${stepIndices.map((stepIndex) => {
          const step = sourceStep(stepIndex);
          return `<div class="step-line"><span class="tiny">step ${escapeHtml(step?.step ?? Number(stepIndex) + 1)}</span><br><span class="action-name">${escapeHtml(actionLabel(step?.action))}</span></div>`;
        }).join("")}</div>`;
    const tr = document.createElement("tr");
    tr.className = index === state.selectedRowIndex ? "selected" : "";
    tr.draggable = true;
    tr.dataset.index = index;
    const realChips = realIndices.map((realIndex) => (
      `<button class="episode-chip" data-remove-real="${realIndex}" data-index="${index}" title="移除此视频段">#${realIndex} ×</button>`
    )).join("");
    tr.innerHTML = `
      <td title="拖拽整行排序">↕ ${index + 1}</td>
      <td>${sourceStepHtml}</td>
      <td>
        <select data-field="real" data-index="${index}">${realOptions}</select>
        <div class="episode-chips">${realChips}</div>
        <div class="tiny">${realIndices.length > 1 ? `${realIndices.length} 个视频段` : real ? escapeHtml(real.task_name || real.episode_id) : ""}</div>
        <div class="tiny">${tail ? `自定义 tail：#${tail.episode_index} ${escapeHtml(tail.camera.split(".").pop())} @ ${tail.timestamps.map((value) => `${value.toFixed(3)}s`).join(" / ")}` : ""}</div>
      </td>
      <td>
        <select data-field="status" data-index="${index}">
          <option value="matched">已配对</option>
          <option value="skipped">跳过</option>
          <option value="unmatched">未配对</option>
        </select>
        <div><span class="status ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span></div>
      </td>
      <td><input data-field="notes" data-index="${index}" value="${escapeAttr(row.notes || "")}"></td>
      <td>
        <div class="ops">
          <button data-op="up" data-index="${index}" title="上移配对组" ${index === 0 ? "disabled" : ""}>↑ 上移</button>
          <button data-op="down" data-index="${index}" title="下移配对组" ${index === state.rows.length - 1 ? "disabled" : ""}>↓ 下移</button>
          <button data-op="merge-up" data-index="${index}" title="并入上一个配对组" ${index === 0 ? "disabled" : ""}>合并上方</button>
          <button data-op="merge-down" data-index="${index}" title="与下一个配对组合并" ${index === state.rows.length - 1 ? "disabled" : ""}>合并下方</button>
          <button data-op="split" data-index="${index}" title="将合并组拆回独立 step" ${stepIndices.length <= 1 ? "disabled" : ""}>拆分</button>
          <button data-op="delete" data-index="${index}" title="删除此配对组">删除</button>
        </div>
      </td>`;
    tr.addEventListener("click", (event) => {
      if (event.target && ["SELECT", "INPUT", "BUTTON"].includes(event.target.tagName)) return;
      state.selectedRowIndex = index;
      renderAlignment();
      renderRealEpisodes();
      renderSelected();
    });
    tr.addEventListener("dragstart", handleDragStart);
    tr.addEventListener("dragover", handleDragOver);
    tr.addEventListener("dragleave", handleDragLeave);
    tr.addEventListener("drop", handleDrop);
    tr.addEventListener("dragend", handleDragEnd);
    body.appendChild(tr);
    tr.querySelector('[data-field="real"]').value = realIndices.length === 1 ? realIndices[0] : "";
    tr.querySelector('[data-field="status"]').value = row.status || "unmatched";
  });
  body.querySelectorAll("select[data-field]").forEach((control) => {
    control.addEventListener("change", updateRowFromControl);
  });
  body.querySelectorAll('input[data-field="notes"], input[data-field="manual-name"]').forEach((control) => {
    control.addEventListener("input", updateRowFromControl);
    control.addEventListener("focus", rememberEditableField);
  });
  body.querySelectorAll("button[data-op]").forEach((button) => {
    button.addEventListener("click", handleRowOp);
  });
  body.querySelectorAll("button[data-remove-real]").forEach((button) => {
    button.addEventListener("click", removeRealEpisode);
  });
  const matched = state.rows.filter((row) => row.status === "matched" && realEpisodeIndices(row).length).length;
  const sourceCount = state.rows.reduce((total, row) => total + sourceStepIndices(row).length, 0);
  el("alignmentMeta").textContent = `${sourceCount} 个 traj steps · ${state.rows.length} 个配对组 · ${matched} 个已配对`;
}

function updateRowFromControl(event) {
  const target = event.target;
  const row = state.rows[Number(target.dataset.index)];
  if (!row) return;
  if (target.dataset.field === "real") {
    if (target.value !== "") {
      assignRealEpisode(Number(target.value), Number(target.dataset.index), "replace");
    }
    return;
  } else if (target.dataset.field === "status") {
    pushHistory("修改配对状态");
    row.status = target.value;
    if (row.status === "unmatched") {
      row.real_episode_indices = [];
      syncPrimaryRealEpisode(row);
      setNotice("已解除该 step 的真机 episode，episode 已恢复为未使用。", "ok");
    }
  } else if (target.dataset.field === "notes") {
    row.notes = target.value;
    state.selectedRowIndex = Number(target.dataset.index);
    return;
  } else if (target.dataset.field === "manual-name") {
    row.manual_name = target.value;
    row.action = {name: target.value || "manual_inserted", base_name: "manual_inserted", node_ids: []};
    state.selectedRowIndex = Number(target.dataset.index);
    return;
  }
  state.selectedRowIndex = Number(target.dataset.index);
  renderWorkspace();
}

function rememberEditableField(event) {
  pushHistory(event.target.dataset.field === "manual-name" ? "修改空行名称" : "修改备注");
}

function removeRealEpisode(event) {
  event.stopPropagation();
  const rowIndex = Number(event.currentTarget.dataset.index);
  const realIndex = Number(event.currentTarget.dataset.removeReal);
  const row = state.rows[rowIndex];
  if (!row) return;
  pushHistory("移除真机 episode");
  row.real_episode_indices = realEpisodeIndices(row).filter((value) => value !== realIndex);
  syncPrimaryRealEpisode(row);
  if (!row.real_episode_indices.length && row.status === "matched") row.status = "unmatched";
  state.selectedRowIndex = rowIndex;
  renderWorkspace();
}

function handleRowOp(event) {
  event.stopPropagation();
  const index = Number(event.target.dataset.index);
  const op = event.target.dataset.op;
  if (op === "up" && index > 0) {
    pushHistory("上移配对组");
    [state.rows[index - 1], state.rows[index]] = [state.rows[index], state.rows[index - 1]];
    state.selectedRowIndex = index - 1;
  } else if (op === "down" && index < state.rows.length - 1) {
    pushHistory("下移配对组");
    [state.rows[index + 1], state.rows[index]] = [state.rows[index], state.rows[index + 1]];
    state.selectedRowIndex = index + 1;
  } else if (op === "merge-up" && index > 0) {
    pushHistory("合并上方 traj step");
    mergeRows(index - 1, index);
    state.selectedRowIndex = index - 1;
  } else if (op === "merge-down" && index < state.rows.length - 1) {
    pushHistory("合并下方 traj step");
    mergeRows(index, index + 1);
    state.selectedRowIndex = index;
  } else if (op === "split") {
    pushHistory("拆分 traj step 组");
    splitRow(index);
  } else if (op === "delete") {
    pushHistory("删除配对组");
    state.rows.splice(index, 1);
    state.selectedRowIndex = Math.max(0, Math.min(index, state.rows.length - 1));
  }
  renderWorkspace();
}

function mergeRows(targetIndex, sourceIndex) {
  const target = state.rows[targetIndex];
  const source = state.rows[sourceIndex];
  if (!target || !source) return;
  target.traj_step_indices = [...sourceStepIndices(target), ...sourceStepIndices(source)];
  target.traj_step_index = target.traj_step_indices[0] ?? null;
  target.real_episode_indices = [...new Set([...realEpisodeIndices(target), ...realEpisodeIndices(source)])];
  syncPrimaryRealEpisode(target);
  target.observation_tail = source.observation_tail || target.observation_tail || null;
  if (target.real_episode_indices.length) {
    target.status = "matched";
    target.observation_tail = null;
  }
  target.notes = [target.notes, source.notes].filter(Boolean).join("; ");
  state.rows.splice(sourceIndex, 1);
}

function splitRow(index) {
  const row = state.rows[index];
  const indices = sourceStepIndices(row);
  if (!row || indices.length <= 1) return;
  const replacement = indices.map((stepIndex, offset) => {
    const step = sourceStep(stepIndex);
    return {
      id: `split_${Date.now()}_${stepIndex}`,
      traj_step_index: stepIndex,
      traj_step_indices: [stepIndex],
      source_step: step?.step ?? Number(stepIndex) + 1,
      action: step?.action || {},
      real_episode_index: offset === 0 ? row.real_episode_index : null,
      real_episode_indices: offset === 0 ? realEpisodeIndices(row) : [],
      status: offset === 0 ? row.status : "unmatched",
      notes: offset === 0 ? row.notes : "",
      observation_tail: offset === indices.length - 1 ? row.observation_tail || null : null,
      include: true,
    };
  });
  state.rows.splice(index, 1, ...replacement);
  state.selectedRowIndex = index;
}

function newManualRow() {
  return {
    id: `manual_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    traj_step_index: null,
    traj_step_indices: [],
    source_step: null,
    action: {name: "manual_inserted", base_name: "manual_inserted", node_ids: []},
    real_episode_index: null,
    real_episode_indices: [],
    manual_name: "manual_inserted",
    status: "unmatched",
    notes: "",
    observation_tail: null,
    include: true,
  };
}

function handleDragStart(event) {
  if (["INPUT", "SELECT", "BUTTON"].includes(event.target.tagName)) {
    event.preventDefault();
    return;
  }
  state.dragIndex = Number(event.currentTarget.dataset.index);
  event.currentTarget.classList.add("dragging");
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", String(state.dragIndex));
}

function handleDragOver(event) {
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  event.currentTarget.classList.add("drop-target");
}

function handleDragLeave(event) {
  event.currentTarget.classList.remove("drop-target");
}

function handleDrop(event) {
  event.preventDefault();
  const fromIndex = state.dragIndex;
  const toIndex = Number(event.currentTarget.dataset.index);
  if (fromIndex === null || fromIndex === toIndex || !state.rows[fromIndex]) return;
  pushHistory("拖拽移动配对组");
  const [moved] = state.rows.splice(fromIndex, 1);
  state.rows.splice(toIndex, 0, moved);
  state.selectedRowIndex = toIndex;
  state.dragIndex = null;
  renderWorkspace();
}

function handleDragEnd() {
  state.dragIndex = null;
  document.querySelectorAll("tr.dragging, tr.drop-target").forEach((row) => {
    row.classList.remove("dragging", "drop-target");
  });
}

function renderSelected() {
  const row = currentRow();
  const episode = currentEpisode();
  const realIndices = row ? realEpisodeIndices(row) : [];
  const tail = observationTail(row);
  const reals = realIndices.map(realByIndex).filter(Boolean);
  const groupedSteps = row ? sourceStepIndices(row).map((index) => sourceStep(index)).filter(Boolean) : [];
  el("previewMeta").textContent = row
    ? `配对组 ${state.selectedRowIndex + 1} · ${groupedSteps.length} 个 traj steps · ${reals.length} 个视频段`
    : "";
  const previewSelect = el("previewEpisodeSelect");
  previewSelect.innerHTML = reals.length
    ? reals.map((real) => `<option value="${real.episode_index}">视频段 #${real.episode_index}</option>`).join("")
    : `<option value="">无视频段</option>`;
  previewSelect.disabled = reals.length === 0;
  const tailPicker = el("tailPicker");
  const canSelectTail = Boolean(row && sourceStepIndices(row).length === 0 && realIndices.length === 0);
  tailPicker.hidden = !canSelectTail;
  if (canSelectTail) {
    const tailEpisodeSelect = el("tailEpisodeSelect");
    const availableEpisodes = state.oss?.episodes || [];
    tailEpisodeSelect.innerHTML = availableEpisodes.length
      ? availableEpisodes.map((item) => (
          `<option value="${item.episode_index}">#${item.episode_index} · ${escapeHtml(item.task_name || item.episode_id)}</option>`
        )).join("")
      : `<option value="">请先加载 OSS</option>`;
    if (tail && realByIndex(tail.episode_index)) {
      tailEpisodeSelect.value = String(tail.episode_index);
    }
    el("tailCameraSelect").value = tail?.camera || "observation.images.head_rgb";
    el("tailFrame1").value = tail ? String(tail.timestamps[0]) : "";
    el("tailFrame2").value = tail ? String(tail.timestamps[1]) : "";
    el("tailSelectionMeta").textContent = tail
      ? `已保存：episode #${tail.episode_index} · ${tail.camera.split(".").pop()}，${tail.timestamps.map((value) => `${value.toFixed(3)}s`).join(" / ")}`
      : "尚未保存";
  }
  el("selectedJson").textContent = JSON.stringify({row, real_episodes: reals}, null, 2);
  el("viewGraphJson").textContent = JSON.stringify({
    task: episode?.task,
    merged_steps: groupedSteps.map((step) => ({
      step: step?.step,
      action: step?.action,
      event: step?.event,
      teacher_reason: step?.teacher_reason,
    })),
  }, null, 2);
}

async function loadTailEpisode() {
  const row = currentRow();
  const episodeIndex = el("tailEpisodeSelect").value;
  if (!row || sourceStepIndices(row).length !== 0 || realEpisodeIndices(row).length !== 0) {
    setNotice("只有没有配对 episode 的新增 step 可以设置自定义 observation tail。", "error");
    return;
  }
  if (episodeIndex === "") {
    setNotice("请先选择来源 episode。", "error");
    return;
  }
  await loadPreviewVideos(Number(episodeIndex));
  setNotice(`已加载 episode #${episodeIndex}，请播放选定相机视频并记录两个时间点。`, "ok");
}

function captureTailFrame(slot) {
  const camera = el("tailCameraSelect").value;
  const video = el({
    "observation.images.head_rgb": "videoHead",
    "observation.images.left_wrist_rgb": "videoLeft",
    "observation.images.right_wrist_rgb": "videoRight",
  }[camera] || "videoHead");
  const selectedEpisode = Number(el("tailEpisodeSelect").value);
  if (!video.src || state.loadedPreviewEpisodeIndex !== selectedEpisode) {
    setNotice("请先加载 observation tail 来源视频。", "error");
    return;
  }
  el(slot === 0 ? "tailFrame1" : "tailFrame2").value = video.currentTime.toFixed(3);
}

function seekTailFrame(slot) {
  const value = Number(el(slot === 0 ? "tailFrame1" : "tailFrame2").value);
  if (!Number.isFinite(value) || value < 0) {
    setNotice("帧时间必须是非负数字。", "error");
    return;
  }
  ["videoHead", "videoLeft", "videoRight"].forEach((id) => {
    const video = el(id);
    if (video.src) video.currentTime = value;
  });
}

function saveTailFrames() {
  const row = currentRow();
  const episodeIndex = Number(el("tailEpisodeSelect").value);
  const camera = el("tailCameraSelect").value;
  const rawTimestamps = [el("tailFrame1").value, el("tailFrame2").value];
  const timestamps = rawTimestamps.map(Number).sort((a, b) => a - b);
  if (!row || sourceStepIndices(row).length !== 0 || realEpisodeIndices(row).length !== 0) {
    setNotice("只有没有配对 episode 的新增 step 可以设置自定义 observation tail。", "error");
    return;
  }
  if (!Number.isInteger(episodeIndex) || !realByIndex(episodeIndex)) {
    setNotice("请选择有效的来源 episode。", "error");
    return;
  }
  if (
    rawTimestamps.some((value) => value.trim() === "")
    || timestamps.some((value) => !Number.isFinite(value) || value < 0)
    || timestamps[0] === timestamps[1]
  ) {
    setNotice("请选择两个不同的、非负的帧时间。", "error");
    return;
  }
  const selectedVideo = el({
    "observation.images.head_rgb": "videoHead",
    "observation.images.left_wrist_rgb": "videoLeft",
    "observation.images.right_wrist_rgb": "videoRight",
  }[camera] || "videoHead");
  if (
    state.loadedPreviewEpisodeIndex === episodeIndex
    && Number.isFinite(selectedVideo.duration)
    && timestamps[1] > selectedVideo.duration
  ) {
    setNotice(`帧时间超出视频长度 ${selectedVideo.duration.toFixed(3)}s。`, "error");
    return;
  }
  pushHistory("设置自定义 observation tail");
  row.observation_tail = {
    episode_index: episodeIndex,
    timestamps,
    camera,
    source: "manual_frame_selection",
  };
  renderWorkspace();
  setNotice(`已保存 episode #${episodeIndex} 的两帧，供下一步 previous_tail 使用。`, "ok");
}

function clearTailFrames() {
  const row = currentRow();
  if (!row || !observationTail(row)) return;
  pushHistory("清除自定义 observation tail");
  row.observation_tail = null;
  renderWorkspace();
  setNotice("已清除自定义 observation tail。", "ok");
}

async function loadPreviewVideos(episodeIndex) {
  setNotice("Caching preview assets...");
  const payload = await api("/api/cache-episode", {
    method: "POST",
    body: JSON.stringify({
      oss_root: el("ossRoot").value.trim(),
      episode_index: Number(episodeIndex),
    }),
  });
  setVideo("videoHead", payload.video_urls["observation.images.head_rgb"]);
  setVideo("videoLeft", payload.video_urls["observation.images.left_wrist_rgb"]);
  setVideo("videoRight", payload.video_urls["observation.images.right_wrist_rgb"]);
  state.loadedPreviewEpisodeIndex = Number(episodeIndex);
  return payload;
}

async function previewSelected() {
  const row = currentRow();
  const previewRealIndex = el("previewEpisodeSelect").value;
  if (!row || previewRealIndex === "") {
    setNotice("No real episode selected.", "error");
    return;
  }
  const payload = await loadPreviewVideos(Number(previewRealIndex));
  el("selectedJson").textContent = JSON.stringify({
    row,
    descriptor: payload.descriptor,
    parquet_summary: payload.parquet_summary,
  }, null, 2);
  setNotice("Preview loaded.", "ok");
}

function setVideo(id, src) {
  const node = el(id);
  node.src = src || "";
  if (src) node.load();
}

async function saveAlignment() {
  if (!state.trajectory || !state.oss) {
    setNotice("Load trajectory and OSS first.", "error");
    return;
  }
  if (!state.canSave || !state.selectedFile) {
    setNotice("原始 traj 文件不可用，当前保存结果只能展示，不能再次保存。", "error");
    return;
  }
  setNotice("Saving...");
  const payload = await api("/api/save-alignment", {
    method: "POST",
    body: JSON.stringify({
      trajectory_file: state.selectedFile,
      episode_index: state.selectedEpisodeIndex,
      oss_root: el("ossRoot").value.trim(),
      rows: state.rows,
      real_episodes: state.oss.episodes,
    }),
  });
  state.loadedSavedPath = payload.path;
  el("savedAlignmentPath").value = payload.path;
  await loadSavedFiles();
  setNotice(`Saved ${payload.step_count} steps to ${payload.path}`, "ok");
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

el("loadTrajectory").addEventListener("click", () => loadTrajectory().catch((err) => setNotice(err.message, "error")));
el("loadSavedAlignment").addEventListener("click", () => loadSavedAlignment().catch((err) => setNotice(err.message, "error")));
el("loadOss").addEventListener("click", () => loadOss().catch((err) => setNotice(err.message, "error")));
el("save").addEventListener("click", () => saveAlignment().catch((err) => setNotice(err.message, "error")));
el("preview").addEventListener("click", () => previewSelected().catch((err) => setNotice(err.message, "error")));
el("loadTailEpisode").addEventListener("click", () => loadTailEpisode().catch((err) => setNotice(err.message, "error")));
el("saveTailFrames").addEventListener("click", saveTailFrames);
el("clearTailFrames").addEventListener("click", clearTailFrames);
document.querySelectorAll("button[data-tail-capture]").forEach((button) => {
  button.addEventListener("click", () => captureTailFrame(Number(button.dataset.tailCapture)));
});
document.querySelectorAll("button[data-tail-seek]").forEach((button) => {
  button.addEventListener("click", () => seekTailFrame(Number(button.dataset.tailSeek)));
});
el("undo").addEventListener("click", undoLastOperation);
el("autoAlign").addEventListener("click", () => resetAlignment(true));
el("addRow").addEventListener("click", () => {
  pushHistory("增加空行");
  state.rows.push(newManualRow());
  state.selectedRowIndex = state.rows.length - 1;
  renderWorkspace();
});
el("trajectorySelect").addEventListener("change", () => {
  state.selectedFile = el("trajectorySelect").value;
  loadTrajectory().catch((err) => setNotice(err.message, "error"));
});
el("savedAlignmentPath").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    loadSavedAlignment().catch((err) => setNotice(err.message, "error"));
  }
});
el("episodeSelect").addEventListener("change", () => {
  state.loadedSavedPath = null;
  state.displaySourceName = null;
  state.canSave = true;
  el("save").disabled = false;
  state.selectedEpisodeIndex = Number(el("episodeSelect").value);
  resetAlignment(false);
});
el("realEpisodeSearch").addEventListener("input", renderRealEpisodes);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoLastOperation();
  }
});

Promise.all([loadFiles(), loadSavedFiles()])
  .then(([, initialAlignmentPath]) => initialAlignmentPath ? loadSavedAlignment(initialAlignmentPath) : loadTrajectory())
  .catch((err) => setNotice(err.message, "error"));
</script>
</body>
</html>"""
