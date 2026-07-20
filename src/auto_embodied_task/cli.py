from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constraints import available_modifiers
from .generator import ALL_TASK_TYPES, GenerationConfig, TaskGenerator
from .graph_io import load_view_graphs_jsonl, write_tasks_jsonl


def _csv(value: str) -> tuple[str, ...]:
    if value.strip() == "":
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _materials_from_file(path: str) -> tuple[str, ...]:
    materials = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        materials.append(item)
    return tuple(materials)


def _materials_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.materials_file:
        return _materials_from_file(args.materials_file)
    return _csv(args.materials or "")


def _json_from_file(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-embodied-task")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate task JSONL from view graph JSONL.")
    generate.add_argument("--view-graph", required=True, help="Input view graph JSONL path.")
    generate.add_argument("--output", required=True, help="Output task JSONL path.")
    generate.add_argument("--layout", choices=("all", "indoor", "tabletop"), default="all")
    generate.add_argument("--arms", choices=("single", "double"), default=None)
    generate.add_argument(
        "--task-types",
        default=",".join(ALL_TASK_TYPES),
        help=f"Comma separated task types. Available: all,{','.join(ALL_TASK_TYPES)}",
    )
    generate.add_argument(
        "--settings",
        default="",
        help=f"Comma separated extra settings. Available: all,{','.join(available_modifiers())}",
    )
    generate.add_argument("--no-base", action="store_true", help="Only emit extra-setting variants.")
    generate.add_argument("--max-tasks", type=int, default=100)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--actor", default="<char0>")
    generate.add_argument("--max-pairs-per-scene", type=int, default=200)

    create_task_graph = subparsers.add_parser(
        "create-task-view-graph",
        help="Create a view graph JSON package from a material list and activity description.",
    )
    materials = create_task_graph.add_mutually_exclusive_group(required=True)
    materials.add_argument("--materials", help="Comma separated available material names.")
    materials.add_argument("--materials-file", help="Text file with one material per line.")
    create_task_graph.add_argument(
        "--material-properties",
        default=None,
        help="JSON file defining per-material category, properties, and states.",
    )
    create_task_graph.add_argument("--scene", required=True, help="Scene description, e.g. office desktop.")
    create_task_graph.add_argument("--layout", choices=("indoor", "tabletop"), required=True)
    create_task_graph.add_argument("--arms", choices=("single", "double"), required=True)
    create_task_graph.add_argument("--output", required=True, help="Output view graph JSONL path.")
    create_task_graph.add_argument("--package-output", default=None, help="Optional full JSON package output path.")
    create_task_graph.add_argument(
        "--task-hint",
        "--task",
        "--activity",
        dest="task_hint",
        default=None,
        help="Optional activity/goal used to construct the view graph, e.g. 整理桌面.",
    )
    create_task_graph.add_argument("--scene-id", default=None)
    create_task_graph.add_argument("--env-id", default=None)
    create_task_graph.add_argument("--append", action="store_true", help="Append one JSONL scene instead of overwriting.")
    create_task_graph.add_argument("--provider", choices=("openai", "qwen", "compatible"), default="qwen")
    create_task_graph.add_argument(
        "--model",
        default=None,
        help="API model name, e.g. qwen3.6-plus/qwen3.7-plus/qwen-plus/gpt-4o-mini. Defaults depend on provider.",
    )
    create_task_graph.add_argument(
        "--api-key-env",
        default=None,
        help="API key environment variable. Defaults to DASHSCOPE_API_KEY for qwen, OPENAI_API_KEY otherwise.",
    )
    create_task_graph.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI-compatible chat completions URL. Defaults to provider-specific URL.",
    )
    create_task_graph.add_argument("--timeout-seconds", type=int, default=60)
    create_task_graph.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=False,
        help="Enable Qwen thinking mode for view graph synthesis. Disabled by default for stricter JSON output.",
    )
    create_task_graph.add_argument(
        "--no-enable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen thinking mode. This is the default.",
    )

    serve_graph = subparsers.add_parser(
        "serve-view-graph",
        help="Run a local web UI for creating and editing view graphs.",
    )
    serve_graph.add_argument("--host", default="127.0.0.1")
    serve_graph.add_argument("--port", type=int, default=8765)
    serve_graph.add_argument("--open-browser", action="store_true")

    serve_trajectory = subparsers.add_parser(
        "serve-trajectory",
        help="Run a local web UI for replaying trajectory JSONL files.",
    )
    serve_trajectory.add_argument("--trajectory", default=None, help="Optional initial trajectory JSONL path.")
    serve_trajectory.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory of trajectory JSONL files. Defaults to the project outputs directory.",
    )
    serve_trajectory.add_argument("--host", default="127.0.0.1")
    serve_trajectory.add_argument("--port", type=int, default=8766)
    serve_trajectory.add_argument(
        "--base-path",
        default="",
        help="URL path prefix for reverse-proxy deployment, for example /traj.",
    )
    serve_trajectory.add_argument("--open-browser", action="store_true")

    serve_real_alignment = subparsers.add_parser(
        "serve-real-alignment",
        help="Run a local web UI for aligning symbolic trajectories with real robot OSS episodes.",
    )
    serve_real_alignment.add_argument(
        "--trajectory-dir",
        default=None,
        help="Directory of trajectory JSONL files. Defaults to the project outputs directory.",
    )
    serve_real_alignment.add_argument(
        "--saved-dir",
        default=None,
        help="Directory for aligned JSONL outputs. Defaults to auto_embodied_task/saved.",
    )
    serve_real_alignment.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for cached OSS videos/parquets. Defaults to ~/.cache/auto_embodied_task/real_alignment.",
    )
    serve_real_alignment.add_argument(
        "--alignment",
        default=None,
        help="Optional saved aligned JSONL path to load when the UI opens.",
    )
    serve_real_alignment.add_argument("--host", default="127.0.0.1")
    serve_real_alignment.add_argument("--port", type=int, default=8767)
    serve_real_alignment.add_argument("--oss-region", default="cn-shanghai")
    serve_real_alignment.add_argument("--oss-endpoint", default=None)
    serve_real_alignment.add_argument("--ossutil-bin", default="ossutil")
    serve_real_alignment.add_argument("--open-browser", action="store_true")

    serve_evaluation_replies = subparsers.add_parser(
        "serve-evaluation-replies",
        help="Run a local web UI for reviewing real-observation evaluation JSONL results.",
    )
    serve_evaluation_replies.add_argument(
        "--evaluation-dir",
        default=None,
        help="Directory containing evaluation JSONL and sibling summary files.",
    )
    serve_evaluation_replies.add_argument("--host", default="127.0.0.1")
    serve_evaluation_replies.add_argument("--port", type=int, default=8771)
    serve_evaluation_replies.add_argument("--base-path", default="")
    serve_evaluation_replies.add_argument("--open-browser", action="store_true")

    edit_graph = subparsers.add_parser(
        "edit-view-graph",
        help="Apply an abstract constraint profile to existing view graph JSONL.",
    )
    edit_graph.add_argument("--input", required=True, help="Input view graph JSONL path.")
    edit_graph.add_argument("--profile", required=True, help="Abstract constraint profile JSON path.")
    edit_graph.add_argument("--output", required=True, help="Output profiled view graph JSONL path.")
    edit_graph.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of randomized profiled variants to emit per input graph.",
    )
    edit_graph.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible batches.")
    edit_graph.add_argument(
        "--placement-edge-constraints",
        default=None,
        help="Optional JSON file that forbids or whitelists profile-generated spatial edges.",
    )

    collect = subparsers.add_parser(
        "collect-trajectories",
        help="Run the symbolic harness over generated tasks and write trajectory JSONL.",
    )
    collect.add_argument("--view-graph", required=True, help="Input view graph JSONL path.")
    collect.add_argument("--tasks", required=True, help="Generated task JSONL path.")
    collect.add_argument("--output", required=True, help="Output trajectory JSONL path.")
    collect.add_argument(
        "--backend",
        choices=("symbolic", "robotwin"),
        default="symbolic",
        help="Execution backend. robotwin executes and measures actions in RoboTwin/SAPIEN.",
    )
    collect.add_argument("--robotwin-root", default="/home/wmq/project/bench/RoboTwin")
    collect.add_argument("--robotwin-task-config", default="task_config/demo_clean.yml")
    collect.add_argument("--robotwin-asset-map", default=None)
    collect.add_argument("--robotwin-output-dir", default="outputs/robotwin")
    collect.add_argument("--robotwin-seed", type=int, default=7)
    collect.add_argument("--robotwin-render", action="store_true")
    collect.add_argument(
        "--robotwin-execution-mode",
        choices=("strict", "assisted"),
        default="strict",
        help=(
            "strict rejects actions without measured physical success; assisted allows "
            "pose/drive/kinematic fallbacks for trajectory reproduction."
        ),
    )
    collect.add_argument(
        "--mode",
        choices=("replay", "teacher"),
        default="replay",
        help="Trajectory collection mode. replay uses ground_truth_plan; teacher calls a model each step.",
    )
    collect.add_argument("--max-episodes", type=int, default=None)
    collect.add_argument("--max-steps", type=int, default=None)
    collect.add_argument("--teacher-provider", choices=("openai", "qwen", "compatible"), default="qwen")
    collect.add_argument("--teacher-model", default=None)
    collect.add_argument(
        "--teacher-api-key-env",
        default=None,
        help="Teacher API key environment variable. Defaults to DASHSCOPE_API_KEY for qwen, OPENAI_API_KEY otherwise.",
    )
    collect.add_argument("--teacher-api-base-url", default=None)
    collect.add_argument(
        "--teacher-api-style",
        choices=("chat_completions", "responses"),
        default="chat_completions",
    )
    collect.add_argument("--teacher-timeout-seconds", type=int, default=60)
    collect.add_argument("--teacher-temperature", type=float, default=0.0)
    collect.add_argument("--teacher-max-api-attempts", type=int, default=1)
    collect.add_argument("--teacher-retry-backoff-seconds", type=float, default=5.0)
    collect.add_argument("--teacher-retry-max-seconds", type=float, default=60.0)
    collect.add_argument(
        "--failure-injection",
        choices=("none", "once", "probability", "all"),
        default="none",
        help="Optionally inject failed_<action> events during collection.",
    )
    collect.add_argument(
        "--failure-actions",
        default="all",
        help="Comma separated action names eligible for injection, or all.",
    )
    collect.add_argument("--failure-probability", type=float, default=0.0)
    collect.add_argument("--max-failures-per-episode", type=int, default=1)
    collect.add_argument("--failure-seed", type=int, default=None)
    collect.add_argument(
        "--placement-edge-constraints",
        default=None,
        help="Optional JSON file that forbids or whitelists placement edges for putin/puton valid actions.",
    )

    eval_real = subparsers.add_parser(
        "evaluate-real-trajectories",
        help="Evaluate saved real-aligned trajectories with image/video observations.",
    )
    eval_real.add_argument("--input", required=True, help="Input aligned JSONL from saved/.")
    eval_real.add_argument(
        "--output",
        required=True,
        help="Output evaluation JSONL base path. A timestamp is appended automatically.",
    )
    eval_real.add_argument(
        "--provider",
        choices=(
            "openai",
            "qwen",
            "compatible",
            "mr_openai",
            "mr_anthropic",
            "mr_google",
        ),
        default="qwen",
    )
    eval_real.add_argument("--model", default=None)
    eval_real.add_argument("--model-name", default=None, help="Display name stored in records and summary.")
    eval_real.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "API key environment variable. Defaults to DASHSCOPE_API_KEY for qwen, "
            "MR_API_KEY for mr_* providers, and OPENAI_API_KEY otherwise."
        ),
    )
    eval_real.add_argument("--api-base-url", default=None)
    eval_real.add_argument(
        "--api-style",
        choices=(
            "auto",
            "chat_completions",
            "responses",
            "anthropic_messages",
            "gemini_generate_content",
        ),
        default="auto",
        help="Wire protocol. auto selects the protocol required by each provider.",
    )
    eval_real.add_argument("--timeout-seconds", type=int, default=120)
    eval_real.add_argument("--temperature", type=float, default=0.0)
    eval_real.add_argument("--max-output-tokens", type=int, default=2048)
    eval_real.add_argument("--max-api-attempts", type=int, default=1)
    eval_real.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    eval_real.add_argument("--retry-max-seconds", type=float, default=60.0)
    eval_real.add_argument(
        "--modes",
        default="obs_only,visible_graph_only,graph_only,obs_plus_graph,wrong_graph_plus_obs",
        help="Comma separated eval modes.",
    )
    eval_real.add_argument(
        "--history-source",
        choices=("teacher", "inference"),
        default="teacher",
        help="Use replayed teacher history or each mode's accumulated model inference history.",
    )
    eval_real.add_argument(
        "--valid-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the graph-derived valid_actions candidate list in model requests.",
    )
    eval_real.add_argument(
        "--soft-optimal-beta",
        type=float,
        default=1.0,
        help="Inverse temperature for relaxed-cost soft optimal action scoring.",
    )
    eval_real.add_argument("--frame-count", type=int, default=2)
    eval_real.add_argument("--observation-window-seconds", type=float, default=0.5)
    eval_real.add_argument(
        "--frame-sampling",
        choices=("head", "previous_tail"),
        default="head",
    )
    eval_real.add_argument(
        "--cameras",
        default="observation.images.head_rgb,observation.images.left_wrist_rgb,observation.images.right_wrist_rgb",
        help="Comma separated camera video keys to sample.",
    )
    eval_real.add_argument("--max-steps", type=int, default=None)
    eval_real.add_argument("--dry-run", action="store_true", help="Write request summaries without calling a model.")
    eval_real.add_argument("--fail-fast", action="store_true", help="Stop on the first model request error.")
    eval_real.add_argument("--oss-region", default="cn-shanghai")
    eval_real.add_argument("--oss-endpoint", default=None)
    eval_real.add_argument("--cache-dir", default=None)

    eval_graph_rollout = subparsers.add_parser(
        "evaluate-view-graph-rollouts",
        help="Run closed-loop model evaluation from saved initial view graphs.",
    )
    eval_graph_rollout.add_argument("--input", required=True, help="Input aligned JSONL from saved/.")
    eval_graph_rollout.add_argument(
        "--output",
        required=True,
        help="Output evaluation JSONL base path. A timestamp is appended automatically.",
    )
    eval_graph_rollout.add_argument(
        "--provider",
        choices=("openai", "qwen", "compatible", "mr_openai", "mr_anthropic", "mr_google"),
        default="qwen",
    )
    eval_graph_rollout.add_argument("--model", default=None)
    eval_graph_rollout.add_argument("--model-name", default=None)
    eval_graph_rollout.add_argument("--api-key-env", default=None)
    eval_graph_rollout.add_argument("--api-base-url", default=None)
    eval_graph_rollout.add_argument(
        "--api-style",
        choices=("auto", "chat_completions", "responses", "anthropic_messages", "gemini_generate_content"),
        default="auto",
    )
    eval_graph_rollout.add_argument("--timeout-seconds", type=int, default=120)
    eval_graph_rollout.add_argument("--temperature", type=float, default=0.0)
    eval_graph_rollout.add_argument("--max-output-tokens", type=int, default=2048)
    eval_graph_rollout.add_argument("--max-api-attempts", type=int, default=1)
    eval_graph_rollout.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    eval_graph_rollout.add_argument("--retry-max-seconds", type=float, default=60.0)
    eval_graph_rollout.add_argument(
        "--valid-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include graph-derived valid_actions in each closed-loop model request.",
    )
    eval_graph_rollout.add_argument("--soft-optimal-beta", type=float, default=1.0)
    eval_graph_rollout.add_argument("--max-steps", type=int, default=100)
    eval_graph_rollout.add_argument("--history-window", type=int, default=8)
    eval_graph_rollout.add_argument("--max-consecutive-model-errors", type=int, default=3)
    eval_graph_rollout.add_argument(
        "--failure-injection",
        choices=("none", "once", "probability", "all"),
        default="none",
        help=(
            "Inject failed_<action> events only for selected actions that would normally "
            "execute successfully."
        ),
    )
    eval_graph_rollout.add_argument(
        "--failure-actions",
        default="all",
        help="Comma separated action names eligible for injection, or all.",
    )
    eval_graph_rollout.add_argument("--failure-probability", type=float, default=0.0)
    eval_graph_rollout.add_argument("--max-failures-per-episode", type=int, default=1)
    eval_graph_rollout.add_argument("--failure-seed", type=int, default=None)
    eval_graph_rollout.add_argument(
        "--graph-disturbance-file",
        default=None,
        help=(
            "Optional JSON/JSONL schedule of external graph changes applied before the "
            "specified step observation."
        ),
    )
    eval_graph_rollout.add_argument("--fail-fast", action="store_true")

    eval_graph_manifest = subparsers.add_parser(
        "evaluate-view-graph-intervention-manifest",
        help="Run manifest conditions as independent closed-loop view-graph rollouts.",
    )
    eval_graph_manifest.add_argument("--manifest", required=True)
    eval_graph_manifest.add_argument("--output-dir", required=True)
    eval_graph_manifest.add_argument(
        "--conditions",
        default="all",
        help="Comma separated condition ids, or all.",
    )
    eval_graph_manifest.add_argument(
        "--provider",
        choices=("openai", "qwen", "compatible", "mr_openai", "mr_anthropic", "mr_google"),
        default="qwen",
    )
    eval_graph_manifest.add_argument("--model", default=None)
    eval_graph_manifest.add_argument("--model-name", default=None)
    eval_graph_manifest.add_argument("--api-key-env", default=None)
    eval_graph_manifest.add_argument("--api-base-url", default=None)
    eval_graph_manifest.add_argument(
        "--api-style",
        choices=("auto", "chat_completions", "responses", "anthropic_messages", "gemini_generate_content"),
        default="auto",
    )
    eval_graph_manifest.add_argument("--timeout-seconds", type=int, default=120)
    eval_graph_manifest.add_argument("--temperature", type=float, default=0.0)
    eval_graph_manifest.add_argument("--max-output-tokens", type=int, default=2048)
    eval_graph_manifest.add_argument("--max-api-attempts", type=int, default=1)
    eval_graph_manifest.add_argument("--retry-backoff-seconds", type=float, default=5.0)
    eval_graph_manifest.add_argument("--retry-max-seconds", type=float, default=60.0)
    eval_graph_manifest.add_argument(
        "--valid-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_graph_manifest.add_argument("--soft-optimal-beta", type=float, default=1.0)
    eval_graph_manifest.add_argument("--max-steps", type=int, default=100)
    eval_graph_manifest.add_argument("--history-window", type=int, default=8)
    eval_graph_manifest.add_argument("--max-consecutive-model-errors", type=int, default=3)
    eval_graph_manifest.add_argument("--fail-fast", action="store_true")

    validate_robotwin = subparsers.add_parser(
        "validate-robotwin-task",
        help="Compile a view graph/task/trajectory and report RoboTwin physical coverage.",
    )
    validate_robotwin.add_argument("--view-graph", required=True)
    validate_robotwin.add_argument("--tasks", required=True)
    validate_robotwin.add_argument("--trajectory", default=None)
    validate_robotwin.add_argument("--asset-map", default=None)
    validate_robotwin.add_argument("--robotwin-root", default=None)
    validate_robotwin.add_argument("--seed", type=int, default=7)
    validate_robotwin.add_argument("--output", required=True)

    replay_robotwin = subparsers.add_parser(
        "replay-robotwin-trajectory",
        help="Replay real-aligned steps from saved/ in the RoboTwin 2.0 physical backend.",
    )
    replay_robotwin.add_argument("--view-graph", required=True)
    replay_robotwin.add_argument("--tasks", required=True)
    replay_robotwin.add_argument("--trajectory", required=True)
    replay_robotwin.add_argument("--asset-map", default=None)
    replay_robotwin.add_argument("--robotwin-root", default="/home/wmq/project/bench/RoboTwin")
    replay_robotwin.add_argument("--task-config", default="task_config/demo_clean.yml")
    replay_robotwin.add_argument("--seed", type=int, default=7)
    replay_robotwin.add_argument("--render", action="store_true")
    replay_robotwin.add_argument(
        "--execution-mode",
        choices=("strict", "assisted"),
        default="strict",
        help=(
            "strict produces physical acceptance evidence; assisted prioritizes "
            "successful reproduction of an existing aligned trajectory."
        ),
    )
    replay_robotwin.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        graphs = load_view_graphs_jsonl(args.view_graph)
        config = GenerationConfig(
            task_types=_csv(args.task_types),
            settings=_csv(args.settings),
            include_base=not args.no_base,
            arms=args.arms,
            layout=args.layout,
            max_tasks=args.max_tasks,
            seed=args.seed,
            actor=args.actor,
            max_pairs_per_scene=args.max_pairs_per_scene,
        )
        tasks = TaskGenerator(config).generate(graphs)
        count = write_tasks_jsonl(tasks, args.output)
        print(f"Wrote {count} tasks to {args.output}")
        return 0
    if args.command == "create-task-view-graph":
        from .layout_synthesis import (
            TaskViewGraphSynthesisConfig,
            synthesize_task_view_graph,
            write_task_view_graph_package,
            write_view_graph_jsonl,
        )

        config = TaskViewGraphSynthesisConfig(
            materials=_materials_from_args(args),
            scene=args.scene,
            layout=args.layout,
            arms=args.arms,
            material_properties=_json_from_file(args.material_properties),
            task_hint=args.task_hint,
            scene_id=args.scene_id,
            env_id=args.env_id,
            provider=args.provider,
            model=args.model,
            api_key_env=args.api_key_env,
            api_base_url=args.api_base_url,
            timeout_seconds=args.timeout_seconds,
            enable_thinking=args.enable_thinking,
        )
        package = synthesize_task_view_graph(config)
        graph = package["view_graph"]
        write_view_graph_jsonl(graph, args.output, append=args.append)
        if args.package_output:
            write_task_view_graph_package(package, args.package_output)
            print(f"Wrote view graph package to {args.package_output}")
        print(f"Wrote view graph {graph['scene_id']} to {args.output}")
        return 0
    if args.command == "serve-view-graph":
        from .view_graph_server import serve_view_graph_app

        serve_view_graph_app(
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "serve-trajectory":
        from .trajectory_server import serve_trajectory_app

        serve_trajectory_app(
            trajectory_path=args.trajectory,
            trajectory_dir=args.trajectory_dir,
            host=args.host,
            port=args.port,
            base_path=args.base_path,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "serve-real-alignment":
        from .real_alignment_server import serve_real_alignment_app

        serve_real_alignment_app(
            trajectory_dir=args.trajectory_dir,
            saved_dir=args.saved_dir,
            cache_dir=args.cache_dir,
            alignment_path=args.alignment,
            host=args.host,
            port=args.port,
            oss_region=args.oss_region,
            oss_endpoint=args.oss_endpoint,
            ossutil_bin=args.ossutil_bin,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "serve-evaluation-replies":
        from .evaluation_reply_server import serve_evaluation_reply_app

        serve_evaluation_reply_app(
            evaluation_dir=args.evaluation_dir,
            host=args.host,
            port=args.port,
            base_path=args.base_path,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "edit-view-graph":
        from .profile_editor import edit_view_graphs_with_profile

        results = edit_view_graphs_with_profile(
            input_path=args.input,
            profile_path=args.profile,
            output_path=args.output,
            num_samples=args.num_samples,
            seed=args.seed,
            placement_edge_constraints_path=args.placement_edge_constraints,
        )
        print(f"Wrote {len(results)} profiled view graphs to {args.output}")
        return 0
    if args.command == "collect-trajectories":
        from .harness import FailureInjectionConfig, TeacherPolicyConfig, collect_symbolic_trajectories

        teacher_config = None
        if args.mode == "teacher":
            teacher_config = TeacherPolicyConfig(
                provider=args.teacher_provider,
                model=args.teacher_model,
                api_key_env=args.teacher_api_key_env,
                api_base_url=args.teacher_api_base_url,
                timeout_seconds=args.teacher_timeout_seconds,
                temperature=args.teacher_temperature,
                max_attempts=args.teacher_max_api_attempts,
                retry_backoff_seconds=args.teacher_retry_backoff_seconds,
                retry_max_seconds=args.teacher_retry_max_seconds,
                api_style=args.teacher_api_style,
            )
        failure_injection = FailureInjectionConfig(
            mode=args.failure_injection,
            actions=_csv(args.failure_actions),
            probability=args.failure_probability,
            max_failures_per_episode=args.max_failures_per_episode,
            seed=args.failure_seed,
        )
        backend_factory = None
        if args.backend == "robotwin":
            from .robotwin_adapter import RoboTwinBackend, RoboTwinBackendConfig

            backend_config = RoboTwinBackendConfig(
                robotwin_root=Path(args.robotwin_root),
                task_config=Path(args.robotwin_task_config),
                asset_map_path=Path(args.robotwin_asset_map) if args.robotwin_asset_map else None,
                output_dir=Path(args.robotwin_output_dir),
                seed=args.robotwin_seed,
                render=args.robotwin_render,
                execution_mode=args.robotwin_execution_mode,
            )

            def backend_factory(graph, task, placement_constraints):
                return RoboTwinBackend(graph, task, backend_config, placement_constraints=placement_constraints)

        result = collect_symbolic_trajectories(
            view_graph_path=args.view_graph,
            tasks_path=args.tasks,
            output_path=args.output,
            mode=args.mode,
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            teacher_config=teacher_config,
            failure_injection=failure_injection,
            placement_edge_constraints_path=args.placement_edge_constraints,
            backend_factory=backend_factory,
        )
        print(f"Wrote {result.count} {args.backend} trajectories to {result.output_path}")
        return 0
    if args.command == "evaluate-real-trajectories":
        from .real_observation_eval import RealObservationEvalConfig, evaluate_real_trajectories

        config = RealObservationEvalConfig(
            provider=args.provider,
            model=args.model,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
            api_base_url=args.api_base_url,
            api_style=args.api_style,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_api_attempts=args.max_api_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_max_seconds=args.retry_max_seconds,
            modes=_csv(args.modes),
            history_source=args.history_source,
            include_valid_actions=args.valid_actions,
            soft_optimal_beta=args.soft_optimal_beta,
            frame_count=args.frame_count,
            cameras=_csv(args.cameras),
            observation_window_seconds=args.observation_window_seconds,
            frame_sampling=args.frame_sampling,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
            oss_region=args.oss_region,
            oss_endpoint=args.oss_endpoint,
            cache_dir=args.cache_dir,
        )
        result = evaluate_real_trajectories(
            input_path=args.input,
            output_path=args.output,
            config=config,
        )
        print(f"Wrote {result['count']} evaluation records to {result['output_path']}")
        print(f"Wrote evaluation summary to {result['summary_path']}")
        return 0
    if args.command == "evaluate-view-graph-rollouts":
        from .view_graph_rollout_eval import ViewGraphRolloutEvalConfig, evaluate_view_graph_rollouts

        config = ViewGraphRolloutEvalConfig(
            provider=args.provider,
            model=args.model,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
            api_base_url=args.api_base_url,
            api_style=args.api_style,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_api_attempts=args.max_api_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_max_seconds=args.retry_max_seconds,
            include_valid_actions=args.valid_actions,
            soft_optimal_beta=args.soft_optimal_beta,
            max_steps=args.max_steps,
            history_window=args.history_window,
            max_consecutive_model_errors=args.max_consecutive_model_errors,
            failure_injection=args.failure_injection,
            failure_actions=_csv(args.failure_actions),
            failure_probability=args.failure_probability,
            max_failures_per_episode=args.max_failures_per_episode,
            failure_seed=args.failure_seed,
            graph_disturbance_file=args.graph_disturbance_file,
            fail_fast=args.fail_fast,
        )
        result = evaluate_view_graph_rollouts(
            input_path=args.input,
            output_path=args.output,
            config=config,
        )
        print(f"Wrote {result['count']} closed-loop records to {result['output_path']}")
        print(f"Wrote evaluation summary to {result['summary_path']}")
        return 0
    if args.command == "evaluate-view-graph-intervention-manifest":
        from .view_graph_rollout_eval import (
            ViewGraphRolloutEvalConfig,
            evaluate_view_graph_intervention_manifest,
        )

        config = ViewGraphRolloutEvalConfig(
            provider=args.provider,
            model=args.model,
            model_name=args.model_name,
            api_key_env=args.api_key_env,
            api_base_url=args.api_base_url,
            api_style=args.api_style,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            max_api_attempts=args.max_api_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_max_seconds=args.retry_max_seconds,
            include_valid_actions=args.valid_actions,
            soft_optimal_beta=args.soft_optimal_beta,
            max_steps=args.max_steps,
            history_window=args.history_window,
            max_consecutive_model_errors=args.max_consecutive_model_errors,
            fail_fast=args.fail_fast,
        )
        result = evaluate_view_graph_intervention_manifest(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            config=config,
            condition_ids=_csv(args.conditions),
        )
        print(f"Completed {result['condition_count']} intervention conditions")
        print(f"Wrote intervention suite summary to {result['suite_summary_path']}")
        return 0
    if args.command == "validate-robotwin-task":
        from .robotwin_adapter import validate_robotwin_task

        report = validate_robotwin_task(
            view_graph_path=args.view_graph,
            tasks_path=args.tasks,
            output_path=args.output,
            asset_map_path=args.asset_map,
            trajectory_path=args.trajectory,
            robotwin_root=args.robotwin_root,
            seed=args.seed,
        )
        print(f"Wrote RoboTwin validation report to {args.output}")
        print(
            f"ok={report['ok']} mapped={report['coverage']['nodes']['mapped']}/"
            f"{report['coverage']['nodes']['total']} unsupported_actions="
            f"{len(report['unsupported_actions'])}"
        )
        return 0 if report["ok"] else 1
    if args.command == "replay-robotwin-trajectory":
        from .robotwin_adapter import replay_robotwin_trajectory

        report = replay_robotwin_trajectory(
            view_graph_path=args.view_graph,
            tasks_path=args.tasks,
            trajectory_path=args.trajectory,
            asset_map_path=args.asset_map,
            robotwin_root=args.robotwin_root,
            task_config=args.task_config,
            seed=args.seed,
            render=args.render,
            execution_mode=args.execution_mode,
            output_dir=args.output_dir,
        )
        print(f"RoboTwin replay success={report['success']} steps={report['step_count']}")
        print(f"Wrote physical replay artifacts to {args.output_dir}")
        return 0 if report["success"] else 1
    parser.error(f"Unknown command: {args.command}")
    return 2
