from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constraints import available_modifiers
from .generator import ALL_TASK_TYPES, GenerationConfig, TaskGenerator
from .graph_io import load_view_graphs_jsonl, write_tasks_jsonl
from .harness import FailureInjectionConfig, TeacherPolicyConfig, collect_symbolic_trajectories
from .layout_synthesis import (
    TaskViewGraphSynthesisConfig,
    synthesize_task_view_graph,
    write_task_view_graph_package,
    write_view_graph_jsonl,
)
from .profile_editor import edit_view_graphs_with_profile
from .trajectory_server import serve_trajectory_app
from .view_graph_server import serve_view_graph_app


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
    collect.add_argument("--teacher-timeout-seconds", type=int, default=60)
    collect.add_argument("--teacher-temperature", type=float, default=0.0)
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
        serve_view_graph_app(
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "serve-trajectory":
        serve_trajectory_app(
            trajectory_path=args.trajectory,
            trajectory_dir=args.trajectory_dir,
            host=args.host,
            port=args.port,
            base_path=args.base_path,
            open_browser=args.open_browser,
        )
        return 0
    if args.command == "edit-view-graph":
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
        teacher_config = None
        if args.mode == "teacher":
            teacher_config = TeacherPolicyConfig(
                provider=args.teacher_provider,
                model=args.teacher_model,
                api_key_env=args.teacher_api_key_env,
                api_base_url=args.teacher_api_base_url,
                timeout_seconds=args.teacher_timeout_seconds,
                temperature=args.teacher_temperature,
            )
        failure_injection = FailureInjectionConfig(
            mode=args.failure_injection,
            actions=_csv(args.failure_actions),
            probability=args.failure_probability,
            max_failures_per_episode=args.max_failures_per_episode,
            seed=args.failure_seed,
        )
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
        )
        print(f"Wrote {result.count} symbolic trajectories to {result.output_path}")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2
