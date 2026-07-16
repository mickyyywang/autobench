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
        default="obs_only,graph_only,obs_plus_graph,wrong_graph_plus_obs",
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

    export_og = subparsers.add_parser(
        "export-omnigibson-config",
        help="Export a view graph into an OmniGibson-compatible JSON config package.",
    )
    export_og.add_argument("--view-graph", required=True, help="Input view graph JSONL path.")
    export_og.add_argument("--asset-map", default=None, help="Optional Autobench-to-OmniGibson asset map JSON.")
    export_og.add_argument("--output-dir", required=True, help="Directory for og_config.json and reports.")
    export_og.add_argument(
        "--no-primitive-fallback",
        action="store_true",
        help="Fail unmapped assets instead of writing PrimitiveObject fallback records.",
    )

    validate_og = subparsers.add_parser(
        "validate-omnigibson-task",
        help="Validate whether a view graph and task can be exported to the OmniGibson adapter layer.",
    )
    validate_og.add_argument("--view-graph", required=True, help="Input view graph JSONL path.")
    validate_og.add_argument("--tasks", required=True, help="Task JSONL path.")
    validate_og.add_argument("--asset-map", default=None, help="Optional Autobench-to-OmniGibson asset map JSON.")
    validate_og.add_argument("--trajectory", default=None, help="Optional trajectory JSONL to validate action parsing.")
    validate_og.add_argument("--output", required=True, help="Validation report JSON output path.")
    validate_og.add_argument(
        "--no-primitive-fallback",
        action="store_true",
        help="Treat missing asset map entries as errors.",
    )

    replay_og = subparsers.add_parser(
        "replay-omnigibson-trajectory",
        help="Replay an Autobench trajectory through the OmniGibson adapter state-level backend.",
    )
    replay_og.add_argument("--config", required=True, help="og_config.json exported by export-omnigibson-config.")
    replay_og.add_argument("--trajectory", required=True, help="Teacher trajectory JSONL path.")
    replay_og.add_argument("--mode", choices=("state",), default="state")
    replay_og.add_argument("--output-dir", required=True, help="Replay report output directory.")

    export_bddl = subparsers.add_parser(
        "export-bddl",
        help="Export BDDL-compatible goal predicates plus custom predicate reports.",
    )
    export_bddl.add_argument("--view-graph", required=True, help="Input view graph JSONL path.")
    export_bddl.add_argument("--tasks", required=True, help="Task JSONL path.")
    export_bddl.add_argument("--asset-map", default=None, help="Accepted for CLI symmetry; currently only used by validation.")
    export_bddl.add_argument("--output-dir", required=True, help="BDDL package output directory.")

    run_og = subparsers.add_parser(
        "run-omnigibson-config",
        help="Optionally load an exported config in an installed OmniGibson runtime.",
    )
    run_og.add_argument("--config", required=True, help="og_config.json exported by export-omnigibson-config.")
    run_og.add_argument("--output-dir", default=None, help="Runtime report directory. Defaults to config directory.")
    run_og.add_argument(
        "--no-apply-initial-state",
        action="store_true",
        help="Create the environment without attempting initial state application.",
    )

    dump_obs = subparsers.add_parser(
        "dump-omnigibson-observation",
        help="Load an exported OmniGibson config and dump real robot sensor observations.",
    )
    dump_obs.add_argument("--config", required=True, help="og_config.json exported by export-omnigibson-config.")
    dump_obs.add_argument("--trajectory", default=None, help="Optional Autobench trajectory JSONL to replay best-effort.")
    dump_obs.add_argument("--output-dir", required=True, help="Directory for observation files and reports.")
    dump_obs.add_argument("--robot-model", default="fetch", help="OmniGibson robot model to insert if config has none.")
    dump_obs.add_argument(
        "--modalities",
        default="rgb,depth",
        help="Comma separated observation modalities, e.g. rgb,depth,seg_instance.",
    )
    dump_obs.add_argument("--image-width", type=int, default=128)
    dump_obs.add_argument("--image-height", type=int, default=128)
    dump_obs.add_argument("--steps", type=int, default=1, help="Random environment steps before dumping observations.")
    dump_obs.add_argument(
        "--force",
        action="store_true",
        help="Bypass runtime preflight blockers and attempt to start Isaac/OmniGibson anyway.",
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
            base_path=args.base_path,
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
    if args.command == "export-omnigibson-config":
        from .omnigibson_adapter import export_omnigibson_config

        paths = export_omnigibson_config(
            view_graph_path=args.view_graph,
            asset_map_path=args.asset_map,
            output_dir=args.output_dir,
            allow_primitive_fallback=not args.no_primitive_fallback,
        )
        print(f"Wrote OmniGibson config package to {args.output_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0
    if args.command == "validate-omnigibson-task":
        from .omnigibson_adapter import validate_omnigibson_task

        report = validate_omnigibson_task(
            view_graph_path=args.view_graph,
            tasks_path=args.tasks,
            asset_map_path=args.asset_map,
            output_path=args.output,
            trajectory_path=args.trajectory,
            allow_primitive_fallback=not args.no_primitive_fallback,
        )
        print(f"Wrote OmniGibson validation report to {args.output}")
        print(f"ok={report['ok']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
        return 0
    if args.command == "replay-omnigibson-trajectory":
        from .omnigibson_adapter import replay_omnigibson_trajectory

        paths = replay_omnigibson_trajectory(
            config_path=args.config,
            trajectory_path=args.trajectory,
            output_dir=args.output_dir,
            mode=args.mode,
        )
        print(f"Wrote state-level replay reports to {args.output_dir}")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0
    if args.command == "export-bddl":
        from .omnigibson_adapter.goal_mapper import export_bddl_package
        from .omnigibson_adapter.io_utils import load_tasks_jsonl, require_single_graph

        graph = require_single_graph(load_view_graphs_jsonl(args.view_graph), args.view_graph)
        tasks = load_tasks_jsonl(args.tasks)
        output_dir = Path(args.output_dir)
        for task in tasks:
            target_dir = output_dir if len(tasks) == 1 else output_dir / task.task_id
            export_bddl_package(task=task, graph=graph, output_dir=target_dir)
        print(f"Wrote BDDL package for {len(tasks)} task(s) to {args.output_dir}")
        return 0
    if args.command == "run-omnigibson-config":
        from .omnigibson_adapter.runtime import run_omnigibson_config

        report = run_omnigibson_config(
            config_path=args.config,
            output_dir=args.output_dir,
            apply_initial_state=not args.no_apply_initial_state,
        )
        print(f"OmniGibson runtime ok={report.get('ok')} stage={report.get('stage')}")
        if report.get("error"):
            print(report["error"])
        return 0 if report.get("ok") else 1
    if args.command == "dump-omnigibson-observation":
        from .omnigibson_adapter.runtime import dump_omnigibson_observation

        report = dump_omnigibson_observation(
            config_path=args.config,
            output_dir=args.output_dir,
            trajectory_path=args.trajectory,
            robot_model=args.robot_model,
            modalities=_csv(args.modalities),
            image_width=args.image_width,
            image_height=args.image_height,
            steps=args.steps,
            force=args.force,
        )
        print(f"OmniGibson observation ok={report.get('ok')} stage={report.get('stage')}")
        if report.get("message"):
            print(report["message"])
        if report.get("error"):
            print(report["error"])
        if report.get("files"):
            print(f"wrote {len(report['files'])} observation files to {args.output_dir}")
        return 0 if report.get("ok") else 1
    parser.error(f"Unknown command: {args.command}")
    return 2
