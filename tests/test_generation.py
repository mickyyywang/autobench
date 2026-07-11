from __future__ import annotations

import contextlib
import copy
from http.server import ThreadingHTTPServer
import io
import json
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib import request
import unittest
from unittest.mock import patch

from auto_embodied_task.cli import main
from auto_embodied_task.generator import GenerationConfig, TaskGenerator
from auto_embodied_task.graph_io import load_view_graphs_jsonl
from auto_embodied_task.harness import (
    FailureInjectionConfig,
    ParsedAction,
    PlacementEdgeConstraints,
    ScriptedTeacherPolicy,
    SymbolicBackend,
    SymbolicHarness,
    _teacher_user_prompt,
    load_placement_edge_constraints,
)
from auto_embodied_task.layout_synthesis import (
    TaskViewGraphSynthesisConfig,
    _validate_task_view_graph_package,
    build_task_view_graph_prompt,
)
from auto_embodied_task.models import TaskRecord, ViewGraph
from auto_embodied_task.trajectory_server import _TrajectoryAppHandler, trajectory_replay_payload
from auto_embodied_task.view_graph_server import _ViewGraphAppHandler, _config_from_payload, _render_app_html


SCENE = {
    "scene_id": "test_scene",
    "env_id": 1,
    "layout": "indoor",
    "robot": {"arms": "single"},
    "nodes": [
        {"id": "kitchen", "name": "kitchen", "category": "room"},
        {"id": "table", "name": "table", "category": "surface", "room": "kitchen", "properties": ["SURFACES"]},
        {"id": "box", "name": "box", "category": "container", "room": "kitchen", "properties": ["CONTAINERS"]},
        {"id": "book", "name": "book", "category": "object", "room": "kitchen", "properties": ["OCCLUDER"]},
        {"id": "apple", "name": "apple", "category": "food", "room": "kitchen", "parent": "box", "properties": ["GRABBABLE", "MOVABLE"]},
        {"id": "spoon", "name": "spoon", "category": "tool", "room": "kitchen", "properties": ["GRABBABLE", "MOVABLE"]},
    ],
    "edges": [
        {"from": "table", "to": "kitchen", "relation": "INSIDE"},
        {"from": "box", "to": "kitchen", "relation": "INSIDE"},
        {"from": "apple", "to": "box", "relation": "INSIDE"},
        {"from": "box", "to": "apple", "relation": "OCCLUDES"},
        {"from": "spoon", "to": "table", "relation": "ON"},
        {"from": "book", "to": "spoon", "relation": "OCCLUDES"},
    ],
}


class GenerationTest(unittest.TestCase):
    def test_generate_base_and_setting_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scene.jsonl"
            path.write_text(json.dumps(SCENE) + "\n", encoding="utf-8")
            graphs = load_view_graphs_jsonl(path)
            config = GenerationConfig(
                task_types=("multi_object",),
                settings=("spatial", "temporal", "failure_recovery"),
                max_tasks=8,
                seed=3,
            )
            tasks = TaskGenerator(config).generate(graphs)

        self.assertGreaterEqual(len(tasks), 4)
        self.assertTrue(any("spatial" in task.settings for task in tasks))
        self.assertTrue(any("temporal" in task.settings for task in tasks))
        self.assertTrue(any("failure_recovery" in task.settings for task in tasks))
        self.assertTrue(all(task.task_completion_criterion for task in tasks))
        spatial_tasks = [task for task in tasks if "spatial" in task.settings]
        setting_tasks = [task for task in tasks if task.settings]
        leaked_phrases = ("Spatial clue", "that is on", "Use memory", "First locate", "fails")
        self.assertTrue(all(phrase not in task.task for task in setting_tasks for phrase in leaked_phrases))
        self.assertTrue(all("constraint_subtasks" in task.metadata for task in setting_tasks))
        self.assertTrue(any("(INSIDE, apple, box)" in task.task_completion_criterion for task in spatial_tasks))
        self.assertTrue(any("(OCCLUDED_BY, apple, box)" in task.task_completion_criterion for task in spatial_tasks))

    def test_generate_long_horizon_tasks_use_graph_constraints(self) -> None:
        scene = copy.deepcopy(SCENE)
        for node in scene["nodes"]:
            if node["id"] == "box":
                node["properties"] = ["CONTAINERS", "CAN_OPEN"]
                node["states"] = ["CLOSED"]
            if node["id"] == "book":
                node["properties"] = ["GRABBABLE", "MOVABLE", "OCCLUDER"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scene.jsonl"
            path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            graphs = load_view_graphs_jsonl(path)
            config = GenerationConfig(
                task_types=("long_horizon",),
                settings=("spatial", "temporal", "failure_recovery"),
                max_tasks=8,
                seed=1,
            )
            tasks = TaskGenerator(config).generate(graphs)

        base_tasks = [task for task in tasks if task.task_type == "long_horizon" and not task.settings]
        self.assertTrue(base_tasks)
        base = base_tasks[0]
        self.assertEqual(base.metadata["long_horizon"]["num_steps"], 2)
        self.assertTrue(base.metadata["long_horizon"]["uses_constraints"])
        self.assertIn("STEP_1", base.task_completion_criterion)
        self.assertIn("STEP_2", base.task_completion_criterion)
        self.assertGreaterEqual(len(base.ground_truth_plan), 8)
        self.assertTrue(any("[move_aside]" in action for action in base.ground_truth_plan))
        self.assertTrue(any("[open]" in action for action in base.ground_truth_plan))
        self.assertTrue(any("[close]" in action for action in base.ground_truth_plan))
        self.assertTrue(base.objects["placements"])
        subtask_types = {item["type"] for item in base.metadata["constraint_subtasks"]}
        self.assertIn("resolve_access_constraint", subtask_types)
        self.assertIn("ordered_placement", subtask_types)
        self.assertTrue(any("temporal" in task.settings for task in tasks))
        failure_task = next(task for task in tasks if "failure_recovery" in task.settings)
        self.assertTrue(any("[failed_grab]" in action for action in failure_task.ground_truth_plan))

    def test_cli_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "scene.jsonl"
            output_path = Path(tmpdir) / "tasks.jsonl"
            input_path.write_text(json.dumps(SCENE) + "\n", encoding="utf-8")
            code = main(
                [
                    "generate",
                    "--view-graph",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--task-types",
                    "navigation,manipulation",
                    "--settings",
                    "memory",
                    "--max-tasks",
                    "6",
                ]
            )
            self.assertEqual(code, 0)
            lines = output_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 6)
        payloads = [json.loads(line) for line in lines]
        payload = payloads[0]
        self.assertIn("ground_truth_plan", payload)
        self.assertIn("task_completion_criterion", payload)
        memory_payloads = [item for item in payloads if "memory" in item["settings"]]
        self.assertTrue(memory_payloads)
        for memory_payload in memory_payloads:
            metadata = memory_payload["metadata"]
            self.assertNotIn("RECALL", memory_payload["task_completion_criterion"])
            self.assertEqual(metadata["memory_constraint"]["mode"], "prior_observation")
            self.assertTrue(metadata["memory_constraint"]["not_initial_state"])
            self.assertEqual(metadata["memory_episode"]["type"], "prior_observation")
            self.assertTrue(metadata["memory_episode"]["not_initial_state"])
            self.assertEqual(metadata["constraint_subtasks"][0]["type"], "retrieve_prior_observation")
        apple_memory = next(
            item
            for item in memory_payloads
            if item["metadata"]["memory_constraint"]["remember_object"] == "apple"
        )
        self.assertEqual(apple_memory["metadata"]["memory_constraint"]["remember_relation"], "INSIDE")
        self.assertEqual(apple_memory["metadata"]["memory_constraint"]["remember_anchor"], "box")

    def test_cli_collects_symbolic_trajectory_with_memory_and_failure(self) -> None:
        scene = {
            "scene_id": "harness_scene",
            "env_id": "harness_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "parent": "table",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
                {
                    "id": "apple",
                    "name": "apple",
                    "category": "food",
                    "parent": "box",
                    "properties": ["GRABBABLE", "MOVABLE"],
                },
            ],
            "edges": [
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "apple", "to": "box", "relation": "INSIDE"},
                {"from": "box", "to": "apple", "relation": "OCCLUDES"},
            ],
        }
        task = {
            "task_id": "harness_task_1",
            "scene_id": "harness_scene",
            "env_id": "harness_scene",
            "layout": "tabletop",
            "arms": "single",
            "task_type": "manipulation",
            "task": "Put the apple on the table.",
            "task_completion_criterion": "(ON, apple, table)",
            "ground_truth_plan": [
                "<char0> [look] <box> (box)",
                "<char0> [open] <box> (box)",
                "<char0> [reach] <apple> (apple)",
                "<char0> [failed_grab] <apple> (apple)",
                "<char0> [recover] <apple> (apple)",
                "<char0> [grab] <apple> (apple)",
                "<char0> [reach] <table> (table)",
                "<char0> [puton] <apple> (apple) <table> (table)",
            ],
            "objects": {"object": "apple", "target": "table", "relation": "ON"},
            "settings": ["memory", "temporal", "failure_recovery"],
            "metadata": {
                "memory_constraint": {
                    "mode": "prior_observation",
                    "not_initial_state": True,
                    "remember_object": "apple",
                    "remember_anchor": "box",
                    "remember_relation": "INSIDE",
                    "memory_fact": "(INSIDE, apple, box)",
                },
                "memory_episode": {
                    "type": "prior_observation",
                    "not_initial_state": True,
                    "observations": [{"object": "apple", "anchor": "box", "relation": "INSIDE"}],
                },
                "constraint_subtasks": [
                    {
                        "setting": "temporal",
                        "type": "ordered_placement",
                        "step": 1,
                        "object": "apple",
                        "target": "table",
                        "relation": "ON",
                    },
                    {
                        "setting": "failure_recovery",
                        "type": "inject_failure",
                        "action": "grab",
                        "object": "apple",
                    },
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            task_path = Path(tmpdir) / "tasks.jsonl"
            trajectory_path = Path(tmpdir) / "trajectories.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            code = main(
                [
                    "collect-trajectories",
                    "--view-graph",
                    str(graph_path),
                    "--tasks",
                    str(task_path),
                    "--output",
                    str(trajectory_path),
                ]
            )
            self.assertEqual(code, 0)
            generated = sorted(Path(tmpdir).glob("trajectories_*.jsonl"))
            episodes = [json.loads(line) for line in generated[0].read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertTrue(episode["success"])
        initial_visible = {item["id"] for item in episode["initial_observation"]["visible_nodes"]}
        self.assertNotIn("apple", initial_visible)
        self.assertIn("box", initial_visible)
        self.assertEqual(episode["metrics"]["memory"]["remembered_anchor"], "box")
        self.assertTrue(episode["metrics"]["memory"]["object_hidden_initially"])
        self.assertTrue(episode["metrics"]["memory"]["used_remembered_anchor"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["failure_observed"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["recovered_after_failure"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["retried_failed_action"])
        self.assertTrue(episode["metrics"]["temporal"]["ordered"])
        self.assertEqual(
            episode["final_state"]["nodes"]["apple"]["location"],
            {"relation": "ON", "target": "table"},
        )
        self.assertTrue(any(step["event"]["status"] == "failure" for step in episode["trajectory"]))

    def test_cli_collect_trajectories_does_not_overwrite_existing_output(self) -> None:
        scene = {
            "scene_id": "timestamp_scene",
            "env_id": "timestamp_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [{"from": "apple", "to": "table", "relation": "ON"}],
        }
        task = {
            "task_id": "timestamp_task",
            "scene_id": "timestamp_scene",
            "env_id": "timestamp_scene",
            "layout": "tabletop",
            "arms": "single",
            "task_type": "manipulation",
            "task": "Keep apple on table.",
            "task_completion_criterion": "(ON, apple, table)",
            "ground_truth_plan": [],
            "objects": {"object": "apple", "target": "table", "relation": "ON"},
            "settings": [],
            "metadata": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            task_path = Path(tmpdir) / "tasks.jsonl"
            trajectory_path = Path(tmpdir) / "trajectories.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            trajectory_path.write_text("old trajectories\n", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "collect-trajectories",
                        "--view-graph",
                        str(graph_path),
                        "--tasks",
                        str(task_path),
                        "--output",
                        str(trajectory_path),
                    ]
                )

            generated = sorted(Path(tmpdir).glob("trajectories_*.jsonl"))
            original = trajectory_path.read_text(encoding="utf-8")
            generated_payloads = [
                json.loads(line)
                for line in generated[0].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(code, 0)
        self.assertEqual(original, "old trajectories\n")
        self.assertEqual(len(generated), 1)
        self.assertIn(str(generated[0]), stdout.getvalue())
        self.assertEqual(len(generated_payloads), 1)
        self.assertEqual(generated_payloads[0]["episode_id"], "timestamp_task")

    def test_symbolic_backend_contract(self) -> None:
        scene = copy.deepcopy(SCENE)
        for node in scene["nodes"]:
            if node["id"] == "box":
                node["properties"] = ["CONTAINERS", "CAN_OPEN", "OCCLUDER"]
                node["states"] = ["CLOSED"]
        graph = ViewGraph.from_dict(scene, fallback_scene_id="contract_scene")
        task = TaskRecord(
            task_id="backend_contract",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Move apple to table.",
            task_completion_criterion="(ON, apple, table)",
            ground_truth_plan=[],
            objects={"object": "apple", "target": "table", "relation": "ON"},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        self.assertEqual(backend.name, "symbolic")
        initial = backend.observe()
        initial_nodes = {item["id"]: item for item in initial["visible_nodes"]}
        self.assertEqual(initial_nodes["box"]["properties"], ["CONTAINERS", "CAN_OPEN", "OCCLUDER"])
        self.assertEqual(initial_nodes["box"]["states"], ["CLOSED"])
        self.assertTrue(initial_nodes["box"]["openable"])
        self.assertTrue(initial_nodes["box"]["container"])
        self.assertFalse(initial_nodes["box"]["surface"])
        self.assertEqual(initial_nodes["box"]["occludes_hidden_count"], 1)
        self.assertEqual(initial["robot"]["hands"]["capacity"], 1)
        self.assertEqual(initial["robot"]["hands"]["free_count"], 1)
        self.assertEqual(
            initial["robot"]["hands"]["slots"],
            [
                {"name": "left", "available": True, "holding": None},
                {"name": "right", "available": False, "holding": None},
            ],
        )
        initial_payload = json.loads(_teacher_user_prompt(task, initial, []))
        self.assertIn({"name": "open", "node_ids": ["box"], "object": "box", "effect_hint": "may_reveal_hidden"}, initial_payload["valid_actions"])
        visible = {item["id"] for item in initial["visible_nodes"]}
        self.assertNotIn("apple", visible)
        self.assertFalse(backend.success())

        self.assertEqual(backend.step(ParsedAction("open", ["box"]))["status"], "success")
        after_open = backend.observe()
        after_open_nodes = {item["id"]: item for item in after_open["visible_nodes"]}
        self.assertEqual(after_open_nodes["box"]["occludes_hidden_count"], 0)
        after_open_edges = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in after_open["visible_edges"]
        }
        self.assertIn(("apple", "box", "INSIDE"), after_open_edges)
        self.assertEqual(backend.step(ParsedAction("grab", ["apple"]))["status"], "success")
        after_grab_edges = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in backend.observe()["visible_edges"]
        }
        after_grab_hands = backend.observe()["robot"]["hands"]
        self.assertEqual(after_grab_hands["free_count"], 0)
        self.assertEqual(after_grab_hands["slots"][0], {"name": "left", "available": True, "holding": {"id": "apple", "name": "apple"}})
        self.assertNotIn(("apple", "box", "INSIDE"), after_grab_edges)
        self.assertNotIn(("apple", "table", "ON"), after_grab_edges)
        self.assertEqual(backend.step(ParsedAction("puton", ["apple", "table"]))["status"], "success")
        after_puton_edges = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in backend.observe()["visible_edges"]
        }
        self.assertIn(("apple", "table", "ON"), after_puton_edges)
        self.assertNotIn(("apple", "box", "INSIDE"), after_puton_edges)
        self.assertEqual(backend.observe()["robot"]["hands"]["free_count"], 1)
        self.assertTrue(backend.success())
        self.assertEqual(
            backend.snapshot()["nodes"]["apple"]["location"],
            {"relation": "ON", "target": "table"},
        )
        self.assertTrue(backend.metrics(initial)["success"])

    def test_closed_movable_openable_container_occluder_must_be_opened(self) -> None:
        scene = {
            "scene_id": "closed_movable_occluder_scene",
            "env_id": "closed_movable_occluder_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "folder",
                    "name": "folder",
                    "category": "container",
                    "properties": ["GRABBABLE", "MOVABLE", "CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "paper", "relation": "OCCLUDES"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="closed_movable_occluder_scene")
        task = TaskRecord(
            task_id="closed_movable_occluder_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Reveal the paper.",
            task_completion_criterion="(VISIBLE, paper)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        initial = backend.observe()
        initial_visible = {item["id"] for item in initial["visible_nodes"]}
        self.assertNotIn("paper", initial_visible)
        folder = next(item for item in initial["visible_nodes"] if item["id"] == "folder")
        self.assertTrue(folder["openable"])
        self.assertFalse(folder["open"])
        self.assertEqual(folder["occludes_hidden_count"], 1)

        payload = json.loads(_teacher_user_prompt(task, initial, []))
        self.assertIn(
            {"name": "open", "node_ids": ["folder"], "object": "folder", "effect_hint": "may_reveal_hidden"},
            payload["valid_actions"],
        )
        self.assertNotIn(
            {"name": "move_aside", "node_ids": ["folder"], "object": "folder", "effect_hint": "may_reveal_hidden"},
            payload["valid_actions"],
        )
        self.assertEqual(backend.step(ParsedAction("move_aside", ["folder"]))["failure_type"], "requires_open")
        self.assertEqual(backend.step(ParsedAction("clear", ["folder"]))["failure_type"], "unsupported_resolution")

        self.assertEqual(backend.step(ParsedAction("open", ["folder"]))["status"], "success")
        after_open = backend.observe()
        self.assertIn("paper", {item["id"] for item in after_open["visible_nodes"]})
        self.assertTrue(backend.success())

    def test_teacher_uses_plain_putdown_actions_for_held_blocker(self) -> None:
        scene = {
            "scene_id": "revealing_blocker_putdown_scene",
            "env_id": "revealing_blocker_putdown_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "桌面", "name": "桌面", "category": "surface", "properties": ["SURFACES"]},
                {"id": "书", "name": "书", "category": "object", "properties": ["GRABBABLE", "MOVABLE", "OCCLUDER"]},
                {"id": "黑色笔", "name": "黑色笔", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "书", "to": "桌面", "relation": "ON"},
                {"from": "黑色笔", "to": "桌面", "relation": "ON"},
                {"from": "书", "to": "黑色笔", "relation": "OCCLUDES"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="revealing_blocker_putdown_scene")
        task = TaskRecord(
            task_id="revealing_blocker_putdown_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="double",
            task_type="manual_ready_goal",
            task="把黑色笔放进铅笔盒。",
            task_completion_criterion="(VISIBLE, 黑色笔)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)
        before = backend.observe()
        self.assertNotIn("黑色笔", {item["id"] for item in before["visible_nodes"]})
        self.assertEqual(backend.step(ParsedAction("grab", ["书"]))["status"], "success")
        after = backend.observe()
        self.assertNotIn("黑色笔", {item["id"] for item in after["visible_nodes"]})

        payload = json.loads(
            _teacher_user_prompt(
                task,
                after,
                [
                    {
                        "action": {"name": "grab", "base_name": "grab", "node_ids": ["书"]},
                        "event": {"status": "success"},
                        "new_visible_nodes": [],
                    }
                ],
            )
        )

        puton_action = {"name": "puton", "node_ids": ["书", "桌面"], "object": "书", "target": "桌面"}
        self.assertIn(puton_action, payload["valid_actions"])
        returned_puton = next(action for action in payload["valid_actions"] if action == puton_action)
        self.assertTrue(all("priority" not in action for action in payload["valid_actions"]))
        self.assertNotIn("effect_hint", returned_puton)

    def test_symbolic_observation_exposes_non_openable_container_affordance(self) -> None:
        scene = copy.deepcopy(SCENE)
        scene["edges"] = [
            edge
            for edge in scene["edges"]
            if not (edge["from"] == "book" and edge["to"] == "spoon" and edge["relation"] == "OCCLUDES")
        ]
        graph = ViewGraph.from_dict(scene, fallback_scene_id="affordance_scene")
        task = TaskRecord(
            task_id="backend_affordances",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Inspect scene affordances.",
            task_completion_criterion="(VISIBLE, box)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        nodes = {item["id"]: item for item in backend.observe()["visible_nodes"]}

        self.assertEqual(nodes["box"]["properties"], ["CONTAINERS"])
        self.assertEqual(nodes["box"]["states"], [])
        self.assertFalse(nodes["box"]["openable"])
        self.assertTrue(nodes["box"]["container"])
        self.assertFalse(nodes["box"]["grabbable"])
        self.assertTrue(nodes["spoon"]["grabbable"])
        self.assertTrue(nodes["spoon"]["movable"])

    def test_open_container_occluder_reveals_occluded_nodes(self) -> None:
        scene = copy.deepcopy(SCENE)
        for node in scene["nodes"]:
            if node["id"] == "box":
                node["properties"] = ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "HIDDEN"]
                node["states"] = ["CLOSED"]
        graph = ViewGraph.from_dict(scene, fallback_scene_id="open_container_occluder_scene")
        task = TaskRecord(
            task_id="open_container_occluder",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Find the apple.",
            task_completion_criterion="(VISIBLE, apple)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        initial = backend.observe()
        initial_visible = {item["id"] for item in initial["visible_nodes"]}
        initial_nodes = {item["id"]: item for item in initial["visible_nodes"]}
        self.assertNotIn("apple", initial_visible)
        self.assertEqual(initial_nodes["box"]["occludes_hidden_count"], 1)
        initial_payload = json.loads(_teacher_user_prompt(task, initial, []))
        self.assertIn({"name": "open", "node_ids": ["box"], "object": "box", "effect_hint": "may_reveal_hidden"}, initial_payload["valid_actions"])
        self.assertNotIn({"name": "inspect", "node_ids": ["box"], "object": "box", "effect_hint": "may_reveal_hidden"}, initial_payload["valid_actions"])

        self.assertEqual(backend.step(ParsedAction("inspect", ["box"]))["status"], "success")
        self.assertNotIn("apple", {item["id"] for item in backend.observe()["visible_nodes"]})
        self.assertEqual(backend.step(ParsedAction("open", ["box"]))["status"], "success")
        after_open = backend.observe()
        after_open_visible = {item["id"] for item in after_open["visible_nodes"]}
        after_open_nodes = {item["id"]: item for item in after_open["visible_nodes"]}

        self.assertIn("apple", after_open_visible)
        self.assertEqual(after_open_nodes["box"]["occludes_hidden_count"], 0)
        self.assertEqual(
            backend.snapshot()["nodes"]["apple"]["location"],
            {"relation": "INSIDE", "target": "box"},
        )

        harness_task = TaskRecord(
            task_id="open_container_occluder_harness",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Find the apple.",
            task_completion_criterion="(VISIBLE, apple)",
            ground_truth_plan=["<char0> [open] <box> (box)"],
            objects={},
            metadata={},
        )
        episode = SymbolicHarness(graph, harness_task).run()
        open_step = episode["trajectory"][0]
        initial_graph = episode["initial_view_graph"]
        self.assertIn("initial_state", episode)
        self.assertIn("apple", {item["id"] for item in initial_graph["nodes"]})
        self.assertIn({"from": "box", "to": "apple", "relation": "OCCLUDES"}, initial_graph["edges"])
        self.assertIn("apple", {item["id"] for item in open_step["post_observation"]["visible_nodes"]})
        self.assertEqual([item["id"] for item in open_step["new_visible_nodes"]], ["apple"])

    def test_teacher_prompt_allows_only_visible_nodes(self) -> None:
        graph = ViewGraph.from_dict(copy.deepcopy(SCENE), fallback_scene_id="teacher_allowed_scene")
        task = TaskRecord(
            task_id="teacher_allowed",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Put the apple on the table.",
            task_completion_criterion="(ON, apple, table)",
            ground_truth_plan=[],
            objects={"object": "apple", "target": "table"},
            metadata={},
        )
        observation = SymbolicBackend(graph, task).observe()

        payload = json.loads(_teacher_user_prompt(task, observation, []))

        self.assertNotIn("map_layout", payload["current_observation"])
        self.assertIn("box", payload["allowed_node_ids"])
        self.assertNotIn("apple", payload["allowed_node_ids"])
        self.assertNotIn("apple", payload["allowed_node_names"])
        box = next(item for item in payload["current_observation"]["visible_nodes"] if item["id"] == "box")
        self.assertNotIn("occludes_hidden_count", box)
        self.assertIn("action_catalog", payload)
        self.assertIn("valid_actions", payload)
        self.assertNotIn({"name": "clear", "node_ids": ["box"], "object": "box", "effect_hint": "may_reveal_hidden"}, payload["valid_actions"])
        self.assertTrue(any("valid_actions" in item for item in payload["action_constraints"]))
        self.assertTrue(any("new_visible_nodes" in item for item in payload["action_constraints"]))

    def test_teacher_valid_actions_prioritize_open_before_grab(self) -> None:
        scene = {
            "scene_id": "open_before_grab_scene",
            "env_id": "open_before_grab_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "GRABBABLE", "MOVABLE"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="open_before_grab_scene")
        task = TaskRecord(
            task_id="open_before_grab_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="double",
            task_type="manipulation",
            task="Open the box.",
            task_completion_criterion="(OPEN, box)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        payload = json.loads(_teacher_user_prompt(task, SymbolicBackend(graph, task).observe(), []))
        actions = payload["valid_actions"]

        open_box_index = actions.index({"name": "open", "node_ids": ["box"], "object": "box"})
        grab_apple_index = actions.index({"name": "grab", "node_ids": ["apple"], "object": "apple"})
        grab_box_index = actions.index({"name": "grab", "node_ids": ["box"], "object": "box"})
        self.assertLess(open_box_index, grab_apple_index)
        self.assertLess(open_box_index, grab_box_index)

    def test_teacher_payload_includes_robot_hand_state(self) -> None:
        scene = {
            "scene_id": "hands_scene",
            "env_id": "hands_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "spoon", "name": "spoon", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "spoon", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="hands_scene")
        task = TaskRecord(
            task_id="hands_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="double",
            task_type="manipulation",
            task="Put the apple on the table.",
            task_completion_criterion="(ON, apple, table)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)
        self.assertEqual(backend.step(ParsedAction("grab", ["apple"]))["status"], "success")

        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))
        hands = payload["current_observation"]["robot"]["hands"]

        self.assertEqual(hands["capacity"], 2)
        self.assertEqual(hands["occupied_count"], 1)
        self.assertEqual(hands["free_count"], 1)
        self.assertEqual(hands["held_object_ids"], ["apple"])
        self.assertEqual(hands["slots"][0], {"name": "left", "available": True, "holding": {"id": "apple", "name": "apple"}})
        self.assertTrue(hands["slots"][1]["available"])
        self.assertIsNone(hands["slots"][1]["holding"])
        self.assertEqual(payload["action_catalog"]["attach"]["hand_usage"]["required_free_hands"], 2)
        self.assertEqual(payload["action_catalog"]["open"]["hand_usage"]["required_free_hands"], 2)
        self.assertEqual(payload["action_catalog"]["close"]["hand_usage"]["required_free_hands"], 2)
        self.assertEqual(payload["action_catalog"]["grab"]["hand_usage"]["result"], "occupies_one_hand")
        self.assertTrue(payload["action_catalog"]["puton"]["hand_usage"]["held_object_required"])
        self.assertTrue(any("robot.hands" in item for item in payload["action_constraints"]))

    def test_teacher_does_not_open_while_holding_and_puton_requires_surface(self) -> None:
        scene = {
            "scene_id": "held_open_and_puton_scene",
            "env_id": "held_open_and_puton_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN"], "states": ["OPEN"]},
                {"id": "drawer", "name": "drawer", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN"], "states": ["CLOSED"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "drawer", "to": "table", "relation": "ON"},
                {"from": "apple", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="held_open_and_puton_scene")
        task = TaskRecord(
            task_id="held_open_and_puton_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="double",
            task_type="manipulation",
            task="Move apple.",
            task_completion_criterion="(ON, apple, table)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        self.assertEqual(backend.step(ParsedAction("grab", ["apple"]))["status"], "success")
        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))

        self.assertIn({"name": "puton", "node_ids": ["apple", "table"], "object": "apple", "target": "table"}, payload["valid_actions"])
        self.assertIn({"name": "putin", "node_ids": ["apple", "box"], "object": "apple", "target": "box"}, payload["valid_actions"])
        self.assertNotIn({"name": "puton", "node_ids": ["apple", "box"], "object": "apple", "target": "box"}, payload["valid_actions"])
        self.assertFalse(any(action["name"] == "open" and action.get("object") == "drawer" for action in payload["valid_actions"]))
        self.assertFalse(any(action["name"] == "close" and action.get("object") == "box" for action in payload["valid_actions"]))

        self.assertEqual(backend.step(ParsedAction("puton", ["apple", "box"]))["failure_type"], "not_surface")
        self.assertEqual(backend.step(ParsedAction("open", ["drawer"]))["failure_type"], "hands_occupied")
        self.assertEqual(backend.step(ParsedAction("close", ["box"]))["failure_type"], "hands_occupied")

    def test_container_max_items_blocks_putin_and_valid_actions(self) -> None:
        scene = {
            "scene_id": "container_capacity_scene",
            "env_id": "container_capacity_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN"],
                    "states": ["OPEN"],
                    "max_items": 1,
                },
                {"id": "banana", "name": "banana", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "banana", "to": "box", "relation": "INSIDE"},
                {"from": "apple", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="container_capacity_scene")
        task = TaskRecord(
            task_id="container_capacity_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="double",
            task_type="manual_ready_goal",
            task="Keep the box within capacity.",
            task_completion_criterion={"and": [["AT_MOST_INSIDE", "box", 1]]},
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        box = next(node for node in backend.observe()["visible_nodes"] if node["id"] == "box")
        self.assertEqual(box["max_items"], 1)
        self.assertEqual(box["item_count"], 1)
        self.assertTrue(box["is_full"])
        self.assertTrue(backend.success())

        self.assertEqual(backend.step(ParsedAction("grab", ["apple"]))["status"], "success")
        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))
        self.assertNotIn(
            {"name": "putin", "node_ids": ["apple", "box"], "object": "apple", "target": "box"},
            payload["valid_actions"],
        )

        event = backend.step(ParsedAction("putin", ["apple", "box"]))
        self.assertEqual(event["status"], "failure")
        self.assertEqual(event["failure_type"], "container_full")

    def test_placement_edge_constraints_fail_at_backend_not_valid_actions(self) -> None:
        scene = {
            "scene_id": "placement_constraints_scene",
            "env_id": "placement_constraints_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "桌面", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "收纳盒", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "pencil_case", "name": "铅笔盒", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "red_pen", "name": "红色笔", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "pencil_case", "to": "table", "relation": "ON"},
                {"from": "red_pen", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="placement_constraints_scene")
        task = TaskRecord(
            task_id="placement_constraints_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manual_ready_goal",
            task="把红色笔放进铅笔盒。",
            task_completion_criterion="(INSIDE, red_pen, pencil_case)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        constraints = PlacementEdgeConstraints.from_json(
            {
                "nonexistent_edges": [
                    {"object": "红色笔", "target": "收纳盒", "action": "putin"},
                ]
            }
        )
        backend = SymbolicBackend(graph, task, placement_edge_constraints=constraints)

        self.assertEqual(backend.step(ParsedAction("grab", ["red_pen"]))["status"], "success")
        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))

        self.assertIn(
            {"name": "putin", "node_ids": ["red_pen", "box"], "object": "red_pen", "target": "box"},
            payload["valid_actions"],
        )
        self.assertIn(
            {
                "name": "putin",
                "node_ids": ["red_pen", "pencil_case"],
                "object": "red_pen",
                "target": "pencil_case",
            },
            payload["valid_actions"],
        )
        event = backend.step(ParsedAction("putin", ["red_pen", "box"]))
        self.assertEqual(event["status"], "failure")
        self.assertEqual(event["failure_type"], "disallowed_placement_edge")
        self.assertFalse(event["injected"])

    def test_placement_edge_constraints_load_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "placement_constraints.json"
            path.write_text(
                json.dumps(
                    {
                        "forbidden_edges": [
                            {"from": "红色笔", "to": "收纳盒", "relation": "IN"},
                            {"object": "*", "target": "桌面", "action": "puton"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            constraints = load_placement_edge_constraints(path)

        self.assertFalse(
            constraints.allows(
                source_id="red_pen",
                source_name="红色笔",
                target_id="box",
                target_name="收纳盒",
                relation="INSIDE",
            )
        )
        self.assertFalse(
            constraints.allows(
                source_id="anything",
                source_name=None,
                target_id="table",
                target_name="桌面",
                relation="ON",
            )
        )

    def test_symbolic_backend_assembles_decomposed_parts(self) -> None:
        scene = {
            "scene_id": "assemble_scene",
            "env_id": "assemble_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "pen",
                    "name": "pen",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE", "DECOMPOSABLE"],
                    "states": ["DECOMPOSED"],
                },
                {"id": "pen_body", "name": "pen body", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"], "part_of": "pen"},
                {"id": "pen_cap", "name": "pen cap", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"], "part_of": "pen"},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "pen_body", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_cap", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_body", "to": "table", "relation": "ON"},
                {"from": "pen_cap", "to": "box", "relation": "INSIDE"},
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "box", "to": "pen_cap", "relation": "OCCLUDES"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="assemble_scene")
        task = TaskRecord(
            task_id="assemble_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manual_ready_goal",
            task="Assemble the pen.",
            task_completion_criterion={
                "and": [
                    {"predicate": "ASSEMBLED", "args": ["pen"]},
                    {"predicate": "ATTACHED", "args": ["pen_cap", "pen_body"]},
                ]
            },
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        visible = {item["id"] for item in backend.observe()["visible_nodes"]}
        self.assertNotIn("pen", visible)
        self.assertIn("pen_body", visible)
        self.assertNotIn("pen_cap", visible)
        self.assertEqual(backend.step(ParsedAction("open", ["box"]))["status"], "success")
        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))
        self.assertIn({"name": "attach", "node_ids": ["pen_body", "pen_cap"], "object": "pen_body", "target": "pen_cap"}, payload["valid_actions"])
        self.assertFalse(backend.success())
        event = backend.step(ParsedAction("attach", ["pen_cap", "pen_body"]))

        self.assertEqual(event["status"], "success")
        self.assertEqual(event["action_model"]["name"], "attach")
        self.assertTrue(backend.success())
        snapshot = backend.snapshot()
        self.assertTrue(snapshot["nodes"]["pen"]["assembled"])
        self.assertEqual(snapshot["nodes"]["pen_cap"]["attached_to"], "pen_body")
        self.assertEqual(snapshot["nodes"]["pen_cap"]["location"], {"relation": None, "target": None})
        self.assertEqual(snapshot["nodes"]["pen_body"]["location"], {"relation": None, "target": None})
        active_edges = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in snapshot["active_occlusion_edges"]
        }
        self.assertNotIn(("box", "pen_cap", "OCCLUDES"), active_edges)
        visible_edges = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in backend.observe()["visible_edges"]
        }
        self.assertNotIn(("pen_cap", "box", "INSIDE"), visible_edges)
        self.assertNotIn(("pen_body", "table", "ON"), visible_edges)
        self.assertNotIn(("box", "pen_cap", "OCCLUDES"), visible_edges)
        self.assertIn(("pen_cap", "pen", "PART_OF"), visible_edges)
        self.assertIn(("pen_cap", "pen_body", "ATTACHED"), visible_edges)
        self.assertIn("pen_body", {item["id"] for item in backend.observe()["visible_nodes"]})

        self.assertEqual(backend.step(ParsedAction("grab", ["pen"]))["status"], "success")
        self.assertEqual(backend.step(ParsedAction("putin", ["pen", "box"]))["status"], "success")
        self.assertEqual(backend.step(ParsedAction("close", ["box"]))["status"], "success")
        visible_after_close = {item["id"] for item in backend.observe()["visible_nodes"]}
        self.assertNotIn("pen", visible_after_close)
        self.assertNotIn("pen_body", visible_after_close)
        self.assertNotIn("pen_cap", visible_after_close)

    def test_static_parts_are_not_attachable(self) -> None:
        scene = {
            "scene_id": "static_parts_scene",
            "env_id": "static_parts_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "drawer",
                    "name": "drawer",
                    "category": "furniture",
                    "properties": ["STORAGE_UNIT", "SURFACES", "DECOMPOSABLE", "STATIC"],
                    "states": ["DECOMPOSED"],
                },
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                    "part_of": "drawer",
                },
                {
                    "id": "drawer_middle",
                    "name": "drawer middle",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                    "part_of": "drawer",
                },
            ],
            "edges": [
                {"from": "drawer_top", "to": "drawer", "relation": "PART_OF"},
                {"from": "drawer_middle", "to": "drawer", "relation": "PART_OF"},
                {"from": "drawer_top", "to": "table", "relation": "BENEATH"},
                {"from": "drawer_middle", "to": "table", "relation": "BENEATH"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="static_parts_scene")
        task = TaskRecord(
            task_id="static_parts_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manual_ready_goal",
            task="Use the drawer.",
            task_completion_criterion="",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))
        attach_actions = [action for action in payload["valid_actions"] if action["name"] == "attach"]

        self.assertEqual(attach_actions, [])
        event = backend.step(ParsedAction("attach", ["drawer_top", "drawer_middle"]))
        self.assertEqual(event["status"], "failure")
        self.assertEqual(event["failure_type"], "not_attachable")

    def test_symbolic_backend_assemble_parent_action(self) -> None:
        scene = {
            "scene_id": "assemble_parent_scene",
            "env_id": "assemble_parent_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "pen",
                    "name": "pen",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE", "DECOMPOSABLE"],
                    "states": ["CAPPED"],
                },
                {"id": "pen_body", "name": "pen body", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"], "part_of": "pen"},
                {"id": "pen_cap", "name": "pen cap", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"], "part_of": "pen"},
            ],
            "edges": [
                {"from": "pen_body", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_cap", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_body", "to": "table", "relation": "ON"},
                {"from": "pen_cap", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="assemble_parent_scene")
        task = TaskRecord(
            task_id="assemble_parent_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manual_ready_goal",
            task="Assemble the pen.",
            task_completion_criterion={
                "and": [
                    {"predicate": "ASSEMBLED", "args": ["pen"]},
                    {"predicate": "ATTACHED", "args": ["pen_cap", "pen"]},
                    {"predicate": "ATTACHED", "args": ["pen_body", "pen"]},
                ]
            },
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        self.assertFalse(backend.success())
        event = backend.step(ParsedAction("assemble", ["pen"]))

        self.assertEqual(event["status"], "success")
        self.assertEqual(event["action_model"]["name"], "assemble")
        self.assertTrue(backend.success())
        snapshot = backend.snapshot()
        self.assertTrue(snapshot["nodes"]["pen"]["assembled"])
        self.assertEqual(snapshot["nodes"]["pen_cap"]["attached_to"], "pen")

    def test_structured_goal_supports_compound_predicates(self) -> None:
        scene = {
            "scene_id": "structured_goal_scene",
            "env_id": "structured_goal_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN"],
                    "states": ["CLOSED"],
                },
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "apple", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="structured_goal_scene")
        task = TaskRecord(
            task_id="structured_goal_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Organize the apple with the box secured.",
            task_completion_criterion={
                "and": [
                    ["INSIDE", "apple", "box"],
                    {"predicate": "CLOSED", "args": ["box"]},
                ]
            },
            ground_truth_plan=[
                "<char0> [open] <box> (box)",
                "<char0> [grab] <apple> (apple)",
                "<char0> [putin] <apple> (apple) <box> (box)",
                "<char0> [close] <box> (box)",
            ],
            objects={},
            metadata={},
        )

        episode = SymbolicHarness(graph, task).run()

        self.assertTrue(episode["success"])
        self.assertTrue(episode["metrics"]["goal"]["success"])
        self.assertEqual(episode["metrics"]["goal"]["predicates"], ["INSIDE", "CLOSED"])
        self.assertEqual(len(episode["metrics"]["goal"]["checks"]), 2)
        self.assertEqual(
            episode["final_state"]["nodes"]["apple"]["location"],
            {"relation": "INSIDE", "target": "box"},
        )
        self.assertFalse(episode["final_state"]["nodes"]["box"]["open"])
        putin_step = next(step for step in episode["trajectory"] if step["action"]["name"] == "putin")
        self.assertEqual(putin_step["event"]["action_model"]["name"], "putin")
        self.assertTrue(putin_step["event"]["action_model"]["preconditions"])
        self.assertTrue(putin_step["event"]["action_model"]["effects"])

    def test_pressable_object_supports_press_action_and_pressed_goal(self) -> None:
        scene = {
            "scene_id": "press_button_scene",
            "env_id": "press_button_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "button", "name": "button", "category": "object", "properties": ["PRESSABLE"]},
            ],
            "edges": [
                {"from": "button", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="press_button_scene")
        task = TaskRecord(
            task_id="press_button_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Press the button.",
            task_completion_criterion="(PRESSED, button)",
            ground_truth_plan=["<char0> [press] <button> (button)"],
            objects={},
            metadata={},
        )

        initial_observation = SymbolicBackend(graph, task).observe()
        teacher_payload = json.loads(_teacher_user_prompt(task, initial_observation, []))
        self.assertIn({"name": "press", "node_ids": ["button"], "object": "button"}, teacher_payload["valid_actions"])
        self.assertIn({"name": "stop", "node_ids": []}, teacher_payload["valid_actions"])
        self.assertEqual(teacher_payload["action_catalog"]["press"]["parameters"], ["object"])
        self.assertEqual(teacher_payload["action_catalog"]["stop"]["parameters"], [])

        episode = SymbolicHarness(graph, task).run()

        self.assertTrue(episode["success"])
        self.assertTrue(episode["final_state"]["nodes"]["button"]["pressed"])
        self.assertEqual(episode["metrics"]["goal"]["predicates"], ["PRESSED"])
        self.assertEqual(episode["trajectory"][0]["event"]["action_model"]["name"], "press")

    def test_pressed_times_goal_counts_successful_press_events(self) -> None:
        scene = {
            "scene_id": "press_button_count_scene",
            "env_id": "press_button_count_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "button", "name": "button", "category": "object", "properties": ["PRESSABLE"]},
            ],
            "edges": [
                {"from": "button", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="press_button_count_scene")
        plan = ["<char0> [press] <button> (button)" for _ in range(8)]
        task = TaskRecord(
            task_id="press_button_count_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Press the button eight times.",
            task_completion_criterion="(PRESSED_TIMES, button, 8)",
            ground_truth_plan=plan,
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        for _ in range(7):
            self.assertEqual(backend.step(ParsedAction("press", ["button"]))["status"], "success")
            self.assertFalse(backend.success())
        self.assertEqual(backend.step(ParsedAction("press", ["button"]))["status"], "success")
        self.assertTrue(backend.success())

        episode = SymbolicHarness(graph, task).run()

        self.assertTrue(episode["success"])
        self.assertEqual(len(episode["trajectory"]), 8)
        self.assertEqual(episode["metrics"]["goal"]["predicates"], ["PRESSED_TIMES"])
        self.assertTrue(episode["metrics"]["goal"]["checks"][0]["success"])

    def test_pressed_sequence_goal_checks_successful_press_order(self) -> None:
        scene = {
            "scene_id": "press_button_sequence_scene",
            "env_id": "press_button_sequence_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "blue", "name": "blue", "category": "object", "properties": ["PRESSABLE"]},
                {"id": "red", "name": "red", "category": "object", "properties": ["PRESSABLE"]},
            ],
            "edges": [
                {"from": "blue", "to": "table", "relation": "ON"},
                {"from": "red", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="press_button_sequence_scene")
        task = TaskRecord(
            task_id="press_button_sequence_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Press blue, then red, then blue.",
            task_completion_criterion={"and": [{"predicate": "PRESSED_SEQUENCE", "args": [["blue", "red", "blue"]]}]},
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)

        self.assertEqual(backend.step(ParsedAction("press", ["blue"]))["status"], "success")
        self.assertFalse(backend.success())
        self.assertEqual(backend.step(ParsedAction("press", ["red"]))["status"], "success")
        self.assertFalse(backend.success())
        self.assertEqual(backend.step(ParsedAction("press", ["blue"]))["status"], "success")
        self.assertTrue(backend.success())

        wrong_backend = SymbolicBackend(graph, task)
        self.assertEqual(wrong_backend.step(ParsedAction("press", ["red"]))["status"], "success")
        self.assertEqual(wrong_backend.step(ParsedAction("press", ["blue"]))["status"], "success")
        self.assertEqual(wrong_backend.step(ParsedAction("press", ["blue"]))["status"], "success")
        self.assertFalse(wrong_backend.success())

    def test_structured_goal_supports_negated_occlusion(self) -> None:
        scene = {
            "scene_id": "negated_goal_scene",
            "env_id": "negated_goal_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "book",
                    "name": "book",
                    "category": "object",
                    "properties": ["OCCLUDER", "MOVABLE", "GRABBABLE"],
                },
                {"id": "paper", "name": "paper", "category": "object", "properties": ["MOVABLE"]},
            ],
            "edges": [
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "paper", "relation": "OCCLUDES"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="negated_goal_scene")
        task = TaskRecord(
            task_id="negated_goal_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Make the paper accessible.",
            task_completion_criterion={"not": ["OCCLUDES", "book", "paper"]},
            ground_truth_plan=["<char0> [move_aside] <book> (book)"],
            objects={},
            metadata={},
        )

        episode = SymbolicHarness(graph, task).run()

        self.assertTrue(episode["success"])
        self.assertTrue(episode["metrics"]["goal"]["success"])
        self.assertEqual(episode["metrics"]["goal"]["predicates"], ["OCCLUDES"])
        self.assertTrue(episode["final_state"]["nodes"]["paper"]["visible"])
        move_step = episode["trajectory"][0]
        self.assertEqual(move_step["event"]["action_model"]["name"], "move_aside")

    def test_movable_occluder_is_revealed_only_by_move_aside(self) -> None:
        scene = {
            "scene_id": "movable_occluder_resolution_scene",
            "env_id": "movable_occluder_resolution_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["OCCLUDER", "MOVABLE", "GRABBABLE"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "paper", "relation": "OCCLUDES"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="movable_occluder_resolution_scene")
        task = TaskRecord(
            task_id="movable_occluder_resolution_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Reveal the paper.",
            task_completion_criterion="(VISIBLE, paper)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        backend = SymbolicBackend(graph, task)
        initial_payload = json.loads(_teacher_user_prompt(task, backend.observe(), []))

        self.assertIn(
            {"name": "move_aside", "node_ids": ["book"], "object": "book", "effect_hint": "may_reveal_hidden"},
            initial_payload["valid_actions"],
        )
        self.assertNotIn(
            {"name": "clear", "node_ids": ["book"], "object": "book", "effect_hint": "may_reveal_hidden"},
            initial_payload["valid_actions"],
        )
        self.assertEqual(backend.step(ParsedAction("clear", ["book"]))["failure_type"], "unsupported_resolution")
        self.assertNotIn("paper", {item["id"] for item in backend.observe()["visible_nodes"]})
        self.assertEqual(backend.step(ParsedAction("grab", ["book"]))["status"], "success")
        self.assertNotIn("paper", {item["id"] for item in backend.observe()["visible_nodes"]})
        self.assertEqual(backend.step(ParsedAction("move_aside", ["book"]))["status"], "success")
        self.assertIn("paper", {item["id"] for item in backend.observe()["visible_nodes"]})

    def test_failure_injection_excludes_perception_actions(self) -> None:
        all_actions = FailureInjectionConfig(mode="all")
        for action in ("look", "observe", "inspect", "clear", "recover", "stop"):
            self.assertFalse(all_actions.allows(action))
            self.assertFalse(FailureInjectionConfig(mode="all", actions=(action,)).allows(action))

        self.assertTrue(all_actions.allows("grab"))
        self.assertTrue(all_actions.allows("move_aside"))

    def test_teacher_valid_actions_allow_recover_or_stop_after_failure(self) -> None:
        scene = {
            "scene_id": "recover_priority_scene",
            "env_id": "recover_priority_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="recover_priority_scene")
        task = TaskRecord(
            task_id="recover_priority_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manipulation",
            task="Put apple on the table.",
            task_completion_criterion="(ON, apple, table)",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        history = [
            {
                "step": 1,
                "action": {"name": "failed_grab", "base_name": "grab", "node_ids": ["apple"]},
                "event": {"status": "failure", "failure_type": "injected"},
                "success_after_step": False,
            }
        ]

        payload = json.loads(_teacher_user_prompt(task, SymbolicBackend(graph, task).observe(), history))

        self.assertEqual(
            payload["valid_actions"],
            [{"name": "recover", "node_ids": []}, {"name": "stop", "node_ids": []}],
        )
        self.assertEqual(payload["action_catalog"]["recover"]["parameters"], [])
        self.assertFalse(payload["action_catalog"]["recover"]["failure_injectable"])
        self.assertEqual(payload["action_catalog"]["stop"]["parameters"], [])
        self.assertFalse(payload["action_catalog"]["stop"]["failure_injectable"])

        injected_history = [
            {
                "step": 1,
                "action": {"name": "failed_putin", "base_name": "putin", "node_ids": ["apple"]},
                "requested_action": {"name": "putin", "base_name": "putin", "node_ids": ["apple", "box"]},
                "event": {"status": "failure", "failure_type": "injected"},
                "success_after_step": False,
            }
        ]

        injected_payload = json.loads(_teacher_user_prompt(task, SymbolicBackend(graph, task).observe(), injected_history))

        self.assertEqual(
            injected_payload["valid_actions"],
            [{"name": "recover", "node_ids": []}, {"name": "stop", "node_ids": []}],
        )

        precondition_failure_history = [
            {
                "step": 1,
                "action": {"name": "grab", "base_name": "grab", "node_ids": ["apple"]},
                "event": {"status": "failure", "failure_type": "not_reachable", "injected": False},
                "success_after_step": False,
            }
        ]

        precondition_failure_payload = json.loads(
            _teacher_user_prompt(task, SymbolicBackend(graph, task).observe(), precondition_failure_history)
        )
        precondition_failure_action_names = {
            action["name"] for action in precondition_failure_payload["valid_actions"]
        }

        self.assertNotIn("recover", precondition_failure_action_names)
        self.assertIn("grab", precondition_failure_action_names)
        self.assertIn("stop", precondition_failure_action_names)

        recovered_history = [
            *injected_history,
            {
                "step": 2,
                "action": {"name": "recover", "base_name": "recover", "node_ids": []},
                "requested_action": {"name": "recover", "base_name": "recover", "node_ids": []},
                "event": {"status": "success", "action": "recover", "node_ids": [], "recovered": True},
                "success_after_step": False,
            },
        ]

        recovered_payload = json.loads(_teacher_user_prompt(task, SymbolicBackend(graph, task).observe(), recovered_history))

        self.assertEqual(
            recovered_payload["valid_actions"],
            [
                {
                    "name": "putin",
                    "node_ids": ["apple", "box"],
                    "object": "apple",
                    "target": "box",
                }
            ],
        )

    def test_teacher_mode_runs_closed_loop_with_scripted_policy(self) -> None:
        scene = {
            "scene_id": "teacher_scene",
            "env_id": "teacher_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="teacher_scene")
        task = TaskRecord(
            task_id="teacher_task",
            scene_id="teacher_scene",
            env_id="teacher_scene",
            layout="tabletop",
            arms="single",
            task_type="manipulation",
            task="Put apple into box.",
            task_completion_criterion="(INSIDE, apple, box)",
            ground_truth_plan=[],
            objects={"object": "apple", "target": "box", "relation": "INSIDE"},
            metadata={},
        )
        policy = ScriptedTeacherPolicy(
            [
                {"reason": "take the object", "action": {"name": "grab", "object": "apple"}},
                {"reason": "place it in the container", "action": {"name": "putin", "object": "apple", "target": "box"}},
                {"reason": "goal is satisfied", "action": {"name": "stop"}},
            ]
        )

        episode = SymbolicHarness(graph, task, mode="teacher", max_steps=5, teacher_policy=policy).run()

        self.assertTrue(episode["success"])
        self.assertEqual(episode["mode"], "teacher")
        self.assertEqual([step["action"]["name"] for step in episode["trajectory"]], ["grab", "putin", "stop"])
        self.assertEqual(episode["trajectory"][0]["action"]["name"], "grab")
        self.assertTrue(episode["trajectory"][-1]["event"]["stopped"])
        self.assertTrue(episode["trajectory"][-1]["success_after_step"])
        self.assertIn("teacher_response", episode["trajectory"][0])
        self.assertNotIn("source_plan_step", episode["trajectory"][0])
        self.assertEqual(
            episode["final_state"]["nodes"]["apple"]["location"],
            {"relation": "INSIDE", "target": "box"},
        )

    def test_teacher_stop_before_goal_is_failure(self) -> None:
        scene = {
            "scene_id": "teacher_stop_failure_scene",
            "env_id": "teacher_stop_failure_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="teacher_stop_failure_scene")
        task = TaskRecord(
            task_id="teacher_stop_failure_task",
            scene_id="teacher_stop_failure_scene",
            env_id="teacher_stop_failure_scene",
            layout="tabletop",
            arms="single",
            task_type="manipulation",
            task="Put apple into box.",
            task_completion_criterion="(INSIDE, apple, box)",
            ground_truth_plan=[],
            objects={"object": "apple", "target": "box", "relation": "INSIDE"},
            metadata={},
        )
        policy = ScriptedTeacherPolicy([{"reason": "end early", "action": {"name": "stop"}}])

        episode = SymbolicHarness(graph, task, mode="teacher", max_steps=5, teacher_policy=policy).run()

        self.assertFalse(episode["success"])
        self.assertEqual([step["action"]["name"] for step in episode["trajectory"]], ["stop"])
        self.assertTrue(episode["trajectory"][0]["event"]["stopped"])
        self.assertFalse(episode["trajectory"][0]["success_after_step"])

    def test_teacher_mode_can_inject_failure_and_recover(self) -> None:
        scene = {
            "scene_id": "teacher_failure_scene",
            "env_id": "teacher_failure_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        graph = ViewGraph.from_dict(scene, fallback_scene_id="teacher_failure_scene")
        task = TaskRecord(
            task_id="teacher_failure_task",
            scene_id="teacher_failure_scene",
            env_id="teacher_failure_scene",
            layout="tabletop",
            arms="single",
            task_type="manipulation",
            task="Put apple into box.",
            task_completion_criterion="(INSIDE, apple, box)",
            ground_truth_plan=[],
            objects={"object": "apple", "target": "box", "relation": "INSIDE"},
            metadata={},
        )
        policy = ScriptedTeacherPolicy(
            [
                {"reason": "try to take the object", "action": {"name": "grab", "object": "apple"}},
                {"reason": "recover from the failed grab", "action": {"name": "recover"}},
                {"reason": "retry the object", "action": {"name": "grab", "object": "apple"}},
                {"reason": "place it in the container", "action": {"name": "putin", "object": "apple", "target": "box"}},
                {"reason": "goal is satisfied", "action": {"name": "stop"}},
            ]
        )

        episode = SymbolicHarness(
            graph,
            task,
            mode="teacher",
            max_steps=6,
            teacher_policy=policy,
            failure_injection=FailureInjectionConfig(mode="once", actions=("grab",), seed=3),
        ).run()

        self.assertTrue(episode["success"])
        self.assertEqual(
            [step["action"]["name"] for step in episode["trajectory"]],
            ["failed_grab", "recover", "grab", "putin", "stop"],
        )
        self.assertEqual(episode["trajectory"][0]["event"]["status"], "failure")
        self.assertEqual(episode["trajectory"][0]["event"]["failure_type"], "injected")
        self.assertEqual(episode["trajectory"][0]["requested_action"]["name"], "grab")
        self.assertEqual(episode["trajectory"][1]["action"]["node_ids"], [])
        self.assertEqual(episode["trajectory"][1]["event"]["action"], "recover")
        self.assertTrue(episode["trajectory"][1]["event"]["recovered"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["failure_observed"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["recovered_after_failure"])
        self.assertTrue(episode["metrics"]["failure_recovery"]["retried_failed_action"])

    def test_cli_collects_teacher_trajectory_from_mock_api(self) -> None:
        scene = {
            "scene_id": "teacher_cli_scene",
            "env_id": "teacher_cli_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"]},
                {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "apple", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        task = {
            "task_id": "teacher_cli_task",
            "scene_id": "teacher_cli_scene",
            "env_id": "teacher_cli_scene",
            "layout": "tabletop",
            "arms": "single",
            "task_type": "manipulation",
            "task": "Put apple into box.",
            "task_completion_criterion": "(INSIDE, apple, box)",
            "ground_truth_plan": [],
            "objects": {"object": "apple", "target": "box", "relation": "INSIDE"},
            "settings": [],
            "metadata": {},
        }

        responses = [
            {"reason": "grab apple", "action": {"name": "grab", "object": "apple"}},
            {"reason": "put apple into box", "action": {"name": "putin", "object": "apple", "target": "box"}},
            {"reason": "goal is satisfied", "action": {"name": "stop"}},
        ]
        create_calls = []

        def fake_create(**kwargs):
            create_calls.append(kwargs)
            payload = responses[len(create_calls) - 1]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(payload)),
                    )
                ]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=fake_create),
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            task_path = Path(tmpdir) / "tasks.jsonl"
            trajectory_path = Path(tmpdir) / "trajectories.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            task_path.write_text(json.dumps(task) + "\n", encoding="utf-8")

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}), patch(
                "auto_embodied_task.harness.OpenAI",
                return_value=fake_client,
            ) as openai_cls:
                code = main(
                    [
                        "collect-trajectories",
                        "--view-graph",
                        str(graph_path),
                        "--tasks",
                        str(task_path),
                        "--output",
                        str(trajectory_path),
                        "--mode",
                        "teacher",
                        "--teacher-provider",
                        "qwen",
                        "--teacher-model",
                        "qwen3.6-plus",
                        "--teacher-api-key-env",
                        "DASHSCOPE_API_KEY",
                        "--max-steps",
                        "4",
                    ]
                )
            generated = sorted(Path(tmpdir).glob("trajectories_*.jsonl"))
            episodes = [json.loads(line) for line in generated[0].read_text(encoding="utf-8").splitlines()]

        self.assertEqual(code, 0)
        openai_cls.assert_called_once()
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(openai_cls.call_args.kwargs["timeout"], 60)
        self.assertEqual(len(create_calls), 3)
        self.assertEqual(create_calls[0]["model"], "qwen3.6-plus")
        self.assertEqual(create_calls[0]["extra_body"], {"enable_thinking": False})
        self.assertEqual(create_calls[0]["response_format"], {"type": "json_object"})
        teacher_payload = json.loads(create_calls[0]["messages"][1]["content"])
        self.assertNotIn("map_layout", teacher_payload["current_observation"])
        self.assertIn("visible_nodes", teacher_payload["current_observation"])
        self.assertIn("allowed_node_ids", teacher_payload)
        self.assertIn("apple", teacher_payload["allowed_node_ids"])
        self.assertIn("action_catalog", teacher_payload)
        self.assertIn("valid_actions", teacher_payload)
        self.assertIn({"name": "grab", "node_ids": ["apple"], "object": "apple"}, teacher_payload["valid_actions"])
        self.assertIn("Choose one action object", teacher_payload["action_constraints"][0])
        self.assertEqual(len(episodes), 1)
        episode = episodes[0]
        self.assertTrue(episode["success"])
        self.assertEqual(episode["mode"], "teacher")
        self.assertEqual([step["action"]["name"] for step in episode["trajectory"]], ["grab", "putin", "stop"])

    def test_trajectory_server_builds_frames_from_visible_observations(self) -> None:
        episode = {
            "episode_id": "episode_1",
            "scene_id": "scene_1",
            "env_id": "scene_1",
            "layout": "tabletop",
            "mode": "teacher",
            "task": "Find the apple.",
            "success": True,
            "initial_view_graph": {
                "scene_id": "scene_1",
                "env_id": "scene_1",
                "layout": "tabletop",
                "nodes": [
                    {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"], "states": []},
                    {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS"], "states": []},
                    {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE"], "states": []},
                ],
                "edges": [
                    {"from": "apple", "to": "box", "relation": "INSIDE"},
                    {"from": "box", "to": "apple", "relation": "OCCLUDES"},
                ],
            },
            "initial_observation": {
                "visible_nodes": [
                    {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"], "states": []}
                ],
                "visible_edges": [],
                "held_objects": [],
            },
            "trajectory": [
                {
                    "step": 1,
                    "action": {"name": "inspect", "base_name": "inspect", "node_ids": ["box"]},
                    "event": {"status": "success", "action": "inspect"},
                    "post_observation": {
                        "visible_nodes": [
                            {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"], "states": []},
                            {
                                "id": "apple",
                                "name": "apple",
                                "category": "food",
                                "properties": ["GRABBABLE", "MOVABLE"],
                                "states": [],
                                "reachable": True,
                            },
                        ],
                        "visible_edges": [{"from": "apple", "to": "table", "relation": "ON"}],
                        "held_objects": [],
                    },
                    "new_visible_nodes": [{"id": "apple", "name": "apple"}],
                    "success_after_step": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectories.jsonl"
            path.write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")

            payload = trajectory_replay_payload(path)

        self.assertEqual(payload["episode_count"], 1)
        replay_episode = payload["episodes"][0]
        initial_graph = replay_episode["initial_view_graph"]
        self.assertFalse(initial_graph["limited"])
        self.assertEqual(initial_graph["source"], "initial_view_graph")
        self.assertEqual({node["id"] for node in initial_graph["nodes"]}, {"table", "box", "apple"})
        self.assertIn({"from": "box", "to": "apple", "relation": "OCCLUDES"}, initial_graph["edges"])
        frames = replay_episode["frames"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["kind"], "initial")
        self.assertEqual([node["id"] for node in frames[0]["view_graph"]["nodes"]], ["table"])
        self.assertEqual(frames[1]["kind"], "post_action")
        self.assertEqual({node["id"] for node in frames[1]["view_graph"]["nodes"]}, {"table", "apple"})
        self.assertEqual(frames[1]["view_graph"]["edges"], [{"from": "apple", "to": "table", "relation": "ON"}])
        self.assertEqual(frames[1]["new_visible_nodes"], [{"id": "apple", "name": "apple"}])

    def test_trajectory_server_endpoint_serves_replay_payload(self) -> None:
        episode = {
            "episode_id": "server_episode",
            "scene_id": "server_scene",
            "initial_observation": {"visible_nodes": [], "visible_edges": []},
            "trajectory": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectories.jsonl"
            path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
            server = ThreadingHTTPServer(("127.0.0.1", 0), _TrajectoryAppHandler)
            server.trajectory_dir = Path(tmpdir)
            server.trajectory_path = path
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                files_url = f"http://127.0.0.1:{server.server_port}/api/trajectory-files"
                with request.urlopen(files_url, timeout=5) as response:
                    files_data = json.loads(response.read().decode("utf-8"))
                trajectory_url = f"http://127.0.0.1:{server.server_port}/api/trajectories?file=trajectories.jsonl"
                with request.urlopen(trajectory_url, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(files_data["selected"], "trajectories.jsonl")
        self.assertEqual([item["name"] for item in files_data["files"]], ["trajectories.jsonl"])
        self.assertEqual(data["episode_count"], 1)
        self.assertEqual(data["trajectory_file"], "trajectories.jsonl")
        self.assertEqual(data["episodes"][0]["episode_id"], "server_episode")
        self.assertEqual(data["episodes"][0]["frame_count"], 1)

    def test_view_graph_server_create_endpoint_calls_backend(self) -> None:
        package = {
            "view_graph": {
                "scene_id": "server_scene",
                "env_id": "server_scene",
                "layout": "tabletop",
                "robot": {"arms": "single"},
                "nodes": [
                    {"id": "desk", "name": "desk", "category": "surface", "properties": ["SURFACES"]},
                    {"id": "apple", "name": "apple", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                ],
                "edges": [{"from": "apple", "to": "desk", "relation": "ON"}],
            }
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ViewGraphAppHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch("auto_embodied_task.view_graph_server.synthesize_task_view_graph", return_value=package) as synthesize:
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/api/create-view-graph"
                body = json.dumps(
                    {
                        "materials_text": "apple\nbox\n",
                        "material_properties_text": '{"materials": {"apple": {"properties": ["GRABBABLE", "MOVABLE"]}}}',
                        "scene": "office desktop",
                        "task_hint": "整理桌面",
                        "layout": "tabletop",
                        "arms": "single",
                        "provider": "qwen",
                        "model": "qwen3.6-plus",
                        "api_key": "sk-test",
                        "api_key_env": "DASHSCOPE_API_KEY",
                        "timeout_seconds": "45",
                    }
                ).encode("utf-8")
                req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(data["view_graph"]["scene_id"], "server_scene")
        synthesize.assert_called_once()
        config = synthesize.call_args.args[0]
        self.assertEqual(config.materials, ("apple", "box"))
        self.assertEqual(config.scene, "office desktop")
        self.assertEqual(config.task_hint, "整理桌面")
        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.api_key_env, "DASHSCOPE_API_KEY")
        self.assertEqual(config.timeout_seconds, 45)
        self.assertFalse(config.enable_thinking)

    def test_view_graph_server_edit_endpoint_profiles_graph_samples(self) -> None:
        scene = {
            "scene_id": "server_profile_scene",
            "env_id": "server_profile_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE", "COPYABLE"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "server_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
            "temporal": {"enabled": True, "causal_chain_steps": 2},
            "memory": {"enabled": True, "num_memory_items": 1, "num_similar_distractors": 1},
            "failure_recovery": {"enabled": True, "num_failures": 1},
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ViewGraphAppHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/edit-view-graph"
            body = json.dumps(
                {"view_graph": scene, "profile": profile, "num_samples": 2, "seed": 11}
            ).encode("utf-8")
            req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(data["num_samples"], 2)
        self.assertEqual(len(data["view_graphs"]), 2)
        self.assertEqual(data["view_graph"]["scene_id"], data["view_graphs"][0]["scene_id"])
        self.assertEqual(len({item["scene_id"] for item in data["view_graphs"]}), 2)
        metadata = data["view_graphs"][0]["metadata"]
        self.assertEqual(metadata["requested_constraint_profile"]["profile_id"], "server_profile")
        self.assertEqual(metadata["difficulty_tags"]["spatial"][0], "spatial.num_occluded_objects=1")
        self.assertEqual(set(metadata["difficulty_tags"]), {"spatial"})
        self.assertEqual(set(metadata["achieved_constraint_profile"]), {"spatial"})
        self.assertEqual(set(metadata["profile_constraints"]), {"spatial"})

    def test_view_graph_server_edit_endpoint_respects_placement_constraints(self) -> None:
        scene = {
            "scene_id": "server_constrained_profile_scene",
            "env_id": "server_constrained_profile_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "folder", "name": "folder", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "drawer_top", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "server_constrained_profile",
            "target_object": "book",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        constraints = {
            "nonexistent_edges": [
                {"from": "drawer_top", "to": "book", "relation": "OCCLUDES"},
            ]
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ViewGraphAppHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/api/edit-view-graph"
            body = json.dumps(
                {
                    "view_graph": scene,
                    "profile": profile,
                    "placement_edge_constraints": constraints,
                    "seed": 1,
                }
            ).encode("utf-8")
            req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in data["view_graph"]["edges"]
        }
        self.assertIn(("folder", "book", "OCCLUDES"), edge_triples)
        self.assertNotIn(("drawer_top", "book", "OCCLUDES"), edge_triples)
        self.assertEqual(data["placement_edge_constraints"]["forbidden_edges"], constraints["nonexistent_edges"])
        self.assertEqual(
            data["view_graph"]["metadata"]["placement_edge_constraints"]["forbidden_edges"],
            constraints["nonexistent_edges"],
        )

    def test_view_graph_server_treats_raw_key_in_env_field_as_api_key(self) -> None:
        config = _config_from_payload(
            {
                "materials_text": "apple\n",
                "material_properties_text": "{}",
                "scene": "office desktop",
                "layout": "tabletop",
                "arms": "single",
                "api_key_env": "sk-raw-key",
            }
        )

        self.assertEqual(config.api_key, "sk-raw-key")
        self.assertIsNone(config.api_key_env)

    def test_view_graph_server_html_supports_file_loading_and_raw_api_key(self) -> None:
        html = _render_app_html()

        self.assertIn('id="materials-file"', html)
        self.assertIn('id="material-properties-file"', html)
        self.assertIn('id="view-graph-file"', html)
        self.assertIn('id="import-view-graph"', html)
        self.assertIn("Import Existing View Graph", html)
        self.assertIn("Import View Graph", html)
        self.assertIn("Profile Edit", html)
        self.assertIn('id="apply-profile-btn"', html)
        self.assertIn('id="profile-num-samples"', html)
        self.assertIn('id="profile-placement-constraints"', html)
        self.assertIn('id="placement-constraints-file"', html)
        self.assertIn('type="range"', html)
        self.assertIn('id="spatial-decomposed-parents"', html)
        self.assertNotIn('id="temporal-enabled"', html)
        self.assertNotIn('id="memory-enabled"', html)
        self.assertNotIn('id="failure-enabled"', html)
        self.assertIn('id="download-profiled-btn"', html)
        self.assertIn('/api/edit-view-graph', html)
        self.assertIn("difficultyTagText", html)
        self.assertIn("profiledSamples", html)
        self.assertIn("profileBaseGraph", html)
        self.assertIn("profileSourceGraph", html)
        self.assertIn("from base graph", html)
        self.assertIn("parseOptionalJsonTextarea", html)
        self.assertIn("placement_edge_constraints", html)
        self.assertIn("parseExistingViewGraphText", html)
        self.assertIn("setLoadedGraph", html)
        self.assertIn('id="api-key"', html)
        self.assertIn('id="enable-thinking"', html)
        self.assertNotIn('id="enable-thinking" type="checkbox" checked', html)
        self.assertIn('id="map-view-btn"', html)
        self.assertIn("buildMapLayout", html)
        self.assertIn("renderMapLegend", html)
        self.assertIn("mapNodeStyle", html)
        self.assertIn("buildParentById", html)
        self.assertIn("applyNestedPositions", html)
        self.assertIn("placeNodesInGrid", html)
        self.assertIn("relationOrientation", html)
        self.assertIn("relationLabelForParent", html)
        self.assertIn("occlusion-box", html)
        self.assertIn('"BENEATH"', html)
        self.assertIn("beneath", html)
        self.assertIn("orderNodesByRelations", html)
        self.assertIn("isMapRelation", html)
        self.assertNotIn("graph.edges.indexOf(edge)", html)
        self.assertNotIn("map-edge", html)
        self.assertNotIn("appendMapEdgeLabel", html)
        self.assertIn("Edit Graph", html)
        self.assertIn("Download JSONL", html)
        self.assertIn("Add Node", html)
        self.assertIn("Delete Node", html)
        self.assertIn("Add Edge", html)
        self.assertIn("Duplicate node name", html)
        self.assertIn("isNodeNameTaken", html)
        self.assertNotIn("Custom relation", html)
        self.assertNotIn('"PRESSES"', html)
        self.assertNotIn('"CLAMPS"', html)
        self.assertIn("No graph loaded. Create a graph first.", html)
        self.assertNotIn('scene_id: "empty"', html)

    def test_cli_rejects_removed_create_view_graph_command(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["create-view-graph", "--help"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_cli_profiles_view_graph_from_abstract_profile(self) -> None:
        profile = {
            "profile_id": "abstract_profile_test",
            "spatial": {
                "enabled": True,
                "num_occluded_objects": 2,
                "occlusion_depth": 2,
            },
            "temporal": {
                "enabled": True,
                "causal_chain_steps": 4,
            },
            "memory": {
                "enabled": True,
                "num_memory_items": 2,
                "num_similar_distractors": 2,
            },
            "failure_recovery": {
                "enabled": True,
                "num_failures": 2,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            scene = copy.deepcopy(SCENE)
            for node in scene["nodes"]:
                if node["id"] == "box":
                    node["properties"] = ["CONTAINERS", "CAN_OPEN", "OCCLUDER"]
                    node["states"] = ["CLOSED"]
                if node["id"] == "book":
                    node["properties"] = ["OCCLUDER", "MOVABLE"]
                if {"GRABBABLE", "MOVABLE"}.issubset(set(node.get("properties", []))):
                    node["properties"].append("COPYABLE")
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--num-samples",
                    "3",
                    "--seed",
                    "7",
                ]
            )
            payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            payload = payloads[0]
            graphs = load_view_graphs_jsonl(output_path)

        self.assertEqual(code, 0)
        self.assertEqual(len(graphs), 3)
        self.assertEqual(len({graph.scene_id for graph in graphs}), 3)
        self.assertEqual(len(payloads), 3)
        metadata = payload["metadata"]
        self.assertEqual(metadata["requested_constraint_profile"]["profile_id"], "abstract_profile_test")
        self.assertEqual(metadata["profile_sample_index"], 1)
        achieved = metadata["achieved_constraint_profile"]
        self.assertEqual(achieved["spatial"]["num_occluded_objects"], 2)
        self.assertGreaterEqual(achieved["spatial"]["occlusion_depth"], 2)
        occlusion = metadata["profile_constraints"]["spatial"]["occlusion"]
        self.assertEqual(len(occlusion["objects"]), 2)
        self.assertTrue(all(item["depth"] == 2 for item in occlusion["objects"]))
        self.assertTrue(all(len(item["layers"]) == 2 for item in occlusion["objects"]))
        self.assertEqual(set(achieved["spatial"]["occlusion_depths"]), {item["object"] for item in occlusion["objects"]})
        self.assertEqual(set(achieved), {"spatial"})
        self.assertEqual(set(metadata["profile_constraints"]), {"spatial"})
        self.assertIn("profile_constraints", metadata)
        self.assertEqual(metadata["difficulty_tags"]["spatial"][0], "spatial.num_occluded_objects=2")
        self.assertIn("spatial.occlusion_depth=2", metadata["difficulty_tags"]["spatial"])
        self.assertEqual(set(metadata["difficulty_tags"]), {"spatial"})
        self.assertTrue(metadata["graph_edits"])

    def test_profile_spatial_uses_only_existing_occluder_affordances(self) -> None:
        scene = {
            "scene_id": "plain_blocker_scene",
            "env_id": "plain_blocker_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "no_synthetic_occluder_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        self.assertEqual(payload["metadata"]["achieved_constraint_profile"]["spatial"]["num_occluded_objects"], 0)
        self.assertEqual(payload["metadata"]["achieved_constraint_profile"]["spatial"]["occlusion_depth"], 0)
        self.assertFalse(any(edge["relation"] == "OCCLUDES" for edge in payload["edges"]))
        book = next(node for node in payload["nodes"] if node["id"] == "book")
        self.assertNotIn("OCCLUDER", book.get("properties", []))

    def test_profile_spatial_requires_occluder_and_resolution_affordance(self) -> None:
        scene = {
            "scene_id": "occluder_affordance_scene",
            "env_id": "occluder_affordance_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {"id": "folder", "name": "folder", "category": "object", "properties": ["OCCLUDER"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN"]},
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "occluder_affordance_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        occlusion = payload["metadata"]["profile_constraints"]["spatial"]["occlusion"]["objects"][0]
        self.assertEqual(occlusion["layers"][0]["blocker"], "book")
        self.assertEqual(occlusion["layers"][0]["resolution_action"], "move_aside")
        edge_triples = {(edge["from"], edge["to"], edge["relation"]) for edge in payload["edges"]}
        self.assertIn(("book", "paper", "OCCLUDES"), edge_triples)
        self.assertNotIn(("folder", "paper", "OCCLUDES"), edge_triples)
        self.assertNotIn(("box", "paper", "OCCLUDES"), edge_triples)

    def test_profile_editor_ignores_non_spatial_dimensions(self) -> None:
        scene = {
            "scene_id": "memory_no_copy_scene",
            "env_id": "memory_no_copy_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
            ],
            "edges": [{"from": "paper", "to": "table", "relation": "ON"}],
        }
        profile = {
            "profile_id": "memory_no_copy_profile",
            "target_object": "paper",
            "spatial": {"enabled": False},
            "memory": {"enabled": True, "num_memory_items": 1, "num_similar_distractors": 2},
            "temporal": {"enabled": True, "causal_chain_steps": 3},
            "failure_recovery": {"enabled": True, "num_failures": 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        self.assertEqual(payload["metadata"]["achieved_constraint_profile"], {})
        self.assertEqual(payload["metadata"]["difficulty_tags"], {})
        self.assertEqual(payload["metadata"]["profile_constraints"], {})
        self.assertFalse(any(node.get("similar_to") == "paper" for node in payload["nodes"]))

    def test_profile_spatial_selection_excludes_parent_part_conflicts(self) -> None:
        scene = {
            "scene_id": "part_conflict_scene",
            "env_id": "part_conflict_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "pen", "name": "pen", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "pen_body",
                    "name": "pen body",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "part_of": "pen",
                },
                {
                    "id": "pen_cap",
                    "name": "pen cap",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "part_of": "pen",
                },
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "folder", "name": "folder", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {"id": "binder", "name": "binder", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
            ],
            "edges": [
                {"from": "pen", "to": "table", "relation": "ON"},
                {"from": "pen_body", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_cap", "to": "pen", "relation": "PART_OF"},
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "binder", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "part_conflict_profile",
            "target_object": "pen",
            "spatial": {"enabled": True, "num_occluded_objects": 2, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "3",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        objects = [
            item["object"]
            for item in payload["metadata"]["profile_constraints"]["spatial"]["occlusion"]["objects"]
        ]
        self.assertIn("pen", objects)
        self.assertNotIn("pen_body", objects)
        self.assertNotIn("pen_cap", objects)

    def test_profile_spatial_decomposes_only_decomposable_parents(self) -> None:
        scene = {
            "scene_id": "decompose_scene",
            "env_id": "decompose_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {
                    "id": "pen",
                    "name": "pen",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE", "DECOMPOSABLE"],
                    "states": ["CAPPED"],
                },
                {
                    "id": "pen_body",
                    "name": "pen body",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "part_of": "pen",
                },
                {
                    "id": "pen_cap",
                    "name": "pen cap",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "part_of": "pen",
                    "states": ["ATTACHED"],
                },
                {
                    "id": "marker",
                    "name": "marker",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "states": ["CAPPED"],
                },
                {
                    "id": "marker_cap",
                    "name": "marker cap",
                    "category": "tool",
                    "properties": ["GRABBABLE", "MOVABLE"],
                    "part_of": "marker",
                    "states": ["ATTACHED"],
                },
            ],
            "edges": [
                {"from": "pen", "to": "table", "relation": "ON"},
                {"from": "pen_body", "to": "pen", "relation": "PART_OF"},
                {"from": "pen_cap", "to": "pen", "relation": "PART_OF"},
                {"from": "marker", "to": "table", "relation": "ON"},
                {"from": "marker_cap", "to": "marker", "relation": "PART_OF"},
            ],
        }
        profile = {
            "profile_id": "decompose_profile",
            "target_object": "pen",
            "spatial": {
                "enabled": True,
                "num_occluded_objects": 0,
                "occlusion_depth": 0,
                "num_decomposed_parents": 2,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "4",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        edge_triples = {(edge["from"], edge["to"], edge["relation"]) for edge in payload["edges"]}
        self.assertNotIn(("pen", "table", "ON"), edge_triples)
        self.assertIn(("pen_body", "table", "ON"), edge_triples)
        self.assertIn(("pen_cap", "table", "ON"), edge_triples)
        self.assertIn(("pen_body", "pen", "PART_OF"), edge_triples)
        self.assertIn(("pen_cap", "pen", "PART_OF"), edge_triples)
        self.assertIn(("marker", "table", "ON"), edge_triples)
        self.assertNotIn(("marker_cap", "table", "ON"), edge_triples)
        metadata = payload["metadata"]
        self.assertEqual(metadata["profile_primary_object"], "pen_body")
        self.assertEqual(metadata["achieved_constraint_profile"]["spatial"]["num_decomposed_parents"], 1)
        self.assertEqual(metadata["achieved_constraint_profile"]["spatial"]["decomposed_parents"], ["pen"])
        self.assertIn("spatial.num_decomposed_parents=1", metadata["difficulty_tags"]["spatial"])
        decomposition = metadata["profile_constraints"]["spatial"]["decomposition"]
        self.assertEqual(decomposition["requested_num_decomposed_parents"], 2)
        self.assertEqual(decomposition["num_decomposed_parents"], 1)
        self.assertEqual(decomposition["parents"][0]["parent"], "pen")
        marker = next(node for node in payload["nodes"] if node["id"] == "marker")
        pen = next(node for node in payload["nodes"] if node["id"] == "pen")
        pen_cap = next(node for node in payload["nodes"] if node["id"] == "pen_cap")
        marker_cap = next(node for node in payload["nodes"] if node["id"] == "marker_cap")
        self.assertIn("DECOMPOSED", pen.get("states", []))
        self.assertNotIn("CAPPED", pen.get("states", []))
        self.assertNotIn("ATTACHED", pen_cap.get("states", []))
        self.assertIn("CAPPED", marker.get("states", []))
        self.assertIn("ATTACHED", marker_cap.get("states", []))
        self.assertNotIn("DECOMPOSABLE", marker.get("properties", []))
        self.assertTrue(decomposition["parents"][0]["removed_assembly_states"])
        self.assertTrue(any(edit["type"] == "decompose_parent" and edit["parent"] == "pen" for edit in metadata["graph_edits"]))

        graph = ViewGraph.from_dict(payload, fallback_scene_id="decompose_scene")
        task = TaskRecord(
            task_id="decompose_visibility_task",
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms="single",
            task_type="manual_ready_goal",
            task="Inspect decomposed visibility.",
            task_completion_criterion="",
            ground_truth_plan=[],
            objects={},
            metadata={},
        )
        visible = {item["id"] for item in SymbolicBackend(graph, task).observe()["visible_nodes"]}
        self.assertNotIn("pen", visible)
        self.assertIn("pen_body", visible)
        self.assertIn("pen_cap", visible)
        self.assertIn("marker", visible)

    def test_profile_spatial_decomposed_container_parts_can_occlude(self) -> None:
        scene = {
            "scene_id": "decompose_drawer_occlusion_scene",
            "env_id": "decompose_drawer_occlusion_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "drawer",
                    "name": "drawer",
                    "category": "furniture",
                    "properties": ["STORAGE_UNIT", "SURFACES", "DECOMPOSABLE"],
                },
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                    "part_of": "drawer",
                },
                {
                    "id": "drawer_middle",
                    "name": "drawer middle",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                    "part_of": "drawer",
                },
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "drawer", "to": "table", "relation": "ON"},
                {"from": "drawer_top", "to": "drawer", "relation": "PART_OF"},
                {"from": "drawer_middle", "to": "drawer", "relation": "PART_OF"},
            ],
        }
        profile = {
            "profile_id": "decompose_drawer_occlusion_profile",
            "target_object": "paper",
            "spatial": {
                "enabled": True,
                "num_occluded_objects": 1,
                "occlusion_depth": 1,
                "num_decomposed_parents": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "2",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        edge_triples = {(edge["from"], edge["to"], edge["relation"]) for edge in payload["edges"]}
        self.assertNotIn(("drawer", "table", "ON"), edge_triples)
        self.assertIn(("drawer_top", "table", "ON"), edge_triples)
        self.assertIn(("drawer_middle", "table", "ON"), edge_triples)
        occluders = [
            edge["from"]
            for edge in payload["edges"]
            if edge["to"] == "paper" and edge["relation"] == "OCCLUDES"
        ]
        self.assertEqual(len(occluders), 1)
        self.assertIn(occluders[0], {"drawer_top", "drawer_middle"})
        metadata = payload["metadata"]
        self.assertEqual(metadata["achieved_constraint_profile"]["spatial"]["num_decomposed_parents"], 1)
        self.assertEqual(metadata["achieved_constraint_profile"]["spatial"]["num_occluded_objects"], 1)

    def test_profile_editor_respects_placement_edge_constraints_for_occlusion(self) -> None:
        scene = {
            "scene_id": "constrained_occlusion_scene",
            "env_id": "constrained_occlusion_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "folder", "name": "folder", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "drawer_top", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "constrained_occlusion_profile",
            "target_object": "book",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        constraints = {
            "nonexistent_edges": [
                {"from": "drawer_top", "to": "book", "relation": "OCCLUDES"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            constraint_path = Path(tmpdir) / "placement_constraints.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            constraint_path.write_text(json.dumps(constraints), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--placement-edge-constraints",
                    str(constraint_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in payload["edges"]
        }
        self.assertIn(("folder", "book", "OCCLUDES"), edge_triples)
        self.assertNotIn(("drawer_top", "book", "OCCLUDES"), edge_triples)
        occlusion = payload["metadata"]["profile_constraints"]["spatial"]["occlusion"]
        self.assertEqual(occlusion["objects"][0]["layers"][0]["blocker"], "folder")

    def test_profile_editor_respects_container_max_items_for_occlusion(self) -> None:
        scene = {
            "scene_id": "capacity_occlusion_scene",
            "env_id": "capacity_occlusion_scene",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "pen", "name": "pen", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                    "max_items": 1,
                },
                {
                    "id": "drawer_bottom",
                    "name": "drawer bottom",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                    "max_items": 1,
                },
            ],
            "edges": [
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "pen", "to": "table", "relation": "ON"},
                {"from": "drawer_top", "to": "table", "relation": "ON"},
                {"from": "drawer_bottom", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "capacity_occlusion_profile",
            "target_object": "book",
            "spatial": {"enabled": True, "num_occluded_objects": 3, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "4",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        occlusion_sources = [
            edge["from"]
            for edge in payload["edges"]
            if edge["relation"] == "OCCLUDES"
        ]
        self.assertEqual(len(occlusion_sources), 2)
        self.assertLessEqual(occlusion_sources.count("drawer_top"), 1)
        self.assertLessEqual(occlusion_sources.count("drawer_bottom"), 1)
        achieved = payload["metadata"]["achieved_constraint_profile"]["spatial"]
        self.assertEqual(achieved["num_occluded_objects"], 2)

    def test_profile_editor_blocks_each_drawer_layer_from_occluding_book(self) -> None:
        scene = {
            "scene_id": "desk_drawer_book_constraints",
            "env_id": "desk_drawer_book_constraints",
            "layout": "tabletop",
            "robot": {"arms": "double"},
            "nodes": [
                {"id": "桌面", "name": "桌面", "category": "surface", "properties": ["SURFACES"]},
                {"id": "书", "name": "书", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "文件夹", "name": "文件夹", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {
                    "id": "抽屉第一层",
                    "name": "抽屉第一层",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                },
                {
                    "id": "抽屉第二层",
                    "name": "抽屉第二层",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                },
                {
                    "id": "抽屉第三层",
                    "name": "抽屉第三层",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "书", "to": "桌面", "relation": "ON"},
                {"from": "文件夹", "to": "桌面", "relation": "ON"},
                {"from": "抽屉第一层", "to": "桌面", "relation": "BENEATH"},
                {"from": "抽屉第二层", "to": "桌面", "relation": "BENEATH"},
                {"from": "抽屉第三层", "to": "桌面", "relation": "BENEATH"},
            ],
        }
        profile = {
            "profile_id": "desk_drawer_book_constraints",
            "target_object": "书",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        constraints = {
            "nonexistent_edges": [
                {"from": "抽屉第一层", "to": "书", "relation": "OCCLUDES"},
                {"from": "抽屉第二层", "to": "书", "relation": "OCCLUDES"},
                {"from": "抽屉第三层", "to": "书", "relation": "OCCLUDES"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            constraint_path = Path(tmpdir) / "placement_constraints.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene, ensure_ascii=False) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
            constraint_path.write_text(json.dumps(constraints, ensure_ascii=False), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--placement-edge-constraints",
                    str(constraint_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in payload["edges"]
        }
        self.assertIn(("文件夹", "书", "OCCLUDES"), edge_triples)
        for drawer_layer in ("抽屉第一层", "抽屉第二层", "抽屉第三层"):
            self.assertNotIn((drawer_layer, "书", "OCCLUDES"), edge_triples)
        self.assertEqual(payload["metadata"]["placement_edge_constraints"]["forbidden_edges"], constraints["nonexistent_edges"])

    def test_profile_spatial_container_occluder_uses_occludes_relation(self) -> None:
        scene = {
            "scene_id": "container_occlusion_scene",
            "env_id": "container_occlusion_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "box",
                    "name": "box",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "container_occlusion_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        occlusion = payload["metadata"]["profile_constraints"]["spatial"]["occlusion"]["objects"][0]
        self.assertEqual(occlusion["layers"][0]["type"], "occluder")
        self.assertEqual(occlusion["layers"][0]["blocker"], "box")
        self.assertEqual(occlusion["layers"][0]["resolution_action"], "open")
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in payload["edges"]
        }
        self.assertNotIn(("paper", "box", "INSIDE"), edge_triples)
        self.assertIn(("box", "paper", "OCCLUDES"), edge_triples)
        self.assertNotIn(("paper", "table", "ON"), edge_triples)
        self.assertIn(("box", "table", "ON"), edge_triples)
        self.assertEqual(
            payload["metadata"]["achieved_constraint_profile"]["spatial"]["occlusion_depths"]["paper"],
            1,
        )

    def test_profile_spatial_static_nodes_can_occlude_but_are_not_occlusion_targets(self) -> None:
        scene = {
            "scene_id": "static_occluder_scene",
            "env_id": "static_occluder_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {
                    "id": "folder",
                    "name": "folder",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                    "states": ["CLOSED"],
                },
                {
                    "id": "drawer_top",
                    "name": "drawer top",
                    "category": "container",
                    "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER", "STATIC"],
                    "states": ["CLOSED"],
                },
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "drawer_top", "to": "table", "relation": "ON"},
            ],
        }
        profile = {
            "profile_id": "static_occluder_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "1",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in payload["edges"]
        }
        self.assertIn(("folder", "paper", "OCCLUDES"), edge_triples)
        self.assertIn(("drawer_top", "folder", "OCCLUDES"), edge_triples)
        self.assertFalse(
            any(edge["to"] == "drawer_top" and edge["relation"] == "OCCLUDES" for edge in payload["edges"])
        )
        self.assertEqual(
            payload["metadata"]["achieved_constraint_profile"]["spatial"]["occlusion_depths"]["paper"],
            2,
        )

    def test_profile_spatial_replaces_existing_occluders_for_each_object(self) -> None:
        scene = {
            "scene_id": "replace_occluder_scene",
            "env_id": "replace_occluder_scene",
            "layout": "tabletop",
            "robot": {"arms": "single"},
            "nodes": [
                {"id": "table", "name": "table", "category": "surface", "properties": ["SURFACES"]},
                {"id": "paper", "name": "paper", "category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                {"id": "book", "name": "book", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {"id": "folder", "name": "folder", "category": "object", "properties": ["OCCLUDER", "MOVABLE"]},
                {"id": "box", "name": "box", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN"]},
            ],
            "edges": [
                {"from": "paper", "to": "table", "relation": "ON"},
                {"from": "book", "to": "table", "relation": "ON"},
                {"from": "folder", "to": "table", "relation": "ON"},
                {"from": "box", "to": "table", "relation": "ON"},
                {"from": "book", "to": "paper", "relation": "OCCLUDES"},
                {"from": "folder", "to": "paper", "relation": "OCCLUDES"},
                {"from": "paper", "to": "folder", "relation": "LEFT_OF"},
            ],
        }
        profile = {
            "profile_id": "replace_occluder_profile",
            "target_object": "paper",
            "spatial": {"enabled": True, "num_occluded_objects": 1, "occlusion_depth": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "scene.jsonl"
            profile_path = Path(tmpdir) / "profile.json"
            output_path = Path(tmpdir) / "profiled.jsonl"
            graph_path.write_text(json.dumps(scene) + "\n", encoding="utf-8")
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            code = main(
                [
                    "edit-view-graph",
                    "--input",
                    str(graph_path),
                    "--profile",
                    str(profile_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "5",
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(code, 0)
        paper_occluders = [
            edge["from"]
            for edge in payload["edges"]
            if edge["to"] == "paper" and edge["relation"] == "OCCLUDES"
        ]
        edge_triples = {(edge["from"], edge["to"], edge["relation"]) for edge in payload["edges"]}
        self.assertEqual(len(paper_occluders), 1)
        self.assertNotIn(("paper", "table", "ON"), edge_triples)
        self.assertNotIn(("paper", "folder", "LEFT_OF"), edge_triples)
        self.assertTrue(
            any(edit["type"] == "replace_occluder" and edit["target"] == "paper" for edit in payload["metadata"]["graph_edits"])
        )
        self.assertTrue(
            any(edit["type"] == "remove_visible_spatial_edges_for_occlusion" and edit["target"] == "paper" for edit in payload["metadata"]["graph_edits"])
        )

    def test_cli_rejects_local_task_view_graph_provider(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    "create-task-view-graph",
                    "--materials",
                    "apple, box",
                    "--scene",
                    "office desktop",
                    "--layout",
                    "tabletop",
                    "--arms",
                    "single",
                    "--output",
                    "unused.jsonl",
                    "--provider",
                    "local",
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_view_graph_prompt_uses_activity_input_without_task_output(self) -> None:
        prompt = build_task_view_graph_prompt(
            TaskViewGraphSynthesisConfig(
                materials=("红色笔", "书"),
                scene="办公桌面",
                layout="tabletop",
                arms="single",
                task_hint="整理桌面",
                material_properties={
                    "红色笔": {"properties": ["GRABBABLE", "MOVABLE"]},
                    "红色笔笔帽": {"properties": ["GRABBABLE", "MOVABLE"], "part_of": "红色笔"},
                },
            )
        )

        self.assertIn("整理桌面", prompt)
        self.assertIn("红色笔", prompt)
        self.assertIn("不要翻译成英文", prompt)
        self.assertIn("Relation Skill", prompt)
        self.assertIn("Properties Skill", prompt)
        self.assertIn("初始 view graph 不输出 `INSIDE` 边", prompt)
        self.assertIn("初始 view graph 只保留这个能力属性，不输出 `OCCLUDES` 边", prompt)
        self.assertIn("遮挡关系由后续 profile editor 根据难度 profile 添加", prompt)
        self.assertIn("`COPYABLE`", prompt)
        self.assertIn("spatial profile editor 不会复制节点", prompt)
        self.assertIn("`DECOMPOSABLE`", prompt)
        self.assertIn("声明部件清单", prompt)
        self.assertIn("红色笔 -> 红色笔笔帽", prompt)
        self.assertIn("每个保留在 `view_graph.nodes` 中的节点", prompt)
        self.assertIn("同一个物料可以用整体 node 参与对外关系", prompt)
        self.assertIn("不能混用", prompt)
        self.assertIn("如果输出部件 nodes", prompt)
        self.assertNotIn("implicit_environment", prompt)
        self.assertNotIn('"selected_materials"', prompt)
        self.assertNotIn('"task_definition"', prompt)
        self.assertNotIn('"instruction"', prompt)
        self.assertNotIn('"success_criteria"', prompt)

    def test_explicit_tabletop_material_stays_input_support(self) -> None:
        package = {
            "view_graph": {
                "scene_id": "explicit_tabletop",
                "env_id": "explicit_tabletop",
                "layout": "tabletop",
                "robot": {"arms": "single"},
                "nodes": [
                    {"id": "桌面", "name": "桌面", "category": "surface", "properties": ["SURFACES"]},
                    {"id": "苹果", "name": "苹果", "category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                ],
                "edges": [{"from": "苹果", "to": "桌面", "relation": "ON"}],
            }
        }

        _validate_task_view_graph_package(package, ("桌面", "苹果"), {"桌面": {"properties": ["SURFACES"]}})
        graph = ViewGraph.from_dict(package["view_graph"], fallback_scene_id="explicit_tabletop")

        tabletop = graph.get("桌面")
        self.assertEqual(tabletop.metadata["source"], "input")
        self.assertFalse(tabletop.is_implicit_environment)
        self.assertTrue(tabletop.can_be_task_target)
        self.assertNotIn("normalized_tabletop_support_nodes", package["view_graph"].get("metadata", {}))

    def test_validate_view_graph_removes_parent_external_edges_when_parts_are_external(self) -> None:
        package = {
            "view_graph": {
                "scene_id": "parent_part_conflict",
                "env_id": "parent_part_conflict",
                "layout": "tabletop",
                "robot": {"arms": "single"},
                "nodes": [
                    {"id": "desk", "name": "desk", "category": "surface", "properties": ["SURFACES"]},
                    {"id": "pen", "name": "pen", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
                    {
                        "id": "pen_cap",
                        "name": "pen cap",
                        "category": "tool",
                        "part_of": "pen",
                        "properties": ["GRABBABLE", "MOVABLE"],
                    },
                ],
                "edges": [
                    {"from": "pen", "to": "desk", "relation": "ON"},
                    {"from": "pen_cap", "to": "desk", "relation": "ON"},
                    {"from": "pen_cap", "to": "pen", "relation": "PART_OF"},
                ],
            }
        }

        _validate_task_view_graph_package(package, ("desk", "pen"), {"desk": {"properties": ["SURFACES"]}})
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in package["view_graph"]["edges"]
        }

        self.assertNotIn(("pen", "desk", "ON"), edge_triples)
        self.assertIn(("pen_cap", "desk", "ON"), edge_triples)
        self.assertIn(("pen_cap", "pen", "PART_OF"), edge_triples)
        self.assertEqual(package["view_graph"]["metadata"]["removed_parent_part_external_conflict_edges"], 1)

    def test_cli_creates_task_view_graph_from_api_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "generated.jsonl"
            package_path = Path(tmpdir) / "package.json"
            materials_path = Path(tmpdir) / "materials.txt"
            properties_path = Path(tmpdir) / "material_properties.json"
            materials_path.write_text(
                "# editable office material list\n"
                "apple\n"
                "box\n"
                "\n"
                "book\n"
                "pen\n"
                "clip\n"
                "paper\n"
                "desk\n",
                encoding="utf-8",
            )
            properties_path.write_text(
                json.dumps(
                    {
                        "materials": {
                            "apple": {"category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
                            "box": {
                                "category": "container",
                                "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                                "states": ["CLOSED"],
                            },
                            "book": {"category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                            "pen": {
                                "category": "tool",
                                "properties": ["GRABBABLE", "MOVABLE"],
                                "parts": [
                                    {
                                        "id": "pen_cap",
                                        "category": "tool",
                                        "properties": ["GRABBABLE", "MOVABLE"],
                                    }
                                ],
                            },
                            "clip": {"category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
                            "paper": {"category": "object", "properties": ["GRABBABLE", "MOVABLE"]},
                            "desk": {"category": "surface", "properties": ["SURFACES"]},
                            "drawer": {"category": "furniture", "properties": ["STORAGE_UNIT", "SURFACES"]},
                            "drawer_top": {
                                "category": "container",
                                "properties": ["CONTAINERS", "CAN_OPEN"],
                                "states": ["CLOSED"],
                                "part_of": "drawer",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            api_package = {
                "view_graph": {
                    "scene_id": "office_desktop_api",
                    "env_id": "office_desktop_api",
                    "layout": "tabletop",
                    "robot": {"arms": "double", "start": "front"},
                    "description": "API-created office desktop task graph.",
                    "nodes": [
                        {"id": "desk", "name": "desk", "category": "surface", "properties": ["SURFACES"]},
                        {
                            "id": "apple",
                            "name": "apple",
                            "category": "food",
                            "parent": "box",
                            "properties": ["GRABBABLE", "MOVABLE"],
                        },
                        {
                            "id": "book",
                            "name": "book",
                            "category": "object",
                            "parent": "desk",
                            "properties": ["GRABBABLE", "MOVABLE"],
                        },
                        {
                            "id": "box",
                            "name": "box",
                            "category": "container",
                            "parent": "desk",
                            "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"],
                            "states": ["CLOSED"],
                        },
                        {
                            "id": "pen",
                            "name": "pen",
                            "category": "tool",
                            "parent": "desk",
                            "properties": ["GRABBABLE", "MOVABLE"],
                        },
                        {
                            "id": "clip",
                            "name": "clip",
                            "category": "tool",
                            "parent": "desk",
                            "properties": ["GRABBABLE", "MOVABLE"],
                        },
                        {
                            "id": "paper",
                            "name": "paper",
                            "category": "object",
                            "parent": "desk",
                            "properties": ["GRABBABLE", "MOVABLE"],
                        },
                    ],
                    "edges": [
                        {"from": "apple", "to": "box", "relation": "INSIDE"},
                        {"from": "box", "to": "apple", "relation": "OCCLUDES"},
                        {"from": "box", "to": "desk", "relation": "ON"},
                        {"from": "book", "to": "desk", "relation": "ON"},
                        {"from": "pen", "to": "desk", "relation": "ON"},
                        {"from": "clip", "to": "desk", "relation": "ON"},
                        {"from": "paper", "to": "desk", "relation": "ON"},
                        {"from": "desk", "to": "pen", "relation": "OCCLUDES"},
                    ],
                },
            }

            completion = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(api_package)),
                    )
                ]
            )
            create_calls = []

            def fake_create(**kwargs):
                create_calls.append(kwargs)
                return completion

            fake_client = SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=fake_create),
                )
            )

            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}), patch(
                "auto_embodied_task.layout_synthesis.OpenAI",
                return_value=fake_client,
            ) as openai_cls:
                code = main(
                    [
                        "create-task-view-graph",
                        "--materials-file",
                        str(materials_path),
                        "--material-properties",
                        str(properties_path),
                        "--scene",
                        "office desktop",
                        "--layout",
                        "tabletop",
                        "--arms",
                        "double",
                        "--output",
                        str(output_path),
                        "--package-output",
                        str(package_path),
                        "--provider",
                        "qwen",
                        "--model",
                        "qwen3.6-plus",
                    ]
                )
            self.assertEqual(code, 0)
            openai_cls.assert_called_once()
            self.assertEqual(openai_cls.call_args.kwargs["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(openai_cls.call_args.kwargs["timeout"], 60)
            self.assertTrue(create_calls)
            self.assertEqual(create_calls[0]["model"], "qwen3.6-plus")
            self.assertEqual(create_calls[0]["extra_body"], {"enable_thinking": False})
            graphs = load_view_graphs_jsonl(output_path)
            package = json.loads(package_path.read_text(encoding="utf-8"))

        self.assertEqual(len(graphs), 1)
        graph = graphs[0]
        self.assertEqual(graph.layout, "tabletop")
        self.assertEqual(graph.metadata["task_synthesis_provider"], "qwen")
        self.assertEqual(graph.metadata["task_synthesis_model"], "qwen3.6-plus")
        self.assertNotIn("task_definition", graph.metadata)
        self.assertNotIn("selected_materials", graph.metadata)
        self.assertNotIn("task_definition", package)
        self.assertNotIn("selected_materials", package)
        box = graph.get("box")
        self.assertTrue(box.is_container)
        self.assertTrue(box.is_openable)
        self.assertIn("CLOSED", box.states)
        self.assertEqual(
            package["view_graph"]["metadata"]["input_materials"],
            ["apple", "box", "book", "pen", "clip", "paper", "desk"],
        )
        edge_triples = {
            (edge["from"], edge["to"], edge["relation"])
            for edge in package["view_graph"]["edges"]
        }
        self.assertNotIn(("apple", "box", "INSIDE"), edge_triples)
        self.assertNotIn(("box", "apple", "OCCLUDES"), edge_triples)
        self.assertIn(("pen_cap", "pen", "PART_OF"), edge_triples)
        self.assertIn(("book", "desk", "ON"), edge_triples)
        self.assertNotIn(("desk", "pen", "OCCLUDES"), edge_triples)
        self.assertIn("desk", graph.nodes)
        self.assertEqual(graph.get("desk").metadata["source"], "input")
        self.assertIn("pen", graph.nodes)
        self.assertNotIn("桌面", graph.nodes)
        self.assertNotIn("implicit_tabletop", graph.nodes)
        self.assertIn("pen_cap", graph.nodes)
        self.assertEqual(package["view_graph"]["metadata"]["added_declared_part_nodes"], ["pen_cap"])
        self.assertEqual(
            package["view_graph"]["metadata"]["added_declared_part_edges"],
            [{"from": "pen_cap", "to": "pen", "relation": "PART_OF"}],
        )
        self.assertEqual(package["view_graph"]["metadata"]["removed_inside_edges"], 1)
        self.assertEqual(package["view_graph"]["metadata"]["removed_occlusion_edges"], 2)
        self.assertNotIn("removed_invalid_relation_affordance_edges", package["view_graph"]["metadata"])
        self.assertNotIn("normalized_tabletop_support_nodes", package["view_graph"]["metadata"])
        self.assertNotIn("removed_parent_nodes_with_explicit_parts", package["view_graph"]["metadata"])
        self.assertNotIn("removed_parent_edges_with_explicit_parts", package["view_graph"]["metadata"])
        self.assertNotIn("added_tabletop_on_edges", package["view_graph"]["metadata"])
        self.assertNotIn("added_implicit_tabletop_on_edges", package["view_graph"]["metadata"])
        self.assertNotIn("auto_augmented_occlusion_edges", package["view_graph"]["metadata"])
        self.assertTrue({"apple", "box", "book", "pen", "pen_cap", "clip", "paper", "desk"}.issubset(graph.nodes))
        tasks = TaskGenerator(GenerationConfig(task_types=("multi_object",), max_tasks=1)).generate(graphs)
        self.assertTrue(any("[open]" in action for action in tasks[0].ground_truth_plan))
        self.assertTrue(any("[close]" in action for action in tasks[0].ground_truth_plan))


if __name__ == "__main__":
    unittest.main()
