from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import random
from typing import Iterable

from .constraints import build_modifiers
from .models import Node, TaskRecord, ViewGraph
from .planner import PlanBuilder
from .templates import (
    LONG_HORIZON_TEMPLATES,
    MANIPULATION_INSIDE_TEMPLATES,
    MANIPULATION_ON_TEMPLATES,
    MULTI_INSIDE_TEMPLATES,
    MULTI_ON_TEMPLATES,
    NAVIGATION_TEMPLATES,
    choose_template,
)


ALL_TASK_TYPES = ("long_horizon", "navigation", "manipulation", "multi_object")


@dataclass
class GenerationConfig:
    task_types: tuple[str, ...] = ALL_TASK_TYPES
    settings: tuple[str, ...] = ()
    include_base: bool = True
    arms: str | None = None
    layout: str = "all"
    max_tasks: int | None = 100
    seed: int = 0
    actor: str = "<char0>"
    max_pairs_per_scene: int = 200


@dataclass
class _Counter:
    value: int = 0

    def next(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}_{self.value:06d}"


class TaskGenerator:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.rng = random.Random(self.config.seed)
        self.plan_builder = PlanBuilder(actor=self.config.actor)
        self.modifiers = build_modifiers(self.config.settings)

    def generate(self, graphs: Iterable[ViewGraph]) -> list[TaskRecord]:
        tasks: list[TaskRecord] = []
        counter = _Counter()
        for graph in graphs:
            if self.config.layout != "all" and graph.layout != self.config.layout:
                continue
            for base_task in self._base_tasks_for_graph(graph, counter):
                if self.config.include_base:
                    tasks.append(base_task)
                    if self._is_full(tasks):
                        return tasks
                for modifier in self.modifiers:
                    variant = modifier.apply(base_task, graph, self.rng)
                    if variant is None:
                        continue
                    variant.task_id = counter.next(f"{graph.scene_id}_{modifier.name}")
                    tasks.append(variant)
                    if self._is_full(tasks):
                        return tasks
        return tasks

    def _is_full(self, tasks: list[TaskRecord]) -> bool:
        return self.config.max_tasks is not None and len(tasks) >= self.config.max_tasks

    def _base_tasks_for_graph(self, graph: ViewGraph, counter: _Counter) -> Iterable[TaskRecord]:
        task_types = self.config.task_types
        if "all" in task_types:
            task_types = ALL_TASK_TYPES
        arms = self._arms_for_graph(graph)
        if "long_horizon" in task_types:
            yield from self._long_horizon_tasks(graph, arms, counter)
        if "navigation" in task_types:
            yield from self._navigation_tasks(graph, arms, counter)
        if "manipulation" in task_types:
            yield from self._manipulation_tasks(graph, arms, counter)
        if "multi_object" in task_types:
            yield from self._multi_object_tasks(graph, arms, counter)

    def _arms_for_graph(self, graph: ViewGraph) -> str:
        selected = self.config.arms or str(graph.robot.get("arms", "single"))
        if selected not in {"single", "double"}:
            raise ValueError(f"arms must be single or double, got {selected!r}")
        return selected

    def _navigation_tasks(self, graph: ViewGraph, arms: str, counter: _Counter) -> Iterable[TaskRecord]:
        for target in graph.navigation_targets:
            task_text = choose_template(self.rng, NAVIGATION_TEMPLATES).format(object=target.name)
            yield TaskRecord(
                task_id=counter.next(f"{graph.scene_id}_navigation"),
                scene_id=graph.scene_id,
                env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
                layout=graph.layout,
                arms=arms,
                task_type="navigation",
                task=task_text,
                task_completion_criterion=f"(CLOSE, robot, {target.name})",
                ground_truth_plan=self.plan_builder.navigate_to(graph, target),
                objects={"target": target.id},
                metadata={"target_name": target.name},
            )

    def _manipulation_tasks(self, graph: ViewGraph, arms: str, counter: _Counter) -> Iterable[TaskRecord]:
        emitted = 0
        for obj in graph.grabbable_objects:
            for target in graph.placement_targets:
                if obj.id == target.id:
                    continue
                if not self._reasonable_single_object_pair(obj, target):
                    continue
                relation = graph.relation_for_placement(target)
                templates = MANIPULATION_INSIDE_TEMPLATES if relation == "INSIDE" else MANIPULATION_ON_TEMPLATES
                task_text = choose_template(self.rng, templates).format(object=obj.name, target=target.name)
                yield TaskRecord(
                    task_id=counter.next(f"{graph.scene_id}_manipulation"),
                    scene_id=graph.scene_id,
                    env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
                    layout=graph.layout,
                    arms=arms,
                    task_type="manipulation",
                    task=task_text,
                    task_completion_criterion=self._placement_criterion(obj, target, relation),
                    ground_truth_plan=self.plan_builder.manipulation_plan(graph, obj, target, relation),
                    objects={"object": obj.id, "target": target.id, "relation": relation},
                    metadata={"object_name": obj.name, "target_name": target.name},
                )
                emitted += 1
                if emitted >= self.config.max_pairs_per_scene:
                    return

    def _multi_object_tasks(self, graph: ViewGraph, arms: str, counter: _Counter) -> Iterable[TaskRecord]:
        emitted = 0
        for obj1, obj2 in combinations(graph.grabbable_objects, 2):
            for target in graph.placement_targets:
                if target.id in {obj1.id, obj2.id}:
                    continue
                if not self._reasonable_single_object_pair(obj1, target):
                    continue
                if not self._reasonable_single_object_pair(obj2, target):
                    continue
                relation = graph.relation_for_placement(target)
                templates = MULTI_INSIDE_TEMPLATES if relation == "INSIDE" else MULTI_ON_TEMPLATES
                task_text = choose_template(self.rng, templates).format(
                    object1=obj1.name,
                    object2=obj2.name,
                    target=target.name,
                )
                objects = [obj1, obj2]
                yield TaskRecord(
                    task_id=counter.next(f"{graph.scene_id}_multi_object"),
                    scene_id=graph.scene_id,
                    env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
                    layout=graph.layout,
                    arms=arms,
                    task_type="multi_object",
                    task=task_text,
                    task_completion_criterion="".join(
                        self._placement_criterion(obj, target, relation) for obj in objects
                    ),
                    ground_truth_plan=self.plan_builder.multi_object_plan(graph, objects, target, relation, arms),
                    objects={"objects": [obj1.id, obj2.id], "target": target.id, "relation": relation},
                    metadata={
                        "object_names": [obj1.name, obj2.name],
                        "target_name": target.name,
                    },
                )
                emitted += 1
                if emitted >= self.config.max_pairs_per_scene:
                    return

    def _long_horizon_tasks(self, graph: ViewGraph, arms: str, counter: _Counter) -> Iterable[TaskRecord]:
        emitted = 0
        targets = sorted(graph.placement_targets, key=self._long_horizon_target_key)
        for target in targets:
            placements = self._long_horizon_placements(graph, target)
            if len(placements) < 2:
                continue
            yield self._build_long_horizon_task(graph, arms, counter, placements)
            emitted += 1
            if emitted >= self.config.max_pairs_per_scene:
                return

    def _long_horizon_placements(self, graph: ViewGraph, target: Node) -> list[tuple[Node, Node, str]]:
        relation = graph.relation_for_placement(target)
        candidates: list[tuple[int, str, str, Node]] = []
        for obj in graph.grabbable_objects:
            if not self._reasonable_single_object_pair(obj, target):
                continue
            if self._is_structural_part_with_movable_parent(graph, obj):
                continue
            if self._is_part_related(graph, obj, target):
                continue
            if self._is_already_placed(graph, obj, target, relation):
                continue
            score = self._long_horizon_object_score(graph, obj)
            candidates.append((-score, obj.name, obj.id, obj))

        candidates.sort()
        if not candidates or -candidates[0][0] <= 0:
            return []
        selected = [item[3] for item in candidates[:3]]
        if len(selected) < 2:
            return []
        return [(obj, target, relation) for obj in selected]

    def _build_long_horizon_task(
        self,
        graph: ViewGraph,
        arms: str,
        counter: _Counter,
        placements: list[tuple[Node, Node, str]],
    ) -> TaskRecord:
        opened: set[str] = set()
        opened_order: list[Node] = []
        plan: list[str] = []
        criteria: list[str] = []
        placement_records: list[dict[str, str | int]] = []
        precondition_records: list[dict[str, str | int]] = []
        constraint_subtasks: list[dict[str, str | int]] = []

        def open_once(node: Node) -> list[str]:
            already_open = node.id in opened
            steps = self.plan_builder.open_if_needed(node, opened)
            if steps and not already_open:
                opened_order.append(node)
            return steps

        for step, (obj, target, relation) in enumerate(placements, start=1):
            access_constraints = self._access_constraints_for_object(graph, obj)
            part_nodes = self._part_nodes(graph, obj.id)
            resolved_by_open: set[str] = set()
            resolved_blockers: set[str] = set()

            for constraint in access_constraints:
                anchor = constraint["anchor"]
                record = {
                    "step": step,
                    "kind": constraint["kind"],
                    "relation": constraint["relation"],
                    "object": obj.name,
                    "anchor": anchor.name,
                }
                precondition_records.append(record)
                if constraint["kind"] == "source_container" and anchor.is_openable:
                    plan.extend(self.plan_builder.navigate_to(graph, anchor))
                    plan.extend(open_once(anchor))
                    resolved_by_open.add(anchor.id)
                    constraint_subtasks.append(
                        {
                            **record,
                            "type": "open_source_container",
                            "setting": "long_horizon",
                        }
                    )
                    continue
                if anchor.id in resolved_by_open:
                    constraint_subtasks.append(
                        {
                            **record,
                            "type": "resolved_by_open_source",
                            "setting": "long_horizon",
                        }
                    )
                    continue
                if anchor.id in resolved_blockers:
                    constraint_subtasks.append(
                        {
                            **record,
                            "type": "already_resolved_access_constraint",
                            "setting": "long_horizon",
                        }
                    )
                    continue
                action = None
                if anchor.is_container and anchor.is_openable:
                    plan.extend(self.plan_builder.navigate_to(graph, anchor))
                    plan.extend(open_once(anchor))
                    action = "open"
                elif anchor.is_movable:
                    plan.extend(self.plan_builder.navigate_to(graph, anchor))
                    action = "move_aside"
                    plan.append(self.plan_builder.action(action, anchor))
                else:
                    constraint_subtasks.append(
                        {
                            **record,
                            "type": "unresolved_access_constraint",
                            "setting": "long_horizon",
                        }
                    )
                    continue
                resolved_blockers.add(anchor.id)
                constraint_subtasks.append(
                    {
                        **record,
                        "type": "resolve_access_constraint",
                        "setting": "long_horizon",
                        "action": action,
                    }
                )

            if part_nodes:
                plan.extend(self.plan_builder.navigate_to(graph, obj))
                for part in part_nodes[:2]:
                    plan.append(self.plan_builder.action("inspect", obj, part))
                    constraint_subtasks.append(
                        {
                            "setting": "long_horizon",
                            "type": "verify_part_state",
                            "step": step,
                            "object": obj.name,
                            "part": part.name,
                            "relation": "PART_OF",
                        }
                    )

            plan.extend(self.plan_builder.navigate_to(graph, obj))
            plan.append(self.plan_builder.action("grab", obj))
            plan.extend(self.plan_builder.navigate_to(graph, target))
            plan.extend(open_once(target))
            plan.append(self.plan_builder.action("putin" if relation == "INSIDE" else "puton", obj, target))

            criteria.append(
                f"STEP_{step}: {self._placement_criterion(obj, target, relation)}"
            )
            placement_records.append(
                {
                    "step": step,
                    "object": obj.id,
                    "object_name": obj.name,
                    "target": target.id,
                    "target_name": target.name,
                    "relation": relation,
                }
            )
            constraint_subtasks.append(
                {
                    "setting": "long_horizon",
                    "type": "ordered_placement",
                    "step": step,
                    "object": obj.name,
                    "target": target.name,
                    "relation": relation,
                    "criterion": criteria[-1],
                }
            )

        for node in reversed(opened_order):
            plan.extend(self.plan_builder.close_if_needed(node, opened))

        task_text = choose_template(self.rng, LONG_HORIZON_TEMPLATES).format(
            placements=self._format_long_horizon_placements(placements)
        )
        object_ids = [obj.id for obj, _, _ in placements]
        target_ids = [target.id for _, target, _ in placements]
        return TaskRecord(
            task_id=counter.next(f"{graph.scene_id}_long_horizon"),
            scene_id=graph.scene_id,
            env_id=graph.env_id if graph.env_id is not None else graph.scene_id,
            layout=graph.layout,
            arms=arms,
            task_type="long_horizon",
            task=task_text,
            task_completion_criterion=" ".join(criteria),
            ground_truth_plan=plan,
            objects={
                "objects": object_ids,
                "targets": target_ids,
                "target": target_ids[0],
                "relation": placements[0][2],
                "placements": [
                    {"object": item["object"], "target": item["target"], "relation": item["relation"]}
                    for item in placement_records
                ],
            },
            metadata={
                "object_names": [obj.name for obj, _, _ in placements],
                "target_names": [target.name for _, target, _ in placements],
                "placements": placement_records,
                "long_horizon": {
                    "num_steps": len(placements),
                    "uses_constraints": bool(precondition_records),
                    "preconditions": precondition_records,
                },
                "constraint_subtasks": constraint_subtasks,
            },
        )

    @staticmethod
    def _reasonable_single_object_pair(obj: Node, target: Node) -> bool:
        if not obj.is_grabbable or not obj.is_movable:
            return False
        if not target.can_be_task_target:
            return False
        if obj.is_room or target.is_room:
            return False
        if obj.id == target.id:
            return False
        return True

    @staticmethod
    def _placement_criterion(obj: Node, target: Node, relation: str) -> str:
        return f"(CLOSE, robot, {obj.name})(CLOSE, robot, {target.name})({relation}, {obj.name}, {target.name})"

    @staticmethod
    def _long_horizon_target_key(target: Node) -> tuple[int, int, str, str]:
        return (
            0 if target.is_openable else 1,
            0 if target.is_container else 1,
            target.name,
            target.id,
        )

    def _long_horizon_object_score(self, graph: ViewGraph, obj: Node) -> int:
        score = 0
        access_constraints = self._access_constraints_for_object(graph, obj)
        score += 4 * len(access_constraints)
        score += 2 * len(self._part_nodes(graph, obj.id))
        if graph.spatial_context(obj.id) is not None:
            score += 1
        return score

    def _access_constraints_for_object(self, graph: ViewGraph, obj: Node) -> list[dict[str, Node | str]]:
        constraints: list[dict[str, Node | str]] = []
        seen: set[tuple[str, str, str]] = set()

        def add(kind: str, relation: str, anchor: Node) -> None:
            if anchor.id == obj.id or anchor.is_room:
                return
            key = (kind, relation, anchor.id)
            if key in seen:
                return
            seen.add(key)
            constraints.append({"kind": kind, "relation": relation, "anchor": anchor})

        context = graph.spatial_context(obj.id)
        if context is not None:
            relation, _, anchor = context
            if (relation in {"INSIDE", "IN"} or (obj.parent == anchor.id and anchor.is_container)) and anchor.is_openable:
                add("source_container", "INSIDE", anchor)

        incoming_relations = {
            "OCCLUDES": ("occlusion", "OCCLUDED_BY"),
            "PARTIALLY_OCCLUDES": ("occlusion", "OCCLUDED_BY"),
            "BLOCKS": ("occlusion", "OCCLUDED_BY"),
            "HIDES": ("occlusion", "OCCLUDED_BY"),
            "COVERS": ("occlusion", "OCCLUDED_BY"),
        }
        for edge in graph.incoming(obj.id):
            mapping = incoming_relations.get(edge.relation)
            if mapping is None:
                continue
            kind, relation = mapping
            anchor = graph.get(edge.source)
            if kind == "occlusion" and anchor.is_openable and self._is_inside_anchor(graph, obj, anchor):
                add("source_container", "INSIDE", anchor)
            add(kind, relation, anchor)

        metadata_relations = {
            "occluded_by": ("occlusion", "OCCLUDED_BY"),
            "blocked_by": ("occlusion", "OCCLUDED_BY"),
            "hidden_by": ("occlusion", "OCCLUDED_BY"),
            "covered_by": ("occlusion", "OCCLUDED_BY"),
        }
        for key, (kind, relation) in metadata_relations.items():
            if key not in obj.metadata:
                continue
            for anchor in self._metadata_anchor_nodes(graph, obj.metadata[key]):
                add(kind, relation, anchor)

        return constraints

    @staticmethod
    def _metadata_anchor_nodes(graph: ViewGraph, raw_value) -> list[Node]:
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        anchors = []
        for value in values:
            anchor_id = str(value)
            if anchor_id in graph.nodes:
                anchors.append(graph.get(anchor_id))
                continue
            for node in graph.nodes.values():
                if node.name == anchor_id:
                    anchors.append(node)
                    break
        return anchors

    def _part_nodes(self, graph: ViewGraph, parent_id: str) -> list[Node]:
        parts: list[Node] = []
        seen: set[str] = set()
        for edge in graph.incoming(parent_id):
            if edge.relation != "PART_OF":
                continue
            part = graph.get(edge.source)
            parts.append(part)
            seen.add(part.id)
        for node in graph.nodes.values():
            if node.id in seen:
                continue
            if str(node.metadata.get("part_of", "")) == parent_id:
                parts.append(node)
                seen.add(node.id)
        return sorted(parts, key=lambda item: (item.name, item.id))

    def _is_structural_part_with_movable_parent(self, graph: ViewGraph, obj: Node) -> bool:
        parent_id = self._parent_object_id(graph, obj.id)
        if parent_id is None or parent_id not in graph.nodes:
            return False
        parent = graph.get(parent_id)
        return parent.is_grabbable and parent.is_movable

    def _is_part_related(self, graph: ViewGraph, obj: Node, target: Node) -> bool:
        return self._parent_object_id(graph, obj.id) == target.id or self._parent_object_id(graph, target.id) == obj.id

    @staticmethod
    def _parent_object_id(graph: ViewGraph, node_id: str) -> str | None:
        node = graph.get(node_id)
        parent_id = node.metadata.get("part_of")
        if parent_id is not None:
            return str(parent_id)
        for edge in graph.outgoing(node_id):
            if edge.relation == "PART_OF":
                return edge.target
        return None

    @staticmethod
    def _is_already_placed(graph: ViewGraph, obj: Node, target: Node, relation: str) -> bool:
        if obj.parent == target.id:
            return True
        wanted = {"INSIDE", "IN"} if relation == "INSIDE" else {"ON"}
        return any(edge.target == target.id and edge.relation in wanted for edge in graph.outgoing(obj.id))

    @staticmethod
    def _is_inside_anchor(graph: ViewGraph, obj: Node, anchor: Node) -> bool:
        if obj.parent == anchor.id and anchor.is_container:
            return True
        return any(
            edge.target == anchor.id and edge.relation in {"INSIDE", "IN"}
            for edge in graph.outgoing(obj.id)
        )

    @staticmethod
    def _format_long_horizon_placements(placements: list[tuple[Node, Node, str]]) -> str:
        phrases = []
        for obj, target, relation in placements:
            preposition = "into" if relation == "INSIDE" else "on"
            phrases.append(f"place {obj.name} {preposition} {target.name}")
        if len(phrases) == 1:
            return phrases[0]
        return ", then ".join(phrases)
