from __future__ import annotations

from random import Random

from ..models import TaskRecord, ViewGraph
from .base import TaskModifier, register_modifier


@register_modifier
class TemporalConstraint(TaskModifier):
    name = "temporal"

    def apply(self, task: TaskRecord, graph: ViewGraph, rng: Random) -> TaskRecord | None:
        if task.task_type == "navigation":
            return None
        if task.task_type == "long_horizon":
            return self._apply_to_long_horizon(task, graph)
        if task.task_type == "multi_object":
            return self._apply_to_multi_object(task, graph)
        if task.task_type == "manipulation":
            return self._apply_to_single_object(task, graph)
        return None

    def _apply_to_long_horizon(self, task: TaskRecord, graph: ViewGraph) -> TaskRecord | None:
        placements = task.objects.get("placements", [])
        if not placements:
            return None

        updated = task.clone()
        updated.settings.append(self.name)
        step_criteria = []
        ordered_steps = []
        for index, placement in enumerate(placements, start=1):
            object_id = placement.get("object")
            target_id = placement.get("target")
            relation = str(placement.get("relation", "ON"))
            if object_id not in graph.nodes or target_id not in graph.nodes:
                return None
            obj = graph.get(object_id)
            target = graph.get(target_id)
            criterion = (
                f"STEP_{index}: (CLOSE, robot, {obj.name})(CLOSE, robot, {target.name})"
                f"({relation}, {obj.name}, {target.name})"
            )
            step_criteria.append(criterion)
            ordered_steps.append(
                {
                    "setting": self.name,
                    "type": "ordered_placement",
                    "step": index,
                    "object": obj.name,
                    "target": target.name,
                    "relation": relation,
                    "criterion": criterion,
                }
            )

        updated.task_completion_criterion = " ".join(step_criteria)
        updated.metadata["temporal_constraint"] = {
            "ordered_steps": "long_horizon_placements",
            "num_steps": len(ordered_steps),
        }
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + ordered_steps
        return updated

    def _apply_to_multi_object(self, task: TaskRecord, graph: ViewGraph) -> TaskRecord | None:
        object_ids = task.objects.get("objects", [])
        target_id = task.objects.get("target")
        relation = task.objects.get("relation")
        if len(object_ids) < 2 or target_id not in graph.nodes:
            return None
        target = graph.get(target_id)
        objects = [graph.get(object_id) for object_id in object_ids if object_id in graph.nodes]
        if len(objects) < 2:
            return None

        updated = task.clone()
        updated.settings.append(self.name)
        step_criteria = []
        for index, obj in enumerate(objects, start=1):
            step_criteria.append(
                f"STEP_{index}: (CLOSE, robot, {obj.name})(CLOSE, robot, {target.name})"
                f"({relation}, {obj.name}, {target.name})"
            )
        updated.task_completion_criterion = " ".join(step_criteria)
        updated.metadata["temporal_constraint"] = {
            "ordered_objects": [obj.name for obj in objects],
            "target": target.name,
            "relation": relation,
        }
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + [
            {
                "setting": self.name,
                "type": "ordered_placement",
                "step": index,
                "object": obj.name,
                "target": target.name,
                "relation": relation,
                "criterion": step_criteria[index - 1],
            }
            for index, obj in enumerate(objects, start=1)
        ]
        return updated

    def _apply_to_single_object(self, task: TaskRecord, graph: ViewGraph) -> TaskRecord | None:
        object_id = task.objects.get("object")
        target_id = task.objects.get("target")
        relation = task.objects.get("relation")
        if object_id not in graph.nodes or target_id not in graph.nodes:
            return None
        obj = graph.get(object_id)
        target = graph.get(target_id)

        updated = task.clone()
        updated.settings.append(self.name)
        updated.task_completion_criterion = (
            f"STEP_1: (CLOSE, robot, {obj.name}) "
            f"STEP_2: (CLOSE, robot, {target.name})({relation}, {obj.name}, {target.name})"
        )
        updated.metadata["temporal_constraint"] = {
            "ordered_steps": ["locate_object", "place_object"],
            "object": obj.name,
            "target": target.name,
            "relation": relation,
        }
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + [
            {
                "setting": self.name,
                "type": "ordered_locate",
                "step": 1,
                "object": obj.name,
                "criterion": f"STEP_1: (CLOSE, robot, {obj.name})",
            },
            {
                "setting": self.name,
                "type": "ordered_placement",
                "step": 2,
                "object": obj.name,
                "target": target.name,
                "relation": relation,
                "criterion": f"STEP_2: (CLOSE, robot, {target.name})({relation}, {obj.name}, {target.name})",
            },
        ]
        return updated
