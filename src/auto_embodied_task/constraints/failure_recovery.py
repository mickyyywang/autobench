from __future__ import annotations

from random import Random

from ..models import TaskRecord, ViewGraph
from .base import TaskModifier, register_modifier


@register_modifier
class FailureRecovery(TaskModifier):
    name = "failure_recovery"

    def apply(self, task: TaskRecord, graph: ViewGraph, rng: Random) -> TaskRecord | None:
        if task.task_type == "navigation":
            return None
        updated = task.clone()
        failure_index = self._first_action_index(updated.ground_truth_plan, "[grab]")
        failed_action = "grab"
        if failure_index is None:
            failure_index = self._first_action_index(updated.ground_truth_plan, "[open]")
            failed_action = "open"
        if failure_index is None:
            return None

        object_name = updated.metadata.get("object_name")
        if object_name is None and updated.metadata.get("object_names"):
            object_name = updated.metadata["object_names"][0]
        if object_name is None:
            object_name = "the target object"

        failed_step = updated.ground_truth_plan[failure_index].replace(f"[{failed_action}]", f"[failed_{failed_action}]")
        recovery_step = failed_step.replace(f"[failed_{failed_action}]", "[recover]")
        updated.ground_truth_plan = (
            updated.ground_truth_plan[:failure_index]
            + [failed_step, recovery_step]
            + updated.ground_truth_plan[failure_index:]
        )
        updated.settings.append(self.name)
        updated.metadata["failure_recovery"] = {
            "failed_action": failed_action,
            "failure_plan_index": failure_index,
            "recovery_policy": "recover_and_retry_same_action",
        }
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + [
            {
                "setting": self.name,
                "type": "inject_failure",
                "action": failed_action,
                "object": object_name,
                "plan_index": failure_index,
                "plan_step": failed_step,
            },
            {
                "setting": self.name,
                "type": "recover_and_retry",
                "action": failed_action,
                "object": object_name,
                "plan_step": recovery_step,
            },
        ]
        return updated

    @staticmethod
    def _first_action_index(plan: list[str], token: str) -> int | None:
        for index, step in enumerate(plan):
            if token in step:
                return index
        return None
