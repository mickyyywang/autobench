from __future__ import annotations

from random import Random

from ..models import TaskRecord, ViewGraph
from .base import TaskModifier, register_modifier


@register_modifier
class MemoryConstraint(TaskModifier):
    name = "memory"
    OUTGOING_RELATIONS = ("INSIDE", "IN", "ON", "CLOSE", "NEAR")
    INCOMING_RELATIONS = {
        "CONTAINS": "INSIDE",
        "HAS_INSIDE": "INSIDE",
        "HOLDS": "INSIDE",
        "SUPPORTS": "ON",
        "HAS_ON": "ON",
        "CLOSE": "CLOSE",
        "NEAR": "NEAR",
    }

    def apply(self, task: TaskRecord, graph: ViewGraph, rng: Random) -> TaskRecord | None:
        node_id = self._primary_object_id(task)
        if node_id is None or node_id not in graph.nodes:
            return None
        context = self._memory_context(graph, node_id)
        if context is None:
            return None
        relation, node, anchor = context
        fact = f"({relation}, {node.name}, {anchor.name})"
        actor = self._actor_from_plan(task)
        updated = task.clone()
        updated.settings.append(self.name)
        updated.metadata["memory_constraint"] = {
            "mode": "prior_observation",
            "not_initial_state": True,
            "remember_object": node.name,
            "remember_anchor": anchor.name,
            "remember_relation": relation,
            "memory_fact": fact,
        }
        updated.metadata["memory_episode"] = {
            "type": "prior_observation",
            "not_initial_state": True,
            "observations": [
                {
                    "object": node.name,
                    "anchor": anchor.name,
                    "relation": relation,
                    "fact": fact,
                }
            ],
            "observation_plan": [
                f"{actor} [look] <{anchor.name}> ({anchor.id})",
                f"{actor} [observe] <{node.name}> ({node.id}) <{anchor.name}> ({anchor.id})",
            ],
        }
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + [
            {
                "setting": self.name,
                "type": "retrieve_prior_observation",
                "object": node.name,
                "anchor": anchor.name,
                "relation": relation,
                "memory_fact": fact,
                "source": "metadata.memory_episode",
            }
        ]
        return updated

    def _memory_context(self, graph: ViewGraph, node_id: str):
        node = graph.get(node_id)
        for relation in self.OUTGOING_RELATIONS:
            for edge in graph.outgoing(node_id):
                if edge.relation != relation:
                    continue
                anchor = graph.get(edge.target)
                if anchor.id != node_id:
                    return self._canonical_relation(relation), node, anchor

        for edge in graph.incoming(node_id):
            relation = self.INCOMING_RELATIONS.get(edge.relation)
            if relation is None:
                continue
            anchor = graph.get(edge.source)
            if anchor.id != node_id:
                return relation, node, anchor

        return graph.spatial_context(node_id)

    @staticmethod
    def _canonical_relation(relation: str) -> str:
        if relation == "IN":
            return "INSIDE"
        return relation

    @staticmethod
    def _actor_from_plan(task: TaskRecord) -> str:
        if not task.ground_truth_plan:
            return "<char0>"
        first_step = task.ground_truth_plan[0]
        action_start = first_step.find(" [")
        if action_start <= 0:
            return "<char0>"
        return first_step[:action_start]

    @staticmethod
    def _primary_object_id(task: TaskRecord) -> str | None:
        if "object" in task.objects:
            return task.objects["object"]
        objects = task.objects.get("objects")
        if objects:
            return objects[0]
        return task.objects.get("target")
