from __future__ import annotations

from random import Random

from ..models import TaskRecord, ViewGraph
from .base import TaskModifier, register_modifier


@register_modifier
class SpatialConstraint(TaskModifier):
    name = "spatial"
    # Spatial constraints here mean constrained object states, not ordinary
    # support/direction facts such as ON table or LEFT_OF bowl.
    DIRECT_RELATIONS = {
        "INSIDE": ("INSIDE", "containment"),
        "IN": ("INSIDE", "containment"),
        "CONTAINED_BY": ("INSIDE", "containment"),
        "OCCLUDED_BY": ("OCCLUDED_BY", "occlusion"),
        "PARTIALLY_OCCLUDED_BY": ("OCCLUDED_BY", "occlusion"),
        "BLOCKED_BY": ("OCCLUDED_BY", "occlusion"),
        "HIDDEN_BY": ("OCCLUDED_BY", "occlusion"),
        "COVERED_BY": ("OCCLUDED_BY", "occlusion"),
    }
    INVERSE_RELATIONS = {
        "CONTAINS": ("INSIDE", "containment"),
        "HAS_INSIDE": ("INSIDE", "containment"),
        "OCCLUDES": ("OCCLUDED_BY", "occlusion"),
        "PARTIALLY_OCCLUDES": ("OCCLUDED_BY", "occlusion"),
        "BLOCKS": ("OCCLUDED_BY", "occlusion"),
        "HIDES": ("OCCLUDED_BY", "occlusion"),
        "COVERS": ("OCCLUDED_BY", "occlusion"),
    }
    METADATA_RELATIONS = {
        "contained_by": ("INSIDE", "containment"),
        "inside": ("INSIDE", "containment"),
        "occluded_by": ("OCCLUDED_BY", "occlusion"),
        "blocked_by": ("OCCLUDED_BY", "occlusion"),
        "hidden_by": ("OCCLUDED_BY", "occlusion"),
        "covered_by": ("OCCLUDED_BY", "occlusion"),
    }

    def apply(self, task: TaskRecord, graph: ViewGraph, rng: Random) -> TaskRecord | None:
        facts = []
        for node_id in self._object_ids(task):
            facts.extend(self._constraint_facts_for_object(graph, node_id))

        if not facts:
            return None

        updated = task.clone()
        updated.settings.append(self.name)
        for _, relation, _, node_name, anchor_name in facts:
            criterion = f"({relation}, {node_name}, {anchor_name})"
            if criterion not in updated.task_completion_criterion:
                updated.task_completion_criterion += criterion
        updated.metadata["spatial_constraint"] = [
            {"kind": kind, "relation": relation, "object": node_name, "anchor": anchor_name}
            for _, relation, kind, node_name, anchor_name in facts
        ]
        existing_subtasks = list(updated.metadata.get("constraint_subtasks", []))
        updated.metadata["constraint_subtasks"] = existing_subtasks + [
            {
                "setting": self.name,
                "type": "resolve_spatial_state",
                "kind": kind,
                "object": node_name,
                "anchor": anchor_name,
                "relation": relation,
                "criterion": f"({relation}, {node_name}, {anchor_name})",
            }
            for _, relation, kind, node_name, anchor_name in facts
        ]
        return updated

    def _constraint_facts_for_object(self, graph: ViewGraph, node_id: str) -> list[tuple[str, str, str, str, str]]:
        if node_id not in graph.nodes:
            return []
        node = graph.get(node_id)
        facts: list[tuple[str, str, str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for edge in graph.outgoing(node_id):
            if edge.relation not in self.DIRECT_RELATIONS:
                continue
            anchor = graph.get(edge.target)
            relation, kind = self.DIRECT_RELATIONS[edge.relation]
            self._append_fact(facts, seen, node_id, relation, kind, node.name, anchor)

        for edge in graph.incoming(node_id):
            if edge.relation not in self.INVERSE_RELATIONS:
                continue
            anchor = graph.get(edge.source)
            relation, kind = self.INVERSE_RELATIONS[edge.relation]
            self._append_fact(facts, seen, node_id, relation, kind, node.name, anchor)

        for key, (relation, kind) in self.METADATA_RELATIONS.items():
            if key not in node.metadata:
                continue
            for anchor in self._metadata_anchors(graph, node.metadata[key]):
                self._append_fact(facts, seen, node_id, relation, kind, node.name, anchor)

        return facts

    @staticmethod
    def _append_fact(
        facts: list[tuple[str, str, str, str, str]],
        seen: set[tuple[str, str, str]],
        node_id: str,
        relation: str,
        kind: str,
        node_name: str,
        anchor,
    ) -> None:
        if anchor.id == node_id or anchor.is_room:
            return
        key = (node_id, relation, anchor.id)
        if key in seen:
            return
        seen.add(key)
        facts.append((node_id, relation, kind, node_name, anchor.name))

    @staticmethod
    def _metadata_anchors(graph: ViewGraph, raw_value) -> list:
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

    @staticmethod
    def _object_ids(task: TaskRecord) -> list[str]:
        ids: list[str] = []
        if "target" in task.objects:
            ids.append(task.objects["target"])
        if "object" in task.objects:
            ids.append(task.objects["object"])
        ids.extend(task.objects.get("objects", []))
        return list(dict.fromkeys(ids))
