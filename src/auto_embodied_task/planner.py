from __future__ import annotations

from .models import Node, ViewGraph


def _label(node: Node) -> str:
    return f"<{node.name}> ({node.id})"


def _action(actor: str, action: str, *nodes: Node | str) -> str:
    rendered = []
    for node in nodes:
        if isinstance(node, Node):
            rendered.append(_label(node))
        else:
            rendered.append(str(node))
    suffix = " " + " ".join(rendered) if rendered else ""
    return f"{actor} [{action}]{suffix}"


class PlanBuilder:
    def __init__(self, actor: str = "<char0>") -> None:
        self.actor = actor

    def action(self, action: str, *nodes: Node | str) -> str:
        return _action(self.actor, action, *nodes)

    def navigate_to(self, graph: ViewGraph, target: Node) -> list[str]:
        if graph.layout == "tabletop":
            support = graph.support_of(target.id)
            plan = []
            if support is not None:
                _, parent = support
                plan.append(_action(self.actor, "look", parent))
            plan.append(_action(self.actor, "reach", target))
            return plan

        room = graph.room_of(target.id)
        plan = []
        if room is not None and room.id != target.id:
            plan.append(_action(self.actor, "walk", room))
        plan.append(_action(self.actor, "walk", target))
        return plan

    def open_if_needed(self, target: Node, opened: set[str]) -> list[str]:
        if target.id in opened:
            return []
        if not target.is_openable:
            return []
        opened.add(target.id)
        return [_action(self.actor, "open", target)]

    def close_if_needed(self, target: Node, opened: set[str]) -> list[str]:
        if target.id not in opened or not target.is_openable:
            return []
        opened.remove(target.id)
        return [_action(self.actor, "close", target)]

    def manipulation_plan(self, graph: ViewGraph, obj: Node, target: Node, relation: str) -> list[str]:
        opened: set[str] = set()
        plan = []
        plan.extend(self.navigate_to(graph, obj))
        plan.append(_action(self.actor, "grab", obj))
        plan.extend(self.navigate_to(graph, target))
        plan.extend(self.open_if_needed(target, opened))
        plan.append(_action(self.actor, "putin" if relation == "INSIDE" else "puton", obj, target))
        plan.extend(self.close_if_needed(target, opened))
        return plan

    def multi_object_plan(
        self,
        graph: ViewGraph,
        objects: list[Node],
        target: Node,
        relation: str,
        arms: str,
    ) -> list[str]:
        opened: set[str] = set()
        plan = []
        if arms == "double":
            for obj in objects:
                plan.extend(self.navigate_to(graph, obj))
                plan.append(_action(self.actor, "grab", obj))
            plan.extend(self.navigate_to(graph, target))
            plan.extend(self.open_if_needed(target, opened))
            for obj in objects:
                plan.append(_action(self.actor, "putin" if relation == "INSIDE" else "puton", obj, target))
            plan.extend(self.close_if_needed(target, opened))
            return plan

        for obj in objects:
            plan.extend(self.navigate_to(graph, obj))
            plan.append(_action(self.actor, "grab", obj))
            plan.extend(self.navigate_to(graph, target))
            plan.extend(self.open_if_needed(target, opened))
            plan.append(_action(self.actor, "putin" if relation == "INSIDE" else "puton", obj, target))
        plan.extend(self.close_if_needed(target, opened))
        return plan
