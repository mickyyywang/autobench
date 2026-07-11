from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import copy


MAX_ITEMS_FIELDS = ("max_items", "item_capacity", "max_capacity", "capacity")


def normalize_property(value: str) -> str:
    return value.strip().upper().replace(" ", "_")


def normalize_relation(value: str) -> str:
    return value.strip().upper().replace(" ", "_")


def parse_max_items(data: dict[str, Any] | None) -> int | None:
    if not isinstance(data, dict):
        return None
    for key in MAX_ITEMS_FIELDS:
        raw_value = data.get(key)
        if isinstance(raw_value, dict):
            raw_value = (
                raw_value.get("max_items")
                if raw_value.get("max_items") is not None
                else raw_value.get("items", raw_value.get("max"))
            )
        if isinstance(raw_value, bool):
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return None


@dataclass
class Node:
    id: str
    name: str
    category: str = "object"
    properties: tuple[str, ...] = ()
    room: str | None = None
    parent: str | None = None
    states: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        node_id = data.get("id")
        if node_id is None:
            raise ValueError(f"Node is missing id: {data}")
        raw_name = data.get("name") or data.get("class_name") or data.get("instance_name")
        if raw_name is None:
            raise ValueError(f"Node is missing name/class_name: {data}")
        category = data.get("category", "object")
        if category == "Rooms":
            category = "room"
        props = tuple(normalize_property(str(p)) for p in data.get("properties", ()))
        states = tuple(normalize_property(str(s)) for s in data.get("states", ()))
        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "id",
                "name",
                "class_name",
                "instance_name",
                "category",
                "properties",
                "states",
                "room",
                "parent",
            }
        }
        return cls(
            id=str(node_id),
            name=str(raw_name),
            category=str(category).lower(),
            properties=props,
            room=str(data["room"]) if data.get("room") is not None else None,
            parent=str(data["parent"]) if data.get("parent") is not None else None,
            states=states,
            metadata=metadata,
        )

    def has_property(self, *properties: str) -> bool:
        wanted = {normalize_property(p) for p in properties}
        return any(prop in wanted for prop in self.properties)

    @property
    def is_room(self) -> bool:
        return self.category in {"room", "rooms"}

    @property
    def is_grabbable(self) -> bool:
        if self.is_room:
            return False
        if self.has_property("GRABBABLE"):
            return True
        if self.properties:
            return False
        return self.category in {"prop", "food", "tool"}

    @property
    def is_movable(self) -> bool:
        if self.has_property("MOVABLE"):
            return True
        if self.properties:
            return False
        return self.is_grabbable

    @property
    def is_container(self) -> bool:
        if self.has_property("CONTAINERS"):
            return True
        if self.properties:
            return False
        return self.category in {"container", "receptacle"}

    @property
    def is_surface(self) -> bool:
        if self.has_property("SURFACE", "SURFACES"):
            return True
        if self.properties:
            return False
        return self.category in {"surface", "furniture", "table", "counter", "workspace"}

    @property
    def is_openable(self) -> bool:
        return self.has_property("CAN_OPEN") or "CLOSED" in self.states or "OPEN" in self.states

    @property
    def max_items(self) -> int | None:
        return parse_max_items(self.metadata)

    @property
    def is_pressable(self) -> bool:
        return self.has_property("PRESSABLE")

    @property
    def is_implicit_environment(self) -> bool:
        source = str(self.metadata.get("source", ""))
        return source == "implicit_environment" or bool(
            self.metadata.get("implicit") or self.metadata.get("implicit_environment")
        )

    @property
    def can_be_task_target(self) -> bool:
        return not self.is_implicit_environment and not self.is_room and (self.is_container or self.is_surface)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Edge":
        source = data.get("from") if "from" in data else data.get("from_id", data.get("source"))
        target = data.get("to") if "to" in data else data.get("to_id", data.get("target"))
        relation = data.get("relation", data.get("relation_type"))
        if source is None or target is None or relation is None:
            raise ValueError(f"Edge needs from/to/relation fields: {data}")
        metadata = {
            k: v
            for k, v in data.items()
            if k not in {"from", "from_id", "source", "to", "to_id", "target", "relation", "relation_type"}
        }
        return cls(
            source=str(source),
            target=str(target),
            relation=normalize_relation(str(relation)),
            metadata=metadata,
        )


@dataclass
class ViewGraph:
    scene_id: str
    layout: str
    nodes: dict[str, Node]
    edges: list[Edge] = field(default_factory=list)
    env_id: int | str | None = None
    robot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_scene_id: str) -> "ViewGraph":
        nodes = {node.id: node for node in (Node.from_dict(item) for item in data.get("nodes", []))}
        edges = [Edge.from_dict(item) for item in data.get("edges", [])]
        scene_id = str(data.get("scene_id") or data.get("id") or fallback_scene_id)
        layout = str(data.get("layout", "indoor")).lower()
        env_id = data.get("env_id", data.get("environment_id", scene_id))
        metadata = {
            k: v
            for k, v in data.items()
            if k not in {"scene_id", "id", "layout", "nodes", "edges", "env_id", "environment_id", "robot", "metadata"}
        }
        raw_metadata = data.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        elif raw_metadata is not None:
            metadata["metadata"] = raw_metadata
        graph = cls(
            scene_id=scene_id,
            layout=layout,
            nodes=nodes,
            edges=edges,
            env_id=env_id,
            robot=dict(data.get("robot", {})),
            metadata=metadata,
        )
        graph._validate_edges()
        return graph

    def _validate_edges(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Unknown edge source {edge.source!r} in scene {self.scene_id}")
            if edge.target not in self.nodes:
                raise ValueError(f"Unknown edge target {edge.target!r} in scene {self.scene_id}")

    def get(self, node_id: str) -> Node:
        return self.nodes[str(node_id)]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == str(node_id)]

    def incoming(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target == str(node_id)]

    @property
    def rooms(self) -> list[Node]:
        return [node for node in self.nodes.values() if node.is_room]

    @property
    def grabbable_objects(self) -> list[Node]:
        return sorted(
            [node for node in self.nodes.values() if node.is_grabbable and node.is_movable],
            key=lambda item: (item.name, item.id),
        )

    @property
    def navigation_targets(self) -> list[Node]:
        return sorted(
            [node for node in self.nodes.values() if not node.is_room and not node.is_implicit_environment],
            key=lambda item: (item.name, item.id),
        )

    @property
    def placement_targets(self) -> list[Node]:
        return sorted(
            [node for node in self.nodes.values() if node.can_be_task_target],
            key=lambda item: (item.name, item.id),
        )

    def room_of(self, node_id: str) -> Node | None:
        node = self.get(node_id)
        if node.room and node.room in self.nodes and self.nodes[node.room].is_room:
            return self.nodes[node.room]

        room_relations = {"INSIDE", "IN", "ON", "CONTAINS", "HAS"}
        for edge in self.outgoing(node_id):
            target = self.nodes[edge.target]
            if target.is_room and edge.relation in room_relations:
                return target
        for edge in self.incoming(node_id):
            source = self.nodes[edge.source]
            if source.is_room and edge.relation in room_relations:
                return source
        return None

    def support_of(self, node_id: str) -> tuple[str, Node] | None:
        node = self.get(node_id)
        if node.parent and node.parent in self.nodes:
            return "ON", self.nodes[node.parent]

        useful = {
            "ON",
            "INSIDE",
            "IN",
            "CLOSE",
            "NEAR",
            "LEFT_OF",
            "RIGHT_OF",
            "FRONT_OF",
            "BEHIND",
        }
        for edge in self.outgoing(node_id):
            target = self.nodes[edge.target]
            if not target.is_room and edge.relation in useful:
                return edge.relation, target
        return None

    def spatial_context(self, node_id: str) -> tuple[str, Node, Node] | None:
        node = self.get(node_id)
        support = self.support_of(node_id)
        if support is not None:
            relation, parent = support
            return relation, node, parent
        room = self.room_of(node_id)
        if room is not None:
            return "INSIDE", node, room
        return None

    def relation_for_placement(self, target: Node) -> str:
        if target.is_container:
            return "INSIDE"
        return "ON"


@dataclass
class TaskRecord:
    task_id: str
    scene_id: str
    env_id: int | str
    layout: str
    arms: str
    task_type: str
    task: str
    task_completion_criterion: Any
    ground_truth_plan: list[str]
    objects: dict[str, Any] = field(default_factory=dict)
    settings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "TaskRecord":
        return copy.deepcopy(self)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
