from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .models import ViewGraph, parse_max_items
from .placement_constraints import PlacementEdgeConstraints, load_placement_edge_constraints


LOCATION_RELATIONS = {"ON", "INSIDE", "IN"}
OCCLUSION_RELATIONS = {"OCCLUDES", "PARTIALLY_OCCLUDES", "BLOCKS", "HIDES", "COVERS"}
VISIBLE_PLACEMENT_RELATIONS = LOCATION_RELATIONS | {
    "BENEATH",
    "UNDER",
    "BELOW",
    "CONTAINS",
    "HAS_INSIDE",
    "SUPPORTS",
    "HAS_ON",
}
VISIBLE_RELATIVE_RELATIONS = {"LEFT_OF", "RIGHT_OF", "FRONT_OF", "BEHIND", "CLOSE", "NEAR"}
DECOMPOSED_STATE = "DECOMPOSED"
DECOMPOSED_ASSEMBLY_STATES = {"ASSEMBLED", "ATTACHED", "CAPPED"}
STATIC_PROPERTIES = {"STATIC", "FIXED", "IMMOVABLE"}


@dataclass
class ProfileEditResult:
    graph: dict[str, Any]
    requested_profile: dict[str, Any]
    achieved_profile: dict[str, Any]
    constraints: dict[str, Any]
    graph_edits: list[dict[str, Any]]


def edit_view_graphs_with_profile(
    *,
    input_path: str | Path,
    profile_path: str | Path,
    output_path: str | Path,
    num_samples: int = 1,
    seed: int | None = None,
    placement_edge_constraints_path: str | Path | None = None,
    placement_edge_constraints: PlacementEdgeConstraints | None = None,
) -> list[ProfileEditResult]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    profile = load_constraint_profile(profile_path)
    constraints = placement_edge_constraints
    if placement_edge_constraints_path is not None:
        constraints = load_placement_edge_constraints(placement_edge_constraints_path)
    graphs = load_view_graph_records_jsonl(input_path)
    seed_source = random.Random(seed)
    results = []
    for graph in graphs:
        for sample_index in range(num_samples):
            sample_rng = random.Random(seed_source.randrange(0, 2**63))
            results.append(
                edit_view_graph_with_profile(
                    graph,
                    profile,
                    rng=sample_rng,
                    sample_index=sample_index,
                    num_samples=num_samples,
                    placement_edge_constraints=constraints,
                )
            )
    write_view_graph_records_jsonl((result.graph for result in results), output_path)
    return results


def load_constraint_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return payload


def edit_view_graph_with_profile(
    graph: dict[str, Any],
    profile: dict[str, Any],
    rng: random.Random | None = None,
    sample_index: int = 0,
    num_samples: int = 1,
    placement_edge_constraints: PlacementEdgeConstraints | None = None,
) -> ProfileEditResult:
    editor = _ProfileEditor(
        graph,
        profile,
        rng or random.Random(),
        sample_index,
        num_samples,
        placement_edge_constraints,
    )
    return editor.apply()


def load_view_graph_records_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSONL line: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_no}: every JSONL line must be an object")
            records.append((line_no, payload))
    if not records:
        return []
    if any("nodes" in record for _, record in records):
        graphs = []
        for line_no, record in records:
            if "nodes" not in record:
                raise ValueError(f"{source}:{line_no}: mixed direct graph and streaming records are not supported")
            graphs.append(copy.deepcopy(record))
        return graphs

    scenes: dict[str, dict[str, Any]] = {}
    for line_no, record in records:
        record_type = str(record.get("record_type", record.get("type", ""))).lower()
        scene_id = str(record.get("scene_id", "default"))
        scene = scenes.setdefault(scene_id, {"scene_id": scene_id, "nodes": [], "edges": []})
        if record_type == "scene":
            scene.update({k: v for k, v in record.items() if k not in {"record_type", "type"}})
        elif record_type == "node":
            scene["nodes"].append({k: v for k, v in record.items() if k not in {"record_type", "type", "scene_id"}})
        elif record_type == "edge":
            scene["edges"].append({k: v for k, v in record.items() if k not in {"record_type", "type", "scene_id"}})
        else:
            raise ValueError(f"{source}:{line_no}: expected nodes or record_type scene/node/edge")
    return [copy.deepcopy(scene) for _, scene in sorted(scenes.items(), key=lambda item: item[0])]


def write_view_graph_records_jsonl(graphs: Iterable[dict[str, Any]], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for graph in graphs:
            handle.write(json.dumps(graph, ensure_ascii=False) + "\n")
            count += 1
    return count


class _ProfileEditor:
    def __init__(
        self,
        graph: dict[str, Any],
        profile: dict[str, Any],
        rng: random.Random,
        sample_index: int,
        num_samples: int,
        placement_edge_constraints: PlacementEdgeConstraints | None,
    ) -> None:
        if not isinstance(profile, dict):
            raise ValueError("constraint profile must be a JSON object")
        self.graph = copy.deepcopy(graph)
        self.profile = copy.deepcopy(profile)
        self.rng = rng
        self.sample_index = sample_index
        self.num_samples = num_samples
        self.placement_edge_constraints = placement_edge_constraints or PlacementEdgeConstraints()
        self.nodes = self.graph.setdefault("nodes", [])
        self.edges = self.graph.setdefault("edges", [])
        if not isinstance(self.nodes, list) or not isinstance(self.edges, list):
            raise ValueError("view graph needs list fields: nodes and edges")
        self.edits: list[dict[str, Any]] = []
        self.constraints: dict[str, Any] = {}
        self.primary_object_id: str | None = None
        self.preferred_container_id: str | None = None

    def apply(self) -> ProfileEditResult:
        self._apply_sample_identity()
        self._validate_unique_node_ids()
        self.primary_object_id = self._pick_primary_object()
        if self.primary_object_id is None and self._needs_primary_object():
            self.primary_object_id = self._create_grabbable_node("profile_object")

        if self._enabled("spatial"):
            self.constraints["spatial"] = self._apply_spatial()

        self._normalize_occlusion_edges()
        achieved = self._achieved_profile()
        self._add_layered_drawer_occlusions()
        metadata = self.graph.setdefault("metadata", {})
        metadata["requested_constraint_profile"] = copy.deepcopy(self.profile)
        metadata["achieved_constraint_profile"] = achieved
        metadata["difficulty_tags"] = _difficulty_tags(achieved)
        metadata["profile_constraints"] = copy.deepcopy(self.constraints)
        metadata["graph_edits"] = copy.deepcopy(self.edits)
        metadata["profile_sample_index"] = self.sample_index + 1
        if not self.placement_edge_constraints.is_empty():
            metadata["placement_edge_constraints"] = self.placement_edge_constraints.to_json()
        if self.primary_object_id is not None:
            metadata["profile_primary_object"] = self.primary_object_id

        ViewGraph.from_dict(self.graph, fallback_scene_id=str(self.graph.get("scene_id", "profiled_scene")))
        return ProfileEditResult(
            graph=self.graph,
            requested_profile=copy.deepcopy(self.profile),
            achieved_profile=achieved,
            constraints=copy.deepcopy(self.constraints),
            graph_edits=copy.deepcopy(self.edits),
        )

    def _add_layered_drawer_occlusions(self) -> None:
        """Add directional open-state occlusion for three vertically stacked drawers.

        These structural edges are added after ordinary profile occlusion is
        normalised so they do not consume the profile's single-occluder budget
        and do not remove the drawers' normal placement edges.
        """

        layer_markers = ("第一层", "第二层", "第三层")
        for parent_id in sorted(self._node_ids()):
            part_ids = self._direct_part_ids(parent_id)
            ordered: list[str] = []
            for marker in layer_markers:
                matches = [
                    part_id
                    for part_id in part_ids
                    if marker in str(self._node(part_id).get("name", part_id))
                    and self._is_container(self._node(part_id))
                    and self._is_openable(self._node(part_id))
                ]
                if len(matches) != 1:
                    ordered = []
                    break
                ordered.append(matches[0])
            if len(ordered) != 3:
                continue

            for upper_index, source_id in enumerate(ordered[:-1]):
                for target_id in ordered[upper_index + 1 :]:
                    existing = next(
                        (
                            edge
                            for edge in self.edges
                            if isinstance(edge, dict)
                            and str(edge.get("from", edge.get("source"))) == source_id
                            and str(edge.get("to", edge.get("target"))) == target_id
                            and self._relation(edge) == "OCCLUDES"
                        ),
                        None,
                    )
                    if existing is not None:
                        existing["resolution_action"] = "close"
                        continue
                    edge = {
                        "from": source_id,
                        "to": target_id,
                        "relation": "OCCLUDES",
                        "resolution_action": "close",
                    }
                    self.edges.append(edge)
                    self.edits.append({"type": "add_layered_drawer_occlusion", **copy.deepcopy(edge)})

    def _apply_sample_identity(self) -> None:
        if self.num_samples <= 1:
            return
        metadata = self.graph.setdefault("metadata", {})
        original_scene_id = str(self.graph.get("scene_id", "profiled_scene"))
        original_env_id = self.graph.get("env_id", original_scene_id)
        suffix = f"profile_{self.sample_index + 1:03d}"
        metadata.setdefault("source_scene_id", original_scene_id)
        metadata.setdefault("source_env_id", original_env_id)
        self.graph["scene_id"] = f"{original_scene_id}_{suffix}"
        self.graph["env_id"] = f"{original_env_id}_{suffix}"

    def _apply_spatial(self) -> dict[str, Any]:
        section = self._section("spatial")
        num_occluded = max(
            self._int(section, "num_occluded_objects"),
            self._int(section, "occluded_objects"),
        )
        requested_depth = max(self._int(section, "occlusion_depth"), self._int(section, "max_depth"))
        occlusion_depth = max(requested_depth, 1 if num_occluded > 0 else 0)
        num_decomposed = max(
            self._int(section, "num_decomposed_parents"),
            self._int(section, "decomposed_parents"),
            self._int(section, "num_decompositions"),
        )
        constraints: dict[str, Any] = {}

        if num_decomposed > 0:
            constraints["decomposition"] = self._apply_spatial_decomposition(num_decomposed)

        if self.primary_object_id is not None:
            constraints["object"] = self.primary_object_id

        if num_occluded > 0:
            target_id = self._require_primary_object()
            constraints["object"] = target_id
            victims = self._ensure_spatial_victims(num_occluded)
            victim_ids = set(victims)
            chains = []
            for victim_id in victims:
                blockers = self._ensure_blocker_nodes(occlusion_depth, exclude=victim_ids, target_id=victim_id)
                previous = victim_id
                layers = []
                for blocker_id in blockers:
                    layers.append(self._add_spatial_occlusion_layer(blocker_id, previous))
                    previous = blocker_id
                chains.append({"object": victim_id, "depth": len(layers), "layers": layers})
            achieved_chains = [item for item in chains if item["depth"] > 0]
            constraints["occlusion"] = {
                "requested_num_occluded_objects": num_occluded,
                "requested_depth": occlusion_depth,
                "num_occluded_objects": len(achieved_chains),
                "depth": min((item["depth"] for item in achieved_chains), default=0),
                "objects": chains,
            }

        return constraints

    def _apply_spatial_decomposition(self, requested: int) -> dict[str, Any]:
        parent_ids = self._ensure_decomposable_parents(requested)
        parents = []
        for parent_id in parent_ids:
            record = self._decompose_parent(parent_id)
            if self.primary_object_id == parent_id and record.get("parts"):
                replacement = str(record["parts"][0])
                self.primary_object_id = replacement
                record["primary_object_replaced_by"] = replacement
            parents.append(record)
        achieved = [record for record in parents if self._is_parent_decomposed(str(record.get("parent")))]
        return {
            "requested_num_decomposed_parents": requested,
            "num_decomposed_parents": len(achieved),
            "parents": parents,
        }

    def _achieved_profile(self) -> dict[str, Any]:
        achieved: dict[str, Any] = {}
        if self._enabled("spatial"):
            spatial = self.constraints.get("spatial", {}).get("occlusion", {})
            decomposition = self.constraints.get("spatial", {}).get("decomposition", {})
            profile_depths = {
                str(item.get("object")): self._occlusion_depth(str(item.get("object")))
                for item in spatial.get("objects", [])
                if item.get("object")
            }
            profile_victims = [node_id for node_id, depth in profile_depths.items() if depth > 0]
            decomposed_parent_ids = [
                str(item.get("parent"))
                for item in decomposition.get("parents", [])
                if item.get("parent") and self._is_parent_decomposed(str(item.get("parent")))
            ]
            achieved["spatial"] = {
                "num_occluded_objects": len(profile_victims),
                "occlusion_depth": min((depth for depth in profile_depths.values() if depth > 0), default=0),
                "occlusion_depths": profile_depths,
                "primary_object_occlusion_depth": (
                    self._occlusion_depth(self.primary_object_id) if self.primary_object_id is not None else 0
                ),
                "num_decomposed_parents": len(decomposed_parent_ids),
                "decomposed_parents": decomposed_parent_ids,
            }
        return achieved

    def _section(self, name: str) -> dict[str, Any]:
        raw = self.profile.get(name, {})
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"profile.{name} must be an object")
        return raw

    def _enabled(self, name: str) -> bool:
        section = self._section(name)
        return bool(section) and bool(section.get("enabled", True))

    def _needs_primary_object(self) -> bool:
        if self._enabled("spatial"):
            section = self._section("spatial")
            return max(self._int(section, "num_occluded_objects"), self._int(section, "occluded_objects")) > 0
        return False

    @staticmethod
    def _int(section: dict[str, Any], key: str, default: int = 0) -> int:
        value = section.get(key, default)
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"profile value {key!r} must be an integer") from exc
        return max(parsed, 0)

    def _validate_unique_node_ids(self) -> None:
        seen = set()
        for node in self.nodes:
            node_id = node.get("id")
            if node_id is None:
                raise ValueError(f"node is missing id: {node}")
            if str(node_id) in seen:
                raise ValueError(f"duplicate node id: {node_id}")
            seen.add(str(node_id))

    def _node(self, node_id: str) -> dict[str, Any]:
        for node in self.nodes:
            if str(node.get("id")) == str(node_id):
                return node
        raise KeyError(node_id)

    def _node_ids(self) -> set[str]:
        return {str(node.get("id")) for node in self.nodes}

    def _pick_primary_object(self) -> str | None:
        hinted = self.profile.get("target_object") or self.profile.get("object")
        if hinted is not None:
            resolved = self._resolve_node_id(str(hinted))
            if resolved is not None:
                if not self._can_participate_external(resolved):
                    raise ValueError(
                        f"target_object {hinted!r} conflicts with existing parent/part external relations"
                    )
                if self._needs_primary_object() and not self._can_be_occlusion_target(resolved):
                    raise ValueError(f"target_object {hinted!r} cannot be an occlusion target")
                return resolved
        candidates = [
            node
            for node in self.nodes
            if self._is_grabbable(node)
            and self._can_be_occlusion_target(str(node["id"]))
            and self._can_participate_external(str(node["id"]))
        ]
        if not candidates:
            return None
        return str(self.rng.choice(candidates)["id"])

    def _resolve_node_id(self, value: str) -> str | None:
        for node in self.nodes:
            if str(node.get("id")) == value or str(node.get("name")) == value:
                return str(node["id"])
        return None

    def _require_primary_object(self) -> str:
        if self.primary_object_id is None:
            raise ValueError("profile needs at least one grabbable/movable object in the view graph")
        return self.primary_object_id

    def _ensure_spatial_victims(self, count: int) -> list[str]:
        victims = [
            str(node["id"])
            for node in self.nodes
            if self._is_grabbable(node)
            and self._can_be_occlusion_target(str(node["id"]))
            and self._can_participate_external(str(node["id"]))
        ]
        self.rng.shuffle(victims)
        if self.primary_object_id and self.primary_object_id in victims:
            victims.remove(self.primary_object_id)
            victims.insert(0, self.primary_object_id)
        return self._select_part_exclusive(victims, count)

    def _ensure_blocker_nodes(self, count: int, exclude: set[str], target_id: str) -> list[str]:
        excluded = self._expand_part_exclusions(exclude)
        candidates = [
            str(node["id"])
            for node in self.nodes
            if str(node.get("id")) not in excluded
            and not self._is_room(node)
            and not self._is_surface(node)
            and self._can_occlude(node)
            and self._can_participate_external(str(node["id"]))
        ]
        self.rng.shuffle(candidates)
        selected: list[str] = []
        blocked = set(excluded)
        current_target_id = target_id
        for index in range(count):
            must_be_targetable = index < count - 1
            for node_id in candidates:
                if node_id in blocked:
                    continue
                if must_be_targetable and not self._can_be_occlusion_target(node_id):
                    continue
                if not self._can_participate_external(node_id, selected):
                    continue
                if not self._allows_edge(node_id, current_target_id, "OCCLUDES"):
                    continue
                if self._occlusion_resolution_action(self._node(node_id)) == "open" and not self._container_has_capacity_for(
                    node_id,
                    current_target_id,
                ):
                    continue
                selected.append(node_id)
                blocked.update(self._part_conflict_set(node_id))
                current_target_id = node_id
                break
            else:
                break
        return selected

    def _ensure_decomposable_parents(self, count: int) -> list[str]:
        candidates = [
            str(node["id"])
            for node in self.nodes
            if self._is_decomposable_parent(node)
            and not self._is_parent_decomposed(str(node["id"]))
        ]
        hinted = self.profile.get("target_object") or self.profile.get("object")
        hinted_id = self._resolve_node_id(str(hinted)) if hinted is not None else None
        self.rng.shuffle(candidates)
        if hinted_id in candidates:
            candidates.remove(hinted_id)
            candidates.insert(0, hinted_id)

        selected: list[str] = []
        blocked: set[str] = set()
        for parent_id in candidates:
            if parent_id in blocked:
                continue
            selected.append(parent_id)
            blocked.update(self._part_family(parent_id))
            if len(selected) >= count:
                break
        return selected

    def _decompose_parent(self, parent_id: str) -> dict[str, Any]:
        parts = self._direct_part_ids(parent_id)
        removed_edges = self._remove_external_edges_for_node(parent_id)
        removed_parent_field = self._remove_parent_field_for_decomposition(parent_id)
        removed_states = self._remove_decomposed_assembly_states(parent_id, parts)
        added_decomposed_state = self._mark_decomposed_state(parent_id)
        added_edges: list[dict[str, Any]] = []

        if removed_parent_field is not None:
            added_edges.extend(
                self._add_decomposed_relation_for_parts(
                    parent_id,
                    parts,
                    str(removed_parent_field["relation"]),
                    str(removed_parent_field["to"]),
                    parent_was_source=True,
                )
            )
        for edge in removed_edges:
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            relation = self._relation(edge)
            if source == parent_id:
                added_edges.extend(
                    self._add_decomposed_relation_for_parts(
                        parent_id,
                        parts,
                        relation,
                        target,
                        parent_was_source=True,
                    )
                )
            elif target == parent_id:
                added_edges.extend(
                    self._add_decomposed_relation_for_parts(
                        parent_id,
                        parts,
                        relation,
                        source,
                        parent_was_source=False,
                    )
                )

        record = {
            "type": "decompose_parent",
            "parent": parent_id,
            "parts": parts,
            "removed_parent_edges": removed_edges,
            "removed_parent_field": removed_parent_field,
            "removed_assembly_states": removed_states,
            "added_decomposed_state": added_decomposed_state,
            "added_part_edges": added_edges,
        }
        self.edits.append(copy.deepcopy(record))
        return record

    def _mark_decomposed_state(self, parent_id: str) -> bool:
        node = self._node(parent_id)
        states = node.setdefault("states", [])
        normalized = {_normalize(state) for state in states}
        if DECOMPOSED_STATE in normalized:
            return False
        states.append(DECOMPOSED_STATE)
        self.edits.append({"type": "mark_decomposed_state", "parent": parent_id, "state": DECOMPOSED_STATE})
        return True

    def _remove_decomposed_assembly_states(self, parent_id: str, parts: list[str]) -> list[dict[str, Any]]:
        removed: list[dict[str, Any]] = []
        for node_id in [parent_id, *parts]:
            if node_id not in self._node_ids():
                continue
            node = self._node(node_id)
            states = node.get("states")
            if not isinstance(states, list):
                continue
            kept_states = []
            removed_states = []
            for state in states:
                if _normalize(state) in DECOMPOSED_ASSEMBLY_STATES:
                    removed_states.append(state)
                else:
                    kept_states.append(state)
            if not removed_states:
                continue
            if kept_states:
                node["states"] = kept_states
            else:
                node.pop("states", None)
            removed.append({"node": node_id, "states": removed_states})
        if removed:
            self.edits.append(
                {
                    "type": "remove_decomposed_assembly_states",
                    "parent": parent_id,
                    "removed": copy.deepcopy(removed),
                }
            )
        return removed

    def _add_decomposed_relation_for_parts(
        self,
        parent_id: str,
        parts: list[str],
        relation: str,
        other_id: str,
        *,
        parent_was_source: bool,
    ) -> list[dict[str, Any]]:
        relation = _normalize(relation)
        part_ids = list(parts)
        if relation in OCCLUSION_RELATIONS and parent_was_source:
            part_ids = [part_id for part_id in part_ids if self._can_occlude(self._node(part_id))][:1]

        added = []
        family = self._part_family(parent_id)
        for part_id in part_ids:
            if other_id in family:
                continue
            source_id = part_id if parent_was_source else other_id
            target_id = other_id if parent_was_source else part_id
            if source_id == target_id:
                continue
            if source_id not in self._node_ids() or target_id not in self._node_ids():
                continue
            if relation in OCCLUSION_RELATIONS:
                if not self._allows_edge(source_id, target_id, relation):
                    continue
                self._remove_incoming_occlusion_edges(target_id)
            if self._add_edge(source_id, target_id, relation):
                added.append({"from": source_id, "to": target_id, "relation": relation})
        return added

    def _add_spatial_occlusion_layer(self, occluder_id: str, target_id: str) -> dict[str, Any]:
        occluder = self._node(occluder_id)
        resolution_action = self._occlusion_resolution_action(occluder)
        if resolution_action is None:
            raise ValueError(f"node {occluder_id!r} cannot occlude because it lacks an occlusion affordance")
        if not self._can_be_occlusion_target(target_id):
            raise ValueError(f"node {target_id!r} cannot be an occlusion target")
        if not self._allows_edge(occluder_id, target_id, "OCCLUDES"):
            raise ValueError(f"OCCLUDES from {occluder_id!r} to {target_id!r} is disallowed")
        if resolution_action == "open" and not self._container_has_capacity_for(occluder_id, target_id):
            raise ValueError(f"container {occluder_id!r} is at max_items capacity")
        if resolution_action == "open":
            self._ensure_closed_container_state(occluder_id)
        removed = self._remove_incoming_occlusion_edges(target_id)
        removed_visible = self._remove_visible_spatial_edges_for_occlusion(target_id)
        self._add_edge(occluder_id, target_id, "OCCLUDES")
        return {
            "type": "occluder",
            "blocker": occluder_id,
            "target": target_id,
            "relation": "OCCLUDES",
            "resolution_action": resolution_action,
            "removed_previous_occluders": removed,
            "removed_visible_spatial_edges": removed_visible,
        }

    def _select_part_exclusive(
        self,
        candidates: list[str],
        count: int,
        exclude: set[str] | None = None,
    ) -> list[str]:
        selected: list[str] = []
        blocked = self._expand_part_exclusions(exclude or set())
        for node_id in candidates:
            if node_id in blocked:
                continue
            if not self._can_participate_external(node_id, selected):
                continue
            selected.append(node_id)
            blocked.update(self._part_conflict_set(node_id))
            if len(selected) >= count:
                break
        return selected

    def _expand_part_exclusions(self, node_ids: set[str]) -> set[str]:
        expanded: set[str] = set()
        for node_id in node_ids:
            expanded.update(self._part_conflict_set(node_id))
        return expanded

    def _part_conflict_set(self, node_id: str) -> set[str]:
        if node_id not in self._node_ids():
            return {node_id}
        return {node_id} | self._part_ancestors(node_id) | self._part_descendants(node_id)

    def _part_family(self, node_id: str) -> set[str]:
        if node_id not in self._node_ids():
            return {node_id}
        root = self._part_root(node_id)
        family = {root}
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                current = str(node["id"])
                parent = self._part_parent(current)
                if parent in family and current not in family:
                    family.add(current)
                    changed = True
        return family

    def _part_ancestors(self, node_id: str) -> set[str]:
        ancestors: set[str] = set()
        current = node_id
        while current not in ancestors:
            parent = self._part_parent(current)
            if parent is None or parent not in self._node_ids():
                return ancestors
            ancestors.add(parent)
            current = parent
        return ancestors

    def _part_descendants(self, node_id: str) -> set[str]:
        descendants: set[str] = set()
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                current = str(node["id"])
                if current == node_id or current in descendants:
                    continue
                parent = self._part_parent(current)
                if parent == node_id or parent in descendants:
                    descendants.add(current)
                    changed = True
        return descendants

    def _part_root(self, node_id: str) -> str:
        current = node_id
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            parent = self._part_parent(current)
            if parent is None or parent not in self._node_ids():
                return current
            current = parent
        return current

    def _part_parent(self, node_id: str) -> str | None:
        if node_id not in self._node_ids():
            return None
        node = self._node(node_id)
        raw_parent = node.get("part_of")
        if raw_parent is not None:
            resolved = self._resolve_node_id(str(raw_parent))
            if resolved is not None:
                return resolved
        for edge in self.edges:
            if str(edge.get("from", edge.get("source"))) == node_id and self._relation(edge) == "PART_OF":
                parent = str(edge.get("to", edge.get("target")))
                if parent in self._node_ids():
                    return parent
        return None

    def _direct_part_ids(self, parent_id: str) -> list[str]:
        parts = []
        for node in self.nodes:
            node_id = str(node["id"])
            if node_id != parent_id and self._part_parent(node_id) == parent_id:
                parts.append(node_id)
        return parts

    def _has_external_relation(self, node_id: str) -> bool:
        for edge in self.edges:
            if not isinstance(edge, dict) or self._relation(edge) == "PART_OF":
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            if source == node_id or target == node_id:
                return True
        return False

    def _has_implicit_parent_relation(self, node_id: str) -> bool:
        if node_id not in self._node_ids():
            return False
        raw_parent = self._node(node_id).get("parent")
        return raw_parent is not None and self._resolve_node_id(str(raw_parent)) is not None

    def _has_transferable_parent_relation(self, node_id: str) -> bool:
        return self._has_external_relation(node_id) or self._has_implicit_parent_relation(node_id)

    def _is_parent_decomposed(self, parent_id: str) -> bool:
        if parent_id not in self._node_ids():
            return False
        parts = self._direct_part_ids(parent_id)
        if not parts:
            return False
        if self._has_external_relation(parent_id) or self._has_implicit_parent_relation(parent_id):
            return False
        return all(self._has_external_relation(part_id) for part_id in parts)

    def _ensure_container(self, exclude: set[str] | None = None) -> str:
        exclude = exclude or set()
        hinted = self.profile.get("container") or self._section("spatial").get("container")
        if hinted is not None:
            resolved = self._resolve_node_id(str(hinted))
            if resolved is not None and resolved not in exclude and self._can_participate_external(resolved):
                self._ensure_openable_container(resolved)
                return resolved
        excluded = self._expand_part_exclusions(exclude)
        candidates = [
            node
            for node in self.nodes
            if str(node.get("id")) not in excluded
            and self._is_container(node)
            and self._can_participate_external(str(node["id"]))
        ]
        if candidates:
            node = self.rng.choice(candidates)
            node_id = str(node.get("id"))
            self._ensure_openable_container(node_id)
            return node_id
        container_id = self._unique_id("profile_container")
        node = {
            "id": container_id,
            "name": container_id,
            "category": "container",
            "properties": ["CONTAINERS", "CAN_OPEN"],
            "states": ["CLOSED"],
            "source": "profile_editor",
        }
        self.nodes.append(node)
        self.edits.append({"type": "add_node", "id": container_id, "category": "container"})
        self._move(container_id, "ON", self._ensure_surface())
        return container_id

    def _ensure_surface(self) -> str:
        candidates = [node for node in self.nodes if self._is_surface(node)]
        candidates = [node for node in candidates if self._can_participate_external(str(node["id"]))]
        if candidates:
            node = self.rng.choice(candidates)
            return str(node["id"])
        for node in sorted(self.nodes, key=lambda item: (str(item.get("name", item.get("id"))), str(item.get("id")))):
            if self._is_surface(node) and self._can_participate_external(str(node["id"])):
                return str(node["id"])
        surface_id = self._unique_id("profile_surface")
        node = {
            "id": surface_id,
            "name": surface_id,
            "category": "surface",
            "properties": ["SURFACES"],
            "source": "profile_editor",
        }
        self.nodes.append(node)
        self.edits.append({"type": "add_node", "id": surface_id, "category": "surface"})
        return surface_id

    def _create_grabbable_node(
        self,
        prefix: str,
        properties: tuple[str, ...] = ("GRABBABLE", "MOVABLE"),
    ) -> str:
        node_id = self._unique_id(prefix)
        node = {
            "id": node_id,
            "name": node_id,
            "category": "object",
            "properties": list(properties),
            "source": "profile_editor",
        }
        self.nodes.append(node)
        self.edits.append({"type": "add_node", "id": node_id, "category": "object"})
        self._move(node_id, "ON", self._ensure_surface())
        return node_id

    def _unique_id(self, prefix: str) -> str:
        existing = self._node_ids()
        index = 1
        while f"{prefix}_{index}" in existing:
            index += 1
        return f"{prefix}_{index}"

    def _move(self, source_id: str, relation: str, target_id: str) -> None:
        self._require_external_participant(source_id, additional={target_id})
        self._require_external_participant(target_id, additional={source_id})
        self._remove_location_edges(source_id)
        self._add_edge(source_id, target_id, relation)

    def _remove_location_edges(self, source_id: str) -> None:
        remaining = []
        removed = []
        for edge in self.edges:
            if str(edge.get("from", edge.get("source"))) == source_id and self._relation(edge) in LOCATION_RELATIONS:
                removed.append(copy.deepcopy(edge))
                continue
            remaining.append(edge)
        if removed:
            self.edges[:] = remaining
            self.edits.append({"type": "remove_location_edges", "source": source_id, "edges": removed})

    def _remove_incoming_occlusion_edges(self, target_id: str) -> list[dict[str, Any]]:
        remaining = []
        removed = []
        for edge in self.edges:
            if str(edge.get("to", edge.get("target"))) == target_id and self._relation(edge) in OCCLUSION_RELATIONS:
                removed.append(copy.deepcopy(edge))
                continue
            remaining.append(edge)
        if removed:
            self.edges[:] = remaining
            self.edits.append({"type": "replace_occluder", "target": target_id, "removed": removed})
        return removed

    def _remove_visible_spatial_edges_for_occlusion(self, target_id: str) -> list[dict[str, Any]]:
        remaining = []
        removed = []
        for edge in self.edges:
            if not isinstance(edge, dict):
                remaining.append(edge)
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            relation = self._relation(edge)
            remove = False
            if source == target_id and relation in VISIBLE_PLACEMENT_RELATIONS:
                remove = True
            elif target == target_id and relation in {"CONTAINS", "HAS_INSIDE", "SUPPORTS", "HAS_ON"}:
                remove = True
            elif (source == target_id or target == target_id) and relation in VISIBLE_RELATIVE_RELATIONS:
                remove = True
            if remove:
                removed.append(copy.deepcopy(edge))
                continue
            remaining.append(edge)
        if removed:
            self.edges[:] = remaining
            self.edits.append({"type": "remove_visible_spatial_edges_for_occlusion", "target": target_id, "edges": removed})
        return removed

    def _remove_external_edges_for_node(self, node_id: str) -> list[dict[str, Any]]:
        remaining = []
        removed = []
        for edge in self.edges:
            if not isinstance(edge, dict) or self._relation(edge) == "PART_OF":
                remaining.append(edge)
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            if source == node_id or target == node_id:
                removed.append(copy.deepcopy(edge))
                continue
            remaining.append(edge)
        if removed:
            self.edges[:] = remaining
            self.edits.append({"type": "remove_parent_external_edges_for_decomposition", "parent": node_id, "edges": removed})
        return removed

    def _remove_parent_field_for_decomposition(self, node_id: str) -> dict[str, Any] | None:
        node = self._node(node_id)
        raw_parent = node.get("parent")
        if raw_parent is None:
            return None
        parent_id = self._resolve_node_id(str(raw_parent))
        if parent_id is None:
            return None
        relation = "INSIDE" if self._is_container(self._node(parent_id)) else "ON"
        del node["parent"]
        record = {"from": node_id, "to": parent_id, "relation": relation, "field": "parent"}
        self.edits.append({"type": "remove_parent_field_for_decomposition", "parent": node_id, "removed": record})
        return record

    def _normalize_occlusion_edges(self) -> None:
        first_incoming: dict[str, dict[str, Any]] = {}
        duplicate_incoming: dict[str, list[dict[str, Any]]] = {}
        forbidden_edges: list[dict[str, Any]] = []
        kept_edges = []
        for edge in self.edges:
            if not isinstance(edge, dict) or self._relation(edge) not in OCCLUSION_RELATIONS:
                kept_edges.append(edge)
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            relation = self._relation(edge)
            if not self._allows_edge(source, target, relation):
                forbidden_edges.append(copy.deepcopy(edge))
                continue
            if target not in first_incoming:
                first_incoming[target] = edge
                kept_edges.append(edge)
                continue
            duplicate_incoming.setdefault(target, []).append(copy.deepcopy(edge))
        if duplicate_incoming:
            self.edges[:] = kept_edges
            self.edits.append({"type": "dedupe_incoming_occluders", "removed": duplicate_incoming})
        if forbidden_edges:
            self.edges[:] = kept_edges
            self.edits.append({"type": "remove_forbidden_occlusion_edges", "removed": forbidden_edges})
        for target_id in sorted(first_incoming):
            self._remove_visible_spatial_edges_for_occlusion(target_id)

    def _add_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        relation = _normalize(relation)
        for edge in self.edges:
            if (
                str(edge.get("from", edge.get("source"))) == source_id
                and str(edge.get("to", edge.get("target"))) == target_id
                and self._relation(edge) == relation
            ):
                return False
        if relation != "PART_OF":
            self._require_external_participant(source_id, additional={target_id})
            self._require_external_participant(target_id, additional={source_id})
        self.edges.append({"from": source_id, "to": target_id, "relation": relation})
        self.edits.append({"type": "add_edge", "from": source_id, "to": target_id, "relation": relation})
        return True

    def _allows_edge(self, source_id: str, target_id: str, relation: str) -> bool:
        if self.placement_edge_constraints.is_empty():
            return True
        if source_id not in self._node_ids() or target_id not in self._node_ids():
            return True
        source = self._node(source_id)
        target = self._node(target_id)
        return self.placement_edge_constraints.allows(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            source_name=str(source.get("name")) if source.get("name") is not None else None,
            target_name=str(target.get("name")) if target.get("name") is not None else None,
        )

    def _container_has_capacity_for(self, container_id: str, target_id: str) -> bool:
        if container_id not in self._node_ids():
            return True
        node = self._node(container_id)
        if not self._is_container(node):
            return True
        max_items = parse_max_items(node)
        if max_items is None:
            return True
        return self._container_item_count(container_id, exclude={target_id}) < max_items

    def _container_item_count(self, container_id: str, exclude: set[str] | None = None) -> int:
        exclude = exclude or set()
        items: set[str] = set()
        for node_id in self._node_ids():
            if node_id == container_id or node_id in exclude:
                continue
            relation, target = self._location_of(node_id)
            if relation in {"INSIDE", "IN"} and target == container_id:
                items.add(node_id)
        for edge in self.edges:
            if not isinstance(edge, dict) or self._relation(edge) not in OCCLUSION_RELATIONS:
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            resolution_action = edge.get("resolution_action")
            if resolution_action is None and source in self._node_ids():
                resolution_action = self._occlusion_resolution_action(self._node(source))
            resolution_action = str(resolution_action or "").strip().lower().replace("-", "_")
            if source == container_id and resolution_action != "open":
                continue
            if source == container_id and target != container_id and target not in exclude:
                items.add(target)
        return len(items)

    def _location_of(self, node_id: str) -> tuple[str | None, str | None]:
        for edge in self.edges:
            if str(edge.get("from", edge.get("source"))) == node_id and self._relation(edge) in LOCATION_RELATIONS:
                return self._relation(edge), str(edge.get("to", edge.get("target")))
        node = self._node(node_id)
        parent = node.get("parent")
        if parent is not None and str(parent) in self._node_ids():
            parent_node = self._node(str(parent))
            return ("INSIDE" if self._is_container(parent_node) else "ON"), str(parent)
        return None, None

    def _require_external_participant(self, node_id: str, additional: set[str] | None = None) -> None:
        conflicts = self._external_participant_conflicts(node_id, additional or set())
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise ValueError(f"node {node_id!r} conflicts with parent/part external participant(s): {joined}")

    def _can_participate_external(self, node_id: str, additional: Iterable[str] | None = None) -> bool:
        return not self._external_participant_conflicts(node_id, set(additional or ()))

    def _external_participant_conflicts(self, node_id: str, additional: set[str]) -> set[str]:
        if node_id not in self._node_ids():
            return set()
        participants = self._external_relation_participants() | set(additional)
        participants.discard(node_id)
        return (self._part_ancestors(node_id) | self._part_descendants(node_id)) & participants

    def _external_relation_participants(self) -> set[str]:
        node_ids = self._node_ids()
        participants: set[str] = set()
        for edge in self.edges:
            if not isinstance(edge, dict) or self._relation(edge) == "PART_OF":
                continue
            source = str(edge.get("from", edge.get("source")))
            target = str(edge.get("to", edge.get("target")))
            if source in node_ids:
                participants.add(source)
            if target in node_ids:
                participants.add(target)
        return participants

    def _occlusion_depth(self, node_id: str) -> int:
        def visit(current: str, seen: set[str]) -> int:
            depths = []
            for edge in self.edges:
                if str(edge.get("to", edge.get("target"))) != current or self._relation(edge) not in OCCLUSION_RELATIONS:
                    continue
                source = str(edge.get("from", edge.get("source")))
                if source in seen:
                    continue
                depths.append(1 + visit(source, seen | {source}))
            return max(depths, default=0)

        return visit(node_id, {node_id})

    def _ensure_openable_container(self, node_id: str) -> None:
        node = self._node(node_id)
        self._ensure_property(node, "CONTAINERS")
        self._ensure_property(node, "CAN_OPEN")
        states = node.setdefault("states", [])
        normalized = {_normalize(item) for item in states}
        if "OPEN" not in normalized and "CLOSED" not in normalized:
            states.append("CLOSED")
            self.edits.append({"type": "add_state", "node": node_id, "state": "CLOSED"})

    def _ensure_closed_container_state(self, node_id: str) -> None:
        node = self._node(node_id)
        if not self._is_container(node):
            raise ValueError(f"node {node_id!r} is not a container")
        states = node.setdefault("states", [])
        normalized = [_normalize(item) for item in states]
        if "CLOSED" in normalized:
            return
        if "OPEN" in normalized:
            raise ValueError(f"node {node_id!r} is explicitly OPEN and cannot be used as an open-resolved occluder")
        states[:] = [state for state in states if _normalize(state) != "OPEN"]
        states.append("CLOSED")
        self.edits.append({"type": "set_state", "node": node_id, "state": "CLOSED"})

    def _ensure_property(self, node: dict[str, Any], prop: str) -> None:
        prop = _normalize(prop)
        properties = node.setdefault("properties", [])
        if prop not in {_normalize(item) for item in properties}:
            properties.append(prop)
            self.edits.append({"type": "add_property", "node": str(node.get("id")), "property": prop})

    @staticmethod
    def _relation(edge: dict[str, Any]) -> str:
        return _normalize(edge.get("relation", edge.get("relation_type", "")))

    @staticmethod
    def _has_property(node: dict[str, Any], *properties: str) -> bool:
        wanted = {_normalize(prop) for prop in properties}
        return any(_normalize(prop) in wanted for prop in node.get("properties", []))

    def _is_grabbable(self, node: dict[str, Any]) -> bool:
        return not self._is_room(node) and self._has_property(node, "GRABBABLE") and self._has_property(node, "MOVABLE")

    def _is_decomposable_parent(self, node: dict[str, Any]) -> bool:
        node_id = str(node.get("id"))
        return (
            not self._is_room(node)
            and self._has_property(node, "DECOMPOSABLE", "CAN_DECOMPOSE")
            and bool(self._direct_part_ids(node_id))
            and self._has_transferable_parent_relation(node_id)
        )

    @staticmethod
    def _is_room(node: dict[str, Any]) -> bool:
        return str(node.get("category", "")).lower() in {"room", "rooms"}

    def _is_container(self, node: dict[str, Any]) -> bool:
        category = str(node.get("category", "")).lower()
        return self._has_property(node, "CONTAINERS") or category in {"container", "receptacle"}

    def _can_occlude(self, node: dict[str, Any]) -> bool:
        return self._occlusion_resolution_action(node) is not None

    def _occlusion_resolution_action(self, node: dict[str, Any]) -> str | None:
        if self._is_room(node) or self._is_surface(node):
            return None
        if not self._has_property(node, "OCCLUDER"):
            return None
        if self._is_container(node):
            if self._is_openable(node) and self._is_closed_container(node):
                return "open"
            if self._has_property(node, "MOVABLE"):
                return "move_aside"
            return None
        if self._has_property(node, "MOVABLE"):
            return "move_aside"
        return None

    def _is_openable(self, node: dict[str, Any]) -> bool:
        states = {_normalize(state) for state in node.get("states", [])}
        return self._has_property(node, "CAN_OPEN") or bool(states.intersection({"OPEN", "CLOSED"}))

    def _can_be_occlusion_target(self, node_id: str) -> bool:
        if node_id not in self._node_ids():
            return False
        return not self._is_static(self._node(node_id))

    def _is_static(self, node: dict[str, Any]) -> bool:
        return self._has_property(node, *STATIC_PROPERTIES)

    def _is_closed_container(self, node: dict[str, Any]) -> bool:
        if not self._is_container(node):
            return False
        states = {_normalize(state) for state in node.get("states", [])}
        if "CLOSED" in states:
            return True
        return self._has_property(node, "CAN_OPEN") and "OPEN" not in states

    def _is_surface(self, node: dict[str, Any]) -> bool:
        category = str(node.get("category", "")).lower()
        return self._has_property(node, "SURFACE", "SURFACES") or category in {
            "surface",
            "furniture",
            "table",
            "counter",
            "workspace",
        }


def _normalize(value: Any) -> str:
    return str(value).strip().upper().replace(" ", "_")


def _difficulty_tags(achieved: dict[str, Any]) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}
    spatial = achieved.get("spatial")
    if isinstance(spatial, dict):
        tags["spatial"] = [
            f"spatial.num_occluded_objects={int(spatial.get('num_occluded_objects', 0) or 0)}",
            f"spatial.occlusion_depth={int(spatial.get('occlusion_depth', 0) or 0)}",
            f"spatial.num_decomposed_parents={int(spatial.get('num_decomposed_parents', 0) or 0)}",
        ]
    return tags
