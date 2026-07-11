from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from openai import APITimeoutError, OpenAI

from .models import ViewGraph


MATERIAL_CORE_FIELDS = {"id", "name", "category", "properties", "states", "part_of", "parent", "parts"}
NODE_CORE_FIELDS = {"name", "category", "properties", "states", "parts", "part_of"}


@dataclass
class TaskViewGraphSynthesisConfig:
    materials: tuple[str, ...]
    scene: str
    layout: str
    arms: str
    material_properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_hint: str | None = None
    scene_id: str | None = None
    env_id: str | int | None = None
    provider: str = "qwen"
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    api_base_url: str | None = None
    timeout_seconds: int = 60
    enable_thinking: bool = False


def synthesize_task_view_graph(config: TaskViewGraphSynthesisConfig) -> dict[str, Any]:
    """Create a view graph package from a material list and activity hint.

    The returned package has one top-level JSON field: `view_graph`.
    """

    if config.layout not in {"indoor", "tabletop"}:
        raise ValueError(f"layout must be indoor or tabletop, got {config.layout!r}")
    if config.arms not in {"single", "double"}:
        raise ValueError(f"arms must be single or double, got {config.arms!r}")
    materials = tuple(_clean_object_name(item) for item in config.materials if _clean_object_name(item))
    if not materials:
        raise ValueError("at least one material is required")

    material_properties = _normalise_material_properties(config.material_properties)
    effective = TaskViewGraphSynthesisConfig(
        **{**config.__dict__, "materials": materials, "material_properties": material_properties}
    )
    if effective.provider not in {"openai", "qwen", "compatible"}:
        raise ValueError(f"Unknown provider {effective.provider!r}; use openai, qwen, or compatible")

    package = _synthesize_task_view_graph_api(effective)
    _validate_task_view_graph_package(package, materials, material_properties)
    metadata = package["view_graph"].setdefault("metadata", {})
    metadata.setdefault("input_materials", list(materials))
    if effective.task_hint:
        metadata.setdefault("activity", effective.task_hint)
    return package


def write_view_graph_jsonl(graph: dict[str, Any], path: str | Path, append: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(graph, ensure_ascii=False) + "\n")


def write_task_view_graph_package(package: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(package, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_task_view_graph_prompt(config: TaskViewGraphSynthesisConfig) -> str:
    materials = ", ".join(config.materials)
    scene_id = config.scene_id or _default_scene_id(config.scene, config.layout)
    env_id = config.env_id if config.env_id is not None else scene_id
    activity = config.task_hint or config.scene
    material_properties = json.dumps(config.material_properties, ensure_ascii=False, sort_keys=True)
    input_tokens = {_clean_object_name(item) for item in config.materials}
    declared_parts = _declared_parts_prompt(config.material_properties, input_tokens)
    relation_skill = _read_prompt_skill("relation.md")
    properties_skill = _read_prompt_skill("properties.md")
    return f"""请根据输入物料和任务目标构建一个具身场景 view graph，并只输出严格 JSON。

只返回一个 JSON object。

输出结构：
{{
  "view_graph": {{
    "scene_id": string,
    "env_id": string or number,
    "layout": "indoor" or "tabletop",
    "robot": {{"arms": "single" or "double", "start": string}},
    "description": string,
    "nodes": [
      {{"id": string, "name": string, "category": string, "properties": [string], "room": optional string, "parent": optional string, "part_of": optional string, "states": optional [string], "source": optional "input" or "part"}}
    ],
    "edges": [
      {{"from": string, "to": string, "relation": string}}
    ]
  }}
}}

输入：
- 物料列表：{materials}
- 场景描述：{config.scene}
- 任务/目标：{activity}
- layout：{config.layout}
- 机器人手臂：{config.arms}
- scene_id：{scene_id}
- env_id：{env_id}
- 物料属性 JSON：{material_properties}
- 声明部件清单：{declared_parts}

硬规则：
- 你只负责构建 view graph。
- 任务/目标只作为构图输入，用来判断物体位置、状态和关系。
- 物料列表中的每个物料以及它的所有部件都必须出现在 view_graph.nodes 中。
- `物料属性 JSON` 中通过 `parts` 或 `part_of` 声明的部件必须逐个输出为 node，不能省略。
- 每个声明部件必须有一条 `PART_OF` edge 指向它的父物体或父部件。
- 输入物料节点的 source 标为 "input"。
- 输入物料是中文时，节点 `id` 和 `name` 必须保持原始中文物料名，不要翻译成英文。

{relation_skill}

{properties_skill}
"""


def _read_prompt_skill(filename: str) -> str:
    return (Path(__file__).with_name("prompt_skills") / filename).read_text(encoding="utf-8").strip()


def _synthesize_task_view_graph_api(config: TaskViewGraphSynthesisConfig) -> dict[str, Any]:
    model = config.model or os.environ.get("AUTO_EMBODIED_TASK_MODEL") or _default_model_for_provider(config.provider)
    content = _chat_completion_json_content(
        provider=config.provider,
        model=model,
        api_key=config.api_key,
        api_key_env=_api_key_env_for_provider(config.provider, config.api_key_env),
        api_base_url=_api_base_url_for_provider(config.provider, config.api_base_url),
        timeout_seconds=config.timeout_seconds,
        enable_thinking=config.enable_thinking,
        system_prompt=(
            "你生成机器可读的具身场景 view graph。只输出严格 JSON。"
        ),
        user_prompt=build_task_view_graph_prompt(config),
    )
    payload = _extract_json_object(content)
    package = payload if "view_graph" in payload else {"view_graph": payload}
    view_graph = package.setdefault("view_graph", {})
    view_graph.setdefault("metadata", {})["task_synthesis_provider"] = config.provider
    view_graph["metadata"]["task_synthesis_model"] = model
    return package


def _chat_completion_json_content(
    *,
    provider: str,
    model: str,
    api_key: str | None,
    api_key_env: str,
    api_base_url: str,
    timeout_seconds: int,
    enable_thinking: bool,
    system_prompt: str,
    user_prompt: str,
) -> str:
    key = api_key or os.environ.get(api_key_env)
    if not key:
        raise RuntimeError(f"Missing API key. Set {api_key_env} or pass api_key.")
    client = OpenAI(
        api_key=key,
        base_url=api_base_url,
        timeout=timeout_seconds,
    )
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if provider == "qwen":
        create_kwargs["extra_body"] = {"enable_thinking": enable_thinking}

    try:
        completion = client.chat.completions.create(
            **create_kwargs,
        )
    except (TimeoutError, APITimeoutError) as exc:
        raise RuntimeError(
            f"{provider} API request timed out after {timeout_seconds} seconds. "
            "Increase --timeout-seconds or use a faster model."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{provider} API request failed: {exc}") from exc

    if not completion.choices:
        raise RuntimeError(f"{provider} API returned no choices")
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError(f"{provider} API returned empty content")
    return content


def _validate_graph_dict(graph: dict[str, Any]) -> None:
    ViewGraph.from_dict(graph, fallback_scene_id=str(graph.get("scene_id", "generated_scene")))


def _validate_task_view_graph_package(
    package: dict[str, Any],
    input_materials: tuple[str, ...],
    material_properties: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not isinstance(package, dict):
        raise ValueError("View graph package must be a JSON object")
    package.pop("task_definition", None)
    package.pop("selected_materials", None)

    graph = package.get("view_graph")
    if not isinstance(graph, dict):
        raise ValueError("Package is missing view_graph object")
    removed_non_material_nodes, removed_non_material_edges = _remove_non_material_nodes(
        graph,
        input_materials,
        material_properties or {},
    )
    added_declared_part_nodes, added_declared_part_edges = _ensure_declared_part_nodes(
        graph,
        input_materials,
        material_properties or {},
    )
    removed_inside_edges = _remove_inside_edges(graph)
    removed_occlusion_edges = _remove_occlusion_edges(graph)
    removed_invalid_relation_edges = _remove_invalid_relation_affordance_edges(graph)
    removed_parent_part_conflict_edges = _remove_parent_part_external_conflict_edges(graph)
    _validate_graph_dict(graph)
    for material_id in input_materials:
        if not _material_present_or_represented_by_parts(graph, material_id):
            raise ValueError(f"Input material {material_id!r} is not present as a view_graph node or part nodes")

    graph_metadata = graph.setdefault("metadata", {})
    graph_metadata.pop("task_definition", None)
    graph_metadata.pop("selected_materials", None)
    if removed_non_material_nodes:
        graph_metadata["removed_non_material_nodes"] = removed_non_material_nodes
    if removed_non_material_edges:
        graph_metadata["removed_non_material_edges"] = removed_non_material_edges
    if added_declared_part_nodes:
        graph_metadata["added_declared_part_nodes"] = added_declared_part_nodes
    if added_declared_part_edges:
        graph_metadata["added_declared_part_edges"] = added_declared_part_edges
    if removed_inside_edges:
        graph_metadata["removed_inside_edges"] = removed_inside_edges
    if removed_occlusion_edges:
        graph_metadata["removed_occlusion_edges"] = removed_occlusion_edges
    if removed_invalid_relation_edges:
        graph_metadata["removed_invalid_relation_affordance_edges"] = removed_invalid_relation_edges
    if removed_parent_part_conflict_edges:
        graph_metadata["removed_parent_part_external_conflict_edges"] = removed_parent_part_conflict_edges


def _remove_non_material_nodes(
    graph: dict[str, Any],
    input_materials: tuple[str, ...],
    material_properties: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("view_graph.nodes must be a list")
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")

    input_tokens = {_clean_object_name(item) for item in input_materials}
    declared_part_tokens = _declared_part_tokens(material_properties, input_tokens)
    node_by_id = {
        str(node.get("id")): node
        for node in nodes
        if isinstance(node, dict) and node.get("id") is not None
    }
    allowed_ids: set[str] = set()
    input_node_ids: set[str] = set()
    part_node_ids: set[str] = set()

    for node_id, node in node_by_id.items():
        identities = _node_identity_tokens(node)
        if identities.intersection(input_tokens):
            allowed_ids.add(node_id)
            input_node_ids.add(node_id)
        elif identities.intersection(declared_part_tokens):
            allowed_ids.add(node_id)
            part_node_ids.add(node_id)

    changed = True
    while changed:
        changed = False
        for node_id, node in node_by_id.items():
            if node_id in allowed_ids:
                continue
            parent_ref = _optional_text(node.get("part_of"))
            if parent_ref and _node_reference_allowed(parent_ref, node_by_id, allowed_ids, input_tokens):
                allowed_ids.add(node_id)
                part_node_ids.add(node_id)
                changed = True
        for edge in edges:
            if not isinstance(edge, dict) or _normalise_affordance(edge.get("relation")) != "PART_OF":
                continue
            source = _edge_source(edge)
            target = _edge_target(edge)
            if source in node_by_id and source not in allowed_ids and _node_reference_allowed(
                target,
                node_by_id,
                allowed_ids,
                input_tokens,
            ):
                allowed_ids.add(source)
                part_node_ids.add(source)
                changed = True

    disallowed_ids = set(node_by_id) - allowed_ids
    kept_nodes = []
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id")) in disallowed_ids:
            continue
        if isinstance(node, dict):
            node_id = str(node.get("id"))
            if node_id in input_node_ids:
                node["source"] = "input"
                node.pop("implicit_environment", None)
                _merge_matching_declared_node_fields(node, material_properties)
            elif node_id in part_node_ids:
                node["source"] = "part"
            if str(node.get("parent", "")) in disallowed_ids:
                node.pop("parent", None)
        kept_nodes.append(node)

    kept_edges = [
        edge
        for edge in edges
        if not (
            isinstance(edge, dict)
            and (_edge_source(edge) in disallowed_ids or _edge_target(edge) in disallowed_ids)
        )
    ]
    graph["nodes"] = kept_nodes
    graph["edges"] = kept_edges
    return len(disallowed_ids), len(edges) - len(kept_edges)


def _ensure_declared_part_nodes(
    graph: dict[str, Any],
    input_materials: tuple[str, ...],
    material_properties: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("view_graph.nodes must be a list")
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")

    input_tokens = {_clean_object_name(item) for item in input_materials}
    declared_parts = _declared_part_items(material_properties, input_tokens)
    if not declared_parts:
        return [], []

    added_nodes: list[str] = []
    added_edges: list[dict[str, str]] = []
    for part_id, part_item in declared_parts.items():
        parent_token = _clean_object_name(str(part_item.get("part_of", "")))
        if not parent_token:
            continue
        parent_id = _ensure_declared_parent_node(
            graph,
            parent_token,
            input_tokens,
            material_properties,
            added_nodes,
        )
        if parent_id is None:
            continue
        node_id = _find_node_id_by_token(graph, part_id)
        if node_id is None:
            node = _node_from_declared_material(part_id, part_item, source="part")
            node["part_of"] = parent_id
            nodes.append(node)
            node_id = str(node["id"])
            added_nodes.append(node_id)
        else:
            node = _node_by_id(graph, node_id)
            if node is not None:
                node.setdefault("source", "part")
                node.setdefault("part_of", parent_id)
                _merge_declared_node_fields(node, part_item)
        if not _has_edge(graph, node_id, parent_id, "PART_OF"):
            edge = {"from": node_id, "to": parent_id, "relation": "PART_OF"}
            edges.append(edge)
            added_edges.append(edge)
    return added_nodes, added_edges


def _ensure_declared_parent_node(
    graph: dict[str, Any],
    parent_token: str,
    input_tokens: set[str],
    material_properties: dict[str, dict[str, Any]],
    added_nodes: list[str],
) -> str | None:
    parent_id = _find_node_id_by_token(graph, parent_token)
    if parent_id is not None:
        return parent_id
    if parent_token not in input_tokens and parent_token not in material_properties:
        return None
    parent_item = material_properties.get(parent_token, {"name": parent_token, "properties": []})
    node = _node_from_declared_material(parent_token, parent_item, source="input" if parent_token in input_tokens else "part")
    graph.setdefault("nodes", []).append(node)
    parent_id = str(node["id"])
    added_nodes.append(parent_id)
    parent_parent = _optional_text(parent_item.get("part_of"))
    if parent_parent:
        grandparent_id = _ensure_declared_parent_node(
            graph,
            _clean_object_name(parent_parent),
            input_tokens,
            material_properties,
            added_nodes,
        )
        if grandparent_id is not None:
            node["part_of"] = grandparent_id
            graph.setdefault("edges", []).append({"from": parent_id, "to": grandparent_id, "relation": "PART_OF"})
    return parent_id


def _node_from_declared_material(material_id: str, item: dict[str, Any], *, source: str) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": material_id,
        "name": str(item.get("name") or material_id),
        "category": str(item.get("category", "object")),
        "properties": list(item.get("properties", [])),
        "source": source,
    }
    if item.get("states"):
        node["states"] = list(item.get("states", []))
    node.update(_declared_node_extra_fields(item))
    return node


def _merge_declared_node_fields(node: dict[str, Any], item: dict[str, Any]) -> None:
    node.setdefault("name", str(item.get("name") or node.get("id")))
    if "category" not in node and item.get("category") is not None:
        node["category"] = str(item["category"])
    if not node.get("properties") and item.get("properties"):
        node["properties"] = list(item.get("properties", []))
    if "states" not in node and item.get("states"):
        node["states"] = list(item.get("states", []))
    for key, value in _declared_node_extra_fields(item).items():
        node.setdefault(key, value)


def _merge_matching_declared_node_fields(
    node: dict[str, Any],
    material_properties: dict[str, dict[str, Any]],
) -> None:
    for token in _node_identity_tokens(node):
        item = material_properties.get(token)
        if item is not None:
            _merge_declared_node_fields(node, item)
            return


def _declared_node_extra_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(value)
        for key, value in item.items()
        if key not in NODE_CORE_FIELDS
    }


def _find_node_id_by_token(graph: dict[str, Any], token: str) -> str | None:
    clean_token = _clean_object_name(token)
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and clean_token in _node_identity_tokens(node):
            return str(node.get("id"))
    return None


def _node_by_id(graph: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and str(node.get("id")) == node_id:
            return node
    return None


def _has_edge(graph: dict[str, Any], source_id: str, target_id: str, relation: str) -> bool:
    relation = _normalise_affordance(relation)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if _edge_source(edge) == source_id and _edge_target(edge) == target_id and _normalise_affordance(edge.get("relation")) == relation:
            return True
    return False


def _remove_inside_edges(graph: dict[str, Any]) -> int:
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")
    kept_edges = []
    removed = 0
    for edge in edges:
        if isinstance(edge, dict) and _normalise_affordance(edge.get("relation")) in {"INSIDE", "IN"}:
            removed += 1
            continue
        kept_edges.append(edge)
    if removed:
        graph["edges"] = kept_edges
    return removed


def _remove_occlusion_edges(graph: dict[str, Any]) -> int:
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")
    occlusion_relations = {"OCCLUDES", "PARTIALLY_OCCLUDES", "BLOCKS", "HIDES", "COVERS"}
    kept_edges = []
    removed = 0
    for edge in edges:
        if isinstance(edge, dict) and _normalise_affordance(edge.get("relation")) in occlusion_relations:
            removed += 1
            continue
        kept_edges.append(edge)
    if removed:
        graph["edges"] = kept_edges
    return removed


def _material_present_or_represented_by_parts(graph: dict[str, Any], material_id: str) -> bool:
    token = _clean_object_name(material_id)
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if token in _node_identity_tokens(node):
            return True
        part_of = _optional_text(node.get("part_of"))
        if part_of and _clean_object_name(part_of) == token:
            return True
    return False


def _declared_part_tokens(
    material_properties: dict[str, dict[str, Any]],
    input_tokens: set[str],
) -> set[str]:
    return set(_declared_part_items(material_properties, input_tokens))


def _declared_part_items(
    material_properties: dict[str, dict[str, Any]],
    input_tokens: set[str],
) -> dict[str, dict[str, Any]]:
    allowed_parent_tokens = set(input_tokens)
    parts: dict[str, dict[str, Any]] = {}
    changed = True
    while changed:
        changed = False
        for material_id, item in material_properties.items():
            if not isinstance(item, dict):
                continue
            parent = _optional_text(item.get("part_of"))
            if not parent or _clean_object_name(parent) not in allowed_parent_tokens:
                continue
            part_id = _clean_object_name(str(material_id))
            if part_id in parts:
                continue
            part_item = dict(item)
            part_item["part_of"] = _clean_object_name(parent)
            parts[part_id] = part_item
            allowed_parent_tokens.add(part_id)
            if item.get("name") is not None:
                allowed_parent_tokens.add(_clean_object_name(str(item["name"])))
            changed = True
    return parts


def _declared_parts_prompt(
    material_properties: dict[str, dict[str, Any]],
    input_tokens: set[str],
) -> str:
    parts = _declared_part_items(material_properties, input_tokens)
    if not parts:
        return "无"
    by_parent: dict[str, list[str]] = {}
    for part_id, item in parts.items():
        parent = _clean_object_name(str(item.get("part_of", "")))
        by_parent.setdefault(parent, []).append(part_id)
    return "; ".join(
        f"{parent} -> {', '.join(sorted(children))}"
        for parent, children in sorted(by_parent.items())
    )


def _node_identity_tokens(node: dict[str, Any]) -> set[str]:
    tokens = {_clean_object_name(str(node.get("id", "")))}
    if node.get("name") is not None:
        tokens.add(_clean_object_name(str(node["name"])))
    return {token for token in tokens if token}


def _node_reference_allowed(
    reference: str,
    node_by_id: dict[str, dict[str, Any]],
    allowed_ids: set[str],
    input_tokens: set[str],
) -> bool:
    if reference in allowed_ids:
        return True
    reference_token = _clean_object_name(reference)
    if reference_token in input_tokens:
        return True
    for allowed_id in allowed_ids:
        node = node_by_id.get(allowed_id)
        if node and reference_token in _node_identity_tokens(node):
            return True
    return False


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _remove_invalid_relation_affordance_edges(graph: dict[str, Any]) -> int:
    nodes = {str(node.get("id")): node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("id") is not None}
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")
    kept_edges = []
    removed = 0
    for edge in edges:
        if not isinstance(edge, dict):
            kept_edges.append(edge)
            continue
        relation = _normalise_affordance(edge.get("relation"))
        source = nodes.get(_edge_source(edge))
        target = nodes.get(_edge_target(edge))
        if relation == "ON" and not _node_has_any_property(target, {"SURFACE", "SURFACES"}):
            removed += 1
            continue
        if relation in {"INSIDE", "IN"} and not _node_has_any_property(
            target,
            {"CONTAINERS"},
        ):
            removed += 1
            continue
        if relation == "OCCLUDES" and not _node_has_any_property(source, {"OCCLUDER"}):
            removed += 1
            continue
        kept_edges.append(edge)
    if removed:
        graph["edges"] = kept_edges
    return removed


def _remove_parent_part_external_conflict_edges(graph: dict[str, Any]) -> int:
    nodes = {
        str(node.get("id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    edges = graph.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("view_graph.edges must be a list")

    parent_by_child: dict[str, str] = {}
    for node_id, node in nodes.items():
        parent = _optional_text(node.get("part_of"))
        if parent in nodes:
            parent_by_child[node_id] = parent
    for edge in edges:
        if not isinstance(edge, dict) or _normalise_affordance(edge.get("relation")) != "PART_OF":
            continue
        source = _edge_source(edge)
        target = _edge_target(edge)
        if source in nodes and target in nodes:
            parent_by_child[source] = target

    if not parent_by_child:
        return 0

    children_by_parent: dict[str, set[str]] = {}
    for child, parent in parent_by_child.items():
        children_by_parent.setdefault(parent, set()).add(child)

    def descendants(node_id: str) -> set[str]:
        found: set[str] = set()
        stack = list(children_by_parent.get(node_id, set()))
        while stack:
            child = stack.pop()
            if child in found:
                continue
            found.add(child)
            stack.extend(children_by_parent.get(child, set()))
        return found

    external_participants: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict) or _normalise_affordance(edge.get("relation")) == "PART_OF":
            continue
        source = _edge_source(edge)
        target = _edge_target(edge)
        if source in nodes:
            external_participants.add(source)
        if target in nodes:
            external_participants.add(target)

    conflicted_parents = {
        node_id
        for node_id in nodes
        if node_id in external_participants and descendants(node_id).intersection(external_participants)
    }
    if not conflicted_parents:
        return 0

    kept_edges = []
    removed = 0
    for edge in edges:
        if not isinstance(edge, dict) or _normalise_affordance(edge.get("relation")) == "PART_OF":
            kept_edges.append(edge)
            continue
        if _edge_source(edge) in conflicted_parents or _edge_target(edge) in conflicted_parents:
            removed += 1
            continue
        kept_edges.append(edge)
    if removed:
        graph["edges"] = kept_edges
    return removed


def _node_has_any_property(node: dict[str, Any] | None, properties: set[str]) -> bool:
    if not isinstance(node, dict):
        return False
    node_properties = {_normalise_affordance(prop) for prop in node.get("properties", [])}
    return bool(node_properties.intersection(properties))


def _edge_source(edge: dict[str, Any]) -> str:
    return str(edge.get("from", edge.get("from_id", edge.get("source", ""))))


def _edge_target(edge: dict[str, Any]) -> str:
    return str(edge.get("to", edge.get("to_id", edge.get("target", ""))))


def _normalise_material_properties(raw_value: Any) -> dict[str, dict[str, Any]]:
    if not raw_value:
        return {}
    if isinstance(raw_value, dict) and "materials" in raw_value:
        raw_value = raw_value["materials"]

    entries: list[tuple[str | None, Any]]
    if isinstance(raw_value, dict):
        entries = [(str(key), value) for key, value in raw_value.items()]
    elif isinstance(raw_value, list):
        entries = [(None, value) for value in raw_value]
    else:
        raise ValueError("material_properties must be an object, a list, or an object with a materials field")

    normalised: dict[str, dict[str, Any]] = {}
    for fallback_id, value in entries:
        if not isinstance(value, dict):
            raise ValueError(f"material property entry must be an object: {value!r}")
        material_id = _clean_object_name(str(value.get("id") or fallback_id or value.get("name") or ""))
        if not material_id:
            raise ValueError(f"material property entry is missing id/name: {value!r}")
        props = [_normalise_affordance(item) for item in value.get("properties", [])]
        states = [_normalise_affordance(item) for item in value.get("states", [])]
        item: dict[str, Any] = {
            "name": str(value.get("name") or material_id.replace("_", "")),
            "properties": props,
        }
        item.update(_material_extra_fields(value))
        if value.get("category") is not None:
            item["category"] = str(value["category"]).lower()
        if states:
            item["states"] = states
        parent_id = value.get("part_of", value.get("parent"))
        if parent_id is not None:
            item["part_of"] = _clean_object_name(str(parent_id))
        parts = _normalise_declared_parts(material_id, value.get("parts", []))
        if parts:
            item["parts"] = [part_id for part_id, _part in parts]
            for part_id, part_item in parts:
                existing = normalised.get(part_id)
                if existing is None:
                    normalised[part_id] = part_item
                else:
                    existing.setdefault("part_of", material_id)
        existing_item = normalised.get(material_id)
        if existing_item is not None and "part_of" in existing_item and "part_of" not in item:
            item["part_of"] = existing_item["part_of"]
        normalised[material_id] = item
    return normalised


def _normalise_declared_parts(
    parent_id: str,
    raw_parts: Any,
) -> list[tuple[str, dict[str, Any]]]:
    if not raw_parts:
        return []
    if not isinstance(raw_parts, list):
        raise ValueError(f"parts for {parent_id!r} must be a list")

    parts: list[tuple[str, dict[str, Any]]] = []
    for raw_part in raw_parts:
        if isinstance(raw_part, str):
            part_id = _clean_object_name(raw_part)
            if not part_id:
                continue
            parts.append(
                (
                    part_id,
                    {
                        "name": raw_part,
                        "properties": [],
                        "part_of": parent_id,
                    },
                )
            )
            continue
        if not isinstance(raw_part, dict):
            raise ValueError(f"part entries for {parent_id!r} must be strings or objects: {raw_part!r}")

        part_id = _clean_object_name(str(raw_part.get("id") or raw_part.get("name") or ""))
        if not part_id:
            raise ValueError(f"part entry for {parent_id!r} is missing id/name: {raw_part!r}")
        part_item: dict[str, Any] = {
            "name": str(raw_part.get("name") or part_id.replace("_", "")),
            "properties": [_normalise_affordance(item) for item in raw_part.get("properties", [])],
            "part_of": _clean_object_name(str(raw_part.get("part_of", raw_part.get("parent", parent_id)))),
        }
        part_item.update(_material_extra_fields(raw_part))
        if raw_part.get("category") is not None:
            part_item["category"] = str(raw_part["category"]).lower()
        states = [_normalise_affordance(item) for item in raw_part.get("states", [])]
        if states:
            part_item["states"] = states
        parts.append((part_id, part_item))
    return parts


def _material_extra_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): copy.deepcopy(item)
        for key, item in value.items()
        if key not in MATERIAL_CORE_FIELDS
    }


def _normalise_affordance(value: Any) -> str:
    return str(value).strip().upper().replace(" ", "_")


def _default_model_for_provider(provider: str) -> str:
    if provider == "qwen":
        return "qwen3.6-plus"
    if provider == "openai":
        return "gpt-4o-mini"
    return "gpt-4o-mini"


def _api_key_env_for_provider(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "qwen":
        return "DASHSCOPE_API_KEY"
    return "OPENAI_API_KEY"


def _api_base_url_for_provider(provider: str, override: str | None) -> str:
    if override:
        return override
    env_value = os.environ.get("AUTO_EMBODIED_TASK_API_BASE_URL")
    if env_value:
        return env_value
    if provider == "qwen":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://api.openai.com/v1"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("API response must be a JSON object")
    return parsed


def _clean_object_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _default_scene_id(scene: str, layout: str) -> str:
    return f"{_slug(scene)}_{layout}_generated"


def _slug(value: str) -> str:
    chars = []
    previous_separator = False
    for char in value.lower().strip():
        if char.isalnum():
            chars.append(char)
            previous_separator = False
        elif not previous_separator:
            chars.append("_")
            previous_separator = True
    slug = "".join(chars).strip("_")
    return slug or "scene"
