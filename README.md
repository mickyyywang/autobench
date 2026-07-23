# auto_embodied_task

`auto_embodied_task` builds tabletop/indoor embodied-task assets from material specs, view graphs, manual task JSONL, and symbolic teacher trajectory collection. It follows the ET-Plan-Bench output style while keeping graph editing, profile-based occlusion, task definition, and trajectory replay as separate steps.

Core output fields are compatible with ET-Plan-Bench:

- `task`
- `task_completion_criterion`
- `env_id`
- `ground_truth_plan`

Manual task files and the optional generator also use `task_id`, `scene_id`, `layout`, `arms`, `task_type`, `settings`, `objects`, and `metadata` for downstream filtering and future modules.

## Quick Start

Current workflow:

1. Define materials and material properties under `task_specs_cn/`.
2. Manually define task JSONL under `outputs/`.
3. Use `serve-view-graph` to create/edit a view graph and apply the spatial profile that adds occlusion.
4. Use `collect-trajectories` in teacher mode to collect trajectories.
5. Use `serve-trajectory` to replay and inspect collected trajectories.

### 1. Materials and Properties

Create a material list and a matching property JSON, for example:

- `task_specs_cn/化妆品收纳A_materials.txt`
- `task_specs_cn/化妆品收纳A_material_properties.json`

The material list is one material id per line. The properties file defines affordances such as `GRABBABLE`, `MOVABLE`, `CONTAINERS`, `CAN_OPEN`, `OCCLUDER`, `DECOMPOSABLE`, `SURFACES`, `parts`, and optional extra fields such as `max_items`.

### 2. Manual Tasks

Manually write tasks into `outputs/<scene>_tasks.jsonl`. Each line is one task record. For manual-ready teacher tasks, the usual shape is:

```json
{"task_id":"整理办公桌面A_3","scene_id":"整理办公桌面A_3","env_id":"整理办公桌面A_3","layout":"tabletop","arms":"double","task_type":"manual_ready_goal","task":"整理办公桌。","task_completion_criterion":{"and":[["INSIDE","红色笔","铅笔盒"],["CLOSED","铅笔盒"]]},"ground_truth_plan":[],"settings":["spatial","manual_ready"],"metadata":{"manual_goal_only":true}}
```

In this workflow, `ground_truth_plan` is normally empty. Teacher collection uses `task_completion_criterion` as the source of truth for success.

### 3. View Graph and Profile Editing

Run the local view-graph UI from the bench root:

```bash
cd /home/wmq/project/bench
PYTHONPATH=auto_embodied_task/src python -m auto_embodied_task serve-view-graph \
  --host 127.0.0.1 \
  --port 8765 \
  --open-browser
```

Use the UI to load or type the materials/properties, create or edit the base view graph, and apply the spatial profile. The profile step adds occlusion (`OCCLUDES`) and decomposition variants. Save the profiled graph JSONL under `auto_embodied_task/view_graph/`, with scene ids matching the manual task JSONL.

### 4. Teacher Trajectory Collection

For current manual-ready tasks, use `collect-trajectories` in `teacher` mode. The launch configuration `Debug manual ready teacher tabletopa` in `/home/wmq/project/.vscode/launch.json` is the reference set of parameters. Equivalent command shape:

```bash
cd /home/wmq/project/bench
PYTHONPATH=auto_embodied_task/src python -m auto_embodied_task collect-trajectories \
  --view-graph auto_embodied_task/view_graph/整理办公桌面A_3.jsonl \
  --tasks auto_embodied_task/outputs/整理办公桌面A_3_tasks.jsonl \
  --output auto_embodied_task/outputs/整理办公桌面A_3_teacher_trajectories.jsonl \
  --mode teacher \
  --teacher-provider qwen \
  --teacher-model qwen3.7-plus \
  --teacher-api-key-env DASHSCOPE_API_KEY \
  --teacher-temperature 0 \
  --max-steps 100 \
  --failure-injection probability \
  --failure-actions all \
  --failure-probability 0.8 \
  --max-failures-per-episode 10 \
  --failure-seed 7
```

Set `DASHSCOPE_API_KEY` in the shell or debugger environment. Do not hard-code API keys in docs or committed config.

To run one dynamic condition during collection, embed conditions under
`metadata.collection_conditions` in the executable `outputs/*_tasks.jsonl`
TaskRecord and select exactly one condition. The condition is checked before
every teacher observation and is applied at most once per episode. For the
toy-organizing task, the red-box condition is:

```bash
PYTHONPATH=src python -m auto_embodied_task collect-trajectories \
  --view-graph view_graph/整理玩具A.jsonl \
  --tasks outputs/整理玩具A_tasks.jsonl \
  --output outputs/整理玩具A_red_condition_teacher_trajectories.jsonl \
  --mode teacher \
  --teacher-provider mr_openai \
  --teacher-model gpt-5.5 \
  --teacher-api-key-env MR_API_KEY \
  --teacher-api-style responses \
  --teacher-temperature 0 \
  --max-steps 100 \
  --condition-id red_box_max_items_add_red_fishing_toy
```

`整理玩具A_tasks.jsonl` contains only this red-box condition. The separately
defined `整理玩具B_tasks.jsonl` contains only
`purple_box_max_items_add_green_plush`: its initial view graph must omit
`绿色毛绒玩具`, and the purple box reaching `max_items` adds that object. Each
`add_object` node must be absent from its task's initial view graph. Collection
validates this requirement before the first teacher call. The collection runtime intentionally supports only
`on_container_max_items_reached -> add_object` with
`success_policy.type=existing_task_goal`; it does not support relocation,
copying, or dynamic goal inheritance. Collection conditions currently require
the symbolic backend.

`--condition-file` remains available for standalone condition suites. When it
is omitted, `--condition-id` is resolved from each task record's
`metadata.collection_conditions`.

Closed-loop evaluation intervention manifests support two broader `add_object`
patterns. A new copy or same-class object can inherit the source object's local
goal, including its OR branches:

Natural add-object names are declared separately from existing materials and
properties in `task_specs_cn/copy_objects.json`:

```json
{
  "schema_version": 1,
  "inherit_copy_disabled_task_groups": ["整理玩具A", "整理玩具B"],
  "task_groups": {
    "整理办公桌面B": {
      "蓝色笔": {
        "id": "铅笔",
        "name": "铅笔"
      }
    }
  }
}
```

The view-graph creator does not read this registry, so `铅笔` remains absent
from the initial graph and existing material/property files remain unchanged.
The registry must cover every material marked `COPYABLE` in each enabled task
group. A task group listed in `inherit_copy_disabled_task_groups` must not have a
`task_groups` entry: the generator skips copy-template coverage for it and omits
the `add_object_inherit_source_goal` disturbance by setting that condition to
`eligible=false` and `graph_disturbance=null`. In particular, only
`add_object_existing_task_goal_at_capacity` is eligible as an add-object
condition for `整理玩具A/B`; each task's late object is already named in the
task success goal but is absent from its initial view graph.
The closed-loop manifest generator considers only `COPYABLE` placement sources,
reads the new object's natural id/name from this registry, and adds `copy_from`
automatically. At runtime, `category`, `properties`, `states`, `max_items`, and
all other attributes come from the actual source node in the profiled view graph;
registry values cannot override them. Thus a magazine copied from an openable
book stays openable, and a registered identity cannot add behavior absent from
its source node. The generator does not synthesize
`<source>__added_copy` names. Repeated source ids such as
`胡萝卜_1` and `胡萝卜_2` resolve through the base `胡萝卜` template. Templates do
not alter the initial task criterion. `inherit_from` dynamically adds the source
object's projected goal only after the new object appears. If that goal includes
`ASSEMBLED`, the generator also resolves every direct `PART_OF` node through the
same registry, spawns the new parent and its copied parts on the staging surface,
and inherits both the assembly and placement requirements.

```json
{
  "trigger": {
    "type": "on_object_goal_satisfied",
    "node_id": "蓝色笔",
    "required_predicates": [
      {"predicate": "CLOSED", "args": ["铅笔盒"]}
    ]
  },
  "graph_disturbance": {
    "operation": "add_object",
    "object": {"id": "铅笔", "name": "铅笔", "copy_from": "蓝色笔"},
    "relation": "ON",
    "target": "桌面",
    "success_policy": {"type": "inherit_from", "source_node_id": "蓝色笔"}
  }
}
```

If a task object is already named in the initial task completion criterion but
is absent from the initial view graph, use
`on_container_max_items_reached` with
`"success_policy": {"type": "existing_task_goal"}`. When the object is
added, the task criterion is unchanged. In the inheritance pattern, the
source object's projected criterion is dynamically conjoined to the effective
criterion with the new object substituted for the source. The closed-loop
intervention report records the previous criterion, added expression, and
effective criterion under `details.goal_update`.

### 5. Trajectory Replay UI

Run the replay visualizer from the bench root:

```bash
cd /home/wmq/project/bench
PYTHONPATH=auto_embodied_task/src python -m auto_embodied_task serve-trajectory \
  --trajectory-dir auto_embodied_task/outputs \
  --host 127.0.0.1 \
  --port 8766 \
  --open-browser
```

The UI lists trajectory JSONL files in `auto_embodied_task/outputs` and replays each episode step by step.

## Optional: Built-in Task Generation

The project still includes a built-in `generate` command, but the current benchmark workflow usually writes `outputs/*_tasks.jsonl` manually instead of generating tasks automatically.

```bash
cd /home/wmq/project/bench/auto_embodied_task
PYTHONPATH=src python -m auto_embodied_task generate \
  --view-graph examples/indoor_view_graph.jsonl \
  --output outputs/indoor_tasks.jsonl \
  --layout indoor \
  --arms single \
  --task-types long_horizon,navigation,manipulation,multi_object \
  --settings spatial,memory,temporal,failure_recovery \
  --max-tasks 40 \
  --seed 7
```

For tabletop scenes:

```bash
PYTHONPATH=src python -m auto_embodied_task generate \
  --view-graph examples/tabletop_view_graph.jsonl \
  --output outputs/tabletop_tasks.jsonl \
  --layout tabletop \
  --arms double \
  --settings spatial,temporal \
  --max-tasks 30
```

A richer multi-scene example is available at `examples/rich_view_graphs.jsonl`. It contains indoor and tabletop scenes with ordinary support edges plus constrained states such as containment and occlusion.

```bash
PYTHONPATH=src python -m auto_embodied_task generate \
  --view-graph examples/rich_view_graphs.jsonl \
  --output outputs/rich_tasks.jsonl \
  --layout all \
  --task-types all \
  --settings spatial,memory,temporal,failure_recovery \
  --max-tasks 200
```

To create the graph input from a material list, use `create-task-view-graph`. This command asks an API model to produce one JSON package with:

- `view_graph`: the graph consumed by profile editing, trajectory collection, and the optional task generator.

The activity goal, such as `整理桌面` or `prepare ingredients`, is an input to graph construction. All materials from `--materials-file` are expected to appear in one generated view graph, either as whole-object nodes or as explicit part nodes. Graph creation only uses input materials and parts of input materials; it must not add extra environment, room, desk, tray, shelf, or support nodes unless those objects are listed as input materials. For `tabletop` layout, list `桌面` or another support surface as a material and give it `SURFACES`; validation will not add an implicit tabletop node. In the current workflow, benchmark tasks are usually written manually in `outputs/*_tasks.jsonl`; the generated or edited view graph is used directly by `collect-trajectories`.

It writes the graph as JSONL for profile editing, trajectory collection, or optional downstream task generation, and can also write the full package JSON:

```bash
PYTHONPATH=src python -m auto_embodied_task create-task-view-graph \
  --materials-file task_specs_cn/organize_the_office_desk_materials.txt \
  --material-properties task_specs_cn/organize_the_office_desk_material_properties.json \
  --scene "办公桌面" \
  --task "整理办公桌" \
  --layout tabletop \
  --arms double \
  --provider qwen \
  --model qwen3.6-plus \
  --output examples/generated_office_task_graph.jsonl \
  --package-output examples/generated_office_task_package.json
```

View graph creation uses OpenAI-compatible chat completions and validates the returned package before writing output. For Qwen/DashScope-style endpoints, pass `--provider qwen`; override `--model` with the exact deployed model name, for example `qwen3.6-plus` or `qwen3.7-plus` if that is the model exposed by your endpoint. Qwen thinking mode is disabled by default for stricter JSON output; pass `--enable-thinking` only when you want model-side deliberation:

```bash
DASHSCOPE_API_KEY=... PYTHONPATH=src python -m auto_embodied_task create-task-view-graph \
  --materials-file task_specs_cn/prepare_ingredients_materials.txt \
  --material-properties task_specs_cn/prepare_ingredients_material_properties.json \
  --scene "厨房备餐台" \
  --layout tabletop \
  --arms single \
  --provider qwen \
  --model qwen3.6-plus \
  --task "准备食材" \
  --output examples/generated_cooking_task_graph.jsonl \
  --package-output examples/generated_cooking_task_package.json
```

Use `--provider compatible --api-base-url ... --api-key-env ... --model ...` for other VLM/LLM services that expose the same chat completions shape.

The graph-construction prompt is written in Chinese and explicitly asks the model to preserve Chinese material names in node `id` and `name`. Relation and property enums remain English (`ON`, `PART_OF`, `GRABBABLE`, `MOVABLE`, etc.) so downstream parsing stays stable.

Detailed relation and property guidance is kept in small prompt skill files:

- `src/auto_embodied_task/prompt_skills/relation.md`
- `src/auto_embodied_task/prompt_skills/properties.md`

## Material Properties JSON

`--materials-file` lists the available material ids, one per line. `--material-properties` is an optional JSON file passed to the API prompt so the model can assign categories, affordances, states, and part relations consistently:

```json
{
  "materials": {
    "apple": {"category": "food", "properties": ["GRABBABLE", "MOVABLE"]},
    "book": {"category": "object", "properties": ["GRABBABLE", "MOVABLE", "OCCLUDER"]},
    "blue_pen": {
      "category": "tool",
      "properties": ["GRABBABLE", "MOVABLE", "DECOMPOSABLE"],
      "states": ["CAPPED"],
      "parts": [
        {"id": "blue_pen_body", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"]},
        {"id": "blue_pen_cap", "category": "tool", "properties": ["GRABBABLE", "MOVABLE"], "states": ["ATTACHED"]}
      ]
    },
    "box": {"category": "container", "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"], "states": ["CLOSED"]},
    "drawer": {
      "category": "furniture",
      "properties": ["STORAGE_UNIT", "SURFACES"],
      "parts": [
        {"id": "drawer_top", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"], "states": ["CLOSED"]},
        {"id": "drawer_middle", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"], "states": ["CLOSED"]},
        {"id": "drawer_bottom", "category": "container", "properties": ["CONTAINERS", "CAN_OPEN", "OCCLUDER"], "states": ["CLOSED"]}
      ]
    }
  }
}
```

The API prompt describes these fields as follows:

- `GRABBABLE` and `MOVABLE`: allow a material to become a grasped/moved task object. A movable container or surface with an `INSIDE`/`ON` payload rejects `grab` with `non_empty_payload` unless it also has `CARRY_CONTENTS` (or the compatible alias `STABLE_TRANSPORT`).
- `SURFACES`: allows a material to become an `ON` placement target. It can be combined with `STORAGE_UNIT` on a drawer parent when the drawer top/body can support objects.
- `CONTAINERS`: allows a material to become a future containment/placement target. Initial `create-task-view-graph` output does not keep `INSIDE` edges.
- `CAN_OPEN` plus `states` such as `["OPEN"]` or `["CLOSED"]`: makes the planner emit open/close actions when placing into that target.
- `OCCLUDER`: marks that a node can act as an occluding object later. Initial view graph construction keeps this as an ability property only; profiled graph editing adds `OCCLUDES` edges from the requested spatial difficulty profile.
- `COPYABLE`: reserved for downstream task/harness memory distractor logic. Spatial profile editing does not copy nodes.
- `DECOMPOSABLE`: marks that a parent node may be split by profile editing into its existing `PART_OF` child nodes for spatial difficulty. Objects without this property are never decomposed automatically.
- `parts`: describes possible child part nodes. The model can represent external relations through the whole-object node or through its part nodes, but not both. If part nodes are emitted, non-`PART_OF` relations for that material should use the concrete part nodes; if the whole-object node carries those relations, its part nodes should keep only structural `PART_OF` relations until a later `DECOMPOSABLE` spatial profile decomposes the parent.
- `part_of`: the inverse form for manually listed child materials. For example, `drawer_top` can point back to parent `drawer`.

If a material is missing from the JSON file, the model still receives the material id from `--materials-file` and must infer reasonable properties from context.

Task-specific Chinese material specs are available under `task_specs_cn/`. For each new view-graph construction task, create a pair such as `<task_id>_materials.txt` and `<task_id>_material_properties.json` before using the view-graph UI or running `create-task-view-graph`.

Initial view graph construction should preserve only raw visible scene structure: support, part structure, relative positions, object properties, and open/closed states. `create-task-view-graph` removes any `INSIDE`, `OCCLUDES`, `BLOCKS`, `HIDES`, or `COVERS` edges returned by the model; containment/occlusion difficulty is added later by `edit-view-graph` from an explicit profile. `CLOSED` states are kept because a closed container alone is not hidden state without an object inside it. Every retained node should participate in at least one edge. For a material with parts, non-`PART_OF` relations should use either the parent node or the part nodes, not both. If part nodes are emitted, external relations should connect to those concrete parts. Any generated node that is neither an input material nor a part of an input material is removed during graph validation. If the model omits a part declared in `material_properties.parts` or a declared `part_of` chain rooted at an input material, validation adds the missing part node and a structural `PART_OF` edge from the material spec; it does not invent external placement for that part. `ON` targets must have `SURFACES`.

## View Graph Editing

For an end-to-end browser workflow, run the local web UI:

```bash
PYTHONPATH=src python -m auto_embodied_task serve-view-graph \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765/`. The page can create a view graph through the backend `create-task-view-graph` path, or skip creation and import an existing direct view graph JSON, package JSON with `view_graph`, or JSONL file. It then immediately inspects and edits the loaded nodes and edges in the same browser workflow. The canvas has both `Graph` and `Map` views: `Graph` shows the relation network, while `Map` renders a top-down room/table layout from `room`, `parent`, `ON`, `BENEATH`, `INSIDE`, `PART_OF`, `LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, `BEHIND`, `NEAR`, and `OCCLUDES` edges without drawing relation lines. Edits to nodes or edges re-render both views from the same current graph object; Graph drag positions are only for the network view, while Map positions are recomputed from relations. Top-level surfaces become map zones, `ON` and `BENEATH` children are drawn inside their targets with `on`/`beneath` labels, `INSIDE` children and part nodes are drawn inside their parent node, relative-position relations arrange objects without overlap, and `OCCLUDES` places the occluded target inside the occluder with a red dashed box. The editor can select and drag nodes in `Graph` view, inspect the inferred `Map` layout, edit node metadata, add/delete nodes, add/delete edges, change `from`/`to` endpoints, and change relations such as `ON`, `BENEATH`, `INSIDE`, `OCCLUDES`, and `PART_OF`. Node names must be unique; adding or renaming to a duplicate name is rejected. Materials and material properties can be typed directly or loaded from local `.txt`/`.json` files. For model access, fill either `API key` with a raw key or `API key env` with an environment variable name such as `DASHSCOPE_API_KEY`. The `Profile Edit` panel snapshots the created/imported graph as a base graph and applies only spatial difficulty settings to that base graph through sliders; temporal, memory, and failure-recovery constraints are task/harness-level settings instead of view-graph edits. Repeated applies reroll from the base graph instead of stacking on the previous profiled graph. `Num samples` is a free numeric input. When samples is greater than one, the server returns randomized variants, loads the first one into the editor, and keeps the full batch available through `Download Samples JSONL`. Use `Download JSONL` to save only the currently loaded graph.

### Profiled View Graphs

After creating or hand-editing an initial view graph, use `edit-view-graph` to make randomized graph variants that satisfy an abstract constraint profile. This step is local and does not call a model API.

Example profile:

```json
{
  "profile_id": "desk_spatial2",
  "spatial": {
    "enabled": true,
    "num_occluded_objects": 2,
    "occlusion_depth": 2,
    "num_decomposed_parents": 1
  }
}
```

Batch-generate profiled graph variants:

```bash
PYTHONPATH=src python -m auto_embodied_task edit-view-graph \
  --input view_graph/办公桌面_tabletop_generated_edited.jsonl \
  --profile profiles/desk_profile.json \
  --output view_graph/desk_profiled.jsonl \
  --num-samples 10 \
  --seed 7
```

When `--num-samples` is greater than one, each randomized sample gets a unique `scene_id` / `env_id` suffix so manual tasks and trajectory collection can keep the variants separate. Omit `--seed` to get a different random batch each run; pass `--seed` for reproducible experiments. In the spatial profile, `num_occluded_objects` counts distinct occluded target nodes, while `occlusion_depth` controls the required occlusion-chain depth for each selected target. `num_decomposed_parents` counts how many existing parent objects should be decomposed into their `PART_OF` child nodes; only parents that already have `DECOMPOSABLE` and existing part nodes can be selected. Decomposition removes the selected parent node's non-`PART_OF` external relations and recreates them on its direct part nodes, so the view graph directly shows parts participating in the world while the parent remains as a structural owner. Every spatial occlusion layer is represented as a canonical `OCCLUDES` edge, including container blockers; profile editing never writes spatial occlusion as `INSIDE`. Before adding an ordinary profile occluder for a target, profile editing removes existing incoming profile occlusion edges for that target, so one object has at most one direct profile occluder. It also removes the occluded target's visible placement edges (`ON`, `INSIDE`, `BENEATH`, and related variants) and visible relative-position edges (`LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, `BEHIND`, `CLOSE`, `NEAR`) while preserving structural `PART_OF` and occlusion-chain edges. Three-layer openable storage units additionally receive structural directional edges: first layer to second/third and second layer to third, each with `resolution_action: close`. These edges activate only while the upper layer is open, may give a lower drawer multiple structural blockers, preserve the drawer's placement edge, and propagate invisibility through the lower drawer's `INSIDE` descendants. Profile editing does not add `OCCLUDER` or `DECOMPOSABLE` to ordinary objects. Each output graph stores `requested_constraint_profile`, `achieved_constraint_profile`, `difficulty_tags`, `profile_constraints`, and `graph_edits` in `metadata`. Difficulty tags are spatial-only, for example `spatial.num_occluded_objects=2`, `spatial.occlusion_depth=2`, and `spatial.num_decomposed_parents=1`; temporal, memory, and failure-recovery difficulty should be represented in the task/harness stage rather than as view-graph edits.

## View Graph JSONL

Each JSONL line can define one complete scene:

```json
{"scene_id":"apartment_0","env_id":0,"layout":"indoor","robot":{"arms":"single","start":"entry"},"nodes":[{"id":"kitchen","name":"kitchen","category":"room"},{"id":"plate","name":"plate","category":"object","room":"kitchen","properties":["GRABBABLE","MOVABLE"]},{"id":"sink","name":"sink","category":"container","room":"kitchen","properties":["CONTAINERS"]}],"edges":[{"from":"plate","to":"kitchen","relation":"INSIDE"},{"from":"sink","to":"kitchen","relation":"INSIDE"}]}
```

Required scene fields:

- `scene_id`: stable scene identifier.
- `layout`: `indoor` or `tabletop`.
- `nodes`: graph nodes.
- `edges`: graph edges.

Common node fields:

- `id`: unique node id in the scene.
- `name`: human-readable object class or room name. VirtualHome-style `class_name` is also accepted.
- `category`: for example `room`, `object`, `surface`, `container`, `appliance`, `furniture`.
- `properties`: use ET/VirtualHome style values such as `GRABBABLE`, `MOVABLE`, `SURFACES`, `CONTAINERS`, `CAN_OPEN`, `OCCLUDER`, `COPYABLE`, `DECOMPOSABLE`.
- `room`: optional room id for indoor scenes.
- `parent`: optional parent/support id, only when the parent is another input material or an input material part.
- `part_of`: optional parent material/part id for component nodes.

Common edge fields:

- `from` or `from_id`
- `to` or `to_id`
- `relation` or `relation_type`, for example `INSIDE`, `ON`, `BENEATH`, `CLOSE`, `LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, `BEHIND`.

The format is a lightweight, task-generation-oriented view graph inspired by ET-Plan-Bench/VirtualHome scene graphs. It keeps compatible affordance names such as `GRABBABLE`, `MOVABLE`, `SURFACES`, `CONTAINERS`, and accepts VirtualHome-style aliases such as `class_name`, `from_id`, `to_id`, and `relation_type`, but it is not the full ET-Plan-Bench scene graph dump.

Good view graph examples should include a mix of:

- Graph-construction scope: generated view graphs contain input material nodes and their part nodes. Support surfaces such as `桌面` must be listed as input materials.
- Task affordances: `GRABBABLE`/`MOVABLE` objects and `CONTAINERS`/`SURFACES` placement targets.
- Memory context: stable visible support or relative-position facts that can be recalled without being exposed in `task`.
- Spatial constraints: profiled variants add canonical `OCCLUDES` layers and optional `DECOMPOSABLE` parent-to-part decompositions later.
- Distractors: nearby objects with ordinary `ON`, `LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, or `BEHIND` relations so constrained objects are not the only objects in the graph.

## Task Types

This section applies to the optional `generate` command. Manual-ready tasks in `outputs/*_tasks.jsonl` can ignore these built-in task types.

- `long_horizon`: builds ordered multi-step cleanup/organization tasks from existing graph constraints. It prefers objects that are inside openable containers, occluded, or have parts, and writes the ordered decomposition to `metadata.constraint_subtasks`.
- `navigation`: find or move close to an object.
- `manipulation`: move one grabbable object into or onto a target.
- `multi_object`: move two grabbable objects into or onto one target. `--arms single` produces sequential plans; `--arms double` can pick both objects before placing them.

Use `--task-types all` to enable all built-in task types. The CLI default also includes `long_horizon`.

## Extra Settings

This section also applies mainly to the optional `generate` path. In the current manual workflow, settings are usually represented directly in task JSONL metadata and in the profile/harness stage.

Each setting lives in its own module under `src/auto_embodied_task/constraints`.

- `spatial`: requires task objects to have constrained spatial states such as containment or occlusion. It recognizes relations such as `INSIDE`/`CONTAINS` and `OCCLUDED_BY`/`OCCLUDES`, then writes canonical criteria like `(OCCLUDED_BY, apple, box)`.
- `memory`: stores a prior-observation episode in metadata. It does not alter `task_completion_criterion` and must not be treated as an initial state.
- `temporal`: rewrites applicable criteria with explicit `STEP_1`, `STEP_2`, etc. and stores the ordered subtask resolution in metadata.
- `failure_recovery`: injects a recoverable failed action into the reference plan and records the recovery subtask in metadata.

Extra settings do not alter the natural-language `task` field. The task text stays goal-only; full constraint resolution lives in structured fields such as `settings`, `task_completion_criterion`, `ground_truth_plan`, and `metadata.constraint_subtasks`.

Use `--settings all` to enable all built-in settings. Settings are generated as separate task variants by default, so the base tasks remain available unless `--no-base` is passed.

## Trajectory Collection and Replay

Use `collect-trajectories` for trajectory collection. The current manual-ready workflow uses `teacher` mode with manually written `outputs/*_tasks.jsonl`; see `Debug manual ready teacher tabletopa` in `/home/wmq/project/.vscode/launch.json` for the maintained debug arguments.

The key files are:

- `--view-graph`: profiled graph JSONL from `view_graph/`.
- `--tasks`: manual task JSONL from `outputs/`.
- `--output`: trajectory JSONL written back under `outputs/`.

Use `serve-trajectory` for replay visualization:

```bash
cd /home/wmq/project/bench
PYTHONPATH=auto_embodied_task/src python -m auto_embodied_task serve-trajectory \
  --trajectory-dir auto_embodied_task/outputs \
  --host 127.0.0.1 \
  --port 8766 \
  --open-browser
```

Harness internals, supported action names, observation shape, failure injection behavior, and evaluator details should be checked directly in code, primarily `src/auto_embodied_task/harness.py`, `src/auto_embodied_task/harness_bp.py`, and `src/auto_embodied_task/goal.py`.

### RoboTwin 2.0 backend

The simulator-neutral harness accepts `--backend symbolic|robotwin`. Static compilation does not import SAPIEN and is the first check for every new scene:

```bash
PYTHONPATH=src python -m auto_embodied_task validate-robotwin-task \
  --view-graph view_graph/整理办公桌面B_1.jsonl \
  --tasks outputs/整理办公桌面B_1_tasks.jsonl \
  --trajectory saved/整理办公桌面B_1_teacher_trajectories_20260712_231519__galaxea_r1lite_20260713_165639_192.168.31.142__aligned_20260715_214950.jsonl \
  --asset-map exp/robotwin/config/b1_asset_map.json \
  --output exp/robotwin/validation/current_static.json
```

The physical runtime uses the isolated environment `/home/wmq/.conda/envs/robotwin` and local RoboTwin checkout `/home/wmq/project/bench/RoboTwin`. Replay the existing failure/recovery episode with:

```bash
env -u DISPLAY CUDA_HOME=/home/wmq/.conda/envs/robotwin PYTHONPATH=src \
  /home/wmq/.conda/envs/robotwin/bin/python -m auto_embodied_task \
  replay-robotwin-trajectory \
  --view-graph view_graph/整理办公桌面B_1.jsonl \
  --tasks outputs/整理办公桌面B_1_tasks.jsonl \
  --trajectory saved/整理办公桌面B_1_teacher_trajectories_20260712_231519__galaxea_r1lite_20260713_165639_192.168.31.142__aligned_20260715_214950.jsonl \
  --asset-map exp/robotwin/config/b1_asset_map.json \
  --robotwin-root /home/wmq/project/bench/RoboTwin \
  --execution-mode assisted \
  --output-dir exp/robotwin/experiments/b1_assisted_replay_49_attempt4 --seed 7
```

RoboTwin action execution is selected at runtime. Use `--execution-mode assisted`
when reproducing an existing aligned trajectory and allowing the adapter's
pose/drive/kinematic fallbacks. Use `--execution-mode strict` for physical
acceptance and closed-loop evaluation; failed contact or predicates return an
error instead of being committed. An assisted report may have `success=true`,
but always records `strict_physical_acceptance=false`. Closed-loop collection
uses the equivalent `--robotwin-execution-mode strict|assisted` option and
defaults to `strict`.

`failed_grab`, `failed_putin`, `failed_open`, `failed_close`, and `failed_attach` only count as failures when their physical evidence passes the corresponding verifier. Goal predicates are read from SAPIEN poses, AABBs, articulation positions, contacts, and fixed drives. Egocentric visibility uses `head_camera` actor segmentation; view-graph `OCCLUDES` counterfactuals use the stable `overview_camera` actor segmentation.

Physical replay only accepts a trajectory with `real_alignment` metadata from `saved/`; passing the teacher trajectory from `outputs/` is rejected. The replay records one observation before the first aligned action (`step_0000`) and one after every aligned action. Each `trajectory[i].observation` in `replay_report.json` contains the structured scene state, visible nodes, visibility ratios, robot joint/end-effector state, and paths to the camera artifacts. `observations.jsonl` contains the same observation sequence; `observations/step_NNNN/` contains `observation.json` plus RGB PNG, depth NPY/PNG, and raw/visualized actor-segmentation NPY/PNG files for `head_camera`, `overview_camera`, `front_camera`, `left_camera`, and `right_camera`. `head_camera` is a 640x360, 72-degree vertical-FOV onboard view; the former high rear head view is retained as `overview_camera`. `acceptance.all_aligned_steps_executed` and `acceptance.all_observations_captured` are required for replay success.

The archived seed-7 artifact at `exp/robotwin/archive/legacy_functional_proxy_replay/replay_report.json` is the earlier functional-proxy baseline: 49/49 real-aligned actions and 50/50 observations. It is not an acceptance result for the current official-object and corrected-articulation scene. The current successful assisted replay is `exp/robotwin/experiments/b1_assisted_replay_49_attempt4/`; its reproducibly sampled dense action captures are indexed by `dense_sampling_manifest.json`. Current configuration, audits, visual reviews, and new experiments live under `exp/robotwin/`; environment details are recorded in `exp/robotwin/docs/environment.md`.

## Real Trajectory Evaluation with MR Models

`evaluate-real-trajectories` supports the internal MR multimodal gateways described by the examples in `/home/wmq/project/mr_model`. Set `MR_API_KEY`, select an `mr_*` provider, and pass the exact model name:

```bash
export MR_API_KEY=...
PYTHONPATH=src python -m auto_embodied_task evaluate-real-trajectories \
  --input saved/aligned_trajectory.jsonl \
  --output evaluations/real_eval_gpt55.jsonl \
  --provider mr_openai \
  --model gpt-5.5 \
  --modes obs_only \
  --history-source inference \
  --frame-sampling previous_tail
```

The MR providers are:

- `mr_openai`: `gpt-5.5`, `gpt-5.5-pro-2026-04-23`, `gpt-5.4-2026-03-05`, `gpt-5-2025-08-07`.
- `mr_anthropic`: `claude-opus-4-8`, `claude-fable-5`, `claude-sonnet-5`, `claude-opus-4-7`, `claude-sonnet-4-6-20260217`, `claude-opus-4-6-20260205`.
- `mr_google`: `gemini-3.1-pro-preview`, `gemini-3-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`.

The provider selects the correct MR URL, authentication header, image payload, and response parser automatically. `--api-base-url` remains available for overriding a gateway. Image- and video-generation models are not action-evaluation models and are intentionally excluded. The Gemini Robotics entry is also excluded because its calling protocol is not documented in the supplied examples.

Every evaluation run appends a local timestamp to the requested output basename, for example `real_eval_gpt55_20260715_164530.jsonl`. Its summary uses the same basename with `__summary.json`, so repeated runs do not overwrite one another.

With `--frame-sampling previous_tail`, an unmatched `manual_inserted` step can be evaluated from the nearest preceding matched/custom observation tail. Its own custom observation tail remains the input observation for the following step. The evaluator preserves graph-generated `valid_actions` and, when necessary, appends the manual gold action instead of changing the global post-recovery retry rule. Evaluation records expose the two sources separately as `generated_valid_actions` and `valid_actions_added`.

## Extending Settings

Create a new module in `auto_embodied_task/constraints`, subclass `TaskModifier`, and register it:

```python
from .base import TaskModifier, register_modifier

@register_modifier
class MySetting(TaskModifier):
    name = "my_setting"

    def apply(self, task, graph, rng):
        updated = task.clone()
        updated.settings.append(self.name)
        return updated
```

Then import the module in `constraints/__init__.py` so the CLI can find it.


  cd /home/wmq/project/bench/auto_embodied_task
  scripts/evaluate_real_models_visible_graph_only.sh


MODES='obs_only,visible_graph_only' HISTORY_SOURCE='teacher' ./scripts/evaluate_real_models_valid_compare.sh both


› 不可执行在真实场景中是执行之后才知道不可执行，所以仍然是候选。我觉得没有问题。


• 这样定义是合理的。valid_actions 表示“基于动作前 observation 可以合理尝试的动作”，而不是“执行器保证成功的动
  作”。

  三个指标因此形成清晰分工：

  - action_admissibility：动作在执行前是否合理可尝试。
  - soft_optimal_action_score：反事实执行后是否成功、是否推动任务接近目标。
  - recovery：动作失败后是否识别失败并正确恢复。

  因此某个动作可以同时满足：

  action_admissibility = 1
  soft_optimal_action_score = 0

  例如尝试把物品放入看起来可用、但实际已满的容器。这不是指标矛盾，而是在分别衡量“选择是否合理”和“结果是否有
  效”。


cd /home/wmq/project/bench/auto_embodied_task
./scripts/evaluate_view_graph_models_closed_loop.sh both





如何生成manifests？
  生成脚本：auto_embodied_task/scripts/generate_saved_intervention_manifests.py:492
  它会针对每个 episode 独立分析：
  - task_completion_criterion
  - initial_view_graph
  - teacher 成功动作序列
  - alignment-only 状态变化
  - 合法目标与错误目标
  - 可见物体、容器状态和遮挡关系

  然后独立选择：

  - state_regression
  - completed_subgoal_rollback
  - wrong_container_relocation
  - add_occlusion
  - add_object_inherit_source_goal（整理玩具A/B保留条目，但为 eligible=false）
  - add_object_existing_task_goal_at_capacity（仅满足数据前置条件时 eligible）
  - 每种动作类型一次的 failure injection

  不再包含针对 cosmetics/office/tabletop 的固定物体模板。错误容器也会排除该物体的所有合法目标，并优先选择轨迹中实际使用过、干扰发生时可见的容器。

  生成后默认进行符号回放，验证：

  - trigger 能被 teacher 轨迹触发
  - 干扰确实改变图状态
  - 被移动物体仍然可见
  - 恢复动作可以成功
  - 新增遮挡能够被解除
  - 新增 copy 在 staging 位置尚未成功，按继承目标归位后成功

  manifest v5 的 add_occlusion 不固定发生在 step 1。运行时从第 3 个模型 step
  起，在任务进度 10%～80% 的窗口内持续检查候选，并预演遮挡和恢复。v5 还加入
  两类 add_object：源目标满足后动态继承局部成功标准，以及容器达到 max_items 后
  引入 task goal 已引用但初始图缺失的对象。任务组可以只启用第二类；整理玩具A/B就是
  这种情况。旧 episode 不满足第二类前置条件时，
  条目保留为 eligible=false，并记录 ineligible_reason。

  先预览而不写文件：

  cd /home/wmq/project/bench/auto_embodied_task

  python scripts/generate_saved_intervention_manifests.py \
    --saved-dir saved \
    --output-dir exp/intervention_manifests \
    --dry-run

  生成新 episode，保留已有 manifest：

  python scripts/generate_saved_intervention_manifests.py \
    --saved-dir saved \
    --output-dir exp/intervention_manifests

  重新生成并覆盖全部：

  python scripts/generate_saved_intervention_manifests.py \
    --saved-dir saved \
    --output-dir exp/intervention_manifests \
    --overwrite



view graph闭环评测：
  cd /home/wmq/project/bench/auto_embodied_task
  ./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
