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

- `GRABBABLE` and `MOVABLE`: allow a material to become a grasped/moved task object.
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

When `--num-samples` is greater than one, each randomized sample gets a unique `scene_id` / `env_id` suffix so manual tasks and trajectory collection can keep the variants separate. Omit `--seed` to get a different random batch each run; pass `--seed` for reproducible experiments. In the spatial profile, `num_occluded_objects` counts distinct occluded target nodes, while `occlusion_depth` controls the required occlusion-chain depth for each selected target. `num_decomposed_parents` counts how many existing parent objects should be decomposed into their `PART_OF` child nodes; only parents that already have `DECOMPOSABLE` and existing part nodes can be selected. Decomposition removes the selected parent node's non-`PART_OF` external relations and recreates them on its direct part nodes, so the view graph directly shows parts participating in the world while the parent remains as a structural owner. Every spatial occlusion layer is represented as a canonical `OCCLUDES` edge, including container blockers; profile editing never writes spatial occlusion as `INSIDE`. Before adding an occluder for a target, profile editing removes existing incoming occlusion edges for that target, so one object has at most one direct occluder. It also removes the occluded target's visible placement edges (`ON`, `INSIDE`, `BENEATH`, and related variants) and visible relative-position edges (`LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, `BEHIND`, `CLOSE`, `NEAR`) while preserving structural `PART_OF` and occlusion-chain edges. Profile editing does not add `OCCLUDER` or `DECOMPOSABLE` to ordinary objects. Each output graph stores `requested_constraint_profile`, `achieved_constraint_profile`, `difficulty_tags`, `profile_constraints`, and `graph_edits` in `metadata`. Difficulty tags are spatial-only, for example `spatial.num_occluded_objects=2`, `spatial.occlusion_depth=2`, and `spatial.num_decomposed_parents=1`; temporal, memory, and failure-recovery difficulty should be represented in the task/harness stage rather than as view-graph edits.

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
