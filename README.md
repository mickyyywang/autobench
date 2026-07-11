# auto_embodied_task

`auto_embodied_task` generates embodied task benchmark JSONL from JSONL view graphs. It follows the ET-Plan-Bench output style while making task settings modular.

Core output fields are compatible with ET-Plan-Bench:

- `task`
- `task_completion_criterion`
- `env_id`
- `ground_truth_plan`

The generator also writes `task_id`, `scene_id`, `layout`, `arms`, `task_type`, `settings`, `objects`, and `metadata` for downstream filtering and future modules.

## Quick Start

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

- `view_graph`: the graph consumed by the existing task generator.

The activity goal, such as `整理桌面` or `prepare ingredients`, is an input to graph construction. All materials from `--materials-file` are expected to appear in one generated view graph, either as whole-object nodes or as explicit part nodes. Graph creation only uses input materials and parts of input materials; it must not add extra environment, room, desk, tray, shelf, or support nodes unless those objects are listed as input materials. For `tabletop` layout, list `桌面` or another support surface as a material and give it `SURFACES`; validation will not add an implicit tabletop node. Benchmark tasks are still generated later by the `generate` command from the view graph.

It writes the graph as JSONL for downstream generation, and can also write the full package JSON:

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

Task-specific Chinese material specs are available under `task_specs_cn/`. For each new view-graph construction task, create a pair such as `<task_id>_materials.txt` and `<task_id>_material_properties.json` before running `create-task-view-graph`.

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

When `--num-samples` is greater than one, each randomized sample gets a unique `scene_id` / `env_id` suffix so downstream task generation can keep the variants separate. Omit `--seed` to get a different random batch each run; pass `--seed` for reproducible experiments. In the spatial profile, `num_occluded_objects` counts distinct occluded target nodes, while `occlusion_depth` controls the required occlusion-chain depth for each selected target. `num_decomposed_parents` counts how many existing parent objects should be decomposed into their `PART_OF` child nodes; only parents that already have `DECOMPOSABLE` and existing part nodes can be selected. Decomposition removes the selected parent node's non-`PART_OF` external relations and recreates them on its direct part nodes, so the view graph directly shows parts participating in the world while the parent remains as a structural owner. Every spatial occlusion layer is represented as a canonical `OCCLUDES` edge, including container blockers; profile editing never writes spatial occlusion as `INSIDE`. Before adding an occluder for a target, profile editing removes existing incoming occlusion edges for that target, so one object has at most one direct occluder. It also removes the occluded target's visible placement edges (`ON`, `INSIDE`, `BENEATH`, and related variants) and visible relative-position edges (`LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, `BEHIND`, `CLOSE`, `NEAR`) while preserving structural `PART_OF` and occlusion-chain edges. Profile editing does not add `OCCLUDER` or `DECOMPOSABLE` to ordinary objects. Each output graph stores `requested_constraint_profile`, `achieved_constraint_profile`, `difficulty_tags`, `profile_constraints`, and `graph_edits` in `metadata`. Difficulty tags are spatial-only, for example `spatial.num_occluded_objects=2`, `spatial.occlusion_depth=2`, and `spatial.num_decomposed_parents=1`; temporal, memory, and failure-recovery difficulty should be generated later as task/harness constraints.

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

- Graph-construction scope: generated task graphs contain input material nodes and their part nodes. Support surfaces such as `桌面` must be listed as input materials.
- Task affordances: `GRABBABLE`/`MOVABLE` objects and `CONTAINERS`/`SURFACES` placement targets.
- Memory context: stable visible support or relative-position facts that can be recalled without being exposed in `task`.
- Spatial constraints: profiled variants add canonical `OCCLUDES` layers and optional `DECOMPOSABLE` parent-to-part decompositions later.
- Distractors: nearby objects with ordinary `ON`, `LEFT_OF`, `RIGHT_OF`, `FRONT_OF`, or `BEHIND` relations so constrained objects are not the only objects in the graph.

## Task Types

- `long_horizon`: builds ordered multi-step cleanup/organization tasks from existing graph constraints. It prefers objects that are inside openable containers, occluded, or have parts, and writes the ordered decomposition to `metadata.constraint_subtasks`.
- `navigation`: find or move close to an object.
- `manipulation`: move one grabbable object into or onto a target.
- `multi_object`: move two grabbable objects into or onto one target. `--arms single` produces sequential plans; `--arms double` can pick both objects before placing them.

Use `--task-types all` to enable all built-in task types. The CLI default also includes `long_horizon`.

## Extra Settings

Each setting lives in its own module under `src/auto_embodied_task/constraints`.

- `spatial`: requires task objects to have constrained spatial states such as containment or occlusion. It recognizes relations such as `INSIDE`/`CONTAINS` and `OCCLUDED_BY`/`OCCLUDES`, then writes canonical criteria like `(OCCLUDED_BY, apple, box)`.
- `memory`: stores a prior-observation episode in metadata. It does not alter `task_completion_criterion` and must not be treated as an initial state.
- `temporal`: rewrites applicable criteria with explicit `STEP_1`, `STEP_2`, etc. and stores the ordered subtask resolution in metadata.
- `failure_recovery`: injects a recoverable failed action into the reference plan and records the recovery subtask in metadata.

Extra settings do not alter the natural-language `task` field. The task text stays goal-only; full constraint resolution lives in structured fields such as `settings`, `task_completion_criterion`, `ground_truth_plan`, and `metadata.constraint_subtasks`.

Use `--settings all` to enable all built-in settings. Settings are generated as separate task variants by default, so the base tasks remain available unless `--no-base` is passed.

## Symbolic Trajectory Harness

After generating task JSONL, collect closed-loop trajectories with the built-in symbolic harness:

```bash
PYTHONPATH=src python -m auto_embodied_task collect-trajectories \
  --view-graph examples/generated_office_task_graph.jsonl \
  --tasks outputs/generated_office_tasks.jsonl \
  --output outputs/generated_office_trajectories.jsonl \
  --mode replay
```

The first backend is `symbolic`, so it does not require a simulator. In `replay` mode it replays each task's `ground_truth_plan` through a high-level state machine that tracks object locations, open/closed containers, visibility, reachability, held objects, memory episodes, and recoverable failures. This makes the graph state executable before connecting a physics or robot simulator.

For Guava-style teacher collection, use `teacher` mode. This calls an OpenAI-compatible teacher model at every step, executes the returned JSON action in the symbolic backend, then feeds the next observation back to the teacher:

```bash
DASHSCOPE_API_KEY=... PYTHONPATH=src python -m auto_embodied_task collect-trajectories \
  --view-graph outputs/debug_office_task_graph.jsonl \
  --tasks outputs/debug_office_tasks.jsonl \
  --output outputs/debug_office_teacher_trajectories.jsonl \
  --mode teacher \
  --teacher-provider qwen \
  --teacher-model qwen3.6-plus \
  --teacher-api-key-env DASHSCOPE_API_KEY \
  --max-steps 20
```

Teacher responses must be JSON, for example:

```json
{"reason":"the remembered object is inside the box, so open it first","action":{"name":"open","object":"box"}}
```

Supported action names are `look`, `observe`, `inspect`, `reach`, `walk`, `open`, `close`, `press`, `grab`, `pick`, `attach`, `assemble`, `puton`, `putin`, `move_aside`, `recover`, and `stop`. Teacher trajectories include `teacher_response` on each step. In `teacher` mode the model should choose `stop` when the current state satisfies the success condition; `stop` ends the episode and submits the current state for evaluator scoring.

The harness is routed through a small `WorldBackend` contract with `observe`, `step`, `snapshot`, `success`, and `metrics` methods. The current `SymbolicBackend` implements that contract by wrapping the in-memory `SemanticWorld`. A future simulator backend should implement the same contract while internally calling Isaac, MuJoCo, ManiSkill, or another environment.

Each output JSONL line is one episode with:

- `initial_observation`: visible nodes/edges, held objects, memory episode, and an inferred `map_layout`.
- `trajectory`: one record per step, including the observation before action, parsed action, execution event, state hashes, and whether the task is already successful.
- `final_state`: symbolic world state after replay.
- `metrics`: success plus spatial, temporal, memory, and failure-recovery checks.

`memory` settings are exposed as prior-observation context while the remembered object can be hidden from the initial current observation until the anchor container/surface is inspected or opened. `failure_recovery` steps such as `[failed_grab]` produce injected failure events, and `[recover]` plus the retried action are recorded in the trajectory metrics.

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
