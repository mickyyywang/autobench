# ViewGraphBench：可控干预与失败恢复评测总览

> 更新时间：2026-07-21  
> 用途：交给 Claude Fable 撰写 AAAI paper draft 的项目事实底稿。  
> 论文主线：在长程任务执行中进行可验证、可复现、保持可解的干预，评测 embodied foundation model 能否发现偏差、定位受影响对象、采取恢复动作并最终完成任务。  
> 当前状态：真机开环评测、可执行 View Graph 闭环、六条件 intervention manifests、能力指标和批量实验脚本已实现；统一版本的最终模型结果仍需重跑。模拟器 RL 是计划扩展，不是当前已完成结果。

## 0. Paper in one page

### Working title

**ViewGraphBench: Controlled Mid-Execution Interventions for Evaluating Embodied Foundation Models**

备选：

- **When the World Changes Mid-Task: Benchmarking Recovery with Executable View Graphs**
- **Beyond Nominal Success: Executable View-Graph Interventions for Embodied Agent Evaluation**
- **ViewGraphBench: Stress-Testing Embodied Brains under Action Failures and World-State Disturbances**

### TL;DR

**ViewGraphBench turns partially observable scene graphs into executable test environments that inject verified action failures and world-state disturbances during long-horizon tasks, measuring whether embodied models detect, ground, recover from, and ultimately overcome each intervention.**

### 核心问题

标准 success rate 回答：

> 在环境按预期发展时，模型能否完成任务？

本文要补充回答：

> 当动作没有产生预期效果，或者世界在任务中途偏离了模型已经建立的状态时，模型是否真的在观察环境、更新信念、修复受损子目标并继续完成任务？

### 核心竞争力

View Graph 的主要价值不是“把图片换成图”，而是让干预成为 benchmark 的一等公民：

1. **Executable**：模型动作和外生干预都会改变同一个明确的世界状态。
2. **Partially observable**：模型只看到当前可见投影，不能直接读取被遮挡对象或完整全局状态。
3. **Intervention-ready**：可以在 episode 中途注入动作失败、状态回退、子目标回滚、错误位置和遮挡。
4. **Verified**：干预生成时验证触发条件、状态变化、恢复动作和任务可解性。
5. **Auditable**：每次干预记录对象、时机、前后 state hash、goal cost、planning cost 和可见性变化。
6. **Outcome-grounded**：恢复不是回答一道 failure QA；模型必须在改变后的状态中行动，并继续走到正确完成和 `stop`。

最稳妥的贡献表述是：

> We introduce an executable, partially observable view-graph benchmark for controlled mid-execution interventions, unifying action-outcome failures and exogenous world-state disturbances in long-horizon embodied-brain evaluation.

## 1. 文献驱动的动机

### 1.1 现有 benchmark 已经很好地覆盖 nominal execution

本文不能把现有工作概括成“没有闭环”或“只有短任务”。这既不准确，也会削弱论文可信度。

- [ALFRED](https://openaccess.thecvf.com/content_CVPR_2020/html/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.html) 已经提供部分可观测、平均约 50 步的语言条件 household tasks。
- [TEACh](https://arxiv.org/abs/2110.00534) 加入了对话、歧义消解和交互式任务执行。
- [Habitat 2.0](https://arxiv.org/abs/2106.14405) 和 [BEHAVIOR-1K](https://arxiv.org/abs/2403.09227) 支持大规模、物理驱动的 household mobile manipulation。
- [CALVIN](https://arxiv.org/abs/2112.03227) 测量连续控制策略完成语言技能链的能力。
- [LIBERO](https://arxiv.org/abs/2306.03310) 关注机器人 lifelong learning 和知识迁移。
- [VLABench](https://arxiv.org/abs/2412.18194) 覆盖 world knowledge、implicit instruction 和复合长程操作。
- [RoboCasa365](https://arxiv.org/abs/2603.04356) 已扩展到 365 个 household tasks 和 2,500+ kitchen scenes。

这些 benchmark 证明 simulator closed-loop、long-horizon task 和大规模训练本身都不是本文的 novelty。

但 nominal evaluation 一般从固定或随机化的初始条件开始，让 agent 在没有专门中途破坏的环境中执行，然后报告 success、subtask completion 或 efficiency。它能够很好地测量整体能力，却不一定回答：模型成功是否依赖环境始终符合预期，以及一个已经建立的内部状态被外界否定后模型会怎么做。

### 1.2 受控鲁棒性评测已经存在，但多为 episode-level distribution shift

[LIBERO-Plus](https://arxiv.org/abs/2510.13626) 是必须认真对照的最近工作。它在 object layout、camera viewpoint、robot initial state、language、lighting、background 和 sensor noise 七个维度进行系统扰动，并显示 nominal success 可能掩盖严重 brittleness。

LIBERO-Plus 的贡献说明“controlled perturbation”本身也不是本文首创。本文与其主要区别是干预发生的**时间和语义层级**：

- LIBERO-Plus 主要改变 episode 的初始条件、视觉分布、语言或传感器条件；
- ViewGraphBench 在模型自己的 rollout 中途，依据当前状态和刚刚成功的动作触发干预；
- 干预可以撤销已经完成的子目标或改变已经被模型使用过的世界事实；
- 评测重点不是 OOD success drop 本身，而是 intervention 后的 detection、grounding、repair、resumption 和 stopping。

因此本文研究的是 **mid-execution resilience**，而不是一般的 distributional robustness。

### 1.3 Failure detection and recovery 也不是空白

最接近的 failure 工作包括：

- [AHA / FailGen](https://arxiv.org/abs/2410.00371)：从成功 demonstration 程序化产生失败轨迹，训练 VLM 检测并解释 manipulation failure。
- [FailSafe](https://arxiv.org/abs/2510.01642)：为 VLA 生成失败场景和可执行的低层 recovery actions，并在 ManiSkill 中提高任务表现。
- [RoboRepair](https://arxiv.org/abs/2410.18893)：在机器人程序失败后，从当前世界状态生成 recovery program，避免重复已完成步骤。
- [RoboFailRing](https://aclanthology.org/2026.acl-long.602/)：在执行过程中进行更及时的 failure detection，并用 grounded failure report 改善原因和修复推理。
- [RoboBench](https://arxiv.org/abs/2510.17801)：将 execution/planning failure analysis 纳入 embodied-brain 的五维能力评测。

因此不能声称本文首次提出 failure recovery evaluation。更准确的区别是：

1. 现有 failure 数据常聚焦低层轨迹偏移、抓取/姿态错误、一次失败的识别或 recovery action；
2. 本文同时处理**内生 action-outcome failure**和**外生 world-state disturbance**；
3. 干预发生在长程 task context 中，可能破坏访问条件、已完成目标或对象位置；
4. 恢复正确性由后续可执行状态和最终 goal 决定，而不只是与一条文本解释或 ground-truth correction 做相似度匹配。

### 1.4 高层 embodied-brain 诊断也已有强基线

[Embodied Agent Interface](https://arxiv.org/abs/2410.07166) 将 LLM embodied decision making 分为 goal interpretation、subgoal decomposition、action sequencing 和 transition modeling，并提供 hallucination、affordance 和 planning error 等细粒度指标。[EmbodiedBench](https://arxiv.org/abs/2502.09560) 在四个环境和 1,128 个任务上测量 vision-driven MLLM 能力。[RoboBench](https://arxiv.org/abs/2510.17801) 更直接把 MLLM 定义为 embodied brain，在 5 个维度、14 项能力和 6,092 个 QA pairs 上评测 instruction、perception、planning、affordance 和 failure analysis。

这些工作意味着本文也不能声称首次“超越 success rate”或首次按大脑能力分类。

本文补充的是**在线干预后的行为证据**：模型每一步选择动作，动作改变世界，外生事件再改变世界，下一 observation 来自新状态。模型是否理解失败，最终要看它之后做了什么以及是否恢复任务，而不是只看它能否正确回答 failure-analysis question。

### 1.5 本文实际瞄准的空缺

本文应把 novelty 限定在以下组合：

1. **Mid-execution interventions**：干预不是只在 reset 前改变初始分布，而是在任务执行中改变模型已接触过的世界状态。
2. **Unified failure taxonomy**：统一 action failure、state regression、goal rollback、wrong relocation 和 information loss/occlusion。
3. **State-dependent triggers**：干预依据模型自己的成功动作、当前 task progress 和可见性触发，而不是固定 teacher 时间轴强行注入。
4. **Solvability-preserving verification**：干预前先在后端副本中验证状态变化和至少一种可执行恢复路径。
5. **Closed-loop recovery validation**：检测和解释只是中间能力，最终需要通过 action、state restoration、goal progress 和 correct stop 验证。
6. **Episode-specific generation**：从每个 episode 的 goal、initial graph 和成功 trajectory 自动设计合法干预，不依赖固定物体模板。
7. **Partially observable evaluation**：策略只能看到 `visible_graph_only`，不能读取完整 backend state；主实验不提供 `valid_actions`。

在没有完成更系统文献检索前，应使用：

> “a less explored intersection of intervention-based robustness testing and high-level embodied-brain evaluation”

而不是：

> “the first benchmark for failure recovery”

### 1.6 为什么 View Graph 适合做这种 benchmark

在真机上系统制造“刚放好的东西被别人移走”“已经打开的抽屉又关上”“目标被临时遮挡”等事件，成本高、初始条件难复现，并且不同模型很难遇到完全相同的干预。

完整物理 simulator 可以做到一部分，但如果研究对象是高层 foundation model，结果会同时受到 motion planner、grasp controller、接触物理和视觉 domain gap 影响。View Graph 刻意选择语义层 abstraction：

- 保留 task-relevant object identity、relation、visibility、container、held、assembled 和 capacity state；
- 把低层动作封装为有成功/失败结果的语义 transition；
- 允许在不重建完整数字孪生的情况下精确修改某个世界事实；
- 每次修改可以计算对 goal facts、planning prerequisites 和 visible information 的影响；
- 相同 manifest 可以对多个 foundation models 重复执行。

View Graph 的价值因此是**干预分辨率和可审计性**，而不只是速度。

### 1.7 真机 observation 对照的角色：支持性验证，而不是主故事

[SIMPLER](https://simpler-env.github.io/simpler.pdf) 强调：便宜的模拟评测不能自动被当作真实世界替代品，需要 paired sim-and-real experiments 验证 relative policy performance 和 ranking 是否相关。

本文同样不能默认 View Graph 能代表真实 observation。已有 aligned real-robot trajectories 提供一个 supporting validation：在同一个 task、step 和 gold state 上，分别给模型真实图像和 `visible_graph_only`，比较其 action selection、recovery、exploration 和 stopping。

这个实验的作用是支持以下前提：

> View Graph 至少保留了足够的高层决策信息，使后续 intervention benchmark 不只是一个与真实观察完全脱节的文字游戏。

它不是论文主贡献，也不证明端到端 graph extraction 已解决。固定真机轨迹必须使用 `teacher history`；使用 inference history 会产生 history–observation inconsistency。

### 1.8 Closest-work comparison

| Work family | 干预时机 | 主要异常 | 评测对象 | recovery 如何验证 | 与本文的关键边界 |
| --- | --- | --- | --- | --- | --- |
| ALFRED / BEHAVIOR-1K / CALVIN / RoboCasa365 | 任务 reset 后进行 nominal rollout | 初始状态变化和执行难度，不以统一中途破坏为核心 | 完整 agent/policy | task/subtask success | 提供长程闭环任务，但不把 runtime intervention 当作实验自变量 |
| LIBERO-Plus | 主要改变 evaluation episode 的布局、视角、机器人初态、语言、光照、背景和噪声 | distribution shift / robustness perturbation | VLA policy | perturbation 下 task success | 最重要的 robustness 近邻；本文关注模型已执行若干步后世界事实被否定及其修复过程。最终论文需从原始实现再次核实其 perturbation timing |
| AHA / FailGen | 对成功 demonstration 进行程序化失败生成 | manipulation failure trajectory | failure-detection VLM | detection、explanation 及 downstream benefit | 本文让同一个决策模型在干预后的新状态中继续行动，直到恢复或失败 |
| FailSafe | 在 execution stage 制造低层偏差 | object/robot pose deviation 和低层 corrective action | VLM/VLA recovery module | corrective action 与 closed-loop improvement | 本文的干预和动作处在高层语义状态层，并覆盖外生状态回退、目标回滚和遮挡 |
| RoboRepair | 程序执行出错后 | 当前 world state 与原计划不一致 | recovery-program generator | recovery program 的执行效率/成功 | 本文不生成一次性完整修复程序，而是逐 observation 决策并承受后续部分可观测性 |
| EAI / EmbodiedBench / RoboBench | 构造 task、module 或 QA 样本 | planning、affordance、transition 和 failure reasoning errors | plan、MLLM QA 或模块能力 | matching、simulated transition 或细粒度分数 | 本文把 failure reasoning 变成可执行 rollout 中的行为结果 |
| **ViewGraphBench** | **模型自己的 rollout 中途，按实时 action/state trigger** | **action-outcome failure + exogenous semantic world change** | **高层 embodied policy** | **确定性状态转移、局部状态恢复、全局 goal completion 和 stop** | **研究 intervention 后的 detect → ground → repair → resume，而非只测扰动后的平均 success** |

这张表是定位框架，不应直接替代逐篇事实核查。特别是 2025–2026 的并发工作发展很快，最终 Related Work 必须重新检查 paper、supplement 和公开代码中的实际 intervention timing。

### 1.9 Research questions

- **RQ1 — Nominal competence**：不同 embodied foundation models 在无干预的 `visible_graph_only + no-valid` 闭环中能否完成长程任务并正确停止？
- **RQ2 — Intervention resilience**：五类中途干预分别造成多大 success/progress/efficiency drop？
- **RQ3 — Failure understanding**：模型能否检测 action failure，并正确 grounding 失败动作及对象？
- **RQ4 — State repair and resumption**：模型能否修复被回退、移错或遮挡的状态，同时保留其他已完成子目标并继续完成全局任务？
- **RQ5 — Capability diagnosis**：recovery、exploration、action selection 和 stop metrics 能否解释 intervention success 的模型差异？
- **RQ6 — Proxy validity（支持性）**：在 aligned real observations 上，`visible_graph_only` 与 `obs_only` 的模型排序和能力分数有多一致？
- **RQ7 — Downstream RL（计划）**：使用 intervention rollouts 和 graph-derived progress signals 训练，能否提升未见干预下的恢复能力？

## 2. Intervention taxonomy

本文把异常分成两个大类：

```text
Endogenous execution failure
  └── action_failure_once_per_action_type

Exogenous world-state disturbance
  ├── state_regression
  ├── completed_subgoal_rollback
  ├── wrong_container_relocation
  ├── add_occlusion
  └── add_object
```

两类失败需要的 reasoning 不同：

- action failure：意图正确，但 action outcome 与预期不一致；需要检测、grounding、retry 或 replan。
- state disturbance：过去动作可能确实成功，但之后世界又变化；不能把它错误归因于原动作失败，而应重新观察并修复当前状态。

### 2.1 八个独立 condition 条目

每个 v5 manifest 有 8 个 condition 条目；每个可执行 condition 都从同一个 `initial_view_graph` 重新开始并清空模型 history。旧 episode 当前有 7 个可执行 condition；不满足第二类 `add_object` 数据前置条件的条目保留为 `eligible: false`，runner 会跳过。

| Condition | 触发与改变 | 测量重点 |
| --- | --- | --- |
| `baseline` | 无干预 | nominal competence |
| `action_failure_once_per_action_type` | 模型选择的、本来可成功的每种 action name 首次执行时注入一次失败 | failure detection、grounding、retry/replan |
| `state_regression` | episode-specific 状态动作成功后，在下一 observation 前恢复该状态，例如 `open -> false` | 是否重新检查访问状态并重做必要动作 |
| `completed_subgoal_rollback` | 目标 placement 成功后，把物体移回原 surface | 是否发现已完成目标被破坏并恢复 |
| `wrong_container_relocation` | 目标 placement 成功后，把物体移入错误但合理的容器 | 是否纠正 plausible wrong state |
| `add_occlusion` | 在中程撤销一个已完成 placement，并在临时位置加入可解除遮挡 | 主动探索与目标修复 |
| `add_object_inherit_source_goal` | 源对象局部目标满足后，引入其 copy/同类物体，并动态继承相同局部成功标准 | 新物料出现后的目标扩展与继续整理 |
| `add_object_existing_task_goal_at_capacity` | 容器达到 `max_items` 后，引入已写入 task goal、但初始图中不存在的物体 | 满容器下的重新规划；仅对满足数据前置条件的 episode 可执行 |

### 2.2 Action failure

注入失败前，runner 会在 backend 副本中执行预测动作。只有本来能成功的动作才会变成 `failed_<action>`，避免把模型自身的不可执行动作错误计为 injected failure。

当前正式 condition 使用：

```text
mode = all
deduplication_scope = action_name
only_normally_successful_actions = true
```

因此不是“第一个动作失败”，也不是 27 个 action signatures 都失败，而是模型实际使用的每一种可注错 action type 最多失败一次。一个 rollout 可出现多个 failures。

### 2.3 State regression

该 condition 选择 teacher 成功轨迹中一个具有状态效果的动作，例如：

```text
open(收纳盒第一层)
move_aside(大碗)
```

模型在自己的 rollout 中成功完成相同动作后，干预在下一 observation 之前把状态回退。它通常破坏 access prerequisite，而不一定直接破坏 goal fact，所以应主要增加 planning cost，而不是强行增加 goal completion cost。

### 2.4 Completed-subgoal rollback

该 condition 在模型完成一个严格 goal placement 后，把物体移回 episode 中合法的 surface。它明确破坏一个已经满足的目标事实，测试模型是否持续维护 task state，而不是“做过一次就永久认为完成”。

### 2.5 Wrong-container relocation

该 condition 把刚正确放置的物体移动到错误但 plausible 的容器。错误目标会排除该物体所有合法目标，并优先使用本 episode 真实出现过的 placement target。

这比简单移回桌面更难：模型必须识别对象仍然存在、容器也合理，但当前 relation 与 task goal 不一致。

### 2.6 Dynamic add-occlusion

`add_occlusion` 不在初始图直接制造干扰。它从 step 3 开始，在 direct goal progress 为 10%–80% 的窗口中寻找当前可用候选，并且最多注入一次。condition 名仍为 `add_occlusion`，运行时会物化为 `relocate_and_add_occlusion`：先把目标移出正确位置，再建立遮挡。

候选必须满足：

- source 和 target 当前均可见且未被拿着；
- target 当前已经满足一个具体的 `ON`/`INSIDE` goal branch，但全局任务尚未完成；
- source 和 target 位于同一个根 surface，且该 surface 不等于 target 的正确 placement；
- 新 edge 当前未激活且确实会隐藏 target；
- 干预后 goal completion cost 与 planning cost 均上升；
- 后端副本验证 `open(source)` 或 `move_aside(source)` 能揭示 target，并可继续 `grab` 后恢复记录的原始 placement。

实际解除动作：

```text
if source supports open and is closed:
    resolution = open
elif source supports move_aside and is not moved aside:
    resolution = move_aside
else:
    try next candidate
```

实际 source、target 和 resolution 写入：

```text
disturbances_applied[].runtime_selection
disturbances_applied[].spec
```

`spec.previous_location` 和 `runtime_selection.staging_location` 记录正确位置与临时位置。不同模型走到不同状态时可以选中不同候选；这是 state-dependent intervention，而不是复用 teacher step 的固定遮挡。

### 2.7 Dynamic add-object

闭环 manifest 支持两类 `add_object`：

- `on_object_goal_satisfied + inherit_from`：新 copy/同类物体继承源对象投影后的局部 goal；runtime 把替换 subject 后的表达式动态并入 effective criterion。
- `on_container_max_items_reached + existing_task_goal`：新增物体必须已经出现在原始 `task_completion_criterion` 中、但不在初始 view graph；runtime 不改 criterion。

生成器不会为第二类伪造 goal。若旧 episode 的初始图已经包含 task goal 中的所有对象，则对应条目标记为 `eligible: false` 并写入 `ineligible_reason`。第一类只选择可由独立 copy 完成的纯 `ON`/`INSIDE` 局部 placement goal，避免继承无法复制的 assembly/attachment 前置结构。

## 3. View Graph execution model

### 3.1 完整状态与可见投影

系统内部维护完整状态 \(G_t\)，模型只接收：

\[
O_t^{VG}=P_{visible}(G_t).
\]

`visible_graph_only` 当前字段：

| 组成 | 暴露内容 |
| --- | --- |
| `visible_nodes` | `id`、`name`、`category`、`open`、`assembled`、`pressed`、`is_full`（存在时） |
| `visible_edges` | 两端都可见时的 `from`、`to`、`relation` |
| `held_objects` | `id`、`name` |
| `robot.hands` | 手部占用状态 |

不保留：

- `map_layout`；
- 不可见节点及其 edges；
- 完整 properties/affordance/profile metadata；
- 精确 `max_items` 和容器 item count。

容量仅保留当前 `is_full`。任务文本和 success criterion 会命名目标对象，但不泄露其隐藏位置和当前 relation。

三层抽屉使用有方向的动态结构遮挡：第一层打开时遮挡第二、三层，第二层打开时遮挡第三层，第三层不会反向遮挡上层。图中以 `OCCLUDES(source, target, resolution_action=close)` 表示；边仅在 source 为 `OPEN` 时激活。下层抽屉不可见时，其所有 `INSIDE` 后代也不可见；关闭上层后，系统依据其他仍打开的上层抽屉重新计算可见性。`close` 因此也可以成为 exploration opportunity，并在揭示 goal-relevant 下层抽屉或内容时获得 information gain / soft-optimal 收益。

### 3.2 动作空间

```text
look, observe, inspect, reach, walk,
open, close, press, grab, attach,
puton, putin, move_aside, stop
```

- `pick -> grab`，`place_in -> putin`，`place_on -> puton`。
- `assemble` 已从模型动作空间移除，装配使用 `attach`。
- `attach` 在 no-valid 模式下参数无序匹配；`putin` / `puton` 有序。
- `recover` 目前仍存在于 legacy catalog，但 request 把 recovery 定义成结构化判断，闭环 evaluator 拒绝把 `recover` 当作物理 action。正式冻结前建议把 legacy catalog entry 也删除，避免 prompt 矛盾。

### 3.3 闭环顺序

```text
G_t
  -> intervention_runtime.before_step()       # 外生状态变化
  -> backend.observe()
  -> visible projection O_t
  -> model predicts action and recovery flag
  -> optional injected action failure
  -> backend.step(action)
  -> score state/action/capabilities
  -> intervention_runtime.after_model_action()
  -> G_{t+1}
```

`before_step()` 负责检查 trigger 并在当前 observation 生成前修改 backend，因此模型能看到变化后的状态，但不会收到“发生了某干预”的文字提示。

`after_model_action()` 推进 manifest runtime 的 trigger/cleanup 状态，不负责执行模型动作本身。

## 4. 数据集状态

当前 [`saved/`](../saved) 顶层有 **16 个 aligned episodes**，共 **840 条 trajectory records**；单 episode 为 **44–61** 条，均值 **52.5**。

| Task family | Episodes | 长程结构 |
| --- | ---: | --- |
| 化妆品收纳 B_1/B_4/B_9/B_10 | 4 | 多容器、遮挡、打开、装配和多物体收纳 |
| 整理办公桌面 B_1/B_4/B_7/B_8/B_11/B_12/B_13/B_14 | 8 | 文具入盒并关闭、书本/废纸归位、容量限制 |
| 整理餐桌 A_3/A_4/A_5/A_6 | 4 | 嵌套/叠放、托盘、餐具架和菜篮 |

每个 aligned episode 包含 task、结构化 completion criterion、`initial_view_graph`、teacher trajectory、真机 observation alignment 和执行 metadata。

[`exp/intervention_manifests/`](./intervention_manifests) 中已有 13 个旧 episode 的 manifest；B_12～B_14 尚未纳入这批旧 manifest。当前版本：

```text
manifest_version = 5
generation_algorithm = episode_semantic_v5_with_add_object
```

生成器逐 episode 分析 goal、initial graph、teacher action sequence、合法/错误目标和遮挡候选，不使用 cosmetics/office/table 的固定对象模板。

## 5. Evaluation protocols

### 5.1 主评测：View Graph intervention closed loop

- 从同一 initial graph 为每个 condition 重置 backend；
- `visible_graph_only`；
- `history_source=inference`；
- `no-valid-actions`；
- 默认 max 100 steps，history window 8；
- 连续 3 次 API/解析错误后终止；
- goal 满足后仍需模型输出 `stop` 才算完整成功。

正式规模：

```text
13 episodes × 5 models × 7 eligible conditions = 455 independent rollouts
```

### 5.2 支持性评测：real-observation open loop

每个 aligned step 从 teacher gold state 开始，模型只做一步反事实 action；下一 step 不延续预测状态。

主 comparison：

```text
obs_only vs visible_graph_only
teacher history
no valid actions
```

它用于验证 representation fidelity，不报告闭环 task success。`inference history` 不适用于固定真机 observation。

## 6. Metrics

### 6.1 当前四个能力维度

| 能力 | 指标 | 方向 |
| --- | --- | --- |
| Action selection | `action_admissibility_rate` | ↑ |
| Action selection | `soft_optimal_action_score` | ↑ |
| Failure recovery | `recovery_detection_f1` | ↑ |
| Failure recovery | `recovery_grounding_accuracy` | ↑ |
| Active exploration | `exploration_opportunity_recall` | ↑ |
| Active exploration | `normalized_goal_information_gain` | ↑ |
| Completion judgment | `premature_stop_rate` | ↓ |
| Completion judgment | `completion_stop_recall` | ↑ |

`action_admissibility` 只判断动作前是否合理可尝试，不保证执行成功。`normalized_goal_information_gain` 是目标隐藏节点 reveal coverage，不是 Shannon entropy。

### 6.2 Goal cost 与 planning cost

当前 capability metric version 为 **v5**：

- `goal_completion_cost`：直接未满足 goal facts；
- `planning_cost`：goal actions 加去重的访问 prerequisites，例如 `open`、`move_aside`、`grab`。

soft-optimal：

\[
s(a)=\exp[-\beta(C_{plan}(a)-C^*_{plan})], \quad \beta=1.
\]

normalized progress 使用 goal cost：

\[
progress=clip((C_{goal}^{init}-C_{goal}^{final})/C_{goal}^{init},0,1).
\]

这一区分对 intervention 很重要：state regression 可以只增加 planning prerequisites；v4 的 compound add-occlusion 会同时撤销 placement goal fact，并增加解除遮挡的 planning prerequisite。

### 6.3 闭环 outcome metrics

当前已经输出：

- `task_success_rate`；
- `goal_ever_satisfied_rate`；
- `final_goal_satisfied_rate`；
- `normalized_goal_progress`；
- `teacher_normalized_efficiency`；
- `action_executability_rate`；
- `average_step_count`；
- termination reasons；
- `intervention_applied_rate` 和实际 intervention count。

### 6.4 为 intervention paper 建议新增/离线统计的核心指标

以下最能支撑主故事，其中一部分可从现有 step records 离线计算，但尚未全部作为 summary 字段实现：

1. **Intervention-conditioned success**：只在干预实际发生的 rollout 上统计最终成功。
2. **Robustness ratio**：同 model/episode 下 `success_condition / success_baseline` 或 paired success delta。
3. **Recovery latency**：从 intervention step 到受损 goal/planning state 首次恢复所需步数。
4. **Progress recovery ratio**：最终恢复了干预造成的多少 goal/planning cost 增量。
5. **Repair action precision**：干预后首次有效动作是否针对正确对象和 blocker。
6. **Regression recurrence**：是否反复执行无关动作或重新破坏已恢复子目标。
7. **Post-recovery efficiency**：相对同 episode baseline 多出的 action overhead。
8. **Resumption success**：局部状态修复后是否继续完成剩余全局目标，而不是停在局部恢复。

论文不能只报告 condition success。若某个模型从未执行 manifest 的 trigger action，干预不会发生；必须同时报告 `intervention_applied_rate`。

## 7. Experiment design

### Experiment A — Nominal baseline

5 models × 13 episodes，`visible_graph_only + no-valid`，报告 success、progress、efficiency、executability 和 stop behavior。

### Experiment B — Controlled intervention suite

对每个 model/episode 比较 baseline 与五类 intervention。主要表应包含：

```text
applied rate
task success
success delta from paired baseline
final goal progress
recovery detection / grounding
recovery latency
post-recovery efficiency
```

### Experiment C — Intervention-type analysis

比较：

- action failure：是否知道 action outcome 失败；
- state regression：是否更新已观察过的 access state；
- goal rollback：是否维护已完成子目标；
- wrong relocation：是否能纠正 plausible-but-wrong relation；
- occlusion：是否能从目标消失推断需要探索。

该分析是论文最重要的 capability decomposition。

### Experiment D — Valid-actions ablation

比较 valid/no-valid，用于量化 oracle candidate list 隐藏了多少 action generation 和 grounding 难度。主结果仍应 no-valid。

### Experiment E — Real-to-graph supporting validation

在真机 aligned trajectory 上比较 `obs_only` 与 `visible_graph_only`：

- paired per-episode gap；
- model rank correlation；
- task-family gap；
- capability-specific gap。

### Experiment F — RL extension（未实现）

可以用 intervention rollouts 构造 curriculum 或 reward：

```text
r_t = goal_progress_delta
    + planning_cost_delta
    + correct_recovery_bonus
    + goal_relevant_exploration_bonus
    + correct_stop_bonus
    - non_executable_penalty
    - step_penalty
```

训练 policy 仍只能看到 partial observation；full graph 只用于 transition、reward 和 verification。

## 8. Models

批量脚本当前配置：

- `qwen3.6-plus`
- `gpt-5.5`
- `gpt-5.4-2026-03-05`
- `claude-opus-4-7`
- `gemini-3.1-pro-preview`

正式论文记录 exact model ID、provider、调用日期、temperature=0 和 retry config。

## 9. Experiment scripts

从项目根目录运行：

```bash
cd /home/wmq/project/bench/auto_embodied_task
```

### 9.1 生成和验证 manifests

只设计和验证，不写文件：

```bash
python scripts/generate_saved_intervention_manifests.py \
  --saved-dir saved \
  --output-dir exp/intervention_manifests \
  --dry-run
```

覆盖重建全部：

```bash
python scripts/generate_saved_intervention_manifests.py \
  --saved-dir saved \
  --output-dir exp/intervention_manifests \
  --overwrite
```

正式数据不建议使用 `--no-validate`。

### 9.2 正式 closed-loop intervention evaluation

先查看 65 个 model/manifest commands：

```bash
DRY_RUN=1 ./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
```

运行全部 390 rollouts：

```bash
./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
```

筛模型：

```bash
MODEL_FILTER='qwen3.6-plus' \
./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
```

筛 condition：

```bash
CONDITIONS='add_occlusion' \
./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
```

筛 episode：

```bash
MANIFEST_FILTER='化妆品收纳B_1,整理餐桌A_3' \
./scripts/evaluate_all_view_graph_manifest_closed_loop.sh
```

关键环境变量：

| 变量 | 默认值 |
| --- | --- |
| `CONDITIONS` | `all` |
| `MAX_STEPS` | `100` |
| `HISTORY_WINDOW` | `8` |
| `MAX_CONSECUTIVE_MODEL_ERRORS` | `3` |
| `SOFT_OPTIMAL_BETA` | `1.0` |
| `FAIL_FAST` | `0` |
| `STOP_ON_ERROR` | `0` |

### 9.3 单 condition debug

VS Code：

```text
Debug View Graph Closed Loop: qwen3.6-plus (failure + disturbance)
```

或：

```bash
MANIFEST='/home/wmq/project/bench/auto_embodied_task/exp/intervention_manifests/化妆品收纳B_1_intervention_manifest.json' \
MODEL='qwen3.6-plus' \
./scripts/evaluate_view_graph_intervention_manifest.sh state_regression
```

### 9.4 Real observation supporting experiment

```bash
MODES='obs_only,visible_graph_only' \
HISTORY_SOURCE='teacher' \
./scripts/evaluate_real_models_valid_compare.sh no-valid
```

valid/no-valid ablation：

```bash
MODES='obs_only,visible_graph_only' \
HISTORY_SOURCE='teacher' \
./scripts/evaluate_real_models_valid_compare.sh both
```

### 9.5 Visualization

```bash
./scripts/serve_evaluation_replies.sh
```

默认：

```text
http://127.0.0.1:8771/brian_eval/
```

## 10. Output and versioning

输出自动附加时间戳，不覆盖旧结果。每个 JSONL 有 `__summary.json`；manifest runner 另有 suite summary。

模型响应诊断保存在每条 record 的 `response_metadata` 中；Gemini 会记录每次请求的 `finishReason`、`finishMessage`、usage 和 token 上限。`parse_repair=trailing_closing_braces` 表示只移除了完整首个 JSON 后多余的右花括号。

最终论文结果锁定：

```text
capability_metric_version = 5
closed_loop_metric_version = 3
manifest_version = 5
soft_optimal_beta = 1.0
```

`evaluations/` 中已有历史结果，但部分生成于 history、action catalog、exploration、recovery、visible projection、dynamic occlusion 和 cost split 修复之前，不应直接用于最终论文表格。

最近实现验证：

- full tests：166 passed；另有 1 个 inventory check 因 B_12～B_14 尚无 manifest 而失败；
- 13/13 manifests semantic validation passed；
- add-object inherit：13/13 trigger、动态 goal 扩展、grab + restore 验证通过；existing-task-goal 类型在旧 episode 中均因不存在“goal 已引用但初始图缺失”的对象而标记为 ineligible；
- add-occlusion：13/13 completed placement relocated，target hidden，goal/planning cost increased，并通过 reveal + grab + restore 验证；
- completed rollback 和 wrong relocation：13/13 goal/planning cost increased；
- state regression：13/13 state changed and planning cost increased。

这些是 benchmark implementation checks，不是模型结果。

## 11. Code map

| 功能 | 文件 |
| --- | --- |
| Symbolic backend/actions/failure | [`src/auto_embodied_task/harness.py`](../src/auto_embodied_task/harness.py) |
| Real open-loop/projection/metrics | [`src/auto_embodied_task/real_observation_eval.py`](../src/auto_embodied_task/real_observation_eval.py) |
| Closed loop/intervention runtime | [`src/auto_embodied_task/view_graph_rollout_eval.py`](../src/auto_embodied_task/view_graph_rollout_eval.py) |
| Manifest generator | [`scripts/generate_saved_intervention_manifests.py`](../scripts/generate_saved_intervention_manifests.py) |
| Formal batch evaluation | [`scripts/evaluate_all_view_graph_manifest_closed_loop.sh`](../scripts/evaluate_all_view_graph_manifest_closed_loop.sh) |
| Real-observation evaluation | [`scripts/evaluate_real_models_valid_compare.sh`](../scripts/evaluate_real_models_valid_compare.sh) |
| Visualization | [`src/auto_embodied_task/evaluation_reply_server.py`](../src/auto_embodied_task/evaluation_reply_server.py) |
| Detailed intervention notes | [`view_graph_closed_loop_failure_and_disturbance.md`](./view_graph_closed_loop_failure_and_disturbance.md) |
| Detailed metric notes | [`real_trajectory_evaluation_metrics_and_usage.md`](./real_trajectory_evaluation_metrics_and_usage.md) |

## 12. 推荐论文结构

1. **Introduction**：nominal success 不等于 resilience；机器人必须处理 action-outcome error 和外生 world change。
2. **Related Work**：nominal embodied benchmarks、robustness perturbations、failure detection/recovery、embodied-brain diagnostics。
3. **ViewGraphBench**：partial executable graph 与 task/goal/action formalization。
4. **Controlled Intervention Suite**：五种 intervention、trigger、verification 和 solvability。
5. **Metrics**：detection → grounding → repair → resumption → completion。
6. **Experiments**：baseline、condition deltas、capability diagnosis、real-to-graph supporting validation。
7. **Optional RL**：只有有真实训练结果时加入。
8. **Limitations**。

推荐主图：

- Figure 1：nominal rollout 与五类 intervention 的统一状态机。
- Figure 2：一次 completed rollback 或 wrong relocation 的完整时间线。
- Table 1：dataset/manifest statistics。
- Table 2：baseline 与五 conditions 的 applied rate、success、progress、latency 和 efficiency。
- Table 3：模型的 detection/grounding/repair/resumption 分解。
- Table 4：obs-only vs visible-graph supporting validation。

统计以 episode 为聚类单位做 paired bootstrap；不要把同一 episode 的几十个 steps 当成独立样本。

## 13. 不能过度声称

1. 不是第一个 long-horizon、closed-loop 或 failure-recovery benchmark。
2. 不是第一个 controlled perturbation benchmark；LIBERO-Plus 是重要近邻。
3. 当前图来自构造、编辑和 alignment，没有独立评测自动 graph extraction。
4. 符号语义执行不等同于真实接触物理；本文隔离的是 high-level brain。
5. 13 episodes、3 task families 仍是小规模 pilot，最好继续扩展。
6. 部分 trigger 是 policy-dependent，必须报告 applied rate。
7. planning cost 是 heuristic，不是 exact shortest path。
8. exploration 目前主要覆盖 goal-related `open` / `move_aside`。
9. teacher trajectory 用于 manifest 设计/验证和效率参照，但不会把未来 teacher action 提供给闭环模型。
10. RL 尚无当前仓库结果，不能写成已完成贡献。

## 14. Claude Fable 文献调研任务

不要让 Claude Fable 直接根据本文档自由补引用。建议连同下面的 prompt 一起提供：

```text
You are preparing the Related Work and Introduction for an AAAI paper about
controlled mid-execution interventions for embodied foundation models.

The proposed benchmark uses an executable, partially observable view graph to
inject action failures and exogenous world-state disturbances during a model's
own long-horizon rollout. Conditions include action failure, state regression,
completed-subgoal rollback, wrong-container relocation, and dynamic occlusion.
Recovery is validated by subsequent executable actions and final task outcome.

Conduct a source-grounded literature review using the latest primary papers.
At minimum compare:
1. nominal interactive benchmarks: ALFRED, TEACh, Habitat 2.0, BEHAVIOR-1K,
   CALVIN, LIBERO, VLABench, RoboCasa365;
2. robustness/perturbation benchmarks: LIBERO-Plus and any 2025-2026 successors;
3. failure detection/recovery: AHA/FailGen, FailSafe, RoboRepair, RoboFailRing,
   and relevant real-robot recovery benchmarks;
4. high-level embodied-brain diagnostics: Embodied Agent Interface,
   EmbodiedBench, RoboBench, RoboVQA;
5. proxy validation: SIMPLER and related real-to-sim evaluation work.

For every paper, verify from the primary source:
- whether perturbations occur before reset or during execution;
- whether the evaluated unit is QA, a complete plan, a high-level policy, or a
  low-level VLA;
- whether recovery is text-matched, action-matched, or validated by closed-loop
  task completion;
- whether observations are real, simulated visual, full symbolic state, or
  partially observable state;
- whether intervention solvability and actual trigger coverage are verified.

Return:
A. a comparison table;
B. the five closest papers and precise non-overlapping differences;
C. defensible novelty wording without “first” unless exhaustively supported;
D. BibTeX from official proceedings/arXiv;
E. any paper that materially invalidates the proposed positioning.

Do not invent benchmark limitations, citations, model results, or completed RL
experiments. Treat real-image vs view-graph evaluation as supporting proxy
validation; the primary story is controlled intervention and failure recovery.
```

### 当前最需要 Claude 核查的近邻

1. **LIBERO-Plus**：确认全部 perturbations 是否都在 episode reset/initialization 阶段，以及是否存在运行中 target relocation。
2. **FailSafe**：明确其 failure injection 是 low-level pose deviation、恢复 action 的粒度，以及与 high-level semantic state rollback 的边界。
3. **RoboRepair**：检查 11-task recovery-program benchmark 是否包含外生 state change 和最终 closed-loop execution。
4. **RoboBench**：准确描述 MLLM-as-world-simulator 与本文确定性 executable backend 的区别。
5. **2026 concurrent work**：重点搜索 runtime perturbation、subgoal rollback、object relocation、occlusion recovery、embodied resilience benchmark。

如果这些近邻中已有工作完整覆盖“长程执行中、state-dependent、solvability-verified semantic interventions + partial observation + closed-loop recovery outcome”，论文必须进一步收缩 novelty 或把贡献转向 real-observation calibration、episode-specific manifest generation 和 capability metrics。
