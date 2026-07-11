# Properties Skill

只使用这些 properties 枚举：

- `GRABBABLE`: 可以被机器人抓取。
- `MOVABLE`: 可以被移动；可操作任务物体通常同时需要 `GRABBABLE` 和 `MOVABLE`。
- `SURFACES`: 可以作为 `ON` 的目标；没有该属性的节点不能作为 `ON` 目标。
- `CONTAINERS`: 可以作为后续放入任务或 profile editor 的容器目标；初始 view graph 不输出 `INSIDE` 边。
- `CAN_OPEN`: 可以打开/关闭；如果使用它，通常需要在 `states` 中写 `OPEN` 或 `CLOSED`。
- `PRESSABLE`: 可以被机器人按下；适合按钮、开关等，按下后可用 `PRESSED` 作为任务完成谓词。
- `OCCLUDER`: 有遮挡能力的节点。初始 view graph 只保留这个能力属性，不输出 `OCCLUDES` 边；遮挡关系由后续 profile editor 根据难度 profile 添加。
- `COPYABLE`: 预留给后续 task/harness 层面的 memory 干扰物逻辑；spatial profile editor 不会复制节点。
- `DECOMPOSABLE`: 允许 profile editor 在 spatial 难度中把父物体拆成已有部件节点参与对外关系；没有该属性的父物体不能被自动拆解。

`parts` 和 `part_of` 用来描述结构部件。生成 view graph 时可以让父物料整体 node 参与对外关系，也可以让部件 nodes 参与对外关系；两者不能混用。如果输出部件 nodes，就用部件 nodes 表达该物料的对外位置或相对位置关系，父物体 node 只通过 `PART_OF` 管理结构。如果整体 node 参与这些对外关系，部件 nodes 只保留结构 `PART_OF`；后续只有带 `DECOMPOSABLE` 的父物体才能由 profile editor 拆成部件参与对外关系。
