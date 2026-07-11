# Relation Skill

只使用这些 relation 枚举：

- `ON`: 物体在另一个输入物料或输入物料 part 表示的支撑面上；目标节点必须有 `SURFACES` 属性。
- `BENEATH`: 物体在另一个输入物料或输入物料 part 的下方或下层。
- `PART_OF`: 子部件属于父物体；只有父物体 node 被保留时才使用这条结构边。
- `LEFT_OF` / `RIGHT_OF` / `FRONT_OF` / `BEHIND`: 当前视角下的相对位置。
- `CONNECTED`: 只在房间或区域本身是输入物料时使用。

节点参与规则：

- 每个保留在 `view_graph.nodes` 中的节点都必须至少出现在一条 edge 的 `from` 或 `to` 中；`parent` / `part_of` 字段不能替代 edge。
- 同一个物料可以用整体 node 参与对外关系，也可以用部件 nodes 参与对外关系，但不能混用。
- 如果用部件 nodes 参与对外关系。整体 node 只能通过 `PART_OF` 连接这些部件。
- 如果用整体 node 参与对外关系。部件 nodes 只能通过 `PART_OF` 连接整体。