# 规范化定义目录

存放经 Loader 解析、规范化后的定义文件。

## 当前文件

| 文件 | 说明 |
|------|------|
| `xinsuo.json` | 由 `心所定义.docx` 导出的结构化心所定义，共 **40 条** |
| `xinsuo_duizhi.json` | 由 `心所对治关系.docx` 导出的对治关系（完全互斥），共 **40 条**（40 心所各 1 条 source） |
| `suoyuan_xiang_si.json` | 由 `所缘:想:思的定义.docx` 导出的提取定义 |

字段：`name`、`suoyuan`、`shengqi_nian`、`ye`、`xiandai_xinli`。Schema 见 `definitions/schemas/xinsuo.schema.json`。

对治关系字段：`source`、`target`、`description`、`relation_type`（统一为「对治」）、`is_direct`。Schema 见 `definitions/schemas/relation.schema.json`。

`source` 与 `target` 均已规范化为与 `xinsuo.json` 一致的标准名称（如 `痴`→`无明`，`无嗔`→`无瞋`）。
