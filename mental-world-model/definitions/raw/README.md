# 原始定义文件目录

将用户提供的原始定义文件放置于此目录。**修改 raw 中的文档后，下次 `MentalWorldModel.load()` 会自动同步到对应的 JSON 文件**（当 raw 文件比 JSON 更新时）。

## 同步策略

| 定义 | 同步目标 JSON | raw 匹配模式 |
|------|---------------|--------------|
| 心所 | `normalized/xinsuo.json` | `*心所定义*.docx` / `.md` / `.json` |
| 对治关系 | `normalized/xinsuo_duizhi.json` | `*对治*.docx` / `.md` / `.json` |
| 所缘/想/思 | `normalized/suoyuan_xiang_si.json` | `*所缘*想*思*.docx` / `.md` / `.json` |
| 心所种子库 | `seeds/xinsuo_seeds.json` | `*种子*.docx` / `*种子库*.docx` / `.md` / `.json` |

规则：

1. **raw 比 JSON 新** → 重新解析 raw，覆盖写入 JSON
2. **仅存在 JSON** → 直接读取 JSON（兼容无 raw 的仓库）
3. **仅存在 raw** → 解析并生成 JSON
4. **同类型匹配到多个 raw 文件** → 报错，请只保留一个

## 支持的格式

- **Word（`.docx`）** — 推荐；心所/对治/所缘为段落结构，种子库为表格（列：类别 | 心所 | 默认种子分数）
- **Markdown（`.md`）** — 需含 ` ```json ` 代码块
- **JSON（`.json`）** — 结构与对应 normalized JSON 一致

## 种子库表格示例（Word）

| 类别 | 心所 | 默认种子分数 |
| ---- | ---- | -----------: |
| 善 | 信 | 0.60 |
| | 精进 | 0.55 |

类别列可留空，沿用上一行的类别。
