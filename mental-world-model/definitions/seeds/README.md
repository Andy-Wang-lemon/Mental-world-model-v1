# 心所种子库权重目录

存放 40 心所的种子库权重 `W_seed`（程序读取的 JSON），参与最终分合成：

```
final = S_col × W_seed
```

**总开关**：`engine/seeds/config.py` → `SEED_LIBRARY_ENABLED`（**默认 False**）。

- **关闭**：不加载本目录 JSON 到评分乘子；提取阶段跳过路由 H；EMA 不更新。
- **开启**：加载 `xinsuo_seeds.json` 作为 `W_default`，并与运行时 `seed_state.json` 合并使用；路由 Top-6、最终分 × W_seed、交互后 EMA 更新均生效。

## 源文件与同步

**推荐在 `definitions/raw/` 维护源文档**（如 `心所种子库.docx`），表格列：

| 类别 | 心所 | 默认种子分数 |

`MentalWorldModel.load()` 时（**种子库开启时**）若 raw 比本目录的 `xinsuo_seeds.json` 更新，会自动重新解析并覆盖此文件。

也可直接编辑 `xinsuo_seeds.json`（须覆盖全部 40 个心所）。

加载入口：`DefinitionLoader.load_xinsuo_seed_definitions()` → `build_default_seed_weights()`。
