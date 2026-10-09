# LLM 适配说明

心智世界模型与具体大语言模型**解耦**：评分、提取、冲突消解均只调用 `LlmClient.complete()`。

## 架构

```
MentalWorldModel
    └── LlmClient (Protocol)
            ├── MockLlmClient          测试 / 开发
            ├── HttpChatCompletionsClient   内置 HTTP 通用适配
            └── 你的自定义适配器        SDK、gRPC、私有协议等
```

## 内置 HTTP 适配

`HttpChatCompletionsClient` 不预设厂商地址或模型名。你提供：

| 配置 | 环境变量 | 说明 |
|------|----------|------|
| API 根地址 | `MWM_LLM_BASE_URL` | 必填 |
| 模型标识 | `MWM_LLM_MODEL` | 必填 |
| 鉴权密钥 | `MWM_LLM_API_KEY` | 可选 |
| 对话路径 | `MWM_LLM_CHAT_PATH` | 默认 `/chat/completions` |
| 鉴权头 | `MWM_LLM_AUTH_HEADER` | 默认 `Authorization` |
| 鉴权前缀 | `MWM_LLM_AUTH_PREFIX` | 默认 `Bearer ` |
| 额外头 | `MWM_LLM_EXTRA_HEADERS` | JSON 对象 |
| JSON 响应格式 | `MWM_LLM_JSON_RESPONSE` | 默认 `true` |

请求：`POST {BASE_URL}{CHAT_PATH}`，body 含 `model`、`messages`、`temperature`；
可选 `response_format: {type: json_object}`。

响应：解析 `choices[0].message.content` 为 JSON dict。

## 自定义适配器

实现 `complete(messages, *, schema=None) -> dict` 即可：

- `messages`：`[{"role": "user"|"assistant"|"system", "content": "..."}]`
- `schema`：期望返回结构的 JSON Schema；可在 prompt 中自行约束输出
- 返回值：必须是 `dict`（如 `{"score": 0.8}`、`{"winner": "信"}`）

注入：

```python
create_llm_client(client=your_adapter)
```

## 向后兼容

`OpenAICompatibleClient` 为 `HttpChatCompletionsClient` 的别名，旧代码可继续使用。
