"""LLM messages 辅助函数。"""


def user_message_content(messages: list[dict[str, str]]) -> str:
    """取 messages 中最后一条 user 消息内容；无 user 时回退到最后一条。"""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message["content"]
    return messages[-1]["content"] if messages else ""
