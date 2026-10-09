"""定义规范化器。"""

from typing import Any

from mental_world_model.utils.xinsuo_names import (
    normalize_relation_type,
    normalize_xinsuo_name,
)


class DefinitionNormalizer:
    """规范化定义文本与结构。"""

    def normalize(self, definition: Any) -> Any:
        """规范化定义对象或列表。"""
        if isinstance(definition, list):
            if not definition:
                return definition
            first = definition[0]
            if isinstance(first, dict) and "name" in first:
                return [self.normalize_xinsuo_dict(item) for item in definition]
            if isinstance(first, dict) and "source" in first:
                return [self.normalize_relation_dict(item) for item in definition]
            return definition

        if isinstance(definition, dict):
            if "suoyuan_jing" in definition:
                return self.normalize_suoyuan_xiang_si_dict(definition)
            if "name" in definition:
                return self.normalize_xinsuo_dict(definition)
            if "source" in definition:
                return self.normalize_relation_dict(definition)

        return definition

    def normalize_name(self, value: str) -> str:
        """规范化心所名称。"""
        return normalize_xinsuo_name(self.normalize_text(value))

    def normalize_text(self, value: str) -> str:
        """规范化文本：统一换行、去掉行尾空白、去掉首尾空白。"""
        if not isinstance(value, str):
            return value
        text = value.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def normalize_xinsuo_dict(self, item: dict[str, Any]) -> dict[str, Any]:
        """规范化单条心所 dict（仅处理已存在的字段）。"""
        result = dict(item)
        if "name" in item:
            result["name"] = self.normalize_name(str(item["name"]))
        for field in ("suoyuan", "shengqi_nian", "ye", "xiandai_xinli"):
            if field in item:
                result[field] = self.normalize_text(str(item[field]))
        return result

    def normalize_relation_dict(self, item: dict[str, Any]) -> dict[str, Any]:
        """规范化单条关系 dict（仅处理已存在的字段）。"""
        result = dict(item)
        if "source" in item:
            result["source"] = self.normalize_name(str(item["source"]))
        if "target" in item:
            result["target"] = self.normalize_name(str(item["target"]))
        if "relation_type" in item:
            result["relation_type"] = normalize_relation_type(
                self.normalize_text(str(item["relation_type"]))
            )
        if "description" in item:
            result["description"] = self.normalize_text(str(item["description"]))
        if "is_direct" in item:
            result["is_direct"] = bool(item["is_direct"])
        return result

    def normalize_suoyuan_xiang_si_dict(self, item: dict[str, Any]) -> dict[str, str]:
        """规范化所缘/想/思定义 dict（仅处理已存在的字段）。"""
        result = dict(item)
        for field in ("suoyuan_jing", "xiang", "si", "summary"):
            if field in item:
                result[field] = self.normalize_text(str(item[field]))
        return result
