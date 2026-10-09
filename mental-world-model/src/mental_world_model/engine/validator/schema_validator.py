"""基于 definitions/schemas/ 的 JSON Schema 运行时校验（stdlib 子集实现）。"""

import json
from pathlib import Path
from typing import Any

_SCHEMA_FILES = {
    "xinsuo": "xinsuo.schema.json",
    "relation": "relation.schema.json",
    "suoyuan_xiang_si": "suoyuan_xiang_si.schema.json",
}


class SchemaValidator:
    """加载并缓存 JSON Schema，对原始 dict 做结构校验。"""

    def __init__(self, schemas_dir: str | Path) -> None:
        self._schemas_dir = Path(schemas_dir)
        self._schemas: dict[str, dict[str, Any]] = {}

    def _get_schema(self, schema_key: str) -> dict[str, Any]:
        if schema_key not in self._schemas:
            schema_path = self._schemas_dir / _SCHEMA_FILES[schema_key]
            with open(schema_path, encoding="utf-8") as file:
                self._schemas[schema_key] = json.load(file)
        return self._schemas[schema_key]

    def validate_instance(self, schema_key: str, instance: Any) -> list[str]:
        """校验单个实例，返回人类可读错误列表。"""
        schema = self._get_schema(schema_key)
        errors: list[str] = []
        self._validate(instance, schema, path="", errors=errors)
        return errors

    def validate_list(self, schema_key: str, items: list[Any]) -> list[str]:
        """校验列表中的每一项。"""
        errors: list[str] = []
        for index, item in enumerate(items):
            for message in self.validate_instance(schema_key, item):
                errors.append(f"[{index}] {message}")
        return errors

    def _validate(
        self,
        instance: Any,
        schema: dict[str, Any],
        *,
        path: str,
        errors: list[str],
    ) -> None:
        expected_type = schema.get("type")
        if expected_type == "object":
            self._validate_object(instance, schema, path=path, errors=errors)
        elif expected_type == "string":
            if not isinstance(instance, str):
                errors.append(f"{path or '根'}: 应为 string")
        elif expected_type == "boolean":
            if not isinstance(instance, bool):
                errors.append(f"{path or '根'}: 应为 boolean")
        elif expected_type is not None:
            errors.append(f"{path or '根'}: 不支持的 schema type {expected_type!r}")

    def _validate_object(
        self,
        instance: Any,
        schema: dict[str, Any],
        *,
        path: str,
        errors: list[str],
    ) -> None:
        if not isinstance(instance, dict):
            errors.append(f"{path or '根'}: 应为 object")
            return

        for field_name in schema.get("required", []):
            if field_name not in instance:
                errors.append(f"{path + field_name if path else field_name}: 缺少必填字段")

        properties = schema.get("properties", {})
        for key, value in instance.items():
            field_path = f"{path}{key}" if path else key
            if key not in properties:
                if schema.get("additionalProperties") is False:
                    errors.append(f"{field_path}: 不允许的额外字段")
                continue

            field_schema = properties[key]
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(
                    f"{field_path}: 值 {value!r} 不在允许范围 {field_schema['enum']}"
                )

            field_type = field_schema.get("type")
            if field_type == "string":
                if not isinstance(value, str):
                    errors.append(f"{field_path}: 应为 string")
                elif field_schema.get("minLength", 0) > 0 and not value.strip():
                    errors.append(f"{field_path}: 字符串不能为空")
                elif (
                    "minLength" in field_schema
                    and isinstance(value, str)
                    and len(value.strip()) < field_schema["minLength"]
                ):
                    errors.append(
                        f"{field_path}: 长度不足 {field_schema['minLength']}"
                    )
            elif field_type == "boolean" and not isinstance(value, bool):
                errors.append(f"{field_path}: 应为 boolean")
