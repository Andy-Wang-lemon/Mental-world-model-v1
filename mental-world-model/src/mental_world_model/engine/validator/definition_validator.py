"""定义校验器。"""

from pathlib import Path
from typing import Any

from mental_world_model.core.relations.relation import Relation
from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.models import SuoyuanXiangSiDefinitions
from mental_world_model.engine.validator.schema_validator import SchemaValidator
from mental_world_model.engine.validator.validation_result import (
    ScoreConflict,
    ValidationResult,
)

_XINSUO_REQUIRED_FIELDS = (
    "name",
    "suoyuan",
    "shengqi_nian",
    "ye",
    "xiandai_xinli",
)

_RELATION_REQUIRED_FIELDS = (
    "source",
    "target",
    "description",
    "relation_type",
    "is_direct",
)

_SUOYUAN_XIANG_SI_REQUIRED_FIELDS = (
    "suoyuan_jing",
    "xiang",
    "si",
    "summary",
)

_RELATION_TYPES = frozenset({"对治"})

MUTEX_RELATION_TYPES = frozenset({"对治"})

_RELATION_TYPE_ALIASES = frozenset({"障碍"})


class DefinitionValidator:
    """校验定义合法性（JSON Schema 结构 + 语义规则）。"""

    def __init__(self, schemas_dir: str | Path | None = None) -> None:
        self._schema = SchemaValidator(schemas_dir) if schemas_dir else None

    def validate(self, definition: Any) -> ValidationResult:
        """校验定义对象。"""
        if isinstance(definition, Xinsuo):
            return self.validate_xinsuo(definition)
        if isinstance(definition, Relation):
            return self.validate_relation(definition)
        if isinstance(definition, SuoyuanXiangSiDefinitions):
            return self.validate_suoyuan_xiang_si(definition)
        if isinstance(definition, list):
            return self._validate_list(definition)
        if isinstance(definition, dict):
            if "suoyuan_jing" in definition:
                return self.validate_suoyuan_xiang_si(definition)
            if "name" in definition:
                return self.validate_xinsuo(definition)
            if "source" in definition:
                return self.validate_relation(definition)
        return ValidationResult(valid=False, errors=["无法识别的定义类型"])

    def validate_xinsuo(
        self,
        definition: Xinsuo | dict[str, Any],
        known_names: set[str] | None = None,
    ) -> ValidationResult:
        """校验心所（Xinsuo）定义。"""
        item = definition.to_dict() if isinstance(definition, Xinsuo) else definition

        errors: list[str] = []
        if self._schema is not None:
            errors.extend(self._schema.validate_instance("xinsuo", item))

        for field_name in _XINSUO_REQUIRED_FIELDS:
            if field_name not in item:
                errors.append(f"缺少字段：{field_name}")
            elif not isinstance(item[field_name], str) or not item[field_name].strip():
                errors.append(f"字段为空：{field_name}")

        if known_names is not None and "name" in item and item["name"] in known_names:
            errors.append(f"重复的心所名称：{item['name']}")

        return ValidationResult(valid=not errors, errors=errors)

    def validate_xinsuo_list(
        self,
        definitions: list[Xinsuo | dict[str, Any]],
    ) -> ValidationResult:
        """校验心所定义列表（含名称唯一性）。"""
        errors: list[str] = []
        seen_names: set[str] = set()

        for index, definition in enumerate(definitions):
            result = self.validate_xinsuo(definition, known_names=seen_names)
            if not result.valid:
                errors.extend(f"[{index}] {error}" for error in result.errors)
            name = (
                definition.name
                if isinstance(definition, Xinsuo)
                else definition.get("name")
            )
            if name:
                seen_names.add(name)

        return ValidationResult(valid=not errors, errors=errors)

    def validate_relation(
        self,
        definition: Relation | dict[str, Any],
        xinsuo_names: set[str] | None = None,
    ) -> ValidationResult:
        """校验关系（Relation）定义。"""
        item = definition.to_dict() if isinstance(definition, Relation) else definition

        errors: list[str] = []
        if self._schema is not None:
            schema_item = dict(item)
            if schema_item.get("relation_type") in _RELATION_TYPE_ALIASES:
                schema_item = {**schema_item, "relation_type": "对治"}
            errors.extend(self._schema.validate_instance("relation", schema_item))

        for field_name in _RELATION_REQUIRED_FIELDS:
            if field_name not in item:
                errors.append(f"缺少字段：{field_name}")

        if "relation_type" in item:
            normalized_type = item["relation_type"]
            if normalized_type in _RELATION_TYPE_ALIASES:
                normalized_type = "对治"
            if normalized_type not in _RELATION_TYPES:
                errors.append(f"无效的关系类型：{item['relation_type']}")

        if xinsuo_names is not None:
            for endpoint in ("source", "target"):
                if endpoint in item and item[endpoint] not in xinsuo_names:
                    errors.append(f"{endpoint} 不在心所列表中：{item[endpoint]}")

        return ValidationResult(valid=not errors, errors=errors)

    def validate_relation_list(
        self,
        definitions: list[Relation | dict[str, Any]],
        xinsuo_names: set[str] | None = None,
    ) -> ValidationResult:
        """校验关系定义列表。"""
        errors: list[str] = []
        for index, definition in enumerate(definitions):
            result = self.validate_relation(definition, xinsuo_names=xinsuo_names)
            if not result.valid:
                errors.extend(f"[{index}] {error}" for error in result.errors)

        if xinsuo_names is not None:
            sources: list[str] = []
            for definition in definitions:
                if isinstance(definition, Relation):
                    sources.append(definition.source)
                else:
                    sources.append(str(definition.get("source", "")))
            missing = sorted(xinsuo_names - set(sources))
            duplicate = sorted(
                name for name in set(sources) if sources.count(name) > 1
            )
            if missing:
                errors.append(f"缺少以心所为 source 的对治关系：{', '.join(missing)}")
            if duplicate:
                errors.append(f"重复 source 的对治关系：{', '.join(duplicate)}")
            if len(definitions) != len(xinsuo_names):
                errors.append(
                    f"对治关系条数应为 {len(xinsuo_names)}，当前为 {len(definitions)}"
                )

        return ValidationResult(valid=not errors, errors=errors)

    def validate_suoyuan_xiang_si(
        self,
        definition: SuoyuanXiangSiDefinitions | dict[str, Any],
    ) -> ValidationResult:
        """校验所缘/想/思定义。"""
        item = (
            definition.to_dict()
            if isinstance(definition, SuoyuanXiangSiDefinitions)
            else definition
        )

        errors: list[str] = []
        if self._schema is not None:
            errors.extend(self._schema.validate_instance("suoyuan_xiang_si", item))

        for field_name in _SUOYUAN_XIANG_SI_REQUIRED_FIELDS:
            if field_name not in item:
                errors.append(f"缺少字段：{field_name}")
            elif not isinstance(item[field_name], str) or not item[field_name].strip():
                errors.append(f"字段为空：{field_name}")

        return ValidationResult(valid=not errors, errors=errors)

    def check_score_conflicts(
        self,
        scores: dict[str, float],
        relations: list[Relation],
        *,
        threshold: float = 0,
    ) -> list[ScoreConflict]:
        """检查打分结果中的互斥冲突。"""
        active = {
            name: score
            for name, score in scores.items()
            if score > threshold
        }
        conflicts: list[ScoreConflict] = []
        seen_pairs: set[tuple[str, str]] = set()

        for relation in relations:
            if relation.relation_type not in MUTEX_RELATION_TYPES:
                continue
            source = relation.source
            target = relation.target
            if source not in active or target not in active:
                continue

            pair = tuple(sorted((source, target)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            conflicts.append(
                ScoreConflict(
                    name_a=source,
                    name_b=target,
                    score_a=active[source],
                    score_b=active[target],
                    description=relation.description,
                )
            )

        return conflicts

    def _validate_list(self, definitions: list[Any]) -> ValidationResult:
        if not definitions:
            return ValidationResult(valid=True)

        first = definitions[0]
        if isinstance(first, Xinsuo) or (
            isinstance(first, dict) and "name" in first
        ):
            return self.validate_xinsuo_list(definitions)
        if isinstance(first, Relation) or (
            isinstance(first, dict) and "source" in first
        ):
            return self.validate_relation_list(definitions)
        return ValidationResult(valid=False, errors=["无法识别的定义列表类型"])
