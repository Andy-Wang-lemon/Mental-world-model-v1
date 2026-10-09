"""从原文提取所缘境，并判定显式想 / 隐式百法路径。"""

from mental_world_model.core.xinsuo.model import Xinsuo
from mental_world_model.engine.extraction.models import (
    IMPLICIT_SI_PLACEHOLDER,
    IMPLICIT_XIANG_PLACEHOLDER,
    ExtractedContext,
    ExtractedField,
    SuoyuanXiangSiDefinitions,
)
from mental_world_model.engine.extraction.prompts import (
    build_suoyuan_messages,
    build_xiang_evidence_messages,
    build_xiang_si_messages,
)
from mental_world_model.engine.extraction.router import SuoyuanXinsuoRouter
from mental_world_model.engine.extraction.validators import (
    normalize_suoyuan_response,
    normalize_xiang_si_field,
)
from mental_world_model.engine.seeds.config import is_seed_library_enabled
from mental_world_model.llm.client import LlmClient
from mental_world_model.llm.usage import llm_call_context

_SUOYUAN_SCHEMA = {
    "type": "object",
    "properties": {
        "suoyuan": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
    },
    "required": ["suoyuan"],
}

_XIANG_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_explicit_xiang": {"type": "boolean"},
        "xiang": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["text"],
        },
        "has_explicit_si": {"type": "boolean"},
        "si": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["text"],
        },
    },
    "required": ["has_explicit_xiang", "xiang"],
}

_XIANG_SI_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["text", "source", "confidence"],
}

_XIANG_SI_SCHEMA = {
    "type": "object",
    "properties": {
        "xiang": _XIANG_SI_FIELD_SCHEMA,
        "si": _XIANG_SI_FIELD_SCHEMA,
    },
    "required": ["xiang", "si"],
}


class ContextExtractor:
    """所缘境 → 判定显式想 → 显式摘想/思 或 隐式占位（不推理想/思）。"""

    def __init__(
        self,
        llm_client: LlmClient,
        definitions: SuoyuanXiangSiDefinitions,
        xinsuo_list: list[Xinsuo],
        seed_weights: dict[str, float] | None = None,
    ) -> None:
        self._llm = llm_client
        self._definitions = definitions
        self._xinsuo_list = xinsuo_list
        self._xinsuo_by_name = {item.name: item for item in xinsuo_list}
        self._seed_weights = seed_weights or {}
        self._router = SuoyuanXinsuoRouter(llm_client)

    def extract(self, yuanwen: str) -> ExtractedContext:
        """提取所缘，并决定 explicit / implicit 评分路径。"""
        suoyuan = self._extract_suoyuan(yuanwen)

        if is_seed_library_enabled():
            hint_names, relevance = self._router.route(
                yuanwen,
                suoyuan.text,
                self._xinsuo_list,
                self._seed_weights,
            )
        else:
            hint_names = []
            relevance = {name: 0.0 for name in self._xinsuo_by_name}

        path, xiang, si = self._resolve_xiang_si_path(yuanwen, suoyuan.text)

        return ExtractedContext(
            yuanwen=yuanwen,
            suoyuan=suoyuan,
            xiang=xiang,
            si=si,
            scoring_path=path,
            routed_hint_names=hint_names,
            router_relevance=relevance,
        )

    def _extract_suoyuan(self, yuanwen: str):
        messages = build_suoyuan_messages(yuanwen, self._definitions)
        with llm_call_context(stage="extraction.suoyuan"):
            response = self._llm.complete(messages, schema=_SUOYUAN_SCHEMA)
        if "suoyuan" not in response:
            raise ValueError("LLM 提取响应缺少 suoyuan")
        return normalize_suoyuan_response(response["suoyuan"])

    def _resolve_xiang_si_path(
        self,
        yuanwen: str,
        suoyuan_text: str,
    ) -> tuple[str, ExtractedField, ExtractedField]:
        messages = build_xiang_evidence_messages(
            yuanwen, suoyuan_text, self._definitions
        )
        with llm_call_context(stage="extraction.xiang_evidence"):
            response = self._llm.complete(messages, schema=_XIANG_EVIDENCE_SCHEMA)

        has_xiang = bool(response.get("has_explicit_xiang"))
        has_si = bool(response.get("has_explicit_si"))
        raw_xiang = response.get("xiang") or {"text": ""}
        raw_si = response.get("si") or {"text": ""}

        if not has_xiang:
            return (
                "implicit",
                ExtractedField(
                    text=IMPLICIT_XIANG_PLACEHOLDER, source="原文", confidence=1.0
                ),
                ExtractedField(
                    text=IMPLICIT_SI_PLACEHOLDER, source="原文", confidence=1.0
                ),
            )

        # 显式：强制按原文护栏；若无法落地则改走隐式（禁止推理编造）
        xiang = normalize_xiang_si_field(
            {
                "text": raw_xiang.get("text", ""),
                "source": "原文",
                "confidence": 1.0,
            },
            yuanwen=yuanwen,
        )
        if xiang.source != "原文" or not xiang.text.strip():
            return (
                "implicit",
                ExtractedField(
                    text=IMPLICIT_XIANG_PLACEHOLDER, source="原文", confidence=1.0
                ),
                ExtractedField(
                    text=IMPLICIT_SI_PLACEHOLDER, source="原文", confidence=1.0
                ),
            )

        if has_si and str(raw_si.get("text", "")).strip():
            si = normalize_xiang_si_field(
                {
                    "text": raw_si.get("text", ""),
                    "source": "原文",
                    "confidence": 1.0,
                },
                yuanwen=yuanwen,
            )
            if si.source != "原文":
                si = ExtractedField(text="", source="原文", confidence=1.0)
        else:
            si = ExtractedField(text="", source="原文", confidence=1.0)

        return "explicit", xiang, si

    def _extract_xiang_si_legacy(
        self,
        yuanwen: str,
        suoyuan_text: str,
        hint_xinsuo: list[Xinsuo],
    ):
        """保留旧接口供测试；新主路径不再推理想/思。"""
        messages = build_xiang_si_messages(
            yuanwen,
            suoyuan_text,
            self._definitions,
            hint_xinsuo,
            self._seed_weights,
        )
        with llm_call_context(stage="extraction.xiang_si"):
            response = self._llm.complete(messages, schema=_XIANG_SI_SCHEMA)
        for field_name in ("xiang", "si"):
            if field_name not in response:
                raise ValueError(f"LLM 提取响应缺少 {field_name}")

        xiang = normalize_xiang_si_field(response["xiang"], yuanwen=yuanwen)
        si = normalize_xiang_si_field(response["si"], yuanwen=yuanwen)
        return xiang, si
