"""定义加载器。"""

import json
import re
from pathlib import Path
from typing import Any, Callable

from mental_world_model.engine.loader.docx_reader import parse_xinsuo_docx
from mental_world_model.engine.loader.duizhi_reader import parse_duizhi_docx
from mental_world_model.engine.loader.seed_reader import parse_seeds_docx
from mental_world_model.engine.loader.suoyuan_xiang_si_reader import parse_suoyuan_xiang_si_docx

_SUPPORTED_RAW_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".docx"}


class AmbiguousDocxError(FileNotFoundError):
    """raw 目录下匹配到多个源文件，无法唯一确定。"""


class DefinitionLoader:
    """从文件系统加载原始定义。"""

    def load_raw_file(self, path: str | Path) -> Any:
        """按扩展名加载单个原始定义文件。"""
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            return self.load_json(file_path)
        if suffix in (".yaml", ".yml"):
            return self.load_yaml(file_path)
        if suffix == ".md":
            return self.load_markdown(file_path)
        if suffix == ".docx":
            return self._load_docx_by_name(file_path)

        raise ValueError(f"不支持的文件格式：{file_path}")

    def load_raw_directory(self, path: str | Path) -> dict[str, Any]:
        """加载目录下所有支持的原始定义文件，键为文件名（不含扩展名）。"""
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"不是目录：{directory}")

        results: dict[str, Any] = {}
        for file_path in sorted(directory.iterdir()):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in _SUPPORTED_RAW_SUFFIXES:
                continue
            results[file_path.stem] = self.load_raw_file(file_path)
        return results

    def load_json(self, path: str | Path) -> Any:
        """加载 JSON 格式定义文件。"""
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def load_markdown(self, path: str | Path) -> Any:
        """加载 Markdown 定义文件（提取首个 ```json 代码块）。"""
        content = Path(path).read_text(encoding="utf-8")
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if not match:
            raise ValueError(f"Markdown 文件未包含 JSON 代码块：{path}")
        return json.loads(match.group(1))

    def load_yaml(self, path: str | Path) -> Any:
        """加载 YAML 格式定义文件（需要 ``pip install PyYAML``）。"""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "加载 YAML 需要 PyYAML：pip install PyYAML"
            ) from exc

        with open(path, encoding="utf-8") as file:
            return yaml.safe_load(file)

    def load_docx(self, path: str | Path, output_path: str | Path | None = None) -> list[dict]:
        """加载 Word 格式心所定义文件。"""
        data = parse_xinsuo_docx(path)
        if output_path is not None:
            self.export_normalized_json(data, output_path)
        return data

    def load_duizhi_docx(self, path: str | Path, output_path: str | Path | None = None) -> list[dict]:
        """加载 Word 格式心所对治关系文件。"""
        data = parse_duizhi_docx(path)
        if output_path is not None:
            self.export_normalized_json(data, output_path)
        return data

    def load_suoyuan_xiang_si_docx(
        self,
        path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, str]:
        """加载 Word 格式所缘境/想/思定义文件。"""
        data = parse_suoyuan_xiang_si_docx(path)
        if output_path is not None:
            self.export_normalized_json(data, output_path)
        return data

    def load_seeds_docx(self, path: str | Path, output_path: str | Path | None = None) -> dict:
        """加载 Word 格式心所种子库文件。"""
        data = parse_seeds_docx(path)
        if output_path is not None:
            self.export_normalized_json(data, output_path)
        return data

    def load_xinsuo_definitions(self, definitions_dir: str | Path) -> list[dict]:
        """加载心所定义：raw 更新时同步到 normalized JSON。"""
        return self._load_with_raw_sync(
            definitions_dir=definitions_dir,
            normalized_path=Path(definitions_dir) / "normalized" / "xinsuo.json",
            raw_globs=["*心所定义*.docx", "*心所定义*.md", "*心所定义*.json"],
            parse_raw=self._parse_xinsuo_raw,
            label="心所定义",
        )

    def load_duizhi_definitions(self, definitions_dir: str | Path) -> list[dict]:
        """加载对治关系定义：raw 更新时同步到 normalized JSON。"""
        return self._load_with_raw_sync(
            definitions_dir=definitions_dir,
            normalized_path=Path(definitions_dir) / "normalized" / "xinsuo_duizhi.json",
            raw_globs=["*对治*.docx", "*对治*.md", "*对治*.json"],
            parse_raw=self._parse_duizhi_raw,
            label="心所对治关系",
        )

    def load_suoyuan_xiang_si_definitions(self, definitions_dir: str | Path) -> dict[str, str]:
        """加载所缘/想/思定义：raw 更新时同步到 normalized JSON。"""
        return self._load_with_raw_sync(
            definitions_dir=definitions_dir,
            normalized_path=Path(definitions_dir) / "normalized" / "suoyuan_xiang_si.json",
            raw_globs=[
                "*所缘*想*思*.docx",
                "*所缘*想*思*.md",
                "*所缘*想*思*.json",
            ],
            parse_raw=self._parse_suoyuan_xiang_si_raw,
            label="所缘/想/思定义",
        )

    def load_xinsuo_seed_definitions(self, definitions_dir: str | Path) -> dict:
        """加载心所种子库：raw 更新时同步到 seeds/xinsuo_seeds.json。"""
        return self._load_with_raw_sync(
            definitions_dir=definitions_dir,
            normalized_path=Path(definitions_dir) / "seeds" / "xinsuo_seeds.json",
            raw_globs=["*种子*.docx", "*种子库*.docx", "*种子*.md", "*种子库*.md", "*种子*.json"],
            parse_raw=self._parse_seeds_raw,
            label="心所种子库",
        )

    def export_normalized_json(self, data: Any, path: str | Path) -> None:
        """将结构化数据导出为 normalized JSON 文件。"""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _parse_xinsuo_raw(self, raw_path: Path) -> list[dict]:
        suffix = raw_path.suffix.lower()
        if suffix == ".docx":
            return parse_xinsuo_docx(raw_path)
        if suffix == ".md":
            data = self.load_markdown(raw_path)
            if not isinstance(data, list):
                raise ValueError(f"心所定义 Markdown 应包含 JSON 数组：{raw_path}")
            return data
        return self.load_json(raw_path)

    def _parse_duizhi_raw(self, raw_path: Path) -> list[dict]:
        suffix = raw_path.suffix.lower()
        if suffix == ".docx":
            return parse_duizhi_docx(raw_path)
        if suffix == ".md":
            data = self.load_markdown(raw_path)
            if not isinstance(data, list):
                raise ValueError(f"对治关系 Markdown 应包含 JSON 数组：{raw_path}")
            return data
        return self.load_json(raw_path)

    def _parse_suoyuan_xiang_si_raw(self, raw_path: Path) -> dict[str, str]:
        suffix = raw_path.suffix.lower()
        if suffix == ".docx":
            return parse_suoyuan_xiang_si_docx(raw_path)
        if suffix == ".md":
            data = self.load_markdown(raw_path)
            if not isinstance(data, dict):
                raise ValueError(f"所缘/想/思 Markdown 应包含 JSON 对象：{raw_path}")
            return data
        return self.load_json(raw_path)

    def _parse_seeds_raw(self, raw_path: Path) -> dict:
        suffix = raw_path.suffix.lower()
        if suffix == ".docx":
            return parse_seeds_docx(raw_path)
        if suffix == ".md":
            data = self.load_markdown(raw_path)
            if not isinstance(data, dict):
                raise ValueError(f"种子库 Markdown 应包含 JSON 对象：{raw_path}")
            return data
        return self.load_json(raw_path)

    def _load_docx_by_name(self, file_path: Path) -> Any:
        """根据 docx 文件名推断类型并加载。"""
        name = file_path.name
        if "种子" in name:
            return self.load_seeds_docx(file_path)
        if "对治" in name:
            return self.load_duizhi_docx(file_path)
        if "所缘" in name or ("想" in name and "思" in name):
            return self.load_suoyuan_xiang_si_docx(file_path)
        return self.load_docx(file_path)

    def _find_raw_candidate(
        self,
        raw_dir: Path,
        globs: list[str],
        *,
        label: str,
    ) -> Path | None:
        candidates: list[Path] = []
        seen: set[Path] = set()
        for pattern in globs:
            for path in sorted(raw_dir.glob(pattern)):
                if path.name.startswith("~$"):
                    continue
                resolved = path.resolve()
                if path.is_file() and resolved not in seen:
                    seen.add(resolved)
                    candidates.append(path)

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        names = ", ".join(path.name for path in candidates)
        raise AmbiguousDocxError(
            f"{label} 在 raw 目录匹配到多个源文件，无法唯一确定：{names}。"
            "请只保留一个匹配文件。"
        )

    @staticmethod
    def _raw_is_newer_than_normalized(raw_path: Path, normalized_path: Path) -> bool:
        if not normalized_path.exists():
            return True
        return raw_path.stat().st_mtime > normalized_path.stat().st_mtime

    def _load_with_raw_sync(
        self,
        *,
        definitions_dir: str | Path,
        normalized_path: Path,
        raw_globs: list[str],
        parse_raw: Callable[[Path], Any],
        label: str,
    ) -> Any:
        """raw 源文件较新时解析并写入 normalized；否则读取 normalized JSON。"""
        root = Path(definitions_dir)
        raw_dir = root / "raw"
        raw_path = self._find_raw_candidate(raw_dir, raw_globs, label=label)

        if raw_path is not None and self._raw_is_newer_than_normalized(raw_path, normalized_path):
            data = parse_raw(raw_path)
            self.export_normalized_json(data, normalized_path)
            return data

        if normalized_path.exists():
            return self.load_json(normalized_path)

        if raw_path is not None:
            data = parse_raw(raw_path)
            self.export_normalized_json(data, normalized_path)
            return data

        raise FileNotFoundError(
            f"未找到{label}：缺少 {normalized_path}，且 raw 目录下无匹配的源文件"
        )
