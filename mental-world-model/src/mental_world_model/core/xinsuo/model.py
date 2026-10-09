"""心所（Xinsuo）。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Xinsuo:
    """心所。

    字段结构由用户提供的原始定义文件决定。
    """

    name: str
    suoyuan: str
    shengqi_nian: str
    ye: str
    xiandai_xinli: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Xinsuo":
        """从字典构造心所对象。"""
        return cls(
            name=data["name"],
            suoyuan=data["suoyuan"],
            shengqi_nian=data["shengqi_nian"],
            ye=data["ye"],
            xiandai_xinli=data["xiandai_xinli"],
        )

    def to_dict(self) -> dict[str, str]:
        """转换为字典。"""
        return {
            "name": self.name,
            "suoyuan": self.suoyuan,
            "shengqi_nian": self.shengqi_nian,
            "ye": self.ye,
            "xiandai_xinli": self.xiandai_xinli,
        }
