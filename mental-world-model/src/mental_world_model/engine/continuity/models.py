"""连续事件状态模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class BiasFrame:
    """开放解释倾向（认知连续性 H 的一条）。"""

    id: str
    text: str
    mass: float = 0.0
    support: int = 0
    last_seen: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "mass": self.mass,
            "support": self.support,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BiasFrame":
        return cls(
            id=str(raw.get("id") or ""),
            text=str(raw.get("text") or "").strip(),
            mass=float(raw.get("mass") or 0.0),
            support=int(raw.get("support") or 0),
            last_seen=str(raw.get("last_seen") or ""),
        )


@dataclass
class MentalEpisode:
    """单事件心理快照（存独立 M_event，避免重复加权）。"""

    event_id: str
    theta: float
    timestamp: str
    scores: dict[str, float] = field(default_factory=dict)
    text_summary: str = ""
    # 相对上一事件的叙事间隔（天）；首事件为 0
    gap_from_previous_days: float = 0.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "theta": self.theta,
            "timestamp": self.timestamp,
            "scores": {k: float(v) for k, v in self.scores.items()},
            "text_summary": self.text_summary,
            "gap_from_previous_days": float(self.gap_from_previous_days),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "MentalEpisode":
        scores_raw = raw.get("scores") or {}
        scores = (
            {str(k): float(v) for k, v in scores_raw.items()}
            if isinstance(scores_raw, dict)
            else {}
        )
        try:
            gap = max(0.0, float(raw.get("gap_from_previous_days") or 0.0))
        except (TypeError, ValueError):
            gap = 0.0
        return cls(
            event_id=str(raw.get("event_id") or ""),
            theta=float(raw.get("theta") or 0.0),
            timestamp=str(raw.get("timestamp") or ""),
            scores=scores,
            text_summary=str(raw.get("text_summary") or ""),
            gap_from_previous_days=gap,
        )


@dataclass
class ContinuityState:
    """人物连续状态：H + 心理 episode 列表。"""

    person_id: str = "default"
    cognitive_frames: list[BiasFrame] = field(default_factory=list)
    mental_episodes: list[MentalEpisode] = field(default_factory=list)
    event_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "person_id": self.person_id,
            "cognitive_frames": [f.to_dict() for f in self.cognitive_frames],
            "mental_episodes": [e.to_dict() for e in self.mental_episodes],
            "event_count": self.event_count,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ContinuityState":
        if not isinstance(raw, dict):
            return cls()
        frames_raw = raw.get("cognitive_frames") or []
        eps_raw = raw.get("mental_episodes") or []
        frames = [
            BiasFrame.from_dict(item)
            for item in frames_raw
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        episodes = [
            MentalEpisode.from_dict(item)
            for item in eps_raw
            if isinstance(item, dict)
        ]
        return cls(
            person_id=str(raw.get("person_id") or "default"),
            cognitive_frames=frames,
            mental_episodes=episodes,
            event_count=int(raw.get("event_count") or 0),
            updated_at=str(raw.get("updated_at") or ""),
        )

    def top_cognitive_frames(self, k: int) -> list[BiasFrame]:
        ranked = sorted(self.cognitive_frames, key=lambda f: f.mass, reverse=True)
        return ranked[: max(0, k)]
