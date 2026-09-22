"""离线身份结果映射与实时说话人身份确认状态。

聚类标签只在单个会话内有效，本模块只管理“连续判决后再展示姓名”的会话态，
不缓存数据库内容。每次达到质量门槛的语音段仍直接查询共享 pgvector 数据库。
"""
from dataclasses import dataclass


def apply_offline_identities(
    segments: list[dict],
    mappings: list[dict],
) -> list[dict]:
    """Apply cluster matches and time-local identity anchors to segments."""
    mapping_by_label = {mapping["label"]: mapping for mapping in mappings}
    for segment in segments:
        cluster_mapping = mapping_by_label.get(segment.get("speaker"))
        if cluster_mapping and cluster_mapping.get("name"):
            segment["speaker_name"] = cluster_mapping["name"]

        votes = {}
        start = float(segment["start"])
        end = float(segment["end"])
        anchors = (
            cluster_mapping.get("_identity_anchors", [])
            if cluster_mapping is not None
            else []
        )
        for anchor in anchors:
            overlap = sum(
                max(0.0, min(end, interval_end) - max(start, interval_start))
                for interval_start, interval_end in anchor["intervals"]
            )
            if overlap <= 0:
                continue
            speaker_id = anchor["speaker_id"]
            vote = votes.setdefault(speaker_id, {
                "duration": 0.0,
                "score": float("-inf"),
                "name": anchor["name"],
            })
            vote["duration"] += overlap
            if anchor["score"] is not None:
                vote["score"] = max(vote["score"], anchor["score"])
        if votes:
            winner = max(
                votes.values(),
                key=lambda vote: (vote["duration"], vote["score"]),
            )
            segment["speaker_name"] = winner["name"]

    return [
        {
            key: value
            for key, value in mapping.items()
            if key != "_identity_anchors"
        }
        for mapping in mappings
    ]


@dataclass
class _IdentityState:
    observed_id: str | None = None
    observed_name: str | None = None
    consecutive: int = 0
    confirmed_id: str | None = None
    confirmed_name: str | None = None


class RealtimeIdentityTracker:
    """按聚类标签累计连续识别结果，并抑制单次误命中。"""

    def __init__(self, required_confirmations: int = 2):
        if required_confirmations < 1:
            raise ValueError("实时身份确认次数必须大于等于 1")
        self._required = required_confirmations
        self._states: dict[str, _IdentityState] = {}

    def observe(self, label: str, match: dict | None) -> str | None:
        state = self._states.setdefault(label, _IdentityState())
        speaker_id = match.get("speaker_id") if match else None
        speaker_name = match.get("name") if match else None

        if speaker_id == state.observed_id:
            state.consecutive += 1
            state.observed_name = speaker_name
        else:
            state.observed_id = speaker_id
            state.observed_name = speaker_name
            state.consecutive = 1

        if state.consecutive >= self._required:
            state.confirmed_id = state.observed_id
            state.confirmed_name = state.observed_name
        elif speaker_id == state.confirmed_id and speaker_id is not None:
            # 同一身份改名后无需再次隐藏；数据库仍是每段直查。
            state.confirmed_name = speaker_name
        return state.confirmed_name

    def confirmed_name(self, label: str) -> str | None:
        """读取已确认姓名，不改变连续识别计数。"""
        state = self._states.get(label)
        return state.confirmed_name if state is not None else None

    def clear(self) -> None:
        self._states.clear()
