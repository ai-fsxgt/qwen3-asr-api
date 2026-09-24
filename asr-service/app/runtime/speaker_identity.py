"""把离线声纹识别结果映射到转写分段。"""


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
