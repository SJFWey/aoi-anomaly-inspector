from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aoi.artifacts import environment_snapshot, sha256_file, write_json_atomic, write_jsonl_atomic

write_json = write_json_atomic
write_jsonl = write_jsonl_atomic


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_records(records: list[dict[str, Any]], model: str, category: str, run_id: str) -> dict[str, Any]:
    latencies = sorted(float(record["latency_ms"]) for record in records)
    num_images = len(records)
    num_ng = sum(1 for record in records if record["decision"] == "NG")
    p95_index = int(0.95 * (len(latencies) - 1)) if latencies else 0
    return {
        "model": model,
        "category": category,
        "run_id": run_id,
        "num_images": num_images,
        "num_ng": num_ng,
        "ng_rate": float(num_ng / num_images) if num_images else 0.0,
        "avg_latency_ms": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_ms": float(latencies[p95_index]) if latencies else 0.0,
        "created_at": now_utc(),
    }


__all__ = [
    "environment_snapshot",
    "now_utc",
    "sha256_file",
    "summarize_records",
    "write_json",
    "write_jsonl",
]
