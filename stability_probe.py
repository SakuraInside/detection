from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=4) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe backend stability and RAM SLA")
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--stream-id", default="main")
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--period-sec", type=float, default=2.0)
    args = parser.parse_args()

    samples = []
    deadline = time.time() + max(1, args.minutes) * 60
    metrics_url = f"{args.backend}/api/metrics?stream_id={args.stream_id}"
    while time.time() < deadline:
        data = fetch_json(metrics_url)
        process = data.get("process") or {}
        pipeline = data.get("pipeline") or {}
        stats = pipeline.get("stats") or {}
        samples.append(
            {
                "t": time.time(),
                "rss": int(process.get("rss_analytics_sum_bytes") or 0),
                "inf_ms": float(stats.get("inference_ms_avg") or 0.0),
                "decode_fps": float(stats.get("decode_fps") or 0.0),
                "render_fps": float(stats.get("render_fps") or 0.0),
            }
        )
        time.sleep(max(0.5, args.period_sec))

    rss_values = [s["rss"] for s in samples]
    inf_values = [s["inf_ms"] for s in samples]
    out = {
        "samples": len(samples),
        "rss_max_bytes": max(rss_values) if rss_values else 0,
        "rss_mean_bytes": int(statistics.fmean(rss_values)) if rss_values else 0,
        "rss_sla_ok": (max(rss_values) < 1_000_000_000) if rss_values else False,
        "inference_ms_p95": sorted(inf_values)[int(len(inf_values) * 0.95)] if inf_values else 0.0,
        "decode_fps_min": min((s["decode_fps"] for s in samples), default=0.0),
        "render_fps_min": min((s["render_fps"] for s in samples), default=0.0),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

