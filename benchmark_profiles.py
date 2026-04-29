"""Simple benchmark runner for legacy/hybrid/external profiles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from runtime_env import APP_HOST_DEFAULT, APP_PORT_DEFAULT

ROOT = Path(__file__).resolve().parent


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_app(base_url: str, timeout_sec: float = 60.0) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            http_json("GET", f"{base_url}/api/info", timeout=2.0)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("app startup timeout")


def run_profile(profile: str, host: str, port: int, duration_sec: int) -> dict:
    base_url = f"http://{host}:{port}"
    proc = subprocess.Popen(
        [sys.executable, "run_stack.py", "--profile", profile, "--host", host, "--port", str(port)],
        cwd=str(ROOT),
    )
    try:
        wait_app(base_url)
        files = http_json("GET", f"{base_url}/api/files").get("files", [])
        if not files:
            raise RuntimeError("no videos found in data/")
        video_path = files[0]["path"]
        http_json("POST", f"{base_url}/api/open", {"path": video_path})
        http_json("POST", f"{base_url}/api/play", {})

        samples: list[dict] = []
        started = time.time()
        while time.time() - started < duration_sec:
            info = http_json("GET", f"{base_url}/api/info")
            metrics = http_json("GET", f"{base_url}/api/metrics")
            samples.append({"info": info, "metrics": metrics})
            time.sleep(1.0)

        if not samples:
            raise RuntimeError("no benchmark samples collected")
        return summarize_profile(profile, samples, video_path)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except Exception:
                proc.kill()


def summarize_profile(profile: str, samples: list[dict], video_path: str) -> dict:
    decode_fps = [float(s["info"]["stats"].get("decode_fps", 0.0)) for s in samples]
    render_fps = [float(s["info"]["stats"].get("render_fps", 0.0)) for s in samples]
    inf_ms = [float(s["info"]["stats"].get("inference_ms_avg", 0.0)) for s in samples]
    rss = [
        int(s["metrics"]["process"]["rss_bytes"])
        for s in samples
        if s["metrics"]["process"].get("rss_bytes") is not None
    ]
    scheduler_mode = samples[-1]["info"]["stats"].get("scheduler_mode", "local")
    return {
        "profile": profile,
        "video_path": video_path,
        "samples": len(samples),
        "decode_fps_avg": round(sum(decode_fps) / max(1, len(decode_fps)), 2),
        "render_fps_avg": round(sum(render_fps) / max(1, len(render_fps)), 2),
        "inference_ms_avg": round(sum(inf_ms) / max(1, len(inf_ms)), 2),
        "rss_mb_avg": round((sum(rss) / max(1, len(rss))) / (1024 * 1024), 2) if rss else None,
        "rss_mb_peak": round((max(rss) / (1024 * 1024)), 2) if rss else None,
        "scheduler_mode": scheduler_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runtime profiles")
    parser.add_argument("--profiles", default="legacy,hybrid,external")
    parser.add_argument("--duration-sec", default=20, type=int)
    parser.add_argument("--host", default=APP_HOST_DEFAULT)
    parser.add_argument("--port", default=APP_PORT_DEFAULT, type=int)
    parser.add_argument("--output", default="logs/benchmark_profiles.json")
    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    results = []
    for p in profiles:
        print(f"[benchmark] running profile={p}")
        try:
            res = run_profile(p, args.host, args.port, args.duration_sec)
        except (RuntimeError, urllib.error.URLError) as exc:
            res = {"profile": p, "error": str(exc)}
        results.append(res)

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    print(f"[benchmark] saved: {out_path}")


if __name__ == "__main__":
    main()
