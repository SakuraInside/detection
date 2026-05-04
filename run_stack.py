"""Run migration stack in legacy/hybrid/external profile."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from runtime_env import APP_HOST_DEFAULT, APP_PORT_DEFAULT, VIDEO_BRIDGE_ADDR
from runtime_profiles import (
    INFERENCE_RPC_ADDR,
    RUNTIME_CONTROL_ADDR,
    RUNTIME_INGEST_ADDR,
    apply_profile,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def _split_host_port(addr: str) -> tuple[str, int]:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _wait_tcp_ready(host: str, port: int, timeout_sec: float = 45.0) -> None:
    deadline = time.time() + timeout_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except Exception as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"service {host}:{port} not ready: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Integra migration stack")
    parser.add_argument(
        "--profile",
        choices=["legacy", "hybrid", "external", "rust-video"],
        default="external",
    )
    parser.add_argument("--host", default=APP_HOST_DEFAULT)
    parser.add_argument("--port", default=APP_PORT_DEFAULT, type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--no-apply-profile", action="store_true")
    args = parser.parse_args()

    if args.profile in {"hybrid", "external", "rust-video"} and not shutil.which("cargo"):
        print(
            "[run_stack] error: команда `cargo` не найдена в PATH.\n"
            "  Профили hybrid/external собирают runtime-core; rust-video — video-bridge (декод в Rust).\n"
            "  Установите toolchain: https://rustup.rs/\n"
            "  Или: python run_stack.py --profile legacy  # только Python\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.no_apply_profile:
        apply_profile(CONFIG_PATH, args.profile)
        print(f"[run_stack] applied profile={args.profile} to {CONFIG_PATH}")

    procs: list[subprocess.Popen] = []
    try:
        if args.profile == "rust-video":
            procs.append(
                subprocess.Popen(
                    [
                        "cargo",
                        "run",
                        "--manifest-path",
                        str(ROOT / "video-bridge" / "Cargo.toml"),
                        "--",
                        "--listen",
                        VIDEO_BRIDGE_ADDR,
                    ],
                    cwd=str(ROOT),
                )
            )
            print(f"[run_stack] started video-bridge ({VIDEO_BRIDGE_ADDR})")
            vb_host, vb_port = _split_host_port(VIDEO_BRIDGE_ADDR)
            _wait_tcp_ready(vb_host, vb_port, timeout_sec=120.0)
            print("[run_stack] video-bridge TCP is ready")

        if args.profile in {"hybrid", "external"}:
            env = os.environ.copy()
            env["RUNTIME_INGEST_ADDR"] = RUNTIME_INGEST_ADDR
            env["RUNTIME_CONTROL_ADDR"] = RUNTIME_CONTROL_ADDR
            if args.profile == "external":
                env["INFERENCE_RPC_ADDR"] = INFERENCE_RPC_ADDR
            procs.append(
                subprocess.Popen(
                    ["cargo", "run"],
                    cwd=str(ROOT / "runtime-core"),
                    env=env,
                )
            )
            print("[run_stack] started runtime-core")
            ihost, iport = _split_host_port(RUNTIME_INGEST_ADDR)
            chost, cport = _split_host_port(RUNTIME_CONTROL_ADDR)
            _wait_tcp_ready(ihost, iport, timeout_sec=60.0)
            _wait_tcp_ready(chost, cport, timeout_sec=60.0)
            print("[run_stack] runtime-core ports are ready")

        if args.profile == "external":
            ihost, iport = _split_host_port(INFERENCE_RPC_ADDR)
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "run_inference_service.py",
                        "--host",
                        ihost,
                        "--port",
                        str(iport),
                    ],
                    cwd=str(ROOT),
                )
            )
            print("[run_stack] started inference service")
            # На первом запуске сервис может дольше стартовать из-за скачивания весов.
            _wait_tcp_ready(ihost, iport, timeout_sec=300.0)
            print("[run_stack] inference service port is ready")

        app_cmd = [
            sys.executable,
            "run.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.reload:
            app_cmd.append("--reload")
        app_proc = subprocess.Popen(app_cmd, cwd=str(ROOT))
        procs.append(app_proc)
        print(f"[run_stack] started app at http://{args.host}:{args.port}")

        app_rc = app_proc.wait()
        if app_rc != 0:
            raise RuntimeError(f"app exited with code {app_rc}")
    except KeyboardInterrupt:
        print("[run_stack] interrupted, shutting down...")
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in reversed(procs):
            if p.poll() is None:
                try:
                    p.wait(timeout=4)
                except Exception:
                    p.kill()


if __name__ == "__main__":
    main()
