"""Entrypoint Integra Native.

Запускает только нативные процессы (Python в рантайме = 0):
  1. video-bridge   — Rust crate, декод видео (OpenCV), TCP :9876
  2. backend_gateway — Rust crate, HTTP/WS UI + FFI инференс через integra_ffi

Python используется только как launcher (этот файл) — никаких импортов
app.* / app.pipeline.* / app.analyzer.* больше не существует в рантайме.

Для запуска требуется:
  • cargo (Rust toolchain)
  • собранный integra_ffi.dll / libintegra_ffi.so (см. native/README.md)
  • frontend в static/ (index.html, app.js, styles.css)
  • Windows + video-bridge: нужны clang.exe и libclang.dll (полный LLVM или VS «Clang tools»);
    pip libclang недостаточен. См. INTEGRA_LLVM_BIN в run.py / сообщения при ошибке.

Использование:
    python run.py                       # debug-сборка (быстрее cargo, дольше start)
    python run.py --release             # release-сборка (рекомендуется)
    python run.py --release --no-bridge # gateway без video-bridge (smoke-тест)
    python run.py --release --bridge-build-only  # только cargo build video-bridge (OPENCV_/LLVM как у полного run)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _find_integra_ffi(root: Path) -> Path | None:
    """Найти integra_ffi.{dll,so,dylib}, чтобы выставить INTEGRA_FFI_PATH."""
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform == "darwin"
    lib_name = (
        "integra_ffi.dll"
        if is_windows
        else "libintegra_ffi.dylib"
        if is_macos
        else "libintegra_ffi.so"
    )
    candidates = [
        root / "native" / "build-msvc" / "RelWithDebInfo" / lib_name,
        root / "native" / "build-msvc" / "Release" / lib_name,
        root / "native" / "build" / "RelWithDebInfo" / lib_name,
        root / "native" / "build" / "Release" / lib_name,
        root / "native" / "build" / lib_name,
        root / lib_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _path_segments_need_msys_strip(segment: str) -> bool:
    s = segment.lower().replace("/", "\\")
    markers = (
        "\\msys64\\",
        "\\msys32\\",
        "mingw64",
        "mingw32",
        "\\ucrt64\\",
        "\\clang64\\",
    )
    return any(m in s for m in markers)


def _strip_msys_from_path(path_val: str) -> str:
    """Убрать MSYS2/MinGW из PATH: иначе libclang для opencv-rust цепляет gcc-заголовки и падает на MSVC OpenCV."""
    parts = [p for p in path_val.split(os.pathsep) if p.strip() and not _path_segments_need_msys_strip(p)]
    return os.pathsep.join(parts)


def _llvm_clang_and_libclang_dirs(bindir: Path) -> tuple[Path, Path] | None:
    """В официальном LLVM для Windows clang.exe в bin/, libclang.dll часто в lib/. Возвращает (bin, dir_with_libclang_dll)."""
    bindir = bindir.resolve()
    if not (bindir / "clang.exe").is_file():
        return None
    if (bindir / "libclang.dll").is_file():
        return bindir, bindir
    libdir = bindir.parent / "lib"
    if (libdir / "libclang.dll").is_file():
        return bindir, libdir.resolve()
    return None


def _try_llvm_layout(path: Path) -> tuple[Path, Path] | None:
    """path может быть .../LLVM, .../LLVM/bin или .../LLVM/lib."""
    p = path.resolve()
    if (p / "clang.exe").is_file():
        return _llvm_clang_and_libclang_dirs(p)
    if (p / "bin" / "clang.exe").is_file():
        return _llvm_clang_and_libclang_dirs(p / "bin")
    # только libclang.dll в lib/ (иногда задают LIBCLANG_PATH так)
    if p.name.lower() == "lib" and (p / "libclang.dll").is_file():
        b = p.parent / "bin"
        if (b / "clang.exe").is_file():
            return _llvm_clang_and_libclang_dirs(b)
    return None


def _ensure_clang_for_opencvrs(env: dict[str, str]) -> bool:
    """opencv-binding-generator ищет clang.exe в PATH и загружает libclang.dll (clang-sys: LIBCLANG_PATH = каталог с DLL).

    В LLVM для Windows DLL часто лежит в lib/, а clang.exe в bin/ — это нормально.
    """
    if not sys.platform.startswith("win"):
        return True

    manual = os.environ.get("INTEGRA_LLVM_BIN", "").strip()
    candidates: list[Path] = []

    if manual:
        candidates.append(Path(manual))

    candidates.extend(
        [
            Path(r"C:\Program Files\LLVM\bin"),
            Path(r"C:\Program Files\LLVM"),
            Path(r"C:\Program Files (x86)\LLVM\bin"),
            Path(r"C:\Program Files (x86)\LLVM"),
        ]
    )

    try:
        vs_roots = [
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Microsoft Visual Studio",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        ]
        for vs_root in vs_roots:
            if not vs_root.is_dir():
                continue
            for year in ("2022", "2019", "2017"):
                for edition in ("Community", "Professional", "Enterprise", "BuildTools", "Preview"):
                    p = vs_root / year / edition / "VC" / "Tools" / "Llvm" / "x64" / "bin"
                    candidates.append(p)
    except Exception:
        pass

    def apply_layout(clang_bin: Path, libclang_dir: Path) -> None:
        env["LIBCLANG_PATH"] = str(libclang_dir)
        # clang.exe должен быть в PATH; каталог с DLL — для загрузки рядом с процессом при необходимости
        env["PATH"] = (
            str(clang_bin)
            + os.pathsep
            + str(libclang_dir)
            + os.pathsep
            + env.get("PATH", "")
        )

    cur_lp = env.get("LIBCLANG_PATH", "").strip()
    if cur_lp:
        layout = _try_llvm_layout(Path(cur_lp))
        if layout:
            cb, lcd = layout
            apply_layout(cb, lcd)
            print(f"[run.py] LIBCLANG_PATH / LLVM: clang={cb}, libclang.dll={lcd}", flush=True)
            return True
        print(
            f"[run.py] LIBCLANG_PATH={cur_lp} не удалось сопоставить с clang.exe + libclang.dll — ищем стандартные пути.",
            flush=True,
        )

    seen: set[str] = set()
    for d in candidates:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        layout = _try_llvm_layout(d)
        if layout:
            cb, lcd = layout
            apply_layout(cb, lcd)
            print(f"[run.py] LLVM toolchain → clang={cb}, LIBCLANG_PATH={lcd}", flush=True)
            return True

    w = shutil.which("clang.exe") or shutil.which("clang")
    if w:
        wp = Path(w).resolve().parent
        layout = _llvm_clang_and_libclang_dirs(wp)
        if layout:
            cb, lcd = layout
            apply_layout(cb, lcd)
            print(f"[run.py] LLVM из PATH → clang={cb}, LIBCLANG_PATH={lcd}", flush=True)
            return True

    print(
        "[run.py] ОШИБКА: не найдены clang.exe и libclang.dll (ожидается LLVM: bin\\clang.exe + lib\\libclang.dll).\n"
        "         Установите «LLVM» с https://releases.llvm.org/\n"
        "         или задайте INTEGRA_LLVM_BIN=C:\\\\Program Files\\\\LLVM  (или ...\\\\LLVM\\\\bin)",
        file=sys.stderr,
        flush=True,
    )
    return False


def _normalize_opencv_msvc_crt(env: dict[str, str]) -> None:
    """opencv-rs принимает только OPENCV_MSVC_CRT=static|dynamic (не md/mt)."""
    if not sys.platform.startswith("win"):
        return
    v = env.get("OPENCV_MSVC_CRT", "").strip().lower()
    if v not in ("static", "dynamic"):
        env["OPENCV_MSVC_CRT"] = "dynamic"


def _boost_opencv_for_opencvrs(root: Path, env: dict[str, str]) -> None:
    """Если пользователь не задал OPENCV_*, пробуем локальный build/opencv (Windows MSVC world)."""
    if env.get("OPENCV_INCLUDE_PATHS") and env.get("OPENCV_LINK_PATHS") and env.get(
        "OPENCV_LINK_LIBS"
    ):
        return
    if not sys.platform.startswith("win"):
        return
    inc = root / "build" / "opencv" / "include"
    lib = root / "build" / "opencv" / "x64" / "vc17" / "lib"
    bin_dir = root / "build" / "opencv" / "x64" / "vc17" / "bin"
    if not inc.is_dir() or not lib.is_dir():
        return
    libs = sorted(lib.glob("opencv_world*.lib"))
    if not libs:
        return
    # Берём самую «свежую» по имени (opencv_world4140.lib vs модули).
    lib_file = max(libs, key=lambda p: p.name)
    env["OPENCV_INCLUDE_PATHS"] = str(inc.resolve())
    env["OPENCV_LINK_PATHS"] = str(lib.resolve())
    env["OPENCV_LINK_LIBS"] = lib_file.stem
    env.setdefault("OPENCV_MSVC_CRT", "dynamic")
    if bin_dir.is_dir():
        env["PATH"] = str(bin_dir.resolve()) + os.pathsep + env.get("PATH", "")


def _parse_listen(addr: str) -> tuple[str, int]:
    host, _, port_s = addr.partition(":")
    if not host or not port_s:
        raise ValueError(f"bad INTEGRA_VIDEO_BRIDGE_ADDR: {addr!r}")
    return host, int(port_s)


def _prepare_video_bridge_build_env(root: Path, env: dict[str, str]) -> bool:
    """OPENCV_* из build/opencv, при необходимости strip MSYS из PATH и LLVM для opencv-rs на Windows.

    Возвращает False, если на Windows не найдены clang.exe + libclang.dll.
    """
    _boost_opencv_for_opencvrs(root, env)
    if not env.get("OPENCV_INCLUDE_PATHS"):
        print(
            "[run.py] Предупреждение: OPENCV_INCLUDE_PATHS не задан и не найден build/opencv.\n"
            "         video-bridge (opencv-rust) может не собраться. Задайте OPENCV_INCLUDE_PATHS,\n"
            "         OPENCV_LINK_PATHS, OPENCV_LINK_LIBS (см. https://github.com/twistedfall/opencv-rust#getting-opencv).",
            file=sys.stderr,
        )

    if sys.platform.startswith("win"):
        if os.environ.get("INTEGRA_KEEP_MSYS_IN_PATH", "").lower() not in ("1", "true", "yes"):
            old = env.get("PATH", "")
            env["PATH"] = _strip_msys_from_path(old)
            if env["PATH"] != old:
                print(
                    "[run.py] из PATH убраны сегменты MSYS/MinGW (нужно для opencv-rust + MSVC OpenCV).\n"
                    "         Чтобы оставить как было: INTEGRA_KEEP_MSYS_IN_PATH=1",
                    flush=True,
                )
        if not _ensure_clang_for_opencvrs(env):
            return False
        _normalize_opencv_msvc_crt(env)
    return True


def _sync_opencv_dnn_cuda_env(root: Path, env: dict[str, str]) -> None:
    """Если в config.json model.device содержит «cuda», включаем CUDA-бэкенд OpenCV DNN (см. native/src/opencv_dnn_engine.cpp).

    Уже заданный INTEGRA_OPENCV_DNN_CUDA не перезаписываем. Если OpenCV собран без cuDNN, движок сам откатится на CPU.
    """
    if env.get("INTEGRA_OPENCV_DNN_CUDA", "").strip():
        return
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        dev = str(cfg.get("model", {}).get("device", "")).lower()
        if "cuda" in dev:
            env["INTEGRA_OPENCV_DNN_CUDA"] = "1"
            print(
                "[run.py] model.device указывает CUDA → INTEGRA_OPENCV_DNN_CUDA=1 для OpenCV DNN.\n"
                "         Нужен OpenCV, собранный с CUDA/cuDNN; иначе останется CPU (~100+ мс на кадр).",
                flush=True,
            )
    except Exception:
        pass


def _read_bridge_target_fps(root: Path) -> int:
    """Частота кадров video-bridge (0 = без лимита). Берём из config.json pipeline.target_fps."""
    cfg_path = root / "config.json"
    try:
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        tf = int(cfg.get("pipeline", {}).get("target_fps", 30))
        return max(0, min(tf, 120))
    except Exception:
        return 30


def _wait_tcp_listening(addr: str, timeout_sec: float = 20.0) -> bool:
    host, port = _parse_listen(addr)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Integra Native — Rust-only launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--release", action="store_true", help="cargo build --release (рекомендуется)"
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="не запускать video-bridge (smoke-тест gateway отдельно)",
    )
    parser.add_argument(
        "--bridge-addr",
        default="127.0.0.1:9876",
        help="адрес TCP-сервера video-bridge (BGR-потока)",
    )
    parser.add_argument(
        "--bridge-wait-sec",
        type=float,
        default=25.0,
        help="сколько ждать, пока :9876 начнёт принимать соединения",
    )
    parser.add_argument(
        "--bridge-build-only",
        action="store_true",
        help="только собрать video-bridge (те же OPENCV_/LLVM/PATH, что при обычном run.py); без gateway и без integra_ffi",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    env: dict[str, str] = dict(os.environ)
    env["INTEGRA_PROJECT_ROOT"] = str(root)
    env["INTEGRA_BACKEND_HOST"] = str(args.host)
    env["INTEGRA_BACKEND_PORT"] = str(args.port)
    env["INTEGRA_VIDEO_BRIDGE_ADDR"] = args.bridge_addr

    if args.bridge_build_only and args.no_bridge:
        parser.error("--bridge-build-only несовместим с --no-bridge")

    if args.bridge_build_only:
        if not _prepare_video_bridge_build_env(root, env):
            raise SystemExit(3)
        if args.release:
            build_cmd = ["cargo", "build", "--release", "--manifest-path", "Cargo.toml"]
        else:
            build_cmd = ["cargo", "build", "--manifest-path", "Cargo.toml"]
        print("[run.py] только сборка video-bridge (--bridge-build-only) …", flush=True)
        br = subprocess.run(
            build_cmd,
            cwd=str(root / "video-bridge"),
            env=env,
        )
        if br.returncode != 0:
            print(
                "[run.py] Сборка video-bridge не удалась.\n"
                "         • Если вы запускали голый «cargo build» в video-bridge: задайте OPENCV_* или используйте этот скрипт.\n"
                "         • Если в логе opencv build-script: panic / Failed to run the bindings generator / gapi / internal error:\n"
                "           обновите крейт opencv в video-bridge/Cargo.toml и cargo update (OpenCV 4.14 требует свежий binding-generator).\n"
                "         • Если «cuda_runtime.h file not found» при модуле cudalegacy: в Cargo.toml opencv отключены CUDA-модули или добавьте CUDA Toolkit в OPENCV_CLANG_ARGS — см. video-bridge/Cargo.toml.",
                file=sys.stderr,
                flush=True,
            )
        raise SystemExit(br.returncode)

    ffi_lib = _find_integra_ffi(root)
    if ffi_lib is None:
        print(
            "[run.py] integra_ffi не найдена.\n"
            "         Соберите native: cmake --build native/build-msvc --config RelWithDebInfo --target integra_ffi\n"
            "         или укажите INTEGRA_FFI_PATH вручную.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    env["INTEGRA_FFI_PATH"] = str(ffi_lib)
    env["PATH"] = str(ffi_lib.parent) + os.pathsep + env.get("PATH", "")
    _sync_opencv_dnn_cuda_env(root, env)

    bridge_proc: subprocess.Popen | None = None
    if not args.no_bridge:
        if not _prepare_video_bridge_build_env(root, env):
            raise SystemExit(3)

        if args.release:
            build_cmd = ["cargo", "build", "--release", "--manifest-path", "Cargo.toml"]
            target_dir = "release"
        else:
            build_cmd = ["cargo", "build", "--manifest-path", "Cargo.toml"]
            target_dir = "debug"
        print("[run.py] cargo build video-bridge …", flush=True)
        br = subprocess.run(
            build_cmd,
            cwd=str(root / "video-bridge"),
            env=env,
        )
        if br.returncode != 0:
            print(
                "[run.py] Сборка video-bridge не удалась — gateway не сможет открыть видео.\n"
                "         Если «Can't find clang binary»: установите LLVM for Windows (clang.exe + libclang.dll)\n"
                "           или компонент VS «C++ Clang tools», либо INTEGRA_LLVM_BIN=C:\\\\path\\\\to\\\\LLVM\\\\bin.\n"
                "           • не задавайте LIBCLANG_PATH на MinGW; переоткройте shell без MSYS в PATH, или\n"
                "           • установите LLVM for Windows / компонент «Clang» в Visual Studio — run.py подхватит сам, или\n"
                "           • INTEGRA_KEEP_MSYS_IN_PATH=1 только если осознанно отлаживаете конфликт.\n"
                "         Либо только пересобрать мост:  python run.py --release --bridge-build-only\n"
                "         Если в логе build-script opencv — panic при генерации модулей (gapi и т.д.): обновите крейт opencv / привяжите OpenCV к поддерживаемой opencv-rust версии.\n"
                "         Либо smoke без видео:  python run.py --no-bridge",
                file=sys.stderr,
            )
            raise SystemExit(br.returncode)

        bridge_exe = root / "video-bridge" / "target" / target_dir / (
            "video-bridge.exe" if sys.platform.startswith("win") else "video-bridge"
        )
        if not bridge_exe.is_file():
            print(f"[run.py] нет бинарника после сборки: {bridge_exe}", file=sys.stderr)
            raise SystemExit(1)

        bridge_tf = _read_bridge_target_fps(root)
        bridge_args = [
            str(bridge_exe),
            "--listen",
            args.bridge_addr,
            "--target-fps",
            str(bridge_tf),
        ]
        bridge_proc = subprocess.Popen(bridge_args, cwd=str(root / "video-bridge"), env=env)
        print(
            f"[run.py] video-bridge: {args.bridge_addr} (pid={bridge_proc.pid}) target_fps={bridge_tf}",
            flush=True,
        )

        if not _wait_tcp_listening(args.bridge_addr, timeout_sec=args.bridge_wait_sec):
            print(
                f"[run.py] За {args.bridge_wait_sec}s порт {args.bridge_addr} не открылся.\n"
                "         Проверьте, что video-bridge не упал (окно/лог), и что OpenCV DLL в PATH.",
                file=sys.stderr,
            )
            if bridge_proc.poll() is not None:
                print(
                    f"[run.py] video-bridge уже завершился с кодом {bridge_proc.returncode}.",
                    file=sys.stderr,
                )
            try:
                bridge_proc.terminate()
                bridge_proc.wait(timeout=2)
            except Exception:
                bridge_proc.kill()
            raise SystemExit(1)

    gw_cmd = ["cargo", "run", "--bin", "backend_gateway"]
    if args.release:
        gw_cmd.insert(2, "--release")

    print(f"[run.py] FFI: {ffi_lib}")
    print(f"[run.py] backend-gateway: {args.host}:{args.port}")

    try:
        gw_proc = subprocess.Popen(gw_cmd, cwd=str(root / "runtime-core"), env=env)
        rc = gw_proc.wait()
        raise SystemExit(rc)
    except KeyboardInterrupt:
        if bridge_proc is not None:
            try:
                bridge_proc.send_signal(signal.SIGINT)
            except Exception:
                pass
        raise SystemExit(130)
    finally:
        if bridge_proc is not None:
            try:
                bridge_proc.terminate()
                bridge_proc.wait(timeout=3)
            except Exception:
                try:
                    bridge_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
