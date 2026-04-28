"""Точка входа в приложение.

Примеры запуска:
    python run.py                    # запуск на http://127.0.0.1:8000
    python run.py --host 0.0.0.0     # открыть доступ в локальной сети
    python run.py --port 8080        # запуск на другом порту
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    # CLI-параметры удобны для локального запуска без правки кода.
    parser = argparse.ArgumentParser(description="Integra-LOST: abandoned object detector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true", help="горячая перезагрузка для разработки")
    args = parser.parse_args()

    # Запускаем ASGI-сервер и передаем путь к приложению FastAPI.
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
