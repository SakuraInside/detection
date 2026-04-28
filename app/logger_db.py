"""Журнал событий в SQLite.

Запись идет через фоновый поток, чтобы не блокировать инференс.
Одна connection в WAL-режиме достаточно быстра для нашего потока событий.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    video_pos_ms  REAL    NOT NULL,
    type          TEXT    NOT NULL,
    track_id      INTEGER,
    cls_id        INTEGER,
    cls_name      TEXT,
    confidence    REAL,
    bbox_x1       REAL,
    bbox_y1       REAL,
    bbox_x2       REAL,
    bbox_y2       REAL,
    snapshot_path TEXT,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(type);
"""


@dataclass
class EventRow:
    ts: float
    video_pos_ms: float
    type: str
    track_id: Optional[int]
    cls_id: Optional[int]
    cls_name: Optional[str]
    confidence: Optional[float]
    bbox: Optional[tuple[float, float, float, float]]
    snapshot_path: Optional[str]
    note: Optional[str] = None


class EventLogger:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Одна connection + lock достаточно для нашей нагрузки.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(SCHEMA)
        # WAL улучшает конкурентное чтение/запись, NORMAL снижает sync-накладные расходы.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.commit()
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Optional[EventRow]]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="event-logger", daemon=True)
        self._worker.start()

    def log(self, row: EventRow) -> None:
        # Неблокирующая запись в очередь.
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Если диск/БД тормозит, вытесняем самую старую запись.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(row)

    def _run(self) -> None:
        # Фоновый consumer: вытаскивает события из очереди и пишет в БД.
        while not self._stop.is_set():
            try:
                row = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if row is None:
                break
            self._insert(row)

    def _insert(self, row: EventRow) -> None:
        # Если bbox отсутствует, пишем NULL-значения в координаты.
        bx = row.bbox or (None, None, None, None)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events (ts, video_pos_ms, type, track_id, cls_id, cls_name,
                                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                    snapshot_path, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.ts,
                    row.video_pos_ms,
                    row.type,
                    row.track_id,
                    row.cls_id,
                    row.cls_name,
                    row.confidence,
                    bx[0],
                    bx[1],
                    bx[2],
                    bx[3],
                    row.snapshot_path,
                    row.note,
                ),
            )
            self._conn.commit()

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        # Для UI возвращаем последние события в обратном порядке.
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT id, ts, video_pos_ms, type, track_id, cls_id, cls_name,
                       confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, snapshot_path, note
                FROM events ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return rows

    def clear(self) -> None:
        # Полная очистка журнала событий.
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()

    def close(self) -> None:
        # Отправляем "сентинел" в очередь и дожидаемся завершения worker-потока.
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        with self._lock:
            self._conn.close()


def now() -> float:
    # Вынесено в функцию для единообразия и удобства возможного мокирования.
    return time.time()


def iter_classnames(class_ids: Iterable[int], names: dict[int, str]) -> list[str]:
    # Преобразует список id классов в человекочитаемые имена.
    return [names.get(i, str(i)) for i in class_ids]
