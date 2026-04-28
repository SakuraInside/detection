# Integra-LOST · Детектор оставленного / исчезнувшего предмета

Реализация сценария «Оставленный / исчезнувший предмет» из
**Постановления Правительства РФ № 969 от 26 сентября 2016 г.**
Веб-интерфейс + YOLOv11 (Ultralytics) + BoT-SORT-трекер +
конечный автомат событий с двойным временным окном
(стабилизация → подтверждение оставленности).

## Архитектура

Конвейер из 4 потоков (см. `app/pipeline.py`):

```
[Decoder thread]  ── bounded queue ──>  [Inference worker]  ──>  latest_jpeg
   cv2.VideoCapture                       YOLOv11 + BoT-SORT
   MKV / MP4 / AVI                        ↓
                                          Analyzer (FSM)
                                          ↓                    ┌── WebSocket  → UI
                                          AnalyzerEvent ──────┤── EventLogger → SQLite
                                                              └── snapshot    → JPEG
```

Состояния трекинга (FSM):

```
NONE → CANDIDATE → STATIC → UNATTENDED → ALARM_ABANDONED
                                            ↓ (track lost ≥ disappear_grace_sec)
                                          ALARM_DISAPPEARED
```

* `STATIC` — смещение центроида ≤ `static_displacement_px` за окно
  `static_window_sec`.
* `UNATTENDED` — `owner_left_sec` секунд рядом нет человека
  (`owner_proximity_px` или пересечение bbox).
* `ALARM_ABANDONED` — объект простоял один ≥ `abandon_time_sec`.
* `ALARM_DISAPPEARED` — раз поднятая тревога продолжает «гореть»,
  если объект пропал из кадра ≥ `disappear_grace_sec`.

## Установка

```powershell
# 1. (рекомендуется) создать venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. установить PyTorch с CUDA (выберите команду под ваш CUDA build,
#    см. https://pytorch.org/get-started/locally/ ).  Пример для CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. остальные зависимости
pip install -r requirements.txt
```

При первом запуске Ultralytics автоматически скачает веса
`yolo11x.pt`. На RTX 4090 24 GB можно оставить как есть; на менее
мощных GPU (например, 4 GB VRAM) откройте настройки и выберите
`yolo11s.pt` или `yolo11m.pt` — все параметры применяются на лету.

## Запуск

```powershell
python run.py                 # http://127.0.0.1:8000
python run.py --host 0.0.0.0  # доступ из локальной сети
```

Положите `.mkv` файлы в `./data/`, либо вставьте абсолютный путь
в поле «Открыть» в UI.

## Веб-интерфейс

* Кнопки **Воспр. / Пауза**, ползунок seek, индикаторы FPS и
  времени инференса.
* Список файлов из `./data/` + ввод абсолютного пути.
* Боковая панель «Активные тревоги» — список треков в состоянии
  `ALARM_ABANDONED` / `ALARM_DISAPPEARED`.
* Журнал событий с миниатюрами (SQLite + JPEG-снапшоты в
  `logs/snapshots/`).
* Сворачиваемая панель настроек: модель, FSM-пороги, UI-флаги.
  Все изменения применяются без перезапуска (модель —
  с тёплой переинициализацией весов).

## Производительность

Конвейер не «копит» кадры под нагрузкой: декодер сбрасывает
*самые старые* кадры, а инференс всегда работает с самым свежим.
Это правильное поведение для live-наблюдения: latency держится
низким, а событие «появление/исчезновение» не теряется,
потому что оно поднимается до сброса (FSM работает на
ts инференса, не на ts отображения).

Для 30+ FPS на HD-видео:
* `model.weights` — `yolo11x.pt` (4090) или `yolo11s/m.pt` (≤6 GB VRAM)
* `model.imgsz` — 640 (быстрее) или 960 (точнее)
* `model.half` — `true` (FP16, ~2× ускорение, требует GPU)
* экспорт в TensorRT (опционально, см. ниже)

### Экспорт в TensorRT (необязательно)

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt').export(format='engine', half=True, imgsz=640)"
```

Затем в `config.json` (или в UI) поменяйте `model.weights` на
`yolo11x.engine`.

## Настройки (выдержка из `config.json`)

| Секция     | Параметр                  | Назначение                                                       |
| ---------- | ------------------------- | ---------------------------------------------------------------- |
| `model`    | `weights`, `imgsz`, `half`| Модель YOLO, размер входа, FP16                                  |
| `model`    | `object_classes`          | COCO id «оставляемых» предметов (24 backpack, 26 handbag, 28 suitcase, 39 bottle, 41 cup, 63 laptop…) |
| `analyzer` | `static_window_sec`       | Длительность окна, на котором проверяется неподвижность          |
| `analyzer` | `abandon_time_sec`        | Тайм-аут до тревоги «оставлен»                                   |
| `analyzer` | `owner_left_sec`          | Сколько объект должен оставаться один без человека рядом         |
| `analyzer` | `disappear_grace_sec`     | После скольких секунд пропавший «оставленный» помечается исчезшим|

## Структура проекта

```
integra/
├── run.py              # uvicorn-обвязка
├── config.json         # настройки по умолчанию (живая конфигурация)
├── requirements.txt
├── README.md
├── app/
│   ├── main.py         # FastAPI: REST + WS + MJPEG
│   ├── pipeline.py     # потоки decode/inference + рендер оверлея
│   ├── detector.py     # обёртка YOLOv11 + BoT-SORT
│   ├── analyzer.py     # FSM «оставленный/исчезнувший»
│   ├── logger_db.py    # SQLite-логгер событий
│   └── config.py       # типизированные настройки и hot-reload
├── static/             # одностраничный UI (Tailwind CDN)
├── data/               # сюда кладём .mkv видеофайлы
├── logs/
│   ├── events.db       # SQLite журнал
│   └── snapshots/      # JPEG-снапшоты тревог
└── models/             # сюда можно класть собственные .pt / .engine
```

## Лицензия

Внутренний учебно-исследовательский проект.
Использует Ultralytics YOLOv11 (AGPL-3.0).
