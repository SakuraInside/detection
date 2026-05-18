"""
Экспорт YOLO11s .pt → .onnx для integra native pipeline.
Требует: pip install ultralytics
Запуск:  python scripts/export_onnx.py [--weights models/yolo11s.pt] [--imgsz 640] [--half]
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO11 → ONNX")
    p.add_argument("--weights", default="models/yolo11s.pt")
    p.add_argument("--out", default="", help="Путь выхода .onnx (по умолчанию рядом с весами)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 экспорт (ONNX + fp16 weights)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--no-simplify", dest="simplify", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics не установлен. Запустите: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: файл весов не найден: {weights}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else weights.with_suffix(".onnx")

    print(f"Загружаем модель: {weights}")
    model = YOLO(str(weights))

    print(f"Экспортируем → {out_path}  (imgsz={args.imgsz}, half={args.half}, opset={args.opset}, simplify={args.simplify})")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
    )

    result_path = Path(exported) if exported else out_path
    if result_path != out_path and result_path.exists():
        import shutil
        shutil.move(str(result_path), str(out_path))
        result_path = out_path

    if not result_path.exists():
        print(f"ERROR: экспорт завершён, но файл не найден: {result_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = result_path.stat().st_size / 1024 / 1024
    print(f"\nГотово: {result_path}  ({size_mb:.1f} MB)")
    print(f"Теперь задайте в config.json:")
    print(f'  "engine": "onnx",')
    print(f'  "model_path": "{result_path.as_posix()}"')


if __name__ == "__main__":
    main()
