"""pyinfer — Python-воркер инференса для Integra-LOST.

YOLOv11 (onnxruntime, CPU) + ByteTrack (контур людей) + class-agnostic объекты
сцены (MOG2 + IoU-трекер) + поведенческая FSM. Говорит по TCP-протоколу
infer_worker (см. runtime-core/src/bin/infer_worker.rs), поэтому подключается к
существующему Rust backend_gateway без изменений в нём.
"""
