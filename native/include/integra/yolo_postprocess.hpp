#pragma once

#include "integra/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace integra {

/// Пороги постобработки (совместимо с Ultralytics export ONNX).
struct PostprocessParams {
  float conf_threshold = 0.25f;
  float nms_iou_threshold = 0.45f;
  int num_classes = 80;
  /// Число якорей в выходе (640×640 → обычно 8400; для другого imgsz задайте из shape тензора).
  int num_anchors = 8400;
};

/// Эвристика выхода YOLO [1,d1,d2]: true = каналы по dim1 (84×8400), false = якоря по dim1 (8400×84).
bool yolo_output_channel_first(std::int64_t d1, std::int64_t d2);

/// Имя класса COCO по индексу 0..79 (или пустая строка).
std::string coco80_class_name(int cls_id);

/// Декодирование одного «слоя» YOLOv8: тензор [1, 4+nc, N] в памяти как CHW:
/// индекс элемента (channel c, anchor i) = `data[c * N + i]`, N = num_anchors.
/// Боксы — центр + размер в пикселях относительно входа (imgsz × imgsz).
void decode_yolov8_chw(const float* chw, const PostprocessParams& pp, float input_w, float input_h,
                       std::vector<Detection>& out);

/// Без batch: выход `(dim1, dim2)` в row-major. Если `dim1_is_channel` — формат [C,N]
/// (как Ultralytics `(1,84,8400)`); иначе [N,C] `(1,8400,84)` — внутри делается транспозиция
/// в scratch (как CHW), затем decode.
void decode_yolov8_flat(const float* data, int dim1, int dim2, bool dim1_is_channel,
                        PostprocessParams pp, float input_w, float input_h,
                        std::vector<float>& scratch, std::vector<Detection>& out);

/// Класс-aware greedy NMS (как в типичном постпроцессе детекции).
void nms_greedy_xyxy(std::vector<Detection>& dets, float iou_threshold);

/// Проверка decode+NMS на синтетическом буфере (для CI / ручного smoke).
bool selftest_yolo_postprocess();

}  // namespace integra
