#pragma once

#include "integra/types.hpp"

#include <vector>

namespace integra {

// ---------------------------------------------------------------------------
// Frame-level anti-noise filter (one-shot, без межкадровой памяти).
//
// Применяется ПОСЛЕ inference / NMS, но ДО трекера и FSM.
// Срезает: слишком маленькие боксы, дегенеративные aspect ratio, мелочёвку
// с низкой уверенностью, переполнение детекций. Пороги уверенности
// разделены для person и остальных классов (person выше — меньше FP на мебель).
//
// Источник параметров — config.json / IntegraConfig: `conf_threshold`, `min_box_size_px`,
// опционально `ignore_det_norm_*` (см. integra_ffi.h).
// tune_from_postprocess() выравнивает min_conf с порогом модели (без «двойного» завышения).
// ---------------------------------------------------------------------------

struct FrameFilterConfig {
  // Для объектов: после tune_from_postprocess() совпадает с conf YOLO (см. .cpp).
  float min_conf = 0.51f;
  // Для person: выше базового conf YOLO (см. tune_from_postprocess) — меньше «person» на стол/стену.
  float min_conf_person = 0.56f;
  // Минимальная сторона bbox в пикселях кадра.
  int min_box_px = 20;
  // Минимальная доля площади bbox от площади кадра.
  float min_area_ratio = 0.0004f;
  // Максимальная доля площади bbox от площади кадра.
  float max_area_ratio = 0.25f;
  // Допустимый aspect ratio (w/h).
  float min_aspect = 0.15f;
  float max_aspect = 5.0f;
  // Жёсткий предел числа детекций на кадр (после фильтрации, top-by-conf).
  int max_detections = 48;
  // > 0: отбрасываем боксы, касающиеся края на N px. 0 = не резать.
  int border_px = 0;
  // Нормализованный 0..1 прямоугольник: не-person с центроидом внутри — отброс. x2<=x1 => выкл.
  float ignore_det_norm_x1 = 0.f;
  float ignore_det_norm_y1 = 0.f;
  float ignore_det_norm_x2 = 0.f;
  float ignore_det_norm_y2 = 0.f;

  // Как в Python `detector.py`: порог confidence от центроида Y + послабление у краёв кадра.
  // Если false — используются фиксированные пороги по class_id ниже в apply_frame_filter.
  bool use_regional_class_conf = false;
  float upper_region_y_ratio = 0.62f;
  float min_conf_upper = 0.22f;
  float min_conf_lower = 0.30f;
  float bottom_region_y_ratio = 0.88f;
  float min_conf_bottom = 0.26f;
  int conf_border_relax_px = 0;
  float min_conf_border = 0.20f;
  float person_min_conf_border = 0.18f;

  static constexpr int kClassMinConfMax = 16;
  int class_min_conf_count = 0;
  int class_min_conf_ids[kClassMinConfMax]{};
  float class_min_conf_thresholds[kClassMinConfMax]{};

  /// Выравнивает min_conf с порогом постпроцесса YOLO (см. реализацию в .cpp).
  void tune_from_postprocess(float conf_threshold);
};

/// Возвращает true, если класс попадает в whitelist целевых объектов.
///   • person_class_id всегда «релевантен».
///   • Если object_classes непустой — whitelist строгий.
///   • Иначе — доменный default (рюкзак / сумка / чемодан / бутылка / …).
bool is_relevant_class(int cls_id, int person_class_id,
                       const std::vector<int>& object_classes);

/// Возвращает копию входа, оставив только детекции, прошедшие фильтр.
/// Сохраняет original conf / class / bbox; не трогает track_id.
std::vector<Detection> apply_frame_filter(const std::vector<Detection>& in,
                                           int frame_w, int frame_h,
                                           int person_class_id,
                                           const FrameFilterConfig& ff);

/// Жадное схлопывание сильно перекрывающихся боксов без учёта class_id
/// (оставляет более уверенную детекцию). Снижает «роя» ложных классов на одном фоне.
void merge_cross_class_by_iou(std::vector<Detection>& dets, float iou_threshold);

/// COCO backpack / handbag / suitcase: один угол коробки часто даёт два класса с низким conf —
/// оставляем одну детекцию с максимальным confidence (person не трогаем).
void merge_luggage_cross_class_by_iou(std::vector<Detection>& dets, int person_class_id,
                                      float iou_threshold);

/// То же, но только среди детекций с class_id != person_class_id (person не трогаем).
void merge_cross_class_objects_only(std::vector<Detection>& dets, int person_class_id,
                                    float iou_threshold);

/// Схлопывание bottle / wine glass / cup / bowl: высокий IoU **или** соседство в ряд/стопку
/// между разными классами посуды (бутылка+стакан). Пара **bottle+bottle** — только по IoU
/// (см. `tableware_adjacent_any_class`), порог вызова должен быть высоким, иначе сливаются соседние канистры.
void merge_tableware_cross_iou(std::vector<Detection>& dets, int person_class_id,
                               float iou_threshold);

/// Схлопывание дубликатов одного класса: **только IoU ≥ порога** (почти один и тот же бокс).
/// Геометрическая склейка соседних объектов отключена — отдельные треки, как в эталоне Python.
void merge_same_class_objects_only(std::vector<Detection>& dets, int person_class_id,
                                    float iou_threshold);

/// Жадное подавление дублей YOLO на одном физическом кластере канистр (COCO 39): оставляет
/// более уверенный бокс, если IoU с уже принятым ≥ порога (NMS по одному классу).
void suppress_duplicate_bottles_by_iou(std::vector<Detection>& dets, int person_class_id,
                                       float iou_threshold);

}  // namespace integra
