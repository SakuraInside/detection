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
// разделены для person и остальных классов (person обычно нужно жёстче).
//
// Источник параметров — config.json / IntegraConfig: `conf_threshold` и `min_box_size_px`.
// tune_from_postprocess() выравнивает min_conf с порогом модели (без «двойного» завышения).
// ---------------------------------------------------------------------------

struct FrameFilterConfig {
  // Для объектов: после tune_from_postprocess() совпадает с conf YOLO (см. .cpp).
  float min_conf = 0.51f;
  // Для person: обычно conf_threshold + небольшой запас.
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

}  // namespace integra
