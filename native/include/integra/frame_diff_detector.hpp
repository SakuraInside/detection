#pragma once

#include "integra/geom.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/video/background_segm.hpp>
#include <vector>

namespace integra {

/// Структура для хранения информации об обнаруженном регионе-кандидате сцены.
struct ChangedRegion {
  BBoxXYXY bbox;           // Ограничивающий прямоугольник (в координатах исходного кадра)
  float pixel_change;      // Заполненность маски в ROI (0..255) — грубая метрика
  float gradient_change;   // То же (совместимость со старым интерфейсом)
  float combined_score;    // Комбинированный скор (для сортировки/уверенности)
};

/// Детектор class-agnostic регионов-кандидатов сцены.
///
/// Реализован поверх модели фона (MOG2) на уменьшенном grayscale-кадре:
///  * вся стадия обнаружения векторизована внутри OpenCV (без per-pixel циклов
///    на полном разрешении — именно они «вешали» поток);
///  * модель фона с низкой скоростью адаптации удерживает оставленный
///    неподвижный предмет как «передний план» достаточно долго, чтобы
///    поведенческая FSM подтвердила статику (простой diff соседних кадров на это
///    не способен — у неподвижного объекта нет межкадровой разницы).
class FrameDiffDetector {
 public:
  explicit FrameDiffDetector(int buffer_size = 10);

  /// Добавить новый кадр и вернуть регионы-кандидаты (координаты — исходный кадр).
  /// pixel_threshold/gradient_threshold оставлены для совместимости ABI; чувстви-
  /// тельность задаётся моделью фона (varThreshold) и порогом площади.
  std::vector<ChangedRegion> process_frame(const cv::Mat& bgr_frame,
                                           float pixel_threshold = 20.f,
                                           float gradient_threshold = 15.f,
                                           int min_region_area_px = 100);

  /// Последний обработанный (уменьшенный, grayscale) кадр.
  cv::Mat get_last_frame() const;

  /// Условная «глубина» истории модели фона.
  int buffer_size() const { return history_; }

  /// Сбросить модель фона (например, при переключении видео / seek).
  void reset();

 private:
  /// Слить соседние регионы в один bbox.
  std::vector<ChangedRegion> merge_regions(const std::vector<ChangedRegion>& regions,
                                           float iou_threshold = 0.3f);

  void rebuild_model();

  int history_;
  cv::Ptr<cv::BackgroundSubtractorMOG2> bg_;
  cv::Mat last_small_;       // последний обработанный grayscale-кадр (уменьшенный)
  double inv_scale_ = 1.0;   // множитель для возврата bbox к исходному размеру
};

}  // namespace integra
