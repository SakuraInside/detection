#pragma once

#include "integra/types.hpp"

#include <memory>
#include <vector>

namespace integra {

/// Параметры ByteTrack (соответствуют оригинальному BYTETracker).
struct ByteTrackParams {
  /// Порог «высокой уверенности»: эти детекции идут в первую ассоциацию.
  float track_high_thresh = 0.50f;
  /// Нижняя граница «второго прохода» BYTE: детекции в [low, high) спасают потерянные треки.
  float track_low_thresh = 0.10f;
  /// Новый трек заводим только при уверенности не ниже этого порога.
  float new_track_thresh = 0.60f;
  /// IoU-порог матча первого прохода (cost = 1 - IoU; матч при cost <= match_thresh).
  float match_thresh = 0.80f;
  /// Сколько кадров держим потерянный трек до удаления (масштабируется частотой кадров).
  int track_buffer = 30;
  /// Частота кадров источника — влияет на time-to-live потерянных треков.
  float frame_rate = 30.f;
};

/// Полноценный ByteTrack: предсказание Калмана (xyah) + двухстадийная ассоциация
/// (BYTE) + венгерское назначение + жизненный цикл трека (New/Tracked/Lost/Removed).
///
/// Используется ТОЛЬКО для людей (class-based контур). Class-agnostic объекты сцены
/// трекаются отдельным `IouTracker`.
class ByteTracker {
 public:
  explicit ByteTracker(ByteTrackParams params = {});
  ~ByteTracker();
  ByteTracker(const ByteTracker&) = delete;
  ByteTracker& operator=(const ByteTracker&) = delete;
  ByteTracker(ByteTracker&&) noexcept;
  ByteTracker& operator=(ByteTracker&&) noexcept;

  /// Проставляет `det.track_id` у активных (подтверждённых) треков; неподтверждённые
  /// детекции остаются с track_id = -1 (как в оригинальном ByteTrack — ID появляется
  /// со второго кадра). Координаты — в пространстве полного кадра (xyxy).
  void update(std::vector<Detection>& dets, int frame_w = 0, int frame_h = 0);

  void reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace integra
