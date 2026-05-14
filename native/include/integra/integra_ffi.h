// integra_ffi.h — C ABI поверх integra_core.
//
// Этот заголовок предназначен для интеграции из Rust (через bindgen / cc) или
// любого другого языка без C++. Все типы — POD; все строки — UTF-8,
// null-terminated. Memory ownership: pipeline владеет своими ресурсами;
// все указатели в payload_json валидны ТОЛЬКО на время вызова callback.
//
// Контракты потокобезопасности:
//   • integra_pipeline_push_frame() — НЕ потокобезопасен для одного pipeline.
//     Caller обязан сериализовать вызовы (один поток на pipeline).
//   • Разные pipelines независимы и могут вызываться параллельно.
//   • integra_pipeline_create/destroy — потокобезопасны (mutex внутри).
//
// Шаринг тяжёлых ресурсов: при kind="tensorrt" повторный create с тем же
// model_path использует общий ICudaEngine из кэша (см. SharedTRTEngine).

#ifndef INTEGRA_FFI_H
#define INTEGRA_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(INTEGRA_FFI_BUILD)
#    define INTEGRA_API __declspec(dllexport)
#  else
#    define INTEGRA_API __declspec(dllimport)
#  endif
#else
#  if defined(INTEGRA_FFI_BUILD)
#    define INTEGRA_API __attribute__((visibility("default")))
#  else
#    define INTEGRA_API
#  endif
#endif

// При несовпадении ABI integra_pipeline_create вернёт NULL.
#define INTEGRA_FFI_ABI_VERSION 4

/// Максимум пар (class_id → min confidence) из `model.class_min_conf` для нативного фильтра.
#define INTEGRA_CLASS_MIN_CONF_MAX 16

// ---------------------------------------------------------------------------
// Pipeline handle (opaque).
// ---------------------------------------------------------------------------
typedef struct IntegraPipeline IntegraPipeline;

// ---------------------------------------------------------------------------
// Callback type. Вызывается синхронно из integra_pipeline_push_frame().
//
//   type_tag    — "event"  : alarm event (abandoned / disappeared / …)
//                 "frame_result" : полный результат кадра (tracks + persons + stats)
//   payload_json — UTF-8 JSON, null-terminated, ВАЛИДЕН ТОЛЬКО НА ВРЕМЯ ВЫЗОВА.
//   user_data    — то, что было передано в push_frame.
// ---------------------------------------------------------------------------
typedef void (*IntegraEventCb)(const char* type_tag,
                                const char* payload_json,
                                void* user_data);

// ---------------------------------------------------------------------------
// Config. Все поля заполняются caller'ом.
// Нулевые/пустые значения трактуются как «default» там, где это разумно.
// ---------------------------------------------------------------------------
typedef struct {
  // ---- inference engine ----
  const char* engine_kind;      // "tensorrt" | "opencv" | "onnx" | "stub". NULL => "tensorrt".
  const char* model_path;       // путь к .engine (TRT) / .onnx (opencv / onnx). NULL => пусто.
  int         input_size;       // 640 / 960 / 0=без resize. <=0 => 640.

  // ---- postprocess (YOLO) ----
  float       conf_threshold;   // <=0 => 0.25
  float       nms_iou_threshold; // <=0 => 0.45
  int         num_classes;      // <=0 => 80
  int         num_anchors;      // <=0 => 8400

  // ---- class whitelist / anti-noise ----
  int         person_class_id;  // COCO person = 0
  const int*  object_classes;   // массив class_id; NULL/len=0 => доменный default (backpack/handbag/…)
  int         object_classes_len;
  int         min_box_size_px;  // <=0 => 20

  // ---- analyzer (FSM) ----
  double      static_displacement_px;
  double      static_window_sec;
  double      abandon_time_sec;
  double      owner_proximity_px;
  double      owner_left_sec;
  double      disappear_grace_sec;
  double      min_object_area_px;
  int         centroid_history_maxlen;
  int         max_active_tracks;

  // ---- optional spatial mask (нормализованные 0..1 координаты кадра) ----
  // Если ignore_det_norm_x2 <= ignore_det_norm_x1 или y2 <= y1 — маска выключена.
  // Иначе детекции с class_id != person_class_id и центроидом bbox внутри прямоугольника
  // отбрасываются (типично: зона приёмной, где модель даёт хаос; люди не режутся).
  double      ignore_det_norm_x1;
  double      ignore_det_norm_y1;
  double      ignore_det_norm_x2;
  double      ignore_det_norm_y2;

  // ---- identity ----
  const char* camera_id;        // включается в event JSON. NULL => "main".

  // ---- frame filter: вертикальные пороги conf как в Python detector.py (0 = выкл., старая эвристика) ----
  int         use_regional_class_conf;
  float       upper_region_y_ratio;
  float       min_conf_upper;
  float       min_conf_lower;
  float       bottom_region_y_ratio;
  float       min_conf_bottom;
  int         border_relax_px;
  float       min_conf_border;
  float       person_min_conf_border;

  // ---- IouTracker (нативный путь; не путать с Ultralytics YAML) ----
  float       tracker_iou_match_threshold;  /* <=0 => 0.35 */
  int         tracker_max_missed_frames;    /* <=0 => 10 */
  int         tracker_soft_centroid_match;  /* 0 = только жёсткий IoU, 1 = + мягкий центроид */

  // ---- per-class min confidence (COCO class_id), до INTEGRA_CLASS_MIN_CONF_MAX пар ----
  int         class_min_conf_count;
  int         class_min_conf_class_ids[INTEGRA_CLASS_MIN_CONF_MAX];
  float       class_min_conf_thresholds[INTEGRA_CLASS_MIN_CONF_MAX];
} IntegraConfig;

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

// Создать pipeline. Возвращает NULL при ошибке инициализации или ABI mismatch.
// abi_version: caller обязан передать INTEGRA_FFI_ABI_VERSION.
INTEGRA_API IntegraPipeline* integra_pipeline_create(int abi_version,
                                                      const IntegraConfig* cfg);

// Уничтожить pipeline. Безопасно вызывать с NULL.
INTEGRA_API void integra_pipeline_destroy(IntegraPipeline* p);

// ---------------------------------------------------------------------------
// Frame processing.
//
// bgr_data: row-major BGR uint8, height*width*3 bytes, плотный (без padding).
// pts_ms:   media timestamp кадра в миллисекундах (используется в SceneAnalyzer).
// cb:       вызывается синхронно для каждого alarm event и в конце для
//           frame_result. Можно передать NULL — тогда события и tracks
//           молча выкидываются (полезно для warm-up / benchmark).
//
// Возвращает:
//   0   — успешно;
//   -1  — pipeline == NULL;
//   -2  — bad frame (width/height <=0 или bgr_data == NULL);
//   -3  — inference failed (см. stderr);
//   -4  — ABI mismatch / pipeline недоступен.
// ---------------------------------------------------------------------------
INTEGRA_API int integra_pipeline_push_frame(IntegraPipeline* p,
                                             const uint8_t* bgr_data,
                                             int width,
                                             int height,
                                             int64_t pts_ms,
                                             IntegraEventCb cb,
                                             void* user_data);

// Сбросить трекер и FSM (например, при seek / переключении видео).
INTEGRA_API void integra_pipeline_reset(IntegraPipeline* p);

// ---------------------------------------------------------------------------
// Diagnostics.
// ---------------------------------------------------------------------------

// Возвращает версию сборки FFI (null-terminated, static storage).
INTEGRA_API const char* integra_ffi_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // INTEGRA_FFI_H
