#include "integra/frame_filter.hpp"

#include <algorithm>

namespace integra {

void FrameFilterConfig::tune_from_postprocess(float conf_threshold) {
  // Важно: YOLO-постпроцесс уже отфильтровал по `conf_threshold`.
  // Здесь не поднимаем порог выше модели — иначе «глушим» все детекции.
  // Person слегка строже (шумный класс); точечные классы — в apply_frame_filter ниже.
  const float c = std::clamp(conf_threshold, 0.01f, 0.99f);
  min_conf = c;
  min_conf_person = std::min(0.99f, c + 0.05f);
}

bool is_relevant_class(int cls_id, int person_class_id,
                       const std::vector<int>& object_classes) {
  if (cls_id == person_class_id) {
    return true;
  }
  if (!object_classes.empty()) {
    return std::find(object_classes.begin(), object_classes.end(), cls_id) !=
           object_classes.end();
  }
  // Доменные дефолтные классы abandoned-object детекции (COCO).
  switch (cls_id) {
    case 24:  // backpack
    case 26:  // handbag
    case 28:  // suitcase
    case 39:  // bottle
    case 40:  // wine glass
    case 41:  // cup
    case 45:  // bowl
      return true;
    default:
      return false;
  }
}

std::vector<Detection> apply_frame_filter(const std::vector<Detection>& in,
                                           int frame_w, int frame_h,
                                           int person_class_id,
                                           const FrameFilterConfig& ff) {
  std::vector<Detection> out;
  out.reserve(in.size());

  const float fw = static_cast<float>(std::max(1, frame_w));
  const float fh = static_cast<float>(std::max(1, frame_h));
  const float frame_area = fw * fh;

  for (const auto& d : in) {
    const float x1 = d.bbox.x1;
    const float y1 = d.bbox.y1;
    const float x2 = d.bbox.x2;
    const float y2 = d.bbox.y2;
    const float bw = x2 - x1;
    const float bh = y2 - y1;
    if (!(bw > 0.0f && bh > 0.0f)) continue;
    if (bw < static_cast<float>(ff.min_box_px) ||
        bh < static_cast<float>(ff.min_box_px)) {
      continue;
    }
    const float ar = bw / std::max(1.0f, bh);
    if (ar < ff.min_aspect || ar > ff.max_aspect) continue;
    if (ff.border_px > 0) {
      const float b = static_cast<float>(ff.border_px);
      if (x1 < b || y1 < b || x2 > (fw - b) || y2 > (fh - b)) continue;
    }
    const float area_ratio = (bw * bh) / std::max(1.0f, frame_area);
    if (area_ratio < ff.min_area_ratio || area_ratio > ff.max_area_ratio) continue;
    const float conf_thr_base =
        (d.class_id == person_class_id) ? ff.min_conf_person : ff.min_conf;
    // COCO handbag (26): сверху путается с цилиндром — умеренный минимум поверх базы.
    float conf_thr =
        (d.class_id == 26) ? std::max(conf_thr_base, 0.52f) : conf_thr_base;
    if (d.class_id == 24) {
      conf_thr = std::max(conf_thr, 0.48f);
    }
    if (d.class_id == 28) {
      conf_thr = std::max(conf_thr, 0.48f);
    }
    // COCO bottle (39): слабый скор на дальних ракурсах — чуть ниже базы, но не ниже пола.
    if (d.class_id == 39) {
      conf_thr = std::max(0.28f, conf_thr_base - 0.02f);
    }
    if (d.confidence < conf_thr) continue;

    out.push_back(d);
  }

  if (static_cast<int>(out.size()) > ff.max_detections) {
    std::sort(out.begin(), out.end(),
              [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
              });
    out.resize(static_cast<std::size_t>(ff.max_detections));
  }

  return out;
}

}  // namespace integra
