#include "integra/frame_filter.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace integra {

namespace {

struct Dsu {
  std::vector<int> parent;
  explicit Dsu(int n) : parent(static_cast<std::size_t>(n)) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  int find(int x) {
    if (parent[static_cast<std::size_t>(x)] != x) {
      parent[static_cast<std::size_t>(x)] = find(parent[static_cast<std::size_t>(x)]);
    }
    return parent[static_cast<std::size_t>(x)];
  }
  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) {
      parent[static_cast<std::size_t>(b)] = a;
    }
  }
};

/// Геометрия «в ряд / стопка» для бутылки+посуды (bottle_loose — мягче зазоры, если есть класс 39).
static bool tableware_row_stack_geometry(const BBoxXYXY& A, const BBoxXYXY& B,
                                         bool bottle_loose) {
  const float wa = std::max(1.f, A.x2 - A.x1);
  const float ha = std::max(1.f, A.y2 - A.y1);
  const float wb = std::max(1.f, B.x2 - B.x1);
  const float hb = std::max(1.f, B.y2 - B.y1);
  const float minw = std::min(wa, wb);
  const float minh = std::min(ha, hb);

  const float y1o = std::max(A.y1, B.y1);
  const float y2o = std::min(A.y2, B.y2);
  const float yf = std::max(0.f, y2o - y1o) / minh;

  const float x1o = std::max(A.x1, B.x1);
  const float x2o = std::min(A.x2, B.x2);
  const float xf = std::max(0.f, x2o - x1o) / minw;

  float wg = 0.f;
  if (A.x2 < B.x1) {
    wg = B.x1 - A.x2;
  } else if (B.x2 < A.x1) {
    wg = A.x1 - B.x2;
  }

  float hg = 0.f;
  if (A.y2 < B.y1) {
    hg = B.y1 - A.y2;
  } else if (B.y2 < A.y1) {
    hg = A.y1 - B.y2;
  }

  const float row_wg_max = bottle_loose ? 0.46f : 0.22f;
  const float y_need = bottle_loose ? 0.26f : 0.38f;
  if (yf >= y_need && wg <= row_wg_max * minw) {
    return true;
  }
  const float stack_hg_max = bottle_loose ? 0.40f : 0.23f;
  const float x_need = bottle_loose ? 0.26f : 0.38f;
  if (xf >= x_need && hg <= stack_hg_max * minh) {
    return true;
  }
  return false;
}

/// То же по геометрии, но разные классы 39/40/41/45 (bottle vs cup на одной группе).
bool tableware_adjacent_any_class(const Detection& a, const Detection& b) {
  if (a.class_id == 39 && b.class_id == 39) {
    return false;
  }
  return tableware_row_stack_geometry(a.bbox, b.bbox,
                                      a.class_id == 39 || b.class_id == 39);
}

bool tableware_cross_merge_pair(const Detection& a, const Detection& b, float iou_threshold) {
  const float iou = iou_xyxy(a.bbox, b.bbox);
  if (iou >= iou_threshold) {
    return true;
  }
  return tableware_adjacent_any_class(a, b);
}

bool same_class_merge_pair(const Detection& a, const Detection& b, float iou_threshold) {
  if (a.class_id != b.class_id) {
    return false;
  }
  const float iou = iou_xyxy(a.bbox, b.bbox);
  // Только сильное перекрытие (дубликаты одной детекции). Без геометрии «в ряд» —
  // как основной поток VisionOCR `detector.py` (склейка там только IoU ≥ merge_iou_threshold
  // при подмешивании ROI, не между соседними физическими объектами).
  return iou >= iou_threshold;
}

/// Касание границы кадра в пределах relax px (как `border_relax_px` в Python detector).
bool touches_conf_relax_border(const BBoxXYXY& b, int fw, int fh, int relax_px) {
  if (relax_px <= 0 || fw <= 0 || fh <= 0) {
    return false;
  }
  const float br = static_cast<float>(relax_px);
  const float wf = static_cast<float>(fw);
  const float hf = static_cast<float>(fh);
  return b.x1 <= br || b.y1 <= br || b.x2 >= (wf - br) || b.y2 >= (hf - br);
}

/// Минимальный conf по вертикали + послабление у края (логика VisionOCR `detector.py`).
float regional_conf_threshold(const FrameFilterConfig& ff,
                              int cls_id,
                              int person_class_id,
                              float cy,
                              int fw,
                              int fh,
                              const BBoxXYXY& bbox) {
  const float fh_f = std::max(1.f, static_cast<float>(fh));
  const float upper_cut = fh_f * ff.upper_region_y_ratio;
  const float bottom_cut = fh_f * ff.bottom_region_y_ratio;
  float min_c = ff.min_conf_lower;
  if (cy <= upper_cut) {
    min_c = ff.min_conf_upper;
  } else if (cy >= bottom_cut) {
    min_c = ff.min_conf_bottom;
  }
  const bool touch = touches_conf_relax_border(bbox, fw, fh, ff.conf_border_relax_px);
  if (touch) {
    min_c = std::min(min_c, ff.min_conf_border);
  }
  if (cls_id == person_class_id && touch) {
    min_c = std::min(min_c, ff.person_min_conf_border);
  }
  return min_c;
}

float class_min_override_for(int cls_id, const FrameFilterConfig& ff) {
  for (int i = 0; i < ff.class_min_conf_count && i < FrameFilterConfig::kClassMinConfMax; ++i) {
    if (ff.class_min_conf_ids[i] == cls_id) {
      return ff.class_min_conf_thresholds[i];
    }
  }
  return 0.f;
}

}  // namespace

void FrameFilterConfig::tune_from_postprocess(float conf_threshold) {
  // Важно: YOLO-постпроцесс уже отфильтровал по `conf_threshold`.
  // Объекты: не поднимаем min_conf выше модели (двойной порог).
  // Person: чуть выше conf модели — баланс FP на мебель / пропуски дальних людей.
  const float c = std::clamp(conf_threshold, 0.01f, 0.99f);
  min_conf = c;
  min_conf_person = std::min(0.99f, c + 0.03f);
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

    // Багаж (24/26/28): микро-боксы на текстуре коробки/кромке — почти всегда FP.
    constexpr int kLuggageMinSidePx = 38;
    if ((d.class_id == 24 || d.class_id == 26 || d.class_id == 28) &&
        (bw < static_cast<float>(kLuggageMinSidePx) || bh < static_cast<float>(kLuggageMinSidePx))) {
      continue;
    }

    // Опциональная маска (статичная камера): зона приёмной / хаотичный стол.
    if (d.class_id != person_class_id && ff.ignore_det_norm_x2 > ff.ignore_det_norm_x1 &&
        ff.ignore_det_norm_y2 > ff.ignore_det_norm_y1) {
      float cx = 0.f;
      float cy = 0.f;
      centroid_xyxy(d.bbox, &cx, &cy);
      const float nx = cx / fw;
      const float ny = cy / fh;
      if (nx >= ff.ignore_det_norm_x1 && nx <= ff.ignore_det_norm_x2 && ny >= ff.ignore_det_norm_y1 &&
          ny <= ff.ignore_det_norm_y2) {
        continue;
      }
    }

    // COCO person (0): геометрия типичного силуэта — режет «person» на столы, стены,
    // крупные горизонтальные пятна (низкая уверенность модели не спасает без этого).
    if (d.class_id == person_class_id) {
      const float ph_ar = bw / std::max(1.0f, bh);
      const float area_ratio_p = (bw * bh) / std::max(1.0f, frame_area);
      if (ph_ar > 1.58f) {
        continue;
      }
      if (ph_ar < 0.08f) {
        continue;
      }
      if (area_ratio_p > 0.25f) {
        continue;
      }
    }

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

    // Верх площади по классу: гигантские FP (стойка целиком, стена, ковёр как «сумка»).
    if (d.class_id != person_class_id) {
      if (d.class_id == 39 && area_ratio > 0.048f) {
        continue;
      }
      if ((d.class_id == 40 || d.class_id == 41 || d.class_id == 45) && area_ratio > 0.036f) {
        continue;
      }
      if ((d.class_id == 24 || d.class_id == 26 || d.class_id == 28) && area_ratio > 0.095f) {
        continue;
      }
    }

    // Центр верха (проём двери): слабый багаж на фоне — FP; не трогаем посуду и боковые зоны
    // (стол с бутылками часто в верхней части кадра при высоком угле камеры).
    if (d.class_id != person_class_id &&
        (d.class_id == 24 || d.class_id == 26 || d.class_id == 28)) {
      float ucx = 0.f;
      float ucy = 0.f;
      centroid_xyxy(d.bbox, &ucx, &ucy);
      const float unx = ucx / fw;
      const float uny = ucy / fh;
      if (uny < 0.36f && unx > 0.34f && unx < 0.66f && area_ratio < 0.014f &&
          d.confidence < 0.58f) {
        continue;
      }
    }

    // Центр верха: бутылка/стакан на фоне проёма (лестница, стеллаж) — частый FP TRT;
    // боковой стол с канистрами не попадает (nx обычно < 0.38).
    // Стаканы/бокалы у «верхнего центра» (монитор/блик) — не режем COCO bottle (39): там
    // реальные канистры на стойке под углом камеры.
    if (d.class_id != person_class_id &&
        (d.class_id == 40 || d.class_id == 41 || d.class_id == 45)) {
      float tcx = 0.f;
      float tcy = 0.f;
      centroid_xyxy(d.bbox, &tcx, &tcy);
      const float tnx = tcx / fw;
      const float tny = tcy / fh;
      if (tny < 0.38f && tnx > 0.40f && tnx < 0.66f && area_ratio < 0.017f &&
          d.confidence < 0.62f) {
        continue;
      }
    }

    // Блик на полу у нижней кромки: широкая низкая полоска, малая доля кадра.
    if (d.class_id != person_class_id && frame_h > 0) {
      const float fh2 = static_cast<float>(frame_h);
      const float barfloor = bw / std::max(1.0f, bh);
      if (y2 > fh2 * 0.902f && bh < 62.f && bw > 85.f && barfloor > 1.82f &&
          area_ratio < 0.0058f) {
        continue;
      }
      float tcx = 0.f;
      float tcy = 0.f;
      centroid_xyxy(d.bbox, &tcx, &tcy);
      const float ny = tcy / fh;
      const float bar = bw / std::max(1.0f, bh);
      // Широкое низкое пятно в нижней зоне кадра — блик плитки / отражение двери.
      if (ny > 0.56f && y2 > fh2 * 0.40f && bh < fh2 * 0.17f && bar > 1.12f &&
          area_ratio < 0.0145f && area_ratio > 0.00038f) {
        continue;
      }
    }

    // COCO bottle (39): без «ветвления» standing/lying — оно ломало реальные канистры под углом.
    // Только явные FP: ультра-узкий штырь, микро-пятно, плоский телефон, огромная панель.
    if (d.class_id == 39) {
      const float mx = std::max(bw, bh);
      const float mn = std::max(1.0f, std::min(bw, bh));
      if (mx / mn > 6.8f) {
        continue;
      }
      const float area_px = bw * bh;
      // Слабые/мелкие канистры на столе (высокий ракурс) — не отсекать агрессивно.
      if (area_px < 300.f) {
        continue;
      }
      const float bar = bw / std::max(1.0f, bh);
      // «Плоская» бутылка: только совсем вытянутый бокс без площади (телефон/блик).
      if (bar > 1.55f && area_px < 22000.f) {
        continue;
      }
      if (area_ratio > 0.019f && bar > 0.70f && area_px > 85000.f) {
        continue;
      }
      const float nx = (x1 + x2) * 0.5f / fw;
      const float ny = (y1 + y2) * 0.5f / fh;
      const float elong = mx / mn;
      // Вертикальный блик на стекле у двери (центр кадра): не канистра.
      if (elong >= 2.6f && elong <= 9.5f && nx > 0.34f && nx < 0.66f && ny < 0.62f &&
          area_ratio < 0.028f && d.confidence < 0.62f) {
        continue;
      }
      // Слабая «бутылка» у левого верха — симметрия правого правила (блик/камера).
      if (nx < 0.37f && ny < 0.54f && area_ratio < 0.009f && d.confidence < 0.58f) {
        continue;
      }
      // Слабая «бутылка» у правого верха — часто монитор/блик стойки, не ёмкость.
      if (nx > 0.63f && ny < 0.54f && area_ratio < 0.009f && d.confidence < 0.58f) {
        continue;
      }
    }

    // bottle / wine glass / cup / bowl: «щели» и тонкие полоски на стекле/телефоне (правая сторона).
    if (d.class_id == 39 || d.class_id == 40 || d.class_id == 41 || d.class_id == 45) {
      const float mx = std::max(bw, bh);
      const float mn = std::max(1.0f, std::min(bw, bh));
      if (mn < 24.f && mx / mn > 3.0f) {
        continue;
      }
    }

    // Cup / bowl: только очень «пластинчатые» крупные боксы (монитор как миска).
    if (d.class_id == 41 || d.class_id == 45) {
      const float mx = std::max(bw, bh);
      const float mn = std::max(1.0f, std::min(bw, bh));
      const float area_ratio_cb = (bw * bh) / std::max(1.0f, frame_area);
      if (mx / mn > 3.35f && area_ratio_cb > 0.010f) {
        continue;
      }
    }

    // Handbag: очень вытянутый прямоугольник — часто клавиатура/монитор в профиль.
    if (d.class_id == 26) {
      const float mx = std::max(bw, bh);
      const float mn = std::max(1.0f, std::min(bw, bh));
      if (mx / mn > 6.0f) {
        continue;
      }
    }

    float conf_thr = 0.f;
    if (ff.use_regional_class_conf) {
      float rcx = 0.f;
      float rcy = 0.f;
      centroid_xyxy(d.bbox, &rcx, &rcy);
      conf_thr = regional_conf_threshold(ff, d.class_id, person_class_id, rcy, frame_w, frame_h,
                                           d.bbox);
      // Региональные пороги совпадают с посудой/полом — для сумки/чемодана это слишком мягко
      // (YOLO цепляет угол коробки, таз → handbag ~0.5). Держим отдельный пол для багажа.
      if (d.class_id == 26) {
        conf_thr = std::max(conf_thr, 0.56f);
      } else if (d.class_id == 28) {
        conf_thr = std::max(conf_thr, 0.55f);
      } else if (d.class_id == 24) {
        conf_thr = std::max(conf_thr, 0.52f);
      }
    } else {
      const float conf_thr_base =
          (d.class_id == person_class_id) ? ff.min_conf_person : ff.min_conf;
      // COCO handbag (26): сверху путается с цилиндром — умеренный минимум поверх базы.
      conf_thr = (d.class_id == 26) ? std::max(conf_thr_base, 0.52f) : conf_thr_base;
      if (d.class_id == 24) {
        conf_thr = std::max(conf_thr, 0.48f);
      }
      if (d.class_id == 28) {
        conf_thr = std::max(conf_thr, 0.48f);
      }
      if (d.class_id == 40) {
        conf_thr = std::max(conf_thr, 0.42f);
      }
      if (d.class_id == 39) {
        conf_thr = std::max(0.43f, conf_thr_base);
      }
      if (d.class_id == 41 || d.class_id == 45) {
        conf_thr = std::max(conf_thr, 0.42f);
      }
      {
        const float ap = bw * bh;
        if ((d.class_id == 40 || d.class_id == 41) && ap < 14000.f) {
          conf_thr = std::max(conf_thr, 0.50f);
        }
      }
    }
    const float cls_ov = class_min_override_for(d.class_id, ff);
    if (cls_ov > 0.f) {
      conf_thr = std::max(conf_thr, cls_ov);
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

void merge_cross_class_by_iou(std::vector<Detection>& dets, float iou_threshold) {
  if (dets.size() < 2) {
    return;
  }
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  std::vector<Detection> keep;
  keep.reserve(dets.size());
  for (const auto& d : dets) {
    bool ok = true;
    for (const auto& k : keep) {
      if (iou_xyxy(d.bbox, k.bbox) >= iou_threshold) {
        ok = false;
        break;
      }
    }
    if (ok) {
      keep.push_back(d);
    }
  }
  dets.swap(keep);
}

void merge_cross_class_objects_only(std::vector<Detection>& dets, int person_class_id,
                                    float iou_threshold) {
  std::vector<Detection> persons;
  std::vector<Detection> objects;
  persons.reserve(dets.size());
  objects.reserve(dets.size());
  for (auto& d : dets) {
    if (d.class_id == person_class_id) {
      persons.push_back(std::move(d));
    } else {
      objects.push_back(std::move(d));
    }
  }
  merge_cross_class_by_iou(objects, iou_threshold);
  dets.clear();
  dets.reserve(persons.size() + objects.size());
  for (auto& p : persons) {
    dets.push_back(std::move(p));
  }
  for (auto& o : objects) {
    dets.push_back(std::move(o));
  }
}

void merge_luggage_cross_class_by_iou(std::vector<Detection>& dets, int person_class_id,
                                      float iou_threshold) {
  std::vector<Detection> persons;
  std::vector<Detection> luggage;
  std::vector<Detection> other;
  persons.reserve(dets.size());
  luggage.reserve(8);
  other.reserve(dets.size());
  for (auto& d : dets) {
    if (d.class_id == person_class_id) {
      persons.push_back(std::move(d));
    } else if (d.class_id == 24 || d.class_id == 26 || d.class_id == 28) {
      luggage.push_back(std::move(d));
    } else {
      other.push_back(std::move(d));
    }
  }
  merge_cross_class_by_iou(luggage, iou_threshold);
  dets.clear();
  dets.reserve(persons.size() + luggage.size() + other.size());
  for (auto& p : persons) {
    dets.push_back(std::move(p));
  }
  for (auto& l : luggage) {
    dets.push_back(std::move(l));
  }
  for (auto& o : other) {
    dets.push_back(std::move(o));
  }
}

void merge_tableware_cross_iou(std::vector<Detection>& dets, int person_class_id,
                               float iou_threshold) {
  std::vector<Detection> persons;
  std::vector<Detection> tableware;
  std::vector<Detection> other;
  persons.reserve(dets.size());
  tableware.reserve(dets.size());
  other.reserve(dets.size());
  for (auto& d : dets) {
    if (d.class_id == person_class_id) {
      persons.push_back(std::move(d));
    } else if (d.class_id == 39 || d.class_id == 40 || d.class_id == 41 || d.class_id == 45) {
      tableware.push_back(std::move(d));
    } else {
      other.push_back(std::move(d));
    }
  }
  if (tableware.size() > 1) {
    const int n = static_cast<int>(tableware.size());
    Dsu dsu(n);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (tableware_cross_merge_pair(tableware[static_cast<std::size_t>(i)],
                                        tableware[static_cast<std::size_t>(j)], iou_threshold)) {
          dsu.unite(i, j);
        }
      }
    }
    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      groups[dsu.find(i)].push_back(i);
    }
    std::vector<Detection> merged_tw;
    merged_tw.reserve(groups.size());
    for (const auto& kv : groups) {
      const std::vector<int>& idx = kv.second;
      int best = idx[0];
      for (int id : idx) {
        if (tableware[static_cast<std::size_t>(id)].confidence >
            tableware[static_cast<std::size_t>(best)].confidence) {
          best = id;
        }
      }
      merged_tw.push_back(std::move(tableware[static_cast<std::size_t>(best)]));
    }
    tableware.swap(merged_tw);
  }
  dets.clear();
  dets.reserve(persons.size() + tableware.size() + other.size());
  for (auto& p : persons) {
    dets.push_back(std::move(p));
  }
  for (auto& t : tableware) {
    dets.push_back(std::move(t));
  }
  for (auto& o : other) {
    dets.push_back(std::move(o));
  }
}

void merge_same_class_objects_only(std::vector<Detection>& dets, int person_class_id,
                                   float iou_threshold) {
  std::vector<Detection> persons;
  std::vector<Detection> objects;
  persons.reserve(dets.size());
  objects.reserve(dets.size());
  for (auto& d : dets) {
    if (d.class_id == person_class_id) {
      persons.push_back(std::move(d));
    } else {
      objects.push_back(std::move(d));
    }
  }
  if (objects.size() < 2) {
    dets.clear();
    dets.reserve(persons.size() + objects.size());
    for (auto& p : persons) {
      dets.push_back(std::move(p));
    }
    for (auto& o : objects) {
      dets.push_back(std::move(o));
    }
    return;
  }

  const int n = static_cast<int>(objects.size());
  Dsu dsu(n);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (same_class_merge_pair(objects[static_cast<std::size_t>(i)],
                                objects[static_cast<std::size_t>(j)], iou_threshold)) {
        dsu.unite(i, j);
      }
    }
  }

  std::unordered_map<int, std::vector<int>> groups;
  groups.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    groups[dsu.find(i)].push_back(i);
  }

  std::vector<Detection> merged;
  merged.reserve(groups.size());
  for (const auto& kv : groups) {
    const std::vector<int>& idx = kv.second;
    int best = idx[0];
    for (int id : idx) {
      if (objects[static_cast<std::size_t>(id)].confidence >
          objects[static_cast<std::size_t>(best)].confidence) {
        best = id;
      }
    }
    merged.push_back(std::move(objects[static_cast<std::size_t>(best)]));
  }
  objects.swap(merged);

  dets.clear();
  dets.reserve(persons.size() + objects.size());
  for (auto& p : persons) {
    dets.push_back(std::move(p));
  }
  for (auto& o : objects) {
    dets.push_back(std::move(o));
  }
}

void suppress_duplicate_bottles_by_iou(std::vector<Detection>& dets, int person_class_id,
                                       float iou_threshold) {
  std::vector<Detection> persons;
  std::vector<Detection> bottles;
  std::vector<Detection> rest;
  persons.reserve(dets.size());
  bottles.reserve(dets.size());
  rest.reserve(dets.size());
  for (auto& d : dets) {
    if (d.class_id == person_class_id) {
      persons.push_back(std::move(d));
    } else if (d.class_id == 39) {
      bottles.push_back(std::move(d));
    } else {
      rest.push_back(std::move(d));
    }
  }
  if (bottles.size() > 1U) {
    merge_cross_class_by_iou(bottles, iou_threshold);
  }
  dets.clear();
  dets.reserve(persons.size() + bottles.size() + rest.size());
  for (auto& p : persons) {
    dets.push_back(std::move(p));
  }
  for (auto& b : bottles) {
    dets.push_back(std::move(b));
  }
  for (auto& r : rest) {
    dets.push_back(std::move(r));
  }
}

}  // namespace integra
