#include "integra/yolo_postprocess.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace integra {

namespace {

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// COCO 80 (индекс = class_id).
static const std::array<const char*, 80> kCoco80 = {
    "person",        "bicycle",      "car",           "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",          "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",          "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",      "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",       "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",         "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",      "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",         "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",            "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",     "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",         "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

}  // namespace

bool yolo_output_channel_first(std::int64_t d1, std::int64_t d2) {
  if (d1 <= 0 || d2 <= 0) {
    return true;
  }
  if (d1 < 512 && d2 >= d1 * 8) {
    return true;
  }
  if (d2 < 512 && d1 >= d2 * 8) {
    return false;
  }
  return d1 < d2;
}

std::string coco80_class_name(int cls_id) {
  if (cls_id < 0 || cls_id >= 80) {
    return {};
  }
  return std::string(kCoco80[static_cast<std::size_t>(cls_id)]);
}

void decode_yolov8_chw(const float* chw, const PostprocessParams& pp, float input_w, float input_h,
                       std::vector<Detection>& out) {
  out.clear();
  const int nc = pp.num_classes;
  const int n = pp.num_anchors;
  if (nc <= 0 || n <= 0 || chw == nullptr) {
    return;
  }

  out.reserve(static_cast<std::size_t>(n) / 10);
  for (int i = 0; i < n; ++i) {
    const float cx = chw[0 * n + i];
    const float cy = chw[1 * n + i];
    const float bw = chw[2 * n + i];
    const float bh = chw[3 * n + i];

    int best_c = 0;
    float best_s = 0.f;
    for (int c = 0; c < nc; ++c) {
      const float s = sigmoid(chw[(4 + c) * n + i]);
      if (s > best_s) {
        best_s = s;
        best_c = c;
      }
    }
    if (best_s < pp.conf_threshold) {
      continue;
    }

    const float x1 = cx - 0.5f * bw;
    const float y1 = cy - 0.5f * bh;
    const float x2 = cx + 0.5f * bw;
    const float y2 = cy + 0.5f * bh;

    Detection d;
    d.track_id = -1;
    d.class_id = best_c;
    d.cls_name = coco80_class_name(best_c);
    if (d.cls_name.empty()) {
      d.cls_name = "cls_" + std::to_string(best_c);
    }
    d.confidence = best_s;
    d.bbox = {x1, y1, x2, y2};
    (void)input_w;
    (void)input_h;
    out.push_back(d);
  }
}

void decode_yolov8_flat(const float* data, int dim1, int dim2, bool dim1_is_channel,
                        PostprocessParams pp, float input_w, float input_h,
                        std::vector<float>& scratch, std::vector<Detection>& out) {
  if (data == nullptr || dim1 <= 0 || dim2 <= 0) {
    out.clear();
    return;
  }
  if (dim1_is_channel) {
    const int C = dim1;
    const int N = dim2;
    if (C < 4) {
      out.clear();
      return;
    }
    pp.num_classes = C - 4;
    pp.num_anchors = N;
    decode_yolov8_chw(data, pp, input_w, input_h, out);
    return;
  }
  const int N = dim1;
  const int C = dim2;
  if (C < 4) {
    out.clear();
    return;
  }
  pp.num_classes = C - 4;
  pp.num_anchors = N;
  scratch.resize(static_cast<std::size_t>(C) * static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    for (int c = 0; c < C; ++c) {
      scratch[static_cast<std::size_t>(c) * static_cast<std::size_t>(N) + static_cast<std::size_t>(i)] =
          data[static_cast<std::size_t>(i) * static_cast<std::size_t>(C) + static_cast<std::size_t>(c)];
    }
  }
  decode_yolov8_chw(scratch.data(), pp, input_w, input_h, out);
}

void nms_greedy_xyxy(std::vector<Detection>& dets, float iou_threshold) {
  if (dets.empty()) {
    return;
  }
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

  std::vector<Detection> keep;
  keep.reserve(dets.size());
  for (const auto& d : dets) {
    bool ok = true;
    for (const auto& k : keep) {
      if (d.class_id != k.class_id) {
        continue;
      }
      if (iou_xyxy(d.bbox, k.bbox) > iou_threshold) {
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

bool selftest_yolo_postprocess() {
  PostprocessParams pp;
  pp.num_anchors = 10;
  pp.num_classes = 80;
  pp.conf_threshold = 0.25f;
  pp.nms_iou_threshold = 0.45f;

  std::vector<float> buf(static_cast<std::size_t>(4 + pp.num_classes) *
                             static_cast<std::size_t>(pp.num_anchors),
                         0.f);
  const int i = 5;
  buf[static_cast<std::size_t>(0 * pp.num_anchors + i)] = 320.f;
  buf[static_cast<std::size_t>(1 * pp.num_anchors + i)] = 320.f;
  buf[static_cast<std::size_t>(2 * pp.num_anchors + i)] = 100.f;
  buf[static_cast<std::size_t>(3 * pp.num_anchors + i)] = 100.f;
  buf[static_cast<std::size_t>((4 + 39) * pp.num_anchors + i)] = 5.f;

  std::vector<Detection> out;
  decode_yolov8_chw(buf.data(), pp, 640.f, 640.f, out);
  nms_greedy_xyxy(out, pp.nms_iou_threshold);
  return out.size() == 1 && out[0].class_id == 39 && out[0].confidence > 0.25f;
}

}  // namespace integra
