#include "integra/frame_diff_detector.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace integra {

namespace {
// Длинная сторона рабочего кадра стадии diff. Полный кадр (Full HD) сюда не
// доходит: всё считается на маленьком grayscale — это и снимает нагрузку, и
// гасит пиксельный шум (меньше ложных регионов).
constexpr int kWorkLongEdge = 480;
// Скорость адаптации фона. Низкая → оставленный предмет долго «передний план».
constexpr double kLearningRate = 0.002;
}  // namespace

FrameDiffDetector::FrameDiffDetector(int buffer_size)
    : history_(std::max(2, buffer_size)) {
  rebuild_model();
}

void FrameDiffDetector::rebuild_model() {
  // detectShadows=false: тени не нужны и только плодят полутона/ложные пятна.
  bg_ = cv::createBackgroundSubtractorMOG2(
      /*history=*/std::max(120, history_ * 12),
      /*varThreshold=*/16.0,
      /*detectShadows=*/false);
}

void FrameDiffDetector::reset() {
  rebuild_model();
  last_small_.release();
  inv_scale_ = 1.0;
}

cv::Mat FrameDiffDetector::get_last_frame() const { return last_small_.clone(); }

std::vector<ChangedRegion> FrameDiffDetector::process_frame(
    const cv::Mat& bgr_frame, float /*pixel_threshold*/,
    float /*gradient_threshold*/, int min_region_area_px) {
  if (bgr_frame.empty() || !bg_) {
    return {};
  }

  // 1) Уменьшаем кадр. Вся стадия обнаружения работает на маленьком grayscale.
  const int long_edge = std::max(bgr_frame.cols, bgr_frame.rows);
  const double scale =
      long_edge > kWorkLongEdge
          ? static_cast<double>(kWorkLongEdge) / static_cast<double>(long_edge)
          : 1.0;
  inv_scale_ = scale > 1e-6 ? 1.0 / scale : 1.0;

  cv::Mat small;
  if (scale < 1.0) {
    cv::resize(bgr_frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
  } else {
    small = bgr_frame;
  }

  cv::Mat gray;
  if (small.channels() == 3) {
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = small;
  }
  // Лёгкое сглаживание — гасит шум сенсора/JPEG.
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
  gray.copyTo(last_small_);

  // 2) Модель фона → маска переднего плана (векторно, внутри OpenCV).
  cv::Mat fg;
  bg_->apply(gray, fg, kLearningRate);

  // 3) Бинаризация + морфология (чистим одиночные пиксели, закрываем дыры).
  cv::threshold(fg, fg, 200, 255, cv::THRESH_BINARY);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(fg, fg, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(fg, fg, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

  // 4) Контуры → регионы. Порог площади масштабируем под уменьшенный кадр.
  const double area_scale = scale * scale;
  const double min_area_small = std::max(
      4.0, static_cast<double>(std::max(1, min_region_area_px)) * area_scale);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(fg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<ChangedRegion> result;
  result.reserve(contours.size());
  for (const auto& contour : contours) {
    const double a = cv::contourArea(contour);
    if (a < min_area_small) {
      continue;
    }
    const cv::Rect r = cv::boundingRect(contour);
    // Плотность заполнения bbox: тонкие «рваные» следы прохода людей → низкая.
    const double fill = a / std::max(1.0, static_cast<double>(r.area()));

    ChangedRegion region;
    region.bbox = {static_cast<float>(r.x * inv_scale_),
                   static_cast<float>(r.y * inv_scale_),
                   static_cast<float>((r.x + r.width) * inv_scale_),
                   static_cast<float>((r.y + r.height) * inv_scale_)};
    const float score = static_cast<float>(std::min(1.0, fill) * 255.0);
    region.pixel_change = score;
    region.gradient_change = score;
    region.combined_score = score;
    result.push_back(region);
  }

  return merge_regions(result, 0.3f);
}

std::vector<ChangedRegion> FrameDiffDetector::merge_regions(
    const std::vector<ChangedRegion>& regions, float iou_threshold) {
  if (regions.empty()) {
    return {};
  }

  std::vector<ChangedRegion> merged = regions;
  std::vector<bool> used(merged.size(), false);
  std::vector<ChangedRegion> result;

  // Сортируем по скору (большие изменения первыми).
  std::sort(merged.begin(), merged.end(),
            [](const ChangedRegion& a, const ChangedRegion& b) {
              return a.combined_score > b.combined_score;
            });

  for (size_t i = 0; i < merged.size(); ++i) {
    if (used[i]) {
      continue;
    }
    ChangedRegion current = merged[i];
    used[i] = true;

    for (size_t j = i + 1; j < merged.size(); ++j) {
      if (used[j]) {
        continue;
      }
      const float x1_max = std::max(current.bbox.x1, merged[j].bbox.x1);
      const float y1_max = std::max(current.bbox.y1, merged[j].bbox.y1);
      const float x2_min = std::min(current.bbox.x2, merged[j].bbox.x2);
      const float y2_min = std::min(current.bbox.y2, merged[j].bbox.y2);

      if (x2_min > x1_max && y2_min > y1_max) {
        const float intersect = (x2_min - x1_max) * (y2_min - y1_max);
        const float area1 = (current.bbox.x2 - current.bbox.x1) *
                            (current.bbox.y2 - current.bbox.y1);
        const float area2 = (merged[j].bbox.x2 - merged[j].bbox.x1) *
                            (merged[j].bbox.y2 - merged[j].bbox.y1);
        const float iou = intersect / std::max(1e-6f, area1 + area2 - intersect);

        if (iou > iou_threshold) {
          current.bbox.x1 = std::min(current.bbox.x1, merged[j].bbox.x1);
          current.bbox.y1 = std::min(current.bbox.y1, merged[j].bbox.y1);
          current.bbox.x2 = std::max(current.bbox.x2, merged[j].bbox.x2);
          current.bbox.y2 = std::max(current.bbox.y2, merged[j].bbox.y2);
          current.pixel_change =
              std::max(current.pixel_change, merged[j].pixel_change);
          current.gradient_change =
              std::max(current.gradient_change, merged[j].gradient_change);
          current.combined_score =
              std::max(current.combined_score, merged[j].combined_score);
          used[j] = true;
        }
      }
    }
    result.push_back(current);
  }

  return result;
}

}  // namespace integra
