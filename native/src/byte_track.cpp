#include "integra/byte_track.hpp"

#include "integra/geom.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace integra {

namespace {

// ---------------------------------------------------------------------------
// Kalman filter (constant-velocity), модель xyah как в оригинальном ByteTrack.
//   state  = [cx, cy, a, h, vcx, vcy, va, vh]   (8)
//   measure= [cx, cy, a, h]                      (4)
// H выбирает первые 4 компоненты, F = I + сдвиг скорости (dt = 1) — поэтому
// проекции/обновления считаются по подблокам ковариации без общих умножений на H.
// ---------------------------------------------------------------------------
using Vec8 = std::array<double, 8>;
using Mat8 = std::array<std::array<double, 8>, 8>;
using Vec4 = std::array<double, 4>;
using Mat4 = std::array<std::array<double, 4>, 4>;

constexpr double kStdWeightPos = 1.0 / 20.0;
constexpr double kStdWeightVel = 1.0 / 160.0;

struct KalmanState {
  Vec8 mean{};
  Mat8 cov{};
};

void kf_initiate(KalmanState& s, const Vec4& meas) {
  s.mean = {meas[0], meas[1], meas[2], meas[3], 0, 0, 0, 0};
  const double h = meas[3];
  const std::array<double, 8> std = {
      2 * kStdWeightPos * h, 2 * kStdWeightPos * h, 1e-2, 2 * kStdWeightPos * h,
      10 * kStdWeightVel * h, 10 * kStdWeightVel * h, 1e-5, 10 * kStdWeightVel * h};
  s.cov = Mat8{};
  for (int i = 0; i < 8; ++i) {
    s.cov[i][i] = std[i] * std[i];
  }
}

void kf_predict(KalmanState& s) {
  const double h = s.mean[3];
  // mean = F * mean  (dt = 1): позиция += скорость.
  for (int i = 0; i < 4; ++i) {
    s.mean[i] += s.mean[i + 4];
  }
  // cov = F cov F^T + Q. F = I + E, где E[i][i+4] = 1 (i=0..3).
  // (F cov)[i][j] = cov[i][j] + (i<4 ? cov[i+4][j] : 0)
  Mat8 fc{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      fc[i][j] = s.cov[i][j] + (i < 4 ? s.cov[i + 4][j] : 0.0);
    }
  }
  // (fc F^T)[i][j] = fc[i][j] + (j<4 ? fc[i][j+4] : 0)
  Mat8 out{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      out[i][j] = fc[i][j] + (j < 4 ? fc[i][j + 4] : 0.0);
    }
  }
  const std::array<double, 8> qstd = {
      kStdWeightPos * h, kStdWeightPos * h, 1e-2, kStdWeightPos * h,
      kStdWeightVel * h, kStdWeightVel * h, 1e-5, kStdWeightVel * h};
  for (int i = 0; i < 8; ++i) {
    out[i][i] += qstd[i] * qstd[i];
  }
  s.cov = out;
}

bool invert4(const Mat4& in, Mat4& out) {
  // Гаусс-Жордан с частичным выбором ведущего элемента.
  std::array<std::array<double, 8>, 4> a{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) a[i][j] = in[i][j];
    a[i][4 + i] = 1.0;
  }
  for (int col = 0; col < 4; ++col) {
    int piv = col;
    double best = std::fabs(a[col][col]);
    for (int r = col + 1; r < 4; ++r) {
      if (std::fabs(a[r][col]) > best) {
        best = std::fabs(a[r][col]);
        piv = r;
      }
    }
    if (best < 1e-12) return false;
    if (piv != col) std::swap(a[piv], a[col]);
    const double d = a[col][col];
    for (int j = 0; j < 8; ++j) a[col][j] /= d;
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      const double f = a[r][col];
      if (f == 0.0) continue;
      for (int j = 0; j < 8; ++j) a[r][j] -= f * a[col][j];
    }
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out[i][j] = a[i][4 + j];
  return true;
}

void kf_update(KalmanState& s, const Vec4& meas) {
  const double h = s.mean[3];
  // project: proj_mean = mean[0..3]; proj_cov = cov[0..3][0..3] + R.
  const std::array<double, 4> rstd = {kStdWeightPos * h, kStdWeightPos * h, 1e-1,
                                      kStdWeightPos * h};
  Mat4 proj{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) proj[i][j] = s.cov[i][j];
  for (int i = 0; i < 4; ++i) proj[i][i] += rstd[i] * rstd[i];

  Mat4 inv{};
  if (!invert4(proj, inv)) {
    return;  // вырожденная ковариация — пропускаем обновление
  }

  // K = cov * H^T * inv = (первые 4 столбца cov) * inv  -> 8x4
  std::array<Vec4, 8> K{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) {
      double acc = 0.0;
      for (int k = 0; k < 4; ++k) acc += s.cov[i][k] * inv[k][j];
      K[i][j] = acc;
    }
  }
  // innovation = meas - proj_mean
  Vec4 innov{meas[0] - s.mean[0], meas[1] - s.mean[1], meas[2] - s.mean[2],
             meas[3] - s.mean[3]};
  for (int i = 0; i < 8; ++i) {
    double acc = 0.0;
    for (int j = 0; j < 4; ++j) acc += K[i][j] * innov[j];
    s.mean[i] += acc;
  }
  // cov = cov - K * (H cov), где H cov = первые 4 строки cov (4x8).
  Mat8 newcov = s.cov;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (int k = 0; k < 4; ++k) acc += K[i][k] * s.cov[k][j];
      newcov[i][j] -= acc;
    }
  }
  s.cov = newcov;
}

// ---------------------------------------------------------------------------
// Конвертация bbox <-> xyah.
// ---------------------------------------------------------------------------
Vec4 xyxy_to_xyah(const BBoxXYXY& b) {
  const double w = std::max(1e-3f, b.x2 - b.x1);
  const double h = std::max(1e-3f, b.y2 - b.y1);
  return {b.x1 + w * 0.5, b.y1 + h * 0.5, w / h, h};
}

BBoxXYXY xyah_to_xyxy(const Vec8& m) {
  const double h = m[3];
  const double w = m[2] * h;
  BBoxXYXY b;
  b.x1 = static_cast<float>(m[0] - w * 0.5);
  b.y1 = static_cast<float>(m[1] - h * 0.5);
  b.x2 = static_cast<float>(m[0] + w * 0.5);
  b.y2 = static_cast<float>(m[1] + h * 0.5);
  return b;
}

enum class TS { kNew = 0, kTracked, kLost, kRemoved };

struct STrack {
  KalmanState kf;
  int track_id = 0;
  TS state = TS::kNew;
  bool is_activated = false;
  float score = 0.f;
  int class_id = 0;
  std::string cls_name;
  int frame_id = 0;
  int start_frame = 0;
  int tracklet_len = 0;
  int matched_det = -1;  // индекс детекции в текущем кадре (для записи track_id)

  BBoxXYXY bbox() const { return xyah_to_xyxy(kf.mean); }
};

// ---------------------------------------------------------------------------
// Венгерский алгоритм (Jonker–Volgenant потенциалы), прямоугольная матрица.
// rowsol[i] = назначенный столбец строки i или -1.
// ---------------------------------------------------------------------------
void solve_lap(const std::vector<std::vector<double>>& cost, int nr, int nc,
               std::vector<int>& rowsol) {
  rowsol.assign(std::max(0, nr), -1);
  if (nr == 0 || nc == 0) return;
  const int n = std::max(nr, nc);
  const double BIG = 1e9;
  std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, 0.0));
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j)
      a[i][j] = (i <= nr && j <= nc) ? cost[i - 1][j - 1] : BIG;

  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);
  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, INF);
    std::vector<char> used(n + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = -1;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  for (int j = 1; j <= n; ++j) {
    const int i = p[j];
    if (i >= 1 && i <= nr && j <= nc) rowsol[i - 1] = j - 1;
  }
}

/// Матч по IoU-дистанции: возвращает пары (track_idx, det_idx) с cost <= thresh.
void associate(const std::vector<STrack*>& tracks, const std::vector<BBoxXYXY>& dets,
               float thresh, std::vector<std::pair<int, int>>& matches,
               std::vector<int>& u_track, std::vector<int>& u_det) {
  matches.clear();
  u_track.clear();
  u_det.clear();
  const int nt = static_cast<int>(tracks.size());
  const int nd = static_cast<int>(dets.size());
  if (nt == 0 || nd == 0) {
    for (int i = 0; i < nt; ++i) u_track.push_back(i);
    for (int j = 0; j < nd; ++j) u_det.push_back(j);
    return;
  }
  std::vector<std::vector<double>> cost(nt, std::vector<double>(nd, 0.0));
  for (int i = 0; i < nt; ++i) {
    const BBoxXYXY tb = tracks[i]->bbox();
    for (int j = 0; j < nd; ++j) {
      cost[i][j] = 1.0 - static_cast<double>(iou_xyxy(tb, dets[j]));
    }
  }
  std::vector<int> rowsol;
  solve_lap(cost, nt, nd, rowsol);
  std::vector<char> det_matched(nd, 0);
  for (int i = 0; i < nt; ++i) {
    const int j = rowsol[i];
    if (j >= 0 && cost[i][j] <= static_cast<double>(thresh)) {
      matches.emplace_back(i, j);
      det_matched[j] = 1;
    } else {
      u_track.push_back(i);
    }
  }
  for (int j = 0; j < nd; ++j)
    if (!det_matched[j]) u_det.push_back(j);
}

}  // namespace

// ---------------------------------------------------------------------------
// ByteTracker::Impl
// ---------------------------------------------------------------------------
struct ByteTracker::Impl {
  ByteTrackParams params;
  int frame_id = 0;
  int next_id = 1;
  int max_time_lost = 30;
  std::vector<STrack> tracked;
  std::vector<STrack> lost;

  explicit Impl(ByteTrackParams p) : params(p) {
    max_time_lost = static_cast<int>(params.frame_rate / 30.0f * params.track_buffer);
    if (max_time_lost < 1) max_time_lost = params.track_buffer;
  }

  void activate(STrack& t, const Detection& d, int det_idx, bool new_id) {
    if (new_id) t.track_id = next_id++;
    kf_initiate(t.kf, xyxy_to_xyah(d.bbox));
    t.state = TS::kTracked;
    t.tracklet_len = 0;
    t.frame_id = frame_id;
    t.start_frame = frame_id;
    t.score = d.confidence;
    t.class_id = d.class_id;
    t.cls_name = d.cls_name;
    t.matched_det = det_idx;
    // На самом первом кадре трек ещё не подтверждён (как в оригинале), кроме самого
    // первого кадра трекера, где все детекции активируются сразу.
    t.is_activated = (frame_id == 1);
  }

  void update_track(STrack& t, const Detection& d, int det_idx) {
    kf_update(t.kf, xyxy_to_xyah(d.bbox));
    t.state = TS::kTracked;
    t.is_activated = true;
    t.tracklet_len += 1;
    t.frame_id = frame_id;
    t.score = d.confidence;
    t.class_id = d.class_id;
    t.cls_name = d.cls_name;
    t.matched_det = det_idx;
  }

  void reactivate(STrack& t, const Detection& d, int det_idx) {
    kf_update(t.kf, xyxy_to_xyah(d.bbox));
    t.state = TS::kTracked;
    t.is_activated = true;
    t.tracklet_len = 0;
    t.frame_id = frame_id;
    t.score = d.confidence;
    t.class_id = d.class_id;
    t.cls_name = d.cls_name;
    t.matched_det = det_idx;
  }
};

ByteTracker::ByteTracker(ByteTrackParams params)
    : impl_(std::make_unique<Impl>(params)) {}
ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

void ByteTracker::reset() {
  const ByteTrackParams p = impl_->params;
  impl_ = std::make_unique<Impl>(p);
}

void ByteTracker::update(std::vector<Detection>& dets, int /*frame_w*/, int /*frame_h*/) {
  auto& I = *impl_;
  I.frame_id += 1;
  for (auto& d : dets) d.track_id = -1;

  // Разделяем детекции на high/low по уверенности.
  std::vector<int> high_idx, low_idx;
  for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
    const float s = dets[i].confidence;
    if (s >= I.params.track_high_thresh) {
      high_idx.push_back(i);
    } else if (s >= I.params.track_low_thresh) {
      low_idx.push_back(i);
    }
  }

  // Предсказание всех существующих треков (tracked + lost).
  for (auto& t : I.tracked) kf_predict(t.kf);
  for (auto& t : I.lost) kf_predict(t.kf);
  for (auto& t : I.tracked) t.matched_det = -1;
  for (auto& t : I.lost) t.matched_det = -1;

  // Пул для первого прохода: подтверждённые tracked + lost; неподтверждённые отдельно.
  std::vector<STrack*> unconfirmed, pool;
  for (auto& t : I.tracked) {
    if (t.is_activated)
      pool.push_back(&t);
    else
      unconfirmed.push_back(&t);
  }
  for (auto& t : I.lost) pool.push_back(&t);

  auto box_of = [&](const std::vector<int>& idx) {
    std::vector<BBoxXYXY> v;
    v.reserve(idx.size());
    for (int i : idx) v.push_back(dets[i].bbox);
    return v;
  };

  // --- Первый проход: pool vs high-detections.
  std::vector<std::pair<int, int>> m1;
  std::vector<int> ut1, ud1;
  associate(pool, box_of(high_idx), I.params.match_thresh, m1, ut1, ud1);
  for (auto [it, id] : m1) {
    STrack* t = pool[it];
    const int di = high_idx[id];
    if (t->state == TS::kTracked)
      I.update_track(*t, dets[di], di);
    else
      I.reactivate(*t, dets[di], di);  // был Lost
  }

  // --- Второй проход (BYTE): оставшиеся ТОЛЬКО-tracked vs low-detections.
  std::vector<STrack*> r_tracked;
  for (int i : ut1) {
    if (pool[i]->state == TS::kTracked) r_tracked.push_back(pool[i]);
  }
  std::vector<std::pair<int, int>> m2;
  std::vector<int> ut2, ud2;
  associate(r_tracked, box_of(low_idx), 0.5f, m2, ut2, ud2);
  for (auto [it, id] : m2) {
    const int di = low_idx[id];
    I.update_track(*r_tracked[it], dets[di], di);
  }
  // Неподобранные tracked → Lost.
  for (int i : ut2) {
    STrack* t = r_tracked[i];
    if (t->state != TS::kLost) {
      t->state = TS::kLost;
      t->is_activated = false;
    }
  }

  // --- Неподтверждённые треки vs оставшиеся high-detections.
  std::vector<int> rem_high;
  rem_high.reserve(ud1.size());
  for (int id : ud1) rem_high.push_back(high_idx[id]);
  std::vector<std::pair<int, int>> m3;
  std::vector<int> ut3, ud3;
  associate(unconfirmed, box_of(rem_high), 0.7f, m3, ut3, ud3);
  for (auto [it, id] : m3) {
    const int di = rem_high[id];
    I.update_track(*unconfirmed[it], dets[di], di);
  }
  for (int i : ut3) {
    unconfirmed[i]->state = TS::kRemoved;  // неподтверждённый без матча — удалить
  }

  // --- Новые треки из оставшихся high-detections.
  std::vector<STrack> activated_new;
  for (int id : ud3) {
    const int di = rem_high[id];
    if (dets[di].confidence < I.params.new_track_thresh) continue;
    STrack t;
    I.activate(t, dets[di], di, /*new_id=*/true);
    activated_new.push_back(std::move(t));
  }

  // --- Жизненный цикл потерянных треков.
  for (auto& t : I.lost) {
    if (t.state == TS::kTracked) continue;  // реактивирован выше
    if (I.frame_id - t.frame_id > I.max_time_lost) t.state = TS::kRemoved;
  }

  // --- Пересборка списков tracked / lost.
  std::vector<STrack> new_tracked, new_lost;
  for (auto& t : I.tracked) {
    if (t.state == TS::kTracked)
      new_tracked.push_back(std::move(t));
    else if (t.state == TS::kLost)
      new_lost.push_back(std::move(t));
    // kRemoved — выбрасываем
  }
  for (auto& t : I.lost) {
    if (t.state == TS::kTracked)
      new_tracked.push_back(std::move(t));  // реактивированный
    else if (t.state == TS::kLost)
      new_lost.push_back(std::move(t));
  }
  for (auto& t : activated_new) new_tracked.push_back(std::move(t));

  I.tracked.swap(new_tracked);
  I.lost.swap(new_lost);

  // --- Проставляем track_id активным трекам по индексу детекции.
  for (auto& t : I.tracked) {
    if (t.is_activated && t.matched_det >= 0 &&
        t.matched_det < static_cast<int>(dets.size())) {
      dets[t.matched_det].track_id = t.track_id;
    }
  }
}

}  // namespace integra
