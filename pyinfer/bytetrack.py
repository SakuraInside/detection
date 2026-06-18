"""Компактный ByteTrack: Kalman (numpy) + двухэтапная ассоциация (lap).

Без torch/ultralytics. Даёт устойчивые track ID для людей (контур M2 по ТЗ).
ID начинаются со 100001 — как на эталонном скрине пользователя.
"""
from __future__ import annotations

import lap
import numpy as np

from .config import TrackerCfg


class KalmanFilter:
    """Kalman для bbox в пространстве (x_center, y_center, aspect, height)."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        pmean = self._update_mat @ mean
        pcov = self._update_mat @ covariance @ self._update_mat.T
        return pmean, pcov + innovation_cov

    def update(self, mean, covariance, measurement):
        pmean, pcov = self.project(mean, covariance)
        kalman_gain = covariance @ self._update_mat.T @ np.linalg.inv(pcov)
        innovation = measurement - pmean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ pcov @ kalman_gain.T
        return new_mean, new_cov


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    shared_kalman = KalmanFilter()
    count = 100000  # первый присвоенный ID будет 100001

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0

    @staticmethod
    def next_id():
        STrack.count += 1
        return STrack.count

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        for st in stracks:
            st.predict()

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret


# ---------- matching helpers ----------

def _ious(atlbrs, btlbrs):
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    a = np.ascontiguousarray(atlbrs, dtype=np.float32)
    b = np.ascontiguousarray(btlbrs, dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def iou_distance(atracks, btracks):
    atlbrs = [t.tlbr for t in atracks]
    btlbrs = [t.tlbr for t in btracks]
    return 1.0 - _ious(atlbrs, btlbrs)


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches = []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    u_track = np.where(x < 0)[0]
    u_det = np.where(y < 0)[0]
    return np.asarray(matches), u_track, u_det


def joint_stracks(tlista, tlistb):
    exists = {t.track_id: 1 for t in tlista}
    res = list(tlista)
    for t in tlistb:
        if t.track_id not in exists:
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


class BYTETracker:
    def __init__(self, cfg: TrackerCfg):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.track_thresh = cfg.high_thresh
        self.det_thresh = cfg.new_thresh
        self.low_thresh = cfg.low_thresh
        self.match_thresh = cfg.match_thresh
        self.max_time_lost = int(cfg.frame_rate / 30.0 * cfg.track_buffer)
        self.kalman_filter = KalmanFilter()

    def update(self, dets: np.ndarray):
        """dets: Nx5 [x1,y1,x2,y2,score]. Возвращает активные STrack."""
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        if dets is None or len(dets) == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        scores = dets[:, 4]
        bboxes = dets[:, :4]

        remain = scores >= self.track_thresh
        low = (scores > self.low_thresh) & (scores < self.track_thresh)
        dets_high = [STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(bboxes[remain], scores[remain])]
        dets_low = [STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(bboxes[low], scores[low])]

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # --- 1-я ассоциация (high conf) ---
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, dets_high)
        matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)
        for itr, idet in matches:
            track = strack_pool[itr]
            det = dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # --- 2-я ассоциация (low conf) ---
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked, dets_low)
        matches, u_track2, u_det_low = linear_assignment(dists, thresh=0.5)
        for itr, idet in matches:
            track = r_tracked[itr]
            det = dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # --- unconfirmed (одиночные high-conf без подтверждения) ---
        dets_remaining = [dets_high[i] for i in u_det]
        dists = iou_distance(unconfirmed, dets_remaining)
        matches, u_unconfirmed, u_det = linear_assignment(dists, thresh=0.7)
        for itr, idet in matches:
            unconfirmed[itr].update(dets_remaining[idet], self.frame_id)
            activated.append(unconfirmed[itr])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        # --- новые треки ---
        for inew in u_det:
            track = dets_remaining[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # --- истёкшие lost ---
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]

        return [t for t in self.tracked_stracks if t.is_activated]

    def predicted_lost(self, max_age: int = 20):
        """Недавно «потерянные» треки с Kalman-предсказанной позицией (tlbr).

        Нужны как зона ПОДАВЛЕНИЯ объект-кандидатов: сидящий человек мерцает в YOLO,
        и на кадрах-пропусках его силуэт ловится MOG2 как «оставленный объект».
        Пока трек жив предсказанием — гасим этот регион. В FSM/UI НЕ отдаём.
        """
        if max_age <= 0:
            return []
        return [t for t in self.lost_stracks if (self.frame_id - t.frame_id) <= max_age]
