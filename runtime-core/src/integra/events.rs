//! Типизированные структуры, в которые парсится payload из FFI-callback.
//!
//! Точно совпадают с JSON-builder'ами в `native/src/integra_ffi.cpp`
//! (см. `event_to_json` / `tracks_array_json` / `persons_array_json` /
//! `frame_result_json`). При изменении C-формата — обновить здесь.

use serde::{Deserialize, Serialize};

/// Алерт от SceneAnalyzer (abandoned / disappeared / …).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlarmEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub camera_id: String,
    pub track_id: i64,
    pub cls_id: i32,
    pub cls_name: String,
    pub confidence: f64,
    pub ts_wall_ms: f64,
    pub video_pos_ms: f64,
    pub bbox: [f32; 4],
    #[serde(default)]
    pub note: String,
    #[serde(skip, default)]
    pub snapshot_jpeg: Option<Vec<u8>>,
}

/// Снимок трека для UI (на каждый кадр).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrackSnapshot {
    pub id: i32,
    pub cls: String,
    pub state: String,
    pub bbox: [f32; 4],
    pub conf: f32,
    pub static_for_sec: f64,
    pub unattended_for_sec: f64,
    pub alarm: bool,
}

/// Детекция человека (без присвоения трека в UI).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PersonDet {
    pub track_id: i32,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FrameStats {
    pub detections: u32,
    pub persons: u32,
    pub infer_ms: f64,
    pub preprocess_ms: f64,
    pub tracker_ms: f64,
    pub analyzer_ms: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FrameResult {
    pub frame_id: u64,
    pub pts_ms: f64,
    pub stats: FrameStats,
    pub tracks: Vec<TrackSnapshot>,
    pub persons: Vec<PersonDet>,
}

/// Сообщение, которое spawn_stream отправляет наружу через mpsc-канал.
#[derive(Debug, Clone)]
pub enum StreamMessage {
    /// Alarm от SceneAnalyzer (по одному на каждое событие на кадр).
    Event(AlarmEvent),
    /// Полный результат кадра (треки + люди + статистика). Идёт после всех событий.
    Frame(FrameResult),
    /// Нефатальная ошибка инференса/парсинга — поток продолжает работать.
    Error(String),
}
