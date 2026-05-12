//! Async-фасад над одним `Pipeline`: bounded-каналы и drop-oldest семантика.
//!
//! Архитектура одного потока (одной камеры):
//!
//!   frame source ──► frame_tx (mpsc bounded(2), try_send → drop-newest при Full)
//!                              │
//!                          [worker] = spawn_blocking
//!                              │
//!                              ▼
//!                       Pipeline::push_frame
//!                              │
//!                              ▼
//!                    events_tx (mpsc bounded(EVENTS_CAP))
//!                              │
//!                              ▼
//!                       consumer (UI / алармы / persist)
//!
//! Почему `mpsc(2)` с try_send drop-newest:
//!   * tokio mpsc не даёт producer'у дропнуть `oldest` — это требует доступа к
//!     `recv` со стороны producer'а, которого у нас нет.
//!   * drop-newest = "если воркер не справляется, мы выкидываем самый свежий
//!     поступающий кадр". Для live-видео это эквивалентно throttling источника
//!     (потеряем кадр, но не задержим следующий).
//!   * Альтернатива (drop-oldest) требует своего queue — реализуем в шаге 4,
//!     если буферизация на 2 окажется недостаточной.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

use super::events::StreamMessage;
use super::ffi::{IntegraError, IntegraLib};
use super::pipeline::{Pipeline, PipelineConfig};
use super::preview_encode;

/// Размер канала событий. 64 — компромисс: всплеск ~32 трека × 1-2 события
/// на смену состояния FSM ещё помещается без блокировки producer'а.
pub const EVENTS_CHANNEL_CAP: usize = 64;

/// Размер канала кадров. 2 — один "in-flight" + один "next".
pub const FRAMES_CHANNEL_CAP: usize = 2;

/// Один сырой BGR-кадр.
///
/// **Семантика владения**: текущий вариант — `Vec<u8>` (owned). Producer
/// клонирует пиксели в эту структуру и отдаёт ownership в канал.
/// На 1080p BGR это ~6 МБ копии. Если станет узким местом — в шаге 4
/// заменим на `Arc<[u8]>` и пул переиспользуемых буферов.
pub struct Frame {
    pub bgr: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pts_ms: i64,
}

pub struct StreamHandle {
    pub frame_tx: mpsc::Sender<Frame>,
    pub events_rx: mpsc::Receiver<StreamMessage>,
    pub join: JoinHandle<()>,
}

impl StreamHandle {
    /// Попытаться положить кадр в очередь. Возвращает `Ok(true)`, если кадр
    /// принят, `Ok(false)`, если очередь полна и кадр выкинут.
    pub fn push_frame_nonblocking(&self, frame: Frame) -> Result<bool, mpsc::error::SendError<Frame>> {
        match self.frame_tx.try_send(frame) {
            Ok(()) => Ok(true),
            Err(mpsc::error::TrySendError::Full(_)) => Ok(false),
            Err(mpsc::error::TrySendError::Closed(f)) => Err(mpsc::error::SendError(f)),
        }
    }

    /// Сигнализирует воркеру завершиться (через закрытие frame_tx).
    /// После возврата handle сам можно дождаться через `.join.await`.
    pub fn shutdown(self) -> JoinHandle<()> {
        drop(self.frame_tx);
        self.join
    }
}

// ---------------------------------------------------------------------------
// spawn_stream — основной entry point.
// ---------------------------------------------------------------------------

/// Создаёт pipeline и запускает blocking-таск, обрабатывающий кадры.
///
/// Возвращает handle с двумя каналами:
///   * `frame_tx`  — куда продьюсер шлёт кадры (`try_send` или `push_frame_nonblocking`);
///   * `events_rx` — откуда консьюмер читает события и FrameResult'ы.
pub fn spawn_stream(
    cfg: PipelineConfig,
    lib: Arc<IntegraLib>,
) -> Result<StreamHandle, IntegraError> {
    // Создание pipeline синхронное (читает .engine, инициализирует CUDA).
    // Если упадёт — не плодим таск.
    let mut pipeline = Pipeline::new(&cfg, lib)?;

    let (frame_tx, mut frame_rx) = mpsc::channel::<Frame>(FRAMES_CHANNEL_CAP);
    let (events_tx, events_rx) = mpsc::channel::<StreamMessage>(EVENTS_CHANNEL_CAP);

    let camera_id = cfg.camera_id.clone();
    let join = tokio::task::spawn_blocking(move || {
        fn alarm_needs_snapshot(kind: &str) -> bool {
            kind.starts_with("alarm_") || kind == "abandoned" || kind == "disappeared"
        }

        info!(camera_id = %camera_id, "integra stream worker started");
        // blocking_recv возвращает None после закрытия sender'а — это сигнал shutdown.
        while let Some(frame) = frame_rx.blocking_recv() {
            let w = frame.width as i32;
            let h = frame.height as i32;
            let pts_ms = frame.pts_ms;
            let need = (frame.width as usize) * (frame.height as usize) * 3;
            if frame.bgr.len() < need {
                let _ = events_tx.blocking_send(StreamMessage::Error(format!(
                    "frame buffer too small: have {}, need {}",
                    frame.bgr.len(),
                    need
                )));
                continue;
            }
            let result = pipeline.push_frame(&frame.bgr, w, h, pts_ms);
            match result {
                Ok(messages) => {
                    let tracks_overlay = messages
                        .iter()
                        .find_map(|m| {
                            if let StreamMessage::Frame(fr) = m {
                                Some(fr.tracks.clone())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_default();
                    let mut to_send = Vec::with_capacity(messages.len());
                    for msg in messages {
                        match msg {
                            StreamMessage::Event(mut ev) => {
                                if alarm_needs_snapshot(&ev.kind) {
                                    let crop = preview_encode::encode_alarm_crop_jpeg(
                                        &frame.bgr,
                                        frame.width,
                                        frame.height,
                                        ev.bbox,
                                        0.14,
                                        720,
                                        82,
                                    );
                                    ev.snapshot_jpeg = crop.or_else(|| {
                                        preview_encode::encode_preview_jpeg(
                                            &frame.bgr,
                                            frame.width,
                                            frame.height,
                                            &tracks_overlay,
                                        )
                                    });
                                }
                                to_send.push(StreamMessage::Event(ev));
                            }
                            other => to_send.push(other),
                        }
                    }
                    for msg in to_send {
                        if events_tx.blocking_send(msg).is_err() {
                            debug!(camera_id = %camera_id, "events consumer closed; stopping worker");
                            return;
                        }
                    }
                }
                Err(e) => {
                    error!(camera_id = %camera_id, error = %e, "push_frame failed");
                    if events_tx
                        .blocking_send(StreamMessage::Error(format!("{e}")))
                        .is_err()
                    {
                        return;
                    }
                }
            }
        }
        info!(camera_id = %camera_id, "integra stream worker stopped (frame_rx closed)");
    });

    Ok(StreamHandle {
        frame_tx,
        events_rx,
        join,
    })
}

// ---------------------------------------------------------------------------
// Unit tests (без реальной DLL — чисто тип-уровень + конфиг).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_default_sane() {
        let c = PipelineConfig::default();
        assert_eq!(c.engine_kind, "tensorrt");
        assert_eq!(c.input_size, 640);
        assert!(c.conf_threshold > 0.0 && c.conf_threshold < 1.0);
        assert!(c.abandon_time_sec > 0.0);
    }

    /// Smoke-тест по реальной DLL. Если `integra_ffi.dll` найден — создаёт
    /// pipeline с engine_kind="stub" (без модели) и прогоняет один 64x64 кадр.
    /// Если DLL не найден — тест пропускается (полезно в CI без C++ сборки).
    #[test]
    fn smoke_load_stub_pipeline() {
        let lib = match crate::integra::ffi::global_lib() {
            Ok(l) => l,
            Err(e) => {
                eprintln!("skip smoke_load_stub_pipeline: {e}");
                return;
            }
        };
        eprintln!("integra_ffi version: {}", lib.version());

        let cfg = PipelineConfig {
            engine_kind: "stub".into(),
            model_path: String::new(),
            input_size: 64,
            conf_threshold: 0.25,
            nms_iou_threshold: 0.45,
            num_classes: 80,
            num_anchors: 8400,
            min_box_size_px: 4,
            ..PipelineConfig::default()
        };
        let mut p = Pipeline::new(&cfg, lib).expect("pipeline must init on stub backend");
        let bgr = vec![128u8; 64 * 64 * 3];
        let msgs = p
            .push_frame(&bgr, 64, 64, 0)
            .expect("stub push_frame must succeed");
        // stub backend ничего не детектит — но FrameResult обязан прийти.
        let frame_msg_count = msgs
            .iter()
            .filter(|m| matches!(m, StreamMessage::Frame(_)))
            .count();
        assert_eq!(frame_msg_count, 1, "expected exactly one FrameResult, got {msgs:?}");
    }

    #[test]
    fn frame_message_roundtrip_json() {
        // Sanity check: то, что C-код пишет, мы можем распарсить.
        let raw = r#"{"frame_id":42,"pts_ms":1234.5,
            "stats":{"detections":3,"persons":1,
                     "infer_ms":12.3,"preprocess_ms":1.1,
                     "tracker_ms":0.2,"analyzer_ms":0.1},
            "tracks":[{"id":1,"cls":"handbag","state":"static",
                       "bbox":[1.0,2.0,3.0,4.0],"conf":0.7,
                       "static_for_sec":5.0,"unattended_for_sec":0.0,"alarm":false}],
            "persons":[{"track_id":-1,"confidence":0.9,
                        "bbox":[10.0,20.0,30.0,40.0]}]}"#;
        let fr: crate::integra::events::FrameResult = serde_json::from_str(raw).unwrap();
        assert_eq!(fr.frame_id, 42);
        assert_eq!(fr.tracks.len(), 1);
        assert_eq!(fr.persons.len(), 1);
    }
}
