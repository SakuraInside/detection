//! Safe Rust обёртка над одним `IntegraPipeline`.
//!
//! Контракт:
//!   * `Pipeline` — `Send`, но **не** `Sync`: C-сторона single-stream per pipeline.
//!     В `spawn_stream` мы это соблюдаем (один blocking-таск владеет pipeline).
//!   * `push_frame()` синхронный, занимает CPU+GPU несколько мс — вызывать только
//!     из `spawn_blocking` или dedicated thread.

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;

use tracing::{debug, warn};

use super::events::{AlarmEvent, FrameResult, StreamMessage};
use super::ffi::{
    push_frame_error_message, IntegraConfig, IntegraError, IntegraLib, IntegraPipelineRaw,
    INTEGRA_FFI_ABI_VERSION,
};

/// Безопасная конфигурация: владеет CString'ами, чтобы C-сторона видела
/// валидные указатели на всё время жизни struct.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub engine_kind: String,
    pub model_path: String,
    pub input_size: i32,

    pub conf_threshold: f32,
    pub nms_iou_threshold: f32,
    pub num_classes: i32,
    pub num_anchors: i32,

    pub person_class_id: i32,
    pub object_classes: Vec<i32>,
    pub min_box_size_px: i32,

    pub static_displacement_px: f64,
    pub static_window_sec: f64,
    pub abandon_time_sec: f64,
    pub owner_proximity_px: f64,
    pub owner_left_sec: f64,
    pub disappear_grace_sec: f64,
    pub min_object_area_px: f64,
    pub centroid_history_maxlen: i32,
    pub max_active_tracks: i32,

    pub camera_id: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            engine_kind: "tensorrt".into(),
            model_path: String::new(),
            input_size: 640,
            conf_threshold: 0.25,
            nms_iou_threshold: 0.45,
            num_classes: 80,
            num_anchors: 8400,
            person_class_id: 0,
            object_classes: Vec::new(),
            min_box_size_px: 20,
            static_displacement_px: 7.0,
            static_window_sec: 3.0,
            abandon_time_sec: 15.0,
            owner_proximity_px: 180.0,
            owner_left_sec: 5.0,
            disappear_grace_sec: 4.0,
            min_object_area_px: 100.0,
            centroid_history_maxlen: 72,
            max_active_tracks: 256,
            camera_id: "main".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// CString-холдер, чтобы выставить C-указатели и не уронить их в процессе вызова.
// ---------------------------------------------------------------------------
struct CStrings {
    engine_kind: CString,
    model_path: CString,
    camera_id: CString,
}

impl CStrings {
    fn from_cfg(cfg: &PipelineConfig) -> Result<Self, IntegraError> {
        Ok(CStrings {
            engine_kind: CString::new(cfg.engine_kind.as_str())
                .map_err(|_| IntegraError::InvalidArgument("engine_kind contains NUL"))?,
            model_path: CString::new(cfg.model_path.as_str())
                .map_err(|_| IntegraError::InvalidArgument("model_path contains NUL"))?,
            camera_id: CString::new(cfg.camera_id.as_str())
                .map_err(|_| IntegraError::InvalidArgument("camera_id contains NUL"))?,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline.
// ---------------------------------------------------------------------------
pub struct Pipeline {
    handle: *mut IntegraPipelineRaw,
    lib: Arc<IntegraLib>,
}

// Pipeline owns the C handle — pointer не делится между потоками.
// Send: пайплайн можно ПЕРЕДАТЬ в другой поток (move).
// Не делаем Sync: одновременные push_frame'ы из двух потоков запрещены.
unsafe impl Send for Pipeline {}

impl Pipeline {
    pub fn new(cfg: &PipelineConfig, lib: Arc<IntegraLib>) -> Result<Self, IntegraError> {
        let cstrs = CStrings::from_cfg(cfg)?;
        let c_cfg = IntegraConfig {
            engine_kind: cstrs.engine_kind.as_ptr(),
            model_path: cstrs.model_path.as_ptr(),
            input_size: cfg.input_size,
            conf_threshold: cfg.conf_threshold,
            nms_iou_threshold: cfg.nms_iou_threshold,
            num_classes: cfg.num_classes,
            num_anchors: cfg.num_anchors,
            person_class_id: cfg.person_class_id,
            object_classes: if cfg.object_classes.is_empty() {
                ptr::null()
            } else {
                cfg.object_classes.as_ptr()
            },
            object_classes_len: cfg.object_classes.len() as i32,
            min_box_size_px: cfg.min_box_size_px,
            static_displacement_px: cfg.static_displacement_px,
            static_window_sec: cfg.static_window_sec,
            abandon_time_sec: cfg.abandon_time_sec,
            owner_proximity_px: cfg.owner_proximity_px,
            owner_left_sec: cfg.owner_left_sec,
            disappear_grace_sec: cfg.disappear_grace_sec,
            min_object_area_px: cfg.min_object_area_px,
            centroid_history_maxlen: cfg.centroid_history_maxlen,
            max_active_tracks: cfg.max_active_tracks,
            camera_id: cstrs.camera_id.as_ptr(),
        };

        let handle = unsafe { (lib.create)(INTEGRA_FFI_ABI_VERSION, &c_cfg as *const _) };
        // cstrs живёт здесь до конца функции — C-указатели валидны во время create.
        drop(cstrs);

        if handle.is_null() {
            return Err(IntegraError::PipelineCreateFailed);
        }
        Ok(Pipeline { handle, lib })
    }

    /// Обработать кадр. `bgr` — плотный BGR uint8 длиной `width*height*3`.
    ///
    /// Возвращает `Vec<StreamMessage>`: сначала все алармы (если есть), затем
    /// один `Frame(FrameResult)` со снимком треков и статистикой.
    pub fn push_frame(
        &mut self,
        bgr: &[u8],
        width: i32,
        height: i32,
        pts_ms: i64,
    ) -> Result<Vec<StreamMessage>, IntegraError> {
        if width <= 0 || height <= 0 {
            return Err(IntegraError::InvalidArgument("width/height must be > 0"));
        }
        let need = (width as usize) * (height as usize) * 3;
        if bgr.len() < need {
            return Err(IntegraError::InvalidArgument(
                "bgr buffer smaller than width*height*3",
            ));
        }

        let mut collector = FrameCollector::default();
        let code = unsafe {
            (self.lib.push_frame)(
                self.handle,
                bgr.as_ptr(),
                width,
                height,
                pts_ms,
                Some(trampoline),
                &mut collector as *mut FrameCollector as *mut c_void,
            )
        };
        if code != 0 {
            return Err(IntegraError::PushFrameFailed(
                code,
                push_frame_error_message(code),
            ));
        }
        Ok(collector.into_messages())
    }

    pub fn reset(&mut self) {
        unsafe { (self.lib.reset)(self.handle) };
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.lib.destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// FrameCollector + trampoline: пробрасываем callback'и C → safe Rust.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct FrameCollector {
    events: Vec<AlarmEvent>,
    frame_result: Option<FrameResult>,
    parse_errors: Vec<String>,
}

impl FrameCollector {
    fn into_messages(self) -> Vec<StreamMessage> {
        let mut out = Vec::with_capacity(self.events.len() + 1 + self.parse_errors.len());
        for e in self.events {
            out.push(StreamMessage::Event(e));
        }
        if let Some(fr) = self.frame_result {
            out.push(StreamMessage::Frame(fr));
        }
        for e in self.parse_errors {
            out.push(StreamMessage::Error(e));
        }
        out
    }
}

/// C-callback. Никогда не panic-ит наружу (catch_unwind), unsafe-к жизни user_data
/// контролируется выше: указатель валиден на всё время push_frame().
extern "C" fn trampoline(
    type_tag: *const c_char,
    payload_json: *const c_char,
    user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(|| {
        if user_data.is_null() || type_tag.is_null() || payload_json.is_null() {
            return;
        }
        let collector = unsafe { &mut *(user_data as *mut FrameCollector) };
        let tag = unsafe { CStr::from_ptr(type_tag) }.to_string_lossy();
        let payload = unsafe { CStr::from_ptr(payload_json) }.to_string_lossy();

        match tag.as_ref() {
            "event" => match serde_json::from_str::<AlarmEvent>(&payload) {
                Ok(ev) => {
                    debug!(?ev, "integra event");
                    collector.events.push(ev);
                }
                Err(e) => {
                    warn!(error=%e, payload=%payload, "failed to parse alarm event");
                    collector.parse_errors.push(format!("event: {e}"));
                }
            },
            "frame_result" => match serde_json::from_str::<FrameResult>(&payload) {
                Ok(fr) => {
                    collector.frame_result = Some(fr);
                }
                Err(e) => {
                    warn!(error=%e, payload=%payload, "failed to parse frame_result");
                    collector.parse_errors.push(format!("frame_result: {e}"));
                }
            },
            other => {
                collector
                    .parse_errors
                    .push(format!("unknown FFI tag: {other}"));
            }
        }
    });
}
