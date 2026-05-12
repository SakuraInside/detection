//! Низкоуровневый FFI-слой: `extern "C"` сигнатуры + `IntegraLib`.
//!
//! Сигнатуры зеркалят `native/include/integra/integra_ffi.h` (ABI v1).
//! `IntegraLib` — handle на загруженную DLL/so; экземпляр один на процесс
//! (через `OnceCell`), чтобы C++ синглтоны (`SharedTRTEngine` кэш и т.п.)
//! не дублировались.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::os::raw::c_double;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

// ---------------------------------------------------------------------------
// ABI constants — должны точно соответствовать integra_ffi.h.
// ---------------------------------------------------------------------------

pub const INTEGRA_FFI_ABI_VERSION: c_int = 1;

// ---------------------------------------------------------------------------
// IntegraConfig — repr(C), один-в-один с C struct.
//
// ВАЖНО: поля по порядку и типу строго совпадают с native/include/integra/integra_ffi.h.
// При изменении C-структуры — поднять INTEGRA_FFI_ABI_VERSION на C-стороне.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug)]
pub struct IntegraConfig {
    // engine
    pub engine_kind: *const c_char,
    pub model_path: *const c_char,
    pub input_size: c_int,
    // postprocess
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32,
    pub num_classes: c_int,
    pub num_anchors: c_int,
    // class whitelist
    pub person_class_id: c_int,
    pub object_classes: *const c_int,
    pub object_classes_len: c_int,
    pub min_box_size_px: c_int,
    // analyzer
    pub static_displacement_px: c_double,
    pub static_window_sec: c_double,
    pub abandon_time_sec: c_double,
    pub owner_proximity_px: c_double,
    pub owner_left_sec: c_double,
    pub disappear_grace_sec: c_double,
    pub min_object_area_px: c_double,
    pub centroid_history_maxlen: c_int,
    pub max_active_tracks: c_int,
    // identity
    pub camera_id: *const c_char,
}

// Pipeline в C — opaque struct; в Rust представляем как enum без вариантов.
#[repr(C)]
pub struct IntegraPipelineRaw {
    _private: [u8; 0],
}

pub type IntegraEventCb = unsafe extern "C" fn(
    type_tag: *const c_char,
    payload_json: *const c_char,
    user_data: *mut c_void,
);

// ---------------------------------------------------------------------------
// Type aliases для сигнатур функций.
// ---------------------------------------------------------------------------

type FnCreate =
    unsafe extern "C" fn(abi_version: c_int, cfg: *const IntegraConfig) -> *mut IntegraPipelineRaw;
type FnDestroy = unsafe extern "C" fn(p: *mut IntegraPipelineRaw);
type FnPushFrame = unsafe extern "C" fn(
    p: *mut IntegraPipelineRaw,
    bgr_data: *const u8,
    width: c_int,
    height: c_int,
    pts_ms: i64,
    cb: Option<IntegraEventCb>,
    user_data: *mut c_void,
) -> c_int;
type FnReset = unsafe extern "C" fn(p: *mut IntegraPipelineRaw);
type FnVersion = unsafe extern "C" fn() -> *const c_char;

// ---------------------------------------------------------------------------
// Ошибки.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum IntegraError {
    #[error("integra_ffi: library load failed at {path:?}: {source}")]
    LoadFailed {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("integra_ffi: library not found in any of {paths:?}")]
    LibraryNotFound { paths: Vec<PathBuf> },
    #[error("integra_ffi: symbol '{symbol}' not found: {source}")]
    MissingSymbol {
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("integra_ffi: pipeline create returned NULL (cfg/ABI/model error — см. stderr)")]
    PipelineCreateFailed,
    #[error("integra_ffi: push_frame failed (code={0}): {1}")]
    PushFrameFailed(i32, &'static str),
    #[error("integra_ffi: invalid argument: {0}")]
    InvalidArgument(&'static str),
    #[error("integra_ffi: JSON parse failed: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("integra_ffi: UTF-8 decode failed: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

pub fn push_frame_error_message(code: i32) -> &'static str {
    match code {
        0 => "ok",
        -1 => "pipeline is NULL",
        -2 => "bad frame (width/height/data)",
        -3 => "inference failed (см. stderr)",
        -4 => "ABI mismatch / context unavailable",
        _ => "unknown error",
    }
}

// ---------------------------------------------------------------------------
// IntegraLib — загруженная DLL/so + типобезопасные function pointers.
// ---------------------------------------------------------------------------

pub struct IntegraLib {
    // Library должен переживать function pointers (иначе они становятся dangling).
    // Храним первым полем — гарантирует drop-order: pointers → library.
    _library: Library,
    pub(super) create: FnCreate,
    pub(super) destroy: FnDestroy,
    pub(super) push_frame: FnPushFrame,
    pub(super) reset: FnReset,
    pub(super) version: FnVersion,
}

// Functions из загруженной DLL — это просто адреса в памяти, безопасно делить
// между потоками (immutable). Сами вызовы — unsafe, контракт обеспечивает caller.
unsafe impl Send for IntegraLib {}
unsafe impl Sync for IntegraLib {}

#[cfg(target_os = "windows")]
const LIB_FILENAME: &str = "integra_ffi.dll";
#[cfg(target_os = "linux")]
const LIB_FILENAME: &str = "libintegra_ffi.so";
#[cfg(target_os = "macos")]
const LIB_FILENAME: &str = "libintegra_ffi.dylib";

/// Список кандидатов, где искать DLL/so по умолчанию.
/// Порядок: `INTEGRA_FFI_PATH` → рядом с exe → разработческие пути → системный поиск.
fn candidate_paths() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(env) = std::env::var("INTEGRA_FFI_PATH") {
        v.push(PathBuf::from(env));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            v.push(parent.join(LIB_FILENAME));
        }
    }
    // dev-сборка из workspace root.
    for rel in &[
        "native/build-msvc/RelWithDebInfo",
        "native/build-msvc/Release",
        "native/build/RelWithDebInfo",
        "native/build/Release",
        "native/build",
        "../native/build-msvc/RelWithDebInfo",
        "../native/build-msvc/Release",
        "../native/build/RelWithDebInfo",
        "../native/build",
    ] {
        v.push(PathBuf::from(rel).join(LIB_FILENAME));
    }
    // bare name → системный поиск (LD_LIBRARY_PATH / PATH / rpath).
    v.push(PathBuf::from(LIB_FILENAME));
    v
}

unsafe fn resolve<T>(lib: &Library, name: &'static [u8]) -> Result<T, IntegraError>
where
    T: Copy,
{
    let sym: Symbol<T> = lib
        .get(name)
        .map_err(|e| IntegraError::MissingSymbol {
            symbol: std::str::from_utf8(name).unwrap_or("?"),
            source: e,
        })?;
    Ok(*sym)
}

impl IntegraLib {
    /// Загрузить из явного пути.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, IntegraError> {
        let path_ref = path.as_ref();
        unsafe {
            let library =
                Library::new(path_ref).map_err(|e| IntegraError::LoadFailed {
                    path: path_ref.to_path_buf(),
                    source: e,
                })?;
            let create: FnCreate = resolve(&library, b"integra_pipeline_create\0")?;
            let destroy: FnDestroy = resolve(&library, b"integra_pipeline_destroy\0")?;
            let push_frame: FnPushFrame = resolve(&library, b"integra_pipeline_push_frame\0")?;
            let reset: FnReset = resolve(&library, b"integra_pipeline_reset\0")?;
            let version: FnVersion = resolve(&library, b"integra_ffi_version\0")?;
            Ok(IntegraLib {
                _library: library,
                create,
                destroy,
                push_frame,
                reset,
                version,
            })
        }
    }

    /// Загрузить из стандартных путей. Возвращает `IntegraError::LibraryNotFound`,
    /// если ни одного валидного кандидата не найдено.
    pub fn load_default() -> Result<Self, IntegraError> {
        let candidates = candidate_paths();
        let mut tried = Vec::new();
        for cand in &candidates {
            if !cand.exists() && cand.is_absolute() {
                // абсолютный путь, файла нет — даже не пытаемся.
                tried.push(cand.clone());
                continue;
            }
            match Self::load_from_path(cand) {
                Ok(l) => return Ok(l),
                Err(IntegraError::LoadFailed { .. }) => {
                    tried.push(cand.clone());
                    continue;
                }
                Err(other) => return Err(other),
            }
        }
        Err(IntegraError::LibraryNotFound { paths: tried })
    }

    /// Версия библиотеки (как возвращает `integra_ffi_version()` из C).
    pub fn version(&self) -> String {
        unsafe {
            let raw = (self.version)();
            if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Process-wide singleton: один IntegraLib на процесс — чтобы C-сторона
// (например, SharedTRTEngine weak-cache) не дублировалась.
// ---------------------------------------------------------------------------

static GLOBAL_LIB: OnceCell<Arc<IntegraLib>> = OnceCell::new();

pub fn global_lib() -> Result<Arc<IntegraLib>, IntegraError> {
    if let Some(lib) = GLOBAL_LIB.get() {
        return Ok(lib.clone());
    }
    let lib = Arc::new(IntegraLib::load_default()?);
    let _ = GLOBAL_LIB.set(lib.clone());
    Ok(lib)
}
