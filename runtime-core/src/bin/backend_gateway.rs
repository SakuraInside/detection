//! backend_gateway: единственный пользовательский backend.
//!
//! Поток данных:
//!   video-bridge (Rust + opencv) :9876  ──TCP/BGR──►  bridge_worker
//!                                                          │
//!                                                          ├──► pipeline (FFI: TRT/OpenCV DNN)
//!                                                          │       │
//!                                                          │       └──► events_consumer ──► state + WS
//!                                                          │
//!                                                          └──► MJPEG render (BGR→RGB + bbox overlay → JPEG)
//!                                                                  │
//!                                                                  └──► /video_feed
//!
//! Python в рантайме = 0 процессов.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::process::Command;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use sysinfo::System;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::time::{self, Duration};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use runtime_core::integra::{
    self, encode_preview_jpeg_with_options, FrameResult, IntegraLib, PersonDet, StreamMessage,
    TrackSnapshot,
};

// ---------------------------------------------------------------------------
// State.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PipelineStats {
    detections: u32,
    persons: u32,
    infer_ms: f64,
    preprocess_ms: f64,
    tracker_ms: f64,
    analyzer_ms: f64,
    frames_processed: u64,
    events_total: u64,
    last_frame_id: u64,
    /// Скорость приёма кадров с video-bridge (плавность видео в UI).
    video_bridge_fps: f64,
    last_video_frame_at: Option<Instant>,
    /// Скорость кадров, полностью обработанных FFI (инференс + треки).
    pipeline_process_fps: f64,
    last_pipeline_frame_at: Option<Instant>,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            detections: 0,
            persons: 0,
            infer_ms: 0.0,
            preprocess_ms: 0.0,
            tracker_ms: 0.0,
            analyzer_ms: 0.0,
            frames_processed: 0,
            events_total: 0,
            last_frame_id: 0,
            video_bridge_fps: 0.0,
            last_video_frame_at: None,
            pipeline_process_fps: 0.0,
            last_pipeline_frame_at: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct GatewayInfo {
    loaded: bool,
    playing: bool,
    video_path: Option<String>,
    fps: f64,
    frame_count: i64,
    current_frame: i64,
    width: i32,
    height: i32,
    latest_jpeg: Option<Vec<u8>>,
    tracks: Vec<TrackSnapshot>,
    persons: Vec<PersonDet>,
    stats: PipelineStats,
}

struct SessionHandle {
    frame_tx: mpsc::Sender<integra::Frame>,
    _events_task: JoinHandle<()>,
}

/// Всё состояние ОДНОГО потока (камера/файл). Несколько таких живут параллельно
/// в `AppState.streams`, ключ — `stream_id` ("cam1"/"cam2"/…). Это даёт несколько
/// одновременно открытых видео/RTSP, каждое со своим мостом, worker-сессией,
/// превью и таймлайном.
struct Stream {
    id: String,
    info: RwLock<GatewayInfo>,
    /// Зеркало `info.playing` без async-лока — читается в горячем bridge read-loop
    /// каждый кадр (см. историю фикса троттла моста).
    playing: AtomicBool,
    playback_gen: AtomicU64,
    session: Mutex<Option<SessionHandle>>,
    /// Sender команд в активный bridge_worker этого потока (seek/pause/play/close).
    bridge_cmd_tx: Mutex<Option<std::sync::mpsc::Sender<serde_json::Value>>>,
}

impl Stream {
    fn new(id: &str) -> Arc<Self> {
        Arc::new(Self {
            id: id.to_string(),
            info: RwLock::new(GatewayInfo::default()),
            playing: AtomicBool::new(false),
            playback_gen: AtomicU64::new(0),
            session: Mutex::new(None),
            bridge_cmd_tx: Mutex::new(None),
        })
    }
}

#[derive(Clone)]
struct AppState {
    /// Карта потоков по stream_id. Создаётся лениво при первом /api/open|/api/streams.
    streams: Arc<RwLock<HashMap<String, Arc<Stream>>>>,
    events: Arc<Mutex<VecDeque<serde_json::Value>>>,
    root: PathBuf,
    bridge_addr: String,
    ws_tx: broadcast::Sender<String>,
    lib: Option<Arc<IntegraLib>>,
    infer_worker_addr: Option<String>,
    started_at: Instant,
    snapshots_dir: PathBuf,
    /// `config.json` → `ui.show_persons`: рисовать ли bbox людей на MJPEG (ложные person на фоне иначе мешают).
    preview_draw_person_boxes: Arc<AtomicBool>,
    preview_max_long_edge: Arc<AtomicU32>,
    preview_jpeg_quality: Arc<AtomicU32>,
    preview_encode_max_fps: Arc<AtomicU32>,
    metrics_history: Arc<Mutex<VecDeque<serde_json::Value>>>,
}

/// stream_id по умолчанию, когда клиент не указал явно.
const DEFAULT_STREAM_ID: &str = "cam1";

fn sid(q: &StreamQuery) -> String {
    q.stream_id
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_STREAM_ID.to_string())
}

impl AppState {
    /// Получить (создать при отсутствии) поток по id.
    async fn stream(&self, id: &str) -> Arc<Stream> {
        if let Some(s) = self.streams.read().await.get(id) {
            return s.clone();
        }
        let mut w = self.streams.write().await;
        w.entry(id.to_string())
            .or_insert_with(|| Stream::new(id))
            .clone()
    }

    /// Получить поток только если он уже существует.
    async fn try_stream(&self, id: &str) -> Option<Arc<Stream>> {
        self.streams.read().await.get(id).cloned()
    }

    /// Снимок всех потоков (для тикера статуса / api_streams).
    async fn all_streams(&self) -> Vec<Arc<Stream>> {
        self.streams.read().await.values().cloned().collect()
    }
}

#[derive(Debug, Deserialize)]
struct OpenRequest {
    path: String,
    #[allow(dead_code)]
    stream_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SeekRequest {
    frame: i64,
}

#[derive(Debug, Deserialize)]
struct LimitQuery {
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct StreamQuery {
    #[allow(dead_code)]
    stream_id: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct PreviewSettings {
    max_long_edge: u32,
    jpeg_quality: u8,
    max_fps: u32,
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,backend_gateway=info".to_string()),
        )
        .init();

    let host = std::env::var("INTEGRA_BACKEND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("INTEGRA_BACKEND_PORT")
        .ok()
        .and_then(|x| x.parse::<u16>().ok())
        .unwrap_or(8000);
    let addr: SocketAddr = format!("{host}:{port}").parse()?;

    let root = std::env::var("INTEGRA_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);

    let infer_worker_addr = std::env::var("INTEGRA_INFER_WORKER_ADDR").ok();

    // Загружаем integra_ffi.dll/.so. Если задан infer_worker — FFI в нём, gateway без FFI.
    let lib: Option<Arc<IntegraLib>> = if infer_worker_addr.is_some() {
        info!("infer_worker mode: FFI not loaded in gateway");
        None
    } else {
        match integra::ffi::global_lib() {
            Ok(l) => {
                info!(version = %l.version(), "integra_ffi loaded");
                Some(l)
            }
            Err(e) => {
                error!(error = %e, "failed to load integra_ffi; build native/integra_ffi or set INTEGRA_FFI_PATH");
                return Err(anyhow::anyhow!("integra_ffi not loaded: {e}"));
            }
        }
    };

    let (ws_tx, _) = broadcast::channel::<String>(128);

    let snapshots_dir = root.join("logs").join("snapshots").join("main");
    if let Err(e) = fs::create_dir_all(&snapshots_dir) {
        warn!(error = %e, dir = %snapshots_dir.display(), "failed to create snapshots dir");
    }

    let cfg0 = read_config_json(&root).unwrap_or_else(|| serde_json::json!({}));
    let preview_draw_person_boxes = Arc::new(AtomicBool::new(
        cfg0.get("ui")
            .and_then(|u| u.get("show_persons"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    ));
    let preview_settings = preview_settings_from_config(&cfg0);
    let preview_max_long_edge = Arc::new(AtomicU32::new(preview_settings.max_long_edge));
    let preview_jpeg_quality = Arc::new(AtomicU32::new(preview_settings.jpeg_quality as u32));
    let preview_encode_max_fps = Arc::new(AtomicU32::new(preview_settings.max_fps));

    let state = AppState {
        streams: Arc::new(RwLock::new(HashMap::new())),
        events: Arc::new(Mutex::new(VecDeque::with_capacity(1024))),
        root,
        bridge_addr: std::env::var("INTEGRA_VIDEO_BRIDGE_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:9876".to_string()),
        ws_tx,
        lib,
        infer_worker_addr,
        started_at: Instant::now(),
        snapshots_dir,
        preview_draw_person_boxes,
        preview_max_long_edge,
        preview_jpeg_quality,
        preview_encode_max_fps,
        metrics_history: Arc::new(Mutex::new(VecDeque::with_capacity(180))),
    };

    // Периодическая рассылка `status` через WS (как Python делал по таймеру).
    spawn_ws_status_ticker(state.clone());

    let app = Router::new()
        .route("/health", get(health))
        .route("/", get(root_index))
        .route("/static/*path", get(static_asset))
        .route("/logs/*path", get(logs_asset))
        .route("/api/info", get(api_info))
        .route("/api/open", post(api_open))
        .route("/api/play", post(api_play))
        .route("/api/pause", post(api_pause))
        .route("/api/seek", post(api_seek))
        .route("/api/metrics", get(api_metrics))
        .route("/api/files", get(api_files))
        .route("/api/streams", get(api_streams).post(api_create_stream))
        .route("/api/close_stream", post(api_close_stream))
        .route("/api/settings", get(api_settings).put(api_put_settings))
        .route("/api/events", get(api_events).delete(api_clear_events))
        .route("/video_snapshot", get(video_snapshot))
        .route("/video_feed", get(video_feed))
        .route("/ws", get(ws_handler))
        .with_state(state);

    info!(%addr, "backend-gateway listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Static / index / health.
// ---------------------------------------------------------------------------

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status":"ok","service":"backend-rust"}))
}

async fn root_index(State(state): State<AppState>) -> impl IntoResponse {
    let index = state.root.join("static").join("index.html");
    match fs::read_to_string(index) {
        Ok(s) => (
            StatusCode::OK,
            [("content-type", "text/html; charset=utf-8")],
            s,
        )
            .into_response(),
        Err(_) => (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            "backend-rust is running".to_string(),
        )
            .into_response(),
    }
}

async fn static_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    serve_asset(&state.root.join("static"), &path).await
}

async fn logs_asset(
    AxumPath(path): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    // Папка snapshots может ещё не существовать (Rust бэкенд их пока не пишет).
    // Отвечаем 404 — frontend это переживёт (placeholder/alt).
    serve_asset(&state.root.join("logs"), &path).await
}

async fn serve_asset(root: &FsPath, rel_path: &str) -> axum::response::Response {
    let rel = rel_path.trim_start_matches('/');
    if rel.is_empty() {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let file_path = root.join(rel);
    let canon = match file_path.canonicalize() {
        Ok(p) => p,
        Err(_) => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    let root_canon = match root.canonicalize() {
        Ok(p) => p,
        Err(_) => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    if !canon.starts_with(&root_canon) {
        return (StatusCode::FORBIDDEN, "forbidden").into_response();
    }
    let body = match fs::read(&canon) {
        Ok(b) => b,
        Err(_) => return (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    let ct = content_type_for_path(&canon);
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, ct),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}

fn content_type_for_path(path: &FsPath) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

// ---------------------------------------------------------------------------
// /api/info, /api/metrics, /api/files, /api/streams, /api/settings, /api/events.
// ---------------------------------------------------------------------------

async fn api_info(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    match state.try_stream(&id).await {
        Some(st) => Json(info_payload(&id, &*st.info.read().await)),
        None => Json(info_payload(&id, &GatewayInfo::default())),
    }
}

fn info_payload(stream_id: &str, info: &GatewayInfo) -> serde_json::Value {
    serde_json::json!({
        "stream_id": stream_id,
        "loaded": info.loaded,
        "playing": info.playing,
        "video_path": info.video_path,
        "fps": info.fps,
        "frame_count": info.frame_count,
        "current_frame": info.current_frame,
        "duration_sec": if info.fps > 0.0 { (info.frame_count as f64) / info.fps } else { 0.0 },
        "current_sec": if info.fps > 0.0 { (info.current_frame as f64) / info.fps } else { 0.0 },
        "width": info.width,
        "height": info.height,
        "stats": stats_payload(&info.stats),
        "tracks": tracks_payload(&info.tracks),
        "persons": persons_payload(&info.persons),
    })
}

fn persons_payload(persons: &[PersonDet]) -> serde_json::Value {
    serde_json::Value::Array(
        persons
            .iter()
            .map(|p| {
                serde_json::json!({
                    "track_id": p.track_id,
                    "confidence": p.confidence,
                    "bbox": p.bbox,
                })
            })
            .collect(),
    )
}

fn stats_payload(s: &PipelineStats) -> serde_json::Value {
    serde_json::json!({
        "decoded": s.frames_processed,
        "inferred": s.frames_processed,
        "events": s.events_total,
        "inference_ms_avg": s.infer_ms,
        "stage_preprocess_ms_ema": s.preprocess_ms,
        "stage_infer_ms_ema": s.infer_ms,
        "stage_tracker_ms_ema": s.tracker_ms,
        "stage_analyzer_ms_ema": s.analyzer_ms,
        "native_enabled": true,
        "native_failures": 0,
        "native_fallbacks": 0,
        "scheduler_mode": "runtime-core-ffi",
        "detections": s.detections,
        "persons": s.persons,
        "decode_fps": s.video_bridge_fps,
        "render_fps": s.video_bridge_fps,
        "pipeline_fps": s.pipeline_process_fps,
        "dropped_decode": 0,
        "dropped_render": 0,
    })
}

/// Первая строка CSV от nvidia-smi (GPU 0). Поля справа — числовые; имя может содержать запятые.
fn gpu_metrics_from_nvidia_smi() -> Option<serde_json::Value> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();
    
    match output {
        Ok(out) => {
            if !out.status.success() {
                warn!("nvidia-smi returned non-zero status");
                return None;
            }
            
            let stdout = match String::from_utf8(out.stdout) {
                Ok(s) => s,
                Err(e) => {
                    warn!("nvidia-smi output is not valid UTF-8: {}", e);
                    return None;
                }
            };
            
            let line = stdout.lines().next()?.trim();
            if line.is_empty() {
                warn!("nvidia-smi output is empty");
                return None;
            }
            
            let parts: Vec<&str> = line
                .split(',')
                .map(|s| s.trim().trim_matches('"'))
                .collect();
                
            if parts.len() < 4 {
                warn!("nvidia-smi output has unexpected format: expected >= 4 fields, got {}", parts.len());
                return None;
            }
            
            let n = parts.len();
            let mem_total_mib: u64 = match parts[n - 1].parse() {
                Ok(v) => v,
                Err(e) => {
                    warn!("Failed to parse GPU memory total: {}", e);
                    return None;
                }
            };
            let mem_used_mib: u64 = match parts[n - 2].parse() {
                Ok(v) => v,
                Err(e) => {
                    warn!("Failed to parse GPU memory used: {}", e);
                    return None;
                }
            };
            let util: f64 = match parts[n - 3].parse() {
                Ok(v) => v,
                Err(e) => {
                    warn!("Failed to parse GPU utilization: {}", e);
                    return None;
                }
            };
            
            let name = parts[..n - 3].join(", ");
            const MIB: u64 = 1024 * 1024;
            
            Some(serde_json::json!({
                "available": true,
                "name": name,
                "util_percent": util,
                "memory_used_bytes": mem_used_mib.saturating_mul(MIB),
                "memory_total_bytes": mem_total_mib.saturating_mul(MIB),
            }))
        }
        Err(e) => {
            warn!("Failed to run nvidia-smi: {}", e);
            None
        }
    }
}

fn tracks_payload(tracks: &[TrackSnapshot]) -> serde_json::Value {
    serde_json::Value::Array(
        tracks
            .iter()
            .map(|t| {
                serde_json::json!({
                    "id": t.id,
                    "cls": t.cls,
                    "state": t.state,
                    "bbox": t.bbox,
                    "conf": t.conf,
                    "static_for_sec": t.static_for_sec,
                    "unattended_for_sec": t.unattended_for_sec,
                    "alarm": t.alarm,
                })
            })
            .collect(),
    )
}

async fn api_metrics(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    let info_clone = match state.try_stream(&id).await {
        Some(st) => st.info.read().await.clone(),
        None => GatewayInfo::default(),
    };

    // Системные метрики через sysinfo (синхронные, держим минимум информации).
    let mut sys = System::new_all();
    sys.refresh_all();
    // Refresh CPU дважды — sysinfo требует это для корректного % (между измерениями
    // ждать MINIMUM_CPU_UPDATE_INTERVAL = 200ms). Делаем blocking sleep на пару мс,
    // чтобы не залипать: погрешность приемлема.
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let proc_info = sys.process(pid);
    let rss = proc_info.map(|p| p.memory()).unwrap_or(0); // bytes
    let cpu_pct = proc_info.map(|p| p.cpu_usage()).unwrap_or(0.0);

    let uptime_sec = state.started_at.elapsed().as_secs_f64();
    let latest_jpeg_bytes = info_clone.latest_jpeg.as_ref().map(|j| j.len()).unwrap_or(0) as u64;
    let frame_bytes = if info_clone.width > 0 && info_clone.height > 0 {
        (info_clone.width as u64)
            .saturating_mul(info_clone.height as u64)
            .saturating_mul(3)
    } else {
        0
    };
    let cfg = read_config_json(&state.root).unwrap_or_else(|| serde_json::json!({}));
    let memory_warning_bytes = cfg
        .get("pipeline")
        .and_then(|p| p.get("memory_chart_warning_bytes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(419_430_400);
    let memory_critical_bytes = cfg
        .get("pipeline")
        .and_then(|p| p.get("memory_chart_critical_bytes"))
        .and_then(|v| v.as_u64())
        .unwrap_or(524_288_000);
    let preview_max_long_edge = state.preview_max_long_edge.load(Ordering::Relaxed);
    let preview_jpeg_quality = state
        .preview_jpeg_quality
        .load(Ordering::Relaxed)
        .clamp(35, 95);
    let preview_encode_max_fps = state
        .preview_encode_max_fps
        .load(Ordering::Relaxed)
        .clamp(1, 60);

    let mut processes_top_rss: Vec<serde_json::Value> = sys
        .processes()
        .iter()
        .map(|(pid, process)| {
            serde_json::json!({
                "pid": pid.as_u32(),
                "name": process.name(),
                "rss_bytes": process.memory(),
            })
        })
        .filter(|p| p.get("rss_bytes").and_then(|v| v.as_u64()).unwrap_or(0) >= 80 * 1024 * 1024)
        .collect();
    processes_top_rss.sort_by_key(|p| {
        std::cmp::Reverse(p.get("rss_bytes").and_then(|v| v.as_u64()).unwrap_or(0))
    });
    processes_top_rss.truncate(8);

    let rss_history = {
        let mut history = state.metrics_history.lock().await;
        history.push_back(serde_json::json!({
            "t": uptime_sec,
            "rss_total_bytes": rss,
            "rss_pipeline_bytes": rss,
            "rss_ema_bytes": rss,
        }));
        while history.len() > 180 {
            history.pop_front();
        }
        history.iter().cloned().collect::<Vec<_>>()
    };

    Json(serde_json::json!({
        "process": {
            "pid": std::process::id(),
            "rss_bytes": rss,
            "vms_bytes": rss,
            "private_bytes": rss,
            "cpu_percent": cpu_pct,
            "rss_pipeline_bytes": rss,
            "rss_analytics_sum_bytes": rss,
            "rss_ema_bytes": rss,
            "rss_peak_recent_bytes": rss,
            "cuda_memory_allocated_bytes": null,
            "cuda_memory_reserved_bytes": null,
            "preview_memory_bytes_est": info_clone.latest_jpeg.as_ref().map(|j| j.len()).unwrap_or(0),
            "rss_history": [],
            "memory_breakdown": [
                {"kind":"process_rss","label":"backend-gateway","pid": std::process::id(), "bytes": rss},
            ],
            "mode": "rust-gateway",
        },
        "system": {
            "cpu_percent": sys.global_cpu_info().cpu_usage(),
            "ram_used_bytes": sys.used_memory(),
            "ram_total_bytes": sys.total_memory(),
            "ram_percent": if sys.total_memory() > 0 { 100.0 * (sys.used_memory() as f64) / (sys.total_memory() as f64) } else { 0.0 },
            "gpu": gpu_metrics_from_nvidia_smi().unwrap_or_else(|| {
                serde_json::json!({
                    "available": false,
                    "name": null,
                    "util_percent": 0,
                    "memory_used_bytes": 0,
                    "memory_total_bytes": 0,
                })
            }),
            "processes_top_rss": [],
        },
        "pipeline": {
            "stats": stats_payload(&info_clone.stats),
            "video": {
                "path": info_clone.video_path,
                "fps": info_clone.fps,
                "width": info_clone.width,
                "height": info_clone.height,
                "loaded": info_clone.loaded,
                "playing": info_clone.playing,
            },
            "latest_frame": {
                "frame_id": info_clone.stats.last_frame_id,
                "detections": info_clone.stats.detections,
                "persons": info_clone.stats.persons,
                "inference_ms": info_clone.stats.infer_ms,
            },
            "thresholds": {},
            "queues": {},
            "buffers": {},
            "uptime_sec": uptime_sec,
        }
    }))
}

async fn api_files(State(state): State<AppState>) -> impl IntoResponse {
    let data_dir = state.root.join("data");
    let _ = fs::create_dir_all(&data_dir);
    let mut files = Vec::new();
    let allowed = [".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"];
    if let Ok(rd) = fs::read_dir(&data_dir) {
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_file() {
                continue;
            }
            let ext = p
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| format!(".{}", x.to_lowercase()))
                .unwrap_or_default();
            if !allowed.contains(&ext.as_str()) {
                continue;
            }
            if let Ok(md) = e.metadata() {
                files.push(serde_json::json!({
                    "name": p.file_name().and_then(|x| x.to_str()).unwrap_or(""),
                    "path": p.to_string_lossy(),
                    "size_mb": ((md.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0
                }));
            }
        }
    }
    Json(serde_json::json!({"data_dir": data_dir.to_string_lossy(), "files": files}))
}

async fn api_streams(State(state): State<AppState>) -> impl IntoResponse {
    let mut out = Vec::new();
    for st in state.all_streams().await {
        let info = st.info.read().await;
        out.push(serde_json::json!({
            "stream_id": st.id,
            "loaded": info.loaded,
            "playing": info.playing,
            "video_path": info.video_path,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
        }));
    }
    // Стабильный порядок (cam1, cam2, …) — иначе сетка панелей «прыгает».
    out.sort_by(|a, b| {
        a.get("stream_id").and_then(|v| v.as_str()).unwrap_or("")
            .cmp(b.get("stream_id").and_then(|v| v.as_str()).unwrap_or(""))
    });
    Json(serde_json::json!({ "streams": out }))
}

#[derive(Debug, Deserialize)]
struct CreateStreamRequest {
    stream_id: Option<String>,
}

/// Создать (зарегистрировать) пустой поток — фронт зовёт при «+ Добавить поток».
async fn api_create_stream(
    State(state): State<AppState>,
    Json(req): Json<CreateStreamRequest>,
) -> impl IntoResponse {
    let id = req
        .stream_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_STREAM_ID.to_string());
    let _ = state.stream(&id).await; // get-or-create
    Json(serde_json::json!({"ok": true, "stream_id": id}))
}

/// Закрыть и удалить поток (остановить мост/сессию).
async fn api_close_stream(
    State(state): State<AppState>,
    Json(req): Json<CreateStreamRequest>,
) -> impl IntoResponse {
    let id = req
        .stream_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_STREAM_ID.to_string());
    if let Some(st) = state.try_stream(&id).await {
        // Останавливаем мост (смена gen → read-loop выходит) и worker-сессию.
        st.playing.store(false, Ordering::SeqCst);
        st.playback_gen.fetch_add(1, Ordering::SeqCst);
        *st.session.lock().await = None;
        *st.bridge_cmd_tx.lock().await = None;
    }
    state.streams.write().await.remove(&id);
    Json(serde_json::json!({"ok": true, "stream_id": id}))
}

async fn api_settings(State(state): State<AppState>) -> impl IntoResponse {
    let cfg = read_config_json(&state.root).unwrap_or_else(|| serde_json::json!({}));
    Json(cfg)
}

async fn api_put_settings(
    State(state): State<AppState>,
    Json(patch): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut cur = read_config_json(&state.root).unwrap_or_else(|| serde_json::json!({}));
    merge_json(&mut cur, &patch);
    if let Err(e) = write_config_json(&state.root, &cur) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to save config: {e}"),
        )
            .into_response();
    }
    state.preview_draw_person_boxes.store(
        cur.get("ui")
            .and_then(|u| u.get("show_persons"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        Ordering::Relaxed,
    );
    let preview_settings = preview_settings_from_config(&cur);
    state
        .preview_max_long_edge
        .store(preview_settings.max_long_edge, Ordering::Relaxed);
    state
        .preview_jpeg_quality
        .store(preview_settings.jpeg_quality as u32, Ordering::Relaxed);
    state
        .preview_encode_max_fps
        .store(preview_settings.max_fps, Ordering::Relaxed);
    Json(cur).into_response()
}

async fn api_events(
    State(state): State<AppState>,
    Query(q): Query<LimitQuery>,
    _stream: Query<StreamQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(200).clamp(1, 1000);
    let ev = state.events.lock().await;
    let n = ev.len();
    let start = n.saturating_sub(limit);
    let slice: Vec<serde_json::Value> = ev.iter().skip(start).cloned().collect();
    Json(serde_json::json!({ "events": slice }))
}

async fn api_clear_events(State(state): State<AppState>, _stream: Query<StreamQuery>) -> impl IntoResponse {
    state.events.lock().await.clear();
    Json(serde_json::json!({"ok":true}))
}

// ---------------------------------------------------------------------------
// /api/open, /api/play, /api/pause, /api/seek.
// ---------------------------------------------------------------------------

/// Разрешить открытие видеофайла по пути. Сервер слушает только `127.0.0.1`,
/// поэтому открываем любой локальный видеофайл по абсолютному пути (ограничение
/// `data/` снято по требованию пользователя). Базовые проверки оставляем: файл
/// существует, это обычный файл, расширение — видео. Относительные пути по-преж-
/// нему резолвятся относительно корня проекта (тайлы из `data/` работают как есть).
/// URL живого потока (RTSP/RTMP/HTTP-MJPEG/…) — отдаём как есть, без файловых
/// проверок: OpenCV/FFmpeg откроет это в video-bridge напрямую.
fn is_stream_url(raw: &str) -> bool {
    let lower = raw.trim().to_ascii_lowercase();
    [
        "rtsp://", "rtsps://", "rtmp://", "rtmps://", "http://", "https://",
        "udp://", "tcp://", "rtp://", "srt://", "mms://",
    ]
    .iter()
    .any(|p| lower.starts_with(p))
}

fn resolve_open_video_path(root: &FsPath, raw: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("path is required".to_string());
    }
    if is_stream_url(raw) {
        return Ok(PathBuf::from(raw));
    }
    let joined = if FsPath::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };
    let canon_file = joined
        .canonicalize()
        .map_err(|_| "video file not found or inaccessible".to_string())?;
    if !canon_file.is_file() {
        return Err("path is not a regular file".to_string());
    }
    let ext_ok = canon_file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_ascii_lowercase().as_str(),
                "mkv" | "mp4" | "avi" | "mov" | "webm" | "m4v"
            )
        })
        .unwrap_or(false);
    if !ext_ok {
        return Err(
            "unsupported file type (expected a video: mkv/mp4/avi/mov/webm/m4v)".to_string(),
        );
    }
    Ok(canon_file)
}

async fn api_open(State(state): State<AppState>, Json(req): Json<OpenRequest>) -> impl IntoResponse {
    if req.path.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "path is required").into_response();
    }
    let abs_path = match resolve_open_video_path(&state.root, &req.path) {
        Ok(p) => p,
        Err(msg) => return (StatusCode::FORBIDDEN, msg).into_response(),
    };
    let path_for_workers = abs_path.to_string_lossy().into_owned();
    let stream_id = req
        .stream_id
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_STREAM_ID.to_string());
    let st = state.stream(&stream_id).await;
    {
        let mut info = st.info.write().await;
        info.loaded = true;
        info.playing = true;
        st.playing.store(true, Ordering::SeqCst);
        info.video_path = Some(path_for_workers.clone());
        info.tracks.clear();
        info.persons.clear();
        info.stats = PipelineStats::default();
    }

    // Стартуем (или переоткрываем) сессию инференса для ЭТОГО потока.
    if let Err(e) = open_pipeline_session(state.clone(), st.clone(), path_for_workers.clone()).await {
        error!(error = %e, "failed to open pipeline session");
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("pipeline open: {e}"))
            .into_response();
    }

    // Подключаемся к video-bridge для получения кадров этого потока.
    start_bridge_stream(state.clone(), st.clone(), path_for_workers.clone());

    push_event(
        &state,
        serde_json::json!({
            "ts": now_secs(),
            "stream_id": stream_id,
            "video_pos_ms": 0.0,
            "type": "system_open",
            "track_id": null,
            "cls_name": "",
            "confidence": null,
            "bbox": [0,0,0,0],
            "note": format!("opened {}", path_for_workers),
        }),
    )
    .await;

    Json(serde_json::json!({"ok":true,"path":path_for_workers,"stream_id":stream_id})).into_response()
}

async fn api_play(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    if let Some(st) = state.try_stream(&id).await {
        st.info.write().await.playing = true;
        st.playing.store(true, Ordering::SeqCst);
        send_bridge_command(&st, serde_json::json!({"cmd": "play"})).await;
    }
    Json(serde_json::json!({"playing":true,"stream_id":id}))
}

async fn api_pause(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    if let Some(st) = state.try_stream(&id).await {
        st.info.write().await.playing = false;
        st.playing.store(false, Ordering::SeqCst);
        send_bridge_command(&st, serde_json::json!({"cmd": "pause"})).await;
    }
    Json(serde_json::json!({"playing":false,"stream_id":id}))
}

async fn api_seek(
    State(state): State<AppState>,
    Query(q): Query<StreamQuery>,
    Json(req): Json<SeekRequest>,
) -> impl IntoResponse {
    let id = sid(&q);
    let mut sent = false;
    if let Some(st) = state.try_stream(&id).await {
        // Optimistic update — UI обновит timeline сразу. Реальная позиция придёт
        // от video-bridge в meta следующего кадра (через ≤ длительность 1 кадра).
        st.info.write().await.current_frame = req.frame.max(0);
        sent = send_bridge_command(
            &st,
            serde_json::json!({"cmd": "seek", "frame": req.frame.max(0)}),
        )
        .await;
    }
    Json(serde_json::json!({"ok": sent, "frame": req.frame.max(0), "stream_id": id}))
}

async fn send_bridge_command(st: &Arc<Stream>, cmd: serde_json::Value) -> bool {
    let tx = {
        let guard = st.bridge_cmd_tx.lock().await;
        guard.as_ref().cloned()
    };
    match tx {
        Some(tx) => tx.send(cmd).is_ok(),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// MJPEG / snapshot.
// ---------------------------------------------------------------------------

async fn video_snapshot(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    let jpeg = match state.try_stream(&id).await {
        Some(st) => st.info.read().await.latest_jpeg.clone(),
        None => None,
    };
    if let Some(jpeg) = jpeg {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")],
            jpeg,
        )
            .into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn video_feed(State(state): State<AppState>, Query(q): Query<StreamQuery>) -> impl IntoResponse {
    let id = sid(&q);
    let body_stream = stream! {
        let boundary = "frame";
        let _last_preview_encode = Instant::now()
            .checked_sub(StdDuration::from_secs(1))
            .unwrap_or_else(Instant::now);
        loop {
            let maybe = match state.try_stream(&id).await {
                Some(st) => st.info.read().await.latest_jpeg.clone(),
                None => None,
            };
            if let Some(jpeg) = maybe {
                let mut chunk = Vec::with_capacity(jpeg.len() + 128);
                chunk.extend_from_slice(
                    format!("--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpeg.len()).as_bytes(),
                );
                chunk.extend_from_slice(&jpeg);
                chunk.extend_from_slice(b"\r\n");
                yield Ok::<Vec<u8>, std::io::Error>(chunk);
            }
            time::sleep(Duration::from_millis(50)).await;
        }
    };
    let body = axum::body::Body::from_stream(body_stream);
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "multipart/x-mixed-replace; boundary=frame"),
            (header::CACHE_CONTROL, "no-store"),
            (header::PRAGMA, "no-cache"),
        ],
        body,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// WebSocket.
// ---------------------------------------------------------------------------

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |sock| ws_loop(sock, state))
}

async fn ws_loop(mut ws: WebSocket, state: AppState) {
    // hello — текущий снэпшот по КАЖДОМУ открытому потоку.
    for st in state.all_streams().await {
        let hello = {
            let info = st.info.read().await;
            serde_json::json!({"type":"hello","stream_id": st.id, "info": info_payload(&st.id, &info)})
                .to_string()
        };
        if ws.send(Message::Text(hello)).await.is_err() {
            return;
        }
    }

    let mut rx = state.ws_tx.subscribe();
    loop {
        let recv = rx.recv().await;
        match recv {
            Ok(s) => {
                if ws.send(Message::Text(s)).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

fn spawn_ws_status_ticker(state: AppState) {
    tokio::spawn(async move {
        let mut tick = time::interval(Duration::from_millis(500));
        loop {
            tick.tick().await;
            // Статус по каждому открытому потоку (фронт раскладывает по панелям).
            for st in state.all_streams().await {
                let payload = {
                    let info = st.info.read().await;
                    serde_json::json!({"type":"status","stream_id": st.id, "info": info_payload(&st.id, &info)})
                        .to_string()
                };
                let _ = state.ws_tx.send(payload);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// FFI pipeline session.
// ---------------------------------------------------------------------------

async fn open_pipeline_session(
    state: AppState,
    st: Arc<Stream>,
    _video_path: String,
) -> anyhow::Result<()> {
    // Берём настройки из config.json (если есть) — иначе разумные дефолты.
    let cfg_json = read_config_json(&state.root).unwrap_or_else(|| serde_json::json!({}));
    let pcfg = build_pipeline_config(&state.root, &cfg_json);

    info!(stream = %st.id, engine = %pcfg.engine_kind, model = %pcfg.model_path, "creating integra pipeline");

    // Каждый поток открывает СВОЁ соединение к infer_worker (воркер — поток на
    // подключение, инференс сериализован общим локом). Так несколько камер
    // обрабатываются параллельно одной моделью.
    let stream_handle = if let Some(ref addr) = state.infer_worker_addr {
        info!(addr = %addr, stream = %st.id, "connecting to infer_worker via TCP");
        integra::spawn_tcp_stream(addr.clone())
            .map_err(|e| anyhow::anyhow!("tcp_stream connect failed: {e}"))?
    } else {
        let lib = state.lib.clone().ok_or_else(|| anyhow::anyhow!("integra_ffi not loaded"))?;
        integra::spawn_stream(pcfg, lib)?
    };
    let integra::StreamHandle {
        frame_tx,
        mut events_rx,
        join: _join,
    } = stream_handle;

    let events_arc = state.events.clone();
    let ws_tx = state.ws_tx.clone();
    let snapshots_dir = state.snapshots_dir.clone();
    let st_task = st.clone();

    let events_task = tokio::spawn(async move {
        let stream_id = st_task.id.clone();
        while let Some(msg) = events_rx.recv().await {
            match msg {
                StreamMessage::Event(ev) => {
                    // Сохраняем snapshot для alarm-событий: последний JPEG из info.latest_jpeg
                    // уже содержит bbox-overlay (отрисован в bridge_worker).
                    let snapshot_path = if is_alarm_event(&ev.kind) {
                        if let Some(ref bytes) = ev.snapshot_jpeg {
                            write_snapshot(&snapshots_dir, &ev.kind, ev.track_id, bytes)
                        } else {
                            let latest = st_task.info.read().await.latest_jpeg.clone();
                            latest.and_then(|jpeg| {
                                write_snapshot(&snapshots_dir, &ev.kind, ev.track_id, &jpeg)
                            })
                        }
                    } else {
                        None
                    };

                    let json_ev = serde_json::json!({
                        "ts": now_secs(),
                        "stream_id": stream_id.clone(),
                        "video_pos_ms": ev.video_pos_ms,
                        "type": ev.kind,
                        "track_id": ev.track_id,
                        "cls_id": ev.cls_id,
                        "cls_name": ev.cls_name,
                        "confidence": ev.confidence,
                        "bbox": ev.bbox,
                        "note": ev.note,
                        "snapshot_path": snapshot_path,
                    });
                    {
                        let mut e = events_arc.lock().await;
                        e.push_back(json_ev.clone());
                        while e.len() > 1000 {
                            e.pop_front();
                        }
                    }
                    {
                        let mut info = st_task.info.write().await;
                        info.stats.events_total = info.stats.events_total.saturating_add(1);
                    }
                    let _ = ws_tx.send(
                        serde_json::json!({"type":"event","stream_id": stream_id.clone(), "event": json_ev})
                            .to_string(),
                    );
                }
                StreamMessage::Frame(fr) => {
                    update_info_from_frame(&st_task.info, &fr).await;
                }
                StreamMessage::Error(e) => {
                    warn!(stream = %stream_id, error = %e, "integra stream error");
                }
            }
        }
        info!(stream = %st_task.id, "integra events consumer stopped");
    });

    let new_session = SessionHandle {
        frame_tx,
        _events_task: events_task,
    };

    // Замена старой сессии этого потока (старая frame_tx уйдёт в drop → worker завершится).
    *st.session.lock().await = Some(new_session);
    Ok(())
}

async fn update_info_from_frame(info_arc: &RwLock<GatewayInfo>, fr: &FrameResult) {
    let mut info = info_arc.write().await;
    info.tracks = fr.tracks.clone();
    info.persons = fr.persons.clone();
    info.stats.last_frame_id = fr.frame_id;
    info.stats.frames_processed = info.stats.frames_processed.saturating_add(1);

    let now = Instant::now();
    if let Some(prev) = info.stats.last_pipeline_frame_at {
        let dt = now.duration_since(prev).as_secs_f64().max(1e-6);
        let inst_fps = 1.0 / dt;
        let a = 0.25_f64;
        info.stats.pipeline_process_fps =
            a * inst_fps + (1.0 - a) * info.stats.pipeline_process_fps;
    }
    info.stats.last_pipeline_frame_at = Some(now);

    info.stats.detections = fr.stats.detections;
    info.stats.persons = fr.stats.persons;
    // EMA по простой формуле α = 0.2.
    let a = 0.2_f64;
    info.stats.infer_ms = a * fr.stats.infer_ms + (1.0 - a) * info.stats.infer_ms;
    info.stats.preprocess_ms = a * fr.stats.preprocess_ms + (1.0 - a) * info.stats.preprocess_ms;
    info.stats.tracker_ms = a * fr.stats.tracker_ms + (1.0 - a) * info.stats.tracker_ms;
    info.stats.analyzer_ms = a * fr.stats.analyzer_ms + (1.0 - a) * info.stats.analyzer_ms;
}

/// COCO-80 (Ultralytics / YOLO): имя класса из `model.class_min_conf` → id для нативного FFI.
fn coco_class_id_from_label(name: &str) -> Option<i32> {
    const LABELS: &[(&str, i32)] = &[
        ("person", 0),
        ("bicycle", 1),
        ("car", 2),
        ("motorcycle", 3),
        ("airplane", 4),
        ("bus", 5),
        ("train", 6),
        ("truck", 7),
        ("boat", 8),
        ("traffic light", 9),
        ("fire hydrant", 10),
        ("stop sign", 11),
        ("parking meter", 12),
        ("bench", 13),
        ("bird", 14),
        ("cat", 15),
        ("dog", 16),
        ("horse", 17),
        ("sheep", 18),
        ("cow", 19),
        ("elephant", 20),
        ("bear", 21),
        ("zebra", 22),
        ("giraffe", 23),
        ("backpack", 24),
        ("umbrella", 25),
        ("handbag", 26),
        ("tie", 27),
        ("suitcase", 28),
        ("frisbee", 29),
        ("skis", 30),
        ("snowboard", 31),
        ("sports ball", 32),
        ("kite", 33),
        ("baseball bat", 34),
        ("baseball glove", 35),
        ("skateboard", 36),
        ("surfboard", 37),
        ("tennis racket", 38),
        ("bottle", 39),
        ("wine glass", 40),
        ("cup", 41),
        ("fork", 42),
        ("knife", 43),
        ("spoon", 44),
        ("bowl", 45),
        ("banana", 46),
        ("apple", 47),
        ("sandwich", 48),
        ("orange", 49),
        ("broccoli", 50),
        ("carrot", 51),
        ("hot dog", 52),
        ("pizza", 53),
        ("donut", 54),
        ("cake", 55),
        ("chair", 56),
        ("couch", 57),
        ("potted plant", 58),
        ("bed", 59),
        ("dining table", 60),
        ("toilet", 61),
        ("tv", 62),
        ("laptop", 63),
        ("mouse", 64),
        ("remote", 65),
        ("keyboard", 66),
        ("cell phone", 67),
        ("microwave", 68),
        ("oven", 69),
        ("toaster", 70),
        ("sink", 71),
        ("refrigerator", 72),
        ("book", 73),
        ("clock", 74),
        ("vase", 75),
        ("scissors", 76),
        ("teddy bear", 77),
        ("hair drier", 78),
        ("toothbrush", 79),
    ];
    LABELS
        .iter()
        .find(|(label, _)| *label == name)
        .map(|(_, id)| *id)
}

fn parse_native_class_min_conf(model: &serde_json::Value) -> Vec<(i32, f32)> {
    let mut out = Vec::new();
    let Some(obj) = model.get("class_min_conf").and_then(|v| v.as_object()) else {
        return out;
    };
    for (name, val) in obj {
        let Some(id) = coco_class_id_from_label(name.as_str()) else {
            continue;
        };
        let Some(th) = val.as_f64() else {
            continue;
        };
        if out.len() >= 16 {
            break;
        }
        out.push((id, th as f32));
    }
    out
}

fn build_pipeline_config(root: &FsPath, cfg: &serde_json::Value) -> integra::PipelineConfig {
    let mut p = integra::PipelineConfig::default();

    // engine: по умолчанию tensorrt; переопределение — `native_analytics.engine` в config.json
    // или переменная окружения INTEGRA_ENGINE_KIND (например `opencv` для отладки без .engine).
    let native = cfg.get("native_analytics");
    let env_engine = std::env::var("INTEGRA_ENGINE_KIND").ok();
    let engine_default = "tensorrt";
    let engine = env_engine
        .or_else(|| {
            native
                .and_then(|n| n.get("engine"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| engine_default.to_string());
    p.engine_kind = engine;

    // model_path: из native_analytics.model_path, относительно root.
    let model_rel = native
        .and_then(|n| n.get("model_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("models/yolo11n_fp16.engine");
    let model_abs = root.join(model_rel);
    p.model_path = model_abs.to_string_lossy().into_owned();

    p.input_size = native
        .and_then(|n| n.get("input_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(640) as i32;

    // postprocess — из model.*
    if let Some(model) = cfg.get("model") {
        p.conf_threshold = model
            .get("conf")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25) as f32;
        p.nms_iou_threshold = model
            .get("iou")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.45) as f32;
        p.person_class_id = model
            .get("person_class")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;
        p.min_box_size_px = model
            .get("min_box_size_px")
            .and_then(|v| v.as_i64())
            .unwrap_or(20) as i32;
        if let Some(arr) = model.get("object_classes").and_then(|v| v.as_array()) {
            p.object_classes = arr
                .iter()
                .filter_map(|v| v.as_i64().map(|x| x as i32))
                .collect();
        }

        // Порт `VisionOCR/.../detector.py`: вертикальные min_conf — только по явному флагу
        // (иначе поля вроде upper_region_y_ratio в JSON сами по себе меняют поведение).
        if model.get("use_native_regional_conf").and_then(|v| v.as_bool()) == Some(true) {
            p.use_regional_class_conf = true;
            p.upper_region_y_ratio = model
                .get("upper_region_y_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.62) as f32;
            p.min_conf_upper = model
                .get("min_conf_upper")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.22) as f32;
            p.min_conf_lower = model
                .get("min_conf_lower")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.30) as f32;
            p.bottom_region_y_ratio = model
                .get("bottom_region_y_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.88) as f32;
            p.min_conf_bottom = model
                .get("min_conf_bottom")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.26) as f32;
            p.border_relax_px = model
                .get("border_relax_px")
                .and_then(|v| v.as_i64())
                .unwrap_or(24) as i32;
            p.min_conf_border = model
                .get("min_conf_border")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.20) as f32;
            p.person_min_conf_border = model
                .get("person_min_conf_border")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.18) as f32;
        }
        p.class_min_conf = parse_native_class_min_conf(model);
    }

    // analyzer — из analyzer.*
    if let Some(a) = cfg.get("analyzer") {
        if let Some(v) = a.get("static_displacement_px").and_then(|v| v.as_f64()) {
            p.static_displacement_px = v;
        }
        if let Some(v) = a.get("static_window_sec").and_then(|v| v.as_f64()) {
            p.static_window_sec = v;
        }
        if let Some(v) = a.get("abandon_time_sec").and_then(|v| v.as_f64()) {
            p.abandon_time_sec = v;
        }
        if let Some(v) = a.get("owner_proximity_px").and_then(|v| v.as_f64()) {
            p.owner_proximity_px = v;
        }
        if let Some(v) = a.get("owner_left_sec").and_then(|v| v.as_f64()) {
            p.owner_left_sec = v;
        }
        if let Some(v) = a.get("disappear_grace_sec").and_then(|v| v.as_f64()) {
            p.disappear_grace_sec = v;
        }
        if let Some(v) = a.get("min_object_area_px").and_then(|v| v.as_f64()) {
            p.min_object_area_px = v;
        }
        if let Some(arr) = a.get("ignore_detection_norm_rect").and_then(|v| v.as_array()) {
            if arr.len() == 4 {
                let mut nums = [0.0f64; 4];
                let mut ok = true;
                for (i, item) in arr.iter().enumerate() {
                    if let Some(x) = item.as_f64() {
                        nums[i] = x;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok && nums[2] > nums[0] && nums[3] > nums[1] {
                    p.ignore_det_norm_x1 = nums[0];
                    p.ignore_det_norm_y1 = nums[1];
                    p.ignore_det_norm_x2 = nums[2];
                    p.ignore_det_norm_y2 = nums[3];
                }
            }
        }
        if let Some(v) = a.get("tracker_iou_match_threshold").and_then(|v| v.as_f64()) {
            p.tracker_iou_match_threshold = v as f32;
        }
        if let Some(v) = a.get("tracker_max_missed_frames").and_then(|v| v.as_i64()) {
            p.tracker_max_missed_frames = v as i32;
        }
        if let Some(v) = a.get("tracker_soft_centroid_match").and_then(|v| v.as_bool()) {
            p.tracker_soft_centroid_match = v;
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Video-bridge consumer: тянет кадры по TCP, кодит JPEG с overlay, пушит в pipeline.
// ---------------------------------------------------------------------------

fn start_bridge_stream(state: AppState, st: Arc<Stream>, video_path: String) {
    let my_gen = st.playback_gen.fetch_add(1, Ordering::SeqCst) + 1;
    let bridge_addr = state.bridge_addr.clone();

    // Канал команд из API-handlers в этот воркер. Воркер пишет JSON-строки в сокет
    // video-bridge; bridge применяет их перед чтением следующего кадра.
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<serde_json::Value>();

    // Регистрируем sender в потоке — старый автоматически перетрётся (его receiver уйдёт
    // в drop вместе с предыдущей spawn_blocking task, которая уже завершилась по playback_gen).
    {
        let st_cmd = st.clone();
        let tx = cmd_tx.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                *st_cmd.bridge_cmd_tx.lock().await = Some(tx);
            });
        });
    }

    tokio::task::spawn_blocking(move || {
        let mut sock = match std::net::TcpStream::connect(&bridge_addr) {
            Ok(s) => s,
            Err(e) => {
                error!(addr=%bridge_addr, stream=%st.id, error=%e, "failed to connect to video-bridge");
                st.playing.store(false, Ordering::SeqCst);
                let st_err = st.clone();
                tokio::runtime::Handle::current().spawn(async move {
                    st_err.info.write().await.playing = false;
                });
                return;
            }
        };
        let _ = sock.set_nodelay(true);
        let _ = sock.set_read_timeout(Some(StdDuration::from_millis(1500)));
        let open = serde_json::json!({
            "cmd": "open",
            "path": video_path,
            "prefer_hw_decode": true
        });
        let line = format!("{open}\n");
        if std::io::Write::write_all(&mut sock, line.as_bytes()).is_err() {
            return;
        }
        let mut reader = match sock.try_clone() {
            Ok(s) => BufReader::new(s),
            Err(_) => return,
        };
        let mut hs = String::new();
        if reader.read_line(&mut hs).ok().unwrap_or(0) == 0 {
            return;
        }
        let hs_v: serde_json::Value = serde_json::from_str(hs.trim()).unwrap_or_default();
        if !hs_v.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            let msg = hs_v
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("video-bridge handshake failed")
                .to_string();
            warn!(stream=%st.id, handshake=%hs.trim(), "video-bridge handshake failed");
            // Сообщаем UI: поток не открылся (главное для диагностики RTSP).
            st.playing.store(false, Ordering::SeqCst);
            let st_err = st.clone();
            let st_state = state.clone();
            let sid_err = st.id.clone();
            tokio::runtime::Handle::current().spawn(async move {
                st_err.info.write().await.playing = false;
                push_event(
                    &st_state,
                    serde_json::json!({
                        "ts": now_secs(),
                        "stream_id": sid_err,
                        "video_pos_ms": 0.0,
                        "type": "system_error",
                        "track_id": null, "cls_name": "", "confidence": null,
                        "bbox": [0,0,0,0],
                        "note": format!("Источник не открылся: {msg}"),
                    }),
                )
                .await;
            });
            return;
        }
        {
            let st_hs = st.clone();
            let fps = hs_v.get("fps").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let frames = hs_v.get("frames").and_then(|v| v.as_i64()).unwrap_or(0);
            let width = hs_v.get("width").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let height = hs_v.get("height").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            tokio::runtime::Handle::current().spawn(async move {
                let mut info = st_hs.info.write().await;
                info.fps = fps;
                info.frame_count = frames;
                info.width = width;
                info.height = height;
            });
        }

        let mut last_preview_encode = Instant::now()
            .checked_sub(StdDuration::from_secs(1))
            .unwrap_or_else(Instant::now);

        // Клонируем frame_tx ОДИН раз: tokio mpsc::Sender::try_send синхронный, так
        // что в цикле кадры уходят воркеру без async-лока на st.session (раньше
        // это был ещё один block_in_place(block_on) на каждый кадр).
        let frame_tx: Option<mpsc::Sender<integra::Frame>> =
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    st.session.lock().await.as_ref().map(|s| s.frame_tx.clone())
                })
            });

        // fps-EMA считаем локально (без лока); в общий state пишем только на такте
        // превью (≤ preview_encode_max_fps) — для UI этого с запасом достаточно.
        let mut local_fps = 0.0_f64;
        let mut last_video_at: Option<Instant> = None;
        // Кодирование превью (downscale+overlay+JPEG, десятки мс на 1080p@q92)
        // вынесено с потока read-loop: раньше оно шло ИНЛАЙН и гейтило приём
        // кадров (decode залипал на ~12fps при 25fps-кодировании). Теперь кодируем
        // в отдельной задаче; guard пропускает такт, если предыдущий кадр ещё
        // кодируется — превью само себя троттлит, а декод идёт в реальном времени.
        let preview_busy = Arc::new(AtomicBool::new(false));

        loop {
            if st.playback_gen.load(Ordering::SeqCst) != my_gen {
                break;
            }

            // Драйним команды от API-handlers и шлём их в video-bridge.
            while let Ok(cmd) = cmd_rx.try_recv() {
                let line = format!("{cmd}\n");
                if let Err(e) = std::io::Write::write_all(&mut sock, line.as_bytes()) {
                    warn!(error = %e, "failed to forward command to video-bridge");
                    break;
                }
                let _ = std::io::Write::flush(&mut sock);
            }
            if !st.playing.load(Ordering::SeqCst) {
                std::thread::sleep(StdDuration::from_millis(30));
                continue;
            }

            let mut meta = String::new();
            if reader.read_line(&mut meta).ok().unwrap_or(0) == 0 {
                break;
            }
            let mv: serde_json::Value = match serde_json::from_str(meta.trim()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let w = mv.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let h = mv.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let pts_ms = mv.get("pos_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let frame_id = mv.get("frame_id").and_then(|v| v.as_i64()).unwrap_or(0);
            if w == 0 || h == 0 {
                continue;
            }
            let mut len_buf = [0_u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut bgr = vec![0_u8; len];
            if reader.read_exact(&mut bgr).is_err() {
                break;
            }
            if bgr.len() != w * h * 3 {
                continue;
            }

            // 0. Скорость приёма кадров с video-bridge (локальный EMA, без лока).
            let now = Instant::now();
            if let Some(prev) = last_video_at {
                let dt = now.duration_since(prev).as_secs_f64().max(1e-6);
                local_fps = 0.25 * (1.0 / dt) + 0.75 * local_fps;
            }
            last_video_at = Some(now);

            // 1. Кадр воркеру (drop-newest): прямой try_send, без async-лока.
            let bgr_shared = Arc::new(bgr);
            if let Some(tx) = &frame_tx {
                let _ = tx.try_send(integra::Frame {
                    bgr: bgr_shared.clone(),
                    width: w as u32,
                    height: h as u32,
                    pts_ms: pts_ms as i64,
                });
            }

            let max_preview_fps = state
                .preview_encode_max_fps
                .load(Ordering::Relaxed)
                .clamp(1, 60);
            let preview_period = StdDuration::from_secs_f64(1.0 / max_preview_fps as f64);
            // Пропускаем такт, если ещё не пришло время ИЛИ предыдущее превью всё
            // ещё кодируется (guard) — read-loop никогда не ждёт энкодер.
            if last_preview_encode.elapsed() < preview_period
                || preview_busy.load(Ordering::Acquire)
            {
                continue;
            }
            last_preview_encode = Instant::now();
            preview_busy.store(true, Ordering::Release);

            // Кодирование вынесено в отдельную задачу: снимок треков (async-read),
            // затем тяжёлый encode на blocking-пуле. read-loop сразу идёт за
            // следующим кадром — декод не зависит от частоты/качества превью.
            // Настройки превью — общие (state), кадр/треки — этого потока (st).
            let st_pv = st.clone();
            let busy = preview_busy.clone();
            let bgr_for_preview = bgr_shared.clone();
            let draw_person_boxes = state.preview_draw_person_boxes.load(Ordering::Relaxed);
            let max_long_edge = state.preview_max_long_edge.load(Ordering::Relaxed);
            let jpeg_quality =
                state.preview_jpeg_quality.load(Ordering::Relaxed).clamp(35, 95) as u8;
            let (pw, ph, pframe, fps_now) = (w as u32, h as u32, frame_id, local_fps);
            tokio::runtime::Handle::current().spawn(async move {
                let (tracks_snapshot, persons_snapshot) = {
                    let g = st_pv.info.read().await;
                    (g.tracks.clone(), g.persons.clone())
                };
                let jpeg = tokio::task::spawn_blocking(move || {
                    encode_preview_jpeg_with_options(
                        &*bgr_for_preview, pw, ph,
                        &tracks_snapshot, &persons_snapshot,
                        draw_person_boxes, max_long_edge, jpeg_quality,
                    )
                })
                .await
                .ok()
                .flatten();
                {
                    let mut info = st_pv.info.write().await;
                    info.current_frame = pframe;
                    info.stats.video_bridge_fps = fps_now;
                    if let Some(jpeg) = jpeg {
                        info.latest_jpeg = Some(jpeg);
                    }
                }
                busy.store(false, Ordering::Release);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Misc helpers.
// ---------------------------------------------------------------------------

async fn push_event(state: &AppState, ev: serde_json::Value) {
    let mut e = state.events.lock().await;
    e.push_back(ev.clone());
    while e.len() > 1000 {
        e.pop_front();
    }
    drop(e);
    let _ = state
        .ws_tx
        .send(serde_json::json!({"type":"event","stream_id":"main","event":ev}).to_string());
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn is_alarm_event(kind: &str) -> bool {
    // Тревожные события ТЗ (object_left / person_interaction — контекстные, без снапшота).
    if matches!(
        kind,
        "object_unattended" | "object_removed" | "object_missing"
    ) {
        return true;
    }
    // Legacy-фоллбэк для старых сборок FFI (alarm_* / abandoned / disappeared).
    kind.starts_with("alarm_") || kind == "abandoned" || kind == "disappeared"
}

fn write_snapshot(dir: &FsPath, kind: &str, track_id: i64, jpeg: &[u8]) -> Option<String> {
    let _ = fs::create_dir_all(dir);
    let ts = chrono_compact_now();
    // Очищаем kind от символов, опасных для FS.
    let safe_kind: String = kind
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    let fname = format!("{safe_kind}_{track_id}_{ts}.jpg");
    let full = dir.join(&fname);
    if let Err(e) = fs::write(&full, jpeg) {
        warn!(error = %e, path = %full.display(), "failed to write snapshot");
        return None;
    }
    // Относительный URL для фронтенда (mount /logs/*).
    Some(format!("/logs/snapshots/main/{fname}"))
}

fn chrono_compact_now() -> String {
    // Формат YYYYMMDD_HHMMSS_mmm — совпадает со схемой Python-логгера.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    // Минимальная конверсия time-since-epoch → UTC компоненты без chrono.
    // (точности секунд достаточно; миллисекунды берём из subsec_millis).
    let (y, mo, d, h, mi, s) = epoch_to_utc(secs);
    format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}_{millis:03}")
}

fn epoch_to_utc(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    // Простейшая прокрутка дней / месяцев с учётом високосных.
    let days_from_epoch = secs / 86400;
    let sec_of_day = secs % 86400;
    let h = (sec_of_day / 3600) as u32;
    let mi = ((sec_of_day % 3600) / 60) as u32;
    let s = (sec_of_day % 60) as u32;
    // 1970-01-01 = day 0.
    let mut year: i32 = 1970;
    let mut days = days_from_epoch as i64;
    loop {
        let yd = if is_leap(year) { 366 } else { 365 };
        if days >= yd {
            days -= yd;
            year += 1;
        } else {
            break;
        }
    }
    let mlen = [31, 28 + if is_leap(year) { 1 } else { 0 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month: u32 = 1;
    for &dl in &mlen {
        if days >= dl {
            days -= dl;
            month += 1;
        } else {
            break;
        }
    }
    let day = (days + 1) as u32;
    (year, month, day, h, mi, s)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn read_config_json(root: &FsPath) -> Option<serde_json::Value> {
    let p = root.join("config.json");
    let raw = fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

fn preview_settings_from_config(cfg: &serde_json::Value) -> PreviewSettings {
    let pipeline = cfg.get("pipeline");
    let max_long_edge = pipeline
        .and_then(|p| p.get("preview_max_long_edge"))
        .and_then(|v| v.as_u64())
        .unwrap_or(960)
        .clamp(320, 1920) as u32;
    let jpeg_quality = pipeline
        .and_then(|p| p.get("preview_jpeg_quality"))
        .and_then(|v| v.as_u64())
        .unwrap_or(70)
        .clamp(35, 95) as u8;
    let max_fps = pipeline
        .and_then(|p| p.get("preview_encode_max_fps"))
        .and_then(|v| v.as_u64())
        .unwrap_or(12)
        .clamp(1, 60) as u32;
    PreviewSettings {
        max_long_edge,
        jpeg_quality,
        max_fps,
    }
}

fn write_config_json(root: &FsPath, v: &serde_json::Value) -> anyhow::Result<()> {
    let p = root.join("config.json");
    let tmp = root.join("config.json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    fs::rename(tmp, p)?;
    Ok(())
}

fn merge_json(dst: &mut serde_json::Value, patch: &serde_json::Value) {
    match (dst, patch) {
        (serde_json::Value::Object(d), serde_json::Value::Object(p)) => {
            for (k, v) in p {
                match d.get_mut(k) {
                    Some(cur) => merge_json(cur, v),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, p) => *d = p.clone(),
    }
}

#[cfg(test)]
mod open_path_tests {
    use super::{fs, resolve_open_video_path};

    #[test]
    fn resolve_accepts_video_under_data_absolute() {
        let tmp = std::env::temp_dir().join(format!("integra_open_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        let vid = tmp.join("data").join("clip.mp4");
        fs::write(&vid, b"x").unwrap();
        let got = resolve_open_video_path(&tmp, vid.to_str().unwrap()).unwrap();
        assert!(got.ends_with("clip.mp4"));
    }

    #[test]
    fn resolve_accepts_relative_data_path() {
        let tmp = std::env::temp_dir().join(format!("integra_open_rel_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        fs::write(tmp.join("data").join("x.webm"), b"x").unwrap();
        let got = resolve_open_video_path(&tmp, "data/x.webm").unwrap();
        assert!(got.ends_with("x.webm"));
    }

    #[test]
    fn resolve_rejects_file_outside_data() {
        let tmp = std::env::temp_dir().join(format!("integra_open_out_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        let outside = tmp.join("outside.mkv");
        fs::write(&outside, b"x").unwrap();
        let abs = outside.canonicalize().unwrap();
        let err = resolve_open_video_path(&tmp, abs.to_str().unwrap()).unwrap_err();
        assert!(err.contains("data/"));
    }

    #[test]
    fn resolve_rejects_relative_traversal_out_of_data() {
        let tmp = std::env::temp_dir().join(format!("integra_open_trav_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        fs::write(tmp.join("secret.txt"), b"no").unwrap();
        fs::write(tmp.join("data").join("ok.mp4"), b"x").unwrap();
        let err = resolve_open_video_path(&tmp, "data/../secret.txt").unwrap_err();
        assert!(err.contains("data/"));
    }
}
