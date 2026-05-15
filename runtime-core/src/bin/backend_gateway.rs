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

use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::process::Command;
use std::net::SocketAddr;
use std::path::{Path as FsPath, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State};
use axum::http::{header, Method, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use sysinfo::System;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::time::{self, Duration};
use tokio::task::JoinHandle;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};

use runtime_core::integra::{
    self, encode_preview_jpeg, FrameResult, IntegraLib, PersonDet, StreamMessage, TrackSnapshot,
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

#[derive(Clone)]
struct AppState {
    info: Arc<RwLock<GatewayInfo>>,
    events: Arc<Mutex<VecDeque<serde_json::Value>>>,
    root: PathBuf,
    bridge_addr: String,
    playback_gen: Arc<AtomicU64>,
    session: Arc<Mutex<Option<SessionHandle>>>,
    ws_tx: broadcast::Sender<String>,
    lib: Option<Arc<IntegraLib>>,
    infer_worker_addr: Option<String>,
    started_at: Instant,
    snapshots_dir: PathBuf,
    /// Sender команд в активный bridge_worker (seek/pause/play/close).
    /// При новом /api/open перетирается на свежий sender; старая bridge_worker-сессия
    /// сама завершится (playback_gen) и её mpsc::Receiver уйдёт в drop.
    bridge_cmd_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<serde_json::Value>>>>,
    /// `config.json` → `ui.show_persons`: рисовать ли bbox людей на MJPEG (ложные person на фоне иначе мешают).
    preview_draw_person_boxes: Arc<AtomicBool>,
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
    let root = match root.canonicalize() {
        Ok(p) => {
            info!(root = %p.display(), "project root (canonical)");
            p
        }
        Err(e) => {
            warn!(
                error = %e,
                path = %root.display(),
                "project root could not be canonicalized; using path as given"
            );
            root
        }
    };

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

    let state = AppState {
        info: Arc::new(RwLock::new(GatewayInfo::default())),
        events: Arc::new(Mutex::new(VecDeque::with_capacity(1024))),
        root,
        bridge_addr: std::env::var("INTEGRA_VIDEO_BRIDGE_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:9876".to_string()),
        playback_gen: Arc::new(AtomicU64::new(0)),
        session: Arc::new(Mutex::new(None)),
        ws_tx,
        lib,
        infer_worker_addr,
        started_at: Instant::now(),
        snapshots_dir,
        bridge_cmd_tx: Arc::new(Mutex::new(None)),
        preview_draw_person_boxes,
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
        .route("/api/upload_video", post(api_upload_video))
        .route("/api/streams", get(api_streams))
        .route("/api/settings", get(api_settings).put(api_put_settings))
        .route("/api/events", get(api_events).delete(api_clear_events))
        .route("/video_snapshot", get(video_snapshot))
        .route("/video_feed", get(video_feed))
        .route("/ws", get(ws_handler))
        .with_state(state)
        // CORS: UI может открываться не с того origin (другой порт / file:// + meta redirect и т.д.).
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([
                    Method::GET,
                    Method::POST,
                    Method::PUT,
                    Method::DELETE,
                    Method::OPTIONS,
                ])
                .allow_headers(Any),
        )
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024));

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

async fn api_info(State(state): State<AppState>, _q: Query<StreamQuery>) -> impl IntoResponse {
    let info = state.info.read().await;
    Json(info_payload(&info))
}

fn info_payload(info: &GatewayInfo) -> serde_json::Value {
    serde_json::json!({
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
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let line = stdout.lines().next()?.trim();
    if line.is_empty() {
        return None;
    }
    let parts: Vec<&str> = line
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    if parts.len() < 4 {
        return None;
    }
    let n = parts.len();
    let mem_total_mib: u64 = parts[n - 1].parse().ok()?;
    let mem_used_mib: u64 = parts[n - 2].parse().ok()?;
    let util: f64 = parts[n - 3].parse().ok()?;
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

async fn api_metrics(State(state): State<AppState>, _q: Query<StreamQuery>) -> impl IntoResponse {
    let info = state.info.read().await;
    let info_clone = info.clone();
    drop(info);

    // Системные метрики через sysinfo (синхронные, держим минимум информации).
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu();
    // Refresh CPU дважды — sysinfo требует это для корректного % (между измерениями
    // ждать MINIMUM_CPU_UPDATE_INTERVAL = 200ms). Делаем blocking sleep на пару мс,
    // чтобы не залипать: погрешность приемлема.
    std::thread::sleep(StdDuration::from_millis(50));
    sys.refresh_cpu();

    let pid = sysinfo::Pid::from_u32(std::process::id());
    sys.refresh_process(pid);
    let proc_info = sys.process(pid);
    let rss = proc_info.map(|p| p.memory()).unwrap_or(0); // bytes
    let cpu_pct = proc_info.map(|p| p.cpu_usage()).unwrap_or(0.0);

    let uptime_sec = state.started_at.elapsed().as_secs_f64();

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
                let fname = p.file_name().and_then(|x| x.to_str()).unwrap_or("");
                files.push(serde_json::json!({
                    "name": fname,
                    "path": format!("data/{fname}"),
                    "size_mb": ((md.len() as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0
                }));
            }
        }
    }
    Json(serde_json::json!({"data_dir": data_dir.to_string_lossy(), "files": files}))
}

/// Имя файла из multipart + расширение из белого списка (как в `api_files`).
fn sanitize_upload_filename(raw: &str) -> Option<String> {
    let base = FsPath::new(raw).file_name()?.to_str()?;
    if base.is_empty() || base.contains('\0') {
        return None;
    }
    if base.len() > 240 {
        return None;
    }
    let ext = FsPath::new(base)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())?;
    let allowed = ["mkv", "mp4", "avi", "mov", "webm", "m4v"];
    if !allowed.contains(&ext.as_str()) {
        return None;
    }
    Some(base.to_string())
}

/// POST multipart, поле `file` — сохранение в `<project>/data/` (как ожидает `static/app.js`).
async fn api_upload_video(State(state): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
    let data_dir = state.root.join("data");
    if let Err(e) = fs::create_dir_all(&data_dir) {
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("data/: {e}")).into_response();
    }

    while let Some(mut field) = match multipart.next_field().await {
        Ok(f) => f,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("multipart: {e}")).into_response(),
    } {
        if field.name() != Some("file") {
            continue;
        }
        let orig_name = field.file_name().map(|s| s.to_string()).unwrap_or_default();
        let safe_name = if orig_name.trim().is_empty() {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            format!("upload_{ts}.mkv")
        } else {
            match sanitize_upload_filename(&orig_name) {
                Some(s) => s,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        "invalid file name or extension (allowed: .mkv .mp4 .avi .mov .webm .m4v)",
                    )
                        .into_response();
                }
            }
        };

        let mut dest = data_dir.join(&safe_name);
        if dest.exists() {
            let stem = FsPath::new(&safe_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("video");
            let ext = FsPath::new(&safe_name)
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("mkv");
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            dest = data_dir.join(format!("{stem}_{ts}.{ext}"));
        }

        let mut outfile = match tokio::fs::File::create(&dest).await {
            Ok(f) => f,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, format!("create: {e}")).into_response(),
        };
        let mut total: u64 = 0;
        const MAX: u64 = 512 * 1024 * 1024;
        loop {
            match field.chunk().await {
                Ok(Some(chunk)) => {
                    total += chunk.len() as u64;
                    if total > MAX {
                        let _ = tokio::fs::remove_file(&dest).await;
                        return (
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "file too large (max 512 MiB)",
                        )
                            .into_response();
                    }
                    if let Err(e) = tokio::io::AsyncWriteExt::write_all(&mut outfile, &chunk).await {
                        let _ = tokio::fs::remove_file(&dest).await;
                        return (StatusCode::INTERNAL_SERVER_ERROR, format!("write: {e}")).into_response();
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tokio::fs::remove_file(&dest).await;
                    return (StatusCode::BAD_REQUEST, format!("read: {e}")).into_response();
                }
            }
        }
        if total == 0 {
            let _ = tokio::fs::remove_file(&dest).await;
            return (StatusCode::BAD_REQUEST, "empty file").into_response();
        }

        let name = dest
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(safe_name.as_str())
            .to_string();

        info!(%name, bytes = total, "upload_video saved");

        let rel_path = format!("data/{name}");

        return Json(serde_json::json!({
            "ok": true,
            "path": rel_path,
            "name": name,
        }))
        .into_response();
    }

    (StatusCode::BAD_REQUEST, "missing file field").into_response()
}

async fn api_streams(State(state): State<AppState>) -> impl IntoResponse {
    let info = state.info.read().await;
    Json(serde_json::json!({
        "streams": [{
            "stream_id": "main",
            "loaded": info.loaded,
            "playing": info.playing,
            "video_path": info.video_path,
        }]
    }))
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

/// Каталог `data_root` — префикс пути `file` (оба пути в одном виде компонентов).
fn path_is_under_data_root(data_root: &FsPath, file: &FsPath) -> bool {
    let mut dr = data_root.components();
    let mut fc = file.components();
    loop {
        match (dr.next(), fc.next()) {
            (None, None) => return true,
            (None, Some(_)) => return true,
            (Some(_), None) => return false,
            (Some(a), Some(b)) if a == b => {}
            _ => return false,
        }
    }
}

/// Каталог `data/` (канонический) содержит файл `path` (канонический).
/// Дополнительно: сравнение по строке с разделителем — иногда надёжнее, чем только `components()`.
fn file_is_inside_data_dir(data_dir: &FsPath, path: &FsPath) -> bool {
    let Ok(dc) = data_dir.canonicalize() else {
        return false;
    };
    let Ok(pc) = path.canonicalize() else {
        return false;
    };
    if path_is_under_data_root(&dc, &pc) {
        return true;
    }
    let d = dc.to_string_lossy();
    let p = pc.to_string_lossy();
    let sep = std::path::MAIN_SEPARATOR;
    let d_trim = d.trim_end_matches(sep);
    let p_trim = p.trim_end_matches(sep);
    p_trim == d_trim || p_trim.starts_with(&format!("{d_trim}{sep}"))
}

/// Безопасное имя файла (без `/`, `\`, `..`) для поиска только в `<root>/data/`.
fn safe_data_basename(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    if raw.contains('/') || raw.contains('\\') || raw.contains("..") {
        return None;
    }
    let name = FsPath::new(raw).file_name()?.to_str()?.to_string();
    if name != raw {
        return None;
    }
    let ext = FsPath::new(&name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())?;
    let allowed = ["mkv", "mp4", "avi", "mov", "webm", "m4v"];
    if !allowed.contains(&ext.as_str()) {
        return None;
    }
    Some(name)
}

/// Разрешить открытие только видеофайлов внутри `<project>/data/` (после `canonicalize`).
/// Иначе клиент с доступом к HTTP API может передать абсолютный путь и заставить
/// video-bridge читать произвольные файлы на машине с бэкендом.
fn resolve_open_video_path(root: &FsPath, raw: &str) -> Result<PathBuf, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("path is required".to_string());
    }

    let root_candidates: Vec<PathBuf> = {
        let mut v = vec![root.to_path_buf()];
        match root.canonicalize() {
            Ok(c) if c.as_path() != root => v.push(c),
            _ => {}
        }
        v
    };

    let joined = if FsPath::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };

    match joined.canonicalize() {
        Ok(canon_file) => {
            for r in &root_candidates {
                let data_dir = r.join("data");
                let _ = fs::create_dir_all(&data_dir);
                if let Ok(canon_data) = data_dir.canonicalize() {
                    if file_is_inside_data_dir(&canon_data, &canon_file) && canon_file.is_file() {
                        return Ok(canon_file);
                    }
                }
            }
        }
        Err(_) if FsPath::new(raw).is_absolute() => {
            return Err("video file not found or inaccessible".to_string());
        }
        Err(_) => {}
    }

    // 2) Только имя файла: `clip.mkv` → единственный кандидат в `<root>/data/clip.mkv`.
    if let Some(base) = safe_data_basename(raw) {
        for r in &root_candidates {
            let candidate = r.join("data").join(&base);
            if let Ok(canon_file) = candidate.canonicalize() {
                let data_dir = r.join("data");
                let _ = fs::create_dir_all(&data_dir);
                if let Ok(canon_data) = data_dir.canonicalize() {
                    if file_is_inside_data_dir(&canon_data, &canon_file) && canon_file.is_file() {
                        return Ok(canon_file);
                    }
                }
            }
        }
    }

    Err("video path must be inside the project data/ directory".to_string())
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
    {
        let mut info = state.info.write().await;
        info.loaded = true;
        info.playing = true;
        info.video_path = Some(path_for_workers.clone());
        info.tracks.clear();
        info.persons.clear();
        info.stats = PipelineStats::default();
    }

    // Стартуем (или переоткрываем) FFI сессию.
    if let Err(e) = open_pipeline_session(state.clone(), path_for_workers.clone()).await {
        error!(error = %e, "failed to open pipeline session");
        return (StatusCode::INTERNAL_SERVER_ERROR, format!("pipeline open: {e}"))
            .into_response();
    }

    // Подключаемся к video-bridge для получения кадров.
    start_bridge_stream(state.clone(), path_for_workers.clone());

    push_event(
        &state,
        serde_json::json!({
            "ts": now_secs(),
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

    Json(serde_json::json!({"ok":true,"path":path_for_workers})).into_response()
}

async fn api_play(State(state): State<AppState>) -> impl IntoResponse {
    {
        let mut info = state.info.write().await;
        info.playing = true;
    }
    send_bridge_command(&state, serde_json::json!({"cmd": "play"})).await;
    Json(serde_json::json!({"playing":true}))
}

async fn api_pause(State(state): State<AppState>) -> impl IntoResponse {
    {
        let mut info = state.info.write().await;
        info.playing = false;
    }
    send_bridge_command(&state, serde_json::json!({"cmd": "pause"})).await;
    Json(serde_json::json!({"playing":false}))
}

async fn api_seek(State(state): State<AppState>, Json(req): Json<SeekRequest>) -> impl IntoResponse {
    // Optimistic update — UI обновит timeline сразу. Реальная позиция придёт
    // от video-bridge в meta следующего кадра (через ≤ длительность 1 кадра).
    {
        let mut info = state.info.write().await;
        info.current_frame = req.frame.max(0);
    }
    let sent = send_bridge_command(
        &state,
        serde_json::json!({"cmd": "seek", "frame": req.frame.max(0)}),
    )
    .await;
    Json(serde_json::json!({"ok": sent, "frame": req.frame.max(0)}))
}

async fn send_bridge_command(state: &AppState, cmd: serde_json::Value) -> bool {
    let tx = {
        let guard = state.bridge_cmd_tx.lock().await;
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

async fn video_snapshot(State(state): State<AppState>, _stream: Query<StreamQuery>) -> impl IntoResponse {
    let info = state.info.read().await;
    if let Some(jpeg) = info.latest_jpeg.clone() {
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")],
            jpeg,
        )
            .into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn video_feed(State(state): State<AppState>, _stream: Query<StreamQuery>) -> impl IntoResponse {
    let body_stream = stream! {
        let boundary = "frame";
        loop {
            let maybe = {
                let info = state.info.read().await;
                info.latest_jpeg.clone()
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
    // hello — текущий снэпшот.
    let hello = {
        let info = state.info.read().await;
        serde_json::json!({"type":"hello","stream_id":"main","info": info_payload(&info)}).to_string()
    };
    if ws.send(Message::Text(hello)).await.is_err() {
        return;
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
            // Не пушим тяжёлые статусы, если нет подписчиков (broadcast::send отдаст Err).
            let payload = {
                let info = state.info.read().await;
                serde_json::json!({"type":"status","stream_id":"main","info": info_payload(&info)})
                    .to_string()
            };
            let _ = state.ws_tx.send(payload);
        }
    });
}

// ---------------------------------------------------------------------------
// FFI pipeline session.
// ---------------------------------------------------------------------------

async fn open_pipeline_session(state: AppState, _video_path: String) -> anyhow::Result<()> {
    // Берём настройки из config.json (если есть) — иначе разумные дефолты.
    let cfg_json = read_config_json(&state.root).unwrap_or_else(|| serde_json::json!({}));
    let pcfg = build_pipeline_config(&state.root, &cfg_json);

    info!(engine = %pcfg.engine_kind, model = %pcfg.model_path, "creating integra pipeline");

    let stream_handle = if let Some(ref addr) = state.infer_worker_addr {
        info!(addr = %addr, "connecting to infer_worker via TCP");
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

    let info_arc = state.info.clone();
    let events_arc = state.events.clone();
    let ws_tx = state.ws_tx.clone();
    let snapshots_dir = state.snapshots_dir.clone();

    let events_task = tokio::spawn(async move {
        while let Some(msg) = events_rx.recv().await {
            match msg {
                StreamMessage::Event(ev) => {
                    // Сохраняем snapshot для alarm-событий: последний JPEG из info.latest_jpeg
                    // уже содержит bbox-overlay (отрисован в bridge_worker).
                    let snapshot_path = if is_alarm_event(&ev.kind) {
                        if let Some(ref bytes) = ev.snapshot_jpeg {
                            write_snapshot(&snapshots_dir, &ev.kind, ev.track_id, bytes)
                        } else {
                            let latest = info_arc.read().await.latest_jpeg.clone();
                            latest.and_then(|jpeg| {
                                write_snapshot(&snapshots_dir, &ev.kind, ev.track_id, &jpeg)
                            })
                        }
                    } else {
                        None
                    };

                    let json_ev = serde_json::json!({
                        "ts": now_secs(),
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
                        let mut info = info_arc.write().await;
                        info.stats.events_total = info.stats.events_total.saturating_add(1);
                    }
                    let _ = ws_tx.send(
                        serde_json::json!({"type":"event","stream_id":"main","event":json_ev})
                            .to_string(),
                    );
                }
                StreamMessage::Frame(fr) => {
                    update_info_from_frame(&info_arc, &fr).await;
                }
                StreamMessage::Error(e) => {
                    warn!(error = %e, "integra stream error");
                }
            }
        }
        info!("integra events consumer stopped");
    });

    let new_session = SessionHandle {
        frame_tx,
        _events_task: events_task,
    };

    // Замена старой сессии (старая frame_tx уйдёт в drop → spawn_stream worker завершится).
    let mut slot = state.session.lock().await;
    *slot = Some(new_session);
    Ok(())
}

async fn update_info_from_frame(info_arc: &Arc<RwLock<GatewayInfo>>, fr: &FrameResult) {
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

fn start_bridge_stream(state: AppState, video_path: String) {
    let my_gen = state.playback_gen.fetch_add(1, Ordering::SeqCst) + 1;
    let bridge_addr = state.bridge_addr.clone();

    // Канал команд из API-handlers в этот воркер. Воркер пишет JSON-строки в сокет
    // video-bridge; bridge применяет их перед чтением следующего кадра.
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<serde_json::Value>();

    // Регистрируем sender в AppState — старый автоматически перетрётся (его receiver уйдёт
    // в drop вместе с предыдущей spawn_blocking task, которая уже завершилась по playback_gen).
    {
        let state_cmd = state.bridge_cmd_tx.clone();
        let tx = cmd_tx.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                *state_cmd.lock().await = Some(tx);
            });
        });
    }

    tokio::task::spawn_blocking(move || {
        let mut sock = match std::net::TcpStream::connect(&bridge_addr) {
            Ok(s) => s,
            Err(e) => {
                error!(addr=%bridge_addr, error=%e, "failed to connect to video-bridge");
                let info = state.info.clone();
                tokio::runtime::Handle::current().spawn(async move {
                    info.write().await.playing = false;
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
            warn!(handshake=%hs.trim(), "video-bridge handshake failed");
            return;
        }
        {
            let info = state.info.clone();
            let fps = hs_v.get("fps").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let frames = hs_v.get("frames").and_then(|v| v.as_i64()).unwrap_or(0);
            let width = hs_v.get("width").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let height = hs_v.get("height").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            tokio::runtime::Handle::current().spawn(async move {
                let mut info = info.write().await;
                info.fps = fps;
                info.frame_count = frames;
                info.width = width;
                info.height = height;
            });
        }

        loop {
            if state.playback_gen.load(Ordering::SeqCst) != my_gen {
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
            let playing = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    state.info.read().await.playing
                })
            });
            if !playing {
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

            // 0. Скорость приёма кадров с video-bridge (плавность видео; не путать с pipeline_process_fps).
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut info = state.info.write().await;
                    let now = Instant::now();
                    if let Some(prev) = info.stats.last_video_frame_at {
                        let dt = now.duration_since(prev).as_secs_f64().max(1e-6);
                        let inst_fps = 1.0 / dt;
                        let a = 0.25_f64;
                        info.stats.video_bridge_fps =
                            a * inst_fps + (1.0 - a) * info.stats.video_bridge_fps;
                    }
                    info.stats.last_video_frame_at = Some(now);
                });
            });

            let bgr_shared = Arc::new(bgr);
            push_frame_to_pipeline(
                &state,
                bgr_shared.clone(),
                w as u32,
                h as u32,
                pts_ms as i64,
            );

            // 2. Снимок треков и людей (для overlay).
            let (tracks_snapshot, persons_snapshot) = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let g = state.info.read().await;
                    (g.tracks.clone(), g.persons.clone())
                })
            });

            // 3. BGR → RGB, даунскейл, bbox, JPEG.
            let draw_person_boxes = state.preview_draw_person_boxes.load(Ordering::Relaxed);
            let jpeg = encode_preview_jpeg(
                &*bgr_shared,
                w as u32,
                h as u32,
                &tracks_snapshot,
                &persons_snapshot,
                draw_person_boxes,
            );

            if let Some(jpeg) = jpeg {
                let info = state.info.clone();
                tokio::runtime::Handle::current().spawn(async move {
                    let mut info = info.write().await;
                    info.current_frame = frame_id;
                    info.latest_jpeg = Some(jpeg);
                });
            }
        }
    });
}

fn push_frame_to_pipeline(state: &AppState, bgr: Arc<Vec<u8>>, width: u32, height: u32, pts_ms: i64) {
    let session_arc = state.session.clone();
    // try_send из блокирующего контекста — берём lock в block_on'ом.
    let _ = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async move {
            let g = session_arc.lock().await;
            if let Some(s) = g.as_ref() {
                let frame = integra::Frame {
                    bgr,
                    width,
                    height,
                    pts_ms,
                };
                let _ = s.frame_tx.try_send(frame);
            }
        });
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
    // C++ pipeline шлёт kind вида "alarm_abandoned" / "alarm_disappeared"
    // (см. analyticsd_main.cpp::serialize_alarm). Также принимаем legacy-имена
    // "abandoned" / "disappeared" — на всякий случай.
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
    fn resolve_accepts_basename_only() {
        let tmp = std::env::temp_dir().join(format!("integra_open_base_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        fs::write(tmp.join("data").join("only.mp4"), b"x").unwrap();
        let got = resolve_open_video_path(&tmp, "only.mp4").unwrap();
        assert!(got.ends_with("only.mp4"));
    }

    #[test]
    fn resolve_accepts_data_relative_from_api_files() {
        let tmp = std::env::temp_dir().join(format!("integra_open_relapi_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("data")).unwrap();
        fs::write(tmp.join("data").join("clip.mkv"), b"x").unwrap();
        let got = resolve_open_video_path(&tmp, "data/clip.mkv").unwrap();
        assert!(got.ends_with("clip.mkv"));
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
