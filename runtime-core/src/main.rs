use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameMeta {
    camera_id: String,
    frame_id: u64,
    pts_ms: u64,
    width: u32,
    height: u32,
    // На этом этапе Rust-контур хранит только metadata/state.
    // Пиксели должны жить в GPU/SHM пути и не раздувать RAM.
    shm_slot: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceRequest {
    camera_id: String,
    frame_id: u64,
    pts_ms: u64,
    roi: Option<[u32; 4]>,
    shm_slot: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceResult {
    camera_id: String,
    frame_id: u64,
    inference_ms: f32,
    detections: usize,
    persons: usize,
}

#[derive(Debug, Default)]
struct CameraState {
    last_frame_id: u64,
    dropped_frames: u64,
    // Compact ring-buffer состояния вместо хранения кадров.
    recent_track_counts: VecDeque<usize>,
    recent_inf_ms: VecDeque<f32>,
}

#[derive(Debug, Default)]
struct RuntimeStats {
    queued_for_infer: u64,
    dropped_for_backpressure: u64,
    completed_infer: u64,
}

#[derive(Debug, Default)]
struct ControlState {
    last_granted_infer_frame: HashMap<String, u64>,
    camera_inf_ms_ema: HashMap<String, f32>,
    camera_activity_score: HashMap<String, f32>,
    global_overload_level: u8,
    scheduler_pending_batch: usize,
    scheduler_dropped_total: u64,
    scheduler_queued_total: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "runtime_core=info,runtime-core=info,info".to_string()),
        )
        .init();

    if let Ok(addr) = std::env::var("INFERENCE_RPC_ADDR") {
        match probe_python_inference(&addr).await {
            Ok(names_count) => info!(%addr, names = names_count, "python inference service probe ok"),
            Err(err) => warn!(%addr, error = %err, "python inference service probe failed"),
        }
    }

    // Центральные bounded очереди.
    let (ingest_tx, ingest_rx) = mpsc::channel::<FrameMeta>(512);
    let (infer_tx, infer_rx) = mpsc::channel::<InferenceRequest>(256);
    let (result_tx, result_rx) = mpsc::channel::<InferenceResult>(256);
    let control_state = Arc::new(Mutex::new(ControlState::default()));

    if let Ok(addr) = std::env::var("RUNTIME_INGEST_ADDR") {
        tokio::spawn(run_runtime_ingest_server(
            addr.clone(),
            ingest_tx.clone(),
            result_tx.clone(),
            control_state.clone(),
        ));
        info!(%addr, "runtime ingest bridge server enabled");
    } else {
        // Демо-генератор: имитируем несколько камер.
        tokio::spawn(simulate_ingest(ingest_tx.clone(), 8, 8));
    }
    if let Ok(addr) = std::env::var("RUNTIME_CONTROL_ADDR") {
        tokio::spawn(run_runtime_control_server(addr.clone(), control_state.clone()));
        info!(%addr, "runtime control server enabled");
    }
    tokio::spawn(simulate_inference_worker(infer_rx, result_tx));
    scheduler_loop(ingest_rx, infer_tx, result_rx, control_state).await
}

async fn probe_python_inference(addr: &str) -> Result<usize> {
    let stream = TcpStream::connect(addr).await?;
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let request = json!({
        "id": 1,
        "op": "health"
    });
    writer
        .write_all(format!("{}\n", request).as_bytes())
        .await?;
    writer.flush().await?;
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    if line.trim().is_empty() {
        anyhow::bail!("empty response");
    }
    let resp: serde_json::Value = serde_json::from_str(&line)?;
    if !resp.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
        anyhow::bail!("service replied with error: {resp}");
    }
    let names_count = resp
        .get("names")
        .and_then(|v| v.as_object())
        .map(|m| m.len())
        .unwrap_or(0);
    Ok(names_count)
}

async fn scheduler_loop(
    mut ingest_rx: mpsc::Receiver<FrameMeta>,
    infer_tx: mpsc::Sender<InferenceRequest>,
    mut result_rx: mpsc::Receiver<InferenceResult>,
    control_state: Arc<Mutex<ControlState>>,
) -> Result<()> {
    let mut cameras: HashMap<String, CameraState> = HashMap::new();
    let mut stats = RuntimeStats::default();
    let mut batch_tick = time::interval(Duration::from_millis(10));
    let mut report_tick = time::interval(Duration::from_secs(2));
    let mut pending_batch: Vec<FrameMeta> = Vec::with_capacity(32);

    loop {
        tokio::select! {
            _ = batch_tick.tick() => {
                if pending_batch.is_empty() {
                    update_control_runtime_health(&control_state, 0, stats.queued_for_infer, stats.dropped_for_backpressure);
                    continue;
                }
                // Центральный scheduler: ограничивает batch размер и выброс старых задач при перегрузе.
                let keep_from = pending_batch.len().saturating_sub(16);
                let to_send = pending_batch.split_off(keep_from);
                for frame in to_send {
                    let req = InferenceRequest {
                        camera_id: frame.camera_id.clone(),
                        frame_id: frame.frame_id,
                        pts_ms: frame.pts_ms,
                        roi: None,
                        shm_slot: frame.shm_slot.clone(),
                    };
                    match infer_tx.try_send(req) {
                        Ok(_) => stats.queued_for_infer += 1,
                        Err(_) => {
                            stats.dropped_for_backpressure += 1;
                            if let Some(cam) = cameras.get_mut(&frame.camera_id) {
                                cam.dropped_frames += 1;
                            }
                        }
                    }
                }
                update_control_runtime_health(
                    &control_state,
                    pending_batch.len(),
                    stats.queued_for_infer,
                    stats.dropped_for_backpressure,
                );
            }
            maybe_frame = ingest_rx.recv() => {
                let Some(frame) = maybe_frame else {
                    break;
                };
                let cam = cameras.entry(frame.camera_id.clone()).or_default();
                cam.last_frame_id = frame.frame_id;
                pending_batch.push(frame);
                update_control_runtime_health(
                    &control_state,
                    pending_batch.len(),
                    stats.queued_for_infer,
                    stats.dropped_for_backpressure,
                );
            }
            maybe_result = result_rx.recv() => {
                let Some(result) = maybe_result else {
                    break;
                };
                stats.completed_infer += 1;
                update_control_camera_latency(&control_state, &result.camera_id, result.inference_ms);
                if let Some(cam) = cameras.get_mut(&result.camera_id) {
                    push_ring_f32(&mut cam.recent_inf_ms, result.inference_ms, 120);
                    push_ring_usize(&mut cam.recent_track_counts, result.detections, 120);
                }
            }
            _ = report_tick.tick() => {
                let cams = cameras.len();
                info!(
                    cameras = cams,
                    queued = stats.queued_for_infer,
                    dropped = stats.dropped_for_backpressure,
                    done = stats.completed_infer,
                    "runtime-core scheduler tick"
                );
                if stats.dropped_for_backpressure > 0 {
                    warn!("backpressure active: dropped inference requests detected");
                }
            }
        }
    }

    Ok(())
}

fn push_ring_f32(ring: &mut VecDeque<f32>, value: f32, max_len: usize) {
    ring.push_back(value);
    while ring.len() > max_len {
        ring.pop_front();
    }
}

fn push_ring_usize(ring: &mut VecDeque<usize>, value: usize, max_len: usize) {
    ring.push_back(value);
    while ring.len() > max_len {
        ring.pop_front();
    }
}

async fn simulate_ingest(tx: mpsc::Sender<FrameMeta>, cameras: usize, fps: u64) {
    let mut frame_id = 0_u64;
    let period = Duration::from_millis((1000.0 / fps as f64) as u64);
    let mut tick = time::interval(period);
    loop {
        tick.tick().await;
        frame_id = frame_id.wrapping_add(1);
        for i in 0..cameras {
            let payload = FrameMeta {
                camera_id: format!("cam-{i:03}"),
                frame_id,
                pts_ms: frame_id.saturating_mul(1000 / fps.max(1)),
                width: 1920,
                height: 1080,
                shm_slot: None,
            };
            if tx.send(payload).await.is_err() {
                return;
            }
        }
    }
}

async fn simulate_inference_worker(
    mut rx: mpsc::Receiver<InferenceRequest>,
    tx: mpsc::Sender<InferenceResult>,
) {
    while let Some(req) = rx.recv().await {
        time::sleep(Duration::from_millis(12)).await;
        let out = InferenceResult {
            camera_id: req.camera_id,
            frame_id: req.frame_id,
            inference_ms: 11.8,
            detections: 2,
            persons: 1,
        };
        if tx.send(out).await.is_err() {
            return;
        }
    }
}

async fn run_runtime_ingest_server(
    addr: String,
    ingest_tx: mpsc::Sender<FrameMeta>,
    result_tx: mpsc::Sender<InferenceResult>,
    control_state: Arc<Mutex<ControlState>>,
) {
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(err) => {
            warn!(%addr, error=%err, "failed to bind runtime ingest bridge");
            return;
        }
    };
    loop {
        let (socket, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(err) => {
                warn!(error=%err, "runtime ingest accept failed");
                continue;
            }
        };
        let ingest_tx = ingest_tx.clone();
        let result_tx = result_tx.clone();
        let control_state = control_state.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(socket).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<serde_json::Value>(&line) {
                            Ok(v) => {
                                let typ = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                                if typ == "frame_observed" {
                                    let msg = FrameMeta {
                                        camera_id: v
                                            .get("camera_id")
                                            .and_then(|x| x.as_str())
                                            .unwrap_or("unknown")
                                            .to_string(),
                                        frame_id: v.get("frame_id").and_then(|x| x.as_u64()).unwrap_or(0),
                                        pts_ms: v.get("pos_ms").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64,
                                        width: v.get("width").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                                        height: v.get("height").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                                        shm_slot: None,
                                    };
                                    let _ = ingest_tx.send(msg).await;
                                } else if typ == "inference_result" {
                                    let msg = InferenceResult {
                                        camera_id: v
                                            .get("camera_id")
                                            .and_then(|x| x.as_str())
                                            .unwrap_or("unknown")
                                            .to_string(),
                                        frame_id: v.get("frame_id").and_then(|x| x.as_u64()).unwrap_or(0),
                                        inference_ms: v.get("inference_ms").and_then(|x| x.as_f64()).unwrap_or(0.0)
                                            as f32,
                                        detections: v.get("detections").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                                        persons: v.get("persons").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                                    };
                                    let events = v.get("events").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                                    update_control_camera_activity(&control_state, &msg.camera_id, msg.detections, events);
                                    update_control_camera_latency(&control_state, &msg.camera_id, msg.inference_ms);
                                    let _ = result_tx.send(msg).await;
                                }
                            }
                            Err(err) => {
                                warn!(%peer, error=%err, "invalid ingest bridge json");
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        warn!(%peer, error=%err, "runtime ingest read failed");
                        break;
                    }
                }
            }
        });
    }
}

async fn run_runtime_control_server(addr: String, state: Arc<Mutex<ControlState>>) {
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(err) => {
            warn!(%addr, error=%err, "failed to bind runtime control server");
            return;
        }
    };
    loop {
        let (socket, peer) = match listener.accept().await {
            Ok(v) => v,
            Err(err) => {
                warn!(error=%err, "runtime control accept failed");
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            let (reader, mut writer) = socket.into_split();
            let mut lines = BufReader::new(reader).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let response = match serde_json::from_str::<serde_json::Value>(&line) {
                            Ok(v) => handle_control_message(v, &state),
                            Err(err) => json!({
                                "type": "error",
                                "error": format!("invalid json: {err}")
                            }),
                        };
                        let out = format!("{response}\n");
                        if writer.write_all(out.as_bytes()).await.is_err() {
                            break;
                        }
                        if writer.flush().await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(err) => {
                        warn!(%peer, error=%err, "runtime control read failed");
                        break;
                    }
                }
            }
        });
    }
}

fn update_control_camera_latency(state: &Arc<Mutex<ControlState>>, camera_id: &str, inference_ms: f32) {
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let prev = st
        .camera_inf_ms_ema
        .get(camera_id)
        .copied()
        .unwrap_or(inference_ms.max(0.0));
    let next = (prev * 0.85) + (inference_ms.max(0.0) * 0.15);
    st.camera_inf_ms_ema.insert(camera_id.to_string(), next);
}

fn update_control_camera_activity(
    state: &Arc<Mutex<ControlState>>,
    camera_id: &str,
    detections: usize,
    events: usize,
) {
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    let prev = st
        .camera_activity_score
        .get(camera_id)
        .copied()
        .unwrap_or(0.0);
    let signal = (detections.min(20) as f32 * 0.1) + (events.min(10) as f32 * 0.5);
    let next = (prev * 0.85) + signal;
    st.camera_activity_score.insert(camera_id.to_string(), next.min(10.0));
}

fn update_control_runtime_health(
    state: &Arc<Mutex<ControlState>>,
    pending_batch_len: usize,
    queued_total: u64,
    dropped_total: u64,
) {
    let mut st = match state.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    st.scheduler_pending_batch = pending_batch_len;
    st.scheduler_queued_total = queued_total;
    st.scheduler_dropped_total = dropped_total;
    let drop_ratio = if queued_total > 0 {
        dropped_total as f32 / queued_total as f32
    } else {
        0.0
    };
    st.global_overload_level = if pending_batch_len >= 128 || drop_ratio >= 0.30 {
        2
    } else if pending_batch_len >= 48 || drop_ratio >= 0.10 {
        1
    } else {
        0
    };
}

fn handle_control_message(v: serde_json::Value, state: &Arc<Mutex<ControlState>>) -> serde_json::Value {
    let typ = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    if typ != "should_infer" {
        return json!({
            "type": "error",
            "error": format!("unsupported message type: {typ}")
        });
    }
    let camera_id = v
        .get("camera_id")
        .and_then(|x| x.as_str())
        .unwrap_or("unknown")
        .to_string();
    let frame_id = v.get("frame_id").and_then(|x| x.as_u64()).unwrap_or(0);
    let interval = v
        .get("default_interval")
        .and_then(|x| x.as_u64())
        .unwrap_or(1)
        .max(1);
    let (infer, target_interval, latency_ema, overload_level, priority, max_roi_count) = {
        let mut st = match state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let latency_ema = st.camera_inf_ms_ema.get(&camera_id).copied().unwrap_or(0.0);
        let activity = st.camera_activity_score.get(&camera_id).copied().unwrap_or(0.0);
        let latency_boost = if latency_ema >= 40.0 {
            3_u64
        } else if latency_ema >= 28.0 {
            2_u64
        } else if latency_ema >= 18.0 {
            1_u64
        } else {
            0_u64
        };
        let overload_boost = match st.global_overload_level {
            2 => 3_u64,
            1 => 1_u64,
            _ => 0_u64,
        };
        let activity_relief = if activity >= 4.0 { 1_u64 } else { 0_u64 };
        let target_interval = interval
            .saturating_add(latency_boost)
            .saturating_add(overload_boost)
            .saturating_sub(activity_relief)
            .max(1);
        let last = st.last_granted_infer_frame.get(&camera_id).copied().unwrap_or(0);
        let infer = if last == 0 || frame_id.saturating_sub(last) >= target_interval {
            st.last_granted_infer_frame.insert(camera_id.clone(), frame_id);
            true
        } else {
            false
        };
        let priority = if activity >= 5.0 && st.global_overload_level == 0 {
            "high"
        } else if st.global_overload_level >= 2 {
            "low"
        } else {
            "normal"
        };
        let max_roi_count = if st.global_overload_level >= 2 {
            1_u64
        } else if st.global_overload_level == 1 {
            2_u64
        } else if activity >= 5.0 {
            4_u64
        } else {
            3_u64
        };
        (
            infer,
            target_interval,
            latency_ema,
            st.global_overload_level,
            priority,
            max_roi_count,
        )
    };
    json!({
        "type": "should_infer_result",
        "camera_id": camera_id,
        "frame_id": frame_id,
        "infer": infer,
        "target_interval": target_interval,
        "priority": priority,
        "max_roi_count": max_roi_count,
        "latency_ema_ms": latency_ema,
        "overload_level": overload_level
    })
}
