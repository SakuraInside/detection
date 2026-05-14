//! infer_worker — изолированный процесс инференса.
//!
//! Загружает integra_ffi.dll (CUDA + модель) и принимает BGR-кадры по TCP.
//! После запуска он владеет всей тяжёлой C++ памятью (~400-500 МБ с CUDA),
//! а backend_gateway работает без FFI и занимает ~80-100 МБ.
//!
//! Запуск (автоматически через run.py):
//!   infer_worker.exe --listen 127.0.0.1:9910
//!
//! Протокол (см. tcp_client.rs):
//!   Клиент → сервер: {\"width\":W,\"height\":H,\"pts_ms\":T}\n + u32-LE + BGR
//!   Сервер → клиент: {\"type\":\"event\",\"payload\":{…}}\n     (0..N)
//!                    {\"type\":\"frame_result\",\"payload\":{…}}\n (1 per frame)

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde_json::json;
use tracing::{error, info, warn};

use runtime_core::config::{build_pipeline_config, read_config_json};
use runtime_core::integra::ffi::global_lib;
use runtime_core::integra::{Pipeline, StreamMessage};

#[derive(Parser, Debug)]
#[command(name = "infer_worker")]
struct Args {
    /// Адрес TCP-сервера для приёма кадров
    #[arg(long, default_value = "127.0.0.1:9910")]
    listen: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,infer_worker=info".to_string()),
        )
        .init();

    let args = Args::parse();

    let root = std::env::var("INTEGRA_PROJECT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().expect("cwd"));

    // Загружаем integra_ffi.dll — здесь живут CUDA и модель (~400+ МБ).
    let lib = match global_lib() {
        Ok(l) => l,
        Err(e) => {
            error!(error = %e, "infer_worker: failed to load integra_ffi");
            return Err(anyhow::anyhow!("{e}"));
        }
    };
    info!(version = %lib.version(), "integra_ffi loaded");

    let cfg_json = read_config_json(&root).unwrap_or_else(|| serde_json::json!({}));
    let pipeline_cfg = build_pipeline_config(&root, &cfg_json);
    info!(engine = %pipeline_cfg.engine_kind, model = %pipeline_cfg.model_path, "pipeline config built");

    let listener = TcpListener::bind(&args.listen)?;
    info!(addr = %args.listen, "infer_worker listening");

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "infer_worker: accept failed");
                continue;
            }
        };
        let peer = stream.peer_addr().ok();
        info!(?peer, "infer_worker: client connected");

        let lib_clone = lib.clone();
        let pcfg = pipeline_cfg.clone();

        std::thread::spawn(move || {
            if let Err(e) = handle_client(stream, pcfg, lib_clone) {
                warn!(?peer, error = %e, "infer_worker: client session ended with error");
            } else {
                info!(?peer, "infer_worker: client disconnected");
            }
        });
    }

    Ok(())
}

fn handle_client(
    stream: std::net::TcpStream,
    cfg: runtime_core::integra::PipelineConfig,
    lib: std::sync::Arc<runtime_core::integra::IntegraLib>,
) -> Result<()> {
    let _ = stream.set_nodelay(true);
    let write_stream = stream.try_clone()?;
    let mut writer = std::io::BufWriter::new(write_stream);
    let mut reader = BufReader::new(stream);

    let mut pipeline = Pipeline::new(&cfg, lib)?;

    loop {
        // Читаем meta-строку
        let mut meta_line = String::new();
        if reader.read_line(&mut meta_line)? == 0 {
            break;
        }
        let meta: serde_json::Value = match serde_json::from_str(meta_line.trim()) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "infer_worker: bad meta json, skipping line");
                continue;
            }
        };

        let w = meta.get("width").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let h = meta.get("height").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        let pts_ms = meta.get("pts_ms").and_then(|v| v.as_i64()).unwrap_or(0);

        if w <= 0 || h <= 0 {
            continue;
        }

        // Читаем BGR (4-байтная длина + пиксели)
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut bgr = vec![0u8; len];
        reader.read_exact(&mut bgr)?;

        if bgr.len() != (w * h * 3) as usize {
            warn!(got = bgr.len(), expected = w*h*3, "infer_worker: frame size mismatch, skipping");
            continue;
        }

        // Инференс
        match pipeline.push_frame(&bgr, w, h, pts_ms) {
            Ok(messages) => {
                for msg in messages {
                    let json_line = match &msg {
                        StreamMessage::Event(ev) => {
                            match serde_json::to_value(ev) {
                                Ok(v) => json!({"type":"event","payload":v}).to_string(),
                                Err(e) => json!({"type":"error","msg":e.to_string()}).to_string(),
                            }
                        }
                        StreamMessage::Frame(fr) => {
                            match serde_json::to_value(fr) {
                                Ok(v) => json!({"type":"frame_result","payload":v}).to_string(),
                                Err(e) => json!({"type":"error","msg":e.to_string()}).to_string(),
                            }
                        }
                        StreamMessage::Error(e) => {
                            json!({"type":"error","msg":e}).to_string()
                        }
                    };
                    writer.write_all(json_line.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
                writer.flush()?;
            }
            Err(e) => {
                error!(error = %e, "infer_worker: push_frame failed");
                let err_line = json!({"type":"error","msg":e.to_string()}).to_string();
                writer.write_all(err_line.as_bytes())?;
                writer.write_all(b"\n")?;
                writer.flush()?;
            }
        }
    }

    Ok(())
}
