//! Декод видео в Rust, выдача BGR-кадров в TCP. Используется backend_gateway.
//!
//! Протокол:
//! 1) Клиент: `{"cmd":"open","path":"/path.mp4","seek_frame":null,"prefer_hw_decode":true}\n`
//! 2) Сервер: handshake JSON + `\n`
//! 3) Кадры: JSON meta + `\n` + 4 байта LE u32 (длина BGR) + BGR (H*W*3)
//!
//! После handshake клиент может слать команды (одна на строку, `\n`-terminated):
//!   {"cmd":"seek","frame":N}     — переместиться к кадру N
//!   {"cmd":"seek","ms":12345.0}  — переместиться к позиции в миллисекундах
//!   {"cmd":"pause"}              — приостановить выдачу кадров (соединение остаётся)
//!   {"cmd":"play"}               — возобновить выдачу
//!   {"cmd":"close"}              — корректно завершить сессию
//!
//! Команды применяются перед чтением следующего кадра (задержка ≤ длительности одного кадра).

use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpListener};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use opencv::core::Mat;
use opencv::prelude::*;
use opencv::videoio::{
    VideoCapture, VideoCaptureTrait, CAP_FFMPEG, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES, CAP_PROP_POS_MSEC,
};
use serde::Serialize;
use serde_json::json;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(name = "video-bridge")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9876")]
    listen: String,
    /// 0 = не ограничивать
    #[arg(long, default_value_t = 0_u32)]
    target_fps: u32,
}

#[derive(Serialize)]
struct Handshake {
    r#type: &'static str,
    ok: bool,
    width: i32,
    height: i32,
    fps: f64,
    frames: i64,
    /// Клиент просил включить HW decode (OpenCV CAP_PROP_HW_ACCELERATION).
    prefer_hw_decode: bool,
    /// Удалось применить CAP_PROP_HW_ACCELERATION (best-effort).
    hw_decode_set_ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

#[derive(Serialize)]
struct FrameMeta {
    frame_id: u64,
    pos_ms: f64,
    width: i32,
    height: i32,
}

#[derive(Debug)]
enum SessionCommand {
    SeekFrame(i32),
    SeekMs(f64),
    Pause,
    Play,
    Close,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,video_bridge=info".to_string()),
        )
        .init();

    let args = Args::parse();
    let listener =
        TcpListener::bind(&args.listen).with_context(|| format!("bind {}", args.listen))?;
    info!(addr = %args.listen, "video-bridge listening (Python: pipeline.rust_video_bridge + this address)");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "accept failed");
                continue;
            }
        };
        let peer = stream
            .peer_addr()
            .unwrap_or_else(|_| std::net::SocketAddr::from(([0, 0, 0, 0], 0)));
        stream.set_nodelay(true).ok();
        info!(%peer, "client connected");

        let mut reader = BufReader::new(stream.try_clone()?);
        let mut cmd_line = String::new();
        if reader.read_line(&mut cmd_line).unwrap_or(0) == 0 {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(cmd_line.trim()).unwrap_or(json!({}));
        if v.get("cmd").and_then(|x| x.as_str()) != Some("open") {
            write_handshake_error(&mut stream, "expected {\"cmd\":\"open\",\"path\":...}")?;
            continue;
        }
        let path = match v.get("path").and_then(|x| x.as_str()) {
            Some(p) => p.to_string(),
            None => {
                write_handshake_error(&mut stream, "missing path")?;
                continue;
            }
        };
        let seek_frame = v
            .get("seek_frame")
            .and_then(|x| x.as_u64())
            .map(|x| x as i32);
        let prefer_hw_decode = v
            .get("prefer_hw_decode")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);

        match run_session(
            &mut stream,
            &path,
            seek_frame,
            args.target_fps,
            prefer_hw_decode,
        ) {
            Ok(()) => info!(%peer, "session ended"),
            Err(e) => warn!(%peer, error = %e, "session error"),
        }
    }

    Ok(())
}

fn write_handshake_error(stream: &mut std::net::TcpStream, msg: &str) -> Result<()> {
    let h = Handshake {
        r#type: "handshake",
        ok: false,
        width: 0,
        height: 0,
        fps: 0.0,
        frames: 0,
        prefer_hw_decode: false,
        hw_decode_set_ok: false,
        message: Some(msg.to_string()),
    };
    writeln!(stream, "{}", serde_json::to_string(&h)?)?;
    stream.flush()?;
    Ok(())
}

fn run_session(
    stream: &mut std::net::TcpStream,
    path: &str,
    seek_frame: Option<i32>,
    target_fps: u32,
    prefer_hw_decode: bool,
) -> Result<()> {
    if !Path::new(path).exists() {
        write_handshake_error(stream, &format!("file not found: {path}"))?;
        return Ok(());
    }

    let mut cap =
        VideoCapture::from_file(path, CAP_FFMPEG).map_err(|e| anyhow!("opencv open: {e}"))?;
    if !cap.is_opened()? {
        write_handshake_error(stream, "VideoCapture failed")?;
        return Ok(());
    }

    // OpenCV 4.x: CAP_PROP_HW_ACCELERATION + VIDEO_ACCELERATION_ANY (1.0) — best-effort.
    const CAP_PROP_HW_ACCELERATION: i32 = 53;
    const VIDEO_ACCELERATION_ANY: f64 = 1.0;
    let mut hw_decode_set_ok = false;
    if prefer_hw_decode {
        hw_decode_set_ok = cap
            .set(CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY)
            .unwrap_or(false);
        if hw_decode_set_ok {
            info!("HW video acceleration requested and set on capture");
        } else {
            warn!("prefer_hw_decode set but CAP_PROP_HW_ACCELERATION not applied (driver/build)");
        }
    }

    if let Some(sf) = seek_frame {
        cap.set(CAP_PROP_POS_FRAMES, sf as f64).ok();
    }

    let w = cap.get(CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(CAP_PROP_FPS).unwrap_or(30.0);
    let frames = cap.get(CAP_PROP_FRAME_COUNT)? as i64;

    // Темп выдачи кадров: не быстрее реального FPS потока из контейнера.
    // Иначе при FPS в метаданных ~6 и pipeline.target_fps=30 за 1 с стены читается 30 кадров
    // файла → ~5× ускорение относительно «живого» времени ролика.
    let stream_fps = if fps.is_finite() && fps > 0.5 && fps <= 240.0 {
        fps
    } else {
        30.0
    };
    let pace_fps = if target_fps > 0 {
        (target_fps as f64).min(stream_fps).max(1.0)
    } else {
        stream_fps
    };
    if target_fps > 0 && pace_fps + 0.001 < target_fps as f64 {
        info!(
            stream_fps,
            pace_fps,
            target_fps,
            "playback pacing capped to stream FPS (avoid fast-forward)"
        );
    }

    let hs = Handshake {
        r#type: "handshake",
        ok: true,
        width: w,
        height: h,
        fps,
        frames,
        prefer_hw_decode,
        hw_decode_set_ok,
        message: None,
    };
    writeln!(stream, "{}", serde_json::to_string(&hs)?)?;
    stream.flush()?;

    // Поднимаем отдельный command-reader thread: он читает JSON-команды от клиента
    // по `\n` и шлёт их в основной цикл через mpsc-канал. Когда основной цикл
    // завершается, мы делаем shutdown(Read) — read_line возвращает Ok(0), thread выходит.
    let (cmd_tx, cmd_rx) = mpsc::channel::<SessionCommand>();
    let cmd_stream = stream
        .try_clone()
        .map_err(|e| anyhow!("clone stream: {e}"))?;
    let cmd_thread = std::thread::spawn(move || {
        let mut reader = BufReader::new(cmd_stream);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let v: serde_json::Value = match serde_json::from_str(line.trim()) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let cmd = match v.get("cmd").and_then(|x| x.as_str()) {
                        Some("seek") => {
                            if let Some(f) = v.get("frame").and_then(|x| x.as_i64()) {
                                SessionCommand::SeekFrame(f as i32)
                            } else if let Some(ms) = v.get("ms").and_then(|x| x.as_f64()) {
                                SessionCommand::SeekMs(ms)
                            } else {
                                continue;
                            }
                        }
                        Some("pause") => SessionCommand::Pause,
                        Some("play") => SessionCommand::Play,
                        Some("close") => {
                            let _ = cmd_tx.send(SessionCommand::Close);
                            break;
                        }
                        _ => continue,
                    };
                    if cmd_tx.send(cmd).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let period = if pace_fps > 0.0 {
        Some(Duration::from_secs_f64(1.0 / pace_fps))
    } else {
        None
    };
    let mut next_tick = Instant::now();
    let mut frame_id: u64 = 0;
    let mut paused = false;
    let mut mat = Mat::default();

    let result: Result<()> = (|| loop {
        // Драйним команды, накопившиеся между кадрами.
        let mut should_close = false;
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                SessionCommand::SeekFrame(f) => {
                    let target = f.max(0) as f64;
                    if cap.set(CAP_PROP_POS_FRAMES, target).unwrap_or(false) {
                        frame_id = f.max(0) as u64;
                        info!(frame = f, "seek by frame");
                    } else {
                        warn!(frame = f, "seek by frame failed");
                    }
                    // Сбрасываем темпинг — иначе после seek next_tick может быть в будущем.
                    next_tick = Instant::now();
                }
                SessionCommand::SeekMs(ms) => {
                    if cap.set(CAP_PROP_POS_MSEC, ms.max(0.0)).unwrap_or(false) {
                        info!(ms = ms, "seek by ms");
                        // frame_id обновится из cap на следующем кадре (через pos_msec).
                    } else {
                        warn!(ms = ms, "seek by ms failed");
                    }
                    next_tick = Instant::now();
                }
                SessionCommand::Pause => {
                    paused = true;
                    info!("paused");
                }
                SessionCommand::Play => {
                    paused = false;
                    next_tick = Instant::now();
                    info!("resumed");
                }
                SessionCommand::Close => {
                    info!("close requested by client");
                    should_close = true;
                }
            }
        }
        if should_close {
            return Ok(());
        }

        if paused {
            // Не декодируем и не пишем — соединение остаётся живым, клиент в любой момент
            // может прислать seek/play/close. Минимальный sleep, чтобы CPU не крутился вхолостую.
            std::thread::sleep(Duration::from_millis(30));
            continue;
        }

        if let Some(p) = period {
            next_tick += p;
            let now = Instant::now();
            if now < next_tick {
                std::thread::sleep(next_tick - now);
            } else {
                next_tick = now;
            }
        }

        let ok = cap.read(&mut mat)?;
        if !ok || mat.empty() {
            return Ok(());
        }

        if !mat.is_continuous() {
            let mut cont = Mat::default();
            mat.copy_to(&mut cont)
                .map_err(|e| anyhow!("copy_to: {e}"))?;
            mat = cont;
        }

        frame_id = frame_id.saturating_add(1);
        let pos_ms = cap.get(CAP_PROP_POS_MSEC).unwrap_or(0.0);
        let rows = mat.rows();
        let cols = mat.cols();
        let slice = mat.data_bytes().map_err(|e| anyhow!("data_bytes: {e}"))?;

        let meta = FrameMeta {
            frame_id,
            pos_ms,
            width: cols,
            height: rows,
        };
        writeln!(stream, "{}", serde_json::to_string(&meta)?)?;
        let len = slice.len() as u32;
        stream.write_all(&len.to_le_bytes())?;
        stream.write_all(slice)?;
        if let Err(e) = stream.flush() {
            // Клиент закрыл соединение — корректно выходим.
            warn!(error = %e, "stream flush failed; closing");
            return Ok(());
        }
    })();

    // Корректно гасим command thread: после Shutdown::Both read_line вернёт Ok(0) или Err.
    let _ = stream.shutdown(Shutdown::Both);
    let _ = cmd_thread.join();
    result
}
