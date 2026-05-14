//! TCP-клиент к infer_worker.
//!
//! Реализует тот же интерфейс что `spawn_stream` (возвращает `StreamHandle`),
//! но вместо прямого вызова FFI отправляет BGR-кадры в отдельный процесс
//! `infer_worker` по TCP и читает обратно JSON-результаты.
//!
//! Протокол (newline-delimited JSON + binary):
//!   → {\"width\":W,\"height\":H,\"pts_ms\":T}\n + u32-LE + BGR bytes
//!   ← {\"type\":\"event\",\"payload\":{…}}\n          (0..N per frame)
//!   ← {\"type\":\"frame_result\",\"payload\":{…}}\n   (1 per frame)
//!
//! Snapshot-кроп для alarm-событий пропускается в этом режиме:
//! gateway использует `latest_jpeg` как fallback (поведение настроено в gateway).

use std::io::{BufRead, BufReader, Write};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use super::events::{AlarmEvent, FrameResult, StreamMessage};
use super::stream::{Frame, StreamHandle, EVENTS_CHANNEL_CAP, FRAMES_CHANNEL_CAP};

/// Подключается к infer_worker на `addr` и запускает worker-поток.
/// Возвращает `StreamHandle` с теми же каналами что и `spawn_stream`.
pub fn spawn_tcp_stream(addr: String) -> Result<StreamHandle, std::io::Error> {
    let sock = connect_with_retry(&addr, 3, Duration::from_millis(500))?;
    let _ = sock.set_nodelay(true);

    let (frame_tx, mut frame_rx) = mpsc::channel::<Frame>(FRAMES_CHANNEL_CAP);
    let (events_tx, events_rx) = mpsc::channel::<StreamMessage>(EVENTS_CHANNEL_CAP);

    let join: JoinHandle<()> = tokio::task::spawn_blocking(move || {
        let mut write_sock = match sock.try_clone() {
            Ok(s) => s,
            Err(e) => { error!(error=%e, "tcp_client: clone socket failed"); return; }
        };
        let mut reader = BufReader::new(sock);

        info!(%addr, "tcp_client: connected to infer_worker");

        let mut frames_sent: u64 = 0;
        loop {
            let frame = match frame_rx.blocking_recv() {
                Some(f) => f,
                None => {
                    info!("tcp_client: frame sender dropped, disconnecting");
                    break;
                }
            };

            // 1. Отправляем meta
            let meta = serde_json::json!({
                "width": frame.width,
                "height": frame.height,
                "pts_ms": frame.pts_ms,
            });
            let meta_line = format!("{meta}\n");
            if let Err(e) = write_sock.write_all(meta_line.as_bytes()) {
                error!(error=%e, frames_sent, "tcp_client: write meta failed, infer_worker disconnected");
                break;
            }

            // 2. Отправляем BGR (4-байтная длина + пиксели)
            let bgr = frame.bgr.as_slice();
            let len_bytes = (bgr.len() as u32).to_le_bytes();
            if let Err(e) = write_sock.write_all(&len_bytes) {
                error!(error=%e, frames_sent, "tcp_client: write len failed");
                break;
            }
            if let Err(e) = write_sock.write_all(bgr) {
                error!(error=%e, frames_sent, "tcp_client: write bgr failed");
                break;
            }
            if let Err(e) = write_sock.flush() {
                error!(error=%e, frames_sent, "tcp_client: flush failed");
                break;
            }
            frames_sent += 1;

            // 3. Читаем ответы до получения frame_result (или error)
            let mut got_frame = false;
            while !got_frame {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        warn!(frames_sent, "tcp_client: infer_worker closed connection (EOF)");
                        return;
                    }
                    Err(e) => {
                        error!(error=%e, frames_sent, "tcp_client: read error from infer_worker");
                        return;
                    }
                    Ok(_) => {}
                }
                let v: serde_json::Value = match serde_json::from_str(line.trim()) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error=%e, line=%line.trim(), "tcp_client: bad json from infer_worker");
                        continue;
                    }
                };
                let typ = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
                match typ {
                    "event" => {
                        match serde_json::from_value::<AlarmEvent>(v["payload"].clone()) {
                            Ok(ev) => {
                                if events_tx.blocking_send(StreamMessage::Event(ev)).is_err() {
                                    return;
                                }
                            }
                            Err(e) => warn!(error=%e, "tcp_client: parse AlarmEvent failed"),
                        }
                    }
                    "frame_result" => {
                        match serde_json::from_value::<FrameResult>(v["payload"].clone()) {
                            Ok(fr) => {
                                if events_tx.blocking_send(StreamMessage::Frame(fr)).is_err() {
                                    return;
                                }
                            }
                            Err(e) => warn!(error=%e, "tcp_client: parse FrameResult failed"),
                        }
                        got_frame = true;
                    }
                    "error" => {
                        let msg = v.get("msg").and_then(|x| x.as_str()).unwrap_or("unknown").to_string();
                        warn!(msg=%msg, frames_sent, "tcp_client: infer_worker reported error");
                        if events_tx.blocking_send(StreamMessage::Error(msg)).is_err() {
                            return;
                        }
                        got_frame = true;
                    }
                    other => warn!(tag=%other, "tcp_client: unknown message type"),
                }
            }
        }
        info!(frames_sent, "tcp_client: session ended");
    });

    Ok(StreamHandle { frame_tx, events_rx, join })
}

fn connect_with_retry(
    addr: &str,
    retries: usize,
    delay: Duration,
) -> Result<std::net::TcpStream, std::io::Error> {
    let mut last_err = None;
    for attempt in 0..=retries {
        match std::net::TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                if attempt < retries {
                    warn!(addr=%addr, attempt=attempt+1, error=%e, "tcp_client: connect retry");
                    std::thread::sleep(delay);
                }
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap())
}
