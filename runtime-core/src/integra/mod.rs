//! Safe Rust обёртка над C ABI из `native/include/integra/integra_ffi.h`.
//!
//! Дерево модулей:
//!   * [`ffi`]      — низкоуровневые `extern "C"` сигнатуры + `IntegraLib`
//!                    (загрузка DLL/so в рантайме через `libloading`).
//!   * [`pipeline`] — safe `Pipeline { handle, lib }` с `push_frame()` и
//!                    trampoline-callback для сбора событий.
//!   * [`events`]   — типизированные структуры (deserialize из JSON):
//!                    `AlarmEvent`, `TrackSnapshot`, `PersonDet`, `FrameResult`.
//!   * [`stream`]   — async-фасад `spawn_stream(cfg)` с bounded-каналами
//!                    (drop-oldest для frames, FIFO для events).

pub mod events;
pub mod ffi;
pub mod pipeline;
pub(crate) mod preview_encode;
pub mod stream;
pub mod tcp_client;

pub use events::{AlarmEvent, FrameResult, FrameStats, PersonDet, StreamMessage, TrackSnapshot};
pub use ffi::{IntegraError, IntegraLib};
pub use pipeline::{Pipeline, PipelineConfig};
pub use preview_encode::{encode_alarm_crop_jpeg, encode_preview_jpeg};
pub use stream::{spawn_stream, Frame, StreamHandle};
pub use tcp_client::spawn_tcp_stream;
