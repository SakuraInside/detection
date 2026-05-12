//! runtime-core: оркестратор Integra Native (Rust ↔ C++/CUDA через FFI).
//!
//! Этот crate имеет одновременно несколько бинарей (`main.rs`, `bin/backend_gateway.rs`)
//! и общую library-секцию для модулей, которые они переиспользуют.
//!
//! Публичные модули:
//!   * [`integra`] — safe Rust обёртка над `integra_ffi` (C ABI к `integra_core`).
//!     Загрузка DLL/so в рантайме, типизированные события, `spawn_stream` с
//!     bounded-каналами и drop-oldest семантикой для live-видео.

pub mod integra;
