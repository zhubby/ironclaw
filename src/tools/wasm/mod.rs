//! WASM sandbox for untrusted tool execution.
//!
//! This module provides Wasmtime-based sandboxed execution for tools,
//! following patterns from NEAR blockchain and modern WASM best practices:
//!
//! - **Compile once, instantiate fresh**: Tools are validated and compiled
//!   at registration time. Each execution creates a fresh instance.
//!
//! - **Fuel metering**: CPU usage is limited via Wasmtime's fuel system.
//!
//! - **Memory limits**: Memory growth is bounded via ResourceLimiter.
//!
//! - **Minimal host API**: Only log, time, and optional workspace read.
//!
//! - **Capability-based security**: Features are opt-in via Capabilities.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Tool Registration                            │
//! │  WASM bytes → Validate → Compile (AOT) → PreparedModule (cached)   │
//! └─────────────────────────────────────────────────────────────────────┘
//!                                 │
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Tool Execution                               │
//! │  JSON params → WasmToolWrapper → Fresh Instance → Execute → Result │
//! │                      ↓                    ↓                         │
//! │               ResourceLimiter        HostState                      │
//! │               (memory, fuel)      (log, time, workspace)           │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Security Constraints
//!
//! | Threat | Mitigation |
//! |--------|------------|
//! | CPU exhaustion | Fuel metering |
//! | Memory exhaustion | ResourceLimiter, 10MB default |
//! | Infinite loops | Epoch interruption + tokio timeout |
//! | Filesystem access | No WASI FS, only host workspace_read |
//! | Network access | No network host functions |
//! | Log spam | Max 1000 entries, 4KB per message |
//! | Path traversal | Validate paths (no `..`, no `/` prefix) |
//! | Trap recovery | Discard instance, never reuse |
//! | Side channels | Fresh instance per execution |
//!
//! # Example
//!
//! ```ignore
//! use near_agent::tools::wasm::{WasmToolRuntime, WasmRuntimeConfig, WasmToolWrapper};
//! use near_agent::tools::wasm::host::Capabilities;
//! use std::sync::Arc;
//!
//! // Create runtime
//! let runtime = Arc::new(WasmToolRuntime::new(WasmRuntimeConfig::default())?);
//!
//! // Prepare a tool from WASM bytes
//! let wasm_bytes = std::fs::read("my_tool.wasm")?;
//! let prepared = runtime.prepare("my_tool", &wasm_bytes, None).await?;
//!
//! // Create wrapper with minimal capabilities
//! let tool = WasmToolWrapper::new(runtime, prepared, Capabilities::default());
//!
//! // Execute (implements Tool trait)
//! let output = tool.execute(serde_json::json!({"input": "test"}), &ctx).await?;
//! ```

mod error;
mod host;
mod limits;
mod runtime;
mod wrapper;

pub use error::{TrapCode, TrapInfo, WasmError};
pub use host::{Capabilities, HostState, LogEntry, LogLevel, WorkspaceCapability, WorkspaceReader};
pub use limits::{
    DEFAULT_FUEL_LIMIT, DEFAULT_MEMORY_LIMIT, DEFAULT_TIMEOUT, FuelConfig, ResourceLimits,
    WasmResourceLimiter,
};
pub use runtime::{PreparedModule, WasmRuntimeConfig, WasmToolRuntime};
pub use wrapper::WasmToolWrapper;
