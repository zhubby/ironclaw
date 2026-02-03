//! Host functions for WASM sandbox.
//!
//! Implements a minimal, security-focused host API following VMLogic patterns
//! from NEAR blockchain. The principle is: deny by default, grant minimal capabilities.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::tools::wasm::error::WasmError;

/// Maximum log entries per execution (prevents log spam attacks).
const MAX_LOG_ENTRIES: usize = 1000;

/// Maximum bytes per log message.
const MAX_LOG_MESSAGE_BYTES: usize = 4096;

/// Log levels matching the WIT interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// A single log entry from WASM execution.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp_millis: u64,
}

/// Capabilities that can be granted to a WASM tool.
///
/// By default, tools have NO capabilities. Each must be explicitly granted.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    /// If Some, tool can read from workspace at these paths.
    /// Empty vec means workspace access granted but no paths allowed yet.
    /// None means workspace access completely disabled.
    pub workspace_read: Option<WorkspaceCapability>,
}

/// Workspace read capability configuration.
#[derive(Clone, Default)]
pub struct WorkspaceCapability {
    /// Allowed path prefixes (e.g., ["context/", "daily/"]).
    /// Empty means all paths allowed (within safety constraints).
    pub allowed_prefixes: Vec<String>,
    /// Function to actually read from workspace.
    /// This is injected by the runtime to avoid coupling to workspace impl.
    pub reader: Option<Arc<dyn WorkspaceReader>>,
}

impl std::fmt::Debug for WorkspaceCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceCapability")
            .field("allowed_prefixes", &self.allowed_prefixes)
            .field("reader", &self.reader.is_some())
            .finish()
    }
}

/// Trait for reading from workspace (allows mocking in tests).
pub trait WorkspaceReader: Send + Sync {
    fn read(&self, path: &str) -> Option<String>;
}

/// Host state maintained during WASM execution.
///
/// This is the "VMLogic" equivalent, it tracks all side effects and enforces limits.
#[derive(Debug)]
pub struct HostState {
    /// Collected log entries.
    logs: Vec<LogEntry>,
    /// Whether logging is still allowed (false after MAX_LOG_ENTRIES).
    logging_enabled: bool,
    /// Granted capabilities.
    capabilities: Capabilities,
    /// Count of log entries dropped due to rate limiting.
    logs_dropped: usize,
}

impl HostState {
    /// Create a new host state with the given capabilities.
    pub fn new(capabilities: Capabilities) -> Self {
        Self {
            logs: Vec::new(),
            logging_enabled: true,
            capabilities,
            logs_dropped: 0,
        }
    }

    /// Create a minimal host state with no capabilities.
    pub fn minimal() -> Self {
        Self::new(Capabilities::default())
    }

    /// Log a message from WASM.
    ///
    /// Returns Ok(()) if logged, Err if rate limited or too long.
    pub fn log(&mut self, level: LogLevel, message: String) -> Result<(), WasmError> {
        if !self.logging_enabled {
            self.logs_dropped += 1;
            return Ok(()); // Silently drop, don't fail execution
        }

        if self.logs.len() >= MAX_LOG_ENTRIES {
            self.logging_enabled = false;
            self.logs_dropped += 1;
            tracing::warn!(
                "WASM log limit reached ({} entries), further logs dropped",
                MAX_LOG_ENTRIES
            );
            return Ok(());
        }

        // Truncate overly long messages
        let message = if message.len() > MAX_LOG_MESSAGE_BYTES {
            let mut truncated = message[..MAX_LOG_MESSAGE_BYTES].to_string();
            truncated.push_str("... (truncated)");
            truncated
        } else {
            message
        };

        let timestamp_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.logs.push(LogEntry {
            level,
            message,
            timestamp_millis,
        });

        Ok(())
    }

    /// Get current timestamp in milliseconds.
    pub fn now_millis(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Read from workspace if capability granted.
    pub fn workspace_read(&self, path: &str) -> Result<Option<String>, WasmError> {
        // Check if workspace capability is granted
        let capability = match &self.capabilities.workspace_read {
            Some(cap) => cap,
            None => return Ok(None), // No capability, return None
        };

        // Validate path (security critical)
        validate_workspace_path(path)?;

        // Check allowed prefixes if any are specified
        if !capability.allowed_prefixes.is_empty() {
            let allowed = capability
                .allowed_prefixes
                .iter()
                .any(|prefix| path.starts_with(prefix));
            if !allowed {
                tracing::debug!(
                    path = path,
                    allowed = ?capability.allowed_prefixes,
                    "WASM workspace read denied: path not in allowed prefixes"
                );
                return Ok(None);
            }
        }

        // Actually read from workspace
        match &capability.reader {
            Some(reader) => Ok(reader.read(path)),
            None => Ok(None), // No reader configured
        }
    }

    /// Get collected logs after execution.
    pub fn take_logs(&mut self) -> Vec<LogEntry> {
        std::mem::take(&mut self.logs)
    }

    /// Get number of logs dropped due to rate limiting.
    pub fn logs_dropped(&self) -> usize {
        self.logs_dropped
    }
}

/// Validate a workspace path for security.
///
/// Blocks path traversal attacks and absolute paths.
fn validate_workspace_path(path: &str) -> Result<(), WasmError> {
    // Block absolute paths
    if path.starts_with('/') {
        return Err(WasmError::PathTraversalBlocked(
            "absolute paths not allowed".to_string(),
        ));
    }

    // Block path traversal
    if path.contains("..") {
        return Err(WasmError::PathTraversalBlocked(
            "parent directory references not allowed".to_string(),
        ));
    }

    // Block null bytes
    if path.contains('\0') {
        return Err(WasmError::PathTraversalBlocked(
            "null bytes not allowed".to_string(),
        ));
    }

    // Block Windows-style absolute paths (just in case)
    if path.len() >= 2 && path.chars().nth(1) == Some(':') {
        return Err(WasmError::PathTraversalBlocked(
            "Windows-style paths not allowed".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tools::wasm::host::{
        Capabilities, HostState, LogLevel, MAX_LOG_ENTRIES, MAX_LOG_MESSAGE_BYTES,
        WorkspaceCapability, WorkspaceReader, validate_workspace_path,
    };
    use std::sync::Arc;

    struct MockReader {
        content: String,
    }

    impl WorkspaceReader for MockReader {
        fn read(&self, _path: &str) -> Option<String> {
            Some(self.content.clone())
        }
    }

    #[test]
    fn test_logging_basic() {
        let mut state = HostState::minimal();
        state
            .log(LogLevel::Info, "test message".to_string())
            .unwrap();

        let logs = state.take_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].level, LogLevel::Info);
        assert_eq!(logs[0].message, "test message");
    }

    #[test]
    fn test_logging_rate_limit() {
        let mut state = HostState::minimal();

        // Fill up to limit
        for i in 0..MAX_LOG_ENTRIES {
            state
                .log(LogLevel::Debug, format!("message {}", i))
                .unwrap();
        }

        // This should be dropped silently
        state
            .log(LogLevel::Info, "should be dropped".to_string())
            .unwrap();

        assert_eq!(state.take_logs().len(), MAX_LOG_ENTRIES);
        assert_eq!(state.logs_dropped(), 1);
    }

    #[test]
    fn test_logging_truncation() {
        let mut state = HostState::minimal();

        let long_message = "x".repeat(MAX_LOG_MESSAGE_BYTES + 1000);
        state.log(LogLevel::Info, long_message).unwrap();

        let logs = state.take_logs();
        assert!(logs[0].message.len() <= MAX_LOG_MESSAGE_BYTES + 20); // +20 for truncation suffix
        assert!(logs[0].message.ends_with("... (truncated)"));
    }

    #[test]
    fn test_now_millis() {
        let state = HostState::minimal();
        let now = state.now_millis();
        // Should be a reasonable timestamp (after 2020)
        assert!(now > 1577836800000); // Jan 1, 2020
    }

    #[test]
    fn test_workspace_read_no_capability() {
        let state = HostState::minimal();
        let result = state.workspace_read("context/test.md").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_workspace_read_with_capability() {
        let reader = Arc::new(MockReader {
            content: "test content".to_string(),
        });

        let capabilities = Capabilities {
            workspace_read: Some(WorkspaceCapability {
                allowed_prefixes: vec![],
                reader: Some(reader),
            }),
        };

        let state = HostState::new(capabilities);
        let result = state.workspace_read("context/test.md").unwrap();
        assert_eq!(result, Some("test content".to_string()));
    }

    #[test]
    fn test_workspace_read_prefix_restriction() {
        let reader = Arc::new(MockReader {
            content: "test content".to_string(),
        });

        let capabilities = Capabilities {
            workspace_read: Some(WorkspaceCapability {
                allowed_prefixes: vec!["context/".to_string()],
                reader: Some(reader),
            }),
        };

        let state = HostState::new(capabilities);

        // Allowed prefix
        let result = state.workspace_read("context/test.md").unwrap();
        assert!(result.is_some());

        // Disallowed prefix
        let result = state.workspace_read("secrets/api_key.txt").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_path_validation_blocks_traversal() {
        assert!(validate_workspace_path("../etc/passwd").is_err());
        assert!(validate_workspace_path("context/../secrets").is_err());
        assert!(validate_workspace_path("context/test/../../secrets").is_err());
    }

    #[test]
    fn test_path_validation_blocks_absolute() {
        assert!(validate_workspace_path("/etc/passwd").is_err());
        assert!(validate_workspace_path("/context/test.md").is_err());
    }

    #[test]
    fn test_path_validation_blocks_null_bytes() {
        assert!(validate_workspace_path("context/test\0.md").is_err());
    }

    #[test]
    fn test_path_validation_blocks_windows_paths() {
        assert!(validate_workspace_path("C:\\Windows\\System32").is_err());
        assert!(validate_workspace_path("D:secrets").is_err());
    }

    #[test]
    fn test_path_validation_allows_valid_paths() {
        assert!(validate_workspace_path("context/test.md").is_ok());
        assert!(validate_workspace_path("daily/2024-01-15.md").is_ok());
        assert!(validate_workspace_path("projects/alpha/notes.md").is_ok());
        assert!(validate_workspace_path("MEMORY.md").is_ok());
    }
}
