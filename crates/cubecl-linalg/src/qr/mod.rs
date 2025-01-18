mod base;

// contains QR decomposition kernels.
pub mod kernels;

pub use base::*;

/// Tests for QR decomposition kernels
#[cfg(feature = "export_tests")]
pub mod tests;
