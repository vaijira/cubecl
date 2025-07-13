#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil)]

mod base;
/// Contains QR kernels
pub mod kernels;
/// Tests for QR kernels
#[cfg(feature = "export_tests")]
pub mod tests;

pub use base::*;
