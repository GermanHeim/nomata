//! Separation equipment models.
//!
//! This module contains models for equipment that separates streams:
//!
//! - [`FlashSeparator`] - Vapor-liquid equilibrium separator

mod flash_separator;

pub use flash_separator::{FlashOutput, FlashSeparator, FlashSeparatorBuilder, FlashSeparatorVar};
