//! Common chemical engineering unit operation models.
//!
//! This module provides typed implementations of standard unit operations
//! organized as individual files:
//!
//! - **CSTR**: Continuous Stirred Tank Reactor with reaction kinetics
//! - **PFR**: Plug Flow Reactor with spatial discretization
//! - **HeatExchanger**: Shell-and-tube heat exchanger
//! - **FlashSeparator**: Vapor-liquid equilibrium separator
//! - **Mixer**: Combines multiple inlet streams
//! - **Splitter**: Divides one stream into multiple outlets
//! - **Compressor**: Gas compression with isentropic efficiency
//! - **Pump**: Liquid pumping with head calculations
//! - **MAP**: Component mapper for lumping/delumping operations
//!
//! All models use the type system to enforce correct connections and
//! proper variable roles (Parameter, Algebraic, Differential).
//!
//! ## Thermodynamics Integration
//!
//! When the `thermodynamics` feature is enabled, models can automatically
//! compute physical properties (density, heat capacity, vapor pressure, etc.)
//! using the CoolProp library through rfluids.
//!
//! ## Example
//!
//! ```
//! use nomata::models::CSTR;
//! use nomata::BalanceBuilder;
//!
//! let cstr = CSTR::new(100.0, 1.0, 350.0)
//!     .with_kinetics(1e8, 8000.0)
//!     .with_thermodynamics(-45000.0, 1000.0, 4184.0);
//!
//! // Build balance equations with compile-time verification
//! let mass_eq = cstr.mass_balance();
//! ```

mod compressor;
mod cstr;
mod flash_separator;
mod heat_exchanger;
mod map;
mod mixer;
mod pfr;
mod pump;
mod splitter;

pub use compressor::Compressor;
pub use cstr::CSTR;
pub use flash_separator::FlashSeparator;
pub use heat_exchanger::HeatExchanger;
pub use map::MAP;
pub use mixer::Mixer;
pub use pfr::PFR;
pub use pump::Pump;
pub use splitter::Splitter;
