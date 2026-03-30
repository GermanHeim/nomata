//! Stream manipulation models.
//!
//! This module contains models for combining, dividing, and transforming streams:
//!
//! - [`Mixer`] - Combines N inlet streams into one outlet
//! - [`Splitter`] - Divides one inlet stream into N outlets
//! - [`Map`] - Lumps or delumps component representations

mod map;
mod mixer;
mod splitter;

pub use map::{ComponentMapping, Map, MapBuilder, MapVar, MappingDirection};
pub use mixer::{Mixer, MixerBuilder};
pub use splitter::{Splitter, SplitterBuilder};

/// Selects one scalar property within a single stream port.
///
/// Used with [`Mixer::inlet_var`], [`Mixer::outlet_var`],
/// [`Splitter::inlet_var`], and [`Splitter::outlet_var`] to avoid hard-coded
/// index offsets when reading variables from multi-port units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortVar {
    /// Flow rate \[kg/s or mol/s\]
    Flow,
    /// Temperature \[K\]
    Temperature,
    /// Pressure \[Pa\]
    Pressure,
}
