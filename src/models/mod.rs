//! Unit operation models for process simulation.
//!
//! This module provides implementations of common chemical engineering unit operations,
//! organized into logical categories:
//!
//! ## Pressure Changers ([`pressure_changers`])
//! - [`Pump`] - Liquid pressure increase with efficiency
//! - [`Compressor`] - Gas pressure increase with isentropic efficiency
//! - [`Valve`] - Adiabatic pressure reduction (isenthalpic throttle)
//!
//! ## Heat Transfer ([`heat_transfer`])
//! - [`Heater`] - Single-stream heater/cooler with heat duty or temperature spec
//! - [`HeatExchanger`] - Counter-current two-stream heat exchanger
//!
//! ## Separators ([`separators`])
//! - [`FlashSeparator`] - Vapor-liquid equilibrium separator
//!
//! ## Stream Operations ([`stream_ops`])
//! - [`Mixer<N>`](Mixer) - Combines N inlet streams (const generic)
//! - [`Splitter<N>`](Splitter) - Divides one stream into N outlets (const generic)
//! - [`Map`] - Lumps or delumps component representations
//!
//! All models implement the `Process` trait for type-safe stream processing:
//! ```ignore
//! let outlet = pump.process(inlet)?;
//! let outputs: [Stream<MassFlow>; 3] = splitter.process(feed)?;
//! let vapor = flash.process(feed)?.vapor;
//! ```

pub mod heat_transfer;
pub mod pressure_changers;
pub mod separators;
pub mod stream_ops;

// Re-export all models at the top level for convenience
pub use heat_transfer::{HeatExchanger, HeatExchangerBuilder, Heater, HeaterBuilder};
pub use pressure_changers::{Compressor, CompressorBuilder, Pump, PumpBuilder, Valve, ValveBuilder};
pub use separators::{FlashOutput, FlashSeparator, FlashSeparatorBuilder};
pub use stream_ops::{ComponentMapping, Map, MapBuilder, MappingDirection, Mixer, MixerBuilder, Splitter, SplitterBuilder};

