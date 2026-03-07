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

pub use map::{ComponentMapping, Map, MapBuilder, MappingDirection};
pub use mixer::{Mixer, MixerBuilder};
pub use splitter::{Splitter, SplitterBuilder};
