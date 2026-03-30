//! Heat transfer equipment models.
//!
//! This module contains models for equipment that transfers heat:
//!
//! - [`HeatExchanger`] - Counter-current two-stream heat exchanger

mod heat_exchanger;

pub use heat_exchanger::{HeatExchanger, HeatExchangerBuilder, HeatExchangerVar};
