//! Valve model for pressure reduction via throttling.
//!
//! A valve reduces the pressure of a stream adiabatically (isenthalpic throttle).
//! For an ideal throttle the enthalpy is conserved: no work and no heat exchange.
//! In this simplified model the outlet temperature equals the inlet temperature,
//! which is exact for ideal gases and a good first approximation for liquids.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MassFlow>::new()
//!     .with_flow(5.0)
//!     .with_temperature(300.0)
//!     .with_pressure(10e5)
//!     .pure("Water")
//!     .build()?;
//!
//! let mut valve = Valve::new("valve-1")
//!     .with_pressure_drop(8e5)
//!     .build()?;
//!
//! let outlet = valve.process(feed)?;
//! ```
//!
//! # Physics
//!
//! The valve is governed by three algebraic equations:
//!
//! 1. **Mass balance**: F_out = F_in
//! 2. **Pressure spec**: P_out = P_in - dP  (or P_out = P_specified)
//! 3. **Isenthalpic**: T_out = T_in
//!
//! where:
//! - F: mass flow rate [kg/s]
//! - P: pressure [Pa]
//! - T: temperature [K]

use crate::{EquationModel, MassFlow, NomataError, NomataResult, Process, Stream};

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;

/// Variable indices for the valve model.
#[derive(Clone, Copy)]
struct ValveVars {
    f_in: usize,
    t_in: usize,
    p_in: usize,
    f_out: usize,
    t_out: usize,
    p_out: usize,
}

impl Default for ValveVars {
    fn default() -> Self {
        ValveVars { f_in: 0, t_in: 1, p_in: 2, f_out: 3, t_out: 4, p_out: 5 }
    }
}

/// Valve for adiabatic pressure reduction.
///
/// The valve model defines algebraic equations that can be solved using
/// either Sequential-Modular (SM) or Equation-Oriented (EO) approaches.
#[derive(Debug)]
pub struct Valve {
    name: String,
    pressure_drop: Option<f64>,
    outlet_pressure: Option<f64>,
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,

    // Variable values: F_in, T_in, P_in, F_out, T_out, P_out
    vars: [f64; 6],
}

impl Valve {
    /// Creates a new valve builder.
    pub fn new(name: &str) -> ValveBuilder {
        ValveBuilder {
            name: name.to_string(),
            pressure_drop: None,
            outlet_pressure: None,
            #[cfg(feature = "thermodynamics")]
            fluid: None,
        }
    }

    /// Gets the valve name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the pressure drop specification [Pa], if any.
    pub fn pressure_drop_spec(&self) -> Option<f64> {
        self.pressure_drop
    }

    /// Gets the outlet pressure specification [Pa], if any.
    pub fn outlet_pressure_spec(&self) -> Option<f64> {
        self.outlet_pressure
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let idx = ValveVars::default();
        let f_in = vars[idx.f_in];
        let t_in = vars[idx.t_in];
        let f_out = vars[idx.f_out];
        let t_out = vars[idx.t_out];
        let p_in = vars[idx.p_in];
        let p_out = vars[idx.p_out];

        let r1 = f_out - f_in;

        let r2 = if let Some(dp) = self.pressure_drop {
            p_out - (p_in - dp)
        } else if let Some(p_spec) = self.outlet_pressure {
            p_out - p_spec
        } else {
            S::from(0.0)
        };

        let r3 = t_out - t_in;

        vec![r1, r2, r3]
    }
}

impl EquationModel for Valve {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_variables(&self) -> usize {
        6
    }

    fn n_equations(&self) -> usize {
        3
    }

    fn variable_names(&self) -> Vec<&str> {
        vec!["F_in", "T_in", "P_in", "F_out", "T_out", "P_out"]
    }

    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        self.residuals_generic(vars)
    }

    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64> {
        self.residuals_generic(vars)
    }

    fn get_variables(&self) -> Vec<f64> {
        self.vars.to_vec()
    }

    fn set_variables(&mut self, vars: &[f64]) {
        for (i, &v) in vars.iter().enumerate().take(6) {
            self.vars[i] = v;
        }
    }

    fn specified_indices(&self) -> Vec<usize> {
        let idx = ValveVars::default();
        vec![idx.f_in, idx.t_in, idx.p_in]
    }

    fn n_inlet_ports(&self) -> usize {
        1
    }

    fn n_outlet_ports(&self) -> usize {
        1
    }

    fn inlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![0, 1, 2] // F_in, T_in, P_in
    }

    fn outlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![3, 4, 5] // F_out, T_out, P_out
    }
}

impl Process for Valve {
    type Input = Stream<MassFlow>;
    type Output = Stream<MassFlow>;

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let inlet_data = input.to_data();
        let idx = ValveVars::default();

        // Set inlet variables
        self.vars[idx.f_in] = inlet_data.flow;
        self.vars[idx.t_in] = inlet_data.temperature;
        self.vars[idx.p_in] = inlet_data.pressure;

        let p_out = if let Some(dp) = self.pressure_drop {
            inlet_data.pressure - dp
        } else if let Some(p) = self.outlet_pressure {
            p
        } else {
            return Err(NomataError::Configuration(
                "Valve requires either pressure_drop or outlet_pressure".to_string(),
            ));
        };

        if p_out <= 0.0 {
            return Err(NomataError::Validation(format!(
                "Valve outlet pressure is non-positive: {p_out} Pa"
            )));
        }

        // Isenthalpic: T_out is found from h(T_out, P_out) = h(T_in, P_in).
        // With a fluid set, look up enthalpy at inlet then invert at outlet
        // pressure via props_ph.  Fall back to T_out = T_in (ideal gas) when
        // the thermodynamics feature is disabled or no fluid is configured.
        #[cfg(feature = "thermodynamics")]
        let outlet_temperature = if let Some(ref fluid) = self.fluid {
            // Get specific enthalpy at inlet conditions
            let h_in = fluid
                .props_pt(inlet_data.pressure, inlet_data.temperature)
                .map(|p| p.enthalpy)
                .unwrap_or(0.0);
            if h_in == 0.0 {
                // Property lookup failed; fall back to ideal-gas approximation
                inlet_data.temperature
            } else {
                // Find T_out such that h(P_out, T_out) = h_in
                fluid.props_ph(p_out, h_in).map(|p| p.temperature).unwrap_or(inlet_data.temperature)
            }
        } else {
            inlet_data.temperature
        };
        #[cfg(not(feature = "thermodynamics"))]
        let outlet_temperature = inlet_data.temperature;

        // Set outlet variables
        self.vars[idx.f_out] = inlet_data.flow;
        self.vars[idx.t_out] = outlet_temperature;
        self.vars[idx.p_out] = p_out;

        Stream::<MassFlow>::new()
            .with_flow(inlet_data.flow)
            .with_temperature(outlet_temperature)
            .with_pressure(p_out)
            .with_composition(
                &inlet_data.components.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &inlet_data.composition,
            )
            .build()
    }
}

/// Builder for Valve.
#[derive(Debug)]
pub struct ValveBuilder {
    name: String,
    pressure_drop: Option<f64>,
    outlet_pressure: Option<f64>,
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,
}

impl ValveBuilder {
    /// Sets the pressure drop across the valve [Pa].
    ///
    /// `P_out = P_in - pressure_drop`
    pub fn with_pressure_drop(mut self, dp: f64) -> Self {
        self.pressure_drop = Some(dp);
        self
    }

    /// Sets the target outlet pressure [Pa].
    pub fn with_outlet_pressure(mut self, pressure: f64) -> Self {
        self.outlet_pressure = Some(pressure);
        self
    }

    /// Sets the fluid for thermodynamic calculations.
    ///
    /// When set, the outlet temperature is computed from the isenthalpic
    /// condition `h(T_out, P_out) = h(T_in, P_in)` using `props_ph`, rather
    /// than the ideal-gas approximation `T_out = T_in`.
    #[cfg(feature = "thermodynamics")]
    pub fn with_fluid(mut self, fluid: impl Into<Fluid>) -> Self {
        self.fluid = Some(fluid.into());
        self
    }

    /// Builds the valve.
    pub fn build(self) -> NomataResult<Valve> {
        if self.pressure_drop.is_none() && self.outlet_pressure.is_none() {
            return Err(NomataError::Configuration(
                "Valve requires either pressure_drop or outlet_pressure".to_string(),
            ));
        }

        Ok(Valve {
            name: self.name,
            pressure_drop: self.pressure_drop,
            outlet_pressure: self.outlet_pressure,
            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,
            vars: [0.0; 6],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valve_equations() {
        let mut valve = Valve::new("valve-1").with_pressure_drop(5e5).build().unwrap();

        let f: f64 = 5.0;
        let t_in: f64 = 350.0;
        let p_in: f64 = 10e5;
        let p_out: f64 = p_in - 5e5;
        let t_out: f64 = t_in; // Isenthalpic

        valve.set_variables(&[f, t_in, p_in, f, t_out, p_out]);

        let residuals = valve.residuals(&valve.get_variables());
        for (i, r) in residuals.iter().enumerate() {
            assert!(r.abs() < 1e-10, "Residual {} = {} (should be 0)", i, r);
        }
    }

    #[test]
    fn test_valve_variable_counts() {
        let valve = Valve::new("valve-1").with_pressure_drop(1e5).build().unwrap();

        assert_eq!(valve.n_variables(), 6);
        assert_eq!(valve.n_equations(), 3);
        assert_eq!(valve.variable_names().len(), 6);
        assert_eq!(valve.specified_indices().len(), 3);
        assert_eq!(valve.free_indices().len(), 3);
    }

    #[test]
    fn test_valve_process() {
        let mut valve = Valve::new("valve-1").with_pressure_drop(8e5).build().unwrap();

        let feed = Stream::<MassFlow>::new()
            .with_flow(5.0)
            .with_temperature(300.0)
            .with_pressure(10e5)
            .build()
            .unwrap();

        let outlet = valve.process(feed).unwrap();
        let data = outlet.to_data();

        assert!((data.flow - 5.0).abs() < 1e-10);
        assert!((data.temperature - 300.0).abs() < 1e-10);
        assert!((data.pressure - 2e5).abs() < 1e-10);
    }

    #[test]
    fn test_valve_outlet_pressure_spec() {
        let mut valve = Valve::new("valve-1").with_outlet_pressure(1e5).build().unwrap();

        let feed = Stream::<MassFlow>::new()
            .with_flow(2.0)
            .with_temperature(400.0)
            .with_pressure(5e5)
            .build()
            .unwrap();

        let outlet = valve.process(feed).unwrap();
        let data = outlet.to_data();

        assert!((data.pressure - 1e5).abs() < 1e-10);
        assert!((data.temperature - 400.0).abs() < 1e-10);
    }
}
