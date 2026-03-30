//! Compressor model for gas pressure increase.
//!
//! A compressor increases the pressure of a gas stream with associated work input.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MassFlow>::new()
//!     .with_flow(1.0)
//!     .with_temperature(300.0)
//!     .with_pressure(1e5)
//!     .pure("Air")
//!     .build()?;
//!
//! let mut compressor = Compressor::new("comp-1")
//!     .with_efficiency(0.72)
//!     .with_pressure_ratio(3.0)
//!     .build()?;
//!
//! let outlet = compressor.process(feed)?;
//! ```
//!
//! # Physics
//!
//! The compressor is governed by the following algebraic equations:
//!
//! 1. **Mass balance**: F_out = F_in
//! 2. **Pressure spec**: P_out = P_in * ratio (or P_out = P_specified)
//! 3. **Isentropic temperature**: T_out_is = T_in * (P_out/P_in)^((gamma-1)/gamma)
//! 4. **Actual temperature**: T_out = T_in + (T_out_is - T_in) / eta
//! 5. **Work equation**: W = F_in * Cp * (T_out - T_in)

use crate::{EquationModel, MassFlow, NomataError, NomataResult, Process, Stream};

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;

/// Named accessor for compressor variables.
///
/// Use with [`Compressor::var`] to read individual variable values by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorVar {
    /// Inlet flow rate \[kg/s\]
    InletFlow,
    /// Inlet temperature \[K\]
    InletTemperature,
    /// Inlet pressure \[Pa\]
    InletPressure,
    /// Outlet flow rate \[kg/s\]
    OutletFlow,
    /// Outlet temperature \[K\]
    OutletTemperature,
    /// Outlet pressure \[Pa\]
    OutletPressure,
    /// Shaft work input \[W\]
    ShaftWork,
}

impl CompressorVar {
    fn index(self) -> usize {
        match self {
            CompressorVar::InletFlow => 0,
            CompressorVar::InletTemperature => 1,
            CompressorVar::InletPressure => 2,
            CompressorVar::OutletFlow => 3,
            CompressorVar::OutletTemperature => 4,
            CompressorVar::OutletPressure => 5,
            CompressorVar::ShaftWork => 6,
        }
    }
}

/// Variable indices for the compressor model.
#[derive(Clone, Copy)]
struct CompressorVars {
    f_in: usize,
    t_in: usize,
    p_in: usize,
    f_out: usize,
    t_out: usize,
    p_out: usize,
    work: usize,
}

impl Default for CompressorVars {
    fn default() -> Self {
        CompressorVars { f_in: 0, t_in: 1, p_in: 2, f_out: 3, t_out: 4, p_out: 5, work: 6 }
    }
}

/// Compressor for gas pressure increase.
///
/// The compressor model defines algebraic equations that can be solved using
/// either Sequential-Modular (SM) or Equation-Oriented (EO) approaches.
#[derive(Debug)]
pub struct Compressor {
    name: String,
    efficiency: f64,
    pressure_ratio: Option<f64>,
    outlet_pressure: Option<f64>,
    gamma: f64,         // Cp/Cv
    heat_capacity: f64, // Cp [J/(kg*K)]
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,

    // Variable values
    vars: [f64; 7], // F_in, T_in, P_in, F_out, T_out, P_out, W
}

impl Compressor {
    /// Creates a new compressor builder.
    pub fn new(name: &str) -> CompressorBuilder {
        CompressorBuilder {
            name: name.to_string(),
            efficiency: 0.72,
            pressure_ratio: None,
            outlet_pressure: None,
            gamma: 1.4,            // Air-like
            heat_capacity: 1005.0, // Air Cp
            #[cfg(feature = "thermodynamics")]
            fluid: None,
        }
    }

    /// Gets the compressor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the compressor efficiency.
    pub fn efficiency(&self) -> f64 {
        self.efficiency
    }

    /// Gets the computed work [W] from the last process call.
    pub fn work(&self) -> f64 {
        self.vars[CompressorVars::default().work]
    }

    /// Returns the current value of a named variable.
    pub fn var(&self, v: CompressorVar) -> f64 {
        self.vars[v.index()]
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let idx = CompressorVars::default();
        let f_in = vars[idx.f_in];
        let t_in = vars[idx.t_in];
        let p_in = vars[idx.p_in];
        let f_out = vars[idx.f_out];
        let t_out = vars[idx.t_out];
        let p_out = vars[idx.p_out];
        let w = vars[idx.work];

        let r1 = f_out - f_in;

        let r2 = if let Some(ratio) = self.pressure_ratio {
            p_out - p_in * ratio
        } else if let Some(p_spec) = self.outlet_pressure {
            p_out - p_spec
        } else {
            S::from(0.0)
        };

        // Use stored f64 for branch condition (p_in is a specified inlet).
        let p_in_f64 = self.vars[idx.p_in];
        let ratio: S = if p_in_f64.abs() > 1e-10 { p_out / p_in } else { S::from(1.0) };
        let exponent = (self.gamma - 1.0) / self.gamma;
        let t_out_isentropic = t_in * ratio.powf(exponent);
        let expected_t_out = t_in + (t_out_isentropic - t_in) / self.efficiency;
        let r3 = t_out - expected_t_out;

        let expected_work = f_in * self.heat_capacity * (t_out - t_in);
        let r4 = w - expected_work;

        vec![r1, r2, r3, r4]
    }
}

impl EquationModel for Compressor {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_variables(&self) -> usize {
        7
    }

    fn n_equations(&self) -> usize {
        4
    }

    fn variable_names(&self) -> Vec<&str> {
        vec!["F_in", "T_in", "P_in", "F_out", "T_out", "P_out", "W"]
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
        for (i, &v) in vars.iter().enumerate().take(7) {
            self.vars[i] = v;
        }
    }

    fn specified_indices(&self) -> Vec<usize> {
        // Inlet variables are specified (known from upstream)
        let idx = CompressorVars::default();
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

impl Process for Compressor {
    type Input = Stream<MassFlow>;
    type Output = Stream<MassFlow>;

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let inlet_data = input.to_data();
        let idx = CompressorVars::default();

        // Set inlet variables from input stream
        self.vars[idx.f_in] = inlet_data.flow;
        self.vars[idx.t_in] = inlet_data.temperature;
        self.vars[idx.p_in] = inlet_data.pressure;

        // Calculate outlet pressure
        let p_out = if let Some(ratio) = self.pressure_ratio {
            inlet_data.pressure * ratio
        } else if let Some(p) = self.outlet_pressure {
            p
        } else {
            return Err(NomataError::Configuration(
                "Compressor requires either pressure_ratio or outlet_pressure".to_string(),
            ));
        };

        let ratio = p_out / inlet_data.pressure;

        // Get gamma and cp from thermodynamics if available
        #[cfg(feature = "thermodynamics")]
        let (gamma, cp) = if let Some(ref fluid) = self.fluid {
            if let Ok(props) = fluid.props_pt(inlet_data.pressure, inlet_data.temperature) {
                let gamma = props.cp / props.cv;
                (gamma, props.cp)
            } else {
                (self.gamma, self.heat_capacity)
            }
        } else {
            (self.gamma, self.heat_capacity)
        };

        #[cfg(not(feature = "thermodynamics"))]
        let (gamma, cp) = (self.gamma, self.heat_capacity);

        // Isentropic temperature ratio
        let exponent = (gamma - 1.0) / gamma;
        let t_out_isentropic = inlet_data.temperature * ratio.powf(exponent);

        // Actual outlet temperature (accounting for efficiency)
        let dt_isentropic = t_out_isentropic - inlet_data.temperature;
        let dt_actual = dt_isentropic / self.efficiency;
        let outlet_temperature = inlet_data.temperature + dt_actual;

        // Work
        let specific_work = cp * dt_actual;
        let work = inlet_data.flow * specific_work;

        // Set outlet variables
        self.vars[idx.f_out] = inlet_data.flow;
        self.vars[idx.t_out] = outlet_temperature;
        self.vars[idx.p_out] = p_out;
        self.vars[idx.work] = work;

        // Verify equations are satisfied
        let residuals = self.residuals(&self.vars);
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        if max_residual > 1e-6 {
            eprintln!(
                "Warning: Compressor '{}' equations not satisfied, max residual = {:.2e}",
                self.name, max_residual
            );
        }

        // Build output stream
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

/// Builder for Compressor.
#[derive(Debug)]
pub struct CompressorBuilder {
    name: String,
    efficiency: f64,
    pressure_ratio: Option<f64>,
    outlet_pressure: Option<f64>,
    gamma: f64,
    heat_capacity: f64,
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,
}

impl CompressorBuilder {
    /// Sets the isentropic efficiency (0-1).
    pub fn with_efficiency(mut self, eta: f64) -> Self {
        self.efficiency = eta;
        self
    }

    /// Sets the pressure ratio (P_out / P_in).
    pub fn with_pressure_ratio(mut self, ratio: f64) -> Self {
        self.pressure_ratio = Some(ratio);
        self
    }

    /// Sets the target outlet pressure [Pa].
    pub fn with_outlet_pressure(mut self, pressure: f64) -> Self {
        self.outlet_pressure = Some(pressure);
        self
    }

    /// Sets the heat capacity ratio (Cp/Cv).
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the heat capacity [J/(kg*K)].
    pub fn with_heat_capacity(mut self, cp: f64) -> Self {
        self.heat_capacity = cp;
        self
    }

    /// Sets the fluid for thermodynamic calculations.
    #[cfg(feature = "thermodynamics")]
    pub fn with_fluid(mut self, fluid: impl Into<Fluid>) -> Self {
        self.fluid = Some(fluid.into());
        self
    }

    /// Builds the compressor.
    pub fn build(self) -> NomataResult<Compressor> {
        if self.efficiency <= 0.0 || self.efficiency > 1.0 {
            return Err(NomataError::Validation(format!(
                "Efficiency must be in (0, 1], got {}",
                self.efficiency
            )));
        }

        if self.pressure_ratio.is_none() && self.outlet_pressure.is_none() {
            return Err(NomataError::Configuration(
                "Compressor requires either pressure_ratio or outlet_pressure".to_string(),
            ));
        }

        Ok(Compressor {
            name: self.name,
            efficiency: self.efficiency,
            pressure_ratio: self.pressure_ratio,
            outlet_pressure: self.outlet_pressure,
            gamma: self.gamma,
            heat_capacity: self.heat_capacity,
            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,
            vars: [0.0; 7],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_equations() {
        let mut compressor = Compressor::new("comp-1")
            .with_efficiency(0.72)
            .with_pressure_ratio(3.0)
            .build()
            .unwrap();

        // Set inlet conditions
        let f: f64 = 1.0; // kg/s
        let t_in: f64 = 300.0; // K
        let p_in: f64 = 1e5; // Pa
        let p_out: f64 = 3e5; // Pa (ratio = 3)
        let gamma: f64 = 1.4;
        let cp: f64 = 1005.0;

        // Calculate expected outlet temperature
        let ratio: f64 = p_out / p_in;
        let exponent: f64 = (gamma - 1.0) / gamma;
        let t_out_is: f64 = t_in * ratio.powf(exponent);
        let t_out: f64 = t_in + (t_out_is - t_in) / 0.72;
        let w: f64 = f * cp * (t_out - t_in);

        compressor.set_variables(&[f, t_in, p_in, f, t_out, p_out, w]);

        let residuals = compressor.residuals(&compressor.get_variables());

        // All residuals should be zero at solution
        for (i, r) in residuals.iter().enumerate() {
            assert!(r.abs() < 1e-6, "Residual {} = {} (should be 0)", i, r);
        }
    }

    #[test]
    fn test_compressor_variable_counts() {
        let compressor = Compressor::new("comp-1").with_pressure_ratio(2.0).build().unwrap();

        assert_eq!(compressor.n_variables(), 7);
        assert_eq!(compressor.n_equations(), 4);
        assert_eq!(compressor.variable_names().len(), 7);
        assert_eq!(compressor.specified_indices().len(), 3);
        assert_eq!(compressor.free_indices().len(), 4);
    }
}
