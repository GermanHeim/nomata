//! Pump model for liquid pressure increase.
//!
//! A pump increases the pressure of a liquid stream with associated work input.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MassFlow>::new()
//!     .with_flow(10.0)
//!     .with_temperature(298.15)
//!     .with_pressure(1e5)
//!     .pure("Water")
//!     .build()?;
//!
//! let mut pump = Pump::new("pump-1")
//!     .with_efficiency(0.75)
//!     .with_outlet_pressure(10e5)
//!     .build()?;
//!
//! let outlet = pump.process(feed)?;
//! ```
//!
//! # Physics
//!
//! The pump is governed by the following algebraic equations:
//!
//! 1. **Mass balance**: F_out = F_in
//! 2. **Pressure spec**: P_out = P_in + dP (or P_out = P_specified)
//! 3. **Work equation**: W = F_in * dP / (rho * eta)
//! 4. **Temperature rise**: T_out = T_in + W / (F_in * Cp)
//!
//! where:
//! - F: mass flow rate [kg/s]
//! - P: pressure [Pa]
//! - T: temperature [K]
//! - W: shaft work [W]
//! - rho: density [kg/m^3]
//! - eta: isentropic efficiency [-]
//! - Cp: heat capacity [J/(kg*K)]

use crate::{EquationModel, MassFlow, NomataError, NomataResult, Process, Stream};

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;

/// Named accessor for pump variables.
///
/// Use with [`Pump::var`] to read individual variable values by name instead
/// of by raw index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PumpVar {
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

impl PumpVar {
    fn index(self) -> usize {
        match self {
            PumpVar::InletFlow => 0,
            PumpVar::InletTemperature => 1,
            PumpVar::InletPressure => 2,
            PumpVar::OutletFlow => 3,
            PumpVar::OutletTemperature => 4,
            PumpVar::OutletPressure => 5,
            PumpVar::ShaftWork => 6,
        }
    }
}

/// Variable indices for the pump model.
#[derive(Clone, Copy)]
struct PumpVars {
    f_in: usize,
    t_in: usize,
    p_in: usize,
    f_out: usize,
    t_out: usize,
    p_out: usize,
    work: usize,
}

impl Default for PumpVars {
    fn default() -> Self {
        PumpVars { f_in: 0, t_in: 1, p_in: 2, f_out: 3, t_out: 4, p_out: 5, work: 6 }
    }
}

/// Pump for liquid pressure increase.
///
/// The pump model defines algebraic equations that can be solved using
/// either Sequential-Modular (SM) or Equation-Oriented (EO) approaches.
#[derive(Debug)]
pub struct Pump {
    name: String,
    efficiency: f64,
    outlet_pressure: Option<f64>,
    pressure_rise: Option<f64>,
    density: f64,       // kg/m^3
    heat_capacity: f64, // J/(kg*K)
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,

    // Variable values
    vars: [f64; 7], // F_in, T_in, P_in, F_out, T_out, P_out, W
}

impl Pump {
    /// Creates a new pump builder.
    pub fn new(name: &str) -> PumpBuilder {
        PumpBuilder {
            name: name.to_string(),
            efficiency: 0.75,
            outlet_pressure: None,
            pressure_rise: None,
            density: 1000.0,
            heat_capacity: 4180.0,
            #[cfg(feature = "thermodynamics")]
            fluid: None,
        }
    }

    /// Gets the pump name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the pump efficiency.
    pub fn efficiency(&self) -> f64 {
        self.efficiency
    }

    /// Gets the computed work [W] from the last process call.
    pub fn work(&self) -> f64 {
        self.vars[PumpVars::default().work]
    }

    /// Returns the current value of a named variable.
    ///
    /// Prefer this over indexing `get_variables()` directly to avoid
    /// hard-coded index magic numbers.
    pub fn var(&self, v: PumpVar) -> f64 {
        self.vars[v.index()]
    }

    /// Gets the outlet pressure specification [Pa], if any.
    pub fn outlet_pressure_spec(&self) -> Option<f64> {
        self.outlet_pressure
    }

    /// Gets the pressure rise specification [Pa], if any.
    pub fn pressure_rise_spec(&self) -> Option<f64> {
        self.pressure_rise
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let idx = PumpVars::default();
        let f_in = vars[idx.f_in];
        let t_in = vars[idx.t_in];
        let p_in = vars[idx.p_in];
        let f_out = vars[idx.f_out];
        let t_out = vars[idx.t_out];
        let p_out = vars[idx.p_out];
        let w = vars[idx.work];

        let dp = p_out - p_in;

        let r1 = f_out - f_in;

        let r2 = if let Some(p_spec) = self.outlet_pressure {
            p_out - p_spec
        } else if let Some(dp_spec) = self.pressure_rise {
            dp - dp_spec
        } else {
            S::from(0.0)
        };

        // Use stored f64 values for branch conditions (specified inlets are fixed).
        let f_in_f64 = self.vars[idx.f_in];
        let (r3, r4) = if f_in_f64.abs() > 1e-10 {
            let expected_work = f_in * dp / self.density / self.efficiency;
            let expected_t_out = t_in + w / f_in / self.heat_capacity;
            (w - expected_work, t_out - expected_t_out)
        } else {
            (w, t_out - t_in)
        };

        vec![r1, r2, r3, r4]
    }
}

impl EquationModel for Pump {
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
        let idx = PumpVars::default();
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

impl Process for Pump {
    type Input = Stream<MassFlow>;
    type Output = Stream<MassFlow>;

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let inlet_data = input.to_data();
        let idx = PumpVars::default();

        // Set inlet variables from input stream
        self.vars[idx.f_in] = inlet_data.flow;
        self.vars[idx.t_in] = inlet_data.temperature;
        self.vars[idx.p_in] = inlet_data.pressure;

        // Calculate outlet pressure
        let p_out = if let Some(p) = self.outlet_pressure {
            p
        } else if let Some(dp) = self.pressure_rise {
            inlet_data.pressure + dp
        } else {
            return Err(NomataError::Configuration(
                "Pump requires either outlet_pressure or pressure_rise".to_string(),
            ));
        };

        let dp = p_out - inlet_data.pressure;

        // Update properties from thermodynamics if available
        #[cfg(feature = "thermodynamics")]
        if let Some(ref fluid) = self.fluid {
            if let Ok(props) = fluid.props_pt(inlet_data.pressure, inlet_data.temperature) {
                // Note: We could update density and cp here for more accuracy
                let _ = props.density;
            }
        }

        // Calculate work and outlet temperature via equations
        let specific_work = dp / (self.density * self.efficiency);
        let dt = specific_work / self.heat_capacity;
        let outlet_temperature = inlet_data.temperature + dt;
        let work = inlet_data.flow * specific_work;

        // Set outlet variables
        self.vars[idx.f_out] = inlet_data.flow;
        self.vars[idx.t_out] = outlet_temperature;
        self.vars[idx.p_out] = p_out;
        self.vars[idx.work] = work;

        // Verify equations are satisfied (residuals should be ~0)
        let residuals = self.residuals(&self.vars);
        let max_residual = residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        if max_residual > 1e-6 {
            eprintln!(
                "Warning: Pump '{}' equations not satisfied, max residual = {:.2e}",
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

/// Builder for Pump.
pub struct PumpBuilder {
    name: String,
    efficiency: f64,
    outlet_pressure: Option<f64>,
    pressure_rise: Option<f64>,
    density: f64,
    heat_capacity: f64,
    #[cfg(feature = "thermodynamics")]
    fluid: Option<Fluid>,
}

impl PumpBuilder {
    /// Sets the isentropic efficiency (0 to 1).
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency = efficiency.clamp(0.01, 1.0);
        self
    }

    /// Sets the outlet pressure [Pa].
    pub fn with_outlet_pressure(mut self, pressure: f64) -> Self {
        self.outlet_pressure = Some(pressure);
        self
    }

    /// Sets the pressure rise [Pa].
    pub fn with_pressure_rise(mut self, dp: f64) -> Self {
        self.pressure_rise = Some(dp);
        self
    }

    /// Sets the density [kg/m^3] for manual property specification.
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Sets the heat capacity [J/(kg*K)] for manual property specification.
    pub fn with_heat_capacity(mut self, cp: f64) -> Self {
        self.heat_capacity = cp;
        self
    }

    /// Sets the fluid for thermodynamic calculations.
    #[cfg(feature = "thermodynamics")]
    pub fn with_fluid(mut self, substance: crate::Pure) -> Self {
        self.fluid = Some(Fluid::new(substance));
        self
    }

    /// Builds the pump.
    pub fn build(self) -> NomataResult<Pump> {
        if self.outlet_pressure.is_none() && self.pressure_rise.is_none() {
            return Err(NomataError::Configuration(
                "Pump requires either outlet_pressure or pressure_rise".to_string(),
            ));
        }

        Ok(Pump {
            name: self.name,
            efficiency: self.efficiency,
            outlet_pressure: self.outlet_pressure,
            pressure_rise: self.pressure_rise,
            density: self.density,
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
    fn test_pump_builder() {
        let pump =
            Pump::new("pump-1").with_efficiency(0.8).with_outlet_pressure(5e5).build().unwrap();

        assert_eq!(pump.name(), "pump-1");
        assert_eq!(pump.efficiency(), 0.8);
    }

    #[test]
    fn test_pump_process() {
        let feed = Stream::<MassFlow>::new()
            .with_flow(10.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let mut pump =
            Pump::new("pump-1").with_efficiency(0.75).with_outlet_pressure(5e5).build().unwrap();

        let outlet = pump.process(feed).unwrap();

        // Check pressure increase
        assert!((outlet.pressure() - 5e5).abs() < 1.0);

        // Check flow conservation
        assert!((outlet.flow() - 10.0).abs() < 1e-10);

        // Check temperature increase (small for liquid)
        assert!(outlet.temperature() > 300.0);
        assert!(outlet.temperature() < 310.0);

        // Check work is positive
        assert!(pump.work() > 0.0);
    }

    #[test]
    fn test_pump_equations() {
        let mut pump =
            Pump::new("pump-1").with_efficiency(0.75).with_outlet_pressure(5e5).build().unwrap();

        // Set up a valid solution point
        let f = 10.0; // kg/s
        let t_in = 300.0; // K
        let p_in = 1e5; // Pa
        let p_out = 5e5; // Pa
        let dp = p_out - p_in;
        let rho = 1000.0;
        let eta = 0.75;
        let cp = 4180.0;
        let w = f * dp / (rho * eta);
        let t_out = t_in + w / (f * cp);

        pump.set_variables(&[f, t_in, p_in, f, t_out, p_out, w]);

        let residuals = pump.residuals(&pump.get_variables());

        // All residuals should be zero at solution
        for (i, r) in residuals.iter().enumerate() {
            assert!(r.abs() < 1e-10, "Residual {} = {} (should be 0)", i, r);
        }
    }

    #[test]
    fn test_pump_variable_counts() {
        let pump = Pump::new("pump-1").with_outlet_pressure(5e5).build().unwrap();

        assert_eq!(pump.n_variables(), 7);
        assert_eq!(pump.n_equations(), 4);
        assert_eq!(pump.variable_names().len(), 7);
        assert_eq!(pump.specified_indices().len(), 3);
        assert_eq!(pump.free_indices().len(), 4);
    }
}
