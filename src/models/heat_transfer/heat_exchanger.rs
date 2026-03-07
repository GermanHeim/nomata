//! Counter-current shell-and-tube heat exchanger model.
//!
//! A heat exchanger transfers heat between a hot stream and a cold stream
//! without mixing them. This model assumes:
//!
//! - Steady-state, counter-current or co-current operation
//! - Constant heat capacities (no phase change within the unit)
//! - Optional pressure drops on each side
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let hot = Stream::<MassFlow>::new()
//!     .with_flow(2.0).with_temperature(450.0).with_pressure(5e5).build()?;
//! let cold = Stream::<MassFlow>::new()
//!     .with_flow(3.0).with_temperature(300.0).with_pressure(2e5).build()?;
//!
//! let mut hex = HeatExchanger::new("hex-1")
//!     .with_hot_outlet_temperature(380.0)
//!     .with_hot_cp(2000.0)
//!     .with_cold_cp(4180.0)
//!     .build()?;
//!
//! // SM solving: [hot_outlet, cold_outlet] = hex.process([hot, cold])?
//! let [hot_out, cold_out] = hex.process([hot, cold])?;
//! ```
//!
//! # Physics
//!
//! Seven algebraic equations for 13 stream variables plus shared duty Q:
//!
//! 1. **Hot mass balance**:  F_hot_out = F_hot_in
//! 2. **Cold mass balance**: F_cold_out = F_cold_in
//! 3. **Hot energy**:        Q = F_hot_in * Cp_hot * (T_hot_in - T_hot_out)
//! 4. **Cold energy**:       Q = F_cold_in * Cp_cold * (T_cold_out - T_cold_in)
//! 5. **Hot pressure**:      P_hot_out  = P_hot_in  - dP_hot
//! 6. **Cold pressure**:     P_cold_out = P_cold_in - dP_cold
//! 7. **Specification**:     T_hot_out = T_spec  OR  T_cold_out = T_spec  OR  Q = Q_spec

use crate::{EquationModel, MassFlow, NomataError, NomataResult, Process, Stream};

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;

/// Variable layout for the heat exchanger:
/// [F_hi, T_hi, P_hi, F_ho, T_ho, P_ho, F_ci, T_ci, P_ci, F_co, T_co, P_co, Q]
///  0     1     2     3     4     5     6     7     8     9     10    11    12
#[derive(Clone, Copy)]
struct HexVars {
    f_hi: usize,
    t_hi: usize,
    p_hi: usize,
    f_ho: usize,
    t_ho: usize,
    p_ho: usize,
    f_ci: usize,
    t_ci: usize,
    p_ci: usize,
    f_co: usize,
    t_co: usize,
    p_co: usize,
    q: usize,
}

impl Default for HexVars {
    fn default() -> Self {
        HexVars {
            f_hi: 0,
            t_hi: 1,
            p_hi: 2,
            f_ho: 3,
            t_ho: 4,
            p_ho: 5,
            f_ci: 6,
            t_ci: 7,
            p_ci: 8,
            f_co: 9,
            t_co: 10,
            p_co: 11,
            q: 12,
        }
    }
}

/// Which property is specified to close the system.
#[derive(Debug, Clone, Copy)]
enum HexSpec {
    HotOutletTemperature(f64),
    ColdOutletTemperature(f64),
    Duty(f64),
}

/// Counter-current heat exchanger.
///
/// Two-inlet, two-outlet model.  Inlet 0 = hot side, inlet 1 = cold side.
/// Outlet 0 = hot side outlet, outlet 1 = cold side outlet.
#[derive(Debug)]
pub struct HeatExchanger {
    name: String,
    spec: HexSpec,
    cp_hot: f64,   // J/(kg*K)
    cp_cold: f64,  // J/(kg*K)
    dp_hot: f64,   // Pa  (positive = pressure drop)
    dp_cold: f64,  // Pa
    #[cfg(feature = "thermodynamics")]
    hot_fluid: Option<Fluid>,
    #[cfg(feature = "thermodynamics")]
    cold_fluid: Option<Fluid>,

    // 13 variable values
    vars: [f64; 13],
}

impl HeatExchanger {
    /// Creates a new heat exchanger builder.
    pub fn new(name: &str) -> HeatExchangerBuilder {
        HeatExchangerBuilder {
            name: name.to_string(),
            spec: None,
            cp_hot: 4180.0,
            cp_cold: 4180.0,
            dp_hot: 0.0,
            dp_cold: 0.0,
            #[cfg(feature = "thermodynamics")]
            hot_fluid: None,
            #[cfg(feature = "thermodynamics")]
            cold_fluid: None,
        }
    }

    /// Gets the exchanger name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the computed heat duty [W] from the last solve or process call.
    pub fn duty(&self) -> f64 {
        self.vars[HexVars::default().q]
    }
}

impl EquationModel for HeatExchanger {
    fn n_variables(&self) -> usize {
        13
    }

    fn n_equations(&self) -> usize {
        7
    }

    fn variable_names(&self) -> Vec<&str> {
        vec!["F_hi", "T_hi", "P_hi", "F_ho", "T_ho", "P_ho",
             "F_ci", "T_ci", "P_ci", "F_co", "T_co", "P_co", "Q"]
    }

    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        let i = HexVars::default();

        let f_hi = vars[i.f_hi];
        let t_hi = vars[i.t_hi];
        let p_hi = vars[i.p_hi];
        let f_ho = vars[i.f_ho];
        let t_ho = vars[i.t_ho];
        let p_ho = vars[i.p_ho];
        let f_ci = vars[i.f_ci];
        let t_ci = vars[i.t_ci];
        let p_ci = vars[i.p_ci];
        let f_co = vars[i.f_co];
        let t_co = vars[i.t_co];
        let p_co = vars[i.p_co];
        let q = vars[i.q];

        // 1. Hot mass balance
        let r1 = f_ho - f_hi;
        // 2. Cold mass balance
        let r2 = f_co - f_ci;
        // 3. Hot side energy: Q = F_hi * Cp_hot * (T_hi - T_ho)
        let r3 = q - f_hi * self.cp_hot * (t_hi - t_ho);
        // 4. Cold side energy: Q = F_ci * Cp_cold * (T_co - T_ci)
        let r4 = q - f_ci * self.cp_cold * (t_co - t_ci);
        // 5. Hot side pressure
        let r5 = p_ho - (p_hi - self.dp_hot);
        // 6. Cold side pressure
        let r6 = p_co - (p_ci - self.dp_cold);
        // 7. Closing specification
        let r7 = match self.spec {
            HexSpec::HotOutletTemperature(t)  => t_ho - t,
            HexSpec::ColdOutletTemperature(t) => t_co - t,
            HexSpec::Duty(q_spec)             => q - q_spec,
        };

        vec![r1, r2, r3, r4, r5, r6, r7]
    }

    fn get_variables(&self) -> Vec<f64> {
        self.vars.to_vec()
    }

    fn set_variables(&mut self, vars: &[f64]) {
        for (k, &v) in vars.iter().enumerate().take(13) {
            self.vars[k] = v;
        }
    }

    fn specified_indices(&self) -> Vec<usize> {
        // Both inlet streams are specified by upstream connections
        vec![0, 1, 2, 6, 7, 8]
    }

    fn n_inlet_ports(&self) -> usize {
        2
    }

    fn n_outlet_ports(&self) -> usize {
        2
    }

    fn inlet_port_indices(&self, port: usize) -> Vec<usize> {
        match port {
            0 => vec![0, 1, 2],  // hot inlet: F_hi, T_hi, P_hi
            1 => vec![6, 7, 8],  // cold inlet: F_ci, T_ci, P_ci
            _ => vec![],
        }
    }

    fn outlet_port_indices(&self, port: usize) -> Vec<usize> {
        match port {
            0 => vec![3, 4, 5],   // hot outlet: F_ho, T_ho, P_ho
            1 => vec![9, 10, 11], // cold outlet: F_co, T_co, P_co
            _ => vec![],
        }
    }
}

impl Process for HeatExchanger {
    type Input = [Stream<MassFlow>; 2];
    type Output = [Stream<MassFlow>; 2];

    /// Process the heat exchanger.
    ///
    /// `input[0]` is the hot-side inlet, `input[1]` is the cold-side inlet.
    /// Returns `[hot_outlet, cold_outlet]`.
    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let [hot, cold] = input;
        let hi = hot.to_data();
        let ci = cold.to_data();
        let i = HexVars::default();

        // Set inlet variables
        self.vars[i.f_hi] = hi.flow;
        self.vars[i.t_hi] = hi.temperature;
        self.vars[i.p_hi] = hi.pressure;
        self.vars[i.f_ci] = ci.flow;
        self.vars[i.t_ci] = ci.temperature;
        self.vars[i.p_ci] = ci.pressure;

        // Calculate Q and outlet temperatures from the specification
        let (q, t_ho, t_co) = match self.spec {
            HexSpec::HotOutletTemperature(t_ho_spec) => {
                let q = hi.flow * self.cp_hot * (hi.temperature - t_ho_spec);
                let dt_cold = if ci.flow * self.cp_cold > 1e-15 {
                    q / (ci.flow * self.cp_cold)
                } else {
                    return Err(NomataError::Configuration(
                        "HeatExchanger cold side flow is zero".to_string(),
                    ));
                };
                (q, t_ho_spec, ci.temperature + dt_cold)
            }
            HexSpec::ColdOutletTemperature(t_co_spec) => {
                let q = ci.flow * self.cp_cold * (t_co_spec - ci.temperature);
                let dt_hot = if hi.flow * self.cp_hot > 1e-15 {
                    q / (hi.flow * self.cp_hot)
                } else {
                    return Err(NomataError::Configuration(
                        "HeatExchanger hot side flow is zero".to_string(),
                    ));
                };
                (q, hi.temperature - dt_hot, t_co_spec)
            }
            HexSpec::Duty(q_spec) => {
                let dt_hot = if hi.flow * self.cp_hot > 1e-15 {
                    q_spec / (hi.flow * self.cp_hot)
                } else {
                    return Err(NomataError::Configuration(
                        "HeatExchanger hot side flow is zero".to_string(),
                    ));
                };
                let dt_cold = if ci.flow * self.cp_cold > 1e-15 {
                    q_spec / (ci.flow * self.cp_cold)
                } else {
                    return Err(NomataError::Configuration(
                        "HeatExchanger cold side flow is zero".to_string(),
                    ));
                };
                (q_spec, hi.temperature - dt_hot, ci.temperature + dt_cold)
            }
        };

        let p_ho = hi.pressure - self.dp_hot;
        let p_co = ci.pressure - self.dp_cold;

        if p_ho <= 0.0 {
            return Err(NomataError::Validation(format!(
                "HeatExchanger '{}': hot outlet pressure non-positive: {p_ho} Pa",
                self.name
            )));
        }
        if p_co <= 0.0 {
            return Err(NomataError::Validation(format!(
                "HeatExchanger '{}': cold outlet pressure non-positive: {p_co} Pa",
                self.name
            )));
        }

        // Store outlet variables
        self.vars[i.f_ho] = hi.flow;
        self.vars[i.t_ho] = t_ho;
        self.vars[i.p_ho] = p_ho;
        self.vars[i.f_co] = ci.flow;
        self.vars[i.t_co] = t_co;
        self.vars[i.p_co] = p_co;
        self.vars[i.q] = q;

        let names_hot: Vec<&str> = hi.components.iter().map(|s| s.as_str()).collect();
        let names_cold: Vec<&str> = ci.components.iter().map(|s| s.as_str()).collect();

        let hot_out = Stream::<MassFlow>::new()
            .with_flow(hi.flow)
            .with_temperature(t_ho)
            .with_pressure(p_ho)
            .with_composition(&names_hot, &hi.composition)
            .build()?;

        let cold_out = Stream::<MassFlow>::new()
            .with_flow(ci.flow)
            .with_temperature(t_co)
            .with_pressure(p_co)
            .with_composition(&names_cold, &ci.composition)
            .build()?;

        Ok([hot_out, cold_out])
    }
}

/// Builder for HeatExchanger.
#[derive(Debug)]
pub struct HeatExchangerBuilder {
    name: String,
    spec: Option<HexSpec>,
    cp_hot: f64,
    cp_cold: f64,
    dp_hot: f64,
    dp_cold: f64,
    #[cfg(feature = "thermodynamics")]
    hot_fluid: Option<Fluid>,
    #[cfg(feature = "thermodynamics")]
    cold_fluid: Option<Fluid>,
}

impl HeatExchangerBuilder {
    /// Specifies the hot-side outlet temperature [K].
    pub fn with_hot_outlet_temperature(mut self, t: f64) -> Self {
        self.spec = Some(HexSpec::HotOutletTemperature(t));
        self
    }

    /// Specifies the cold-side outlet temperature [K].
    pub fn with_cold_outlet_temperature(mut self, t: f64) -> Self {
        self.spec = Some(HexSpec::ColdOutletTemperature(t));
        self
    }

    /// Specifies the heat duty [W].  Positive = heat transferred from hot to cold.
    pub fn with_duty(mut self, q: f64) -> Self {
        self.spec = Some(HexSpec::Duty(q));
        self
    }

    /// Sets the hot-side heat capacity [J/(kg*K)].
    pub fn with_hot_cp(mut self, cp: f64) -> Self {
        self.cp_hot = cp;
        self
    }

    /// Sets the cold-side heat capacity [J/(kg*K)].
    pub fn with_cold_cp(mut self, cp: f64) -> Self {
        self.cp_cold = cp;
        self
    }

    /// Sets the hot-side pressure drop [Pa].
    pub fn with_hot_pressure_drop(mut self, dp: f64) -> Self {
        self.dp_hot = dp;
        self
    }

    /// Sets the cold-side pressure drop [Pa].
    pub fn with_cold_pressure_drop(mut self, dp: f64) -> Self {
        self.dp_cold = dp;
        self
    }

    /// Sets the hot-side fluid for thermodynamic calculations.
    #[cfg(feature = "thermodynamics")]
    pub fn with_hot_fluid(mut self, fluid: impl Into<Fluid>) -> Self {
        self.hot_fluid = Some(fluid.into());
        self
    }

    /// Sets the cold-side fluid for thermodynamic calculations.
    #[cfg(feature = "thermodynamics")]
    pub fn with_cold_fluid(mut self, fluid: impl Into<Fluid>) -> Self {
        self.cold_fluid = Some(fluid.into());
        self
    }

    /// Builds the heat exchanger.
    pub fn build(self) -> NomataResult<HeatExchanger> {
        let spec = self.spec.ok_or_else(|| {
            NomataError::Configuration(
                "HeatExchanger requires a specification: \
                 hot outlet temperature, cold outlet temperature, or duty"
                    .to_string(),
            )
        })?;

        Ok(HeatExchanger {
            name: self.name,
            spec,
            cp_hot: self.cp_hot,
            cp_cold: self.cp_cold,
            dp_hot: self.dp_hot,
            dp_cold: self.dp_cold,
            #[cfg(feature = "thermodynamics")]
            hot_fluid: self.hot_fluid,
            #[cfg(feature = "thermodynamics")]
            cold_fluid: self.cold_fluid,
            vars: [0.0; 13],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_streams() -> (Stream<MassFlow>, Stream<MassFlow>) {
        let hot = Stream::<MassFlow>::new()
            .with_flow(2.0)
            .with_temperature(450.0)
            .with_pressure(5e5)
            .build()
            .unwrap();
        let cold = Stream::<MassFlow>::new()
            .with_flow(3.0)
            .with_temperature(300.0)
            .with_pressure(2e5)
            .build()
            .unwrap();
        (hot, cold)
    }

    #[test]
    fn test_hex_variable_counts() {
        let hex = HeatExchanger::new("hex")
            .with_hot_outlet_temperature(380.0)
            .build()
            .unwrap();

        assert_eq!(hex.n_variables(), 13);
        assert_eq!(hex.n_equations(), 7);
        assert_eq!(hex.specified_indices().len(), 6);
        assert_eq!(hex.free_indices().len(), 7);
        assert_eq!(hex.n_inlet_ports(), 2);
        assert_eq!(hex.n_outlet_ports(), 2);
    }

    #[test]
    fn test_hex_equations_at_solution() {
        let cp_hot = 2000.0;
        let cp_cold = 4180.0;
        let f_hi = 2.0;
        let t_hi = 450.0;
        let p_hi = 5e5;
        let f_ci = 3.0;
        let t_ci = 300.0;
        let p_ci = 2e5;
        let t_ho = 380.0;

        // Compute consistent values
        let q = f_hi * cp_hot * (t_hi - t_ho);
        let t_co = t_ci + q / (f_ci * cp_cold);

        let mut hex = HeatExchanger::new("hex")
            .with_hot_outlet_temperature(t_ho)
            .with_hot_cp(cp_hot)
            .with_cold_cp(cp_cold)
            .build()
            .unwrap();

        hex.set_variables(&[f_hi, t_hi, p_hi, f_hi, t_ho, p_hi,
                             f_ci, t_ci, p_ci, f_ci, t_co, p_ci, q]);

        let res = hex.residuals(&hex.get_variables());
        for (k, r) in res.iter().enumerate() {
            assert!(r.abs() < 1e-6, "Residual {} = {}", k, r);
        }
    }

    #[test]
    fn test_hex_process_hot_outlet_temp() {
        let (hot, cold) = make_streams();

        let mut hex = HeatExchanger::new("hex")
            .with_hot_outlet_temperature(380.0)
            .with_hot_cp(2000.0)
            .with_cold_cp(4180.0)
            .build()
            .unwrap();

        let [hot_out, cold_out] = hex.process([hot, cold]).unwrap();
        let hd = hot_out.to_data();
        let cd = cold_out.to_data();

        assert!((hd.flow - 2.0).abs() < 1e-10);
        assert!((hd.temperature - 380.0).abs() < 1e-6);
        assert!((cd.flow - 3.0).abs() < 1e-10);
        // Cold side gained: Q = 2*2000*(450-380) = 280 000 W => dT = 280000/(3*4180) = 22.3 K
        let expected_cold_out = 300.0 + 2.0 * 2000.0 * (450.0 - 380.0) / (3.0 * 4180.0);
        assert!((cd.temperature - expected_cold_out).abs() < 1e-6);
    }

    #[test]
    fn test_hex_process_duty_spec() {
        let (hot, cold) = make_streams();

        let q_spec = 1e5; // 100 kW

        let mut hex = HeatExchanger::new("hex")
            .with_duty(q_spec)
            .with_hot_cp(2000.0)
            .with_cold_cp(4180.0)
            .build()
            .unwrap();

        let [hot_out, cold_out] = hex.process([hot, cold]).unwrap();
        let hd = hot_out.to_data();
        let cd = cold_out.to_data();

        assert!((hd.temperature - (450.0 - q_spec / (2.0 * 2000.0))).abs() < 1e-6);
        assert!((cd.temperature - (300.0 + q_spec / (3.0 * 4180.0))).abs() < 1e-6);
    }
}
