//! Flash Separator model.
//!
//! A vapor-liquid equilibrium separator using either K-values or thermodynamic calculations.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MolarFlow>::new()
//!     .with_flow(100.0)
//!     .with_temperature(350.0)
//!     .with_pressure(1e5)
//!     .with_composition(&["Light", "Heavy"], &[0.6, 0.4])
//!     .build()?;
//!
//! let mut flash = FlashSeparator::new("flash-1")
//!     .with_k_values(&[3.0, 0.5])
//!     .build()?;
//!
//! let output = flash.process(feed)?;
//! // Access typed outputs: output.vapor, output.liquid
//! ```
//!
//! # Physics
//!
//! The flash separator is governed by the following algebraic equations:
//!
//! 1. **Rachford-Rice equation** (solved for vapor fraction psi):
//!    sum_i [ z_i * (K_i - 1) / (1 + psi * (K_i - 1)) ] = 0
//!
//! 2. **Equilibrium relations**: y_i = K_i * x_i
//!
//! 3. **Material balances**:
//!    - F_vapor = psi * F_feed
//!    - F_liquid = (1 - psi) * F_feed
//!    - x_i = z_i / (1 + psi * (K_i - 1))
//!    - y_i = K_i * x_i
//!
//! where z_i is feed composition, K_i is equilibrium ratio, psi is vapor fraction.

use crate::{EquationModel, FlowBasis, MolarFlow, NomataError, NomataResult, Process, Stream};

/// Output from a flash separator: vapor and liquid streams.
#[derive(Debug)]
pub struct FlashOutput<F: FlowBasis = MolarFlow> {
    /// Vapor outlet stream.
    pub vapor: Stream<F>,
    /// Liquid outlet stream.
    pub liquid: Stream<F>,
}

/// Flash separator for vapor-liquid equilibrium.
///
/// The flash separator model defines algebraic equations that can be solved using
/// either Sequential-Modular (SM) or Equation-Oriented (EO) approaches.
#[derive(Debug)]
pub struct FlashSeparator {
    name: String,
    k_values: Vec<f64>,
    feed_composition: Vec<f64>,

    // EO variables: [F_in, T_in, P_in, F_vap, T_vap, P_vap, F_liq, T_liq, P_liq, psi]
    vars: [f64; 10],

    // Computed compositions (updated when psi changes)
    liquid_composition: Vec<f64>,
    vapor_composition: Vec<f64>,
}

/// Variable indices for FlashSeparator
struct FlashVars {
    f_in: usize,
    t_in: usize,
    p_in: usize,
    f_vap: usize,
    t_vap: usize,
    p_vap: usize,
    f_liq: usize,
    t_liq: usize,
    p_liq: usize,
    psi: usize,
}

impl Default for FlashVars {
    fn default() -> Self {
        Self {
            f_in: 0,
            t_in: 1,
            p_in: 2,
            f_vap: 3,
            t_vap: 4,
            p_vap: 5,
            f_liq: 6,
            t_liq: 7,
            p_liq: 8,
            psi: 9,
        }
    }
}

impl FlashSeparator {
    /// Creates a new flash separator builder.
    pub fn new(name: &str) -> FlashSeparatorBuilder {
        FlashSeparatorBuilder {
            name: name.to_string(),
            k_values: Vec::new(),
            feed_composition: Vec::new(),
        }
    }

    /// Gets the flash separator name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the vapor fraction after calculation.
    pub fn vapor_fraction(&self) -> f64 {
        let idx = FlashVars::default();
        self.vars[idx.psi]
    }

    /// Gets the liquid composition after calculation.
    pub fn liquid_composition(&self) -> &[f64] {
        &self.liquid_composition
    }

    /// Gets the vapor composition after calculation.
    pub fn vapor_composition(&self) -> &[f64] {
        &self.vapor_composition
    }

    /// Gets the K-values.
    pub fn k_values(&self) -> &[f64] {
        &self.k_values
    }

    /// Gets feed flow.
    pub fn feed_flow(&self) -> f64 {
        self.vars[FlashVars::default().f_in]
    }

    /// Gets feed temperature.
    pub fn feed_temperature(&self) -> f64 {
        self.vars[FlashVars::default().t_in]
    }

    /// Gets feed pressure.
    pub fn feed_pressure(&self) -> f64 {
        self.vars[FlashVars::default().p_in]
    }

    /// Performs Rachford-Rice flash calculation.
    fn flash_calculation(&mut self, feed_composition: &[f64]) {
        let n = feed_composition.len();
        if n == 0 || self.k_values.len() != n {
            return;
        }

        // Initial guess for vapor fraction
        let mut psi = 0.5;
        let idx = FlashVars::default();

        // Rachford-Rice iteration
        for _ in 0..50 {
            let mut f = 0.0;
            let mut df = 0.0;

            for i in 0..n {
                let k = self.k_values[i];
                let z = feed_composition[i];
                let km1 = k - 1.0;
                let denom = 1.0 + psi * km1;

                f += z * km1 / denom;
                df -= z * km1 * km1 / (denom * denom);
            }

            if f.abs() < 1e-10 {
                break;
            }

            // Newton step
            let dpsi = -f / df;
            psi += dpsi;
            psi = psi.clamp(0.0, 1.0);

            if dpsi.abs() < 1e-12 {
                break;
            }
        }

        self.vars[idx.psi] = psi;

        // Update outlet flows based on vapor fraction
        let f_in = self.vars[idx.f_in];
        self.vars[idx.f_vap] = f_in * psi;
        self.vars[idx.f_liq] = f_in * (1.0 - psi);

        // Compute compositions
        self.liquid_composition = vec![0.0; n];
        self.vapor_composition = vec![0.0; n];

        for i in 0..n {
            let k = self.k_values[i];
            let z = feed_composition[i];

            let x = z / (1.0 + psi * (k - 1.0));
            let y = k * x;

            self.liquid_composition[i] = x;
            self.vapor_composition[i] = y;
        }
    }

    /// Computes the Rachford-Rice residual for a given vapor fraction.
    ///
    /// This is the core algebraic equation: sum_i [ z_i * (K_i - 1) / (1 + psi * (K_i - 1)) ] = 0
    pub fn rachford_rice_residual(&self, psi: f64) -> f64 {
        self.rachford_rice_residual_generic(psi)
    }

    fn rachford_rice_residual_generic<S: crate::Scalar>(&self, psi: S) -> S {
        if self.feed_composition.is_empty() || self.k_values.len() != self.feed_composition.len() {
            return S::from(0.0);
        }
        let mut f = S::from(0.0);
        for i in 0..self.feed_composition.len() {
            let k = self.k_values[i];
            let z = self.feed_composition[i];
            let km1 = k - 1.0;
            let denom = psi * km1 + 1.0_f64;
            f = f + S::from(z * km1) / denom;
        }
        f
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let f_in = vars[0];
        let t_in = vars[1];
        let p_in = vars[2];
        let f_vap = vars[3];
        let t_vap = vars[4];
        let p_vap = vars[5];
        let f_liq = vars[6];
        let t_liq = vars[7];
        let p_liq = vars[8];
        let psi = vars[9];

        vec![
            self.rachford_rice_residual_generic(psi),
            f_vap - psi * f_in,
            f_liq - (S::from(1.0) - psi) * f_in,
            t_vap - t_in,
            t_liq - t_in,
            p_vap - p_in,
            p_liq - p_in,
        ]
    }
}

impl EquationModel for FlashSeparator {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_variables(&self) -> usize {
        // Variables: F_in, T_in, P_in, F_vap, T_vap, P_vap, F_liq, T_liq, P_liq, psi
        10
    }

    fn n_equations(&self) -> usize {
        // 1. Rachford-Rice equation
        // 2. Vapor flow: F_vap = psi * F_in
        // 3. Liquid flow: F_liq = (1-psi) * F_in
        // 4. Vapor temp: T_vap = T_in
        // 5. Liquid temp: T_liq = T_in
        // 6. Vapor pressure: P_vap = P_in
        // 7. Liquid pressure: P_liq = P_in
        7
    }

    fn variable_names(&self) -> Vec<&str> {
        vec!["F_in", "T_in", "P_in", "F_vap", "T_vap", "P_vap", "F_liq", "T_liq", "P_liq", "psi"]
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
        for (i, &v) in vars.iter().enumerate().take(10) {
            self.vars[i] = v;
        }

        // Clamp psi to [0, 1]
        let idx = FlashVars::default();
        self.vars[idx.psi] = self.vars[idx.psi].clamp(0.0, 1.0);

        // Update compositions based on new vapor fraction
        let psi = self.vars[idx.psi];
        let n = self.feed_composition.len();
        if n > 0 && self.k_values.len() == n {
            self.liquid_composition = vec![0.0; n];
            self.vapor_composition = vec![0.0; n];

            for i in 0..n {
                let k = self.k_values[i];
                let z = self.feed_composition[i];
                let x = z / (1.0 + psi * (k - 1.0));
                let y = k * x;
                self.liquid_composition[i] = x;
                self.vapor_composition[i] = y;
            }
        }
    }

    fn specified_indices(&self) -> Vec<usize> {
        // Inlet variables are specified
        vec![0, 1, 2]
    }

    fn n_inlet_ports(&self) -> usize {
        1
    }

    fn n_outlet_ports(&self) -> usize {
        2 // Vapor (port 0) and Liquid (port 1)
    }

    fn inlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![0, 1, 2]
    }

    fn outlet_port_indices(&self, port: usize) -> Vec<usize> {
        match port {
            0 => vec![3, 4, 5], // Vapor outlet
            1 => vec![6, 7, 8], // Liquid outlet
            _ => vec![3, 4, 5], // Default to vapor
        }
    }
}

impl Process for FlashSeparator {
    type Input = Stream<MolarFlow>;
    type Output = FlashOutput<MolarFlow>;

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let data = input.to_data();

        // Store feed data for equation model
        let idx = FlashVars::default();
        self.vars[idx.f_in] = data.flow;
        self.vars[idx.t_in] = data.temperature;
        self.vars[idx.p_in] = data.pressure;
        self.feed_composition = data.composition.clone();

        // Validate or set default k_values
        if self.k_values.len() != data.composition.len() {
            if self.k_values.is_empty() {
                // Default K-values (all 1.0 means no separation)
                self.k_values = vec![1.0; data.composition.len()];
            } else {
                return Err(NomataError::Validation(format!(
                    "K-values count ({}) doesn't match component count ({})",
                    self.k_values.len(),
                    data.composition.len()
                )));
            }
        }

        // Perform flash calculation (solves Rachford-Rice equation)
        self.flash_calculation(&data.composition);

        // Verify equation is satisfied
        let psi = self.vars[idx.psi];
        let residual = self.rachford_rice_residual(psi);
        if residual.abs() > 1e-8 {
            eprintln!(
                "Warning: FlashSeparator '{}' Rachford-Rice residual = {:.2e}",
                self.name, residual
            );
        }

        // Update outlet values
        let vapor_flow = self.vars[idx.f_vap];
        let liquid_flow = self.vars[idx.f_liq];
        self.vars[idx.t_vap] = data.temperature;
        self.vars[idx.t_liq] = data.temperature;
        self.vars[idx.p_vap] = data.pressure;
        self.vars[idx.p_liq] = data.pressure;

        // Build vapor stream
        let vapor = Stream::<MolarFlow>::new()
            .with_flow(vapor_flow)
            .with_temperature(data.temperature)
            .with_pressure(data.pressure)
            .with_composition(
                &data.components.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &self.vapor_composition,
            )
            .build()?;

        // Build liquid stream
        let liquid = Stream::<MolarFlow>::new()
            .with_flow(liquid_flow)
            .with_temperature(data.temperature)
            .with_pressure(data.pressure)
            .with_composition(
                &data.components.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                &self.liquid_composition,
            )
            .build()?;

        Ok(FlashOutput { vapor, liquid })
    }
}

/// Builder for FlashSeparator.
#[derive(Debug)]
pub struct FlashSeparatorBuilder {
    name: String,
    k_values: Vec<f64>,
    feed_composition: Vec<f64>,
}

impl FlashSeparatorBuilder {
    /// Sets the equilibrium K-values (one per component).
    pub fn with_k_values(mut self, k_values: &[f64]) -> Self {
        self.k_values = k_values.to_vec();
        self
    }

    /// Sets the feed composition (mole fractions, must sum to 1.0).
    pub fn with_feed_composition(mut self, composition: &[f64]) -> Self {
        self.feed_composition = composition.to_vec();
        self
    }

    /// Builds the flash separator.
    pub fn build(self) -> NomataResult<FlashSeparator> {
        // Initialize vars with psi = 0.5 initial guess
        let mut vars = [0.0; 10];
        vars[FlashVars::default().psi] = 0.5;

        Ok(FlashSeparator {
            name: self.name,
            k_values: self.k_values,
            feed_composition: self.feed_composition,
            vars,
            liquid_composition: Vec::new(),
            vapor_composition: Vec::new(),
        })
    }
}
