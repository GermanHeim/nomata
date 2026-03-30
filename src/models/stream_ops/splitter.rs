//! Splitter model for dividing a single stream into multiple outlets.
//!
//! A splitter divides an inlet stream into multiple outlet streams based on
//! specified split fractions. All outlets have the same composition, temperature,
//! and pressure as the inlet.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MassFlow>::new()
//!     .with_flow(10.0)
//!     .with_temperature(300.0)
//!     .with_pressure(1e5)
//!     .pure("Water")
//!     .build()?;
//!
//! // Const generic N specifies number of outlets at compile time
//! let mut splitter = Splitter::<3>::new("split-1")
//!     .with_fractions([0.5, 0.3, 0.2])
//!     .build()?;
//!
//! let outputs: [Stream<MassFlow>; 3] = splitter.process(feed)?;
//! ```
//!
//! # Physics
//!
//! The splitter is governed by the following algebraic equations:
//!
//! 1. **Split equations**: F_out_i = fraction_i * F_in (for each outlet i)
//! 2. **Temperature passthrough**: T_out_i = T_in (for each outlet i)
//! 3. **Pressure passthrough**: P_out_i = P_in (for each outlet i)
//!
//! Note: Mass balance F_in = sum(F_out_i) is automatically satisfied
//! when fractions sum to 1.0.

use crate::{EquationModel, FlowBasis, MassFlow, NomataResult, Process, Stream};
use super::PortVar;

// Helper to normalize fractions and issue warning
fn normalize_and_warn<const N: usize>(fractions: &mut [f64; N], name: &str) {
    let sum: f64 = fractions.iter().sum();
    if (sum - 1.0).abs() > 1e-10 {
        eprintln!(
            "Warning: Splitter '{}' split fractions sum to {} (expected 1.0). Normalizing.",
            name, sum
        );
        for f in fractions.iter_mut() {
            *f /= sum;
        }
    }
}

/// Stream splitter with const generic N outlets.
///
/// The number of outlets is specified at compile time via the const generic parameter N.
///
/// # Variable Layout
///
/// For Splitter<N>:
/// - Indices 0..3: Inlet stream (F, T, P)
/// - Indices 3..3+3N: Outlet streams (N ports x 3 variables each: F, T, P)
///
/// # Example
///
/// ```ignore
/// // 2-way split
/// let mut split2 = Splitter::<2>::new("s1")
///     .with_fractions([0.6, 0.4])
///     .build()?;
///
/// // 3-way split
/// let mut split3 = Splitter::<3>::new("s2")
///     .with_fractions([0.5, 0.3, 0.2])
///     .build()?;
/// ```
#[derive(Debug)]
pub struct Splitter<const N: usize, F: FlowBasis = MassFlow> {
    name: String,
    fractions: [f64; N],
    /// Variables: [F_in, T_in, P_in, F_out1, T_out1, P_out1, F_out2, T_out2, P_out2, ...]
    vars: Vec<f64>,
    _marker: std::marker::PhantomData<F>,
}

impl<const N: usize, F: FlowBasis> Splitter<N, F> {
    /// Creates a new splitter builder with N outlets.
    pub fn new(name: &str) -> SplitterBuilder<N, F> {
        // Default: equal fractions
        let default_fraction = 1.0 / N as f64;
        SplitterBuilder {
            name: name.to_string(),
            fractions: [default_fraction; N],
            _marker: std::marker::PhantomData,
        }
    }

    /// Gets the splitter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a variable from the inlet stream.
    ///
    /// `var` selects `Flow`, `Temperature`, or `Pressure`.
    pub fn inlet_var(&self, var: PortVar) -> f64 {
        self.vars[var as usize]
    }

    /// Returns a variable from the given outlet port.
    ///
    /// `port` is 0-indexed; `var` selects `Flow`, `Temperature`, or `Pressure`.
    pub fn outlet_var(&self, port: usize, var: PortVar) -> f64 {
        let offset = 3 + port * 3 + var as usize;
        self.vars[offset]
    }

    /// Gets the split fractions.
    pub fn fractions(&self) -> &[f64; N] {
        &self.fractions
    }

    /// Total number of variables: 1 inlet * 3 + N outlets * 3
    const fn n_vars() -> usize {
        3 + N * 3
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let f_in = vars[0];
        let t_in = vars[1];
        let p_in = vars[2];

        let mut residuals = Vec::with_capacity(N * 3);
        for i in 0..N {
            let out_base = 3 + i * 3;
            let f_out = vars[out_base];
            let t_out = vars[out_base + 1];
            let p_out = vars[out_base + 2];
            residuals.push(f_out - f_in * self.fractions[i]);
            residuals.push(t_out - t_in);
            residuals.push(p_out - p_in);
        }
        residuals
    }
}

impl<const N: usize, F: FlowBasis> EquationModel for Splitter<N, F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_variables(&self) -> usize {
        Self::n_vars()
    }

    fn n_equations(&self) -> usize {
        // For each outlet: F_out_i = fraction_i * F_in, T_out_i = T_in, P_out_i = P_in
        N * 3
    }

    fn variable_names(&self) -> Vec<&str> {
        let mut names = vec!["F_in", "T_in", "P_in"];
        for i in 0..N {
            names.push(if i == 0 {
                "F_out1"
            } else if i == 1 {
                "F_out2"
            } else {
                "F_outN"
            });
            names.push(if i == 0 {
                "T_out1"
            } else if i == 1 {
                "T_out2"
            } else {
                "T_outN"
            });
            names.push(if i == 0 {
                "P_out1"
            } else if i == 1 {
                "P_out2"
            } else {
                "P_outN"
            });
        }
        names
    }

    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        self.residuals_generic(vars)
    }

    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64> {
        self.residuals_generic(vars)
    }

    fn get_variables(&self) -> Vec<f64> {
        self.vars.clone()
    }

    fn set_variables(&mut self, vars: &[f64]) {
        for (i, &v) in vars.iter().enumerate().take(Self::n_vars()) {
            self.vars[i] = v;
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
        N
    }

    fn inlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![0, 1, 2]
    }

    fn outlet_port_indices(&self, port: usize) -> Vec<usize> {
        let base = 3 + port * 3;
        vec![base, base + 1, base + 2]
    }
}

impl<const N: usize, F: FlowBasis> Process for Splitter<N, F> {
    type Input = Stream<F>;
    type Output = [Stream<F>; N];

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let data = input.to_data();

        // Build each outlet stream
        let mut outlets: [Option<Stream<F>>; N] = std::array::from_fn(|_| None);

        for (i, fraction) in self.fractions.iter().enumerate() {
            outlets[i] = Some(
                Stream::<F>::new()
                    .with_flow(data.flow * fraction)
                    .with_temperature(data.temperature)
                    .with_pressure(data.pressure)
                    .with_composition(
                        &data.components.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                        &data.composition,
                    )
                    .build()?,
            );
        }

        // Convert Option array to array (all are Some at this point)
        Ok(outlets.map(|opt| opt.expect("all outlets should be built")))
    }
}

/// Builder for Splitter.
#[derive(Debug)]
pub struct SplitterBuilder<const N: usize, F: FlowBasis = MassFlow> {
    name: String,
    fractions: [f64; N],
    _marker: std::marker::PhantomData<F>,
}

impl<const N: usize, F: FlowBasis> SplitterBuilder<N, F> {
    /// Sets the split fractions. They will be normalized if they don't sum to 1.
    pub fn with_fractions(mut self, fractions: [f64; N]) -> Self {
        self.fractions = fractions;
        self
    }

    /// Builds the splitter.
    pub fn build(mut self) -> NomataResult<Splitter<N, F>> {
        normalize_and_warn(&mut self.fractions, &self.name);

        Ok(Splitter {
            name: self.name,
            fractions: self.fractions,
            vars: vec![0.0; Splitter::<N, F>::n_vars()],
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MassFlow;

    #[test]
    fn test_splitter_2() {
        let mut splitter =
            Splitter::<2, MassFlow>::new("test").with_fractions([0.6, 0.4]).build().unwrap();

        let feed = Stream::<MassFlow>::new()
            .with_flow(10.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let outputs = splitter.process(feed).unwrap();

        assert!((outputs[0].flow() - 6.0).abs() < 1e-10);
        assert!((outputs[1].flow() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_splitter_3() {
        let mut splitter =
            Splitter::<3, MassFlow>::new("test").with_fractions([0.5, 0.3, 0.2]).build().unwrap();

        let feed = Stream::<MassFlow>::new()
            .with_flow(100.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let outputs = splitter.process(feed).unwrap();

        assert!((outputs[0].flow() - 50.0).abs() < 1e-10);
        assert!((outputs[1].flow() - 30.0).abs() < 1e-10);
        assert!((outputs[2].flow() - 20.0).abs() < 1e-10);
    }
}
