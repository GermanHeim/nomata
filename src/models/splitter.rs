//! Splitter model dividing one stream into multiple outlets.
//!
//! # Example
//!
//! ```
//! use nomata::models::Splitter;
//!
//! // Type-safe initialization with 3 outlets
//! let splitter = Splitter::new()
//!     .with_split_fractions([0.5, 0.3, 0.2]);
//!
//! // Only configured splitters can compute outlets
//! // splitter.compute_outlets();  // Compiles
//! ```

use crate::*;
use std::collections::HashMap;
use std::marker::PhantomData;

// Typed Equation Variable Structs (Generic for autodiff support)

/// Variables for split fraction equation: F_out_i - frac_i * F_in = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct SplitFractionVars<S: Scalar> {
    /// Outlet flow for this split
    pub f_out: S,
    /// frac_i * F_in computed externally
    pub frac_f_in: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for SplitFractionVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["f_out", "frac_f_in"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_out: *vars.get(&format!("{}_f_out", prefix))?,
            frac_f_in: *vars.get(&format!("{}_frac_f_in", prefix))?,
        })
    }
}

impl EquationVars for SplitFractionVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["f_out", "frac_f_in"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for temperature continuity: T_out_i - T_in = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct TempContinuityVars<S: Scalar> {
    /// Outlet temperature
    pub t_out: S,
    /// Inlet temperature
    pub t_in: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for TempContinuityVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["t_out", "t_in"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            t_out: *vars.get(&format!("{}_t_out", prefix))?,
            t_in: *vars.get(&format!("{}_t_in", prefix))?,
        })
    }
}

impl EquationVars for TempContinuityVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["t_out", "t_in"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Phantom type marker for uninitialized state.
pub struct Uninitialized;

/// Phantom type marker for initialized state.
pub struct Initialized;

/// Splitter dividing one stream into multiple outlets.
///
/// This is a multi-port unit with 1 input and N outputs.
/// Type parameter `S` ensures split fractions are set before operation.
pub struct Splitter<const N: usize, S = Uninitialized> {
    // Inlet
    pub inlet_flow: f64,
    pub inlet_temp: f64,
    pub inlet_composition: Vec<f64>,
    pub component_names: Vec<String>,

    // Split fractions (Some when S = Initialized)
    split_fractions: Option<[Var<Parameter>; N]>,

    // Outlets
    pub outlet_flows: [Var<Algebraic>; N],
    pub outlet_temps: [Var<Algebraic>; N],

    _state: PhantomData<S>,
}

impl<const N: usize> Splitter<N, Uninitialized> {
    /// Creates a new splitter with N outlets.
    ///
    /// Call `.with_split_fractions()` to configure split ratios.
    pub fn new() -> Self {
        Splitter {
            inlet_flow: 0.0,
            inlet_temp: 298.15,
            inlet_composition: Vec::new(),
            component_names: Vec::new(),

            split_fractions: None,

            outlet_flows: std::array::from_fn(|_| Var::new(0.0)),
            outlet_temps: std::array::from_fn(|_| Var::new(298.15)),

            _state: PhantomData,
        }
    }

    /// Sets split fractions (must sum to 1.0), transitioning to configured state.
    pub fn with_split_fractions(self, fractions: [f64; N]) -> Splitter<N, Initialized> {
        // Verify sum is approximately 1.0
        let sum: f64 = fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Split fractions must sum to 1.0, got {}", sum);

        Splitter {
            inlet_flow: self.inlet_flow,
            inlet_temp: self.inlet_temp,
            inlet_composition: self.inlet_composition,
            component_names: self.component_names,

            split_fractions: Some(fractions.map(Var::new)),

            outlet_flows: self.outlet_flows,
            outlet_temps: self.outlet_temps,

            _state: PhantomData,
        }
    }
}

impl<const N: usize> Default for Splitter<N, Uninitialized> {
    fn default() -> Self {
        Self::new()
    }
}

// Methods available at any state
impl<const N: usize, S> Splitter<N, S> {
    /// Gets number of outlets (compile-time constant).
    pub const fn n_outlets(&self) -> usize {
        N
    }
}

// Operational methods (only for configured splitters)
impl<const N: usize> Splitter<N, Initialized> {
    /// Computes outlet conditions.
    ///
    /// Only available for configured splitters.
    pub fn compute_outlets(&mut self) {
        let fractions = self
            .split_fractions
            .as_ref()
            .expect("split_fractions should be set for Initialized splitter");
        for (i, frac_var) in fractions.iter().enumerate().take(N) {
            let frac = frac_var.get();
            self.outlet_flows[i] = Var::new(self.inlet_flow * frac);
            self.outlet_temps[i] = Var::new(self.inlet_temp);
        }
    }

    /// Gets split fraction for outlet i (guaranteed to exist).
    pub fn split_fraction(&self, i: usize) -> f64 {
        self.split_fractions
            .as_ref()
            .expect("split_fractions should be set for Initialized splitter")[i]
            .get()
    }

    /// Gets all split fractions.
    pub fn split_fractions(&self) -> Vec<f64> {
        self.split_fractions
            .as_ref()
            .expect("split_fractions should be set for Initialized splitter")
            .iter()
            .map(|f| f.get())
            .collect()
    }

    // TODO: Let user set a deltaP or deltaT if desired

    /// Creates an outlet stream for the specified index.
    ///
    /// Returns a stream with the split flow and same conditions as inlet.
    /// For multicomponent streams, composition is preserved from inlet.
    /// Note: Assumes constant pressure and temperature.
    fn compute_outlet_stream(&self, idx: usize) -> Result<Stream<MolarFlow>, crate::StreamError> {
        assert!(idx < N, "Outlet index {} out of range (max {})", idx, N - 1);

        let flow = self.outlet_flows[idx].get();
        let temp = self.outlet_temps[idx].get();

        // If we have multicomponent composition info, use it
        if !self.component_names.is_empty()
            && let Ok(stream) = Stream::with_composition(
                flow,
                self.component_names.clone(),
                self.inlet_composition.clone(),
            )
        {
            return Ok(stream.at_conditions(temp, 101325.0));
        }

        Err(crate::StreamError::MissingComposition {
            model: "Splitter".to_string(),
            suggestion: "inlet_composition and component_names fields".to_string(),
        })
    }

    /// Returns a reference to the outlet stream at the specified index.
    pub fn outlet_stream(&self, idx: usize) -> crate::OutletRef {
        crate::OutletRef::new("Splitter", &format!("outlet_{}", idx))
    }

    /// Populates the outlet reference at the specified index with the current computed stream.
    pub fn populate_outlet(
        &self,
        idx: usize,
        outlet_ref: &crate::OutletRef,
    ) -> Result<(), crate::StreamError> {
        let stream = self.compute_outlet_stream(idx)?;
        outlet_ref.set(stream);
        Ok(())
    }
}

/// Port-based interface for Splitter.
///
/// Demonstrates how multi-port units expose their ports dynamically.
impl<const N: usize, S> HasPorts for Splitter<N, S> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        (0..N).map(|i| NamedPort::output(&format!("outlet_{}", i), "MolarFlow")).collect()
    }
}

/// Compile-time port specification for Splitter.
///
/// Enables type-safe connections with const generic port indices.
impl<const N: usize, S> PortSpec for Splitter<N, S> {
    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = N;
    const STREAM_TYPE: &'static str = "MolarFlow";
}

/// UnitOp implementation for Splitter.
impl<const N: usize, S> UnitOp for Splitter<N, S> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    #[cfg(not(feature = "autodiff"))]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        let n = N;

        // Mass balance: F_in - sum(F_out_i) = 0
        let mut vars: Vec<String> = vec!["F_in".to_string()];
        vars.extend((0..n).map(|i| format!("F_out_{}", i)));
        let mass_balance = ResidualFunction::from_dynamic(
            &format!("{}_mass_balance", unit_name),
            vars,
            move |v, names| {
                let f_in = v.get(&names[0]).copied().unwrap_or(0.0);
                let sum_out: f64 = (1..=n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                f_in - sum_out
            },
        );
        system.add_algebraic(mass_balance);

        // Split fraction constraints: F_out_i - frac_i * F_in = 0
        for i in 0..N {
            let prefix = format!("{}_split_{}", unit_name, i);
            let split_eq = ResidualFunction::from_typed(
                &format!("{}_split_{}", unit_name, i),
                &prefix,
                |v: SplitFractionVars<f64>| v.f_out - v.frac_f_in,
            );
            system.add_algebraic(split_eq);
        }

        // Temperature continuity: T_out_i - T_in = 0 for all outlets
        for i in 0..N {
            let prefix = format!("{}_temp_{}", unit_name, i);
            let temp_eq = ResidualFunction::from_typed(
                &format!("{}_temp_{}", unit_name, i),
                &prefix,
                |v: TempContinuityVars<f64>| v.t_out - v.t_in,
            );
            system.add_algebraic(temp_eq);
        }
    }

    #[cfg(feature = "autodiff")]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;
        let n = N;

        // Mass balance: F_in - sum(F_out_i) = 0
        // Note: from_dynamic doesn't have autodiff support, kept as-is
        let mut vars: Vec<String> = vec!["F_in".to_string()];
        vars.extend((0..n).map(|i| format!("F_out_{}", i)));
        let mass_balance = ResidualFunction::from_dynamic(
            &format!("{}_mass_balance", unit_name),
            vars,
            move |v, names| {
                let f_in = v.get(&names[0]).copied().unwrap_or(0.0);
                let sum_out: f64 = (1..=n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                f_in - sum_out
            },
        );
        system.add_algebraic(mass_balance);

        // Split fraction constraints: F_out_i - frac_i * F_in = 0
        for i in 0..N {
            let prefix = format!("{}_split_{}", unit_name, i);
            let split_eq = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_split_{}", unit_name, i),
                &prefix,
                |v: SplitFractionVars<f64>| v.f_out - v.frac_f_in,
                |v: SplitFractionVars<Dual64>| v.f_out - v.frac_f_in,
            );
            system.add_algebraic(split_eq);
        }

        // Temperature continuity: T_out_i - T_in = 0 for all outlets
        for i in 0..N {
            let prefix = format!("{}_temp_{}", unit_name, i);
            let temp_eq = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_temp_{}", unit_name, i),
                &prefix,
                |v: TempContinuityVars<f64>| v.t_out - v.t_in,
                |v: TempContinuityVars<Dual64>| v.t_out - v.t_in,
            );
            system.add_algebraic(temp_eq);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splitter_creation() {
        let splitter: Splitter<3, Initialized> =
            Splitter::new().with_split_fractions([0.5, 0.3, 0.2]);
        assert_eq!(splitter.n_outlets(), 3);
        assert_eq!(splitter.split_fraction(0), 0.5);
    }

    #[test]
    fn test_splitter_computation() {
        let mut splitter: Splitter<2, Initialized> =
            Splitter::new().with_split_fractions([0.3, 0.7]);
        splitter.inlet_flow = 100.0;
        splitter.inlet_temp = 350.0;
        splitter.compute_outlets();

        assert_eq!(splitter.outlet_flows[0].get(), 30.0);
        assert_eq!(splitter.outlet_flows[1].get(), 70.0);
    }

    #[test]
    #[should_panic(expected = "Split fractions must sum to 1.0")]
    fn test_splitter_invalid_fractions() {
        let _: Splitter<2, Initialized> = Splitter::new().with_split_fractions([0.6, 0.5]); // Sum > 1.0
    }

    #[test]
    fn test_splitter_outlet_streams() {
        let mut splitter: Splitter<3, Initialized> =
            Splitter::new().with_split_fractions([0.5, 0.3, 0.2]);
        splitter.inlet_flow = 100.0;
        splitter.inlet_temp = 310.0;
        splitter.component_names = vec!["Mixture".to_string()];
        splitter.inlet_composition = vec![1.0];
        splitter.compute_outlets();

        let outlet0_ref = splitter.outlet_stream(0);
        splitter.populate_outlet(0, &outlet0_ref).unwrap();
        let outlet0 = outlet0_ref.get().unwrap();

        let outlet1_ref = splitter.outlet_stream(1);
        splitter.populate_outlet(1, &outlet1_ref).unwrap();
        let outlet1 = outlet1_ref.get().unwrap();

        let outlet2_ref = splitter.outlet_stream(2);
        splitter.populate_outlet(2, &outlet2_ref).unwrap();
        let outlet2 = outlet2_ref.get().unwrap();

        assert_eq!(outlet0.total_flow, 50.0);
        assert_eq!(outlet0.temperature, 310.0);
        assert_eq!(outlet1.total_flow, 30.0);
        assert_eq!(outlet2.total_flow, 20.0);
    }
}
