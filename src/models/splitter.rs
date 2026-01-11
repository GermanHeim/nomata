//! Splitter model dividing one stream into multiple outlets.
//!
//! # Example
//!
//! ```
//! use nomata::{Stream, MolarFlow};
//! use nomata::models::Splitter;
//!
//! // Create inlet stream
//! let inlet = Stream::<MolarFlow, _>::with_composition(
//!     100.0,
//!     vec!["A".to_string(), "B".to_string()],
//!     vec![0.6, 0.4],
//! ).unwrap().at_conditions(298.15, 101325.0);
//!
//! // Initialize splitter with split fractions and connect stream
//! let mut splitter = Splitter::<3>::new()
//!     .with_split_fractions([0.5, 0.3, 0.2]);
//! splitter.set_inlet_stream(inlet);
//!
//! // Get outlet references for later use
//! let outlet_0 = splitter.outlet_stream(0);
//! let outlet_1 = splitter.outlet_stream(1);
//! let outlet_2 = splitter.outlet_stream(2);
//!
//! // After solver runs: splitter.populate_outlet(0, &outlet_0).unwrap();
//! // Then access: outlet_0.get().unwrap();
//! ```

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;
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
    pub inlet_stream: Option<Stream<MolarFlow, InitializedConditions>>,
    pub inlet_flow: f64,
    pub inlet_temp: f64,
    pub inlet_composition: Vec<f64>,
    pub component_names: Vec<String>,

    // Split fractions (Some when S = Initialized)
    split_fractions: Option<[Var<Parameter>; N]>,

    // Outlets
    pub outlet_flows: [Var<Algebraic>; N],
    pub outlet_temps: [Var<Algebraic>; N],

    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<Fluid>,

    _state: PhantomData<S>,
}

impl<const N: usize> Splitter<N, Uninitialized> {
    /// Creates a new splitter with N outlets.
    ///
    /// Call `.with_split_fractions()` to configure split ratios.
    pub fn new() -> Self {
        Splitter {
            inlet_stream: None,
            inlet_flow: 0.0,
            inlet_temp: 298.15,
            inlet_composition: Vec::new(),
            component_names: Vec::new(),

            split_fractions: None,

            outlet_flows: std::array::from_fn(|_| Var::new(0.0)),
            outlet_temps: std::array::from_fn(|_| Var::new(298.15)),

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _state: PhantomData,
        }
    }

    /// Sets split fractions (must sum to 1.0), transitioning to configured state.
    pub fn with_split_fractions(self, fractions: [f64; N]) -> Splitter<N, Initialized> {
        // Verify sum is approximately 1.0
        let sum: f64 = fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Split fractions must sum to 1.0, got {}", sum);

        Splitter {
            inlet_stream: self.inlet_stream,
            inlet_flow: self.inlet_flow,
            inlet_temp: self.inlet_temp,
            inlet_composition: self.inlet_composition,
            component_names: self.component_names,

            split_fractions: Some(fractions.map(Var::new)),

            outlet_flows: self.outlet_flows,
            outlet_temps: self.outlet_temps,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

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

    /// Sets inlet stream and extracts properties.
    ///
    /// Stores the inlet stream and extracts its properties.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::{Stream, MolarFlow, InitializedConditions};
    ///
    /// // Multicomponent stream
    /// let stream = Stream::<MolarFlow, _>::with_composition(
    ///     100.0,
    ///     vec!["N2", "O2"],
    ///     vec![0.79, 0.21],
    /// ).unwrap().at_conditions(298.15, 101325.0);
    ///
    /// let mut splitter = Splitter::new().with_split_fractions([0.6, 0.4]);
    /// splitter.set_inlet_stream(stream);
    /// ```
    pub fn set_inlet_stream(&mut self, stream: Stream<MolarFlow, InitializedConditions>) {
        self.inlet_flow = stream.total_flow;
        self.inlet_temp = stream.temperature;
        self.inlet_composition = stream.composition.clone();
        self.component_names = stream.components.clone();
        self.inlet_stream = Some(stream);
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

        // If inlet_stream exists, use it to create outlet with proper composition
        if let Some(ref inlet) = self.inlet_stream {
            let stream =
                Stream::with_composition(flow, inlet.components.clone(), inlet.composition.clone())
                    .map_err(|_| crate::StreamError::MissingComposition {
                        model: "Splitter".to_string(),
                        suggestion: "Valid inlet stream composition".to_string(),
                    })?;
            return Ok(stream.at_conditions(temp, inlet.pressure));
        }

        // Fallback: use component_names and composition if available
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
            suggestion: "Set inlet stream using set_inlet_stream()".to_string(),
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

        // Simulate what the solver would compute
        splitter.outlet_flows[0] = Var::new(30.0);
        splitter.outlet_flows[1] = Var::new(70.0);
        splitter.outlet_temps[0] = Var::new(350.0);
        splitter.outlet_temps[1] = Var::new(350.0);

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

        // Simulate what the solver would compute
        splitter.outlet_flows = [Var::new(50.0), Var::new(30.0), Var::new(20.0)];
        splitter.outlet_temps = [Var::new(310.0), Var::new(310.0), Var::new(310.0)];

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

    #[cfg(feature = "thermodynamics")]
    #[test]
    fn test_splitter_thermodynamics_support() {
        use crate::thermodynamics::{Fluid, fluids::Pure};

        // Create water stream
        let water = Fluid::new(Pure::Water);
        let stream = Stream::<MolarFlow, _>::from_fluid(100.0, &water, 300.0, 101325.0);

        let mut splitter: Splitter<2, Initialized> =
            Splitter::new().with_split_fractions([0.4, 0.6]);
        splitter.set_inlet_stream(stream);

        // Simulate what the solver would compute
        splitter.outlet_flows = [Var::new(40.0), Var::new(60.0)];
        splitter.outlet_temps = [Var::new(300.0), Var::new(300.0)];

        let outlet0_ref = splitter.outlet_stream(0);
        splitter.populate_outlet(0, &outlet0_ref).unwrap();
        let outlet0 = outlet0_ref.get().unwrap();

        let outlet1_ref = splitter.outlet_stream(1);
        splitter.populate_outlet(1, &outlet1_ref).unwrap();
        let outlet1 = outlet1_ref.get().unwrap();

        // Check that pure streams are created with correct flows and temperatures
        assert_eq!(outlet0.total_flow, 40.0);
        assert_eq!(outlet0.temperature, 300.0);
        assert_eq!(outlet0.pressure, 101325.0);
        assert_eq!(outlet0.components[0], "Water");

        assert_eq!(outlet1.total_flow, 60.0);
        assert_eq!(outlet1.temperature, 300.0);
        assert_eq!(outlet1.pressure, 101325.0);
        assert_eq!(outlet1.components[0], "Water");
    }

    #[cfg(feature = "thermodynamics")]
    #[test]
    fn test_splitter_custom_mixture_support() {
        use crate::thermodynamics::{Fluid, fluids::Pure};
        use std::collections::HashMap;

        // Create a custom mole-based mixture (simplified air)
        let mixture = Fluid::new_mole_based(
            "CustomAir",
            HashMap::from([(Pure::Nitrogen, 0.78), (Pure::Oxygen, 0.21), (Pure::Argon, 0.01)]),
        )
        .unwrap();

        let stream = Stream::<MolarFlow, _>::from_fluid(100.0, &mixture, 298.15, 101325.0);

        let mut splitter: Splitter<2, Initialized> =
            Splitter::new().with_split_fractions([0.3, 0.7]);
        splitter.set_inlet_stream(stream);

        // Simulate what the solver would compute
        splitter.outlet_flows = [Var::new(30.0), Var::new(70.0)];
        splitter.outlet_temps = [Var::new(298.15), Var::new(298.15)];

        let outlet0_ref = splitter.outlet_stream(0);
        splitter.populate_outlet(0, &outlet0_ref).unwrap();
        let outlet0 = outlet0_ref.get().unwrap();

        let outlet1_ref = splitter.outlet_stream(1);
        splitter.populate_outlet(1, &outlet1_ref).unwrap();
        let outlet1 = outlet1_ref.get().unwrap();

        // Check that mixture streams are created with correct flows and multicomponent composition
        assert_eq!(outlet0.total_flow, 30.0);
        assert_eq!(outlet0.temperature, 298.15);
        assert_eq!(outlet0.pressure, 101325.0);
        assert_eq!(outlet0.components.len(), 3);
        assert!(outlet0.components.contains(&"Nitrogen".to_string()));
        assert!(outlet0.components.contains(&"Oxygen".to_string()));
        assert!(outlet0.components.contains(&"Argon".to_string()));

        assert_eq!(outlet1.total_flow, 70.0);
        assert_eq!(outlet1.temperature, 298.15);
        assert_eq!(outlet1.pressure, 101325.0);
        assert_eq!(outlet1.components.len(), 3);
    }

    #[cfg(feature = "thermodynamics")]
    #[test]
    fn test_splitter_with_multicomponent_thermodynamics() {
        use crate::thermodynamics::{Fluid, fluids::Pure};
        use std::collections::HashMap;

        // Create a mole-based mixture of ethylene (C2H4) and propylene (C3H6)
        let mixture = Fluid::new_mole_based(
            "C2H4-C3H6",
            HashMap::from([(Pure::Ethylene, 0.6), (Pure::Propylene, 0.4)]),
        )
        .unwrap();

        // Create a stream directly from the mixture
        let stream = Stream::<MolarFlow, _>::from_fluid(200.0, &mixture, 350.0, 200000.0);

        // Create and configure splitter
        let mut splitter: Splitter<2, Initialized> =
            Splitter::new().with_split_fractions([0.35, 0.65]);

        // Set inlet stream (contains all component information)
        splitter.set_inlet_stream(stream);

        // Simulate what the solver would compute for now
        splitter.outlet_flows = [Var::new(70.0), Var::new(130.0)];
        splitter.outlet_temps = [Var::new(350.0), Var::new(350.0)];

        // Verify inlet properties were extracted correctly
        assert_eq!(splitter.inlet_flow, 200.0);
        assert_eq!(splitter.inlet_temp, 350.0);

        // Verify components are present (order may vary due to HashMap)
        assert_eq!(splitter.component_names.len(), 2);
        assert!(splitter.component_names.contains(&"Ethylene".to_string()));
        assert!(splitter.component_names.contains(&"Propylene".to_string()));

        // Composition should sum to 1.0
        let comp_sum: f64 = splitter.inlet_composition.iter().sum();
        assert!((comp_sum - 1.0).abs() < 1e-6);
        assert_eq!(splitter.inlet_composition.len(), 2);

        // Get outlet streams
        let outlet0_ref = splitter.outlet_stream(0);
        splitter.populate_outlet(0, &outlet0_ref).unwrap();
        let outlet0 = outlet0_ref.get().unwrap();

        let outlet1_ref = splitter.outlet_stream(1);
        splitter.populate_outlet(1, &outlet1_ref).unwrap();
        let outlet1 = outlet1_ref.get().unwrap();

        // Verify split flows
        assert_eq!(outlet0.total_flow, 70.0); // 200 * 0.35
        assert_eq!(outlet1.total_flow, 130.0); // 200 * 0.65

        // Verify temperatures are preserved
        assert_eq!(outlet0.temperature, 350.0);
        assert_eq!(outlet1.temperature, 350.0);

        // Verify pressure is preserved
        assert_eq!(outlet0.pressure, 200000.0);
        assert_eq!(outlet1.pressure, 200000.0);

        // Verify multicomponent composition is preserved (not collapsed to single component)
        assert_eq!(outlet0.components.len(), 2);
        assert_eq!(outlet1.components.len(), 2);
        assert!(outlet0.components.contains(&"Ethylene".to_string()));
        assert!(outlet0.components.contains(&"Propylene".to_string()));
    }
}
