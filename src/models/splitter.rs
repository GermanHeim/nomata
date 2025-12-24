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
use std::marker::PhantomData;

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
        let fractions = self.split_fractions.as_ref().unwrap();
        for (i, frac_var) in fractions.iter().enumerate().take(N) {
            let frac = frac_var.get();
            self.outlet_flows[i] = Var::new(self.inlet_flow * frac);
            self.outlet_temps[i] = Var::new(self.inlet_temp);
        }
    }

    /// Gets split fraction for outlet i (guaranteed to exist).
    pub fn split_fraction(&self, i: usize) -> f64 {
        self.split_fractions.as_ref().unwrap()[i].get()
    }

    /// Gets all split fractions.
    pub fn split_fractions(&self) -> Vec<f64> {
        self.split_fractions.as_ref().unwrap().iter().map(|f| f.get()).collect()
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

/// UnitOp implementation for Splitter.
impl<const N: usize, S> UnitOp for Splitter<N, S> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Mass balance: F_in = sum(F_out_i)
        let mut mass_balance = ResidualFunction::new(&format!("{}_mass_balance", unit_name));
        mass_balance.add_term(EquationTerm::new(1.0, "F_in"));
        for i in 0..N {
            mass_balance.add_term(EquationTerm::new(-1.0, &format!("F_out_{}", i)));
        }
        system.add_algebraic(mass_balance);

        // Split fraction constraints: F_out_i = frac_i * F_in
        for i in 0..N {
            let mut split_eq = ResidualFunction::new(&format!("{}_split_{}", unit_name, i));
            split_eq.add_term(EquationTerm::new(1.0, &format!("F_out_{}", i)));
            split_eq.add_term(EquationTerm::new(-1.0, &format!("frac_{}_F_in", i)));
            system.add_algebraic(split_eq);
        }

        // Temperature continuity: T_out_i = T_in for all outlets
        for i in 0..N {
            let mut temp_eq = ResidualFunction::new(&format!("{}_temp_{}", unit_name, i));
            temp_eq.add_term(EquationTerm::new(1.0, &format!("T_out_{}", i)));
            temp_eq.add_term(EquationTerm::new(-1.0, "T_in"));
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
}
