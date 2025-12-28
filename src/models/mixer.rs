//! Mixer model combining multiple inlet streams.
//!
//! # Example
//!
//! ```
//! use nomata::models::Mixer;
//!
//! // Type-safe initialization with 3 inlets
//! let mixer: Mixer<3> = Mixer::new(2)  // 2 components
//!     .with_inlets_configured();
//!
//! // Only configured mixers can compute outlet conditions
//! // mixer.compute_outlet();  // Compiles
//! ```

use crate::*;
use std::marker::PhantomData;

/// Phantom type marker for uninitialized state.
pub struct Uninitialized;

/// Phantom type marker for initialized state.
pub struct Initialized;

/// Mixer combining multiple inlet streams with perfect mixing.
///
/// This is a multi-port unit with N inputs and 1 output.
/// Type parameter `I` ensures inlet configuration before operation.
pub struct Mixer<const N: usize, I = Initialized> {
    // Inlets (N streams)
    pub inlet_flows: [f64; N],
    pub inlet_temps: [f64; N],
    pub inlet_compositions: Vec<[f64; N]>,

    // Outlet conditions (algebraic)
    pub outlet_flow: Var<Algebraic>,
    pub outlet_temp: Var<Algebraic>,
    pub outlet_composition: Vec<Var<Algebraic>>,

    _state: PhantomData<I>,
}

impl<const N: usize> Mixer<N, Uninitialized> {
    /// Creates a new mixer with N inlets.
    ///
    /// Call `.with_inlets_configured()` to begin using.
    pub fn new(n_components: usize) -> Self {
        Mixer {
            inlet_flows: [0.0; N],
            inlet_temps: [298.15; N],
            inlet_compositions: vec![[0.0; N]; n_components],

            outlet_flow: Var::new(0.0),
            outlet_temp: Var::new(298.15),
            outlet_composition: (0..n_components).map(|_| Var::new(0.0)).collect(),

            _state: PhantomData,
        }
    }

    /// Marks inlets as configured, transitioning to operational state.
    pub fn with_inlets_configured(self) -> Mixer<N, Initialized> {
        Mixer {
            inlet_flows: self.inlet_flows,
            inlet_temps: self.inlet_temps,
            inlet_compositions: self.inlet_compositions,

            outlet_flow: self.outlet_flow,
            outlet_temp: self.outlet_temp,
            outlet_composition: self.outlet_composition,

            _state: PhantomData,
        }
    }
}

// Methods available at any state
impl<const N: usize, I> Mixer<N, I> {
    /// Gets number of inlets (compile-time constant).
    pub const fn n_inlets(&self) -> usize {
        N
    }
}

// Operational methods (only for configured mixers)
impl<const N: usize> Mixer<N, Initialized> {
    /// Sets inlet conditions from Stream objects.
    ///
    /// # Arguments
    ///
    /// * `streams` - Array of N Stream references
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::{Stream, MolarFlow};
    /// use nomata::models::Mixer;
    ///
    /// let stream1 = Stream::<MolarFlow, _>::with_composition(
    ///     100.0,
    ///     vec!["A".to_string(), "B".to_string()],
    ///     vec![1.0, 0.0],
    /// ).unwrap().at_conditions(298.15, 101325.0);
    ///
    /// let stream2 = Stream::<MolarFlow, _>::with_composition(
    ///     100.0,
    ///     vec!["A".to_string(), "B".to_string()],
    ///     vec![0.0, 1.0],
    /// ).unwrap().at_conditions(298.15, 101325.0);
    ///
    /// let mut mixer = Mixer::<2, _>::new(2).with_inlets_configured();
    /// mixer.from_streams(&[&stream1, &stream2]);
    /// mixer.compute_outlet();
    /// ```
    pub fn from_streams<S: crate::StreamType, C>(&mut self, streams: &[&crate::Stream<S, C>]) {
        assert_eq!(streams.len(), N, "Number of streams must match mixer inlet count");

        // Extract flows and temperatures
        for (i, stream) in streams.iter().enumerate() {
            self.inlet_flows[i] = stream.total_flow;
            self.inlet_temps[i] = stream.temperature;
        }

        // Transpose compositions: stream.composition -> inlet_compositions[component][inlet]
        let n_components = self.outlet_composition.len();
        self.inlet_compositions = (0..n_components)
            .map(|comp_idx| {
                let mut comp_in_each_inlet = [0.0; N];
                for (inlet_idx, stream) in streams.iter().enumerate() {
                    comp_in_each_inlet[inlet_idx] = stream.composition[comp_idx];
                }
                comp_in_each_inlet
            })
            .collect();
    }

    /// Sets inlet conditions from arrays of flow, temperature, and compositions.
    ///
    /// # Arguments
    ///
    /// * `flows` - Flow rates for each inlet [N values]
    /// * `temps` - Temperatures for each inlet [N values]  
    /// * `compositions` - Composition of each component in each inlet
    ///   - `compositions[inlet_idx]` = composition vector for that inlet
    ///   - Each composition vector should sum to 1.0
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::models::Mixer;
    ///
    /// let mut mixer = Mixer::<3, _>::new(3).with_inlets_configured();
    /// mixer.set_inlets(
    ///     &[100.0, 100.0, 100.0],           // Flows
    ///     &[298.15, 298.15, 298.15],        // Temperatures
    ///     &[
    ///         vec![1.0, 0.0, 0.0],           // Inlet 0: pure component 0
    ///         vec![0.0, 1.0, 0.0],           // Inlet 1: pure component 1
    ///         vec![0.0, 0.0, 1.0],           // Inlet 2: pure component 2
    ///     ],
    /// );
    /// mixer.compute_outlet();
    /// ```
    pub fn set_inlets(&mut self, flows: &[f64; N], temps: &[f64; N], compositions: &[Vec<f64>]) {
        self.inlet_flows = *flows;
        self.inlet_temps = *temps;

        // Transpose: compositions[inlet][component] -> inlet_compositions[component][inlet]
        let n_components = self.outlet_composition.len();
        self.inlet_compositions = (0..n_components)
            .map(|comp_idx| {
                let mut comp_in_each_inlet = [0.0; N];
                for (inlet_idx, inlet_comp) in compositions.iter().enumerate() {
                    comp_in_each_inlet[inlet_idx] = inlet_comp[comp_idx];
                }
                comp_in_each_inlet
            })
            .collect();
    }

    /// Computes outlet conditions assuming perfect mixing.
    pub fn compute_outlet(&mut self) {
        // Mass balance
        let total_flow: f64 = self.inlet_flows.iter().sum();
        self.outlet_flow = Var::new(total_flow);

        // Energy balance (mass-weighted average)
        if total_flow > 1e-10 {
            let total_enthalpy: f64 =
                self.inlet_flows.iter().zip(&self.inlet_temps).map(|(f, t)| f * t).sum();
            self.outlet_temp = Var::new(total_enthalpy / total_flow);
        }

        // Component balances
        for (comp_idx, comp_flows) in self.inlet_compositions.iter().enumerate() {
            let total_comp: f64 = self.inlet_flows.iter().zip(comp_flows).map(|(f, x)| f * x).sum();
            self.outlet_composition[comp_idx] = Var::new(total_comp / total_flow);
        }
    }
}

/// Port-based interface for Mixer.
///
/// Demonstrates how multi-port units expose their ports dynamically.
impl<const N: usize, I> HasPorts for Mixer<N, I> {
    fn input_ports(&self) -> Vec<NamedPort> {
        (0..N).map(|i| NamedPort::input(&format!("inlet_{}", i), "MolarFlow")).collect()
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MolarFlow")]
    }
}

/// Compile-time port specification for Mixer.
///
/// Enables type-safe connections with const generic port indices.
impl<const N: usize, I> PortSpec for Mixer<N, I> {
    const INPUT_COUNT: usize = N;
    const OUTPUT_COUNT: usize = 1;
    const STREAM_TYPE: &'static str = "MolarFlow";
}

/// UnitOp implementation for Mixer.
///
/// Note: Mixers are typically steady-state devices (no accumulation),
/// so they only contribute algebraic equations.
impl<const N: usize, I> UnitOp for Mixer<N, I> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Mixers can operate in steady-state (typical) or dynamic mode

        if T::IS_STEADY {
            // Steady-state: no accumulation (typical for mixers)
            // Mass balance: 0 = sum(F_in_i) - F_out
            let n = N;
            let mut vars: Vec<String> = (0..n).map(|i| format!("F_in_{}", i)).collect();
            vars.push("F_out".to_string());
            let mass_balance = ResidualFunction::from_dynamic(
                &format!("{}_mass_balance", unit_name),
                vars,
                move |v, names| {
                    let sum_in: f64 =
                        (0..n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                    sum_in - v.get(&names[n]).copied().unwrap_or(0.0)
                },
            );
            system.add_algebraic(mass_balance);

            // Energy balance: 0 = sum(F_in_i*T_in_i) - F_out*T_out
            let mut vars: Vec<String> = (0..n).map(|i| format!("F_in_{}_T_in_{}", i, i)).collect();
            vars.push("F_out_T_out".to_string());
            let energy_balance = ResidualFunction::from_dynamic(
                &format!("{}_energy_balance", unit_name),
                vars,
                move |v, names| {
                    let sum_in: f64 =
                        (0..n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                    sum_in - v.get(&names[n]).copied().unwrap_or(0.0)
                },
            );
            system.add_algebraic(energy_balance);

            // Component balances: 0 = sum(F_in_i*x_in_i_j) - F_out*x_out_j
            for j in 0..self.outlet_composition.len() {
                let jj = j;
                let mut vars: Vec<String> =
                    (0..n).map(|i| format!("F_in_{}_x_in_{}_{}", i, i, jj)).collect();
                vars.push(format!("F_out_x_out_{}", jj));
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, j),
                    vars,
                    move |v, names| {
                        let sum_in: f64 =
                            (0..n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                        sum_in - v.get(&names[n]).copied().unwrap_or(0.0)
                    },
                );
                system.add_algebraic(comp_balance);
            }
        } else {
            // Dynamic: include accumulation terms
            // Mass balance: dM/dt = sum(F_in_i) - F_out
            let n = N;
            let mut vars: Vec<String> = vec!["dM_dt".to_string()];
            vars.extend((0..n).map(|i| format!("F_in_{}", i)));
            vars.push("F_out".to_string());
            let mass_balance = ResidualFunction::from_dynamic(
                &format!("{}_mass_balance", unit_name),
                vars,
                move |v, names| {
                    let dm_dt = v.get(&names[0]).copied().unwrap_or(0.0);
                    let sum_in: f64 =
                        (1..=n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                    let f_out = v.get(&names[n + 1]).copied().unwrap_or(0.0);
                    dm_dt - sum_in + f_out
                },
            );
            system.add_differential(mass_balance);

            // Energy balance: d(M*T)/dt = sum(F_in_i*T_in_i) - F_out*T_out
            let mut vars: Vec<String> = vec!["d_MT_dt".to_string()];
            vars.extend((0..n).map(|i| format!("F_in_{}_T_in_{}", i, i)));
            vars.push("F_out_T_out".to_string());
            let energy_balance = ResidualFunction::from_dynamic(
                &format!("{}_energy_balance", unit_name),
                vars,
                move |v, names| {
                    let d_mt_dt = v.get(&names[0]).copied().unwrap_or(0.0);
                    let sum_in: f64 =
                        (1..=n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                    let f_out_t_out = v.get(&names[n + 1]).copied().unwrap_or(0.0);
                    d_mt_dt - sum_in + f_out_t_out
                },
            );
            system.add_differential(energy_balance);

            // Component balances: d(M*x_j)/dt = sum(F_in_i*x_in_i_j) - F_out*x_out_j
            for j in 0..self.outlet_composition.len() {
                let jj = j;
                let mut vars: Vec<String> = vec![format!("d_Mx_{}_dt", jj)];
                vars.extend((0..n).map(|i| format!("F_in_{}_x_in_{}_{}", i, i, jj)));
                vars.push(format!("F_out_x_out_{}", jj));
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, j),
                    vars,
                    move |v, names| {
                        let d_mx_dt = v.get(&names[0]).copied().unwrap_or(0.0);
                        let sum_in: f64 =
                            (1..=n).map(|i| v.get(&names[i]).copied().unwrap_or(0.0)).sum();
                        let f_out_x_out = v.get(&names[n + 1]).copied().unwrap_or(0.0);
                        d_mx_dt - sum_in + f_out_x_out
                    },
                );
                system.add_differential(comp_balance);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let mixer: Mixer<3, Initialized> = Mixer::new(2).with_inlets_configured();
        assert_eq!(mixer.n_inlets(), 3);
        assert_eq!(mixer.outlet_composition.len(), 2);
    }

    #[test]
    fn test_mixer_computation() {
        let mut mixer: Mixer<2, Initialized> = Mixer::new(1).with_inlets_configured();
        mixer.inlet_flows = [10.0, 20.0];
        mixer.inlet_temps = [300.0, 350.0];
        mixer.compute_outlet();

        assert_eq!(mixer.outlet_flow.get(), 30.0);
        assert!((mixer.outlet_temp.get() - 333.333).abs() < 0.01);
    }
}
