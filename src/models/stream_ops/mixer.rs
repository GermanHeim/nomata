//! Mixer model for combining multiple streams into a single outlet.
//!
//! A mixer combines N inlet streams into one outlet stream. The outlet
//! has the mass/molar-weighted average properties of the inlets.
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let stream1 = Stream::<MassFlow>::new()
//!     .with_flow(5.0)
//!     .with_temperature(300.0)
//!     .with_pressure(1e5)
//!     .pure("Water")
//!     .build()?;
//!
//! let stream2 = Stream::<MassFlow>::new()
//!     .with_flow(3.0)
//!     .with_temperature(350.0)
//!     .with_pressure(1e5)
//!     .pure("Water")
//!     .build()?;
//!
//! // Const generic N specifies number of inlets at compile time
//! let mut mixer = Mixer::<2>::new("mix-1").build()?;
//!
//! let output: Stream<MassFlow> = mixer.process([stream1, stream2])?;
//! ```
//!
//! # Physics
//!
//! The mixer is governed by the following algebraic equations:
//!
//! 1. **Mass balance**: F_out = sum(F_in_i)
//! 2. **Energy balance**: F_out * T_out = sum(F_in_i * T_in_i)
//! 3. **Pressure**: P_out = min(P_in_i) (isobaric mixing at lowest pressure)

use crate::{EquationModel, FlowBasis, MassFlow, NomataResult, Process, Stream};

/// Stream mixer with const generic N inlets.
///
/// The number of inlets is specified at compile time via the const generic parameter N.
///
/// # Variable Layout
///
/// For Mixer<N>:
/// - Indices 0..3N: Inlet streams (N ports x 3 variables each: F, T, P)
/// - Indices 3N..3N+3: Outlet stream (F, T, P)
///
/// # Example
///
/// ```ignore
/// // 2-way mix
/// let mut mix2 = Mixer::<2>::new("m1").build()?;
/// let output = mix2.process([stream1, stream2])?;
///
/// // 3-way mix
/// let mut mix3 = Mixer::<3>::new("m2").build()?;
/// let output = mix3.process([s1, s2, s3])?;
/// ```
#[derive(Debug)]
pub struct Mixer<const N: usize, F: FlowBasis = MassFlow> {
    name: String,
    /// Variables: [F_in1, T_in1, P_in1, F_in2, T_in2, P_in2, ..., F_out, T_out, P_out]
    vars: Vec<f64>,
    _marker: std::marker::PhantomData<F>,
}

impl<const N: usize, F: FlowBasis> Mixer<N, F> {
    /// Creates a new mixer builder with N inlets.
    pub fn new(name: &str) -> MixerBuilder<N, F> {
        MixerBuilder {
            name: name.to_string(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Gets the mixer name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Total number of variables: N inlets * 3 + 1 outlet * 3
    const fn n_vars() -> usize {
        N * 3 + 3
    }
}

impl<const N: usize, F: FlowBasis> EquationModel for Mixer<N, F> {
    fn n_variables(&self) -> usize {
        Self::n_vars()
    }

    fn n_equations(&self) -> usize {
        // Mass balance + Energy balance + Pressure equation
        3
    }

    fn variable_names(&self) -> Vec<&str> {
        let mut names = Vec::with_capacity(Self::n_vars());
        for i in 0..N {
            names.push(if i == 0 { "F_in1" } else if i == 1 { "F_in2" } else { "F_inN" });
            names.push(if i == 0 { "T_in1" } else if i == 1 { "T_in2" } else { "T_inN" });
            names.push(if i == 0 { "P_in1" } else if i == 1 { "P_in2" } else { "P_inN" });
        }
        names.push("F_out");
        names.push("T_out");
        names.push("P_out");
        names
    }

    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        // Extract inlet flows, temps, pressures
        let mut total_flow = 0.0;
        let mut flow_temp_sum = 0.0;
        let mut min_pressure = f64::INFINITY;

        for i in 0..N {
            let f = vars[i * 3];
            let t = vars[i * 3 + 1];
            let p = vars[i * 3 + 2];
            total_flow += f;
            flow_temp_sum += f * t;
            if p < min_pressure {
                min_pressure = p;
            }
        }

        let out_base = N * 3;
        let f_out = vars[out_base];
        let t_out = vars[out_base + 1];
        let p_out = vars[out_base + 2];

        // Residuals
        let r1 = f_out - total_flow; // Mass balance
        let r2 = if total_flow.abs() > 1e-10 {
            f_out * t_out - flow_temp_sum // Energy balance (simplified)
        } else {
            t_out - 298.15 // Default temperature if no flow
        };
        let r3 = p_out - min_pressure; // Pressure = minimum inlet pressure

        vec![r1, r2, r3]
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
        // Inlet variables are specified (from upstream)
        (0..N * 3).collect()
    }

    fn n_inlet_ports(&self) -> usize {
        N
    }

    fn n_outlet_ports(&self) -> usize {
        1
    }

    fn inlet_port_indices(&self, port: usize) -> Vec<usize> {
        let base = port * 3;
        vec![base, base + 1, base + 2]
    }

    fn outlet_port_indices(&self, _port: usize) -> Vec<usize> {
        let base = N * 3;
        vec![base, base + 1, base + 2]
    }
}

impl<const N: usize, F: FlowBasis> Process for Mixer<N, F> {
    type Input = [Stream<F>; N];
    type Output = Stream<F>;

    fn process(&mut self, inputs: Self::Input) -> NomataResult<Self::Output> {
        // Calculate total flow
        let total_flow: f64 = inputs.iter().map(|s| s.flow()).sum();

        // Temperature: flow-weighted average
        let total_temp: f64 = inputs
            .iter()
            .map(|s| s.flow() * s.temperature())
            .sum::<f64>()
            / total_flow;

        // Pressure: minimum of all inlets (isobaric mixing at lowest pressure)
        let min_pressure: f64 = inputs
            .iter()
            .map(|s| s.pressure())
            .fold(f64::INFINITY, f64::min);

        // For composition, we need to collect all unique components
        // and compute flow-weighted mole fractions
        // This is a simplified implementation assuming same components in all streams

        // Get reference composition from first stream
        let first_data = inputs[0].to_data();

        Stream::<F>::new()
            .with_flow(total_flow)
            .with_temperature(total_temp)
            .with_pressure(min_pressure)
            .with_composition(
                &first_data
                    .components
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &first_data.composition,
            )
            .build()
    }
}

/// Builder for Mixer.
#[derive(Debug)]
pub struct MixerBuilder<const N: usize, F: FlowBasis = MassFlow> {
    name: String,
    _marker: std::marker::PhantomData<F>,
}

impl<const N: usize, F: FlowBasis> MixerBuilder<N, F> {
    /// Builds the mixer.
    pub fn build(self) -> NomataResult<Mixer<N, F>> {
        Ok(Mixer {
            name: self.name,
            vars: vec![0.0; Mixer::<N, F>::n_vars()],
            _marker: std::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MassFlow;

    #[test]
    fn test_mixer_2() {
        let mut mixer = Mixer::<2, MassFlow>::new("test").build().unwrap();

        let s1 = Stream::<MassFlow>::new()
            .with_flow(5.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let s2 = Stream::<MassFlow>::new()
            .with_flow(3.0)
            .with_temperature(350.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let output = mixer.process([s1, s2]).unwrap();

        // Total flow = 5 + 3 = 8
        assert!((output.flow() - 8.0).abs() < 1e-10);
        // Weighted avg temp = (5*300 + 3*350) / 8 = 2550/8 = 318.75
        assert!((output.temperature() - 318.75).abs() < 1e-10);
    }

    #[test]
    fn test_mixer_3() {
        let mut mixer = Mixer::<3, MassFlow>::new("test").build().unwrap();

        let s1 = Stream::<MassFlow>::new()
            .with_flow(10.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let s2 = Stream::<MassFlow>::new()
            .with_flow(5.0)
            .with_temperature(350.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let s3 = Stream::<MassFlow>::new()
            .with_flow(5.0)
            .with_temperature(400.0)
            .with_pressure(1e5)
            .pure("Water")
            .build()
            .unwrap();

        let output = mixer.process([s1, s2, s3]).unwrap();

        // Total flow = 10 + 5 + 5 = 20
        assert!((output.flow() - 20.0).abs() < 1e-10);
        // Weighted avg temp = (10*300 + 5*350 + 5*400) / 20 = 6750/20 = 337.5
        assert!((output.temperature() - 337.5).abs() < 1e-10);
    }
}
