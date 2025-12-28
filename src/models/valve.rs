//! Control valve model for flow regulation.

use crate::*;

use std::collections::HashMap;
use std::marker::PhantomData;

// Typed Equation Variable Structs (Generic for autodiff support)

/// Variables for flow calculation: F - Cv * opening * sqrt(dP) = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct ValveFlowVars<S: Scalar> {
    pub flow: S,
    pub cv: S,
    pub opening: S,
    pub sqrt_dp: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for ValveFlowVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["F", "Cv", "opening", "sqrt_dP"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            flow: *vars.get(&format!("{}_F", prefix))?,
            cv: *vars.get(&format!("{}_Cv", prefix))?,
            opening: *vars.get(&format!("{}_opening", prefix))?,
            sqrt_dp: *vars.get(&format!("{}_sqrt_dP", prefix))?,
        })
    }
}

// Backwards compatibility: EquationVars for f64
impl EquationVars for ValveFlowVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["F", "Cv", "opening", "sqrt_dP"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Marker types for Valve initialization states.
pub struct Uninitialized;
pub struct Initialized;

/// Control valve model.
///
/// Models flow through a control valve using the valve equation:
/// F = Cv * opening * sqrt(dP)
///
/// Type parameters enforce compile-time initialization:
/// - `C`: Valve configuration state (Cv and opening)
/// - `P`: Port connection state
///
/// # Example
///
/// ```
/// use nomata::models::Valve;
///
/// // Create and configure a valve
/// let valve = Valve::new()
///     .with_configuration(100.0, 0.5);  // Cv=100, 50% open
///
/// // Access valve properties
/// assert_eq!(valve.cv().unwrap(), 100.0);
/// assert_eq!(valve.opening().unwrap(), 0.5);
/// ```
pub struct Valve<C = Uninitialized, P = Disconnected>
where
    P: PortState,
{
    // State variables
    /// Pressure drop across valve (Pa)
    pub pressure_drop: Var<Algebraic>,
    /// Volumetric flow through valve (m3/s)
    pub flow: Var<Algebraic>,

    // Parameters
    /// Valve coefficient (Cv)
    cv: Option<Var<Parameter>>,
    /// Valve opening fraction (0.0 to 1.0)
    opening: Option<Var<Parameter>>,

    // Ports
    /// Inlet port
    pub inlet: Port<Stream<MassFlow>, Input, P>,
    /// Outlet port
    pub outlet: Port<Stream<MassFlow>, Output, P>,

    _config: PhantomData<C>,
}

impl Valve {
    /// Creates a new valve in uninitialized state.
    pub fn new() -> Self {
        Valve {
            pressure_drop: Var::new(0.0),
            flow: Var::new(0.0),
            cv: None,
            opening: None,
            inlet: Port::new(),
            outlet: Port::new(),
            _config: PhantomData,
        }
    }
}

impl Default for Valve {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: PortState> Valve<Uninitialized, P> {
    /// Sets valve configuration parameters.
    ///
    /// Transitions from Uninitialized to Initialized state.
    ///
    /// # Arguments
    ///
    /// * `cv` - Valve coefficient (flow capacity)
    /// * `opening` - Valve opening fraction (0.0 = closed, 1.0 = fully open)
    ///
    /// # Panics
    ///
    /// Panics if opening is not in range [0.0, 1.0].
    pub fn with_configuration(self, cv: f64, opening: f64) -> Valve<Initialized, P> {
        assert!((0.0..=1.0).contains(&opening), "Valve opening must be between 0.0 and 1.0");

        Valve {
            pressure_drop: self.pressure_drop,
            flow: self.flow,
            cv: Some(Var::new(cv)),
            opening: Some(Var::new(opening)),
            inlet: self.inlet,
            outlet: self.outlet,
            _config: PhantomData,
        }
    }
}

impl<P: PortState> Valve<Initialized, P> {
    /// Gets the valve coefficient.
    pub fn cv(&self) -> Option<f64> {
        self.cv.as_ref().map(|v| v.get())
    }

    /// Gets the valve opening fraction.
    pub fn opening(&self) -> Option<f64> {
        self.opening.as_ref().map(|v| v.get())
    }

    /// Sets a new valve opening.
    ///
    /// # Panics
    ///
    /// Panics if opening is not in range [0.0, 1.0].
    pub fn set_opening(&mut self, opening: f64) {
        assert!((0.0..=1.0).contains(&opening), "Valve opening must be between 0.0 and 1.0");
        if let Some(ref mut v) = self.opening {
            v.set(opening);
        }
    }

    /// Computes flow through the valve given a pressure drop.
    ///
    /// Uses the valve equation: F = Cv * opening * sqrt(dP)
    pub fn compute_flow(&mut self, pressure_drop: f64) {
        self.pressure_drop = Var::new(pressure_drop);

        let cv = self.cv.as_ref().unwrap().get();
        let opening = self.opening.as_ref().unwrap().get();

        // F = Cv * opening * sqrt(dP)
        let flow = cv * opening * pressure_drop.abs().sqrt();
        self.flow = Var::new(flow);
    }
}

impl<C, P: PortState> UnitOp for Valve<C, P> {
    type In = Stream<MassFlow>;
    type Out = Stream<MassFlow>;

    #[cfg(not(feature = "autodiff"))]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Valve flow equation: F - Cv * opening * sqrt(dP) = 0
        let flow_eq = ResidualFunction::from_typed(
            &format!("{}_flow", unit_name),
            unit_name,
            |v: ValveFlowVars<f64>| v.flow - v.cv * v.opening * v.sqrt_dp,
        );
        system.add_algebraic(flow_eq);
    }

    #[cfg(feature = "autodiff")]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        // Valve flow equation: F - Cv * opening * sqrt(dP) = 0
        let flow_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_flow", unit_name),
            unit_name,
            |v: ValveFlowVars<f64>| v.flow - v.cv * v.opening * v.sqrt_dp,
            |v: ValveFlowVars<Dual64>| v.flow - v.cv * v.opening * v.sqrt_dp,
        );
        system.add_algebraic(flow_eq);
    }
}

impl<C, P: PortState> HasPorts for Valve<C, P> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MassFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MassFlow")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valve_creation() {
        let valve: Valve<Initialized> = Valve::new().with_configuration(100.0, 0.5);

        assert_eq!(valve.cv(), Some(100.0));
        assert_eq!(valve.opening(), Some(0.5));
    }

    #[test]
    fn test_valve_flow_computation() {
        let mut valve: Valve<Initialized> = Valve::new().with_configuration(100.0, 1.0);

        // With dP = 100 Pa, Cv = 100, opening = 1.0:
        // F = 100 * 1.0 * sqrt(100) = 100 * 10 = 1000
        valve.compute_flow(100.0);

        assert!((valve.flow.get() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_valve_partial_opening() {
        let mut valve: Valve<Initialized> = Valve::new().with_configuration(100.0, 0.5);

        // With dP = 100 Pa, Cv = 100, opening = 0.5:
        // F = 100 * 0.5 * sqrt(100) = 50 * 10 = 500
        valve.compute_flow(100.0);

        assert!((valve.flow.get() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_valve_set_opening() {
        let mut valve: Valve<Initialized> = Valve::new().with_configuration(100.0, 0.5);

        valve.set_opening(0.75);
        assert_eq!(valve.opening(), Some(0.75));
    }

    #[test]
    #[should_panic(expected = "Valve opening must be between 0.0 and 1.0")]
    fn test_valve_invalid_opening() {
        let _valve: Valve<Initialized> = Valve::new().with_configuration(100.0, 1.5);
    }
}
