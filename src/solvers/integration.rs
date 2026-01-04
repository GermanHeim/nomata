//! Numerical integration for dynamic flowsheets.
//!
//! This module provides tools for integrating numercally
//! dynamic process models over time using the `[differential-equations](https://docs.rs/differential-equations/)`
//! crate.

use crate::solvers::{SolverError, SolverResult};
use crate::{Dynamic, Flowsheet, VariableRegistry};
use differential_equations::methods::{ExplicitRungeKutta, ImplicitRungeKutta};
use differential_equations::ode::{ODE, ODEProblem};
use nalgebra::SVector;

/// A stateful integrator for dynamic flowsheets.
///
/// Combines a flowsheet, variable registry, and current time to provide
/// convenient time-stepping capabilities for dynamic process models.
///
/// # Type Parameters
///
/// * `N` - Maximum number of differential variables the integrator can handle
///
/// # Examples
///
/// ```ignore
/// use nomata::solvers::integration::FlowsheetIntegrator;
/// use nomata::{Flowsheet, VariableRegistry, Dynamic};
///
/// let flowsheet = Flowsheet::<Dynamic>::new();
/// let registry = VariableRegistry::new();
///
/// // Create integrator for up to 32 differential variables
/// let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);
/// integrator.step(0.1)?; // Step forward 0.1 time units
/// ```
pub struct FlowsheetIntegrator<const N: usize> {
    /// The dynamic flowsheet containing equations
    pub flowsheet: Flowsheet<Dynamic>,
    /// Variable registry holding the current state
    pub registry: VariableRegistry,
    /// Current simulation time
    pub time: f64,
    /// Integration method to use
    pub method: IntegrationMethod,
}

/// Integration methods available for time-stepping.
///
/// Each method has different stability and accuracy characteristics.
///
/// For details on each method, refer to the [`differential-equations`](https://docs.rs/differential-equations/) crate documentation.
///
/// # Variants
///
/// - `Dopri5`: Adaptive explicit Dormand-Prince 5(4) method
/// - `Radau5`: Adaptive implicit Radau5 method
/// - `GaussLegendre4`: Adaptive implicit Gauss-Legendre 4th order
/// - `GaussLegendre6`: Adaptive implicit Gauss-Legendre 6th order
/// - `LobattoIIIC2`: Adaptive implicit Lobatto IIIC 2nd order
/// - `LobattoIIIC4`: Adaptive implicit Lobatto IIIC 4th order
/// - `RK4`: Fixed-step explicit 4th-order Runge-Kutta
/// - `Euler`: Fixed-step explicit Forward Euler
/// - `Heun`: Fixed-step explicit Heun method
/// - `Midpoint`: Fixed-step explicit Midpoint method
/// - `Ralston`: Fixed-step explicit Ralston method
///
/// For stiff systems, implicit methods are recommended.
/// For non-stiff systems, explicit methods may be more efficient.
#[derive(Debug, Clone, Copy)]
pub enum IntegrationMethod {
    // Adaptive explicit methods
    /// Dormand-Prince 5(4) method
    Dopri5,
    // Adaptive implicit methods
    /// Radau5 method
    Radau5,
    /// Gauss-Legendre 4th order
    GaussLegendre4,
    /// Gauss-Legendre 6th order
    GaussLegendre6,
    /// Lobatto IIIC 2nd order
    LobattoIIIC2,
    /// Lobatto IIIC 4th order
    LobattoIIIC4,
    // Fixed-step explicit methods
    /// 4th-order Runge-Kutta
    RK4,
    /// Forward Euler
    Euler,
    /// Heun method
    Heun,
    /// Midpoint method
    Midpoint,
    /// Ralston method
    Ralston,
}

/// ODE wrapper for flowsheet integration using differential-equations crate.
pub struct FlowsheetODE<'a, const N: usize> {
    flowsheet: &'a Flowsheet<Dynamic>,
    registry: &'a VariableRegistry,
}

impl<const N: usize> ODE<f64, SVector<f64, N>> for FlowsheetODE<'_, N> {
    fn diff(&self, t: f64, y: &SVector<f64, N>, dydt: &mut SVector<f64, N>) {
        let n_diff = self.flowsheet.equation_system.differential_count();

        // These conditions should have been checked by step() before creating the ODE problem.
        // If they occur here, it indicates an internal bug or misuse.
        if n_diff == 0 {
            panic!("Internal error: ODE called with no differential equations");
        }
        if n_diff > N {
            panic!(
                "Internal error: ODE called with {} differential variables but capacity is {}",
                n_diff, N
            );
        }

        // Get current full state
        let mut state = self.registry.get_all_values();

        // Set differential variables to y
        for i in 0..n_diff {
            state[i] = y[i];
        }

        // Derivatives vector (zero for differential vars)
        let derivatives = vec![0.0; n_diff];

        // Evaluate residuals
        let residuals = self.flowsheet.equation_system.evaluate_residuals(&state, &derivatives, t);

        // dydt is the differential residuals
        for i in 0..n_diff {
            dydt[i] = residuals[i];
        }
    }
}

impl<const N: usize> FlowsheetIntegrator<N> {
    /// Creates a new flowsheet integrator.
    ///
    /// # Arguments
    ///
    /// * `flowsheet` - Dynamic flowsheet with harvested equations
    /// * `registry` - Variable registry with initialized state
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::solvers::integration::FlowsheetIntegrator;
    /// use nomata::{Flowsheet, VariableRegistry, Dynamic};
    ///
    /// let flowsheet = Flowsheet::<Dynamic>::new();
    /// let registry = VariableRegistry::new();
    /// let integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);
    ///
    /// assert_eq!(integrator.time(), 0.0);
    /// ```
    pub fn new(flowsheet: Flowsheet<Dynamic>, registry: VariableRegistry) -> Self {
        FlowsheetIntegrator { flowsheet, registry, time: 0.0, method: IntegrationMethod::Dopri5 }
    }

    /// Creates a new integrator with specified initial time.
    pub fn with_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Sets the integration method.
    pub fn with_method(mut self, method: IntegrationMethod) -> Self {
        self.method = method;
        self
    }

    /// Gets the current simulation time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Gets a reference to the flowsheet.
    pub fn flowsheet(&self) -> &Flowsheet<Dynamic> {
        &self.flowsheet
    }

    /// Gets a mutable reference to the flowsheet.
    pub fn flowsheet_mut(&mut self) -> &mut Flowsheet<Dynamic> {
        &mut self.flowsheet
    }

    /// Gets a reference to the variable registry.
    pub fn registry(&self) -> &VariableRegistry {
        &self.registry
    }

    /// Gets a mutable reference to the variable registry.
    pub fn registry_mut(&mut self) -> &mut VariableRegistry {
        &mut self.registry
    }

    /// Takes a single integration step forward in time.
    ///
    /// Updates the state variables using the specified integration method
    /// and advances the simulation time.
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step size
    ///
    /// # Returns
    ///
    /// Ok(()) if step succeeded, Err otherwise
    ///
    /// # Examples
    ///
    /// ```ignore
    /// integrator.step(0.1)?;  // Step forward 0.1 time units
    /// println!("Time: {}", integrator.time());  // Should be 0.1
    /// ```
    pub fn step(&mut self, dt: f64) -> SolverResult<()> {
        // Get number of differential variables
        let n_diff = self.flowsheet.equation_system.differential_count();

        if n_diff == 0 {
            return Err(SolverError::NoEquations);
        }

        if n_diff > N {
            return Err(SolverError::TooManyDifferentialVariables(n_diff, N));
        }

        // Create ODE wrapper
        let ode = FlowsheetODE::<N> { flowsheet: &self.flowsheet, registry: &self.registry };

        // Get initial differential state
        let state = self.registry.get_all_values();
        let mut y0 = SVector::<f64, N>::zeros();
        for i in 0..n_diff {
            y0[i] = state[i];
        }

        // Set up ODE problem
        let t0 = self.time;
        let tf = self.time + dt;
        let problem = ODEProblem::new(&ode, t0, tf, y0);

        // Choose solver based on method
        let solution = match self.method {
            // Adaptive explicit
            IntegrationMethod::Dopri5 => {
                let mut solver = ExplicitRungeKutta::dopri5().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            // Adaptive implicit
            IntegrationMethod::Radau5 => {
                let mut solver = ImplicitRungeKutta::radau5().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            IntegrationMethod::GaussLegendre4 => {
                let mut solver = ImplicitRungeKutta::gauss_legendre_4().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            IntegrationMethod::GaussLegendre6 => {
                let mut solver = ImplicitRungeKutta::gauss_legendre_6().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            IntegrationMethod::LobattoIIIC2 => {
                let mut solver = ImplicitRungeKutta::lobatto_iiic_2().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            IntegrationMethod::LobattoIIIC4 => {
                let mut solver = ImplicitRungeKutta::lobatto_iiic_4().rtol(1e-6).atol(1e-8);
                problem.solve(&mut solver)
            }
            // Fixed-step explicit
            IntegrationMethod::RK4 => {
                let mut solver = ExplicitRungeKutta::rk4(dt);
                problem.solve(&mut solver)
            }
            IntegrationMethod::Euler => {
                let mut solver = ExplicitRungeKutta::euler(dt);
                problem.solve(&mut solver)
            }
            IntegrationMethod::Heun => {
                let mut solver = ExplicitRungeKutta::heun(dt);
                problem.solve(&mut solver)
            }
            IntegrationMethod::Midpoint => {
                let mut solver = ExplicitRungeKutta::midpoint(dt);
                problem.solve(&mut solver)
            }
            IntegrationMethod::Ralston => {
                let mut solver = ExplicitRungeKutta::ralston(dt);
                problem.solve(&mut solver)
            }
        };

        match solution {
            Ok(sol) => {
                // Update registry with final state
                let mut new_state = self.registry.get_all_values();
                let final_y = &sol.y[sol.y.len() - 1];
                for i in 0..n_diff {
                    new_state[i] = final_y[i];
                }
                self.registry.set_all_values(&new_state);

                // Update time
                self.time = tf;
                Ok(())
            }
            Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
        }
    }

    #[cfg(not(feature = "solvers"))]
    pub fn step(&mut self, _dt: f64) -> SolverResult<()> {
        Err(SolverError::FeatureNotEnabled)
    }

    /// Integrates from current time to specified final time.
    ///
    /// Takes multiple steps as needed to reach t_final.
    ///
    /// # Arguments
    ///
    /// * `t_final` - Target final time
    /// * `dt` - Fixed time step size
    ///
    /// # Returns
    ///
    /// History of (time, state) pairs
    pub fn integrate_to(&mut self, t_final: f64, dt: f64) -> SolverResult<Vec<(f64, Vec<f64>)>> {
        let mut history = Vec::new();

        // Save initial state
        history.push((self.time, self.registry.get_all_values()));

        while self.time < t_final {
            let step_size = dt.min(t_final - self.time);
            self.step(step_size)?;

            // Save state after each step
            history.push((self.time, self.registry.get_all_values()));
        }

        Ok(history)
    }

    #[cfg(not(feature = "solvers"))]
    pub fn integrate_to(&mut self, _t_final: f64, _dt: f64) -> SolverResult<Vec<(f64, Vec<f64>)>> {
        Err(SolverError::FeatureNotEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Flowsheet, VariableRegistry};

    #[test]
    fn test_integrator_creation() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);
        assert_eq!(integrator.time(), 0.0);
    }

    #[test]
    fn test_integrator_with_time() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry).with_time(5.0);
        assert_eq!(integrator.time(), 5.0);
    }

    #[test]
    fn test_integrator_with_method() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry)
            .with_method(IntegrationMethod::Euler);

        // Just check it compiles and runs
        assert_eq!(integrator.time(), 0.0);
    }

    #[test]
    fn test_integrator_step() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);
        let result = integrator.step(0.1);

        // Should fail with no equations
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SolverError::NoEquations));
    }

    #[test]
    fn test_integrator_multiple_methods() {
        let _flowsheet = Flowsheet::<Dynamic>::new();
        let _registry = VariableRegistry::new();

        let methods = vec![
            IntegrationMethod::Dopri5,
            IntegrationMethod::RK4,
            IntegrationMethod::Euler,
            IntegrationMethod::Heun,
            IntegrationMethod::Midpoint,
            IntegrationMethod::Ralston,
        ];

        for method in methods {
            let flowsheet = Flowsheet::<Dynamic>::new();
            let registry = VariableRegistry::new();
            let mut integrator =
                FlowsheetIntegrator::<32>::new(flowsheet, registry).with_method(method);
            let result = integrator.step(0.01);
            assert!(result.is_err(), "Method {:?} should fail with no equations", method);
            assert!(matches!(result.unwrap_err(), SolverError::NoEquations));
        }
    }

    #[test]
    fn test_integrator_method_switching() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);

        // Start with Euler
        integrator = integrator.with_method(IntegrationMethod::Euler);
        let result1 = integrator.step(0.01);
        assert!(result1.is_err());

        // Switch to RK4
        integrator = integrator.with_method(IntegrationMethod::RK4);
        let result2 = integrator.step(0.01);
        assert!(result2.is_err());
    }

    #[test]
    fn test_integrator_integrate_to() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);

        let result = integrator.integrate_to(0.1, 0.02);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SolverError::NoEquations));
    }

    #[test]
    fn test_integrator_time_advancement() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry).with_time(5.0);

        let result1 = integrator.step(0.5);
        assert!(result1.is_err());

        let result2 = integrator.step(1.0);
        assert!(result2.is_err());
    }

    #[test]
    #[cfg(not(feature = "solvers"))]
    fn test_integrator_without_solvers() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);

        let result = integrator.step(0.1);
        assert!(result.is_err());

        let history_result = integrator.integrate_to(1.0, 0.1);
        assert!(history_result.is_err());
    }

    #[test]
    fn test_integrator_accessors() {
        let flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        let mut integrator = FlowsheetIntegrator::<32>::new(flowsheet, registry);

        // Test flowsheet access
        assert!(integrator.flowsheet().equations().total_equations() == 0);

        // Test registry access
        assert_eq!(integrator.registry().get_all_values().len(), 0);

        // Test mutable access
        integrator.flowsheet_mut();
        integrator.registry_mut();
    }

    #[test]
    fn test_integrator_capacity_exceeded() {
        use crate::ResidualFunction;

        // Create a flowsheet with 3 differential equations
        let mut flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        // Add 3 differential equations (simple dummy equations)
        let eq1 = ResidualFunction::from_dynamic("diff1", vec!["x1".to_string()], |vars, names| {
            *vars.get(&names[0]).unwrap_or(&0.0) - 1.0
        });
        let eq2 = ResidualFunction::from_dynamic("diff2", vec!["x2".to_string()], |vars, names| {
            *vars.get(&names[0]).unwrap_or(&0.0) - 2.0
        });
        let eq3 = ResidualFunction::from_dynamic("diff3", vec!["x3".to_string()], |vars, names| {
            *vars.get(&names[0]).unwrap_or(&0.0) - 3.0
        });

        flowsheet.equation_system.add_differential(eq1);
        flowsheet.equation_system.add_differential(eq2);
        flowsheet.equation_system.add_differential(eq3);

        // Verify we have 3 differential equations
        assert_eq!(flowsheet.equation_system.differential_count(), 3);

        // Create an integrator with capacity of only 2 variables
        let mut integrator = FlowsheetIntegrator::<2>::new(flowsheet, registry);

        // Attempt to step should fail with TooManyDifferentialVariables
        let result = integrator.step(0.1);
        assert!(result.is_err());

        match result.unwrap_err() {
            SolverError::TooManyDifferentialVariables(actual, capacity) => {
                assert_eq!(actual, 3);
                assert_eq!(capacity, 2);
            }
            other => panic!("Expected TooManyDifferentialVariables, got {:?}", other),
        }
    }

    #[test]
    fn test_integrator_actually_solves_equations() {
        use crate::ResidualFunction;

        // Create a simple ODE: dx/dt = -x, with x(0) = 1.0
        // Analytical solution: x(t) = e^(-t)
        // At t=1.0: x(1) aprox 0.368

        let mut flowsheet = Flowsheet::<Dynamic>::new();
        let registry = VariableRegistry::new();

        // Register variable with initial value
        let _x_id = registry.register(1.0, "x");

        // Create differential equation: dx/dt = -x
        // Residual form: dx/dt = -x
        let eq =
            ResidualFunction::from_dynamic("decay", vec!["x".to_string()], move |vars, names| {
                let x = *vars.get(&names[0]).unwrap_or(&0.0);
                -x // Returns -x, so dx/dt = -x (exponential decay)
            });

        flowsheet.equation_system.add_differential(eq);

        // Create integrator
        let mut integrator = FlowsheetIntegrator::<4>::new(flowsheet, registry);

        // Get initial value
        let initial_value = integrator.registry().get_all_values()[0];
        assert!((initial_value - 1.0).abs() < 1e-10, "Initial value should be 1.0");

        // Integrate from t=0 to t=1.0
        let result = integrator.integrate_to(1.0, 0.1);
        assert!(result.is_ok(), "Integration should succeed");

        // Get final value
        let final_value = integrator.registry().get_all_values()[0];

        // The value should have changed significantly
        assert!(
            (final_value - 1.0).abs() > 0.1,
            "Value should have changed from initial value, got {}",
            final_value
        );

        // For exponential decay dx/dt = -x, x(1) = e^(-1) ≈ 0.368
        println!("Initial: {}, Final: {}, Time: {}", initial_value, final_value, integrator.time());
        assert_eq!(integrator.time(), 1.0, "Time should be 1.0");

        // Verify we're solving the correct equation (exponential decay)
        let expected = 1.0_f64.exp().recip(); // e^(-1) aprox 0.368
        assert!(
            (final_value - expected).abs() < 0.01,
            "Expected x(1) aprox {}, got {}",
            expected,
            final_value
        );
    }
}
