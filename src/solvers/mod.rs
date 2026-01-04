//! Numerical solvers for process simulation systems.
//!
//! This module provides numerical solution methods for steady-state and dynamic
//! process models, with specialized support for flowsheets with recycle streams.
//!
//! # Solution Paradigms
//!
//! Process flowsheets can be solved using two fundamentally different approaches:
//!
//! ## Sequential-Modular (SM)
//!
//! The flowsheet is solved unit-by-unit in a fixed sequence, exactly like how
//! an engineer would calculate it by hand:
//!
//! 1. Each block takes its inlet streams
//! 2. Solves its internal equations (independently)
//! 3. Produces outlet streams
//! 4. Passes results downstream
//!
//! For flowsheets with recycle streams, we iterate on "tear streams" until convergence.
//! Typical convergence acceleration methods include:
//! - **Wegstein**: Fixed-point acceleration (default, recommended)
//! - **Broyden**: Quasi-Newton approximation
//! - **Damping/relaxation**: Simple under-relaxation
//!
//! ## Equation-Oriented (EO)
//!
//! The entire flowsheet is written as one large system of equations and solved
//! simultaneously using Newton-Raphson or similar methods:
//!
//! 1. Collect all equations from all units
//! 2. Form a global residual function F(x) = 0
//! 3. Solve using Newton-Raphson with analytical or numerical Jacobian
//!
//! # Submodules
//!
//! - [`recycle`]: Tear-stream solvers for flowsheets with cyclic topology
//!   (Wegstein, Newton, Broyden). Works with user-provided computation closures.
//!
//! # Solvers Provided
//!
//! - [`RecycleSolver`](recycle::RecycleSolver): Iterative convergence for tear streams
//! - [`SteadyStateSolver`]: Newton-Raphson for algebraic equation systems
//! - [`NewtonRaphson`]: General-purpose Newton solver with user-provided Jacobian
//!
//! # Usage Pattern
//!
//! For Sequential-Modular simulation with recycles, provide a closure that
//! computes the flowsheet given tear stream values:
//!
//! ```ignore
//! use nomata::solvers::recycle::RecycleSolver;
//!
//! // Define sequential calculation: inlet -> mixer -> reactor -> separator -> outlet
//! let compute_flowsheet = |tear: &[f64]| -> Vec<f64> {
//!     let recycle_flow = tear[0];
//!     
//!     // Mixer
//!     let mixed_flow = fresh_feed + recycle_flow;
//!     
//!     // Reactor
//!     let reactor_out = mixed_flow * (1.0 - conversion);
//!     
//!     // Separator
//!     let recycle_out = reactor_out * split_fraction;
//!     
//!     vec![recycle_out]  // Return computed tear stream values
//! };
//!
//! let mut solver = RecycleSolver::new(100, 1e-6);
//! let solution = solver.solve(vec![0.0], compute_flowsheet)?;
//! ```

// Submodules
pub mod integration;
pub mod recycle;

use nalgebra::{DMatrix, DVector};

/// Result type for solver operations.
pub type SolverResult<T> = Result<T, SolverError>;

/// Errors that can occur during solving.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// Maximum iterations exceeded without convergence
    #[error("Maximum iterations exceeded")]
    MaxIterationsExceeded,
    /// Singular Jacobian matrix encountered
    #[error("Singular Jacobian matrix")]
    SingularJacobian,
    /// Step size too small
    #[error("Step size too small")]
    StepSizeTooSmall,
    /// Solution diverged
    #[error("Solution diverged")]
    Diverged,
    /// Invalid initial conditions
    #[error("Invalid initial conditions")]
    InvalidInitialConditions,
    /// ODE solver failed
    #[error("ODE solver failed: {0}")]
    ODESolverFailed(String),
    /// ODE/DAE solver feature not enabled
    #[error("ODE/DAE solver requires the 'solvers' feature to be enabled")]
    FeatureNotEnabled,
    /// No variables to solve in the equation system
    #[error("No variables to solve. Did you forget to call harvest_equations()?")]
    NoVariablesToSolve,
    /// No differential equations to integrate
    #[error("No differential equations to integrate")]
    NoEquations,
    /// Too many differential variables for integrator capacity
    #[error("Problem has {0} differential variables but integrator capacity is {1}")]
    TooManyDifferentialVariables(usize, usize),
    /// Invalid tear stream configuration
    #[error("Invalid tear stream: {0}")]
    InvalidTearStream(String),
}

/// Newton-Raphson solver for nonlinear algebraic equations.
///
/// Solves systems of the form F(x) = 0 using the Newton-Raphson method:
/// x_{k+1} = x_k - J^{-1} F(x_k)
///
/// where J is the Jacobian matrix partialF/partialx.
pub struct NewtonRaphson {
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Relaxation factor (1.0 = full Newton step)
    pub relaxation: f64,
}

impl NewtonRaphson {
    /// Creates a new Newton-Raphson solver with default settings.
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        NewtonRaphson { tolerance, max_iterations, relaxation: 1.0 }
    }

    /// Creates a solver with relaxation (damped Newton method).
    pub fn with_relaxation(tolerance: f64, max_iterations: usize, relaxation: f64) -> Self {
        NewtonRaphson { tolerance, max_iterations, relaxation }
    }

    /// Solves the nonlinear system F(x) = 0.
    ///
    /// # Arguments
    ///
    /// * `f` - Function computing residuals and Jacobian
    /// * `x0` - Initial guess
    ///
    /// # Returns
    ///
    /// The solution vector if convergence is achieved.
    pub fn solve<F>(&self, f: F, x0: &[f64]) -> SolverResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> (Vec<f64>, Vec<Vec<f64>>),
    {
        let n = x0.len();
        let mut x = x0.to_vec();

        for iteration in 0..self.max_iterations {
            // Evaluate residuals and Jacobian
            let (residuals, jacobian_data) = f(&x);

            // Check convergence
            let norm = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();
            if norm < self.tolerance {
                return Ok(x);
            }

            {
                // Convert to nalgebra types
                let r_vec = DVector::from_vec(residuals);
                let j_mat = DMatrix::from_row_slice(n, n, &jacobian_data.concat());

                // Solve J * dx = -F
                let decomp = j_mat.lu();
                let dx = match decomp.solve(&(-r_vec)) {
                    Some(sol) => sol,
                    None => return Err(SolverError::SingularJacobian),
                };

                // Update: x = x + alpha * dx
                for i in 0..n {
                    x[i] += self.relaxation * dx[i];
                }
            }

            // Check for divergence
            if iteration > 10 && norm > 1e10 {
                return Err(SolverError::Diverged);
            }
        }

        Err(SolverError::MaxIterationsExceeded)
    }

    /// Solves the nonlinear system and returns solution with statistics.
    ///
    /// # Arguments
    ///
    /// * `f` - Function computing residuals and Jacobian
    /// * `x0` - Initial guess
    ///
    /// # Returns
    ///
    /// Tuple of (solution vector, solver statistics) if convergence is achieved.
    pub fn solve_with_stats<F>(&self, f: F, x0: &[f64]) -> SolverResult<(Vec<f64>, SolverStats)>
    where
        F: Fn(&[f64]) -> (Vec<f64>, Vec<Vec<f64>>),
    {
        let n = x0.len();

        let mut x = x0.to_vec();

        let mut iterations;
        let mut function_evals = 0;
        let mut jacobian_evals = 0;
        let mut final_residual;

        for iteration in 0..self.max_iterations {
            iterations = iteration + 1;

            // Evaluate residuals and Jacobian

            {
                let (residuals, jacobian_data) = f(&x);
                function_evals += 1;
                jacobian_evals += 1;

                // Check convergence
                let norm = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();
                final_residual = norm;
                if norm < self.tolerance {
                    let stats =
                        SolverStats { iterations, function_evals, jacobian_evals, final_residual };
                    return Ok((x, stats));
                }

                // Convert to nalgebra types
                let r_vec = DVector::from_vec(residuals);
                let j_mat = DMatrix::from_row_slice(n, n, &jacobian_data.concat());

                // Solve J * dx = -F
                let decomp = j_mat.lu();
                let dx = match decomp.solve(&(-r_vec)) {
                    Some(sol) => sol,
                    None => return Err(SolverError::SingularJacobian),
                };

                // Update: x = x + alpha * dx
                for i in 0..n {
                    x[i] += self.relaxation * dx[i];
                }
            }

            // Check for divergence
            if iteration > 10 && final_residual > 1e10 {
                return Err(SolverError::Diverged);
            }
        }

        Err(SolverError::MaxIterationsExceeded)
    }
}

/// Configuration for DAE solvers.
pub struct DAESolverConfig {
    /// Absolute tolerance
    pub atol: f64,
    /// Relative tolerance
    pub rtol: f64,
    /// Maximum step size
    pub max_step: f64,
    /// Initial step size
    pub initial_step: f64,
}

impl Default for DAESolverConfig {
    fn default() -> Self {
        DAESolverConfig { atol: 1e-6, rtol: 1e-6, max_step: 0.1, initial_step: 0.01 }
    }
}

/// Statistics from a solver run.
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of function evaluations
    pub function_evals: usize,
    /// Number of Jacobian evaluations
    pub jacobian_evals: usize,
    /// Final residual norm
    pub final_residual: f64,
}

impl SolverStats {
    /// Creates new solver statistics.
    pub fn new() -> Self {
        SolverStats { iterations: 0, function_evals: 0, jacobian_evals: 0, final_residual: 0.0 }
    }
}

impl Default for SolverStats {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level flowsheet solver.
///
/// Integrates a dynamic flowsheet over time by solving the underlying DAE system.
/// Requires the `solvers` feature to be enabled.
///
/// ODE solver with user-provided Jacobian function.
pub fn solve_ode_with_jacobian<const N: usize, F, J>(
    f: F,
    jacobian: J,
    y0: Vec<f64>,
    t_span: (f64, f64),
    method: integration::IntegrationMethod,
) -> SolverResult<Vec<(f64, Vec<f64>)>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
{
    #[cfg(feature = "solvers")]
    {
        use differential_equations::{
            methods::{ExplicitRungeKutta, ImplicitRungeKutta},
            ode::{ODE, ODEProblem},
        };
        use nalgebra::SVector;

        // Create ODE wrapper
        struct UserODE<F, J> {
            f: F,
            jacobian: J,
            n: usize,
        }

        impl<const M: usize, F, J> ODE<f64, SVector<f64, M>> for UserODE<F, J>
        where
            F: Fn(f64, &[f64]) -> Vec<f64>,
            J: Fn(f64, &[f64]) -> Vec<Vec<f64>>,
        {
            fn diff(&self, t: f64, y: &SVector<f64, M>, dydt: &mut SVector<f64, M>) {
                let y_slice = &y.as_slice()[..self.n];
                let result = (self.f)(t, y_slice);
                for i in 0..self.n {
                    dydt[i] = result[i];
                }
            }

            fn jacobian(
                &self,
                t: f64,
                y: &SVector<f64, M>,
                dfdy: &mut differential_equations::prelude::Matrix<f64>,
            ) {
                let y_slice = y.as_slice();
                let jac = (self.jacobian)(t, &y_slice[..self.n]);
                for i in 0..self.n {
                    for j in 0..self.n {
                        dfdy[(i, j)] = jac[i][j];
                    }
                }
            }
        }

        let n = y0.len();
        if n != N {
            return Err(SolverError::InvalidInitialConditions);
        }

        let mut y0_svector = SVector::<f64, N>::zeros();
        for i in 0..n {
            y0_svector[i] = y0[i];
        }

        let ode = UserODE { f, jacobian, n };
        let problem = ODEProblem::new(&ode, t_span.0, t_span.1, y0_svector);

        match method {
            integration::IntegrationMethod::RK4 => {
                let mut solver = ExplicitRungeKutta::rk4(0.01);
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::Dopri5 => {
                let mut solver = ExplicitRungeKutta::dopri5();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::Euler => {
                let mut solver = ExplicitRungeKutta::euler(0.01);
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::Radau5 => {
                let mut solver = ImplicitRungeKutta::radau5();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::GaussLegendre4 => {
                let mut solver = ImplicitRungeKutta::gauss_legendre_4();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::GaussLegendre6 => {
                let mut solver = ImplicitRungeKutta::gauss_legendre_6();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::LobattoIIIC2 => {
                let mut solver = ImplicitRungeKutta::lobatto_iiic_2();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::LobattoIIIC4 => {
                let mut solver = ImplicitRungeKutta::lobatto_iiic_4();
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            // Fixed-step explicit methods
            integration::IntegrationMethod::Heun => {
                let mut solver = ExplicitRungeKutta::heun(0.01);
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::Midpoint => {
                let mut solver = ExplicitRungeKutta::midpoint(0.01);
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
            integration::IntegrationMethod::Ralston => {
                let mut solver = ExplicitRungeKutta::ralston(0.01);
                match problem.solve(&mut solver) {
                    Ok(solution) => {
                        let mut result = Vec::new();
                        for i in 0..solution.t.len() {
                            let t = solution.t[i];
                            let y_vec = (0..n).map(|j| solution.y[i][j]).collect();
                            result.push((t, y_vec));
                        }
                        Ok(result)
                    }
                    Err(e) => Err(SolverError::ODESolverFailed(e.to_string())),
                }
            }
        }
    }

    #[cfg(not(feature = "solvers"))]
    {
        Err(SolverError::FeatureNotEnabled)
    }
}

/// Solves a steady-state flowsheet (algebraic equations only).
///
/// # Example
///
/// ```ignore
/// use nomata::{Flowsheet, Steady, VariableRegistry, solvers::solve_steady_state};
///
/// let mut flowsheet = Flowsheet::<Steady>::new();
/// let registry = VariableRegistry::new();
///
/// // ... build flowsheet, harvest equations ...
///
/// let solution = solve_steady_state(&flowsheet, &registry)?;
/// ```
pub fn solve_steady_state<T: crate::TimeDomain>(
    flowsheet: &crate::Flowsheet<T>,
    registry: &crate::VariableRegistry,
) -> SolverResult<Vec<f64>> {
    let equations = flowsheet.equations();
    let initial_guess = registry.get_all_values();

    // Define combined residual and Jacobian function
    let f = |x: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let derivatives = vec![0.0; x.len()];
        let residuals = equations.evaluate_residuals(x, &derivatives, 0.0);
        let jacobian = equations.compute_jacobian(x);
        (residuals, jacobian)
    };

    // Solve using Newton-Raphson
    let solver = NewtonRaphson::new(1e-6, 100);
    solver.solve(f, &initial_guess)
}

/// Unified flowsheet solution that can represent different solver results.
#[derive(Debug, Clone)]
pub enum FlowsheetSolution {
    /// Solution from steady-state solver (no recycle loops)
    SteadyState {
        /// Converged variable values
        values: Vec<f64>,
        /// Solver statistics
        stats: SolverStats,
    },
    /// Solution from recycle solver (with recycle loops)
    Recycle {
        /// Converged tear stream values
        tear_values: Vec<f64>,
        /// Solver statistics
        stats: SolverStats,
    },
}

/// Unified solver API for flowsheets.
///
/// Automatically detects flowsheet topology and chooses the appropriate solver:
/// - **Acyclic flowsheets**: Uses steady-state Newton-Raphson solver
/// - **Cyclic flowsheets**: Uses recycle/tear-stream solver (future extension)
///
/// # Example
///
/// ```ignore
/// use nomata::{Flowsheet, Steady, solvers::solve};
///
/// let mut flowsheet = Flowsheet::<Steady>::new();
/// // ... build flowsheet and harvest equations ...
///
/// let solution = solve(&flowsheet)?;
///
/// match solution {
///     FlowsheetSolution::SteadyState { values, stats } => {
///         println!("Solved in {} iterations", stats.iterations);
///         // Use values...
///     }
///     FlowsheetSolution::Recycle { tear_values, stats } => {
///         // Handle recycle case...
///     }
/// }
/// ```
/// Configuration for the unified solver.
///
/// Allows users to customize solver behavior for both acyclic and cyclic flowsheets.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Solution method for recycle convergence
    pub recycle_method: recycle::SolverMethod,
    /// Maximum iterations for solvers
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Damping factor for fixed-point iterations (0.0 to 1.0)
    pub damping: f64,
    /// Whether to try fallback methods on failure
    pub enable_fallback: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            recycle_method: recycle::SolverMethod::Wegstein,
            max_iterations: 200,
            tolerance: 1e-6,
            damping: 0.5,
            enable_fallback: true,
        }
    }
}

impl SolverConfig {
    /// Creates a new solver configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the recycle solver method.
    pub fn with_method(mut self, method: recycle::SolverMethod) -> Self {
        self.recycle_method = method;
        self
    }

    /// Sets the maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the damping factor for fixed-point iterations.
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping.clamp(0.0, 1.0);
        self
    }

    /// Enables or disables fallback to alternative methods on failure.
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }
}

/// Unified flowsheet solver with default configuration.
///
/// Automatically detects flowsheet topology and chooses the appropriate solver:
/// - **Acyclic flowsheets**: Uses steady-state Newton-Raphson solver
/// - **Cyclic flowsheets**: Uses recycle/tear-stream solver with Wegstein method
///
/// For custom configuration, use [`solve_with_config`] instead.
pub fn solve<T: crate::TimeDomain>(
    flowsheet: &crate::Flowsheet<T>,
) -> SolverResult<FlowsheetSolution> {
    solve_with_config(flowsheet, &SolverConfig::default())
}

/// Unified flowsheet solver with custom configuration.
///
/// Allows users to specify solver method, tolerances, and other parameters.
///
/// # Example
///
/// ```ignore
/// use nomata::{Flowsheet, Steady, solvers::{solve_with_config, SolverConfig}};
/// use nomata::solvers::recycle::SolverMethod;
///
/// let flowsheet = Flowsheet::<Steady>::new();
/// let config = SolverConfig::new()
///     .with_method(SolverMethod::Newton)
///     .with_tolerance(1e-8)
///     .with_max_iterations(500);
///
/// let solution = solve_with_config(&flowsheet, &config)?;
/// ```
pub fn solve_with_config<T: crate::TimeDomain>(
    flowsheet: &crate::Flowsheet<T>,
    config: &SolverConfig,
) -> SolverResult<FlowsheetSolution> {
    // Check if flowsheet has cycles
    let cycles = flowsheet.detect_cycles();

    if cycles.is_empty() {
        // No cycles - use steady-state solver
        solve_steady_state_unified(flowsheet)
    } else {
        // Has cycles - use recycle solver
        solve_with_recycle(flowsheet, config, cycles.len())
    }
}

/// Internal function to solve flowsheets with recycle loops.
///
/// Uses the RecycleSolver with the configured method to converge tear streams.
fn solve_with_recycle<T: crate::TimeDomain>(
    flowsheet: &crate::Flowsheet<T>,
    config: &SolverConfig,
    num_cycles: usize,
) -> SolverResult<FlowsheetSolution> {
    use recycle::RecycleSolver;

    // Create registry from equations
    let registry = flowsheet.create_registry();
    let equations = flowsheet.equations();

    // Get initial guess from registry
    let initial_guess = registry.get_all_values();
    let n_vars = initial_guess.len();

    if n_vars == 0 {
        // No variables to solve - this is likely a configuration error
        // (e.g., forgot to call harvest_equations())
        return Err(SolverError::NoVariablesToSolve);
    }

    // Create the recycle solver with user-configured method
    let mut solver =
        RecycleSolver::with_method(config.recycle_method, config.max_iterations, config.tolerance);

    // Capture damping factor
    let damping = config.damping;

    // Define the flowsheet computation function for non-autodiff path
    #[cfg(not(feature = "autodiff"))]
    let compute_flowsheet = |tear_values: &[f64]| -> Vec<f64> {
        // Start with current tear stream values
        let mut state = tear_values.to_vec();

        // Ensure state vector is the right size
        if state.len() < n_vars {
            state.resize(n_vars, 0.0);
        }

        // Evaluate the equation residuals
        let derivatives = vec![0.0; state.len()];
        let residuals = equations.evaluate_residuals(&state, &derivatives, 0.0);

        // For fixed-point iteration: x_new = x - alpha * F(x)
        for i in 0..state.len().min(residuals.len()) {
            state[i] -= damping * residuals[i];
        }

        // Return updated values (only the tear stream portion)
        state[..tear_values.len()].to_vec()
    };

    // Define the flowsheet computation function for autodiff path
    #[cfg(feature = "autodiff")]
    let compute_flowsheet = |tear_values: &[num_dual::Dual64]| -> Vec<num_dual::Dual64> {
        use num_dual::Dual64;

        // Extract real parts for equation evaluation
        let real_values: Vec<f64> = tear_values.iter().map(|d| d.re).collect();

        // Start with current tear stream values
        let mut state = real_values.clone();

        // Ensure state vector is the right size
        if state.len() < n_vars {
            state.resize(n_vars, 0.0);
        }

        // Evaluate the equation residuals
        let derivatives = vec![0.0; state.len()];
        let residuals = equations.evaluate_residuals(&state, &derivatives, 0.0);

        // For fixed-point iteration: x_new = x - alpha * F(x)
        for i in 0..state.len().min(residuals.len()) {
            state[i] -= damping * residuals[i];
        }

        // Return updated values as Dual64 (only the tear stream portion)
        state[..tear_values.len()].iter().map(|&v| Dual64::from(v)).collect()
    };

    // Solve the recycle system
    match solver.solve(initial_guess.clone(), compute_flowsheet) {
        Ok(solution) => {
            let stats = SolverStats {
                iterations: solution.iterations,
                function_evals: solution.iterations,
                jacobian_evals: if config.recycle_method == recycle::SolverMethod::Wegstein {
                    0
                } else {
                    solution.iterations
                },
                final_residual: solution.final_residual,
            };

            Ok(FlowsheetSolution::Recycle { tear_values: solution.tear_stream_values, stats })
        }
        Err(recycle::RecycleError::ConvergenceFailure { iterations, final_residual: _ })
            if config.enable_fallback =>
        {
            // Try falling back to Newton method for difficult cases
            let mut newton_solver = RecycleSolver::with_method(
                recycle::SolverMethod::Newton,
                config.max_iterations,
                config.tolerance,
            );

            match newton_solver.solve(initial_guess, compute_flowsheet) {
                Ok(solution) => {
                    let stats = SolverStats {
                        iterations: iterations + solution.iterations,
                        function_evals: iterations + solution.iterations,
                        jacobian_evals: solution.iterations,
                        final_residual: solution.final_residual,
                    };

                    Ok(FlowsheetSolution::Recycle {
                        tear_values: solution.tear_stream_values,
                        stats,
                    })
                }
                Err(_) => Err(SolverError::MaxIterationsExceeded),
            }
        }
        Err(recycle::RecycleError::ConvergenceFailure { .. }) => {
            Err(SolverError::MaxIterationsExceeded)
        }
        Err(recycle::RecycleError::InvalidTearStream(msg)) => Err(SolverError::InvalidTearStream(
            format!("{} (detected {} cycle(s))", msg, num_cycles),
        )),
        Err(recycle::RecycleError::FeatureNotEnabled) => Err(SolverError::FeatureNotEnabled),
    }
}

/// Internal function to solve acyclic flowsheets.
fn solve_steady_state_unified<T: crate::TimeDomain>(
    flowsheet: &crate::Flowsheet<T>,
) -> SolverResult<FlowsheetSolution> {
    // Create registry from equations
    let registry = flowsheet.create_registry();

    let equations = flowsheet.equations();
    let initial_guess = registry.get_all_values();

    // Define combined residual and Jacobian function
    let f = |x: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
        let derivatives = vec![0.0; x.len()];
        let residuals = equations.evaluate_residuals(x, &derivatives, 0.0);
        let jacobian = equations.compute_jacobian(x);
        (residuals, jacobian)
    };

    // Solve using Newton-Raphson with statistics
    let solver = NewtonRaphson::new(1e-6, 100);
    let (solution, stats) = solver.solve_with_stats(f, &initial_guess)?;

    Ok(FlowsheetSolution::SteadyState { values: solution, stats })
}

/// Steady-State Flowsheet Solver
///
/// Specialized Newton-Raphson solver for steady-state flowsheets.
///
/// This solver is optimized for `Flowsheet<Steady>` systems where:
/// - All time derivatives are zero
/// - Only algebraic equations need to be satisfied
/// - The system is represented as F(x) = 0
///
/// # Type-Level Guarantee
///
/// This solver only accepts `Flowsheet<Steady>`, preventing accidental use
/// on dynamic systems that require ODE/DAE solvers.
///
/// # Example
///
/// ```
/// use nomata::{Flowsheet, Steady, VariableRegistry};
/// use nomata::solvers::SteadyStateSolver;
///
/// let registry = VariableRegistry::new();
/// let flowsheet = Flowsheet::<Steady>::new();
///
/// let solver = SteadyStateSolver::new(1e-6, 100);
/// // solver.solve(&flowsheet, &registry, &initial_guess)?;
/// ```
pub struct SteadyStateSolver {
    /// Convergence tolerance for residuals
    pub tolerance: f64,
    /// Maximum Newton iterations
    pub max_iterations: usize,
    /// Damping factor (1.0 = full Newton step, <1.0 = damped)
    pub damping: f64,
    /// Finite difference step for Jacobian approximation
    pub fd_step: f64,
}

impl SteadyStateSolver {
    /// Creates a new steady-state solver with default damping.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Convergence criterion (L2 norm of residuals)
    /// * `max_iterations` - Maximum number of Newton iterations
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        SteadyStateSolver { tolerance, max_iterations, damping: 1.0, fd_step: 1e-8 }
    }

    /// Creates a solver with custom damping factor.
    ///
    /// Damping improves convergence for difficult problems:
    /// - `damping = 1.0`: Full Newton steps (fast but may diverge)
    /// - `damping < 1.0`: Damped steps (slower but more robust)
    pub fn with_damping(tolerance: f64, max_iterations: usize, damping: f64) -> Self {
        SteadyStateSolver { tolerance, max_iterations, damping, fd_step: 1e-8 }
    }

    /// Solves the steady-state flowsheet equations.
    ///
    /// This method is only available for `Flowsheet<Steady>`, enforced at compile time.
    /// Dynamic flowsheets must use ODE/DAE solvers instead.
    ///
    /// # Arguments
    ///
    /// * `residuals_fn` - Function that computes F(x) given state vector x
    /// * `x0` - Initial guess for the solution
    ///
    /// # Returns
    ///
    /// The solution vector satisfying F(x) aprox 0 within tolerance.
    ///
    /// # Errors
    ///
    /// - `MaxIterationsExceeded`: Failed to converge within max_iterations
    /// - `SingularJacobian`: Jacobian is singular (non-invertible)
    /// - `Diverged`: Solution is diverging instead of converging
    pub fn solve<F>(&self, residuals_fn: F, x0: &[f64]) -> SolverResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        {
            let n = x0.len();
            let mut x = x0.to_vec();
            for iteration in 0..self.max_iterations {
                // Evaluate residuals at current point
                let residuals = residuals_fn(&x);
                // Check convergence
                let residual_norm: f64 = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();
                if residual_norm < self.tolerance {
                    return Ok(x);
                }
                // Check for divergence
                if iteration > 10 && residual_norm > 1e10 {
                    return Err(SolverError::Diverged);
                }
                // Compute Jacobian using finite differences
                let jacobian = self.compute_jacobian(&residuals_fn, &x, &residuals);
                // Solve J * dx = -F using LU decomposition
                let r_vec = DVector::from_vec(residuals);
                let j_mat = DMatrix::from_row_slice(n, n, &jacobian.concat());
                let decomp = j_mat.lu();
                let dx = match decomp.solve(&(-r_vec)) {
                    Some(sol) => sol,
                    None => return Err(SolverError::SingularJacobian),
                };
                // Update with damping: x_{k+1} = x_k + alpha * dx
                for i in 0..n {
                    x[i] += self.damping * dx[i];
                }
            }
            Err(SolverError::MaxIterationsExceeded)
        }
        #[cfg(not(feature = "solvers"))]
        {
            panic!("The 'solvers' feature must be enabled to use this solver.");
        }
    }

    /// Computes the Jacobian matrix using finite differences.
    ///
    /// J[i][j] = partialF_i/partialx_j aprox (F_i(x + h*e_j) - F_i(x)) / h
    fn compute_jacobian<F>(&self, f: &F, x: &[f64], f0: &[f64]) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        let mut jacobian = vec![vec![0.0; n]; n];

        for j in 0..n {
            let mut x_perturbed = x.to_vec();
            x_perturbed[j] += self.fd_step;

            let f_perturbed = f(&x_perturbed);

            for i in 0..n {
                jacobian[i][j] = (f_perturbed[i] - f0[i]) / self.fd_step;
            }
        }

        jacobian
    }
}

impl Default for SteadyStateSolver {
    fn default() -> Self {
        Self::new(1e-6, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_raphson_linear() {
        // Solve x - 5 = 0
        let f = |x: &[f64]| {
            let residual = vec![x[0] - 5.0];
            let jacobian = vec![vec![1.0]];
            (residual, jacobian)
        };

        let solver = NewtonRaphson::new(1e-6, 10);
        let solution = solver.solve(f, &[0.0]).unwrap();

        assert!((solution[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_raphson_nonlinear() {
        // Solve x^2 - 4 = 0 (solution: x = 2)
        let f = |x: &[f64]| {
            let residual = vec![x[0] * x[0] - 4.0];
            let jacobian = vec![vec![2.0 * x[0]]]; // df/dx = 2x
            (residual, jacobian)
        };

        let solver = NewtonRaphson::new(1e-6, 20);
        let solution = solver.solve(f, &[1.0]).unwrap();

        assert!((solution[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_solver_stats() {
        let stats = SolverStats::new();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.function_evals, 0);
    }

    #[test]
    fn test_dae_solver_config() {
        let config = DAESolverConfig::default();
        assert_eq!(config.atol, 1e-6);
        assert_eq!(config.rtol, 1e-6);
    }

    #[test]
    fn test_solver_error_display() {
        let err = SolverError::FeatureNotEnabled;
        assert_eq!(err.to_string(), "ODE/DAE solver requires the 'solvers' feature to be enabled");

        let err = SolverError::NoVariablesToSolve;
        assert_eq!(
            err.to_string(),
            "No variables to solve. Did you forget to call harvest_equations()?"
        );

        let err = SolverError::InvalidTearStream("test message".to_string());
        assert_eq!(err.to_string(), "Invalid tear stream: test message");
    }

    #[test]
    fn test_solve_ode_with_jacobian() {
        // Test ODE: dy/dt = -y, y(0) = 1
        // Analytical solution: y(t) = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let jacobian = |_t: f64, _y: &[f64]| vec![vec![-1.0]];

        let y0 = vec![1.0];
        let t_span = (0.0, 1.0);

        let result = solve_ode_with_jacobian::<1, _, _>(
            f,
            jacobian,
            y0,
            t_span,
            integration::IntegrationMethod::RK4,
        )
        .unwrap();

        // Check that we got some results
        assert!(!result.is_empty());

        // Check initial condition
        assert!((result[0].0 - 0.0).abs() < 1e-10);
        assert!((result[0].1[0] - 1.0).abs() < 1e-10);

        // Check final value is approximately e^(-1) aprox 0.3679
        let final_t = result.last().unwrap().0;
        let final_y = result.last().unwrap().1[0];
        let analytical = (-final_t).exp();
        assert!((final_y - analytical).abs() < 0.01); // Loose tolerance for integration
    }

    #[test]
    fn test_steady_state_solver_creation() {
        let solver = SteadyStateSolver::new(1e-6, 50);
        assert_eq!(solver.tolerance, 1e-6);
        assert_eq!(solver.max_iterations, 50);
        assert_eq!(solver.damping, 1.0);
    }

    #[test]
    fn test_steady_state_solver_with_damping() {
        let solver = SteadyStateSolver::with_damping(1e-8, 200, 0.5);
        assert_eq!(solver.damping, 0.5);
    }

    #[test]
    fn test_steady_state_solver_linear_system() {
        // Solve: x + y = 3, 2x - y = 0
        // Solution: x = 1, y = 2
        let solver = SteadyStateSolver::new(1e-10, 100);

        let residuals = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let solution = solver.solve(residuals, &[0.0, 0.0]).unwrap();

        assert!((solution[0] - 1.0).abs() < 1e-6);
        assert!((solution[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_steady_state_solver_nonlinear_system() {
        // Solve: x^2 + y^2 = 5, x - y = 1
        // Solution: x = 2, y = 1 (one of the solutions)
        let solver = SteadyStateSolver::new(1e-10, 100);

        let residuals = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 5.0, x[0] - x[1] - 1.0];

        let solution = solver.solve(residuals, &[1.5, 0.5]).unwrap();

        // Check solution satisfies equations
        let res = residuals(&solution);
        assert!(res[0].abs() < 1e-6);
        assert!(res[1].abs() < 1e-6);
    }

    #[test]
    fn test_steady_state_solver_max_iterations() {
        let solver = SteadyStateSolver::new(1e-20, 2); // Very tight tolerance, few iterations

        let residuals = |x: &[f64]| vec![x[0] * x[0] - 2.0]; // x = sqrt(2)

        let result = solver.solve(residuals, &[0.5]);
        assert!(matches!(result, Err(SolverError::MaxIterationsExceeded)));
    }

    #[test]
    fn test_steady_state_solver_damping() {
        // Difficult problem that benefits from damping
        let solver = SteadyStateSolver::with_damping(1e-6, 100, 0.3);

        let residuals = |x: &[f64]| vec![x[0] * x[0] - 4.0]; // x = 2

        let solution = solver.solve(residuals, &[0.1]).unwrap();
        assert!((solution[0].abs() - 2.0).abs() < 1e-4);
    }
}
