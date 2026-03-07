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
pub mod recycle;

use crate::EquationModel;
#[cfg(feature = "autodiff")]
use crate::autodiff::compute_jacobian;
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
    /// No variables to solve in the equation system
    #[error("No variables to solve. Did you forget to call harvest_equations()?")]
    NoVariablesToSolve,
    /// No equations in the system
    #[error("No equations to solve")]
    NoEquations,
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
        let mut final_residual;

        for (eval_count, iteration) in (0..self.max_iterations).enumerate() {
            iterations = iteration + 1;
            let function_evals = eval_count + 1;
            let jacobian_evals = eval_count + 1;

            // Evaluate residuals and Jacobian
            {
                let (residuals, jacobian_data) = f(&x);

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

/// Typed identifier for a unit in an EOFlowsheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitId(pub usize);

/// Typed identifier for a connection in an EOFlowsheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId(pub usize);

/// Stream state: flow, temperature, pressure.
#[derive(Debug, Clone, Copy)]
pub struct StreamData {
    /// Mass or molar flow rate
    pub flow: f64,
    /// Temperature (K)
    pub temperature: f64,
    /// Pressure (Pa)
    pub pressure: f64,
}

impl StreamData {
    /// Creates a new stream with the given properties.
    pub fn new(flow: f64, temperature: f64, pressure: f64) -> Self {
        StreamData { flow, temperature, pressure }
    }
}

/// Reference to an outlet port of a unit.
#[derive(Debug, Clone, Copy)]
pub struct OutletPort {
    /// The unit that has the outlet
    pub unit_id: UnitId,
    /// Port index (0 for first outlet, 1 for second, etc.)
    pub port: usize,
}

/// Reference to an inlet port of a unit.
#[derive(Debug, Clone, Copy)]
pub struct InletPort {
    /// The unit that has the inlet
    pub unit_id: UnitId,
    /// Port index (0 for first inlet, 1 for second, etc.)
    pub port: usize,
}

/// Object-safe wrapper for [`EquationModel`].
///
/// This trait mirrors `EquationModel` but is dyn-compatible, enabling
/// heterogeneous collections of unit operations inside flowsheets.
pub trait EquationModelDyn {
    /// Number of variables in the model.
    fn n_variables(&self) -> usize;
    /// Number of equations (residuals) in the model.
    fn n_equations(&self) -> usize;
    /// Names of the variables, in order.
    fn variable_names(&self) -> Vec<&str>;
    /// Compute residuals given variable values.
    fn residuals(&self, vars: &[f64]) -> Vec<f64>;
    /// Get the current variable values.
    fn get_variables(&self) -> Vec<f64>;
    /// Set variable values.
    fn set_variables(&mut self, vars: &[f64]);
    /// Returns indices of variables that are free (to be solved).
    fn free_indices(&self) -> Vec<usize>;
    /// Returns the outlet stream values (flow, temp, pressure) for port 0.
    fn get_outlet(&self) -> (f64, f64, f64) {
        let vars = self.get_variables();
        let n = vars.len();
        if n >= 6 {
            (vars[n - 3], vars[n - 2], vars[n - 1])
        } else if n >= 3 {
            (vars[0], vars[1], vars[2])
        } else {
            (0.0, 0.0, 0.0)
        }
    }
    /// Compute residuals with dual numbers for automatic differentiation.
    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64>;
}

impl<T: EquationModel> EquationModelDyn for T {
    fn n_variables(&self) -> usize {
        EquationModel::n_variables(self)
    }
    fn n_equations(&self) -> usize {
        EquationModel::n_equations(self)
    }
    fn variable_names(&self) -> Vec<&str> {
        EquationModel::variable_names(self)
    }
    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        EquationModel::residuals(self, vars)
    }
    fn get_variables(&self) -> Vec<f64> {
        EquationModel::get_variables(self)
    }
    fn set_variables(&mut self, vars: &[f64]) {
        EquationModel::set_variables(self, vars)
    }
    fn free_indices(&self) -> Vec<usize> {
        EquationModel::free_indices(self)
    }
    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64> {
        EquationModel::residuals_dual(self, vars)
    }
}

/// Solves a single [`EquationModel`] using Newton-Raphson.
///
/// When the `autodiff` feature is enabled, the Jacobian is computed via
/// forward-mode automatic differentiation; otherwise central finite differences
/// are used as a fallback.
///
/// # Arguments
///
/// * `model` - The equation model to solve (free variables are determined by
///   [`EquationModel::free_indices`]).
/// * `tolerance` - L2-norm convergence tolerance for residuals.
/// * `max_iterations` - Maximum Newton iterations before failure.
///
/// # Returns
///
/// [`SolverStats`] on convergence; a [`SolverError`] otherwise.
///
/// # Example
///
/// ```ignore
/// use nomata::prelude::*;
/// let mut pump = Pump::new("pump").with_outlet_pressure(5e5).build()?;
/// feed_stream.connect_to(&mut pump);
/// let stats = solve_equation_model(&mut pump, 1e-10, 100)?;
/// println!("Solved in {} iterations", stats.iterations);
/// ```
pub fn solve_equation_model<M: EquationModel>(
    model: &mut M,
    tolerance: f64,
    max_iterations: usize,
) -> SolverResult<SolverStats> {
    let all_vars = model.get_variables();
    let free_indices = model.free_indices();
    let n_free = free_indices.len();
    let n_eqs = model.n_equations();

    if n_free == 0 {
        return Err(SolverError::NoVariablesToSolve);
    }
    if n_eqs == 0 {
        return Err(SolverError::NoEquations);
    }

    let mut all_vars = all_vars;
    let mut x: Vec<f64> = free_indices.iter().map(|&i| all_vars[i]).collect();

    for iteration in 0..max_iterations {
        // Reconstruct full variable vector.
        for (j, &idx) in free_indices.iter().enumerate() {
            all_vars[idx] = x[j];
        }

        let residuals = model.residuals(&all_vars);
        let norm = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        if norm < tolerance {
            model.set_variables(&all_vars);
            return Ok(SolverStats {
                iterations: iteration + 1,
                function_evals: iteration + 1,
                jacobian_evals: iteration + 1,
                final_residual: norm,
            });
        }

        // Compute Jacobian: AD when available, central FD otherwise.
        #[cfg(feature = "autodiff")]
        let jacobian = {
            let all_vars_snap = all_vars.clone();
            let dual_fn = |free_vars: &[num_dual::Dual64]| -> Vec<num_dual::Dual64> {
                let mut vars: Vec<num_dual::Dual64> =
                    all_vars_snap.iter().map(|&v| num_dual::Dual64::from(v)).collect();
                for (j, &idx) in free_indices.iter().enumerate() {
                    vars[idx] = free_vars[j];
                }
                model.residuals_dual(&vars)
            };
            compute_jacobian(dual_fn, &x)
        };

        #[cfg(feature = "autodiff")]
        let jac_entries = jacobian.entries;

        #[cfg(not(feature = "autodiff"))]
        let jac_entries = {
            let free_indices_snap = free_indices.clone();
            let all_vars_snap = all_vars.clone();
            let residual_fn_fd = |free_vars: &[f64]| -> Vec<f64> {
                let mut vars = all_vars_snap.clone();
                for (j, &idx) in free_indices_snap.iter().enumerate() {
                    vars[idx] = free_vars[j];
                }
                model.residuals(&vars)
            };
            let mut entries = vec![0.0f64; n_eqs * n_free];
            for i in 0..n_free {
                let h = 1e-8 * (1.0 + x[i].abs());
                let mut x_p = x.clone();
                x_p[i] += h;
                let f_p = residual_fn_fd(&x_p);
                let mut x_m = x.clone();
                x_m[i] -= h;
                let f_m = residual_fn_fd(&x_m);
                for j in 0..n_eqs {
                    entries[j * n_free + i] = (f_p[j] - f_m[j]) / (2.0 * h);
                }
            }
            entries
        };

        let r_vec = DVector::from_vec(residuals);
        let j_mat = DMatrix::from_row_slice(n_eqs, n_free, &jac_entries);

        let dx = if n_eqs == n_free {
            let decomp = j_mat.lu();
            match decomp.solve(&(-&r_vec)) {
                Some(sol) => sol,
                None => return Err(SolverError::SingularJacobian),
            }
        } else {
            j_mat
                .svd(true, true)
                .solve(&(-&r_vec), 1e-12)
                .map_err(|_| SolverError::SingularJacobian)?
        };

        for i in 0..n_free {
            x[i] += dx[i];
        }

        if iteration > 10 && norm > 1e10 {
            return Err(SolverError::Diverged);
        }
    }

    Err(SolverError::MaxIterationsExceeded)
}

/// Statistics from a completed sequential-modular solve.
#[derive(Debug, Clone)]
pub struct SMSolverStats {
    /// Total Newton iterations summed across all units.
    pub total_iterations: usize,
    /// Number of unit operations solved.
    pub units_solved: usize,
    /// Maximum residual norm observed across all units at convergence.
    pub max_unit_residual: f64,
}

/// Equation-Oriented (EO) flowsheet.
///
/// Assembles all unit equations into one system and solves simultaneously
/// using Newton-Raphson. Supports forward-mode AD Jacobian evaluation via
/// the `autodiff` feature; falls back to central finite differences when that
/// feature is absent.
pub struct EOFlowsheet {
    /// Units indexed by UnitId.
    pub(crate) units: Vec<(String, Box<dyn EquationModelDyn>)>,
    /// Connections: (from_unit, from_var_indices, to_unit, to_var_indices).
    pub(crate) connections: Vec<(usize, Vec<usize>, usize, Vec<usize>)>,
    /// Feeds: (unit_index, var_indices, values).
    pub(crate) feeds: Vec<(usize, Vec<usize>, Vec<f64>)>,
}

impl EOFlowsheet {
    /// Creates a new empty flowsheet.
    pub fn new() -> Self {
        EOFlowsheet { units: Vec::new(), connections: Vec::new(), feeds: Vec::new() }
    }

    /// Adds a unit to the flowsheet and returns its [`UnitId`].
    pub fn add<M: EquationModel + 'static>(&mut self, name: &str, unit: M) -> UnitId {
        let id = UnitId(self.units.len());
        self.units.push((name.to_string(), Box::new(unit)));
        id
    }

    /// Returns the variable offset for a given unit index.
    pub(crate) fn var_offset(&self, unit_idx: usize) -> usize {
        self.units[..unit_idx].iter().map(|(_, u)| u.n_variables()).sum()
    }

    /// Total number of variables across all units.
    pub fn total_variables(&self) -> usize {
        self.units.iter().map(|(_, u)| u.n_variables()).sum()
    }

    /// Total number of equations (unit equations + connection constraints).
    pub fn total_equations(&self) -> usize {
        let unit_eqs: usize = self.units.iter().map(|(_, u)| u.n_equations()).sum();
        let conn_eqs: usize =
            self.connections.iter().map(|(_, fv, _, tv)| fv.len().min(tv.len())).sum();
        unit_eqs + conn_eqs
    }

    /// Connects the outlet variables of `src` to the inlet variables of `dst`.
    ///
    /// `src_vars` and `dst_vars` list the variable indices (within each unit's
    /// own variable vector) that should be equated.
    pub fn connect(
        &mut self,
        src: UnitId,
        src_vars: Vec<usize>,
        dst: UnitId,
        dst_vars: Vec<usize>,
    ) -> ConnectionId {
        let id = ConnectionId(self.connections.len());
        self.connections.push((src.0, src_vars, dst.0, dst_vars));
        id
    }

    /// Sets feed stream values for a unit.
    ///
    /// The solver will fix these variables at the given values throughout the solve.
    pub fn set_feed(&mut self, unit: UnitId, var_indices: Vec<usize>, values: Vec<f64>) {
        self.feeds.push((unit.0, var_indices, values));
    }

    /// Returns all variables as a flat vector, applying feeds.
    pub(crate) fn collect_vars(&self) -> Vec<f64> {
        let mut vars: Vec<f64> = self.units.iter().flat_map(|(_, u)| u.get_variables()).collect();
        for (unit_idx, var_idxs, values) in &self.feeds {
            let offset = self.var_offset(*unit_idx);
            for (vi, val) in var_idxs.iter().zip(values.iter()) {
                vars[offset + vi] = *val;
            }
        }
        vars
    }

    /// Sets all variables from a flat vector back into each unit.
    pub(crate) fn set_all_variables(&mut self, vars: &[f64]) {
        let mut offset = 0;
        for (_, unit) in &mut self.units {
            let n = unit.n_variables();
            unit.set_variables(&vars[offset..offset + n]);
            offset += n;
        }
    }

    /// Evaluates the full system of residuals (unit equations + connections).
    pub fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        let mut all_residuals = Vec::new();
        // Unit equations.
        let mut offset = 0;
        for (_, unit) in &self.units {
            let n = unit.n_variables();
            all_residuals.extend(unit.residuals(&vars[offset..offset + n]));
            offset += n;
        }
        // Connection constraints: src_var == dst_var.
        for (src_idx, src_vars, dst_idx, dst_vars) in &self.connections {
            let src_off = self.var_offset(*src_idx);
            let dst_off = self.var_offset(*dst_idx);
            for (&sv, &dv) in src_vars.iter().zip(dst_vars.iter()) {
                all_residuals.push(vars[src_off + sv] - vars[dst_off + dv]);
            }
        }
        all_residuals
    }

    /// Evaluate residuals with dual numbers for forward-mode AD.
    #[cfg(feature = "autodiff")]
    pub fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64> {
        let mut all_residuals = Vec::new();
        let mut offset = 0;
        for (_, unit) in &self.units {
            let n = unit.n_variables();
            all_residuals.extend(unit.residuals_dual(&vars[offset..offset + n]));
            offset += n;
        }
        for (src_idx, src_vars, dst_idx, dst_vars) in &self.connections {
            let src_off = self.var_offset(*src_idx);
            let dst_off = self.var_offset(*dst_idx);
            for (&sv, &dv) in src_vars.iter().zip(dst_vars.iter()) {
                let diff = vars[src_off + sv] - vars[dst_off + dv];
                all_residuals.push(diff);
            }
        }
        all_residuals
    }

    /// Solves the entire flowsheet simultaneously using Newton-Raphson.
    ///
    /// The solver determines free variables (not pinned as feeds), assembles
    /// the combined residual vector, and iterates until convergence.
    pub fn solve(&mut self, tolerance: f64, max_iterations: usize) -> SolverResult<SolverStats> {
        // Collect all variables; apply feeds.
        let mut all_vars = self.collect_vars();
        let n_total = all_vars.len();

        // Determine which variable indices are free (not pinned by feeds).
        let mut pinned: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (unit_idx, var_idxs, _) in &self.feeds {
            let offset = self.var_offset(*unit_idx);
            for vi in var_idxs {
                pinned.insert(offset + vi);
            }
        }

        let free_indices: Vec<usize> = (0..n_total).filter(|i| !pinned.contains(i)).collect();
        let n_free = free_indices.len();
        let n_eqs = self.total_equations();

        if n_free == 0 {
            return Err(SolverError::NoVariablesToSolve);
        }
        if n_eqs == 0 {
            return Err(SolverError::NoEquations);
        }

        let mut x: Vec<f64> = free_indices.iter().map(|&i| all_vars[i]).collect();

        for iteration in 0..max_iterations {
            // Reconstruct full variable vector.
            for (j, &idx) in free_indices.iter().enumerate() {
                all_vars[idx] = x[j];
            }
            // Re-apply feeds.
            for (unit_idx, var_idxs, values) in &self.feeds {
                let offset = self.var_offset(*unit_idx);
                for (vi, val) in var_idxs.iter().zip(values.iter()) {
                    all_vars[offset + vi] = *val;
                }
            }

            let residuals = self.residuals(&all_vars);
            let norm = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

            if norm < tolerance {
                for (j, &idx) in free_indices.iter().enumerate() {
                    all_vars[idx] = x[j];
                }
                for (unit_idx, var_idxs, values) in &self.feeds {
                    let offset = self.var_offset(*unit_idx);
                    for (vi, val) in var_idxs.iter().zip(values.iter()) {
                        all_vars[offset + vi] = *val;
                    }
                }
                self.set_all_variables(&all_vars);

                return Ok(SolverStats {
                    iterations: iteration + 1,
                    function_evals: iteration + 1,
                    jacobian_evals: iteration + 1,
                    final_residual: norm,
                });
            }

            // Compute Jacobian.
            #[cfg(feature = "autodiff")]
            let jacobian = {
                let all_vars_snap = all_vars.clone();
                let free_indices_snap = free_indices.clone();
                let dual_fn = |free_vars: &[num_dual::Dual64]| -> Vec<num_dual::Dual64> {
                    let mut vars: Vec<num_dual::Dual64> =
                        all_vars_snap.iter().map(|&v| num_dual::Dual64::from(v)).collect();
                    for (j, &idx) in free_indices_snap.iter().enumerate() {
                        vars[idx] = free_vars[j];
                    }
                    for (unit_idx, var_idxs, values) in &self.feeds {
                        let offset = self.var_offset(*unit_idx);
                        for (vi, val) in var_idxs.iter().zip(values.iter()) {
                            vars[offset + vi] = num_dual::Dual64::from(*val);
                        }
                    }
                    self.residuals_dual(&vars)
                };
                compute_jacobian(dual_fn, &x)
            };

            #[cfg(feature = "autodiff")]
            let jac_entries = jacobian.entries;

            #[cfg(not(feature = "autodiff"))]
            let jac_entries = {
                let free_indices_snap = free_indices.clone();
                let all_vars_snap = all_vars.clone();
                let feeds_snap: Vec<(usize, Vec<usize>, Vec<f64>)> =
                    self.feeds.iter().map(|(u, vi, vals)| (*u, vi.clone(), vals.clone())).collect();
                let residual_fn_fd = |free_vars: &[f64]| -> Vec<f64> {
                    let mut vars = all_vars_snap.clone();
                    for (j, &idx) in free_indices_snap.iter().enumerate() {
                        vars[idx] = free_vars[j];
                    }
                    for (unit_idx, var_idxs, values) in &feeds_snap {
                        let offset = self.var_offset(*unit_idx);
                        for (vi, val) in var_idxs.iter().zip(values.iter()) {
                            vars[offset + vi] = *val;
                        }
                    }
                    self.residuals(&vars)
                };
                let mut entries = vec![0.0f64; n_eqs * n_free];
                for i in 0..n_free {
                    let h = 1e-8 * (1.0 + x[i].abs());
                    let mut x_p = x.clone();
                    x_p[i] += h;
                    let f_p = residual_fn_fd(&x_p);
                    let mut x_m = x.clone();
                    x_m[i] -= h;
                    let f_m = residual_fn_fd(&x_m);
                    for j in 0..n_eqs {
                        entries[j * n_free + i] = (f_p[j] - f_m[j]) / (2.0 * h);
                    }
                }
                entries
            };

            let r_vec = DVector::from_vec(residuals);
            let j_mat = DMatrix::from_row_slice(n_eqs, n_free, &jac_entries);

            let dx = if n_eqs == n_free {
                let decomp = j_mat.lu();
                match decomp.solve(&(-&r_vec)) {
                    Some(sol) => sol,
                    None => return Err(SolverError::SingularJacobian),
                }
            } else {
                j_mat
                    .svd(true, true)
                    .solve(&(-&r_vec), 1e-12)
                    .map_err(|_| SolverError::SingularJacobian)?
            };

            for i in 0..n_free {
                x[i] += dx[i];
            }

            if iteration > 10 && norm > 1e10 {
                return Err(SolverError::Diverged);
            }
        }

        Err(SolverError::MaxIterationsExceeded)
    }

    /// Gets variables for a unit by UnitId.
    pub fn variables(&self, unit: UnitId) -> Vec<f64> {
        self.units[unit.0].1.get_variables()
    }

    /// Gets outlet stream values (flow, temperature, pressure) for a unit.
    pub fn outlet(&self, unit: UnitId) -> (f64, f64, f64) {
        self.units[unit.0].1.get_outlet()
    }

    /// Prints a human-readable summary of the flowsheet.
    pub fn describe(&self) {
        println!(
            "EOFlowsheet({} units, {} connections):",
            self.units.len(),
            self.connections.len()
        );
        for (i, (name, unit)) in self.units.iter().enumerate() {
            println!("  [{}] {}: {} vars, {} eqs", i, name, unit.n_variables(), unit.n_equations());
        }
    }
}

impl Default for EOFlowsheet {
    fn default() -> Self {
        Self::new()
    }
}

/// Solves a single boxed [`EquationModelDyn`] unit using Newton-Raphson.
///
/// Used internally by [`SMFlowsheet::solve`] to solve each unit in sequence.
fn solve_unit_dyn(
    unit: &mut dyn EquationModelDyn,
    tolerance: f64,
    max_iterations: usize,
) -> SolverResult<SolverStats> {
    let free_indices = unit.free_indices();
    let n_free = free_indices.len();
    let n_eqs = unit.n_equations();

    if n_free == 0 {
        return Err(SolverError::NoVariablesToSolve);
    }
    if n_eqs == 0 {
        return Err(SolverError::NoEquations);
    }

    let base_vars = unit.get_variables();
    let mut x: Vec<f64> = free_indices.iter().map(|&i| base_vars[i]).collect();

    for iteration in 0..max_iterations {
        let mut vars = base_vars.clone();
        for (j, &idx) in free_indices.iter().enumerate() {
            vars[idx] = x[j];
        }

        let residuals = unit.residuals(&vars);
        let norm = residuals.iter().map(|r| r * r).sum::<f64>().sqrt();

        if norm < tolerance {
            unit.set_variables(&vars);
            return Ok(SolverStats {
                iterations: iteration + 1,
                function_evals: iteration + 1,
                jacobian_evals: iteration + 1,
                final_residual: norm,
            });
        }

        // Compute Jacobian: AD when available, forward FD otherwise.
        #[cfg(feature = "autodiff")]
        let jacobian = {
            let base_vars_snap = base_vars.clone();
            let free_indices_snap = free_indices.clone();
            let dual_fn = |free_vars: &[num_dual::Dual64]| -> Vec<num_dual::Dual64> {
                let mut dual_vars: Vec<num_dual::Dual64> =
                    base_vars_snap.iter().map(|&v| num_dual::Dual64::from(v)).collect();
                for (j, &idx) in free_indices_snap.iter().enumerate() {
                    dual_vars[idx] = free_vars[j];
                }
                unit.residuals_dual(&dual_vars)
            };
            compute_jacobian(dual_fn, &x)
        };

        #[cfg(feature = "autodiff")]
        let jac_entries = jacobian.entries;

        #[cfg(not(feature = "autodiff"))]
        let jac_entries = {
            let mut entries = vec![0.0; n_eqs * n_free];
            for j in 0..n_free {
                let step = 1e-8 * (1.0 + x[j].abs());
                let mut vars_p = vars.clone();
                vars_p[free_indices[j]] += step;
                let r_p = unit.residuals(&vars_p);
                for i in 0..n_eqs {
                    entries[i * n_free + j] = (r_p[i] - residuals[i]) / step;
                }
            }
            entries
        };

        let r_vec = DVector::from_vec(residuals);
        let j_mat = DMatrix::from_row_slice(n_eqs, n_free, &jac_entries);

        let dx = if n_eqs == n_free {
            match j_mat.lu().solve(&(-&r_vec)) {
                Some(sol) => sol,
                None => return Err(SolverError::SingularJacobian),
            }
        } else {
            j_mat
                .svd(true, true)
                .solve(&(-&r_vec), 1e-12)
                .map_err(|_| SolverError::SingularJacobian)?
        };

        for i in 0..n_free {
            x[i] += dx[i];
        }

        if iteration > 10 && norm > 1e10 {
            return Err(SolverError::Diverged);
        }
    }

    Err(SolverError::MaxIterationsExceeded)
}

/// Sequential-Modular (SM) flowsheet solver.
///
/// Solves units one at a time in topological order. Each unit is solved
/// independently using Newton-Raphson; outlet stream values are then
/// propagated to the inlets of downstream units before they are solved.
pub struct SMFlowsheet {
    /// Units in execution order, with their names.
    units: Vec<(String, Box<dyn EquationModelDyn>)>,
    /// Propagation connections: (src_unit, src_vars, dst_unit, dst_vars).
    connections: Vec<(usize, Vec<usize>, usize, Vec<usize>)>,
}

impl SMFlowsheet {
    /// Creates a new empty SMFlowsheet.
    pub fn new() -> Self {
        SMFlowsheet { units: Vec::new(), connections: Vec::new() }
    }

    /// Adds a unit in sequential execution order. Returns its [`UnitId`].
    pub fn add<M: EquationModel + 'static>(&mut self, name: &str, unit: M) -> UnitId {
        let id = UnitId(self.units.len());
        self.units.push((name.to_string(), Box::new(unit)));
        id
    }

    /// Adds a propagation connection between units.
    pub fn connect(
        &mut self,
        src: UnitId,
        src_vars: Vec<usize>,
        dst: UnitId,
        dst_vars: Vec<usize>,
    ) -> ConnectionId {
        let id = ConnectionId(self.connections.len());
        self.connections.push((src.0, src_vars, dst.0, dst_vars));
        id
    }

    /// Solves all units in sequence until convergence, propagating stream values.
    ///
    /// Returns aggregated [`SMSolverStats`] covering all units.
    pub fn solve(&mut self, tolerance: f64, max_iterations: usize) -> SolverResult<SMSolverStats> {
        let mut total_iterations = 0;
        let mut max_residual = 0.0f64;

        // Propagate connections before each unit solve.
        let n_units = self.units.len();
        for unit_idx in 0..n_units {
            // Propagate inlet values from upstream units.
            for (src_idx, src_vars, dst_idx, dst_vars) in &self.connections {
                if *dst_idx == unit_idx {
                    let src_vals: Vec<f64> = {
                        let src_unit = &self.units[*src_idx].1;
                        let src_full = src_unit.get_variables();
                        src_vars.iter().map(|&vi| src_full[vi]).collect()
                    };
                    let dst_unit = &mut self.units[*dst_idx].1;
                    let mut dst_full = dst_unit.get_variables();
                    for (&di, val) in dst_vars.iter().zip(src_vals.iter()) {
                        dst_full[di] = *val;
                    }
                    dst_unit.set_variables(&dst_full);
                }
            }

            // Solve this unit.
            let stats = solve_unit_dyn(self.units[unit_idx].1.as_mut(), tolerance, max_iterations)?;
            total_iterations += stats.iterations;
            if stats.final_residual > max_residual {
                max_residual = stats.final_residual;
            }
        }

        Ok(SMSolverStats {
            total_iterations,
            units_solved: n_units,
            max_unit_residual: max_residual,
        })
    }
}

impl Default for SMFlowsheet {
    fn default() -> Self {
        Self::new()
    }
}

pub use recycle::RecycleSolver;

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
    fn test_solver_error_display() {
        let err = SolverError::NoVariablesToSolve;
        assert_eq!(
            err.to_string(),
            "No variables to solve. Did you forget to call harvest_equations()?"
        );

        let err = SolverError::InvalidTearStream("test message".to_string());
        assert_eq!(err.to_string(), "Invalid tear stream: test message");
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
