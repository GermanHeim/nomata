//! Tear-stream solver for flowsheets with recycle loops.
//!
//! This module provides iterative convergence methods for solving flowsheets
//! containing recycle streams (cyclic topology). The solver iterates on
//! "tear streams" (user-specified streams that break the cycle) until the
//! guessed and computed values converge.
//!
//! # Convergence Methods
//!
//! - **Wegstein** (default): Fixed-point acceleration method. Fast and robust
//!   for most problems. Recommended starting point.
//! - **Newton-Raphson**: Uses finite differences (or autodiff) for Jacobian.
//!   Faster convergence near solution, requires `solvers` feature.
//! - **Broyden**: Quasi-Newton method, approximates Jacobian iteratively.
//!   Good balance between Wegstein and Newton.
//!
//! # Sequential-Modular Philosophy
//!
//! This solver is designed for **Sequential-Modular (SM)** simulation:
//!
//! 1. The user provides a closure that computes the flowsheet given tear stream values
//! 2. Inside the closure, units are executed in sequence (topological order)
//! 3. The solver iterates until tear stream convergence
//!
//! # Example
//!
//! ```ignore
//! use nomata::solvers::recycle::{RecycleSolver, SolverMethod};
//!
//! // Define the sequential calculation
//! let compute_flowsheet = |tear: &[f64]| -> Vec<f64> {
//!     let recycle_flow = tear[0];
//!     
//!     // Execute units in sequence
//!     let mixed = fresh_feed + recycle_flow;      // Mixer
//!     let reacted = mixed * (1.0 - conversion);   // Reactor
//!     let recycle_out = reacted * split_fraction; // Separator
//!     
//!     vec![recycle_out]  // Return computed tear stream
//! };
//!
//! // Solve with Wegstein acceleration
//! let mut solver = RecycleSolver::new(100, 1e-6);
//! let solution = solver.solve(vec![0.0], compute_flowsheet)?;
//!
//! println!("Converged in {} iterations", solution.iterations);
//! println!("Recycle flow: {:.2}", solution.tear_stream_values[0]);
//! ```
//!
//! # With Automatic Differentiation
//!
//! When the `autodiff` feature is enabled, the closure should accept `&[Dual64]`:
//!
//! ```ignore
//! use num_dual::Dual64;
//!
//! let compute_flowsheet = |tear: &[Dual64]| -> Vec<Dual64> {
//!     // Same logic, but using Dual64 for automatic derivatives
//!     let recycle = tear[0];
//!     let mixed = Dual64::from(fresh_feed) + recycle;
//!     // ...
//! };
//! ```

use nalgebra::{DMatrix, DVector};

#[cfg(feature = "autodiff")]
use num_dual::*;

/// Wegstein acceleration method for recycle convergence.
///
/// Accelerates the convergence of fixed-point iterations using the
/// Wegstein method. Given g(x) where fixed point is x = g(x),
/// the method computes: x_next = q*x + (1-q)*g(x)
///
/// # Algorithm
///
/// The acceleration factor q is computed from:
/// a = (x3 - x2) / (x2 - x1)
/// q = a / (a - 1)
pub struct WegsteinAccelerator {
    /// History of x values for computing acceleration
    x_history: Vec<Vec<f64>>,
    /// History of g(x) values
    g_history: Vec<Vec<f64>>,
    /// Minimum q value for stability
    q_min: f64,
    /// Maximum q value for stability  
    q_max: f64,
}

impl WegsteinAccelerator {
    /// Creates a new Wegstein accelerator.
    ///
    /// # Arguments
    ///
    /// * `q_min` - Minimum q value (typically -5.0)
    /// * `q_max` - Maximum q value (typically 0.0 for damping)
    pub fn new(q_min: f64, q_max: f64) -> Self {
        WegsteinAccelerator { x_history: Vec::new(), g_history: Vec::new(), q_min, q_max }
    }

    /// Creates a default Wegstein accelerator.
    pub fn with_defaults() -> Self {
        Self::new(-5.0, 0.0)
    }

    /// Computes the next iteration using Wegstein acceleration.
    ///
    /// # Arguments
    ///
    /// * `x` - Current guess
    /// * `gx` - Result from function evaluation: g(x)
    ///
    /// # Returns
    ///
    /// The accelerated value for the next iteration
    pub fn accelerate(&mut self, x: &[f64], gx: &[f64]) -> Vec<f64> {
        let n = x.len();
        assert_eq!(n, gx.len(), "Vector dimensions must match");

        // Store history
        self.x_history.push(x.to_vec());
        self.g_history.push(gx.to_vec());

        // First two iterations: use direct substitution to build history
        if self.x_history.len() < 3 {
            return gx.to_vec();
        }

        let len = self.x_history.len();
        let x1 = &self.x_history[len - 3];
        let x2 = &self.x_history[len - 2];
        let x3 = x;

        let mut x_next = vec![0.0; n];

        for i in 0..n {
            // Compute a = (x3 - x2) / (x2 - x1)
            let dx = x2[i] - x1[i];
            let a = if dx.abs() > 1e-14 { (x3[i] - x2[i]) / dx } else { 0.0 };

            // Compute q = a / (a - 1)
            let q = if (a - 1.0).abs() > 1e-14 {
                a / (a - 1.0)
            } else {
                0.0 // Direct substitution if a ~ 1
            };

            // Bound q for stability
            let q_bounded = q.clamp(self.q_min, self.q_max);

            // Wegstein update: x_next = q*x + (1-q)*g(x)
            x_next[i] = q_bounded * x3[i] + (1.0 - q_bounded) * gx[i];
        }

        x_next
    }

    /// Resets the accelerator state.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.g_history.clear();
    }
}

/// Method selection for recycle convergence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverMethod {
    /// Wegstein acceleration (fixed-point, no Jacobian needed)
    /// Recommended for sequential-modular flowsheet solving
    #[default]
    Wegstein,
    /// Newton-Raphson with finite difference Jacobian
    /// Better for monolithic/tightly coupled systems
    Newton,
    /// Broyden's method (quasi-Newton, approximates Jacobian)
    /// Hybrid between Wegstein and Newton
    Broyden,
}

/// Compute Jacobian using automatic differentiation with dual numbers.
///
/// Uses num-dual for proper forward-mode AD with exact derivatives.
#[cfg(feature = "autodiff")]
fn compute_jacobian_autodiff<F>(func: &F, x: &[f64], n: usize) -> Vec<Vec<f64>>
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    // Compute columns of the Jacobian using forward-mode AD
    let columns: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            // Create dual number vector where x[j] has derivative 1.0
            let x_dual: Vec<Dual64> = x
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    if i == j {
                        Dual64::new(val, 1.0) // Differentiate with respect to x[j]
                    } else {
                        Dual64::from(val) // Other variables are constants
                    }
                })
                .collect();

            // Evaluate function with dual numbers to get derivatives
            let result = func(&x_dual);

            // Extract column j: for each row i, compute dg_i/dx_j - delta_ij
            result
                .iter()
                .enumerate()
                .map(|(i, res)| {
                    let dg_dx = res.eps;
                    // Jacobian of f = g(x) - x is J = dg/dx - I
                    if i == j { dg_dx - 1.0 } else { dg_dx }
                })
                .collect()
        })
        .collect();

    // Convert from column-major to row-major
    (0..n).map(|i| columns.iter().map(|col| col[i]).collect()).collect()
}

/// Dispatcher when autodiff feature is enabled.
#[cfg(feature = "autodiff")]
fn compute_jacobian<F>(func: &F, x: &[f64], n: usize) -> Vec<Vec<f64>>
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    compute_jacobian_autodiff(func, x, n)
}

/// Compute Jacobian using finite differences (fallback when autodiff not available).
#[cfg(not(feature = "autodiff"))]
fn compute_jacobian<F>(func: &F, x: &[f64], n: usize) -> Vec<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut jacobian = vec![vec![0.0; n]; n];
    let gx = func(x);

    for j in 0..n {
        let h = 1e-8 * (1.0 + x[j].abs());
        let mut x_perturbed = x.to_vec();
        x_perturbed[j] += h;

        let gx_perturbed = func(&x_perturbed);

        for i in 0..n {
            // Forward difference: (f(x+h) - f(x)) / h
            jacobian[i][j] = (gx_perturbed[i] - gx[i]) / h;
            if i == j {
                jacobian[i][j] -= 1.0;
            }
        }
    }

    jacobian
}

/// Linear system solver using nalgebra.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, RecycleError> {
    let n = b.len();
    let mut a_data = Vec::with_capacity(n * n);
    for row in a.iter().take(n) {
        for &val in row.iter().take(n) {
            a_data.push(val);
        }
    }

    let a_mat = DMatrix::from_row_slice(n, n, &a_data);
    let b_vec = DVector::from_vec(b.to_vec());

    match a_mat.lu().solve(&b_vec) {
        Some(x) => Ok(x.data.as_vec().clone()),
        None => {
            Err(RecycleError::ConvergenceFailure { iterations: 0, final_residual: f64::INFINITY })
        }
    }
}

/// Fallback when solvers feature is disabled.
#[cfg(not(feature = "solvers"))]
fn solve_linear_system(_a: &[Vec<f64>], _b: &[f64]) -> Result<Vec<f64>, RecycleError> {
    Err(RecycleError::FeatureNotEnabled)
}

/// Newton-Raphson solver implementation.
struct NewtonSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl NewtonSolver {
    fn new(max_iterations: usize, tolerance: f64) -> Self {
        NewtonSolver { max_iterations, tolerance }
    }

    /// Solve with autodiff support
    #[cfg(feature = "autodiff")]
    pub fn solve<F>(&mut self, x: Vec<f64>, func: F) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[Dual64]) -> Vec<Dual64>,
    {
        let n = x.len();
        let mut iteration_history = Vec::new();
        let mut x = x;

        for iteration in 0..self.max_iterations {
            // Evaluate function: f(x) = g(x) - x
            // Convert x to Dual64 with zero derivative for function evaluation
            let x_dual: Vec<Dual64> = x.iter().map(|&v| Dual64::from(v)).collect();
            let gx_dual = func(&x_dual);

            // Extract real parts
            let mut fx = vec![0.0; n];
            for i in 0..n {
                fx[i] = gx_dual[i].re - x[i];
            }

            // Compute residual norm
            let residual_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();

            iteration_history.push(IterationInfo {
                iteration,
                residual_norm,
                tear_stream_values: x.clone(),
            });

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(RecycleSolution {
                    converged: true,
                    iterations: iteration + 1,
                    final_residual: residual_norm,
                    tear_stream_values: x,
                    iteration_history,
                });
            }

            // Compute Jacobian using autodiff
            let jacobian = compute_jacobian(&func, &x, n);

            // Solve J * dx = -f for dx
            let neg_fx = fx.iter().map(|&v| -v).collect::<Vec<f64>>();

            let dx = match solve_linear_system(&jacobian, &neg_fx) {
                Ok(dx) => dx,
                Err(_) => {
                    return Err(RecycleError::ConvergenceFailure {
                        iterations: iteration + 1,
                        final_residual: residual_norm,
                    });
                }
            };

            // Update: x_new = x_old + dx
            for i in 0..n {
                x[i] += dx[i];
            }
        }

        Err(RecycleError::ConvergenceFailure {
            iterations: self.max_iterations,
            final_residual: iteration_history
                .last()
                .expect("Recycle solver must run at least one iteration")
                .residual_norm,
        })
    }

    /// Solve without autodiff
    #[cfg(not(feature = "autodiff"))]
    pub fn solve<F>(&mut self, x: Vec<f64>, func: F) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        let mut iteration_history = Vec::new();
        let mut x = x;

        for iteration in 0..self.max_iterations {
            // Evaluate function: f(x) = g(x) - x
            let gx = func(&x);
            let mut fx = vec![0.0; n];
            for i in 0..n {
                fx[i] = gx[i] - x[i];
            }

            // Compute residual norm
            let residual_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();

            iteration_history.push(IterationInfo {
                iteration,
                residual_norm,
                tear_stream_values: x.clone(),
            });

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(RecycleSolution {
                    converged: true,
                    iterations: iteration + 1,
                    final_residual: residual_norm,
                    tear_stream_values: x,
                    iteration_history,
                });
            }

            // Compute Jacobian using finite differences
            let jacobian = compute_jacobian(&func, &x, n);

            // Solve J * dx = -f for dx
            let neg_fx = fx.iter().map(|&v| -v).collect::<Vec<f64>>();

            let dx = match solve_linear_system(&jacobian, &neg_fx) {
                Ok(dx) => dx,
                Err(_) => {
                    return Err(RecycleError::ConvergenceFailure {
                        iterations: iteration + 1,
                        final_residual: residual_norm,
                    });
                }
            };

            // Update: x_new = x_old + dx
            for i in 0..n {
                x[i] += dx[i];
            }
        }

        Err(RecycleError::ConvergenceFailure {
            iterations: self.max_iterations,
            final_residual: iteration_history
                .last()
                .expect("Recycle solver must run at least one iteration")
                .residual_norm,
        })
    }
}

/// Broyden's method implementation.
struct BroydenSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl BroydenSolver {
    fn new(max_iterations: usize, tolerance: f64) -> Self {
        BroydenSolver { max_iterations, tolerance }
    }

    /// Solve with autodiff support (Broyden doesn't use autodiff but signature must match)
    #[cfg(feature = "autodiff")]
    pub fn solve<F>(&mut self, x: Vec<f64>, func: F) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[Dual64]) -> Vec<Dual64>,
    {
        let n = x.len();
        let mut iteration_history = Vec::new();
        let mut x = x;

        // Initialize inverse Jacobian approximation as identity
        let mut b_inv = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect::<Vec<Vec<f64>>>();

        // Evaluate initial function
        let x_dual: Vec<Dual64> = x.iter().map(|&v| Dual64::from(v)).collect();
        let gx_dual = func(&x_dual);
        let mut fx = vec![0.0; n];
        for i in 0..n {
            fx[i] = gx_dual[i].re - x[i];
        }

        for iteration in 0..self.max_iterations {
            // Compute residual norm
            let residual_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();

            iteration_history.push(IterationInfo {
                iteration,
                residual_norm,
                tear_stream_values: x.clone(),
            });

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(RecycleSolution {
                    converged: true,
                    iterations: iteration + 1,
                    final_residual: residual_norm,
                    tear_stream_values: x,
                    iteration_history,
                });
            }

            // Compute step: dx = -B^{-1} * f
            let mut dx = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    dx[i] -= b_inv[i][j] * fx[j];
                }
            }

            // Update x
            for i in 0..n {
                x[i] += dx[i];
            }

            // Evaluate new function (autodiff version)
            let x_dual: Vec<Dual64> = x.iter().map(|&v| Dual64::from(v)).collect();
            let gx_new_dual = func(&x_dual);
            let mut fx_new = vec![0.0; n];
            for i in 0..n {
                fx_new[i] = gx_new_dual[i].re - x[i];
            }

            // Compute df = f_new - f_old
            let mut df = vec![0.0; n];
            for i in 0..n {
                df[i] = fx_new[i] - fx[i];
            }

            // Broyden update: B^{-1} += (dx - B^{-1}*df) * dx^T * B^{-1} / (dx^T * B^{-1} * df)
            let mut b_inv_df = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    b_inv_df[i] += b_inv[i][j] * df[j];
                }
            }

            let mut denominator = 0.0;
            for i in 0..n {
                denominator += dx[i] * b_inv_df[i];
            }

            if denominator.abs() > 1e-14 {
                for i in 0..n {
                    for j in 0..n {
                        b_inv[i][j] += (dx[i] - b_inv_df[i]) * dx[j] / denominator;
                    }
                }
            }

            fx = fx_new;
        }

        Err(RecycleError::ConvergenceFailure {
            iterations: self.max_iterations,
            final_residual: iteration_history
                .last()
                .expect("Recycle solver must run at least one iteration")
                .residual_norm,
        })
    }

    /// Solve without autodiff
    #[cfg(not(feature = "autodiff"))]
    pub fn solve<F>(&mut self, x: Vec<f64>, func: F) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        let mut iteration_history = Vec::new();
        let mut x = x;

        // Initialize inverse Jacobian approximation as identity
        let mut b_inv = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect::<Vec<Vec<f64>>>();

        // Evaluate initial function
        let gx = func(&x);
        let mut fx = vec![0.0; n];
        for i in 0..n {
            fx[i] = gx[i] - x[i];
        }

        for iteration in 0..self.max_iterations {
            // Compute residual norm
            let residual_norm = fx.iter().map(|v| v * v).sum::<f64>().sqrt();

            iteration_history.push(IterationInfo {
                iteration,
                residual_norm,
                tear_stream_values: x.clone(),
            });

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(RecycleSolution {
                    converged: true,
                    iterations: iteration + 1,
                    final_residual: residual_norm,
                    tear_stream_values: x,
                    iteration_history,
                });
            }

            // Compute step: dx = -B^{-1} * f
            let mut dx = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    dx[i] -= b_inv[i][j] * fx[j];
                }
            }

            // Update x
            for i in 0..n {
                x[i] += dx[i];
            }

            // Evaluate new function
            let gx_new = func(&x);
            let mut fx_new = vec![0.0; n];
            for i in 0..n {
                fx_new[i] = gx_new[i] - x[i];
            }

            // Compute df = f_new - f_old
            let mut df = vec![0.0; n];
            for i in 0..n {
                df[i] = fx_new[i] - fx[i];
            }

            // Broyden update: B^{-1} += (dx - B^{-1}*df) * dx^T * B^{-1} / (dx^T * B^{-1} * df)
            let mut b_inv_df = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    b_inv_df[i] += b_inv[i][j] * df[j];
                }
            }

            let mut denominator = 0.0;
            for i in 0..n {
                denominator += dx[i] * b_inv_df[i];
            }

            if denominator.abs() > 1e-14 {
                for i in 0..n {
                    for j in 0..n {
                        b_inv[i][j] += (dx[i] - b_inv_df[i]) * dx[j] / denominator;
                    }
                }
            }

            fx = fx_new;
        }

        Err(RecycleError::ConvergenceFailure {
            iterations: self.max_iterations,
            final_residual: iteration_history
                .last()
                .expect("Recycle solver must run at least one iteration")
                .residual_norm,
        })
    }
}

/// Recycle solver for flowsheets with cyclic topology.
///
/// Supports multiple solution methods:
/// - **Wegstein** (default): Sequential-modular, no Jacobian
/// - **Newton-Raphson**: Monolithic, finite difference Jacobian
/// - **Broyden**: Hybrid quasi-Newton approach
pub struct RecycleSolver {
    /// Solution method
    method: SolverMethod,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Wegstein accelerator (used only for Wegstein method)
    accelerator: WegsteinAccelerator,
}

impl RecycleSolver {
    /// Creates a new recycle solver with Wegstein method (default).
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        RecycleSolver {
            method: SolverMethod::Wegstein,
            max_iterations,
            tolerance,
            accelerator: WegsteinAccelerator::with_defaults(),
        }
    }

    /// Creates a recycle solver with specified method.
    ///
    /// # Arguments
    ///
    /// * `method` - Solution method (Wegstein, Newton, or Broyden)
    /// * `max_iterations` - Maximum iterations
    /// * `tolerance` - Convergence tolerance
    pub fn with_method(method: SolverMethod, max_iterations: usize, tolerance: f64) -> Self {
        RecycleSolver {
            method,
            max_iterations,
            tolerance,
            accelerator: WegsteinAccelerator::with_defaults(),
        }
    }

    /// Creates a recycle solver with custom Wegstein parameters.
    pub fn with_wegstein(max_iterations: usize, tolerance: f64, q_min: f64, q_max: f64) -> Self {
        RecycleSolver {
            method: SolverMethod::Wegstein,
            max_iterations,
            tolerance,
            accelerator: WegsteinAccelerator::new(q_min, q_max),
        }
    }

    /// Solves a flowsheet with recycle streams using the selected method.
    ///
    /// # Arguments
    ///
    /// * `tear_stream_guess` - Initial guess for tear stream variables
    /// * `compute_flowsheet` - Function that computes flowsheet given tear stream values.
    ///   Must accept `&[Dual64]` and work generically over numeric types
    ///
    /// # Returns
    ///
    /// Converged tear stream values
    #[cfg(feature = "autodiff")]
    pub fn solve<F>(
        &mut self,
        tear_stream_guess: Vec<f64>,
        compute_flowsheet: F,
    ) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[Dual64]) -> Vec<Dual64>,
    {
        match self.method {
            SolverMethod::Wegstein => {
                // Wegstein needs f64 version, so we wrap the dual function
                let f64_func = |x: &[f64]| -> Vec<f64> {
                    let x_dual: Vec<Dual64> = x.iter().map(|&v| Dual64::from(v)).collect();
                    let result = compute_flowsheet(&x_dual);
                    result.iter().map(|d| d.re).collect()
                };
                self.solve_wegstein(tear_stream_guess, f64_func)
            }
            SolverMethod::Newton => {
                let mut newton = NewtonSolver::new(self.max_iterations, self.tolerance);
                newton.solve(tear_stream_guess, compute_flowsheet)
            }
            SolverMethod::Broyden => {
                let mut broyden = BroydenSolver::new(self.max_iterations, self.tolerance);
                broyden.solve(tear_stream_guess, compute_flowsheet)
            }
        }
    }

    /// Solves a flowsheet with recycle streams (without autodiff).
    ///
    /// # Arguments
    ///
    /// * `tear_stream_guess` - Initial guess for tear stream variables
    /// * `compute_flowsheet` - Function that computes flowsheet given tear stream values
    ///
    /// # Returns
    ///
    /// Converged tear stream values
    #[cfg(not(feature = "autodiff"))]
    pub fn solve<F>(
        &mut self,
        tear_stream_guess: Vec<f64>,
        compute_flowsheet: F,
    ) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        match self.method {
            SolverMethod::Wegstein => self.solve_wegstein(tear_stream_guess, compute_flowsheet),
            SolverMethod::Newton => {
                let mut newton = NewtonSolver::new(self.max_iterations, self.tolerance);
                newton.solve(tear_stream_guess, compute_flowsheet)
            }
            SolverMethod::Broyden => {
                let mut broyden = BroydenSolver::new(self.max_iterations, self.tolerance);
                broyden.solve(tear_stream_guess, compute_flowsheet)
            }
        }
    }

    fn solve_wegstein<F>(
        &mut self,
        mut tear_stream_guess: Vec<f64>,
        compute_flowsheet: F,
    ) -> Result<RecycleSolution, RecycleError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        self.accelerator.reset();

        let mut iteration_history = Vec::new();

        for iteration in 0..self.max_iterations {
            // Evaluate flowsheet with current guess
            let tear_stream_computed = compute_flowsheet(&tear_stream_guess);

            // Compute residual
            let residual: Vec<f64> = tear_stream_guess
                .iter()
                .zip(tear_stream_computed.iter())
                .map(|(guess, computed)| computed - guess)
                .collect();

            let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();

            iteration_history.push(IterationInfo {
                iteration,
                residual_norm,
                tear_stream_values: tear_stream_guess.clone(),
            });

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(RecycleSolution {
                    converged: true,
                    iterations: iteration + 1,
                    final_residual: residual_norm,
                    tear_stream_values: tear_stream_guess,
                    iteration_history,
                });
            }

            // Apply Wegstein acceleration
            tear_stream_guess =
                self.accelerator.accelerate(&tear_stream_guess, &tear_stream_computed);
        }

        // Did not converge
        Err(RecycleError::ConvergenceFailure {
            iterations: self.max_iterations,
            final_residual: iteration_history
                .last()
                .expect("Recycle solver must run at least one iteration")
                .residual_norm,
        })
    }
}

/// Solution from recycle solver.
#[derive(Debug)]
pub struct RecycleSolution {
    /// Whether the solver converged
    pub converged: bool,
    /// Number of iterations required
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: f64,
    /// Converged tear stream values
    pub tear_stream_values: Vec<f64>,
    /// History of all iterations
    pub iteration_history: Vec<IterationInfo>,
}

/// Information about a single iteration.
#[derive(Debug, Clone)]
pub struct IterationInfo {
    /// Iteration number
    pub iteration: usize,
    /// Residual norm at this iteration
    pub residual_norm: f64,
    /// Tear stream values at this iteration
    pub tear_stream_values: Vec<f64>,
}

/// Errors that can occur during recycle solving.
#[derive(Debug)]
pub enum RecycleError {
    /// Solver failed to converge within maximum iterations
    ConvergenceFailure { iterations: usize, final_residual: f64 },
    /// Invalid tear stream specification
    InvalidTearStream(String),
    /// Required feature is not enabled
    FeatureNotEnabled,
}

impl std::fmt::Display for RecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecycleError::ConvergenceFailure { iterations, final_residual } => {
                write!(
                    f,
                    "Recycle solver failed to converge after {} iterations (residual: {:.6e})",
                    iterations, final_residual
                )
            }
            RecycleError::InvalidTearStream(msg) => write!(f, "Invalid tear stream: {}", msg),
            RecycleError::FeatureNotEnabled => {
                write!(f, "Newton/Broyden methods require the 'solvers' feature to be enabled")
            }
        }
    }
}

impl std::error::Error for RecycleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wegstein_simple() {
        let mut accel = WegsteinAccelerator::new(-5.0, 0.5);

        // Simple iteration: x = 0.5*x + 2
        // Fixed point: x = 4
        let x_old = vec![0.0];
        let x_new = vec![2.0]; // f(0) = 2

        let x_next = accel.accelerate(&x_old, &x_new);

        // First iteration should be direct substitution
        assert_eq!(x_next[0], 2.0);
    }

    #[test]
    fn test_wegstein_convergence() {
        let mut accel = WegsteinAccelerator::new(-5.0, 0.5);

        let mut x = vec![0.0];

        // Iteration: x = 0.5*x + 2, fixed point at x=4
        // Use damped substitution for stability
        for iter in 0..50 {
            let x_new = vec![0.5 * x[0] + 2.0];
            x = accel.accelerate(&x, &x_new);
            if (x[0] - 4.0).abs() < 1e-6 {
                println!("Converged at iteration {}: x = {:.6}", iter, x[0]);
                break;
            }
        }

        // Should converge eventually with damping
        assert!((x[0] - 4.0).abs() < 1e-3, "x = {}, expected ~4.0", x[0]);
    }

    #[test]
    #[cfg(not(feature = "autodiff"))]
    fn test_recycle_solver() {
        let mut solver = RecycleSolver::new(50, 1e-6);

        // Simple recycle: x = 0.9*x + 1
        // Fixed point: x = 10
        let result = solver.solve(vec![0.0], |x| vec![x[0] * 0.9 + 1.0]);

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);
        assert!((solution.tear_stream_values[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(feature = "autodiff")]
    fn test_recycle_solver() {
        use num_dual::Dual64;

        let mut solver = RecycleSolver::new(50, 1e-6);

        // Simple recycle: x = 0.9*x + 1
        // Fixed point: x = 10
        let result = solver
            .solve(vec![0.0], |x: &[Dual64]| vec![Dual64::from(0.9) * x[0] + Dual64::from(1.0)]);

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);
        assert!((solution.tear_stream_values[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    #[cfg(not(feature = "autodiff"))]
    fn test_recycle_solver_multivariable() {
        let mut solver = RecycleSolver::new(100, 1e-6);

        // Two-variable system:
        // x1 = 0.8*x1 + 0.1*x2 + 1
        // x2 = 0.2*x1 + 0.7*x2 + 2
        let result = solver.solve(vec![0.0, 0.0], |x| {
            vec![0.8 * x[0] + 0.1 * x[1] + 1.0, 0.2 * x[0] + 0.7 * x[1] + 2.0]
        });

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);

        // Fixed point solution:
        // x1 = 0.8*x1 + 0.1*x2 + 1  =>  0.2*x1 - 0.1*x2 = 1
        // x2 = 0.2*x1 + 0.7*x2 + 2  =>  -0.2*x1 + 0.3*x2 = 2
        // Solving: x1 = 12.5 , x2 = 15.0
        assert!((solution.tear_stream_values[0] - 12.5).abs() < 0.01);
        assert!((solution.tear_stream_values[1] - 15.0).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "autodiff")]
    fn test_recycle_solver_multivariable() {
        use num_dual::Dual64;

        let mut solver = RecycleSolver::new(100, 1e-6);

        // Two-variable system:
        // x1 = 0.8*x1 + 0.1*x2 + 1
        // x2 = 0.2*x1 + 0.7*x2 + 2
        let result = solver.solve(vec![0.0, 0.0], |x: &[Dual64]| {
            let c08 = Dual64::from(0.8);
            let c01 = Dual64::from(0.1);
            let c02 = Dual64::from(0.2);
            let c07 = Dual64::from(0.7);
            let c10 = Dual64::from(1.0);
            let c20 = Dual64::from(2.0);
            vec![c08 * x[0] + c01 * x[1] + c10, c02 * x[0] + c07 * x[1] + c20]
        });

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);

        // Fixed point solution: x1 = 12.5 , x2 = 15.0
        assert!((solution.tear_stream_values[0] - 12.5).abs() < 0.01);
        assert!((solution.tear_stream_values[1] - 15.0).abs() < 0.01);
    }

    #[test]
    #[cfg(not(feature = "autodiff"))]
    fn test_recycle_solver_failure() {
        let mut solver = RecycleSolver::new(5, 1e-10);

        // Very strict tolerance - should fail
        let result = solver.solve(vec![0.0], |x| vec![x[0] * 0.9 + 1.0]);

        assert!(result.is_err());
    }

    #[test]
    fn test_newton_method() {
        let mut solver = RecycleSolver::with_method(SolverMethod::Newton, 50, 1e-6);

        // Simple recycle: x = 0.9*x + 1
        // Fixed point: x = 10
        let result = solver.solve(vec![0.0], |x| vec![x[0] * 0.9 + 1.0]);

        if let Err(ref e) = result {
            eprintln!("Newton method failed: {:?}", e);
        }

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);
        assert!((solution.tear_stream_values[0] - 10.0).abs() < 1e-5);

        // Newton should converge faster than Wegstein
        println!("Newton converged in {} iterations", solution.iterations);
        assert!(solution.iterations < 15, "Newton took {} iterations", solution.iterations);
    }

    #[test]
    fn test_broyden_method() {
        let mut solver = RecycleSolver::with_method(SolverMethod::Broyden, 50, 1e-6);

        // Two-variable system
        let result = solver.solve(vec![0.0, 0.0], |x| {
            vec![x[0] * 0.8 + x[1] * 0.1 + 1.0, x[0] * 0.2 + x[1] * 0.7 + 2.0]
        });

        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.converged);
        assert!((solution.tear_stream_values[0] - 12.5).abs() < 0.01);
        assert!((solution.tear_stream_values[1] - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_solver_method_selection() {
        // Test that we can create solvers with different methods
        let _wegstein = RecycleSolver::new(100, 1e-6);
        let _newton = RecycleSolver::with_method(SolverMethod::Newton, 100, 1e-6);
        let _broyden = RecycleSolver::with_method(SolverMethod::Broyden, 100, 1e-6);

        // Verify default
        assert_eq!(SolverMethod::default(), SolverMethod::Wegstein);
    }
}
