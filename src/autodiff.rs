//! Automatic differentiation support using `num-dual`.
//!
//! This module provides integration with the `num-dual` crate for automatic
//! differentiation, enabling Jacobian computation for DAE systems.
//!
//! # Example
//!
//! ```ignore
//! use nomata::autodiff::*;
//! use num_dual::Dual64;
//!
//! // Compute Jacobian of f(x,y) = [x^2, xy]
//! let f = |vars: &[Dual64]| vec![vars[0] * vars[0], vars[0] * vars[1]];
//! let jac = compute_jacobian(f, &[2.0, 3.0]);
//! ```

use num_dual::*;

/// Jacobian matrix representation for a system of equations.
///
/// This represents partialF/partialx where F is a vector of residual functions
/// and x is a vector of variables.
pub struct Jacobian {
    /// Number of equations (rows)
    pub n_equations: usize,
    /// Number of variables (columns)
    pub n_variables: usize,
    /// Jacobian entries in row-major order
    pub entries: Vec<f64>,
}

impl Jacobian {
    /// Creates a new Jacobian matrix with zeros.
    pub fn zeros(n_equations: usize, n_variables: usize) -> Self {
        Jacobian { n_equations, n_variables, entries: vec![0.0; n_equations * n_variables] }
    }

    /// Gets the element at position (i, j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries[i * self.n_variables + j]
    }

    /// Sets the element at position (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.entries[i * self.n_variables + j] = value;
    }

    /// Returns true if the Jacobian is square.
    pub fn is_square(&self) -> bool {
        self.n_equations == self.n_variables
    }
}

/// Computes the Jacobian of a residual function using forward-mode automatic differentiation.
///
/// # Arguments
///
/// * `f` - The residual function taking a slice of variables and returning a Vec of residuals
/// * `x` - The point at which to evaluate the Jacobian
///
/// # Returns
///
/// A Jacobian matrix containing all partial derivatives.
pub fn compute_jacobian<F>(f: F, x: &[f64]) -> Jacobian
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let n_vars = x.len();
    let mut n_eqs = 0;
    let mut columns = Vec::new();

    // For each variable, compute derivatives
    for j in 0..n_vars {
        // Create dual number array with one independent variable
        let mut x_dual: Vec<Dual64> = x.iter().map(|&v| Dual64::from(v)).collect();
        x_dual[j] = Dual64::from(x[j]).derivative();

        // Evaluate function
        let residuals = f(&x_dual);

        if j == 0 {
            n_eqs = residuals.len();
        }

        // Extract derivatives (column j of Jacobian)
        let column: Vec<f64> = residuals.iter().map(|r| r.eps).collect();
        columns.push(column);
    }

    // Convert column-major to row-major
    let mut entries = vec![0.0; n_eqs * n_vars];
    for i in 0..n_eqs {
        for j in 0..n_vars {
            entries[i * n_vars + j] = columns[j][i];
        }
    }

    Jacobian { n_equations: n_eqs, n_variables: n_vars, entries }
}

/// Computes the Jacobian matrix for an equation system using numerical differentiation.
///
/// Since the equation system uses closures that operate on `HashMap<String, f64>`,
/// we cannot directly apply automatic differentiation. This function uses finite
/// differences to approximate the Jacobian.
///
/// # Arguments
///
/// * `system` - The equation system to differentiate
/// * `var_names` - The ordered list of variable names corresponding to the state vector
/// * `state` - The state vector at which to evaluate the Jacobian
///
/// # Returns
///
/// Dense Jacobian matrix ∂F/∂x as `Vec<Vec<f64>>`
///
/// # Note
///
/// For true automatic differentiation with closures, consider writing the equations
/// as functions of `Dual64` and using `compute_jacobian` directly.
pub fn compute_jacobian_numerical<T: crate::TimeDomain>(
    system: &crate::EquationSystem<T>,
    var_names: &[String],
    state: &[f64],
) -> Vec<Vec<f64>> {
    use std::collections::HashMap;

    let n_eqs = system.total_equations();
    let n_vars = state.len();
    let eps = 1e-8; // Finite difference step size

    let mut jacobian = vec![vec![0.0; n_vars]; n_eqs];

    // Build base state HashMap
    let build_state = |values: &[f64]| -> HashMap<String, f64> {
        var_names.iter().cloned().zip(values.iter().copied()).collect()
    };

    // Evaluate residuals at base state
    let base_state = build_state(state);
    let base_residuals: Vec<f64> = system
        .differential_equations()
        .iter()
        .chain(system.algebraic_equations().iter())
        .map(|eq| eq.evaluate(&base_state))
        .collect();

    // For each variable, compute ∂F/∂x_i using central differences
    for i in 0..n_vars {
        let mut state_plus = state.to_vec();
        state_plus[i] += eps;
        let state_plus_map = build_state(&state_plus);

        let residuals_plus: Vec<f64> = system
            .differential_equations()
            .iter()
            .chain(system.algebraic_equations().iter())
            .map(|eq| eq.evaluate(&state_plus_map))
            .collect();

        for j in 0..n_eqs {
            jacobian[j][i] = (residuals_plus[j] - base_residuals[j]) / eps;
        }
    }

    jacobian
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobian_creation() {
        let jac = Jacobian::zeros(3, 2);
        assert_eq!(jac.n_equations, 3);
        assert_eq!(jac.n_variables, 2);
        assert_eq!(jac.entries.len(), 6);
    }

    #[test]
    fn test_jacobian_access() {
        let mut jac = Jacobian::zeros(2, 2);
        jac.set(0, 1, 5.0);
        assert_eq!(jac.get(0, 1), 5.0);
    }

    #[test]
    fn test_compute_jacobian_simple() {
        // Test function: f(x, y) = [x^2, xy]
        let f = |vars: &[Dual64]| {
            let x = vars[0];
            let y = vars[1];
            vec![x * x, x * y]
        };

        let point = vec![2.0, 3.0];
        let jac = compute_jacobian(f, &point);

        // Jacobian should be:
        // [[2x,   0  ]   [[4, 0]
        //  [ y,   x  ]] =  [3, 2]]
        assert_eq!(jac.n_equations, 2);
        assert_eq!(jac.n_variables, 2);
        assert!((jac.get(0, 0) - 4.0).abs() < 1e-10); // partial(x^2)/partialx = 2x = 4
        assert!((jac.get(0, 1) - 0.0).abs() < 1e-10); // partial(x^2)/partialy = 0
        assert!((jac.get(1, 0) - 3.0).abs() < 1e-10); // partial(xy)/partialx = y = 3
        assert!((jac.get(1, 1) - 2.0).abs() < 1e-10); // partial(xy)/partialy = x = 2
    }
}
