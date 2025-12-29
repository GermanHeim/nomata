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

/// Computes the Jacobian matrix for an equation system using central numerical differentiation.
///
/// This function uses central finite differences: (f(x+h) - f(x-h)) / (2h).
/// While it requires two evaluations per variable, it provides second-order accuracy O(h^2),
/// which is more stable for complex chemical process models.
///
/// # Arguments
///
/// * `system` - The equation system to differentiate
/// * `var_names` - The ordered list of variable names corresponding to the state vector
/// * `state` - The state vector at which to evaluate the Jacobian
///
/// # Returns
///
/// A `Jacobian` struct containing the partial derivatives.
pub fn compute_jacobian_numerical<T: crate::TimeDomain>(
    system: &crate::EquationSystem<T>,
    var_names: &[String],
    state: &[f64],
) -> Jacobian {
    use std::collections::HashMap;

    let n_eqs = system.total_equations();
    let n_vars = state.len();
    let base_eps: f64 = 1e-8;

    let mut jacobian = Jacobian::zeros(n_eqs, n_vars);

    // Build state HashMap helper
    let build_state = |values: &[f64]| -> HashMap<String, f64> {
        var_names.iter().cloned().zip(values.iter().copied()).collect()
    };

    // Helper to evaluate all residuals at a given point
    let evaluate_all = |vals: &[f64]| -> Vec<f64> {
        let state_map = build_state(vals);
        system
            .differential_equations()
            .iter()
            .chain(system.algebraic_equations().iter())
            .map(|eq| eq.evaluate(&state_map))
            .collect()
    };

    // For each variable, compute partialF/partialx_i using central differences
    for i in 0..n_vars {
        // Scale epsilon relative to variable magnitude
        let h = base_eps.max(base_eps * state[i].abs());

        // Evaluation at x + h
        let mut state_plus = state.to_vec();
        state_plus[i] += h;
        let residuals_plus = evaluate_all(&state_plus);

        // Evaluation at x - h
        let mut state_minus = state.to_vec();
        state_minus[i] -= h;
        let residuals_minus = evaluate_all(&state_minus);

        for j in 0..n_eqs {
            // Central difference formula: (f(x+h) - f(x-h)) / (2h)
            let deriv = (residuals_plus[j] - residuals_minus[j]) / (2.0 * h);
            jacobian.set(j, i, deriv);
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
