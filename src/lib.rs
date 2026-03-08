#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use thiserror::Error;

// Core modules
pub mod models;

// Optional feature modules
#[cfg(feature = "autodiff")]
pub mod autodiff;

#[cfg(feature = "solvers")]
pub mod solvers;

#[cfg(feature = "thermodynamics")]
pub mod thermodynamics;

#[cfg(feature = "thermodynamics")]
pub use thermodynamics::{
    Substance,
    fluids::{PredefinedMix, Pure},
};

/// Trait for processing a unit operation with typed inputs and outputs.
///
/// This is the primary interface for running unit operations.
/// Each unit defines its own input and output types, ensuring
/// type-safe connections at compile time.
///
/// # Examples
///
/// Single inlet/outlet (Pump, Heater, Compressor):
/// ```ignore
/// impl Process for Pump {
///     type Input = Stream<MassFlow>;
///     type Output = Stream<MassFlow>;
/// }
/// ```
///
/// Multiple outlets with const generics (Splitter):
/// ```ignore
/// impl<const N: usize> Process for Splitter<N> {
///     type Input = Stream<MassFlow>;
///     type Output = [Stream<MassFlow>; N];
/// }
/// ```
///
/// Multiple inlets with const generics (Mixer):
/// ```ignore
/// impl<const N: usize> Process for Mixer<N> {
///     type Input = [Stream<MassFlow>; N];
///     type Output = Stream<MassFlow>;
/// }
/// ```
pub trait Process {
    /// The input type for this unit operation.
    type Input;

    /// The output type for this unit operation.
    type Output;

    /// Processes the input and returns the output.
    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output>;
}

/// A scalar type that supports the arithmetic operations required for generic
/// residual computations, enabling both direct evaluation (with `f64`) and
/// automatic differentiation (with [`num_dual::Dual64`]).
///
/// Both `f64` and `num_dual::Dual64` (with the `autodiff` feature) implement
/// this trait, so model residuals written generically over `Scalar` can be
/// evaluated with either type without code duplication.
pub trait Scalar:
    Copy
    + From<f64>
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::Add<f64, Output = Self>
    + std::ops::Sub<f64, Output = Self>
    + std::ops::Mul<f64, Output = Self>
    + std::ops::Div<f64, Output = Self>
{
    /// Raises `self` to a floating-point power.
    fn powf(self, exp: f64) -> Self;
}

impl Scalar for f64 {
    fn powf(self, exp: f64) -> f64 {
        f64::powf(self, exp)
    }
}

/// Trait for models that define algebraic equations for solving.
///
/// Models implementing this trait expose their governing equations as functions
/// that can be differentiated via autodiff and solved using either Sequential-Modular
/// (SM) or Equation-Oriented (EO) approaches.
///
/// # Variable Layout
///
/// Models define their variables as a flat array. The `variable_names()` method
/// returns the names in order, and equations work with this array.
///
/// # Example
///
/// ```ignore
/// impl EquationModel for Pump {
///     fn n_variables(&self) -> usize { 4 }  // F_in, P_in, F_out, P_out
///     fn n_equations(&self) -> usize { 2 }  // mass balance, energy equation
///     
///     fn variable_names(&self) -> Vec<&str> {
///         vec!["F_in", "P_in", "F_out", "P_out"]
///     }
///     
///     fn residuals(&self, vars: &[f64]) -> Vec<f64> {
///         let [f_in, p_in, f_out, p_out] = [vars[0], vars[1], vars[2], vars[3]];
///         vec![
///             f_out - f_in,                           // Mass balance
///             p_out - (p_in + self.pressure_rise()),  // Pressure equation
///         ]
///     }
/// }
/// ```
pub trait EquationModel {
    /// Number of variables in the model.
    fn n_variables(&self) -> usize;

    /// Number of equations (residuals) in the model.
    fn n_equations(&self) -> usize;

    /// Names of the variables, in order.
    fn variable_names(&self) -> Vec<&str>;

    /// Compute residuals given variable values.
    ///
    /// At the solution, all residuals should be zero.
    fn residuals(&self, vars: &[f64]) -> Vec<f64>;

    /// Compute residuals with dual numbers for automatic differentiation.
    ///
    /// Enables exact Jacobian columns via forward-mode AD. Each call returns
    /// the residual vector where dual derivatives carry the partial derivatives
    /// of each residual with respect to one independent variable.
    ///
    /// Only available when the `autodiff` feature is enabled.
    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64>;

    /// Get the current variable values (initial guess or solution).
    fn get_variables(&self) -> Vec<f64>;

    /// Set variable values (after solving).
    fn set_variables(&mut self, vars: &[f64]);

    /// Returns indices of variables that are specified (known inputs).
    /// These should not be modified by the solver.
    fn specified_indices(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Returns indices of variables that are free (to be solved).
    fn free_indices(&self) -> Vec<usize> {
        let specified = self.specified_indices();
        (0..self.n_variables()).filter(|i| !specified.contains(i)).collect()
    }

    /// Returns the name of this unit operation.
    fn name(&self) -> &str {
        "unit"
    }

    /// Returns the number of inlet ports. Default is 1.
    fn n_inlet_ports(&self) -> usize {
        1
    }

    /// Returns the number of outlet ports. Default is 1.
    fn n_outlet_ports(&self) -> usize {
        1
    }

    /// Returns indices of inlet stream variables [F, T, P] for a specific port.
    /// Default assumes standard layout: port 0 uses indices 0, 1, 2.
    fn inlet_port_indices(&self, port: usize) -> Vec<usize> {
        let base = port * 3;
        vec![base, base + 1, base + 2]
    }

    /// Returns indices of outlet stream variables [F, T, P] for a specific port.
    /// Default assumes standard layout: port 0 uses indices 3, 4, 5 (after single inlet).
    fn outlet_port_indices(&self, port: usize) -> Vec<usize> {
        let base = self.n_inlet_ports() * 3 + port * 3;
        vec![base, base + 1, base + 2]
    }

    /// Returns indices of inlet stream variables [F, T, P].
    /// Default assumes standard layout: indices 0, 1, 2.
    fn inlet_indices(&self) -> Vec<usize> {
        self.inlet_port_indices(0)
    }

    /// Returns indices of outlet stream variables [F, T, P].
    /// Default assumes standard layout: indices 3, 4, 5.
    fn outlet_indices(&self) -> Vec<usize> {
        self.outlet_port_indices(0)
    }

    /// Sets inlet stream properties (F, T, P) for a specific port.
    fn set_inlet_port(&mut self, port: usize, flow: f64, temperature: f64, pressure: f64) {
        let mut vars = self.get_variables();
        let indices = self.inlet_port_indices(port);
        if indices.len() >= 3 {
            vars[indices[0]] = flow;
            vars[indices[1]] = temperature;
            vars[indices[2]] = pressure;
        }
        self.set_variables(&vars);
    }

    /// Sets inlet stream properties (F, T, P) for port 0.
    fn set_inlet(&mut self, flow: f64, temperature: f64, pressure: f64) {
        self.set_inlet_port(0, flow, temperature, pressure);
    }

    /// Gets outlet stream properties (F, T, P) for a specific port.
    fn get_outlet_port(&self, port: usize) -> (f64, f64, f64) {
        let vars = self.get_variables();
        let indices = self.outlet_port_indices(port);
        if indices.len() >= 3 {
            (vars[indices[0]], vars[indices[1]], vars[indices[2]])
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Gets outlet stream properties (F, T, P) for port 0.
    fn get_outlet(&self) -> (f64, f64, f64) {
        self.get_outlet_port(0)
    }
}

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::models::*;
    #[cfg(feature = "solvers")]
    pub use crate::solvers::StreamData as SolverStreamData;
    #[cfg(feature = "solvers")]
    pub use crate::solvers::{
        ConnectionId, EOFlowsheet, InletPort, IntoInletSpec, IntoOutletSpec, OutletPort,
        SMFlowsheet, SMSolverStats, UnitId,
        solve_equation_model,
    };
    pub use crate::{
        Algebraic,
        // Balance types and equations
        BalanceType,
        ComponentBalance,
        Connected,
        Disconnected,
        EnergyBalance,
        Equation,
        // Equation building
        EquationBuilder,
        EquationModel,
        EquationTerm,
        FlowBasis,
        // Core types
        Flowsheet,
        Inlet,
        MassBalance,
        MassFlow,
        MolarFlow,
        MomentumBalance,
        // Error types
        NomataError,
        NomataResult,
        Outlet,
        Parameter,
        // Port types (legacy - to be removed)
        Port,
        PortDirection,
        PortState,
        Process,
        // Scalar type for generic residuals
        Scalar,
        SolveMode,
        // Time domain
        Steady,
        // Stream types
        Stream,
        StreamData,
        TimeDomain,
        Var,
        // Variable types
        VarId,
        VariableRegistry,
        VariableRole,
        connect_ports,
    };

    #[cfg(feature = "thermodynamics")]
    pub use crate::{PredefinedMix, Pure, Substance};
}

// Error Types

/// Main error type for Nomata operations.
#[derive(Error, Debug)]
pub enum NomataError {
    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Unit error: {0}")]
    Unit(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Solver error: {0}")]
    Solver(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Not solved: flowsheet must be solved before accessing results")]
    NotSolved,

    #[error("Variable not found: {0}")]
    VariableNotFound(usize),

    #[error("Port not found: {0}")]
    PortNotFound(String),

    #[error("Unit not found: {0}")]
    UnitNotFound(String),
}

#[cfg(feature = "solvers")]
impl From<solvers::SolverError> for NomataError {
    fn from(err: solvers::SolverError) -> Self {
        NomataError::Solver(err.to_string())
    }
}

/// Result type for Nomata operations.
pub type NomataResult<T> = Result<T, NomataError>;

// Variable Roles (Type-Level)

/// Marker trait for variable roles. Sealed - only Parameter, Algebraic, Differential implement it.
pub trait VariableRole: Clone + Copy + 'static {
    /// Human-readable name for debugging.
    const NAME: &'static str;
}

/// Parameter: constant value that doesn't change during simulation.
///
/// Parameters represent physical properties, rate constants, or design specifications.
/// They cannot have time derivatives.
///
/// # Examples
/// - Rate constants (k)
/// - Densities (rho)
/// - Heat capacities (Cp)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parameter;
impl VariableRole for Parameter {
    const NAME: &'static str = "Parameter";
}

/// Algebraic: variable without time derivative (instantaneous relationship).
///
/// Algebraic variables are computed from equations without accumulation terms.
/// They cannot have time derivatives in balanced equations.
///
/// # Examples
/// - Temperature (T)
/// - Pressure (P)
/// - Concentration (C)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Algebraic;
impl VariableRole for Algebraic {
    const NAME: &'static str = "Algebraic";
}

// Typed Variable ID

/// A type-safe identifier for a variable in the registry.
///
/// The type parameter `R` encodes the variable's role at compile time,
/// preventing misuse (e.g., taking derivatives of parameters).
///
/// # Type Safety
///
/// ```compile_fail
/// use nomata::prelude::*;
///
/// let mut registry = VariableRegistry::new();
/// let param: VarId<Parameter> = registry.register_parameter(1.0);
///
/// // ERROR: Parameter variables cannot be differentiated
/// registry.differentiate(&param);
/// ```
#[derive(Debug)]
pub struct VarId<R: VariableRole> {
    index: usize,
    _role: PhantomData<R>,
}

impl<R: VariableRole> Clone for VarId<R> {
    fn clone(&self) -> Self {
        VarId { index: self.index, _role: PhantomData }
    }
}

impl<R: VariableRole> Copy for VarId<R> {}

impl<R: VariableRole> PartialEq for VarId<R> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<R: VariableRole> Eq for VarId<R> {}

impl<R: VariableRole> std::hash::Hash for VarId<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<R: VariableRole> VarId<R> {
    /// Returns the raw index (for internal use).
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the role name.
    pub fn role_name(&self) -> &'static str {
        R::NAME
    }
}

// Variable Registry (Type-Safe)

/// Metadata for a registered variable.
#[derive(Debug, Clone)]
struct VariableEntry {
    value: f64,
    name: Option<String>,
}

/// A type-safe registry for process variables.
///
/// Variables are registered with their role (Parameter, Algebraic, Differential)
/// and receive a typed ID that prevents misuse at compile time.
///
/// # Example
///
/// ```
/// use nomata::prelude::*;
///
/// let mut registry = VariableRegistry::new();
///
/// // Register typed variables
/// let k: VarId<Parameter> = registry.register_parameter(0.5);
/// let t: VarId<Algebraic> = registry.register_algebraic(300.0);
/// // Access values
/// assert_eq!(registry.get(&k), 0.5);
/// assert_eq!(registry.get(&t), 300.0);
///
/// // Update values
/// registry.set(&t, 350.0);
/// assert_eq!(registry.get(&t), 350.0);
/// ```
#[derive(Debug, Default)]
pub struct VariableRegistry {
    entries: Vec<VariableEntry>,
    parameter_indices: Vec<usize>,
    algebraic_indices: Vec<usize>,
}

impl VariableRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        VariableRegistry::default()
    }

    /// Registers a parameter variable (time-invariant constant).
    pub fn register_parameter(&mut self, value: f64) -> VarId<Parameter> {
        self.register_parameter_named(value, None)
    }

    /// Registers a named parameter variable.
    pub fn register_parameter_named(&mut self, value: f64, name: Option<&str>) -> VarId<Parameter> {
        let index = self.entries.len();
        self.entries.push(VariableEntry { value, name: name.map(|s| s.to_string()) });
        self.parameter_indices.push(index);
        VarId { index, _role: PhantomData }
    }

    /// Registers an algebraic variable (no time derivative).
    pub fn register_algebraic(&mut self, value: f64) -> VarId<Algebraic> {
        self.register_algebraic_named(value, None)
    }

    /// Registers a named algebraic variable.
    pub fn register_algebraic_named(&mut self, value: f64, name: Option<&str>) -> VarId<Algebraic> {
        let index = self.entries.len();
        self.entries.push(VariableEntry { value, name: name.map(|s| s.to_string()) });
        self.algebraic_indices.push(index);
        VarId { index, _role: PhantomData }
    }

    /// Gets the value of a variable.
    pub fn get<R: VariableRole>(&self, id: &VarId<R>) -> f64 {
        self.entries[id.index].value
    }

    /// Sets the value of a variable.
    pub fn set<R: VariableRole>(&mut self, id: &VarId<R>, value: f64) {
        self.entries[id.index].value = value;
    }

    /// Gets the name of a variable (if set).
    pub fn name<R: VariableRole>(&self, id: &VarId<R>) -> Option<&str> {
        self.entries[id.index].name.as_deref()
    }

    /// Returns the number of parameters.
    pub fn parameter_count(&self) -> usize {
        self.parameter_indices.len()
    }

    /// Returns the number of algebraic variables.
    pub fn algebraic_count(&self) -> usize {
        self.algebraic_indices.len()
    }

    /// Returns the total number of variables.
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Gets all values as a vector (for solvers).
    pub fn get_all_values(&self) -> Vec<f64> {
        self.entries.iter().map(|e| e.value).collect()
    }

    /// Sets all values from a vector (for solvers).
    pub fn set_all_values(&mut self, values: &[f64]) {
        for (i, v) in values.iter().enumerate() {
            if i < self.entries.len() {
                self.entries[i].value = *v;
            }
        }
    }

    /// Gets values of all algebraic variables.
    pub fn get_algebraic_values(&self) -> Vec<f64> {
        self.algebraic_indices.iter().map(|&i| self.entries[i].value).collect()
    }
}

// Convenience Var wrapper (runtime value with compile-time role)

/// A typed variable holding a value with compile-time role enforcement.
///
/// This is a convenience wrapper around `Rc<RefCell<f64>>` that tracks the role.
#[derive(Debug, Clone)]
pub struct Var<R: VariableRole> {
    value: Rc<RefCell<f64>>,
    _role: PhantomData<R>,
}

impl<R: VariableRole> Var<R> {
    /// Creates a new variable with the given initial value.
    pub fn new(value: f64) -> Self {
        Var { value: Rc::new(RefCell::new(value)), _role: PhantomData }
    }

    /// Gets the current value.
    pub fn get(&self) -> f64 {
        *self.value.borrow()
    }

    /// Sets the value.
    pub fn set(&self, value: f64) {
        *self.value.borrow_mut() = value;
    }
}

// Balance Types (Type-Level Conservation Laws)

/// Marker trait for balance/conservation law types.
///
/// Different balance types are incompatible at compile time,
/// preventing accidental mixing of mass and energy equations.
pub trait BalanceType: 'static {
    /// Human-readable name for the balance type.
    const NAME: &'static str;
}

/// Mass balance (conservation of mass).
///
/// Used for equations of the form: d(mass)/dt = mass_in - mass_out
#[derive(Debug, Clone, Copy)]
pub struct MassBalance;
impl BalanceType for MassBalance {
    const NAME: &'static str = "MassBalance";
}

/// Energy balance (conservation of energy).
///
/// Used for equations of the form: d(U)/dt = H_in - H_out + Q - W
#[derive(Debug, Clone, Copy)]
pub struct EnergyBalance;
impl BalanceType for EnergyBalance {
    const NAME: &'static str = "EnergyBalance";
}

/// Component balance (conservation of species).
///
/// Used for equations of the form: d(n_i)/dt = F_in*z_i - F_out*x_i + r_i*V
#[derive(Debug, Clone, Copy)]
pub struct ComponentBalance;
impl BalanceType for ComponentBalance {
    const NAME: &'static str = "ComponentBalance";
}

/// Momentum balance (conservation of momentum).
///
/// Used for equations of the form: d(momentum)/dt = forces
#[derive(Debug, Clone, Copy)]
pub struct MomentumBalance;
impl BalanceType for MomentumBalance {
    const NAME: &'static str = "MomentumBalance";
}

// Typed Equations

/// A typed balance equation parameterized by balance type.
///
/// The type parameter `B` encodes which conservation law this equation represents,
/// preventing accidental mixing of different balance types.
///
/// # Example
///
/// ```
/// use nomata::prelude::*;
///
/// // Create a mass balance equation
/// let mass_eq: Equation<MassBalance> = Equation::new("reactor_mass_balance");
///
/// // Create an energy balance equation
/// let energy_eq: Equation<EnergyBalance> = Equation::new("reactor_energy_balance");
///
/// // These are different types and cannot be confused!
/// ```
#[derive(Debug)]
pub struct Equation<B: BalanceType> {
    name: String,
    residual: f64,
    terms: Vec<EquationTerm>,
    _balance_type: PhantomData<B>,
}

impl<B: BalanceType> Equation<B> {
    /// Creates a new equation with the given name.
    pub fn new(name: &str) -> Self {
        Equation {
            name: name.to_string(),
            residual: 0.0,
            terms: Vec::new(),
            _balance_type: PhantomData,
        }
    }

    /// Gets the equation name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the current residual value.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Sets the residual value.
    pub fn set_residual(&mut self, value: f64) {
        self.residual = value;
    }

    /// Checks if the equation is satisfied (residual near zero).
    pub fn is_satisfied(&self, tolerance: f64) -> bool {
        self.residual.abs() < tolerance
    }

    /// Adds a term to the equation.
    pub fn add_term(&mut self, term: EquationTerm) {
        self.terms.push(term);
    }

    /// Gets the terms.
    pub fn terms(&self) -> &[EquationTerm] {
        &self.terms
    }

    /// Returns the balance type name.
    pub fn balance_type(&self) -> &'static str {
        B::NAME
    }

    /// Evaluates the equation given variable values.
    pub fn evaluate(&self, var_values: &HashMap<String, f64>) -> f64 {
        self.terms.iter().map(|term| term.evaluate(var_values)).sum()
    }
}

impl<B: BalanceType> Clone for Equation<B> {
    fn clone(&self) -> Self {
        Equation {
            name: self.name.clone(),
            residual: self.residual,
            terms: self.terms.clone(),
            _balance_type: PhantomData,
        }
    }
}

/// A term in a balance equation.
///
/// Represents coefficient * variable, e.g., "-1.0 * F_out" or "1.0 * dV_dt"
#[derive(Debug, Clone)]
pub struct EquationTerm {
    /// Coefficient multiplying the variable.
    pub coefficient: f64,
    /// Variable name or description.
    pub variable: String,
}

impl EquationTerm {
    /// Creates a new equation term.
    pub fn new(coefficient: f64, variable: &str) -> Self {
        EquationTerm { coefficient, variable: variable.to_string() }
    }

    /// Evaluates the term given variable values.
    pub fn evaluate(&self, var_values: &HashMap<String, f64>) -> f64 {
        self.coefficient * var_values.get(&self.variable).copied().unwrap_or(0.0)
    }
}

// Time Domain (Type-Level)

/// Marker trait for time domain.
pub trait TimeDomain: Clone + 'static {
    /// Whether this is steady-state (no time derivatives).
    const IS_STEADY: bool;
}

/// Steady-state: no time derivatives allowed.
#[derive(Debug, Clone, Copy)]
pub struct Steady;
impl TimeDomain for Steady {
    const IS_STEADY: bool = true;
}

// Flow Basis (Type-Level Stream Classification)

/// Marker trait for stream flow basis.
pub trait FlowBasis: Clone + 'static {
    /// Human-readable name.
    const NAME: &'static str;
}

/// Molar flow basis (mol/s, kmol/h, etc.).
#[derive(Debug, Clone, Copy)]
pub struct MolarFlow;
impl FlowBasis for MolarFlow {
    const NAME: &'static str = "MolarFlow";
}

/// Mass flow basis (kg/s, lb/h, etc.).
#[derive(Debug, Clone, Copy)]
pub struct MassFlow;
impl FlowBasis for MassFlow {
    const NAME: &'static str = "MassFlow";
}

// Typed Stream

/// Stream data (runtime).
#[derive(Debug, Clone)]
pub struct StreamData {
    pub flow: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub composition: Vec<f64>,
    pub components: Vec<String>,
}

impl Default for StreamData {
    fn default() -> Self {
        StreamData {
            flow: 0.0,
            temperature: 298.15,
            pressure: 101325.0,
            composition: Vec::new(),
            components: Vec::new(),
        }
    }
}

/// A typed process stream.
///
/// The type parameter `F` enforces the flow basis at compile time,
/// preventing connections between incompatible streams.
///
/// # Type Safety
///
/// Streams with different flow bases cannot be mixed in operations that
/// require type matching:
///
/// ```
/// use nomata::prelude::*;
///
/// // MolarFlow and MassFlow streams are distinct types
/// let molar: Stream<MolarFlow> = Stream::new().with_flow(100.0).build().unwrap();
/// let mass: Stream<MassFlow> = Stream::new().with_flow(50.0).build().unwrap();
///
/// // Each stream can be used independently
/// assert_eq!(molar.flow(), 100.0);
/// assert_eq!(mass.flow(), 50.0);
/// ```
#[derive(Debug, Clone)]
pub struct Stream<F: FlowBasis = MolarFlow> {
    id: usize,
    data: Rc<RefCell<StreamData>>,
    is_initialized: bool,
    _flow_basis: PhantomData<F>,
}

impl<F: FlowBasis> Stream<F> {
    /// Creates a new stream builder.
    pub fn new() -> StreamBuilder<F> {
        StreamBuilder::new()
    }

    /// Creates a simple feed stream with flow, temperature, and pressure.
    ///
    /// This is a convenience method for creating streams without the builder pattern.
    ///
    /// # Arguments
    ///
    /// * `flow` - Flow rate [kg/s or mol/s depending on F]
    /// * `temperature` - Temperature [K]
    /// * `pressure` - Pressure [Pa]
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::prelude::*;
    ///
    /// let feed = Stream::<MassFlow>::feed(10.0, 300.0, 1e5);
    /// assert_eq!(feed.flow(), 10.0);
    /// assert_eq!(feed.temperature(), 300.0);
    /// assert_eq!(feed.pressure(), 1e5);
    /// ```
    pub fn feed(flow: f64, temperature: f64, pressure: f64) -> Self {
        Stream::new()
            .with_flow(flow)
            .with_temperature(temperature)
            .with_pressure(pressure)
            .build()
            .expect("Simple feed stream should always be valid")
    }

    /// Returns the stream ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns whether this stream has been initialized/computed.
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Gets the total flow rate.
    pub fn flow(&self) -> f64 {
        self.data.borrow().flow
    }

    /// Gets the temperature [K].
    pub fn temperature(&self) -> f64 {
        self.data.borrow().temperature
    }

    /// Gets the pressure [Pa].
    pub fn pressure(&self) -> f64 {
        self.data.borrow().pressure
    }

    /// Gets the composition (mole or mass fractions).
    pub fn composition(&self) -> Vec<f64> {
        self.data.borrow().composition.clone()
    }

    /// Gets the component names.
    pub fn components(&self) -> Vec<String> {
        self.data.borrow().components.clone()
    }

    /// Gets the number of components.
    pub fn n_components(&self) -> usize {
        self.data.borrow().components.len()
    }

    /// Gets the flow of a specific component.
    pub fn component_flow(&self, index: usize) -> f64 {
        let data = self.data.borrow();
        data.flow * data.composition.get(index).copied().unwrap_or(0.0)
    }

    /// Gets the raw data (for SM execution).
    pub fn to_data(&self) -> StreamData {
        self.data.borrow().clone()
    }
}

impl<F: FlowBasis> Default for Stream<F> {
    fn default() -> Self {
        Stream::new().build().unwrap()
    }
}

/// Builder for creating typed streams.
pub struct StreamBuilder<F: FlowBasis = MolarFlow> {
    flow: f64,
    temperature: f64,
    pressure: f64,
    components: Vec<String>,
    composition: Vec<f64>,
    _flow_basis: PhantomData<F>,
}

impl<F: FlowBasis> StreamBuilder<F> {
    fn new() -> Self {
        StreamBuilder {
            flow: 0.0,
            temperature: 298.15,
            pressure: 101325.0,
            components: Vec::new(),
            composition: Vec::new(),
            _flow_basis: PhantomData,
        }
    }

    /// Sets the total flow rate.
    pub fn with_flow(mut self, flow: f64) -> Self {
        self.flow = flow;
        self
    }

    /// Sets the temperature [K].
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets the pressure [Pa].
    pub fn with_pressure(mut self, pressure: f64) -> Self {
        self.pressure = pressure;
        self
    }

    /// Sets temperature and pressure together.
    pub fn with_conditions(mut self, temperature: f64, pressure: f64) -> Self {
        self.temperature = temperature;
        self.pressure = pressure;
        self
    }

    /// Sets the composition using component names and fractions.
    pub fn with_composition(mut self, components: &[&str], fractions: &[f64]) -> Self {
        self.components = components.iter().map(|s| s.to_string()).collect();
        self.composition = fractions.to_vec();
        self
    }

    /// Sets composition from a pure component.
    pub fn pure(mut self, component: &str) -> Self {
        self.components = vec![component.to_string()];
        self.composition = vec![1.0];
        self
    }

    /// Builds the stream, validating the configuration.
    pub fn build(self) -> NomataResult<Stream<F>> {
        // Validate composition
        if !self.composition.is_empty() {
            if self.composition.len() != self.components.len() {
                return Err(NomataError::Validation(format!(
                    "Composition length {} doesn't match component count {}",
                    self.composition.len(),
                    self.components.len()
                )));
            }

            let sum: f64 = self.composition.iter().sum();
            if (sum - 1.0).abs() > 1e-6 && sum > 0.0 {
                return Err(NomataError::Validation(format!(
                    "Composition must sum to 1.0, got {}",
                    sum
                )));
            }

            if self.composition.iter().any(|&x| x < 0.0) {
                return Err(NomataError::Validation(
                    "Composition fractions cannot be negative".to_string(),
                ));
            }
        }

        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        Ok(Stream {
            id: COUNTER.fetch_add(1, Ordering::SeqCst),
            data: Rc::new(RefCell::new(StreamData {
                flow: self.flow,
                temperature: self.temperature,
                pressure: self.pressure,
                composition: self.composition,
                components: self.components,
            })),
            is_initialized: true,
            _flow_basis: PhantomData,
        })
    }
}

impl<F: FlowBasis> Default for StreamBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

// Port System (Type-Safe with Linear Ownership)

/// Marker trait for port direction.
pub trait PortDirection: 'static {
    const NAME: &'static str;
}

/// Inlet port direction.
#[derive(Debug, Clone, Copy)]
pub struct Inlet;
impl PortDirection for Inlet {
    const NAME: &'static str = "Inlet";
}

/// Outlet port direction.
#[derive(Debug, Clone, Copy)]
pub struct Outlet;
impl PortDirection for Outlet {
    const NAME: &'static str = "Outlet";
}

/// Marker trait for port connection state.
pub trait PortState: 'static {}

/// Port is available for connection.
#[derive(Debug, Clone, Copy)]
pub struct Disconnected;
impl PortState for Disconnected {}

/// Port has been connected (consumed).
#[derive(Debug, Clone, Copy)]
pub struct Connected;
impl PortState for Connected {}

/// A type-safe port with linear ownership semantics.
///
/// # Type Parameters
///
/// - `F`: Flow basis (MolarFlow, MassFlow)
/// - `D`: Direction (Inlet, Outlet)
/// - `S`: State (Disconnected, Connected)
///
/// # Linear Ownership
///
/// Ports are **consumed** when connected. A `Disconnected` port becomes
/// `Connected` after use, preventing double-connection at compile time.
///
/// ```
/// use nomata::prelude::*;
///
/// let outlet: Port<MolarFlow, Outlet, Disconnected> = Port::new("out");
/// let inlet: Port<MolarFlow, Inlet, Disconnected> = Port::new("in");
///
/// // Ports are consumed when connected
/// let (out_conn, in_conn) = connect_ports(outlet, inlet);
///
/// // After connection, the ports have Connected state
/// // and cannot be connected again (they're consumed by value)
/// ```
#[derive(Debug)]
pub struct Port<F: FlowBasis, D: PortDirection, S: PortState> {
    name: String,
    unit_id: Option<usize>,
    stream_data: Rc<RefCell<Option<StreamData>>>,
    _flow_basis: PhantomData<F>,
    _direction: PhantomData<D>,
    _state: PhantomData<S>,
}

impl<F: FlowBasis, D: PortDirection> Port<F, D, Disconnected> {
    /// Creates a new disconnected port.
    pub fn new(name: &str) -> Self {
        Port {
            name: name.to_string(),
            unit_id: None,
            stream_data: Rc::new(RefCell::new(None)),
            _flow_basis: PhantomData,
            _direction: PhantomData,
            _state: PhantomData,
        }
    }

    /// Creates a port with a unit ID.
    pub fn with_unit(name: &str, unit_id: usize) -> Self {
        Port {
            name: name.to_string(),
            unit_id: Some(unit_id),
            stream_data: Rc::new(RefCell::new(None)),
            _flow_basis: PhantomData,
            _direction: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<F: FlowBasis, D: PortDirection, S: PortState> Port<F, D, S> {
    /// Gets the port name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the unit ID (if assigned).
    pub fn unit_id(&self) -> Option<usize> {
        self.unit_id
    }

    /// Gets the stream data (if available).
    pub fn stream_data(&self) -> Option<StreamData> {
        self.stream_data.borrow().clone()
    }

    /// Sets the stream data.
    pub fn set_stream_data(&self, data: StreamData) {
        *self.stream_data.borrow_mut() = Some(data);
    }
}

/// Connects an outlet port to an inlet port, consuming both.
///
/// # Type Safety
///
/// This function enforces:
/// - Stream type compatibility (same flow basis)
/// - Direction correctness (output -> input)
/// - Single-use (ports are consumed, preventing double-connection)
///
/// # Returns
///
/// Returns the connected ports (with `Connected` state).
pub fn connect_ports<F: FlowBasis>(
    output: Port<F, Outlet, Disconnected>,
    input: Port<F, Inlet, Disconnected>,
) -> (Port<F, Outlet, Connected>, Port<F, Inlet, Connected>) {
    // Share stream data between connected ports
    let shared_data = output.stream_data.clone();

    let out_connected = Port {
        name: output.name,
        unit_id: output.unit_id,
        stream_data: shared_data.clone(),
        _flow_basis: PhantomData,
        _direction: PhantomData,
        _state: PhantomData,
    };

    let in_connected = Port {
        name: input.name,
        unit_id: input.unit_id,
        stream_data: shared_data,
        _flow_basis: PhantomData,
        _direction: PhantomData,
        _state: PhantomData,
    };

    (out_connected, in_connected)
}

// Solve Mode

/// Solving approach for the flowsheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMode {
    /// Sequential Modular: solve units one-by-one, iterate on tear streams for cycles.
    SequentialModular,
    /// Equation-Oriented: assemble all equations and solve simultaneously.
    EquationOriented,
}

impl Default for SolveMode {
    fn default() -> Self {
        SolveMode::SequentialModular
    }
}

// Equation Builder (Type-Safe)

/// Type-erased residual function.
pub struct ResidualFn {
    pub name: String,
    pub evaluate: Box<dyn Fn(&VariableRegistry) -> f64>,
}

impl std::fmt::Debug for ResidualFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualFn").field("name", &self.name).finish()
    }
}

/// Builder for equation system with typed variables.
#[derive(Debug, Default)]
pub struct EquationBuilder {
    /// Algebraic equations.
    pub algebraic: Vec<ResidualFn>,
}

impl EquationBuilder {
    /// Creates a new equation builder.
    pub fn new() -> Self {
        EquationBuilder::default()
    }

    /// Adds an algebraic equation using typed variable IDs.
    pub fn add_algebraic<F>(&mut self, name: &str, residual: F)
    where
        F: Fn(&VariableRegistry) -> f64 + 'static,
    {
        self.algebraic.push(ResidualFn { name: name.to_string(), evaluate: Box::new(residual) });
    }

    /// Returns total equation count.
    pub fn equation_count(&self) -> usize {
        self.algebraic.len()
    }

    /// Evaluates all residuals.
    pub fn evaluate(&self, registry: &VariableRegistry) -> Vec<f64> {
        self.algebraic.iter().map(|eq| (eq.evaluate)(registry)).collect()
    }
}

/// A typed process flowsheet container.
///
/// The type parameter `T` enforces time domain constraints:
/// - `Flowsheet<Steady>`: Only algebraic equations, no d/dt terms
///
/// Note: The new `Process` trait provides a more direct API for stream processing.
/// Use `unit.process(input)` directly for most use cases.
pub struct Flowsheet<T: TimeDomain = Steady> {
    solve_mode: SolveMode,
    streams: HashMap<usize, StreamData>,
    is_solved: bool,
    registry: VariableRegistry,
    _time_domain: PhantomData<T>,
}

impl<T: TimeDomain> std::fmt::Debug for Flowsheet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flowsheet")
            .field("solve_mode", &self.solve_mode)
            .field("stream_count", &self.streams.len())
            .field("is_solved", &self.is_solved)
            .field("is_steady", &T::IS_STEADY)
            .finish()
    }
}

impl<T: TimeDomain> Flowsheet<T> {
    /// Creates a new flowsheet.
    pub fn new() -> Self {
        Flowsheet {
            solve_mode: SolveMode::SequentialModular,
            streams: HashMap::new(),
            is_solved: false,
            registry: VariableRegistry::new(),
            _time_domain: PhantomData,
        }
    }

    /// Creates a flowsheet with specified solve mode.
    pub fn with_mode(solve_mode: SolveMode) -> Self {
        let mut fs = Self::new();
        fs.solve_mode = solve_mode;
        fs
    }

    /// Gets the solve mode.
    pub fn solve_mode(&self) -> SolveMode {
        self.solve_mode
    }

    /// Sets the solve mode.
    pub fn set_solve_mode(&mut self, mode: SolveMode) {
        self.solve_mode = mode;
        self.is_solved = false;
    }

    /// Returns stream count.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Returns whether solved.
    pub fn is_solved(&self) -> bool {
        self.is_solved
    }

    /// Gets the variable registry.
    pub fn registry(&self) -> &VariableRegistry {
        &self.registry
    }

    /// Gets mutable variable registry.
    pub fn registry_mut(&mut self) -> &mut VariableRegistry {
        &mut self.registry
    }

    /// Adds a stream to the flowsheet.
    pub fn add_stream<F: FlowBasis>(&mut self, stream: &Stream<F>) -> usize {
        let id = stream.id();
        self.streams.insert(id, stream.to_data());
        self.is_solved = false;
        id
    }

    /// Gets stream data by ID.
    pub fn get_stream(&self, stream_id: usize) -> Option<&StreamData> {
        self.streams.get(&stream_id)
    }
}

impl<T: TimeDomain> Default for Flowsheet<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_variable_registry() {
        let mut registry = VariableRegistry::new();

        let k: VarId<Parameter> = registry.register_parameter(0.5);
        let t: VarId<Algebraic> = registry.register_algebraic(300.0);

        assert_eq!(registry.get(&k), 0.5);
        assert_eq!(registry.get(&t), 300.0);

        registry.set(&t, 350.0);
        assert_eq!(registry.get(&t), 350.0);

        assert_eq!(registry.parameter_count(), 1);
        assert_eq!(registry.algebraic_count(), 1);
    }

    #[test]
    fn test_typed_stream() {
        let stream: Stream<MolarFlow> = Stream::new()
            .with_flow(100.0)
            .with_composition(&["A", "B"], &[0.6, 0.4])
            .with_conditions(300.0, 5e5)
            .build()
            .unwrap();

        assert_eq!(stream.flow(), 100.0);
        assert_eq!(stream.temperature(), 300.0);
    }

    #[test]
    fn test_stream_validation() {
        let result: NomataResult<Stream<MolarFlow>> =
            Stream::new().with_composition(&["A", "B"], &[0.6]).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_port_connection_type_safety() {
        let out: Port<MolarFlow, Outlet, Disconnected> = Port::new("out");
        let inp: Port<MolarFlow, Inlet, Disconnected> = Port::new("in");

        let (out_conn, in_conn) = connect_ports(out, inp);

        // Verify they're connected
        assert_eq!(out_conn.name(), "out");
        assert_eq!(in_conn.name(), "in");
    }

    #[test]
    fn test_flowsheet_creation() {
        let flowsheet: Flowsheet<Steady> = Flowsheet::new();
        assert_eq!(flowsheet.stream_count(), 0);
        assert!(Steady::IS_STEADY);
    }

    #[test]
    fn test_equation_creation() {
        let mass_eq: Equation<MassBalance> = Equation::new("reactor_mass");
        let energy_eq: Equation<EnergyBalance> = Equation::new("reactor_energy");

        assert_eq!(mass_eq.name(), "reactor_mass");
        assert_eq!(mass_eq.balance_type(), "MassBalance");
        assert_eq!(energy_eq.balance_type(), "EnergyBalance");
    }

    #[test]
    fn test_equation_residual() {
        let mut eq: Equation<MassBalance> = Equation::new("test");
        assert_eq!(eq.residual(), 0.0);

        eq.set_residual(1e-10);
        assert!(eq.is_satisfied(1e-8));

        eq.set_residual(0.1);
        assert!(!eq.is_satisfied(1e-8));
    }

    #[test]
    fn test_equation_terms() {
        let mut eq: Equation<MassBalance> = Equation::new("mass_balance");

        // dV/dt = F_in - F_out
        eq.add_term(EquationTerm::new(1.0, "dV_dt"));
        eq.add_term(EquationTerm::new(-1.0, "F_in"));
        eq.add_term(EquationTerm::new(1.0, "F_out"));

        assert_eq!(eq.terms().len(), 3);

        // Evaluate with values
        let mut vars = HashMap::new();
        vars.insert("dV_dt".to_string(), 0.0);
        vars.insert("F_in".to_string(), 100.0);
        vars.insert("F_out".to_string(), 100.0);

        // At steady state: 0 = 100 - 100 = 0
        let residual = eq.evaluate(&vars);
        assert!((residual).abs() < 1e-10);
    }
}
