#![cfg_attr(not(doctest), doc = include_str!("../README.md"))]

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

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

// Reexport fluid substance types for easier access
#[cfg(feature = "thermodynamics")]
pub use thermodynamics::{
    Substance,
    fluids::{PredefinedMix, Pure},
};

// Reexports from differential-equations crate for ODE solving
#[cfg(feature = "solvers")]
pub use differential_equations::methods::{ExplicitRungeKutta, ImplicitRungeKutta};

/// Variable Registry: Shared State Management
/// Unique identifier for a variable in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);

impl VarId {
    /// Gets the index value.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Central registry for all variable values.
///
/// In a process simulator, variables need shared state so that:
/// - Solvers can modify values
/// - Unit operations can read updated values
/// - The equation system references the same variables
///
/// # Examples
///
/// ```
/// use nomata::{VariableRegistry, Algebraic};
///
/// let registry = VariableRegistry::new();
/// let temp = registry.create_algebraic(298.15);
///
/// // Solver updates the value
/// registry.set(temp.id(), 310.0);
///
/// // Unit operation reads the updated value
/// assert_eq!(registry.get(temp.id()), 310.0);
/// ```
#[derive(Debug, Clone)]
pub struct VariableRegistry {
    /// Storage for all variable values
    values: Rc<RefCell<Vec<f64>>>,
    /// Metadata: variable roles (for validation)
    roles: Rc<RefCell<Vec<&'static str>>>,
}

impl VariableRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        VariableRegistry {
            values: Rc::new(RefCell::new(Vec::new())),
            roles: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Registers a new variable and returns its ID.
    fn register(&self, initial_value: f64, role: &'static str) -> VarId {
        let mut values = self.values.borrow_mut();
        let mut roles = self.roles.borrow_mut();
        let id = VarId(values.len());
        values.push(initial_value);
        roles.push(role);
        id
    }

    /// Creates a new parameter variable.
    pub fn create_parameter(&self, value: f64) -> Var<Parameter> {
        let id = self.register(value, "Parameter");
        Var::from_id(id, self.clone())
    }

    /// Creates a new algebraic variable.
    pub fn create_algebraic(&self, value: f64) -> Var<Algebraic> {
        let id = self.register(value, "Algebraic");
        Var::from_id(id, self.clone())
    }

    /// Creates a new differential variable.
    pub fn create_differential(&self, value: f64) -> Var<Differential> {
        let id = self.register(value, "Differential");
        Var::from_id(id, self.clone())
    }

    /// Gets the current value of a variable.
    pub fn get(&self, id: VarId) -> f64 {
        self.values.borrow()[id.0]
    }

    /// Sets the value of a variable.
    pub fn set(&self, id: VarId, value: f64) {
        self.values.borrow_mut()[id.0] = value;
    }

    /// Sets the value of a variable by its index in the registry.
    pub fn set_by_index(&self, index: usize, value: f64) {
        assert!(index < self.len(), "Index {} out of bounds (max {})", index, self.len() - 1);
        self.values.borrow_mut()[index] = value;
    }

    /// Gets the role of a variable (for debugging/validation).
    pub fn get_role(&self, id: VarId) -> &'static str {
        self.roles.borrow()[id.0]
    }

    /// Returns the total number of variables.
    pub fn len(&self) -> usize {
        self.values.borrow().len()
    }

    /// Returns whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.values.borrow().is_empty()
    }

    /// Gets all values as a vector (for solvers).
    pub fn get_all_values(&self) -> Vec<f64> {
        self.values.borrow().clone()
    }

    /// Sets all values from a vector (for solvers).
    pub fn set_all_values(&self, values: &[f64]) {
        assert_eq!(values.len(), self.len(), "Value vector length mismatch");
        self.values.borrow_mut().copy_from_slice(values);
    }

    /// Initializes the registry from an equation system.
    ///
    /// Creates variables in the registry corresponding to all variables
    /// in the equation system, with appropriate roles (differential/algebraic).
    ///
    /// # Arguments
    /// * `equations` - The equation system to initialize from
    /// * `initial_conditions` - Optional initial values for all variables.
    ///   If `None`, all variables are initialized to 0.0.
    ///   Must have exactly `equations.variable_count()` elements if provided.
    ///
    /// # Panics
    /// Panics if the registry is not empty or if `initial_conditions` length
    /// doesn't match the number of variables in the equation system.
    ///
    /// # Examples
    /// ```ignore
    /// let registry = VariableRegistry::new();
    /// registry.initialize_from_equations(flowsheet.equations(), None);
    ///
    /// // For custom initial conditions, use the stream-level initial conditions API:
    /// use std::collections::HashMap;
    /// let mut conditions = HashMap::new();
    /// conditions.insert("F_in".to_string(), 10.0);
    /// conditions.insert("T".to_string(), 350.0);
    /// let initial_conditions = flowsheet.set_stream_initial_conditions("CSTR-101", &conditions);
    /// let registry = flowsheet.create_registry_with_initial_conditions(&initial_conditions);
    /// ```
    pub fn initialize_from_equations<T: TimeDomain>(
        &self,
        equations: &EquationSystem<T>,
        initial_conditions: Option<&[f64]>,
    ) {
        // Validate registry is empty
        assert!(self.is_empty(), "Registry must be empty when initializing from equations");

        let var_count = equations.variable_count();

        // Validate initial conditions if provided
        if let Some(conditions) = initial_conditions {
            assert_eq!(
                conditions.len(),
                var_count,
                "Initial conditions length ({}) must match variable count ({})",
                conditions.len(),
                var_count
            );
        }

        // Create variables with appropriate roles and initial values
        for i in 0..var_count {
            let initial_value = initial_conditions.map(|conditions| conditions[i]).unwrap_or(0.0);

            if equations.is_variable_differential(i) {
                self.create_differential(initial_value);
            } else {
                self.create_algebraic(initial_value);
            }
        }
    }
}

impl Default for VariableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Type Layer: Core Traits
/// Marker trait for variable roles in process models.
///
/// Variables can be algebraic, differential, or parameters. The role determines
/// what operations are valid and how the variable behaves in equations.
///
/// # Variable Role Hierarchy
///
/// - **Parameters**: Constants with respect to time (e.g., rate constants, physical properties)
/// - **Algebraic**: Instantaneous relationships that may vary in time
/// - **Differential**: Accumulations with time derivatives
pub trait VariableRole {}

/// Parameters are constants with respect to time.
///
/// Parameters represent physical properties, rate constants, or design specifications
/// that remain fixed throughout a simulation. Unlike algebraic variables, parameters:
///
/// - Cannot change during time integration
/// - Are known a priori (not computed from equations)
/// - May be used in sensitivity analysis or optimization
///
/// # Examples
///
/// ```
/// use nomata::{Var, Parameter};
///
/// let rate_constant: Var<Parameter> = Var::new(0.5);  // k = 0.5 1/s
/// let density: Var<Parameter> = Var::new(1000.0);      // rho = 1000 kg/m^3
/// ```
///
/// # Extensibility
///
/// The distinction between parameters and variables enables:
/// - Automatic differentiation w.r.t. parameters (sensitivity analysis)
/// - Parameter estimation / optimization workflows
/// - Clear separation of model structure from parameter values
#[derive(Debug, Clone, Copy)]
pub struct Parameter;
impl VariableRole for Parameter {}

/// Algebraic variables cannot have time derivatives.
///
/// Algebraic variables represent instantaneous relationships and constraints
/// that must hold at all times, such as equilibrium conditions or
/// constitutive equations. Unlike parameters, algebraic variables:
///
/// - Are computed from other variables via equations
/// - May change during time integration (but without explicit d/dt terms)
/// - Represent intensive properties or derived quantities
#[derive(Debug, Clone, Copy)]
pub struct Algebraic;
impl VariableRole for Algebraic {}

/// Differential variables must appear in balance equations.
///
/// Differential variables represent accumulations (mass, energy, momentum)
/// whose time derivatives define the dynamic behavior of the system.
#[derive(Debug, Clone, Copy)]
pub struct Differential;
impl VariableRole for Differential {}

/// Marker trait for time domain of models.
///
/// Models can be either steady-state (no time derivatives) or dynamic
/// (time derivatives allowed).
pub trait TimeDomain {
    /// Returns true if this is a steady-state model.
    const IS_STEADY: bool;
}

/// Steady-state models have no time variation.
///
/// All derivatives with respect to time are zero, and the system is at
/// equilibrium or steady operation.
#[derive(Debug)]
pub struct Steady;
impl TimeDomain for Steady {
    const IS_STEADY: bool = true;
}

/// Dynamic models evolve in time.
///
/// Differential variables have non-zero time derivatives, and the system
/// state changes according to balance equations.
#[derive(Debug)]
pub struct Dynamic;
impl TimeDomain for Dynamic {
    const IS_STEADY: bool = false;
}

/// Component Layer: Variables
/// A typed variable in a process model.
///
/// Variables now use an **index-based system** to enable shared state:
/// - Variables reference an ID in a central `VariableRegistry`
/// - Solvers can modify values through the registry
/// - Unit operations read updated values
/// - The type parameter `R` still enforces role-based correctness
///
/// This is the standard approach in industrial simulators:
/// - gPROMS: Variable IDs in global state
/// - Aspen: Central variable registry
/// - Modelica: Compiles to vector of state variables
///
/// # Variable Roles
///
/// - `Var<Parameter>`: Time-invariant constants (rate constants, physical properties)
/// - `Var<Algebraic>`: Computed variables without time derivatives (temperature, pressure)
/// - `Var<Differential>`: State variables with accumulation (holdup, energy)
///
/// # Examples
///
/// ```
/// use nomata::{VariableRegistry, Var, Algebraic, Differential};
///
/// let registry = VariableRegistry::new();
/// let temp: Var<Algebraic> = registry.create_algebraic(298.15);
/// let mass: Var<Differential> = registry.create_differential(1000.0);
///
/// // Variables share state through the registry
/// assert_eq!(temp.get(), 298.15);
/// temp.set(310.0);
/// assert_eq!(temp.get(), 310.0);
/// ```
#[derive(Debug, Clone)]
pub struct Var<R: VariableRole> {
    /// Index into the variable registry
    id: VarId,
    /// Reference to the shared registry
    registry: VariableRegistry,
    /// Phantom data to carry the role type
    _role: PhantomData<R>,
}

impl<R: VariableRole> Var<R> {
    /// Creates a new variable reference.
    ///
    /// **Internal use only.** Users should create variables through
    /// `VariableRegistry::create_*()` methods for proper shared state management.
    fn from_id(id: VarId, registry: VariableRegistry) -> Self {
        Var { id, registry, _role: PhantomData }
    }

    /// Creates a standalone variable (legacy API).
    ///
    /// ** WARNING**: This creates an isolated variable in its own registry.
    /// Changes won't be shared with solvers or other parts of the system.
    ///
    /// **Prefer**: `VariableRegistry::create_*()` for production code.
    ///
    /// This method exists for backward compatibility and simple examples.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Var, Algebraic};
    ///
    /// // Legacy API (isolated state)
    /// let temp: Var<Algebraic> = Var::new(298.15);
    ///
    /// // Preferred API (shared state)
    /// use nomata::VariableRegistry;
    /// let registry = VariableRegistry::new();
    /// let temp_shared: Var<Algebraic> = registry.create_algebraic(298.15);
    /// ```
    pub fn new(value: f64) -> Self {
        let registry = VariableRegistry::new();
        let id = registry.register(value, std::any::type_name::<R>());
        Var { id, registry, _role: PhantomData }
    }

    /// Gets the variable's ID.
    pub fn id(&self) -> VarId {
        self.id
    }

    /// Gets the current value of the variable.
    pub fn get(&self) -> f64 {
        self.registry.get(self.id)
    }

    /// Sets the value of the variable.
    pub fn set(&self, value: f64) {
        self.registry.set(self.id, value)
    }

    /// Gets a reference to the registry (for advanced use).
    pub fn registry(&self) -> &VariableRegistry {
        &self.registry
    }
}

/// Time derivative representation for differential variables.
///
/// This type can only be constructed from `Var<Differential>`, ensuring
/// that derivatives are only taken of differential variables.
pub struct Derivative {
    /// The time derivative value (dv/dt)
    pub value: f64,
}

impl Derivative {
    /// Creates a new derivative with the given rate of change.
    fn new(value: f64) -> Self {
        Derivative { value }
    }
}

/// Extension trait for differential variables.
///
/// Only implemented for `Var<Differential>`, this trait provides the
/// `derivative()` method that computes or represents the time derivative.
pub trait CanDifferentiate {
    /// Computes the time derivative of this variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Var, Differential, CanDifferentiate};
    ///
    /// let holdup: Var<Differential> = Var::new(100.0);
    /// let dhdt = holdup.derivative();
    /// ```
    fn derivative(&self) -> Derivative;
}

impl CanDifferentiate for Var<Differential> {
    fn derivative(&self) -> Derivative {
        // For differential variables, the derivative is tracked as part of the
        // time integration. This returns the current derivative state from the registry.
        // In a full implementation, derivatives would be computed by the ODE solver.
        let current_value = self.get();
        Derivative::new(current_value)
    }
}

// Note: The key invariant is that CanDifferentiate is NOT implemented for
// Var<Algebraic>, making it a compile-time error to call .derivative() on
// algebraic variables.

/// Component Layer: Streams
/// Marker trait for physical stream types.
///
/// Different stream types represent different physical quantities and
/// cannot be mixed without explicit conversion.
pub trait StreamType {}

/// Mass flow rate stream (kg/s or similar units).
#[derive(Debug, Clone, Copy)]
pub struct MassFlow;
impl StreamType for MassFlow {}

/// Molar flow rate stream (mol/s or similar units).
#[derive(Debug, Clone, Copy)]
pub struct MolarFlow;
impl StreamType for MolarFlow {}

/// Temperature stream (K or similar units).
#[derive(Debug, Clone, Copy)]
pub struct Temperature;
impl StreamType for Temperature {}

/// Pressure stream (Pa or similar units).
#[derive(Debug, Clone, Copy)]
pub struct Pressure;
impl StreamType for Pressure {}

/// Process stream with composition, temperature, and pressure tracking.
///
/// Streams carry material with component compositions and thermodynamic conditions.
/// The type parameter `S` enforces compatibility between connections (MolarFlow or MassFlow).
/// The type parameter `C` tracks whether temperature and pressure have been initialized.
///
/// Can represent both pure component and multi-component streams.
///
/// # Type Parameters
///
/// - `S`: Stream type (MolarFlow or MassFlow)
///
/// # Invariants
///
/// - Compositions must sum to 1.0 (enforced by validation methods)
/// - Number of components matches number of component names
/// - Component flows sum to total flow
///
/// # Examples
///
/// ```
/// use nomata::{Stream, MolarFlow, UninitializedConditions};
///
/// // Water-Ethanol mixture
/// let mut stream = Stream::<MolarFlow, UninitializedConditions>::new(
///     100.0,  // Total flow: 100 mol/s
///     vec!["Water".to_string(), "Ethanol".to_string()],
/// )
/// .at_conditions(298.15, 101325.0);
///
/// // Set composition (mole fractions)
/// stream.set_composition(vec![0.6, 0.4]).unwrap();
///
/// // Get individual component flows
/// assert_eq!(stream.component_flow(0), 60.0);  // Water: 60 mol/s
/// assert_eq!(stream.component_flow(1), 40.0);  // Ethanol: 40 mol/s
/// ```
/// Error type for stream-related operations.
#[derive(Error, Debug)]
pub enum StreamError {
    #[error(
        "Missing composition data for {model} outlet stream. Use {suggestion} to provide inlet composition."
    )]
    MissingComposition { model: String, suggestion: String },
    #[error("{model} requires composition calculation from {calculation_type}.")]
    RequiresCalculation { model: String, calculation_type: String },
    #[error("Stream not available: {reason}. {suggestion}")]
    NotAvailable { reason: String, suggestion: String },
}

/// Reference to an outlet stream that can be queried after solving.
///
/// This allows building a flowsheet declaratively without computing streams immediately.
/// After solving the flowsheet, call `get()` to retrieve the computed stream.
///
/// # Examples
///
/// ```
/// use nomata::{OutletRef, MolarFlow};
///
/// // Create reference (doesn't compute stream yet)
/// let outlet_ref: OutletRef<MolarFlow> = OutletRef::new("Heater-101", "outlet");
///
/// // Build flowsheet...
/// // Solve flowsheet...
///
/// // Then access stream
/// // let stream = outlet_ref.get()?;
/// ```
#[derive(Debug, Clone)]
pub struct OutletRef<S: StreamType + Clone = MolarFlow> {
    /// Unit name or ID
    unit_name: String,
    /// Port name
    port_name: String,
    /// Cached stream data (populated after solving)
    stream: std::cell::RefCell<Option<Stream<S>>>,
}

impl<S: StreamType + Clone> OutletRef<S> {
    /// Creates a new outlet reference.
    pub fn new(unit_name: &str, port_name: &str) -> Self {
        OutletRef {
            unit_name: unit_name.to_string(),
            port_name: port_name.to_string(),
            stream: std::cell::RefCell::new(None),
        }
    }

    /// Gets the stream data (after solving).
    pub fn get(&self) -> Result<Stream<S>, StreamError> {
        self.stream.borrow().as_ref().cloned().ok_or_else(|| StreamError::NotAvailable {
            reason: format!("Stream from {}.{} not computed", self.unit_name, self.port_name),
            suggestion: "Solve the flowsheet first".to_string(),
        })
    }

    /// Sets the stream data (internal, called by solver).
    pub fn set(&self, stream: Stream<S>) {
        *self.stream.borrow_mut() = Some(stream);
    }

    /// Checks if stream data is available.
    pub fn is_available(&self) -> bool {
        self.stream.borrow().is_some()
    }

    /// Clears the cached stream (e.g., before re-solving).
    pub fn clear(&self) {
        *self.stream.borrow_mut() = None;
    }
}

#[derive(Debug, Clone)]
pub struct Stream<S: StreamType, C = InitializedConditions> {
    /// Total flow rate [kg/s or mol/s depending on S]
    pub total_flow: f64,
    /// Component compositions [mass or mole fractions]
    pub composition: Vec<f64>,
    /// Component names or identifiers
    pub components: Vec<String>,
    /// Temperature \[K\]
    pub temperature: f64,
    /// Pressure \[Pa\]
    pub pressure: f64,
    /// Optional fluid object for thermodynamic calculations
    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<crate::thermodynamics::Fluid>,
    /// Phantom data to carry the stream type
    _stream_type: PhantomData<S>,
    /// Phantom data to track condition initialization
    _condition_state: PhantomData<C>,
}

/// Phantom type marker for uninitialized stream conditions.
#[derive(Debug, Clone, Copy)]
pub struct UninitializedConditions;

/// Phantom type marker for initialized stream conditions.
#[derive(Debug, Clone, Copy)]
pub struct InitializedConditions;

impl<S: StreamType> Stream<S, UninitializedConditions> {
    /// Creates a new multi-component stream with uninitialized conditions.
    ///
    /// Temperature and pressure must be set before the stream can be used.
    ///
    /// # Arguments
    ///
    /// * `total_flow` - Total flow rate
    /// * `components` - Component names/identifiers
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Stream, MolarFlow, UninitializedConditions};
    ///
    /// let stream = Stream::<MolarFlow, UninitializedConditions>::new(
    ///     100.0,
    ///     vec!["N2".to_string(), "O2".to_string(), "Ar".to_string()],
    /// )
    /// .at_conditions(298.15, 101325.0);  // Must set conditions!
    /// ```
    pub fn new(total_flow: f64, components: Vec<String>) -> Self {
        let n = components.len();
        Stream {
            total_flow,
            composition: vec![0.0; n],
            components,
            temperature: 0.0,
            pressure: 0.0,
            #[cfg(feature = "thermodynamics")]
            fluid: None,
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        }
    }

    /// Creates a stream with specified composition (conditions still uninitialized).
    ///
    /// # Arguments
    ///
    /// * `total_flow` - Total flow rate
    /// * `components` - Component names
    /// * `composition` - Component fractions (must sum to 1.0)
    pub fn with_composition(
        total_flow: f64,
        components: impl IntoIterator<Item = impl Into<String>>,
        composition: Vec<f64>,
    ) -> Result<Self, String> {
        let component_vec: Vec<String> = components.into_iter().map(|s| s.into()).collect();
        let n = component_vec.len();
        if composition.len() != n {
            return Err(format!(
                "Composition length {} does not match {} components",
                composition.len(),
                n
            ));
        }
        if composition.iter().any(|&x| x < 0.0) {
            return Err("Composition fractions cannot be negative".to_string());
        }
        let sum: f64 = composition.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("Composition sum {} deviates from 1.0 (tolerance: 1e-6)", sum));
        }
        Ok(Stream {
            total_flow,
            composition,
            components: component_vec,
            temperature: 0.0,
            pressure: 0.0,
            #[cfg(feature = "thermodynamics")]
            fluid: None,
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        })
    }

    /// Sets temperature and pressure, transitioning to initialized state.
    pub fn at_conditions(
        self,
        temperature: f64,
        pressure: f64,
    ) -> Stream<S, InitializedConditions> {
        Stream {
            total_flow: self.total_flow,
            composition: self.composition,
            components: self.components,
            temperature,
            pressure,
            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        }
    }
}

// Methods available for both states
impl<S: StreamType, C> Stream<S, C> {
    /// Number of components in the stream.
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Gets the name of a specific component.
    pub fn component_name(&self, index: usize) -> &str {
        &self.components[index]
    }

    /// Gets the composition of a specific component.
    pub fn get_composition(&self, index: usize) -> f64 {
        self.composition[index]
    }
}

// Methods only available for initialized streams
impl<S: StreamType> Stream<S, InitializedConditions> {
    /// Creates a pure component stream with specified conditions.
    pub fn pure(total_flow: f64, component: String, temperature: f64, pressure: f64) -> Self {
        Stream {
            total_flow,
            composition: vec![1.0],
            components: vec![component],
            temperature,
            pressure,
            #[cfg(feature = "thermodynamics")]
            fluid: None,
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        }
    }

    /// Creates a new stream from a thermodynamics Fluid (requires thermodynamics feature).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::{Stream, MolarFlow, InitializedConditions, thermodynamics::{Fluid, fluids::Pure}};
    /// use std::collections::HashMap;
    ///
    /// // Pure component stream
    /// let water = Fluid::new(Pure::Water);
    /// let stream = Stream::<MolarFlow, InitializedConditions>::from_fluid(
    ///     100.0,
    ///     &water,
    ///     300.0,
    ///     101325.0,
    /// );
    ///
    /// // Multi-component mixture
    /// let air = Fluid::new_mole_based(
    ///     "CustomAir",
    ///     HashMap::from([
    ///         (Pure::Nitrogen, 0.79),
    ///         (Pure::Oxygen, 0.21),
    ///     ]),
    /// ).unwrap();
    /// let stream = Stream::<MolarFlow, InitializedConditions>::from_fluid(
    ///     150.0,
    ///     &air,
    ///     298.15,
    ///     101325.0,
    /// );
    /// ```
    #[cfg(feature = "thermodynamics")]
    pub fn from_fluid(
        total_flow: f64,
        fluid: &crate::thermodynamics::Fluid,
        temperature: f64,
        pressure: f64,
    ) -> Self {
        let (components, composition) = fluid.get_components();

        Stream {
            total_flow,
            composition,
            components,
            temperature,
            pressure,
            fluid: Some(fluid.clone()),
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        }
    }

    /// Creates a new stream with components and their compositions (requires thermodynamics feature).
    ///
    /// This is the primary constructor for creating streams with thermodynamic support.
    /// It accepts a vector of (component, mole fraction) tuples, enabling both pure
    /// component streams and arbitrary multi-component mixtures.
    ///
    /// # Arguments
    ///
    /// * `total_flow` - Total flow rate in the stream's units (mol/s or kg/s)
    /// * `components` - Vector of (component, mole fraction) pairs
    /// * `temperature` - Temperature in Kelvin
    /// * `pressure` - Pressure in Pascals
    ///
    /// # Type-Safety
    ///
    /// Components can be:
    /// - Pure fluids: `Pure::Water`, `Pure::Nitrogen`, etc.
    /// - Predefined mixtures: `PredefinedMix::Air`, etc.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use nomata::{Stream, MolarFlow, InitializedConditions, Pure};
    ///
    /// // Pure component stream
    /// let water = Stream::<MolarFlow, InitializedConditions>::new(
    ///     100.0,
    ///     vec![(Pure::Water, 1.0)],
    ///     300.0,
    ///     101325.0,
    /// );
    ///
    /// // Multi-component mixture (custom air composition)
    /// let air = Stream::<MolarFlow, InitializedConditions>::new(
    ///     150.0,
    ///     vec![
    ///         (Pure::Nitrogen, 0.79),
    ///         (Pure::Oxygen, 0.21),
    ///     ],
    ///     298.15,
    ///     101325.0,
    /// );
    ///
    /// // Natural gas mixture
    /// let natural_gas = Stream::<MolarFlow, InitializedConditions>::new(
    ///     200.0,
    ///     vec![
    ///         (Pure::Methane, 0.95),
    ///         (Pure::Ethane, 0.03),
    ///         (Pure::Propane, 0.02),
    ///     ],
    ///     288.15,
    ///     5e6,
    /// );
    /// ```
    #[cfg(feature = "thermodynamics")]
    pub fn new(
        total_flow: f64,
        components: Vec<(crate::thermodynamics::fluids::Pure, f64)>,
        temperature: f64,
        pressure: f64,
    ) -> Self {
        use crate::thermodynamics::Fluid;
        use std::collections::HashMap;

        let component_names: Vec<String> =
            components.iter().map(|(pure, _)| format!("{:?}", pure)).collect();

        let composition: Vec<f64> = components.iter().map(|(_, fraction)| *fraction).collect();

        // Create Fluid object
        let fluid = if components.len() == 1 {
            Some(Fluid::new(components[0].0))
        } else {
            let comp_map: HashMap<_, _> = components.into_iter().collect();
            Fluid::new_mole_based("mixture", comp_map).ok()
        };

        Stream {
            total_flow,
            composition,
            components: component_names,
            temperature,
            pressure,
            fluid,
            _stream_type: PhantomData,
            _condition_state: PhantomData,
        }
    }

    /// Sets the composition (mole or mass fractions).
    ///
    /// # Arguments
    ///
    /// * `composition` - Component fractions (must sum to 1.0 +- tolerance)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Length doesn't match number of components
    /// - Sum deviates from 1.0 by more than 1e-6
    /// - Any fraction is negative
    pub fn set_composition(&mut self, composition: Vec<f64>) -> Result<(), String> {
        if composition.len() != self.n_components() {
            return Err(format!(
                "Composition length {} does not match {} components",
                composition.len(),
                self.n_components()
            ));
        }

        // Check for negative values
        if composition.iter().any(|&x| x < 0.0) {
            return Err("Composition fractions cannot be negative".to_string());
        }

        // Check sum
        let sum: f64 = composition.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(format!("Composition sum {} deviates from 1.0 (tolerance: 1e-6)", sum));
        }

        self.composition = composition;
        Ok(())
    }

    /// Gets the flow rate of a specific component.
    ///
    /// # Arguments
    ///
    /// * `index` - Component index
    ///
    /// # Returns
    ///
    /// Component flow = total_flow * composition\[index\]
    pub fn component_flow(&self, index: usize) -> f64 {
        self.total_flow * self.composition[index]
    }

    /// Gets all component flows.
    pub fn component_flows(&self) -> Vec<f64> {
        self.composition.iter().map(|&x| self.total_flow * x).collect()
    }

    /// Sets component flows directly (recomputes total and composition).
    ///
    /// # Arguments
    ///
    /// * `flows` - Individual component flows
    pub fn set_component_flows(&mut self, flows: Vec<f64>) -> Result<(), String> {
        if flows.len() != self.n_components() {
            return Err(format!(
                "Flow vector length {} does not match {} components",
                flows.len(),
                self.n_components()
            ));
        }

        self.total_flow = flows.iter().sum();

        if self.total_flow == 0.0 {
            self.composition = vec![0.0; self.n_components()];
        } else {
            self.composition = flows.iter().map(|&f| f / self.total_flow).collect();
        }

        Ok(())
    }

    /// Validates that the stream is physically consistent.
    ///
    /// Checks:
    /// - Composition sums to 1.0
    /// - No negative values
    /// - Total flow is non-negative
    pub fn validate(&self) -> Result<(), String> {
        if self.total_flow < 0.0 {
            return Err("Total flow cannot be negative".to_string());
        }

        if self.composition.iter().any(|&x| x < 0.0) {
            return Err("Composition fractions cannot be negative".to_string());
        }

        let sum: f64 = self.composition.iter().sum();
        if (sum - 1.0).abs() > 1e-6 && self.total_flow > 0.0 {
            return Err(format!("Composition sum {} deviates from 1.0 (tolerance: 1e-6)", sum));
        }

        Ok(())
    }
}

/// Trait for unit operations that can be automatically initialized from stream data.
///
/// When a stream is connected to a unit, this trait allows automatic property computation
/// from the stream's thermodynamic state (P, T, composition).
///
/// # Examples
///
/// ```
/// use nomata::{Stream, MolarFlow, FromStream, StreamType};
///
/// // Implement for your unit operation
/// struct MyUnit {
///     temp: f64,
/// }
///
/// impl FromStream for MyUnit {
///     fn from_stream<S: StreamType, C>(stream: &Stream<S, C>) -> Self {
///         MyUnit { temp: stream.temperature }
///     }
/// }
/// ```
pub trait FromStream: Sized {
    /// Create an initialized unit from stream data.
    ///
    /// This method should extract relevant properties (flow, temperature, etc.)
    /// from the stream and return a fully initialized unit.
    fn from_stream<S: StreamType, C>(stream: &Stream<S, C>) -> Self;
}

/// Graph Layer: Unit Operations
/// Port Ownership and Linear Connection Tracking
///
/// Marker trait for port states in linear type tracking.
///
/// Ports can be in two states:
/// - `Disconnected`: Available for connection
/// - `Connected`: Already connected (cannot be reused)
///
/// This enables compile-time enforcement of "connect exactly once" semantics.
pub trait PortState {}

/// Port is available for connection.
#[derive(Debug, Clone, Copy)]
pub struct Disconnected;
impl PortState for Disconnected {}

/// Port is already connected (consumed).
#[derive(Debug, Clone, Copy)]
pub struct Connected;
impl PortState for Connected {}

/// A typed port with ownership tracking.
///
/// Ports are parametrized by:
/// - `S`: Stream type (e.g., `MolarFlow`, `MassFlow`)
/// - `D`: Direction marker (`Input` or `Output`)
/// - `P`: Port state (`Disconnected` or `Connected`)
///
/// # Linear Usage
///
/// Ports follow linear type semantics - they can only be connected once.
/// After connection, the port transitions from `Disconnected` to `Connected`,
/// preventing reuse.
///
/// # Examples
///
/// ```
/// use nomata::{Port, Input, Output, Disconnected, MolarFlow, Stream};
///
/// // Create disconnected ports
/// let inlet: Port<Stream<MolarFlow>, Input, Disconnected> = Port::new();
/// let outlet: Port<Stream<MolarFlow>, Output, Disconnected> = Port::new();
///
/// // After connection, ports become Connected and cannot be reused
/// ```
pub struct Port<S, D, P: PortState> {
    _stream: PhantomData<S>,
    _direction: PhantomData<D>,
    _state: PhantomData<P>,
}

impl<S, D, P: PortState> Port<S, D, P> {
    /// Creates a new port in the given state.
    pub fn new() -> Self {
        Port { _stream: PhantomData, _direction: PhantomData, _state: PhantomData }
    }
}

impl<S, D, P: PortState> Default for Port<S, D, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker type for input ports.
#[derive(Debug, Clone, Copy)]
pub struct Input;

/// Marker type for output ports.
#[derive(Debug, Clone, Copy)]
pub struct Output;

/// Connects two ports with linear ownership transfer.
///
/// This function:
/// 1. Enforces stream type compatibility (output -> input)
/// 2. Requires both ports to be `Disconnected`
/// 3. Consumes both ports, returning them in `Connected` state
/// 4. Prevents double-connection at compile time
///
/// # Type Parameters
///
/// - `S`: Stream type (must match between output and input)
///
/// # Examples
///
/// ```
/// use nomata::{Port, Input, Output, Disconnected, MolarFlow, Stream, connect_ports};
///
/// let outlet: Port<Stream<MolarFlow>, Output, Disconnected> = Port::new();
/// let inlet: Port<Stream<MolarFlow>, Input, Disconnected> = Port::new();
///
/// // Connect the ports (consumes both, returns Connected versions)
/// let (outlet_connected, inlet_connected) = connect_ports(outlet, inlet);
///
/// // Cannot connect again - ports are consumed!
/// // connect_ports(outlet_connected, inlet_connected); // Won't compile
/// ```
pub fn connect_ports<S>(
    _output: Port<S, Output, Disconnected>,
    _input: Port<S, Input, Disconnected>,
) -> (Port<S, Output, Connected>, Port<S, Input, Connected>) {
    // Ports are consumed (moved), preventing reuse
    let output_connected =
        Port { _stream: PhantomData, _direction: PhantomData, _state: PhantomData };
    let input_connected =
        Port { _stream: PhantomData, _direction: PhantomData, _state: PhantomData };
    (output_connected, input_connected)
}

/// Compile-Time Port Counting with Const Generics (Rust 2024)
///
/// Unit operation with compile-time port count enforcement.
///
/// This leverages Rust 2024's const generics to encode the number of input
/// and output ports in the type system, preventing port count mismatches
/// at compile time.
///
/// # Type Parameters
///
/// * `IN` - Number of input ports (const generic)
/// * `OUT` - Number of output ports (const generic)
/// * `S` - Stream type
///
/// # Examples
///
/// ```
/// use nomata::{UnitWithPortCounts, Stream, MolarFlow};
///
/// // Mixer: 2 inputs, 1 output
/// type Mixer = UnitWithPortCounts<2, 1, Stream<MolarFlow>>;
/// let mixer = Mixer::new();
/// assert_eq!(mixer.input_count(), 2);
/// assert_eq!(mixer.output_count(), 1);
///
/// // Splitter: 1 input, 2 outputs
/// type Splitter = UnitWithPortCounts<1, 2, Stream<MolarFlow>>;
/// let splitter = Splitter::new();
/// assert_eq!(splitter.input_count(), 1);
/// assert_eq!(splitter.output_count(), 2);
/// ```
///
/// # Compile-Time Safety
///
/// ```compile_fail
/// use nomata::{UnitWithPortCounts, Stream, MolarFlow};
///
/// // This would fail to compile - array size mismatch
/// let unit: UnitWithPortCounts<2, 1, Stream<MolarFlow>> = UnitWithPortCounts::new();
/// let wrong_inputs: [f64; 3] = [1.0, 2.0, 3.0]; // Expected 2, got 3
/// unit.set_inputs(wrong_inputs); // Won't compile!
/// ```
#[derive(Debug, Clone)]
pub struct UnitWithPortCounts<const IN: usize, const OUT: usize, S> {
    /// Input port values (size enforced by const generic)
    pub inputs: [f64; IN],
    /// Output port values (size enforced by const generic)
    pub outputs: [f64; OUT],
    /// Stream type marker
    _stream_type: PhantomData<S>,
}

impl<const IN: usize, const OUT: usize, S> UnitWithPortCounts<IN, OUT, S> {
    /// Creates a new unit with all ports initialized to zero.
    pub fn new() -> Self {
        UnitWithPortCounts { inputs: [0.0; IN], outputs: [0.0; OUT], _stream_type: PhantomData }
    }

    /// Returns the number of input ports (compile-time constant).
    pub const fn input_count(&self) -> usize {
        IN
    }

    /// Returns the number of output ports (compile-time constant).
    pub const fn output_count(&self) -> usize {
        OUT
    }

    /// Sets input port values (size enforced at compile time).
    pub fn set_inputs(&mut self, values: [f64; IN]) {
        self.inputs = values;
    }

    /// Sets output port values (size enforced at compile time).
    pub fn set_outputs(&mut self, values: [f64; OUT]) {
        self.outputs = values;
    }

    /// Gets a reference to input port values.
    pub fn get_inputs(&self) -> &[f64; IN] {
        &self.inputs
    }

    /// Gets a reference to output port values.
    pub fn get_outputs(&self) -> &[f64; OUT] {
        &self.outputs
    }
}

impl<const IN: usize, const OUT: usize, S> Default for UnitWithPortCounts<IN, OUT, S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for common unit operation configurations.
pub mod port_configs {
    use super::*;

    /// Single input, single output (SISO) unit
    /// Examples: Reactor, Heat Exchanger, Pump, Compressor
    pub type SISO<S> = UnitWithPortCounts<1, 1, S>;

    /// Multiple input, single output (MISO) unit
    /// Examples: Mixer (2 inputs -> 1 output)
    pub type Mixer2<S> = UnitWithPortCounts<2, 1, S>;

    /// Multiple input, single output (3 inputs)
    /// Examples: Three-way mixer
    pub type Mixer3<S> = UnitWithPortCounts<3, 1, S>;

    /// Single input, multiple output (SIMO) unit
    /// Examples: Splitter (1 input -> 2 outputs)
    pub type Splitter2<S> = UnitWithPortCounts<1, 2, S>;

    /// Single input, multiple output (3 outputs)
    /// Examples: Three-way splitter
    pub type Splitter3<S> = UnitWithPortCounts<1, 3, S>;

    /// Distillation column: 1 feed, 2 products (overhead + bottoms)
    pub type Distillation<S> = UnitWithPortCounts<1, 2, S>;

    /// Flash separator: 1 feed, 2 products (vapor + liquid)
    pub type Flash<S> = UnitWithPortCounts<1, 2, S>;

    /// Heat exchanger with bypass: 2 inputs (hot + cold), 2 outputs
    pub type HeatExchangerWithBypass<S> = UnitWithPortCounts<2, 2, S>;
}

/// Port-Based Architecture for Multi-Port Units
///
/// A named port with runtime identification.
///
/// Unlike the type-level `Port<S, D, P>`, this represents a port at runtime
/// with a unique name and stream type information.
#[derive(Debug, Clone)]
pub struct NamedPort {
    /// Port name (e.g., "inlet", "outlet1", "overhead", "bottoms")
    pub name: String,
    /// Direction: true for input, false for output
    pub is_input: bool,
    /// Stream type identifier (for runtime checking)
    pub stream_type: &'static str,
}

impl NamedPort {
    /// Creates a new input port.
    pub fn input(name: &str, stream_type: &'static str) -> Self {
        NamedPort { name: name.to_string(), is_input: true, stream_type }
    }

    /// Creates a new output port.
    pub fn output(name: &str, stream_type: &'static str) -> Self {
        NamedPort { name: name.to_string(), is_input: false, stream_type }
    }
}

/// Port-based unit operation trait.
///
/// This trait allows units to have multiple input and output ports,
/// addressing the limitation of the simple `UnitOp` trait which assumes
/// exactly one input and one output.
///
/// # Design Motivation
///
/// Real process units often have multiple ports:
/// - **Mixer**: Multiple inlets, one outlet
/// - **Splitter**: One inlet, multiple outlets
/// - **Distillation Column**: One feed, overhead + bottoms
/// - **Heat Exchanger**: Hot/cold inlet/outlet pairs
///
/// # Examples
///
/// ```
/// use nomata::{HasPorts, NamedPort};
///
/// struct Mixer {
///     num_inlets: usize,
/// }
///
/// impl HasPorts for Mixer {
///     fn input_ports(&self) -> Vec<NamedPort> {
///         (0..self.num_inlets)
///             .map(|i| NamedPort::input(&format!("inlet_{}", i), "MolarFlow"))
///             .collect()
///     }
///
///     fn output_ports(&self) -> Vec<NamedPort> {
///         vec![NamedPort::output("outlet", "MolarFlow")]
///     }
/// }
/// ```
pub trait HasPorts {
    /// Returns all input ports for this unit.
    fn input_ports(&self) -> Vec<NamedPort>;

    /// Returns all output ports for this unit.
    fn output_ports(&self) -> Vec<NamedPort>;

    /// Gets a specific input port by name.
    fn get_input_port(&self, name: &str) -> Option<NamedPort> {
        self.input_ports().into_iter().find(|p| p.name == name)
    }

    /// Gets a specific output port by name.
    fn get_output_port(&self, name: &str) -> Option<NamedPort> {
        self.output_ports().into_iter().find(|p| p.name == name)
    }

    /// Gets the total number of ports.
    fn port_count(&self) -> (usize, usize) {
        (self.input_ports().len(), self.output_ports().len())
    }
}

/// Port-specific connection information.
///
/// Unlike the unit-level connection in `connect()`, this tracks
/// connections between specific named ports on units.
#[derive(Debug, Clone)]
pub struct PortConnection {
    /// Source unit name/ID
    pub from_unit: String,
    /// Source port name
    pub from_port: String,
    /// Destination unit name/ID
    pub to_unit: String,
    /// Destination port name
    pub to_port: String,
    /// Stream type (must match)
    pub stream_type: &'static str,
}

impl PortConnection {
    /// Creates a new port connection.
    pub fn new(
        from_unit: &str,
        from_port: &str,
        to_unit: &str,
        to_port: &str,
        stream_type: &'static str,
    ) -> Self {
        PortConnection {
            from_unit: from_unit.to_string(),
            from_port: from_port.to_string(),
            to_unit: to_unit.to_string(),
            to_port: to_port.to_string(),
            stream_type,
        }
    }
}

/// Connects two specific ports on different units.
///
/// This is the port-specific version of `connect()` that allows
/// connecting multi-port units like mixers and splitters.
///
/// # Examples
///
/// ```ignore
/// use nomata::{connect_named_ports, HasPorts};
///
/// let mixer = Mixer::new(3);  // 3 inlets, 1 outlet
/// let reactor = Reactor::new();
///
/// // Connect mixer's outlet to reactor's inlet
/// let conn = connect_named_ports(
///     &mixer, "outlet",
///     &reactor, "inlet",
/// ).expect("Port connection failed");
/// ```
pub fn connect_named_ports<U1, U2>(
    upstream: &U1,
    upstream_port: &str,
    downstream: &U2,
    downstream_port: &str,
) -> Result<PortConnection, String>
where
    U1: HasPorts,
    U2: HasPorts,
{
    // Get the output port from upstream unit
    let out_port = upstream
        .get_output_port(upstream_port)
        .ok_or_else(|| format!("Output port '{}' not found on upstream unit", upstream_port))?;

    // Get the input port from downstream unit
    let in_port = downstream
        .get_input_port(downstream_port)
        .ok_or_else(|| format!("Input port '{}' not found on downstream unit", downstream_port))?;

    // Check stream type compatibility
    if out_port.stream_type != in_port.stream_type {
        return Err(format!(
            "Stream type mismatch: {} (output) vs {} (input)",
            out_port.stream_type, in_port.stream_type
        ));
    }

    // Create connection record
    Ok(PortConnection::new(
        "upstream", // In real use, pass actual unit names
        upstream_port,
        "downstream",
        downstream_port,
        out_port.stream_type,
    ))
}

/// Unified port specification trait for compile-time port checking.
///
/// This trait allows units to declare their ports at compile time,
/// enabling type-safe connections with const generic port indices.
///
/// # Type Parameters
///
/// * `IN`: Number of input ports
/// * `OUT`: Number of output ports
/// * `S`: Stream type for all ports (homogeneous for simplicity)
pub trait PortSpec {
    /// Number of input ports.
    const INPUT_COUNT: usize;

    /// Number of output ports.
    const OUTPUT_COUNT: usize;

    /// Stream type identifier.
    const STREAM_TYPE: &'static str;
}

/// Unified connection function with compile-time port index checking.
///
/// This function provides type-safe connections for units that implement `PortSpec`,
/// ensuring port indices are valid at compile time while supporting multiple ports.
///
/// # Examples
///
/// ```ignore
/// use nomata::{connect, PortSpec};
///
/// // Units implementing PortSpec
/// let mixer: Mixer<2> = Mixer::new(1);  // 2 inputs, 1 output
/// let reactor: Reactor = Reactor::new();  // 1 input, 1 output
///
/// // Connect with compile-time index checking
/// connect(&mixer, 0, &reactor, 0);  // outlet 0 -> inlet 0
/// ```
///
/// # Type Safety
///
/// - Port indices must be const (compile-time known)
/// - Index bounds are checked at compile time
/// - Stream types must match
pub fn connect<U1, U2>(
    upstream: &U1,
    upstream_port_idx: usize,
    downstream: &U2,
    downstream_port_idx: usize,
) -> Result<PortConnection, String>
where
    U1: PortSpec + HasPorts,
    U2: PortSpec + HasPorts,
{
    // Compile-time bounds checking
    if upstream_port_idx >= U1::OUTPUT_COUNT {
        return Err(format!(
            "Output port index {} out of bounds (max {})",
            upstream_port_idx,
            U1::OUTPUT_COUNT - 1
        ));
    }
    if downstream_port_idx >= U2::INPUT_COUNT {
        return Err(format!(
            "Input port index {} out of bounds (max {})",
            downstream_port_idx,
            U2::INPUT_COUNT - 1
        ));
    }

    // Stream type compatibility check
    if U1::STREAM_TYPE != U2::STREAM_TYPE {
        return Err(format!(
            "Stream type mismatch: {} (output) vs {} (input)",
            U1::STREAM_TYPE,
            U2::STREAM_TYPE
        ));
    }

    // Get port names from HasPorts implementation
    let upstream_ports = upstream.output_ports();
    let downstream_ports = downstream.input_ports();

    if upstream_port_idx >= upstream_ports.len() || downstream_port_idx >= downstream_ports.len() {
        return Err("Port index mismatch with HasPorts implementation".to_string());
    }

    let upstream_port = &upstream_ports[upstream_port_idx];
    let downstream_port = &downstream_ports[downstream_port_idx];

    // Create connection record
    Ok(PortConnection::new(
        "upstream",
        &upstream_port.name,
        "downstream",
        &downstream_port.name,
        upstream_port.stream_type,
    ))
}

/// Unit operation with explicit port tracking (legacy).
///
/// This trait extends the basic `UnitOp` with explicit port identity.
/// Implementors must specify the number and types of their ports.
///
/// **Note**: This is retained for backward compatibility. For new code,
/// prefer `HasPorts` which supports multi-port units.
///
/// # Examples
///
/// ```ignore
/// struct Reactor {
///     inlet: Port<Stream<MolarFlow>, Input, Disconnected>,
///     outlet: Port<Stream<MolarFlow>, Output, Disconnected>,
/// }
///
/// impl UnitOpWithPorts for Reactor {
///     type Ports = (Port<Stream<MolarFlow>, Input, Disconnected>,
///                   Port<Stream<MolarFlow>, Output, Disconnected>);
/// }
/// ```
pub trait UnitOpWithPorts {
    /// The ports of this unit operation as a tuple.
    /// Use tuples for multiple ports: `(Port<S1, I>, Port<S2, O>)`
    type Ports;
}

/// Trait for unit operations in a process flowsheet (legacy single-port).
///
/// Each unit operation must declare its input and output stream types,
/// enabling compile-time verification of connections.
///
/// # Limitations
///
/// This trait assumes each unit has **exactly one input and one output**.
/// This works for simple units like reactors and heat exchangers, but
/// fails for multi-port units:
///
/// -  **Mixers**: Multiple inputs, one output
/// -  **Splitters**: One input, multiple outputs
/// -  **Distillation**: One feed, multiple products (overhead + bottoms)
///
/// # Migration Path
///
/// For units with multiple ports, implement `HasPorts` instead:
///
/// ```ignore
/// // OLD: Single-port unit
/// impl UnitOp for Reactor {
///     type In = Stream<MolarFlow>;
///     type Out = Stream<MolarFlow>;
/// }
///
/// // NEW: Multi-port unit
/// impl HasPorts for Mixer {
///     fn input_ports(&self) -> Vec<NamedPort> {
///         vec![
///             NamedPort::input("inlet1", "MolarFlow"),
///             NamedPort::input("inlet2", "MolarFlow"),
///         ]
///     }
///     fn output_ports(&self) -> Vec<NamedPort> {
///         vec![NamedPort::output("outlet", "MolarFlow")]
///     }
/// }
/// ```
///
/// # Examples
///
/// ```ignore
/// use nomata::{UnitOp, Stream, MolarFlow};
/// use std::marker::PhantomData;
///
/// pub struct Reactor {
///     _marker: PhantomData<()>,
/// }
///
/// impl UnitOp for Reactor {
///     type In = Stream<MolarFlow>;
///     type Out = Stream<MolarFlow>;
/// }
/// ```
/// Component System: For hierarchical modules with isolated component sets
/// Represents a set of chemical components for a process unit or module.
///
/// Component sets allow different parts of a flowsheet to track different
/// species, reducing equation complexity. For example:
/// - C2 Splitter module: tracks {ethane, ethylene}
/// - C3 Splitter module: tracks {propane, propylene}
///
/// # Examples
///
/// ```
/// use nomata::ComponentSet;
///
/// let c2_components = ComponentSet::new(vec!["ethane", "ethylene"]);
/// let c3_components = ComponentSet::new(vec!["propane", "propylene"]);
///
/// assert_eq!(c2_components.count(), 2);
/// assert!(c2_components.contains("ethane"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```

#[derive(Debug, Clone)]
pub struct ComponentSet {
    components: Vec<String>,
}

impl ComponentSet {
    /// Creates a new component set from a list of component names.
    pub fn new(components: Vec<&str>) -> Self {
        ComponentSet { components: components.iter().map(|s| s.to_string()).collect() }
    }

    /// Creates an empty component set.
    pub fn empty() -> Self {
        ComponentSet { components: Vec::new() }
    }

    /// Returns the number of components in this set.
    pub fn count(&self) -> usize {
        self.components.len()
    }

    /// Checks if a component is in this set.
    pub fn contains(&self, component: &str) -> bool {
        self.components.iter().any(|c| c == component)
    }

    /// Returns a slice of all component names.
    pub fn components(&self) -> &[String] {
        &self.components
    }

    /// Adds a component to the set if it doesn't already exist.
    pub fn add(&mut self, component: &str) {
        if !self.contains(component) {
            self.components.push(component.to_string());
        }
    }
}

/// Component Mapping: Lumping and Delumping
///
/// Direction of component transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingDirection {
    /// Lumping: Combine detailed components into lumped representation
    /// Example: MA + PD -> MAPD
    Lumping,
    /// Delumping: Split lumped component into detailed components
    /// Example: MAPD -> MA + PD
    Delumping,
}

/// A rule defining how components map between representations.
///
/// Component mappings enable interfacing between models with different
/// levels of detail. Common use cases:
/// - Lumping similar species to reduce equation count
/// - Delumping pseudo-components at module boundaries
/// - Interfacing detailed and simplified models
///
/// # Examples
///
/// ```
/// use nomata::ComponentMapping;
///
/// // MAPD splits into MA and PD with 60:40 ratio
/// let mapping = ComponentMapping::new(
///     "MAPD",
///     vec![("MA", 0.6), ("PD", 0.4)]
/// );
///
/// assert_eq!(mapping.lumped_component(), "MAPD");
/// assert_eq!(mapping.detailed_count(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ComponentMapping {
    /// The lumped (pseudo) component name
    lumped: String,
    /// Detailed components with their split/combine fractions
    /// For delumping: fractions sum to 1.0 (how to split)
    /// For lumping: relative weights (how to combine)
    detailed: Vec<(String, f64)>,
}

impl ComponentMapping {
    /// Creates a new component mapping.
    ///
    /// # Arguments
    ///
    /// * `lumped` - Name of the lumped pseudo-component
    /// * `detailed` - List of (component_name, fraction) pairs
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::ComponentMapping;
    ///
    /// // C4 lumps into C4_light and C4_heavy
    /// let c4_mapping = ComponentMapping::new(
    ///     "C4_mix",
    ///     vec![("n-butane", 0.7), ("iso-butane", 0.3)]
    /// );
    /// ```
    pub fn new(lumped: &str, detailed: Vec<(&str, f64)>) -> Self {
        ComponentMapping {
            lumped: lumped.to_string(),
            detailed: detailed.iter().map(|(name, frac)| (name.to_string(), *frac)).collect(),
        }
    }

    /// Returns the lumped component name.
    pub fn lumped_component(&self) -> &str {
        &self.lumped
    }

    /// Returns the detailed components and their fractions.
    pub fn detailed_components(&self) -> &[(String, f64)] {
        &self.detailed
    }

    /// Returns the number of detailed components.
    pub fn detailed_count(&self) -> usize {
        self.detailed.len()
    }

    /// Gets the fraction for a specific detailed component.
    pub fn get_fraction(&self, component: &str) -> Option<f64> {
        self.detailed.iter().find(|(name, _)| name == component).map(|(_, frac)| *frac)
    }

    /// Validates that fractions sum to approximately 1.0.
    pub fn validate_fractions(&self) -> bool {
        let sum: f64 = self.detailed.iter().map(|(_, frac)| frac).sum();
        (sum - 1.0).abs() < 1e-6
    }
}

/// A mapper unit operation that transforms between component representations.
///
/// This unit performs lumping (combining components) or delumping (splitting
/// components) operations, enabling connections between modules with different
/// component detail levels.
///
/// # Physics
///
/// The mapper conserves total mass/moles:
/// - **Lumping**: F_lumped = Σ(F_detailed_i)
/// - **Delumping**: F_detailed_i = fraction_i * F_lumped
///
/// # Examples
///
/// ```
/// use nomata::{ComponentMapper, ComponentMapping, MappingDirection, Steady};
///
/// let mut mapper = ComponentMapper::<Steady>::new("MAPD_to_MA_PD");
/// mapper.set_direction(MappingDirection::Delumping);
///
/// // Define the mapping: MAPD -> 60% MA + 40% PD
/// let mapping = ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]);
/// mapper.add_mapping(mapping);
///
/// assert_eq!(mapper.mapping_count(), 1);
/// assert_eq!(mapper.direction(), MappingDirection::Delumping);
/// ```
#[derive(Debug)]
pub struct ComponentMapper<T: TimeDomain> {
    /// Mapper name
    name: String,
    /// Transformation direction
    direction: MappingDirection,
    /// Component mapping rules
    mappings: Vec<ComponentMapping>,
    /// Inlet flowrate (lumped or detailed, depending on direction)
    inlet_flow: Var<Algebraic>,
    /// Outlet flowrate (detailed or lumped, depending on direction)
    outlet_flow: Var<Algebraic>,
    /// Phantom data for time domain
    _time_domain: PhantomData<T>,
}

impl<T: TimeDomain> ComponentMapper<T> {
    /// Creates a new component mapper.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{ComponentMapper, Dynamic};
    ///
    /// let mapper = ComponentMapper::<Dynamic>::new("C4_lumper");
    /// assert_eq!(mapper.name(), "C4_lumper");
    /// ```
    pub fn new(name: &str) -> Self {
        let registry = VariableRegistry::new();
        ComponentMapper {
            name: name.to_string(),
            direction: MappingDirection::Lumping,
            mappings: Vec::new(),
            inlet_flow: registry.create_algebraic(0.0),
            outlet_flow: registry.create_algebraic(0.0),
            _time_domain: PhantomData,
        }
    }

    /// Gets the mapper name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the transformation direction.
    pub fn set_direction(&mut self, direction: MappingDirection) {
        self.direction = direction;
    }

    /// Gets the transformation direction.
    pub fn direction(&self) -> MappingDirection {
        self.direction
    }

    /// Adds a component mapping rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{ComponentMapper, ComponentMapping, Steady};
    ///
    /// let mut mapper = ComponentMapper::<Steady>::new("mapper");
    /// mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));
    /// mapper.add_mapping(ComponentMapping::new("C4", vec![("n-C4", 0.7), ("i-C4", 0.3)]));
    ///
    /// assert_eq!(mapper.mapping_count(), 2);
    /// ```
    pub fn add_mapping(&mut self, mapping: ComponentMapping) {
        self.mappings.push(mapping);
    }

    /// Returns the number of mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Gets a reference to all mappings.
    pub fn mappings(&self) -> &[ComponentMapping] {
        &self.mappings
    }

    /// Gets the inlet flowrate variable.
    pub fn inlet_flow(&self) -> &Var<Algebraic> {
        &self.inlet_flow
    }

    /// Gets the outlet flowrate variable.
    pub fn outlet_flow(&self) -> &Var<Algebraic> {
        &self.outlet_flow
    }

    /// Computes the total number of input components.
    pub fn input_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Lumping => {
                // Input is detailed components
                self.mappings.iter().map(|m| m.detailed_count()).sum()
            }
            MappingDirection::Delumping => {
                // Input is lumped components
                self.mappings.len()
            }
        }
    }

    /// Computes the total number of output components.
    pub fn output_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Delumping => {
                // Output is detailed components
                self.mappings.iter().map(|m| m.detailed_count()).sum()
            }
            MappingDirection::Lumping => {
                // Output is lumped components
                self.mappings.len()
            }
        }
    }
}

impl<T: TimeDomain> UnitOp for ComponentMapper<T> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<TD: TimeDomain>(&self, system: &mut EquationSystem<TD>, unit_name: &str) {
        match self.direction {
            MappingDirection::Lumping => {
                // For each mapping: F_lumped = Σ(F_detailed_i)
                for mapping in self.mappings.iter() {
                    let lumped_name = mapping.lumped_component().to_string();
                    let detailed_names: Vec<String> = mapping
                        .detailed_components()
                        .iter()
                        .map(|(name, _)| format!("F_{}", name))
                        .collect();
                    let n_detailed = detailed_names.len();

                    let mut vars = vec![format!("F_{}", lumped_name)];
                    vars.extend(detailed_names);

                    let lumping_eq = ResidualFunction::from_dynamic(
                        &format!("{}_lump_{}", unit_name, lumped_name),
                        vars,
                        move |v, names| {
                            let f_lumped = v.get(&names[0]).copied().unwrap_or(0.0);
                            let sum_detailed: f64 = (1..=n_detailed)
                                .map(|i| v.get(&names[i]).copied().unwrap_or(0.0))
                                .sum();
                            f_lumped - sum_detailed
                        },
                    );
                    system.add_algebraic(lumping_eq);
                }
            }
            MappingDirection::Delumping => {
                // For each mapping and detailed component: F_detailed_i = fraction_i * F_lumped
                for mapping in &self.mappings {
                    let lumped_name = mapping.lumped_component().to_string();
                    for (detailed_name, fraction) in mapping.detailed_components() {
                        let frac = *fraction;
                        let vars =
                            vec![format!("F_{}", detailed_name), format!("F_{}", lumped_name)];
                        let delumping_eq = ResidualFunction::from_dynamic(
                            &format!("{}_delump_{}_to_{}", unit_name, lumped_name, detailed_name),
                            vars,
                            move |v, names| {
                                let f_detailed = v.get(&names[0]).copied().unwrap_or(0.0);
                                let f_lumped = v.get(&names[1]).copied().unwrap_or(0.0);
                                f_detailed - frac * f_lumped
                            },
                        );
                        system.add_algebraic(delumping_eq);
                    }
                }
            }
        }
    }
}

/// Hierarchical Module System
///
/// A hierarchical module containing an internal flowsheet.
///
/// Modules enable composition and component isolation, similar to Aspen Plus
/// Hierarchies. Each module:
/// - Contains an internal flowsheet
/// - Declares its own component set (reducing equations)
/// - Exposes input/output ports (interface to parent flowsheet)
/// - Can be nested within other modules or flowsheets
///
/// # Use Cases
///
/// 1. **Component Isolation**: Different modules track different species
///    ```ignore
///    let c2_splitter = Module::new("C2_Split")
///        .with_components(vec!["ethane", "ethylene"]);
///    
///    let c3_splitter = Module::new("C3_Split")
///        .with_components(vec!["propane", "propylene"]);
///    ```
///
/// 2. **Reusable Subsystems**: Define once, instantiate multiple times
///    ```ignore
///    fn create_distillation_train() -> Module<Dynamic> {
///        let mut module = Module::new("DistTrain");
///        // ... build internal flowsheet ...
///        module
///    }
///    ```
///
/// 3. **Equation Reduction**: Only equations for declared components
///
/// # Examples
///
/// ```
/// use nomata::{Module, Dynamic, ComponentSet};
///
/// let mut c2_module = Module::<Dynamic>::new("C2_Splitter");
/// c2_module.set_components(ComponentSet::new(vec!["ethane", "ethylene"]));
///
/// // Internal flowsheet is accessible
/// assert_eq!(c2_module.flowsheet().unit_count(), 0);
/// ```
#[derive(Debug)]
pub struct Module<T: TimeDomain> {
    /// Module name
    pub name: String,
    /// Internal flowsheet containing the module's unit operations
    flowsheet: Flowsheet<T>,
    /// Component set for this module
    component_set: ComponentSet,
    /// Input port definitions
    input_ports: Vec<NamedPort>,
    /// Output port definitions
    output_ports: Vec<NamedPort>,
}

impl<T: TimeDomain> Module<T> {
    /// Creates a new empty module with the given name.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Module, Dynamic};
    ///
    /// let module = Module::<Dynamic>::new("C2_Splitter");
    /// assert_eq!(module.name(), "C2_Splitter");
    /// ```
    pub fn new(name: &str) -> Self {
        Module {
            name: name.to_string(),
            flowsheet: Flowsheet::new(),
            component_set: ComponentSet::empty(),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
        }
    }

    /// Gets the module name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets a reference to the internal flowsheet.
    pub fn flowsheet(&self) -> &Flowsheet<T> {
        &self.flowsheet
    }

    /// Gets a mutable reference to the internal flowsheet.
    pub fn flowsheet_mut(&mut self) -> &mut Flowsheet<T> {
        &mut self.flowsheet
    }

    /// Sets the component set for this module.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Module, Dynamic, ComponentSet};
    ///
    /// let mut module = Module::<Dynamic>::new("C2_Split");
    /// module.set_components(ComponentSet::new(vec!["ethane", "ethylene"]));
    ///
    /// assert_eq!(module.component_count(), 2);
    /// ```
    pub fn set_components(&mut self, components: ComponentSet) {
        self.component_set = components;
    }

    /// Gets a reference to the component set.
    pub fn components(&self) -> &ComponentSet {
        &self.component_set
    }

    /// Returns the number of components in this module.
    pub fn component_count(&self) -> usize {
        self.component_set.count()
    }

    /// Adds an input port to the module interface.
    ///
    /// Input ports define how external streams enter the module.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Module, Dynamic, NamedPort};
    ///
    /// let mut module = Module::<Dynamic>::new("Mixer");
    /// module.add_input_port(NamedPort::input("feed_1", "MolarFlow"));
    /// module.add_input_port(NamedPort::input("feed_2", "MolarFlow"));
    ///
    /// assert_eq!(module.input_port_count(), 2);
    /// ```
    pub fn add_input_port(&mut self, port: NamedPort) {
        self.input_ports.push(port);
    }

    /// Adds an output port to the module interface.
    ///
    /// Output ports define how internal streams exit the module.
    pub fn add_output_port(&mut self, port: NamedPort) {
        self.output_ports.push(port);
    }

    /// Returns the number of input ports.
    pub fn input_port_count(&self) -> usize {
        self.input_ports.len()
    }

    /// Returns the number of output ports.
    pub fn output_port_count(&self) -> usize {
        self.output_ports.len()
    }

    /// Gets the input ports.
    pub fn input_ports(&self) -> &[NamedPort] {
        &self.input_ports
    }

    /// Gets the output ports.
    pub fn output_ports(&self) -> &[NamedPort] {
        &self.output_ports
    }

    /// Harvests equations from all units in the internal flowsheet.
    ///
    /// This collects all equations from the module's internal units,
    /// respecting the module's component set for equation reduction.
    ///
    /// Note: Currently harvests all equations from registered units.
    /// Future enhancement: Filter equations based on component_set.
    pub fn harvest_internal_equations(&mut self) {
        // Currently, this method provides a way to trigger equation harvesting
        // for the internal flowsheet. In practice, unit-specific equation
        // harvesting should be done via flowsheet.harvest_equations().
        //
        // The module structure allows for future equation filtering based
        // on the component_set, which would reduce equation count for
        // subsystems that only track a subset of components.
        //
        // For now, the equation system is already populated when units
        // call build_equations through flowsheet.harvest_equations().
    }

    /// Gets the total number of equations in this module.
    ///
    /// Includes both the internal flowsheet equations and any
    /// interface constraints.
    pub fn total_equations(&self) -> usize {
        self.flowsheet.equation_system.total_equations()
    }
}

// Modules can act as unit operations in a parent flowsheet
impl<T: TimeDomain> UnitOp for Module<T> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<TD: TimeDomain>(&self, system: &mut EquationSystem<TD>, unit_name: &str) {
        // When a module is used as a unit operation, it contributes all its
        // internal flowsheet equations to the parent system.
        //
        // This enables hierarchical modeling where modules act as black-box
        // unit operations in a larger flowsheet.

        let prefix = unit_name.to_string();

        // Copy all differential equations from internal flowsheet
        for eq in self.flowsheet.equation_system.differential_equations() {
            let original_vars = eq.variable_names();
            let renamed_vars: Vec<String> =
                original_vars.iter().map(|v| format!("{}_{}", prefix, v)).collect();

            // Create a mapping closure that remaps variable names
            let old_vars = original_vars.clone();
            let new_prefix = prefix.clone();
            let inner_fn = eq.clone_eval_fn();

            let renamed_eq = ResidualFunction::from_dynamic(
                &format!("{}_{}", prefix, eq.name()),
                renamed_vars,
                move |var_values, _names| {
                    // Remap variable names: parent_unit_var -> var
                    let mut remapped: std::collections::HashMap<String, f64> =
                        std::collections::HashMap::new();
                    for old_var in &old_vars {
                        let new_var = format!("{}_{}", new_prefix, old_var);
                        if let Some(&val) = var_values.get(&new_var) {
                            remapped.insert(old_var.clone(), val);
                        }
                    }
                    inner_fn(&remapped)
                },
            );
            system.add_differential(renamed_eq);
        }

        // Copy all algebraic equations from internal flowsheet
        for eq in self.flowsheet.equation_system.algebraic_equations() {
            let original_vars = eq.variable_names();
            let renamed_vars: Vec<String> =
                original_vars.iter().map(|v| format!("{}_{}", prefix, v)).collect();

            // Create a mapping closure that remaps variable names
            let old_vars = original_vars.clone();
            let new_prefix = prefix.clone();
            let inner_fn = eq.clone_eval_fn();

            let renamed_eq = ResidualFunction::from_dynamic(
                &format!("{}_{}", prefix, eq.name()),
                renamed_vars,
                move |var_values, _names| {
                    // Remap variable names: parent_unit_var -> var
                    let mut remapped: std::collections::HashMap<String, f64> =
                        std::collections::HashMap::new();
                    for old_var in &old_vars {
                        let new_var = format!("{}_{}", new_prefix, old_var);
                        if let Some(&val) = var_values.get(&new_var) {
                            remapped.insert(old_var.clone(), val);
                        }
                    }
                    inner_fn(&remapped)
                },
            );
            system.add_algebraic(renamed_eq);
        }

        // Future enhancement: Add interface mapping equations that connect
        // the module's input/output ports to the internal flowsheet variables
    }
}

/// Trait for Unit Operations
///
/// Trait for unit operations in a process flowsheet.
///
/// Unit operations must:
/// 1. Declare input/output stream types (for compile-time type checking)
/// 2. Generate their own equations (for automatic equation harvesting)
///
/// # Equation Harvesting
///
/// The `build_equations()` method allows units to populate an equation system
/// with their specific physics. This enables a generic simulation loop:
///
/// ```ignore
/// // 1. Create flowsheet and add units
/// let mut flowsheet = Flowsheet::<Dynamic>::new();
/// let reactor = CSTR::new(10.0, 1.0, 298.15);
///
/// // 2. Register units
/// flowsheet.add_unit_with_equations("reactor", &reactor);
///
/// // 3. Solve (equations are harvested automatically)
/// solve_flowsheet(&mut flowsheet);
/// ```
pub trait UnitOp {
    /// The type of the input stream(s) this unit accepts
    type In;
    /// The type of the output stream(s) this unit produces
    type Out;

    /// Populates the equation system with this unit's equations.
    ///
    /// Each unit operation implements its own physics by adding:
    /// - Differential equations (for state variables)
    /// - Algebraic equations (for constraints)
    ///
    /// # Arguments
    ///
    /// * `system` - The equation system to populate
    /// * `unit_name` - A unique name for this unit instance (for equation naming)
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str);
}

/// Conservation Law Enforcement Traits
///
/// Marker trait indicating a unit operation conserves mass.
///
/// Units implementing this trait **must** provide a mass balance equation.
/// This enables compile-time verification that mass conservation is enforced.
///
/// # Type-Level Enforcement
///
/// By requiring this trait in generic contexts, we can ensure at compile time
/// that only units with proper mass balances are used:
///
/// ```ignore
/// fn validate_unit<U: ConservesMass>(unit: &U) {
///     // Compiler guarantees this unit has a mass balance
/// }
/// ```
///
/// # Examples
///
/// ```
/// use nomata::{UnitOp, ConservesMass, Equation, MassBalance, EquationSystem, TimeDomain};
/// # use nomata::{Stream, MolarFlow};
/// # use std::marker::PhantomData;
///
/// struct Reactor;
///
/// impl UnitOp for Reactor {
///     type In = Stream<MolarFlow>;
///     type Out = Stream<MolarFlow>;
///     fn build_equations<T: TimeDomain>(&self, _: &mut EquationSystem<T>, _: &str) {}
/// }
///
/// impl ConservesMass for Reactor {
///     fn mass_balance(&self) -> Equation<MassBalance> {
///         Equation::new("reactor_mass_balance")
///     }
/// }
/// ```
pub trait ConservesMass: UnitOp {
    /// Returns the mass balance equation for this unit.
    ///
    /// The mass balance must satisfy: accumulation = input - output + generation
    fn mass_balance(&self) -> Equation<MassBalance>;
}

/// Marker trait indicating a unit operation conserves energy.
///
/// Units implementing this trait **must** provide an energy balance equation.
/// This enables compile-time verification that energy conservation is enforced.
///
/// # Examples
///
/// ```
/// use nomata::{UnitOp, ConservesEnergy, Equation, EnergyBalance, EquationSystem, TimeDomain};
/// # use nomata::{Stream, MolarFlow};
///
/// struct HeatExchanger;
///
/// impl UnitOp for HeatExchanger {
///     type In = Stream<MolarFlow>;
///     type Out = Stream<MolarFlow>;
///     fn build_equations<T: TimeDomain>(&self, _: &mut EquationSystem<T>, _: &str) {}
/// }
///
/// impl ConservesEnergy for HeatExchanger {
///     fn energy_balance(&self) -> Equation<EnergyBalance> {
///         Equation::new("hx_energy_balance")
///     }
/// }
/// ```
pub trait ConservesEnergy: UnitOp {
    /// Returns the energy balance equation for this unit.
    ///
    /// The energy balance must satisfy: accumulation = input - output + heat - work
    fn energy_balance(&self) -> Equation<EnergyBalance>;
}

/// Marker trait indicating a unit operation conserves momentum.
///
/// Less commonly used than mass/energy, but important for fluid dynamics.
pub trait ConservesMomentum: UnitOp {
    /// Returns the momentum balance equation for this unit.
    fn momentum_balance(&self) -> Equation<MomentumBalance>;
}

/// Composite trait for units that satisfy all fundamental conservation laws.
///
/// A unit implementing `FullyConservative` must conserve mass, energy, and momentum.
/// This is the strongest conservation guarantee.
///
/// # Examples
///
/// ```ignore
/// fn validate_conservative_unit<U: FullyConservative>(unit: &U) {
///     // Compiler guarantees mass, energy, AND momentum conservation
///     let mass_eq = unit.mass_balance();
///     let energy_eq = unit.energy_balance();
///     let momentum_eq = unit.momentum_balance();
/// }
/// ```
pub trait FullyConservative: ConservesMass + ConservesEnergy + ConservesMomentum {}

/// Blanket implementation: any unit with all three conservation laws is fully conservative.
impl<T> FullyConservative for T where T: ConservesMass + ConservesEnergy + ConservesMomentum {}

/// Helper function to validate a unit satisfies mass conservation at runtime.
///
/// While the trait provides compile-time guarantees, this function checks
/// the actual residual value of the mass balance equation.
///
/// # Arguments
///
/// * `unit` - A unit operation that conserves mass
/// * `tolerance` - Maximum allowed residual
///
/// # Returns
///
/// `Ok(())` if mass balance residual < tolerance, `Err(...)` otherwise
pub fn validate_mass_conservation<U: ConservesMass>(
    unit: &U,
    tolerance: f64,
) -> Result<(), String> {
    let balance = unit.mass_balance();
    if balance.is_satisfied(tolerance) {
        Ok(())
    } else {
        Err(format!(
            "Mass balance '{}' not satisfied: residual = {} (tolerance: {})",
            balance.name,
            balance.residual(),
            tolerance
        ))
    }
}

/// Helper function to validate a unit satisfies energy conservation at runtime.
pub fn validate_energy_conservation<U: ConservesEnergy>(
    unit: &U,
    tolerance: f64,
) -> Result<(), String> {
    let balance = unit.energy_balance();
    if balance.is_satisfied(tolerance) {
        Ok(())
    } else {
        Err(format!(
            "Energy balance '{}' not satisfied: residual = {} (tolerance: {})",
            balance.name,
            balance.residual(),
            tolerance
        ))
    }
}

/// Validates all conservation laws for a fully conservative unit.
pub fn validate_full_conservation<U: FullyConservative>(
    unit: &U,
    tolerance: f64,
) -> Result<(), String> {
    validate_mass_conservation(unit, tolerance)?;
    validate_energy_conservation(unit, tolerance)?;

    let momentum = unit.momentum_balance();
    if !momentum.is_satisfied(tolerance) {
        return Err(format!(
            "Momentum balance '{}' not satisfied: residual = {} (tolerance: {})",
            momentum.name,
            momentum.residual(),
            tolerance
        ));
    }

    Ok(())
}

// Graph Layer: Flowsheet and Topology
///
/// Unique identifier for a unit operation in a flowsheet.
///
/// Unit IDs allow tracking of connections even when the original
/// unit operation objects are not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitId(pub usize);

/// Unique identifier for a port on a unit operation.
///
/// Port IDs uniquely identify connection endpoints in the flowsheet graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortId {
    /// The unit operation this port belongs to
    pub unit: UnitId,
    /// The port index on that unit (0 for single-port units)
    pub port_index: usize,
}

/// Unique identifier for an edge (connection) in the flowsheet.
///
/// Edge IDs allow tracking individual connections between units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub usize);

/// Direction of material flow in a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowDirection {
    /// Flow from upstream to downstream
    Forward,
    /// Reverse flow (e.g., recycle streams)
    Reverse,
}

/// A connection (edge) between two unit operations in a flowsheet.
///
/// Edges represent material, energy, or information flow between units.
#[derive(Debug, Clone)]
pub struct Connection {
    /// Unique identifier for this connection
    pub id: EdgeId,
    /// Source port (output)
    pub from: PortId,
    /// Destination port (input)
    pub to: PortId,
    /// Direction of flow
    pub direction: FlowDirection,
    /// Optional connection name/label
    pub label: Option<String>,
}

impl Connection {
    /// Creates a new connection between two ports.
    pub fn new(id: EdgeId, from: PortId, to: PortId) -> Self {
        Connection { id, from, to, direction: FlowDirection::Forward, label: None }
    }

    /// Sets a label for this connection.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Sets the flow direction.
    pub fn with_direction(mut self, direction: FlowDirection) -> Self {
        self.direction = direction;
        self
    }
}

/// Information about a unit operation registered in the flowsheet.
#[derive(Debug, Clone)]
pub struct UnitInfo {
    /// Unique identifier
    pub id: UnitId,
    /// Optional name/label
    pub name: Option<String>,
    /// Number of input ports
    pub num_inputs: usize,
    /// Number of output ports
    pub num_outputs: usize,
}

impl UnitInfo {
    /// Creates unit info for a unit with single input and output.
    pub fn new(id: UnitId) -> Self {
        UnitInfo { id, name: None, num_inputs: 1, num_outputs: 1 }
    }

    /// Sets the unit name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Sets the number of ports.
    pub fn with_ports(mut self, num_inputs: usize, num_outputs: usize) -> Self {
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
        self
    }
}

/// A flowsheet containing unit operations, connections, and equations.
///
/// The flowsheet acts as a graph data structure that stores:
/// - Unit operations (nodes)
/// - Connections between units (edges)
/// - The complete equation system
///
/// # Examples
///
/// ```
/// use nomata::{Flowsheet, Dynamic, UnitId, PortId, EdgeId, models::CSTR};
///
/// let mut flowsheet = Flowsheet::<Dynamic>::new();
///
/// // Register units with their operations
/// let reactor = CSTR::new(100.0, 1.0, 350.0);
/// let reactor_info = flowsheet.add_unit("reactor", reactor);
/// // Note: separator would need a separator unit, omitted for brevity
///
/// // Create connection (would need separator unit first)
/// // let from = PortId { unit: reactor_info.id, port_index: 0 };
/// // let to = PortId { unit: separator_info.id, port_index: 0 };
/// // flowsheet.add_connection(from, to);
///
/// assert_eq!(flowsheet.unit_count(), 1);
/// ```
impl<T: TimeDomain + std::fmt::Debug> std::fmt::Debug for Flowsheet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Flowsheet")
            .field("units", &self.units)
            .field("connections", &self.connections)
            .field("next_unit_id", &self.next_unit_id)
            .field("next_edge_id", &self.next_edge_id)
            .field("equation_system", &self.equation_system)
            .field("pending_harvests_count", &self.pending_harvests.len())
            .finish()
    }
}

pub struct Flowsheet<T: TimeDomain> {
    /// Registry of all unit operations
    units: Vec<UnitInfo>,
    /// Adjacency list: connections between units
    connections: Vec<Connection>,
    /// Stream data at ports (keyed by PortId)
    stream_data: HashMap<PortId, StreamData>,
    /// Next available unit ID
    next_unit_id: usize,
    /// Next available edge ID
    next_edge_id: usize,
    /// Complete equation system for this flowsheet
    pub equation_system: EquationSystem<T>,
    /// Pending unit operations to be harvested
    pending_harvests: Vec<HarvestFn<T>>,
}

type HarvestFn<T> = Box<dyn Fn(&mut EquationSystem<T>)>;

/// Stream data stored at a port.
///
/// This stores the physical properties needed to initialize unit operations.
#[derive(Debug, Clone)]
pub struct StreamData {
    pub flow: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub composition: Vec<f64>,
    pub components: Vec<String>,
}

impl<T: TimeDomain> Flowsheet<T> {
    /// Creates a new empty flowsheet.
    pub fn new() -> Self {
        Flowsheet {
            units: Vec::new(),
            connections: Vec::new(),
            stream_data: HashMap::new(),
            next_unit_id: 0,
            next_edge_id: 0,
            equation_system: EquationSystem::new(),
            pending_harvests: Vec::new(),
        }
    }

    /// Registers a new unit operation in the flowsheet and stores it for equation harvesting.
    ///
    /// The unit operation is stored internally and its equations will be harvested
    /// when `harvest_equations()` is called.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Dynamic, models::CSTR};
    ///
    /// let mut flowsheet = Flowsheet::<Dynamic>::new();
    /// let reactor = CSTR::new(100.0, 1.0, 350.0);
    /// flowsheet.add_unit("CSTR-101", reactor);
    /// // Equations are harvested later with flowsheet.harvest_equations()
    /// ```
    pub fn add_unit<U>(&mut self, name: &str, unit: U) -> UnitInfo
    where
        U: UnitOp + 'static,
    {
        let id = UnitId(self.next_unit_id);
        self.next_unit_id += 1;

        let info = UnitInfo { id, name: Some(name.to_string()), num_inputs: 1, num_outputs: 1 };
        self.units.push(info.clone());

        // Store the harvest closure
        let name_clone = name.to_string();
        let harvest_closure = Box::new(move |system: &mut EquationSystem<T>| {
            unit.build_equations(system, &name_clone);
        });
        self.pending_harvests.push(harvest_closure);

        info
    }

    /// Updates an existing unit's information.
    ///
    /// Useful for modifying unit metadata after creation.
    pub fn update_unit(&mut self, id: UnitId, f: impl FnOnce(&mut UnitInfo)) {
        if let Some(unit) = self.units.iter_mut().find(|u| u.id == id) {
            f(unit);
        }
    }

    /// Adds a connection between two ports.
    ///
    /// This is the runtime component that works alongside compile-time
    /// type checking in `connect()` and `connect_ports()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Dynamic, UnitId, PortId, models::CSTR};
    ///
    /// let mut flowsheet = Flowsheet::<Dynamic>::new();
    /// let u1 = CSTR::new(100.0, 1.0, 350.0);
    /// let u1_info = flowsheet.add_unit("unit1", u1);
    /// // Note: would need actual unit2, omitted for brevity
    ///
    /// // let from = PortId { unit: u1_info.id, port_index: 0 };
    /// // let to = PortId { unit: u2_info.id, port_index: 0 };
    /// // flowsheet.add_connection(from, to);
    /// ```
    pub fn add_connection(&mut self, from: PortId, to: PortId) -> &mut Connection {
        let edge_id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        let connection = Connection::new(edge_id, from, to);
        self.connections.push(connection);
        self.connections.last_mut().unwrap()
    }

    /// Gets the number of registered units.
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Gets the number of connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Finds all connections from a specific unit.
    ///
    /// Returns edges where this unit is the source.
    pub fn outgoing_connections(&self, unit: UnitId) -> Vec<&Connection> {
        self.connections.iter().filter(|conn| conn.from.unit == unit).collect()
    }

    /// Finds all connections to a specific unit.
    ///
    /// Returns edges where this unit is the destination.
    pub fn incoming_connections(&self, unit: UnitId) -> Vec<&Connection> {
        self.connections.iter().filter(|conn| conn.to.unit == unit).collect()
    }

    /// Gets all connections in the flowsheet.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Gets all units in the flowsheet.
    pub fn units(&self) -> &[UnitInfo] {
        &self.units
    }

    /// Finds a unit by its ID.
    pub fn get_unit(&self, id: UnitId) -> Option<&UnitInfo> {
        self.units.iter().find(|u| u.id == id)
    }

    /// Checks if two units are directly connected.
    pub fn are_connected(&self, from: UnitId, to: UnitId) -> bool {
        self.connections.iter().any(|conn| conn.from.unit == from && conn.to.unit == to)
    }

    /// Validates the flowsheet topology.
    ///
    /// Checks for common issues like:
    /// - Disconnected units
    /// - Cycles (if not allowed)
    /// - Port connection completeness
    pub fn validate(&self) -> Result<(), String> {
        // Check for disconnected units
        for unit in &self.units {
            let has_input = !self.incoming_connections(unit.id).is_empty();
            let has_output = !self.outgoing_connections(unit.id).is_empty();

            if !has_input && !has_output {
                let name = unit.name.as_deref().unwrap_or("unnamed");
                return Err(format!("Unit '{}' ({:?}) is disconnected", name, unit.id));
            }
        }

        Ok(())
    }

    /// Stores stream data at a specific port.
    ///
    /// This allows units to query their inlet stream properties.
    pub fn set_stream_at_port<S: StreamType, C>(&mut self, port: PortId, stream: &Stream<S, C>) {
        let data = StreamData {
            flow: stream.total_flow,
            temperature: stream.temperature,
            pressure: stream.pressure,
            composition: stream.composition.clone(),
            components: stream.components.clone(),
        };
        self.stream_data.insert(port, data);
    }

    /// Retrieves stream data at a specific port.
    ///
    /// Returns None if no stream data has been set at this port.
    pub fn get_stream_at_port(&self, port: &PortId) -> Option<&StreamData> {
        self.stream_data.get(port)
    }

    /// Detects cycles in the flowsheet using depth-first search.
    ///
    /// Returns a list of cycles found, where each cycle is a vector of unit IDs
    /// forming a closed loop.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let cycles = flowsheet.detect_cycles();
    /// if !cycles.is_empty() {
    ///     println!("Found {} recycle loop(s)", cycles.len());
    /// }
    /// ```
    pub fn detect_cycles(&self) -> Vec<Vec<UnitId>> {
        use std::collections::{HashMap, HashSet};

        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        // Build adjacency list
        let mut adjacency: HashMap<UnitId, Vec<UnitId>> = HashMap::new();
        for conn in &self.connections {
            adjacency.entry(conn.from.unit).or_default().push(conn.to.unit);
        }

        fn dfs(
            node: UnitId,
            adjacency: &HashMap<UnitId, Vec<UnitId>>,
            visited: &mut HashSet<UnitId>,
            rec_stack: &mut HashSet<UnitId>,
            path: &mut Vec<UnitId>,
            cycles: &mut Vec<Vec<UnitId>>,
        ) {
            visited.insert(node);
            rec_stack.insert(node);
            path.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        dfs(neighbor, adjacency, visited, rec_stack, path, cycles);
                    } else if rec_stack.contains(&neighbor) {
                        // Found a cycle - extract it from path
                        if let Some(cycle_start) = path.iter().position(|&id| id == neighbor) {
                            let cycle = path[cycle_start..].to_vec();
                            cycles.push(cycle);
                        }
                    }
                }
            }

            path.pop();
            rec_stack.remove(&node);
        }

        // Check all units as potential starting points
        for unit in &self.units {
            if !visited.contains(&unit.id) {
                dfs(unit.id, &adjacency, &mut visited, &mut rec_stack, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// Finds tear streams for recycle loops.
    ///
    /// A tear stream is a connection that, if removed, breaks a cycle.
    /// This is essential for sequential-modular flowsheet simulation.
    ///
    /// Returns a list of connection IDs that can serve as tear streams.
    pub fn find_tear_streams(&self) -> Vec<EdgeId> {
        let cycles = self.detect_cycles();
        let mut tear_streams = Vec::new();

        for cycle in cycles {
            // Simple heuristic: tear the last connection in each cycle
            if cycle.len() >= 2 {
                let from_unit = cycle[cycle.len() - 1];
                let to_unit = cycle[0];

                // Find the connection between these units
                if let Some(conn) = self
                    .connections
                    .iter()
                    .find(|c| c.from.unit == from_unit && c.to.unit == to_unit)
                {
                    tear_streams.push(conn.id);
                }
            }
        }

        tear_streams
    }

    /// Harvests equations from all registered unit operations.
    ///
    /// This method processes all units that were added with `add_unit(name, unit)`
    /// and populates the flowsheet's equation system with their physics equations.
    ///
    /// Call this method after adding all units to the flowsheet.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Dynamic, models::CSTR};
    ///
    /// let mut flowsheet = Flowsheet::<Dynamic>::new();
    /// let reactor = CSTR::new(10.0, 1.0, 298.15);
    ///
    /// flowsheet.add_unit("CSTR-101", reactor);
    /// flowsheet.harvest_equations(); // Harvests all units
    /// ```
    pub fn harvest_equations(&mut self) {
        for harvest in self.pending_harvests.drain(..) {
            harvest(&mut self.equation_system);
        }
    }

    /// Gets the equation system for this flowsheet.
    pub fn equations(&self) -> &EquationSystem<T> {
        &self.equation_system
    }

    /// Gets a mutable reference to the equation system.
    pub fn equations_mut(&mut self) -> &mut EquationSystem<T> {
        &mut self.equation_system
    }

    /// Creates and initializes a VariableRegistry from the harvested equations.
    ///
    /// This method should be called after all equations have been harvested.
    /// It creates a registry with all variables from the equation system,
    /// properly initialized with appropriate roles (differential/algebraic).
    ///
    /// # Returns
    ///
    /// A fully initialized VariableRegistry ready for use with solvers.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Steady, VariableRegistry};
    ///
    /// let mut flowsheet = Flowsheet::<Steady>::new();
    /// // ... harvest equations ...
    /// let registry = flowsheet.create_registry();
    /// ```
    pub fn create_registry(&self) -> VariableRegistry {
        let registry = VariableRegistry::new();
        registry.initialize_from_equations(&self.equation_system, None);
        registry
    }

    /// Creates a VariableRegistry with custom initial conditions specified by variable name.
    ///
    /// This allows setting initial conditions in a more user-friendly way by specifying
    /// values for specific variables by name, rather than having to know the exact
    /// order of variables in the equation system.
    ///
    /// # Arguments
    ///
    /// * `initial_conditions` - A map from variable names to initial values.
    ///   Variables not specified in the map will be initialized to 0.0.
    ///
    /// # Returns
    ///
    /// A fully initialized VariableRegistry with the specified initial conditions,
    /// or an error if any variable names are not found in the equation system.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Steady, models::CSTR};
    /// use std::collections::HashMap;
    ///
    /// let mut flowsheet = Flowsheet::<Steady>::new();
    /// let reactor = CSTR::new(100.0, 1.0, 350.0);
    /// flowsheet.add_unit("CSTR-101", reactor);
    /// flowsheet.harvest_equations();
    ///
    /// let mut initial_conditions = HashMap::new();
    /// initial_conditions.insert("CSTR-101_F_in".to_string(), 100.0);
    /// initial_conditions.insert("CSTR-101_T_in".to_string(), 350.0);
    ///
    /// let registry = flowsheet.create_registry_with_initial_conditions(&initial_conditions).unwrap();
    /// ```
    pub fn create_registry_with_initial_conditions(
        &self,
        initial_conditions: &std::collections::HashMap<String, f64>,
    ) -> Result<VariableRegistry, Vec<String>> {
        let registry = VariableRegistry::new();

        let var_count = self.equation_system.variable_count();
        let mut values = vec![0.0; var_count];

        let mut missing = Vec::new();

        // Set custom initial conditions by name
        for (name, &value) in initial_conditions {
            if let Some(index) = self.equation_system.variable_index(name) {
                values[index] = value;
            } else {
                missing.push(name.clone());
            }
        }

        if !missing.is_empty() {
            return Err(missing);
        }

        registry.initialize_from_equations(&self.equation_system, Some(&values));
        Ok(registry)
    }

    /// Sets initial values in an existing VariableRegistry by variable name.
    ///
    /// This method allows updating the values in an already initialized registry
    /// by specifying values for specific variables by name.
    ///
    /// # Arguments
    ///
    /// * `registry` - The registry to update
    /// * `initial_conditions` - A map from variable names to values
    ///
    /// # Returns
    ///
    /// Ok(()) if all variables were found and set, or an error with the list of missing variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Steady, models::CSTR};
    /// use std::collections::HashMap;
    ///
    /// let mut flowsheet = Flowsheet::<Steady>::new();
    /// let reactor = CSTR::new(100.0, 1.0, 350.0);
    /// flowsheet.add_unit("CSTR-101", reactor);
    /// flowsheet.harvest_equations();
    ///
    /// let mut registry = flowsheet.create_registry();
    ///
    /// let mut updates = HashMap::new();
    /// updates.insert("CSTR-101_F_in".to_string(), 100.0);
    /// updates.insert("CSTR-101_T_in".to_string(), 350.0);
    ///
    /// flowsheet.set_registry_values_by_name(&registry, &updates).unwrap();
    /// ```
    pub fn set_registry_values_by_name(
        &self,
        registry: &VariableRegistry,
        initial_conditions: &std::collections::HashMap<String, f64>,
    ) -> Result<(), Vec<String>> {
        let mut missing = Vec::new();

        for (name, &value) in initial_conditions {
            if let Some(index) = self.equation_system.variable_index(name) {
                registry.set_by_index(index, value);
            } else {
                missing.push(name.clone());
            }
        }

        if !missing.is_empty() { Err(missing) } else { Ok(()) }
    }

    /// Sets initial conditions for a unit's stream variables.
    ///
    /// This is a convenience method for setting initial conditions for common
    /// stream properties like flow rates, temperatures, and compositions.
    /// The method automatically constructs variable names based on the unit name
    /// and common naming conventions.
    ///
    /// # Arguments
    ///
    /// * `unit_name` - The name of the unit (e.g., "CSTR-101")
    /// * `stream_conditions` - A map from stream property names to values.
    ///   Common properties include: "F_in", "F_out", "T_in", "T", "C_in", "C", "V", etc.
    ///
    /// # Returns
    ///
    /// A HashMap that can be passed to `create_registry_with_initial_conditions`.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Flowsheet, Steady, models::CSTR};
    /// use std::collections::HashMap;
    ///
    /// let mut flowsheet = Flowsheet::<Steady>::new();
    /// let reactor = CSTR::new(100.0, 1.0, 350.0);
    /// flowsheet.add_unit("CSTR-101", reactor);
    /// flowsheet.harvest_equations();
    ///
    /// let mut stream_conditions = HashMap::new();
    /// stream_conditions.insert("F_in".to_string(), 100.0);
    /// stream_conditions.insert("T_in".to_string(), 350.0);
    /// stream_conditions.insert("C_in".to_string(), 1.0);
    /// stream_conditions.insert("V".to_string(), 100.0);
    /// stream_conditions.insert("C".to_string(), 0.5);
    /// stream_conditions.insert("T".to_string(), 360.0);
    ///
    /// let initial_conditions = flowsheet.set_stream_initial_conditions("CSTR-101", &stream_conditions);
    /// let registry = flowsheet.create_registry_with_initial_conditions(&initial_conditions).unwrap();
    /// ```
    pub fn set_stream_initial_conditions(
        &self,
        unit_name: &str,
        stream_conditions: &std::collections::HashMap<String, f64>,
    ) -> std::collections::HashMap<String, f64> {
        let mut initial_conditions = std::collections::HashMap::new();

        for (property, &value) in stream_conditions {
            let var_name = format!("{}_{}", unit_name, property);
            initial_conditions.insert(var_name, value);
        }

        initial_conditions
    }
}

impl<T: TimeDomain> Default for Flowsheet<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced connection function that registers edges in a flowsheet.
///
/// This function:
/// 1. Performs compile-time type checking (stream compatibility)
/// 2. Registers the connection in the flowsheet graph at runtime
///
/// # Examples
///
/// ```ignore
/// use nomata::{Flowsheet, Dynamic, UnitId, PortId, connect_with_flowsheet};
///
/// let mut flowsheet = Flowsheet::<Dynamic>::new();
/// let reactor_info = flowsheet.add_unit("reactor");
/// let separator_info = flowsheet.add_unit("separator");
///
/// // Type-safe connection that also updates the graph
/// connect_with_flowsheet(&reactor, &separator, &mut flowsheet, reactor_info.id, separator_info.id);
/// ```
pub fn connect_with_flowsheet<U1, U2, T>(
    _upstream: &U1,
    _downstream: &U2,
    flowsheet: &mut Flowsheet<T>,
    upstream_id: UnitId,
    downstream_id: UnitId,
) -> EdgeId
where
    U1: UnitOp,
    U2: UnitOp<In = U1::Out>,
    T: TimeDomain,
{
    // Compile-time check: U2::In must equal U1::Out (enforced by trait bound)

    // Runtime registration: add edge to flowsheet graph
    let from = PortId { unit: upstream_id, port_index: 0 };
    let to = PortId { unit: downstream_id, port_index: 0 };

    let connection = flowsheet.add_connection(from, to);
    connection.id
}

/// Enhanced port connection function that registers edges in a flowsheet.
///
/// Combines linear type checking with runtime graph registration.
pub fn connect_ports_with_flowsheet<S, T>(
    output: Port<S, Output, Disconnected>,
    input: Port<S, Input, Disconnected>,
    flowsheet: &mut Flowsheet<T>,
    from: PortId,
    to: PortId,
) -> (Port<S, Output, Connected>, Port<S, Input, Connected>, EdgeId)
where
    T: TimeDomain,
{
    // Compile-time checks (via type parameters)
    let (out_connected, in_connected) = connect_ports(output, input);

    // Runtime registration
    let connection = flowsheet.add_connection(from, to);
    let edge_id = connection.id;

    (out_connected, in_connected, edge_id)
}

/// Connects a stream to a unit operation, automatically initializing the unit from stream data.
///
/// This function:
/// 1. Stores the stream data in the flowsheet at the specified port
/// 2. Automatically calls the unit's `initialize_from_stream` method if it implements `FromStream`
///
/// # Type Parameters
///
/// - `S`: Stream type (e.g., MolarFlow, MassFlow)
/// - `C`: Stream condition state
/// - `U`: Unit operation type that implements `FromStream`
/// - `T`: Time domain
///
/// # Examples
///
/// ```ignore
/// use nomata::{Stream, MolarFlow, Flowsheet, Dynamic, PortId, UnitId};
///
/// let stream = Stream::<MolarFlow, _>::pure(100.0, "Water".to_string(), 300.0, 101325.0);
/// let mut flowsheet = Flowsheet::<Dynamic>::new();
///
/// let port_id = PortId { unit: UnitId(0), port_index: 0 };
///
/// // Automatically creates and initializes heater from stream properties
/// let heater = Heater::from_stream(&stream);
/// flowsheet.set_stream_at_port(port_id, &stream);
/// ```
pub fn connect_stream_to_unit<S: StreamType, C, U, T>(
    stream: &Stream<S, C>,
    flowsheet: &mut Flowsheet<T>,
    port_id: PortId,
) -> U
where
    U: FromStream,
    T: TimeDomain,
{
    flowsheet.set_stream_at_port(port_id, stream);
    U::from_stream(stream)
}

/// Graph Layer: Model Container
///
/// A process model with a specific time domain.
///
/// The type parameter `T` determines whether time derivatives are allowed.
///
/// # Examples
///
/// ```
/// use nomata::{Model, Steady, Dynamic};
///
/// let steady_model: Model<Steady> = Model::new();
/// let dynamic_model: Model<Dynamic> = Model::new();
/// ```
pub struct Model<T: TimeDomain> {
    /// Phantom data to carry the time domain type
    _time_domain: PhantomData<T>,
}

impl<T: TimeDomain> Model<T> {
    /// Creates a new model with the specified time domain.
    pub fn new() -> Self {
        Model { _time_domain: PhantomData }
    }
}

impl<T: TimeDomain> Default for Model<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for dynamic models.
///
/// Only implemented for `Model<Dynamic>`, this trait provides methods
/// that are only meaningful for time-varying models.
pub trait DynamicOps {
    /// Advances the model forward in time (placeholder for future implementation).
    fn step(&mut self, dt: f64);
}

impl DynamicOps for Model<Dynamic> {
    fn step(&mut self, dt: f64) {
        // Time-stepping implementation using simple forward Euler method.
        //
        // This provides basic integration capability. For production use,
        // consider using higher-order methods (RK4, BDF) from the solvers
        // module or differential-equations crate.
        //
        // The implementation requires:
        // 1. A flowsheet with harvested equations
        // 2. A variable registry with initialized state
        // 3. Residual functions that can be evaluated
        //
        // Since Model<T> currently doesn't store a flowsheet or registry
        // (it's a marker type for compile-time checking), actual time-stepping
        // is done through the solver module functions like solve_ode() or
        // by manually implementing:
        //
        // ```ignore
        // // Pseudo-code for what this would do:
        // let y = registry.get_differential_values();
        // let dydt = evaluate_residuals(&flowsheet.equations(), &y);
        // let y_new = y + dt * dydt;
        // registry.set_differential_values(&y_new);
        // ```
        //
        // See examples/ode_integration.rs and examples/solver_with_autodiff.rs
        // for working integration examples.

        // For now, this is a no-op to maintain API stability.
        // Users should use the solver functions directly with their flowsheet.
        let _ = dt; // Acknowledge parameter
    }
}

// Note: DynamicOps is NOT implemented for Model<Steady>, making it
// impossible to call time-stepping methods on steady-state models.

/// Equation Layer: Balance Equations and Residuals
///
/// Marker trait for types of balance equations.
///
/// Different balance types represent different conservation laws
/// that must be satisfied in process models.
pub trait BalanceType {}

/// Mass balance equation type.
///
/// Represents conservation of mass: accumulation = in - out + generation
#[derive(Debug, Clone, Copy)]
pub struct MassBalance;
impl BalanceType for MassBalance {}

/// Energy balance equation type.
///
/// Represents conservation of energy: accumulation = in - out + generation/consumption
#[derive(Debug, Clone, Copy)]
pub struct EnergyBalance;
impl BalanceType for EnergyBalance {}

/// Component balance equation type.
///
#[derive(Debug, Clone, Copy)]
/// Represents conservation of individual chemical species.
pub struct ComponentBalance;
impl BalanceType for ComponentBalance {}

/// Momentum balance equation type.
///
/// Represents conservation of momentum (for fluid flow).
#[derive(Debug, Clone, Copy)]
pub struct MomentumBalance;
impl BalanceType for MomentumBalance {}

/// Conservation Law Closure Enforcement
///
/// Marker trait indicating a balance equation has all required terms.
///
/// This trait is only implemented when the balance builder has been
/// properly closed with all necessary terms (input, output, accumulation).
pub trait ConservationClosed {}

/// Phantom type markers for builder states
pub struct NeedsInput;
pub struct HasInput;
pub struct NeedsOutput;
pub struct HasOutput;
pub struct NeedsAccumulation;
pub struct Closed;

/// Builder for constructing type-safe balance equations.
///
/// Uses phantom types to enforce at compile time that all required
/// terms are provided before the balance can be used.
///
/// # Type Parameters
///
/// - `B`: The balance type (MassBalance, EnergyBalance, etc.)
/// - `I`: Input state (NeedsInput or HasInput)
/// - `O`: Output state (NeedsOutput or HasOutput)  
/// - `A`: Accumulation state (NeedsAccumulation or Closed)
///
/// # Examples
///
/// ```
/// use nomata::BalanceBuilder;
///
/// let balance = BalanceBuilder::mass("reactor")
///     .with_input(100.0)    // Flow in
///     .with_output(95.0)    // Flow out
///     .with_generation(5.0) // Generation term
///     .with_accumulation(10.0) // Holdup derivative
///     .close();  // Verifies: in - out + gen = accum
/// ```
pub struct BalanceBuilder<B: BalanceType, I, O, A> {
    name: String,
    input: f64,
    output: f64,
    generation: f64,
    accumulation: f64,
    _balance: PhantomData<B>,
    _input_state: PhantomData<I>,
    _output_state: PhantomData<O>,
    _accum_state: PhantomData<A>,
}

impl<B: BalanceType> BalanceBuilder<B, NeedsInput, NeedsOutput, NeedsAccumulation> {
    /// Creates a new balance builder.
    pub fn new(name: &str) -> Self {
        BalanceBuilder {
            name: name.to_string(),
            input: 0.0,
            output: 0.0,
            generation: 0.0,
            accumulation: 0.0,
            _balance: PhantomData,
            _input_state: PhantomData,
            _output_state: PhantomData,
            _accum_state: PhantomData,
        }
    }
}

/// Helper functions to create specific balance types
impl BalanceBuilder<MassBalance, NeedsInput, NeedsOutput, NeedsAccumulation> {
    /// Creates a new mass balance builder.
    pub fn mass(name: &str) -> Self {
        Self::new(name)
    }
}

impl BalanceBuilder<EnergyBalance, NeedsInput, NeedsOutput, NeedsAccumulation> {
    /// Creates a new energy balance builder.
    pub fn energy(name: &str) -> Self {
        Self::new(name)
    }
}

impl BalanceBuilder<ComponentBalance, NeedsInput, NeedsOutput, NeedsAccumulation> {
    /// Creates a new component balance builder.
    pub fn component(name: &str) -> Self {
        Self::new(name)
    }
}

impl BalanceBuilder<MomentumBalance, NeedsInput, NeedsOutput, NeedsAccumulation> {
    /// Creates a new momentum balance builder.
    pub fn momentum(name: &str) -> Self {
        Self::new(name)
    }
}

impl<B: BalanceType, O, A> BalanceBuilder<B, NeedsInput, O, A> {
    /// Adds input term(s) to the balance.
    pub fn with_input(self, input: f64) -> BalanceBuilder<B, HasInput, O, A> {
        BalanceBuilder {
            name: self.name,
            input,
            output: self.output,
            generation: self.generation,
            accumulation: self.accumulation,
            _balance: PhantomData,
            _input_state: PhantomData,
            _output_state: PhantomData,
            _accum_state: PhantomData,
        }
    }
}

impl<B: BalanceType, I, A> BalanceBuilder<B, I, NeedsOutput, A> {
    /// Adds output term(s) to the balance.
    pub fn with_output(self, output: f64) -> BalanceBuilder<B, I, HasOutput, A> {
        BalanceBuilder {
            name: self.name,
            input: self.input,
            output,
            generation: self.generation,
            accumulation: self.accumulation,
            _balance: PhantomData,
            _input_state: PhantomData,
            _output_state: PhantomData,
            _accum_state: PhantomData,
        }
    }
}

impl<B: BalanceType, I, O> BalanceBuilder<B, I, O, NeedsAccumulation> {
    /// Adds accumulation term to the balance.
    pub fn with_accumulation(self, accumulation: f64) -> BalanceBuilder<B, I, O, Closed> {
        BalanceBuilder {
            name: self.name,
            input: self.input,
            output: self.output,
            generation: self.generation,
            accumulation,
            _balance: PhantomData,
            _input_state: PhantomData,
            _output_state: PhantomData,
            _accum_state: PhantomData,
        }
    }
}

impl<B: BalanceType, I, O, A> BalanceBuilder<B, I, O, A> {
    /// Adds generation/consumption term (optional).
    pub fn with_generation(mut self, generation: f64) -> Self {
        self.generation = generation;
        self
    }
}

impl<B: BalanceType> BalanceBuilder<B, HasInput, HasOutput, Closed> {
    /// Closes the balance and verifies conservation.
    ///
    /// Returns an equation only if: input - output + generation = accumulation
    pub fn close(self) -> Result<Equation<B>, String> {
        let residual = self.input - self.output + self.generation - self.accumulation;
        let tolerance = 1e-10;

        if residual.abs() > tolerance {
            return Err(format!(
                "Conservation law violated for '{}': in={}, out={}, gen={}, accum={}, residual={}",
                self.name, self.input, self.output, self.generation, self.accumulation, residual
            ));
        }

        Ok(Equation::new(&self.name))
    }

    /// Closes the balance without runtime verification (for symbolic setup).
    ///
    /// Use this when defining model structure, not numerical values.
    pub fn close_symbolic(self) -> Equation<B> {
        Equation::new(&self.name)
    }
}

impl<B: BalanceType> ConservationClosed for BalanceBuilder<B, HasInput, HasOutput, Closed> {}

/// A balance equation in a process model.
///
/// Balance equations represent conservation laws and must be satisfied
/// by the model. The type parameter `B` indicates which conservation
/// law is being enforced.
///
/// # Conservation Closure
///
/// For compile-time verification that balances actually close, use
/// `BalanceBuilder` instead of constructing directly:
///
/// ```ignore
/// use nomata::{BalanceBuilder, MassBalance};
///
/// let balance = BalanceBuilder::<MassBalance>::new("reactor")
///     .with_input(100.0)
///     .with_output(90.0)
///     .with_generation(0.0)
///     .with_accumulation(10.0)
///     .close(); // Enforces: 100 - 90 + 0 = 10
/// ```
///
/// # Examples
///
/// ```
/// use nomata::{Equation, MassBalance};
///
/// let mass_balance: Equation<MassBalance> = Equation::new("mass_balance");
/// ```
#[derive(Debug)]
pub struct Equation<B: BalanceType> {
    /// Descriptive name for the equation
    pub name: String,
    /// The residual value (should be zero at solution)
    residual: f64,
    /// Phantom data to carry the balance type
    _balance_type: PhantomData<B>,
}

impl<B: BalanceType> Equation<B> {
    /// Creates a new balance equation with the given name.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Equation, MassBalance, EnergyBalance};
    ///
    /// let mass_eq: Equation<MassBalance> = Equation::new("reactor_mass");
    /// let energy_eq: Equation<EnergyBalance> = Equation::new("reactor_energy");
    /// ```
    pub fn new(name: &str) -> Self {
        Equation { name: name.to_string(), residual: 0.0, _balance_type: PhantomData }
    }

    /// Gets the current residual value.
    ///
    /// A residual of zero indicates the equation is satisfied.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Sets the residual value.
    ///
    /// This would typically be computed from the balance equation terms.
    pub fn set_residual(&mut self, value: f64) {
        self.residual = value;
    }

    /// Checks if the equation is satisfied (residual near zero).
    pub fn is_satisfied(&self, tolerance: f64) -> bool {
        self.residual.abs() < tolerance
    }
}

/// Closure-Based Residual Functions
///
/// Type alias for residual evaluation functions.
///
/// A residual function takes a variable name -> value mapping and returns
/// the residual value (should be zero at solution).
pub type ResidualFn =
    std::sync::Arc<dyn Fn(&std::collections::HashMap<String, f64>) -> f64 + Send + Sync>;

/// Type alias for dual-number residual evaluation functions (autodiff support).
#[cfg(feature = "autodiff")]
pub type DualResidualFn = std::sync::Arc<
    dyn Fn(&std::collections::HashMap<String, num_dual::Dual64>) -> num_dual::Dual64 + Send + Sync,
>;

/// Trait for scalar types that can be used in equation computations.
///
/// This trait is implemented for `f64` and `Dual64` (when autodiff is enabled),
/// allowing the same equation struct and closure to work with both types.
///
/// When the `autodiff` feature is enabled, this trait uses `num_dual::DualNum<f64>`
/// as a supertrait, which provides all mathematical operations automatically.
///
/// # Example
///
/// ```
/// use nomata::Scalar;
///
/// fn compute_residual<S: Scalar>(f_in: S, f_out: S) -> S {
///     f_in - f_out
/// }
///
/// let result = compute_residual(100.0_f64, 50.0_f64);
/// assert!((result - 50.0).abs() < 1e-10);
/// ```
#[cfg(feature = "autodiff")]
pub trait Scalar: num_dual::DualNum<f64> + Copy + Send + Sync + 'static {}

#[cfg(feature = "autodiff")]
impl<T: num_dual::DualNum<f64> + Copy + Send + Sync + 'static> Scalar for T {}

/// Trait for scalar types that can be used in equation computations.
///
/// When the `autodiff` feature is disabled, this is a marker trait for f64.
/// The Vars structs need this bound to compile, but math operations use
/// f64's inherent methods directly.
#[cfg(not(feature = "autodiff"))]
pub trait Scalar:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + From<f64>
    + Send
    + Sync
    + 'static
{
}

#[cfg(not(feature = "autodiff"))]
impl Scalar for f64 {}

/// Trait for typed equation variable structs that work with any scalar type.
///
/// Implement this trait for equation variable structs that should support
/// automatic differentiation. The struct should be generic over the scalar type.
///
/// # Example
///
/// ```
/// use nomata::{EquationVarsGeneric, Scalar};
/// use std::collections::HashMap;
///
/// struct MassBalanceVars<S: Scalar> {
///     f_in: S,
///     f_out: S,
/// }
///
/// impl<S: Scalar> EquationVarsGeneric<S> for MassBalanceVars<S> {
///     fn base_names() -> &'static [&'static str] {
///         &["F_in", "F_out"]
///     }
///
///     fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
///         Some(Self {
///             f_in: *vars.get(&format!("{}_F_in", prefix))?,
///             f_out: *vars.get(&format!("{}_F_out", prefix))?,
///         })
///     }
/// }
/// ```
pub trait EquationVarsGeneric<S: Scalar>: Sized {
    /// Returns the base variable names (without prefix).
    fn base_names() -> &'static [&'static str];

    /// Constructs the variable set from a HashMap using prefixed names.
    fn from_map(vars: &std::collections::HashMap<String, S>, prefix: &str) -> Option<Self>;
}

/// A residual function for a differential-algebraic equation system.
///
/// This represents a single equation that must be satisfied (residual = 0).
/// Uses typed equation structs for compile-time safety.
///
/// # Example
///
/// ```
/// use nomata::{ResidualFunction, EquationVars};
/// use std::collections::HashMap;
///
/// struct MassBalanceVars {
///     f_in: f64,
///     f_out: f64,
/// }
///
/// impl EquationVars for MassBalanceVars {
///     fn base_names() -> &'static [&'static str] {
///         &["F_in", "F_out"]
///     }
///     fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
///         Some(Self {
///             f_in: *vars.get(&format!("{}_F_in", prefix))?,
///             f_out: *vars.get(&format!("{}_F_out", prefix))?,
///         })
///     }
/// }
///
/// let mass_balance = ResidualFunction::from_typed(
///     "mass_balance",
///     "unit",
///     |v: MassBalanceVars| v.f_in - v.f_out,
/// );
/// ```
pub struct ResidualFunction {
    /// Name of the residual function
    pub name: String,
    /// Variables referenced by this function
    variables: Vec<String>,
    /// The evaluation closure (Arc for cloning)
    eval_fn: ResidualFn,
    /// Dual-number evaluation closure for automatic differentiation
    #[cfg(feature = "autodiff")]
    dual_eval_fn: Option<DualResidualFn>,
}

impl std::fmt::Debug for ResidualFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualFunction")
            .field("name", &self.name)
            .field("variables", &self.variables)
            .field("eval_fn", &"<closure>")
            .finish()
    }
}

impl ResidualFunction {
    /// Evaluates the residual using a variable name -> value mapping.
    ///
    /// # Arguments
    ///
    /// * `var_values` - HashMap from variable name to current value
    ///
    /// # Returns
    ///
    /// The computed residual value (should be zero at solution).
    pub fn evaluate(&self, var_values: &std::collections::HashMap<String, f64>) -> f64 {
        (self.eval_fn)(var_values)
    }

    /// Gets all variable names referenced by this residual function.
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.clone()
    }

    /// Gets the name of this residual function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Clones the evaluation function (for use in Module hierarchies).
    ///
    /// This allows creating a new ResidualFunction that wraps the same
    /// underlying closure with variable renaming.
    pub fn clone_eval_fn(&self) -> ResidualFn {
        self.eval_fn.clone()
    }

    /// Creates a residual function from a typed equation struct.
    ///
    /// This provides compile-time type safety for equation variables.
    /// The struct must implement `EquationVars` which defines the
    /// variable names and how to construct the struct from a HashMap.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the residual (for debugging/display)
    /// * `prefix` - Prefix to prepend to variable names (e.g., unit name)
    /// * `f` - Function that computes the residual given the typed variables
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::{ResidualFunction, EquationVars};
    /// use std::collections::HashMap;
    ///
    /// // Define typed variables for mass balance
    /// struct MassBalanceVars {
    ///     f_in: f64,
    ///     f_out: f64,
    /// }
    ///
    /// impl EquationVars for MassBalanceVars {
    ///     fn base_names() -> &'static [&'static str] {
    ///         &["F_in", "F_out"]
    ///     }
    ///
    ///     fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
    ///         Some(Self {
    ///             f_in: *vars.get(&format!("{}_F_in", prefix))?,
    ///             f_out: *vars.get(&format!("{}_F_out", prefix))?,
    ///         })
    ///     }
    /// }
    ///
    /// // Create type-safe residual function
    /// let mass_balance = ResidualFunction::from_typed(
    ///     "reactor_mass_balance",
    ///     "reactor",
    ///     |v: MassBalanceVars| v.f_in - v.f_out,
    /// );
    /// ```
    #[cfg(not(feature = "autodiff"))]
    pub fn from_typed<V, F>(name: &str, prefix: &str, f: F) -> Self
    where
        V: EquationVars,
        F: Fn(V) -> f64 + Send + Sync + 'static,
    {
        let var_names: Vec<String> =
            V::base_names().iter().map(|n| format!("{}_{}", prefix, n)).collect();

        let prefix_owned = prefix.to_string();
        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| {
                V::from_map(vars, &prefix_owned).map(&f).unwrap_or(0.0)
            }),
        }
    }

    /// Creates a residual function from a typed equation struct (with autodiff support).
    ///
    /// When the `autodiff` feature is enabled, this stores both f64 and Dual64
    /// evaluation closures to enable automatic differentiation for Jacobian computation.
    #[cfg(feature = "autodiff")]
    pub fn from_typed<V, F>(name: &str, prefix: &str, f: F) -> Self
    where
        V: EquationVars,
        F: Fn(V) -> f64 + Send + Sync + 'static,
    {
        let var_names: Vec<String> =
            V::base_names().iter().map(|n| format!("{}_{}", prefix, n)).collect();

        let prefix_owned = prefix.to_string();
        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| {
                V::from_map(vars, &prefix_owned).map(&f).unwrap_or(0.0)
            }),
            dual_eval_fn: None, // Use from_typed_autodiff for autodiff support
        }
    }

    /// Creates a residual function with full autodiff support.
    ///
    /// This constructor takes separate closures for f64 and Dual64 evaluation,
    /// enabling true automatic differentiation for Jacobian computation.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the residual (for debugging/display)
    /// * `prefix` - Prefix to prepend to variable names (e.g., unit name)
    /// * `f` - Function that computes the residual with f64 variables
    /// * `f_dual` - Function that computes the residual with Dual64 variables
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::{ResidualFunction, EquationVars};
    /// use num_dual::Dual64;
    /// use std::collections::HashMap;
    ///
    /// // f64 version
    /// struct MassBalanceVars { f_in: f64, f_out: f64 }
    /// // Dual64 version  
    /// struct MassBalanceVarsDual { f_in: Dual64, f_out: Dual64 }
    ///
    /// let mass_balance = ResidualFunction::from_typed_autodiff(
    ///     "mass_balance",
    ///     "unit",
    ///     &["F_in", "F_out"],
    ///     |vars: &HashMap<String, f64>, prefix: &str| {
    ///         let f_in = vars.get(&format!("{}_F_in", prefix)).copied().unwrap_or(0.0);
    ///         let f_out = vars.get(&format!("{}_F_out", prefix)).copied().unwrap_or(0.0);
    ///         f_in - f_out
    ///     },
    ///     |vars: &HashMap<String, Dual64>, prefix: &str| {
    ///         let zero = Dual64::from(0.0);
    ///         let f_in = vars.get(&format!("{}_F_in", prefix)).copied().unwrap_or(zero);
    ///         let f_out = vars.get(&format!("{}_F_out", prefix)).copied().unwrap_or(zero);
    ///         f_in - f_out
    ///     },
    /// );
    /// ```
    #[cfg(feature = "autodiff")]
    pub fn from_typed_autodiff<F, FDual>(
        name: &str,
        prefix: &str,
        base_names: &[&str],
        f: F,
        f_dual: FDual,
    ) -> Self
    where
        F: Fn(&std::collections::HashMap<String, f64>, &str) -> f64 + Send + Sync + 'static,
        FDual: Fn(&std::collections::HashMap<String, num_dual::Dual64>, &str) -> num_dual::Dual64
            + Send
            + Sync
            + 'static,
    {
        let var_names: Vec<String> =
            base_names.iter().map(|n| format!("{}_{}", prefix, n)).collect();

        let prefix_owned = prefix.to_string();
        let prefix_dual = prefix.to_string();
        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| f(vars, &prefix_owned)),
            dual_eval_fn: Some(std::sync::Arc::new(move |vars| f_dual(vars, &prefix_dual))),
        }
    }

    /// Creates a residual function from a generic typed equation struct.
    ///
    /// This is the recommended approach for autodiff support. The equation struct
    /// must be generic over the scalar type (implementing `EquationVarsGeneric<S>`),
    /// and the closure must be generic over `S: Scalar`.
    ///
    /// This allows a single struct definition and closure to work with both
    /// f64 (for evaluation) and Dual64 (for automatic differentiation).
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the residual (for debugging/display)
    /// * `prefix` - Prefix to prepend to variable names (e.g., unit name)
    /// * `f` - Generic function that computes the residual for any scalar type
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::{ResidualFunction, EquationVarsGeneric, Scalar};
    /// use std::collections::HashMap;
    ///
    /// struct MassBalanceVars<S: Scalar> {
    ///     f_in: S,
    ///     f_out: S,
    /// }
    ///
    /// impl<S: Scalar> EquationVarsGeneric<S> for MassBalanceVars<S> {
    ///     fn base_names() -> &'static [&'static str] {
    ///         &["F_in", "F_out"]
    ///     }
    ///
    ///     fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
    ///         Some(Self {
    ///             f_in: *vars.get(&format!("{}_F_in", prefix))?,
    ///             f_out: *vars.get(&format!("{}_F_out", prefix))?,
    ///         })
    ///     }
    /// }
    ///
    /// // Single closure works for both f64 and Dual64
    /// let mass_balance = ResidualFunction::from_typed_generic::<MassBalanceVars<f64>, _>(
    ///     "mass_balance",
    ///     "unit",
    ///     |v| v.f_in - v.f_out,
    /// );
    /// ```
    #[cfg(feature = "autodiff")]
    pub fn from_typed_generic<V, F>(name: &str, prefix: &str, f: F) -> Self
    where
        V: EquationVarsGeneric<f64>,
        F: Fn(V) -> f64 + Clone + Send + Sync + 'static,
    {
        let var_names: Vec<String> =
            V::base_names().iter().map(|n| format!("{}_{}", prefix, n)).collect();

        let prefix_f64 = prefix.to_string();
        let f_clone = f.clone();

        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| {
                V::from_map(vars, &prefix_f64).map(&f_clone).unwrap_or(0.0)
            }),
            dual_eval_fn: None, // See from_typed_generic_with_dual for full autodiff
        }
    }

    /// Creates a residual function with full generic autodiff support.
    ///
    /// Takes a generic closure and separate struct types for f64 and Dual64.
    /// The Dual64 struct must implement `EquationVarsGeneric<Dual64>`.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the residual
    /// * `prefix` - Prefix for variable names
    /// * `f` - Closure for f64 evaluation
    /// * `f_dual` - Closure for Dual64 evaluation
    ///
    /// # Type Parameters
    ///
    /// * `V` - The f64 variable struct type
    /// * `VDual` - The Dual64 variable struct type
    /// * `F` - The f64 closure type
    /// * `FDual` - The Dual64 closure type
    #[cfg(feature = "autodiff")]
    pub fn from_typed_generic_with_dual<V, VDual, F, FDual>(
        name: &str,
        prefix: &str,
        f: F,
        f_dual: FDual,
    ) -> Self
    where
        V: EquationVarsGeneric<f64>,
        VDual: EquationVarsGeneric<num_dual::Dual64>,
        F: Fn(V) -> f64 + Send + Sync + 'static,
        FDual: Fn(VDual) -> num_dual::Dual64 + Send + Sync + 'static,
    {
        let var_names: Vec<String> =
            V::base_names().iter().map(|n| format!("{}_{}", prefix, n)).collect();

        let prefix_f64 = prefix.to_string();
        let prefix_dual = prefix.to_string();

        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| {
                V::from_map(vars, &prefix_f64).map(&f).unwrap_or(0.0)
            }),
            dual_eval_fn: Some(std::sync::Arc::new(move |vars| {
                VDual::from_map(vars, &prefix_dual)
                    .map(&f_dual)
                    .unwrap_or_else(|| num_dual::Dual64::from(0.0))
            })),
        }
    }

    /// Evaluates the residual using dual numbers for automatic differentiation.
    ///
    /// Returns `None` if no dual evaluation function is available.
    #[cfg(feature = "autodiff")]
    pub fn evaluate_dual(
        &self,
        var_values: &std::collections::HashMap<String, num_dual::Dual64>,
    ) -> Option<num_dual::Dual64> {
        self.dual_eval_fn.as_ref().map(|f| f(var_values))
    }

    /// Checks if this residual function supports automatic differentiation.
    #[cfg(feature = "autodiff")]
    pub fn supports_autodiff(&self) -> bool {
        self.dual_eval_fn.is_some()
    }

    /// Creates a residual function from a dynamic equation struct.
    ///
    /// Use this for equations with runtime-determined variable counts
    /// (e.g., mixers with N inlets, reactors with N components).
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the residual (for debugging/display)
    /// * `var_names` - List of variable names this function depends on
    /// * `f` - Function that computes the residual given the typed variables
    ///
    /// # Example
    ///
    /// ```
    /// use nomata::ResidualFunction;
    /// use std::collections::HashMap;
    ///
    /// struct SumVars {
    ///     inputs: Vec<f64>,
    ///     output: f64,
    /// }
    ///
    /// impl SumVars {
    ///     fn from_map(vars: &HashMap<String, f64>, var_names: &[String]) -> Option<Self> {
    ///         let n = var_names.len() - 1;
    ///         let inputs: Option<Vec<f64>> = var_names[..n]
    ///             .iter()
    ///             .map(|name| vars.get(name).copied())
    ///             .collect();
    ///         Some(Self {
    ///             inputs: inputs?,
    ///             output: *vars.get(&var_names[n])?,
    ///         })
    ///     }
    /// }
    ///
    /// let var_names = vec!["in_0".to_string(), "in_1".to_string(), "out".to_string()];
    /// let sum_eq = ResidualFunction::from_dynamic(
    ///     "sum",
    ///     var_names.clone(),
    ///     move |vars, names| {
    ///         SumVars::from_map(vars, names)
    ///             .map(|v| v.inputs.iter().sum::<f64>() - v.output)
    ///             .unwrap_or(0.0)
    ///     },
    /// );
    /// ```
    #[cfg(not(feature = "autodiff"))]
    pub fn from_dynamic<F>(name: &str, var_names: Vec<String>, f: F) -> Self
    where
        F: Fn(&std::collections::HashMap<String, f64>, &[String]) -> f64 + Send + Sync + 'static,
    {
        let var_names_clone = var_names.clone();
        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| f(vars, &var_names_clone)),
        }
    }

    /// Creates a residual function from a dynamic equation struct (with autodiff support).
    #[cfg(feature = "autodiff")]
    pub fn from_dynamic<F>(name: &str, var_names: Vec<String>, f: F) -> Self
    where
        F: Fn(&std::collections::HashMap<String, f64>, &[String]) -> f64 + Send + Sync + 'static,
    {
        let var_names_clone = var_names.clone();
        ResidualFunction {
            name: name.to_string(),
            variables: var_names,
            eval_fn: std::sync::Arc::new(move |vars| f(vars, &var_names_clone)),
            dual_eval_fn: None, // Dynamic equations don't support autodiff by default
        }
    }
}

/// Trait for typed equation variable sets.
///
/// Implement this trait for structs that represent the inputs to an equation.
/// This provides compile-time type safety: forgetting a variable or using
/// the wrong name causes a compile error.
///
/// # Example
///
/// ```
/// use nomata::EquationVars;
/// use std::collections::HashMap;
///
/// struct ArrheniusVars {
///     k: f64,
///     k0: f64,
///     ea: f64,
///     t: f64,
/// }
///
/// impl EquationVars for ArrheniusVars {
///     fn base_names() -> &'static [&'static str] {
///         &["k", "k0", "Ea", "T"]
///     }
///
///     fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
///         Some(Self {
///             k: *vars.get(&format!("{}_k", prefix))?,
///             k0: *vars.get(&format!("{}_k0", prefix))?,
///             ea: *vars.get(&format!("{}_Ea", prefix))?,
///             t: *vars.get(&format!("{}_T", prefix))?,
///         })
///     }
/// }
/// ```
pub trait EquationVars: Sized {
    /// Returns the base variable names (without prefix).
    ///
    /// These are used to construct the full variable names by prepending
    /// the unit/component prefix.
    fn base_names() -> &'static [&'static str];

    /// Constructs the variable set from a HashMap using prefixed names.
    ///
    /// Returns `None` if any required variable is missing.
    fn from_map(vars: &std::collections::HashMap<String, f64>, prefix: &str) -> Option<Self>;
}

/// Trait for typed equation variable sets with dual number support.
///
/// Implement this trait alongside `EquationVars` to enable automatic
/// differentiation for Jacobian computation.
#[cfg(feature = "autodiff")]
pub trait EquationVarsDual: Sized {
    /// The corresponding f64 variable struct type.
    type F64Vars: EquationVars;

    /// Constructs the variable set from a HashMap of dual numbers.
    ///
    /// Returns `None` if any required variable is missing.
    fn from_map_dual(
        vars: &std::collections::HashMap<String, num_dual::Dual64>,
        prefix: &str,
    ) -> Option<Self>;
}

/// A system of equations (DAE system).
///
/// Represents a complete differential-algebraic equation system for a
/// process model, with both differential and algebraic equations.
///
/// # Variable Mapping
///
/// The equation system maintains a mapping from variable names (strings)
/// to indices in the state/derivative vectors. This allows equations to
/// reference variables by name (e.g., "F_in", "T_reactor") and have them
/// correctly resolved during evaluation.
#[derive(Debug)]
pub struct EquationSystem<T: TimeDomain> {
    /// Differential equations (time derivatives)
    differential_equations: Vec<ResidualFunction>,
    /// Algebraic equations (constraints)
    algebraic_equations: Vec<ResidualFunction>,
    /// Mapping from variable name to index in state vector
    variable_indices: std::collections::HashMap<String, usize>,
    /// Ordered list of variable names (index -> name)
    variable_names: Vec<String>,
    /// Which variables are differential (true) vs algebraic (false)
    variable_is_differential: Vec<bool>,
    /// Phantom data for time domain
    _time_domain: PhantomData<T>,
}

impl<T: TimeDomain> EquationSystem<T> {
    /// Creates a new equation system.
    pub fn new() -> Self {
        EquationSystem {
            differential_equations: Vec::new(),
            algebraic_equations: Vec::new(),
            variable_indices: std::collections::HashMap::new(),
            variable_names: Vec::new(),
            variable_is_differential: Vec::new(),
            _time_domain: PhantomData,
        }
    }

    /// Registers a variable by name and returns its index.
    ///
    /// If the variable already exists, returns its existing index.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique variable name (e.g., "F_in", "T_reactor")
    /// * `is_differential` - Whether this is a differential variable
    ///
    /// # Returns
    ///
    /// The index of the variable in the state vector.
    pub fn register_variable(&mut self, name: &str, is_differential: bool) -> usize {
        if let Some(&idx) = self.variable_indices.get(name) {
            return idx;
        }
        let idx = self.variable_names.len();
        self.variable_indices.insert(name.to_string(), idx);
        self.variable_names.push(name.to_string());
        self.variable_is_differential.push(is_differential);
        idx
    }

    /// Registers multiple variables from an equation.
    ///
    /// Automatically extracts variable names from the equation and registers them.
    /// Works with both expression-based and legacy term-based equations.
    fn register_variables_from_equation(
        &mut self,
        equation: &ResidualFunction,
        is_differential: bool,
    ) {
        // Use the unified variable_names() method which handles both Expr and legacy terms
        for name in equation.variable_names() {
            self.register_variable(&name, is_differential);
        }
    }

    /// Gets the index of a variable by name.
    pub fn variable_index(&self, name: &str) -> Option<usize> {
        self.variable_indices.get(name).copied()
    }

    /// Gets the name of a variable by index.
    pub fn variable_name(&self, index: usize) -> Option<&str> {
        self.variable_names.get(index).map(|s| s.as_str())
    }

    /// Returns whether a variable at the given index is differential.
    pub fn is_variable_differential(&self, index: usize) -> bool {
        self.variable_is_differential.get(index).copied().unwrap_or(false)
    }

    /// Gets the total number of registered variables.
    pub fn variable_count(&self) -> usize {
        self.variable_names.len()
    }

    /// Gets all registered variable names in order.
    pub fn all_variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Builds a variable name -> value mapping from state and derivative vectors.
    ///
    /// This is used by `evaluate_residuals_mapped` to look up values by name.
    pub fn build_value_mapping(
        &self,
        state: &[f64],
        derivatives: &[f64],
    ) -> std::collections::HashMap<String, f64> {
        let mut mapping = std::collections::HashMap::new();

        for (idx, name) in self.variable_names.iter().enumerate() {
            if self.variable_is_differential.get(idx).copied().unwrap_or(false) {
                // Differential variable: value from derivatives vector
                if idx < derivatives.len() {
                    mapping.insert(name.clone(), derivatives[idx]);
                }
            } else {
                // Algebraic variable: value from state vector
                if idx < state.len() {
                    mapping.insert(name.clone(), state[idx]);
                }
            }
        }

        mapping
    }

    /// Adds a differential equation to the system.
    ///
    /// Differential equations relate time derivatives to other variables.
    /// Variables referenced in the equation are automatically registered.
    pub fn add_differential(&mut self, equation: ResidualFunction) {
        self.register_variables_from_equation(&equation, true);
        self.differential_equations.push(equation);
    }

    /// Adds an algebraic equation to the system.
    ///
    /// Algebraic equations are constraints that must hold instantaneously.
    /// Variables referenced in the equation are automatically registered.
    pub fn add_algebraic(&mut self, equation: ResidualFunction) {
        self.register_variables_from_equation(&equation, false);
        self.algebraic_equations.push(equation);
    }

    /// Gets the number of differential equations.
    pub fn differential_count(&self) -> usize {
        self.differential_equations.len()
    }

    /// Gets the number of algebraic equations.
    pub fn algebraic_count(&self) -> usize {
        self.algebraic_equations.len()
    }

    /// Gets the total number of equations.
    pub fn total_equations(&self) -> usize {
        self.differential_count() + self.algebraic_count()
    }

    /// Evaluates all residuals given state and derivative vectors.
    ///
    /// For a DAE system: F(x, dx/dt, t) = 0
    ///
    /// # Arguments
    ///
    /// * `state` - State variable values x
    /// * `derivatives` - Time derivative values dx/dt
    /// * `registry` - Variable registry for looking up values
    ///
    /// # Returns
    ///
    /// Vector of residuals [F_diff..., F_alg...]
    ///
    /// # Example
    ///
    /// ```ignore
    /// let residuals = system.evaluate_residuals(&state, &derivatives, &registry);
    /// assert!(residuals.iter().all(|&r| r.abs() < 1e-6)); // Check convergence
    /// ```
    pub fn evaluate_residuals(&self, state: &[f64], derivatives: &[f64], _time: f64) -> Vec<f64> {
        // Build value mapping from state and derivative vectors
        // For proper evaluation, we need to map variable names to their values
        let var_values = self.build_value_mapping_full(state, derivatives);

        let mut residuals = Vec::with_capacity(self.total_equations());

        // Evaluate differential equations using name-based lookup
        for eq in &self.differential_equations {
            let r = eq.evaluate(&var_values);
            residuals.push(r);
        }

        // Evaluate algebraic equations using name-based lookup
        for eq in &self.algebraic_equations {
            let r = eq.evaluate(&var_values);
            residuals.push(r);
        }

        residuals
    }

    /// Builds a complete value mapping where all variables can access both
    /// state values and derivative values based on their context.
    ///
    /// This handles the complexity of equations that may reference both
    /// differential variables (for their derivatives) and algebraic variables.
    fn build_value_mapping_full(
        &self,
        state: &[f64],
        derivatives: &[f64],
    ) -> std::collections::HashMap<String, f64> {
        let mut mapping = std::collections::HashMap::new();

        // First pass: register state values for all variables
        for (idx, name) in self.variable_names.iter().enumerate() {
            if idx < state.len() {
                mapping.insert(name.clone(), state[idx]);
            }
        }

        // Second pass: for differential variables, also register derivative values
        // using a naming convention (e.g., "dV_dt" for the derivative of "V")
        for (idx, name) in self.variable_names.iter().enumerate() {
            if self.variable_is_differential.get(idx).copied().unwrap_or(false)
                && idx < derivatives.len()
            {
                // Register derivative with "d...dt" prefix
                let deriv_name = format!("d{}_dt", name);
                mapping.insert(deriv_name, derivatives[idx]);

                // Also handle variants like "dV/dt" or "d(VC)/dt"
                if name.starts_with("d") && (name.contains("_dt") || name.contains("/dt")) {
                    mapping.insert(name.clone(), derivatives[idx]);
                }
            }
        }

        mapping
    }

    /// Computes the Jacobian matrix dF/dx for the equation system.
    ///
    /// This is required for implicit solvers (Newton-Raphson, BDF, etc.)
    ///
    /// Uses finite difference approximation since `ResidualFunction` closures
    /// operate on `HashMap<String, f64>` which cannot be directly differentiated.
    ///
    /// # Returns
    ///
    /// Dense Jacobian matrix as `Vec<Vec<f64>>`
    #[cfg(not(feature = "autodiff"))]
    pub fn compute_jacobian(&self, state: &[f64]) -> Vec<Vec<f64>> {
        self.compute_jacobian_numerical(state)
    }

    /// Computes the Jacobian matrix dF/dx for the equation system.
    ///
    /// When the `autodiff` feature is enabled and equations support it,
    /// uses automatic differentiation with dual numbers for exact derivatives.
    /// Falls back to finite differences for equations without dual support.
    ///
    /// # Returns
    ///
    /// Dense Jacobian matrix as `Vec<Vec<f64>>`
    #[cfg(feature = "autodiff")]
    pub fn compute_jacobian(&self, state: &[f64]) -> Vec<Vec<f64>> {
        // Check if all equations support autodiff
        let all_support_autodiff = self
            .differential_equations
            .iter()
            .chain(self.algebraic_equations.iter())
            .all(|eq| eq.supports_autodiff());

        if all_support_autodiff {
            self.compute_jacobian_autodiff(state)
        } else {
            self.compute_jacobian_numerical(state)
        }
    }

    /// Computes Jacobian using finite difference approximation.
    fn compute_jacobian_numerical(&self, state: &[f64]) -> Vec<Vec<f64>> {
        // Use finite difference approximation: dF_i/dx_j = (F_i(x+h*e_j) - F_i(x)) / h
        let n = self.total_equations();
        let mut jacobian = vec![vec![0.0; n]; n];
        let h = 1e-8; // Step size for finite differences
        let derivatives = vec![0.0; self.differential_count()];

        // Evaluate at current state
        let f0 = self.evaluate_residuals(state, &derivatives, 0.0);

        // Perturb each variable and compute partial derivatives
        for j in 0..n {
            let mut state_pert = state.to_vec();
            state_pert[j] += h;
            let f_pert = self.evaluate_residuals(&state_pert, &derivatives, 0.0);

            for i in 0..n {
                jacobian[i][j] = (f_pert[i] - f0[i]) / h;
            }
        }

        jacobian
    }

    /// Computes Jacobian using automatic differentiation with dual numbers.
    #[cfg(feature = "autodiff")]
    fn compute_jacobian_autodiff(&self, state: &[f64]) -> Vec<Vec<f64>> {
        use num_dual::Dual64;

        let n_vars = state.len();
        let n_eqs = self.total_equations();

        // Compute columns of the Jacobian using forward-mode AD
        let columns: Vec<Vec<f64>> = (0..n_vars)
            .map(|j| {
                // Build dual number mapping where variable j has derivative 1.0
                let dual_vars: std::collections::HashMap<String, Dual64> = self
                    .variable_names
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx < state.len())
                    .map(|(idx, name)| {
                        let dual = if idx == j {
                            Dual64::new(state[idx], 1.0)
                        } else {
                            Dual64::from(state[idx])
                        };
                        (name.clone(), dual)
                    })
                    .collect();

                // Evaluate all equations with dual numbers and extract derivatives
                self.differential_equations
                    .iter()
                    .chain(self.algebraic_equations.iter())
                    .map(|eq| eq.evaluate_dual(&dual_vars).map(|r| r.eps).unwrap_or(0.0))
                    .collect()
            })
            .collect();

        // Convert from column-major to row-major
        (0..n_eqs).map(|i| columns.iter().map(|col| col[i]).collect()).collect()
    }

    /// Gets references to all differential equations.
    pub fn differential_equations(&self) -> &[ResidualFunction] {
        &self.differential_equations
    }

    /// Gets references to all algebraic equations.
    pub fn algebraic_equations(&self) -> &[ResidualFunction] {
        &self.algebraic_equations
    }
}

impl<T: TimeDomain> Default for EquationSystem<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for creating mass balance equations.
///
/// Only implemented for `Var<Differential>`, since mass balances involve
/// time derivatives of holdup variables.
pub trait MassBalanceOps {
    /// Creates a mass balance equation for this differential variable.
    ///
    /// The general form is: d(holdup)/dt = flow_in - flow_out + generation
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Var, Differential, MassBalanceOps, Equation, MassBalance};
    ///
    /// let holdup: Var<Differential> = Var::new(100.0);
    /// let balance: Equation<MassBalance> = holdup.mass_balance("reactor_mass");
    /// ```
    fn mass_balance(&self, name: &str) -> Equation<MassBalance>;
}

impl MassBalanceOps for Var<Differential> {
    fn mass_balance(&self, name: &str) -> Equation<MassBalance> {
        // In a real implementation, this would set up the derivative term
        Equation::new(name)
    }
}

/// Extension trait for creating energy balance equations.
///
/// Only implemented for `Var<Differential>`, since energy balances involve
/// time derivatives of internal energy or enthalpy.
pub trait EnergyBalanceOps {
    /// Creates an energy balance equation for this differential variable.
    ///
    /// The general form is: d(U)/dt = Q + W + H_in - H_out
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::{Var, Differential, EnergyBalanceOps, Equation, EnergyBalance};
    ///
    /// let energy: Var<Differential> = Var::new(5000.0);
    /// let balance: Equation<EnergyBalance> = energy.energy_balance("reactor_energy");
    /// ```
    fn energy_balance(&self, name: &str) -> Equation<EnergyBalance>;
}

impl EnergyBalanceOps for Var<Differential> {
    fn energy_balance(&self, name: &str) -> Equation<EnergyBalance> {
        // In a real implementation, this would set up the energy derivative term
        Equation::new(name)
    }
}

// Note: These balance operations are NOT implemented for Var<Algebraic>,
// making it a compile-time error to create balance equations for algebraic variables.

#[cfg(test)]
mod tests {
    use super::*;

    /// Variable Registry Tests
    #[test]
    fn test_variable_registry_creation() {
        let registry = VariableRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_variable_registry_shared_state() {
        let registry = VariableRegistry::new();
        let temp = registry.create_algebraic(298.15);

        // Create another reference to the same variable
        let temp_ref = temp.clone();

        // Modify through one reference
        temp.set(310.0);

        // Both references see the change
        assert_eq!(temp.get(), 310.0);
        assert_eq!(temp_ref.get(), 310.0);
    }

    #[test]
    fn test_equation_harvesting() {
        use crate::models::CSTR;

        let mut flowsheet = Flowsheet::<Dynamic>::new();
        let reactor = CSTR::new(100.0, 1.0, 350.0);

        // Before harvesting
        assert_eq!(flowsheet.equations().total_equations(), 0);

        // Harvest equations
        flowsheet.add_unit("CSTR-101", reactor);
        flowsheet.harvest_equations();

        // After harvesting
        assert_eq!(flowsheet.equations().differential_count(), 3); // mass, component, energy
        assert_eq!(flowsheet.equations().algebraic_count(), 2); // arrhenius, rate
        assert_eq!(flowsheet.equations().total_equations(), 5);
    }

    #[test]
    fn test_multiple_unit_harvesting() {
        use crate::models::{CSTR, Mixer};

        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let reactor1 = CSTR::new(100.0, 1.0, 350.0);
        let reactor2 = CSTR::new(50.0, 0.5, 340.0);
        let mixer: Mixer<2> = Mixer::new(1).with_inlets_configured(); // 2 inlets, 1 component

        flowsheet.add_unit("CSTR-101", reactor1);
        flowsheet.add_unit("CSTR-102", reactor2);
        flowsheet.add_unit("MIX-301", mixer);
        flowsheet.harvest_equations();

        // In Dynamic mode:
        // - 2 CSTRs: 2 * (3 differential + 2 algebraic) = 10 equations
        // - 1 Mixer<2> with 1 component: 3 differential (mass, energy, 1 component)
        // Total: 13 equations (9 differential + 4 algebraic)
        assert_eq!(flowsheet.equations().total_equations(), 13);
        assert_eq!(flowsheet.equations().differential_count(), 9); // 6 from reactors + 3 from mixer
        assert_eq!(flowsheet.equations().algebraic_count(), 4); // All from reactors' kinetics
    }

    #[test]
    fn test_component_set_creation() {
        let c2_components = ComponentSet::new(vec!["ethane", "ethylene"]);
        assert_eq!(c2_components.count(), 2);
        assert!(c2_components.contains("ethane"));
        assert!(c2_components.contains("ethylene"));
        assert!(!c2_components.contains("propane"));
    }

    #[test]
    fn test_component_set_add() {
        let mut components = ComponentSet::empty();
        assert_eq!(components.count(), 0);

        components.add("ethane");
        assert_eq!(components.count(), 1);
        assert!(components.contains("ethane"));

        // Adding duplicate doesn't increase count
        components.add("ethane");
        assert_eq!(components.count(), 1);
    }

    #[test]
    fn test_module_creation() {
        let module = Module::<Dynamic>::new("C2_Splitter");
        assert_eq!(module.name(), "C2_Splitter");
        assert_eq!(module.component_count(), 0);
        assert_eq!(module.input_port_count(), 0);
        assert_eq!(module.output_port_count(), 0);
        assert_eq!(module.flowsheet().unit_count(), 0);
    }

    #[test]
    fn test_module_with_components() {
        let mut module = Module::<Steady>::new("C3_Splitter");
        module.set_components(ComponentSet::new(vec!["propane", "propylene"]));

        assert_eq!(module.component_count(), 2);
        assert!(module.components().contains("propane"));
        assert!(module.components().contains("propylene"));
    }

    #[test]
    fn test_module_ports() {
        let mut module = Module::<Dynamic>::new("Mixer");

        module.add_input_port(NamedPort::input("feed_1", "MolarFlow"));
        module.add_input_port(NamedPort::input("feed_2", "MolarFlow"));
        module.add_output_port(NamedPort::output("product", "MolarFlow"));

        assert_eq!(module.input_port_count(), 2);
        assert_eq!(module.output_port_count(), 1);
        assert_eq!(module.input_ports().len(), 2);
        assert_eq!(module.output_ports().len(), 1);
    }

    #[test]
    fn test_module_internal_flowsheet() {
        use crate::models::CSTR;

        let mut module = Module::<Dynamic>::new("Reaction_Section");
        module.set_components(ComponentSet::new(vec!["reactant", "product"]));

        // Add units to internal flowsheet
        let reactor1 = CSTR::new(100.0, 1.0, 350.0);
        let reactor2 = CSTR::new(50.0, 0.5, 340.0);
        module.flowsheet_mut().add_unit("CSTR-101", reactor1);
        module.flowsheet_mut().add_unit("CSTR-102", reactor2);

        // Harvest equations from all units
        module.flowsheet_mut().harvest_equations();

        assert_eq!(module.flowsheet().unit_count(), 2);
        assert!(module.total_equations() > 0);
    }

    #[test]
    fn test_module_as_unit_op() {
        use crate::UnitOp;
        use crate::models::CSTR;

        let mut module = Module::<Steady>::new("TestModule");

        // Add equations to the module's internal flowsheet
        let reactor = CSTR::new(100.0, 1.0, 350.0);
        module.flowsheet_mut().add_unit("CSTR-101", reactor);
        module.flowsheet_mut().harvest_equations();

        // Verify module has internal equations
        assert!(module.total_equations() > 0);
        let internal_count = module.total_equations();

        // Now use the module as a unit operation in a parent flowsheet
        let mut parent_system = EquationSystem::<Steady>::new();
        module.build_equations(&mut parent_system, "module_instance");

        // The parent system should now have all the module's equations
        assert_eq!(parent_system.total_equations(), internal_count);
    }

    #[test]
    fn test_variable_registry_bulk_operations() {
        let registry = VariableRegistry::new();

        let t1 = registry.create_algebraic(100.0);
        let t2 = registry.create_algebraic(200.0);
        let t3 = registry.create_algebraic(300.0);

        // Get all values
        let values = registry.get_all_values();
        assert_eq!(values, vec![100.0, 200.0, 300.0]);

        // Modify via bulk operation
        let new_values = vec![150.0, 250.0, 350.0];
        registry.set_all_values(&new_values);

        // Variables see the changes
        assert_eq!(t1.get(), 150.0);
        assert_eq!(t2.get(), 250.0);
        assert_eq!(t3.get(), 350.0);
    }

    #[test]
    fn test_variable_registry_role_tracking() {
        let registry = VariableRegistry::new();

        let param = registry.create_parameter(0.5);
        let alg = registry.create_algebraic(298.15);
        let diff = registry.create_differential(1000.0);

        assert_eq!(registry.get_role(param.id()), "Parameter");
        assert_eq!(registry.get_role(alg.id()), "Algebraic");
        assert_eq!(registry.get_role(diff.id()), "Differential");
    }

    #[test]
    fn test_variable_id_index_access() {
        let registry = VariableRegistry::new();
        let var = registry.create_algebraic(42.0);

        let id = var.id();
        assert_eq!(id.index(), 0); // First variable

        let var2 = registry.create_algebraic(99.0);
        assert_eq!(var2.id().index(), 1); // Second variable
    }

    /// Original Variable Tests
    #[test]
    fn test_algebraic_variable() {
        let temp: Var<Algebraic> = Var::new(298.15);
        assert_eq!(temp.get(), 298.15);
    }

    #[test]
    fn test_differential_variable() {
        let mass: Var<Differential> = Var::new(100.0);
        assert_eq!(mass.get(), 100.0);
    }

    #[test]
    fn test_differential_derivative() {
        let holdup: Var<Differential> = Var::new(50.0);
        let _dhdt = holdup.derivative(); // Should compile
    }

    #[test]
    fn test_stream_creation() {
        let stream =
            Stream::<MolarFlow, UninitializedConditions>::new(10.0, vec!["Component".to_string()])
                .at_conditions(298.15, 101325.0);
        assert_eq!(stream.total_flow, 10.0);
        assert_eq!(stream.temperature, 298.15);
        assert_eq!(stream.pressure, 101325.0);
    }

    #[test]
    fn test_model_creation() {
        let _steady: Model<Steady> = Model::new();
        let _dynamic: Model<Dynamic> = Model::new();
    }

    #[test]
    fn test_dynamic_model_step() {
        let mut model: Model<Dynamic> = Model::new();
        model.step(0.1); // Should compile for dynamic models
    }

    #[test]
    fn test_equation_creation() {
        let mass_eq: Equation<MassBalance> = Equation::new("test_mass");
        assert_eq!(mass_eq.name, "test_mass");
        assert_eq!(mass_eq.residual(), 0.0);
    }

    #[test]
    fn test_equation_residual() {
        let mut energy_eq: Equation<EnergyBalance> = Equation::new("test_energy");
        energy_eq.set_residual(10.5);
        assert_eq!(energy_eq.residual(), 10.5);
        assert!(!energy_eq.is_satisfied(1e-6));

        energy_eq.set_residual(1e-8);
        assert!(energy_eq.is_satisfied(1e-6));
    }

    #[test]
    fn test_equation_term_is_removed() {
        // EquationTerm has been removed in favor of closure-based ResidualFunction
        let residual = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["accumulation".to_string(), "flow_in".to_string(), "flow_out".to_string()],
            |v, names| {
                let accum = v.get(&names[0]).copied().unwrap_or(0.0);
                let flow_in = v.get(&names[1]).copied().unwrap_or(0.0);
                let flow_out = v.get(&names[2]).copied().unwrap_or(0.0);
                accum - flow_in + flow_out
            },
        );

        assert_eq!(residual.variable_names().len(), 3);
    }

    #[test]
    fn test_residual_function() {
        let residual = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["accumulation".to_string(), "flow_in".to_string(), "flow_out".to_string()],
            |v, names| {
                let accum = v.get(&names[0]).copied().unwrap_or(0.0);
                let flow_in = v.get(&names[1]).copied().unwrap_or(0.0);
                let flow_out = v.get(&names[2]).copied().unwrap_or(0.0);
                accum - flow_in + flow_out
            },
        );

        assert_eq!(residual.variable_names().len(), 3);

        // Evaluate: 100 - 50 + 40 = 90
        let mut vars = std::collections::HashMap::new();
        vars.insert("accumulation".to_string(), 100.0);
        vars.insert("flow_in".to_string(), 50.0);
        vars.insert("flow_out".to_string(), 40.0);
        let result = residual.evaluate(&vars);
        assert_eq!(result, 90.0);
    }

    #[test]
    fn test_equation_system() {
        let mut system: EquationSystem<Dynamic> = EquationSystem::new();

        let diff_eq =
            ResidualFunction::from_dynamic("diff_1", vec!["dhdt".to_string()], |v, names| {
                v.get(&names[0]).copied().unwrap_or(0.0)
            });
        let alg_eq =
            ResidualFunction::from_dynamic("alg_1", vec!["constraint".to_string()], |v, names| {
                v.get(&names[0]).copied().unwrap_or(0.0)
            });

        system.add_differential(diff_eq);
        system.add_algebraic(alg_eq);

        assert_eq!(system.differential_count(), 1);
        assert_eq!(system.algebraic_count(), 1);
        assert_eq!(system.total_equations(), 2);
    }

    #[test]
    fn test_equation_system_variable_mapping() {
        // Test that variable names are properly registered and mapped
        let mut system: EquationSystem<Steady> = EquationSystem::new();

        // Create a simple mass balance: 0 = F_in - F_out
        let mass_balance = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["F_in".to_string(), "F_out".to_string()],
            |v, names| {
                let f_in = v.get(&names[0]).copied().unwrap_or(0.0);
                let f_out = v.get(&names[1]).copied().unwrap_or(0.0);
                -f_in + f_out
            },
        );

        // Create a simple energy balance: 0 = Q - F*(T_out - T_in)
        let energy_balance = ResidualFunction::from_dynamic(
            "energy_balance",
            vec!["Q".to_string(), "F_T_out".to_string(), "F_T_in".to_string()],
            |v, names| {
                let q = v.get(&names[0]).copied().unwrap_or(0.0);
                let f_t_out = v.get(&names[1]).copied().unwrap_or(0.0);
                let f_t_in = v.get(&names[2]).copied().unwrap_or(0.0);
                q - f_t_out + f_t_in
            },
        );

        system.add_algebraic(mass_balance);
        system.add_algebraic(energy_balance);

        // Check variable registration
        assert_eq!(system.variable_count(), 5);
        assert!(system.variable_index("F_in").is_some());
        assert!(system.variable_index("F_out").is_some());
        assert!(system.variable_index("Q").is_some());
        assert!(system.variable_index("F_T_out").is_some());
        assert!(system.variable_index("F_T_in").is_some());
    }

    #[test]
    fn test_equation_system_evaluate_with_mapping() {
        use std::collections::HashMap;

        // Test the name-based evaluation
        let mass_balance = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["F_in".to_string(), "F_out".to_string()],
            |v, names| {
                let f_in = v.get(&names[0]).copied().unwrap_or(0.0);
                let f_out = v.get(&names[1]).copied().unwrap_or(0.0);
                -f_in + f_out
            },
        );

        // At steady state: F_in = F_out = 100
        let mut var_values = HashMap::new();
        var_values.insert("F_in".to_string(), 100.0);
        var_values.insert("F_out".to_string(), 100.0);

        let residual = mass_balance.evaluate(&var_values);
        assert!((residual - 0.0).abs() < 1e-10, "Expected 0, got {}", residual);

        // Non-steady state: F_in = 100, F_out = 80
        var_values.insert("F_out".to_string(), 80.0);
        let residual = mass_balance.evaluate(&var_values);
        assert!((residual - (-20.0)).abs() < 1e-10, "Expected -20, got {}", residual);
    }

    #[test]
    fn test_equation_system_evaluate_residuals() {
        let mut system: EquationSystem<Steady> = EquationSystem::new();

        // Create a simple equation: 0 = x - y (i.e., x = y)
        let eq1 = ResidualFunction::from_dynamic(
            "equality",
            vec!["x".to_string(), "y".to_string()],
            |v, names| {
                let x = v.get(&names[0]).copied().unwrap_or(0.0);
                let y = v.get(&names[1]).copied().unwrap_or(0.0);
                x - y
            },
        );

        system.add_algebraic(eq1);

        // Register variables with known indices
        let x_idx = system.variable_index("x").unwrap();
        let y_idx = system.variable_index("y").unwrap();

        // Create state vector
        let mut state = vec![0.0; system.variable_count()];
        state[x_idx] = 5.0;
        state[y_idx] = 5.0;

        let residuals = system.evaluate_residuals(&state, &[], 0.0);
        assert_eq!(residuals.len(), 1);
        assert!((residuals[0] - 0.0).abs() < 1e-10, "Expected 0, got {}", residuals[0]);

        // Now make them unequal: x=5, y=3 => residual = 5 - 3 = 2
        state[y_idx] = 3.0;
        let residuals = system.evaluate_residuals(&state, &[], 0.0);
        assert!((residuals[0] - 2.0).abs() < 1e-10, "Expected 2, got {}", residuals[0]);
    }

    #[test]
    fn test_closure_arrhenius_evaluation() {
        use std::collections::HashMap;

        // Test Arrhenius equation: k - k0 * exp(-Ea / (R * T)) = 0
        // Using R = 8.314 J/(mol·K)
        let residual = ResidualFunction::from_dynamic(
            "arrhenius",
            vec!["k".to_string(), "k0".to_string(), "Ea".to_string(), "T".to_string()],
            |vars, _names| {
                let k = vars.get("k").copied().unwrap_or(0.0);
                let k0 = vars.get("k0").copied().unwrap_or(0.0);
                let ea = vars.get("Ea").copied().unwrap_or(0.0);
                let t = vars.get("T").copied().unwrap_or(0.0);
                k - k0 * (-ea / (8.314 * t)).exp()
            },
        );

        // Test case: k0 = 1e8, Ea = 50000 J/mol, T = 350 K
        // k_expected = 1e8 * exp(-50000 / (8.314 * 350)) = 1e8 * exp(-17.16)
        //            ≈ 1e8 * 3.48e-8 ≈ 3.48
        let k0: f64 = 1e8;
        let ea: f64 = 50000.0;
        let t: f64 = 350.0;
        let r: f64 = 8.314;
        let k_expected = k0 * (-ea / (r * t)).exp();

        let mut values = HashMap::new();
        values.insert("k0".to_string(), k0);
        values.insert("Ea".to_string(), ea);
        values.insert("T".to_string(), t);
        values.insert("k".to_string(), k_expected);

        // At the correct k, residual should be ~0
        let res = residual.evaluate(&values);
        assert!(res.abs() < 1e-6, "Expected ~0, got {}", res);

        // With wrong k, residual should be non-zero
        values.insert("k".to_string(), 1.0);
        let res = residual.evaluate(&values);
        assert!((res - (1.0 - k_expected)).abs() < 1e-6, "Wrong residual: {}", res);
    }

    #[test]
    fn test_closure_reaction_rate() {
        use std::collections::HashMap;

        // First-order rate: r = k * C
        // Residual: r - k * C = 0
        let residual = ResidualFunction::from_dynamic(
            "rate",
            vec!["r".to_string(), "k".to_string(), "C".to_string()],
            |vars, _names| {
                let r = vars.get("r").copied().unwrap_or(0.0);
                let k = vars.get("k").copied().unwrap_or(0.0);
                let c = vars.get("C").copied().unwrap_or(0.0);
                r - k * c
            },
        );

        let mut values = HashMap::new();
        values.insert("k".to_string(), 0.1); // rate constant
        values.insert("C".to_string(), 2.0); // concentration
        values.insert("r".to_string(), 0.2); // correct rate = 0.1 * 2.0

        // At solution: residual = 0
        let res = residual.evaluate(&values);
        assert!(res.abs() < 1e-10, "Expected 0, got {}", res);

        // With wrong rate
        values.insert("r".to_string(), 0.5);
        let res = residual.evaluate(&values);
        assert!((res - 0.3).abs() < 1e-10, "Expected 0.3, got {}", res);
    }

    #[test]
    fn test_closure_mass_balance() {
        use std::collections::HashMap;

        // Dynamic mass balance: dV/dt - (F_in - F_out) = 0
        let residual = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["dV_dt".to_string(), "F_in".to_string(), "F_out".to_string()],
            |vars, _names| {
                let dv_dt = vars.get("dV_dt").copied().unwrap_or(0.0);
                let f_in = vars.get("F_in").copied().unwrap_or(0.0);
                let f_out = vars.get("F_out").copied().unwrap_or(0.0);
                dv_dt - (f_in - f_out)
            },
        );

        // At steady state: F_in = F_out = 100, dV/dt = 0
        let mut values = HashMap::new();
        values.insert("F_in".to_string(), 100.0);
        values.insert("F_out".to_string(), 100.0);
        values.insert("dV_dt".to_string(), 0.0);

        let res = residual.evaluate(&values);
        assert!(res.abs() < 1e-10, "Expected 0, got {}", res);

        // Filling: F_in > F_out, dV/dt > 0
        values.insert("F_in".to_string(), 120.0);
        values.insert("F_out".to_string(), 100.0);
        values.insert("dV_dt".to_string(), 20.0); // correct: 120 - 100 = 20
        let res = residual.evaluate(&values);
        assert!(res.abs() < 1e-10, "Expected 0, got {}", res);
    }

    #[test]
    fn test_mass_balance_ops() {
        let holdup: Var<Differential> = Var::new(100.0);
        let balance = holdup.mass_balance("reactor_mass");
        assert_eq!(balance.name, "reactor_mass");
    }

    #[test]
    fn test_energy_balance_ops() {
        let energy: Var<Differential> = Var::new(5000.0);
        let balance = energy.energy_balance("reactor_energy");
        assert_eq!(balance.name, "reactor_energy");
    }

    #[test]
    fn test_parameter_creation() {
        // Parameters are time-invariant constants
        let rate_constant: Var<Parameter> = Var::new(0.5);
        assert_eq!(rate_constant.get(), 0.5);

        let density: Var<Parameter> = Var::new(1000.0);
        assert_eq!(density.get(), 1000.0);
    }

    #[test]
    fn test_parameter_vs_algebraic() {
        // Both can store values, but semantics differ
        let param: Var<Parameter> = Var::new(42.0); // Fixed constant
        let alg: Var<Algebraic> = Var::new(42.0); // Computed variable

        assert_eq!(param.get(), alg.get());
        // Parameters cannot have derivatives (enforced by not implementing CanDifferentiate)
    }

    #[test]
    fn test_balance_builder_valid() {
        // Valid balance: in - out + gen = accum (100 - 90 + 0 = 10)
        let result = BalanceBuilder::mass("reactor")
            .with_input(100.0)
            .with_output(90.0)
            .with_accumulation(10.0)
            .close();
        assert!(result.is_ok());
    }

    #[test]
    fn test_balance_builder_with_generation() {
        // Valid balance with generation: 100 - 95 + 5 = 10
        let result = BalanceBuilder::mass("reactor")
            .with_input(100.0)
            .with_output(95.0)
            .with_generation(5.0)
            .with_accumulation(10.0)
            .close();
        assert!(result.is_ok());
    }

    #[test]
    fn test_balance_builder_invalid() {
        // Invalid balance: 100 - 90 + 0 ≠ 20 (should be 10)
        let result = BalanceBuilder::mass("reactor")
            .with_input(100.0)
            .with_output(90.0)
            .with_accumulation(20.0)
            .close();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Conservation law violated"));
    }

    #[test]
    fn test_balance_builder_energy() {
        // Energy balance: 1000 - 800 + 200 = 400
        let result = BalanceBuilder::energy("heater")
            .with_input(1000.0)
            .with_output(800.0)
            .with_generation(200.0)
            .with_accumulation(400.0)
            .close();
        assert!(result.is_ok());
    }

    #[test]
    fn test_balance_builder_symbolic() {
        // Symbolic balance (no runtime verification)
        let eq = BalanceBuilder::component("reactor_A")
            .with_input(0.0)
            .with_output(0.0)
            .with_accumulation(0.0)
            .close_symbolic();
        assert_eq!(eq.name, "reactor_A");
    }

    /// Flowsheet Tests
    #[test]
    fn test_flowsheet_creation() {
        let flowsheet: Flowsheet<Dynamic> = Flowsheet::new();
        assert_eq!(flowsheet.unit_count(), 0);
        assert_eq!(flowsheet.connection_count(), 0);
    }

    #[test]
    fn test_flowsheet_add_units() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let reactor = models::CSTR::new(100.0, 1.0, 350.0);
        let separator = models::FlashSeparator::new(300.0, 2);
        let reactor_id = flowsheet.add_unit("reactor", reactor);
        let separator_id = flowsheet.add_unit("separator", separator);

        assert_eq!(flowsheet.unit_count(), 2);
        assert_eq!(reactor_id.id.0, 0);
        assert_eq!(separator_id.id.0, 1);
        assert_eq!(reactor_id.name, Some("reactor".to_string()));
        assert_eq!(separator_id.name, Some("separator".to_string()));
    }

    #[test]
    fn test_flowsheet_add_connection() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);

        let from = PortId { unit: u1.id, port_index: 0 };
        let to = PortId { unit: u2.id, port_index: 0 };

        let edge_id = flowsheet.add_connection(from, to).id;

        assert_eq!(flowsheet.connection_count(), 1);
        assert_eq!(edge_id.0, 0);
    }

    #[test]
    fn test_flowsheet_connection_with_label() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);

        let from = PortId { unit: u1.id, port_index: 0 };
        let to = PortId { unit: u2.id, port_index: 0 };

        {
            let conn = flowsheet.add_connection(from, to);
            conn.label = Some("main_stream".to_string());
        }

        let connections = flowsheet.connections();
        assert_eq!(connections[0].label, Some("main_stream".to_string()));
    }

    #[test]
    fn test_flowsheet_outgoing_connections() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit3 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);
        let u3 = flowsheet.add_unit("unit3", unit3);

        let from1 = PortId { unit: u1.id, port_index: 0 };
        let to2 = PortId { unit: u2.id, port_index: 0 };
        let to3 = PortId { unit: u3.id, port_index: 0 };

        flowsheet.add_connection(from1, to2);
        flowsheet.add_connection(from1, to3);

        let outgoing = flowsheet.outgoing_connections(u1.id);
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn test_flowsheet_incoming_connections() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit3 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);
        let u3 = flowsheet.add_unit("unit3", unit3);

        let from1 = PortId { unit: u1.id, port_index: 0 };
        let from2 = PortId { unit: u2.id, port_index: 0 };
        let to3 = PortId { unit: u3.id, port_index: 0 };

        flowsheet.add_connection(from1, to3);
        flowsheet.add_connection(from2, to3);

        let incoming = flowsheet.incoming_connections(u3.id);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_flowsheet_are_connected() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit3 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);
        let u3 = flowsheet.add_unit("unit3", unit3);

        let from = PortId { unit: u1.id, port_index: 0 };
        let to = PortId { unit: u2.id, port_index: 0 };

        flowsheet.add_connection(from, to);

        assert!(flowsheet.are_connected(u1.id, u2.id));
        assert!(!flowsheet.are_connected(u1.id, u3.id));
        assert!(!flowsheet.are_connected(u2.id, u1.id)); // Direction matters
    }

    #[test]
    fn test_flowsheet_get_unit() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();
        let unit = models::CSTR::new(100.0, 1.0, 350.0);
        let info = flowsheet.add_unit("test_unit", unit);

        // Update port counts
        flowsheet.update_unit(info.id, |u| {
            u.num_inputs = 2;
            u.num_outputs = 1;
        });

        let retrieved = flowsheet.get_unit(info.id).unwrap();
        assert_eq!(retrieved.name, Some("test_unit".to_string()));
        assert_eq!(retrieved.num_inputs, 2);
        assert_eq!(retrieved.num_outputs, 1);
    }

    #[test]
    fn test_flowsheet_validate_disconnected() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let _u1 = flowsheet.add_unit("unit1", unit1);
        let _u2 = flowsheet.add_unit("unit2", unit2);
        // No connections added

        let result = flowsheet.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("disconnected"));
    }

    #[test]
    fn test_flowsheet_validate_connected() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        let unit1 = models::CSTR::new(100.0, 1.0, 350.0);
        let unit2 = models::CSTR::new(100.0, 1.0, 350.0);
        let u1 = flowsheet.add_unit("unit1", unit1);
        let u2 = flowsheet.add_unit("unit2", unit2);

        let from = PortId { unit: u1.id, port_index: 0 };
        let to = PortId { unit: u2.id, port_index: 0 };

        flowsheet.add_connection(from, to);

        // Validation passes because:
        // - u1 has output (connected to u2)
        // - u2 has input (from u1)
        // Both units have at least one connection
        let result = flowsheet.validate();
        assert!(result.is_ok(), "Validation should pass for properly connected units");
    }

    #[test]
    fn test_connection_struct() {
        let edge_id = EdgeId(42);
        let from = PortId { unit: UnitId(1), port_index: 0 };
        let to = PortId { unit: UnitId(2), port_index: 0 };

        let conn = Connection::new(edge_id, from, to)
            .with_label("test_stream")
            .with_direction(FlowDirection::Reverse);

        assert_eq!(conn.id.0, 42);
        assert_eq!(conn.from.unit.0, 1);
        assert_eq!(conn.to.unit.0, 2);
        assert_eq!(conn.label, Some("test_stream".to_string()));
        assert_eq!(conn.direction, FlowDirection::Reverse);
    }

    #[test]
    fn test_unit_info() {
        let id = UnitId(5);
        let info = UnitInfo::new(id).with_name("mixer").with_ports(3, 1);

        assert_eq!(info.id.0, 5);
        assert_eq!(info.name, Some("mixer".to_string()));
        assert_eq!(info.num_inputs, 3);
        assert_eq!(info.num_outputs, 1);
    }

    #[test]
    fn test_flowsheet_with_equations() {
        let mut flowsheet = Flowsheet::<Dynamic>::new();

        // Add units
        let unit = models::CSTR::new(100.0, 1.0, 350.0);
        let _u1 = flowsheet.add_unit("reactor", unit);

        // Add equations to the system
        let residual = ResidualFunction::from_dynamic(
            "mass_balance",
            vec!["accumulation".to_string()],
            |v, names| v.get(&names[0]).copied().unwrap_or(0.0),
        );

        flowsheet.equation_system.add_differential(residual);

        assert_eq!(flowsheet.equation_system.differential_count(), 1);
        assert_eq!(flowsheet.unit_count(), 1);
    }

    /// Multi-Component Stream Tests
    #[test]
    fn test_multicomponent_stream_creation() {
        let stream = Stream::<MolarFlow, UninitializedConditions>::new(
            100.0,
            vec!["Water".to_string(), "Ethanol".to_string()],
        )
        .at_conditions(298.15, 101325.0);

        assert_eq!(stream.total_flow, 100.0);
        assert_eq!(stream.n_components(), 2);
        assert_eq!(stream.component_name(0), "Water");
        assert_eq!(stream.component_name(1), "Ethanol");
    }

    #[test]
    fn test_multicomponent_stream_set_composition() {
        let mut stream = Stream::<MolarFlow, UninitializedConditions>::new(
            100.0,
            vec!["N2".to_string(), "O2".to_string()],
        )
        .at_conditions(298.15, 101325.0);

        // Valid composition (sums to 1.0)
        let result = stream.set_composition(vec![0.78, 0.22]);
        assert!(result.is_ok());
        assert_eq!(stream.composition[0], 0.78);
        assert_eq!(stream.composition[1], 0.22);

        // Invalid composition (doesn't sum to 1.0)
        let result = stream.set_composition(vec![0.5, 0.4]);
        assert!(result.is_err());

        // Negative composition
        let result = stream.set_composition(vec![1.2, -0.2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multicomponent_stream_component_flows() {
        let mut stream = Stream::<MolarFlow, UninitializedConditions>::new(
            100.0,
            vec!["Water".to_string(), "Ethanol".to_string()],
        )
        .at_conditions(298.15, 101325.0);

        stream.set_composition(vec![0.6, 0.4]).unwrap();

        assert_eq!(stream.component_flow(0), 60.0); // Water: 60 mol/s
        assert_eq!(stream.component_flow(1), 40.0); // Ethanol: 40 mol/s

        let flows = stream.component_flows();
        assert_eq!(flows, vec![60.0, 40.0]);
    }

    #[test]
    fn test_multicomponent_stream_set_component_flows() {
        let mut stream = Stream::<MolarFlow, UninitializedConditions>::new(
            0.0,
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .at_conditions(298.15, 101325.0);

        stream.set_component_flows(vec![30.0, 50.0, 20.0]).unwrap();

        assert_eq!(stream.total_flow, 100.0);
        assert_eq!(stream.get_composition(0), 0.3);
        assert_eq!(stream.get_composition(1), 0.5);
        assert_eq!(stream.get_composition(2), 0.2);
    }

    #[test]
    fn test_multicomponent_stream_pure() {
        let stream = Stream::<MassFlow>::pure(50.0, "Water".to_string(), 298.15, 101325.0);

        assert_eq!(stream.total_flow, 50.0);
        assert_eq!(stream.n_components(), 1);
        assert_eq!(stream.composition[0], 1.0);
        assert_eq!(stream.component_name(0), "Water");
    }

    #[test]
    fn test_multicomponent_stream_mix() {
        use crate::models::Mixer;

        let mut mixer = Mixer::<2, _>::new(2).with_inlets_configured();
        mixer.inlet_flows = [100.0, 50.0];
        mixer.inlet_temps = [298.15, 298.15];
        // inlet_compositions[component_idx][inlet_idx]
        // Component 0 (A): 0.8 in inlet 0, 0.4 in inlet 1
        // Component 1 (B): 0.2 in inlet 0, 0.6 in inlet 1
        mixer.inlet_compositions = vec![
            [0.8, 0.4], // Component A
            [0.2, 0.6], // Component B
        ];

        // Simulate what the solver would compute
        mixer.outlet_flow = Var::new(150.0);
        mixer.outlet_temp = Var::new(298.15);
        mixer.outlet_composition = vec![Var::new(0.667), Var::new(0.333)];

        assert_eq!(mixer.outlet_flow.get(), 150.0);
        // A: (100*0.8 + 50*0.4) / 150 = (80 + 20) / 150 = 0.667
        // B: (100*0.2 + 50*0.6) / 150 = (20 + 30) / 150 = 0.333
        assert!(
            (mixer.outlet_composition[0].get() - 0.667).abs() < 1e-3,
            "Expected 0.667, got {}",
            mixer.outlet_composition[0].get()
        );
        assert!(
            (mixer.outlet_composition[1].get() - 0.333).abs() < 1e-3,
            "Expected 0.333, got {}",
            mixer.outlet_composition[1].get()
        );
    }

    #[test]
    fn test_multicomponent_stream_validation() {
        let stream = Stream::<MolarFlow, _>::with_composition(
            100.0,
            vec!["A".to_string(), "B".to_string()],
            vec![0.6, 0.4],
        )
        .unwrap()
        .at_conditions(298.15, 101325.0);

        assert!(stream.validate().is_ok());

        // Create invalid stream with negative flow
        let bad_stream =
            Stream::<MolarFlow, UninitializedConditions>::new(-10.0, vec!["A".to_string()])
                .at_conditions(298.15, 101325.0);
        assert!(bad_stream.validate().is_err());
    }

    // Conservation Law Trait Tests
    // Define test unit operations for conservation trait testing
    struct TestReactor;

    impl UnitOp for TestReactor {
        type In = Stream<MolarFlow>;
        type Out = Stream<MolarFlow>;
        fn build_equations<T: TimeDomain>(&self, _: &mut EquationSystem<T>, _: &str) {}
    }

    impl ConservesMass for TestReactor {
        fn mass_balance(&self) -> Equation<MassBalance> {
            let mut eq = Equation::new("reactor_mass");
            eq.set_residual(0.0); // Balanced
            eq
        }
    }

    impl ConservesEnergy for TestReactor {
        fn energy_balance(&self) -> Equation<EnergyBalance> {
            let mut eq = Equation::new("reactor_energy");
            eq.set_residual(0.0); // Balanced
            eq
        }
    }

    #[test]
    fn test_conservation_traits() {
        let reactor = TestReactor;

        // Test mass conservation
        let mass_eq = reactor.mass_balance();
        assert_eq!(mass_eq.name, "reactor_mass");

        // Test energy conservation
        let energy_eq = reactor.energy_balance();
        assert_eq!(energy_eq.name, "reactor_energy");
    }

    #[test]
    fn test_validate_mass_conservation() {
        let reactor = TestReactor;

        // Should pass (residual = 0.0)
        let result = validate_mass_conservation(&reactor, 1e-6);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_energy_conservation() {
        let reactor = TestReactor;

        // Should pass (residual = 0.0)
        let result = validate_energy_conservation(&reactor, 1e-6);
        assert!(result.is_ok());
    }

    struct UnbalancedUnit;

    impl UnitOp for UnbalancedUnit {
        type In = Stream<MolarFlow>;
        type Out = Stream<MolarFlow>;
        fn build_equations<T: TimeDomain>(&self, _: &mut EquationSystem<T>, _: &str) {}
    }

    impl ConservesMass for UnbalancedUnit {
        fn mass_balance(&self) -> Equation<MassBalance> {
            let mut eq = Equation::new("unbalanced_mass");
            eq.set_residual(10.0); // NOT balanced
            eq
        }
    }

    #[test]
    fn test_validate_mass_conservation_fails() {
        let unit = UnbalancedUnit;

        let result = validate_mass_conservation(&unit, 1e-6);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not satisfied"));
    }

    // Test compile-time enforcement via generics
    fn require_mass_conservation<U: ConservesMass>(_unit: &U) -> bool {
        true
    }

    #[test]
    fn test_generic_conservation_requirement() {
        let reactor = TestReactor;

        // This compiles because TestReactor implements ConservesMass
        assert!(require_mass_conservation(&reactor));
    }

    // Test const generic port counting
    #[test]
    fn test_const_port_counting_siso() {
        let unit = UnitWithPortCounts::<1, 1, MolarFlow>::new();
        assert_eq!(unit.input_count(), 1);
        assert_eq!(unit.output_count(), 1);
    }

    #[test]
    fn test_const_port_counting_mixer() {
        let mut mixer = UnitWithPortCounts::<2, 1, MolarFlow>::new();
        mixer.set_inputs([10.0, 20.0]);

        assert_eq!(mixer.input_count(), 2);
        assert_eq!(mixer.output_count(), 1);
        assert_eq!(mixer.get_inputs()[0], 10.0);
        assert_eq!(mixer.get_inputs()[1], 20.0);
    }

    #[test]
    fn test_const_port_counting_splitter() {
        let mut splitter = UnitWithPortCounts::<1, 2, MolarFlow>::new();
        splitter.set_outputs([5.0, 15.0]);

        assert_eq!(splitter.input_count(), 1);
        assert_eq!(splitter.output_count(), 2);
        assert_eq!(splitter.get_outputs()[0], 5.0);
        assert_eq!(splitter.get_outputs()[1], 15.0);
    }

    #[test]
    fn test_const_port_type_aliases() {
        use crate::port_configs::*;

        let _siso: SISO<MolarFlow> = SISO::new();
        let _mixer: Mixer2<MolarFlow> = Mixer2::new();
        let _splitter: Splitter2<MolarFlow> = Splitter2::new();
        let _distillation: Distillation<MolarFlow> = Distillation::new();

        // Type system ensures correct port counts
        let mixer: Mixer2<MolarFlow> = UnitWithPortCounts::<2, 1, MolarFlow>::new();
        assert_eq!(mixer.input_count(), 2);
    }

    #[test]
    fn test_const_port_default() {
        let unit: UnitWithPortCounts<3, 2, MolarFlow> = Default::default();
        assert_eq!(unit.input_count(), 3);
        assert_eq!(unit.output_count(), 2);
    }
}
