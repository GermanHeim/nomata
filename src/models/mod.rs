//! Common chemical engineering unit operation models.
//!
//! This module provides typed implementations of standard unit operations
//! organized as individual files:
//!
//! - **CSTR**: Continuous Stirred Tank Reactor with reaction kinetics
//! - **PFR**: Plug Flow Reactor with spatial discretization
//! - **HeatExchanger**: Shell-and-tube heat exchanger
//! - **FlashSeparator**: Vapor-liquid equilibrium separator
//! - **Mixer**: Combines multiple inlet streams
//! - **Splitter**: Divides one stream into multiple outlets
//! - **Compressor**: Gas compression with isentropic efficiency
//! - **Pump**: Liquid pumping with head calculations
//! - **Valve**: Control valve for flow regulation
//! - **MAP**: Component mapper for lumping/delumping operations
//!
//! All models use the type system to enforce correct connections and
//! proper variable roles (Parameter, Algebraic, Differential).
//!
//! ## Thermodynamics Integration
//!
//! When the `thermodynamics` feature is enabled, models can automatically
//! compute physical properties (density, heat capacity, vapor pressure, etc.)
//! using the CoolProp library through rfluids.
//!
//! ## Example
//!
//! ```
//! use nomata::models::CSTR;
//! use nomata::BalanceBuilder;
//!
//! let cstr = CSTR::new(100.0, 1.0, 350.0)
//!     .with_kinetics(1e8, 8000.0)
//!     .with_thermodynamics(-45000.0, 1000.0, 4184.0);
//!
//! // Build balance equations with compile-time verification
//! let mass_eq = cstr.mass_balance();
//! ```
//!
//! # Creating a New Model
//!
//! This section demonstrates how to create a type-safe unit operation model
//! following the nomata patterns.
//!
//! ## Step 1: Define Typed Equation Variable Structs
//!
//! Each equation needs a struct implementing `EquationVars` that defines
//! which variables participate in the equation. This provides compile-time
//! type safety.
//!
//! ```
//! use nomata::EquationVars;
//! use std::collections::HashMap;
//!
//! /// Variables for a simple valve pressure drop equation:
//! /// dP - Cv * sqrt(F) = 0
//! pub struct ValvePressureDropVars {
//!     pub dp: f64,         // Pressure drop across valve
//!     pub cv_sqrt_f: f64,  // Cv * sqrt(flow)
//! }
//!
//! impl EquationVars for ValvePressureDropVars {
//!     fn base_names() -> &'static [&'static str] {
//!         // Variable names without prefix (unit name is added automatically)
//!         &["dP", "Cv_sqrt_F"]
//!     }
//!
//!     fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
//!         // Extract variables from the solver's HashMap using the unit prefix
//!         Some(Self {
//!             dp: *vars.get(&format!("{}_dP", prefix))?,
//!             cv_sqrt_f: *vars.get(&format!("{}_Cv_sqrt_F", prefix))?,
//!         })
//!     }
//! }
//! ```
//!
//! ## Step 2: Define the Model Struct with Type-State Pattern
//!
//! Use phantom types to enforce initialization order at compile time:
//!
//! ```
//! use nomata::{
//!     Var, Parameter, Algebraic, Differential,
//!     Port, Stream, MassFlow, Input, Output,
//!     PortState, Disconnected,
//! };
//! use std::marker::PhantomData;
//!
//! /// Marker for uninitialized state
//! pub struct Uninitialized;
//! /// Marker for initialized state
//! pub struct Initialized;
//!
//! /// A simple control valve model.
//! ///
//! /// Type parameters:
//! /// - `C`: Configuration state (Uninitialized -> Initialized)
//! /// - `P`: Port connection state (Disconnected -> Connected)
//! pub struct Valve<C = Uninitialized, P = Disconnected>
//! where
//!     P: PortState,
//! {
//!     // State variables (computed by solver)
//!     pub pressure_drop: Var<Algebraic>,
//!
//!     // Parameters (user-specified, fixed during solve)
//!     pub cv: Option<Var<Parameter>>,           // Valve coefficient
//!     pub opening: Option<Var<Parameter>>,      // Valve opening fraction
//!
//!     // Ports for stream connections
//!     pub inlet: Port<Stream<MassFlow>, Input, P>,
//!     pub outlet: Port<Stream<MassFlow>, Output, P>,
//!
//!     // PhantomData to track configuration state
//!     _config: PhantomData<C>,
//! }
//! ```
//!
//! ## Step 3: Implement Constructors with State Transitions
//!
//! The builder pattern enforces that configuration happens before use:
//!
//! ```ignore
//! impl Valve {
//!     /// Creates a new valve in uninitialized state.
//!     pub fn new() -> Self {
//!         Valve {
//!             pressure_drop: Var::new(0.0),
//!             cv: None,
//!             opening: None,
//!             inlet: Port::new(),
//!             outlet: Port::new(),
//!             _config: PhantomData,
//!         }
//!     }
//! }
//!
//! impl<P: PortState> Valve<Uninitialized, P> {
//!     /// Sets valve parameters. Transitions from Uninitialized to Initialized.
//!     pub fn with_configuration(self, cv: f64, opening: f64) -> Valve<Initialized, P> {
//!         Valve {
//!             pressure_drop: self.pressure_drop,
//!             cv: Some(Var::new(cv)),
//!             opening: Some(Var::new(opening)),
//!             inlet: self.inlet,
//!             outlet: self.outlet,
//!             _config: PhantomData,
//!         }
//!     }
//! }
//! ```
//!
//! ## Step 4: Implement UnitOp Trait
//!
//! The `UnitOp` trait defines stream types and equation generation:
//!
//! ```ignore
//! use nomata::{UnitOp, EquationSystem, TimeDomain, ResidualFunction};
//!
//! impl<C, P: PortState> UnitOp for Valve<C, P> {
//!     type In = Stream<MassFlow>;
//!     type Out = Stream<MassFlow>;
//!
//!     fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
//!         // Use from_typed for compile-time type safety
//!         let pressure_eq = ResidualFunction::from_typed(
//!             &format!("{}_pressure_drop", unit_name),
//!             unit_name,
//!             |v: ValvePressureDropVars| v.dp - v.cv_sqrt_f,
//!         );
//!         system.add_algebraic(pressure_eq);
//!     }
//! }
//! ```
//!
//! ## Step 5: Implement HasPorts Trait
//!
//! For flowsheet connectivity, implement `HasPorts`:
//!
//! ```ignore
//! use nomata::{HasPorts, NamedPort};
//!
//! impl<C, P: PortState> HasPorts for Valve<C, P> {
//!     fn input_ports(&self) -> Vec<NamedPort> {
//!         vec![NamedPort::input("inlet", "MassFlow")]
//!     }
//!
//!     fn output_ports(&self) -> Vec<NamedPort> {
//!         vec![NamedPort::output("outlet", "MassFlow")]
//!     }
//! }
//! ```
//!
//! ## Dynamic Variable Counts
//!
//! For models where variable count is determined at runtime (e.g., mixers with
//! N inlets), use `ResidualFunction::from_dynamic` instead:
//!
//! ```ignore
//! // For a mixer with N inlets, variable names are generated at runtime
//! let var_names: Vec<String> = (0..n_inlets)
//!     .map(|i| format!("{}_inlet_{}_F", unit_name, i))
//!     .chain(std::iter::once(format!("{}_outlet_F", unit_name)))
//!     .collect();
//!
//! let mass_balance = ResidualFunction::from_dynamic(
//!     &format!("{}_mass_balance", unit_name),
//!     var_names.clone(),
//!     move |vars: &std::collections::HashMap<String, f64>, names: &[String]| {
//!         // Sum all inlets
//!         let inlet_sum: f64 = names[..names.len()-1]
//!             .iter()
//!             .filter_map(|n| vars.get(n))
//!             .sum();
//!         // Residual: outlet - sum(inlets) = 0
//!         let outlet = vars.get(&names[names.len()-1]).copied().unwrap_or(0.0);
//!         outlet - inlet_sum
//!     },
//! );
//! ```
//!
//! ## Automatic Differentiation Support
//!
//! All models in this module support automatic differentiation when the
//! `autodiff` feature is enabled. This enables efficient Jacobian computation
//! for Newton-based solvers.
//!
//! ### How It Works
//!
//! The autodiff support is built on three key components:
//!
//! 1. **`Scalar` trait**: A trait defining mathematical operations (`exp`, `ln`,
//!    `sqrt`, `powf`, `abs`, `sin`, `cos`) implemented for both `f64` and
//!    `num_dual::Dual64`.
//!
//! 2. **`EquationVarsGeneric<S: Scalar>`**: A generic version of `EquationVars`
//!    that works with any scalar type, allowing the same struct definition to
//!    work with both regular and dual numbers.
//!
//! 3. **`from_typed_generic_with_dual()`**: A constructor that takes two
//!    closures - one for `f64` and one for `Dual64` - enabling automatic
//!    differentiation.
//!
//! ### Creating Autodiff-Enabled Models
//!
//! To create a model that supports autodiff:
//!
//! **Step 1**: Make your Vars struct generic over the scalar type:
//!
//! ```ignore
//! use nomata::{Scalar, EquationVarsGeneric, EquationVars};
//! use std::collections::HashMap;
//!
//! /// Variables for Arrhenius equation: k = k0 * exp(-Ea/(R*T))
//! pub struct ArrheniusVars<S: Scalar> {
//!     pub k: S,
//!     pub k0: S,
//!     pub ea: S,
//!     pub t: S,
//! }
//!
//! impl<S: Scalar> EquationVarsGeneric<S> for ArrheniusVars<S> {
//!     fn base_names() -> &'static [&'static str] {
//!         &["k", "k0", "Ea", "T"]
//!     }
//!
//!     fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
//!         Some(Self {
//!             k: vars.get(&format!("{}_k", prefix))?.clone(),
//!             k0: vars.get(&format!("{}_k0", prefix))?.clone(),
//!             ea: vars.get(&format!("{}_Ea", prefix))?.clone(),
//!             t: vars.get(&format!("{}_T", prefix))?.clone(),
//!         })
//!     }
//! }
//!
//! // Backwards compatibility: implement EquationVars for f64
//! impl EquationVars for ArrheniusVars<f64> {
//!     fn base_names() -> &'static [&'static str] {
//!         <Self as EquationVarsGeneric<f64>>::base_names()
//!     }
//!
//!     fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
//!         <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
//!     }
//! }
//! ```
//!
//! **Step 2**: Use conditional compilation for `build_equations`:
//!
//! ```ignore
//! use nomata::{UnitOp, ResidualFunction, Scalar};
//!
//! // Without autodiff feature
//! #[cfg(not(feature = "autodiff"))]
//! fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
//!     let arrhenius = ResidualFunction::from_typed(
//!         &format!("{}_arrhenius", unit_name),
//!         unit_name,
//!         |v: ArrheniusVars<f64>| v.k - v.k0 * (-v.ea / (8.314 * v.t)).exp(),
//!     );
//!     system.add_algebraic(arrhenius);
//! }
//!
//! // With autodiff feature
//! #[cfg(feature = "autodiff")]
//! fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
//!     use num_dual::Dual64;
//!     
//!     let arrhenius = ResidualFunction::from_typed_generic_with_dual(
//!         &format!("{}_arrhenius", unit_name),
//!         unit_name,
//!         // f64 closure
//!         |v: ArrheniusVars<f64>| v.k - v.k0 * (-v.ea / (8.314 * v.t)).exp(),
//!         // Dual64 closure - .exp() comes from DualNum trait
//!         |v: ArrheniusVars<Dual64>| {
//!             v.k - v.k0 * (-v.ea / (Dual64::from(8.314) * v.t)).exp()
//!         },
//!     );
//!     system.add_algebraic(arrhenius);
//! }
//! ```
//!
//! ### The `Scalar` Trait
//!
//! The `Scalar` trait provides a unified interface for mathematical operations:
//!
//! | Method | Description |
//! |--------|-------------|
//! | `exp(self)` | Exponential function |
//! | `ln(self)` | Natural logarithm |
//! | `sqrt(self)` | Square root |
//! | `powf(self, n)` | Power function |
//! | `abs(self)` | Absolute value |
//! | `sin(self)` | Sine function |
//! | `cos(self)` | Cosine function |
//!
//! For `Dual64`, these methods propagate derivatives correctly, enabling
//! automatic Jacobian computation.
//!
//! ### Which Models Support Autodiff?
//!
//! All models in this module support autodiff for their typed equations:
//!
//! | Model | Typed Equations with Autodiff |
//! |-------|-------------------------------|
//! | `Pump` | Head equation, Power equation |
//! | `Valve` | Flow equation (with sqrt) |
//! | `HeatExchanger` | Energy balances, Heat transfer |
//! | `Compressor` | Isentropic temp, Outlet temp, Power |
//! | `Splitter` | Split fraction, Temperature continuity |
//! | `FlashSeparator` | Mass/energy balances, VLE equations |
//! | `CSTR` | All balances, Arrhenius (with exp) |
//! | `PFR` | Segment mass/energy balances |
//! | `MAP` | Temperature passthrough |
//! | `Mixer` | Uses `from_dynamic` (no autodiff) |
//!
//! **Note**: Equations using `from_dynamic()` (for variable component counts)
//! do not support autodiff. For these equations, the solver falls back to
//! numerical differentiation.
//!
//! ## Variable Roles Summary
//!
//! | Role | Type | Description |
//! |------|------|-------------|
//! | `Parameter` | `Var<Parameter>` | User-specified, fixed during solve |
//! | `Algebraic` | `Var<Algebraic>` | Computed by solver, no time derivative |
//! | `Differential` | `Var<Differential>` | Has time derivative (dynamic models) |
//!
//! ## Best Practices
//!
//! 1. **Type Safety First**: Always use `from_typed()` when variable count is
//!    known at compile time
//! 2. **Meaningful Names**: Variable names should be descriptive
//!    (e.g., `inlet_T`, `outlet_P`)
//! 3. **Prefix Convention**: Use `{unit_name}_{variable}` format for uniqueness
//! 4. **State Transitions**: Use phantom types to enforce initialization order
//! 5. **Documentation**: Document physical meaning of equations in the struct

mod compressor;
mod cstr;
mod flash_separator;
mod heat_exchanger;
mod map;
mod mixer;
mod pfr;
mod pump;
mod splitter;
mod valve;

pub use compressor::Compressor;
pub use cstr::CSTR;
pub use flash_separator::FlashSeparator;
pub use heat_exchanger::HeatExchanger;
pub use map::MAP;
pub use mixer::Mixer;
pub use pfr::PFR;
pub use pump::Pump;
pub use splitter::Splitter;
pub use valve::Valve;
