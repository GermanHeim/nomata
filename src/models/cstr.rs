//! Continuous Stirred Tank Reactor (CSTR) model.
//!
//! # State Variables
//! - Volume (holdup): Differential variable
//! - Concentration: Differential variable
//! - Temperature: Differential variable
//!
//! # Parameters
//! - Pre-exponential factor (k0)
//! - Activation energy (Ea)
//! - Heat of reaction (deltaH)
//!
//! # Balance Equations
//! - Mass balance: dV/dt = F_in - F_out
//! - Component balance: d(VC)/dt = F_in*C_in - F_out*C - V*k*C
//! - Energy balance: d(VrhoCpT)/dt = F_in*rho*Cp*T_in - F_out*rho*Cp*T + V*k*C*(-deltaH) + Q
//!
//! # Example
//!
//! ```
//! use nomata::models::CSTR;
//!
//! // Type-safe initialization using phantom states
//! let cstr = CSTR::new(
//!     100.0,  // Initial volume (L)
//!     1.0,    // Initial concentration (mol/L)
//!     350.0,  // Initial temperature (K)
//! )
//! .with_kinetics(1e8, 8000.0)  // k0 (1/s), Ea (J/mol)
//! .with_thermodynamics(-50000.0, 1000.0, 4184.0);  // deltaH, rho, Cp
//!
//! // Only fully initialized reactors can compute rates
//! // cstr.compute_reaction_rate();  // Compiles
//! ```

use crate::*;
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

// Typed Equation Variable Structs (Generic over scalar type for autodiff)

use crate::{EquationVarsGeneric, Scalar};

/// Variables for steady-state mass balance: F_in - F_out = 0
pub struct SteadyMassBalanceVars<S: Scalar> {
    pub f_in: S,
    pub f_out: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for SteadyMassBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["F_in", "F_out"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
        })
    }
}

impl EquationVars for SteadyMassBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for steady-state component balance: F_in*C_in - F_out*C - V*r = 0
pub struct SteadyComponentBalanceVars<S: Scalar> {
    pub f_in: S,
    pub c_in: S,
    pub f_out: S,
    pub c: S,
    pub v: S,
    pub r: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for SteadyComponentBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["F_in", "C_in", "F_out", "C", "V", "r"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            c_in: *vars.get(&format!("{}_C_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
            c: *vars.get(&format!("{}_C", prefix))?,
            v: *vars.get(&format!("{}_V", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
        })
    }
}

impl EquationVars for SteadyComponentBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for steady-state energy balance
pub struct SteadyEnergyBalanceVars<S: Scalar> {
    pub f_in: S,
    pub f_out: S,
    pub rho: S,
    pub cp: S,
    pub t_in: S,
    pub t: S,
    pub v: S,
    pub r: S,
    pub delta_h: S,
    pub q: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for SteadyEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["F_in", "F_out", "rho", "Cp", "T_in", "T", "V", "r", "deltaH", "Q"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
            rho: *vars.get(&format!("{}_rho", prefix))?,
            cp: *vars.get(&format!("{}_Cp", prefix))?,
            t_in: *vars.get(&format!("{}_T_in", prefix))?,
            t: *vars.get(&format!("{}_T", prefix))?,
            v: *vars.get(&format!("{}_V", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
            delta_h: *vars.get(&format!("{}_deltaH", prefix))?,
            q: *vars.get(&format!("{}_Q", prefix))?,
        })
    }
}

impl EquationVars for SteadyEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic mass balance: dV/dt - (F_in - F_out) = 0
pub struct DynamicMassBalanceVars<S: Scalar> {
    pub dv_dt: S,
    pub f_in: S,
    pub f_out: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for DynamicMassBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dV_dt", "F_in", "F_out"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dv_dt: *vars.get(&format!("{}_dV_dt", prefix))?,
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
        })
    }
}

impl EquationVars for DynamicMassBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic component balance: d(V*C)/dt - (F_in*C_in - F_out*C - V*r) = 0
pub struct DynamicComponentBalanceVars<S: Scalar> {
    pub dvc_dt: S,
    pub f_in: S,
    pub c_in: S,
    pub f_out: S,
    pub c: S,
    pub v: S,
    pub r: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for DynamicComponentBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dVC_dt", "F_in", "C_in", "F_out", "C", "V", "r"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dvc_dt: *vars.get(&format!("{}_dVC_dt", prefix))?,
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            c_in: *vars.get(&format!("{}_C_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
            c: *vars.get(&format!("{}_C", prefix))?,
            v: *vars.get(&format!("{}_V", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
        })
    }
}

impl EquationVars for DynamicComponentBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic energy balance
pub struct DynamicEnergyBalanceVars<S: Scalar> {
    pub dvrhocpt_dt: S,
    pub f_in: S,
    pub f_out: S,
    pub rho: S,
    pub cp: S,
    pub t_in: S,
    pub t: S,
    pub v: S,
    pub r: S,
    pub delta_h: S,
    pub q: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for DynamicEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dVrhoCpT_dt", "F_in", "F_out", "rho", "Cp", "T_in", "T", "V", "r", "deltaH", "Q"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dvrhocpt_dt: *vars.get(&format!("{}_dVrhoCpT_dt", prefix))?,
            f_in: *vars.get(&format!("{}_F_in", prefix))?,
            f_out: *vars.get(&format!("{}_F_out", prefix))?,
            rho: *vars.get(&format!("{}_rho", prefix))?,
            cp: *vars.get(&format!("{}_Cp", prefix))?,
            t_in: *vars.get(&format!("{}_T_in", prefix))?,
            t: *vars.get(&format!("{}_T", prefix))?,
            v: *vars.get(&format!("{}_V", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
            delta_h: *vars.get(&format!("{}_deltaH", prefix))?,
            q: *vars.get(&format!("{}_Q", prefix))?,
        })
    }
}

impl EquationVars for DynamicEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for Arrhenius equation: k - k0*exp(-Ea/(R*T)) = 0
pub struct ArrheniusVars<S: Scalar> {
    pub k: S,
    pub k0: S,
    pub ea: S,
    pub t: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for ArrheniusVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["k", "k0", "Ea", "T"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            k: *vars.get(&format!("{}_k", prefix))?,
            k0: *vars.get(&format!("{}_k0", prefix))?,
            ea: *vars.get(&format!("{}_Ea", prefix))?,
            t: *vars.get(&format!("{}_T", prefix))?,
        })
    }
}

impl EquationVars for ArrheniusVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for reaction rate: r - k*C = 0
pub struct ReactionRateVars<S: Scalar> {
    pub r: S,
    pub k: S,
    pub c: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for ReactionRateVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["r", "k", "C"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            r: *vars.get(&format!("{}_r", prefix))?,
            k: *vars.get(&format!("{}_k", prefix))?,
            c: *vars.get(&format!("{}_C", prefix))?,
        })
    }
}

impl EquationVars for ReactionRateVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Phantom type marker for uninitialized state.
pub struct Uninitialized;

/// Phantom type marker for initialized state.
pub struct Initialized;

/// Continuous Stirred Tank Reactor with single reaction.
///
/// Type parameters enforce compile-time initialization:
/// - `K`: Kinetics state (Uninitialized | Initialized)
/// - `T`: Thermodynamics state (Uninitialized | Initialized)
/// - `PortState`: Port connection state
pub struct CSTR<K = Uninitialized, T = Uninitialized, PortState = Disconnected>
where
    PortState: crate::PortState,
{
    // State variables (Differential)
    pub volume: Var<Differential>,
    pub concentration: Var<Differential>,
    pub temperature: Var<Differential>,

    // Kinetic parameters (Some when K = Initialized)
    k0: Option<Var<Parameter>>, // Pre-exponential factor (1/s)
    activation_energy: Option<Var<Parameter>>, // Ea (J/mol)

    // Thermodynamic parameters (Some when T = Initialized)
    heat_of_reaction: Option<Var<Parameter>>, // deltaH (J/mol)
    density: Option<Var<Parameter>>,          // rho (kg/m^3)
    heat_capacity: Option<Var<Parameter>>,    // Cp (J/kg*K)

    // Algebraic variables (computed)
    pub rate_constant: Var<Algebraic>, // k(T) = k0*exp(-Ea/RT)
    pub reaction_rate: Var<Algebraic>, // r = k*C

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, PortState>,
    pub outlet: Port<Stream<MolarFlow>, Output, PortState>,

    // Operating conditions
    pub feed_flow: f64,          // F_in (L/s)
    pub feed_concentration: f64, // C_in (mol/L)
    pub feed_temperature: f64,   // T_in (K)
    pub heat_duty: f64,          // Q (J/s)

    // Optional: store fluid for thermodynamic calculations
    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<crate::thermodynamics::fluids::Pure>,

    // Phantom data for compile-time state tracking
    _kinetics_state: PhantomData<K>,
    _thermo_state: PhantomData<T>,
}

// Initial constructor - all parameters uninitialized
impl CSTR<Uninitialized, Uninitialized, Disconnected> {
    /// Creates a new CSTR with initial state conditions.
    ///
    /// Kinetics and thermodynamics must be initialized before simulation.
    pub fn new(volume: f64, concentration: f64, temperature: f64) -> Self {
        CSTR {
            volume: Var::new(volume),
            concentration: Var::new(concentration),
            temperature: Var::new(temperature),

            k0: None,
            activation_energy: None,

            heat_of_reaction: None,
            density: None,
            heat_capacity: None,

            rate_constant: Var::new(0.0),
            reaction_rate: Var::new(0.0),

            inlet: Port::new(),
            outlet: Port::new(),

            feed_flow: 0.0,
            feed_concentration: 0.0,
            feed_temperature: 0.0,
            heat_duty: 0.0,

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _kinetics_state: PhantomData,
            _thermo_state: PhantomData,
        }
    }
}

// Kinetics initialization (K: Uninitialized → Initialized)
impl<T, P> CSTR<Uninitialized, T, P>
where
    P: crate::PortState,
{
    /// Sets kinetic parameters, transitioning to kinetics-initialized state.
    pub fn with_kinetics(self, k0: f64, ea: f64) -> CSTR<Initialized, T, P> {
        CSTR {
            volume: self.volume,
            concentration: self.concentration,
            temperature: self.temperature,

            k0: Some(Var::new(k0)),
            activation_energy: Some(Var::new(ea)),

            heat_of_reaction: self.heat_of_reaction,
            density: self.density,
            heat_capacity: self.heat_capacity,

            rate_constant: self.rate_constant,
            reaction_rate: self.reaction_rate,

            inlet: self.inlet,
            outlet: self.outlet,

            feed_flow: self.feed_flow,
            feed_concentration: self.feed_concentration,
            feed_temperature: self.feed_temperature,
            heat_duty: self.heat_duty,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _kinetics_state: PhantomData,
            _thermo_state: self._thermo_state,
        }
    }
}

// Thermodynamics initialization (T: Uninitialized → Initialized)
impl<K, P> CSTR<K, Uninitialized, P>
where
    P: crate::PortState,
{
    /// Sets thermodynamic parameters manually, transitioning to thermo-initialized state.
    pub fn with_thermodynamics(self, delta_h: f64, rho: f64, cp: f64) -> CSTR<K, Initialized, P> {
        CSTR {
            volume: self.volume,
            concentration: self.concentration,
            temperature: self.temperature,

            k0: self.k0,
            activation_energy: self.activation_energy,

            heat_of_reaction: Some(Var::new(delta_h)),
            density: Some(Var::new(rho)),
            heat_capacity: Some(Var::new(cp)),

            rate_constant: self.rate_constant,
            reaction_rate: self.reaction_rate,

            inlet: self.inlet,
            outlet: self.outlet,

            feed_flow: self.feed_flow,
            feed_concentration: self.feed_concentration,
            feed_temperature: self.feed_temperature,
            heat_duty: self.heat_duty,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _kinetics_state: self._kinetics_state,
            _thermo_state: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets thermodynamic parameters from rfluids backend.
    pub fn with_thermodynamics_from_fluid(
        self,
        pure: Pure,
        delta_h: f64,
        temp: f64,
        pressure: f64,
    ) -> Result<CSTR<K, Initialized, P>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;

        Ok(CSTR {
            volume: self.volume,
            concentration: self.concentration,
            temperature: self.temperature,

            k0: self.k0,
            activation_energy: self.activation_energy,

            heat_of_reaction: Some(Var::new(delta_h)),
            density: Some(Var::new(props.density)),
            heat_capacity: Some(Var::new(props.cp)),

            rate_constant: self.rate_constant,
            reaction_rate: self.reaction_rate,

            inlet: self.inlet,
            outlet: self.outlet,

            feed_flow: self.feed_flow,
            feed_concentration: self.feed_concentration,
            feed_temperature: self.feed_temperature,
            heat_duty: self.heat_duty,

            fluid: Some(pure),

            _kinetics_state: self._kinetics_state,
            _thermo_state: PhantomData,
        })
    }
}

// Computation methods (only available when FULLY initialized)
impl<P> CSTR<Initialized, Initialized, P>
where
    P: crate::PortState,
{
    /// Computes rate constant: k = k0 * exp(-Ea/RT)
    ///
    /// Only available for fully initialized reactors.
    pub fn compute_rate_constant(&mut self) {
        const R: f64 = 8.314; // J/(mol*K)
        let k0 = self.k0.as_ref().unwrap().get();
        let ea = self.activation_energy.as_ref().unwrap().get();
        let k = k0 * (-ea / (R * self.temperature.get())).exp();
        self.rate_constant = Var::new(k);
    }

    /// Computes reaction rate: r = k * C
    ///
    /// Only available for fully initialized reactors.
    pub fn compute_reaction_rate(&mut self) {
        let r = self.rate_constant.get() * self.concentration.get();
        self.reaction_rate = Var::new(r);
    }

    /// Gets kinetic parameters (guaranteed to exist).
    pub fn k0(&self) -> f64 {
        self.k0.as_ref().unwrap().get()
    }

    pub fn activation_energy(&self) -> f64 {
        self.activation_energy.as_ref().unwrap().get()
    }

    /// Gets thermodynamic parameters (guaranteed to exist).
    pub fn heat_of_reaction(&self) -> f64 {
        self.heat_of_reaction.as_ref().unwrap().get()
    }

    pub fn density(&self) -> f64 {
        self.density.as_ref().unwrap().get()
    }

    pub fn heat_capacity(&self) -> f64 {
        self.heat_capacity.as_ref().unwrap().get()
    }
}

// Operating condition setters (available at any initialization state)
impl<K, T, P> CSTR<K, T, P>
where
    P: crate::PortState,
{
    /// Sets operating conditions.
    pub fn set_operating_conditions(
        &mut self,
        feed_flow: f64,
        feed_conc: f64,
        feed_temp: f64,
        heat_duty: f64,
    ) {
        self.feed_flow = feed_flow;
        self.feed_concentration = feed_conc;
        self.feed_temperature = feed_temp;
        self.heat_duty = heat_duty;
    }

    /// Creates mass balance equation.
    pub fn mass_balance(&self) -> Equation<MassBalance> {
        self.volume.mass_balance("cstr_mass")
    }

    /// Creates component balance equation.
    pub fn component_balance(&self) -> Equation<MassBalance> {
        self.concentration.mass_balance("cstr_component")
    }

    /// Creates energy balance equation.
    pub fn energy_balance(&self) -> Equation<EnergyBalance> {
        self.temperature.energy_balance("cstr_energy")
    }
}

/// Port-based interface for CSTR.
impl<K, T, P: crate::PortState> HasPorts for CSTR<K, T, P> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MolarFlow")]
    }
}

/// UnitOp implementation with equation harvesting.
#[cfg(not(feature = "autodiff"))]
impl<K, Thermo, P: crate::PortState> UnitOp for CSTR<K, Thermo, P> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // For a CSTR, we have balance equations that differ based on time domain:
        // Dynamic: dV/dt = F_in - F_out, d(V*C)/dt = ..., d(V*rho*Cp*T)/dt = ...
        // Steady:  0 = F_in - F_out, 0 = ..., 0 = ...

        if T::IS_STEADY {
            // Steady-state mode: no accumulation terms

            // Mass balance: F_in - F_out = 0
            let mass_balance = ResidualFunction::from_typed(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: SteadyMassBalanceVars<f64>| v.f_in - v.f_out,
            );
            system.add_algebraic(mass_balance);

            // Component balance: F_in * C_in - F_out * C - V * r = 0
            let component_balance = ResidualFunction::from_typed(
                &format!("{}_component_balance", unit_name),
                unit_name,
                |v: SteadyComponentBalanceVars<f64>| v.f_in * v.c_in - v.f_out * v.c - v.v * v.r,
            );
            system.add_algebraic(component_balance);

            // Energy balance: F_in * rho * Cp * T_in - F_out * rho * Cp * T + V * r * (-deltaH) + Q = 0
            let energy_balance = ResidualFunction::from_typed(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: SteadyEnergyBalanceVars<f64>| {
                    v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                        + v.v * v.r * (-v.delta_h)
                        + v.q
                },
            );
            system.add_algebraic(energy_balance);
        } else {
            // Dynamic mode: include accumulation terms

            // Mass balance: dV/dt - (F_in - F_out) = 0
            let mass_balance = ResidualFunction::from_typed(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: DynamicMassBalanceVars<f64>| v.dv_dt - (v.f_in - v.f_out),
            );
            system.add_differential(mass_balance);

            // Component balance: d(V*C)/dt - (F_in * C_in - F_out * C - V * r) = 0
            let component_balance = ResidualFunction::from_typed(
                &format!("{}_component_balance", unit_name),
                unit_name,
                |v: DynamicComponentBalanceVars<f64>| {
                    v.dvc_dt - (v.f_in * v.c_in - v.f_out * v.c - v.v * v.r)
                },
            );
            system.add_differential(component_balance);

            // Energy balance: d(V*rho*Cp*T)/dt - (F_in*rho*Cp*T_in - F_out*rho*Cp*T + V*r*(-deltaH) + Q) = 0
            let energy_balance = ResidualFunction::from_typed(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: DynamicEnergyBalanceVars<f64>| {
                    v.dvrhocpt_dt
                        - (v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                            + v.v * v.r * (-v.delta_h)
                            + v.q)
                },
            );
            system.add_differential(energy_balance);
        }

        // Algebraic equations (same for both modes)

        // Arrhenius equation: k - k0 * exp(-Ea / (R * T)) = 0
        let arrhenius = ResidualFunction::from_typed(
            &format!("{}_arrhenius", unit_name),
            unit_name,
            |v: ArrheniusVars<f64>| v.k - v.k0 * (-v.ea / (8.314 * v.t)).exp(),
        );
        system.add_algebraic(arrhenius);

        // Reaction rate: r - k * C = 0 (first-order kinetics)
        let rate_eq = ResidualFunction::from_typed(
            &format!("{}_reaction_rate", unit_name),
            unit_name,
            |v: ReactionRateVars<f64>| v.r - v.k * v.c,
        );
        system.add_algebraic(rate_eq);
    }
}

/// UnitOp implementation for CSTR with autodiff support.
#[cfg(feature = "autodiff")]
impl<K, Thermo, P: crate::PortState> UnitOp for CSTR<K, Thermo, P> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::{Dual64, DualNum};

        // For a CSTR, we have balance equations that differ based on time domain:
        // Dynamic: dV/dt = F_in - F_out, d(V*C)/dt = ..., d(V*rho*Cp*T)/dt = ...
        // Steady:  0 = F_in - F_out, 0 = ..., 0 = ...

        if T::IS_STEADY {
            // Steady-state mode: no accumulation terms

            // Mass balance: F_in - F_out = 0
            let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: SteadyMassBalanceVars<f64>| v.f_in - v.f_out,
                |v: SteadyMassBalanceVars<Dual64>| v.f_in - v.f_out,
            );
            system.add_algebraic(mass_balance);

            // Component balance: F_in * C_in - F_out * C - V * r = 0
            let component_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_component_balance", unit_name),
                unit_name,
                |v: SteadyComponentBalanceVars<f64>| v.f_in * v.c_in - v.f_out * v.c - v.v * v.r,
                |v: SteadyComponentBalanceVars<Dual64>| v.f_in * v.c_in - v.f_out * v.c - v.v * v.r,
            );
            system.add_algebraic(component_balance);

            // Energy balance: F_in * rho * Cp * T_in - F_out * rho * Cp * T + V * r * (-deltaH) + Q = 0
            let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: SteadyEnergyBalanceVars<f64>| {
                    v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                        + v.v * v.r * (-v.delta_h)
                        + v.q
                },
                |v: SteadyEnergyBalanceVars<Dual64>| {
                    v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                        + v.v * v.r * (-v.delta_h)
                        + v.q
                },
            );
            system.add_algebraic(energy_balance);
        } else {
            // Dynamic mode: include accumulation terms

            // Mass balance: dV/dt - (F_in - F_out) = 0
            let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: DynamicMassBalanceVars<f64>| v.dv_dt - (v.f_in - v.f_out),
                |v: DynamicMassBalanceVars<Dual64>| v.dv_dt - (v.f_in - v.f_out),
            );
            system.add_differential(mass_balance);

            // Component balance: d(V*C)/dt - (F_in * C_in - F_out * C - V * r) = 0
            let component_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_component_balance", unit_name),
                unit_name,
                |v: DynamicComponentBalanceVars<f64>| {
                    v.dvc_dt - (v.f_in * v.c_in - v.f_out * v.c - v.v * v.r)
                },
                |v: DynamicComponentBalanceVars<Dual64>| {
                    v.dvc_dt - (v.f_in * v.c_in - v.f_out * v.c - v.v * v.r)
                },
            );
            system.add_differential(component_balance);

            // Energy balance: d(V*rho*Cp*T)/dt - (F_in*rho*Cp*T_in - F_out*rho*Cp*T + V*r*(-deltaH) + Q) = 0
            let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: DynamicEnergyBalanceVars<f64>| {
                    v.dvrhocpt_dt
                        - (v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                            + v.v * v.r * (-v.delta_h)
                            + v.q)
                },
                |v: DynamicEnergyBalanceVars<Dual64>| {
                    v.dvrhocpt_dt
                        - (v.f_in * v.rho * v.cp * v.t_in - v.f_out * v.rho * v.cp * v.t
                            + v.v * v.r * (-v.delta_h)
                            + v.q)
                },
            );
            system.add_differential(energy_balance);
        }

        // Algebraic equations (same for both modes)

        // Arrhenius equation: k - k0 * exp(-Ea / (R * T)) = 0
        let arrhenius = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_arrhenius", unit_name),
            unit_name,
            |v: ArrheniusVars<f64>| v.k - v.k0 * (-v.ea / (8.314 * v.t)).exp(),
            |v: ArrheniusVars<Dual64>| v.k - v.k0 * (-v.ea / (Dual64::from(8.314) * v.t)).exp(),
        );
        system.add_algebraic(arrhenius);

        // Reaction rate: r - k * C = 0 (first-order kinetics)
        let rate_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_reaction_rate", unit_name),
            unit_name,
            |v: ReactionRateVars<f64>| v.r - v.k * v.c,
            |v: ReactionRateVars<Dual64>| v.r - v.k * v.c,
        );
        system.add_algebraic(rate_eq);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cstr_creation() {
        let cstr = CSTR::new(100.0, 1.0, 350.0);
        assert_eq!(cstr.volume.get(), 100.0);
        assert_eq!(cstr.concentration.get(), 1.0);
        assert_eq!(cstr.temperature.get(), 350.0);
    }

    #[test]
    fn test_cstr_phantom_type_transitions() {
        // Test kinetics initialization
        let cstr = CSTR::new(100.0, 1.0, 350.0).with_kinetics(1e8, 8000.0);
        assert_eq!(cstr.k0.as_ref().unwrap().get(), 1e8);
        assert_eq!(cstr.activation_energy.as_ref().unwrap().get(), 8000.0);

        // Test thermodynamics initialization
        let cstr = CSTR::new(100.0, 1.0, 350.0).with_thermodynamics(-50000.0, 1000.0, 4184.0);
        assert_eq!(cstr.heat_of_reaction.as_ref().unwrap().get(), -50000.0);
        assert_eq!(cstr.density.as_ref().unwrap().get(), 1000.0);
        assert_eq!(cstr.heat_capacity.as_ref().unwrap().get(), 4184.0);

        // Test full initialization
        let cstr = CSTR::new(100.0, 1.0, 350.0)
            .with_kinetics(1e8, 8000.0)
            .with_thermodynamics(-50000.0, 1000.0, 4184.0);
        assert_eq!(cstr.k0.as_ref().unwrap().get(), 1e8);
        assert_eq!(cstr.density(), 1000.0);
    }

    #[test]
    fn test_rate_constant_calculation() {
        let mut cstr = CSTR::new(100.0, 1.0, 350.0)
            .with_kinetics(1e8, 8000.0)
            .with_thermodynamics(-50000.0, 1000.0, 4184.0);

        cstr.compute_rate_constant();
        assert!(cstr.rate_constant.get() > 0.0);
    }

    #[test]
    #[cfg(feature = "thermodynamics")]
    fn test_thermodynamic_initialization() {
        use crate::thermodynamics::fluids::Pure;

        let result = CSTR::new(100.0, 1.0, 350.0)
            .with_kinetics(1e8, 8000.0)
            .with_thermodynamics_from_fluid(Pure::Water, -50000.0, 350.0, 101325.0);

        assert!(result.is_ok());
        let cstr = result.unwrap();
        assert!(cstr.density() > 0.0);
        assert!(cstr.heat_capacity() > 0.0);
    }
}
