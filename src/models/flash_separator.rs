//! Flash Separator model.
//!
//! Flash separator for vapor-liquid equilibrium.
//!
//! # State Variables
//! - Liquid holdup
//! - Vapor fraction
//! - Temperature
//! - Pressure
//!
//! # Equations
//! - Mass balance: dM/dt = F_in - L - V
//! - Component balances: x_i, y_i from equilibrium
//! - Energy balance
//! - VLE relations: y_i = K_i * x_i
//!
//! # Example
//!
//! ```
//! use nomata::models::FlashSeparator;
//!
//! // Type-safe initialization
//! let flash = FlashSeparator::new(10.0, 2)  // volume, n_components
//!     .with_k_values(vec![3.0, 0.5])
//!     .with_initial_state(350.0, 101325.0);  // T, P
//!
//! // Only fully initialized separators can perform flash calculations
//! // flash.flash_calculation();  // ✓ Compiles
//! ```

use crate::*;
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::fluids::Pure;

// Typed Equation Variable Structs (Generic over scalar type for autodiff)

use crate::{EquationVarsGeneric, Scalar};

/// Variables for steady-state mass balance: 0 = F_in - V - L
pub struct FlashMassBalanceVars<S: Scalar> {
    pub f_in: S,
    pub v: S,
    pub l: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for FlashMassBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["f_in", "v", "l"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_in: *vars.get(&format!("{}_f_in", prefix))?,
            v: *vars.get(&format!("{}_v", prefix))?,
            l: *vars.get(&format!("{}_l", prefix))?,
        })
    }
}

impl EquationVars for FlashMassBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for steady-state energy balance: 0 = F_in*H_in - V*H_v - L*H_l
pub struct FlashEnergyBalanceVars<S: Scalar> {
    pub f_in: S,
    pub h_in: S,
    pub v: S,
    pub h_v: S,
    pub l: S,
    pub h_l: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for FlashEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["f_in", "h_in", "v", "h_v", "l", "h_l"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            f_in: *vars.get(&format!("{}_f_in", prefix))?,
            h_in: *vars.get(&format!("{}_h_in", prefix))?,
            v: *vars.get(&format!("{}_v", prefix))?,
            h_v: *vars.get(&format!("{}_h_v", prefix))?,
            l: *vars.get(&format!("{}_l", prefix))?,
            h_l: *vars.get(&format!("{}_h_l", prefix))?,
        })
    }
}

impl EquationVars for FlashEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic mass balance: dM/dt = F_in - V - L
pub struct FlashDynMassBalanceVars<S: Scalar> {
    pub dm_dt: S,
    pub f_in: S,
    pub v: S,
    pub l: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for FlashDynMassBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dm_dt", "f_in", "v", "l"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dm_dt: *vars.get(&format!("{}_dm_dt", prefix))?,
            f_in: *vars.get(&format!("{}_f_in", prefix))?,
            v: *vars.get(&format!("{}_v", prefix))?,
            l: *vars.get(&format!("{}_l", prefix))?,
        })
    }
}

impl EquationVars for FlashDynMassBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic energy balance: d(M*H)/dt = F_in*H_in - V*H_v - L*H_l
pub struct FlashDynEnergyBalanceVars<S: Scalar> {
    pub d_mh_dt: S,
    pub f_in: S,
    pub h_in: S,
    pub v: S,
    pub h_v: S,
    pub l: S,
    pub h_l: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for FlashDynEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["d_mh_dt", "f_in", "h_in", "v", "h_v", "l", "h_l"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            d_mh_dt: *vars.get(&format!("{}_d_mh_dt", prefix))?,
            f_in: *vars.get(&format!("{}_f_in", prefix))?,
            h_in: *vars.get(&format!("{}_h_in", prefix))?,
            v: *vars.get(&format!("{}_v", prefix))?,
            h_v: *vars.get(&format!("{}_h_v", prefix))?,
            l: *vars.get(&format!("{}_l", prefix))?,
            h_l: *vars.get(&format!("{}_h_l", prefix))?,
        })
    }
}

impl EquationVars for FlashDynEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for VLE equation: y_i = K_i * x_i
pub struct VleVars<S: Scalar> {
    pub y: S,
    pub k: S,
    pub x: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for VleVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["y", "k", "x"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            y: *vars.get(&format!("{}_y", prefix))?,
            k: *vars.get(&format!("{}_k", prefix))?,
            x: *vars.get(&format!("{}_x", prefix))?,
        })
    }
}

impl EquationVars for VleVars<f64> {
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

/// Flash separator for vapor-liquid equilibrium.
///
/// Type parameters enforce compile-time initialization:
/// - `E`: Equilibrium data state (Uninitialized | Initialized)  
/// - `S`: State initialization (Uninitialized | Initialized)
pub struct FlashSeparator<E = Uninitialized, S = Uninitialized> {
    // State variables
    pub holdup: Var<Differential>,
    pub temperature: Var<Differential>,
    pub pressure: Var<Algebraic>,
    pub vapor_fraction: Var<Algebraic>,

    // Parameters
    volume: Var<Parameter>,

    // Equilibrium constants (Some when E = Initialized)
    k_values: Option<Vec<Var<Parameter>>>,

    // Component compositions
    pub n_components: usize,
    feed_composition: Vec<f64>,
    pub liquid_composition: Vec<Var<Algebraic>>,
    pub vapor_composition: Vec<Var<Algebraic>>,

    // Flow rates
    pub inlet_flow: Var<Algebraic>,
    pub vapor_flow: Var<Algebraic>,
    pub liquid_flow: Var<Algebraic>,

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, Disconnected>,
    pub vapor_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,
    pub liquid_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,

    #[cfg(feature = "thermodynamics")]
    pub component_names: Vec<String>,

    // Phantom data for compile-time state tracking
    _equilibrium_state: PhantomData<E>,
    _state_init: PhantomData<S>,
}

// Initial constructor
impl FlashSeparator<Uninitialized, Uninitialized> {
    /// Creates a new flash separator with vessel volume and number of components.
    ///
    /// K-values and initial state must be set before flash calculations.
    pub fn new(volume: f64, n_components: usize) -> Self {
        FlashSeparator {
            holdup: Var::new(volume * 0.5),
            temperature: Var::new(298.15),
            pressure: Var::new(101325.0),
            vapor_fraction: Var::new(0.5),

            volume: Var::new(volume),

            k_values: None,

            n_components,
            feed_composition: vec![1.0 / n_components as f64; n_components],
            liquid_composition: (0..n_components).map(|_| Var::new(0.0)).collect(),
            vapor_composition: (0..n_components).map(|_| Var::new(0.0)).collect(),

            inlet_flow: Var::new(0.0),
            vapor_flow: Var::new(0.0),
            liquid_flow: Var::new(0.0),

            inlet: Port::new(),
            vapor_outlet: Port::new(),
            liquid_outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            component_names: Vec::new(),

            _equilibrium_state: PhantomData,
            _state_init: PhantomData,
        }
    }
}

// Equilibrium initialization (E: Uninitialized → Initialized)
impl<S> FlashSeparator<Uninitialized, S> {
    /// Sets equilibrium K-values manually, transitioning to equilibrium-initialized state.
    pub fn with_k_values(self, k_vals: Vec<f64>) -> FlashSeparator<Initialized, S> {
        assert_eq!(
            k_vals.len(),
            self.n_components,
            "K-values length {} must match n_components {}",
            k_vals.len(),
            self.n_components
        );

        FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: Some(k_vals.into_iter().map(Var::new).collect()),

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet_flow: self.inlet_flow,
            vapor_flow: self.vapor_flow,
            liquid_flow: self.liquid_flow,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            #[cfg(feature = "thermodynamics")]
            component_names: self.component_names,

            _equilibrium_state: PhantomData,
            _state_init: self._state_init,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Computes K-values from thermodynamic properties, transitioning to equilibrium-initialized state.
    pub fn with_k_values_from_thermo(
        self,
        components: &[Pure],
        temp: f64,
        pressure: f64,
    ) -> Result<FlashSeparator<Initialized, S>, crate::thermodynamics::ThermoError> {
        assert_eq!(components.len(), self.n_components);

        let mut k_vals = Vec::new();

        for pure in components {
            let fluid = crate::thermodynamics::Fluid::new(*pure);
            let t_crit = fluid.critical_temperature()?;

            if temp >= t_crit {
                return Err(crate::thermodynamics::ThermoError::InvalidInput(format!(
                    "Temperature {} K exceeds critical temperature {} K for {:?}",
                    temp, t_crit, pure
                )));
            }

            let p_sat = Self::compute_vapor_pressure_rfluids(&fluid, temp)?;
            let k = p_sat / pressure;
            k_vals.push(k);
        }

        Ok(FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: Some(k_vals.into_iter().map(Var::new).collect()),

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet_flow: self.inlet_flow,
            vapor_flow: self.vapor_flow,
            liquid_flow: self.liquid_flow,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            component_names: components.iter().map(|p| format!("{:?}", p)).collect(),

            _equilibrium_state: PhantomData,
            _state_init: self._state_init,
        })
    }

    #[cfg(feature = "thermodynamics")]
    fn compute_vapor_pressure_rfluids(
        fluid: &crate::thermodynamics::Fluid,
        temperature: f64,
    ) -> Result<f64, crate::thermodynamics::ThermoError> {
        use rfluids::prelude::*;

        let rfluids_fluid = match &fluid.substance {
            crate::thermodynamics::Substance::Pure(p) => rfluids::fluid::Fluid::from(*p),
            crate::thermodynamics::Substance::PredefinedMix(m) => rfluids::fluid::Fluid::from(*m),
            _ => {
                return Err(crate::thermodynamics::ThermoError::InvalidInput(
                    "Only pure fluids and predefined mixtures supported for VLE".to_string(),
                ));
            }
        };

        let mut state = rfluids_fluid
            .in_state(FluidInput::temperature(temperature), FluidInput::quality(0.5))
            .map_err(|e| {
                crate::thermodynamics::ThermoError::InvalidInput(format!(
                    "Failed to compute saturation state at {} K: {:?}",
                    temperature, e
                ))
            })?;

        let p_sat = state
            .pressure()
            .map_err(|_| crate::thermodynamics::ThermoError::PropertyNotAvailable)?;

        Ok(p_sat)
    }
}

// State initialization (S: Uninitialized → Initialized)
impl<E> FlashSeparator<E, Uninitialized> {
    /// Sets initial temperature and pressure, transitioning to state-initialized.
    pub fn with_initial_state(
        mut self,
        temp: f64,
        pressure: f64,
    ) -> FlashSeparator<E, Initialized> {
        self.temperature = Var::new(temp);
        self.pressure = Var::new(pressure);

        FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: self.k_values,

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet_flow: self.inlet_flow,
            vapor_flow: self.vapor_flow,
            liquid_flow: self.liquid_flow,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            #[cfg(feature = "thermodynamics")]
            component_names: self.component_names,

            _equilibrium_state: self._equilibrium_state,
            _state_init: PhantomData,
        }
    }
}

// Methods available at any state
impl<E, S> FlashSeparator<E, S> {
    /// Sets feed composition (must sum to 1.0).
    pub fn set_feed_composition(&mut self, composition: Vec<f64>) {
        assert_eq!(composition.len(), self.n_components);
        self.feed_composition = composition;
    }

    /// Sets the inlet flow rate from a stream.
    pub fn set_inlet_flow(&mut self, flow: f64) {
        self.inlet_flow = Var::new(flow);
    }

    /// Creates mass balance equation.
    pub fn mass_balance(&self) -> Equation<MassBalance> {
        self.holdup.mass_balance("flash_mass")
    }

    /// Creates energy balance equation.
    pub fn energy_balance(&self) -> Equation<EnergyBalance> {
        self.temperature.energy_balance("flash_energy")
    }
}

// Flash calculation (only available when FULLY initialized)
impl FlashSeparator<Initialized, Initialized> {
    /// Performs Rachford-Rice flash calculation.
    ///
    /// Only available for fully initialized flash separators.
    pub fn flash_calculation(&mut self) {
        let _temp = self.temperature.get();
        let _pressure = self.pressure.get();

        let mut vapor_frac = 0.5; // Initial guess

        // Rachford-Rice iteration (simplified)
        for _ in 0..10 {
            let mut sum = 0.0;
            for i in 0..self.n_components {
                let k = self.k_values.as_ref().unwrap()[i].get();
                let z = self.feed_composition[i];
                sum += z * (k - 1.0) / (1.0 + vapor_frac * (k - 1.0));
            }

            if sum.abs() < 1e-6 {
                break;
            }

            // Newton step
            let mut deriv = 0.0;
            for i in 0..self.n_components {
                let k = self.k_values.as_ref().unwrap()[i].get();
                let z = self.feed_composition[i];
                let denom = 1.0 + vapor_frac * (k - 1.0);
                deriv -= z * (k - 1.0).powi(2) / (denom * denom);
            }

            vapor_frac -= sum / deriv;
            vapor_frac = vapor_frac.clamp(0.0, 1.0);
        }

        self.vapor_fraction = Var::new(vapor_frac);

        // Compute compositions
        for i in 0..self.n_components {
            let k = self.k_values.as_ref().unwrap()[i].get();
            let z = self.feed_composition[i];

            let x = z / (1.0 + vapor_frac * (k - 1.0));
            let y = k * x;

            self.liquid_composition[i] = Var::new(x);
            self.vapor_composition[i] = Var::new(y);
        }
    }

    /// Gets K-value for component i (guaranteed to exist).
    pub fn k_value(&self, i: usize) -> f64 {
        self.k_values.as_ref().unwrap()[i].get()
    }

    /// Gets all K-values.
    pub fn k_values(&self) -> Vec<f64> {
        self.k_values.as_ref().unwrap().iter().map(|k| k.get()).collect()
    }

    /// Creates the vapor outlet stream from flash.
    ///
    /// Returns a stream with vapor composition and conditions.
    /// Flow rate is computed as inlet_flow * vapor_fraction.
    fn compute_vapor_outlet_stream(&self) -> Result<Stream<MolarFlow>, crate::StreamError> {
        let flow = self.inlet_flow.get() * self.vapor_fraction.get();
        let temp = self.temperature.get();
        let pressure = self.pressure.get();

        // Get vapor composition
        let composition: Vec<f64> = self.vapor_composition.iter().map(|v| v.get()).collect();

        #[cfg(feature = "thermodynamics")]
        {
            if !self.component_names.is_empty()
                && let Ok(stream) = Stream::with_composition(
                    flow,
                    self.component_names.clone(),
                    composition.clone(),
                )
            {
                return Ok(stream.at_conditions(temp, pressure));
            }
        }

        // Fallback: create generic multicomponent stream
        let component_names: Vec<String> =
            (0..self.n_components).map(|i| format!("Component_{}", i)).collect();

        Stream::with_composition(flow, component_names, composition)
            .map(|s| s.at_conditions(temp, pressure))
            .map_err(|_| crate::StreamError::RequiresCalculation {
                model: "FlashSeparator (vapor)".to_string(),
                calculation_type: "VLE calculation".to_string(),
            })
    }

    /// Creates the liquid outlet stream from flash.
    ///
    /// Returns a stream with liquid composition and conditions.
    /// Flow rate is computed as inlet_flow * (1 - vapor_fraction).
    fn compute_liquid_outlet_stream(&self) -> Result<Stream<MolarFlow>, crate::StreamError> {
        let flow = self.inlet_flow.get() * (1.0 - self.vapor_fraction.get());
        let temp = self.temperature.get();
        let pressure = self.pressure.get();

        // Get liquid composition
        let composition: Vec<f64> = self.liquid_composition.iter().map(|v| v.get()).collect();

        #[cfg(feature = "thermodynamics")]
        {
            if !self.component_names.is_empty()
                && let Ok(stream) = Stream::with_composition(
                    flow,
                    self.component_names.clone(),
                    composition.clone(),
                )
            {
                return Ok(stream.at_conditions(temp, pressure));
            }
        }

        // Fallback: create generic multicomponent stream
        let component_names: Vec<String> =
            (0..self.n_components).map(|i| format!("Component_{}", i)).collect();

        Stream::with_composition(flow, component_names, composition)
            .map(|s| s.at_conditions(temp, pressure))
            .map_err(|_| crate::StreamError::RequiresCalculation {
                model: "FlashSeparator (liquid)".to_string(),
                calculation_type: "VLE calculation".to_string(),
            })
    }

    /// Returns a reference to the vapor outlet stream.
    pub fn vapor_outlet_stream(&self) -> crate::OutletRef {
        crate::OutletRef::new("FlashSeparator", "vapor")
    }

    /// Returns a reference to the liquid outlet stream.
    pub fn liquid_outlet_stream(&self) -> crate::OutletRef {
        crate::OutletRef::new("FlashSeparator", "liquid")
    }

    /// Populates the vapor outlet reference with the current computed stream.
    pub fn populate_vapor_outlet(
        &self,
        outlet_ref: &crate::OutletRef,
    ) -> Result<(), crate::StreamError> {
        let stream = self.compute_vapor_outlet_stream()?;
        outlet_ref.set(stream);
        Ok(())
    }

    /// Populates the liquid outlet reference with the current computed stream.
    pub fn populate_liquid_outlet(
        &self,
        outlet_ref: &crate::OutletRef,
    ) -> Result<(), crate::StreamError> {
        let stream = self.compute_liquid_outlet_stream()?;
        outlet_ref.set(stream);
        Ok(())
    }
}

/// Port-based interface for FlashSeparator.
impl<E, S> HasPorts for FlashSeparator<E, S> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![
            NamedPort::output("vapor_outlet", "MolarFlow"),
            NamedPort::output("liquid_outlet", "MolarFlow"),
        ]
    }
}

/// Compile-time port specification for FlashSeparator.
///
/// Enables type-safe connections with const generic port indices.
impl<E, S> PortSpec for FlashSeparator<E, S> {
    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = 2;
    const STREAM_TYPE: &'static str = "MolarFlow";
}

/// UnitOp implementation for FlashSeparator.
#[cfg(not(feature = "autodiff"))]
impl<E, S> UnitOp for FlashSeparator<E, S> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        let n_comp = self.n_components;

        if T::IS_STEADY {
            // Steady-state flash: no accumulation
            // Overall mass balance: 0 = F_in - V - L
            let mass_balance = ResidualFunction::from_typed(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: FlashMassBalanceVars<f64>| v.f_in - v.v - v.l,
            );
            system.add_algebraic(mass_balance);

            // Component balances: 0 = F_in*z_i - V*y_i - L*x_i
            for i in 0..n_comp {
                let vars =
                    vec![format!("F_in_z_{}", i), format!("V_y_{}", i), format!("L_x_{}", i)];
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, i),
                    vars,
                    move |v, names| {
                        let f_in_z = v.get(&names[0]).copied().unwrap_or(0.0);
                        let v_y = v.get(&names[1]).copied().unwrap_or(0.0);
                        let l_x = v.get(&names[2]).copied().unwrap_or(0.0);
                        f_in_z - v_y - l_x
                    },
                );
                system.add_algebraic(comp_balance);
            }

            // Energy balance: 0 = F_in*H_in - V*H_v - L*H_l
            let energy_balance = ResidualFunction::from_typed(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: FlashEnergyBalanceVars<f64>| v.f_in * v.h_in - v.v * v.h_v - v.l * v.h_l,
            );
            system.add_algebraic(energy_balance);
        } else {
            // Dynamic flash: include accumulation terms
            // Overall mass balance: dM/dt = F_in - V - L
            let mass_balance = ResidualFunction::from_typed(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: FlashDynMassBalanceVars<f64>| v.dm_dt - v.f_in + v.v + v.l,
            );
            system.add_differential(mass_balance);

            // Component balances: d(M*z_i)/dt = F_in*z_i - V*y_i - L*x_i
            for i in 0..n_comp {
                let vars = vec![
                    format!("d_Mz_{}_dt", i),
                    format!("F_in_z_{}", i),
                    format!("V_y_{}", i),
                    format!("L_x_{}", i),
                ];
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, i),
                    vars,
                    move |v, names| {
                        let d_mz_dt = v.get(&names[0]).copied().unwrap_or(0.0);
                        let f_in_z = v.get(&names[1]).copied().unwrap_or(0.0);
                        let v_y = v.get(&names[2]).copied().unwrap_or(0.0);
                        let l_x = v.get(&names[3]).copied().unwrap_or(0.0);
                        d_mz_dt - f_in_z + v_y + l_x
                    },
                );
                system.add_differential(comp_balance);
            }

            // Energy balance: d(M*H)/dt = F_in*H_in - V*H_v - L*H_l
            let energy_balance = ResidualFunction::from_typed(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: FlashDynEnergyBalanceVars<f64>| {
                    v.d_mh_dt - v.f_in * v.h_in + v.v * v.h_v + v.l * v.h_l
                },
            );
            system.add_differential(energy_balance);
        }

        // VLE equilibrium relations: y_i = K_i * x_i
        for i in 0..n_comp {
            let prefix = format!("{}_vle_{}", unit_name, i);
            let vle = ResidualFunction::from_typed(
                &format!("{}_vle_{}", unit_name, i),
                &prefix,
                |v: VleVars<f64>| v.y - v.k * v.x,
            );
            system.add_algebraic(vle);
        }

        // Rachford-Rice equation: sum(z_i*(K_i - 1)/(1 + V_frac*(K_i - 1))) = 0
        let rr_vars: Vec<String> = (0..n_comp).map(|i| format!("z_{}_K_term_{}", i, i)).collect();
        let rachford_rice = ResidualFunction::from_dynamic(
            &format!("{}_rachford_rice", unit_name),
            rr_vars,
            move |v, names| names.iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum(),
        );
        system.add_algebraic(rachford_rice);

        // Summation constraints: sum(x_i) = 1
        let mut x_vars: Vec<String> = (0..n_comp).map(|i| format!("x_{}", i)).collect();
        x_vars.push("one".to_string());
        let x_sum = ResidualFunction::from_dynamic(
            &format!("{}_x_sum", unit_name),
            x_vars,
            move |v, names| {
                let sum: f64 =
                    names[..n_comp].iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum();
                let one = v.get(&names[n_comp]).copied().unwrap_or(1.0);
                sum - one
            },
        );
        system.add_algebraic(x_sum);

        // Summation constraints: sum(y_i) = 1
        let mut y_vars: Vec<String> = (0..n_comp).map(|i| format!("y_{}", i)).collect();
        y_vars.push("one".to_string());
        let y_sum = ResidualFunction::from_dynamic(
            &format!("{}_y_sum", unit_name),
            y_vars,
            move |v, names| {
                let sum: f64 =
                    names[..n_comp].iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum();
                let one = v.get(&names[n_comp]).copied().unwrap_or(1.0);
                sum - one
            },
        );
        system.add_algebraic(y_sum);
    }
}

/// UnitOp implementation for FlashSeparator with autodiff support.
#[cfg(feature = "autodiff")]
impl<E, S> UnitOp for FlashSeparator<E, S> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        let n_comp = self.n_components;

        if T::IS_STEADY {
            // Steady-state flash: no accumulation
            // Overall mass balance: 0 = F_in - V - L
            let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: FlashMassBalanceVars<f64>| v.f_in - v.v - v.l,
                |v: FlashMassBalanceVars<Dual64>| v.f_in - v.v - v.l,
            );
            system.add_algebraic(mass_balance);

            // Component balances: 0 = F_in*z_i - V*y_i - L*x_i
            // (from_dynamic equations don't have autodiff support)
            for i in 0..n_comp {
                let vars =
                    vec![format!("F_in_z_{}", i), format!("V_y_{}", i), format!("L_x_{}", i)];
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, i),
                    vars,
                    move |v, names| {
                        let f_in_z = v.get(&names[0]).copied().unwrap_or(0.0);
                        let v_y = v.get(&names[1]).copied().unwrap_or(0.0);
                        let l_x = v.get(&names[2]).copied().unwrap_or(0.0);
                        f_in_z - v_y - l_x
                    },
                );
                system.add_algebraic(comp_balance);
            }

            // Energy balance: 0 = F_in*H_in - V*H_v - L*H_l
            let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: FlashEnergyBalanceVars<f64>| v.f_in * v.h_in - v.v * v.h_v - v.l * v.h_l,
                |v: FlashEnergyBalanceVars<Dual64>| v.f_in * v.h_in - v.v * v.h_v - v.l * v.h_l,
            );
            system.add_algebraic(energy_balance);
        } else {
            // Dynamic flash: include accumulation terms
            // Overall mass balance: dM/dt = F_in - V - L
            let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_mass_balance", unit_name),
                unit_name,
                |v: FlashDynMassBalanceVars<f64>| v.dm_dt - v.f_in + v.v + v.l,
                |v: FlashDynMassBalanceVars<Dual64>| v.dm_dt - v.f_in + v.v + v.l,
            );
            system.add_differential(mass_balance);

            // Component balances: d(M*z_i)/dt = F_in*z_i - V*y_i - L*x_i
            // (from_dynamic equations don't have autodiff support)
            for i in 0..n_comp {
                let vars = vec![
                    format!("d_Mz_{}_dt", i),
                    format!("F_in_z_{}", i),
                    format!("V_y_{}", i),
                    format!("L_x_{}", i),
                ];
                let comp_balance = ResidualFunction::from_dynamic(
                    &format!("{}_component_{}_balance", unit_name, i),
                    vars,
                    move |v, names| {
                        let d_mz_dt = v.get(&names[0]).copied().unwrap_or(0.0);
                        let f_in_z = v.get(&names[1]).copied().unwrap_or(0.0);
                        let v_y = v.get(&names[2]).copied().unwrap_or(0.0);
                        let l_x = v.get(&names[3]).copied().unwrap_or(0.0);
                        d_mz_dt - f_in_z + v_y + l_x
                    },
                );
                system.add_differential(comp_balance);
            }

            // Energy balance: d(M*H)/dt = F_in*H_in - V*H_v - L*H_l
            let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_energy_balance", unit_name),
                unit_name,
                |v: FlashDynEnergyBalanceVars<f64>| {
                    v.d_mh_dt - v.f_in * v.h_in + v.v * v.h_v + v.l * v.h_l
                },
                |v: FlashDynEnergyBalanceVars<Dual64>| {
                    v.d_mh_dt - v.f_in * v.h_in + v.v * v.h_v + v.l * v.h_l
                },
            );
            system.add_differential(energy_balance);
        }

        // VLE equilibrium relations: y_i = K_i * x_i
        for i in 0..n_comp {
            let prefix = format!("{}_vle_{}", unit_name, i);
            let vle = ResidualFunction::from_typed_generic_with_dual(
                &format!("{}_vle_{}", unit_name, i),
                &prefix,
                |v: VleVars<f64>| v.y - v.k * v.x,
                |v: VleVars<Dual64>| v.y - v.k * v.x,
            );
            system.add_algebraic(vle);
        }

        // Rachford-Rice equation: sum(z_i*(K_i - 1)/(1 + V_frac*(K_i - 1))) = 0
        // (from_dynamic equations don't have autodiff support)
        let rr_vars: Vec<String> = (0..n_comp).map(|i| format!("z_{}_K_term_{}", i, i)).collect();
        let rachford_rice = ResidualFunction::from_dynamic(
            &format!("{}_rachford_rice", unit_name),
            rr_vars,
            move |v, names| names.iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum(),
        );
        system.add_algebraic(rachford_rice);

        // Summation constraints: sum(x_i) = 1
        // (from_dynamic equations don't have autodiff support)
        let mut x_vars: Vec<String> = (0..n_comp).map(|i| format!("x_{}", i)).collect();
        x_vars.push("one".to_string());
        let x_sum = ResidualFunction::from_dynamic(
            &format!("{}_x_sum", unit_name),
            x_vars,
            move |v, names| {
                let sum: f64 =
                    names[..n_comp].iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum();
                let one = v.get(&names[n_comp]).copied().unwrap_or(1.0);
                sum - one
            },
        );
        system.add_algebraic(x_sum);

        // Summation constraints: sum(y_i) = 1
        // (from_dynamic equations don't have autodiff support)
        let mut y_vars: Vec<String> = (0..n_comp).map(|i| format!("y_{}", i)).collect();
        y_vars.push("one".to_string());
        let y_sum = ResidualFunction::from_dynamic(
            &format!("{}_y_sum", unit_name),
            y_vars,
            move |v, names| {
                let sum: f64 =
                    names[..n_comp].iter().map(|n| v.get(n).copied().unwrap_or(0.0)).sum();
                let one = v.get(&names[n_comp]).copied().unwrap_or(1.0);
                sum - one
            },
        );
        system.add_algebraic(y_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_separator_creation() {
        let flash = FlashSeparator::new(10.0, 3);
        assert_eq!(flash.volume.get(), 10.0);
        assert!(flash.k_values.is_none());
        assert_eq!(flash.n_components, 3);
    }

    #[test]
    fn test_flash_phantom_type_transitions() {
        // Test equilibrium initialization
        let flash: FlashSeparator<Initialized, Initialized> = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);
        assert_eq!(flash.k_value(0), 3.0);
        assert_eq!(flash.k_value(1), 0.5);

        // Test state initialization
        let flash = FlashSeparator::new(10.0, 2).with_initial_state(350.0, 101325.0);
        assert_eq!(flash.temperature.get(), 350.0);
        assert_eq!(flash.pressure.get(), 101325.0);

        // Test full initialization
        let flash = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);
        assert_eq!(flash.k_value(0), 3.0);
        assert_eq!(flash.temperature.get(), 350.0);
    }

    #[test]
    fn test_flash_calculation() {
        let mut flash = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);

        flash.set_feed_composition(vec![0.5, 0.5]);
        flash.flash_calculation();

        // Light component should be enriched in vapor
        assert!(flash.vapor_composition[0].get() > flash.liquid_composition[0].get());
        // Heavy component should be enriched in liquid
        assert!(flash.liquid_composition[1].get() > flash.vapor_composition[1].get());
    }

    #[cfg(feature = "thermodynamics")]
    #[test]
    fn test_thermodynamic_initialization() {
        use crate::thermodynamics::fluids::Pure;

        let result = FlashSeparator::new(10.0, 1).with_k_values_from_thermo(
            &[Pure::Water],
            298.15,
            101325.0,
        );

        assert!(result.is_ok());
        let flash = result.unwrap().with_initial_state(298.15, 101325.0);

        let k_water = flash.k_value(0);
        // Water at 25°C, 1 atm: K ≈ 0.031
        assert!(k_water > 0.025 && k_water < 0.04, "K_water = {}", k_water);
    }

    #[test]
    fn test_flash_outlet_streams() {
        let mut flash = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 200000.0);

        // Set inlet flow and feed composition
        flash.inlet_flow = Var::new(100.0);
        flash.set_feed_composition(vec![0.5, 0.5]);
        flash.flash_calculation(); // Compute vapor fraction and compositions

        let vapor_ref = flash.vapor_outlet_stream();
        flash.populate_vapor_outlet(&vapor_ref).unwrap();
        let vapor = vapor_ref.get().unwrap();

        let liquid_ref = flash.liquid_outlet_stream();
        flash.populate_liquid_outlet(&liquid_ref).unwrap();
        let liquid = liquid_ref.get().unwrap();

        // Temperature and pressure match flash conditions
        assert_eq!(vapor.temperature, 350.0);
        assert_eq!(vapor.pressure, 200000.0);
        assert_eq!(liquid.temperature, 350.0);
        assert_eq!(liquid.pressure, 200000.0);

        // Flow rates computed from inlet and calculated vapor fraction
        let vf = flash.vapor_fraction.get();
        assert!((vapor.total_flow - 100.0 * vf).abs() < 1e-10);
        assert!((liquid.total_flow - 100.0 * (1.0 - vf)).abs() < 1e-10);

        // Total outlet flow should equal inlet
        assert!((vapor.total_flow + liquid.total_flow - 100.0).abs() < 1e-10);

        // Verify multicomponent streams have correct number of components
        assert_eq!(vapor.n_components(), 2);
        assert_eq!(liquid.n_components(), 2);

        // Light component (K=3.0) enriched in vapor
        assert!(vapor.get_composition(0) > liquid.get_composition(0));
        // Heavy component (K=0.5) enriched in liquid
        assert!(liquid.get_composition(1) > vapor.get_composition(1));

        // Compositions should sum to 1.0
        let vapor_sum: f64 = (0..2).map(|i| vapor.get_composition(i)).sum();
        let liquid_sum: f64 = (0..2).map(|i| liquid.get_composition(i)).sum();
        assert!((vapor_sum - 1.0).abs() < 1e-6);
        assert!((liquid_sum - 1.0).abs() < 1e-6);
    }
}
