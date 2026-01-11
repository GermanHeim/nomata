//! Heat Exchanger model.
//!
//! Counter-current shell-and-tube heat exchanger.
//!
//! # State Variables
//! - Hot side temperature profile
//! - Cold side temperature profile
//!
//! # Parameters
//! - Heat transfer coefficient (U)
//! - Heat transfer area (A)
//! - Hot/cold flow rates
//! - Heat capacities
//!
//! # Example
//!
//! ```
//! use nomata::models::HeatExchanger;
//!
//! // Type-safe initialization
//! let hex = HeatExchanger::new(10.0, 500.0)  // area, U
//!     .with_hot_side(1.0, 1000.0, 4184.0)   // flow, density, Cp
//!     .with_cold_side(1.0, 1000.0, 4184.0); // flow, density, Cp
//!
//! // Only fully initialized exchangers can compute heat duty
//! // hex.compute_heat_duty();  // Compiles
//! ```

use crate::*;
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

// Typed Equation Variable Structs (Generic for autodiff support)

/// Variables for hot side energy balance: Q - m_hot*Cp_hot*dT_hot = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct HotEnergyBalanceVars<S: Scalar> {
    pub q: S,
    pub m_hot: S,
    pub cp_hot: S,
    pub dt_hot: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for HotEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "m_hot", "Cp_hot", "dT_hot"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            q: *vars.get(&format!("{}_Q", prefix))?,
            m_hot: *vars.get(&format!("{}_m_hot", prefix))?,
            cp_hot: *vars.get(&format!("{}_Cp_hot", prefix))?,
            dt_hot: *vars.get(&format!("{}_dT_hot", prefix))?,
        })
    }
}

impl EquationVars for HotEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "m_hot", "Cp_hot", "dT_hot"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for cold side energy balance: Q - m_cold*Cp_cold*dT_cold = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct ColdEnergyBalanceVars<S: Scalar> {
    pub q: S,
    pub m_cold: S,
    pub cp_cold: S,
    pub dt_cold: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for ColdEnergyBalanceVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "m_cold", "Cp_cold", "dT_cold"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            q: *vars.get(&format!("{}_Q", prefix))?,
            m_cold: *vars.get(&format!("{}_m_cold", prefix))?,
            cp_cold: *vars.get(&format!("{}_Cp_cold", prefix))?,
            dt_cold: *vars.get(&format!("{}_dT_cold", prefix))?,
        })
    }
}

impl EquationVars for ColdEnergyBalanceVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "m_cold", "Cp_cold", "dT_cold"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for heat transfer equation: Q - U*A*LMTD = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct HeatTransferVars<S: Scalar> {
    pub q: S,
    pub u: S,
    pub a: S,
    pub lmtd: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for HeatTransferVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "U", "A", "LMTD"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            q: *vars.get(&format!("{}_Q", prefix))?,
            u: *vars.get(&format!("{}_U", prefix))?,
            a: *vars.get(&format!("{}_A", prefix))?,
            lmtd: *vars.get(&format!("{}_LMTD", prefix))?,
        })
    }
}

impl EquationVars for HeatTransferVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["Q", "U", "A", "LMTD"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Phantom type marker for uninitialized state.
pub struct Uninitialized;

/// Phantom type marker for initialized state.
pub struct Initialized;

/// Counter-current shell-and-tube heat exchanger.
///
/// Type parameters enforce compile-time initialization:
/// - `H`: Hot-side properties state (Uninitialized | Initialized)
/// - `C`: Cold-side properties state (Uninitialized | Initialized)
pub struct HeatExchanger<H = Uninitialized, C = Uninitialized> {
    // Geometry (always required)
    area: Var<Parameter>,                // A (m^2)
    heat_transfer_coeff: Var<Parameter>, // U (W/m^2*K)

    // Hot side (Some when H = Initialized)
    pub hot_inlet_temp: Var<Algebraic>,
    pub hot_outlet_temp: Var<Differential>,
    hot_flow_rate: Option<Var<Parameter>>,
    hot_heat_capacity: Option<Var<Parameter>>,
    hot_density: Option<Var<Parameter>>,
    hot_molecular_weight: Option<f64>,

    // Cold side (Some when C = Initialized)
    pub cold_inlet_temp: Var<Algebraic>,
    pub cold_outlet_temp: Var<Differential>,
    cold_flow_rate: Option<Var<Parameter>>,
    cold_heat_capacity: Option<Var<Parameter>>,
    cold_density: Option<Var<Parameter>>,
    cold_molecular_weight: Option<f64>,

    // Composition tracking
    hot_inlet_composition: Vec<f64>,
    hot_component_names: Vec<String>,
    cold_inlet_composition: Vec<f64>,
    cold_component_names: Vec<String>,

    // Heat duty (computed)
    pub heat_duty: Var<Algebraic>,

    // Ports
    pub hot_inlet: Port<Stream<MolarFlow>, Input, Disconnected>,
    pub hot_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,
    pub cold_inlet: Port<Stream<MolarFlow>, Input, Disconnected>,
    pub cold_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,

    #[cfg(feature = "thermodynamics")]
    pub hot_fluid: Option<Pure>,
    #[cfg(feature = "thermodynamics")]
    pub cold_fluid: Option<Pure>,

    // Phantom data
    _hot_state: PhantomData<H>,
    _cold_state: PhantomData<C>,
}

// Initial constructor
impl HeatExchanger<Uninitialized, Uninitialized> {
    /// Creates a new heat exchanger with geometry specified.
    ///
    /// Hot and cold side properties must be initialized before heat duty calculations.
    pub fn new(area: f64, u_coefficient: f64) -> Self {
        HeatExchanger {
            area: Var::new(area),
            heat_transfer_coeff: Var::new(u_coefficient),

            hot_inlet_temp: Var::new(373.15),
            hot_outlet_temp: Var::new(350.0),
            hot_flow_rate: None,
            hot_heat_capacity: None,
            hot_density: None,
            hot_molecular_weight: None,

            cold_inlet_temp: Var::new(298.15),
            cold_outlet_temp: Var::new(320.0),
            cold_flow_rate: None,
            cold_heat_capacity: None,
            cold_density: None,
            cold_molecular_weight: None,

            hot_inlet_composition: vec![1.0],
            hot_component_names: vec!["Unknown".to_string()],
            cold_inlet_composition: vec![1.0],
            cold_component_names: vec!["Unknown".to_string()],

            heat_duty: Var::new(0.0),

            hot_inlet: Port::new(),
            hot_outlet: Port::new(),
            cold_inlet: Port::new(),
            cold_outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            hot_fluid: None,
            #[cfg(feature = "thermodynamics")]
            cold_fluid: None,

            _hot_state: PhantomData,
            _cold_state: PhantomData,
        }
    }
}

// Hot side initialization (H: Uninitialized -> Initialized)
impl<C> HeatExchanger<Uninitialized, C> {
    /// Sets hot side properties manually, transitioning to hot-side-initialized state.
    pub fn with_hot_side(self, flow: f64, density: f64, cp: f64) -> HeatExchanger<Initialized, C> {
        HeatExchanger {
            area: self.area,
            heat_transfer_coeff: self.heat_transfer_coeff,

            hot_inlet_temp: self.hot_inlet_temp,
            hot_outlet_temp: self.hot_outlet_temp,
            hot_flow_rate: Some(Var::new(flow)),
            hot_heat_capacity: Some(Var::new(cp)),
            hot_density: Some(Var::new(density)),
            hot_molecular_weight: None,

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: self.cold_flow_rate,
            cold_heat_capacity: self.cold_heat_capacity,
            cold_density: self.cold_density,
            cold_molecular_weight: self.cold_molecular_weight,

            hot_inlet_composition: self.hot_inlet_composition,
            hot_component_names: self.hot_component_names,
            cold_inlet_composition: self.cold_inlet_composition,
            cold_component_names: self.cold_component_names,

            heat_duty: self.heat_duty,

            hot_inlet: self.hot_inlet,
            hot_outlet: self.hot_outlet,
            cold_inlet: self.cold_inlet,
            cold_outlet: self.cold_outlet,

            #[cfg(feature = "thermodynamics")]
            hot_fluid: self.hot_fluid,
            #[cfg(feature = "thermodynamics")]
            cold_fluid: self.cold_fluid,

            _hot_state: PhantomData,
            _cold_state: self._cold_state,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets hot side properties from thermodynamics.
    pub fn with_hot_side_from_fluid(
        self,
        pure: Pure,
        flow: f64,
        temp: f64,
        pressure: f64,
    ) -> Result<HeatExchanger<Initialized, C>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;

        Ok(HeatExchanger {
            area: self.area,
            heat_transfer_coeff: self.heat_transfer_coeff,

            hot_inlet_temp: self.hot_inlet_temp,
            hot_outlet_temp: self.hot_outlet_temp,
            hot_flow_rate: Some(Var::new(flow)),
            hot_heat_capacity: Some(Var::new(props.cp)),
            hot_density: Some(Var::new(props.density)),
            hot_molecular_weight: fluid_obj.molecular_weight().ok(),

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: self.cold_flow_rate,
            cold_heat_capacity: self.cold_heat_capacity,
            cold_density: self.cold_density,
            cold_molecular_weight: self.cold_molecular_weight,

            hot_inlet_composition: self.hot_inlet_composition,
            hot_component_names: self.hot_component_names,
            cold_inlet_composition: self.cold_inlet_composition,
            cold_component_names: self.cold_component_names,

            heat_duty: self.heat_duty,

            hot_inlet: self.hot_inlet,
            hot_outlet: self.hot_outlet,
            cold_inlet: self.cold_inlet,
            cold_outlet: self.cold_outlet,

            hot_fluid: Some(pure),
            cold_fluid: self.cold_fluid,

            _hot_state: PhantomData,
            _cold_state: self._cold_state,
        })
    }
}

// Cold side initialization (C: Uninitialized -> Initialized)
impl<H> HeatExchanger<H, Uninitialized> {
    /// Sets cold side properties manually, transitioning to cold-side-initialized state.
    pub fn with_cold_side(self, flow: f64, density: f64, cp: f64) -> HeatExchanger<H, Initialized> {
        HeatExchanger {
            area: self.area,
            heat_transfer_coeff: self.heat_transfer_coeff,

            hot_inlet_temp: self.hot_inlet_temp,
            hot_outlet_temp: self.hot_outlet_temp,
            hot_flow_rate: self.hot_flow_rate,
            hot_heat_capacity: self.hot_heat_capacity,
            hot_density: self.hot_density,
            hot_molecular_weight: self.hot_molecular_weight,

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: Some(Var::new(flow)),
            cold_heat_capacity: Some(Var::new(cp)),
            cold_density: Some(Var::new(density)),
            cold_molecular_weight: None,

            hot_inlet_composition: self.hot_inlet_composition,
            hot_component_names: self.hot_component_names,
            cold_inlet_composition: self.cold_inlet_composition,
            cold_component_names: self.cold_component_names,

            heat_duty: self.heat_duty,

            hot_inlet: self.hot_inlet,
            hot_outlet: self.hot_outlet,
            cold_inlet: self.cold_inlet,
            cold_outlet: self.cold_outlet,

            #[cfg(feature = "thermodynamics")]
            hot_fluid: self.hot_fluid,
            #[cfg(feature = "thermodynamics")]
            cold_fluid: self.cold_fluid,

            _hot_state: self._hot_state,
            _cold_state: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets cold side properties from thermodynamics.
    pub fn with_cold_side_from_fluid(
        self,
        pure: Pure,
        flow: f64,
        temp: f64,
        pressure: f64,
    ) -> Result<HeatExchanger<H, Initialized>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;

        Ok(HeatExchanger {
            area: self.area,
            heat_transfer_coeff: self.heat_transfer_coeff,

            hot_inlet_temp: self.hot_inlet_temp,
            hot_outlet_temp: self.hot_outlet_temp,
            hot_flow_rate: self.hot_flow_rate,
            hot_heat_capacity: self.hot_heat_capacity,
            hot_density: self.hot_density,
            hot_molecular_weight: self.hot_molecular_weight,

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: Some(Var::new(flow)),
            cold_heat_capacity: Some(Var::new(props.cp)),
            cold_density: Some(Var::new(props.density)),
            cold_molecular_weight: fluid_obj.molecular_weight().ok(),

            hot_inlet_composition: self.hot_inlet_composition,
            hot_component_names: self.hot_component_names,
            cold_inlet_composition: self.cold_inlet_composition,
            cold_component_names: self.cold_component_names,

            heat_duty: self.heat_duty,

            hot_inlet: self.hot_inlet,
            hot_outlet: self.hot_outlet,
            cold_inlet: self.cold_inlet,
            cold_outlet: self.cold_outlet,

            hot_fluid: self.hot_fluid,
            cold_fluid: Some(pure),

            _hot_state: self._hot_state,
            _cold_state: PhantomData,
        })
    }
}

// Methods available at any state
impl<H, C> HeatExchanger<H, C> {
    /// Gets geometry parameters.
    pub fn area(&self) -> f64 {
        self.area.get()
    }

    pub fn heat_transfer_coeff(&self) -> f64 {
        self.heat_transfer_coeff.get()
    }

    /// Creates energy balance for hot side.
    pub fn hot_side_energy_balance(&self) -> Equation<EnergyBalance> {
        self.hot_outlet_temp.energy_balance("hex_hot_side")
    }

    /// Creates energy balance for cold side.
    pub fn cold_side_energy_balance(&self) -> Equation<EnergyBalance> {
        self.cold_outlet_temp.energy_balance("hex_cold_side")
    }
}

// Fully initialized methods
impl HeatExchanger<Initialized, Initialized> {
    /// Computes outlet temperatures using effectiveness-NTU method.
    ///
    /// Solves the counter-current heat exchanger equations to find outlet temperatures
    /// given inlet temperatures and all other properties.
    pub fn solve_outlet_temperatures(&mut self) {
        // Heat capacity rates (W/K)
        // For molar flow: C = n_dot * Cp_molar = n_dot * MW * Cp_mass
        let c_hot = if let Some(mw) = self.hot_molecular_weight {
            self.hot_flow_rate() * mw * self.hot_heat_capacity()
        } else {
            // Fallback: assume mass flow rate
            self.hot_flow_rate() * self.hot_heat_capacity()
        };

        let c_cold = if let Some(mw) = self.cold_molecular_weight {
            self.cold_flow_rate() * mw * self.cold_heat_capacity()
        } else {
            // Fallback: assume mass flow rate
            self.cold_flow_rate() * self.cold_heat_capacity()
        };

        let c_min = c_hot.min(c_cold);
        let c_max = c_hot.max(c_cold);
        let c_ratio = c_min / c_max;

        // Number of Transfer Units
        let ntu = self.heat_transfer_coeff.get() * self.area.get() / c_min;

        // Effectiveness for counter-current heat exchanger
        let effectiveness = if (c_ratio - 1.0).abs() < 1e-10 {
            // Special case: C_hot = C_cold
            ntu / (1.0 + ntu)
        } else {
            let exp_term = (-ntu * (1.0 - c_ratio)).exp();
            (1.0 - exp_term) / (1.0 - c_ratio * exp_term)
        };

        // Maximum possible heat transfer
        let q_max = c_min * (self.hot_inlet_temp.get() - self.cold_inlet_temp.get());

        // Actual heat transfer
        let q = effectiveness * q_max;
        self.heat_duty = Var::new(q);

        // Compute outlet temperatures from energy balance
        let t_hot_out = self.hot_inlet_temp.get() - q / c_hot;
        let t_cold_out = self.cold_inlet_temp.get() + q / c_cold;

        self.hot_outlet_temp = Var::new(t_hot_out);
        self.cold_outlet_temp = Var::new(t_cold_out);
    }

    /// Computes heat duty using LMTD method.
    ///
    /// Only available for fully initialized heat exchangers.
    pub fn compute_heat_duty(&mut self) {
        let dt1 = self.hot_inlet_temp.get() - self.cold_outlet_temp.get();
        let dt2 = self.hot_outlet_temp.get() - self.cold_inlet_temp.get();

        let lmtd = if (dt1 - dt2).abs() < 1e-6 { dt1 } else { (dt1 - dt2) / (dt1 / dt2).ln() };

        let q = self.heat_transfer_coeff.get() * self.area.get() * lmtd;
        self.heat_duty = Var::new(q);
    }

    /// Gets hot side properties (guaranteed to exist).
    pub fn hot_flow_rate(&self) -> f64 {
        self.hot_flow_rate.as_ref().unwrap().get()
    }

    pub fn hot_density(&self) -> f64 {
        self.hot_density.as_ref().unwrap().get()
    }

    pub fn hot_heat_capacity(&self) -> f64 {
        self.hot_heat_capacity.as_ref().unwrap().get()
    }

    /// Gets cold side properties (guaranteed to exist).
    pub fn cold_flow_rate(&self) -> f64 {
        self.cold_flow_rate.as_ref().unwrap().get()
    }

    pub fn cold_density(&self) -> f64 {
        self.cold_density.as_ref().unwrap().get()
    }

    pub fn cold_heat_capacity(&self) -> f64 {
        self.cold_heat_capacity.as_ref().unwrap().get()
    }

    /// Creates the hot side outlet stream.
    ///
    /// Returns a stream with the hot outlet temperature and flow rate.
    /// For multicomponent streams, composition is preserved from inlet.
    fn compute_hot_outlet_stream(&self) -> Result<Stream<MolarFlow>, crate::StreamError> {
        let flow = self.hot_flow_rate.as_ref().unwrap().get();
        let temp = self.hot_outlet_temp.get();

        // If we have multicomponent composition info, use it
        if !self.hot_component_names.is_empty()
            && self.hot_component_names[0] != "Unknown"
            && let Ok(stream) = Stream::with_composition(
                flow,
                self.hot_component_names.clone(),
                self.hot_inlet_composition.clone(),
            )
        {
            return Ok(stream.at_conditions(temp, 101325.0));
        }

        Err(crate::StreamError::MissingComposition {
            model: "HeatExchanger (hot side)".to_string(),
            suggestion: "set_hot_composition()".to_string(),
        })
    }

    /// Creates the cold side outlet stream.
    ///
    /// Returns a stream with the cold outlet temperature and flow rate.
    /// For multicomponent streams, composition is preserved from inlet.
    fn compute_cold_outlet_stream(&self) -> Result<Stream<MolarFlow>, crate::StreamError> {
        let flow = self.cold_flow_rate.as_ref().unwrap().get();
        let temp = self.cold_outlet_temp.get();

        // If we have multicomponent composition info, use it
        if !self.cold_component_names.is_empty()
            && self.cold_component_names[0] != "Unknown"
            && let Ok(stream) = Stream::with_composition(
                flow,
                self.cold_component_names.clone(),
                self.cold_inlet_composition.clone(),
            )
        {
            return Ok(stream.at_conditions(temp, 101325.0));
        }

        Err(crate::StreamError::MissingComposition {
            model: "HeatExchanger (cold side)".to_string(),
            suggestion: "set_cold_composition()".to_string(),
        })
    }

    /// Returns a reference to the hot outlet stream.
    pub fn hot_outlet_stream(&self) -> crate::OutletRef {
        crate::OutletRef::new("HeatExchanger", "hot_outlet")
    }

    /// Returns a reference to the cold outlet stream.
    pub fn cold_outlet_stream(&self) -> crate::OutletRef {
        crate::OutletRef::new("HeatExchanger", "cold_outlet")
    }

    /// Populates the hot outlet reference with the current computed stream.
    pub fn populate_hot_outlet(
        &self,
        outlet_ref: &crate::OutletRef,
    ) -> Result<(), crate::StreamError> {
        let stream = self.compute_hot_outlet_stream()?;
        outlet_ref.set(stream);
        Ok(())
    }

    /// Populates the cold outlet reference with the current computed stream.
    pub fn populate_cold_outlet(
        &self,
        outlet_ref: &crate::OutletRef,
    ) -> Result<(), crate::StreamError> {
        let stream = self.compute_cold_outlet_stream()?;
        outlet_ref.set(stream);
        Ok(())
    }
}

/// Port-based interface for HeatExchanger.
impl<H, C> HasPorts for HeatExchanger<H, C> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![
            NamedPort::input("hot_inlet", "MolarFlow"),
            NamedPort::input("cold_inlet", "MolarFlow"),
        ]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![
            NamedPort::output("hot_outlet", "MolarFlow"),
            NamedPort::output("cold_outlet", "MolarFlow"),
        ]
    }
}

/// Compile-time port specification for HeatExchanger.
///
/// Enables type-safe connections with const generic port indices.
impl<H, C> PortSpec for HeatExchanger<H, C> {
    const INPUT_COUNT: usize = 2;
    const OUTPUT_COUNT: usize = 2;
    const STREAM_TYPE: &'static str = "MolarFlow";
}

/// UnitOp implementation for HeatExchanger.
impl<H, C> UnitOp for HeatExchanger<H, C> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    #[cfg(not(feature = "autodiff"))]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Hot side energy balance: Q - m_hot*Cp_hot*dT_hot = 0
        let hot_balance = ResidualFunction::from_typed(
            &format!("{}_hot_energy", unit_name),
            unit_name,
            |v: HotEnergyBalanceVars<f64>| v.q - v.m_hot * v.cp_hot * v.dt_hot,
        );
        system.add_algebraic(hot_balance);

        // Cold side energy balance: Q - m_cold*Cp_cold*dT_cold = 0
        let cold_balance = ResidualFunction::from_typed(
            &format!("{}_cold_energy", unit_name),
            unit_name,
            |v: ColdEnergyBalanceVars<f64>| v.q - v.m_cold * v.cp_cold * v.dt_cold,
        );
        system.add_algebraic(cold_balance);

        // Heat transfer equation: Q - U*A*LMTD = 0
        let heat_transfer = ResidualFunction::from_typed(
            &format!("{}_heat_transfer", unit_name),
            unit_name,
            |v: HeatTransferVars<f64>| v.q - v.u * v.a * v.lmtd,
        );
        system.add_algebraic(heat_transfer);
    }

    #[cfg(feature = "autodiff")]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        // Hot side energy balance: Q - m_hot*Cp_hot*dT_hot = 0
        let hot_balance = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_hot_energy", unit_name),
            unit_name,
            |v: HotEnergyBalanceVars<f64>| v.q - v.m_hot * v.cp_hot * v.dt_hot,
            |v: HotEnergyBalanceVars<Dual64>| v.q - v.m_hot * v.cp_hot * v.dt_hot,
        );
        system.add_algebraic(hot_balance);

        // Cold side energy balance: Q - m_cold*Cp_cold*dT_cold = 0
        let cold_balance = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_cold_energy", unit_name),
            unit_name,
            |v: ColdEnergyBalanceVars<f64>| v.q - v.m_cold * v.cp_cold * v.dt_cold,
            |v: ColdEnergyBalanceVars<Dual64>| v.q - v.m_cold * v.cp_cold * v.dt_cold,
        );
        system.add_algebraic(cold_balance);

        // Heat transfer equation: Q - U*A*LMTD = 0
        let heat_transfer = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_heat_transfer", unit_name),
            unit_name,
            |v: HeatTransferVars<f64>| v.q - v.u * v.a * v.lmtd,
            |v: HeatTransferVars<Dual64>| v.q - v.u * v.a * v.lmtd,
        );
        system.add_algebraic(heat_transfer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_exchanger_creation() {
        let hex = HeatExchanger::new(50.0, 500.0);
        assert_eq!(hex.area(), 50.0);
        assert_eq!(hex.heat_transfer_coeff(), 500.0);
    }

    #[test]
    fn test_heat_exchanger_phantom_transitions() {
        // Test hot side initialization
        let hex = HeatExchanger::new(50.0, 500.0).with_hot_side(1.0, 1000.0, 4184.0);
        assert_eq!(hex.hot_flow_rate.as_ref().unwrap().get(), 1.0);
        assert_eq!(hex.hot_density.as_ref().unwrap().get(), 1000.0);

        // Test cold side initialization
        let hex = HeatExchanger::new(50.0, 500.0).with_cold_side(1.0, 1000.0, 4184.0);
        assert_eq!(hex.cold_flow_rate.as_ref().unwrap().get(), 1.0);
        assert_eq!(hex.cold_density.as_ref().unwrap().get(), 1000.0);

        // Test full initialization
        let hex = HeatExchanger::new(50.0, 500.0)
            .with_hot_side(1.0, 1000.0, 4184.0)
            .with_cold_side(1.0, 1000.0, 4184.0);
        assert_eq!(hex.hot_flow_rate(), 1.0);
        assert_eq!(hex.cold_flow_rate(), 1.0);
    }

    #[test]
    fn test_heat_duty_calculation() {
        let mut hex = HeatExchanger::new(50.0, 500.0)
            .with_hot_side(1.0, 1000.0, 4184.0)
            .with_cold_side(1.0, 1000.0, 4184.0);

        hex.hot_inlet_temp = Var::new(373.15);
        hex.hot_outlet_temp = Var::new(350.0);
        hex.cold_inlet_temp = Var::new(298.15);
        hex.cold_outlet_temp = Var::new(320.0);

        hex.compute_heat_duty();
        assert!(hex.heat_duty.get() > 0.0);
    }
}
