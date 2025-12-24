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
use std::marker::PhantomData;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

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

    // Cold side (Some when C = Initialized)
    pub cold_inlet_temp: Var<Algebraic>,
    pub cold_outlet_temp: Var<Differential>,
    cold_flow_rate: Option<Var<Parameter>>,
    cold_heat_capacity: Option<Var<Parameter>>,
    cold_density: Option<Var<Parameter>>,

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

            cold_inlet_temp: Var::new(298.15),
            cold_outlet_temp: Var::new(320.0),
            cold_flow_rate: None,
            cold_heat_capacity: None,
            cold_density: None,

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

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: self.cold_flow_rate,
            cold_heat_capacity: self.cold_heat_capacity,
            cold_density: self.cold_density,

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

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: self.cold_flow_rate,
            cold_heat_capacity: self.cold_heat_capacity,
            cold_density: self.cold_density,

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

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: Some(Var::new(flow)),
            cold_heat_capacity: Some(Var::new(cp)),
            cold_density: Some(Var::new(density)),

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

            cold_inlet_temp: self.cold_inlet_temp,
            cold_outlet_temp: self.cold_outlet_temp,
            cold_flow_rate: Some(Var::new(flow)),
            cold_heat_capacity: Some(Var::new(props.cp)),
            cold_density: Some(Var::new(props.density)),

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

/// UnitOp implementation for HeatExchanger.
impl<H, C> UnitOp for HeatExchanger<H, C> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Hot side energy balance: Q = m_hot * Cp_hot * (T_in_hot - T_out_hot)
        let mut hot_balance = ResidualFunction::new(&format!("{}_hot_energy", unit_name));
        hot_balance.add_term(EquationTerm::new(1.0, "Q"));
        hot_balance.add_term(EquationTerm::new(-1.0, "m_hot_Cp_hot_dT_hot"));
        system.add_algebraic(hot_balance);

        // Cold side energy balance: Q = m_cold * Cp_cold * (T_out_cold - T_in_cold)
        let mut cold_balance = ResidualFunction::new(&format!("{}_cold_energy", unit_name));
        cold_balance.add_term(EquationTerm::new(1.0, "Q"));
        cold_balance.add_term(EquationTerm::new(-1.0, "m_cold_Cp_cold_dT_cold"));
        system.add_algebraic(cold_balance);

        // Heat transfer equation: Q = U*A*LMTD
        let mut heat_transfer = ResidualFunction::new(&format!("{}_heat_transfer", unit_name));
        heat_transfer.add_term(EquationTerm::new(1.0, "Q"));
        heat_transfer.add_term(EquationTerm::new(-1.0, "U_A_LMTD"));
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
