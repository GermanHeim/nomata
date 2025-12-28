//! Pump model for liquid pumping.

use crate::*;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

use std::collections::HashMap;
use std::marker::PhantomData;

// Typed Equation Variable Structs (Generic for autodiff support)

/// Variables for head calculation: H - dP/(rho*g) = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct PumpHeadVars<S: Scalar> {
    pub h: S,
    pub dp_rho_g: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PumpHeadVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["H", "dP_rho_g"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            h: *vars.get(&format!("{}_H", prefix))?,
            dp_rho_g: *vars.get(&format!("{}_dP_rho_g", prefix))?,
        })
    }
}

// Backwards compatibility: EquationVars for f64
impl EquationVars for PumpHeadVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["H", "dP_rho_g"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for power calculation: W - rho*g*H*Q/eta = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct PumpPowerVars<S: Scalar> {
    pub w: S,
    pub rho_g_h_q_eta: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PumpPowerVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["W", "rho_g_H_Q_eta"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            w: *vars.get(&format!("{}_W", prefix))?,
            rho_g_h_q_eta: *vars.get(&format!("{}_rho_g_H_Q_eta", prefix))?,
        })
    }
}

// Backwards compatibility: EquationVars for f64
impl EquationVars for PumpPowerVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["W", "rho_g_H_Q_eta"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Marker types for Pump initialization states.
pub struct Uninitialized;
pub struct Initialized;

/// Centrifugal pump model.
///
/// Type parameters enforce compile-time initialization:
/// - `C`: Pump curve/configuration state (pressures, flow, efficiency, density)
/// - `P`: Port connection state
pub struct Pump<C = Uninitialized, P = Disconnected>
where
    P: PortState,
{
    // State variables
    pub outlet_pressure: Var<Differential>,
    pub power: Var<Algebraic>,

    // Parameters
    pub inlet_pressure: Option<Var<Parameter>>,
    pub volumetric_flow: Option<Var<Parameter>>,
    pub efficiency: Option<Var<Parameter>>,
    pub density: Option<Var<Parameter>>,

    // Computed head (m)
    pub head: Var<Algebraic>,

    // Ports
    pub inlet: Port<Stream<MassFlow>, Input, P>,
    pub outlet: Port<Stream<MassFlow>, Output, P>,

    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<Pure>,

    _c: PhantomData<C>,
}

impl Pump {
    /// Creates a new pump in uninitialized state.
    pub fn new() -> Self {
        Pump {
            outlet_pressure: Var::new(0.0),
            power: Var::new(0.0),

            inlet_pressure: None,
            volumetric_flow: None,
            efficiency: None,
            density: None,

            head: Var::new(0.0),

            inlet: Port::new(),
            outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _c: PhantomData,
        }
    }
}

// Pump configuration initialization
impl<P: PortState> Pump<Uninitialized, P> {
    /// Sets pump configuration parameters. Transitions C from Uninitialized to Initialized.
    pub fn with_configuration(
        self,
        inlet_pressure: f64,
        volumetric_flow: f64,
        efficiency: f64,
        density: f64,
    ) -> Pump<Initialized, P> {
        Pump {
            outlet_pressure: self.outlet_pressure,
            power: self.power,

            inlet_pressure: Some(Var::new(inlet_pressure)),
            volumetric_flow: Some(Var::new(volumetric_flow)),
            efficiency: Some(Var::new(efficiency)),
            density: Some(Var::new(density)),

            head: self.head,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _c: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets pump configuration with density from thermodynamic properties.
    pub fn with_configuration_from_fluid(
        self,
        inlet_pressure: f64,
        volumetric_flow: f64,
        efficiency: f64,
        temperature: f64,
        pure: Pure,
    ) -> Result<Pump<Initialized, P>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(inlet_pressure, temperature)?;

        Ok(Pump {
            outlet_pressure: self.outlet_pressure,
            power: self.power,

            inlet_pressure: Some(Var::new(inlet_pressure)),
            volumetric_flow: Some(Var::new(volumetric_flow)),
            efficiency: Some(Var::new(efficiency)),
            density: Some(Var::new(props.density)),

            head: self.head,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: Some(pure),

            _c: PhantomData,
        })
    }
}

// Operations only available when fully initialized
impl<P: PortState> Pump<Initialized, P> {
    /// Computes pump head and power. Only available for fully initialized pump.
    pub fn compute_pumping(&mut self) {
        let p1 = self.inlet_pressure.as_ref().unwrap().get();
        let p2 = self.outlet_pressure.get();
        let rho = self.density.as_ref().unwrap().get();
        let q = self.volumetric_flow.as_ref().unwrap().get();
        let eta = self.efficiency.as_ref().unwrap().get();

        // Gravitational acceleration (m/s^2)
        const G: f64 = 9.80665;

        // Head (m)
        let h = (p2 - p1) / (rho * G);
        self.head = Var::new(h);

        // Power (W)
        let w = rho * G * h * q / eta;
        self.power = Var::new(w);
    }

    #[cfg(feature = "thermodynamics")]
    /// Updates density from thermodynamic properties. Only available for fully initialized pump.
    pub fn update_density(
        &mut self,
        temp: f64,
        pure: Pure,
    ) -> Result<(), crate::thermodynamics::ThermoError> {
        let pressure = self.inlet_pressure.as_ref().unwrap().get();
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;
        self.density = Some(Var::new(props.density));
        Ok(())
    }
}

impl Default for Pump {
    fn default() -> Self {
        Self::new()
    }
}

/// UnitOp implementation for Pump.
impl<C, P: PortState> UnitOp for Pump<C, P> {
    type In = Stream<MassFlow>;
    type Out = Stream<MassFlow>;

    #[cfg(not(feature = "autodiff"))]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Head calculation: H - dP/(rho*g) = 0
        let head_eq = ResidualFunction::from_typed(
            &format!("{}_head", unit_name),
            unit_name,
            |v: PumpHeadVars<f64>| v.h - v.dp_rho_g,
        );
        system.add_algebraic(head_eq);

        // Power calculation: W - rho*g*H*Q/eta = 0
        let power_eq = ResidualFunction::from_typed(
            &format!("{}_power", unit_name),
            unit_name,
            |v: PumpPowerVars<f64>| v.w - v.rho_g_h_q_eta,
        );
        system.add_algebraic(power_eq);
    }

    #[cfg(feature = "autodiff")]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        // Head calculation: H - dP/(rho*g) = 0
        let head_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_head", unit_name),
            unit_name,
            |v: PumpHeadVars<f64>| v.h - v.dp_rho_g,
            |v: PumpHeadVars<Dual64>| v.h - v.dp_rho_g,
        );
        system.add_algebraic(head_eq);

        // Power calculation: W - rho*g*H*Q/eta = 0
        let power_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_power", unit_name),
            unit_name,
            |v: PumpPowerVars<f64>| v.w - v.rho_g_h_q_eta,
            |v: PumpPowerVars<Dual64>| v.w - v.rho_g_h_q_eta,
        );
        system.add_algebraic(power_eq);
    }
}

impl<C, P: PortState> HasPorts for Pump<C, P> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MassFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MassFlow")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pump_creation() {
        let pump: Pump<Initialized> = Pump::new().with_configuration(101325.0, 0.001, 0.75, 1000.0);

        assert_eq!(pump.inlet_pressure.as_ref().unwrap().get(), 101325.0);
        assert_eq!(pump.volumetric_flow.as_ref().unwrap().get(), 0.001);
        assert_eq!(pump.efficiency.as_ref().unwrap().get(), 0.75);
        assert_eq!(pump.density.as_ref().unwrap().get(), 1000.0);
    }

    #[test]
    fn test_pump_computation() {
        let mut pump: Pump<Initialized> =
            Pump::new().with_configuration(101325.0, 0.001, 0.75, 1000.0);

        pump.outlet_pressure = Var::new(202650.0);
        pump.compute_pumping();

        // Head should be (202650 - 101325) / (1000 * 9.80665) aprox 10.33 m
        assert!((pump.head.get() - 10.33).abs() < 0.01);
    }
}
