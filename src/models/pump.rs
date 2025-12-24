//! Pump model for liquid pumping.

use crate::*;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

use std::marker::PhantomData;

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

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Head calculation: H = (P2 - P1) / (rho * g)
        let mut head_eq = ResidualFunction::new(&format!("{}_head", unit_name));
        head_eq.add_term(EquationTerm::new(1.0, "H"));
        head_eq.add_term(EquationTerm::new(-1.0, "dP_rho_g"));
        system.add_algebraic(head_eq);

        // Power calculation: W = rho * g * H * Q / eta
        let mut power_eq = ResidualFunction::new(&format!("{}_power", unit_name));
        power_eq.add_term(EquationTerm::new(1.0, "W"));
        power_eq.add_term(EquationTerm::new(-1.0, "rho_g_H_Q_eta"));
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
