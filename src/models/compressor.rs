//! Compressor model for gas compression.

use crate::*;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

use std::collections::HashMap;
use std::marker::PhantomData;

// Typed Equation Variable Structs (Generic for autodiff support)

/// Variables for isentropic temperature equation: T2s - T1 * (P2/P1)^((gamma-1)/gamma) = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct IsentropicTempVars<S: Scalar> {
    /// Isentropic outlet temperature
    pub t2s: S,
    /// T1 * (P2/P1)^((gamma-1)/gamma) computed externally
    pub t1_pr_gamma: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for IsentropicTempVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["t2s", "t1_pr_gamma"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            t2s: *vars.get(&format!("{}_t2s", prefix))?,
            t1_pr_gamma: *vars.get(&format!("{}_t1_pr_gamma", prefix))?,
        })
    }
}

impl EquationVars for IsentropicTempVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["t2s", "t1_pr_gamma"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for actual outlet temperature: T2 - T1 - (T2s - T1)/eta = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct OutletTempVars<S: Scalar> {
    /// Actual outlet temperature
    pub t2: S,
    /// Inlet temperature
    pub t1: S,
    /// (T2s - T1) / eta computed externally
    pub dt_isen_eta: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for OutletTempVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["t2", "t1", "dt_isen_eta"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            t2: *vars.get(&format!("{}_t2", prefix))?,
            t1: *vars.get(&format!("{}_t1", prefix))?,
            dt_isen_eta: *vars.get(&format!("{}_dt_isen_eta", prefix))?,
        })
    }
}

impl EquationVars for OutletTempVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["t2", "t1", "dt_isen_eta"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for compressor power: W - m * Cp * (T2 - T1) = 0
///
/// Generic over scalar type `S` to support both f64 and Dual64 for autodiff.
pub struct CompressorPowerVars<S: Scalar> {
    /// Compressor power
    pub w: S,
    /// m * Cp * (T2 - T1) computed externally
    pub m_cp_dt: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for CompressorPowerVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["w", "m_cp_dt"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            w: *vars.get(&format!("{}_w", prefix))?,
            m_cp_dt: *vars.get(&format!("{}_m_cp_dt", prefix))?,
        })
    }
}

impl EquationVars for CompressorPowerVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["w", "m_cp_dt"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Marker types for Compressor initialization states.
pub struct Uninitialized;
pub struct Initialized;

/// Gas compressor with isentropic efficiency.
///
/// Type parameters enforce compile-time initialization:
/// - `C`: Compressor configuration state (pressures, temperature, flow, efficiency, gamma)
/// - `P`: Port connection state
pub struct Compressor<C = Uninitialized, P = Disconnected>
where
    P: PortState,
{
    // State variables
    pub outlet_pressure: Var<Differential>,
    pub outlet_temp: Var<Differential>,
    pub power: Var<Algebraic>,

    // Parameters
    pub inlet_pressure: Option<Var<Parameter>>,
    pub inlet_temp: Option<Var<Parameter>>,
    pub mass_flow: Option<Var<Parameter>>,
    pub isentropic_efficiency: Option<Var<Parameter>>,
    pub gamma: Option<Var<Parameter>>, // Cp/Cv

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, P>,
    pub outlet: Port<Stream<MolarFlow>, Output, P>,

    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<Pure>,

    _c: PhantomData<C>,
}

impl Compressor {
    /// Creates a new compressor in uninitialized state.
    pub fn new() -> Self {
        Compressor {
            outlet_pressure: Var::new(101325.0),
            outlet_temp: Var::new(298.15),
            power: Var::new(0.0),

            inlet_pressure: None,
            inlet_temp: None,
            mass_flow: None,
            isentropic_efficiency: None,
            gamma: None,

            inlet: Port::new(),
            outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _c: PhantomData,
        }
    }
}

// Compressor configuration initialization
impl<P: PortState> Compressor<Uninitialized, P> {
    /// Sets compressor configuration parameters. Transitions C from Uninitialized to Initialized.
    pub fn with_configuration(
        self,
        inlet_pressure: f64,
        inlet_temp: f64,
        mass_flow: f64,
        isentropic_efficiency: f64,
        gamma: f64,
    ) -> Compressor<Initialized, P> {
        Compressor {
            outlet_pressure: self.outlet_pressure,
            outlet_temp: self.outlet_temp,
            power: self.power,

            inlet_pressure: Some(Var::new(inlet_pressure)),
            inlet_temp: Some(Var::new(inlet_temp)),
            mass_flow: Some(Var::new(mass_flow)),
            isentropic_efficiency: Some(Var::new(isentropic_efficiency)),
            gamma: Some(Var::new(gamma)),

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _c: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets compressor configuration with gamma from thermodynamic properties.
    pub fn with_configuration_from_fluid(
        self,
        inlet_pressure: f64,
        inlet_temp: f64,
        mass_flow: f64,
        isentropic_efficiency: f64,
        pure: Pure,
    ) -> Result<Compressor<Initialized, P>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(inlet_pressure, inlet_temp)?;
        let gamma = props.cp / props.cv;

        Ok(Compressor {
            outlet_pressure: self.outlet_pressure,
            outlet_temp: self.outlet_temp,
            power: self.power,

            inlet_pressure: Some(Var::new(inlet_pressure)),
            inlet_temp: Some(Var::new(inlet_temp)),
            mass_flow: Some(Var::new(mass_flow)),
            isentropic_efficiency: Some(Var::new(isentropic_efficiency)),
            gamma: Some(Var::new(gamma)),

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: Some(pure),

            _c: PhantomData,
        })
    }
}

// Operations only available when fully initialized
impl<P: PortState> Compressor<Initialized, P> {
    /// Computes outlet conditions and power requirement. Only available for fully initialized compressor.
    pub fn compute_compression(&mut self) {
        let p1 = self.inlet_pressure.as_ref().unwrap().get();
        let p2 = self.outlet_pressure.get();
        let t1 = self.inlet_temp.as_ref().unwrap().get();
        let eta = self.isentropic_efficiency.as_ref().unwrap().get();
        let gamma = self.gamma.as_ref().unwrap().get();

        // Isentropic outlet temperature
        let t2_isen = t1 * (p2 / p1).powf((gamma - 1.0) / gamma);

        // Actual outlet temperature
        let t2_actual = t1 + (t2_isen - t1) / eta;

        self.outlet_temp = Var::new(t2_actual);

        // Power (assuming ideal gas)
        let cp = gamma * 287.0 / (gamma - 1.0); // For air, R = 287 J/kg*K
        let w = self.mass_flow.as_ref().unwrap().get() * cp * (t2_actual - t1);

        self.power = Var::new(w);
    }

    #[cfg(feature = "thermodynamics")]
    /// Updates gamma using thermodynamic properties. Only available for fully initialized compressor.
    pub fn update_gamma(&mut self, pure: Pure) -> Result<(), crate::thermodynamics::ThermoError> {
        let temp = self.inlet_temp.as_ref().unwrap().get();
        let pressure = self.inlet_pressure.as_ref().unwrap().get();
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;

        let gamma = props.cp / props.cv;

        self.gamma = Some(Var::new(gamma));
        Ok(())
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

/// UnitOp implementation for Compressor.
impl<C, P: PortState> UnitOp for Compressor<C, P> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    #[cfg(not(feature = "autodiff"))]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Isentropic outlet temperature: T2s = T1 * (P2/P1)^((gamma-1)/gamma)
        let isentropic_eq = ResidualFunction::from_typed(
            &format!("{}_isentropic_temp", unit_name),
            unit_name,
            |v: IsentropicTempVars<f64>| v.t2s - v.t1_pr_gamma,
        );
        system.add_algebraic(isentropic_eq);

        // Actual outlet temperature: T2 = T1 + (T2s - T1) / eta
        let actual_temp_eq = ResidualFunction::from_typed(
            &format!("{}_outlet_temp", unit_name),
            unit_name,
            |v: OutletTempVars<f64>| v.t2 - v.t1 - v.dt_isen_eta,
        );
        system.add_algebraic(actual_temp_eq);

        // Power: W = m * Cp * (T2 - T1)
        let power_eq = ResidualFunction::from_typed(
            &format!("{}_power", unit_name),
            unit_name,
            |v: CompressorPowerVars<f64>| v.w - v.m_cp_dt,
        );
        system.add_algebraic(power_eq);
    }

    #[cfg(feature = "autodiff")]
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        // Isentropic outlet temperature: T2s = T1 * (P2/P1)^((gamma-1)/gamma)
        let isentropic_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_isentropic_temp", unit_name),
            unit_name,
            |v: IsentropicTempVars<f64>| v.t2s - v.t1_pr_gamma,
            |v: IsentropicTempVars<Dual64>| v.t2s - v.t1_pr_gamma,
        );
        system.add_algebraic(isentropic_eq);

        // Actual outlet temperature: T2 = T1 + (T2s - T1) / eta
        let actual_temp_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_outlet_temp", unit_name),
            unit_name,
            |v: OutletTempVars<f64>| v.t2 - v.t1 - v.dt_isen_eta,
            |v: OutletTempVars<Dual64>| v.t2 - v.t1 - v.dt_isen_eta,
        );
        system.add_algebraic(actual_temp_eq);

        // Power: W = m * Cp * (T2 - T1)
        let power_eq = ResidualFunction::from_typed_generic_with_dual(
            &format!("{}_power", unit_name),
            unit_name,
            |v: CompressorPowerVars<f64>| v.w - v.m_cp_dt,
            |v: CompressorPowerVars<Dual64>| v.w - v.m_cp_dt,
        );
        system.add_algebraic(power_eq);
    }
}

impl<C, P: PortState> HasPorts for Compressor<C, P> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MolarFlow")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_creation() {
        let comp: Compressor<Initialized> =
            Compressor::new().with_configuration(101325.0, 298.15, 1.0, 0.75, 1.4);

        assert_eq!(comp.isentropic_efficiency.as_ref().unwrap().get(), 0.75);
        assert_eq!(comp.gamma.as_ref().unwrap().get(), 1.4);
    }

    #[test]
    fn test_compression_calculation() {
        let mut comp: Compressor<Initialized> =
            Compressor::new().with_configuration(101325.0, 298.15, 1.0, 0.75, 1.4);

        comp.outlet_pressure = Var::new(303975.0); // 3:1 ratio

        comp.compute_compression();

        assert!(comp.outlet_temp.get() > 298.15);
        assert!(comp.power.get() > 0.0);
    }
}
