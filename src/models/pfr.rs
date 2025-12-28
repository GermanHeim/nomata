//! Plug Flow Reactor (PFR) model.
//!
//! # State Variables
//! - Concentration profile: C(z,t)
//! - Temperature profile: T(z,t)
//!
//! # Parameters
//! - Length
//! - Cross-sectional area
//! - Kinetic constants
//!
//! # PDEs
//! - partialC/partialt = -v*partialC/partialz + D*partial^2C/partialz^2 - r(C,T)
//! - partialT/partialt = -v*partialT/partialz + alpha*partial^2T/partialz^2 + (-deltaH)*r(C,T)/(rho*Cp)

use crate::*;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::{Fluid, fluids::Pure};

use std::collections::HashMap;
use std::marker::PhantomData;

// Typed Equation Variable Structs (Generic over scalar type for autodiff)

use crate::{EquationVarsGeneric, Scalar};

/// Variables for steady-state segment mass balance: 0 = -v*dC/dz + D*d^2C/dz^2 - r
pub struct PfrSteadyMassVars<S: Scalar> {
    pub v_dc_dz: S,
    pub d_d2c_dz2: S,
    pub r: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PfrSteadyMassVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["v_dc_dz", "d_d2c_dz2", "r"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            v_dc_dz: *vars.get(&format!("{}_v_dc_dz", prefix))?,
            d_d2c_dz2: *vars.get(&format!("{}_d_d2c_dz2", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
        })
    }
}

impl EquationVars for PfrSteadyMassVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for steady-state segment energy balance: 0 = -v*dT/dz + alpha*d^2T/dz^2 + r*deltaH/(rho*Cp)
pub struct PfrSteadyEnergyVars<S: Scalar> {
    pub v_dt_dz: S,
    pub alpha_d2t_dz2: S,
    pub r_deltah_rhocp: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PfrSteadyEnergyVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["v_dt_dz", "alpha_d2t_dz2", "r_deltah_rhocp"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            v_dt_dz: *vars.get(&format!("{}_v_dt_dz", prefix))?,
            alpha_d2t_dz2: *vars.get(&format!("{}_alpha_d2t_dz2", prefix))?,
            r_deltah_rhocp: *vars.get(&format!("{}_r_deltah_rhocp", prefix))?,
        })
    }
}

impl EquationVars for PfrSteadyEnergyVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic segment mass balance: dC/dt = -v*dC/dz + D*d^2C/dz^2 - r
pub struct PfrDynMassVars<S: Scalar> {
    pub dc_dt: S,
    pub v_dc_dz: S,
    pub d_d2c_dz2: S,
    pub r: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PfrDynMassVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dc_dt", "v_dc_dz", "d_d2c_dz2", "r"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dc_dt: *vars.get(&format!("{}_dc_dt", prefix))?,
            v_dc_dz: *vars.get(&format!("{}_v_dc_dz", prefix))?,
            d_d2c_dz2: *vars.get(&format!("{}_d_d2c_dz2", prefix))?,
            r: *vars.get(&format!("{}_r", prefix))?,
        })
    }
}

impl EquationVars for PfrDynMassVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Variables for dynamic segment energy balance: dT/dt = -v*dT/dz + alpha*d^2T/dz^2 + r*deltaH/(rho*Cp)
pub struct PfrDynEnergyVars<S: Scalar> {
    pub dt_dt: S,
    pub v_dt_dz: S,
    pub alpha_d2t_dz2: S,
    pub r_deltah_rhocp: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for PfrDynEnergyVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["dt_dt", "v_dt_dz", "alpha_d2t_dz2", "r_deltah_rhocp"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            dt_dt: *vars.get(&format!("{}_dt_dt", prefix))?,
            v_dt_dz: *vars.get(&format!("{}_v_dt_dz", prefix))?,
            alpha_d2t_dz2: *vars.get(&format!("{}_alpha_d2t_dz2", prefix))?,
            r_deltah_rhocp: *vars.get(&format!("{}_r_deltah_rhocp", prefix))?,
        })
    }
}

impl EquationVars for PfrDynEnergyVars<f64> {
    fn base_names() -> &'static [&'static str] {
        <Self as EquationVarsGeneric<f64>>::base_names()
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        <Self as EquationVarsGeneric<f64>>::from_map(vars, prefix)
    }
}

/// Marker types for PFR initialization states.
pub struct Uninitialized;
pub struct Initialized;

/// Plug Flow Reactor with axial dispersion.
///
/// Type parameters enforce compile-time initialization:
/// - `K`: Kinetics state (k0, activation energy)
/// - `G`: Geometry state (length, area, discretization)
/// - `Thermo`: Thermodynamics state (heat of reaction, density, heat capacity)
/// - `P`: Port connection state
pub struct PFR<K = Uninitialized, G = Uninitialized, Thermo = Uninitialized, P = Disconnected>
where
    P: PortState,
{
    pub length: Option<Var<Parameter>>, // L (m)
    pub area: Option<Var<Parameter>>,   // A (m^2)
    pub velocity: Var<Parameter>,       // v (m/s)
    pub dispersion: Var<Parameter>,     // D (m^2/s)
    pub thermal_diff: Var<Parameter>,   // alpha (m^2/s)

    // Kinetics
    pub k0: Option<Var<Parameter>>,
    pub activation_energy: Option<Var<Parameter>>,

    // Thermodynamics
    pub heat_of_reaction: Option<Var<Parameter>>,
    pub density: Option<Var<Parameter>>,
    pub heat_capacity: Option<Var<Parameter>>,

    // Discretized state (for spatial discretization)
    pub n_segments: Option<usize>,
    pub concentration_profile: Vec<Var<Differential>>,
    pub temperature_profile: Vec<Var<Differential>>,

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, P>,
    pub outlet: Port<Stream<MolarFlow>, Output, P>,

    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<Pure>,

    _k: PhantomData<K>,
    _g: PhantomData<G>,
    _thermo: PhantomData<Thermo>,
}

impl Default for PFR {
    fn default() -> Self {
        Self::new()
    }
}

impl PFR {
    /// Creates a new PFR in uninitialized state.
    pub fn new() -> Self {
        PFR {
            length: None,
            area: None,
            velocity: Var::new(1.0),
            dispersion: Var::new(0.01),
            thermal_diff: Var::new(0.001),

            k0: None,
            activation_energy: None,

            heat_of_reaction: None,
            density: None,
            heat_capacity: None,

            n_segments: None,
            concentration_profile: Vec::new(),
            temperature_profile: Vec::new(),

            inlet: Port::new(),
            outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _k: PhantomData,
            _g: PhantomData,
            _thermo: PhantomData,
        }
    }
}

// Kinetics initialization
impl<G, Thermo, P: PortState> PFR<Uninitialized, G, Thermo, P> {
    /// Sets kinetic parameters.
    /// Transitions K from Uninitialized to Initialized.
    pub fn with_kinetics(self, k0: f64, ea: f64) -> PFR<Initialized, G, Thermo, P> {
        PFR {
            length: self.length,
            area: self.area,
            velocity: self.velocity,
            dispersion: self.dispersion,
            thermal_diff: self.thermal_diff,

            k0: Some(Var::new(k0)),
            activation_energy: Some(Var::new(ea)),

            heat_of_reaction: self.heat_of_reaction,
            density: self.density,
            heat_capacity: self.heat_capacity,

            n_segments: self.n_segments,
            concentration_profile: self.concentration_profile,
            temperature_profile: self.temperature_profile,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _k: PhantomData,
            _g: PhantomData,
            _thermo: PhantomData,
        }
    }
}

// Geometry initialization
impl<K, Thermo, P: PortState> PFR<K, Uninitialized, Thermo, P> {
    /// Sets geometry and creates spatial discretization.
    /// Transitions G from Uninitialized to Initialized.
    pub fn with_geometry(
        self,
        length: f64,
        area: f64,
        n_segments: usize,
    ) -> PFR<K, Initialized, Thermo, P> {
        let initial_conc = (0..n_segments).map(|_| Var::new(0.0)).collect();
        let initial_temp = (0..n_segments).map(|_| Var::new(298.15)).collect();

        PFR {
            length: Some(Var::new(length)),
            area: Some(Var::new(area)),
            velocity: self.velocity,
            dispersion: self.dispersion,
            thermal_diff: self.thermal_diff,

            k0: self.k0,
            activation_energy: self.activation_energy,

            heat_of_reaction: self.heat_of_reaction,
            density: self.density,
            heat_capacity: self.heat_capacity,

            n_segments: Some(n_segments),
            concentration_profile: initial_conc,
            temperature_profile: initial_temp,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _k: PhantomData,
            _g: PhantomData,
            _thermo: PhantomData,
        }
    }
}

// Thermodynamics initialization
impl<K, G, P: PortState> PFR<K, G, Uninitialized, P> {
    /// Sets thermodynamic parameters manually.
    /// Transitions Thermo from Uninitialized to Initialized.
    pub fn with_thermodynamics(self, delta_h: f64, rho: f64, cp: f64) -> PFR<K, G, Initialized, P> {
        PFR {
            length: self.length,
            area: self.area,
            velocity: self.velocity,
            dispersion: self.dispersion,
            thermal_diff: self.thermal_diff,

            k0: self.k0,
            activation_energy: self.activation_energy,

            heat_of_reaction: Some(Var::new(delta_h)),
            density: Some(Var::new(rho)),
            heat_capacity: Some(Var::new(cp)),

            n_segments: self.n_segments,
            concentration_profile: self.concentration_profile,
            temperature_profile: self.temperature_profile,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _k: PhantomData,
            _g: PhantomData,
            _thermo: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets thermodynamic properties from fluid database.
    /// Transitions Thermo from Uninitialized to Initialized.
    pub fn with_thermodynamics_from_fluid(
        self,
        pressure: f64,
        temperature: f64,
        delta_h: f64,
        pure: Pure,
    ) -> Result<PFR<K, G, Initialized, P>, crate::thermodynamics::ThermoError> {
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temperature)?;

        Ok(PFR {
            length: self.length,
            area: self.area,
            velocity: self.velocity,
            dispersion: self.dispersion,
            thermal_diff: self.thermal_diff,

            k0: self.k0,
            activation_energy: self.activation_energy,

            heat_of_reaction: Some(Var::new(delta_h)),
            density: Some(Var::new(props.density)),
            heat_capacity: Some(Var::new(props.cp)),

            n_segments: self.n_segments,
            concentration_profile: self.concentration_profile,
            temperature_profile: self.temperature_profile,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: Some(pure),

            _k: PhantomData,
            _g: PhantomData,
            _thermo: PhantomData,
        })
    }
}

// Operations only available when fully initialized
impl<P: PortState> PFR<Initialized, Initialized, Initialized, P> {
    #[cfg(feature = "thermodynamics")]
    /// Updates thermodynamic properties for a segment.
    /// Only available for fully initialized PFR.
    pub fn update_segment_thermo(
        &mut self,
        segment: usize,
        pressure: f64,
        pure: Pure,
    ) -> Result<(), crate::thermodynamics::ThermoError> {
        let temp = self.temperature_profile[segment].get();
        let fluid_obj = Fluid::new(pure);
        let props = fluid_obj.props_pt(pressure, temp)?;

        self.density = Some(Var::new(props.density));
        self.heat_capacity = Some(Var::new(props.cp));
        Ok(())
    }
}
/// UnitOp implementation for PFR.
#[cfg(not(feature = "autodiff"))]
impl<K, G, Thermo, P: PortState> UnitOp for PFR<K, G, Thermo, P> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        // Only build equations if geometry is initialized
        let Some(n_segs) = self.n_segments else { return };
        let Some(length) = &self.length else { return };

        let _dz = length.get() / n_segs as f64;

        // For each segment, add convection-diffusion-reaction equations
        for i in 0..n_segs {
            if T::IS_STEADY {
                // Steady-state: 0 = spatial derivatives + reaction
                // Mass balance: 0 = -v*dC/dz + D*d^2C/dz^2 - r_i
                let prefix = format!("{}_seg_{}_mass", unit_name, i);
                let mass_balance = ResidualFunction::from_typed(
                    &format!("{}_segment_{}_mass", unit_name, i),
                    &prefix,
                    |v: PfrSteadyMassVars<f64>| v.v_dc_dz - v.d_d2c_dz2 + v.r,
                );
                system.add_algebraic(mass_balance);

                // Energy balance: 0 = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let prefix = format!("{}_seg_{}_energy", unit_name, i);
                let energy_balance = ResidualFunction::from_typed(
                    &format!("{}_segment_{}_energy", unit_name, i),
                    &prefix,
                    |v: PfrSteadyEnergyVars<f64>| v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp,
                );
                system.add_algebraic(energy_balance);
            } else {
                // Dynamic: dC/dt = -v*dC/dz + D*d^2C/dz^2 - r_i
                let prefix = format!("{}_seg_{}_mass", unit_name, i);
                let mass_balance = ResidualFunction::from_typed(
                    &format!("{}_segment_{}_mass", unit_name, i),
                    &prefix,
                    |v: PfrDynMassVars<f64>| v.dc_dt + v.v_dc_dz - v.d_d2c_dz2 + v.r,
                );
                system.add_differential(mass_balance);

                // Energy balance: dT_i/dt = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let prefix = format!("{}_seg_{}_energy", unit_name, i);
                let energy_balance = ResidualFunction::from_typed(
                    &format!("{}_segment_{}_energy", unit_name, i),
                    &prefix,
                    |v: PfrDynEnergyVars<f64>| {
                        v.dt_dt + v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp
                    },
                );
                system.add_differential(energy_balance);
            }
        }
    }
}

/// UnitOp implementation for PFR with autodiff support.
#[cfg(feature = "autodiff")]
impl<K, G, Thermo, P: PortState> UnitOp for PFR<K, G, Thermo, P> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        use num_dual::Dual64;

        // Only build equations if geometry is initialized
        let Some(n_segs) = self.n_segments else { return };
        let Some(length) = &self.length else { return };

        let _dz = length.get() / n_segs as f64;

        // For each segment, add convection-diffusion-reaction equations
        for i in 0..n_segs {
            if T::IS_STEADY {
                // Steady-state: 0 = spatial derivatives + reaction
                // Mass balance: 0 = -v*dC/dz + D*d^2C/dz^2 - r_i
                let prefix = format!("{}_seg_{}_mass", unit_name, i);
                let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                    &format!("{}_segment_{}_mass", unit_name, i),
                    &prefix,
                    |v: PfrSteadyMassVars<f64>| v.v_dc_dz - v.d_d2c_dz2 + v.r,
                    |v: PfrSteadyMassVars<Dual64>| v.v_dc_dz - v.d_d2c_dz2 + v.r,
                );
                system.add_algebraic(mass_balance);

                // Energy balance: 0 = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let prefix = format!("{}_seg_{}_energy", unit_name, i);
                let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                    &format!("{}_segment_{}_energy", unit_name, i),
                    &prefix,
                    |v: PfrSteadyEnergyVars<f64>| v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp,
                    |v: PfrSteadyEnergyVars<Dual64>| v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp,
                );
                system.add_algebraic(energy_balance);
            } else {
                // Dynamic: dC/dt = -v*dC/dz + D*d^2C/dz^2 - r_i
                let prefix = format!("{}_seg_{}_mass", unit_name, i);
                let mass_balance = ResidualFunction::from_typed_generic_with_dual(
                    &format!("{}_segment_{}_mass", unit_name, i),
                    &prefix,
                    |v: PfrDynMassVars<f64>| v.dc_dt + v.v_dc_dz - v.d_d2c_dz2 + v.r,
                    |v: PfrDynMassVars<Dual64>| v.dc_dt + v.v_dc_dz - v.d_d2c_dz2 + v.r,
                );
                system.add_differential(mass_balance);

                // Energy balance: dT_i/dt = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let prefix = format!("{}_seg_{}_energy", unit_name, i);
                let energy_balance = ResidualFunction::from_typed_generic_with_dual(
                    &format!("{}_segment_{}_energy", unit_name, i),
                    &prefix,
                    |v: PfrDynEnergyVars<f64>| {
                        v.dt_dt + v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp
                    },
                    |v: PfrDynEnergyVars<Dual64>| {
                        v.dt_dt + v.v_dt_dz - v.alpha_d2t_dz2 - v.r_deltah_rhocp
                    },
                );
                system.add_differential(energy_balance);
            }
        }
    }
}

impl<K, G, Thermo, P: PortState> HasPorts for PFR<K, G, Thermo, P> {
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
    fn test_pfr_creation() {
        let pfr: PFR<Initialized, Initialized, Initialized> = PFR::new()
            .with_kinetics(1e9, 10000.0)
            .with_geometry(10.0, 0.1, 20)
            .with_thermodynamics(-50000.0, 1000.0, 4184.0);

        assert_eq!(pfr.length.as_ref().expect("length should be set after with_geometry").get(), 10.0);
        assert_eq!(pfr.area.as_ref().expect("area should be set after with_geometry").get(), 0.1);
        assert_eq!(pfr.n_segments.unwrap(), 20);
        assert_eq!(pfr.concentration_profile.len(), 20);
        assert_eq!(pfr.temperature_profile.len(), 20);
    }

    #[test]
    fn test_pfr_kinetics() {
        let pfr: PFR<Initialized> = PFR::new().with_kinetics(1e9, 10000.0);

        assert_eq!(pfr.k0.as_ref().unwrap().get(), 1e9);
        assert_eq!(pfr.activation_energy.as_ref().unwrap().get(), 10000.0);
    }
}
