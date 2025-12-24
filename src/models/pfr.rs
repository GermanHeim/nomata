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

use std::marker::PhantomData;

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
                let mut mass_balance =
                    ResidualFunction::new(&format!("{}_segment_{}_mass", unit_name, i));
                mass_balance.add_term(EquationTerm::new(1.0, &format!("v_dC_{}_dz", i))); // convection
                mass_balance.add_term(EquationTerm::new(-1.0, &format!("D_d2C_{}_dz2", i))); // diffusion
                mass_balance.add_term(EquationTerm::new(1.0, &format!("r_{}", i))); // reaction
                system.add_algebraic(mass_balance);

                // Energy balance: 0 = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let mut energy_balance =
                    ResidualFunction::new(&format!("{}_segment_{}_energy", unit_name, i));
                energy_balance.add_term(EquationTerm::new(1.0, &format!("v_dT_{}_dz", i)));
                energy_balance.add_term(EquationTerm::new(-1.0, &format!("alpha_d2T_{}_dz2", i)));
                energy_balance.add_term(EquationTerm::new(-1.0, &format!("r_{}_deltaH_rhoCp", i)));
                system.add_algebraic(energy_balance);
            } else {
                // Dynamic: dC/dt = -v*dC/dz + D*d^2C/dz^2 - r_i
                let mut mass_balance =
                    ResidualFunction::new(&format!("{}_segment_{}_mass", unit_name, i));
                mass_balance.add_term(EquationTerm::new(1.0, &format!("dC_{}_dt", i)));
                mass_balance.add_term(EquationTerm::new(1.0, &format!("v_dC_{}_dz", i))); // convection
                mass_balance.add_term(EquationTerm::new(-1.0, &format!("D_d2C_{}_dz2", i))); // diffusion
                mass_balance.add_term(EquationTerm::new(1.0, &format!("r_{}", i))); // reaction
                system.add_differential(mass_balance);

                // Energy balance: dT_i/dt = -v*dT/dz + alpha*d^2T/dz^2 + r_i*deltaH/(rho*Cp)
                let mut energy_balance =
                    ResidualFunction::new(&format!("{}_segment_{}_energy", unit_name, i));
                energy_balance.add_term(EquationTerm::new(1.0, &format!("dT_{}_dt", i)));
                energy_balance.add_term(EquationTerm::new(1.0, &format!("v_dT_{}_dz", i)));
                energy_balance.add_term(EquationTerm::new(-1.0, &format!("alpha_d2T_{}_dz2", i)));
                energy_balance.add_term(EquationTerm::new(-1.0, &format!("r_{}_deltaH_rhoCp", i)));
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

        assert_eq!(pfr.length.as_ref().unwrap().get(), 10.0);
        assert_eq!(pfr.area.as_ref().unwrap().get(), 0.1);
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
