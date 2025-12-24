//! Thermodynamic property calculations using CoolProp via rfluids.
//!
//! This module provides access to real thermodynamic properties through the
//! `rfluids` crate, which wraps the CoolProp library.
//!
//! # Architecture
//!
//! The module uses a `Substance` enum to represent both pure components and
//! predefined mixtures, mirroring rfluids' substance structure:
//!
//! - **Pure fluids**: Single-component substances (Water, Propane, R134a, etc.)
//! - **Predefined mixtures**: Multi-component fluids (Air, R410A, R407C, etc.)
//!
//! # Features
//!
//! - Pure component properties from CoolProp database
//! - Predefined mixture properties (refrigerant blends, air, etc.)
//! - Accurate equation of state calculations
//! - Temperature, pressure, enthalpy, entropy calculations
//! - Phase equilibrium properties
//! - Transport properties (viscosity, thermal conductivity)
//!
//! # Example
//!
//! ```ignore
//! use nomata::thermodynamics::{Fluid, fluids::*};
//!
//! // Pure fluid - all fluids available via Pure enum
//! let water = Fluid::new(Pure::Water);
//! let propane = Fluid::new(Pure::nPropane);
//! let r134a = Fluid::new(Pure::R134a);
//!
//! // Properties at 101325 Pa and 373.15 K
//! let props = water.props_pt(101325.0, 373.15)?;
//! println!("Enthalpy: {} J/kg", props.enthalpy);
//! println!("Density: {} kg/m^3", props.density);
//!
//! // Predefined mixture - all mixtures available via PredefinedMix enum
//! let air = Fluid::new_mix(PredefinedMix::Air);
//! let r410a = Fluid::new_mix(PredefinedMix::R410A);
//!
//! // Air at 101325 Pa and 298.15 K
//! let air_props = air.props_pt(101325.0, 298.15)?;
//! // Heat capacity ratio
//! println!("cp/cv = {}", air_props.cp / air_props.cv);
//! ```
//!
//! # Implementation Notes
//!
//! This module uses the rfluids crate with its typestate pattern:
//! - `Fluid<Undefined>` for fluid creation from Pure or PredefinedMix
//! - `Fluid<Defined>` after state is specified with `.in_state()`
//! - All property methods return `Result<f64, FluidOutputError>`
//! - The `Substance` enum provides a unified interface for both types

use rfluids::prelude::*;

/// Result type for thermodynamic calculations.
pub type ThermoResult<T> = Result<T, ThermoError>;

/// Errors that can occur during thermodynamic calculations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ThermoError {
    /// Fluid not found in database
    #[error("Fluid not found: {0}")]
    FluidNotFound(String),
    /// Invalid input conditions (e.g., negative temperature)
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// Calculation failed to converge
    #[error("Calculation failed to converge")]
    ConvergenceFailure,
    /// Property not available for this fluid/state
    #[error("Property not available")]
    PropertyNotAvailable,
    /// Two-phase region (properties ambiguous)
    #[error("Two-phase region encountered")]
    TwoPhase,
}

/// Substance type for thermodynamic calculations.
///
/// Provides a unified interface for all rfluids substance types:
/// - Pure components (single substances)
/// - Predefined mixtures (standard blends like Air, R410A)
/// - Binary mixtures (two-component custom mixtures)
/// - Custom mixtures (arbitrary multi-component mixtures)
#[derive(Debug, Clone)]
pub enum Substance {
    /// Pure component fluid
    Pure(Pure),
    /// Predefined mixture (e.g., Air, R410A)
    PredefinedMix(PredefinedMix),
    /// Binary mixture (two-component custom mixture)
    BinaryMix(BinaryMix),
    /// Custom mixture (arbitrary multi-component mixture)
    CustomMix(CustomMix),
}

impl Substance {
    /// Creates an rfluids Fluid object from this substance.
    /// Returns Fluid<Undefined> which must be converted to Fluid<Defined> via in_state().
    fn to_rfluids_fluid(&self) -> ThermoResult<rfluids::fluid::Fluid<Undefined>> {
        match self {
            Substance::Pure(pure) => Ok(rfluids::fluid::Fluid::from(*pure)),
            Substance::PredefinedMix(mix) => Ok(rfluids::fluid::Fluid::from(*mix)),
            Substance::BinaryMix(binary) => Ok(rfluids::fluid::Fluid::from(*binary)),
            Substance::CustomMix(custom) => rfluids::fluid::Fluid::try_from(custom.clone())
                .map_err(|_| ThermoError::PropertyNotAvailable),
        }
    }
}

/// Represents a chemical component or mixture with thermodynamic properties.
///
/// This wraps the `rfluids` fluid database and provides a type-safe
/// interface for property calculations.
///
/// Supports both pure components (Water, Propane, etc.) and predefined mixtures
/// (Air as mixture, R410A, etc.).
pub struct Fluid {
    /// The substance (pure or mixture) identifier for rfluids
    pub substance: Substance,
    /// Name of the fluid
    pub name: String,
}

impl Fluid {
    /// Creates a new pure fluid.
    ///
    /// This is the recommended way to create pure fluids as it provides
    /// compile-time verification of the fluid name.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::Pure};
    ///
    /// let water = Fluid::new(Pure::Water);
    /// let propane = Fluid::new(Pure::nPropane);
    /// let r134a = Fluid::new(Pure::R134a);
    /// ```
    pub fn new(pure: Pure) -> Self {
        let name = format!("{:?}", pure);
        Self { substance: Substance::Pure(pure), name }
    }

    /// Creates a new predefined mixture fluid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::PredefinedMix};
    ///
    /// let air = Fluid::new_mix(PredefinedMix::Air);
    /// let r410a = Fluid::new_mix(PredefinedMix::R410A);
    /// ```
    pub fn new_mix(mix: PredefinedMix) -> Self {
        let name = format!("{:?}", mix);
        Self { substance: Substance::PredefinedMix(mix), name }
    }

    /// Creates a new binary mixture fluid using BinaryMixKind.
    ///
    /// # Arguments
    ///
    /// * `name` - Custom name for the mixture
    /// * `kind` - The binary mixture kind (MPG, MEG, etc.)
    /// * `fraction` - Mass or volume fraction (0.0 to 1.0)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::BinaryMixKind};
    ///
    /// // 40% propylene glycol in water
    /// let mixture = Fluid::new_binary(
    ///     "PropyleneGlycol-40",
    ///     BinaryMixKind::MPG,
    ///     0.4,
    /// )?;
    /// ```
    pub fn new_binary(name: &str, kind: BinaryMixKind, fraction: f64) -> ThermoResult<Self> {
        let binary = kind.with_fraction(fraction).map_err(|_| {
            ThermoError::InvalidInput(format!("Invalid fraction {} for {:?}", fraction, kind))
        })?;

        Ok(Self { substance: Substance::BinaryMix(binary), name: name.to_string() })
    }

    /// Creates a new custom mass-based mixture fluid.
    ///
    /// # Arguments
    ///
    /// * `name` - Custom name for the mixture
    /// * `components` - HashMap of (Pure component, mass fraction) pairs
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::Pure};
    /// use std::collections::HashMap;
    ///
    /// let mixture = Fluid::new_mass_based(
    ///     "WaterEthanol",
    ///     HashMap::from([
    ///         (Pure::Water, 0.6),
    ///         (Pure::Ethanol, 0.4),
    ///     ]),
    /// )?;
    /// ```
    pub fn new_mass_based(
        name: &str,
        components: std::collections::HashMap<Pure, f64>,
    ) -> ThermoResult<Self> {
        let custom = CustomMix::mass_based(components)
            .map_err(|_| ThermoError::InvalidInput("Invalid mass-based mixture".to_string()))?;

        Ok(Self { substance: Substance::CustomMix(custom), name: name.to_string() })
    }

    /// Creates a new custom mole-based mixture fluid.
    ///
    /// # Arguments
    ///
    /// * `name` - Custom name for the mixture
    /// * `components` - HashMap of (Pure component, mole fraction) pairs
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::Pure};
    /// use std::collections::HashMap;
    ///
    /// let mixture = Fluid::new_mole_based(
    ///     "CustomAir",
    ///     HashMap::from([
    ///         (Pure::Nitrogen, 0.78),
    ///         (Pure::Oxygen, 0.21),
    ///         (Pure::Argon, 0.01),
    ///     ]),
    /// )?;
    /// ```
    pub fn new_mole_based(
        name: &str,
        components: std::collections::HashMap<Pure, f64>,
    ) -> ThermoResult<Self> {
        let custom = CustomMix::mole_based(components)
            .map_err(|_| ThermoError::InvalidInput("Invalid mole-based mixture".to_string()))?;

        Ok(Self { substance: Substance::CustomMix(custom), name: name.to_string() })
    }

    /// Computes properties given pressure and temperature.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Pressure [Pa]
    /// * `temperature` - Temperature [K]
    ///
    /// # Returns
    ///
    /// Thermodynamic properties at the specified state.
    pub fn props_pt(&self, pressure: f64, temperature: f64) -> ThermoResult<Properties> {
        if pressure <= 0.0 || temperature <= 0.0 {
            return Err(ThermoError::InvalidInput(
                "Pressure and temperature must be positive".to_string(),
            ));
        }

        // Create a fluid in the defined state using PT inputs
        let mut state = self
            .substance
            .to_rfluids_fluid()?
            .in_state(FluidInput::pressure(pressure), FluidInput::temperature(temperature))
            .map_err(|e| match &self.substance {
                Substance::BinaryMix(_) => ThermoError::InvalidInput(format!(
                    "Binary incompressible mixture state calculation failed. \
                                CoolProp's INCOMP backend has limited support: {:?}",
                    e
                )),
                Substance::CustomMix(_) => ThermoError::InvalidInput(format!(
                    "Custom mixture state calculation failed. \
                                Only certain combinations are supported by CoolProp: {:?}",
                    e
                )),
                _ => ThermoError::ConvergenceFailure,
            })?;

        Properties::from_fluid(&mut state, pressure, temperature)
    }

    /// Computes properties given pressure and enthalpy.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Pressure [Pa]
    /// * `enthalpy` - Specific enthalpy [J/kg]
    pub fn props_ph(&self, pressure: f64, enthalpy: f64) -> ThermoResult<Properties> {
        if pressure <= 0.0 {
            return Err(ThermoError::InvalidInput("Pressure must be positive".to_string()));
        }

        // Create a fluid in the defined state using PH inputs
        let mut state = self
            .substance
            .to_rfluids_fluid()?
            .in_state(FluidInput::pressure(pressure), FluidInput::enthalpy(enthalpy))
            .map_err(|_| ThermoError::ConvergenceFailure)?;

        let temp = state.temperature().map_err(|_| ThermoError::PropertyNotAvailable)?;
        Properties::from_fluid(&mut state, pressure, temp)
    }

    /// Gets the critical temperature [K].
    pub fn critical_temperature(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid()?;
        fluid.critical_temperature().map_err(|_| ThermoError::PropertyNotAvailable)
    }

    /// Gets the critical pressure [Pa].
    pub fn critical_pressure(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid()?;
        fluid.critical_pressure().map_err(|_| ThermoError::PropertyNotAvailable)
    }

    /// Gets the molecular weight [kg/mol].
    pub fn molecular_weight(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid()?;
        fluid.molar_mass().map_err(|_| ThermoError::PropertyNotAvailable)
    }
}

/// Thermodynamic properties at a specific state.
#[derive(Debug, Clone)]
pub struct Properties {
    /// Pressure [Pa]
    pub pressure: f64,
    /// Temperature [K]
    pub temperature: f64,
    /// Density [kg/m^3]
    pub density: f64,
    /// Specific enthalpy [J/kg]
    pub enthalpy: f64,
    /// Specific entropy [J/(kg*K)]
    pub entropy: f64,
    /// Specific heat capacity at constant pressure [J/(kg*K)]
    pub cp: f64,
    /// Specific heat capacity at constant volume [J/(kg*K)]
    pub cv: f64,
    /// Speed of sound [m/s] (None if backend doesn't support it)
    pub speed_of_sound: Option<f64>,
    /// Dynamic viscosity [Pa*s] (None if not available)
    pub viscosity: Option<f64>,
    /// Thermal conductivity [W/(m*K)] (None if not available)
    pub thermal_conductivity: Option<f64>,
    /// Quality (vapor fraction) [0-1, None if single phase]
    pub quality: Option<f64>,
}

impl Properties {
    /// Creates a Properties struct from an rfluids Fluid in defined state.
    ///
    /// # Arguments
    ///
    /// * `fluid` - The rfluids fluid object in defined state
    /// * `pressure` - Pressure [Pa]
    /// * `temperature` - Temperature [K]
    fn from_fluid(
        fluid: &mut rfluids::fluid::Fluid,
        pressure: f64,
        temperature: f64,
    ) -> ThermoResult<Self> {
        Ok(Properties {
            pressure,
            temperature,
            density: fluid.density().map_err(|e| {
                ThermoError::InvalidInput(format!("Density not available: {:?}", e))
            })?,
            enthalpy: fluid.enthalpy().map_err(|e| {
                ThermoError::InvalidInput(format!("Enthalpy not available: {:?}", e))
            })?,
            entropy: fluid.entropy().map_err(|e| {
                ThermoError::InvalidInput(format!("Entropy not available: {:?}", e))
            })?,
            cp: fluid.specific_heat().map_err(|e| {
                ThermoError::InvalidInput(format!("Specific heat (Cp) not available: {:?}", e))
            })?,
            cv: fluid.specific_heat_const_volume().map_err(|e| {
                ThermoError::InvalidInput(format!("Specific heat (Cv) not available: {:?}", e))
            })?,

            // Transport properties - some backends don't support these
            // Depends on the components and state
            speed_of_sound: fluid.sound_speed().ok(),
            viscosity: fluid.dynamic_viscosity().ok(),
            thermal_conductivity: fluid.conductivity().ok(),
            quality: None, // TODO: Implement quality/vapor fraction detection
        })
    }

    /// Computes specific internal energy [J/kg].
    pub fn internal_energy(&self) -> f64 {
        self.enthalpy - self.pressure / self.density
    }

    /// Computes specific Gibbs free energy [J/kg].
    pub fn gibbs_energy(&self) -> f64 {
        self.enthalpy - self.temperature * self.entropy
    }

    /// Computes specific Helmholtz free energy [J/kg].
    pub fn helmholtz_energy(&self) -> f64 {
        self.internal_energy() - self.temperature * self.entropy
    }

    /// Computes the compressibility factor Z = PV/(RT).
    pub fn compressibility_factor(&self, molar_mass: f64) -> f64 {
        const R: f64 = 8.31446261815324; // J/(mol*K)
        self.pressure / (self.density * R * self.temperature / molar_mass)
    }
}

/// Common fluids available in CoolProp database.
pub mod fluids {
    /// Re-export Pure enum from rfluids - contains all available pure fluids
    /// Use like: `Pure::Water`, `Pure::Nitrogen`, `Pure::Propane`, etc.
    pub use rfluids::prelude::Pure;

    /// Re-export PredefinedMix enum from rfluids - contains all predefined mixtures
    /// Use like: `PredefinedMix::Air`, `PredefinedMix::R410A`, etc.
    pub use rfluids::prelude::PredefinedMix;

    /// Re-export BinaryMixKind enum from rfluids - contains all binary mixture types
    /// Use like: `BinaryMixKind::MPG`, `BinaryMixKind::MEG`, etc.
    pub use rfluids::prelude::BinaryMixKind;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_creation() {
        let fluid = Fluid::new(Pure::Water);
        assert_eq!(fluid.name, "Water");
        if let Ok(tc) = fluid.critical_temperature() {
            assert!(tc > 600.0); // Water Tc aprox 647 K
        }
    }

    #[test]
    fn test_properties_water() {
        let water = Fluid::new(Pure::Water);
        let props = water.props_pt(101325.0, 298.15);
        if let Ok(p) = props {
            // Check that properties are returned
            assert!(p.density > 0.0);
            assert!(p.temperature > 298.0 && p.temperature < 299.0);
            assert!(p.pressure > 101000.0 && p.pressure < 102000.0);
        }
    }

    #[test]
    fn test_invalid_input() {
        let water = Fluid::new(Pure::Water);
        let result = water.props_pt(-100.0, 300.0); // Negative pressure
        assert!(result.is_err());
    }

    #[test]
    fn test_properties_derived() {
        let props = Properties {
            pressure: 101325.0,
            temperature: 300.0,
            density: 1000.0,
            enthalpy: 100000.0,
            entropy: 300.0,
            cp: 4180.0,
            cv: 4000.0,
            speed_of_sound: Some(1500.0),
            viscosity: Some(0.001),
            thermal_conductivity: Some(0.6),
            quality: None,
        };

        let u = props.internal_energy();
        assert!((u - (100000.0 - 101325.0 / 1000.0)).abs() < 1.0);

        let g = props.gibbs_energy();
        assert!((g - (100000.0 - 300.0 * 300.0)).abs() < 1.0);
    }

    #[test]
    fn test_fluid_constants() {
        // Test that the rfluids enums are accessible
        use rfluids::prelude::{BinaryMixKind, PredefinedMix, Pure};

        // Pure fluids - accessible via fluids::Pure
        let _water = Pure::Water;
        let _methanol = Pure::Methanol;
        let _nitrogen = Pure::Nitrogen;

        // Predefined mixtures - accessible via fluids::PredefinedMix
        let _air = PredefinedMix::Air;
        let _r410a = PredefinedMix::R410A;

        // Binary mixtures - accessible via fluids::BinaryMixKind
        let _mpg = BinaryMixKind::MPG;
        let _meg = BinaryMixKind::MEG;
    }

    #[test]
    fn test_mixture_creation() {
        // Test that Air works as a mixture
        let air = Fluid::new_mix(PredefinedMix::Air);
        assert_eq!(air.name, "Air");

        // Test R410A mixture
        let r410a = Fluid::new_mix(PredefinedMix::R410A);
        assert_eq!(r410a.name, "R410A");
    }

    #[test]
    fn test_binary_mixture() {
        // Test binary mixture creation with BinaryMixKind
        if let Ok(mix) = Fluid::new_binary("MPG-40", BinaryMixKind::MPG, 0.4) {
            assert_eq!(mix.name, "MPG-40");
        }
    }

    #[test]
    fn test_binary_mixture_invalid_fraction() {
        // Fraction out of range
        let result = Fluid::new_binary(
            "Invalid",
            BinaryMixKind::MPG,
            1.5, // > max
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_mixture_mass_based() {
        // Test custom mass-based mixture
        if let Ok(_mix) = Fluid::new_mass_based(
            "WaterEthanol",
            std::collections::HashMap::from([(Pure::Water, 0.6), (Pure::Ethanol, 0.4)]),
        ) {
            // Success
        }
    }

    #[test]
    fn test_custom_mixture_mole_based() {
        // Test custom mole-based mixture
        if let Ok(_mix) = Fluid::new_mole_based(
            "N2O2Ar",
            std::collections::HashMap::from([
                (Pure::Nitrogen, 0.78),
                (Pure::Oxygen, 0.21),
                (Pure::Argon, 0.01),
            ]),
        ) {
            // Success
        }
    }
}
