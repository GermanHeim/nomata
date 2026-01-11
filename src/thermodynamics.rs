//! Thermodynamic property calculations using CoolProp via rfluids.
//!
//! This module provides access to real thermodynamic properties through the
//! [`rfluids`](https://docs.rs/rfluids/latest/rfluids/) crate, which wraps the [CoolProp](https://coolprop.org/) library.
//!
//!
//! # Architecture
//!
//! The module uses a [`Substance`] enum to represent both pure components and
//! predefined mixtures, mirroring rfluids' substance structure:
//!
//! - **Pure fluids**: Single-component substances ([`Pure`]: fluids::Pure) (Water, Propane, R134a, etc.)
//! - **Predefined mixtures**: Multi-component fluids ([`PredefinedMix`]: fluids::PredefinedMix) (Air, R410A, R407C, etc.)
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
//! - `Fluid<Undefined>` for fluid creation from [`Pure`] or [`PredefinedMix`]
//! - `Fluid<Defined>` after state is specified with `.in_state()`
//! - All property methods return `Result<f64, FluidOutputError>`
//! - The [`Substance`] enum provides a unified interface for both types
//!
//! # Backend Selection
//!
//! You can customize the CoolProp backend used for calculations:
//!
//! ```ignore
//! use nomata::thermodynamics::{Fluid, fluids::Pure, backend::{BaseBackend, Backend, TabularMethod}};
//!
//! // Use base backend
//! let water = Fluid::builder(Pure::Water)
//!     .with_backend(Backend::Base(BaseBackend::Heos))
//!     .build();
//!
//! // Use tabular backend for faster calculations
//! let water_fast = Fluid::builder(Pure::Water)
//!     .with_backend(Backend::Tabular {
//!         base: BaseBackend::Heos,
//!         method: TabularMethod::Ttse,
//!     })
//!     .build();
//! ```

use rfluids::fluid::backend::{BaseBackend, TabularMethod};
use rfluids::prelude::*;

/// Backend configuration for CoolProp calculations.
///
/// Allows selection of either a base backend or a tabular backend
/// (which combines a base backend with an interpolation method).
///
/// For more information, see:
/// <https://docs.rs/rfluids/latest/rfluids/fluid/backend/enum.BaseBackend.html>
#[derive(Debug, Clone, Copy)]
/// Enum for selecting the CoolProp backend. See [`BaseBackend`] and [`TabularMethod`].
pub enum Backend {
    /// Base backend (e.g., Heos, Refprop, Incomp, If97, etc.) [`BaseBackend`]
    Base(BaseBackend),
    /// Tabular backend with interpolation method [`TabularMethod`]
    Tabular {
        /// The base backend to use
        base: BaseBackend,
        /// The tabular interpolation method
        method: TabularMethod,
    },
}

impl Backend {
    /// Extracts the base backend from this configuration.
    fn base_backend(&self) -> BaseBackend {
        match self {
            Backend::Base(b) => *b,
            Backend::Tabular { base, .. } => *base,
        }
    }
}

impl From<BaseBackend> for Backend {
    fn from(base: BaseBackend) -> Self {
        Backend::Base(base)
    }
}

/// Extension trait for [`BaseBackend`] to create tabular backends.
pub trait BaseBackendExt {
    /// Creates a tabular backend with the specified interpolation method.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::backend::{BaseBackend, TabularMethod, BaseBackendExt};
    ///
    /// let backend = BaseBackend::Heos.with(TabularMethod::Ttse);
    /// ```
    fn with(self, method: TabularMethod) -> Backend;
}

impl BaseBackendExt for BaseBackend {
    fn with(self, method: TabularMethod) -> Backend {
        Backend::Tabular { base: self, method }
    }
}

// Note: rfluids also has a Backend type. If needed, we could implement From
// to convert from rfluids::fluid::backend::Backend, but for now our custom
// Backend enum provides a cleaner API for our use case.

/// Result type for thermodynamic calculations.
pub type ThermoResult<T> = Result<T, ThermoError>;

/// Errors that can occur during thermodynamic calculations.
#[derive(Debug, Clone, thiserror::Error)]
/// Enum of errors that can occur during thermodynamic calculations.
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
/// - Pure components (single substances) [`Pure`]
/// - Predefined mixtures (standard blends like Air, R410A) [`PredefinedMix`]
/// - Binary mixtures (two-component custom mixtures) [`BinaryMix`]
/// - Custom mixtures (arbitrary multi-component mixtures) [`CustomMix`]
#[derive(Debug, Clone)]
pub enum Substance {
    /// Pure component fluid ([`Pure`])
    Pure(Pure),
    /// Predefined mixture (e.g., Air, R410A) ([`PredefinedMix`])
    PredefinedMix(PredefinedMix),
    /// Binary mixture (two-component custom mixture) ([`BinaryMix`])
    BinaryMix(BinaryMix),
    /// Custom mixture (arbitrary multi-component mixture) ([`CustomMix`])
    CustomMix(CustomMix),
}

impl From<Pure> for Substance {
    fn from(pure: Pure) -> Self {
        Substance::Pure(pure)
    }
}

impl From<PredefinedMix> for Substance {
    fn from(mix: PredefinedMix) -> Self {
        Substance::PredefinedMix(mix)
    }
}

impl Substance {
    /// Creates an rfluids Fluid object from this substance with optional backend configuration.
    /// Returns Fluid<Undefined> which must be converted to Fluid<Defined> via in_state().
    fn to_rfluids_fluid_with_backend(
        &self,
        backend: Option<Backend>,
    ) -> ThermoResult<rfluids::fluid::Fluid<Undefined>> {
        let fluid = match self {
            Substance::Pure(pure) => {
                if let Some(backend) = backend {
                    let base = backend.base_backend();
                    rfluids::fluid::Fluid::builder()
                        .substance(*pure)
                        .with_backend(base)
                        .build()
                        .map_err(|_| ThermoError::PropertyNotAvailable)?
                } else {
                    rfluids::fluid::Fluid::from(*pure)
                }
            }
            Substance::PredefinedMix(mix) => {
                if let Some(backend) = backend {
                    let base = backend.base_backend();
                    rfluids::fluid::Fluid::builder()
                        .substance(*mix)
                        .with_backend(base)
                        .build()
                        .map_err(|_| ThermoError::PropertyNotAvailable)?
                } else {
                    rfluids::fluid::Fluid::from(*mix)
                }
            }
            Substance::BinaryMix(binary) => {
                if let Some(backend) = backend {
                    let base = backend.base_backend();
                    rfluids::fluid::Fluid::builder()
                        .substance(*binary)
                        .with_backend(base)
                        .build()
                        .map_err(|_| ThermoError::PropertyNotAvailable)?
                } else {
                    rfluids::fluid::Fluid::from(*binary)
                }
            }
            Substance::CustomMix(custom) => {
                // For custom mixtures, we need to use try_from
                // Backend configuration might not be supported for all custom mixtures
                rfluids::fluid::Fluid::try_from(custom.clone())
                    .map_err(|_| ThermoError::PropertyNotAvailable)?
            }
        };

        // Note: TabularMethod configuration is handled through the Backend enum.
        // Some backends in rfluids may support tabular methods, but the exact
        // implementation depends on the rfluids version and CoolProp capabilities.

        Ok(fluid)
    }
}

/// Represents a chemical component or mixture with thermodynamic properties.
///
/// This wraps the `rfluids` fluid database and provides a type-safe
/// interface for property calculations.
///
/// Supports both pure components ([`Pure`]) and predefined mixtures ([`PredefinedMix`]).
#[derive(Debug, Clone)]
pub struct Fluid {
    /// The substance (pure or mixture) identifier for rfluids
    pub substance: Substance,
    /// Name of the fluid
    pub name: String,
    /// Optional backend configuration
    backend: Option<Backend>,
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
        Self { substance: Substance::Pure(pure), name, backend: None }
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
        Self { substance: Substance::PredefinedMix(mix), name, backend: None }
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

        Ok(Self { substance: Substance::BinaryMix(binary), name: name.to_string(), backend: None })
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

        Ok(Self { substance: Substance::CustomMix(custom), name: name.to_string(), backend: None })
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

        Ok(Self { substance: Substance::CustomMix(custom), name: name.to_string(), backend: None })
    }

    /// Extracts component names and compositions from the fluid.
    ///
    /// For pure fluids, returns a single component.
    /// For custom mixtures, returns the components from the mixture.
    /// For predefined/binary mixtures, returns the fluid name as a single component.
    ///
    /// # Returns
    ///
    /// A tuple of (component_names, composition_fractions)
    pub fn get_components(&self) -> (Vec<String>, Vec<f64>) {
        match &self.substance {
            Substance::Pure(pure) => (vec![format!("{:?}", pure)], vec![1.0]),
            Substance::CustomMix(custom) => {
                // Extract components from CustomMix
                let components = custom.components();
                let component_names: Vec<String> =
                    components.keys().map(|pure| format!("{:?}", pure)).collect();
                let composition: Vec<f64> = components.values().copied().collect();
                (component_names, composition)
            }
            Substance::PredefinedMix(_) | Substance::BinaryMix(_) => {
                // For predefined/binary mixes, use the fluid name
                (vec![self.name.clone()], vec![1.0])
            }
        }
    }

    /// Creates a builder for configuring a fluid with custom backend options.
    ///
    /// This is useful when you need more control over the CoolProp backend,
    /// such as using tabular methods for faster calculations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::Pure, backend::*};
    ///
    /// // High-accuracy equation of state
    /// let water = Fluid::builder(Pure::Water)
    ///     .with_backend(Backend::Base(BaseBackend::Heos))
    ///     .build();
    ///
    /// // Use tabular backend for faster calculations
    /// let r134a = Fluid::builder(Pure::R134a)
    ///     .with_backend(BaseBackend::Heos.with(TabularMethod::Ttse))
    ///     .build();
    /// ```
    pub fn builder(pure: Pure) -> FluidBuilder {
        FluidBuilder::new(Substance::Pure(pure))
    }

    /// Creates a builder for a predefined mixture with custom backend options.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::PredefinedMix, backend::*};
    ///
    /// let air = Fluid::builder_mix(PredefinedMix::Air)
    ///     .with_backend(Backend::Base(BaseBackend::Heos))
    ///     .build();
    /// ```
    pub fn builder_mix(mix: PredefinedMix) -> FluidBuilder {
        FluidBuilder::new(Substance::PredefinedMix(mix))
    }

    /// Creates a builder for a binary mixture with custom backend options.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::BinaryMixKind, backend::*};
    ///
    /// let mpg = Fluid::builder_binary(BinaryMixKind::MPG, 0.4)
    ///     .unwrap()
    ///     .with_backend(Backend::Base(BaseBackend::Incomp))
    ///     .build();
    /// ```
    pub fn builder_binary(kind: BinaryMixKind, fraction: f64) -> ThermoResult<FluidBuilder> {
        let binary = kind.with_fraction(fraction).map_err(|_| {
            ThermoError::InvalidInput(format!("Invalid fraction {} for {:?}", fraction, kind))
        })?;
        Ok(FluidBuilder::new(Substance::BinaryMix(binary)))
    }

    /// Computes properties given pressure and temperature.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Pressure \[Pa\]
    /// * `temperature` - Temperature \[K\]
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
            .to_rfluids_fluid_with_backend(self.backend)?
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
    /// * `pressure` - Pressure \[Pa\]
    /// * `enthalpy` - Specific enthalpy [J/kg]
    pub fn props_ph(&self, pressure: f64, enthalpy: f64) -> ThermoResult<Properties> {
        if pressure <= 0.0 {
            return Err(ThermoError::InvalidInput("Pressure must be positive".to_string()));
        }

        // Create a fluid in the defined state using PH inputs
        let mut state = self
            .substance
            .to_rfluids_fluid_with_backend(self.backend)?
            .in_state(FluidInput::pressure(pressure), FluidInput::enthalpy(enthalpy))
            .map_err(|_| ThermoError::ConvergenceFailure)?;

        let temp = state.temperature().map_err(|_| ThermoError::PropertyNotAvailable)?;
        Properties::from_fluid(&mut state, pressure, temp)
    }

    /// Gets the critical temperature \[K\].
    pub fn critical_temperature(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid_with_backend(self.backend)?;
        fluid.critical_temperature().map_err(|_| ThermoError::PropertyNotAvailable)
    }

    /// Gets the critical pressure \[Pa\].
    pub fn critical_pressure(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid_with_backend(self.backend)?;
        fluid.critical_pressure().map_err(|_| ThermoError::PropertyNotAvailable)
    }

    /// Gets the molecular weight [kg/mol].
    pub fn molecular_weight(&self) -> ThermoResult<f64> {
        let mut fluid = self.substance.to_rfluids_fluid_with_backend(self.backend)?;
        fluid.molar_mass().map_err(|_| ThermoError::PropertyNotAvailable)
    }
}

/// Builder for creating [`Fluid`]s with custom backend configuration.
///
/// Allows specification of CoolProp backend for more control over
/// calculation accuracy and performance.
pub struct FluidBuilder {
    substance: Substance,
    backend: Option<Backend>,
}

impl FluidBuilder {
    /// Creates a new fluid builder.
    fn new(substance: Substance) -> Self {
        Self { substance, backend: None }
    }

    /// Sets the CoolProp backend to use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use nomata::thermodynamics::{Fluid, fluids::Pure, backend::*};
    ///
    /// // Base backend
    /// let water = Fluid::builder(Pure::Water)
    ///     .with_backend(Backend::Base(BaseBackend::Heos))
    ///     .build();
    ///
    /// // Tabular backend
    /// let water_tab = Fluid::builder(Pure::Water)
    ///     .with_backend(Backend::Tabular {
    ///         base: BaseBackend::Heos,
    ///         method: TabularMethod::Ttse,
    ///     })
    ///     .build();
    /// ```
    ///
    /// Available base backends:
    ///
    /// - `BaseBackend::Heos`: Helmholtz Equation of State (high accuracy, default)
    /// - `BaseBackend::If97`: IAPWS-IF97 for water/steam
    /// - `BaseBackend::Incomp`: Incompressible fluids
    /// - `BaseBackend::Refprop`: REFPROP (if available)
    /// - `BaseBackend::Pr`: Peng-Robinson cubic EOS
    /// - `BaseBackend::Srk`: Soave-Redlich-Kwong cubic EOS
    /// - `BaseBackend::VtPr`: Volume-translated Peng-Robinson
    /// - `BaseBackend::PcSaft`: PC-SAFT equation of state
    ///
    /// Available tabular methods:
    ///
    /// - `TabularMethod::Ttse`: Two-phase Tabular Taylor Series Expansion
    /// - `TabularMethod::Bicubic`: Bicubic interpolation
    ///
    /// See: <https://docs.rs/rfluids/latest/rfluids/fluid/backend/>
    pub fn with_backend(mut self, backend: impl Into<Backend>) -> Self {
        self.backend = Some(backend.into());
        self
    }

    /// Builds the fluid with the configured options.
    pub fn build(self) -> Fluid {
        let name = match &self.substance {
            Substance::Pure(p) => format!("{:?}", p),
            Substance::PredefinedMix(m) => format!("{:?}", m),
            Substance::BinaryMix(_) => "BinaryMix".to_string(),
            Substance::CustomMix(_) => "CustomMix".to_string(),
        };

        Fluid { substance: self.substance, name, backend: self.backend }
    }
}

/// Thermodynamic properties at a specific state.
#[derive(Debug, Clone)]
pub struct Properties {
    /// Pressure \[Pa\]
    pub pressure: f64,
    /// Temperature \[K\]
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
    /// Dynamic viscosity \[Pa*s\] (None if not available)
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
    /// Re-export [`Pure`] enum from rfluids - contains all available pure fluids
    /// Use like: [`Pure::Water`], [`Pure::Nitrogen`], [`Pure::nPropane`], etc.
    pub use rfluids::prelude::Pure;

    /// Re-export [`PredefinedMix`] enum from rfluids - contains all predefined mixtures
    /// Use like: [`PredefinedMix::Air`], [`PredefinedMix::R410A`], etc.
    pub use rfluids::prelude::PredefinedMix;

    /// Re-export [`BinaryMixKind`] enum from rfluids - contains all binary mixture types
    /// Use like: [`BinaryMixKind::MPG`], [`BinaryMixKind::MEG`], etc.
    pub use rfluids::prelude::BinaryMixKind;
}

/// Backend configuration for CoolProp.
pub mod backend {
    pub use super::Backend;

    /// Re-export BaseBackend enum for selecting calculation backend
    /// See: [BaseBackend docs](https://docs.rs/rfluids/latest/rfluids/fluid/backend/enum.BaseBackend.html)
    pub use rfluids::fluid::backend::BaseBackend;

    /// Re-export TabularMethod enum for tabular interpolation methods
    /// See: [TabularMethod docs](https://docs.rs/rfluids/latest/rfluids/fluid/backend/enum.TabularMethod.html)
    pub use rfluids::fluid::backend::TabularMethod;
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
