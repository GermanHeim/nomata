//! Compressor model for gas compression.

use crate::*;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::Fluid;

#[cfg(feature = "thermodynamics")]
pub use rfluids::prelude::Pure;

use std::collections::HashMap;
use std::marker::PhantomData;
use thiserror::Error;

/// Error type for compressor operations.
#[derive(Error, Debug)]
pub enum CompressorError {
    #[error("Compressor parameter not initialized: {parameter}")]
    UninitializedParameter { parameter: String },
    #[error("Stream error: {0}")]
    StreamError(#[from] crate::StreamError),
    #[cfg(feature = "thermodynamics")]
    #[error("Thermodynamics error: {0}")]
    ThermoError(#[from] crate::thermodynamics::ThermoError),
}

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
    /// Isentropic outlet temperature
    pub t2s: S,
    /// Isentropic efficiency
    pub eta: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for OutletTempVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["t2", "t1", "t2s", "eta"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            t2: *vars.get(&format!("{}_t2", prefix))?,
            t1: *vars.get(&format!("{}_t1", prefix))?,
            t2s: *vars.get(&format!("{}_t2s", prefix))?,
            eta: *vars.get(&format!("{}_eta", prefix))?,
        })
    }
}

impl EquationVars for OutletTempVars<f64> {
    fn base_names() -> &'static [&'static str] {
        &["t2", "t1", "t2s", "eta"]
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

    // Composition (pass-through for non-reactive compressor)
    inlet_composition: Vec<f64>,
    component_names: Vec<String>,

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, P>,
    pub outlet: Port<Stream<MolarFlow>, Output, P>,

    #[cfg(feature = "thermodynamics")]
    pub fluid: Option<Fluid>,

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

            inlet_composition: vec![1.0],
            component_names: vec!["Unknown".to_string()],

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

            inlet_composition: self.inlet_composition,
            component_names: self.component_names,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid: self.fluid,

            _c: PhantomData,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Sets compressor configuration with gamma from thermodynamic properties.
    ///
    /// Automatically handles both pure components and mixtures:
    /// - For a single component (pure): gamma calculated directly from component properties
    /// - For mixtures: uses mole-fraction weighted mixing rules:
    ///   - Cp_mix = sum(y_i * Cp_i)
    ///   - Cv_mix = sum(y_i * Cv_i)
    ///   - gamma_mix = Cp_mix / Cv_mix
    ///
    /// # Arguments
    /// * `components` - Vector of Pure components (single element for pure, multiple for mixture)
    /// * `mole_fractions` - Mole fractions for each component (must sum to 1.0)
    pub fn with_configuration_from_fluid(
        self,
        inlet_pressure: f64,
        inlet_temp: f64,
        mass_flow: f64,
        isentropic_efficiency: f64,
        components: Vec<Pure>,
        mole_fractions: Vec<f64>,
    ) -> Result<Compressor<Initialized, P>, crate::thermodynamics::ThermoError> {
        if components.len() != mole_fractions.len() {
            return Err(crate::thermodynamics::ThermoError::InvalidInput(
                "Number of components must match number of mole fractions".to_string(),
            ));
        }

        let sum: f64 = mole_fractions.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(crate::thermodynamics::ThermoError::InvalidInput(format!(
                "Mole fractions must sum to 1.0, got {}",
                sum
            )));
        }

        // Calculate mixture properties using mixing rules
        let mut cp_mix = 0.0;
        let mut cv_mix = 0.0;
        let mut component_names = Vec::new();

        for (i, pure) in components.iter().enumerate() {
            let fluid = Fluid::new(*pure);
            let props = fluid.props_pt(inlet_pressure, inlet_temp)?;

            cp_mix += mole_fractions[i] * props.cp;
            cv_mix += mole_fractions[i] * props.cv;
            component_names.push(fluid.name.clone());
        }

        let gamma = cp_mix / cv_mix;

        // For pure components, store the fluid object
        let fluid = if components.len() == 1 { Some(Fluid::new(components[0])) } else { None };

        Ok(Compressor {
            outlet_pressure: self.outlet_pressure,
            outlet_temp: self.outlet_temp,
            power: self.power,

            inlet_pressure: Some(Var::new(inlet_pressure)),
            inlet_temp: Some(Var::new(inlet_temp)),
            mass_flow: Some(Var::new(mass_flow)),
            isentropic_efficiency: Some(Var::new(isentropic_efficiency)),
            gamma: Some(Var::new(gamma)),

            inlet_composition: mole_fractions,
            component_names,

            inlet: self.inlet,
            outlet: self.outlet,

            #[cfg(feature = "thermodynamics")]
            fluid,

            _c: PhantomData,
        })
    }
}

// TODO: Improve use of thermodynamic properties for more accurate compression modeling

// Operations only available when fully initialized
impl<P: PortState> Compressor<Initialized, P> {
    /// Computes outlet conditions and power requirement. Only available for fully initialized compressor.
    pub fn compute_compression(&mut self) -> Result<(), CompressorError> {
        let p1 = self
            .inlet_pressure
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "inlet_pressure".to_string(),
            })?
            .get();
        let p2 = self.outlet_pressure.get();
        let t1 = self
            .inlet_temp
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "inlet_temp".to_string(),
            })?
            .get();
        let eta = self
            .isentropic_efficiency
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "isentropic_efficiency".to_string(),
            })?
            .get();
        let gamma = self
            .gamma
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "gamma".to_string(),
            })?
            .get();

        // Isentropic outlet temperature
        let t2_isen = t1 * (p2 / p1).powf((gamma - 1.0) / gamma);

        // Actual outlet temperature
        let t2_actual = t1 + (t2_isen - t1) / eta;

        self.outlet_temp = Var::new(t2_actual);

        // Power (assuming ideal gas)
        let cp = gamma * 287.0 / (gamma - 1.0); // For air, R = 287 J/kg*K
        let mass_flow = self
            .mass_flow
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "mass_flow".to_string(),
            })?
            .get();
        let w = mass_flow * cp * (t2_actual - t1);

        self.power = Var::new(w);
        Ok(())
    }

    #[cfg(feature = "thermodynamics")]
    /// Updates gamma using thermodynamic properties.
    /// Only available for fully initialized compressor.
    ///
    /// Automatically handles both pure components and mixtures based on input.
    ///
    /// # Arguments
    /// * `components` - Vector of Pure components (single element for pure, multiple for mixture)
    /// * `mole_fractions` - Mole fractions for each component (must sum to 1.0)
    pub fn update_gamma(
        &mut self,
        components: Vec<Pure>,
        mole_fractions: Vec<f64>,
    ) -> Result<(), CompressorError> {
        if components.len() != mole_fractions.len() {
            return Err(CompressorError::ThermoError(
                crate::thermodynamics::ThermoError::InvalidInput(
                    "Number of components must match number of mole fractions".to_string(),
                ),
            ));
        }

        if components.is_empty() {
            return Err(CompressorError::ThermoError(
                crate::thermodynamics::ThermoError::InvalidInput(
                    "At least one component must be provided".to_string(),
                ),
            ));
        }

        let sum: f64 = mole_fractions.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(CompressorError::ThermoError(
                crate::thermodynamics::ThermoError::InvalidInput(format!(
                    "Mole fractions must sum to 1.0, got {}",
                    sum
                )),
            ));
        }

        let temp = self
            .inlet_temp
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "inlet_temp".to_string(),
            })?
            .get();
        let pressure = self
            .inlet_pressure
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "inlet_pressure".to_string(),
            })?
            .get();

        // Calculate mixture properties using mixing rules
        let mut cp_mix = 0.0;
        let mut cv_mix = 0.0;

        for (i, pure) in components.iter().enumerate() {
            let fluid = Fluid::new(*pure);
            let props = fluid.props_pt(pressure, temp)?;

            cp_mix += mole_fractions[i] * props.cp;
            cv_mix += mole_fractions[i] * props.cv;
        }

        let gamma = cp_mix / cv_mix;

        self.gamma = Some(Var::new(gamma));
        self.inlet_composition = mole_fractions;

        // Update fluid object for pure components
        if components.len() == 1 {
            self.fluid = Some(Fluid::new(components[0]));
        } else {
            self.fluid = None;
        }

        Ok(())
    }

    /// Computes the outlet stream with compressed conditions (internal use).
    ///
    /// For flowsheet building, use `outlet_stream()` which returns a reference.
    /// This method is called internally by the solver.
    pub fn compute_outlet_stream(&self) -> Result<Stream<MolarFlow>, CompressorError> {
        let flow = self
            .mass_flow
            .as_ref()
            .ok_or_else(|| CompressorError::UninitializedParameter {
                parameter: "mass_flow".to_string(),
            })?
            .get();
        let pressure = self.outlet_pressure.get();
        let temp = self.outlet_temp.get();

        #[cfg(feature = "thermodynamics")]
        {
            if let Some(fluid) = &self.fluid {
                let component_name = fluid.name.clone();
                let stream = Stream::pure(flow, component_name, temp, pressure);
                return Ok(stream);
            }
        }

        // If we have multicomponent composition info, use it
        if !self.component_names.is_empty()
            && self.component_names[0] != "Unknown"
            && let Ok(stream) = Stream::with_composition(
                flow,
                self.component_names.clone(),
                self.inlet_composition.clone(),
            )
        {
            return Ok(stream.at_conditions(temp, pressure));
        }

        Err(CompressorError::StreamError(crate::StreamError::MissingComposition {
            model: "Compressor".to_string(),
            suggestion: "Use FromStream trait to initialize from inlet stream".to_string(),
        }))
    }

    /// Returns a reference to the outlet stream.
    ///
    /// The reference doesn't contain stream data until the flowsheet is solved.
    /// After solving, call `.get()` on the reference to access the stream.
    pub fn outlet_stream(&self) -> crate::OutletRef {
        crate::OutletRef::new("Compressor", "outlet")
    }

    /// Populates the outlet reference with the current computed stream.
    ///
    /// Call this after solving to make results accessible via `outlet_stream().get()`.
    pub fn populate_outlet(&self, outlet_ref: &crate::OutletRef) -> Result<(), CompressorError> {
        let stream = self.compute_outlet_stream()?;
        outlet_ref.set(stream);
        Ok(())
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::FromStream for Compressor<Initialized> {
    fn from_stream<S: crate::StreamType, C>(stream: &crate::Stream<S, C>) -> Self {
        Compressor {
            outlet_pressure: Var::new(stream.pressure * 2.0),
            outlet_temp: Var::new(stream.temperature),
            power: Var::new(0.0),

            inlet_pressure: Some(Var::new(stream.pressure)),
            inlet_temp: Some(Var::new(stream.temperature)),
            mass_flow: Some(Var::new(stream.total_flow)),
            isentropic_efficiency: Some(Var::new(0.85)),
            gamma: Some(Var::new(1.4)),

            inlet_composition: stream.composition.clone(),
            component_names: stream.components.clone(),

            inlet: Port::new(),
            outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            fluid: None,

            _c: PhantomData,
        }
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
            |v: OutletTempVars<f64>| v.t2 - v.t1 - (v.t2s - v.t1) / v.eta,
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
            |v: OutletTempVars<f64>| v.t2 - v.t1 - (v.t2s - v.t1) / v.eta,
            |v: OutletTempVars<Dual64>| v.t2 - v.t1 - (v.t2s - v.t1) / v.eta,
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

/// Compile-time port specification for Compressor.
///
/// Enables type-safe connections with const generic port indices.
impl<C, P: PortState> PortSpec for Compressor<C, P> {
    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = 1;
    const STREAM_TYPE: &'static str = "MolarFlow";
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

        comp.compute_compression().unwrap();

        assert!(comp.outlet_temp.get() > 298.15);
        assert!(comp.power.get() > 0.0);
    }

    #[test]
    fn test_compressor_outlet_stream() {
        let inlet_stream: Stream<MolarFlow> =
            Stream::with_composition(2.0, vec!["Air".to_string()], vec![1.0])
                .unwrap()
                .at_conditions(298.15, 101325.0);
        let mut comp = Compressor::from_stream(&inlet_stream);
        comp.outlet_pressure = Var::new(200000.0);
        comp.compute_compression().unwrap();

        let outlet_ref = comp.outlet_stream();
        comp.populate_outlet(&outlet_ref).unwrap();
        let outlet = outlet_ref.get().unwrap();

        assert_eq!(outlet.total_flow, 2.0);
        assert_eq!(outlet.pressure, comp.outlet_pressure.get());
        assert_eq!(outlet.temperature, comp.outlet_temp.get());
    }

    #[test]
    fn test_compressor_multicomponent_composition() {
        let components = vec!["N2".to_string(), "O2".to_string()];
        let composition = vec![0.79, 0.21];

        let inlet_stream: Stream<MolarFlow> =
            Stream::with_composition(10.0, components.clone(), composition.clone())
                .unwrap()
                .at_conditions(298.15, 101325.0);
        let mut comp = Compressor::from_stream(&inlet_stream);
        comp.outlet_pressure = Var::new(300000.0);
        comp.compute_compression().unwrap();

        let outlet_ref = comp.outlet_stream();
        comp.populate_outlet(&outlet_ref).unwrap();
        let outlet = outlet_ref.get().unwrap();

        assert_eq!(outlet.components, components);
        assert_eq!(outlet.composition, composition);
        assert_eq!(outlet.total_flow, 10.0);
        assert!(outlet.temperature > 298.15); // Should be heated by compression
    }

    #[test]
    #[cfg(feature = "thermodynamics")]
    fn test_compressor_with_pure_thermodynamics() {
        use crate::models::compressor::Pure;

        // Pure nitrogen
        let comp: Result<Compressor<Initialized>, _> = Compressor::new()
            .with_configuration_from_fluid(
                101325.0,
                298.15,
                1.0,
                0.85,
                vec![Pure::Nitrogen],
                vec![1.0],
            );

        assert!(comp.is_ok());
        let comp = comp.unwrap();

        let gamma = comp.gamma.as_ref().unwrap().get();
        assert!(gamma > 1.0, "Gamma should be greater than 1.0, got {}", gamma);
        assert!(comp.fluid.is_some(), "Pure component should have fluid object");
    }

    #[test]
    #[cfg(feature = "thermodynamics")]
    fn test_compressor_with_mixture_thermodynamics() {
        use crate::models::compressor::Pure;

        // Air mixture: 79% N2, 21% O2
        let components = vec![Pure::Nitrogen, Pure::Oxygen];
        let mole_fractions = vec![0.79, 0.21];

        let comp: Result<Compressor<Initialized>, _> = Compressor::new()
            .with_configuration_from_fluid(
                101325.0,
                298.15,
                1.0,
                0.85,
                components,
                mole_fractions.clone(),
            );

        assert!(comp.is_ok());
        let comp = comp.unwrap();

        // Gamma for air should be around 1.4
        let gamma = comp.gamma.as_ref().unwrap().get();
        assert!(gamma > 1.35 && gamma < 1.45, "Expected gamma around 1.4, got {}", gamma);

        assert_eq!(comp.inlet_composition, mole_fractions);
        assert!(comp.fluid.is_none(), "Mixture should not have single fluid object");
    }

    #[test]
    #[cfg(feature = "thermodynamics")]
    fn test_update_gamma() {
        use crate::models::compressor::Pure;

        let mut comp: Compressor<Initialized> =
            Compressor::new().with_configuration(101325.0, 298.15, 1.0, 0.85, 1.4);

        // Update to use thermodynamic properties for a mixture
        let components = vec![Pure::Nitrogen, Pure::Oxygen];
        let mole_fractions = vec![0.79, 0.21];

        let result = comp.update_gamma(components, mole_fractions.clone());
        assert!(result.is_ok());

        let gamma = comp.gamma.as_ref().unwrap().get();
        assert!(gamma > 1.35 && gamma < 1.45, "Expected gamma around around 1.4, got {}", gamma);
        assert_eq!(comp.inlet_composition, mole_fractions);
    }
}
