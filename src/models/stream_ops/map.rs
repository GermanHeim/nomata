//! Component mapper (MAP) for lumping and delumping stream compositions.
//!
//! A MAP unit transforms stream composition between detailed and lumped
//! component representations. Total flow, temperature, and pressure are
//! preserved unchanged. Only the component identities and fractions change.
//!
//! # Use cases
//!
//! - **Lumping**: combine MA + PD (two detailed components) into MAPD (one pseudo-component)
//! - **Delumping**: split MAPD into MA (50 %) and PD (50 %)
//!
//! # Example
//!
//! ```ignore
//! use nomata::prelude::*;
//!
//! let feed = Stream::<MolarFlow>::new()
//!     .with_flow(10.0)
//!     .with_temperature(320.0)
//!     .with_pressure(2e5)
//!     .with_composition(&["MA", "PD"], &[0.5, 0.5])
//!     .build()?;
//!
//! let mut mapper = Map::new("lump-1")
//!     .lumping()
//!     .add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.5), ("PD", 0.5)]))
//!     .build()?;
//!
//! let outlet = mapper.process(feed)?;
//! // outlet has component "MAPD" with fraction 1.0
//! ```
//!
//! # Physics
//!
//! Three aggregate equations (the component transformation is parameter-driven):
//!
//! 1. **Mass balance**: F_out = F_in
//! 2. **Temperature**:  T_out = T_in  (adiabatic, no heat exchange)
//! 3. **Pressure**:     P_out = P_in  (no pressure drop)

use crate::{EquationModel, MolarFlow, NomataError, NomataResult, Process, Stream};

/// Describes how one lumped pseudo-component maps to several detailed components.
///
/// The fractions represent the composition of the pseudo-component in terms of
/// the detailed components (must sum to 1.0 for a valid mapping).
#[derive(Debug, Clone)]
pub struct ComponentMapping {
    lumped: String,
    detailed: Vec<(String, f64)>,
}

impl ComponentMapping {
    /// Creates a new component mapping.
    ///
    /// # Arguments
    ///
    /// * `lumped` - Name of the lumped pseudo-component
    /// * `detailed` - Pairs of (component name, fraction); fractions should sum to 1.0
    pub fn new(lumped: &str, detailed: Vec<(&str, f64)>) -> Self {
        ComponentMapping {
            lumped: lumped.to_string(),
            detailed: detailed.into_iter().map(|(name, frac)| (name.to_string(), frac)).collect(),
        }
    }

    /// Name of the lumped (pseudo) component.
    pub fn lumped_component(&self) -> &str {
        &self.lumped
    }

    /// Detailed components with their fractions.
    pub fn detailed_components(&self) -> &[(String, f64)] {
        &self.detailed
    }

    /// Number of detailed components this pseudo-component expands into.
    pub fn detailed_count(&self) -> usize {
        self.detailed.len()
    }

    /// Returns true if the detailed fractions sum to approximately 1.0.
    pub fn validate_fractions(&self) -> bool {
        let sum: f64 = self.detailed.iter().map(|(_, f)| f).sum();
        (sum - 1.0).abs() < 1e-6
    }
}

/// Direction of the component transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingDirection {
    /// Combine detailed components into lumped pseudo-components.
    Lumping,
    /// Split lumped pseudo-components into detailed components.
    Delumping,
}

/// Variable indices for the MAP model.
#[derive(Clone, Copy)]
struct MapVars {
    f_in: usize,
    t_in: usize,
    p_in: usize,
    f_out: usize,
    t_out: usize,
    p_out: usize,
}

impl Default for MapVars {
    fn default() -> Self {
        MapVars { f_in: 0, t_in: 1, p_in: 2, f_out: 3, t_out: 4, p_out: 5 }
    }
}

/// Component mapper for lumping and delumping stream representations.
///
/// Implements both `EquationModel` (EO) for aggregate flow balances and
/// `Process` (SM) for applying the full component transformation.
#[derive(Debug)]
pub struct Map {
    name: String,
    direction: MappingDirection,
    mappings: Vec<ComponentMapping>,

    // Variable values: F_in, T_in, P_in, F_out, T_out, P_out
    vars: [f64; 6],
}

/// Named accessor for Map unit variables.
///
/// Use with [`Map::var`] to read individual variable values by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapVar {
    /// Inlet flow rate \[mol/s\]
    InletFlow,
    /// Inlet temperature \[K\]
    InletTemperature,
    /// Inlet pressure \[Pa\]
    InletPressure,
    /// Outlet flow rate \[mol/s\]
    OutletFlow,
    /// Outlet temperature \[K\]
    OutletTemperature,
    /// Outlet pressure \[Pa\]
    OutletPressure,
}

impl MapVar {
    fn index(self) -> usize {
        match self {
            MapVar::InletFlow => 0,
            MapVar::InletTemperature => 1,
            MapVar::InletPressure => 2,
            MapVar::OutletFlow => 3,
            MapVar::OutletTemperature => 4,
            MapVar::OutletPressure => 5,
        }
    }
}

impl Map {
    /// Creates a new MAP builder.
    pub fn new(name: &str) -> MapBuilder {
        MapBuilder {
            name: name.to_string(),
            direction: MappingDirection::Lumping,
            mappings: Vec::new(),
        }
    }

    /// Gets the mapper name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the current value of a named variable.
    pub fn var(&self, v: MapVar) -> f64 {
        self.vars[v.index()]
    }

    /// Gets the transformation direction.
    pub fn direction(&self) -> MappingDirection {
        self.direction
    }

    /// Gets the component mappings.
    pub fn mappings(&self) -> &[ComponentMapping] {
        &self.mappings
    }

    /// Number of component mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Returns true if all mapping fractions are valid (sum to 1.0).
    pub fn validate_mappings(&self) -> bool {
        self.mappings.iter().all(|m| m.validate_fractions())
    }

    /// Expected number of inlet components given direction and mappings.
    pub fn input_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Lumping => self.mappings.iter().map(|m| m.detailed_count()).sum(),
            MappingDirection::Delumping => self.mappings.len(),
        }
    }

    /// Expected number of outlet components given direction and mappings.
    pub fn output_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Lumping => self.mappings.len(),
            MappingDirection::Delumping => self.mappings.iter().map(|m| m.detailed_count()).sum(),
        }
    }

    fn residuals_generic<S: crate::Scalar>(&self, vars: &[S]) -> Vec<S> {
        let idx = MapVars::default();
        let r1 = vars[idx.f_out] - vars[idx.f_in];
        let r2 = vars[idx.t_out] - vars[idx.t_in];
        let r3 = vars[idx.p_out] - vars[idx.p_in];
        vec![r1, r2, r3]
    }
}

impl EquationModel for Map {
    fn name(&self) -> &str {
        &self.name
    }

    fn n_variables(&self) -> usize {
        6
    }

    fn n_equations(&self) -> usize {
        3
    }

    fn variable_names(&self) -> Vec<&str> {
        vec!["F_in", "T_in", "P_in", "F_out", "T_out", "P_out"]
    }

    fn residuals(&self, vars: &[f64]) -> Vec<f64> {
        self.residuals_generic(vars)
    }

    #[cfg(feature = "autodiff")]
    fn residuals_dual(&self, vars: &[num_dual::Dual64]) -> Vec<num_dual::Dual64> {
        self.residuals_generic(vars)
    }

    fn get_variables(&self) -> Vec<f64> {
        self.vars.to_vec()
    }

    fn set_variables(&mut self, vars: &[f64]) {
        for (i, &v) in vars.iter().enumerate().take(6) {
            self.vars[i] = v;
        }
    }

    fn specified_indices(&self) -> Vec<usize> {
        let idx = MapVars::default();
        vec![idx.f_in, idx.t_in, idx.p_in]
    }

    fn n_inlet_ports(&self) -> usize {
        1
    }

    fn n_outlet_ports(&self) -> usize {
        1
    }

    fn inlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![0, 1, 2] // F_in, T_in, P_in
    }

    fn outlet_port_indices(&self, _port: usize) -> Vec<usize> {
        vec![3, 4, 5] // F_out, T_out, P_out
    }
}

impl Process for Map {
    type Input = Stream<MolarFlow>;
    type Output = Stream<MolarFlow>;

    fn process(&mut self, input: Self::Input) -> NomataResult<Self::Output> {
        let inlet_data = input.to_data();
        let idx = MapVars::default();

        self.vars[idx.f_in] = inlet_data.flow;
        self.vars[idx.t_in] = inlet_data.temperature;
        self.vars[idx.p_in] = inlet_data.pressure;
        self.vars[idx.f_out] = inlet_data.flow;
        self.vars[idx.t_out] = inlet_data.temperature;
        self.vars[idx.p_out] = inlet_data.pressure;

        // Build a map from inlet component name -> molar fraction
        let inlet_fracs: std::collections::HashMap<&str, f64> = inlet_data
            .components
            .iter()
            .map(|s| s.as_str())
            .zip(inlet_data.composition.iter().copied())
            .collect();

        let (outlet_names, outlet_fracs) = match self.direction {
            MappingDirection::Lumping => {
                // For each mapping, sum the detailed fractions to get the lumped fraction.
                // Components not covered by any mapping are passed through unchanged.
                let mut out_names: Vec<String> = Vec::new();
                let mut out_fracs: Vec<f64> = Vec::new();
                let mut covered: std::collections::HashSet<&str> = std::collections::HashSet::new();

                for mapping in &self.mappings {
                    let lumped_frac: f64 = mapping
                        .detailed_components()
                        .iter()
                        .map(|(name, _)| {
                            covered.insert(name.as_str());
                            inlet_fracs.get(name.as_str()).copied().unwrap_or(0.0)
                        })
                        .sum();
                    out_names.push(mapping.lumped_component().to_string());
                    out_fracs.push(lumped_frac);
                }

                // Pass through any unmapped components
                for (name, frac) in &inlet_fracs {
                    if !covered.contains(*name) {
                        out_names.push((*name).to_string());
                        out_fracs.push(*frac);
                    }
                }

                (out_names, out_fracs)
            }

            MappingDirection::Delumping => {
                // For each mapping, split the lumped component into detailed components
                // using the defined fractions.
                let mut out_names: Vec<String> = Vec::new();
                let mut out_fracs: Vec<f64> = Vec::new();
                let mut covered: std::collections::HashSet<&str> = std::collections::HashSet::new();

                for mapping in &self.mappings {
                    let lumped_name = mapping.lumped_component();
                    covered.insert(lumped_name);
                    let lumped_frac = inlet_fracs.get(lumped_name).copied().unwrap_or(0.0);

                    for (detailed_name, split_frac) in mapping.detailed_components() {
                        out_names.push(detailed_name.clone());
                        out_fracs.push(lumped_frac * split_frac);
                    }
                }

                // Pass through any unmapped components
                for name in &inlet_data.components {
                    if !covered.contains(name.as_str()) {
                        out_names.push(name.clone());
                        out_fracs.push(inlet_fracs.get(name.as_str()).copied().unwrap_or(0.0));
                    }
                }

                (out_names, out_fracs)
            }
        };

        let names_ref: Vec<&str> = outlet_names.iter().map(|s| s.as_str()).collect();

        Stream::<MolarFlow>::new()
            .with_flow(inlet_data.flow)
            .with_temperature(inlet_data.temperature)
            .with_pressure(inlet_data.pressure)
            .with_composition(&names_ref, &outlet_fracs)
            .build()
    }
}

/// Builder for Map.
#[derive(Debug)]
pub struct MapBuilder {
    name: String,
    direction: MappingDirection,
    mappings: Vec<ComponentMapping>,
}

impl MapBuilder {
    /// Sets the direction to lumping (detailed -> lumped).
    pub fn lumping(mut self) -> Self {
        self.direction = MappingDirection::Lumping;
        self
    }

    /// Sets the direction to delumping (lumped -> detailed).
    pub fn delumping(mut self) -> Self {
        self.direction = MappingDirection::Delumping;
        self
    }

    /// Adds a component mapping rule.
    pub fn add_mapping(mut self, mapping: ComponentMapping) -> Self {
        self.mappings.push(mapping);
        self
    }

    /// Builds the MAP unit.
    pub fn build(self) -> NomataResult<Map> {
        if self.mappings.is_empty() {
            return Err(NomataError::Configuration(
                "Map requires at least one ComponentMapping".to_string(),
            ));
        }

        let invalid: Vec<&str> = self
            .mappings
            .iter()
            .filter(|m| !m.validate_fractions())
            .map(|m| m.lumped_component())
            .collect();

        if !invalid.is_empty() {
            return Err(NomataError::Validation(format!(
                "ComponentMapping fractions do not sum to 1.0 for: {}",
                invalid.join(", ")
            )));
        }

        Ok(Map {
            name: self.name,
            direction: self.direction,
            mappings: self.mappings,
            vars: [0.0; 6],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_variable_counts() {
        let mapper = Map::new("m")
            .lumping()
            .add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]))
            .build()
            .unwrap();

        assert_eq!(mapper.n_variables(), 6);
        assert_eq!(mapper.n_equations(), 3);
        assert_eq!(mapper.free_indices().len(), 3);
    }

    #[test]
    fn test_map_equations_at_solution() {
        let mut mapper = Map::new("m")
            .lumping()
            .add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]))
            .build()
            .unwrap();

        // F_out = F_in, T_out = T_in, P_out = P_in
        mapper.set_variables(&[10.0, 300.0, 2e5, 10.0, 300.0, 2e5]);
        let res = mapper.residuals(&mapper.get_variables());
        for (i, r) in res.iter().enumerate() {
            assert!(r.abs() < 1e-10, "Residual {} = {}", i, r);
        }
    }

    #[test]
    fn test_lumping_process() {
        let mut mapper = Map::new("lump")
            .lumping()
            .add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]))
            .build()
            .unwrap();

        let feed = Stream::<MolarFlow>::new()
            .with_flow(10.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .with_composition(&["MA", "PD"], &[0.6, 0.4])
            .build()
            .unwrap();

        let outlet = mapper.process(feed).unwrap();
        let data = outlet.to_data();

        assert!((data.flow - 10.0).abs() < 1e-10);
        assert_eq!(data.components, vec!["MAPD"]);
        assert!((data.composition[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_delumping_process() {
        let mut mapper = Map::new("delump")
            .delumping()
            .add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]))
            .build()
            .unwrap();

        let feed = Stream::<MolarFlow>::new()
            .with_flow(10.0)
            .with_temperature(300.0)
            .with_pressure(1e5)
            .with_composition(&["MAPD"], &[1.0])
            .build()
            .unwrap();

        let outlet = mapper.process(feed).unwrap();
        let data = outlet.to_data();

        assert!((data.flow - 10.0).abs() < 1e-10);
        assert_eq!(data.components, vec!["MA", "PD"]);
        assert!((data.composition[0] - 0.6).abs() < 1e-6);
        assert!((data.composition[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_fractions_rejected() {
        let result = Map::new("bad")
            .lumping()
            .add_mapping(ComponentMapping::new("X", vec![("A", 0.3), ("B", 0.6)]))
            .build();

        assert!(result.is_err());
    }
}
