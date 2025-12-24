//! Component Mapper (MAP) - Lumping and Delumping Unit Operation
//!
//! The MAP unit transforms between detailed and lumped component representations,
//! enabling equation reduction and module interfacing with different detail levels.
//!
//! # Examples
//!
//! ```ignore
//! use nomata::models::MAP;
//! use nomata::{ComponentMapping, MappingDirection};
//!
//! // Create a delumper: MAPD -> MA (60%) + PD (40%)
//! let mut delumper = MAP::new("MAPD_Delumper");
//! delumper.set_direction(MappingDirection::Delumping);
//! delumper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));
//! ```

use crate::{
    Algebraic, ComponentMapping, EquationSystem, EquationTerm, HasPorts, MappingDirection,
    MolarFlow, NamedPort, ResidualFunction, Stream, TimeDomain, UnitOp, Var, VariableRegistry,
};
use std::marker::PhantomData;

/// Component Mapper unit operation (MAP).
///
/// Transforms between detailed and lumped component representations:
/// - **Lumping**: Combine detailed components -> lumped pseudo-component
///   Example: MA + PD -> MAPD
/// - **Delumping**: Split lumped pseudo-component -> detailed components
///   Example: MAPD -> 60% MA + 40% PD
///
/// # Physics
///
/// Conservation of mass/moles:
/// - Lumping: F_lumped = sum(F_detailed_i)
/// - Delumping: F_detailed_i = fraction_i * F_lumped
///
/// # Applications
///
/// 1. **Module Interfacing**: Connect modules with different component detail
/// 2. **Equation Reduction**: Use lumped components in simplified sections
/// 3. **Model Abstraction**: Hide complexity at boundaries
///
/// # Examples
///
/// ```
/// use nomata::models::MAP;
/// use nomata::{ComponentMapping, MappingDirection, Steady};
///
/// let mut mapper = MAP::<Steady>::new("C4_Lumper");
/// mapper.set_direction(MappingDirection::Lumping);
/// mapper.add_mapping(ComponentMapping::new(
///     "C4_mix",
///     vec![("n-butane", 0.5), ("iso-butane", 0.5)]
/// ));
///
/// assert_eq!(mapper.mapping_count(), 1);
/// ```
#[derive(Debug)]
pub struct MAP<T: TimeDomain> {
    /// Unit name
    name: String,
    /// Transformation direction (lumping or delumping)
    direction: MappingDirection,
    /// Component mapping rules
    mappings: Vec<ComponentMapping>,
    /// Inlet flowrate
    inlet_flow: Var<Algebraic>,
    /// Outlet flowrate
    outlet_flow: Var<Algebraic>,
    /// Inlet temperature
    inlet_temp: Var<Algebraic>,
    /// Outlet temperature (assumed same as inlet)
    outlet_temp: Var<Algebraic>,
    /// Phantom data for time domain
    _time_domain: PhantomData<T>,
}

impl<T: TimeDomain> MAP<T> {
    /// Creates a new component mapper.
    ///
    /// # Arguments
    ///
    /// * `name` - Descriptive name for the mapper unit
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::models::MAP;
    /// use nomata::Dynamic;
    ///
    /// let mapper = MAP::<Dynamic>::new("MAPD_Delumper");
    /// assert_eq!(mapper.name(), "MAPD_Delumper");
    /// ```
    pub fn new(name: &str) -> Self {
        let registry = VariableRegistry::new();
        MAP {
            name: name.to_string(),
            direction: MappingDirection::Lumping,
            mappings: Vec::new(),
            inlet_flow: registry.create_algebraic(0.0),
            outlet_flow: registry.create_algebraic(0.0),
            inlet_temp: registry.create_algebraic(298.15),
            outlet_temp: registry.create_algebraic(298.15),
            _time_domain: PhantomData,
        }
    }

    /// Gets the mapper name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sets the transformation direction.
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::models::MAP;
    /// use nomata::{MappingDirection, Steady};
    ///
    /// let mut mapper = MAP::<Steady>::new("mapper");
    /// mapper.set_direction(MappingDirection::Delumping);
    /// assert_eq!(mapper.direction(), MappingDirection::Delumping);
    /// ```
    pub fn set_direction(&mut self, direction: MappingDirection) {
        self.direction = direction;
    }

    /// Gets the transformation direction.
    pub fn direction(&self) -> MappingDirection {
        self.direction
    }

    /// Adds a component mapping rule.
    ///
    /// # Arguments
    ///
    /// * `mapping` - Defines how one lumped component maps to detailed components
    ///
    /// # Examples
    ///
    /// ```
    /// use nomata::models::MAP;
    /// use nomata::{ComponentMapping, Steady};
    ///
    /// let mut mapper = MAP::<Steady>::new("mapper");
    /// mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));
    /// mapper.add_mapping(ComponentMapping::new("C5", vec![("pentane", 1.0)]));
    ///
    /// assert_eq!(mapper.mapping_count(), 2);
    /// ```
    pub fn add_mapping(&mut self, mapping: ComponentMapping) {
        self.mappings.push(mapping);
    }

    /// Returns the number of component mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Gets a reference to all mappings.
    pub fn mappings(&self) -> &[ComponentMapping] {
        &self.mappings
    }

    /// Gets the inlet flowrate variable.
    pub fn inlet_flow(&self) -> &Var<Algebraic> {
        &self.inlet_flow
    }

    /// Gets the outlet flowrate variable.
    pub fn outlet_flow(&self) -> &Var<Algebraic> {
        &self.outlet_flow
    }

    /// Gets the inlet temperature variable.
    pub fn inlet_temp(&self) -> &Var<Algebraic> {
        &self.inlet_temp
    }

    /// Gets the outlet temperature variable.
    pub fn outlet_temp(&self) -> &Var<Algebraic> {
        &self.outlet_temp
    }

    /// Computes the total number of input components.
    ///
    /// - Lumping: detailed components on input
    /// - Delumping: lumped components on input
    pub fn input_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Lumping => self.mappings.iter().map(|m| m.detailed_count()).sum(),
            MappingDirection::Delumping => self.mappings.len(),
        }
    }

    /// Computes the total number of output components.
    ///
    /// - Lumping: lumped components on output
    /// - Delumping: detailed components on output
    pub fn output_component_count(&self) -> usize {
        match self.direction {
            MappingDirection::Delumping => self.mappings.iter().map(|m| m.detailed_count()).sum(),
            MappingDirection::Lumping => self.mappings.len(),
        }
    }

    /// Validates all mappings (fractions should sum to 1.0).
    pub fn validate_mappings(&self) -> bool {
        self.mappings.iter().all(|m| m.validate_fractions())
    }
}

/// Port-based interface for MAP.
impl<T: TimeDomain> HasPorts for MAP<T> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::output("outlet", "MolarFlow")]
    }
}

/// UnitOp implementation for MAP.
impl<T: TimeDomain> UnitOp for MAP<T> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<TD: TimeDomain>(&self, system: &mut EquationSystem<TD>, unit_name: &str) {
        // Temperature passes through unchanged
        let mut temp_eq = ResidualFunction::new(&format!("{}_temperature", unit_name));
        temp_eq.add_term(EquationTerm::new(1.0, "T_out"));
        temp_eq.add_term(EquationTerm::new(-1.0, "T_in"));
        system.add_algebraic(temp_eq);

        match self.direction {
            MappingDirection::Lumping => {
                // For each mapping: F_lumped = sum(F_detailed_i)
                for mapping in &self.mappings {
                    let mut lumping_eq = ResidualFunction::new(&format!(
                        "{}_lump_{}",
                        unit_name,
                        mapping.lumped_component()
                    ));
                    // F_lumped
                    lumping_eq.add_term(EquationTerm::new(
                        1.0,
                        &format!("F_out_{}", mapping.lumped_component()),
                    ));
                    // - sum(F_detailed_i)
                    for (detailed_name, _) in mapping.detailed_components() {
                        lumping_eq
                            .add_term(EquationTerm::new(-1.0, &format!("F_in_{}", detailed_name)));
                    }

                    system.add_algebraic(lumping_eq);
                }
            }
            MappingDirection::Delumping => {
                // For each detailed component: F_detailed_i = fraction_i * F_lumped
                for mapping in &self.mappings {
                    for (detailed_name, fraction) in mapping.detailed_components() {
                        let mut delumping_eq = ResidualFunction::new(&format!(
                            "{}_delump_{}_to_{}",
                            unit_name,
                            mapping.lumped_component(),
                            detailed_name
                        ));
                        // F_detailed_i
                        delumping_eq
                            .add_term(EquationTerm::new(1.0, &format!("F_out_{}", detailed_name)));
                        // - fraction_i * F_lumped
                        delumping_eq.add_term(EquationTerm::new(
                            -fraction,
                            &format!("F_in_{}", mapping.lumped_component()),
                        ));

                        system.add_algebraic(delumping_eq);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dynamic, Steady};

    #[test]
    fn test_map_creation() {
        let mapper = MAP::<Steady>::new("TestMapper");
        assert_eq!(mapper.name(), "TestMapper");
        assert_eq!(mapper.mapping_count(), 0);
        assert_eq!(mapper.direction(), MappingDirection::Lumping);
    }

    #[test]
    fn test_map_direction() {
        let mut mapper = MAP::<Dynamic>::new("mapper");
        mapper.set_direction(MappingDirection::Delumping);
        assert_eq!(mapper.direction(), MappingDirection::Delumping);
    }

    #[test]
    fn test_map_add_mapping() {
        let mut mapper = MAP::<Steady>::new("mapper");
        let mapping = ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]);
        mapper.add_mapping(mapping);

        assert_eq!(mapper.mapping_count(), 1);
        assert_eq!(mapper.mappings()[0].lumped_component(), "MAPD");
        assert_eq!(mapper.mappings()[0].detailed_count(), 2);
    }

    #[test]
    fn test_map_component_counts_lumping() {
        let mut mapper = MAP::<Steady>::new("lumper");
        mapper.set_direction(MappingDirection::Lumping);

        // Add mapping: MAPD ← MA + PD
        mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));

        assert_eq!(mapper.input_component_count(), 2); // MA, PD
        assert_eq!(mapper.output_component_count(), 1); // MAPD
    }

    #[test]
    fn test_map_component_counts_delumping() {
        let mut mapper = MAP::<Steady>::new("delumper");
        mapper.set_direction(MappingDirection::Delumping);

        // Add mapping: MAPD -> MA + PD
        mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));

        assert_eq!(mapper.input_component_count(), 1); // MAPD
        assert_eq!(mapper.output_component_count(), 2); // MA, PD
    }

    #[test]
    fn test_map_multiple_mappings() {
        let mut mapper = MAP::<Steady>::new("multi_mapper");
        mapper.set_direction(MappingDirection::Lumping);

        mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));
        mapper
            .add_mapping(ComponentMapping::new("C4", vec![("n-butane", 0.5), ("iso-butane", 0.5)]));

        assert_eq!(mapper.mapping_count(), 2);
        assert_eq!(mapper.input_component_count(), 4); // MA, PD, n-butane, iso-butane
        assert_eq!(mapper.output_component_count(), 2); // MAPD, C4
    }

    #[test]
    fn test_map_validate_mappings() {
        let mut mapper = MAP::<Steady>::new("validator");

        // Valid mapping
        mapper.add_mapping(ComponentMapping::new("Valid", vec![("A", 0.3), ("B", 0.7)]));
        assert!(mapper.validate_mappings());

        // Add invalid mapping
        mapper.add_mapping(ComponentMapping::new("Invalid", vec![("C", 0.3), ("D", 0.6)]));
        assert!(!mapper.validate_mappings());
    }

    #[test]
    fn test_map_equations_lumping() {
        let mut mapper = MAP::<Steady>::new("lumper");
        mapper.set_direction(MappingDirection::Lumping);
        mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));

        let mut system = EquationSystem::<Steady>::new();
        mapper.build_equations(&mut system, "MAP-101");

        // 1 temperature equation + 1 lumping equation
        assert_eq!(system.algebraic_count(), 2);
        assert_eq!(system.differential_count(), 0);
    }

    #[test]
    fn test_map_equations_delumping() {
        let mut mapper = MAP::<Dynamic>::new("delumper");
        mapper.set_direction(MappingDirection::Delumping);
        mapper.add_mapping(ComponentMapping::new("MAPD", vec![("MA", 0.6), ("PD", 0.4)]));

        let mut system = EquationSystem::<Dynamic>::new();
        mapper.build_equations(&mut system, "MAP-102");

        // 1 temperature equation + 2 delumping equations (one per detailed component)
        assert_eq!(system.algebraic_count(), 3);
        assert_eq!(system.differential_count(), 0);
    }

    #[test]
    fn test_map_ports() {
        let mapper = MAP::<Steady>::new("mapper");
        let inputs = mapper.input_ports();
        let outputs = mapper.output_ports();

        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);
    }
}
