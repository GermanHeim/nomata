//! Flash Separator model.
//!
//! Flash separator for vapor-liquid equilibrium.
//!
//! # State Variables
//! - Liquid holdup
//! - Vapor fraction
//! - Temperature
//! - Pressure
//!
//! # Equations
//! - Mass balance: dM/dt = F_in - L - V
//! - Component balances: x_i, y_i from equilibrium
//! - Energy balance
//! - VLE relations: y_i = K_i * x_i
//!
//! # Example
//!
//! ```
//! use nomata::models::FlashSeparator;
//!
//! // Type-safe initialization
//! let flash = FlashSeparator::new(10.0, 2)  // volume, n_components
//!     .with_k_values(vec![3.0, 0.5])
//!     .with_initial_state(350.0, 101325.0);  // T, P
//!
//! // Only fully initialized separators can perform flash calculations
//! // flash.flash_calculation();  // ✓ Compiles
//! ```

use crate::*;
use std::marker::PhantomData;

#[cfg(feature = "thermodynamics")]
use crate::thermodynamics::fluids::Pure;

/// Phantom type marker for uninitialized state.
pub struct Uninitialized;

/// Phantom type marker for initialized state.
pub struct Initialized;

/// Flash separator for vapor-liquid equilibrium.
///
/// Type parameters enforce compile-time initialization:
/// - `E`: Equilibrium data state (Uninitialized | Initialized)  
/// - `S`: State initialization (Uninitialized | Initialized)
pub struct FlashSeparator<E = Uninitialized, S = Uninitialized> {
    // State variables
    pub holdup: Var<Differential>,
    pub temperature: Var<Differential>,
    pub pressure: Var<Algebraic>,
    pub vapor_fraction: Var<Algebraic>,

    // Parameters
    volume: Var<Parameter>,

    // Equilibrium constants (Some when E = Initialized)
    k_values: Option<Vec<Var<Parameter>>>,

    // Component compositions
    pub n_components: usize,
    feed_composition: Vec<f64>,
    pub liquid_composition: Vec<Var<Algebraic>>,
    pub vapor_composition: Vec<Var<Algebraic>>,

    // Ports
    pub inlet: Port<Stream<MolarFlow>, Input, Disconnected>,
    pub vapor_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,
    pub liquid_outlet: Port<Stream<MolarFlow>, Output, Disconnected>,

    #[cfg(feature = "thermodynamics")]
    pub component_names: Vec<String>,

    // Phantom data for compile-time state tracking
    _equilibrium_state: PhantomData<E>,
    _state_init: PhantomData<S>,
}

// Initial constructor
impl FlashSeparator<Uninitialized, Uninitialized> {
    /// Creates a new flash separator with vessel volume and number of components.
    ///
    /// K-values and initial state must be set before flash calculations.
    pub fn new(volume: f64, n_components: usize) -> Self {
        FlashSeparator {
            holdup: Var::new(volume * 0.5),
            temperature: Var::new(298.15),
            pressure: Var::new(101325.0),
            vapor_fraction: Var::new(0.5),

            volume: Var::new(volume),

            k_values: None,

            n_components,
            feed_composition: vec![1.0 / n_components as f64; n_components],
            liquid_composition: (0..n_components).map(|_| Var::new(0.0)).collect(),
            vapor_composition: (0..n_components).map(|_| Var::new(0.0)).collect(),

            inlet: Port::new(),
            vapor_outlet: Port::new(),
            liquid_outlet: Port::new(),

            #[cfg(feature = "thermodynamics")]
            component_names: Vec::new(),

            _equilibrium_state: PhantomData,
            _state_init: PhantomData,
        }
    }
}

// Equilibrium initialization (E: Uninitialized → Initialized)
impl<S> FlashSeparator<Uninitialized, S> {
    /// Sets equilibrium K-values manually, transitioning to equilibrium-initialized state.
    pub fn with_k_values(self, k_vals: Vec<f64>) -> FlashSeparator<Initialized, S> {
        assert_eq!(
            k_vals.len(),
            self.n_components,
            "K-values length {} must match n_components {}",
            k_vals.len(),
            self.n_components
        );

        FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: Some(k_vals.into_iter().map(Var::new).collect()),

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            #[cfg(feature = "thermodynamics")]
            component_names: self.component_names,

            _equilibrium_state: PhantomData,
            _state_init: self._state_init,
        }
    }

    #[cfg(feature = "thermodynamics")]
    /// Computes K-values from thermodynamic properties, transitioning to equilibrium-initialized state.
    pub fn with_k_values_from_thermo(
        self,
        components: &[Pure],
        temp: f64,
        pressure: f64,
    ) -> Result<FlashSeparator<Initialized, S>, crate::thermodynamics::ThermoError> {
        assert_eq!(components.len(), self.n_components);

        let mut k_vals = Vec::new();

        for pure in components {
            let fluid = crate::thermodynamics::Fluid::new(*pure);
            let t_crit = fluid.critical_temperature()?;

            if temp >= t_crit {
                return Err(crate::thermodynamics::ThermoError::InvalidInput(format!(
                    "Temperature {} K exceeds critical temperature {} K for {:?}",
                    temp, t_crit, pure
                )));
            }

            let p_sat = Self::compute_vapor_pressure_rfluids(&fluid, temp)?;
            let k = p_sat / pressure;
            k_vals.push(k);
        }

        Ok(FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: Some(k_vals.into_iter().map(Var::new).collect()),

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            component_names: components.iter().map(|p| format!("{:?}", p)).collect(),

            _equilibrium_state: PhantomData,
            _state_init: self._state_init,
        })
    }

    #[cfg(feature = "thermodynamics")]
    fn compute_vapor_pressure_rfluids(
        fluid: &crate::thermodynamics::Fluid,
        temperature: f64,
    ) -> Result<f64, crate::thermodynamics::ThermoError> {
        use rfluids::prelude::*;

        let rfluids_fluid = match &fluid.substance {
            crate::thermodynamics::Substance::Pure(p) => rfluids::fluid::Fluid::from(*p),
            crate::thermodynamics::Substance::PredefinedMix(m) => rfluids::fluid::Fluid::from(*m),
            _ => {
                return Err(crate::thermodynamics::ThermoError::InvalidInput(
                    "Only pure fluids and predefined mixtures supported for VLE".to_string(),
                ));
            }
        };

        let mut state = rfluids_fluid
            .in_state(FluidInput::temperature(temperature), FluidInput::quality(0.5))
            .map_err(|e| {
                crate::thermodynamics::ThermoError::InvalidInput(format!(
                    "Failed to compute saturation state at {} K: {:?}",
                    temperature, e
                ))
            })?;

        let p_sat = state
            .pressure()
            .map_err(|_| crate::thermodynamics::ThermoError::PropertyNotAvailable)?;

        Ok(p_sat)
    }
}

// State initialization (S: Uninitialized → Initialized)
impl<E> FlashSeparator<E, Uninitialized> {
    /// Sets initial temperature and pressure, transitioning to state-initialized.
    pub fn with_initial_state(
        mut self,
        temp: f64,
        pressure: f64,
    ) -> FlashSeparator<E, Initialized> {
        self.temperature = Var::new(temp);
        self.pressure = Var::new(pressure);

        FlashSeparator {
            holdup: self.holdup,
            temperature: self.temperature,
            pressure: self.pressure,
            vapor_fraction: self.vapor_fraction,
            volume: self.volume,

            k_values: self.k_values,

            n_components: self.n_components,
            feed_composition: self.feed_composition,
            liquid_composition: self.liquid_composition,
            vapor_composition: self.vapor_composition,

            inlet: self.inlet,
            vapor_outlet: self.vapor_outlet,
            liquid_outlet: self.liquid_outlet,

            #[cfg(feature = "thermodynamics")]
            component_names: self.component_names,

            _equilibrium_state: self._equilibrium_state,
            _state_init: PhantomData,
        }
    }
}

// Methods available at any state
impl<E, S> FlashSeparator<E, S> {
    /// Sets feed composition (must sum to 1.0).
    pub fn set_feed_composition(&mut self, composition: Vec<f64>) {
        assert_eq!(composition.len(), self.n_components);
        self.feed_composition = composition;
    }

    /// Creates mass balance equation.
    pub fn mass_balance(&self) -> Equation<MassBalance> {
        self.holdup.mass_balance("flash_mass")
    }

    /// Creates energy balance equation.
    pub fn energy_balance(&self) -> Equation<EnergyBalance> {
        self.temperature.energy_balance("flash_energy")
    }
}

// Flash calculation (only available when FULLY initialized)
impl FlashSeparator<Initialized, Initialized> {
    /// Performs Rachford-Rice flash calculation.
    ///
    /// Only available for fully initialized flash separators.
    pub fn flash_calculation(&mut self) {
        let _temp = self.temperature.get();
        let _pressure = self.pressure.get();

        let mut vapor_frac = 0.5; // Initial guess

        // Rachford-Rice iteration (simplified)
        for _ in 0..10 {
            let mut sum = 0.0;
            for i in 0..self.n_components {
                let k = self.k_values.as_ref().unwrap()[i].get();
                let z = self.feed_composition[i];
                sum += z * (k - 1.0) / (1.0 + vapor_frac * (k - 1.0));
            }

            if sum.abs() < 1e-6 {
                break;
            }

            // Newton step
            let mut deriv = 0.0;
            for i in 0..self.n_components {
                let k = self.k_values.as_ref().unwrap()[i].get();
                let z = self.feed_composition[i];
                let denom = 1.0 + vapor_frac * (k - 1.0);
                deriv -= z * (k - 1.0).powi(2) / (denom * denom);
            }

            vapor_frac -= sum / deriv;
            vapor_frac = vapor_frac.clamp(0.0, 1.0);
        }

        self.vapor_fraction = Var::new(vapor_frac);

        // Compute compositions
        for i in 0..self.n_components {
            let k = self.k_values.as_ref().unwrap()[i].get();
            let z = self.feed_composition[i];

            let x = z / (1.0 + vapor_frac * (k - 1.0));
            let y = k * x;

            self.liquid_composition[i] = Var::new(x);
            self.vapor_composition[i] = Var::new(y);
        }
    }

    /// Gets K-value for component i (guaranteed to exist).
    pub fn k_value(&self, i: usize) -> f64 {
        self.k_values.as_ref().unwrap()[i].get()
    }

    /// Gets all K-values.
    pub fn k_values(&self) -> Vec<f64> {
        self.k_values.as_ref().unwrap().iter().map(|k| k.get()).collect()
    }
}

/// Port-based interface for FlashSeparator.
impl<E, S> HasPorts for FlashSeparator<E, S> {
    fn input_ports(&self) -> Vec<NamedPort> {
        vec![NamedPort::input("inlet", "MolarFlow")]
    }

    fn output_ports(&self) -> Vec<NamedPort> {
        vec![
            NamedPort::output("vapor_outlet", "MolarFlow"),
            NamedPort::output("liquid_outlet", "MolarFlow"),
        ]
    }
}

/// UnitOp implementation for FlashSeparator.
impl<E, S> UnitOp for FlashSeparator<E, S> {
    type In = Stream<MolarFlow>;
    type Out = Stream<MolarFlow>;

    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, unit_name: &str) {
        if T::IS_STEADY {
            // Steady-state flash: no accumulation
            // Overall mass balance: 0 = F_in - V - L
            let mut mass_balance = ResidualFunction::new(&format!("{}_mass_balance", unit_name));
            mass_balance.add_term(EquationTerm::new(1.0, "F_in"));
            mass_balance.add_term(EquationTerm::new(-1.0, "V"));
            mass_balance.add_term(EquationTerm::new(-1.0, "L"));
            system.add_algebraic(mass_balance);

            // Component balances: 0 = F_in*z_i - V*y_i - L*x_i
            for i in 0..self.n_components {
                let mut comp_balance =
                    ResidualFunction::new(&format!("{}_component_{}_balance", unit_name, i));
                comp_balance.add_term(EquationTerm::new(1.0, &format!("F_in_z_{}", i)));
                comp_balance.add_term(EquationTerm::new(-1.0, &format!("V_y_{}", i)));
                comp_balance.add_term(EquationTerm::new(-1.0, &format!("L_x_{}", i)));
                system.add_algebraic(comp_balance);
            }

            // Energy balance: 0 = F_in*H_in - V*H_v - L*H_l
            let mut energy_balance =
                ResidualFunction::new(&format!("{}_energy_balance", unit_name));
            energy_balance.add_term(EquationTerm::new(1.0, "F_in_H_in"));
            energy_balance.add_term(EquationTerm::new(-1.0, "V_H_v"));
            energy_balance.add_term(EquationTerm::new(-1.0, "L_H_l"));
            system.add_algebraic(energy_balance);
        } else {
            // Dynamic flash: include accumulation terms
            // Overall mass balance: dM/dt = F_in - V - L
            let mut mass_balance = ResidualFunction::new(&format!("{}_mass_balance", unit_name));
            mass_balance.add_term(EquationTerm::new(1.0, "dM_dt"));
            mass_balance.add_term(EquationTerm::new(-1.0, "F_in"));
            mass_balance.add_term(EquationTerm::new(1.0, "V"));
            mass_balance.add_term(EquationTerm::new(1.0, "L"));
            system.add_differential(mass_balance);

            // Component balances: d(M*z_i)/dt = F_in*z_i - V*y_i - L*x_i
            for i in 0..self.n_components {
                let mut comp_balance =
                    ResidualFunction::new(&format!("{}_component_{}_balance", unit_name, i));
                comp_balance.add_term(EquationTerm::new(1.0, &format!("d_Mz_{}_dt", i)));
                comp_balance.add_term(EquationTerm::new(-1.0, &format!("F_in_z_{}", i)));
                comp_balance.add_term(EquationTerm::new(1.0, &format!("V_y_{}", i)));
                comp_balance.add_term(EquationTerm::new(1.0, &format!("L_x_{}", i)));
                system.add_differential(comp_balance);
            }

            // Energy balance: d(M*H)/dt = F_in*H_in - V*H_v - L*H_l
            let mut energy_balance =
                ResidualFunction::new(&format!("{}_energy_balance", unit_name));
            energy_balance.add_term(EquationTerm::new(1.0, "d_MH_dt"));
            energy_balance.add_term(EquationTerm::new(-1.0, "F_in_H_in"));
            energy_balance.add_term(EquationTerm::new(1.0, "V_H_v"));
            energy_balance.add_term(EquationTerm::new(1.0, "L_H_l"));
            system.add_differential(energy_balance);
        }

        // VLE equilibrium relations: y_i = K_i * x_i
        for i in 0..self.n_components {
            let mut vle = ResidualFunction::new(&format!("{}_vle_{}", unit_name, i));
            vle.add_term(EquationTerm::new(1.0, &format!("y_{}", i)));
            vle.add_term(EquationTerm::new(-1.0, &format!("K_{}_x_{}", i, i)));
            system.add_algebraic(vle);
        }

        // Rachford-Rice equation: sum(z_i*(K_i - 1)/(1 + V_frac*(K_i - 1))) = 0
        let mut rachford_rice = ResidualFunction::new(&format!("{}_rachford_rice", unit_name));
        for i in 0..self.n_components {
            rachford_rice.add_term(EquationTerm::new(1.0, &format!("z_{}_K_term_{}", i, i)));
        }
        system.add_algebraic(rachford_rice);

        // Summation constraints
        let mut x_sum = ResidualFunction::new(&format!("{}_x_sum", unit_name));
        for i in 0..self.n_components {
            x_sum.add_term(EquationTerm::new(1.0, &format!("x_{}", i)));
        }
        x_sum.add_term(EquationTerm::new(-1.0, "one"));
        system.add_algebraic(x_sum);

        let mut y_sum = ResidualFunction::new(&format!("{}_y_sum", unit_name));
        for i in 0..self.n_components {
            y_sum.add_term(EquationTerm::new(1.0, &format!("y_{}", i)));
        }
        y_sum.add_term(EquationTerm::new(-1.0, "one"));
        system.add_algebraic(y_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_separator_creation() {
        let flash = FlashSeparator::new(10.0, 3);
        assert_eq!(flash.volume.get(), 10.0);
        assert!(flash.k_values.is_none());
        assert_eq!(flash.n_components, 3);
    }

    #[test]
    fn test_flash_phantom_type_transitions() {
        // Test equilibrium initialization
        let flash: FlashSeparator<Initialized, Initialized> = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);
        assert_eq!(flash.k_value(0), 3.0);
        assert_eq!(flash.k_value(1), 0.5);

        // Test state initialization
        let flash = FlashSeparator::new(10.0, 2).with_initial_state(350.0, 101325.0);
        assert_eq!(flash.temperature.get(), 350.0);
        assert_eq!(flash.pressure.get(), 101325.0);

        // Test full initialization
        let flash = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);
        assert_eq!(flash.k_value(0), 3.0);
        assert_eq!(flash.temperature.get(), 350.0);
    }

    #[test]
    fn test_flash_calculation() {
        let mut flash = FlashSeparator::new(10.0, 2)
            .with_k_values(vec![3.0, 0.5])
            .with_initial_state(350.0, 101325.0);

        flash.set_feed_composition(vec![0.5, 0.5]);
        flash.flash_calculation();

        // Light component should be enriched in vapor
        assert!(flash.vapor_composition[0].get() > flash.liquid_composition[0].get());
        // Heavy component should be enriched in liquid
        assert!(flash.liquid_composition[1].get() > flash.vapor_composition[1].get());
    }

    #[cfg(feature = "thermodynamics")]
    #[test]
    fn test_thermodynamic_initialization() {
        use crate::thermodynamics::fluids::Pure;

        let result = FlashSeparator::new(10.0, 1).with_k_values_from_thermo(
            &[Pure::Water],
            298.15,
            101325.0,
        );

        assert!(result.is_ok());
        let flash = result.unwrap().with_initial_state(298.15, 101325.0);

        let k_water = flash.k_value(0);
        // Water at 25°C, 1 atm: K ≈ 0.031
        assert!(k_water > 0.025 && k_water < 0.04, "K_water = {}", k_water);
    }
}
