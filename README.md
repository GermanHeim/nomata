# Nomata: Typed Process Modeling for Chemical Engineering

<p align="center">
    <img
        width="500"
        src="https://raw.githubusercontent.com/GermanHeim/nomata/main/media/logo.png"
        alt="Nomata logo"
    />
</p>

A correct-by-construction process modeling framework that leverages Rust's type system to enforce structural correctness at compile time.

Nomata is an embedded domain-specific language (EDSL) for chemical process modeling that prevents invalid process models from being representable. Instead of catching modeling errors at runtime, Nomata makes illegal flowsheets fail to compile.

## Key Features

- **Compile-time correctness**: Invalid process models cannot be constructed
- **Type-safe connections**: Unit operations can only be connected if their stream types match
- **Variable role enforcement**: Algebraic and differential variables are distinguished at the type level
- **Time domain consistency**: Steady-state and dynamic models cannot be mixed incorrectly
- **Zero runtime overhead**: All validation happens at compile time

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
nomata = "0.1.0"
```

### Optional Features

Nomata provides optional features for extended functionality:

```toml
[dependencies]
nomata = { version = "0.1.0", features = ["autodiff", "solvers", "thermodynamics"] }
```

Available features:

- **`autodiff`**: Automatic differentiation for Jacobian computation using [`num-dual`](https://docs.rs/num-dual/latest/num_dual/)
- **`solvers`**: Numerical solvers for DAE systems using [`differential-equations`](https://docs.rs/differential-equations/latest/differential_equations/)
- **`thermodynamics`**: CoolProp bindings for thermodynamic properties using [`rfluids`](https://docs.rs/rfluids/latest/rfluids/)
- **`all`**: Enable all features

## Core Concepts

### Variable Roles

Variables are typed by their role in the model, enforcing proper usage:

- **Parameters**: Time-invariant constants (rate constants, physical properties)
- **Algebraic**: Instantaneous relationships without time derivatives
- **Differential**: State variables with accumulation and time derivatives

```rust
use nomata::{VariableRegistry, Var, Parameter, Algebraic, Differential};

// Create a registry to manage shared state
let registry = VariableRegistry::new();

// Parameters: fixed constants
let rate_constant: Var<Parameter> = registry.create_parameter(0.5);
let density: Var<Parameter> = registry.create_parameter(1000.0);

// Algebraic: computed values
let temperature: Var<Algebraic> = registry.create_algebraic(298.15);

// Differential: state variables
let holdup: Var<Differential> = registry.create_differential(100.0);

// This compiles - differential variables can have derivatives
let dhdt = holdup.derivative();

// This would NOT compile - algebraic variables cannot have derivatives
// let dtdt = temperature.derivative(); //  Compile error!

// Parameters also cannot have derivatives (not time-varying)
// let dkdt = rate_constant.derivative(); //  Compile error!
```

**Why distinguish parameters from algebraic variables?**

While both store scalar values, the semantic distinction enables:

- **Sensitivity analysis**: Automatic differentiation w.r.t. parameters
- **Parameter estimation**: Optimization workflows that adjust parameters
- **Model structure clarity**: Separates physics (structure) from values (parameters)
- **Future extensibility**: Different numerical treatments during solution

### Shared State Architecture

Nomata uses an **index-based variable system** to enable solver integration:

```rust
use nomata::{VariableRegistry, Var, Algebraic};

let registry = VariableRegistry::new();
let temp = registry.create_algebraic(298.15);

// Multiple references share the same state
let temp_ref = temp.clone();
temp.set(310.0);
assert_eq!(temp_ref.get(), 310.0);  //  State is shared!

// Solvers can access variables as vectors
let mut state = registry.get_all_values();
state[temp.id().index()] = 320.0;
registry.set_all_values(&state);
assert_eq!(temp.get(), 320.0);  //  Solver updates visible!
```

### Time Domains

Models are typed by their time behavior:

- **Steady-state**: No time derivatives allowed
- **Dynamic**: Time derivatives required for differential variables

```rust
use nomata::{Model, Steady, Dynamic};

let steady_model: Model<Steady> = Model::new();
let dynamic_model: Model<Dynamic> = Model::new();
```

### Streams

Process streams track composition, temperature, and pressure. The unified `Stream` type uses phantom types to enforce that conditions are explicitly initialized:

```rust
use nomata::{Stream, MolarFlow};

// Create multi-component stream with uninitialized conditions
let stream = Stream::<MolarFlow, _>::new(
    100.0,  // total_flow
    vec!["Water".to_string(), "Ethanol".to_string()],
)
.at_conditions(298.15, 101325.0); // Must set T and P!

// Or use with_composition for specific mole fractions
let air = Stream::<MolarFlow, _>::with_composition(
    1000.0,
    vec!["N2", "O2", "Ar"],
    vec![0.78, 0.21, 0.01],
)
.expect("Composition fractions must sum to 1.0")
.at_conditions(298.15, 101325.0);

// Pure component streams
let pure_water = Stream::<MolarFlow, _>::pure(
    100.0,
    "Water".to_string(),
    298.15,  // Temperature required
    101325.0 // Pressure required
);

// Access stream properties
println!("Total flow: {} mol/s", stream.total_flow);
println!("Temperature: {} K", stream.temperature);
println!("Pressure: {} Pa", stream.pressure);
```

### Balance Equations

Balance equations are typed by conservation law:

```rust
use nomata::{Equation, MassBalance, EnergyBalance, Var, Differential};

// Create differential variables
let holdup: Var<Differential> = Var::new(100.0);
let energy: Var<Differential> = Var::new(5000.0);

// Create typed balance equations
let mass_balance: Equation<MassBalance> = Equation::new("reactor_mass");
let energy_balance: Equation<EnergyBalance> = Equation::new("reactor_energy");

// Set residuals (computed from balance terms)
mass_balance.set_residual(0.0);

// Check if satisfied
if mass_balance.is_satisfied(1e-6) {
    println!("Mass balance satisfied!");
}
```

### Residual Functions

Build type-safe residual functions for DAE systems using `EquationVars`:

```rust
use nomata::{ResidualFunction, EquationVars};
use std::collections::HashMap;

// Define typed variables for a mass balance equation
struct MassBalanceVars {
    accumulation: f64,
    flow_in: f64,
    flow_out: f64,
}

impl EquationVars for MassBalanceVars {
    fn base_names() -> &'static [&'static str] {
        &["accumulation", "flow_in", "flow_out"]
    }

    fn from_map(vars: &HashMap<String, f64>, prefix: &str) -> Option<Self> {
        Some(Self {
            accumulation: *vars.get(&format!("{}_accumulation", prefix))?,
            flow_in: *vars.get(&format!("{}_flow_in", prefix))?,
            flow_out: *vars.get(&format!("{}_flow_out", prefix))?,
        })
    }
}

// Create type-safe residual: dN/dt = F_in - F_out
let residual = ResidualFunction::from_typed(
    "mass_balance",
    "reactor",
    |v: MassBalanceVars| v.accumulation - v.flow_in + v.flow_out,
);

// For dynamic variable counts (e.g., N-inlet mixers), use from_dynamic:
let var_names = vec!["inlet_0_F".into(), "inlet_1_F".into(), "outlet_F".into()];
let mixer_balance = ResidualFunction::from_dynamic(
    "mixer_mass_balance",
    var_names,
    |vars, names| {
        let outlet = vars.get(&names[names.len()-1]).copied().unwrap_or(0.0);
        let inlet_sum: f64 = names[..names.len()-1]
            .iter()
            .filter_map(|n| vars.get(n))
            .sum();
        outlet - inlet_sum
    },
);
```

### Automatic Differentiation

With the `autodiff` feature enabled, all unit operation models support automatic differentiation for efficient Jacobian computation:

```rust
use nomata::{Scalar, EquationVarsGeneric};
use std::collections::HashMap;

// Define equation variables generic over scalar type
pub struct ArrheniusVars<S: Scalar> {
    pub k: S,
    pub k0: S,
    pub ea: S,
    pub t: S,
}

impl<S: Scalar> EquationVarsGeneric<S> for ArrheniusVars<S> {
    fn base_names() -> &'static [&'static str] {
        &["k", "k0", "Ea", "T"]
    }

    fn from_map(vars: &HashMap<String, S>, prefix: &str) -> Option<Self> {
        Some(Self {
            k: vars.get(&format!("{}_k", prefix))?.clone(),
            k0: vars.get(&format!("{}_k0", prefix))?.clone(),
            ea: vars.get(&format!("{}_Ea", prefix))?.clone(),
            t: vars.get(&format!("{}_T", prefix))?.clone(),
        })
    }
}

// The Scalar trait provides exp, ln, sqrt, powf, abs, sin, cos
// that work with both f64 and Dual64 for automatic differentiation
```

The `Scalar` trait enables equations like the Arrhenius equation (`k = k0 * exp(-Ea/RT)`) to be automatically differentiated, providing exact Jacobians for Newton-based solvers.

### Complete DAE Systems

Represent full differential-algebraic equation systems:

```rust
use nomata::{EquationSystem, Dynamic};

let mut system: EquationSystem<Dynamic> = EquationSystem::new();

// Add differential equations (balance equations)
system.add_differential(mass_balance_residual);
system.add_differential(energy_balance_residual);

// Add algebraic equations (constitutive relations)
system.add_algebraic(concentration_definition);
system.add_algebraic(temperature_definition);

println!("Total equations: {}", system.total_equations());
println!("Differential: {}", system.differential_count());
println!("Algebraic: {}", system.algebraic_count());
```

### Equation Harvesting

Unit operations know their own physics and can automatically generate equations:

```rust
use nomata::{UnitOp, Flowsheet, Dynamic, models::CSTR, Stream, MolarFlow};

// Generic simulation loop
let mut flowsheet = Flowsheet::<Dynamic>::new();
let reactor = CSTR::new(100.0, 1.0, 350.0);

flowsheet.add_unit("CSTR-101", reactor);
flowsheet.harvest_equations(); // Automatic!

// Equations are now ready for solving
assert_eq!(flowsheet.equations().total_equations(), 5);
```

### Connections Are Type-Checked

Nomata provides a unified, type-safe connection system that enforces stream compatibility at both compile time and runtime.

#### Port Specification

Unit operations declare their ports using the `PortSpec` trait, which specifies port counts and stream types at compile time:

```rust
use nomata::{PortSpec, Stream, MolarFlow, MassFlow};

// Single-port units (1 input, 1 output)
impl PortSpec for Reactor {
    const INPUT_COUNT: usize = 1;
    const OUTPUT_COUNT: usize = 1;
    type STREAM_TYPE = MolarFlow;
}

// Multi-port units (N inputs, 1 output)
impl PortSpec for Mixer {
    const INPUT_COUNT: usize = 3;  // Can mix 3 inlet streams
    const OUTPUT_COUNT: usize = 1;
    type STREAM_TYPE = MassFlow;
}
```

#### Runtime Port Discovery

The `HasPorts` trait provides runtime access to port information:

```rust
use nomata::HasPorts;

let reactor = Reactor::new(/* ... */);
println!("Reactor has {} inputs, {} outputs",
    reactor.input_ports().len(),
    reactor.output_ports().len()
);
```

#### Type-Safe Connections

The unified `connect()` function performs compile-time bounds checking and runtime stream type validation:

```rust
use nomata::connect;

// Compile-time: Checks that reactor has exactly 1 output and separator has exactly 1 input
// Runtime: Validates that both units use compatible stream types
connect(&reactor, &separator)?; // Returns Result<(), ConnectionError>

// Multi-port connections use port indices
connect_at(&mixer, 0, &reactor, 0)?; // Connect mixer's output 0 to reactor's input 0
connect_at(&splitter, 0, &separator, 1)?; // Connect splitter's output 0 to separator's input 1
```

Invalid connections are caught at compile time:

```rust,compile_fail
// ERROR: Reactor has only 1 output, cannot connect output index 2
connect_at(&reactor, 2, &separator, 0);
```

The connection system ensures that:
- **Port bounds are checked at compile time** - Invalid port indices won't compile
- **Stream types are validated at runtime** - Incompatible streams return a `ConnectionError`
- **Zero runtime overhead** - All validation logic is optimized away when successful

## Building from Source

1. Install Rust toolchain using [rustup](https://rustup.rs/).
2. Clone repository:

   ```bash
   git clone https://github.com/GermanHeim/nomata.git
   cd nomata
   ```

3. Build the project:

   ```bash
   cargo build --release
   ```

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/GermanHeim/nomata/blob/main/LICENSE.txt) for more information.
