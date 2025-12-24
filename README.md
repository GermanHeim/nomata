# Nomata: Typed Process Modeling for Chemical Engineering

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

- **`autodiff`**: Automatic differentiation for Jacobian computation using `num-dual`
- **`solvers`**: Numerical solvers for DAE systems using `differential-equations`
- **`thermodynamics`**: CoolProp bindings for thermodynamic properties
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

Build complete residual functions for DAE systems:

```rust
use nomata::{ResidualFunction, EquationTerm};

let mut residual = ResidualFunction::new("mass_balance");

// Add terms: dN/dt = F_in - F_out + generation
residual.add_term(EquationTerm::new(1.0, "accumulation"));
residual.add_term(EquationTerm::new(-1.0, "flow_in"));
residual.add_term(EquationTerm::new(1.0, "flow_out"));
residual.add_term(EquationTerm::new(-1.0, "generation"));

// Evaluate residual
let result = residual.evaluate(&[dhdt, flow_in, flow_out, generation]);
```

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

### Unit Operations and Equation Harvesting

Unit operations know their own physics and can automatically generate equations:

```rust
use nomata::{UnitOp, Flowsheet, Dynamic, models::CSTR, Stream, MolarFlow};

// Define a unit operation trait
trait UnitOp {
    type In;
    type Out;
    
    // Unit builds its own equations
    fn build_equations<T: TimeDomain>(&self, system: &mut EquationSystem<T>, name: &str) {
        // Add mass balance, energy balance, kinetics, etc.
    }
}

// Generic simulation loop
let mut flowsheet = Flowsheet::<Dynamic>::new();
let reactor = CSTR::new(100.0, 1.0, 350.0);

flowsheet.add_unit("CSTR-101");
flowsheet.harvest_equations("CSTR-101", &reactor); // Automatic!

// Equations are now ready for solving
assert_eq!(flowsheet.equations().total_equations(), 5);
```

### Connections Are Type-Checked

```rust
use nomata::{connect, Stream, MolarFlow, MassFlow};

// Connections are validated at compile time
connect(&reactor, &separator); // Only compiles if types match
```

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
