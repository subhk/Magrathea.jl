# Examples

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Ready-to-run scripts</div>
  <h1>Learn Magrathea.jl by running complete examples.</h1>
  <p>
    Each script in the <code>example/</code> directory is a self-contained stability
    calculation you can run and adapt to your own problem.
  </p>
  <div class="magrathea-actions">
    <a class="magrathea-button primary" href="../getting_started/">Get started</a>
    <a class="magrathea-button secondary" href="../analysis/onset_convection/">Analysis modes</a>
  </div>
</div>

## Running Examples

All examples should be run from the repository root with the project environment:

```bash
julia --project=. example/<script_name>.jl
```

Or from the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
include("example/<script_name>.jl")
```

---

## Linear Stability Demo

**File:** `example/linear_stability_demo.jl`

**Purpose:** Basic demonstration of the linear stability solver.

**What it does:**
- Loops over azimuthal wavenumbers ``m = 1, \ldots, 20``
- Computes leading eigenvalues at fixed Rayleigh number
- Displays growth rates and frequencies

**Key concepts:**
- `OnsetParams` configuration
- `solve(OnsetProblem(...))` workflow
- Eigenvalue interpretation

```julia
params = OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10, lmax=60, Nr=64)
result = solve(OnsetProblem(params); nev=8)
result.growth_rate  # replaces real(eigenvalues[1])
```

```julia
# Sample output
m    Re(λ₁)          Im(λ₁)          iterations
------------------------------------------------
 1  -1.23456e-02   5.67890e-01      24
 2  -8.76543e-03   6.12345e-01      28
...
```

**When to use:** First introduction to Magrathea.jl, verifying installation.

---

## Critical Rayleigh Number Scan

**File:** `example/Rac_lm.jl`

**Purpose:** Find critical Rayleigh numbers across azimuthal modes.

**What it does:**
- Scans ``m`` values to find ``Ra_c(m)``
- Uses bisection to find where growth rate = 0
- Identifies the globally most unstable mode

**Key concepts:**
- `find_critical_rayleigh` function
- Parameter sweeps
- Critical mode identification

**Physical insight:** The critical Rayleigh number ``Ra_c`` varies with ``m``, and the minimum determines the first mode to become unstable.

**v2.0 equivalent:**
```julia
params = OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10, lmax=60, Nr=64)
result = solve(OnsetProblem(params); nev=6)
result.growth_rate  # positive = unstable, negative = stable
```

---

## Basic State Onset

**File:** `example/basic_state_onset_example.jl`

**Purpose:** Demonstrate custom basic state usage.

**What it does:**
- Creates a conduction basic state
- Optionally adds meridional temperature variation
- Computes stability with the modified background

**Key concepts:**
- `ChebyshevDiffn` construction
- `basic_state` function with symbolic BCs
- Passing basic state to `OnsetParams`

```julia
# Create basic state using symbolic BCs (recommended)
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Pure conduction
bs = basic_state(cd, χ, E, Ra, Pr)

# With meridional temperature variation (Y₂₀)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))

# With fixed flux at outer boundary
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=Y00(-1.0) + Y20(0.1))

# Use in a stability calculation
params = OnsetParams(..., basic_state=bs)
result = solve(BiglobalProblem(params, bs); nev=8)
```

---

## Boundary-Driven Jet

**File:** `example/boundary_driven_jet.jl`

**Purpose:** Study stability with differential boundary rotation or heating.

**What it does:**
- Creates a basic state with boundary-driven flows
- Computes thermal wind from temperature gradients
- Analyzes modified stability

**Key concepts:**
- Thermal wind balance
- Boundary-driven circulation
- Flow-convection interaction

---

## Non-Axisymmetric Basic State

**File:** `example/nonaxisymmetric_basic_state.jl`

**Purpose:** Create 3D basic states for tri-global analysis.

**What it does:**
- Defines boundary mode amplitudes
- Constructs `BasicState3D`
- Prepares for tri-global coupling

**Key concepts:**
- Symbolic spherical harmonic BCs (`Y20`, `Y22`, etc.)
- `basic_state` convenience function
- `nonaxisymmetric_basic_state` for dictionary syntax

```julia
# Using symbolic BCs (recommended)
bc = Y20(0.1) + Y22(0.05)
bs3d = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# With fixed flux BC
flux = Y00(-1.0) + Y20(0.1) + Y22(0.05)
bs3d = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)

# Alternatively, using dictionary syntax
boundary_modes = Dict(
    (2, 0) => 0.1,    # Axisymmetric Y₂₀
    (2, 2) => 0.05,   # Non-axisymmetric Y₂₂
)
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)
```

---

## Tri-Global Analysis Demo

**File:** `example/triglobal_analysis_demo.jl`

**Purpose:** Framework for mode-coupled stability problems.

**What it does:**
- Sets up `TriglobalParams`
- Builds coupled mode problem
- Estimates problem size
- (Optional) Solves eigenvalue problem

**Key concepts:**
- Mode coupling through non-axisymmetric basic states
- Block matrix structure
- Size estimation before solving

**Note:** Tri-global problems can be very large. Start with small `m_range` and `lmax`.

**v2.0 equivalent:**
```julia
params = OnsetParams(E=1e-5, Pr=1.0, Ra=1.5e7, χ=0.35, lmax=40, Nr=48)
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)
problem = TriglobalProblem(params, bs3d, -2:2)
estimate_size(problem)   # check memory before committing
result = solve(problem; nev=8)
```

---

## MHD Dynamo Example

**File:** `example/mhd_dynamo_example.jl`

**Purpose:** Complete MHD stability analysis workflow.

**What it does:**
- Defines `MHDParams` with magnetic field
- Builds `MHDStabilityOperator`
- Assembles matrices
- Solves eigenvalue problem
- Interprets results

**Key concepts:**
- Lehnert number and magnetic Prandtl number
- Background field types (axial, dipole)
- Magnetic boundary conditions

```julia
params = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 5.0,
    Ra = 1e4, Le = 0.1,           # Weak magnetic field
    ricb = 0.35, m = 2, lmax = 10, N = 16,
    B0_type = axial,
    bci = 1, bco = 1,             # No-slip
    bci_magnetic = 0, bco_magnetic = 0,  # Insulating
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, _ = assemble_mhd_matrices(op)
```

---

## Thermal Wind Test

**File:** `example/test_thermal_wind.jl`

**Purpose:** Verify thermal wind balance implementation.

**What it does:**
- Creates temperature field with latitudinal variation
- Computes thermal wind from geostrophic balance
- Verifies consistency

**Key concepts:**
- Thermal wind equation
- Geostrophic balance
- Verification against analytic solutions

---

## Figure 2 Benchmark

**File:** `example/figure2_benchmark.jl`

**Purpose:** Reproduce published benchmark results.

**What it does:**
- Replicates parameters from Figure 2 of reference paper
- Computes critical curves
- Compares against published values

**Key concepts:**
- Benchmark validation
- Parameter matching
- Quantitative verification

---

## Heat Flux Boundary Condition Examples

### Non-Axisymmetric Flux: Y₂₂ Pattern

**File:** `example/flux_bc_mean_flow.jl`

**Purpose:** Demonstrate self-consistent basic state with non-axisymmetric heat flux boundary conditions.

**What it does:**
- Applies Y₂₂ heat flux pattern at outer boundary (sectoral cooling)
- Computes full geostrophic balance including meridional circulation
- Shows mode coupling through the toroidal-poloidal decomposition
- Compares zonal vs meridional flow components

**Key concepts:**
- `basic_state_selfconsistent` function
- Non-axisymmetric forcing (m ≠ 0)
- Meridional circulation from φ-gradient of temperature
- Block-tridiagonal mode coupling

```julia
# Non-axisymmetric flux pattern
flux = Y00(-1.0) + Y22(-0.2)
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr; flux_bc=flux, verbose=true)

# Access all velocity components
println("Zonal: ", maximum(abs, bs.uphi_coeffs[(3,2)]))
println("Meridional: ", maximum(abs, bs.utheta_coeffs[(2,2)]))
println("Radial: ", maximum(abs, bs.ur_coeffs[(3,2)]))
```

**Physical insight:** The Y₂₂ pattern drives:
- Zonal flow via ∂T/∂θ coupling
- Meridional circulation via ∂T/∂φ (non-zero for m=2)
- Mode coupling cascade: ℓ=2,3,4,... all participate

---

### Axisymmetric Flux: Y₂₀ Pattern

**File:** `example/flux_bc_axisymmetric_flow.jl`

**Purpose:** Demonstrate basic state with axisymmetric heat flux and compare to non-axisymmetric case.

**What it does:**
- Applies Y₂₀ heat flux pattern (latitudinal cooling variation)
- Shows that meridional circulation is exactly zero for m=0
- Computes zonal jets from thermal wind balance
- Highlights differences from Y₂₂ case

**Key concepts:**
- Axisymmetric forcing (m = 0 only)
- No iteration needed (advection term is zero)
- Zonal flow modes Y₁₀, Y₃₀
- Zero meridional circulation

```julia
# Axisymmetric flux pattern
flux = Y00(-1.0) + Y20(-0.2)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)

# Only zonal flow, no meridional
println("Zonal Y₃₀: ", maximum(abs, bs.uphi_coeffs[3]))
# u_θ = u_r = 0 for axisymmetric basic states
```

**Physical insight:** Comparison with Y₂₂:

| Property | Y₂₀ (m=0) | Y₂₂ (m=2) |
|----------|-----------|-----------|
| Zonal flow | Yes | Yes |
| Meridional flow | **No** | **Yes** |
| Iteration needed | No | Yes |

---

## Quick Reference Table

| Script | Complexity | Compute Time | Key Learning |
|--------|------------|--------------|--------------|
| `linear_stability_demo.jl` | Beginner | ~1 min | Basic workflow |
| `Rac_lm.jl` | Beginner | ~5 min | Parameter sweeps |
| `basic_state_onset_example.jl` | Intermediate | ~2 min | Custom basic states |
| `boundary_driven_jet.jl` | Intermediate | ~3 min | Thermal wind |
| `nonaxisymmetric_basic_state.jl` | Intermediate | ~1 min | 3D states |
| `flux_bc_mean_flow.jl` | Intermediate | ~2 min | Non-axisymmetric flux BC |
| `flux_bc_axisymmetric_flow.jl` | Intermediate | ~1 min | Axisymmetric flux BC |
| `triglobal_analysis_demo.jl` | Advanced | ~10+ min | Mode coupling |
| `mhd_dynamo_example.jl` | Advanced | ~5 min | MHD physics |
| `test_thermal_wind.jl` | Intermediate | ~1 min | Verification |
| `figure2_benchmark.jl` | Intermediate | ~10 min | Benchmarking |

---

## Creating Your Own Scripts

Use this template for new analyses:

```julia
#!/usr/bin/env julia
# my_analysis.jl - Description

# Add Magrathea.jl to path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Magrathea
using Printf
using JLD2

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1e7
χ = 0.35
m = 10
lmax = 60
Nr = 64

# === Setup ===
params = OnsetParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m = m, lmax = lmax, Nr = Nr,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# === Solve ===
println("Computing eigenvalues...")
problem = OnsetProblem(params)
result = solve(problem; nev=6)
eigenvalues = result.eigenvalues
eigenvectors = result.eigenvectors

# === Results ===
println("\nResults:")
for (i, λ) in enumerate(eigenvalues)
    @printf("  λ[%d] = %.6e + %.6ei\n", i, real(λ), imag(λ))
end

# === Save ===
@save "my_results.jld2" params eigenvalues eigenvectors
println("\nResults saved to my_results.jld2")
```

### With Custom Basic State

```julia
# === Basic State with Symbolic BCs ===
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Meridional temperature variation
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))

# Or with combined patterns
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Or with fixed flux at outer boundary
flux = Y00(-1.0) + Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)

# Use in problem setup
params = OnsetParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m = m, lmax = lmax, Nr = Nr,
    basic_state = bs,
)
```

### v2.0 Template

The v2.0 API uses `OnsetParams`, `basic_state(params; mode=...)`, typed problem types, and `solve`:

```julia
using Magrathea

# Define parameters
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35,
                     m=4, lmax=30, Nr=64)

# Create basic state (if needed)
bs = basic_state(params; mode=:meridional, amplitude=0.05)

# Create and solve problem
problem = BiglobalProblem(params, bs)
estimate_size(problem)
result = solve(problem; nev=10)

# Analyze results
println("Growth rate: ", result.growth_rate)
println("Frequency:   ", result.frequency)

# Plot (optional)
# using Plots; plot(result)
```

---

## See Also

- [Getting Started](getting_started.md) - Installation and setup
- [Problem Setup](problem_setup.md) - Detailed configuration guide
- [API Reference](reference.md) - Function documentation
