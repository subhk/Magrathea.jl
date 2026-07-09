# Codebase Structure

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">For developers</div>
  <h1>How the Magrathea.jl source is organized.</h1>
  <p>An overview of the source-code layout and architecture to help you navigate and extend the codebase.</p>
</div>

## Directory Layout

```
Magrathea.jl/
├── src/                              # Source code
│   ├── Magrathea.jl                      # Main module — includes submodules, exports public API
│   ├── validation.jl                 # v2.0: Input validation with errors and warnings
│   ├── types.jl                      # v2.0: StabilityResult, problem types, estimate_size
│   ├── solve.jl                      # v2.0: Unified solve() API dispatching on problem type
│   ├── show.jl                       # v2.0: Pretty-printing for all public types
│   │
│   ├── Spectral/
│   │   ├── Spectral.jl               # Entry point
│   │   ├── chebyshev.jl              # ChebyshevDiffn differentiation matrices
│   │   ├── ultraspherical.jl        # Olver-Townsend sparse spectral method
│   │   └── galerkin.jl              # Tau-free ultraspherical-Galerkin radial operators
│   │
│   ├── Operators/
│   │   ├── Operators.jl              # Entry point
│   │   ├── sparse_operator.jl       # Sparse hydrodynamic operators
│   │   └── boundary_conditions.jl   # Mechanical, thermal, magnetic BCs
│   │
│   ├── BasicStates/
│   │   ├── BasicStates.jl            # Entry point
│   │   ├── basic_state.jl           # BasicState, BasicState3D, SphericalHarmonicBC types
│   │   ├── sh_transform.jl          # Real-orthonormal spherical-harmonic transforms (±m)
│   │   ├── advection_diffusion.jl   # Self-consistent basic-state solver
│   │   └── basic_state_operators.jl # Basic-state coupling operators
│   │
│   ├── Stability/
│   │   ├── Stability.jl              # Entry point
│   │   ├── linear.jl                # OnsetParams, LinearStabilityOperator
│   │   ├── solver.jl                # Pluggable generalized-eigenvalue backends
│   │   ├── dof_ownership.jl         # DOF ↔ global-row mapping (distributed assembly)
│   │   ├── velocity.jl              # Velocity reconstruction
│   │   ├── onset.jl                 # Onset convection (no mean flow)
│   │   ├── biglobal.jl              # Biglobal (axisymmetric mean flow)
│   │   └── triglobal.jl             # Triglobal (3D mode coupling)
│   │
│   └── MHD/
│       ├── MHD.jl                    # Entry point
│       ├── types.jl                  # MHDParams, MHDStabilityOperator, BackgroundField enum
│       ├── dipole.jl                 # Dipole background-field helpers
│       ├── operator_functions.jl    # Lorentz, induction, magnetic-diffusion operators
│       ├── assembly.jl              # Tau (sparse) MHD matrix assembly
│       └── galerkin_assembly.jl    # Tau-free Galerkin MHD assembly
│
├── ext/
│   ├── MagratheaRecipesBaseExt/          # Plots.jl recipes (weak dep: RecipesBase)
│   ├── MagratheaMakieExt/                # Makie visualization (weak dep: Makie)
│   └── MagratheaSlepcExt/                # SLEPc/PETSc distributed eigensolver (weak deps: PetscWrap, SlepcWrap)
│
├── test/                             # Test suite
├── example/                          # Example scripts
├── docs/                             # Documentation (Documenter)
└── Project.toml                      # Julia package manifest
```

## Core Architecture

### Module Entry Point (`Magrathea.jl`)

The main module file orchestrates all submodules and exports the public API:

```julia
module Magrathea
    # Dependencies
    using LinearAlgebra, SparseArrays, JLD2, Printf, Random
    using Parameters
    using LinearMaps, WignerSymbols, SpecialFunctions

    # Submodules (included first — core v2.0 files below depend on them)
    include("Spectral/Spectral.jl")        # Chebyshev + ultraspherical + Galerkin discretization
    include("BasicStates/BasicStates.jl")  # Basic state types, SH transforms, coupling operators
    include("Stability/Stability.jl")      # Eigenvalue machinery and analysis modes
    include("Operators/Operators.jl")      # Sparse operators + boundary conditions
    include("MHD/MHD.jl")                  # MHD extension

    # v2.0 core
    include("validation.jl")           # Input validation
    include("types.jl")                # StabilityResult, problem types, estimate_size
    include("solve.jl")                # Unified solve() API
    include("show.jl")                 # Pretty-printing for public types

    export ...
end
```

### Three Analysis Modes

Magrathea.jl provides three distinct analysis modes for different physical scenarios:

| Mode | File | Basic State | Use Case |
|------|------|-------------|----------|
| **Onset** | `Stability/onset.jl` | None (conduction only) | Classical convection onset |
| **Biglobal** | `Stability/biglobal.jl` | Axisymmetric (``m=0``) | Thermal wind effects |
| **Triglobal** | `Stability/triglobal.jl` | Non-axisymmetric | 3D boundary forcing |

## Source File Descriptions

### v2.0 Core Files

#### `types.jl`
Defines the `StabilityResult` return type, common problem parameter types, and `estimate_size` utilities introduced in v2.0.

#### `validation.jl`
Input validation layer introduced in v2.0. Emits structured errors and warnings before problem setup to catch misconfigurations early.

#### `solve.jl`
Unified `solve()` API (v2.0). Dispatches on the problem type (`OnsetProblem`, `BiglobalProblem`, `TriglobalProblem`, `MHDProblem`) to assemble and solve the appropriate generalized eigenproblem, with memory pre-checks.

#### `show.jl`
Pretty-printing methods (`Base.show`, `Base.summary`) for all public types, introduced in v2.0.

### Spectral Submodule (`Spectral/`)

#### `Spectral/chebyshev.jl`
Chebyshev spectral differentiation for radial discretization.

**Key Types:**
```julia
struct ChebyshevDiffn{T<:AbstractFloat}
    n::Int            # Number of points
    domain::Tuple{T,T}
    max_order::Int
    x::Vector{T}      # Collocation points
    D1::Matrix{T}     # First derivative matrix
    D2::Matrix{T}     # Second derivative matrix
    D3::Matrix{T}     # Third derivative (if requested)
    D4::Matrix{T}     # Fourth derivative (if requested)
end
```

**Key Functions:**
- `ChebyshevDiffn(N, [r_i, r_o], nderiv)` - Construct differentiation matrices

#### `Spectral/ultraspherical.jl`
Olver-Townsend sparse spectral method using ultraspherical (Gegenbauer) polynomials for large-scale problems.

#### `Spectral/galerkin.jl`
Tau-free ultraspherical-Galerkin radial operators. Composes the banded ultraspherical primitives (derivative, conversion, multiplication) into a recombined trial basis that carries the boundary conditions, avoiding tau rows (full-rank `B`, no spurious eigenvalues).

### Operators Submodule (`Operators/`)

#### `Operators/sparse_operator.jl`
Sparse hydrodynamic operators using the ultraspherical spectral basis.

#### `Operators/boundary_conditions.jl`
Mechanical, thermal, and magnetic boundary condition application.

### BasicStates Submodule (`BasicStates/`)

#### `BasicStates/basic_state.jl`
Definitions for background temperature and flow states, including the `SphericalHarmonicBC` type (v2.0).

**Key Types:**
```julia
# Axisymmetric basic state (m=0 modes only)
struct BasicState{T<:Real}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int,Vector{T}}     # θ̄_ℓ0(r)
    uphi_coeffs::Dict{Int,Vector{T}}      # ū_φ,ℓ0(r)
    dtheta_dr_coeffs::Dict{Int,Vector{T}}
    duphi_dr_coeffs::Dict{Int,Vector{T}}
end

# Non-axisymmetric basic state (multiple m modes)
struct BasicState3D{T<:Real}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}  # θ̄_ℓm(r)
    dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
end
```

**Key Functions:**
- `conduction_basic_state(cd, χ, lmax_bs)` - Pure conduction profile
- `meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, amplitude)` - With thermal wind
- `nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes)` - 3D state

#### `BasicStates/sh_transform.jl`
Real-orthonormal spherical-harmonic transforms (cos+sin, ±m) and the vector-harmonic horizontal divergence — the foundation for correct non-axisymmetric basic-state advection. Provides in-place, separable synthesis/analysis routines.

#### `BasicStates/advection_diffusion.jl`
Self-consistent advection-diffusion solver for computing basic states.

**Key Functions:**
- `solve_thermal_wind_balance!(bs, E, Ra, Pr)` - Compute axisymmetric thermal wind
- `solve_thermal_wind_balance_3d!(bs3d, E, Ra, Pr)` - Compute 3D thermal wind

#### `BasicStates/basic_state_operators.jl`
Operators for incorporating basic state effects into stability analysis.

**Key Types:**
```julia
struct BasicStateOperators{T}
    # Advection operators: ū·∇u' and u'·∇ū
    advection_matrices::Dict
    # Temperature advection: ū·∇θ' and u'·∇θ̄
    thermal_advection_matrices::Dict
end
```

**Key Functions:**
- `build_basic_state_operators(bs, params)` - Construct operators
- `add_basic_state_operators!(A, B, ops)` - Add to stability matrices

### Stability Submodule (`Stability/`)

#### `Stability/linear.jl`
Core linear stability analysis machinery shared by all modes.

**Key Types:**
```julia
struct OnsetParams{T<:Real, BS}
    E::T              # Ekman number
    Pr::T             # Prandtl number
    Ra::T             # Rayleigh number
    χ::T              # Radius ratio
    m::Int            # Azimuthal wavenumber
    lmax::Int         # Maximum spherical harmonic degree
    Nr::Int           # Radial resolution
    ri::T             # Inner radius
    ro::T             # Outer radius
    L::T              # Gap width (ro - ri)
    mechanical_bc::Symbol
    thermal_bc::Symbol
    use_sparse_weighting::Bool
    equatorial_symmetry::Symbol
    basic_state::BS   # attached basic state, or `nothing`
end

struct LinearStabilityOperator{T}
    params::OnsetParams{T}
    cd::ChebyshevDiffn{T}
    r::Vector{T}
    index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}
    l_sets::Dict{Symbol, Vector{Int}}
    total_dof::Int
    radial_cache::Dict{Tuple{Int,Int}, Matrix{T}}
end
```

**Key Functions:**
- `assemble_matrices(op)` - Build A and B matrices

#### `Stability/solver.jl`
Generalized-eigenvalue solving (`A x = σ B x`) through a pluggable backend interface. The default backend runs in-process; an optional distributed SLEPc/PETSc backend (`backend=:slepc`) is provided by the `MagratheaSlepcExt` extension, loaded with `using PetscWrap, SlepcWrap`.

**Key Functions:**
- `solve_eigenvalue_problem(op; nev, which)` - Compute eigenvalues
- `find_critical_rayleigh(E, Pr, χ, m, lmax, Nr; tol)` - Find critical Ra

#### `Stability/dof_ownership.jl`
DOF ↔ global-row mapping and PETSc row-ownership queries — pure integer bookkeeping over the `index_map` that underpins the distributed (MPI/SLEPc) assembly path.

#### `Stability/velocity.jl`
Reconstruct velocity components from poloidal/toroidal potentials.

**Key Functions:**
- `potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)` - Grid-based velocity

### Analysis Modes

#### `Stability/onset.jl`
Classical convection onset without mean flow.

**Key Types:**
```julia
struct OnsetConvectionParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m::Int, lmax::Int, Nr::Int
    mechanical_bc::Symbol
    thermal_bc::Symbol
end
```

**Key Functions:**
- `solve_onset_problem(params)` - Solve eigenvalue problem
- `find_critical_Ra_onset(params)` - Find critical Rayleigh number
- `find_global_critical_onset(E, Pr, χ; m_range)` - Scan all m modes
- `estimate_onset_problem_size(params)` - Memory/size estimates
- `onset_scaling_laws(E)` - Asymptotic predictions

#### `Stability/biglobal.jl`
Stability analysis with axisymmetric mean flow (thermal wind).

**Key Types:**
```julia
struct BiglobalParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m::Int, lmax::Int, Nr::Int
    basic_state::BasicState{T}
    mechanical_bc::Symbol
    thermal_bc::Symbol
end
```

**Key Functions:**
- `create_conduction_basic_state(params)` - Conduction profile
- `create_thermal_wind_basic_state(params; amplitude)` - With zonal flow
- `solve_biglobal_problem(params)` - Solve with basic state
- `find_critical_Ra_biglobal(params)` - Critical Ra with mean flow
- `compare_onset_vs_biglobal(params)` - Compare to no-flow case
- `sweep_thermal_wind_amplitude(params, amplitudes)` - Parameter study

#### `Stability/triglobal.jl`
Tri-global analysis with non-axisymmetric basic states.

**Key Types:**
```julia
struct TriglobalParams{T<:Real}
    E::T, Pr::T, Ra::T, χ::T
    m_range::UnitRange{Int}  # Coupled perturbation modes
    lmax::Int, Nr::Int
    basic_state_3d::BasicState3D{T}
    mechanical_bc::Symbol
    thermal_bc::Symbol
end

struct CoupledModeProblem{T}
    params::TriglobalParams{T}
    m_range::UnitRange{Int}
    all_m_bs::Vector{Int}
    coupling_graph::Dict{Int,Vector{Int}}
    block_indices::Dict{Int,UnitRange{Int}}
    total_dofs::Int
end
```

**Key Functions:**
- `setup_coupled_mode_problem(params)` - Build coupled problem structure
- `estimate_triglobal_problem_size(params)` - Size/memory estimates
- `solve_triglobal_eigenvalue_problem(params; nev, σ_target, verbose)` - Solve coupled system
- `find_critical_rayleigh_triglobal(params)` - Critical Ra for 3D forcing

### MHD Submodule (`MHD/`)

#### `MHD/types.jl`
`MHDParams` parameter struct and `BackgroundField` enum for selecting the imposed magnetic field geometry.

#### `MHD/dipole.jl`
Dipole magnetic field operators for the background field.

#### `MHD/operator_functions.jl`
Lorentz force, induction, and magnetic diffusion operator terms.

#### `MHD/assembly.jl`
Tau (sparse) MHD matrix assembly — adds magnetic terms to the A/B matrices produced by the Stability submodule.

#### `MHD/galerkin_assembly.jl`
Tau-free ultraspherical-Galerkin assembly of the MHD eigenproblem. Boundary conditions are carried by a recombined trial basis (no tau rows → full-rank `B` → no spurious eigenvalues).

### Extension Packages (`ext/`)

Optional functionality is provided through Julia's extension mechanism (weak dependencies):

- **`MagratheaRecipesBaseExt/`** - Plots.jl recipes, loaded automatically when `RecipesBase` is available
- **`MagratheaMakieExt/`** - Interactive Makie visualization, loaded automatically when a Makie backend is available
- **`MagratheaSlepcExt/`** - Distributed SLEPc/PETSc eigensolver backend, loaded with `using PetscWrap, SlepcWrap` (enables `backend=:slepc`)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Parameters                            │
│  (E, Pr, Ra, χ, m, lmax, Nr, boundary conditions)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Mode Selection                      │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  Onset          │  Biglobal           │  Triglobal              │
│  (no mean flow) │  (axisymmetric)     │  (non-axisymmetric)     │
└────────┬────────┴──────────┬──────────┴────────────┬────────────┘
         │                   │                        │
         ▼                   ▼                        ▼
┌────────────────┐  ┌────────────────┐      ┌────────────────────┐
│ OnsetParams    │  │ BiglobalParams │      │ TriglobalParams    │
│                │  │ + BasicState   │      │ + BasicState3D     │
└────────┬───────┘  └────────┬───────┘      └─────────┬──────────┘
         │                   │                        │
         └───────────────────┼────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Matrix Assembly                              │
│  • Chebyshev / ultraspherical radial discretization             │
│  • Spherical harmonic angular expansion                         │
│  • Boundary condition application                               │
│  • Basic state coupling operators (if applicable)               │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Generalized Eigenvalue Problem                  │
│                      A x = λ B x                                │
│  • Pluggable eigensolver backend (in-process / SLEPc)           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Results                                   │
│  • Eigenvalues (growth rates, frequencies)                      │
│  • Eigenvectors (mode structure)                                │
│  • Field reconstruction (velocity, temperature, magnetic)       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Algorithms

### Thermal Wind Balance

The thermal wind equation connects temperature gradients to zonal flow:

```math
\cos\theta \frac{\partial \bar{u}_\phi}{\partial r} - \frac{\sin\theta}{r} \bar{u}_\phi = -\frac{Ra \cdot E^2}{2 Pr \cdot r_o} \frac{\partial \bar{\Theta}}{\partial \theta}
```

**Implementation:** `solve_thermal_wind_balance!()` in `BasicStates/basic_state.jl`

1. Project temperature gradient onto spherical harmonics
2. Apply coupling coefficients (``\ell \to \ell \pm 1``)
3. Integrate ODE for each velocity mode
4. Apply boundary conditions (no-slip or stress-free)

### Mode Coupling (Triglobal)

Non-axisymmetric basic states couple perturbation modes:

```math
Y_{\ell_1, m_1} \times Y_{\ell_2, m_2} = \sum_{\ell'} G_{\ell_1 \ell_2 \ell'}^{m_1 m_2 m'} Y_{\ell', m_1+m_2}
```

**Implementation:** `setup_coupled_mode_problem()` in `Stability/triglobal.jl`

1. Identify non-zero ``m_{bs}`` modes in basic state
2. Build coupling graph: ``m \leftrightarrow m \pm m_{bs}``
3. Allocate block-sparse matrix structure
4. Assemble diagonal (single-mode) and off-diagonal (coupling) blocks

## Testing

```
test/
├── runtests.jl                  # Test runner (includes all suites below)
├── chebyshev.jl                 # Chebyshev differentiation tests
├── sparse_operator.jl           # Sparse / ultraspherical operator tests
├── galerkin_radial.jl           # Galerkin radial-operator tests
├── sh_transform.jl              # Spherical-harmonic transform tests
├── boundary_conditions.jl       # BC application tests
├── thermal_wind.jl              # Thermal wind balance tests
├── distributed_triglobal.jl     # Coupled triglobal assembly tests
├── mhd_boundary_conditions.jl   # MHD boundary-condition tests
├── type_stability.jl            # Inference + allocation regression tests
├── test_show.jl                 # Pretty-printing tests
└── ...                          # and more (see test/runtests.jl for the full list)
```

Run tests with:
```julia
using Pkg
Pkg.test("Magrathea")
```

## Dependencies

| Package | Type | Purpose |
|---------|------|---------|
| `LinearAlgebra` | stdlib | Standard linear algebra |
| `SparseArrays` | stdlib | Sparse matrix support |
| `Printf` | stdlib | Formatted output for pretty-printing |
| `Random` | stdlib | Random number generation |
| `Logging` | stdlib | Solver progress / diagnostics |
| `Parameters` | direct | `@with_kw` struct macros |
| `JLD2` | direct | Data serialization |
| `WignerSymbols` | direct | Gaunt coefficients for spherical harmonic coupling |
| `SpecialFunctions` | direct | Special mathematical functions |
| `LinearMaps` | direct | Linear operator abstractions |
| `RecipesBase` | weak | Plots.jl plot recipes (`MagratheaRecipesBaseExt`) |
| `Makie` | weak | Interactive visualization (`MagratheaMakieExt`) |
| `PetscWrap`, `SlepcWrap` | weak | Distributed SLEPc/PETSc eigensolver (`MagratheaSlepcExt`) |
| `BenchmarkTools` | test | Performance benchmarking (test extra) |

## Extension Points

### Adding New Basic States

1. Define new function in `BasicStates/basic_state.jl` returning `BasicState` or `BasicState3D`
2. Populate coefficient dictionaries for temperature and velocity
3. Ensure derivatives are computed consistently

### Adding New Physics

1. Create a new submodule directory under `src/` (e.g., `src/MyPhysics/`)
2. Define parameter struct with required fields
3. Implement matrix assembly functions
4. Add operators to the A/B matrices in assembly
5. Include the submodule entry point in `Magrathea.jl` and export as needed

### Custom Boundary Conditions

1. Extend `Operators/boundary_conditions.jl` with new BC type
2. Implement row replacement in `apply_boundary_conditions!()`
3. Add option to parameter structs


## MHD implementation notes

### Overview

This document describes the magnetohydrodynamic (MHD) implementation in Magrathea.jl, which extends the hydrodynamic convection onset solver to include magnetic field interactions.

**Status:** ⚠️ EXPERIMENTAL - Full implementation following Kore structure

**Date:** 2025-10-26

---

## Contents

1. [Physical Model](#physical-model)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Implementation Structure](#implementation-structure)
4. [Usage Examples](#usage-examples)
5. [Benchmark Tests](#benchmark-tests)
6. [References](#references)

---

## Physical Model

### Governing Equations

The MHD equations in a rotating spherical shell:

**Momentum (Navier-Stokes + Lorentz):**
```
∂u/∂t + 2Ω×u = -∇p + E∇²u + Ra/Pr θr̂ + Le²(∇×B)×B₀
```

**Induction:**
```
∂B/∂t = ∇×(u×B₀) + Em∇²B
```

**Heat:**
```
∂θ/∂t + u·∇T₀ = Etherm∇²θ
```

**Incompressibility:**
```
∇·u = 0,  ∇·B = 0
```

### Non-dimensional Parameters

| Parameter | Symbol | Definition | Typical Range |
|-----------|--------|------------|---------------|
| Ekman number | E | ν/(ΩL²) | 10⁻³ - 10⁻⁷ |
| Prandtl | Pr | ν/κ | 0.1 - 10 |
| Magnetic Prandtl | Pm | ν/η | 0.1 - 10 |
| Rayleigh | Ra | αgΔTL³/(νκ) | 10³ - 10⁸ |
| Lehnert | Le | B₀/(√(μρ)ΩL) | 0.01 - 1 |
| Thermal Ekman | Etherm | E/Pr | - |
| Magnetic Ekman | Em | E/Pm | - |

### Field Decomposition

Following Kore, fields are decomposed into toroidal-poloidal forms:

**Velocity:**
```
u = ∇×(∇×(u_pol r̂)) + ∇×(u_tor r̂)
```

**Magnetic Field:**
```
B = ∇×(∇×(f_pol r̂)) + ∇×(g_tor r̂)
```

**Temperature:**
```
T = T₀(r) + θ(r,θ,φ,t)
```

Each scalar function is expanded in spherical harmonics Y_l^m(θ,φ) and discretized radially using ultraspherical (Gegenbauer) polynomials.

---

## Mathematical Formulation

### Eigenvalue Problem

The linear stability analysis leads to:
```
A x = λ B x
```

Where `x = [u_pol, u_tor, f_pol, g_tor, θ]` contains all perturbation fields.

### Matrix Structure

The matrices have block structure:

```
        u     v     f     g     h
    ┌─────────────────────────────┐
  u │ I+C  Coff  0    Lor   -B    │  (2curl NS + Lorentz)
    │                             │
  v │ Coff  I    Lor  0     0     │  (1curl NS + Lorentz)
    │                             │
  f │ Ind  Ind   I    0     0     │  (no-curl induction)
    │                             │
  g │ Ind  Ind   0    I     0     │  (1curl induction)
    │                             │
  h │ Adv  0     0    0     I     │  (heat equation)
    └─────────────────────────────┘
```

Legend:
- I: Time derivative (diagonal blocks in B matrix)
- C: Coriolis (diagonal and off-diagonal)
- Coff: Coriolis off-diagonal coupling
- Lor: Lorentz force
- Ind: Induction
- B: Buoyancy
- Adv: Thermal advection

### Key Couplings

1. **Magnetic → Velocity (Lorentz Force)**
   - Poloidal B → Poloidal u (via toroidal B component)
   - Toroidal B → Toroidal u (via poloidal B component)
   - Strength: Le²

2. **Velocity → Magnetic (Induction)**
   - Poloidal u → Poloidal B (stretching background field)
   - Toroidal u → Toroidal B (shearing background field)
   - Strength: Le

3. **Temperature → Velocity (Buoyancy)**
   - θ → Poloidal u
   - Strength: Ra/Pr

4. **Velocity → Temperature (Advection)**
   - Poloidal u → θ
   - Strength: 1

---

## Implementation Structure

### Files

```
src/MHD/
├── MHD.jl                  # Module entry point
├── types.jl                # MHDParams, MHDStabilityOperator, BackgroundField enum
├── dipole.jl               # Dipole background-field helpers
├── operator_functions.jl   # Lorentz, induction, magnetic-diffusion operators
├── assembly.jl             # Tau (sparse) MHD matrix assembly
└── galerkin_assembly.jl    # Tau-free Galerkin MHD assembly
```

### Key Data Structures

**`MHDParams`** - Physical and numerical parameters
- Contains all dimensionless numbers (E, Pr, Pm, Ra, Le)
- Geometry (ricb, m, lmax, N)
- Boundary conditions (velocity, temperature, magnetic)
- Background field type

**`MHDStabilityOperator`** - Pre-computed radial operators
- All velocity operators (r^k D^n)
- All magnetic field operators
- Background field operators h(r)
- Mode structure (ll_u, ll_v, ll_f, ll_g, ll_h)

**`BackgroundField`** - Enum for field types
- `no_field`: Kinematic dynamo
- `axial`: Uniform axial field
- `dipole`: Dipolar field (future)

### Core Functions

**Lorentz Force Operators:**
```julia
operator_lorentz_poloidal_diagonal(op, l, Le)
operator_lorentz_poloidal_offdiag(op, l, m, offset, Le)
operator_lorentz_toroidal(op, l, Le)
```

**Induction Operators:**
```julia
operator_induction_poloidal_from_u(op, l)
operator_induction_poloidal_from_v(op, l)
operator_induction_toroidal_from_u(op, l, m, offset)
operator_induction_toroidal_from_v(op, l)
```

**Magnetic Diffusion:**
```julia
operator_magnetic_diffusion_poloidal(op, l, Em)
operator_magnetic_diffusion_toroidal(op, l, Em)
```

**Assembly:**
```julia
assemble_mhd_matrices(op)  # Returns (A, B, interior_dofs, info)
```

---

## Usage Examples

### Basic Dynamo Stability

```julia
using Magrathea

# Define parameters
params = MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e4,
    Le = 0.1,          # Weak background field
    ricb = 0.35,
    m = 2,
    lmax = 20,
    N = 32,
    B0_type = axial,
    bci = 1, bco = 1,  # No-slip
    bci_magnetic = 0,  # Insulating boundaries
    bco_magnetic = 0
)

# Build operator and assemble
op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

# Solve eigenvalue problem (solve_eigenvalue_problem is provided by Magrathea)
A_int = A[interior_dofs, interior_dofs]
B_int = B[interior_dofs, interior_dofs]
eigenvalues, _, _ = solve_eigenvalue_problem(A_int, B_int)

# Analyze results
σ = real(eigenvalues[1])
ω = imag(eigenvalues[1])
if real(σ) > 0
    println("Unstable! Dynamo onset detected")
end
```

### Parameter Scan

```julia
# Scan Rayleigh number for fixed magnetic field
Le = 0.1
Ra_values = 10.0.^range(3, 6, length=20)

for Ra in Ra_values
    params = MHDParams(E=1e-3, Pr=1.0, Pm=5.0, Ra=Ra, Le=Le, ...)
    op = MHDStabilityOperator(params)
    A, B, interior_dofs, _ = assemble_mhd_matrices(op)
    eigenvalues, _, _ = solve_eigenvalue_problem(A[interior_dofs, interior_dofs],
                                                B[interior_dofs, interior_dofs])
    println("Ra = $Ra: σ = $(real(eigenvalues[1]))")
end
```

### Kinematic Dynamo (No Background Field)

```julia
# Set Le = 0 for kinematic dynamo
params = MHDParams(
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e5,
    Le = 0.0,          # No background field
    B0_type = no_field,
    ...
)
```

---

## Benchmark Tests

### Christensen et al. (2001) Benchmark

**Case 0:** Non-magnetic convection
- E = 10⁻³, Pr = 1, Ra = 100, Pm = 5, No magnetic field
- Expected: Ra_c ≈ 50-60

**Case 1:** Strong field dynamo
- E = 10⁻³, Pr = 1, Ra = 100, Pm = 5, Le = 1
- Test Lorentz force stabilization

### Jones et al. (2011) Anelastic Benchmark

Future work: Extend to anelastic equations for more realistic planetary conditions.

---

## Physical Insights

### Dynamo Mechanisms

1. **Omega Effect (Differential Rotation)**
   - Toroidal velocity shears poloidal magnetic field
   - Creates toroidal magnetic field
   - Implemented in `operator_induction_toroidal_from_u`

2. **Alpha Effect (Helical Flows)**
   - Helical convection twists toroidal field
   - Creates poloidal magnetic field
   - Emerges from Coriolis-Lorentz interaction

3. **Magnetic Diffusion**
   - Dissipates magnetic field
   - Controlled by Em = E/Pm
   - Large Pm → slow diffusion → easier dynamo

### Stability Regimes

| Le | Ra | Regime |
|----|----|----|
| 0 | < Ra_c | Stable conduction |
| 0 | > Ra_c | Hydrodynamic convection |
| Small | > Ra_c | Weakly magnetic convection |
| O(1) | > Ra_c | Magnetoconvection |
| Large | Any | Magnetically dominated |

---

## Validation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Velocity operators | ✅ Tested | Matches Kore structure |
| Lorentz force | ⚠️ Implemented | Needs validation |
| Induction | ⚠️ Implemented | Needs validation |
| Magnetic diffusion | ⚠️ Implemented | Needs validation |
| Axial field | ⚠️ Implemented | Needs validation |
| Dipole field | ❌ Not implemented | Future work |
| Anelastic | ❌ Not implemented | Future work |

---

## References

### Papers

1. **Christensen et al. (2001)**
   "A numerical dynamo benchmark"
   *Physics of the Earth and Planetary Interiors*, 128(1-4), 25-34

2. **Jones et al. (2011)**
   "Anelastic convection-driven dynamo benchmarks"
   *Icarus*, 216(1), 120-135

3. **Dormy et al. (2004)**
   "MHD flow in a slightly differentially rotating spherical shell"
   *Earth and Planetary Science Letters*, 219(1-2), 79-86

### Codes

1. **Kore**
   - Python implementation: `kore-main/bin/operators.py`
   - Reference for operator structure

2. **PARODY**
   - Fortran dynamo code
   - Benchmark reference

3. **Magic**
   - Spectral dynamo code
   - Alternative benchmark

### Books

1. **Christensen & Wicht (2015)**
   *Numerical Dynamo Simulations*

2. **Jones (2011)**
   *Planetary Magnetic Fields and Fluid Dynamos*
   In: *Treatise on Geophysics*, Vol. 8

---

## Future Development

### High Priority

- [ ] Validate against Christensen benchmark
- [ ] Test with Jones et al. anelastic benchmark (when anelastic added)
- [ ] Implement dipole background field
- [ ] Add more magnetic field geometries

### Medium Priority

- [ ] Conducting inner core (more complex BCs)
- [ ] Variable magnetic diffusivity
- [ ] Compositional convection coupling
- [ ] Hyperdiffusion for high Ra

### Low Priority

- [ ] Torsional oscillations
- [ ] MAC waves
- [ ] Magnetic boundary layers
- [ ] Non-linear terms (DNS)

---

## Contact & Contribution

This MHD implementation follows the Kore structure and uses the corrected spectral multiplication and factorial scaling from the bug fixes (2025-10-26).

For questions or contributions, refer to the main Magrathea.jl documentation.

**Status:** Experimental implementation complete, validation in progress.
