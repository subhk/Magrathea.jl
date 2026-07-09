# Basic States

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Base states</div>
  <h1>Separate the background state from the perturbations.</h1>
  <p>
    Magrathea.jl decouples the steady base state from the perturbations whose stability
    you study, so you can analyze onset against realistic background temperature and
    flow profiles.
  </p>
</div>

## Overview

Two data structures handle base states:

| Type | Use Case | Description |
|------|----------|-------------|
| `BasicState` | Axisymmetric (``m=0``) | Classical onset problems with zonally-symmetric backgrounds |
| `BasicState3D` | Non-axisymmetric | Tri-global analysis with longitudinal variations |

## Quick Start: Symbolic Boundary Conditions

Magrathea.jl provides an intuitive interface for specifying temperature boundary conditions using spherical harmonic notation. Instead of constructing dictionaries manually, use symbolic constructors:

```julia
using Magrathea

# Create Chebyshev differentiation matrices
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Pure conduction (no boundary variation)
bs = basic_state(cd, χ, E, Ra, Pr)

# Meridional temperature variation (equator-pole contrast)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))

# Combined meridional and longitudinal variation
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Fixed heat flux at outer boundary
flux = Y00(-1.0) + Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

The `basic_state()` function automatically selects the appropriate implementation based on the boundary condition structure.

## Symbolic Spherical Harmonic Boundary Conditions

### The `SphericalHarmonicBC` Type

`SphericalHarmonicBC` represents boundary conditions expanded in spherical harmonics:

```math
\bar{T}(r_o, \theta, \phi) = \sum_{\ell,m} A_{\ell m} Y_{\ell m}(\theta, \phi)
```

### Available Constructors

| Function | Pattern | Physical Meaning |
|----------|---------|------------------|
| `Ylm(ℓ, m, amp)` | General ``Y_{\ell m}`` | Any valid mode |
| `Y00(amp)` | Constant | Uniform (monopole) |
| `Y10(amp)` | ``\cos\theta`` | North-south dipole |
| `Y11(amp)` | ``\sin\theta\cos\phi`` | East-west dipole |
| `Y20(amp)` | ``3\cos^2\theta - 1`` | Equator-pole contrast |
| `Y21(amp)` | ``\sin\theta\cos\theta\cos\phi`` | Tesseral quadrupole |
| `Y22(amp)` | ``\sin^2\theta\cos(2\phi)`` | Four-fold longitudinal |
| `Y30`-`Y44` | Higher orders | Complex patterns |

### Combining Harmonics

Use standard arithmetic operators to build complex patterns:

```julia
# Addition: combine multiple modes
bc = Y20(0.1) + Y22(0.05) + Y40(0.02)

# Scalar multiplication: scale amplitude
bc = 0.5 * Y20(0.2)  # Same as Y20(0.1)

# Subtraction and negation
bc = Y20(0.1) - Y22(0.05)
bc = -Y10(0.1)  # Negative amplitude

# Complex combinations
bc = 0.5 * (Y20(1.0) + 2.0 * Y40(0.5))
```

### Physical Interpretation

Common patterns for convection studies:

| Pattern | Physical Scenario |
|---------|-------------------|
| `Y20(amp)` | Differential heating: equator warmer/cooler than poles |
| `Y22(amp)` | Tidal forcing: four-fold longitudinal variation |
| `Y10(amp)` | Hemispherical asymmetry |
| `Y20(a) + Y40(b)` | Multiple latitudinal bands |
| `Y20(a) + Y22(b)` | Combined meridional and longitudinal forcing |

### The `basic_state()` Convenience Function

```julia
basic_state(cd, χ, E, Ra, Pr;
            temperature_bc = nothing,
            flux_bc = nothing,
            mechanical_bc = :no_slip,
            lmax_bs = nothing)
```

**Arguments:**
- `cd` : Chebyshev differentiation structure
- `χ` : Radius ratio ``r_i/r_o``
- `E` : Ekman number
- `Ra` : Rayleigh number
- `Pr` : Prandtl number

**Keyword Arguments:**
- `temperature_bc` : `SphericalHarmonicBC` for fixed temperature at outer boundary
- `flux_bc` : `SphericalHarmonicBC` for fixed heat flux at outer boundary
- `mechanical_bc` : `:no_slip` (default) or `:stress_free`
- `lmax_bs` : Maximum ``\ell`` for expansion (auto-determined if not specified)

**Automatic Dispatch:**

| Boundary Condition | Function Called | Returns |
|-------------------|-----------------|---------|
| None specified | `conduction_basic_state` | `BasicState` |
| Axisymmetric (``m=0`` only) | `meridional_basic_state` | `BasicState` |
| Non-axisymmetric (``m \neq 0``) | `nonaxisymmetric_basic_state` | `BasicState3D` |

### Examples with Symbolic BCs

#### Example: Stress-Free with Y₂₀ Temperature

```julia
bs = basic_state(cd, χ, E, Ra, Pr;
                 temperature_bc = Y20(0.1),
                 mechanical_bc = :stress_free)
```

#### Example: Fixed Flux at Outer Boundary

```julia
# Uniform outward heat flux with meridional modulation
flux = Y00(-1.0) + Y20(0.2)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

#### Example: Tidal Forcing Pattern

```julia
# Four-fold longitudinal pattern (tidal heating)
bc = Y22(0.15)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
# Returns BasicState3D since m=2 ≠ 0
```

#### Example: Complex 3D Pattern

```julia
# Combined meridional and longitudinal forcing
bc = Y20(0.1) + Y22(0.05) + Y21(0.03)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## v2.0 Unified API

In v2.0, all basic state types are accessible through a single `basic_state(params; mode=...)` function. This eliminates the need to manage `ChebyshevDiffn` objects and dispatch manually:

```julia
using Magrathea

# Define parameters once
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=30, Nr=64)

# All modes via a single function:
bs = basic_state(params; mode=:conduction)
bs = basic_state(params; mode=:meridional, amplitude=0.05)
bs = basic_state(params; mode=:selfconsistent, max_iterations=50)
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)
```

The `mode` keyword selects the construction strategy:

| `mode` | Returns | Description |
|--------|---------|-------------|
| `:conduction` | `BasicState` | Pure conductive profile, no flow |
| `:meridional` | `BasicState` | Y₂₀ thermal wind (axisymmetric) |
| `:selfconsistent` | `BasicState3D` | Iterative solver for full geostrophic balance |
| `:nonaxisymmetric` | `BasicState3D` | Laplace-approximation 3D state |

!!! note "Low-level API"
    The low-level functions `conduction_basic_state`, `meridional_basic_state`, `nonaxisymmetric_basic_state`, and `basic_state_selfconsistent` remain fully supported. The unified API is a convenience wrapper.

---

## Axisymmetric States (`BasicState`)

Axisymmetric cases keep only spherical harmonic modes with azimuthal index ``m = 0``.

### Structure

```julia
struct BasicState{T}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int, Vector{T}}
    uphi_coeffs::Dict{Int, Vector{T}}
    dtheta_dr_coeffs::Dict{Int, Vector{T}}
    duphi_dr_coeffs::Dict{Int, Vector{T}}
end
```

### Conduction Basic State

The simplest case is pure conduction with no flow:

```julia
using Magrathea

# Create Chebyshev differentiation matrices
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Build conduction state (fixed temperature BCs)
bs = conduction_basic_state(cd, χ, 6)

# Or with fixed flux at outer boundary
bs = conduction_basic_state(cd, χ, 6;
                            thermal_bc = :fixed_flux,
                            outer_flux = -1.0)
```

**Fixed Temperature Boundary Conditions (default):**

The conduction profile satisfies ``\nabla^2 \bar{T} = 0`` with:
- ``\bar{T}(r_i) = 1`` (hot inner boundary)
- ``\bar{T}(r_o) = 0`` (cold outer boundary)

Solution:
```math
\bar{T}(r) = \frac{r_o/r - 1}{r_o/r_i - 1}
```

**Fixed Flux Boundary Conditions:**

For prescribed heat flux at the outer boundary:
- ``\bar{T}(r_i) = 1`` (hot inner boundary)
- ``\partial\bar{T}/\partial r|_{r_o} = q`` (prescribed flux)

Use `thermal_bc = :fixed_flux` and specify `outer_flux`.

### Meridional Variations

Add a ``Y_{2,0}`` temperature perturbation for pole-equator differential heating:

```julia
# Using symbolic BC (recommended)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.05))

# Or using the low-level function directly
bs_meridional = meridional_basic_state(
    cd,          # Chebyshev differentiation
    χ,           # Radius ratio
    E,           # Ekman number
    Ra,          # Rayleigh number
    Pr,          # Prandtl number
    6,           # lmax_bs
    0.05;        # amplitude
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# With fixed flux at outer boundary
bs_flux = meridional_basic_state(
    cd, χ, E, Ra, Pr, 6, 0.0;
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_flux,
    outer_flux_mean = -1.0,   # Mean (Y00) flux
    outer_flux_Y20 = 0.1,     # Y20 flux variation
)
```

This generates:
- Temperature perturbation: ``\bar{\Theta}_{20}(r) \propto Y_{2,0}(\theta)``
- Thermal wind: ``\bar{u}_\phi(r,\theta)`` from thermal wind balance

### Thermal Wind Balance

When temperature varies with latitude, geostrophic balance requires a zonal flow:

```math
2\Omega \cos\theta \frac{\partial \bar{u}_\phi}{\partial r} = -\frac{Ra \cdot E^2}{Pr \cdot r} \frac{\partial \bar{\Theta}}{\partial \theta}
```

This balance is handled internally by `meridional_basic_state`. For custom
axisymmetric profiles, you can call the solver directly:

```julia
solve_thermal_wind_balance!(
    uphi_coeffs,
    duphi_dr_coeffs,
    theta_coeffs,
    cd, χ, 1.0, Ra, Pr;
    mechanical_bc = :no_slip,
    E = E,
)
```

### Using Basic States in Problems

Pass the basic state to `OnsetParams` and wrap it in the appropriate problem type:

```julia
params = OnsetParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 12,
    lmax = 60,
    Nr = 96,
    basic_state = bs,  # Include the basic state
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

result = solve(BiglobalProblem(params, bs); nev=8)
```

Magrathea.jl automatically augments the linearized operator with advection terms:

```math
\mathbf{u}' \cdot \nabla \bar{\mathbf{u}} + \bar{\mathbf{u}} \cdot \nabla \mathbf{u}'
```

## Fully 3-D States (`BasicState3D`)

`BasicState3D` stores coefficients indexed by ``(\ell, m)`` pairs for non-axisymmetric backgrounds.

### Structure

```julia
struct BasicState3D{T}
    # Grid
    r::Vector{T}
    Nr::Int
    lmax_bs::Int
    mmax_bs::Int

    # Temperature: θ̄_ℓm(r)
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity: ū_r,ℓm(r), ū_θ,ℓm(r), ū_φ,ℓm(r)
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
end
```

### Creating 3-D Basic States

#### Using Symbolic Boundary Conditions (Recommended)

The easiest way to create 3D basic states is with symbolic spherical harmonic notation:

```julia
# Combined meridional and longitudinal pattern
bc = Y20(0.1) + Y22(0.05)
bs3d = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Fixed flux at outer boundary
flux = Y00(-1.0) + Y20(0.1) + Y22(0.05)
bs3d = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

#### Using Dictionary Syntax

Alternatively, use `nonaxisymmetric_basic_state` directly with dictionary syntax:

```julia
boundary_modes = Dict(
    (2, 0) => 0.1,    # Y₂₀ amplitude at boundary
    (2, 2) => 0.05,   # Y₂₂ amplitude at boundary
)

E = 1e-5

# Fixed temperature at outer boundary
bs3d = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr,
    8,                # lmax_bs
    4,                # mmax_bs
    boundary_modes;
    thermal_bc = :fixed_temperature,
)

# Fixed flux at outer boundary
outer_fluxes = Dict(
    (0, 0) => -1.0,   # Mean flux (Y₀₀)
    (2, 0) => 0.1,    # Y₂₀ flux
    (2, 2) => 0.05,   # Y₂₂ flux
)

bs3d_flux = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr,
    8, 4,
    Dict{Tuple{Int,Int},Float64}();  # empty amplitudes
    thermal_bc = :fixed_flux,
    outer_fluxes = outer_fluxes,
)
```

#### Manual Construction

For custom profiles imported from other sources:

```julia
# Initialize empty dictionaries
Nr = 64
lmax_bs = 8
mmax_bs = 3
r = cd.x

theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
dtheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()

# Populate for all (ℓ,m) pairs
for l in 0:lmax_bs
    for m in -min(l, mmax_bs):min(l, mmax_bs)
        theta_coeffs[(l, m)] = zeros(Nr)
        dtheta_dr_coeffs[(l, m)] = zeros(Nr)
    end
end

# Set specific mode amplitudes
theta_coeffs[(2, 0)] .= your_temperature_profile

# Create the BasicState3D
bs3d = BasicState3D(
    r = r,
    Nr = Nr,
    lmax_bs = lmax_bs,
    mmax_bs = mmax_bs,
    theta_coeffs = theta_coeffs,
    dtheta_dr_coeffs = dtheta_dr_coeffs,
    # ... velocity coefficients ...
)
```

### Importing from External Codes

To import coefficients from other simulation codes (e.g., Rayleigh, Magic):

1. **Export spectral coefficients** from the source code
2. **Transform to Magrathea.jl convention** (check normalization)
3. **Populate the dictionaries** with radially interpolated values
4. **Compute derivatives** using Chebyshev differentiation

```julia
# Example: importing from external data
using JLD2

# Load external data
@load "external_basic_state.jld2" theta_lm r_ext

# Interpolate to Magrathea.jl grid
using Interpolations
for (lm, coeffs) in theta_lm
    itp = LinearInterpolation(r_ext, coeffs)
    theta_coeffs[lm] = itp.(cd.x)
    dtheta_dr_coeffs[lm] = cd.D1 * theta_coeffs[lm]
end
```

## Mode Coupling with Basic States

When a non-axisymmetric basic state is present, perturbation modes couple through advection:

```math
Y_{\ell_1, m_1} \times Y_{\ell_2, m_2} = \sum_{\ell'} G_{\ell_1 \ell_2 \ell'}^{m_1 m_2 m'} Y_{\ell', m_1+m_2}
```

Where ``G`` is the Gaunt coefficient computed from Wigner 3j symbols:

```math
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
```

This coupling is handled automatically by `BasicStateOperators`:

```julia
bs_ops = build_basic_state_operators(bs3d, params)
add_basic_state_operators!(A, B, bs_ops, block_indices)
```

## Saving and Loading

Since base states can be expensive to compute, save them with JLD2:

```julia
using JLD2

# Save
@save "basic_states/meridional_l6.jld2" bs

# Load
@load "basic_states/meridional_l6.jld2" bs_loaded

# Use in new problem
params = OnsetParams(..., basic_state = bs_loaded)
```

## Reality Conditions

For real physical fields, spectral coefficients must satisfy:

```math
\bar{f}_{\ell,-m} = (-1)^m \bar{f}_{\ell,m}^*
```

When constructing `BasicState3D` manually, ensure this condition holds:

```julia
for l in 0:lmax_bs
    for m in 1:min(l, mmax_bs)
        theta_coeffs[(l, -m)] = (-1)^m * conj(theta_coeffs[(l, m)])
    end
end
```

## Examples

### Example 1: Meridional Heating with Symbolic BCs

```julia
using Magrathea

# Setup
E = 1e-5
Pr = 1.0
Ra = 1e7
χ = 0.35
Nr = 64

cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Create meridional basic state using symbolic BC
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))

# Verify structure
println("Temperature modes: ", keys(bs.theta_coeffs))
println("Zonal flow modes: ", keys(bs.uphi_coeffs))
```

### Example 2: Fixed Flux Boundary Condition

```julia
# Heat flux at outer boundary: uniform + meridional variation
flux = Y00(-1.0) + Y20(0.2)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

### Example 3: Non-Axisymmetric Boundary Forcing

```julia
# Combined meridional and longitudinal pattern using symbolic BCs
bc = Y20(0.15) + Y22(0.08) + Ylm(3, 2, 0.03)
bs3d = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Alternatively, using dictionary syntax
boundary_modes = Dict(
    (2, 0) => 0.15,     # Equator-pole variation
    (2, 2) => 0.08,     # East-west variation
    (3, 2) => 0.03,     # Higher-order structure
)
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 10, 4, boundary_modes)

# Use with tri-global analysis
tri_params = TriglobalParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m_range = -3:3,
    lmax = 40,
    Nr = Nr,
    basic_state_3d = bs3d,
)
```

### Example 4: Stress-Free Boundaries

```julia
# Stress-free mechanical BCs with Y20 temperature variation
bs = basic_state(cd, χ, E, Ra, Pr;
                 temperature_bc = Y20(0.1),
                 mechanical_bc = :stress_free)
```

### Example 5: Complex 3D Flux Pattern

```julia
# 3D heat flux pattern at outer boundary
flux = Y00(-1.0) + Y20(0.1) + Y22(0.05) + Y40(0.02)
bs3d = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

## Self-Consistent Basic States with Advection

### The Physics of Temperature Advection

For **axisymmetric** (bi-global) basic states, the zonal flow ``\bar{u}_\phi(r,\theta)`` advects only in the ``\phi`` direction, but the temperature ``\bar{T}(r,\theta)`` has no ``\phi`` dependence:

```math
\bar{\mathbf{u}} \cdot \nabla \bar{T} = \frac{\bar{u}_\phi}{r \sin\theta} \frac{\partial \bar{T}}{\partial \phi} = 0
```

This means the standard approach of solving ``\nabla^2 \bar{T} = 0`` (Laplace equation) is **exact** for axisymmetric cases.

For **non-axisymmetric** (tri-global) basic states, the temperature depends on ``\phi``:

```math
\bar{T}(r, \theta, \phi) = \sum_{\ell, m} \bar{T}_{\ell m}(r) Y_{\ell m}(\theta, \phi)
```

Now the advection term is **non-zero**:

```math
\bar{\mathbf{u}} \cdot \nabla \bar{T} = \frac{\bar{u}_\phi}{r \sin\theta} \frac{\partial \bar{T}}{\partial \phi} = \frac{i m \bar{u}_\phi \bar{T}}{r \sin\theta} \neq 0
```

The full steady-state equation becomes:

```math
\kappa \nabla^2 \bar{T} = \bar{\mathbf{u}} \cdot \nabla \bar{T}
```

where ``\kappa`` is the thermal diffusivity.

### When Does This Matter?

The importance of advection is controlled by the **Péclet number**:

```math
\text{Pe} = \frac{UL}{\kappa}
```

| Regime | Péclet Number | Approximation |
|--------|---------------|---------------|
| Low Pe | Pe ≪ 1 | Diffusion dominates: ``\nabla^2 \bar{T} \approx 0`` (Laplace) |
| High Pe | Pe ≫ 1 | Advection dominates: must solve coupled problem |

For most planetary/stellar scenarios with moderate forcing amplitudes, the Laplace approximation is sufficient. Use the self-consistent solver when:

- Non-axisymmetric amplitude > 0.1
- High quantitative accuracy is needed
- Studying strong forcing scenarios
- Benchmarking against other codes

### Using the Self-Consistent Solver

Magrathea.jl provides `basic_state_selfconsistent()` which iteratively solves the coupled advection-diffusion equation:

```julia
using Magrathea

# Setup
cd = ChebyshevDiffn(64, [0.35, 1.0], 4)
E, Pr, Ra, χ = 1e-5, 1.0, 1e7, 0.35

# Non-axisymmetric boundary condition
bc = Y20(0.1) + Y22(0.08)

# Standard solver (Laplace approximation)
bs_standard = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Self-consistent solver (with advection)
bs_sc, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                          temperature_bc=bc,
                                          verbose=true)

println("Converged in $(info.iterations) iterations")
```

### Algorithm

The solver uses **Picard iteration**:

1. **Initialize**: Solve ``\nabla^2 \bar{T}^{(0)} = 0`` (Laplace)
2. **Thermal wind**: Compute ``\bar{u}_\phi^{(n)}`` from ``\bar{T}^{(n)}``
3. **Advection source**: ``S^{(n)} = \frac{1}{\kappa} \bar{u}_\phi^{(n)} \cdot \nabla \bar{T}^{(n)}``
4. **Poisson solve**: ``\nabla^2 \bar{T}^{(n+1)} = S^{(n)}`` with boundary conditions
5. **Check convergence**: ``\|\bar{T}^{(n+1)} - \bar{T}^{(n)}\| < \epsilon``
6. **Repeat** steps 2-5 until converged

### Options

```julia
basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                           temperature_bc = Y20(0.1) + Y22(0.05),
                           flux_bc = nothing,
                           mechanical_bc = :no_slip,
                           lmax_bs = nothing,
                           max_iterations = 20,    # Max Picard iterations
                           tolerance = 1e-8,       # Convergence tolerance
                           verbose = false)        # Print progress
```

### Convergence Information

The solver returns a named tuple with convergence diagnostics:

```julia
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr; temperature_bc=bc)

info.iterations       # Number of iterations used
info.converged        # true if converged, false if hit max_iterations
info.residual_history # Vector of residuals at each iteration
```

### Example: Comparing Standard vs Self-Consistent

```julia
using Magrathea
using Printf

cd = ChebyshevDiffn(64, [0.35, 1.0], 4)
E, Pr, Ra, χ = 1e-5, 1.0, 1e8, 0.35

bc = Y20(0.2) + Y22(0.1)  # Larger amplitudes

# Standard (Laplace)
bs_laplace = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)

# Self-consistent
bs_sc, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                          temperature_bc=bc,
                                          verbose=true)

# Compare Y22 temperature coefficients
T22_laplace = bs_laplace.theta_coeffs[(2, 2)]
T22_sc = bs_sc.theta_coeffs[(2, 2)]

diff = maximum(abs.(T22_laplace .- T22_sc))
@printf("Max difference in T_22: %.4e\n", diff)
```

### Technical Notes

- **Spectral coupling**: The advection term ``\bar{u}_\phi Y_{Lm} \times \bar{T}_{\ell m} Y_{\ell m}`` couples modes through Gaunt coefficients
- **m=0 modes**: Have zero advection (no ``\phi`` dependence)
- **Mode coupling**: The full solver accounts for coupling through the ``(\hat{z}\cdot\nabla)`` operator
- **Convergence**: Typically converges in 2-5 iterations for small amplitudes, more for larger amplitudes
- **Non-convergence**: May indicate the basic state is unstable (not typical for onset studies)

## Full Geostrophic Balance with Meridional Circulation

### Physical Overview

For **non-axisymmetric** basic states (``m \neq 0``), the complete geostrophic balance includes not just zonal flow (``\bar{u}_\phi``) but also **meridional circulation** (``\bar{u}_r``, ``\bar{u}_\theta``).

The full thermal wind equation (curl of geostrophic balance) is:

```math
2\Omega \, (\hat{\mathbf{z}} \cdot \nabla) \bar{\mathbf{u}} = \frac{Ra \cdot E^2}{Pr} \nabla \bar{T} \times \hat{\mathbf{r}}
```

**Component-wise:**

| Component | Equation | Drives |
|-----------|----------|--------|
| ``\phi`` (zonal) | ``2\Omega (\hat{z}\cdot\nabla) \bar{u}_\phi = \frac{Ra E^2}{Pr r} \frac{\partial \bar{T}}{\partial \theta}`` | Zonal jets |
| ``\theta`` (meridional) | ``2\Omega (\hat{z}\cdot\nabla) \bar{u}_\theta = -\frac{Ra E^2}{Pr r \sin\theta} \frac{\partial \bar{T}}{\partial \phi}`` | Meridional flow |
| Continuity | ``\nabla \cdot \bar{\mathbf{u}} = 0`` | Radial flow |

### Why Meridional Circulation Matters

For **axisymmetric** basic states (``m = 0`` only):
- ``\partial \bar{T}/\partial \phi = 0`` → No forcing for ``\bar{u}_\theta``
- Meridional circulation is **exactly zero**
- Only zonal flow exists

For **non-axisymmetric** basic states (``m \neq 0``):
- ``\partial \bar{T}/\partial \phi \propto im \bar{T}`` → **Non-zero forcing**
- Meridional circulation is driven by the ``\phi``-gradient
- Full three-component velocity field required

### The ``(\hat{z}\cdot\nabla)`` Operator and Mode Coupling

The key operator in geostrophic balance is:

```math
(\hat{\mathbf{z}} \cdot \nabla) = \cos\theta \frac{\partial}{\partial r} - \frac{\sin\theta}{r} \frac{\partial}{\partial \theta}
```

In spectral space, this **couples modes ``\ell`` to ``\ell \pm 1``**:

```math
\cos\theta \, Y_{\ell m} = C^+_{\ell m} Y_{\ell+1,m} + C^-_{\ell m} Y_{\ell-1,m}
```

```math
\sin\theta \frac{\partial Y_{\ell m}}{\partial \theta} = A^+_{\ell m} Y_{\ell+1,m} + A^-_{\ell m} Y_{\ell-1,m}
```

This requires solving a **block-tridiagonal system** for all ``\ell`` modes at each azimuthal wavenumber ``m``.

### Toroidal-Poloidal Decomposition

Magrathea.jl uses the **toroidal-poloidal decomposition** for the meridional circulation, which:

1. **Eliminates pressure** from the formulation
2. **Automatically satisfies** the continuity equation ``\nabla \cdot \bar{\mathbf{u}} = 0``
3. **Handles mode coupling** through exact spherical harmonic recurrence relations

The solver builds the full block-tridiagonal system:

```math
\begin{pmatrix}
\ddots & & & \\
& A_{\ell-1} & C_{\ell-1,\ell} & \\
& C_{\ell,\ell-1} & A_\ell & C_{\ell,\ell+1} \\
& & C_{\ell+1,\ell} & A_{\ell+1} \\
& & & & \ddots
\end{pmatrix}
\begin{pmatrix}
\vdots \\ \bar{u}_{\theta,\ell-1} \\ \bar{u}_{\theta,\ell} \\ \bar{u}_{\theta,\ell+1} \\ \vdots
\end{pmatrix}
= \begin{pmatrix}
\vdots \\ F_{\ell-1} \\ F_\ell \\ F_{\ell+1} \\ \vdots
\end{pmatrix}
```

where:
- ``A_\ell``: Diagonal blocks (regularization for numerical stability)
- ``C_{\ell,\ell\pm1}``: Off-diagonal coupling from ``\cos\theta`` and ``\sin\theta \partial/\partial\theta``
- ``F_\ell``: Forcing from ``\partial \bar{T}/\partial \phi``

After solving for ``\bar{u}_\theta``, the radial velocity ``\bar{u}_r`` is computed from continuity.

### Using the Full Solver

The self-consistent solver automatically uses the full geostrophic balance:

```julia
using Magrathea

# Setup
cd = ChebyshevDiffn(32, [0.35, 1.0], 4)
E, Pr, Ra, χ = 1e-4, 1.0, 1e6, 0.35

# Non-axisymmetric flux boundary condition
flux = Y00(-1.0) + Y22(-0.2)

# Self-consistent solver (includes meridional circulation)
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
                                       flux_bc = flux,
                                       verbose = true)

# Access all velocity components
println("Zonal velocity modes: ", keys(bs.uphi_coeffs))
println("Meridional velocity modes: ", keys(bs.utheta_coeffs))
println("Radial velocity modes: ", keys(bs.ur_coeffs))
```

### Controlling the Solver Options

```julia
# Use full mode coupling (default)
solve_meridional_circulation_toroidal_poloidal!(
    ur_coeffs, utheta_coeffs, ...,
    use_full_coupling = true  # Full block-tridiagonal solver
)

# Use diagonal approximation (faster, less accurate)
solve_meridional_circulation_toroidal_poloidal!(
    ur_coeffs, utheta_coeffs, ...,
    use_full_coupling = false  # Simplified diagonal solver
)

# Disable meridional circulation entirely
solve_meridional_circulation_toroidal_poloidal!(
    ur_coeffs, utheta_coeffs, ...,
    include_meridional = false  # Sets u_r = u_θ = 0
)
```

### Coupling Coefficient Functions

Magrathea.jl provides functions for computing the spherical harmonic coupling coefficients:

```julia
# cos(θ) × Y_ℓm coupling
b_minus, b_plus = cos_theta_coupling(ℓ, m)

# sin(θ) × Y_ℓm coupling
a_minus, a_plus = sin_theta_coupling(ℓ, m)

# sin(θ) × ∂Y_ℓm/∂θ coupling
A_minus, A_plus, A_diag = theta_derivative_coupling(ℓ, m)

# ⟨Y_Lm | 1/sinθ | Y_ℓm⟩ Gaunt-like integral
gaunt = inv_sin_theta_gaunt(L, ℓ, m)
```

### Example: Y₂₂ Heat Flux (Non-Axisymmetric)

```julia
# Sectoral heat flux pattern at outer boundary
flux = Y00(-1.0) + Y22(-0.2)
bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr; flux_bc=flux, verbose=true)

# Results show all three velocity components
# Temperature: Y₀₀ (conduction) + Y₂₂ (sectoral)
# Zonal flow: Y₃₂ mode from thermal wind
# Meridional: Multiple modes (ℓ=2,3,4,...) from mode coupling
```

### Example: Y₂₀ Heat Flux (Axisymmetric)

```julia
# Latitudinal heat flux pattern at outer boundary
flux = Y00(-1.0) + Y20(-0.2)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)

# Results show only zonal flow
# Temperature: Y₀₀ (conduction) + Y₂₀ (latitudinal)
# Zonal flow: Y₁₀, Y₃₀ modes from thermal wind
# Meridional: u_r = u_θ = 0 (exactly zero for m=0)
```

### Comparison: Y₂₀ vs Y₂₂

| Property | Y₂₀ (m=0) | Y₂₂ (m=2) |
|----------|-----------|-----------|
| Symmetry | Axisymmetric | Sectoral (4-fold) |
| ``\partial \bar{T}/\partial \phi`` | = 0 | ≠ 0 |
| Advection ``\bar{\mathbf{u}}\cdot\nabla\bar{T}`` | = 0 (no iteration) | ≠ 0 (iteration needed) |
| Zonal flow ``\bar{u}_\phi`` | Yes (Y₁₀, Y₃₀) | Yes (Y₃₂) |
| Meridional ``\bar{u}_\theta``, ``\bar{u}_r`` | **No** | **Yes** (mode coupling) |
| Velocity modes | ``\ell = 1, 3`` | ``\ell = 2, 3, 4, ..., L_{max}`` |

### Physical Interpretation

The full geostrophic solution captures important physics:

1. **Zonal jets** from thermal wind balance (both Y₂₀ and Y₂₂)
2. **Meridional overturning cells** driven by ``\phi``-gradient of temperature (Y₂₂ only)
3. **Mode coupling cascade**: Energy spreads across multiple ``\ell`` modes
4. **Continuity-consistent radial flow**: ``\bar{u}_r`` computed from ``\nabla \cdot \bar{\mathbf{u}} = 0``

This is particularly important for:
- **Tidal forcing** (Y₂₂ patterns from gravitational tides)
- **Heterogeneous boundary heat flux** (CMB variations)
- **Libration-driven flows** in planetary cores

## Checklist

Before using a basic state:

- [ ] Radial grids match between basic state and analysis (`Nr`, `χ`)
- [ ] Coefficients satisfy reality conditions for physical fields
- [ ] All expected ``(\ell, m)`` pairs have entries in dictionaries
- [ ] Derivatives computed consistently with Chebyshev operators
- [ ] Saved JLD2 files reload without conversion warnings

## Next Steps

- **[Tri-Global Analysis](triglobal.md)** - Use 3-D basic states for mode coupling
- **[MHD Extension](mhd_extension.md)** - Add magnetic field effects to basic states

---

!!! info "Example Scripts"
    See the following examples in the `example/` directory:

    - `basic_state_onset_example.jl` - Basic state with symbolic BCs
    - `nonaxisymmetric_basic_state.jl` - 3D basic states with Y₂₂ patterns
    - `flux_bc_mean_flow.jl` - Non-axisymmetric heat flux (Y₂₂) with meridional circulation
    - `flux_bc_axisymmetric_flow.jl` - Axisymmetric heat flux (Y₂₀) showing zero meridional flow
