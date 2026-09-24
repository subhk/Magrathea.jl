# Basic States

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

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

`SphericalHarmonicBC` represents boundary conditions expanded in spherical harmonics.
An amplitude ``A_{\ell m}`` multiplies the unnormalized associated Legendre function:

```math
\bar{T}(r_o, \theta, \phi) = \sum_{\ell,m} A_{\ell m} P_\ell^m(\cos\theta)\cos(m\phi),
```

where ``P_\ell^m`` includes the Condon–Shortley phase ``(-1)^m``, so odd-``m``
patterns carry a minus sign. A `flux_bc` prescribes ``\partial\bar T/\partial r``
at ``r_o`` with the same pattern (the outward heat flux is its negative).

### Available Constructors

| Function | Pattern per unit amplitude | Physical Meaning |
|----------|----------------------------|------------------|
| `Ylm(ℓ, m, amp)` | ``P_\ell^m(\cos\theta)\cos(m\phi)`` | Any valid mode |
| `Y00(amp)` | ``1`` | Uniform (monopole) |
| `Y10(amp)` | ``\cos\theta`` | North-south dipole |
| `Y11(amp)` | ``-\sin\theta\cos\phi`` | East-west dipole |
| `Y20(amp)` | ``(3\cos^2\theta - 1)/2`` | Equator-pole contrast |
| `Y21(amp)` | ``-3\sin\theta\cos\theta\cos\phi`` | Tesseral quadrupole |
| `Y22(amp)` | ``3\sin^2\theta\cos(2\phi)`` | Four-fold longitudinal |
| `Y30`-`Y44` | Higher orders (see docstrings) | Complex patterns |

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
- `lmax_bs` : Maximum ``\ell`` for expansion (auto-determined if not specified).
  An explicit value must retain every nonzero boundary mode; otherwise an
  `ArgumentError` is thrown rather than silently dropping the forcing.

**Automatic Dispatch:**

| Boundary Condition | Function Called | Returns |
|-------------------|-----------------|---------|
| None specified | `conduction_basic_state` | `BasicState` |
| Axisymmetric (``m=0`` only) | shared viscous solver | `BasicState` |
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

## Unified API

With the unified API, all basic state types are accessible through a single `basic_state(params; mode=...)` function. This eliminates the need to manage `ChebyshevDiffn` objects and dispatch manually:

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
| `:selfconsistent` | `BasicState` or `BasicState3D` | Nonlinear Navier–Stokes–Coriolis and thermal transport |
| `:nonaxisymmetric` | `BasicState3D` | Laplace-approximation 3D state |

Each mode imposes its outer-wall forcing as a temperature or a flux, following
the outer wall of `params.thermal_bc`, and holds the inner wall at a fixed
temperature. `params.thermal_bc` must therefore be `:fixed_temperature` or
`(:fixed_temperature, :fixed_flux)`.

!!! note "Low-level API"
    The low-level functions `conduction_basic_state`, `meridional_basic_state`, `nonaxisymmetric_basic_state`, and `basic_state_selfconsistent` remain fully supported. The unified API is a convenience wrapper.

---

## Axisymmetric States (`BasicState`)

Axisymmetric cases keep only spherical harmonic modes with azimuthal index ``m = 0``.

### Structure

See [`BasicState`](@ref) in the API reference for the current fields. Temperature and component dictionaries use radial collocation values. The `flow` field stores the authoritative divergence-free vector representation; use [`mean_flow_velocity`](@ref) for physical velocity components.

### Conduction Basic State

The simplest case is pure conduction with no flow:

```julia
using Magrathea

# Create Chebyshev differentiation matrices
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Build conduction state (fixed temperature BCs)
bs = conduction_basic_state(cd, χ, 6)

# Or with fixed flux at outer boundary (default: the conduction flux)
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
- ``\partial\bar{T}/\partial r|_{r_o} = q`` (prescribed gradient; the outward
  heat flux is ``-q``, so ``q < 0`` carries heat outward)

Use `thermal_bc = :fixed_flux` and specify `outer_flux`. The default,
`outer_flux = -χ/(1 - χ)`, carries the conduction heat flux and reproduces the
fixed-temperature profile; `meridional_basic_state` (`outer_flux_mean`) and
`nonaxisymmetric_basic_state` (the `(0, 0)` flux) use the same default. Pair a
fixed-flux basic state with perturbations that keep the inner temperature fixed,
`thermal_bc = (:fixed_temperature, :fixed_flux)` in `OnsetParams`.

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

This generates the prescribed conductive temperature and a three-component
axisymmetric velocity. Viscous meridional circulation is generally nonzero,
even when the temperature has no azimuthal dependence. A nonzero ``Y_{2,0}``
forcing requires `lmax_bs ≥ 2`; smaller values throw an `ArgumentError`.

### Thermal Wind Balance

The noniterated constructors (`meridional_basic_state`,
`nonaxisymmetric_basic_state`, and `basic_state(cd, ...)`) solve the steady
**Stokes–Coriolis** equations:

```math
2\hat{\mathbf z}\times\bar{\mathbf u}
= -\nabla\bar p + \frac{Ra E^2}{Pr(1-\chi)^3}\,r\bar T\hat{\mathbf r}
+ E\nabla^2\bar{\mathbf u},\qquad \nabla\cdot\bar{\mathbf u}=0.
```

Both inner and outer mechanical boundaries are enforced. The thermal-wind
balance is an interior approximation to these equations; it is insufficient
to impose both boundaries by itself. The model neglects momentum inertia.
They also neglect temperature advection, so their conductive temperature
approximation requires a small thermal Péclet number. Use the self-consistent
solver below to include both nonlinear momentum and thermal transport.

The public Rayleigh number uses shell thickness, while radius and time use
outer radius and inverse rotation rate. The conversion ``Ra/(1-\chi)^3`` is
applied exactly once. Thermal diffusivity in these units is ``E/Pr``.

```julia
velocity = mean_flow_velocity(bs, 0.7, pi/3, 0.0)
velocity.ur, velocity.utheta, velocity.uphi
```

`bs.flow` holds orthonormal vector-harmonic potentials, including every mode
through `lmax_bs`. The component dictionaries remain available as scalar
projections for existing consumers. Tangential vector components are not
band-limited scalar harmonics: use `mean_flow_velocity` for physical fields,
pole regularity, and continuity checks, rather than differentiating the
truncated scalar component projections. Likewise, `compute_full_advection_spectral(bs)`
evaluates ``\bar{\mathbf u}\cdot\nabla\bar T`` from `bs.flow`; its older
form taking component coefficients is approximate and warns once.

The old `solve_thermal_wind_balance!`, `solve_thermal_wind_balance_3d!`, and
`solve_thermal_wind_coupled!` interfaces now return component projections of
the viscous solve. A single-phase component dictionary cannot retain the
complete nonaxisymmetric flow; prefer the basic-state constructors.

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

See [`BasicState3D`](@ref) in the API reference for the current fields. Temperature and component dictionaries use radial collocation values. The `flow` field stores the authoritative divergence-free vector representation; use [`mean_flow_velocity`](@ref) for physical velocity components.

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

Every nonzero entry must satisfy `ℓ ≤ lmax_bs` and `|m| ≤ min(ℓ, mmax_bs)`;
keys `(ℓ, -m)` select the ``\sin(m\phi)`` phase. Modes outside this range throw
an `ArgumentError`.

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

For complex orthonormal harmonics the scalar product coefficient is

```math
G_{123}=(-1)^{m_3}\sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\0&0&0\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\m_1&m_2&-m_3\end{pmatrix}.
```

The stability assembly evaluates the full vector products by angular quadrature,
using native mean-flow potentials when available. Both 2D and 3D use

```math
\mathcal F=\mathbf U\times(\nabla\times\mathbf u)
            +\mathbf u\times(\nabla\times\mathbf U),\qquad
\mathcal G=-\mathbf U\cdot\nabla\Theta-\mathbf u\cdot\nabla\overline T.
```

``\mathcal F`` differs from the linearized momentum-advection force only by a
pressure gradient. If its vector-harmonic components are ``F_R,F_S,F_T``, the
onset matrix receives ``-\ell(\ell+1)r^3[F_R-\partial_r(rF_S)]`` in the poloidal
row and ``\ell(\ell+1)r^2F_T`` in the toroidal row. The heat contribution is
``r^k\mathcal G``, with the same ``k=3`` or ``k=2`` used in the temperature mass
matrix. Radial differentiation applies the product rule to the mean coefficient
before differentiating perturbations, avoiding aliasing of the highest polynomial.

For an axisymmetric `bs` and matching `OnsetParams`:

```julia
op = LinearStabilityOperator(params)
bs_ops = build_basic_state_operators(bs, op, params.m)
# Complete blocks: (l_output, field_output, l_input, field_input)
blocks = bs_ops.blocks
```

Normal matrix assembly includes these blocks automatically. Triglobal assembly
uses the same projection for each pair of signed azimuthal modes. The older
`BasicStateOperators` two-index dictionaries are inspection aliases; use `blocks`
for the complete field couplings.

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

`BasicState3D` stores **real** cosine coefficients at ``(\ell,+m)`` and independent
real sine coefficients at ``(\ell,-m)``. A missing sine coefficient means zero;
it must not be filled with a conjugate of the cosine coefficient.

For ``m>0``, let ``A`` and ``B`` be the stored cosine and sine coefficients and
``s_{\ell m}=\sqrt{(\ell+m)!/(\ell-m)!}`` the public no-factorial to orthonormal
conversion. The complex coefficients used internally are

```math
c_{\ell,m}=s_{\ell m}(A-iB)/\sqrt2,\qquad
c_{\ell,-m}=(-1)^m s_{\ell m}(A+iB)/\sqrt2.
```

These automatically satisfy ``c_{\ell,-m}=(-1)^m c_{\ell,m}^*``. Native vector
potentials are already orthonormal, so their conversion uses ``s_{\ell m}=1``.

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

The self-consistent solver includes nonlinear mean-flow inertia by default
(`momentum_model=:navier_stokes`) in both 2D and 3D. It solves

```math
(\bar{\mathbf u}\cdot\nabla)\bar{\mathbf u}
+2\hat{\mathbf z}\times\bar{\mathbf u}
=-\nabla\bar p+\frac{Ra E^2}{Pr(1-\chi)^3}\,r\bar T\hat{\mathbf r}
+E\nabla^2\bar{\mathbf u},\qquad \nabla\cdot\bar{\mathbf u}=0,
\qquad \frac{E}{Pr}\nabla^2\bar T=\bar{\mathbf u}\cdot\nabla\bar T.
```

The damped Picard iteration solves thermal transport implicitly at fixed
velocity, then solves momentum with frozen nonlinear forcing. It uses
``-({\mathbf U}\cdot\nabla){\mathbf U}={\mathbf U}\times(\nabla\times{\mathbf U})
-\nabla(|{\mathbf U}|^2/2)`` and absorbs the gradient in pressure. Angular
products use an oversampled grid and the native vector-harmonic flow.
Backtracking accepts updates that reduce the fixed-point residual described below.
`momentum_model=:stokes` explicitly retains the earlier weak-inertia model
while still solving thermal transport.

```julia
cd = ChebyshevDiffn(32, [0.35, 1.0], 4)
bs, info = basic_state_selfconsistent(cd, 0.35, 0.01, 30.0, 1.0;
    temperature_bc=Y20(0.01) + Y22(0.01),
    lmax_bs=8, max_iterations=50, tolerance=1e-9)
@assert info.converged
println((info.momentum_residual, info.thermal_residual, info.boundary_residual))
v = mean_flow_velocity(bs, 0.7, pi/3, pi/8)
```

`info.converged` requires momentum, heat, and boundary/gauge residuals to meet
`tolerance`. The momentum and heat residuals are fixed-point residuals: the
largest change of the orthonormal flow potentials and temperature coefficients
under one undamped Picard update, relative to their largest coefficient. The
distance to the steady state is about this residual divided by one minus the
iteration's contraction rate. The collocation defect of the momentum equations
is not used, because its round-off floor grows roughly like ``N^9`` through the
fourth-derivative viscous rows and exceeds `1e-8` near 64 radial nodes. The
boundary residual is the relative defect of the mechanical, gauge, and thermal
boundary conditions.
`info.residual_history` records their maximum, and
`info.momentum_residual_history`, `info.thermal_residual_history`, and
`info.step_history` record each accepted update (a zero step indicates a
failed line search). `info.termination_reason` is `:converged`,
`:max_iterations`, or `:stagnation`.

When `info.converged=false`, the returned state is an incomplete iterate and
must not be treated as a steady equilibrium. Strong forcing may require
smaller `relaxation` or more iterations; Picard convergence is not guaranteed.
Failure to converge does not establish
physical instability. The convenience call `basic_state(params;
mode=:selfconsistent)` throws on nonconvergence unless
`allow_unconverged=true` is explicitly requested.

Boundary conditions apply to every retained real harmonic, including sine
modes generated by transport. Fixed-flux problems retain homogeneous flux on
unforced modes. The purely conductive convenience path returns `nothing` for
`info`; axisymmetric forced states run the coupled iteration too. For 3D,
`basic_state_selfconsistent` defaults to `mmax_bs=lmax_bs`, since nonlinear
products generate azimuthal orders absent from the boundary forcing. An
explicit `mmax_bs` may be used to control that truncation.

## Full Geostrophic Balance with Meridional Circulation

The mean velocity is represented as

```math
\bar{\mathbf u} = \sum_{\ell,m}\left[
\frac{\ell(\ell+1)p_{\ell m}}{r^2}Y_{\ell m}\hat{\mathbf r}
+\frac{p'_{\ell m}}{r}\nabla_hY_{\ell m}
+\frac{t_{\ell m}}{r}\hat{\mathbf r}\times\nabla_hY_{\ell m}\right].
```

This representation enforces incompressibility and vector regularity without
integrating an independent radial continuity equation. Projection of the full
Coriolis cross product couples degrees and cosine/sine phases. Internal
harmonics are orthonormal; public scalar coefficients are converted from the
historical no-factorial normalization on entry and back on output.

At each boundary, no-slip requires ``p=p'=t=0``. Stress-free requires
``p=0``, ``p''-2p'/r=0``, and ``t'-2t/r=0``. With two stress-free boundaries,
zero axial angular momentum fixes the axisymmetric solid-rotation nullspace.

The historical `coupled_thermal_wind` and `include_meridional_flow` constructor
keywords (and `use_full_coupling` of `solve_meridional_circulation_toroidal_poloidal!`)
are accepted for source compatibility but ignored: all values construct the
complete viscous flow, and passing `false` emits a one-time warning. The old
diagonal approximation is no longer used.

For quantitative work, increase radial resolution and `lmax_bs` until the
physical velocity, temperature, and balance residuals converge. Small equation
residuals establish convergence only at the chosen truncation. In a nonlinear
calculation, also increase `mmax_bs`, since transport can generate
azimuthal orders above those in the prescribed boundary forcing. Small Ekman
numbers require enough radial nodes to resolve the viscous boundary layers.

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
    - `flux_bc_axisymmetric_flow.jl` - Axisymmetric heat flux (Y₂₀)
