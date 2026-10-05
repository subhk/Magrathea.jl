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

Choose the physical model separately from its numerical resolution:

| Constructor | Temperature equation | Momentum equation |
|-------------|----------------------|-------------------|
| `conduction_basic_state` | Laplace conduction | Zero velocity |
| `basic_state(cd, ...)`, `meridional_basic_state`, `nonaxisymmetric_basic_state` | Laplace conduction | Viscous Stokes–Coriolis balance |
| `basic_state_selfconsistent` (default) | Advection–diffusion | Navier–Stokes–Coriolis balance, including inertia |
| `basic_state_selfconsistent(...; momentum_model=:stokes)` | Advection–diffusion | Viscous Stokes–Coriolis balance |

The noniterated constructors deliberately neglect thermal advection and momentum
inertia. Increasing their resolution refines that approximation; use the
self-consistent solver when both effects are needed. Both models require
[independent radial and angular refinement](#Checking-spatial-resolution).
The automatic `lmax_bs` is a starting truncation chosen from the boundary
forcing, not a determination of the resolution needed by the resulting flow.

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

# Heterogeneous temperature or heat flux at the inner boundary
bs = basic_state(cd, χ, E, Ra, Pr; inner_temperature_bc=Y20(0.1))
bs = basic_state(cd, χ, E, Ra, Pr; inner_flux_bc=Y11(0.2))
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
            inner_temperature_bc = nothing,
            inner_flux_bc = nothing,
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
- `inner_temperature_bc`, `inner_flux_bc` : the same for the inner boundary (see
  [Inner-Boundary Forcing](@ref))
- `mechanical_bc` : `:no_slip` (default) or `:stress_free`
- `lmax_bs` : Maximum ``\ell`` for expansion (default `max(ℓ_bc + 2, 4)`, where
  `ℓ_bc` is the highest forced degree on either wall).
  An explicit value must retain every nonzero boundary mode; otherwise an
  `ArgumentError` is thrown rather than silently dropping the forcing.

**Automatic Dispatch:**

| Boundary Condition | Function Called | Returns |
|-------------------|-----------------|---------|
| None specified | `conduction_basic_state` | `BasicState` |
| Axisymmetric (``m=0`` only) | shared viscous solver | `BasicState` |
| Non-axisymmetric (``m \neq 0``) | `nonaxisymmetric_basic_state` | `BasicState3D` |

With an inner-boundary condition the harmonics of both walls decide the dispatch.

### Inner-Boundary Forcing

The inner boundary takes the same patterns as the outer one, as a temperature
(`inner_temperature_bc`) or a radial temperature gradient ``\partial\bar T/\partial r``
(`inner_flux_bc`), for example a laterally varying heat flux out of the inner core:

```julia
# Fixed inner temperature with an equator-pole contrast:
# T̄(r_i) = 1 + 0.1 P₂(cosθ)
bs = basic_state(cd, χ, E, Ra, Pr; inner_temperature_bc=Y20(0.1))

# Hemispherical heat flux at the inner boundary, with a warm outer equator
bs3d = basic_state(cd, χ, E, Ra, Pr; inner_flux_bc=Y11(0.2), temperature_bc=Y20(-0.05))
```

Without a `Y00` term the inner wall keeps its mean temperature ``\bar T = 1``, or
for a flux condition the conduction value
``\partial\bar T/\partial r|_{r_i} = -1/(\chi(1-\chi))``; a `Y00` term replaces it.
Fixed flux on both walls leaves the mean temperature undetermined and throws an
`ArgumentError`. The lower-level constructors take `inner_thermal_bc` together
with `inner_amplitudes` (`nonaxisymmetric_basic_state`,
`nonaxisymmetric_basic_state_selfconsistent`), `inner_amplitude` and
`inner_flux_mean` (`meridional_basic_state`), or `inner_flux`
(`conduction_basic_state`); `basic_state_selfconsistent` takes the same symbolic
keywords as `basic_state`.

Each basic state records its inner-wall condition in `inner_thermal_bc`.
Perturbations must satisfy the same physical condition: a fixed inner heat flux
requires `thermal_bc = (:fixed_flux, outer)` in `OnsetParams`, so that the
temperature perturbation has zero flux there; other combinations throw.

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
bs = basic_state(params; mode=:selfconsistent, max_iterations=50)  # throws unless the iteration converges
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)
```

The `mode` keyword selects the construction strategy:

| `mode` | Returns | Description |
|--------|---------|-------------|
| `:conduction` | `BasicState` | Pure conductive profile, no flow |
| `:meridional` | `BasicState` | Conductive temperature and axisymmetric Stokes–Coriolis flow |
| `:selfconsistent` | `BasicState3D` (`BasicState` if `mmax_bs=0`) | Nonlinear Navier–Stokes–Coriolis and thermal transport |
| `:nonaxisymmetric` | `BasicState3D` | Conductive temperature and 3D Stokes–Coriolis flow |

Each mode imposes its forcing as a temperature or a flux, following each wall of
`params.thermal_bc`: `amplitude` (default 0.05) sets the degree-2 pattern on the
outer wall and `inner_amplitude` (default 0) on the inner wall, added to the
conduction mean. `:meridional` uses the axisymmetric ``P_2`` pattern;
`:nonaxisymmetric` and `:selfconsistent` force the orders `m = 1…min(mmax_bs, 2)`
of degree 2 (defaults `mmax_bs=2`, `lmax_bs=4`), and the nonlinear solve retains
azimuthal orders through `lmax_bs`. At least one wall must have a fixed temperature.

```julia
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=30, Nr=64,
                     thermal_bc=(:fixed_flux, :fixed_temperature))
bs = basic_state(params; mode=:meridional, inner_amplitude=0.1)  # P₂ heat-flux pattern at r_i
```

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
basic state with perturbations whose conditions match on each wall, e.g.
`thermal_bc = (:fixed_temperature, :fixed_flux)` in `OnsetParams` for a fixed-flux
outer wall. The inner wall can also have a fixed flux; see
[Inner-Boundary Forcing](@ref).

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

Wrap the basic state and matching `OnsetParams` in the appropriate problem type,
`BiglobalProblem` for a `BasicState` or `TriglobalProblem` for a `BasicState3D`
(`OnsetProblem` rejects a basic state):

```julia
params = OnsetParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 12,
    lmax = 60,
    Nr = bs.Nr,  # must match the basic state's radial grid
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

### Matching mechanical boundary conditions

Mean states attached to a stability problem must satisfy its stationary-wall
conditions. Construction checks the physical velocity traces at both walls:
no-slip requires all three components to vanish; stress-free requires zero
radial velocity and zero tangential viscous traction. A mismatch raises an
`ArgumentError` before assembly. The same check applies to the direct biglobal,
triglobal, and `build_basic_state_operators` entry points, including every
azimuthal mode of a 3D state.

Construct the state and its perturbation problem with the same `mechanical_bc`.
The check uses the actual field rather than a stored boundary label, so a
motionless state is valid with either choice. Prescribed moving walls are not
part of this stationary-wall interface. Native `bs.flow` potentials determine
the velocity when present; custom component-only states are checked using their
radial profiles and derivatives of those profiles. If a state built with matching
conditions still fails, inspect the reported wall residual and increase numerical
precision or resolution.

Thermal forcing is chosen separately: for a mean state with fixed inner
temperature and outer flux, use
`thermal_bc=(:fixed_temperature, :fixed_flux)` in `OnsetParams`. Use the same
outer thermal model when constructing the state and its perturbations.

## Saving and Loading

Since base states can be expensive to compute, save them with JLD2:

```julia
using JLD2

# Save
@save "basic_states/meridional_l6.jld2" bs

# Load (restores the saved variable `bs`)
@load "basic_states/meridional_l6.jld2" bs

# Use in new problem
problem = BiglobalProblem(params, bs)
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

This example constructs the conductive-temperature/Stokes approximation and
inspects its modes. Check the resulting fields by refinement before using them
for quantitative predictions, particularly at small Ekman number.

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

The damped Picard iteration solves thermal transport implicitly, then solves
momentum with frozen nonlinear forcing. Temperature anomalies are advected by the
previous velocity, but the spherical mean temperature is advected by the flow that
the new temperature drives, so the linear buoyancy feedback (advection of the
mean temperature gradient by the buoyancy-driven flow) is solved exactly in each
step and only nonlinear transport is lagged. It uses
``-({\mathbf U}\cdot\nabla){\mathbf U}={\mathbf U}\times(\nabla\times{\mathbf U})
-\nabla(|{\mathbf U}|^2/2)`` and absorbs the gradient in pressure. Angular
products use an oversampled grid and the native vector-harmonic flow.
Backtracking accepts updates that reduce the fixed-point residual described below.
If the Picard iteration stalls, Newton–Krylov steps on the same fixed-point
equation take over (GMRES with finite-difference Jacobian products and a line
search). If those fail too, the boundary anomalies are raised from zero by
natural-parameter continuation, each step starting from the previous steady state.
`momentum_model=:stokes` explicitly retains the earlier weak-inertia model
while still solving thermal transport.

```julia
cd = ChebyshevDiffn(32, [0.35, 1.0], 4)
bs, info = basic_state_selfconsistent(cd, 0.35, 0.01, 30.0, 1.0;
    temperature_bc=Y20(0.01) + Y22(0.01),
    lmax_bs=8, max_iterations=50, tolerance=1e-9)
@assert info.iteration_converged
@assert info.spatial_convergence === :unchecked
println((info.momentum_residual, info.thermal_residual, info.boundary_residual))
v = mean_flow_velocity(bs, 0.7, pi/3, pi/8)
```

`info.iteration_converged` reports whether the momentum, heat, and boundary/gauge
residuals meet `tolerance` at the chosen truncation. `info.converged` is its
backward-compatible alias. Neither establishes spatial convergence:
`info.spatial_convergence` is always `:unchecked`, even when the iteration
converges or the velocity vanishes. Compare independently refined states as
shown below. The momentum and heat residuals are fixed-point residuals: the
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
failed line search). `info.newton_iterations` counts the Newton steps (zero when
Picard converges), `info.forcing_history` records the fraction of the boundary
anomalies at each step, and `info.forcing_reached` is the largest fraction with a
converged steady state. Every Picard or Newton step counts toward
`max_iterations`. `info.termination_reason` is `:converged`,
`:max_iterations`, or `:stagnation`.

When `info.iteration_converged=false`, the returned state is an incomplete iterate and
must not be treated as a steady equilibrium. A steady state need not exist: above
onset, the branch of steady states connected to weak forcing can end in a fold,
beyond which no steady state continues it. The solver then reports
`info.forcing_reached < 1`, the fraction of the requested anomalies it reached.
For example, with stress-free walls at `E = 1e-2` and `Ra = 5e3`, a `Y21 + Y22`
temperature forcing folds at an amplitude of about 0.005. Use weaker forcing, a
lower Rayleigh number, or the noniterated Stokes–Coriolis state there. Failure to
converge does not establish physical instability. The convenience call `basic_state(params;
mode=:selfconsistent)` throws on nonconvergence unless
`allow_unconverged=true` is explicitly requested.

Boundary conditions apply to every retained real harmonic, including sine
modes generated by transport. Fixed-flux problems retain homogeneous flux on
unforced modes. With no boundary forcing specified, the analytical conduction
shortcut retains its existing `info === nothing` return convention because no
iteration was performed. Explicit zero-flow boundary conditions return the same
diagnostic fields as other iterated states; axisymmetric forced states run the
coupled iteration too. For 3D,
`basic_state_selfconsistent` defaults to `mmax_bs=lmax_bs`, since nonlinear
products generate azimuthal orders absent from the boundary forcing. An
explicit `mmax_bs` may be used to control that truncation.

## Checking spatial resolution

Use `mean_flow_resolution(coarse, fine; rtol=1e-3, atol=0)` to compare two states
on the same shell. It measures physical volume ``L^2`` differences using the
native vector harmonics and spectral radial interpolation. Its `converged`
field requires every channel to satisfy `error ≤ atol + rtol * scale`, where
`scale` is the norm of that channel in the finer state.

The report contains `velocity.total`, `velocity.radial`, `velocity.poloidal`
(horizontal poloidal velocity), `velocity.toroidal` (horizontal toroidal
velocity), `temperature.total`, and `temperature.anomaly` (degrees ``\ell>0``).
Checking these separately prevents a dominant zonal flow or radial temperature
profile from concealing errors in the circulation or thermal anomalies. Each
channel exposes `converged`, `error`, `scale`, `tolerance`, and `relative_error`.
Zero-flow conduction states are supported; a nonzero custom velocity must have
native `flow` potentials.

Keep physical parameters, boundary forcing, and the momentum model fixed while
refining. Increase the radial and angular resolutions separately and also check
their combination. For a nonlinear 3D state, increase `mmax_bs` with `lmax_bs`
to retain newly generated azimuthal modes. This modest-forcing example checks
all four edges of a refinement grid; smaller Ekman numbers can require higher
radial and angular resolutions:

```julia
using Magrathea

function refinement_state(Nr, L)
    cd = ChebyshevDiffn(Nr, [0.35, 1.0], 4)
    state, info = basic_state_selfconsistent(cd, 0.35, 0.1, 30.0, 1.0;
        temperature_bc=Y20(0.01) + Y22(0.01),
        lmax_bs=L, mmax_bs=L, tolerance=1e-10, max_iterations=60)
    @assert info.iteration_converged
    return state
end

coarse = refinement_state(20, 6)
radial = refinement_state(28, 6)
angular = refinement_state(20, 8)
fine = refinement_state(28, 8)

checks = (
    radial_at_L6=mean_flow_resolution(coarse, radial; rtol=1e-3, atol=1e-12),
    radial_at_L8=mean_flow_resolution(angular, fine; rtol=1e-3, atol=1e-12),
    angular_at_N20=mean_flow_resolution(coarse, angular; rtol=1e-3, atol=1e-12),
    angular_at_N28=mean_flow_resolution(radial, fine; rtol=1e-3, atol=1e-12),
)
@assert all(check.converged for check in values(checks))
bs = fine
```

If any comparison fails, increase that resolution and repeat both directional
checks at the new finest state. Choose `rtol` and `atol` for the physical accuracy
needed; a passing comparison establishes agreement between those resolutions,
not an error bound against the continuum solution. The comparison does not
change `info.spatial_convergence` or validate the omitted terms of an approximate
physical model. In particular, a small iteration residual can coexist with a
substantial change of the mean flow when `lmax_bs` is increased.

## [Worked example: a Y₂₀ heat flux at the outer boundary](@id y20-heat-flux-example)

This example solves the self-consistent mean-flow equations of
[Self-Consistent Basic States with Advection](#Self-Consistent-Basic-States-with-Advection)
for a shell whose inner boundary is held at a uniform temperature,
``\bar T(r_i)=1``, while the outer boundary carries a prescribed heat flux with a
``Y_2^0`` pattern:

```math
\left.\frac{\partial\bar T}{\partial r}\right|_{r_o}=q_0\left[1+\epsilon P_2(\cos\theta)\right],
\qquad q_0=-\frac{\chi}{1-\chi},\qquad \epsilon=0.3.
```

``q_0`` is the conductive gradient, so the spherically averaged heat flux equals
that of the conduction state and only its latitude dependence drives a flow. With
``\epsilon>0`` the outer boundary loses 30% more heat than average at the poles and
15% less at the equator. Both boundaries are no-slip. With the normalization of
[Available Constructors](#Available-Constructors), `Y00(q₀)` sets the mean of
``\partial\bar T/\partial r`` and `Y20(ε * q₀)` adds ``\epsilon q_0 P_2(\cos\theta)``.
``Ra=3\times10^4`` is about 1.2 times the onset of convection in the conduction
state with fixed-temperature walls, ``Ra_c\approx2.4\times10^4`` at ``m=3``. The
steady state computed here therefore need not be stable; a
[biglobal analysis](analysis/biglobal_stability.md) determines whether it is.
The last line confirms the gradient imposed at the outer boundary:

```@example y20_flux
using Magrathea

χ, E, Ra, Pr = 0.35, 1e-3, 3e4, 1.0
q₀ = -χ / (1 - χ)                  # conductive ∂T̄/∂r at the outer boundary
ε = 0.3
flux = Y00(q₀) + Y20(ε * q₀)       # ∂T̄/∂r(r_o) = q₀[1 + ε P₂(cos θ)]

function y20_state(Nr, lmax_bs)
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr; flux_bc=flux,
        lmax_bs=lmax_bs, max_iterations=80, tolerance=1e-10)
    @assert info.iteration_converged
    return bs
end

bs = y20_state(32, 24)
(mean = bs.dtheta_dr_coeffs[0][end] / sqrt(4π), P₂ = bs.dtheta_dr_coeffs[2][end] * sqrt(5 / (4π)))
```

Refining each truncation separately, as in
[Checking spatial resolution](#Checking-spatial-resolution), changes the velocity
by these relative amounts:

```@example y20_flux
radial = mean_flow_resolution(bs, y20_state(40, 24); rtol=1e-3, atol=1e-14)
angular = mean_flow_resolution(bs, y20_state(32, 32); rtol=1e-3, atol=1e-14)
@assert radial.converged && angular.converged
(radial = radial.velocity.total.relative_error, angular = angular.velocity.total.relative_error)
```

Next, sample the state on a grid of radius ``r`` and colatitude ``\theta``.
[`mean_flow_velocity`](@ref) and [`mean_temperature`](@ref) evaluate the velocity
and temperature. Velocities are in units of ``\Omega r_o``; dividing by ``E``
converts them to the viscous unit ``\nu/r_o``, so they read as Reynolds numbers.
The temperature anomaly removes the spherical mean at each radius. The Stokes
streamfunction ``\psi``, with ``u_r=\partial_\theta\psi/(r^2\sin\theta)`` and
``u_\theta=-\partial_r\psi/(r\sin\theta)``, is integrated from the north pole.
It must vanish again at the south pole, which the last value checks:

```@example y20_flux
rs = range(χ, 1, 121)                    # radius
θs = range(0, π, 241)                    # colatitude
T = [mean_temperature(bs, r, θ) for r in rs, θ in θs]
u = [mean_flow_velocity(bs, r, θ) for r in rs, θ in θs]
ur = getfield.(u, :ur) ./ E              # units of ν/r_o
uφ = getfield.(u, :uphi) ./ E

w = sin.(θs)
T′ = T .- (T * w) ./ sum(w)              # remove the spherical mean at each radius
ψ = zeros(size(ur))                      # ψ = r² ∫₀^θ u_r sin θ′ dθ′
for k in 2:length(θs)
    ψ[:, k] = ψ[:, k-1] .+ rs .^ 2 .* (ur[:, k-1] .* w[k-1] .+ ur[:, k] .* w[k]) .* (step(θs) / 2)
end
X = rs .* sin.(θs)'                      # cylindrical radius s = r sin θ
Z = rs .* cos.(θs)'                      # height z = r cos θ
(uφ_range = extrema(uφ), ψ_south_pole = maximum(abs, ψ[:, end]) / maximum(abs, ψ))
```

Plot the outer heat flux and three meridional sections with CairoMakie:

```@example y20_flux
using CairoMakie

function shell!(ax)
    φ = range(-π / 2, π / 2, 200)
    for a in (χ, 1)
        lines!(ax, a .* cos.(φ), a .* sin.(φ); color=:black, linewidth=1.2)
    end
    lines!(ax, [0, 0, NaN, 0, 0], [χ, 1, NaN, -χ, -1]; color=:black, linewidth=1.2)
    zt = sqrt(1 - χ^2)                   # the tangent cylinder s = χ
    lines!(ax, [χ, χ], [-zt, zt]; color=(:gray25, 0.7), linewidth=0.9, linestyle=:dash)
end

limits = ((-0.03, 1.03), (-1.03, 1.03))
aspect = AxisAspect(1.06 / 2.06)         # keeps the sections circular

function section!(fig, col, F, title, colormap, label)
    m = maximum(abs, F)
    ax = Axis(fig[2, col]; title, limits, aspect)
    hidedecorations!(ax); hidespines!(ax)
    contourf!(ax, X, Z, F; levels=range(-m, m, 22), colormap)
    lv = range(-m, m, 12)[2:end-1]       # solid positive and dashed negative contours
    contour!(ax, X, Z, F; levels=filter(>(0), lv), color=(:black, 0.5), linewidth=0.6)
    contour!(ax, X, Z, F; levels=filter(<(0), lv), color=(:black, 0.5), linewidth=0.6,
             linestyle=:dash)
    shell!(ax)
    Colorbar(fig[3, col]; colormap, limits=(-m, m), label, vertical=false, flipaxis=false,
             ticks=WilkinsonTicks(3), width=Relative(0.85))
    return ax
end

fig = Figure(size=(960, 580), fontsize=14)
Label(fig[0, 1:4], "Steady mean flow driven by a Y₂₀ heat flux at the outer boundary";
      fontsize=17, font=:bold, tellwidth=false)
Label(fig[1, 1:4], L"\chi=0.35,\; E=10^{-3},\; Ra=3\times10^{4},\; Pr=1;\quad \bar{T}(r_i)=1,\quad \partial_r\bar{T}(r_o)=q_0\,[1+0.3\,P_2(\cos\theta)]";
      color=:gray25, tellwidth=false)

# Heat flux out of the outer boundary, q = -∂T̄/∂r, against the height z = cos θ
# so that it lines up with the outer boundary of the sections.
axq = Axis(fig[2, 1]; title="Outer heat flux", limits=((0.75, 1.4), limits[2]), aspect,
           xlabel=L"q/\bar{q}", xticks=[0.85, 1, 1.3], xgridvisible=false,
           ygridvisible=false,
           yticks=(sind.(-90:30:90), ["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"]))
μ = range(-1, 1, 201)
lines!(axq, 1 .+ ε .* (3 .* μ .^ 2 .- 1) ./ 2, μ; color=:firebrick, linewidth=2.5)
vlines!(axq, 1; color=:gray50, linestyle=:dash, linewidth=1)

section!(fig, 2, T′, "Temperature anomaly", :balance, L"\bar{T}-\langle\bar{T}\rangle")
section!(fig, 3, uφ, "Zonal flow", :PuOr, L"\bar{u}_\phi\, r_o/\nu")
axψ = section!(fig, 4, 1e3 .* ψ, "Meridional circulation", :PRGn, L"10^3\,\psi/(\nu r_o)")

# Direction of the meridional velocity (u_s, u_z) on a coarse grid of points.
points, angles, speeds = Point2f[], Float64[], Float64[]
for s in 0.06:0.11:0.98, z in -0.935:0.11:0.935   # symmetric about the equator
    r, θ = hypot(s, z), atan(s, z)
    χ + 0.04 < r < 0.97 || continue
    v = mean_flow_velocity(bs, r, θ)
    us = v.ur * sin(θ) + v.utheta * cos(θ)
    uz = v.ur * cos(θ) - v.utheta * sin(θ)
    push!(points, Point2f(s, z)); push!(angles, atan(uz, us)); push!(speeds, hypot(us, uz))
end
keep = speeds .> 0.03 * maximum(speeds)  # omit the nearly stagnant cell centres
scatter!(axψ, points[keep]; marker=:rtriangle, rotation=angles[keep], markersize=8,
         color=(:black, 0.8))

colgap!(fig.layout, 16)
save("y20_mean_flow.png", fig; px_per_unit=2)
nothing # hide
```

![Outer heat flux and meridional sections of the temperature anomaly, zonal flow and meridional circulation of the steady state driven by a Y₂₀ heat flux](y20_mean_flow.png)

From left to right, the figure shows the imposed outer heat flux and meridional
sections of the temperature anomaly, the zonal flow and the meridional
circulation. The rotation axis is vertical and the dashed vertical line marks the
tangent cylinder ``s=r_i``. Solid contours are positive and dashed contours
negative; the triangles give the direction of the meridional flow.

- **Temperature.** The extra heat loss cools the polar regions
  (``\bar T-\langle\bar T\rangle\approx-0.09`` at the poles) and the deficit warms
  the equator (``\approx+0.04``). The anomaly vanishes on the isothermal inner
  boundary.
- **Zonal flow.** Outside the Ekman layers the flow is in thermal-wind balance,
  ``2\,\partial_z\bar u_\phi=\beta\,\partial_\theta\bar T`` with
  ``\beta=Ra\,E^2/(Pr(1-\chi)^3)``, which this state satisfies to within 2% in the
  interior. With a warm equator, ``\bar u_\phi`` increases away from the equatorial
  plane in both hemispheres. The flow is retrograde at low latitudes
  (``\bar u_\phi r_o/\nu\approx-1.9``) and prograde at high latitudes near the
  outer boundary (``\approx+0.8``).
- **Meridional circulation.** There is one cell in each hemisphere,
  counter-clockwise in the north (``\psi<0``) and its mirror image in the south.
  Cold fluid sinks at high latitudes, flows towards the equator in the Ekman layer
  of the inner boundary, rises at low latitudes and returns poleward in the Ekman
  layer of the outer boundary. Its peak speed is about a sixth of that of the
  zonal flow.

The Reynolds and Péclet numbers are about 2, so inertia and heat advection are
weak but not negligible. The spherical-mean temperature departs from conduction
by only ``3\times10^{-4}``. However, the noniterated Stokes–Coriolis state,
`basic_state(cd, χ, E, Ra, Pr; flux_bc=flux, lmax_bs=24)`, underestimates the zonal
flow by about 10% and the meridional circulation by about 20%. To change the
forcing, edit `flux`. For example, `ε < 0` puts the extra heat loss at the
equator and, to leading order, reverses the anomaly and both flows.

## [Worked example: a Y₂₂ heat flux and a 3-D basic state](@id y22-heat-flux-example)

A heat flux that varies with longitude gives a `BasicState3D`, the basic state of
a [tri-global analysis](triglobal.md). This example keeps the uniform inner
temperature ``\bar T(r_i)=1`` and imposes a ``Y_2^2`` heat-flux pattern at the
outer boundary:

```math
\left.\frac{\partial\bar T}{\partial r}\right|_{r_o}=q_0\left[1+\epsilon\sin^2\theta\cos 2\phi\right],
\qquad q_0=-\frac{\chi}{1-\chi},\qquad \epsilon=0.3.
```

`Y22(a)` is ``3a\sin^2\theta\cos 2\phi``, so the pattern is `Y22(ε * q₀ / 3)`.
Along the equator, the outer boundary loses 30% more heat than average at
longitudes 0° and 180° and 30% less at ±90°. The 3-D heat step solves a dense
system with ``(\ell_{\max}+1)^2N_r`` unknowns. This example therefore uses
``E=10^{-2}``, where `lmax_bs = mmax_bs = 8` is enough for a figure.
``Ra=3\times10^3`` is about 0.6 of the onset of convection in the conduction
state with fixed-temperature walls, ``Ra_c\approx4.8\times10^3`` at ``m=3``. The
last line confirms the imposed gradient:

```@example y22_flux
using Magrathea

χ, E, Ra, Pr = 0.35, 1e-2, 3e3, 1.0
q₀ = -χ / (1 - χ)                  # conductive ∂T̄/∂r at the outer boundary
ε = 0.3
flux = Y00(q₀) + Y22(ε * q₀ / 3)   # ∂T̄/∂r(r_o) = q₀[1 + ε sin²θ cos 2φ]

function y22_state(Nr, lmax_bs)
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    bs, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr; flux_bc=flux,
        lmax_bs=lmax_bs, mmax_bs=lmax_bs, max_iterations=80, tolerance=1e-10)
    @assert info.iteration_converged
    return bs
end

bs = y22_state(24, 8)
(state = typeof(bs), mean = bs.dtheta_dr_coeffs[(0, 0)][end] / sqrt(4π),
 Y22 = bs.dtheta_dr_coeffs[(2, 2)][end] * sqrt(10 / (4π)))
```

The radial truncation is converged to round-off. Raising `lmax_bs` and
`mmax_bs` to 10 changes the velocity by a few parts in a thousand:

```@example y22_flux
radial = mean_flow_resolution(bs, y22_state(32, 8); rtol=1e-2, atol=1e-14)
angular = mean_flow_resolution(bs, y22_state(24, 10); rtol=1e-2, atol=1e-14)
@assert radial.converged && angular.converged
(radial = radial.velocity.total.relative_error, angular = angular.velocity.total.relative_error)
```

Sample the equatorial plane ``\theta=\pi/2`` on a grid of radius and longitude
with [`mean_temperature`](@ref) and [`mean_flow_velocity`](@ref). The temperature
anomaly removes the azimuthal mean at each radius. The forcing is symmetric
about the equator, so ``\bar u_\theta`` vanishes there and the flow in this
plane is horizontal, as the last value confirms:

```@example y22_flux
rs = range(χ, 1, 61)                     # radius
φs = range(0, 2π, 241)                   # longitude
T = [mean_temperature(bs, r, π / 2, φ) for r in rs, φ in φs]
u = [mean_flow_velocity(bs, r, π / 2, φ) for r in rs, φ in φs]
ur = getfield.(u, :ur) ./ E              # units of ν/r_o
uφ = getfield.(u, :uphi) ./ E
T′ = T .- sum(T[:, 1:end-1]; dims=2) ./ (length(φs) - 1)   # remove the azimuthal mean
X = rs .* cos.(φs)'                      # the plane z = 0, seen from the north
Y = rs .* sin.(φs)'
(T′_range = extrema(T′), uθ_over_uφ = maximum(v -> abs(v.utheta), u) / maximum(v -> abs(v.uphi), u))
```

Plot the outer heat flux on a Hammer equal-area map and three equatorial sections:

```@example y22_flux
using CairoMakie

q(θ, φ) = 1 + ε * sin(θ)^2 * cos(2φ)     # heat flux out of the outer boundary / its mean
qmap = cgrad(:balance; rev=true)         # blue where more heat is lost
qrange = (1 - ε, 1 + ε)

function equatorial_section!(fig, col, F, title, colormap, label)
    m = maximum(abs, F)
    ax = Axis(fig[4, col]; title, aspect=DataAspect(), limits=((-1.1, 1.1), (-1.1, 1.1)))
    hidedecorations!(ax); hidespines!(ax)
    contourf!(ax, X, Y, F; levels=range(-m, m, 22), colormap)
    lv = range(-m, m, 12)[2:end-1]       # solid positive and dashed negative contours
    contour!(ax, X, Y, F; levels=filter(>(0), lv), color=(:black, 0.5), linewidth=0.6)
    contour!(ax, X, Y, F; levels=filter(<(0), lv), color=(:black, 0.5), linewidth=0.6,
             linestyle=:dash)
    t = range(0, 2π, 361)
    for a in (χ, 1)
        lines!(ax, a .* cos.(t), a .* sin.(t); color=:black, linewidth=1.2)
    end
    # The heat flux along the equator, as a ring around the shell
    lines!(ax, 1.055 .* cos.(t), 1.055 .* sin.(t); color=q.(π / 2, t), colormap=qmap,
           colorrange=qrange, linewidth=5)
    Colorbar(fig[5, col]; colormap, limits=(-m, m), label, vertical=false, flipaxis=false,
             ticks=WilkinsonTicks(3), width=Relative(0.8))
    return ax
end

fig = Figure(size=(960, 780), fontsize=14)
Label(fig[0, 1:3], "Steady mean flow driven by a Y₂₂ heat flux at the outer boundary";
      fontsize=17, font=:bold, tellwidth=false)
Label(fig[1, 1:3], L"\chi=0.35,\; E=10^{-2},\; Ra=3\times10^{3},\; Pr=1;\quad \bar{T}(r_i)=1,\quad \partial_r\bar{T}(r_o)=q_0\,[1+0.3\,\sin^2\theta\cos 2\phi]";
      color=:gray25, tellwidth=false)

# Hammer equal-area map of the outer boundary, longitude λ and latitude ϕ
hammer(λ, ϕ) = (d = sqrt(1 + cos(ϕ) * cos(λ / 2));
                Point2f(2√2 * cos(ϕ) * sin(λ / 2) / d, √2 * sin(ϕ) / d))
λs = range(-π, π, 181)
ϕs = range(-π / 2, π / 2, 91)
P = [hammer(λ, ϕ) for λ in λs, ϕ in ϕs]
top = fig[2, 1:3] = GridLayout()
axm = Axis(top[1, 1]; title="Heat flux out of the outer boundary", aspect=DataAspect(),
           limits=((-2.9, 2.9), (-1.62, 1.45)), width=440, height=233)
hidedecorations!(axm); hidespines!(axm)
contourf!(axm, first.(P), last.(P), [q(π / 2 - ϕ, λ) for λ in λs, ϕ in ϕs];
          levels=range(qrange..., 13), colormap=qmap)
for λ in -2π/3:π/3:2π/3                  # meridians, labelled below the map
    lines!(axm, [hammer(λ, ϕ) for ϕ in ϕs]; color=(:white, 0.7), linewidth=0.6)
    text!(axm, hammer(λ, 0)[1], -√2; text="$(round(Int, rad2deg(λ)))°", fontsize=11,
          align=(:center, :top), offset=(0, -4), color=:gray25)
end
for ϕ in (-π/3, -π/6, π/6, π/3)          # parallels
    lines!(axm, [hammer(λ, ϕ) for λ in λs]; color=(:white, 0.7), linewidth=0.6)
end
lines!(axm, [hammer(λ, 0) for λ in λs]; color=:black, linewidth=1.2, linestyle=:dash)
lines!(axm, vcat([hammer(-π, ϕ) for ϕ in ϕs], [hammer(π, ϕ) for ϕ in reverse(ϕs)],
                 [hammer(-π, -π / 2)]); color=:black, linewidth=1.2)
Colorbar(top[1, 2]; colormap=qmap, limits=qrange, label=L"q/\bar{q}", height=180,
         ticks=[0.7, 0.85, 1, 1.15, 1.3])

Label(fig[3, 1:3], "Equatorial plane z = 0 seen from the north, rotating counter-clockwise; " *
      "the outer ring repeats the heat flux along the equator"; color=:gray25, tellwidth=false)
equatorial_section!(fig, 1, T′, "Temperature anomaly", :balance,
                    L"\bar{T}-\langle\bar{T}\rangle_\phi")
equatorial_section!(fig, 2, ur, "Radial velocity", :PRGn, L"\bar{u}_r\, r_o/\nu")
axφ = equatorial_section!(fig, 3, uφ, "Azimuthal velocity", :PuOr, L"\bar{u}_\phi\, r_o/\nu")

# Direction of the horizontal flow (u_x, u_y) on a coarse grid of points.
points, angles, speeds = Point2f[], Float64[], Float64[]
for x in -0.96:0.12:0.96, y in -0.96:0.12:0.96
    r, φ = hypot(x, y), atan(y, x)
    χ + 0.04 < r < 0.97 || continue
    v = mean_flow_velocity(bs, r, π / 2, φ)
    ux = v.ur * cos(φ) - v.uphi * sin(φ)
    uy = v.ur * sin(φ) + v.uphi * cos(φ)
    push!(points, Point2f(x, y)); push!(angles, atan(uy, ux)); push!(speeds, hypot(ux, uy))
end
keep = speeds .> 0.03 * maximum(speeds)  # omit nearly stagnant points
scatter!(axφ, points[keep]; marker=:rtriangle, rotation=angles[keep], markersize=7,
         color=(:black, 0.75))

rowsize!(fig.layout, 4, Aspect(1, 1))    # sections as tall as they are wide
rowgap!(fig.layout, 3, 4)
save("y22_mean_flow.png", fig; px_per_unit=2)
nothing # hide
```

![Hammer map of the Y₂₂ outer heat flux and equatorial sections of the temperature anomaly, radial velocity and azimuthal velocity of the steady state](y22_mean_flow.png)

The map shows the imposed heat flux over the whole outer boundary. Blue marks
regions that lose more heat than average and cool the fluid beneath them. The
dashed line is the equator, where the sections below are taken. The sections are
seen from the north, with longitude 0° to the right and increasing
counter-clockwise, the direction of rotation. The ring around each section
repeats the heat flux along the equator, and the triangles show the direction of
the horizontal flow.

- **Temperature.** Cold anomalies form beneath the high-flux longitudes and warm
  ones beneath the low-flux longitudes, with
  ``\bar T-\langle\bar T\rangle_\phi`` between −0.13 and +0.10. Heat advection
  shifts them 20–35° west (clockwise); the azimuthal-mean flow in this plane is
  retrograde, reaching ``\langle\bar u_\phi\rangle_\phi r_o/\nu\approx-0.5``.
- **Radial velocity.** Warm fluid moves outwards and cold fluid inwards: the
  ``m=2`` part of ``\bar u_r`` lies within about 6° of longitude of the
  temperature anomaly, with ``\bar u_r r_o/\nu`` between −2.6 and +2.0.
- **Azimuthal velocity.** Prograde and retrograde jets alternate around the
  shell, with ``\bar u_\phi r_o/\nu`` between −3.2 and +2.8.

Heat advection matters here. The self-consistent flow is about three times
stronger than in the noniterated Stokes–Coriolis state,
`basic_state(cd, χ, E, Ra, Pr; flux_bc=flux, lmax_bs=8)`, which keeps the
conductive temperature. The difference comes from heat transport rather than
momentum inertia: `momentum_model=:stokes` changes the peak velocity by about 1%.
The state `bs` is the input of a tri-global stability analysis: see
[Tri-Global Analysis](triglobal.md).

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
- **[MHD with mean flows](mhd_user_guide.md#Mean-flows:-biglobal-and-triglobal-MHD)** - Basic states with an imposed field and the induced field of their flow

---

!!! info "Example Scripts"
    See the following examples in the `example/` directory:

    - `basic_state_onset_example.jl` - Critical Rayleigh numbers on conduction and meridional (Y₂₀) states
    - `nonaxisymmetric_basic_state.jl` - 3D conduction/Stokes state with Y₂₀, Y₂₂ and Y₃₁ forcing and resolution checks
    - `flux_bc_mean_flow.jl` - Non-axisymmetric heat flux (Y₂₂) with the self-consistent nonlinear solver
    - `flux_bc_axisymmetric_flow.jl` - Axisymmetric heat flux (Y₂₀)
