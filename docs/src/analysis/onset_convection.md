# Onset Convection with No Mean Flow

This section covers the classical linear stability analysis for the onset of thermal convection in rotating spherical shells without any background mean flow. This is the simplest and most fundamental case, where the base state consists only of a conductive temperature profile.

## Physical Problem

### The Basic Setup

Consider a spherical shell of fluid bounded by concentric spheres at radii ``r_i`` (inner) and ``r_o`` (outer), with shell thickness ``L = r_o - r_i``. The system rotates at angular velocity ``\boldsymbol{\Omega} = \Omega \hat{\mathbf{z}}`` about the vertical axis. The inner boundary is maintained at a higher temperature than the outer boundary, creating an unstable temperature stratification.

The **base state** for this problem is:

- **Velocity**: ``\bar{\mathbf{u}} = \mathbf{0}`` (no mean flow)
- **Temperature**: Pure conductive profile ``\bar{T}(r)``
- **Pressure**: Hydrostatic equilibrium

### Conductive Temperature Profile

The conductive base state satisfies ``\nabla^2 \bar{T} = 0`` with fixed temperature boundary conditions:

```math
\bar{T}(r) = \frac{r_o/r - 1}{r_o/r_i - 1} = \frac{\chi}{1-\chi} \left( \frac{r_o}{r} - 1 \right)
```

where ``\chi = r_i/r_o`` is the radius ratio. The temperature gradient:

```math
\frac{d\bar{T}}{dr} = -\frac{r_i r_o}{(r_o - r_i) r^2} = -\frac{\chi}{(1-\chi)^2} \frac{1}{r^2}
```

drives convection when ``Ra`` exceeds the critical value.

### Linearized Perturbation Equations

We consider small perturbations ``(\mathbf{u}', p', \Theta')`` about the base state, where ``\Theta = T - \bar{T}``. The linearized non-dimensional equations are:

**Momentum equation:**
```math
\frac{\partial \mathbf{u}'}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u}' = -\nabla p' + E \nabla^2 \mathbf{u}' + \frac{Ra \cdot E^2}{Pr} \Theta' \hat{\mathbf{r}}
```

**Energy equation:**
```math
\frac{\partial \Theta'}{\partial t} + u_r' \frac{d\bar{T}}{dr} = \frac{E}{Pr} \nabla^2 \Theta'
```

**Continuity:**
```math
\nabla \cdot \mathbf{u}' = 0
```

### Eigenvalue Problem

For single-mode analysis, we seek solutions of the form:

```math
\psi(r, \theta, \phi, t) = e^{im\phi} e^{\sigma t} \sum_\ell \psi_\ell(r) Y_\ell^m(\theta)
```

This leads to the generalized eigenvalue problem:

```math
\mathbf{A} \mathbf{x} = \sigma \mathbf{B} \mathbf{x}
```

where ``\sigma = \sigma_r + i\omega`` is the complex eigenvalue:

| Component | Physical Meaning |
|-----------|------------------|
| ``\sigma_r > 0`` | Unstable (growing perturbation) |
| ``\sigma_r = 0`` | Marginal stability (onset) |
| ``\sigma_r < 0`` | Stable (decaying perturbation) |
| ``\omega`` | Drift frequency (pattern rotation) |

## Implementation in Magrathea.jl

### Step 1: Define Parameters

```julia
using Magrathea

# Define problem parameters
params = OnsetParams(
    # Physical parameters
    E = 1e-5,              # Ekman number
    Pr = 1.0,              # Prandtl number
    Ra = 1e7,              # Rayleigh number (initial guess)

    # Geometry
    χ = 0.35,              # Radius ratio r_i/r_o

    # Spectral resolution
    m = 10,                # Azimuthal wavenumber
    lmax = 60,             # Maximum spherical harmonic degree
    Nr = 64,               # Radial resolution

    # Boundary conditions
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

### Step 2: Build the Linear Stability Operator

```julia
# Build the operator (no basic state = conduction profile)
op = LinearStabilityOperator(params)

# Inspect operator properties
println("Total degrees of freedom: ", op.total_dof)
println("Matrix sparsity: ", 1 - nnz(op.A) / length(op.A))
```

!!! note "No BasicState Required"
    When no `basic_state` argument is provided to `OnsetParams`, Magrathea.jl automatically uses the conductive temperature profile with zero mean flow. This is the default onset convection problem.

### Step 3: Find Leading Eigenvalues

```julia
problem = OnsetProblem(params)
estimate_size(problem)   # print DOFs and estimated memory before solving

result = solve(problem; nev=6)

println("Growth rate: ", result.growth_rate)
println("Frequency:   ", result.frequency)

if result.growth_rate > 0
    println("System is UNSTABLE at Ra = $(params.Ra)")
else
    println("System is STABLE at Ra = $(params.Ra)")
end
```

### Step 4: Find Critical Rayleigh Number

The critical Rayleigh number ``Ra_c`` is the minimum ``Ra`` at which convection onsets:

```julia
# Search for critical Rayleigh number
Ra_c, ω_c, eigvec = find_critical_rayleigh(
    E = params.E,
    Pr = params.Pr,
    χ = params.χ,
    m = params.m,
    lmax = params.lmax,
    Nr = params.Nr;
    Ra_guess = 1e7,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
    nev = 6,
    tol = 1e-8,
)

@info "Critical parameters for m = $(params.m)" Ra_c ω_c
```

### Step 5: Sweep Over Azimuthal Modes

The global critical Rayleigh number requires finding the minimum across all ``m``:

```julia
function find_global_critical(; E, Pr, χ, lmax, Nr, m_range)
    results = Dict{Int, NamedTuple}()

    for m in m_range
        try
            Ra_c, ω_c, _ = find_critical_rayleigh(
                E = E, Pr = Pr, χ = χ, m = m,
                lmax = max(lmax, m + 10),
                Nr = Nr,
                Ra_guess = 1e6,
            )
            results[m] = (Ra_c = Ra_c, ω_c = ω_c)
        catch err
            @warn "Failed for m=$m" err
            results[m] = (Ra_c = NaN, ω_c = NaN)
        end
    end

    # Find global minimum
    valid_results = filter(p -> !isnan(p.second.Ra_c), results)
    m_c = argmin(m -> results[m].Ra_c, keys(valid_results))

    return (m_c = m_c, Ra_c = results[m_c].Ra_c, ω_c = results[m_c].ω_c, all_results = results)
end

# Example usage
critical = find_global_critical(
    E = 1e-5, Pr = 1.0, χ = 0.35,
    lmax = 60, Nr = 64, m_range = 5:25
)

println("Global critical: m_c = $(critical.m_c), Ra_c = $(critical.Ra_c)")
```

## Scaling Laws

At low Ekman number, theoretical asymptotic analysis predicts power-law scalings:

### Critical Rayleigh Number

```math
Ra_c \sim C_{Ra} \cdot E^{-4/3}
```

where ``C_{Ra}`` depends on geometry (radius ratio ``\chi``) and boundary conditions.

### Critical Azimuthal Wavenumber

```math
m_c \sim C_m \cdot E^{-1/3}
```

indicating finer azimuthal structure at lower Ekman numbers.

### Drift Frequency

```math
\omega_c \sim C_\omega \cdot E^{-2/3}
```

(with rotational time scaling) or ``\omega_c \sim C_\omega \cdot E^{2/3}`` (with viscous time scaling).

### Convection Column Width

```math
\delta \sim C_\delta \cdot E^{1/3}
```

The convective columns (thermal Rossby waves) become thinner at lower Ekman numbers.

## Validation Against Published Benchmarks

Magrathea.jl reproduces the published onset values for rotating spherical-shell
convection at radius ratio ``\chi = 0.35`` (no-slip, fixed-temperature,
differential heating, ``Pr = 1``) to **four significant figures**. The reference
values are from the **Kore** code (Barik et al. 2023) and the weakly-nonlinear
onset analysis of the same problem, both tabulated in the shell-thickness
(``d``-based) non-dimensionalization with modified Rayleigh number
``\widetilde{Ra} = Ra \cdot Ek_d``.

Because Magrathea.jl uses an ``r_o``-based Ekman number (see the convention note in
[Mathematical Foundations](../theory/mathematical_foundations.md)), the
literature Ekman number is mapped as ``E = Ek_d \,(1-\chi)^2``, with ``(1-\chi)^2 = 0.4225``:

| ``Ek_d`` (literature) | ``E`` passed to Magrathea.jl | mode ``m`` | ``\widetilde{Ra}`` (Magrathea) | ``\widetilde{Ra}_c`` (ref) | ``\omega`` (Magrathea) | ``\omega_c`` (ref) |
|---|---|---|---|---|---|---|
| ``10^{-3}`` | ``4.225\times10^{-4}`` | ``4`` (global min) | **55.905** | 55.9 | ``-0.02309`` | ``-0.0231`` |
| ``10^{-4}`` | ``4.225\times10^{-5}`` | ``5`` (= ref ``m_c``) | **75.246** | 75.2 | ``-0.01123`` | ``-0.0112`` |

At ``Ek_d = 10^{-3}`` the global critical mode is reproduced exactly (``m_c = 4``,
fully resolution-converged), with the critical Rayleigh number matching to 0.01%
and the drift frequency ``\omega_c`` (in units of ``\Omega``, hence
convention-independent) to 0.04%. At ``Ek_d = 10^{-4}`` the critical Rayleigh
number and drift at the reference critical mode ``m = 5`` match to 0.06% and 0.3%.
This validates the full onset pipeline end to end — the Coriolis, viscous,
buoyancy and thermal-advection operators, the ultraspherical radial
discretization, and the shift-invert eigensolver.

!!! note "Resolution at low Ekman number"
    Global critical-mode selection becomes resolution-sensitive as ``E`` decreases
    (the critical wavenumber grows as ``m_c \sim E^{-1/3}`` and the Ekman boundary
    layer thins as ``E^{1/3}``). At ``Ek_d = 10^{-4}``, modes with ``m > 5`` require
    larger `lmax`/`Nr` to avoid spuriously low critical Rayleigh numbers; use a
    convergence check (vary `lmax`, `Nr`) before trusting the global ``m_c`` at low ``E``.

Reproducing the ``Ek_d = 10^{-3}`` row:

```julia
using Magrathea

χ   = 0.35
gap = 1 - χ
Ek_d = 1e-3
E    = Ek_d * gap^2          # r_o-based Ekman for Magrathea.jl

# scan azimuthal modes; the minimum is the global critical mode
for m in 3:6
    Ra_c, ω_c, _ = Magrathea.find_critical_rayleigh(E, 1.0, χ, m, 24, 36;
        Ra_guess = 5e4, Ra_bracket = (5e3, 5e5), nev = 8)
    Ra_tilde = Ra_c * Ek_d   # modified Rayleigh, comparable to literature
    println("m=$m  R̃a=$(round(Ra_tilde, digits=3))  ω=$(round(ω_c, digits=5))")
end
# → minimum at m=4 with R̃a ≈ 55.9, ω ≈ -0.0231 (matches Barik et al. 2023)
```

## Boundary Condition Effects

### Mechanical Boundary Conditions

| Type | Mathematical Form | Physical Meaning |
|------|-------------------|------------------|
| No-slip | ``\mathbf{u} = 0`` | Fluid adheres to boundary |
| Stress-free | ``P=0``, ``r \partial_r^2 P = 0``, ``-r \partial_r T + T = 0`` | Zero tangential stress |

**No-slip** boundaries are appropriate for:
- Solid inner cores (Earth, terrestrial planets)
- Laboratory experiments with rigid walls

**Stress-free** boundaries are appropriate for:
- Gas-liquid interfaces
- Some stellar convection models

```julia
# No-slip example
params_noslip = OnsetParams(..., mechanical_bc = :no_slip)

# Stress-free example
params_stressfree = OnsetParams(..., mechanical_bc = :stress_free)
```

### Thermal Boundary Conditions

| Type | Mathematical Form | Physical Meaning |
|------|-------------------|------------------|
| Fixed temperature | ``\Theta = 0`` | Isothermal boundary |
| Fixed flux | ``\partial\Theta/\partial r = 0`` | Insulating boundary |

**Fixed temperature** is standard for:
- Core-mantle boundary (CMB)
- Most laboratory setups

**Fixed flux** may be used for:
- Outer boundary with specified heat flux
- Some stellar models

## Mode Structure

### Equatorial Symmetry

Onset modes typically have equatorial symmetry:

- **Equatorially symmetric (ES)**: ``u_r(\theta) = u_r(\pi - \theta)``
- **Equatorially antisymmetric (EA)**: ``u_r(\theta) = -u_r(\pi - \theta)``

In most cases, the critical mode is equatorially symmetric.

### Thermal Rossby Waves

At onset, the convective modes are quasi-geostrophic **thermal Rossby waves**:

- Columnar structure aligned with the rotation axis
- Prograde drift (in the direction of rotation)
- Localized near the tangent cylinder (for small ``\chi``)

### Reconstructing Physical Fields

```julia
# Slice eigenvector into spectral coefficients
poloidal = Dict{Int, Vector{ComplexF64}}()
toroidal = Dict{Int, Vector{ComplexF64}}()
temperature = Dict{Int, Vector{ComplexF64}}()

for ℓ in op.l_sets[:P]
    poloidal[ℓ] = eigvec[op.index_map[(ℓ, :P)]]
end
for ℓ in op.l_sets[:T]
    toroidal[ℓ] = eigvec[op.index_map[(ℓ, :T)]]
end
for ℓ in op.l_sets[:Θ]
    temperature[ℓ] = eigvec[op.index_map[(ℓ, :Θ)]]
end

# If you already have P(r,θ) and T(r,θ) on a grid, you can compute velocities:
# u_r, u_θ, u_φ = potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)
```

## Complete Example

```julia
#!/usr/bin/env julia
# onset_convection_analysis.jl
#
# Classical onset of convection in a rotating spherical shell
# with conductive basic state and no mean flow

using Magrathea
using JLD2
using Printf

# === Physical Parameters ===
E = 1e-5           # Ekman number
Pr = 1.0           # Prandtl number
χ = 0.35           # Radius ratio

# === Numerical Parameters ===
lmax = 60          # Maximum spherical harmonic degree
Nr = 64            # Radial resolution
m_range = 5:25     # Azimuthal modes to scan

# === Sweep over m ===
println("="^60)
println("Onset Convection Analysis (No Mean Flow)")
println("E = $E, Pr = $Pr, χ = $χ")
println("="^60)

results = []

for m in m_range
    @printf("m = %2d: ", m)

    try
        Ra_c, ω_c, eigvec = find_critical_rayleigh(
            E = E, Pr = Pr, χ = χ, m = m,
            lmax = max(lmax, m + 10),
            Nr = Nr,
            Ra_guess = 1e7,
            mechanical_bc = :no_slip,
            thermal_bc = :fixed_temperature,
        )

        push!(results, (m=m, Ra_c=Ra_c, ω_c=ω_c))
        @printf("Ra_c = %.6e, ω_c = %+.6f\n", Ra_c, ω_c)

    catch err
        push!(results, (m=m, Ra_c=NaN, ω_c=NaN))
        @printf("FAILED\n")
    end
end

# === Find Global Critical ===
valid = filter(r -> !isnan(r.Ra_c), results)

if !isempty(valid)
    critical = argmin(r -> r.Ra_c, valid)

    println("\n" * "="^60)
    println("GLOBAL CRITICAL PARAMETERS")
    println("="^60)
    @printf("  Critical mode:      m_c  = %d\n", critical.m)
    @printf("  Critical Rayleigh:  Ra_c = %.8e\n", critical.Ra_c)
    @printf("  Drift frequency:    ω_c  = %+.8f\n", critical.ω_c)

    # Compare to scaling laws
    Ra_scaling = critical.Ra_c * E^(4/3)
    m_scaling = critical.m * E^(1/3)
    @printf("\n  Scaling coefficients:\n")
    @printf("    Ra_c · E^(4/3) = %.4f\n", Ra_scaling)
    @printf("    m_c · E^(1/3)  = %.4f\n", m_scaling)
end

# === Save Results ===
@save "outputs/onset_convection_E$(E)_chi$(χ).jld2" results E Pr χ
println("\nResults saved to outputs/")
```

## v2.0 Complete Example

The v2.0 API replaces `solve_onset_problem(...)` and `find_growth_rate(eigenvalues)` with a unified `solve` call and structured result accessors:

```julia
#!/usr/bin/env julia
# onset_convection_v2.jl
# v2.0-style onset analysis using OnsetProblem + solve

using Magrathea
using Printf

E  = 1e-5
Pr = 1.0
χ  = 0.35

println("="^60)
println("Onset Convection Analysis — v2.0 API")
println("="^60)

results = []

for m in 5:25
    @printf("m = %2d: ", m)

    params = OnsetParams(E=E, Pr=Pr, Ra=1e7, χ=χ,
                         m=m, lmax=max(60, m+10), Nr=64)

    problem = OnsetProblem(params)
    estimate_size(problem)   # prints DOF count; remove if too verbose

    result = solve(problem; nev=6)

    push!(results, (m=m, growth_rate=result.growth_rate,
                    frequency=result.frequency))
    @printf("σ = %+.6e, ω = %+.6f\n",
            result.growth_rate, result.frequency)
end

# Find most-unstable mode
valid = filter(r -> isfinite(r.growth_rate), results)
if !isempty(valid)
    critical = argmax(r -> r.growth_rate, valid)
    println("\n" * "="^60)
    println("MOST UNSTABLE MODE (v2.0)")
    @printf("  m = %d, σ = %.6e, ω = %+.6f\n",
            critical.m, critical.growth_rate, critical.frequency)
end
```

!!! note "Migrating from v1.x"
    Replace `solve_onset_problem(...)` with `solve(OnsetProblem(params); nev=...)`.
    Replace `find_growth_rate(eigenvalues)` with `result.growth_rate`.
    Add `estimate_size(problem)` before large solves to check memory.

## Key References

1. **Chandrasekhar (1961)** - *Hydrodynamic and Hydromagnetic Stability*. Foundational work on rotating convection.

2. **Roberts (1968)** - Showed non-axisymmetric modes onset first with ``Ra_c \sim E^{-4/3}`` scaling.

3. **Busse (1970)** - Annulus model providing physical insight into thermal Rossby waves.

4. **Jones et al. (2000)** - Global asymptotic analysis matching numerical results.

5. **Dormy et al. (2004)** - Comprehensive study with WKB analysis and numerical validation.

6. **Barik et al. (2023)** - Extensive parameter study across radius ratios and Ekman numbers.

## Next Steps

- **[Biglobal Stability Analysis](biglobal_stability.md)** - Add axisymmetric mean flows (thermal wind)
- **[Triglobal Stability Analysis](triglobal_stability.md)** - Non-axisymmetric basic states with mode coupling

---

!!! info "Example Scripts"
    See `example/linear_stability_demo.jl` and `example/Rac_lm.jl` for complete working examples.
