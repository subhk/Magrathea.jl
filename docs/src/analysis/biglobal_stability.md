# Biglobal Stability Analysis with Axisymmetric Mean Flow

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](../getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

Biglobal stability analysis extends the classical onset problem by including an axisymmetric (``m=0``) background flow. This captures scenarios where differential rotation, thermal wind, or imposed zonal jets modify the stability characteristics of the system.

## Physical Motivation

### When to Use Biglobal Analysis

Biglobal analysis is appropriate when:

1. **Thermal wind balance** - Latitudinal temperature variations drive geostrophic zonal flows
2. **Differential rotation** - Inner and outer boundaries rotate at different rates
3. **Imposed zonal jets** - Pre-existing axisymmetric flow structures
4. **Meridional circulation** - Axisymmetric poloidal flows (though often secondary)

### Real-World Applications

| System | Source of Mean Flow |
|--------|---------------------|
| Earth's outer core | Thermal wind from CMB heat flux variations |
| Jupiter's interior | Deep zonal jets extending from atmosphere |
| Solar tachocline | Differential rotation between radiative and convective zones |
| Laboratory experiments | Differentially rotating boundaries |

## Mathematical Formulation

### Base State Structure

An axisymmetric basic state has temperature and all three velocity components
independent of longitude. Viscosity and the wall conditions generally produce
meridional circulation even when the boundary forcing has only `m=0` modes.
The native velocity representation uses vector spherical harmonics;
scalar component projections do not have the same finite harmonic support.

### Thermal Wind Balance

The noniterated `:meridional` constructor solves Stokes–Coriolis momentum
balance with a conductive temperature, enforcing both mechanical boundaries.
Thermal-wind balance is an interior approximation, not the complete boundary
value problem. The self-consistent constructor also solves thermal transport
and includes nonlinear mean-flow inertia by default.

See [mean-state equations](../basic_states.md#Thermal-Wind-Balance) and
[nonlinear steady states](../basic_states.md#Self-Consistent-Basic-States-with-Advection).

### Modified Linearized Equations

With an axisymmetric basic state ``(\overline{\mathbf{u}}, \overline{T})``, the linearized perturbation equations become:

**Momentum:**
```math
\frac{\partial \mathbf{u}'}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u}' + \underbrace{(\mathbf{u}' \cdot \nabla)\overline{\mathbf{u}} + (\overline{\mathbf{u}} \cdot \nabla)\mathbf{u}'}_{\text{advection by/of mean flow}} = -\nabla p' + E \nabla^2 \mathbf{u}' + \frac{Ra \cdot E^2}{Pr(1-\chi)^3} r\Theta' \hat{\mathbf{r}}
```

**Energy:**
```math
\frac{\partial \Theta'}{\partial t} + \mathbf{u}' \cdot \nabla \overline{T} + \overline{\mathbf{u}} \cdot \nabla \Theta' = \frac{E}{Pr} \nabla^2 \Theta'
```

### Azimuthal Mode Decoupling

Because the basic state is axisymmetric (``m_{bs} = 0``), perturbation modes with different azimuthal wavenumbers ``m`` remain decoupled:

```math
Y_\ell^0 \times Y_{\ell'}^m \propto Y_{\ell''}^m
```

This means we can still analyze each ``m`` independently, but the growth rates and eigenmodes are modified by the mean flow.

## The `BasicState` Structure

Magrathea.jl uses the `BasicState` type to store axisymmetric background profiles:

See [`BasicState`](@ref) in the API reference for the current fields. Temperature and component dictionaries use radial collocation values. The `flow` field stores the authoritative divergence-free vector representation; use [`mean_flow_velocity`](@ref) for physical velocity components.

The generated state's `flow` field carries the full velocity used in the
linearization. The thermal gradient, velocity advection, and velocity shear
are included through `src/Stability/mean_flow_coupling.jl`.

## Creating Basic States

### Unified API

With the unified API, use `basic_state(params; mode=...)` instead of constructing `ChebyshevDiffn` manually:

```julia
using Magrathea

params = OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=12, lmax=60, Nr=64)

# Conduction profile (no flow)
bs = basic_state(params; mode=:conduction)

# Meridional thermal wind
bs = basic_state(params; mode=:meridional, amplitude=0.1)

# Solve with unified API
result = solve(BiglobalProblem(params, bs); nev=6)
println("Growth rate: ", result.growth_rate)
println("Frequency:   ", result.frequency)
```

### Method 1: Conduction Profile (No Flow)

The simplest case—useful as a reference or when thermal wind is negligible:

```julia
using Magrathea

# Setup Chebyshev differentiation
Nr = 64
χ = 0.35
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Pure conduction basic state
bs = conduction_basic_state(cd, χ; lmax_bs = 6)
```

This creates a basic state with:
- ``\bar{T}_0(r)`` = conductive profile
- ``\bar{u}_\phi = 0`` everywhere

### Method 2: Meridional Temperature + Thermal Wind

Add latitudinal temperature variations that drive thermal wind:

```julia
# Physical parameters
E = 1e-5
Ra = 1e7
Pr = 1.0

# Create meridional basic state with thermal wind
bs = meridional_basic_state(
    cd,              # Chebyshev differentiation
    χ,               # Radius ratio
    E,               # Ekman number
    Ra,              # Rayleigh number
    Pr,              # Prandtl number
    lmax_bs = 6,     # Max ℓ for basic state
    amplitude = 0.1; # Amplitude of Y₂₀ perturbation
    mechanical_bc = :no_slip,
)
```

This generates:

1. **Temperature**: ``\bar{\Theta}_{20}(r) \cdot Y_2^0(\theta)`` perturbation
2. **Zonal flow**: ``\bar{u}_\phi`` from thermal wind integration

### Custom Boundary Patterns and Imported States

Use the low-level constructor for a prescribed temperature or radial-derivative
pattern:

```julia
cd = ChebyshevDiffn(params.Nr, [params.χ, 1.0], 4)
bs = basic_state(cd, params.χ, params.E, params.Ra, params.Pr;
    temperature_bc=Y20(0.01), mechanical_bc=params.mechanical_bc)
```

For imported states, use the same radial grid, harmonic normalization,
mechanical boundaries, and temperature convention as the perturbation
operator. A hand-built scalar zonal dictionary does not recover the missing
meridional flow. See [Basic States](../basic_states.md) for representation
and interpolation constraints.

## Using Basic States in Stability Analysis

### Pass to OnsetParams

```julia
# Define parameters with basic state
params = OnsetParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1e7,
    χ = 0.35,
    m = 12,
    lmax = 60,
    Nr = 64,
    basic_state = bs,  # Include the axisymmetric basic state
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# Build operator - advection terms automatically included
op = LinearStabilityOperator(params)
```

### Compute Eigenvalues

```julia
# Build basic state with unified constructor
bs = basic_state(params; mode=:meridional, amplitude=0.1)

# Solve biglobal problem
result = solve(BiglobalProblem(params, bs); nev=8)

println("With mean flow:")
println("  Growth rate: σ = ", result.growth_rate)
println("  Drift frequency: ω = ", result.frequency)
```

### Compare to No-Flow Case

```julia
# Reference: no basic state
params_ref = OnsetParams(
    E = 1e-5, Pr = 1.0, Ra = 1e7, χ = 0.35,
    m = 12, lmax = 60, Nr = 64,
    # basic_state omitted → conduction only
)

result_ref = solve(OnsetProblem(params_ref); nev=4)

println("\nComparison:")
println("  Without mean flow: σ = $(result_ref.growth_rate)")
println("  With mean flow:    σ = $(result.growth_rate)")
println("  Difference:        Δσ = $(result.growth_rate - result_ref.growth_rate)")
```

## Physical Effects of Mean Flow

### Stabilization vs Destabilization

Mean flows can either stabilize or destabilize convection:

| Effect | Mechanism | Typical Result |
|--------|-----------|----------------|
| **Advective stabilization** | Mean flow shears convective columns | Increased ``Ra_c`` |
| **Destabilization** | Shear instabilities, resonances | Decreased ``Ra_c`` |
| **Drift modification** | Doppler shift by zonal flow | Changed ``\omega_c`` |

### Thermal Wind Amplitude

The strength of thermal wind scales with the temperature variation amplitude:

```math
\bar{u}_\phi \sim \frac{Ra \cdot E^2}{Pr} \cdot \Delta \bar{\Theta}
```

For weak thermal wind (``\bar{u}_\phi \ll E^{1/3}``), effects are perturbative. For strong thermal wind, significant modifications to onset occur.

### Critical Layer Interactions

When ``\bar{u}_\phi(r_c) = \omega/m`` (matching pattern speed), critical layers form where perturbation energy can be exchanged with the mean flow.

## Boundary-Driven Flows

### Differential Rotation

Boundaries rotating at different rates impose zonal flow:

```julia
# Inner boundary rotating faster
Ω_inner = 1.0  # Reference
Ω_outer = 0.9  # 10% slower

# This requires specialized basic state construction
bs_diff_rot = differential_rotation_basic_state(
    cd, χ, E,
    Ω_inner = Ω_inner,
    Ω_outer = Ω_outer,
)
```

### Boundary Heating Patterns

Laterally varying boundary heat flux:

```julia
# CMB-like heating pattern
boundary_heating = Dict(
    2 => 0.15,  # Y₂₀ amplitude
    4 => 0.05,  # Y₄₀ amplitude
)

bs = boundary_forced_basic_state(
    cd, χ, E, Ra, Pr,
    boundary_modes = boundary_heating,
)
```

## Complete Example: Thermal Wind Stability

```julia
#!/usr/bin/env julia
# biglobal_thermal_wind.jl
#
# Biglobal stability analysis with thermal wind driven by
# latitudinal temperature variations

using Magrathea
using JLD2
using Printf

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1.2e7
χ = 0.35
Nr = 64
lmax = 60

# === Build Chebyshev Operators ===
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# === Create Basic State with Thermal Wind ===
println("="^60)
println("Biglobal Stability Analysis")
println("="^60)

# Sweep thermal wind amplitude
amplitudes = [0.0, 0.05, 0.1, 0.15, 0.2]
m_test = 12

results = []

for amp in amplitudes
    @printf("Amplitude = %.2f: ", amp)

    # Create basic state
    if amp == 0.0
        bs = conduction_basic_state(cd, χ; lmax_bs = 6)
    else
        bs = meridional_basic_state(
            cd, χ, E, Ra, Pr;
            lmax_bs = 6,
            amplitude = amp,
            mechanical_bc = :no_slip,
        )
    end

    # Build problem with basic state
    params = OnsetParams(
        E = E, Pr = Pr, Ra = Ra, χ = χ,
        m = m_test, lmax = lmax, Nr = Nr,
        basic_state = bs,
        mechanical_bc = :no_slip,
        thermal_bc = :fixed_temperature,
    )

    # Find eigenvalues
    result = solve(BiglobalProblem(params, bs); nev=4)

    σ = result.growth_rate
    ω = result.frequency

    push!(results, (amplitude=amp, σ=σ, ω=ω))
    @printf("σ = %+.6e, ω = %+.6f\n", σ, ω)
end

# === Analyze Effect ===
println("\n" * "="^60)
println("Effect of Thermal Wind on Stability")
println("="^60)

σ_ref = results[1].σ  # No mean flow reference

for r in results
    Δσ = r.σ - σ_ref
    effect = Δσ > 0 ? "destabilizing" : "stabilizing"
    @printf("  amp = %.2f: Δσ = %+.4e (%s)\n", r.amplitude, Δσ, effect)
end

# === Find Critical Ra with Mean Flow ===
println("\n" * "="^60)
println("Critical Rayleigh Numbers")
println("="^60)

for amp in [0.0, 0.1, 0.2]
    if amp == 0.0
        bs = conduction_basic_state(cd, χ; lmax_bs = 6)
    else
        bs = meridional_basic_state(
            cd, χ, E, Ra, Pr;
            lmax_bs = 6, amplitude = amp,
        )
    end

    Ra_c, ω_c, _ = find_critical_Ra_biglobal(;
        E = E, Pr = Pr, χ = χ, m = m_test,
        lmax = lmax, Nr = Nr,
        basic_state = bs,
        Ra_guess = Ra,
    )

    @printf("  amp = %.2f: Ra_c = %.6e, ω_c = %+.6f\n", amp, Ra_c, ω_c)
end

# === Save ===
@save "outputs/biglobal_thermal_wind.jld2" results E Pr Ra χ
println("\nResults saved.")
```

## Checklist

Before running biglobal analysis:

- [ ] Chebyshev grid matches between basic state and analysis parameters
- [ ] Basic state temperature coefficients include ``\ell = 0`` (mean profile)
- [ ] Thermal wind computed consistently with temperature perturbation
- [ ] `lmax_bs` in basic state is sufficient for convergence
- [ ] Radial derivatives computed with same Chebyshev operators

## Next Steps

- **[Onset Convection](onset_convection.md)** - Classical problem without mean flow
- **[Triglobal Stability](triglobal_stability.md)** - Non-axisymmetric basic states with mode coupling

---

!!! info "Example Scripts"
    See `example/basic_state_onset_example.jl` and `example/boundary_driven_jet.jl` for working examples.
