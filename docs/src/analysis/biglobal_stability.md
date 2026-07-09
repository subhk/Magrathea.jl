# Biglobal Stability Analysis with Axisymmetric Mean Flow

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

The axisymmetric base state consists of:

**Temperature field:**

```math
\overline{T}(r, \theta) = \overline{T}_0(r) + \sum_{\ell=2,4,...} \overline{\Theta}_{\ell 0}(r) Y_\ell^0(\theta)
```

**Zonal flow:**

```math
\overline{u}_\phi(r, \theta) = \sum_{\ell=1,3,...} \overline{u}_{\phi,\ell 0}(r) Y_\ell^0(\theta)
```

The zonal flow has different parity than temperature: odd-``\ell`` for ``\overline{u}_\phi`` and even-``\ell`` for temperature perturbations (for equatorially symmetric basic states).

### Thermal Wind Balance

When temperature varies with latitude, geostrophic balance requires a zonal flow. The thermal wind equation:

```math
2\Omega \cos\theta \frac{\partial \overline{u}_\phi}{\partial z} = -\frac{g \alpha}{r} \frac{\partial \overline{T}}{\partial \theta}
```

In spherical coordinates with our non-dimensionalization:

```math
2 \cos\theta \frac{\partial \overline{u}_\phi}{\partial r} = -\frac{Ra \cdot E^2}{Pr \cdot r} \frac{\partial \overline{\Theta}}{\partial \theta}
```

This relates the vertical shear of zonal flow to the horizontal temperature gradient.

### Modified Linearized Equations

With an axisymmetric basic state ``(\overline{\mathbf{u}}, \overline{T})``, the linearized perturbation equations become:

**Momentum:**
```math
\frac{\partial \mathbf{u}'}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u}' + \underbrace{(\mathbf{u}' \cdot \nabla)\overline{\mathbf{u}} + (\overline{\mathbf{u}} \cdot \nabla)\mathbf{u}'}_{\text{advection by/of mean flow}} = -\nabla p' + E \nabla^2 \mathbf{u}' + \frac{Ra \cdot E^2}{Pr} \Theta' \hat{\mathbf{r}}
```

**Energy:**
```math
\frac{\partial \Theta'}{\partial t} + u_r' \frac{\partial \overline{T}}{\partial r} + \underbrace{\mathbf{u}' \cdot \nabla \overline{\Theta} + \overline{\mathbf{u}} \cdot \nabla \Theta'}_{\text{advection terms}} = \frac{E}{Pr} \nabla^2 \Theta'
```

### Azimuthal Mode Decoupling

Because the basic state is axisymmetric (``m_{bs} = 0``), perturbation modes with different azimuthal wavenumbers ``m`` remain decoupled:

```math
Y_\ell^0 \times Y_{\ell'}^m \propto Y_{\ell''}^m
```

This means we can still analyze each ``m`` independently, but the growth rates and eigenmodes are modified by the mean flow.

## The `BasicState` Structure

Magrathea.jl uses the `BasicState` type to store axisymmetric background profiles:

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

### Key Properties

| Field | Content | Typical ``\ell`` values |
|-------|---------|----------------------|
| `theta_coeffs` | Temperature ``\bar{\Theta}_{\ell 0}(r)`` | 0, 2, 4, 6, ... |
| `uphi_coeffs` | Zonal flow ``\bar{u}_{\phi,\ell 0}(r)`` | 1, 3, 5, ... |
| `dtheta_dr_coeffs` | Radial derivative ``d\bar{\Theta}_{\ell 0}/dr`` | 0, 2, 4, 6, ... |
| `duphi_dr_coeffs` | Radial derivative ``d\bar{u}_{\phi,\ell 0}/dr`` | 1, 3, 5, ... |

## Creating Basic States

### v2.0 Unified API

In v2.0, use `basic_state(params; mode=...)` instead of constructing `ChebyshevDiffn` manually:

```julia
using Magrathea

params = OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=12, lmax=60, Nr=64)

# Conduction profile (no flow)
bs = basic_state(params; mode=:conduction)

# Meridional thermal wind
bs = basic_state(params; mode=:meridional, amplitude=0.1)

# Solve with v2.0 API
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

### Method 3: Manual Construction

For importing data from simulations or custom profiles:

```julia
# Initialize
Nr = 64
lmax_bs = 6
r = cd.x

# Create coefficient dictionaries
theta_coeffs = Dict{Int, Vector{Float64}}()
dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
uphi_coeffs = Dict{Int, Vector{Float64}}()
duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

# Set temperature profile (example: Y₂₀ variation)
theta_coeffs[0] = conduction_profile.(r, χ)  # ℓ=0 (mean)
theta_coeffs[2] = 0.1 * gaussian_profile.(r)  # ℓ=2 perturbation

# Compute derivatives
dtheta_dr_coeffs[0] = cd.D1 * theta_coeffs[0]
dtheta_dr_coeffs[2] = cd.D1 * theta_coeffs[2]

# Build thermal wind from temperature
solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
    cd, χ, 1.0, Ra, Pr;
    mechanical_bc = :no_slip,
    E = E,
)

# Ensure all ℓ modes are populated
for ℓ in 0:lmax_bs
    theta_coeffs[ℓ] = get(theta_coeffs, ℓ, zeros(Nr))
    dtheta_dr_coeffs[ℓ] = get(dtheta_dr_coeffs, ℓ, zeros(Nr))
    uphi_coeffs[ℓ] = get(uphi_coeffs, ℓ, zeros(Nr))
    duphi_dr_coeffs[ℓ] = get(duphi_dr_coeffs, ℓ, zeros(Nr))
end

# Construct BasicState
bs = BasicState(
    r = r,
    Nr = Nr,
    theta_coeffs = theta_coeffs,
    dtheta_dr_coeffs = dtheta_dr_coeffs,
    uphi_coeffs = uphi_coeffs,
    duphi_dr_coeffs = duphi_dr_coeffs,
    lmax_bs = lmax_bs,
)
```

### Method 4: Import from External Codes

```julia
using JLD2
using Interpolations

# Load data from simulation (e.g., Rayleigh, MagIC)
@load "simulation_output.jld2" T_lm uphi_lm r_sim

# Interpolate to Magrathea.jl grid
for ℓ in [0, 2, 4]
    itp = LinearInterpolation(r_sim, T_lm[ℓ])
    theta_coeffs[ℓ] = itp.(cd.x)
    dtheta_dr_coeffs[ℓ] = cd.D1 * theta_coeffs[ℓ]
end

for ℓ in [1, 3]
    itp = LinearInterpolation(r_sim, uphi_lm[ℓ])
    uphi_coeffs[ℓ] = itp.(cd.x)
    duphi_dr_coeffs[ℓ] = cd.D1 * uphi_coeffs[ℓ]
end
```

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

    Ra_c, ω_c, _ = find_critical_rayleigh(
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
