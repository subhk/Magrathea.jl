# Magnetohydrodynamic Extension

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Magnetohydrodynamics</div>
  <h1>Stability of rotating, conducting fluids in magnetic fields.</h1>
  <p>
    The MHD module studies the linear stability of conducting fluids under rotation,
    thermal gradients, and imposed magnetic fields &mdash; for planetary dynamos,
    stellar convection, and laboratory MHD.
  </p>
</div>

## Overview

The MHD submodule (`Magrathea.MHD`) extends the hydrodynamic solver with:

- **Lorentz force**: Magnetic field effects on fluid motion
- **Induction equation**: Velocity effects on magnetic field evolution
- **Magnetic diffusion**: Ohmic dissipation of magnetic energy
- **Background fields**: Axial and dipolar magnetic field configurations
- **Magnetic boundary conditions**: Insulating, conducting, and perfect conductor options

## Physical Problem

### Governing Equations

The MHD equations in a rotating spherical shell:

**Momentum (Navier-Stokes + Lorentz):**
```math
\frac{\partial \mathbf{u}}{\partial t} + 2\boldsymbol{\Omega} \times \mathbf{u} = -\nabla p + E\nabla^2\mathbf{u} + \frac{Ra \cdot E^2}{Pr} \Theta \hat{\mathbf{r}} + Le^2 (\nabla \times \mathbf{B}) \times \mathbf{B}_0
```

**Induction:**
```math
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u} \times \mathbf{B}_0) + E_m \nabla^2 \mathbf{B}
```

**Heat:**
```math
\frac{\partial \Theta}{\partial t} + \mathbf{u} \cdot \nabla T_0 = \frac{E}{Pr} \nabla^2 \Theta
```

**Constraints:**
```math
\nabla \cdot \mathbf{u} = 0, \quad \nabla \cdot \mathbf{B} = 0
```

### Additional Dimensionless Numbers

| Parameter | Symbol | Definition | Physical Meaning |
|-----------|--------|------------|------------------|
| Magnetic Prandtl | ``Pm`` | ``\nu/\eta`` | Viscous to magnetic diffusivity |
| Lehnert number | ``Le`` | ``B_0/(\sqrt{\mu\rho}\Omega L)`` | Magnetic to rotational forces |
| Magnetic Ekman | ``E_m`` | ``E/Pm = \eta/(\Omega L^2)`` | Magnetic diffusion rate |

### Typical Parameter Values

| Parameter | Earth's Core | Lab Experiments | Simulations |
|-----------|--------------|-----------------|-------------|
| ``E`` | ``10^{-15}`` | ``10^{-3} - 10^{-6}`` | ``10^{-3} - 10^{-7}`` |
| ``Pr`` | 0.1 - 1 | 0.01 - 0.1 | 0.1 - 10 |
| ``Pm`` | ``10^{-6}`` | ``10^{-5}`` | 1 - 10 |
| ``Le`` | ``10^{-4} - 10^{-2}`` | ``10^{-3} - 0.1`` | 0 - 0.1 |

## Quick Start

### Load the MHD Module

```julia
# MHD types, operators, assembly and the eigensolver are all exported by Magrathea
using Magrathea
using LinearAlgebra, SparseArrays
```

### Basic MHD Problem

```julia
# Define parameters
params = MHDParams(
    # Physical parameters
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e5,
    Le = 0.1,           # Background field strength

    # Geometry
    ricb = 0.35,        # Inner core radius
    m = 2,              # Azimuthal wavenumber
    lmax = 15,          # Max spherical harmonic degree
    N = 32,             # Radial resolution
    symm = 1,           # Equatorial symmetry

    # Background field
    B0_type = axial,    # Uniform axial field
    B0_amplitude = 1.0,

    # Boundary conditions
    bci = 1, bco = 1,                      # No-slip velocity
    bci_thermal = 0, bco_thermal = 0,      # Fixed temperature
    bci_magnetic = 0, bco_magnetic = 0,    # Insulating

    # Heating mode
    heating = :differential,
)

# Solve via the high-level API.
# For no-field (hydro) and axial-field MHD, solve() uses a tau-free
# ultraspherical-Galerkin assembly: spurious-free, so the default which=:LR picks
# the physical mode with no σ-targeting. (The dipole case routes through the
# Chebyshev-tau method, whose spurious modes require targeting, e.g. sigma=0.0.)
result = solve(MHDProblem(params); nev=10, which=:LR)

eigenvalues = result.eigenvalues
σ_lead = maximum(real.(eigenvalues))            # growth rate (largest real part)
ω_lead = imag(eigenvalues[argmax(real.(eigenvalues))])

println("Leading eigenvalue: $σ_lead + $(ω_lead)i")
println(σ_lead > 0 ? "System is UNSTABLE" : "System is STABLE")
```

## MHDParams Reference

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `E` | Float64 | Ekman number |
| `Pr` | Float64 | Prandtl number |
| `Pm` | Float64 | Magnetic Prandtl number |
| `Ra` | Float64 | Rayleigh number |
| `Le` | Float64 | Lehnert number (0 for kinematic dynamo) |
| `ricb` | Float64 | Inner core radius ratio |
| `m` | Int | Azimuthal wavenumber |
| `lmax` | Int | Maximum spherical harmonic degree |
| `N` | Int | Radial resolution |

### Background Field Options

```julia
@enum BackgroundField begin
    no_field    # Kinematic dynamo (Le = 0)
    axial       # Uniform axial field B₀ = B₀ẑ
    dipole      # Dipolar field B₀ ∝ (2cosθ r̂ + sinθ θ̂)/r³
end
```

### Boundary Conditions

#### Velocity (Mechanical)

| Value | Type | Conditions |
|-------|------|------------|
| 0 | Stress-free | ``u_r = 0``, ``\partial^2 u/\partial r^2 = 0`` (poloidal); ``-r \partial v/\partial r + v = 0`` (toroidal) |
| 1 | No-slip | ``\mathbf{u} = 0`` at boundary |

#### Temperature (Thermal)

| Value | Type | Condition |
|-------|------|-----------|
| 0 | Fixed temperature | ``\Theta = 0`` |
| 1 | Fixed flux | ``\partial\Theta/\partial r = 0`` |

#### Magnetic

| Value | Type | Condition | Use Case |
|-------|------|-----------|----------|
| 0 | Insulating | ``(l+1)f + r f' = 0`` (CMB), ``l f - r f' = 0`` (ICB) | Earth's mantle |
| 1 | Conducting (finite) | Complex BC with skin depth | Conducting boundaries |
| 2 | Perfect conductor | ``f = 0``, ``E_m(-f'' - 2f'/r + Lf/r^2) = 0`` | Earth's inner core |

## Background Magnetic Fields

### Axial Field

Uniform field aligned with rotation axis:
```math
\mathbf{B}_0 = B_0 \hat{\mathbf{z}}
```

```julia
params = MHDParams(
    ...,
    B0_type = axial,
    B0_amplitude = 1.0,
)
```

### Dipole Field

Dipolar field (requires inner core):
```math
\mathbf{B}_0 = B_0 \frac{1}{r^3} (2\cos\theta \hat{\mathbf{r}} + \sin\theta \hat{\boldsymbol{\theta}})
```

```julia
params = MHDParams(
    ...,
    B0_type = dipole,
    B0_amplitude = 1.0,
    ricb = 0.35,  # Required for dipole
)
```

!!! warning "Dipole Field Requirements"
    Dipole fields require a finite inner core (`ricb > 0`) to avoid singularity at ``r = 0``.

## Matrix Structure

The MHD eigenvalue problem has block structure:

```math
\begin{pmatrix}
A_{uu} & A_{uv} & 0 & A_{uf} & A_{u\Theta} \\
A_{vu} & A_{vv} & A_{vf} & 0 & 0 \\
A_{fu} & A_{fv} & A_{ff} & 0 & 0 \\
A_{gu} & A_{gv} & 0 & A_{gg} & 0 \\
A_{\Theta u} & 0 & 0 & 0 & A_{\Theta\Theta}
\end{pmatrix}
\begin{pmatrix} u \\ v \\ f \\ g \\ \Theta \end{pmatrix}
= \sigma
\begin{pmatrix}
B_{uu} & 0 & 0 & 0 & 0 \\
0 & B_{vv} & 0 & 0 & 0 \\
0 & 0 & B_{ff} & 0 & 0 \\
0 & 0 & 0 & B_{gg} & 0 \\
0 & 0 & 0 & 0 & B_{\Theta\Theta}
\end{pmatrix}
\begin{pmatrix} u \\ v \\ f \\ g \\ \Theta \end{pmatrix}
```

Where:
- ``u`` = poloidal velocity
- ``v`` = toroidal velocity
- ``f`` = poloidal magnetic field
- ``g`` = toroidal magnetic field
- ``\Theta`` = temperature perturbation

### Key Couplings

| Block | Physical Process | Strength |
|-------|------------------|----------|
| ``A_{uf}``, ``A_{vf}`` | Lorentz force (B → u) | ``Le^2`` |
| ``A_{fu}``, ``A_{fv}`` | Induction (u → B) | ``Le`` |
| ``A_{u\Theta}`` | Buoyancy | ``Ra/Pr`` |
| ``A_{\Theta u}`` | Temperature advection | 1 |

## Use Cases

### Case 1: Hydrodynamic Benchmark (No Magnetic Field)

Reproduce Christensen & Wicht (2015) Table 1:

```julia
params = MHDParams(
    E = 4.734e-5,
    Pr = 1.0,
    Pm = 1.0,
    Ra = 1.6e6,
    Le = 0.0,           # No magnetic field
    ricb = 0.35,
    m = 9,
    lmax = 20,
    N = 24,
    B0_type = no_field,
    bci = 1, bco = 1,
    bci_thermal = 0, bco_thermal = 0,
    bci_magnetic = 0, bco_magnetic = 0,
)

result = solve(MHDProblem(params); nev = 10, which = :LR)  # no_field ⇒ Galerkin (spurious-free)
eigenvalues = result.eigenvalues

println("Growth rate: ", real(eigenvalues[1]), " (expect ≈ 0)")
println("Frequency: ", imag(eigenvalues[1]), " (expect ≈ 0.37)")
```

### Case 2: Magnetoconvection with Axial Field

Study how magnetic field stabilizes convection:

```julia
# Scan Lehnert number
Le_values = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
growth_rates = Float64[]

for Le in Le_values
    params = MHDParams(
        E = 1e-3, Pr = 1.0, Pm = 5.0, Ra = 1e5, Le = Le,
        ricb = 0.35, m = 2, lmax = 15, N = 32,
        B0_type = Le > 0 ? axial : no_field,
        bci = 1, bco = 1,
        bci_thermal = 0, bco_thermal = 0,
        bci_magnetic = 0, bco_magnetic = 0,
    )

    # axial + insulating ⇒ Galerkin (spurious-free); :LR picks the physical mode
    eigenvalues = solve(MHDProblem(params); nev = 5, which = :LR).eigenvalues

    push!(growth_rates, real(eigenvalues[1]))
    println("Le = $Le: σ = ", growth_rates[end])
end
```

**Physical insight**: Increasing ``Le`` stabilizes convection due to magnetic tension.

### Case 3: Earth-like Configuration (Perfect Conductor Inner Core)

```julia
params = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 5.0,
    Ra = 1e5, Le = 1e-3,
    ricb = 0.35,
    m = 2, lmax = 15, N = 32,
    B0_type = axial,
    bci = 1, bco = 1,
    bci_thermal = 0, bco_thermal = 0,
    bci_magnetic = 2,    # Perfect conductor at ICB
    bco_magnetic = 0,    # Insulating at CMB
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

println("DOFs with perfect conductor BC: ", length(interior_dofs))
```

## Troubleshooting

### Solver Doesn't Converge

```julia
# Relax tolerance and increase iterations
eigenvalues, _, _ = solve_eigenvalue_problem(A_int, B_int;
    nev = 30,
    tol = 1e-4,
    maxiter = 1000,
)
```

### All Eigenvalues Negative (Stable)

The Rayleigh number is below critical. Increase ``Ra``:

```julia
# Scan Ra to find critical value
Ra_values = [1e4, 5e4, 1e5, 5e5, 1e6]
for Ra in Ra_values
    # ... solve and check growth rate
end
```

### Results Don't Match Kore

Check:
1. Parameter definitions match exactly
2. Boundary conditions are equivalent
3. Resolution is sufficient (`lmax`, `N`)
4. Magrathea.jl uses Boussinesq approximation (no anelastic)

## Performance Considerations

### Matrix Size Estimates

| lmax | N | Approx DOFs | Memory | Solve Time |
|------|---|-------------|--------|------------|
| 10 | 24 | ~600 | <0.1 GB | seconds |
| 20 | 32 | ~2,000 | ~0.3 GB | ~10 sec |
| 30 | 48 | ~5,000 | ~2 GB | ~1 min |
| 50 | 64 | ~15,000 | ~20 GB | ~10 min |

### Tips for Large Problems

1. Start with low resolution for parameter exploration
2. Use sparse matrix storage (default)
3. Increase `nev` only as needed
4. Monitor memory usage with `Base.summarysize(A)`

## Complete Example

See `example/mhd_dynamo_example.jl` for a complete working script:

```julia
#!/usr/bin/env julia
# MHD Dynamo Stability Analysis

using Magrathea

using LinearAlgebra, SparseArrays, Printf

# Parameters
params = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 5.0,
    Ra = 1e4, Le = 0.1,
    ricb = 0.35, m = 2, lmax = 10, N = 16,
    B0_type = axial,
    bci = 1, bco = 1,
    bci_thermal = 0, bco_thermal = 0,
    bci_magnetic = 0, bco_magnetic = 0,
    heating = :differential,
)

# Solve (no_field/axial + insulating ⇒ tau-free Galerkin, spurious-free)
result = solve(MHDProblem(params); nev = 10, which = :LR)
eigenvalues  = result.eigenvalues
eigenvectors = result.eigenvectors

# Results
println("\nLeading eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:5])
    @printf("  %d: σ = %+.6f, ω = %+.6f\n", i, real(λ), imag(λ))
end
```

## References

- Christensen & Wicht (2015), *Numerical Dynamo Simulations*, Treatise on Geophysics Vol. 8
- Jones et al. (2011), *Anelastic convection-driven dynamo benchmarks*, Icarus
- Dormy & Soward (2007), *Mathematical Aspects of Natural Dynamos*

---

!!! info "Additional Documentation"
    See the [MHD User Guide](mhd_user_guide.md) for comprehensive usage documentation, and the [Codebase Structure](codebase_structure.md) page for implementation details.
