# Magnetohydrodynamic Extension

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Magnetohydrodynamics</div>
  <h1>Stability of rotating, conducting fluids in magnetic fields.</h1>
  <p>
    The MHD module studies the linear stability of conducting fluids under rotation,
    thermal gradients, and imposed magnetic fields &mdash; for rotating magnetoconvection,
    stellar convection, and laboratory MHD.
  </p>
</div>

## Overview

The MHD implementation in `Magrathea` extends the hydrodynamic solver with:

- **Lorentz force**: Magnetic field effects on fluid motion
- **Induction equation**: Velocity effects on magnetic field evolution
- **Magnetic diffusion**: Ohmic dissipation of magnetic energy
- **Background fields**: Axial and dipolar magnetic field configurations
- **Magnetic boundary conditions**: Insulating, conducting, and perfect conductor options

## Physical Problem

`MHDProblem` linearizes about a **motionless conductive state** (for mean flows with
an imposed field, see the [MHD guide](mhd_user_guide.md#Mean-flows:-biglobal-and-triglobal-MHD)) with a prescribed,
current-free axial or dipolar field. It does not couple the hydrodynamic nonlinear
mean-flow solver into MHD; explicit `MHDProblem.basic_state` objects are rejected.
`no_field` is hydrodynamic stability, with no magnetic degrees of freedom.

Both imposed fields have spherical-harmonic degree one. Same-type poloidal or
toroidal magnetic/velocity couplings change degree by ±1; mixed-type couplings
preserve degree. Consequently `symm=0` is the direct sum of the two parity sectors.

Native potentials use ``\mathbf{b}=\nabla\times\nabla\times(rf\hat{\mathbf r})+\nabla\times(rg\hat{\mathbf r})`` and harmonics ``Y_l^m/\sqrt{2l+1}``.
The reconstruction routines use this convention for velocity, magnetic field,
and temperature. `N` is the maximum Chebyshev degree (`N+1` coefficients).
`Le` sets the field strength; `B0_amplitude` is a legacy display tag and does not
rescale the field.


### Governing Equations

The MHD equations in a rotating spherical shell:

**Momentum (Navier-Stokes + Lorentz):**
```math
\frac{\partial \mathbf{u}}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u} = -\nabla p + E\nabla^2\mathbf{u} + \frac{Ra \cdot E^2}{Pr\,(1-r_i)^3}\, r\Theta \hat{\mathbf{r}} + Le^2 (\nabla \times \mathbf{B}) \times \mathbf{B}_0
```

Here ``\mathbf B`` is the perturbation field and ``r_i`` is `ricb`. As in the
hydrodynamic solvers, `Ra` is based on the shell thickness and gravity is
proportional to radius; see [Mathematical Foundations](theory/mathematical_foundations.md).

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
| Lehnert number | ``Le`` | ``B_0/(\sqrt{\mu\rho}\,\Omega r_o)`` | Magnetic to rotational forces |
| Magnetic Ekman | ``E_m`` | ``E/Pm = \eta/(\Omega r_o^2)`` | Magnetic diffusion rate |

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
# Insulating or perfectly conducting walls use energy-conserving Galerkin assembly.
# A finite-conductivity inner core or mantle uses coefficient-space tau assembly.
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
| `E` | Real | Ekman number, based on the outer radius |
| `Ra` | Real | Rayleigh number, based on the shell thickness |
| `ricb` | Real | Inner core radius ratio, `0 < ricb < 1` |
| `m` | Int | Azimuthal wavenumber, `m ≥ 0` |
| `lmax` | Int | Maximum spherical harmonic degree, `lmax ≥ m` |
| `N` | Int | Maximum Chebyshev degree (`N + 1` coefficients); even and `≥ 8` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Pr` | `1.0` | Prandtl number |
| `Pm` | `1.0` | Magnetic Prandtl number |
| `Le` | `0.0` | Lehnert number; `0` for `no_field`, `> 0` for an imposed field |
| `symm` | `1` | Equatorial symmetry: `1`, `-1`, or `0` for both parities |
| `B0_type` | `no_field` | Background field: `no_field`, `axial`, or `dipole` |
| `B0_amplitude` | `0.0` | Legacy display tag; does not rescale the field and must be `0` for `no_field` |
| `bci`, `bco` | `1` | Mechanical walls (see below) |
| `bci_thermal`, `bco_thermal` | `0` | Thermal walls |
| `bci_magnetic`, `bco_magnetic` | `0` | Magnetic walls |
| `heating` | `:differential` | `:differential` or `:internal` |
| `mantle_radius` | `nothing` | Outer mantle radius, required and `> 1` when `bco_magnetic=1` |
| `mantle_diffusivity_ratio` | `1` | Mantle/fluid magnetic diffusivity ratio |
| `forcing_frequency` | `0.0` | Legacy keyword; must be zero |

Numeric parameters are promoted to a common type `T` (`MHDParams{T}`).

### Background Field Options

```julia
@enum BackgroundField begin
    no_field    # Hydrodynamics (Le = 0)
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
| 1 | Finite conductor | Evolving core or mantle field, matched to the fluid | Stationary solid with equal permeability |
| 2 | Perfect conductor | ``f=0``, tangential electric field zero | Ideal stationary conducting wall |

For no-slip perfect-conductor walls the toroidal condition is ``g'+g/r=0``.
For stress-free walls the code includes the tangential ``\mathbf{u}\times\mathbf{B}_0``
contribution. `bci_magnetic=1` uses a regular core with the fluid's magnetic
diffusivity. `bco_magnetic=1` adds a finite mantle on
``1\le r\le R_m``, with `mantle_radius=R_m > 1` required and
`mantle_diffusivity_ratio=η_m/η_f > 0` (default `1`). The mantle matches a vacuum
field at its outer surface. Both conducting regions evolve with the same unknown
eigenvalue as the fluid, using tau assembly; `forcing_frequency` must be zero.
See [conducting-region matching](@ref conducting-mantle)
for the interface equations and an example.

`solve` checks the reconstructed magnetic boundary fields by default. Use
`boundary_check=:error` to reject eigenmodes whose physical residuals exceed
`boundary_rtol` and `boundary_atol`; inspect `result.extra.magnetic_boundaries`
or call `magnetic_boundary_residuals(result)`. These checks include angular
components above the retained harmonic cutoff. Radial and angular refinement
are still required to establish convergence.

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
\mathbf{B}_0 = \frac{1}{2r^3} (2\cos\theta \hat{\mathbf{r}} + \sin\theta \hat{\boldsymbol{\theta}})
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
A_{uu} & A_{uv} & A_{uf} & A_{ug} & A_{u\Theta} \\
A_{vu} & A_{vv} & A_{vf} & A_{vg} & 0 \\
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
| ``A_{uf}``, ``A_{ug}``, ``A_{vf}``, ``A_{vg}`` | Lorentz force (B → u) | ``Le^2`` |
| ``A_{fu}``, ``A_{fv}``, ``A_{gu}``, ``A_{gv}`` | Induction (u → B) | 1 (assembled only when ``Le > 0``) |
| ``A_{u\Theta}`` | Buoyancy | ``Ra\,E^2/(Pr\,(1-r_i)^3)`` |
| ``A_{\Theta u}`` | Temperature advection | ``-dT_0/dr``; ``r_i/((1-r_i)r^2)`` for differential heating |

## Use Cases

### Case 1: Hydrodynamic Benchmark (No Magnetic Field)

Reproduce the onset benchmark of Barik et al. (2023) at the shell-thickness
Ekman number Ek_d = 10⁻³ (critical m = 4, R̃aᶜ = 55.9, ωᶜ = -0.0231). `E` is
based on the outer radius, `E = Ek_d (1 - ricb)^2`, and `Ra = R̃a / Ek_d` is based
on the shell thickness:

```julia
params = MHDParams(
    E = 4.225e-4,
    Pr = 1.0,
    Pm = 1.0,
    Ra = 5.59e4,
    Le = 0.0,           # No magnetic field
    ricb = 0.35,
    m = 4,
    lmax = 20,
    N = 24,
    B0_type = no_field,
    bci = 1, bco = 1,
    bci_thermal = 0, bco_thermal = 0,
    bci_magnetic = 0, bco_magnetic = 0,
)

result = solve(MHDProblem(params); nev = 10, which = :LR)  # insulating ⇒ energy Galerkin
eigenvalues = result.eigenvalues

println("Growth rate: ", real(eigenvalues[1]), " (expect ≈ 0)")
println("Frequency: ", imag(eigenvalues[1]), " (expect ≈ -0.0231)")
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

    # insulating ⇒ energy Galerkin; :LR picks the physical mode
    eigenvalues = solve(MHDProblem(params); nev = 5, which = :LR).eigenvalues

    push!(growth_rates, real(eigenvalues[1]))
    println("Le = $Le: σ = ", growth_rates[end])
end
```

**Physical insight**: The effect of ``Le`` is not monotonic. In this scan the growth
rate is essentially unchanged up to ``Le = 10^{-3}``, rises slightly at
``Le = 10^{-2}``, and falls at ``Le = 0.1``: a field can relax the rotational
constraint as well as add magnetic tension.

### Case 3: Ideal Perfect-Conductor Inner Boundary

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

# Low-level tau pencil; solve(MHDProblem(params)) uses the Galerkin assembly here.
op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

println("Differential-equation rows with perfect conductor BC: ", length(interior_dofs))
```

## Troubleshooting

### Solver Doesn't Converge

```julia
# Relax tolerance and increase iterations
result = solve(MHDProblem(params); nev = 30, tol = 1e-4, maxiter = 1000)
```

Passing `sigma` near the expected eigenvalue also helps shift-invert. For small
problems, `backend = :dense` computes the whole spectrum as a cross-check.

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

For `m = 2` with an imposed field, with the dense storage of two complex matrices
reported by `estimate_size(MHDProblem(params))`:

| lmax | N | DOFs (one parity) | DOFs (`symm = 0`) | Dense storage (one parity / both) |
|------|---|-------------------|-------------------|-----------------------------------|
| 10 | 24 | 575 | 1,125 | 0.01 / 0.04 GB |
| 20 | 32 | 1,584 | 3,135 | 0.08 / 0.29 GB |
| 30 | 48 | 3,577 | 7,105 | 0.38 / 1.5 GB |
| 50 | 64 | 7,995 | 15,925 | 1.9 / 7.6 GB |

### Tips for Large Problems

1. Start with low resolution for parameter exploration
2. Call `estimate_size(MHDProblem(params))` before solving; it also estimates the `N` needed for the magnetic boundary layers
3. Use sparse matrix storage (default)
4. Increase `nev` only as needed
5. Monitor memory usage with `Base.summarysize(A)`

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

# Solve (insulating walls ⇒ energy-conserving Galerkin)
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
