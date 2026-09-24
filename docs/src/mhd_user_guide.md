# MHD User Guide

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

**Magrathea.jl MHD Implementation - Comprehensive Usage Documentation**

---

## Table of Contents

1. [Introduction](#Introduction)
2. [Quick Start](#Quick-Start)
3. [Physical Parameters](#Physical-Parameters)
4. [Boundary Conditions](#Boundary-Conditions)
5. [Complete Workflow](#Complete-Workflow)
6. [Common Use Cases](#Common-Use-Cases)
7. [Troubleshooting](#Troubleshooting)
8. [Reference Tables](#Reference-Tables)

---

## Introduction

The MHD implementation in Magrathea.jl solves the **magnetohydrodynamic eigenvalue problem** for rotating spherical shells. This is used to study:

- **Convection onset** in planetary cores
- **Magnetic modification of convection onset** with imposed axial or dipole fields
- **Magnetoconvection** in laboratory experiments
- **Linear stability** about motionless conductive MHD backgrounds

### Mathematical Problem

The code solves the generalized eigenvalue problem:

```
A·v = σ·B·v
```

Where:
- **A**: Spatial operators (Coriolis, buoyancy, Lorentz, diffusion)
- **B**: Time derivative operator
- **σ**: Complex eigenvalue (growth rate + i·frequency)
- **v**: Eigenvector (field amplitudes)

### Key Features

✅ **Spectral accuracy**: Ultraspherical (Gegenbauer) method
✅ **Flexible BCs**: No-slip, stress-free, insulating, perfect conductor
✅ **Background fields**: Axial and dipolar magnetic fields
✅ **Physics checks**: Independent Lorentz/induction, diffusion, wall, and core-matching tests; see [Codebase Structure](codebase_structure.md) for the validation files

---

## Quick Start

### Basic Hydrodynamic Onset (No Magnetic Field)

```julia
using Magrathea
using LinearAlgebra, SparseArrays

# Define parameters (Christensen & Wicht 2015, Table 1)
params = MHDParams(
    E = 4.734e-5,      # Ekman number
    Pr = 1.0,           # Prandtl number
    Pm = 1.0,           # Magnetic Prandtl (irrelevant for Le=0)
    Ra = 1.6e6,         # Rayleigh number
    Le = 0.0,           # NO magnetic field
    ricb = 0.35,        # Inner core radius
    m = 9,              # Azimuthal wavenumber
    lmax = 20,          # Max spherical harmonic degree
    N = 24,             # Radial resolution
    bci = 1, bco = 1,   # No-slip boundaries
    bci_thermal = 0, bco_thermal = 0  # Fixed temperature
)

# Solve via the high-level API. Insulating walls ⇒ energy-conserving Galerkin
# assembly, with boundary constraints built into its trial basis.
result = solve(MHDProblem(params); nev=20, tol=1e-6, which=:LR)

growth_rates = real.(result.eigenvalues)
frequencies  = imag.(result.eigenvalues)

println("Largest growth rate: ", maximum(growth_rates))
println("Critical mode frequency: ", frequencies[argmax(growth_rates)])
```

**Expected Output** (for Ra = Raᶜ ≈ 1.6×10⁶):
- Growth rate: ≈ 0 (marginal stability)
- Frequency: ≈ 0.35-0.40

---

## Eigensolver and model scope

`solve(MHDProblem(params))` is the recommended entry point. With insulating or
perfectly conducting magnetic walls (axial, dipole, or no field) it uses the
energy-conserving Galerkin assembly `Magrathea.assemble_mhd_energy_galerkin`. Each momentum
and induction equation is tested against its own trial basis in the energy inner
product, so the discrete Lorentz force is the exact negative adjoint of the
induction term, Coriolis exchanges no energy, and viscous and ohmic terms only
dissipate. Without buoyancy, no eigenvalue can grow at any resolution. Perfectly
conducting walls with slip need no special treatment, since their motional electric
field enters the weak form directly.

A finite-conductivity inner core (`bci_magnetic = 1`) uses tau assembly, with fluid
residuals in `C⁽⁴⁾` (poloidal velocity) or `C⁽²⁾` (other fields). Only the highest
residual coefficients are replaced by boundary constraints; unknowns remain
Chebyshev coefficients. Tau pencils have infinite algebraic boundary eigenvalues;
shift-invert targeting with `sigma=0.0` can select finite modes near onset. Check
radial and angular convergence for the physical parameters being studied.

Strong fields need enough radial modes to resolve the magnetic (Hartmann) boundary
layers, of thickness ``\sqrt{E\,E_m}/(Le\,B_0)`` at a wall with field ``B_0``. Below
that resolution, the tau pencil shows large spurious growth rates, even without
buoyancy, because Alfvén waves at the truncation scale are under-damped. The
energy-conserving assembly cannot grow spuriously, but its eigenvalues are equally
inaccurate until the layers are resolved. The imposed dipole is ``r_i^{-3}`` times
stronger at the inner wall than at the outer wall (about 23 times for
`ricb = 0.35`), so it needs more radial modes than an axial field of the same `Le`.
For example, at `E = 1e-3`, `Pm = 1`, and `Le = 0.1`, the dipole needs about
`N = 64` with tau assembly, while the axial field is resolved at `N = 24`.
`estimate_size(MHDProblem(params))` prints a rough `N` for the boundary layers, and
`solve` warns when the leading mode's radial spectrum has not decayed
(`result.extra.spectral_tail`). Increase `N` until the leading eigenvalue converges.

The tests compare shell magnetic free decay against independent collocation,
and an equal-diffusivity conducting core against analytical full-sphere decay.
They also check parity separation, current-free Lorentz force, axial induction,
and physical field reconstruction. Independent boundary tests evaluate spherical
strain and the tangential electric field on computed eigenmodes. The slip-wall
EMF uses analytical degree-one harmonic coefficients so forbidden angular
couplings remain exactly zero, including in `Float32`.

The solver linearizes about a **motionless conductive state** with a prescribed,
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

`interior_dofs` returned by tau assembly denotes differential **rows** only.
Solve the full `(A,B)` pencil. Slicing both matrices to these indices removes the
boundary equations without imposing them. Reconstruction requires all coefficients;
Galerkin eigenvectors must first be expanded using their recombination layout.

---

## Physical Parameters

### Dimensionless Numbers

#### Ekman Number (E)

**Definition:** E = ν/(ΩL²)

**Physical meaning:** Ratio of viscous to Coriolis forces

**Typical values:**
- Laboratory: 10⁻³ to 10⁻⁵
- Earth's core: 10⁻¹⁵
- Numerical simulations: 10⁻³ to 10⁻⁶

**What it controls:**
- Small E → Strong rotation effects
- Small E → Thinner boundary layers
- Small E → Higher critical Rayleigh number

#### Rayleigh Number (Ra)

**Definition:** Ra = αgΔTL³/(νκ)

**Physical meaning:** Measure of thermal forcing strength

**Critical value Raᶜ:**
- Depends on E, Pr, geometry, and boundary conditions
- For Earth-like parameters (E~10⁻⁵, χ=0.35): Raᶜ ~ 10⁶

**Parameter scans:**
```julia
# Find critical Rayleigh number
Ra_values = [1e5, 5e5, 1e6, 1.5e6, 2e6]
for Ra in Ra_values
    params = MHDParams(E=4.734e-5, Pr=1.0, Pm=1.0, Ra=Ra, Le=0.0,
                       ricb=0.35, m=9, lmax=20, N=24, ...)
    # Solve and check if growth rate > 0
end
```

#### Lehnert Number (Le)

**Definition:** Le = B₀/(√(μρ)ΩL)

**Physical meaning:** Magnetic field strength relative to rotation

**Typical values:**
- Le = 0: Pure hydrodynamics
- Le ~ 10⁻⁴ - 10⁻²: Weak field (Earth-like)
- Le ~ 0.1: Strong field (laboratory)

**Effect on dynamics:**
- Small Le: Rotation dominates
- Large Le: Magnetic forces compete with rotation
- Le → ∞: Magnetostrophic balance

#### Prandtl Numbers (Pr, Pm)

**Pr = ν/κ** (thermal Prandtl number)
- Liquid metals: Pr ~ 0.01 - 0.1
- Water: Pr ~ 7
- Earth's core: Pr ~ 0.1 - 1

**Pm = ν/η** (magnetic Prandtl number)
- Earth's core: Pm ~ 10⁻⁶ (very small!)
- Laboratory liquid metals: Pm ~ 10⁻⁵ - 10⁻⁴
- **Numerical constraint:** Usually Pm ≥ O(1) for stability

---

## Boundary Conditions

### Mechanical (Velocity) Boundary Conditions

#### No-Slip (bci=1, bco=1)

**Physics:** Fluid sticks to solid boundary

**Mathematical conditions:**
- Poloidal: u = 0, ∂u/∂r = 0
- Toroidal: v = 0

**When to use:**
- Rigid boundaries (Earth's core - solid mantle and inner core)
- Most laboratory experiments
- **Most common choice**

**Example:**
```julia
params = MHDParams(..., bci=1, bco=1)  # No-slip both boundaries
```

#### Stress-Free (bci=0, bco=0)

**Physics:** Zero tangential stress at boundary

**Mathematical conditions:**
- Poloidal: u = 0, ∂²u/∂r² = 0
- Toroidal: -r ∂v/∂r + v = 0

**When to use:**
- Free surfaces (liquid-gas interfaces)
- Simplified models
- **Note:** Magrathea.jl uses Boussinesq approximation (no density stratification)

**Example:**
```julia
params = MHDParams(..., bci=0, bco=0)  # Stress-free both boundaries
```

With stress-free walls at both boundaries, a rigid rotation about the axis
(``m=0``) induces no field and is an exactly neutral mode. The solver removes it
by requiring zero net angular momentum when that momentum is conserved: for
``m=0`` with insulating magnetic walls, and for ``m=0`` or ``m=1`` when `Le = 0`.
Mechanical codes other than 0 and 1 are rejected.

### Thermal Boundary Conditions

#### Fixed Temperature (bci_thermal=0, bco_thermal=0)

**Physics:** Temperature prescribed at boundary (T = 0 for perturbations)

**When to use:**
- High thermal conductivity boundaries
- Classical Rayleigh-Bénard setup
- **Most common choice**

#### Fixed Flux (bci_thermal=1, bco_thermal=1)

**Physics:** Heat flux prescribed (∂T/∂r = 0 for perturbations)

**When to use:**
- Insulating boundaries
- Internally heated systems

### Magnetic Boundary Conditions

#### Insulating (bci_magnetic=0, bco_magnetic=0)

**Physics:** No electrical currents in boundary region

**Mathematical conditions:**
- CMB: (l+1)·f + r·f' = 0
- ICB: l·f - r·f' = 0
- Toroidal: g = 0 (both boundaries)

**When to use:**
- **Earth's mantle** (silicate, electrically insulating)
- Vacuum outside
- **Default choice for most applications**

**Example:**
```julia
params = MHDParams(
    ...,
    bci_magnetic = 0,  # Insulating ICB
    bco_magnetic = 0   # Insulating CMB
)
```

#### Perfect conductor (`bci_magnetic=2` or `bco_magnetic=2`)

A stationary ideal conductor imposes ``f=0`` and zero tangential electric field.
For no-slip velocity this gives ``g'+g/r=0``. There is one constraint for each
magnetic potential at each wall. An extra poloidal diffusion constraint would
overconstrain the second-order radial equation.

For stress-free velocity, the toroidal condition includes the spheroidal
projection of ``\mathbf{u}\times\mathbf{B}_0``:

```math
E_m(g'+g/r) - [\mathbf{u}\times\mathbf{B}_0]_{\mathrm{sph}}=0.
```

#### Finite-conductivity core (`bci_magnetic=1`)

This option evolves the perturbation field in a **stationary solid core with the
same diffusivity and permeability as the fluid**. With no slip at the interface,
``f,f',g,g'`` are continuous. With slip, continuity of tangential electric field
adds the fluid motional EMF to the toroidal derivative condition.

Core potentials have the regular basis
``(r/r_i)^l a_l(x)``, ``x=2(r/r_i)^2-1``, with `N+1` Chebyshev coefficients for
``a_l``. The full vector appends `:fi` and `:gi` sections after the five fluid
sections. The core diffusion equation and the shell equations share the unknown
eigenvalue; no prescribed skin-depth frequency is used. `forcing_frequency` must
remain zero. The imposed dipole is prescribed in the shell; only its perturbation
is continued regularly through the core.

A finite-conductivity mantle (`bco_magnetic=1`) and unequal core/shell diffusivities
are not implemented. For the physical matching conditions, see the
[MagIC inner-core equations](https://magic-sph.github.io/numerics.html#magnetic-boundary-conditions-and-inner-core).

---

## Complete Workflow

### Step 1: Define Parameters

```julia
params = MHDParams(
    # Physical parameters
    E = 1e-3,
    Pr = 1.0,
    Pm = 5.0,
    Ra = 1e5,
    Le = 1e-3,        # Small background field

    # Geometry
    ricb = 0.35,      # Inner core radius
    m = 2,            # Azimuthal wavenumber
    lmax = 15,        # Spherical harmonic truncation
    N = 32,           # Radial resolution
    symm = 1,         # Equatorial symmetry

    # Background field
    B0_type = axial,  # Uniform axial field
    B0_amplitude = 1.0,

    # Boundary conditions
    bci = 1, bco = 1,              # No-slip
    bci_thermal = 0, bco_thermal = 0,  # Fixed T
    bci_magnetic = 2, bco_magnetic = 0, # Perfect conductor ICB

    # Heating
    heating = :differential
)
```

### Step 2: Build Operator

```julia
op = MHDStabilityOperator(params)

println("Operator statistics:")
println("  Matrix size: ", op.matrix_size, " × ", op.matrix_size)
println("  Number of l-modes:")
println("    Poloidal velocity (u): ", length(op.ll_u))
println("    Toroidal velocity (v): ", length(op.ll_v))
println("    Poloidal magnetic (f): ", length(op.ll_f))
println("    Toroidal magnetic (g): ", length(op.ll_g))
println("    Temperature (h): ", length(op.ll_h))
```

### Step 3: Assemble Matrices

```julia
A, B, interior_dofs, info = assemble_mhd_matrices(op)

println("\nMatrix assembly:")
println("  Total DOFs: ", size(A, 1))
println("  Interior DOFs: ", length(interior_dofs))
println("  Sparsity: ", nnz(A), " / ", size(A,1)^2,
        " = ", 100*nnz(A)/size(A,1)^2, "%")
```

### Step 4: Solve Eigenvalue Problem

#### Using the eigenvalue solver

```julia
# Keep the full coefficient-space pencil, including boundary constraints.

# Find eigenvalues with largest real part
σ, v, history = solve_eigenvalue_problem(
    A, B;
    nev=20,      # Number of eigenvalues
    tol=1e-6,    # Tolerance
    which=:LR,   # Largest real part
)

println("\nEigenvalues found:")
for i in 1:length(σ)
    println("  σ[$i] = ", real(σ[i]), " + ", imag(σ[i]), "im")
end
```

#### Using Arpack

```julia
using Arpack

σ, v = eigs(A_int, B_int, nev=20, which=:LR, tol=1e-6)
```

### Step 5: Analyze Results

```julia
# Find critical mode
idx_crit = argmax(real.(σ))

println("\nCritical mode:")
println("  Growth rate: ", real(σ[idx_crit]))
println("  Frequency: ", imag(σ[idx_crit]))
println("  Complex eigenvalue: ", σ[idx_crit])

# Check if unstable
if real(σ[idx_crit]) > 0
    println("  → UNSTABLE")
else
    println("  → STABLE")
end
```

---

## Common Use Cases

### Use Case 1: Hydrodynamic Onset (Benchmark)

**Goal:** Reproduce Christensen & Wicht (2015) Table 1

```julia
# Parameters from published benchmark
E = 4.734e-5
Pr = 1.0
ricb = 0.35
m = 9
lmax = 20
Nr = 24

params = MHDParams(
    E=E, Pr=Pr, Pm=1.0, Ra=1.6e6, Le=0.0,
    ricb=ricb, m=m, lmax=lmax, N=Nr,
    bci=1, bco=1,
    bci_thermal=0, bco_thermal=0,
    bci_magnetic=0, bco_magnetic=0
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

σ, _, _ = solve_eigenvalue_problem(
    A,
    B;
    nev=10, which=:LR,
)

σ_max = maximum(real.(σ))
ω_crit = imag(σ[argmax(real.(σ))])

println("Critical mode:")
println("  Growth rate: ", σ_max, " (should be ≈ 0)")
println("  Frequency: ", ω_crit, " (should be ≈ 0.37)")
```

**Expected results:**
- σ ≈ 0 (marginal stability at Raᶜ)
- ω ≈ 0.37 (prograde thermal wind)

### Use Case 2: MHD with Axial Field

**Goal:** Study magnetoconvection with imposed axial field

```julia
params = MHDParams(
    E=1e-3, Pr=1.0, Pm=5.0,
    Ra=1e5,           # Supercritical
    Le=1e-3,          # Weak field
    ricb=0.35,
    m=2, lmax=15, N=32,
    B0_type=axial,    # Axial background field
    bci=1, bco=1,
    bci_thermal=0, bco_thermal=0,
    bci_magnetic=0, bco_magnetic=0
)

# Scan Lehnert number
Le_values = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
growth_rates = Float64[]

for Le in Le_values
    params_le = MHDParams(
        E=1e-3, Pr=1.0, Pm=5.0, Ra=1e5, Le=Le,
        ricb=0.35, m=2, lmax=15, N=32,
        B0_type=axial,
        bci=1, bco=1,
        bci_thermal=0, bco_thermal=0,
        bci_magnetic=0, bco_magnetic=0
    )

    op = MHDStabilityOperator(params_le)
    A, B, interior_dofs, _ = assemble_mhd_matrices(op)

    σ, _, _ = solve_eigenvalue_problem(
        A,
        B;
        nev=5, which=:LR,
    )

    push!(growth_rates, maximum(real.(σ)))
    println("Le = $Le: σ_max = ", growth_rates[end])
end

# Plot growth rate vs Le (stabilization by magnetic field)
```

**Physical insight:** Increasing Le stabilizes convection (magnetic tension)

### Use Case 3: Perfect Conductor Inner Core

**Goal:** Study an ideal perfectly conducting inner wall

```julia
params = MHDParams(
    E=1e-3, Pr=1.0, Pm=5.0,
    Ra=1e5, Le=1e-3,
    ricb=0.35,
    m=2, lmax=15, N=32,
    B0_type=axial,
    bci=1, bco=1,
    bci_thermal=0, bco_thermal=0,
    bci_magnetic=2,    # ← Perfect conductor ICB (NEW!)
    bco_magnetic=0     #   Insulating CMB
)

op = MHDStabilityOperator(params)
A, B, interior_dofs, info = assemble_mhd_matrices(op)

println("Perfect conductor IC boundary:")
println("  Uses one constraint per magnetic potential at each wall")
println("  Interior DOFs: ", length(interior_dofs))

# Solve eigenvalue problem
σ, _, _ = solve_eigenvalue_problem(
    A,
    B;
    nev=10, which=:LR,
)

println("\nEigenvalues with conducting IC:")
for (i, λ) in enumerate(σ)
    println("  σ[$i] = ", real(λ), " + ", imag(λ), "im")
end
```

**Physics:** Perfect conductor IC affects magnetic boundary layer dynamics

---

## Troubleshooting

### Problem: Solver doesn't converge

**Symptoms:** Krylov iteration throws a convergence error

**Solutions:**
1. Increase `tol` to 1e-5 or 1e-4
2. Increase `nev` to find more eigenvalues
3. Try different `which` option (`:LM`, `:LI`, `:LR`)
4. Check matrix condition number: `cond(Matrix(A_int))`

```julia
# More robust solving
σ, v, history = solve_eigenvalue_problem(
    A, B;
    nev=30,       # More eigenvalues
    tol=1e-4,     # Relaxed tolerance
    maxiter=1000, # More iterations
)
```

### Problem: Growth rates are all negative (stable)

**Diagnosis:** Ra < Raᶜ (system is subcritical)

**Solutions:**
1. Increase Ra until growth rates become positive
2. Scan Ra to find Raᶜ
3. Check if correct mode (m, l) is selected

### Problem: Matrix is singular

**Symptoms:** Zero eigenvalues, solver fails

**Diagnosis:** Boundary conditions may be over-constrained

**Check:**
1. `length(interior_dofs)` should be > 0
2. `rank(B_int)` should equal `size(B_int, 1)`
3. Verify BC settings are compatible

### Problem: Results don't match Kore

**Possible causes:**
1. Different parameter definitions (check E, Pr, Pm carefully)
2. Different boundary conditions
3. Magrathea.jl uses Boussinesq (no anelastic corrections)
4. Resolution too low (increase lmax or N)

**Verification:**
```julia
# Check against Christensen & Wicht (2015) Table 1
# Parameters MUST match exactly
```

---

## Reference Tables

### Table 1: Typical Parameter Ranges

| Parameter | Earth's Core | Lab Experiments | Numerical Simulations |
|-----------|--------------|-----------------|----------------------|
| E | 10⁻¹⁵ | 10⁻³ - 10⁻⁶ | 10⁻³ - 10⁻⁷ |
| Pr | 0.1 - 1 | 0.01 - 0.1 | 0.1 - 10 |
| Pm | 10⁻⁶ | 10⁻⁵ | 1 - 10 |
| Ra/Raᶜ | 10 - 1000 | 1 - 100 | 1 - 100 |
| Le | 10⁻⁴ - 10⁻² | 10⁻³ - 0.1 | 0 - 0.1 |

### Table 2: Boundary Condition Summary

| BC Type | Parameter Value | Physical Scenario | Equations |
|---------|----------------|-------------------|-----------|
| **Velocity** |
| No-slip | bci/bco = 1 | Rigid boundaries | u=0, ∂u/∂r=0 (pol); v=0 (tor) |
| Stress-free | bci/bco = 0 | Free surface | u=0, ∂²u/∂r²=0 (pol); -r ∂v/∂r + v = 0 (tor) |
| **Thermal** |
| Fixed T | bci/bco_thermal = 0 | High conductivity | T = 0 |
| Fixed flux | bci/bco_thermal = 1 | Insulating | ∂T/∂r = 0 |
| **Magnetic** |
| Insulating | bci/bco_magnetic = 0 | Silicate mantle | (l+1)f + r·f' = 0 (CMB) |
| Perfect conductor | bci/bco_magnetic = 2 | Ideal stationary wall | f=0, tangential E=0 |
| Conducting core | bci_magnetic = 1 | Equal diffusivity/permeability | Core diffusion and interface matching |

### Table 3: Matrix Size Estimates

| lmax | N | Approx. DOFs | Memory (GB) | Solve Time |
|------|---|-------------|-------------|------------|
| 10 | 24 | ~600 | <0.1 | seconds |
| 20 | 32 | ~2000 | ~0.3 | ~10 sec |
| 30 | 48 | ~5000 | ~2 | ~1 min |
| 50 | 64 | ~15000 | ~20 | ~10 min |

---

## Getting Help

**Documentation:**
- `?MHDParams` - Parameter structure
- `?MHDStabilityOperator` - Operator construction
- `?assemble_mhd_matrices` - Matrix assembly
- `?apply_magnetic_boundary_conditions!` - Magnetic BCs

**Examples:**
- `test_mhd_basic.jl` - Basic validation
- `test/mhd_physics.jl` - Analytical magnetic physics regressions
- `example/mhd_dynamo_example.jl` - Full workflow

**References:**
- Christensen & Wicht (2015), Treatise on Geophysics, Vol. 8
- Dormy & Soward (2007), "Mathematical Aspects of Natural Dynamos"
- Kore documentation: https://github.com/repepo/kore

**Issues:**
- Report bugs: https://github.com/subhk/Magrathea.jl/issues
- Ask questions: Create discussion on GitHub

---

**Last updated:** October 26, 2025
**Magrathea.jl version:** Development
**Author:** Magrathea.jl Development Team
