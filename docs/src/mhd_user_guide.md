# MHD User Guide

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

**Magrathea.jl MHD Implementation - Comprehensive Usage Documentation**

---

## Table of Contents

1. [Introduction](#Introduction)
2. [Quick Start](#Quick-Start)
3. [Mean flows: biglobal and triglobal MHD](#Mean-flows:-biglobal-and-triglobal-MHD)
4. [Physical Parameters](#Physical-Parameters)
5. [Boundary Conditions](#Boundary-Conditions)
6. [Complete Workflow](#Complete-Workflow)
7. [Common Use Cases](#Common-Use-Cases)
8. [Troubleshooting](#Troubleshooting)
9. [Reference Tables](#Reference-Tables)

---

## Introduction

The MHD implementation in Magrathea.jl solves the **magnetohydrodynamic eigenvalue problem** for rotating spherical shells. This is used to study:

- **Convection onset** in planetary cores
- **Magnetic modification of convection onset** with imposed axial or dipole fields
- **Magnetoconvection** in laboratory experiments
- **Linear stability** about motionless conductive MHD backgrounds (`MHDProblem`)
- **Biglobal and triglobal MHD stability** about self-consistent mean flows with an
  induced mean field (an imposed field in `OnsetParams`; see
  [Mean flows](#Mean-flows:-biglobal-and-triglobal-MHD))

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
✅ **Flexible BCs**: No-slip, stress-free, insulating, perfect conductor, finite-conductivity core or mantle
✅ **Background fields**: Axial and dipolar magnetic fields
✅ **Physics checks**: Independent Lorentz/induction, diffusion, wall, and core-matching tests; see [Codebase Structure](codebase_structure.md) for the validation files

---

## Quick Start

### Basic Hydrodynamic Onset (No Magnetic Field)

```julia
using Magrathea
using LinearAlgebra, SparseArrays

# Onset benchmark of Barik et al. (2023): ricb = 0.35, Pr = 1, no-slip and
# fixed-temperature walls, shell-thickness Ekman number Ek_d = 1e-3. E is based
# on the outer radius, E = Ek_d (1 - ricb)^2; Ra is based on the shell thickness,
# Ra = R̃a / Ek_d with the published critical value R̃a_c = 55.9.
params = MHDParams(
    E = 4.225e-4,       # Ekman number
    Pr = 1.0,           # Prandtl number
    Pm = 1.0,           # Magnetic Prandtl (irrelevant for Le=0)
    Ra = 5.59e4,        # Rayleigh number at the published onset
    Le = 0.0,           # NO magnetic field
    ricb = 0.35,        # Inner core radius
    m = 4,              # Critical azimuthal wavenumber
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

**Expected Output** (Ra at the published critical value):
- Growth rate: ≈ 0 (marginal stability, |σ| < 10⁻⁵)
- Frequency: ≈ -0.0231, the published drift frequency in units of the rotation rate

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

A finite-conductivity inner core (`bci_magnetic = 1`) or mantle
(`bco_magnetic = 1`) uses tau assembly, with fluid
residuals in `C⁽⁴⁾` (poloidal velocity) or `C⁽²⁾` (other fields). Only the highest
residual coefficients are replaced by boundary constraints; unknowns remain
Chebyshev coefficients. Tau pencils have infinite algebraic boundary eigenvalues;
shift-invert targeting with `sigma=0.0` can select finite modes near onset. Check
radial and angular convergence for the physical parameters being studied.

Every returned eigenmode is also checked against the reconstructed physical
magnetic boundary conditions. `solve(...; boundary_check=:warn)` is the default;
`boundary_check=:error` rejects a result that fails, while `:none` disables this
check. `boundary_rtol` (default `1e-6`) and `boundary_atol` (default `0`) set the
tolerances; inspect `result.extra.magnetic_boundaries`. You can repeat the
check independently with `magnetic_boundary_residuals(result; rtol=..., atol=...)`,
or pass an operator and a coefficient vector or matrix. The diagnostic evaluates
the full angular magnetic and electric fields, including components above
`lmax`. This matters at slip walls: satisfying retained tau rows, or the Galerkin
weak form, does not by itself bound the physical boundary residual at finite
angular resolution. A passing boundary check complements eigenvalue refinement
in `N` and `lmax`; it does not establish spatial convergence.

The report's `passed` flag covers all stored modes. `per_mode` contains the
`inner`, `outer`, and (when present) `mantle_outer` wall reports; `maximum`
identifies the worst mode for each wall and component. Components are
`normal_field`, `tangential_field`, and `tangential_electric`, with `nothing` for
conditions that do not apply. Each metric reports its surface ``L^2`` `residual`,
`scale`, `relative_residual`, `tolerance`, and `passed`. The scale includes the
physical wall terms and a bulk RMS field reference; `atol` has absolute
surface-norm units. On distributed workers without eigenvectors, `checked=false`
records that the boundary check was unavailable. A `no_field` problem has no
magnetic unknowns; its report has `applicable=false`.

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
and equal-material core/fluid/mantle configurations against analytical
full-sphere decay. Mantle tests also compare unequal-diffusivity layers with an
independent spherical-Bessel matching problem. At their supported wall types,
the assemblies reproduce analytical spherical-Bessel decay rates. They also
check parity separation, current-free Lorentz force, axial induction, and physical
field reconstruction. Independent boundary tests evaluate spherical strain and the
tangential electric field on computed eigenmodes of both assemblies. The slip-wall
EMF uses analytical degree-one harmonic coefficients so forbidden angular
couplings remain exactly zero, including in `Float32`.

`MHDProblem` linearizes about a **motionless conductive state** with a prescribed,
current-free axial or dipolar field, and rejects explicit `MHDProblem.basic_state`
objects. For an imposed field together with a mean flow, use the collocation
problems described in [Mean flows](#Mean-flows:-biglobal-and-triglobal-MHD).
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

## Mean flows: biglobal and triglobal MHD

An imposed field can also be combined with a mean flow. Set `B0_type`, `Le`, `Pm`, and
`magnetic_bc` in `OnsetParams`: `OnsetProblem`, `BiglobalProblem`, and `TriglobalProblem`
then add the poloidal and toroidal magnetic perturbations `b = ∇×∇×(F𝐫) + ∇×(G𝐫)` to
the collocation operator, with `B0_type` and `Le` normalized as in `MHDParams` and the
magnetic Ekman number `E/Pm`. About a mean state with flow `U`, temperature `T̄` and
mean field `B̄ = B₀ + b̄`, where `b̄` is the field the flow induces, the perturbations obey

```math
\begin{aligned}
\lambda\mathbf u + 2\hat{\mathbf z}\times\mathbf u + (\mathbf U\cdot\nabla)\mathbf u + (\mathbf u\cdot\nabla)\mathbf U
  &= -\nabla p + E\nabla^2\mathbf u + \beta r\theta\hat{\mathbf r}
     + Le^2\left[(\nabla\times\mathbf b)\times\bar{\mathbf B} + \bar{\mathbf J}\times\mathbf b\right],\\
\lambda\mathbf b &= \nabla\times(\mathbf u\times\bar{\mathbf B} + \mathbf U\times\mathbf b) + E_m\nabla^2\mathbf b,
\qquad \bar{\mathbf J} = \nabla\times\bar{\mathbf b},
\end{aligned}
```

with the heat equation unchanged. The mean state must be computed with the same field:
`basic_state(params; mode=...)` with these `OnsetParams` builds it. The `:meridional`
and `:nonaxisymmetric` modes solve the linear steady balance of the buoyancy-driven
flow and its induced field. `:selfconsistent` also includes inertia, thermal
advection, the Lorentz force `Le² J̄×b̄` of the induced field on itself, and its
advection `∇×(U×b̄)`. The lower-level constructors take the same four keywords. The
state stores `b̄` in `bs.field`, in the native potentials of `bs.flow`, and the magnetic
configuration in `bs.magnetic`. Stability problems must use the same configuration;
a mismatch, or a hydrodynamic state with a mean flow, is rejected.

```julia
params = OnsetParams(E=1e-2, Pr=1.0, Ra=2e3, χ=0.35, m=0, lmax=8, Nr=24,
                     B0_type=axial, Le=0.1, Pm=1.0, magnetic_bc=:insulating)
bs3d = basic_state(params; mode=:selfconsistent)   # flow, temperature and b̄
result = solve(TriglobalProblem(params, bs3d, 0:3); nev=6)

# Biglobal, about the axisymmetric part of a state:
p2 = OnsetParams(E=1e-2, Pr=1.0, Ra=2e3, χ=0.35, m=2, lmax=8, Nr=24,
                 B0_type=axial, Le=0.1)
biglobal = solve(BiglobalProblem(p2, basic_state(p2; mode=:meridional)); nev=6)
Br, Bθ, Bφ, r, grid = perturbation_magnetic(biglobal, 1)
```

**Supported walls.** `magnetic_bc` is `:insulating` (default) or `:perfect_conductor`,
for both walls or as an `(inner, outer)` pair. A perfect conductor requires no-slip
walls, since a slipping wall would add a motional EMF to its electric condition.
Finite-conductivity cores and mantles remain specific to `MHDProblem`. With
stress-free insulating walls, only the `m = 0` rigid rotation remains neutral and is
removed, as in `MHDProblem`.

**Resolution.** Like the tau pencil, the collocation pencil shows spurious growing
eigenvalues when the magnetic (Hartmann) boundary layers, of thickness
`√(E·E_m)/(Le·B₀)`, are under-resolved. The dipole is `χ⁻³` stronger at the inner
wall. Onset and biglobal results carry the radial and angular spectral tails of every
eigenvector in `result.extra.spectral_tail`, and a warning names a rough `Nr` when the
leading mode is under-resolved. Increase `Nr` until the leading eigenvalue converges.

**Validation.** Without a mean flow, the collocation eigenvalues converge to those of
`MHDProblem` (`test/mhd_collocation.jl`). Self-consistent mean states satisfy the
steady energy balance, in which the work against the Lorentz force equals the Ohmic
dissipation. The kinetic and magnetic energy budget of coupled triglobal modes closes
to round-off as `Nr` increases (`test/triglobal_physics.jl`).

---

## Physical Parameters

### Dimensionless Numbers

#### Ekman Number (E)

**Definition:** E = ν/(Ω r_o²), based on the outer radius r_o

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

**Definition:** Ra = αgΔTd³/(νκ), based on the shell thickness d = r_o - r_i,
as for `OnsetParams` (see the length-scale note in
[Mathematical Foundations](theory/mathematical_foundations.md))

**Physical meaning:** Measure of thermal forcing strength

**Critical value Raᶜ:**
- Depends on E, Pr, geometry, and boundary conditions, and grows steeply as E
  decreases (asymptotically like E^(-4/3))
- For ricb = 0.35, Pr = 1, no-slip and fixed-temperature walls: Raᶜ ≈ 5.59×10⁴
  at E = 4.225×10⁻⁴ (m = 4) and Raᶜ ≈ 7.52×10⁵ at E = 4.225×10⁻⁵ (m = 5),
  the published R̃aᶜ = Raᶜ·Ek_d = 55.9 and 75.2 of Barik et al. (2023)

**Parameter scans:**
```julia
# Find critical Rayleigh number
Ra_values = [4e4, 5e4, 5.5e4, 6e4, 7e4]
for Ra in Ra_values
    params = MHDParams(E=4.225e-4, Pr=1.0, Pm=1.0, Ra=Ra, Le=0.0,
                       ricb=0.35, m=4, lmax=20, N=24, ...)
    # Solve and check if growth rate > 0
end
```

#### Lehnert Number (Le)

**Definition:** Le = B₀/(√(μρ)Ω r_o), based on the outer radius like E. It enters
the Lorentz force as Le²; the induction coupling has unit strength.

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

Unequal core/fluid diffusivities are not implemented. For the physical matching conditions, see the
[MagIC inner-core equations](https://magic-sph.github.io/numerics.html#magnetic-boundary-conditions-and-inner-core).

#### [Finite-conductivity mantle](@id conducting-mantle)

Set `bco_magnetic=1` and `mantle_radius > 1` to add a **stationary conducting shell** outside the fluid,
ending at ``R_m``. Its magnetic permeability equals the fluid's;
`mantle_diffusivity_ratio=η_m/η_f` is positive and defaults to `1`. The mantle
has no velocity or thermal unknowns. Its magnetic perturbations diffuse with the
same eigenvalue as the fluid and any conducting inner core. Beyond ``R_m`` the
perturbation field is insulating and decays at infinity.

At the fluid–mantle interface ``r=1``, continuity of magnetic field gives
``f=f_m``, ``f'=f_m'``, and ``g=g_m``. Continuity of tangential electric field
uses ``\mathbf E_f=E_m\nabla\times\mathbf b_f-\mathbf u\times\mathbf B_0``
and ``\mathbf E_m=E_m(\eta_m/\eta_f)\nabla\times\mathbf b_m``. Its
spheroidal component is

```math
E_m(g'+g/r)-[\mathbf u\times\mathbf B_0]_{\mathrm{sph}}
=E_m\frac{\eta_m}{\eta_f}(g_m'+g_m/r).
```

The outer mantle surface satisfies ``(l+1)f_m+R_m f_m'=0`` and ``g_m=0``.
The coefficient vector appends `:fm` and `:gm` after the fluid sections and any
`:fi`, `:gi` core sections. Each mantle potential has `N+1` Chebyshev coefficients
on ``[1,R_m]``. Thus `N` refines the fluid, core, and mantle radial representations
together. `perturbation_magnetic(vector, op; region=:mantle)` reconstructs the
mantle field, and `region=:core` reconstructs an included conducting core,
including its regular centre. The default region remains `:fluid`.

```julia
params = MHDParams(
    E=0.01, Pr=1.0, Pm=1.0, Ra=1.0, Le=0.01,
    ricb=0.35, m=1, lmax=4, N=24, B0_type=axial,
    bci_magnetic=1, bco_magnetic=1,
    mantle_radius=1.2, mantle_diffusivity_ratio=3.0,
)
result = solve(MHDProblem(params); backend=:dense, nev=2,
               boundary_check=:error)
```

Slip at a conducting interface uses the specified continuity of the tangential
electric field in the stationary frame, including the fluid motional term.
With a conductivity contrast, this conventional interface model need not reproduce
a resolved no-slip shear layer; no boundary-layer model is added. See
[Rekier, Triana & Buffett (2025)](https://doi.org/10.1029/2024GL113585).

---

## Complete Workflow

This low-level workflow builds and solves the coefficient-tau pencil directly.
For insulating or perfectly conducting walls, `solve(MHDProblem(params))` uses the
energy-conserving Galerkin assembly instead (see
[Eigensolver and model scope](#Eigensolver-and-model-scope)).

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
println("  Differential-equation rows: ", length(interior_dofs))
println("  Sparsity: ", nnz(A), " / ", size(A,1)^2,
        " = ", 100*nnz(A)/size(A,1)^2, "%")
```

### Step 4: Solve Eigenvalue Problem

#### Using the eigenvalue solver

```julia
# Keep the full coefficient-space pencil, including boundary constraints.

# Find eigenvalues with largest real part
σ, v, info = solve_eigenvalue_problem(
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

#### Dense backend for small problems

```julia
σ, v, info = solve_eigenvalue_problem(A, B; nev=20, which=:LR, backend=:dense)
```

The dense backend eliminates the tau rows (the zero rows of `B`) exactly, drops the
infinite eigenvalues, and computes the whole finite spectrum. It costs O(n³), so
use it only for small validation problems.

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

**Goal:** Reproduce the onset benchmark of Barik et al. (2023) at Ek_d = 10⁻³

```julia
# Published critical point: m = 4, R̃aᶜ = Raᶜ·Ek_d = 55.9, ωᶜ = -0.0231
Ek_d = 1e-3
ricb = 0.35
E = Ek_d * (1 - ricb)^2   # outer-radius Ekman number
Pr = 1.0
m = 4
lmax = 20
Nr = 24

params = MHDParams(
    E=E, Pr=Pr, Pm=1.0, Ra=55.9 / Ek_d, Le=0.0,
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
println("  Frequency: ", ω_crit, " (should be ≈ -0.0231)")
```

**Expected results:**
- σ ≈ 0 (marginal stability at Raᶜ, |σ| < 10⁻⁵)
- ω ≈ -0.0231 (the published drift frequency, in units of the rotation rate)

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
        B0_type=Le > 0 ? axial : no_field,   # an imposed field requires Le > 0
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

# Plot growth rate vs Le
```

**Physical insight:** The effect of the field is not monotonic. In this scan the
growth rate is essentially unchanged up to Le = 10⁻³, rises slightly at
Le = 10⁻², and falls at Le = 0.1. A field can relax the rotational constraint as
well as add magnetic tension.

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
println("  Differential-equation rows: ", length(interior_dofs))

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
3. Try a different `which` (`:LR` or `:LI`), or pass a shift `sigma` near the
   expected eigenvalue (for example `sigma=0.0` near onset)
4. For a small problem, cross-check with `backend=:dense`

```julia
# More robust solving
σ, v, info = solve_eigenvalue_problem(
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

### Problem: Singular `B` or infinite eigenvalues

**Symptoms:** `B` is singular; a generalized solver reports infinite eigenvalues

**Diagnosis:** Each tau boundary row of `B` is zero by construction, so the tau pencil
has infinite eigenvalues. Shift-invert (`sigma`) and the dense backend return only
finite ones. The Galerkin pencil used by `solve` for insulating or perfectly
conducting walls has no boundary rows. With stress-free walls, the rigid rotation
is removed by the angular-momentum condition described above.

**Check:**
1. Solve the full `(A, B)` pencil; do not slice it to `interior_dofs`
2. Use supported boundary codes: 0 or 1 for mechanical and thermal walls, 0, 1, or 2 for magnetic walls

### Problem: Results don't match Kore

**Possible causes:**
1. Different parameter definitions (check E, Pr, Pm carefully)
2. Different boundary conditions
3. Magrathea.jl uses Boussinesq (no anelastic corrections)
4. Resolution too low (increase lmax or N)

**Verification:** Kore and most of the literature scale both E and Ra by the
shell thickness d. Pass `E = Ek_d * (1 - ricb)^2` and `Ra = Ra_d` (see the
length-scale note in [Mathematical Foundations](theory/mathematical_foundations.md)),
then reproduce Use Case 1, which `test/published_benchmarks.jl` checks.

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
| Conducting mantle | bco_magnetic = 1 | Finite stationary shell, equal permeability | Mantle diffusion, electric matching, exterior vacuum |

### Table 3: Matrix Size Estimates

Matrix size for `m = 2` with an imposed field (five fields), and the dense storage
of two complex matrices that `estimate_size(MHDProblem(params))` reports.
A single parity is `symm = ±1`; `symm = 0` keeps both.

| lmax | N | DOFs (one parity) | DOFs (`symm = 0`) | Dense storage (one parity / both) |
|------|---|-------------------|-------------------|-----------------------------------|
| 10 | 24 | 575 | 1,125 | 0.01 / 0.04 GB |
| 20 | 32 | 1,584 | 3,135 | 0.08 / 0.29 GB |
| 30 | 48 | 3,577 | 7,105 | 0.38 / 1.5 GB |
| 50 | 64 | 7,995 | 15,925 | 1.9 / 7.6 GB |

The sparse pencils store far less: for `lmax = 15`, `N = 32`, and `m = 2`, the tau
matrix `A` takes about 2 MB, against 21 MB for one dense complex matrix.

---

## Getting Help

**Documentation:**
- `?MHDParams` - Parameter structure
- `?MHDStabilityOperator` - Operator construction
- `?assemble_mhd_matrices` - Tau-pencil assembly
- `?magnetic_boundary_residuals` - Physical magnetic wall checks
- `?Magrathea.apply_magnetic_boundary_conditions!` - Magnetic tau rows (internal)

**Examples:**
- `test/published_benchmarks.jl` - Published onset benchmark
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

**Last updated:** October 3, 2026
**Magrathea.jl version:** Development
**Author:** Magrathea.jl Development Team
