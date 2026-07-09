# Triglobal Stability Analysis with Non-Axisymmetric Mean Flow

Triglobal stability analysis handles the most general case: fully three-dimensional basic states with non-axisymmetric (``m \neq 0``) components. This introduces **mode coupling** between perturbations at different azimuthal wavenumbers, requiring simultaneous solution of coupled modes.

!!! warning "Validation status (read first)"
    The non-axisymmetric basic-state machinery has been **reimplemented on a unified
    vector-spherical-harmonic basis** and is **internally validated**, but it is **not
    yet validated against an external published triglobal benchmark** (none with a
    matching non-dimensionalization has been located). The onset (exact, benchmarked)
    and biglobal-axisymmetric (reduction-validated) paths remain the absolute
    references. Current state of the machinery:

    - **Advection `ū·∇T̄` is exact (divergence form).** Computed as `∇·(ūT̄)` for
      incompressible `ū` via a vector-SH transform (`vecsh_advection`), capturing the
      full triadic (Gaunt) coupling with no scalar-`∂_θ` aliasing. MMS-validated to
      `<1e-10`; the `m=0` axisymmetric path is bit-identical to before.
    - **Azimuthal (φ) advection is captured.** The basic state carries a full real-SH
      `±m` representation (`cos mφ` at `+m`, `sin mφ` at `-m`); the self-consistent
      solver develops the `sin` modes that `ū_φ·∂_φT̄` produces.
    - **One normalization convention.** The real basic state enters the complex-SH
      Gaunt coupling through a single hinge (`_basic_state_complex_profile`,
      `ĉ_{ℓ,m}=(A∓iB)/√2`); the no-factorial ↔ orthonormal mismatch is resolved by a
      boundary rescale (`_sh_rescale`). `m=0` is identity, so the validated paths are
      unchanged.
    - **Coupling convention validated by symmetry.** The real→complex map is verified
      via φ-rotation invariance: a `sin` basic-state mode produces a coupling block of
      the same magnitude as the corresponding `cos` mode (= `-i ×` it), and a rotated
      basic state rephases the block by `e^{-i m_bs φ₀}` without changing its norm
      (test: *"Real→complex coupling: φ-rotation invariance + sin modes"*). This is an
      internal-consistency proof, not an absolute-growth-rate reference.

    - **Meridional circulation carries the `sin` partner.** The solver
      (`solve_meridional_circulation_toroidal_poloidal!`) runs over signed `m`, so a
      `sin`-phase temperature mode (`m<0`) drives its meridional flow (`u_r, u_θ`) with
      the same radial profile as the cosine partner (verified by φ-rotation symmetry,
      test *"Meridional sin partner is the φ-rotation of the cos mode"*); the `m≥0`
      path is bit-identical.

    Treat absolute triglobal growth rates as research-grade pending an external
    benchmark (none convention-matched exists; the published "non-axisymmetric"
    rotating-convection results are onset problems, already validated here).

## Physical Motivation

### When Triglobal Analysis is Required

Triglobal analysis is necessary when the background state breaks axisymmetry:

1. **Hemispheric asymmetry** - Different heat flux between hemispheres
2. **Topographic forcing** - Non-axisymmetric core-mantle boundary
3. **Large-scale convection** - Pre-existing convective patterns modifying stability
4. **Magnetic field effects** - Non-axisymmetric imposed fields
5. **Tidal forcing** - Periodic longitudinal variations
6. **Laboratory experiments** - Asymmetric heating or boundary conditions

### Real-World Applications

| System | Source of Non-Axisymmetry |
|--------|--------------------------|
| Earth's core | CMB heat flux heterogeneity, inner core asymmetry |
| Mercury | 3:2 spin-orbit resonance |
| Io, Europa | Tidal heating patterns |
| Giant planets | Non-axisymmetric deep jets |
| Stars | Active regions, spot coverage |

## Mode Coupling Physics

### The Coupling Mechanism

When the basic state contains ``m_{bs} \neq 0`` components, the advection terms couple different perturbation modes:

```math
\bar{u}_{m_{bs}} \cdot \nabla u'_m \rightarrow u'_{m + m_{bs}} + u'_{m - m_{bs}}
```

**Example**: If ``\bar{\Theta}`` contains ``Y_2^2`` (so ``m_{bs} = 2``):
- Perturbation at ``m = 4`` couples to ``m = 6`` and ``m = 2``
- Perturbation at ``m = 0`` couples to ``m = 2`` and ``m = -2``
- A cascade of couplings connects all modes differing by multiples of ``m_{bs}``

### Gaunt Coefficients

The coupling strength is determined by **Gaunt coefficients**:

```math
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \int Y_{\ell_1}^{m_1} Y_{\ell_2}^{m_2} Y_{\ell_3}^{m_3*} \, d\Omega
```

These are computed from **Wigner 3j symbols**:

```math
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ 0 & 0 & 0 \end{pmatrix}
\begin{pmatrix} \ell_1 & \ell_2 & \ell_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
```

Selection rules:
- ``m_1 + m_2 = m_3``
- ``|\ell_1 - \ell_2| \leq \ell_3 \leq \ell_1 + \ell_2`` (triangle inequality)
- ``\ell_1 + \ell_2 + \ell_3`` must be even

### Block Matrix Structure

The coupled eigenvalue problem has block structure:

```
        m=-2   m=-1   m=0    m=1    m=2
    ┌──────────────────────────────────┐
m=-2│  A₋₂   C₋₂,₋₁  0      0      0   │
    │                                  │
m=-1│ C₋₁,₋₂  A₋₁   C₋₁,₀   0      0   │
    │                                  │
m=0 │  0     C₀,₋₁   A₀    C₀,₁    0   │
    │                                  │
m=1 │  0      0     C₁,₀    A₁    C₁,₂ │
    │                                  │
m=2 │  0      0      0     C₂,₁    A₂  │
    └──────────────────────────────────┘
```

Where:
- ``A_m`` = diagonal blocks (single-mode physics: diffusion, Coriolis, buoyancy)
- ``C_{m,m'}`` = coupling blocks from basic state advection

## Mathematical Formulation

### Full 3D Basic State

The basic state contains all spherical harmonic components:

**Temperature:**
```math
\bar{T}(r, \theta, \phi) = \sum_{\ell=0}^{L_{bs}} \sum_{m=-\ell}^{\ell} \bar{\Theta}_{\ell m}(r) Y_\ell^m(\theta, \phi)
```

**Velocity:**
```math
\bar{\mathbf{u}}(r, \theta, \phi) = \sum_{\ell, m} \left[ \bar{u}_{r,\ell m}(r) Y_\ell^m \hat{\mathbf{r}} + \bar{u}_{\theta,\ell m}(r) \nabla_H Y_\ell^m + \bar{u}_{\phi,\ell m}(r) \hat{\mathbf{r}} \times \nabla_H Y_\ell^m \right]
```

### Coupled Perturbation Equations

For perturbations spanning ``m \in [m_{min}, m_{max}]``:

```math
\frac{\partial \mathbf{u}'_m}{\partial t} + 2\hat{\mathbf{z}} \times \mathbf{u}'_m + \sum_{m'} \left[ (\mathbf{u}'_m \cdot \nabla)\bar{\mathbf{u}}_{m-m'} + (\bar{\mathbf{u}}_{m'} \cdot \nabla)\mathbf{u}'_{m-m'} \right] = \ldots
```

The sum couples modes ``m`` and ``m - m'`` through basic state component ``m'``.

### Generalized Eigenvalue Problem

```math
\begin{pmatrix}
\mathbf{A}_{-2} & \mathbf{C}_{-2,-1} & & & \\
\mathbf{C}_{-1,-2} & \mathbf{A}_{-1} & \mathbf{C}_{-1,0} & & \\
& \mathbf{C}_{0,-1} & \mathbf{A}_0 & \mathbf{C}_{0,1} & \\
& & \mathbf{C}_{1,0} & \mathbf{A}_1 & \mathbf{C}_{1,2} \\
& & & \mathbf{C}_{2,1} & \mathbf{A}_2
\end{pmatrix}
\begin{pmatrix}
\mathbf{x}_{-2} \\ \mathbf{x}_{-1} \\ \mathbf{x}_0 \\ \mathbf{x}_1 \\ \mathbf{x}_2
\end{pmatrix}
= \sigma
\begin{pmatrix}
\mathbf{B}_{-2} & & & & \\
& \mathbf{B}_{-1} & & & \\
& & \mathbf{B}_0 & & \\
& & & \mathbf{B}_1 & \\
& & & & \mathbf{B}_2
\end{pmatrix}
\begin{pmatrix}
\mathbf{x}_{-2} \\ \mathbf{x}_{-1} \\ \mathbf{x}_0 \\ \mathbf{x}_1 \\ \mathbf{x}_2
\end{pmatrix}
```

## The `BasicState3D` Structure

```julia
struct BasicState3D{T}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}

    # Temperature: θ̄_ℓm(r) indexed by (ℓ, m)
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity components: ū_r,ℓm(r), ū_θ,ℓm(r), ū_φ,ℓm(r)
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}}
end
```

### Reality Conditions

For physical (real-valued) fields, coefficients must satisfy:

```math
\bar{f}_{\ell,-m} = (-1)^m \bar{f}_{\ell,m}^*
```

This is automatically enforced when constructing `BasicState3D` from physical data.

## Creating 3D Basic States

### v2.0 Unified API

In v2.0 you build an `OnsetParams`, construct a 3-D basic state with `basic_state(params; mode=…)`, wrap both in a `TriglobalProblem`, and call `solve`. Call `estimate_size` before large triglobal solves:

```julia
using Magrathea

params = OnsetParams(E=1e-5, Pr=1.0, Ra=1.5e7, χ=0.35, m=0, lmax=40, Nr=48)

# Non-axisymmetric 3-D basic state (m≠0 forcing ⇒ BasicState3D)
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)

# Wrap, then check problem size before committing memory
problem = TriglobalProblem(params, bs3d, -2:2)
estimate_size(problem)

# Solve
result = solve(problem; nev=8)
println("Growth rate: ", result.growth_rate)
println("Frequency:   ", result.frequency)
```

For the self-consistent solver with full geostrophic balance:

```julia
bs3d = basic_state(params; mode=:selfconsistent, max_iterations=50)
result = solve(TriglobalProblem(params, bs3d, -5:5); nev=6)
```

### The Full Geostrophic Basic State

For non-axisymmetric basic states, Magrathea.jl computes the **complete velocity field** including:

1. **Zonal flow** ``\bar{u}_\phi`` from thermal wind balance (∂T/∂θ forcing)
2. **Meridional circulation** ``\bar{u}_\theta``, ``\bar{u}_r`` from the φ-component of thermal wind (∂T/∂φ forcing)

The key insight is that for **m ≠ 0** modes:
- ``\partial \bar{T}/\partial \phi \propto im \bar{T} \neq 0`` drives meridional flow
- The operator ``(\hat{z}\cdot\nabla)`` couples modes ``\ell`` to ``\ell\pm1``
- A block-tridiagonal solve is required for full accuracy

See [Basic States: Full Geostrophic Balance](../basic_states.md#full-geostrophic-balance-with-meridional-circulation) for detailed theory.

### Method 1: Self-Consistent Solver (Recommended)

For non-axisymmetric cases, use `basic_state_selfconsistent` which:
- Iteratively solves the advection-diffusion equation
- Computes full meridional circulation using toroidal-poloidal decomposition
- Handles mode coupling exactly

```julia
using Magrathea

# Chebyshev setup
Nr = 64
χ = 0.35
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Non-axisymmetric heat flux BC
flux = Y00(-1.0) + Y22(-0.2)

# Self-consistent solver with full geostrophic balance
bs3d, info = basic_state_selfconsistent(
    cd, χ, E, Ra, Pr;
    flux_bc = flux,
    verbose = true
)

# All three velocity components are computed
println("Zonal modes: ", keys(bs3d.uphi_coeffs))
println("Meridional modes: ", keys(bs3d.utheta_coeffs))
println("Radial modes: ", keys(bs3d.ur_coeffs))
```

### Method 2: Standard Solver (Laplace Approximation)

For quick estimates or small amplitudes, use the standard solver:

```julia
using Magrathea

# Chebyshev setup
Nr = 64
χ = 0.35
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Define non-axisymmetric boundary modes
boundary_modes = Dict(
    (2, 0) => 0.10,   # Y₂₀: pole-equator variation
    (2, 2) => 0.05,   # Y₂₂: east-west variation
    (3, 2) => 0.02,   # Y₃₂: higher order
)

# Create 3D basic state (Laplace approximation)
bs3d = nonaxisymmetric_basic_state(
    cd, χ, E, Ra, Pr, 8, 4, boundary_modes
)
```

### Method 3: Manual Construction

For custom profiles from simulations:

```julia
# Initialize dictionaries
Nr = 64
lmax_bs = 8
mmax_bs = 3
r = cd.x

theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
dtheta_dr_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
# ... other coefficient dictionaries ...

# Populate for all (ℓ, m) pairs
for ℓ in 0:lmax_bs
    for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
        theta_coeffs[(ℓ, m)] = zeros(Nr)
        dtheta_dr_coeffs[(ℓ, m)] = zeros(Nr)
    end
end

# Set specific mode amplitudes
theta_coeffs[(2, 0)] .= your_T20_profile
theta_coeffs[(2, 2)] .= your_T22_profile

# Enforce reality condition
theta_coeffs[(2, -2)] .= conj.(theta_coeffs[(2, 2)])

# Compute derivatives
for (ℓm, coeffs) in theta_coeffs
    dtheta_dr_coeffs[ℓm] = cd.D1 * coeffs
end

# Construct BasicState3D
bs3d = BasicState3D(
    r = r, Nr = Nr,
    lmax_bs = lmax_bs, mmax_bs = mmax_bs,
    theta_coeffs = theta_coeffs,
    dtheta_dr_coeffs = dtheta_dr_coeffs,
    # ... velocity coefficients ...
)
```

### Method 4: Import from Simulation

```julia
using JLD2
using Interpolations

# Load spectral coefficients from external code
@load "simulation_3d.jld2" T_lm u_lm r_sim

# Interpolate to Magrathea.jl grid
for (ℓ, m) in keys(T_lm)
    itp = LinearInterpolation(r_sim, T_lm[(ℓ, m)])
    theta_coeffs[(ℓ, m)] = itp.(cd.x)
    dtheta_dr_coeffs[(ℓ, m)] = cd.D1 * theta_coeffs[(ℓ, m)]
end
```

## Triglobal Analysis Workflow

### Step 1: Create 3D Basic State

```julia
using Magrathea

# Parameters
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# Chebyshev operators
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# Non-axisymmetric boundary forcing
boundary_modes = Dict(
    (2, 0) => 0.10,   # Axisymmetric part
    (2, 2) => 0.08,   # Non-axisymmetric: m = 2
)

bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)
```

### Step 2: Define Triglobal Parameters

```julia
params_triglobal = TriglobalParams(
    # Physical parameters
    E = E,
    Pr = Pr,
    Ra = Ra,
    χ = χ,

    # Mode coupling range
    m_range = -2:2,           # Coupled perturbation modes

    # Resolution
    lmax = 40,
    Nr = Nr,

    # 3D Basic state
    basic_state_3d = bs3d,

    # Boundary conditions
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
```

!!! warning "Mode Range Selection"
    Choose `m_range` to be symmetric and wide enough to capture the coupling cascade. If the basic state has ``m_{bs} = 2``, modes separated by 2 will couple.

### Step 3: Estimate Problem Size

Before solving, check computational requirements:

```julia
size_report = estimate_triglobal_problem_size(params_triglobal)

println("Triglobal Problem Size:")
println("  Number of modes:     ", size_report.num_modes)
println("  Total DOFs:          ", size_report.total_dofs)
println("  Matrix dimensions:   ", size_report.matrix_size, " × ", size_report.matrix_size)
println("  DOFs per mode:       ", size_report.dofs_per_mode)
```

### Typical Problem Sizes

| m_range | lmax | Nr | Approx DOFs |
|---------|------|----|-------------|
| -1:1 | 30 | 32 | ~15,000 |
| -2:2 | 40 | 48 | ~100,000 |
| -3:3 | 45 | 56 | ~250,000 |
| -4:4 | 50 | 64 | ~500,000 |

### Step 4: Build and Solve

```julia
# Build coupled problem
println("Building coupled-mode problem...")
problem = setup_coupled_mode_problem(params_triglobal)

# Inspect coupling structure
println("Coupling graph:")
for (m, neighbors) in sort(problem.coupling_graph)
    println("  m = $m couples to: ", join(neighbors, ", "))
end

# Solve eigenvalue problem
println("Solving eigenvalue problem...")
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(
    params_triglobal;
    nev = 12,            # Number of eigenvalues
    σ_target = 0.0,
    verbose = true,
)

# Results
σ₁ = real(eigenvalues[1])
ω₁ = imag(eigenvalues[1])

println("\nLeading triglobal mode:")
println("  Growth rate: σ = $σ₁")
println("  Drift frequency: ω = $ω₁")
println("  Status: ", σ₁ > 0 ? "UNSTABLE" : "STABLE")
```

**v2.0 API** — use `TriglobalProblem` + `solve`. Always call `estimate_size` before large triglobal solves:

```julia
# Build params + a 3-D basic state
params = OnsetParams(E=1e-5, Pr=1.0, Ra=1.5e7, χ=0.35, m=0, lmax=40, Nr=48)
bs3d = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)

# Wrap in a TriglobalProblem
m_range = -2:2
problem = TriglobalProblem(params, bs3d, m_range)

# Estimate size before allocating
estimate_size(problem)

# Solve
result = solve(problem; nev=12)

println("\nLeading triglobal mode (v2.0):")
println("  Growth rate: σ = ", result.growth_rate)
println("  Drift frequency: ω = ", result.frequency)
println("  Status: ", result.growth_rate > 0 ? "UNSTABLE" : "STABLE")
```

## Post-Processing

### Extract Mode Components

Each eigenvector spans all coupled ``m`` values:

```julia
function extract_mode_component(problem, eigenvector, target_m)
    idx = problem.block_indices[target_m]
    return eigenvector[idx]
end

# Get m=0 component of leading mode
mode0_coeffs = extract_mode_component(problem, eigenvectors[:, 1], 0)

# Get m=2 component
mode2_coeffs = extract_mode_component(problem, eigenvectors[:, 1], 2)
```

### Analyze Mode Energy Distribution

```julia
function mode_energy_distribution(eigenvector, problem)
    energies = Dict{Int, Float64}()

    for m in problem.m_range
        mode_vec = extract_mode_component(problem, eigenvector, m)
        energies[m] = norm(mode_vec)^2
    end

    # Normalize to percentages
    total = sum(values(energies))
    for m in keys(energies)
        energies[m] = 100.0 * energies[m] / total
    end

    return energies
end

# Compute energy distribution
energy_dist = mode_energy_distribution(eigenvectors[:, 1], problem)

println("Energy distribution in leading mode:")
for m in sort(collect(keys(energy_dist)))
    @printf("  m = %+2d: %.1f%%\n", m, energy_dist[m])
end
```

### Reconstruct Physical Fields

```julia
# Each block contains interior DOFs for a single m.
for m in problem.m_range
    mode_vec = extract_mode_component(problem, eigenvectors[:, 1], m)
    # mode_vec contains interior DOFs for this m block.
end
```

## Finding Critical Parameters

### Critical Rayleigh Number

```julia
Ra_c, ω_c, eigvec = find_critical_rayleigh_triglobal(
    E = params_triglobal.E,
    Pr = params_triglobal.Pr,
    χ = params_triglobal.χ,
    m_range = params_triglobal.m_range,
    lmax = params_triglobal.lmax,
    Nr = params_triglobal.Nr,
    basic_state_3d = bs3d;
    Ra_guess = 1e7,
    tol = 1e-3,
)

println("Triglobal critical Rayleigh: Ra_c = $Ra_c")
```

### Parameter Sweeps

```julia
# Sweep basic state amplitude
amplitudes = [0.0, 0.02, 0.05, 0.1, 0.2]
results_sweep = []

for amp in amplitudes
    @printf("Amplitude = %.2f: ", amp)

    if amp == 0.0
        # Axisymmetric only
        boundary_modes = Dict((2, 0) => 0.1)
    else
        # Add non-axisymmetric component
        boundary_modes = Dict(
            (2, 0) => 0.1,
            (2, 2) => amp,
        )
    end

    bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)

    params = TriglobalParams(
        E = E, Pr = Pr, Ra = Ra, χ = χ,
        m_range = -2:2, lmax = 40, Nr = Nr,
        basic_state_3d = bs3d,
    )

    eigenvalues, _ = solve_triglobal_eigenvalue_problem(params; nev=4, verbose=false)

    σ = real(eigenvalues[1])
    ω = imag(eigenvalues[1])

    push!(results_sweep, (amplitude=amp, σ=σ, ω=ω))
    @printf("σ = %+.4e, ω = %+.4f\n", σ, ω)
end
```

## Performance Optimization

### Start Small

Begin with narrow mode range and increase:

```julia
# Quick test run
params_test = TriglobalParams(...,
    m_range = -1:1,
    lmax = 25,
    Nr = 32,
)

# Verify before production run
params_full = TriglobalParams(...,
    m_range = -3:3,
    lmax = 50,
    Nr = 64,
)
```

### Sparse Basic State Storage

Only populate non-zero modes:

```julia
# Good: sparse storage
boundary_modes = Dict((2, 2) => 0.1)  # Only non-zero modes

# Bad: dense storage (unnecessary)
# boundary_modes = Dict((ℓ, m) => 0.0 for ℓ in 0:10, m in -ℓ:ℓ)
```

## Complete Example

```julia
#!/usr/bin/env julia
# triglobal_complete_analysis.jl
#
# Triglobal stability analysis with non-axisymmetric basic state

using Magrathea
using Printf
using JLD2

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# === Setup ===
println("="^60)
println("Triglobal Stability Analysis")
println("Non-Axisymmetric Basic State with Mode Coupling")
println("="^60)

cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

# === Create 3D Basic State ===
boundary_modes = Dict(
    (2, 0) => 0.10,   # Pole-equator (axisymmetric)
    (2, 2) => 0.08,   # East-west (non-axisymmetric)
)

println("\nBasic state modes:")
for ((ℓ, m), amp) in boundary_modes
    println("  Y($ℓ,$m) amplitude = $amp")
end

bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)

# === Triglobal Setup ===
params = TriglobalParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m_range = -2:2,
    lmax = 35,
    Nr = Nr,
    basic_state_3d = bs3d,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# === Problem Size ===
size_report = estimate_triglobal_problem_size(params)
@printf("\nProblem size: %d DOFs across %d modes\n",
    size_report.total_dofs, size_report.num_modes)

# === Build & Solve ===
println("\nBuilding coupled problem...")
problem = setup_coupled_mode_problem(params)

println("Coupling structure:")
for (m, neighbors) in sort(problem.coupling_graph)
    println("  m = $m ↔ ", join(neighbors, ", "))
end

println("\nSolving eigenvalue problem...")
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(params; nev=8)

# === Results ===
println("\n" * "="^60)
println("RESULTS")
println("="^60)

println("\nLeading eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:min(5, length(eigenvalues))])
    @printf("  %d: σ = %+.6e, ω = %+.6f\n", i, real(λ), imag(λ))
end

σ₁ = real(eigenvalues[1])
status = σ₁ > 0 ? "UNSTABLE" : "STABLE"
println("\nSystem is $status at Ra = $Ra")

# === Energy Distribution ===
println("\nEnergy distribution (leading mode):")
total_E = 0.0
mode_E = Dict{Int, Float64}()

for m in params.m_range
    idx = problem.block_indices[m]
    E_m = norm(eigenvectors[idx, 1])^2
    mode_E[m] = E_m
    total_E += E_m
end

for m in sort(collect(keys(mode_E)))
    pct = 100.0 * mode_E[m] / total_E
    @printf("  m = %+2d: %.1f%%\n", m, pct)
end

# === Comparison to Biglobal ===
println("\n" * "="^60)
println("COMPARISON: Triglobal vs Biglobal")
println("="^60)

# Biglobal (axisymmetric basic state only)
bs_axi = meridional_basic_state(cd, χ, E, Ra, Pr;
    lmax_bs = 6, amplitude = 0.1)

params_bi = OnsetParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ,
    m = 0, lmax = 35, Nr = Nr,
    basic_state = bs_axi,
)

result_bi = solve(BiglobalProblem(params_bi, bs_axi); nev=4)
σ_biglobal = result_bi.growth_rate

@printf("\n  Biglobal (m=0 only):  σ = %+.6e\n", σ_biglobal)
@printf("  Triglobal (coupled):  σ = %+.6e\n", σ₁)
@printf("  Difference:           Δσ = %+.6e\n", σ₁ - σ_biglobal)

# === Save Results ===
@save "outputs/triglobal_analysis.jld2" eigenvalues eigenvectors params E Pr Ra χ boundary_modes

println("\nResults saved to outputs/triglobal_analysis.jld2")
```

## Checklist

Before running triglobal analysis:

- [ ] `m_range` is symmetric (e.g., -2:2, not 0:4)
- [ ] `m_range` covers coupling from basic state ``m_{bs}``
- [ ] Problem size fits in available memory
- [ ] Basic state satisfies reality conditions
- [ ] Basic state `mmax_bs` matches expected coupling
- [ ] Solver converges within iteration limit
- [ ] Energy distribution shows expected mode participation

## Comparison of Analysis Types

| Feature | Onset (No Flow) | Biglobal (Axisymm.) | Triglobal (3D) |
|---------|-----------------|---------------------|----------------|
| Basic state ``m`` | 0 only | 0 only | All ``m`` |
| Mode coupling | None | None | Yes |
| Matrix structure | Block diagonal | Block diagonal | Coupled blocks |
| DOFs per ``m`` | ``N_r \times N_\ell`` | ``N_r \times N_\ell`` | ``N_r \times N_\ell`` |
| Total DOFs | Single ``m`` | Single ``m`` | ``\sum_m N_r \times N_\ell`` |
| Memory | Low | Low | High |
| Applications | Classical onset | Thermal wind | CMB heterogeneity |

## Next Steps

- **[Onset Convection](onset_convection.md)** - Classical problem without mean flow
- **[Biglobal Stability](biglobal_stability.md)** - Axisymmetric mean flows

---

!!! info "Example Scripts"
    See the following examples in the `example/` directory:

    - `triglobal_analysis_demo.jl` - Complete triglobal stability workflow
    - `nonaxisymmetric_basic_state.jl` - Creating 3D basic states
    - `flux_bc_mean_flow.jl` - Non-axisymmetric flux BC with Y₂₂ pattern
    - `flux_bc_axisymmetric_flow.jl` - Axisymmetric flux BC with Y₂₀ pattern
