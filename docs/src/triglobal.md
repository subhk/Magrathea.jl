# Tri-Global Instability Analysis

!!! note "Eigensolver setup"
    Eigenvalue examples assume the [SLEPc setup](getting_started.md#SLEPc-setup), including loading the wrappers and calling `slepc_init!`.

## When to Use Tri-Global Analysis

Use tri-global analysis when:

- **Boundary forcing varies with longitude** (e.g., hemispheric heating, topography)
- **Zonal jets** introduce azimuthal shear that couples neighboring modes
- **Lateral heat-flux or temperature variations** inject ``m \neq 0`` components into the base state
- **Large-scale convection patterns** modify the stability of smaller-scale modes

With an imposed field in `OnsetParams` (`B0_type`, `Le`, `Pm`, `magnetic_bc`) the
coupled problem also carries magnetic perturbations, about a mean state with the
induced field of its flow; see the [MHD guide](mhd_user_guide.md#Mean-flows:-biglobal-and-triglobal-MHD).

!!! note
    If your base state is axisymmetric (`BasicState`), use single-mode biglobal analysis (`BiglobalProblem`) for efficiency.

## Mode Coupling Physics

### How Modes Couple

When a basic state has ``m_{bs} \neq 0`` components, perturbations at mode ``m`` couple to ``m \pm m_{bs}``:

```math
\bar{u}_{m_{bs}} \cdot \nabla u'_m \rightarrow u'_{m + m_{bs}} + u'_{m - m_{bs}}
```

For example, if ``\bar{\Theta}`` contains ``Y_{2,2}`` (so ``m_{bs} = 2``):
- Perturbation mode ``m=4`` couples to ``m=2`` and ``m=6``
- Mode ``m=0`` couples to ``m=2`` and ``m=-2``

### Gaunt Coefficients

The coupling strength is determined by Gaunt coefficients:

```math
G_{\ell_1 \ell_2 \ell_3}^{m_1 m_2 m_3} = \int Y_{\ell_1}^{m_1} Y_{\ell_2}^{m_2} Y_{\ell_3}^{m_3*} d\Omega
```

Scalar Gaunt utilities use `WignerSymbols.jl`. The mean-state stability operator
projects the full vector and temperature products by angular quadrature, including
velocity shear, spherical metric terms and the radial equation weights. The same
physical projection is used for axisymmetric and non-axisymmetric states.

## Setting Up a Tri-Global Problem

### Step 1: Create a 3-D Basic State

```julia
using Magrathea

# Chebyshev differentiation
cd = ChebyshevDiffn(64, [0.35, 1.0], 4)

# Non-axisymmetric boundary forcing
boundary_modes = Dict(
    (2, 0) => 0.1,    # Pole-equator variation
    (2, 2) => 0.05,   # East-west variation
)

bs3d = nonaxisymmetric_basic_state(
    cd, 0.35, 1e-5, 1e7, 1.0, 8, 4, boundary_modes
)
```

For an executed example, see the
[Y₂₂ heat-flux worked example](@ref y22-heat-flux-example). It builds a
self-consistent 3-D state, checks its resolution, and plots the boundary heat flux
and equatorial sections.

### Step 2: Define Tri-Global Parameters

```julia
# Shared physical/numerical parameters live in OnsetParams. The base mode is m=0;
# the coupled perturbation modes are supplied by `m_range` when the problem is built.
params = OnsetParams(
    E = 1e-5,
    Pr = 1.0,
    Ra = 1.2e7,
    χ = 0.35,
    m = 0,
    lmax = 40,
    Nr = 64,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)

# Wrap params + the 3-D basic state + the coupled-mode range into a TriglobalProblem.
m_range = -2:2          # coupled perturbation modes (keep symmetric around the base)
problem  = TriglobalProblem(params, bs3d, m_range)
```

!!! warning "Mode Range Selection"
    `m_range` should be symmetric around your primary mode of interest to capture both forward and backward couplings.

### Step 3: Estimate Problem Size

Before solving, check the computational requirements:

```julia
# Prints the number of coupled modes, total DOFs, and the matrix size.
estimate_size(problem)
```

### Typical Problem Sizes

| m_range | lmax | Nr | Approx DOFs |
|---------|------|----|-------------|
| -1:1 | 30 | 32 | ~8,000 |
| -2:2 | 40 | 48 | ~27,000 |
| -4:4 | 50 | 64 | ~81,000 |

These are constraint-reduced sizes with `equatorial_symmetry = :both`. Memory
and run time also depend on the coupling density and sparse-factorization fill.

## Setting Up and Solving

### Build the Coupled Problem

The `TriglobalProblem` built in Step 2 already encodes the coupling: each
perturbation mode `m` couples to its neighbours `m ± m_bs`, where `m_bs` are the
azimuthal wavenumbers present in the 3-D basic state. The resulting block
structure is:

### Understand the Block Structure

The matrices have block structure where each block couples different ``(m, \ell)`` pairs:

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
- ``A_m`` = diagonal blocks: the biglobal operator for order ``m``, including transport by the axisymmetric part of the basic state
- ``C_{m,m'}`` = coupling blocks from the basic state's azimuthal order ``m - m'``

The diagram shows a basic state with ``m_{bs} = 1``. Because the basic state is
real, the ``-m`` diagonal block is the complex conjugate of the ``+m`` block. Each
block uses constraint-reduced coordinates that satisfy the wall conditions
exactly; with stress-free walls the ``|m| \le 1`` blocks also impose zero angular
momentum, removing the neutral rigid-rotation modes.

### Solve the Eigenvalue Problem

```julia
result = solve(problem; nev = 12, sigma = 0.0, verbose = true)

σ₁ = result.growth_rate     # real part of the leading eigenvalue
ω₁ = result.frequency       # imaginary part (drift frequency)

println("Leading tri-global mode:")
println("  Growth rate: ", σ₁)
println("  Drift frequency: ", ω₁)
println("  Status: ", σ₁ > 0 ? "UNSTABLE" : "STABLE")
```

## Post-Processing

`solve` returns a `StabilityResult`: `result.eigenvalues` (all computed values),
`result.eigenvectors` (columns), `result.growth_rate`, `result.frequency`.

!!! note "Per-mode decomposition uses the coupled-mode layout"
    `StabilityResult` does not store the per-`m` block layout. Rebuild it with
    `setup_coupled_mode_problem` from the same parameters: `solve` assembles the
    problem in exactly this layout, so its `block_indices` slice the returned
    eigenvectors. Each block holds constraint-reduced coordinates (the wall
    conditions are eliminated), not raw collocation values.

```julia
p = problem.params
coupled = setup_coupled_mode_problem(TriglobalParams(
    E = p.E, Pr = p.Pr, Ra = p.Ra, χ = p.χ, m_range = problem.m_range,
    lmax = p.lmax, Nr = p.Nr, basic_state_3d = problem.basic_state,
    mechanical_bc = p.mechanical_bc, thermal_bc = p.thermal_bc,
    equatorial_symmetry = p.equatorial_symmetry))
x = leading_mode(result)    # eigenvector of the leading eigenvalue
```

### Extract Mode Components

Each eigenvector spans all coupled modes. Extract individual ``m`` components:

```julia
function extract_mode(coupled, eigenvector, target_m)
    idx = coupled.block_indices[target_m]
    return eigenvector[idx]
end

# Get the m=0 component of the leading mode
mode0_vec = extract_mode(coupled, x, 0)
```

### Reconstruct Physical Fields

```julia
# Velocity on the (r, θ) grid at longitude φ = 0, summed over all coupled m
ur, uθ, uφ = Magrathea.eigenvector_to_velocity_triglobal(collect(x), coupled; φ_slice = 0.0)

# Full 3-D field on Nφ longitudes (more expensive)
ur3, uθ3, uφ3 = Magrathea.eigenvector_to_velocity_triglobal(collect(x), coupled; Nφ = 64)
```

### Analyze Mode Energy Distribution

The reduction bases are orthonormal, so each share below is that of the block's
full radial ``P``, ``T`` and ``\Theta`` coefficients: a participation measure, not a
kinetic energy.

```julia
using LinearAlgebra

# Compute the coefficient-norm share of each m block
function mode_energy(eigenvector, coupled)
    energies = Dict{Int, Float64}()

    for m in coupled.m_range
        mode_vec = extract_mode(coupled, eigenvector, m)
        energies[m] = norm(mode_vec)^2
    end

    # Normalize
    total = sum(values(energies))
    for m in keys(energies)
        energies[m] /= total
    end

    return energies
end

energy_dist = mode_energy(x, coupled)
println("Energy distribution:")
for m in sort(collect(keys(energy_dist)))
    println("  m = $m: ", round(100 * energy_dist[m], digits=1), "%")
end
```

## Finding Critical Parameters

### Critical Rayleigh Number Search

```julia
Ra_c, ω_c, eigvec = find_critical_Ra(problem; Ra_bracket = (1e6, 1e8), tol = 1e-3)

println("Critical Rayleigh number (tri-global): ", Ra_c)
```

### Parameter Sweeps

```julia
# Scan basic state amplitude
amplitudes = [0.01, 0.05, 0.1, 0.2]
results = []

for amp in amplitudes
    boundary_modes = Dict((2, 2) => amp)
    bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)

    params = OnsetParams(E = E, Pr = Pr, Ra = Ra, χ = χ, m = 0, lmax = 40, Nr = 64)
    result = solve(TriglobalProblem(params, bs3d, -2:2); nev = 4, verbose = false)

    push!(results, (amplitude = amp, σ = result.growth_rate, ω = result.frequency))
end
```

## Performance Tips

### Start Small

Begin with narrow `m_range` and increase gradually:

```julia
# Each basic state must be built on its problem's Nr-point radial grid
# Quick test
problem_test = TriglobalProblem(OnsetParams(...; lmax=20, Nr=32), bs3d_Nr32, -1:1)

# Production run
problem_full = TriglobalProblem(OnsetParams(...; lmax=50, Nr=64), bs3d_Nr64, -3:3)
```

### Use Sparse Storage

Keep basic state dictionaries sparse - only populate non-zero modes:

```julia
# Good: Only include active modes
boundary_modes = Dict((2, 2) => 0.1)

# Avoid: Don't fill with zeros
# boundary_modes = Dict((l, m) => 0.0 for l in 0:10, m in -l:l)
```

## Complete Example

```julia
#!/usr/bin/env julia
# triglobal_analysis.jl

using Magrathea
using Printf

# === Parameters ===
E = 1e-5
Pr = 1.0
Ra = 1.5e7
χ = 0.35
Nr = 48

# === Basic State ===
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

boundary_modes = Dict(
    (2, 0) => 0.1,
    (2, 2) => 0.08,
)

bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 8, 4, boundary_modes)

# === Tri-Global Setup ===
params = OnsetParams(
    E = E, Pr = Pr, Ra = Ra, χ = χ, m = 0,
    lmax = 35, Nr = Nr,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature,
)
problem = TriglobalProblem(params, bs3d, -2:2)

# === Check Size ===
estimate_size(problem)

# === Solve ===
println("Solving eigenvalue problem...")
result = solve(problem; nev = 8)
eigenvalues = result.eigenvalues

# === Results ===
println("\n" * "="^50)
println("Leading eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:min(5, length(eigenvalues))])
    @printf("  %d: σ = %+.6e, ω = %+.6f\n", i, real(λ), imag(λ))
end

if real(eigenvalues[1]) > 0
    println("\nSystem is UNSTABLE")
else
    println("\nSystem is STABLE")
end
```

## Checklist

Before running tri-global analysis:

- [ ] `TriglobalProblem` `m_range` is consistent with the basic state's mode content
- [ ] Coupling graph matches physical expectations
- [ ] Estimated problem size is feasible for available memory
- [ ] Basic-state profiles are real cosine (``m > 0``) and sine (``m < 0``) coefficients
- [ ] Solver converges within reasonable iteration limit

## Next Steps

- **[MHD Extension](mhd_extension.md)** - Add magnetic field effects
- **[API Reference](reference.md)** - Complete function documentation

---

!!! info "Example Scripts"
    See `example/triglobal_analysis_demo.jl` for a complete working example.
