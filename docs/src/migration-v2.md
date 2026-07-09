# Migrating to Magrathea.jl v2.0

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">v1.x &rarr; v2.0</div>
  <h1>Migrating to the v2.0 problem / solve API.</h1>
  <p>Everything that changed from v1.x to v2.0, and how to update your code.</p>
</div>

## 1. Basic State API

**Before (v1.x):**
```julia
cd = ChebyshevDiffn(Nr, [chi, 1.0], 4)
bs = conduction_basic_state(cd, chi)
bs = meridional_basic_state(cd, chi, E, Ra, Pr, lmax_bs, amplitude)
bs = nonaxisymmetric_basic_state_selfconsistent(cd, chi, E, Ra, Pr, ...)
```

**After (v2.0):**
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64)
bs = basic_state(params; mode=:conduction)
bs = basic_state(params; mode=:meridional, amplitude=0.05)
bs = basic_state(params; mode=:selfconsistent, max_iterations=50, tol=1e-10)
bs = basic_state(params; mode=:nonaxisymmetric, mmax_bs=2)
```

The `ChebyshevDiffn` is constructed automatically from `params`.

## 2. Solve API

**Before (v1.x):**
```julia
eigenvalues, eigenvectors = solve_onset_problem(onset_params; nev=6)
eigenvalues, eigenvectors = solve_biglobal_problem(biglobal_params; nev=6)
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(triglobal_params; nev=6)
```

**After (v2.0):**
```julia
result = solve(OnsetProblem(params); nev=6)
result = solve(BiglobalProblem(params, bs); nev=6)
result = solve(TriglobalProblem(params, bs3d, -5:5); nev=6)
result = solve(MHDProblem(mhd_params); nev=6)

# Access results:
result.eigenvalues       # Vector{Complex{T}}
result.eigenvectors      # Matrix{Complex{T}}
result.growth_rate       # T (max real part)
result.frequency         # T (imag of fastest-growing)
growth_rate(result)      # same as result.growth_rate
leading_mode(result)     # eigenvector of fastest-growing mode
```

## 3. Source Code Organization

Source files have moved from flat `src/` into subdirectories:

```
src/Spectral/     — ChebyshevDiffn, ultraspherical methods
src/Operators/    — SparseOperator, boundary conditions
src/BasicStates/  — BasicState, advection-diffusion, thermal wind
src/Stability/    — OnsetParams, eigenvalue solvers, analysis modes
src/MHD/          — MHDParams, Lorentz/induction operators, assembly
```

All symbols remain in the `Magrathea` namespace. No import changes needed unless you were including standalone module files directly.

## 4. Removed v1 Compatibility Constructors

Older v1 convenience constructors and tuple-returning mode helpers have been removed. Use `OnsetParams` with the `χ` radius ratio, wrap it in the appropriate problem type, and call `solve`.

## 5. New: Input Validation

Parameter types now validate inputs when wrapped in problem types:

```julia
# These throw ArgumentError when wrapped in OnsetProblem:
OnsetProblem(OnsetParams(E=-1.0, ...))    # E must be positive
OnsetProblem(OnsetParams(χ=1.5, ...))     # χ must be in (0,1)
OnsetProblem(OnsetParams(Nr=4, ...))      # Nr must be >= 8

# These emit warnings:
OnsetProblem(OnsetParams(Nr=12, ...))     # "Nr is very low"
OnsetProblem(OnsetParams(E=0.5, ...))     # "E is unusually large"
```

## 6. New: Pretty-Printing

All public types now have custom `show` methods:

```julia
julia> params
OnsetParams{Float64}
  E  = 0.001    Pr = 1.0    Ra = 100.0    χ = 0.35
  m  = 4         lmax = 30   Nr = 64
  BCs: no_slip | fixed_temperature
  Symmetry: both
```

## 7. New: Problem Size Estimation

```julia
estimate_size(OnsetProblem(params))
# OnsetProblem size estimate
# ├── l-modes: 27 (m=4, lmax=30, both)
# ├── degrees of freedom per mode: 195 (Nr=64, 3 fields)
# ├── matrix size: 5265 × 5265
# └── dense storage estimate: ~0.8 GB
```

## 8. New: Plotting

```julia
# Plots.jl recipes (lightweight)
using Magrathea, Plots
plot(result)                    # eigenvalue spectrum

# Makie (interactive)
using Magrathea, CairoMakie
eigenspectrum(result)           # interactive spectrum with hover
plot_meridional(result, 1)      # meridional slice
plot_radial(result, 1)          # radial profiles
```

Plotting packages are weak dependencies — they're only loaded when you `using Plots` or `using CairoMakie`.
