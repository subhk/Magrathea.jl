# Magrathea.jl

[![CI](https://github.com/subhk/Magrathea.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/Magrathea.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/subhk/Magrathea.jl/actions/workflows/docs.yml/badge.svg)](https://subhk.github.io/Magrathea.jl/)
[![codecov](https://codecov.io/gh/subhk/Magrathea.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/Magrathea.jl)

**Magrathea.jl** is a Julia package for linear stability analysis of convection in rotating spherical shells. It provides spectral methods to solve eigenvalue problems arising in geophysical and astrophysical fluid dynamics.

Named for the planet-building world in *The Hitchhiker's Guide to the Galaxy* — this package constructs planetary interiors, one rotating magnetized shell at a time.

## Features

- **Spectral discretization** — Chebyshev collocation for hydrodynamic stability and ultraspherical operators for MHD
- **Three analysis modes** — onset convection, biglobal (axisymmetric mean flow), and triglobal (non-axisymmetric, mode-coupled) stability
- **Boundary enforcement** — hydrodynamic constraint reduction, energy-conserving MHD Galerkin recombination for insulating or perfectly conducting walls, and MHD tau constraints for finite-conductivity walls
- **Unified solver API** — one `solve(problem)` entry point (extending `CommonSolve.solve`) across all problem types, returning a `StabilityResult`, with a sparse SLEPc backend and a dense backend for small problems
- **Critical-parameter search** — automated bracketing for critical Rayleigh numbers
- **Flexible basic states** — conductive, meridional, non-axisymmetric, and self-consistent nonlinear momentum/thermal steady states, forced by temperature or heat-flux patterns on either boundary

## Installation

Magrathea.jl is not in the General registry; install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Magrathea.jl")
```

See `Project.toml` for Julia compatibility (Julia 1.10 or later in the 1.x series). Sparse eigensolves require complex PETSc/SLEPc libraries and the `PetscWrap`/`SlepcWrap` weak dependencies; follow the [solver setup](https://subhk.github.io/Magrathea.jl/dev/getting_started/#SLEPc-setup). Small validation problems can be solved without PETSc using `solve(problem; backend=:dense)`.

## Quick Start

Onset of rotating convection — find the leading eigenvalues at fixed parameters:

```julia
using Magrathea
import PetscWrap, SlepcWrap
slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
            "-st_pc_factor_mat_solver_type mumps")

# Ekman, Prandtl, Rayleigh, radius ratio, azimuthal wavenumber, truncations
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=30, Nr=64)

problem = OnsetProblem(params)
estimate_size(problem)          # check matrix size before solving
result = solve(problem; nev=6)

result.growth_rate              # leading growth rate
result.frequency                # drift frequency
result.eigenvalues              # full returned spectrum
```

Find the critical Rayleigh number for the onset of convection:

```julia
Ra_c, ω_c, eigenvector_c = find_critical_Ra(OnsetProblem(params))
```

## Analysis Modes

| Mode | Problem type | Mean flow | Use when |
|------|--------------|-----------|----------|
| Onset convection | `OnsetProblem` | none (conductive) | fundamental onset, no background flow |
| Biglobal | `BiglobalProblem` | axisymmetric ($m=0$) | latitudinal structure, modes decoupled |
| Triglobal | `TriglobalProblem` | non-axisymmetric | longitudinal structure, azimuthal orders coupled by the mean state |
| MHD | `MHDProblem` | none; imposed magnetic field | magnetoconvection about a conductive state |
| MHD with mean flow | `BiglobalProblem` / `TriglobalProblem` with `B0_type` in `OnsetParams` | flow, temperature, and induced field | imposed field together with a mean flow |

Biglobal and triglobal analyses run on a basic state built with `basic_state`:

```julia
bs   = basic_state(params; mode=:meridional)        # axisymmetric → BiglobalProblem
bs3d = basic_state(params; mode=:nonaxisymmetric)   # 3-D → TriglobalProblem

result = solve(BiglobalProblem(params, bs))
```

`mode` accepts `:conduction`, `:meridional`, `:nonaxisymmetric`, and `:selfconsistent`.

## MHD Example

Magnetoconvection with a weak axial background field at the onset of convection for the shell-thickness Ekman number Ek_d = 10⁻³ (Barik et al. 2023), with insulating magnetic boundaries (the default):

```julia
# In the initialized SLEPc session above. As in OnsetParams, Ra is based on the
# shell thickness: Ra = R̃a / Ek_d, with Ek_d = E / (1 - ricb)^2 = 1e-3.
params = MHDParams(E=4.225e-4, Pr=1.0, Pm=1.0, Ra=5.59e4, ricb=0.35,
                   m=4, lmax=20, N=24,
                   B0_type=axial, B0_amplitude=1.0, Le=1e-3)

result = solve(MHDProblem(params))
result.growth_rate              # ≈ 0: the weak field barely moves the onset
result.frequency                # ≈ -0.0231
```

A background field requires `Le > 0`. Insulating and perfectly conducting magnetic walls use the energy-conserving Galerkin solver for every field type; finite-conductivity walls use the tau method.

Call `slepc_finalize!()` after the final solve in a session. See the [source map](https://subhk.github.io/Magrathea.jl/dev/codebase_structure/) for the current implementation paths.
