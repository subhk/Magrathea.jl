# Magrathea.jl

[![CI](https://github.com/subhk/Magrathea.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/Magrathea.jl/actions/workflows/ci.yml)
[![Documentation](https://github.com/subhk/Magrathea.jl/actions/workflows/docs.yml/badge.svg)](https://subhk.github.io/Magrathea.jl/)
[![codecov](https://codecov.io/gh/subhk/Magrathea.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/Magrathea.jl)

**Magrathea.jl** is a Julia package for linear stability analysis of convection in rotating spherical shells. It provides spectral methods to solve eigenvalue problems arising in geophysical and astrophysical fluid dynamics.

Named for the planet-building world in *The Hitchhiker's Guide to the Galaxy* — this package constructs planetary interiors, one rotating magnetized shell at a time.

## Features

- **Spectral discretization** — Chebyshev collocation for hydrodynamic stability and ultraspherical operators for MHD
- **Three analysis modes** — onset convection, biglobal (axisymmetric mean flow), and triglobal (non-axisymmetric, mode-coupled) stability
- **Boundary enforcement** — hydrodynamic constraint reduction, insulating-axial-MHD Galerkin recombination, and MHD tau constraints for dipole or conducting walls
- **Unified solver API** — one `solve(problem)` entry point across all problem types, returning a `StabilityResult`.
- **Critical-parameter search** — automated bracketing for critical Rayleigh numbers
- **Flexible basic states** — conductive, meridional, non-axisymmetric, and self-consistent nonlinear momentum/thermal steady states

## Installation

Magrathea.jl is not in the General registry; install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Magrathea.jl")
```

See `Project.toml` for Julia compatibility (Julia 1.10 or later in the 1.x series). Sparse eigensolves require complex PETSc/SLEPc libraries and the `PetscWrap`/`SlepcWrap` weak dependencies; follow the [solver setup](https://subhk.github.io/Magrathea.jl/dev/getting_started/#SLEPc-setup).

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
| Triglobal | `TriglobalProblem` | non-axisymmetric | longitudinal structure, modes coupled via Gaunt coefficients |
| MHD | `MHDProblem` | none; imposed magnetic field | magnetoconvection about a conductive state |

Biglobal and triglobal analyses run on a basic state built with `basic_state`:

```julia
bs   = basic_state(params; mode=:meridional)        # axisymmetric → BiglobalProblem
bs3d = basic_state(params; mode=:nonaxisymmetric)   # 3-D → TriglobalProblem

result = solve(BiglobalProblem(params, bs))
```

`mode` accepts `:conduction`, `:meridional`, `:nonaxisymmetric`, and `:selfconsistent`.

## MHD Example

Magnetoconvection with an axial background field (insulating magnetic boundaries, the default, route through the boundary-recombined Galerkin solver):

```julia
# In the initialized SLEPc session above:
params = MHDParams(E=4.225e-4, Pr=1.0, Pm=1.0, Ra=55.905, ricb=0.35,
                   m=4, lmax=8, N=32,
                   B0_type=axial, B0_amplitude=1.0, Le=1e-3)

result = solve(MHDProblem(params))
result.growth_rate
```

A background field requires `Le > 0`. Dipole fields and perfectly-conducting / finite-conductivity magnetic boundaries are solved via the tau method.

Call `slepc_finalize!()` after the final solve in a session. See the [source map](https://subhk.github.io/Magrathea.jl/dev/codebase_structure/) for the current implementation paths.
