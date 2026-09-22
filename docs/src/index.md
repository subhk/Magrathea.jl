# Magrathea.jl Documentation

```@raw html
<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Linear stability in rotating spherical shells</div>
  <h1>Spectral eigenvalue problems for rotating convection &amp; MHD.</h1>
  <p>Analyze convection onset, stability about axisymmetric or three-dimensional
  mean flows, and magnetoconvection in imposed axial or dipole fields.</p>
</div>
```

[Get started](getting_started.md) · [First problem](problem_setup.md) ·
[Examples](examples.md) · [API reference](reference.md)

## Supported problems

| Problem | Background | Numerical path |
|---------|------------|----------------|
| `OnsetProblem` | Motionless conductive state | Chebyshev collocation with boundary-constraint reduction |
| `BiglobalProblem` | Axisymmetric temperature and three-component velocity | Collocation with mean-flow advection and shear |
| `TriglobalProblem` | Nonaxisymmetric temperature and velocity | Coupled signed azimuthal orders |
| `MHDProblem` | Motionless conductive state and imposed magnetic field | Ultraspherical Galerkin for insulating axial fields; tau for dipole or conducting walls |

The self-consistent hydrodynamic mean-state solver includes nonlinear
momentum inertia and thermal advection in both 2D and 3D. Its Stokes
approximation is available explicitly. MHD currently uses a motionless basic
state; `no_field` gives a hydrodynamic limit without magnetic unknowns.

[Analysis Modes](analysis/index.md) explains which problem to choose.
[Basic States](basic_states.md) describes approximations, boundary forcing,
and convergence checks. The [MHD User Guide](mhd_user_guide.md) covers magnetic
fields and boundaries.

## Quick start

Complete the [SLEPc setup](getting_started.md#SLEPc-setup) first. The supported
sparse eigensolver requires `PetscWrap`, `SlepcWrap`, and compatible complex
PETSc/SLEPc libraries.

```julia
using Magrathea
import PetscWrap, SlepcWrap

slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
            "-st_pc_factor_mat_solver_type mumps")
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=8, Nr=24)
problem = OnsetProblem(params)
estimate_size(problem)
result = solve(problem; nev=6, sigma=0.0)
println((growth_rate=result.growth_rate, frequency=result.frequency))
# At the end of the session, after all solves:
slepc_finalize!()
```

These small truncations demonstrate the API. Increase radial and angular
resolution, vary the spectral target, and check eigenpair residuals before
using a growth rate quantitatively. The fastest-growing returned eigenpair
need not be the fastest-growing eigenpair of the complete spectrum.

Construction and matrix assembly do not require PETSc:

```@example home_assembly
using Magrathea
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=6, Nr=16)
op = LinearStabilityOperator(params)
A, B = assemble_matrices(op)
(size(A), size(B))
```

## Code organization

The source directories share the `Magrathea` namespace.

| Location | Responsibility |
|----------|----------------|
| `src/types.jl`, `src/solve.jl` | All four problem wrappers, results, and unified solve dispatch |
| `src/Spectral/` | Chebyshev and ultraspherical radial tools |
| `src/BasicStates/` | Conductive, viscous, and nonlinear steady mean states |
| `src/Stability/` | Hydro assembly, shared vector mean-flow coupling, and eigenvalue dispatch |
| `src/Operators/` | Coefficient operators and radial boundary helpers |
| `src/MHD/` | Lorentz/induction, magnetic walls/core, MHD assembly and reconstruction |
| `ext/` | PETSc/SLEPc and plotting integrations |

The [Codebase Structure](codebase_structure.md) page maps individual files,
execution paths, and validation tests.

## Physical and numerical conventions

Radii run from `χ` to 1. The Ekman number uses outer radius, while the public
Rayleigh number uses shell thickness; the code converts buoyancy internally
by `Ra / (1 - χ)^3`. See
[Mathematical Foundations](theory/mathematical_foundations.md) before
comparing against a different nondimensionalization.

Hydrodynamic `Nr` counts collocation points. MHD `N` is polynomial degree
and gives `N + 1` radial coefficients. Boundary recombination removes tau
constraints from supported Galerkin pencils; spectral convergence must still
be checked.

Use `estimate_size(problem)` before allocating large systems. Actual memory
and runtime depend on harmonic truncation, mean-flow coupling, boundary
conditions, and sparse factorization.

## Documentation guide

- [Installation and solver setup](getting_started.md)
- [First problem](problem_setup.md)
- [Analysis modes](analysis/index.md)
- [Basic states](basic_states.md) and [triglobal details](triglobal.md)
- [MHD extension](mhd_extension.md) and [MHD user guide](mhd_user_guide.md)
- [Mathematical foundations](theory/mathematical_foundations.md) and [spectral methods](theory/spectral_methods.md)
- [API reference](reference.md), [source map](codebase_structure.md), and [migration guide](migration-v2.md)
- [FAQ](faq.md)

## Requirements and citation

Julia compatibility is declared in `Project.toml` (currently Julia 1.9 and
later in the 1.x series). Sparse eigensolves additionally require the SLEPc
extension. Plotting packages are optional weak dependencies.

When citing a calculation, include the Magrathea commit or package version
used, the parameters, and the numerical truncations. The current package
version is read directly from the project:

```@eval
using Magrathea, TOML, Markdown
project = TOML.parsefile(joinpath(pkgdir(Magrathea), "Project.toml"))
Markdown.parse("Package version: **" * project["version"] * "**.")
```

[Magrathea.jl source repository](https://github.com/subhk/Magrathea.jl).
Released under the MIT License.
