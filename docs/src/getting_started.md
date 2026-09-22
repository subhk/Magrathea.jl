# Getting Started

Install the core package first, then configure SLEPc for sparse eigenvalue
solves. Basic-state construction, matrix assembly, and the documentation
build work without PETSc.

## Installation

Use a Julia version allowed by `Project.toml` (Julia 1.9 or later in the 1.x
series) and clone the repository:

```bash
git clone https://github.com/subhk/Magrathea.jl.git
cd Magrathea.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Alternatively, install into an existing Julia environment:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/Magrathea.jl")
```

The [source map](codebase_structure.md) describes the package, extensions,
examples, and tests. The problem/solve interface is documented in the
[API reference](reference.md).

## SLEPc setup

The supported sparse eigensolver is `backend=:slepc`. Its Julia wrappers are
weak dependencies, so instantiating the core package does not install them.

Install matching PETSc/SLEPc libraries using complex scalars. The shift-invert
configuration below also requires MUMPS. Configure `PETSC_DIR`, `PETSC_ARCH`,
and `SLEPC_DIR` for those installations before loading the wrappers; follow
the [PetscWrap installation instructions](https://github.com/bmxam/PetscWrap.jl#how-to-install-it)
and [SlepcWrap installation instructions](https://github.com/bmxam/SlepcWrap.jl#how-to-install-it)
for library discovery. MPI must match the libraries used by the wrappers.

Add the wrappers to the environment in which you run Magrathea:

```julia
using Pkg
Pkg.add(["PetscWrap", "SlepcWrap"])
```

At the start of each solver session:

```julia
using Magrathea
import PetscWrap, SlepcWrap

slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
            "-st_pc_factor_mat_solver_type mumps")
```

Use `import` for the wrappers to avoid bringing their exported names into
the same namespace as Magrathea's API. The import activates
`MagratheaSlepcExt`. Call `slepc_finalize!()` once all solves in the session
have finished. For a script, a `try ... finally` block can ensure cleanup.

All eigenvalue examples in this documentation assume this setup unless they
explicitly show initialization themselves.

## Verify construction and assembly

This example is executed during the documentation build and needs no PETSc:

```@example installation
using Magrathea
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=6, Nr=16)
problem = OnsetProblem(params)
estimate_size(problem)
op = LinearStabilityOperator(params)
A, B = assemble_matrices(op)
@assert size(A) == size(B) == (op.total_dof, op.total_dof)
nothing # hide
```

For core regression tests, run from the checkout:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Tests include boundary conditions, mean-flow balances, assembly, and
reconstruction. PETSc/MPI runtime checks are conditional; skipped runtime
checks do not validate a PETSc installation.

## Your first calculation

After initializing SLEPc above:

```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=8, Nr=24)
result = solve(OnsetProblem(params); nev=6, sigma=0.0)
println("Growth rate: ", result.growth_rate)
println("Frequency: ", result.frequency)
```

A positive real part of an eigenvalue indicates growth. Its imaginary part
is the temporal frequency; for a mode proportional to
`exp(λ*t + im*φ)`, the phase angular velocity is `-imag(λ)/m` when `m ≠ 0`.

Small truncations are suitable for learning the API. Check resolution,
eigenpair residuals, and spectral targeting for a quantitative result.
See [First Problem](problem_setup.md) for critical searches and reconstruction.

## Running repository examples

Several research scripts assume an initialized solver session. Run them with:

```julia
using Magrathea
import PetscWrap, SlepcWrap
slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
            "-st_pc_factor_mat_solver_type mumps")
try
    include("example/mhd_dynamo_example.jl")
finally
    slepc_finalize!()
end
```

See [Examples](examples.md) for the available files and their scope.

## Building Documentation Locally

From the repository root:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. A normal local build does not deploy.
CI passes `--deploy` explicitly when publishing. To check the directory-style
URLs used on the website without deploying:

```bash
CI=true julia --project=docs docs/make.jl
```

Serve `docs/build/` through a local HTTP server for that preview.

## Troubleshooting

- **Missing SLEPc extension:** install and import both wrappers, then initialize
  SLEPc. `using Magrathea` alone is sufficient for assembly, but not a sparse solve.
- **Library or scalar-type errors:** check the wrapper library paths and that
  PETSc/SLEPc use compatible complex scalars and MPI.
- **Factorization errors:** confirm MUMPS is present for the options above.
- **Insufficient memory:** reduce radial/angular resolution and inspect
  `estimate_size(problem)` before increasing it.
- **No converged eigenpairs:** inspect SLEPc diagnostics and adjust target,
  tolerance, or iteration limit. Do not treat an unconverged solve as stability.

Solver options are documented in the [API reference](reference.md);
mean-state convergence is covered in [Basic States](basic_states.md).
