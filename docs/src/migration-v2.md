# Migrating to the Problem/Solve API

This page describes the current unified interface. Its historical
`migration-v2.md` URL is retained for existing links; it does not identify a
released package version. `Project.toml` is the source of version metadata.

## Construct a problem, then solve

```julia
using Magrathea
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=8, Nr=24)
problem = OnsetProblem(params)
estimate_size(problem)
# After SLEPc initialization:
result = solve(problem; nev=6)
result.eigenvalues
result.eigenvectors
growth_rate(result)
frequency(result)
leading_mode(result)
```

Sparse solves now require the [SLEPc extension](getting_started.md#SLEPc-setup).
Load `PetscWrap` and `SlepcWrap` and call `slepc_init!` before solving.

The lower-level `solve_onset_problem`, `solve_biglobal_problem`, and
`solve_triglobal_eigenvalue_problem` interfaces remain available. Check their
docstrings for their individual return contracts instead of assuming every
solve returns a `StabilityResult`.

## Basic-state construction

```julia
bs = basic_state(params; mode=:conduction)
bs = basic_state(params; mode=:meridional, amplitude=0.01)
bs3d = basic_state(params; mode=:nonaxisymmetric, amplitude=0.01, mmax_bs=2)

biglobal = BiglobalProblem(params, bs)
triglobal = TriglobalProblem(params, bs3d, -2:2)
```

Low-level constructors accepting `ChebyshevDiffn` remain supported and allow
symbolic boundary patterns. Self-consistent construction now includes
nonlinear mean-flow inertia by default. Choose `momentum_model=:stokes`
explicitly for the weak-inertia approximation. Both axisymmetric and 3D
states include viscous meridional circulation.

The authoritative mean velocity is stored in `bs.flow`; use
`mean_flow_velocity` for physical components. Older scalar component
dictionaries are compatibility projections. See [Basic States](basic_states.md)
for residuals and nonconvergence handling.

## Source organization

Files under `src/Spectral/`, `src/BasicStates/`, `src/Stability/`,
`src/Operators/`, and `src/MHD/` share the `Magrathea` namespace.
Problem wrappers live in `src/types.jl`; unified solve dispatch is in
`src/solve.jl`. Optional SLEPc and plotting integrations live in `ext/`.

The [source map](codebase_structure.md) lists the current files, including
the nonlinear mean-flow and shared perturbation-coupling implementations.

## MHD compatibility

`MHDProblem` supports motionless conductive backgrounds. Passing a non-null
mean state is rejected. `no_field` requires `Le=0` and has no magnetic
unknowns; it is not a kinematic-dynamo calculation.

Magnetic inner-boundary flags distinguish insulating (`0`), finite
conducting core (`1`), and perfect conductor (`2`). The outer boundary
supports `0` and `2`; a finite conducting mantle is rejected.

Do not use `A[interior_dofs, interior_dofs]` on MHD tau pencils. Those indices
identify differential-equation rows, and coefficient columns cannot be
removed the same way. Prefer the high-level solve for boundary enforcement
and reconstruction.

## Size estimates and plotting

`estimate_size(problem)` prints actual harmonic counts and unreduced matrix
size. Its dense storage estimate is not an estimate of total sparse
factorization memory. Hydrodynamic `Nr` is a point count, whereas MHD `N`
is polynomial degree.

Plotting integrations activate when their weak dependencies are loaded:

```julia
using Plots
plot(result)

using CairoMakie
eigenspectrum(result)
plot_meridional(result, 1)
plot_radial(result, 1)
```

`plot(result)` and `eigenspectrum(result)` show the eigenvalues of any result;
`plot(results; sweep_param=:E)` plots growth rate against a parameter field of
each result's `params` (a triglobal result has no single `m`).

The field plots read each result's own eigenvector layout: onset and biglobal
collocation blocks, MHD Chebyshev coefficients, or the reduced blocks of every
coupled triglobal `m`. `plot_radial(result, i; field)` draws `|F_ℓ(r)|` per
retained degree for `:poloidal`, `:toroidal`, or `:temperature`; MHD results
also accept `:magnetic_poloidal` and `:magnetic_toroidal`, and triglobal
results draw one curve per coupled `(m, ℓ)`. `plot_meridional(result, i; field,
npoints)` maps the real part of the reconstructed field at `φ = 0`: the result
of `perturbation_temperature` (`:temperature`), `perturbation_velocity`
(`:ur`, `:utheta`, `:uphi`), or, for MHD, `perturbation_magnetic` (`:Br`,
`:Btheta`, `:Bphi`). Triglobal results support the velocity components only.
An eigenvector that does not match its problem's layout raises a
`DimensionMismatch`, and an unsupported field an `ArgumentError`.
