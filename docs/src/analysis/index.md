# Analysis Modes

Choose the problem from the symmetry and physics of the background state.
All eigenvalue examples assume an initialized
[SLEPc session](../getting_started.md#SLEPc-setup).

## Supported problems

| Problem | Mean velocity | Magnetic background | Azimuthal structure |
|---------|---------------|---------------------|---------------------|
| `OnsetProblem` | Zero | None | Independent single `m` |
| `BiglobalProblem` | Axisymmetric, three components | None | Independent single `m` |
| `TriglobalProblem` | Nonaxisymmetric, three components | None | Coupled signed `m_range` |
| `MHDProblem` | Zero | Imposed axial/dipole, or hydrodynamic `no_field` | Independent single `m` |

The MHD solver currently does not combine an imposed magnetic field with a
prescribed hydrodynamic mean flow.

## Onset convection

Use a conductive temperature and zero mean velocity to study classical
convection onset. A fixed `m` solve gives that order's stability;
determining global onset requires scanning azimuthal orders and resolving
each candidate mode.

```julia
using Magrathea
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                     m=2, lmax=8, Nr=24)
result = solve(OnsetProblem(params); nev=6, sigma=0.0)
Ra_c, ω_c, eigenvector_c = find_critical_Ra(OnsetProblem(params); Ra_guess=100.0)
```

[Onset guide](onset_convection.md)

## Biglobal stability

Axisymmetric boundary heating and mean flow preserve each perturbation
azimuthal order. The current linearization includes all mean-velocity
components, their shear, and thermal advection.

```julia
params = OnsetParams(E=1e-2, Pr=1.0, Ra=30.0, χ=0.35,
                     m=1, lmax=6, Nr=24)
bs = basic_state(params; mode=:meridional, amplitude=0.01)
result = solve(BiglobalProblem(params, bs); nev=6)
```

This constructor uses the conductive-temperature/Stokes approximation.
For a nonlinear steady mean state, use `basic_state_selfconsistent`
with axisymmetric forcing and verify its convergence.

[Biglobal guide](biglobal_stability.md)

## Triglobal stability

A nonaxisymmetric background couples perturbation orders through sums with
the background's azimuthal orders. Both signs of perturbation `m` are
supported. The mean-flow coupling uses the same physical vector
linearization as biglobal.

```julia
params = OnsetParams(E=1e-2, Pr=1.0, Ra=30.0, χ=0.35,
                     m=0, lmax=6, Nr=24)
bs3d = basic_state(params; mode=:nonaxisymmetric, amplitude=0.01, mmax_bs=2)
problem = TriglobalProblem(params, bs3d, -2:2)
estimate_size(problem)
result = solve(problem; nev=6)
```

The background truncation and perturbation `m_range` are separate choices.
Nonlinear transport can generate background orders beyond those present in
the boundary forcing. Check both truncations when refining a calculation.

[Triglobal guide](triglobal_stability.md)

## Rotating MHD

```julia
params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.01,
    ricb=0.35, m=1, lmax=6, N=16, B0_type=axial)
result = solve(MHDProblem(params); nev=6, sigma=0.0)
```

Insulating axial cases use boundary-recombined Galerkin assembly. Dipole or
conducting-wall cases use the coefficient-tau formulation, with extra
evolving coefficients for a finite conducting inner core. `no_field`
requires `Le=0` and omits magnetic unknowns.

[MHD guide](../mhd_user_guide.md)

## Numerical cost and interpretation

Use `estimate_size` before solving. Triglobal size increases with the
number of azimuthal blocks, their harmonic counts, and radial resolution.
Sparse-factorization fill and coupling density also affect memory and runtime.

An eigenvalue with positive real part establishes growth about the selected
background at the selected truncation. Quantitative conclusions require
converged mean states, radial/angular refinement, and adequate sampling of
the spectrum. See [Examples](../examples.md) for executable setup examples
and the [source map](../codebase_structure.md) for implementation details.
