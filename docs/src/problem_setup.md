# Setting Up Your First Problem

This guide follows the current `OnsetParams → OnsetProblem → solve` API.
Complete [SLEPc setup](getting_started.md#SLEPc-setup) before executing the
eigenvalue solves.

## Define and inspect a problem

```@example first_problem
using Magrathea
params = OnsetParams(
    E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
    m=2, lmax=8, Nr=24,
    mechanical_bc=:no_slip,
    thermal_bc=:fixed_temperature,
    equatorial_symmetry=:both,
)
problem = OnsetProblem(params)
estimate_size(problem)
```

| Parameter | Meaning | Constraint |
|-----------|---------|------------|
| `E` | Ekman number based on outer radius | Positive |
| `Pr` | Viscosity / thermal diffusivity | Positive |
| `Ra` | Rayleigh number based on shell thickness | Nonnegative in a validated problem |
| `χ` | Inner / outer radius | Between 0 and 1 |
| `m` | Perturbation azimuthal order | Nonnegative |
| `lmax` | Largest retained harmonic degree | At least `m` |
| `Nr` | Radial collocation points | At least 8; convergence usually needs more |
| `mechanical_bc` | Velocity conditions at both walls | `:no_slip` or `:stress_free` |
| `thermal_bc` | Homogeneous perturbation thermal conditions | `:fixed_temperature` or `:fixed_flux` |
| `equatorial_symmetry` | Parity selection | `:both`, `:symmetric`, or `:antisymmetric` |

No-slip fixes all velocity components to zero. Stress-free imposes
impermeability and zero tangential viscous stress. Fixed flux sets the
perturbation radial temperature derivative to zero; it does not require the
basic-state heat flux itself to vanish. See
[boundary conventions](theory/mathematical_foundations.md#Boundary-Conditions)
for the potential-dependent formulas.

Inspect the unreduced collocation pencil:

```@example first_problem
op = LinearStabilityOperator(params)
A, B = assemble_matrices(op)
@assert size(A) == size(B) == (op.total_dof, op.total_dof)
(op.l_sets, size(A))
```

The total size is `Nr` times the sum of retained poloidal, toroidal, and
temperature harmonic counts. The solver eliminates boundary constraints
before its hydrodynamic eigenvalue solve and reconstructs full vectors on
return.

## Solve at fixed Rayleigh number

In an initialized SLEPc session:

```julia
result = solve(problem; nev=6, sigma=0.0, tol=1e-10, maxiter=1000)
println(result.growth_rate)
println(result.frequency)
println(result.eigenvalues)
```

`sigma` controls the spectral target. Results summarize the eigenpairs that
were found; vary targets and resolution when identifying a leading mode.

## Find a critical Rayleigh number

`find_critical_Ra` preserves the problem's boundary and symmetry settings:

```julia
Ra_c, ω_c, eigenvector_c = find_critical_Ra(
    problem; Ra_guess=params.Ra, nev=6, tol=1e-6,
)
```

For onset this returns a tuple: critical Rayleigh number, temporal frequency,
and eigenvector. It searches for a growth-rate sign change and refines the
bracket. To find the global onset, repeat over azimuthal orders and compare
the resolved critical Rayleigh numbers. A single `m` is not a global search.

Biglobal and triglobal critical searches hold their supplied mean state
fixed; rebuild it during a parameter sweep if the intended physical mean
state changes with Rayleigh number. MHD currently requires an explicit
parameter sweep using `solve`.

## Reconstruct fields

For an onset or biglobal result, use the operator-aware helpers:

```julia
velocity = perturbation_velocity(result, 1)
temperature = perturbation_temperature(result, 1)
```

The mode index refers to a column of `result.eigenvectors`. With distributed
solves, reconstruct and plot on rank zero, where eigenvectors are gathered.
The [API reference](reference.md) documents grids and return values.

MHD reconstruction uses full coefficient vectors and its MHD operator.
Triglobal results carry a coupled problem layout; individual blocks must
first be expanded using that layout. Avoid treating reduced vectors as
unreduced spectral arrays.

## Save the result

```julia
using JLD2
mkpath("outputs")
JLD2.jldsave("outputs/onset_case.jld2"; params, result)
```

Record resolution and eigenpair diagnostics with the physical parameters.
Increase `Nr` and `lmax` until growth rates, frequencies, and reconstructed
fields converge.

Continue with [Basic States](basic_states.md), [Analysis Modes](analysis/index.md),
or [Examples](examples.md).
