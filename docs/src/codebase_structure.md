# Codebase Structure

Magrathea defines one main Julia module, `Magrathea`. The directories under `src/` group files included into that namespace; they do not define separate `Magrathea.MHD` or `Magrathea.Stability` modules. Optional integrations are Julia package extensions under `ext/`.

## Public entry points

| Problem | Parameters and background | Numerical path |
|---------|---------------------------|----------------|
| `OnsetProblem` | `OnsetParams`; motionless conductive state | Collocation and boundary-constraint reduction |
| `BiglobalProblem` | `OnsetParams` and axisymmetric `BasicState` | Single azimuthal order with full mean-flow advection and shear |
| `TriglobalProblem` | `OnsetParams`, `BasicState3D`, signed `m_range` | Coupled azimuthal orders with the same vector linearization |
| `MHDProblem` | `MHDParams`; motionless conductive state and imposed field | Energy-conserving Galerkin or coefficient-tau assembly |

All four problem wrappers and `StabilityResult` live in `src/types.jl`. `src/solve.jl` dispatches `solve(problem)` and the supported `find_critical_Ra` searches; `solve` extends `CommonSolve.solve`, so it composes with other packages that use that interface. `MHDProblem` does not accept a prescribed mean flow or expose `find_critical_Ra(MHDProblem(...))`; an imposed field with a mean flow uses the collocation problems with magnetic options in `OnsetParams`.

See [Analysis Modes](analysis/index.md), [Getting Started](getting_started.md), and the [API Reference](reference.md).

## Source map

### Package entry and shared API

| File | Responsibility |
|------|----------------|
| `src/Magrathea.jl` | Dependencies, include order, public exports, and the `CommonSolve.solve` import |
| `src/Stability/magnetic.jl` | `BackgroundField`, magnetic wall options, imposed-field profiles, and the magnetic consistency and resolution checks shared by basic states and the collocation operators |
| `src/types.jl` | Problem wrappers, results, size estimates, `basic_state(params; ...)`, and the plotting and reconstruction function stubs |
| `src/validation.jl` | Parameter and basic-state consistency checks |
| `src/solve.jl` | Solve dispatch (SLEPc or dense backend), critical searches, MHD resolution and magnetic-wall checks, and result reconstruction wrappers |
| `src/show.jl` | Display methods for parameters, problems, and results |

### Radial spectral tools

| File | Responsibility |
|------|----------------|
| `src/Spectral/Spectral.jl` | Includes the spectral files |
| `src/Spectral/chebyshev.jl` | Lobatto nodes, differentiation matrices, and coefficient transforms |
| `src/Spectral/ultraspherical.jl` | Banded derivative, conversion, and multiplication operators |
| `src/Spectral/galerkin.jl` | Boundary-recombined trial bases and radial Galerkin operators |

Hydrodynamic `Nr` counts **collocation points**. MHD `N` is **polynomial degree**, giving `N + 1` coefficients per unrecombined radial block.

### Basic states and nonlinear mean flow

| File | Responsibility |
|------|----------------|
| `src/BasicStates/BasicStates.jl` | Includes the basic-state files |
| `src/BasicStates/basic_state.jl` | `BasicState`, `BasicState3D`, symbolic boundary harmonics, and conductive/Stokes constructors forced by temperature or heat-flux patterns on either wall |
| `src/BasicStates/steady_flow.jl` | Internal `SolenoidalMeanFlow`, viscous Coriolis balance (with the Lorentz force and induction of an imposed field for MHD states), thermal transport, and the `mean_flow_velocity` and `mean_temperature` evaluators |
| `src/BasicStates/nonlinear_flow.jl` | Nonlinear momentum forcing (inertia, and the Lorentz force and EMF of an induced field), fixed-point residuals, damped Picard iteration, and the Newton–Krylov/continuation fallback |
| `src/BasicStates/mean_flow_resolution.jl` | Physical velocity and temperature norms for independent spatial-refinement checks |
| `src/BasicStates/boundary_compatibility.jl` | Checks native or legacy mean-flow traces against stationary mechanical wall conditions |
| `src/BasicStates/advection_diffusion.jl` | Self-consistent constructors, scalar transport helpers, radial Poisson solves, meridional-circulation solves, and angular coupling coefficients |
| `src/BasicStates/basic_state_operators.jl` | Harmonic coupling utilities and axisymmetric stability-assembly adapters |
| `src/BasicStates/sh_transform.jl` | Spherical-harmonic evaluation and transforms |

The noniterated `:meridional` and `:nonaxisymmetric` constructors use a conductive temperature and a Stokes–Coriolis velocity. The `:selfconsistent` constructor includes nonlinear mean-flow inertia and temperature advection by default; `momentum_model=:stokes` explicitly omits momentum inertia.

Both 2D and 3D states can have all three velocity components. The authoritative divergence-free representation is `bs.flow`; use `mean_flow_velocity` and `mean_temperature` for physical values. Scalar component dictionaries are compatibility projections. See [Basic States](basic_states.md) for normalization and convergence requirements.

### Hydrodynamic stability

| File | Responsibility |
|------|----------------|
| `src/Stability/Stability.jl` | Includes the stability files |
| `src/Stability/linear.jl` | `OnsetParams`, `LinearStabilityOperator`, harmonic layouts (with magnetic potentials for an imposed field), collocation assembly, and constraint reduction (including the stress-free zero-angular-momentum condition) |
| `src/Stability/solver.jl` | SLEPc and dense eigenvalue dispatch, SLEPc lifecycle hooks, and critical searches |
| `src/Stability/dof_ownership.jl` | Row ownership and distributed assembly helpers |
| `src/Stability/velocity.jl` | Hydrodynamic perturbation reconstruction, including coupled triglobal modes, and grid-based potential conversion |
| `src/Stability/mean_flow_coupling.jl` | Shared vector advection/shear, thermal, Lorentz-force, and induction coupling for biglobal and triglobal assembly |
| `src/Stability/onset.jl` | Onset parameter adapters, solves, scans, and scaling estimates |
| `src/Stability/biglobal.jl` | Axisymmetric-state solve adapters, critical searches, and comparison utilities |
| `src/Stability/triglobal.jl` | Signed azimuthal blocks, coupling, size estimates, triglobal solves, and critical searches |

Biglobal and triglobal include both advection of perturbations by the mean flow and advection of the mean flow by perturbations, together with thermal transport. They share the physical vector coupling implementation.

### Coefficient operators and boundary helpers

| File | Responsibility |
|------|----------------|
| `src/Operators/Operators.jl` | Includes the coefficient-operator files |
| `src/Operators/sparse_operator.jl` | `SparseOnsetParams`, `SparseStabilityOperator`, and ultraspherical hydro assembly |
| `src/Operators/boundary_conditions.jl` | Radial endpoint constraints and magnetic boundary helper operators |

These coefficient operators coexist with collocation; their presence does not mean that every public solve uses ultraspherical Galerkin assembly.

### Rotating MHD

| File | Responsibility |
|------|----------------|
| `src/MHD/MHD.jl` | Includes the MHD files |
| `src/MHD/types.jl` | Background-field enum, `MHDParams`, `MHDStabilityOperator`, radial operators, and harmonic layouts |
| `src/MHD/dipole.jl` | Radial-power changes for the imposed dipole |
| `src/MHD/operator_functions.jl` | Lorentz, induction, magnetic mass, and diffusion blocks |
| `src/MHD/assembly.jl` | Full coefficient-tau pencil, indexing, and owned-row assembly; includes the MHD boundary file |
| `src/MHD/boundary_conditions.jl` | Magnetic wall constraints, motional electric-field terms, and finite conducting-core/mantle equations |
| `src/MHD/boundary_diagnostics.jl` | Physical magnetic and electric wall residuals, including angular content beyond the retained harmonics |
| `src/MHD/galerkin_assembly.jl` | Petrov–Galerkin assembly and expansion back to full coefficient vectors |
| `src/MHD/energy_galerkin.jl` | Energy-conserving Galerkin assembly used by the MHD solve for insulating or perfectly conducting walls |
| `src/MHD/reconstruct.jl` | Velocity, temperature, and magnetic perturbation reconstruction in the fluid and magnetic solid regions, and spectral-tail resolution measures |

The high-level MHD solve selects:

| Configuration | Discretization |
|---------------|----------------|
| Insulating or perfectly conducting magnetic walls (`no_field`, `axial`, or `dipole`) | Energy-conserving Galerkin: every equation tested against its trial basis in the energy inner product |
| Finite conducting inner core or outer mantle (`bci_magnetic=1` or `bco_magnetic=1`) | Ultraspherical coefficient-tau pencil with additional evolving solid-region coefficients |

`no_field` requires `Le=0` and has no magnetic perturbation unknowns. It is a hydrodynamic limit, not a kinematic-dynamo solver. Imposed `axial` and `dipole` fields require `Le>0`. A finite mantle uses an explicit `mantle_radius>1`, a positive `mantle_diffusivity_ratio`, and an insulating exterior. The mantle is stationary and has the fluid's magnetic permeability.

For tau assembly, `interior_dofs` identifies differential-equation **rows**; it is not a set of removable coefficient columns. Solve the full constrained pencil. Galerkin vectors must be expanded with their recombination layout before reconstruction. The public `solve` handles these paths. See the [MHD User Guide](mhd_user_guide.md) and [boundary conventions](theory/mathematical_foundations.md#Boundary-Conditions).

## Optional extensions

| File | Loaded by | Responsibility |
|------|-----------|----------------|
| `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` | `PetscWrap` and `SlepcWrap` | SLEPc eigenproblems and distributed assembly/reduction |
| `ext/MagratheaSlepcExt/raw_petsc.jl` | The SLEPc extension | Operations missing from the PETSc/SLEPc wrappers |
| `ext/MagratheaRecipesBaseExt/MagratheaRecipesBaseExt.jl` | `RecipesBase`, normally through `Plots` | Plot recipes |
| `ext/MagratheaMakieExt/MagratheaMakieExt.jl` | `Makie`, normally through a plotting backend | Spectrum and field plots |

SLEPc is the supported sparse eigensolver. Load its weak dependencies and call `slepc_init!` before solving. Core construction, assembly, and many physics tests run without PETSc. Every `solve` method also accepts `backend=:dense`, a dense LAPACK eigensolve for small validation problems; it is not a replacement for SLEPc at production sizes.

## Execution paths

1. Construct and validate parameters and a problem wrapper.
2. For biglobal/triglobal, build a mean state and check its convergence.
3. Build harmonic layouts and radial operators; assemble the pencil.
4. Enforce boundaries through hydro constraint reduction, MHD recombination, or MHD tau equations.
5. Solve with the initialized SLEPc extension, or with `backend=:dense` for small problems.
6. Return `StabilityResult` and reconstruct using its operator/layout.

For distributed solves, eigenvalues are available on all ranks, while eigenvectors are gathered to rank zero. Perform field reconstruction and plotting on that rank.

## Tests, examples, and documentation

| Location | Purpose |
|----------|---------|
| `test/runtests.jl` | Core regression entry point |
| `test/published_benchmarks.jl` | Published onset benchmark (Barik et al. 2023) for the onset and MHD solvers |
| `test/thermal_wind.jl` | Viscous mean-flow balance |
| `test/nonlinear_mean_flow.jl` | Nonlinear momentum and thermal residuals |
| `test/mean_flow_coupling.jl` | Independent mean-flow linearization checks |
| `test/mhd_collocation.jl` | Collocation MHD against the MHD solver, self-consistent MHD mean states, and MHD biglobal/triglobal problems |
| `test/triglobal_physics.jl` | Triglobal invariants (uncoupled limit, longitude rotation, second-order coupling), stress-free perturbation walls, and physical-space energy budgets of coupled hydrodynamic and MHD modes |
| `test/inner_boundary_forcing.jl` | Inner-wall temperature and heat-flux patterns for basic states and the matching perturbation condition |
| `test/stress_free_and_flux_bcs.jl` | Removal of the stress-free rigid rotation and per-wall thermal conditions |
| `test/boundary_physics.jl` | Physical mechanical, thermal, and magnetic wall conditions |
| `test/critical_ra_api.jl` | `find_critical_Ra` return contract and Rayleigh bracket keywords |
| `test/correctness_regressions.jl` | Regression checks for confirmed solver, basic-state, and MHD correctness fixes |
| `test/mhd_physics.jl` | Lorentz/induction and MHD spectral checks |
| `test/mhd_boundary_conditions.jl` | MHD wall layouts and conducting-core behavior |
| `test/magnetic_boundary_diagnostics.jl` | Physical magnetic and electric wall residuals, including finite cores and mantles |
| `test/galerkin_radial.jl` | Boundary recombination and radial operators |
| `test/distributed_assembly.jl` | Owned-row assembly consistency without MPI |
| `test/slepc_backend.jl` | Extension dispatch and conditional PETSc runtime checks |
| `example/` | Research scripts; see [Examples](examples.md) for setup requirements |
| `docs/src/` | Documenter Markdown sources |
| `docs/make.jl` | Navigation, local builds, and explicit deployment |
| `docs/Project.toml` | Docs environment: Documenter, and CairoMakie for figures drawn by executed examples |
| `Project.toml` | Package metadata, dependencies, extensions, and compatibility |
| `Manifest.toml` | Resolved dependency versions for a checkout |

Run `julia --project=. -e 'using Pkg; Pkg.test()'` for core tests. PETSc/MPI runtime checks need their external libraries and launcher; a core test pass alone does not validate that runtime. Build the docs using [Getting Started](getting_started.md#Building-Documentation-Locally).
