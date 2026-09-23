# Examples

The repository contains research scripts under `example/`. Their numerical
resolutions and parameter scans can be more expensive than the small examples
below. Complete [SLEPc setup](getting_started.md#SLEPc-setup) before running
scripts that compute eigenvalues.

## Repository scripts

| File | Purpose | Needs SLEPc |
|------|---------|-------------|
| `example/linear_stability_demo.jl` | Onset eigenvalues over azimuthal orders and perturbation reconstruction | Yes (initializes it) |
| `example/Rac_lm.jl` | Self-contained critical Rayleigh numbers per degree for a non-rotating shell | No |
| `example/basic_state_onset_example.jl` | Critical Rayleigh numbers about conductive and meridional basic states | Yes |
| `example/boundary_driven_jet.jl` | Axisymmetric boundary-forced mean flow | No |
| `example/nonaxisymmetric_basic_state.jl` | Nonaxisymmetric temperature and flow construction | No |
| `example/flux_bc_mean_flow.jl` | Nonaxisymmetric flux forcing and self-consistent transport | No |
| `example/flux_bc_axisymmetric_flow.jl` | Axisymmetric flux forcing using the noniterated constructor | No |
| `example/triglobal_analysis_demo.jl` | Coupled azimuthal stability | Final solve only |
| `example/mhd_dynamo_example.jl` | Magnetoconvection with an imposed axial field | Yes |
| `example/figure2_benchmark.jl` | Onset benchmark scan against Barik et al. (2023) at ``Ek_d = 10^{-3}`` | Yes |

The historical `mhd_dynamo_example.jl` filename does not imply a kinematic
dynamo solver: the MHD implementation linearizes about a motionless
conductive state and an imposed current-free field.

`Rac_lm.jl` builds its own small dense eigenproblems and uses the same
Rayleigh-number, gravity and heating conventions as `OnsetParams`; without
rotation each spherical-harmonic degree is independent, and the script checks
itself against the plane-layer limit. `boundary_driven_jet.jl` and
`nonaxisymmetric_basic_state.jl` save a figure when Plots.jl is installed and
skip it otherwise.

The scripts' printed commentary uses thermal-wind language as interpretation;
the constructors themselves solve the viscous mean-flow equations described in
[Basic States](basic_states.md). Axisymmetric forcing can generate meridional
circulation, so its thermal advection is not identically zero.

`linear_stability_demo.jl` calls `slepc_init!` itself and can be run directly
with `julia --project=. example/linear_stability_demo.jl` once PetscWrap and
SlepcWrap are installed. Run the other eigenvalue scripts inside an
initialized session:

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

## Construct all four problem types

This setup example runs during the documentation build without PETSc:

```@example problem_types
using Magrathea
params = OnsetParams(E=1e-2, Pr=1.0, Ra=30.0, χ=0.35,
                     m=1, lmax=6, Nr=24)
onset = OnsetProblem(params)
bs = basic_state(params; mode=:meridional, amplitude=0.01)
biglobal = BiglobalProblem(params, bs)
bs3d = basic_state(params; mode=:nonaxisymmetric, amplitude=0.01, mmax_bs=2)
triglobal = TriglobalProblem(params, bs3d, -2:2)

mhd_params = MHDParams(E=1e-2, Pr=1.0, Pm=1.0, Ra=30.0,
    Le=0.01, ricb=0.35, m=1, lmax=6, N=16, B0_type=axial)
mhd = MHDProblem(mhd_params)
@assert bs.flow !== nothing && bs3d.flow !== nothing
estimate_size(triglobal)
```

The two mean states above use conductive temperatures and Stokes–Coriolis
velocities. Use the self-consistent constructor when nonlinear momentum
inertia and thermal transport matter.

After SLEPc initialization, solve one of these problems:

```julia
result = solve(biglobal; nev=6)
println((result.growth_rate, result.frequency))
```

## Nonlinear mean states in 2D and 3D

The same coupled steady solver handles axisymmetric and nonaxisymmetric
boundary forcing. This example is also executed during the docs build:

```@example nonlinear_states
using Magrathea
cd = ChebyshevDiffn(24, [0.35, 1.0], 4)
for forcing in (Y20(0.01), Y20(0.01) + Y22(0.01))
    bs, info = basic_state_selfconsistent(cd, 0.35, 0.01, 30.0, 1.0;
        temperature_bc=forcing, lmax_bs=4,
        max_iterations=50, tolerance=1e-8)
    @assert info.converged
    println((state=typeof(bs), momentum=info.momentum_residual,
             thermal=info.thermal_residual, boundary=info.boundary_residual))
    println(mean_flow_velocity(bs, 0.7, pi/3, pi/8))
end
```

`momentum_model=:navier_stokes` is the default. Set
`momentum_model=:stokes` to omit nonlinear momentum inertia while retaining
thermal transport. A small residual establishes convergence at the selected
truncation; repeat at higher radial, spherical-degree, and azimuthal
resolution.

For an outer radial-derivative boundary condition, use
`flux_bc=Y00(-1.0) + Y20(-0.01)` (axisymmetric), or add `Y22(-0.01)`
(nonaxisymmetric). The coefficients specify the increasing-radius derivative,
not the outward-normal heat flux. The constructor's inner thermal condition
supplies the temperature reference; this is not an arbitrary two-Neumann
Poisson problem.

## MHD assembly and boundaries

```@example mhd_assembly
using Magrathea
for inner_magnetic in (0, 1, 2)  # insulating, finite core, perfect conductor
    params = MHDParams(E=1e-2, Pr=1.0, Pm=1.0, Ra=30.0,
        Le=0.01, ricb=0.35, m=1, lmax=4, N=12, B0_type=axial,
        bci_magnetic=inner_magnetic, bco_magnetic=0)
    op = MHDStabilityOperator(params)
    A, B, interior_dofs, info = assemble_mhd_matrices(op)
    @assert size(A) == size(B)
    println((inner_magnetic=inner_magnetic, pencil=size(A)))
end
```

This low-level function assembles the coefficient-tau pencil. Keep the full
matrices: `interior_dofs` identifies equation rows, not a square subproblem.
The high-level `solve(MHDProblem(params))` selects Galerkin assembly for
insulating axial cases and tau for conducting walls or dipole fields.

The finite inner core is stationary and has the same magnetic diffusivity
and permeability as the shell in the current model. A finite conducting outer
mantle is unsupported. See [MHD User Guide](mhd_user_guide.md) for boundary
flags and reconstruction.

## Validation examples

The thermal-wind and wall checks live under `test/`:

```bash
julia --project=. -e 'include("test/thermal_wind.jl")'
julia --project=. -e 'include("test/boundary_physics.jl")'
julia --project=. -e 'include("test/nonlinear_mean_flow.jl")'
```

There is no `example/test_thermal_wind.jl`; use the test files above.
See [Codebase Structure](codebase_structure.md) for the complete validation map.
