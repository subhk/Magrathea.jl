using Test
using Logging

include("project_deps.jl")
include("legacy_api.jl")

# Existing tests
include("boundary_conditions.jl")
include("mhd_boundary_conditions.jl")
include("mhd_stress_free_bc.jl")
include("meridional_boundary_conditions.jl")
include("chebyshev.jl")
include("sparse_operator.jl")
include("galerkin_radial.jl")
include("sh_transform.jl")
include("thermal_wind.jl")
include("velocity_reconstruction.jl")
include("audit_fixes.jl")
include("review_regressions.jl")

# v2.0 API tests
include("test_types.jl")
include("type_stability.jl")
include("test_validation.jl")
include("test_show.jl")
include("slepc_backend.jl")
include("dof_ownership.jl")
include("distributed_assembly.jl")
include("distributed_reduction.jl")
include("distributed_onset.jl")
include("distributed_triglobal.jl")

# Coverage tests (construction/builder paths; no eigensolve)
include("test_mhd_operator_coverage.jl")
include("test_onset_solve_helpers.jl")
include("test_bc_bsops_coverage.jl")
include("test_basicstate_coverage.jl")
include("test_solve_assembly_coverage.jl")
include("test_advection_coverage.jl")
include("test_types_state_coverage.jl")
include("test_solve_paths_coverage.jl")

include("perturbation_fields.jl")
