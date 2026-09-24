# ============================================================================
# Unified solve() API for Magrathea.jl v2.0
# ============================================================================

"""
    solve(problem; nev=6, sigma=nothing)

Solve the eigenvalue problem for the given stability problem.
Returns a `StabilityResult` with eigenvalues, eigenvectors, and analysis-specific data.

Automatically warns if the estimated problem memory exceeds 8 GB.

# Example
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=0.35, m=4, lmax=20, Nr=32)
result = solve(OnsetProblem(params); nev=6)
result.growth_rate
```
"""
function solve end

# --- Memory warning helper ---

"""
    _warn_if_large(problem, label)

Run size estimation before a solve and downgrade estimation failures to debug
logs so solving remains possible for partially supported problem types.
"""
function _warn_if_large(problem, label::String)
    try
        _check_memory(problem, label)
    catch err
        @debug "Size estimation failed" label=label exception=(err, catch_backtrace())
    end
end

"""Warn when an onset solve is likely to allocate dense matrices larger than the soft limit."""
function _check_memory(p::OnsetProblem, label)
    total_dof = _hd_total_dof(p.params.m, p.params.lmax, p.params.Nr, p.params.equatorial_symmetry)
    mem = _mem_gb(total_dof)
    if mem > 8.0
        @warn "$label: estimated memory ~$(round(mem; digits=1)) GB exceeds 8 GB — consider reducing lmax or Nr"
    end
    return nothing
end

"""Warn when a biglobal solve is likely to allocate dense matrices larger than the soft limit."""
function _check_memory(p::BiglobalProblem, label)
    total_dof = _hd_total_dof(p.params.m, p.params.lmax, p.params.Nr, p.params.equatorial_symmetry)
    mem = _mem_gb(total_dof)
    if mem > 8.0
        @warn "$label: estimated memory ~$(round(mem; digits=1)) GB exceeds 8 GB — consider reducing lmax or Nr"
    end
    return nothing
end

"""Warn when a triglobal solve is likely to allocate dense matrices larger than the soft limit."""
function _check_memory(p::TriglobalProblem, label)
    total_dof, _ = _triglobal_total_dof(
        p.m_range, p.params.lmax, p.params.Nr, p.params.equatorial_symmetry)
    mem = _mem_gb(total_dof)
    if mem > 8.0
        @warn "$label: estimated memory ~$(round(mem; digits=1)) GB exceeds 8 GB — consider reducing lmax or m_range"
    end
    return nothing
end

"""Warn when an MHD solve is likely to allocate dense matrices larger than the soft limit."""
function _check_memory(p::MHDProblem, label)
    total_dof, _, _, _, _, _ = _mhd_total_dof(p.params)
    mem = _mem_gb(total_dof)
    if mem > 8.0
        @warn "$label: estimated memory ~$(round(mem; digits=1)) GB exceeds 8 GB — consider reducing lmax or N"
    end
    return nothing
end

# ============================================================================
# OnsetProblem solve
# ============================================================================

"""
    solve(problem::OnsetProblem; nev=6, sigma=nothing, tol=1e-10, maxiter=1000, which=:LR)

Solve the onset of convection eigenvalue problem.

Constructs an `OnsetConvectionParams` from the wrapped `OnsetParams`, calls
`solve_onset_problem`, and wraps the result in a `StabilityResult`.

Returns a 4-tuple `(eigenvalues, eigenvectors, operator, info)` are stored in
`result.extra` as `(operator=op, info=info)`.
"""
function solve(problem::OnsetProblem{T};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR,
               backend::Symbol=:slepc) where T

    _warn_if_large(problem, "OnsetProblem")

    onset_params = OnsetConvectionParams(problem.params)

    eigenvalues, eigenvectors, op, info = solve_onset_problem(onset_params;
        nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma, backend=backend)

    # Convert eigenvectors from Vector{Vector{ComplexF64}} to Matrix{Complex{T}}
    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, T)

    return StabilityResult(
        convert(Vector{Complex{T}}, eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, info=info)
    )
end

# ============================================================================
# BiglobalProblem solve
# ============================================================================

"""
    solve(problem::BiglobalProblem; nev=6, sigma=nothing, tol=1e-10, maxiter=1000, which=:LR, verbose=false)

Solve the biglobal stability eigenvalue problem with axisymmetric mean flow.

Constructs `BiglobalParams` from the wrapped `OnsetParams` and `BasicState`,
calls `solve_biglobal_problem`, and wraps the result in a `StabilityResult`.
"""
function solve(problem::BiglobalProblem{T};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR,
               backend::Symbol=:slepc,
               verbose::Bool=false) where T

    _warn_if_large(problem, "BiglobalProblem")

    p = problem.params
    biglobal_params = BiglobalParams(
        E=p.E, Pr=p.Pr, Ra=p.Ra, χ=p.χ,
        m=p.m, lmax=p.lmax, Nr=p.Nr,
        basic_state=problem.basic_state,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry
    )

    eigenvalues, eigenvectors, op, info = solve_biglobal_problem(biglobal_params;
        nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma, backend=backend, verbose=verbose)

    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, T)

    return StabilityResult(
        convert(Vector{Complex{T}}, eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, info=info)
    )
end

# ============================================================================
# TriglobalProblem solve
# ============================================================================

"""
    solve(problem::TriglobalProblem; nev=6, sigma=nothing, tol=1e-8, maxiter=200,
          which=:LR, backend=:slepc, verbose=true)

Solve the triglobal stability eigenvalue problem with non-axisymmetric basic state.

Constructs `TriglobalParams` from the wrapped `OnsetParams`, `BasicState3D`,
and `m_range`, calls `solve_triglobal_eigenvalue_problem`, and wraps the result
in a `StabilityResult`. With `sigma === nothing` the shift is chosen from `which`
like the other problem types.
"""
function solve(problem::TriglobalProblem{T};
               nev::Int=6,
               sigma=nothing,
               tol::Real=1e-8,
               maxiter::Int=200,
               which::Symbol=:LR,
               backend::Symbol=:slepc,
               verbose::Bool=true) where T

    _warn_if_large(problem, "TriglobalProblem")

    p = problem.params
    triglobal_params = TriglobalParams(
        E=p.E, Pr=p.Pr, Ra=p.Ra, χ=p.χ,
        m_range=problem.m_range, lmax=p.lmax, Nr=p.Nr,
        basic_state_3d=problem.basic_state,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry
    )

    eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(triglobal_params;
        σ_target=sigma, nev=nev, verbose=verbose, which=which, tol=tol,
        maxiter=maxiter, backend=backend)

    # eigenvectors is already a Matrix from the triglobal solver
    evec_matrix = Matrix{Complex{T}}(eigenvectors)

    return StabilityResult(
        convert(Vector{Complex{T}}, eigenvalues),
        evec_matrix,
        problem;
        extra=(coupled_modes=collect(problem.m_range),)
    )
end

# ============================================================================
# MHDProblem solve
# ============================================================================

"""
    solve(problem::MHDProblem; nev=6, sigma=nothing, tol=1e-10, maxiter=1000,
          which=:LR, backend=:slepc)

Solve the MHD eigenvalue problem.

Constructs an `MHDStabilityOperator` from the MHD parameters, assembles the
matrices (boundary-recombined Galerkin for insulating walls, tau otherwise),
solves with the selected backend (`:slepc` or `:dense`), and wraps the result in
a `StabilityResult`.

Each returned eigenvector is checked for radial resolution:
`result.extra.spectral_tail[j]` holds the share of mode `j`'s norm in the top
quarter of its Chebyshev coefficients (`radial`) and of its retained degrees
(`angular`). A warning is issued when the leading mode's radial tail exceeds
1e-2: such modes sit at the truncation scale (typically
spurious growth at strong field and low `N`), so increase `N` until the leading
eigenvalue converges. The warning includes a rough `N` for resolving the magnetic
(Hartmann) boundary layers, whose thickness is `√(E·Em)/(Le·B₀)`; `estimate_size`
reports it before solving. A dipole is `ricb⁻³` times stronger at the inner wall
than at the outer wall, so it needs more radial modes than an axial field of equal
`Le`.
"""
function solve(problem::MHDProblem{T, BS};
               nev::Int=6,
               sigma=nothing,
               tol::Float64=1e-10,
               maxiter::Int=1000,
               which::Symbol=:LR,
               backend::Symbol=:slepc) where {T, BS}

    _check_backend(backend)
    _warn_if_large(problem, "MHDProblem")

    mhd_params = problem.params
    op = MHDStabilityOperator(mhd_params)

    if !is_dipole_case(mhd_params.B0_type, mhd_params.ricb) &&
       mhd_params.bci_magnetic == 0 && mhd_params.bco_magnetic == 0
        # Hydro AND axial-field MHD with insulating magnetic BCs: tau-free
        # ultraspherical-Galerkin assembly. (Dipole and conducting/perfect-conductor
        # magnetic BCs are not yet supported by the Galerkin path → tau below.)
        # The recombined basis removes the tau method's spurious eigenvalues, but an
        # under-resolved pencil can still have unphysical growing modes (strong
        # field, low N); the spectral-tail check below flags those. The dipole case
        # still routes through the tau path (see galerkin_assembly.jl).
        A_gal, B_gal, layout = assemble_mhd_galerkin(op)
        if backend === :slepc
            vals_s, vecs_s, _ = Magrathea._solve_generalized_eigen_slepc(
                sparse(A_gal), sparse(B_gal); nev=nev,
                sigma = sigma === nothing ? nothing : ComplexF64(sigma),
                which=which, selection=:maxreal, tol=tol, maxiter=maxiter, verbosity=0)
        else
            vals_s, vecs_s, _ = _dense_generalized_eigen(A_gal, B_gal; nev=nev,
                sigma=sigma, which=which, selection=:maxreal)
        end
        eigenvalues = vals_s
        evecs_full = [reconstruct_mhd_galerkin_full(op, layout, vecs_s[:, j])
                      for j in 1:size(vecs_s, 2)]
        evec_matrix = _eigvecs_to_matrix(eigenvalues, evecs_full, T)
        info = (method = "MHD ultraspherical-Galerkin", n_reduced = layout.nred)
        return StabilityResult(
            convert(Vector{Complex{T}}, eigenvalues),
            evec_matrix,
            problem;
            extra=(operator=op, interior_dofs=collect(1:layout.nred),
                   assembly_info=info, galerkin_layout=layout,
                   spectral_tail=_check_mhd_resolution(op, eigenvalues, evec_matrix))
        )
    end

    if backend === :slepc
        # Distributed MHD tau path: the SLEPc extension assembles (A, B) in
        # distributed PETSc form (owned rows only), applies the tau boundary
        # conditions on the distributed Mats, and runs the EPS solve. Avoids ever
        # materializing the full sparse pencil on a single rank.
        eigenvalues, eigenvectors, info = Magrathea._solve_mhd_slepc(
            op; nev=nev, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
        interior_dofs = Int[]
        info_assembly = info
    else
        A, B, interior_dofs, info_assembly = assemble_mhd_matrices(op)

        eigenvalues, eigenvectors, info = solve_eigenvalue_problem(
            A, B; nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma, backend=backend)
    end

    evec_matrix = _eigvecs_to_matrix(eigenvalues, eigenvectors, T)

    return StabilityResult(
        convert(Vector{Complex{T}}, eigenvalues),
        evec_matrix,
        problem;
        extra=(operator=op, interior_dofs=interior_dofs, assembly_info=info_assembly,
               spectral_tail=_check_mhd_resolution(op, eigenvalues, evec_matrix))
    )
end

"""Radial spectral-tail share above which a leading MHD mode is reported as
under-resolved. Converged modes measure ≲1e-3; spurious truncation-scale growth
measures ~0.4–0.6."""
const _MHD_RADIAL_TAIL_WARN = 1e-2

"""
    _mhd_boundary_layer_N(params) -> Int

Rough radial resolution `N` that resolves the magnetic (Hartmann) boundary layers,
of thickness `√(E·Em)/(Le·B₀)`, where `B₀` is the largest background field at a
wall: 1 for the axial field, and `ricb⁻³` at the inner wall for the dipole. Below it,
Alfvén waves at the truncation scale are under-damped and appear as spurious growing
eigenvalues. The constant was fitted to E = 1e-4 to 1e-3 and Pm = 0.1 to 10, where the
estimate is within about a factor of 2 and errs high for Pm < 1. Returns 0 when no
field is imposed.
"""
function _mhd_boundary_layer_N(p::MHDParams)
    (p.B0_type == no_field || iszero(p.Le)) && return 0
    B0 = is_dipole_case(p.B0_type, p.ricb) ? inv(p.ricb)^3 : one(p.ricb)
    return ceil(Int, 1.8 * sqrt((1 - p.ricb) * p.Le * B0 / sqrt(p.E * p.Em)))
end

"""Compute `_mhd_spectral_tails` for every returned eigenvector (full tau layout)
and warn when the leading mode is radially under-resolved. Workers of a
distributed solve hold no eigenvectors and skip the check."""
function _check_mhd_resolution(op, eigenvalues, evecs::AbstractMatrix)
    size(evecs, 1) == op.matrix_size ||
        return NamedTuple{(:radial, :angular), Tuple{Float64, Float64}}[]
    tails = [_mhd_spectral_tails(op, view(evecs, :, j)) for j in axes(evecs, 2)]
    if !isempty(tails) && tails[1].radial > _MHD_RADIAL_TAIL_WARN
        N = op.params.N; N_layers = _mhd_boundary_layer_N(op.params)
        hint = N_layers > N ?
            " The magnetic boundary layers need roughly N ≳ $N_layers (a rough " *
            "estimate; see `estimate_size`)." : ""
        @warn "MHD leading eigenmode is under-resolved: $(round(100 * tails[1].radial; sigdigits=2))% " *
              "of its norm lies in the top quarter of the Chebyshev coefficients, so its " *
              "eigenvalue $(eigenvalues[1]) is likely a truncation artefact. Increase N " *
              "(currently $N) until the leading eigenvalue converges." * hint
    end
    return tails
end

# ============================================================================
# Utility: convert Vector{Vector} to Matrix
# ============================================================================

"""Normalize solver eigenvectors to a dense `Matrix{Complex{T}}` for `StabilityResult`.

Split into multiple-dispatch methods so the eigenvector container type (matrix for
the Galerkin/triglobal paths, vector-of-vectors for the tau paths) is resolved at
compile time instead of via runtime `isa` branches."""
function _eigvecs_to_matrix(eigenvalues, eigenvectors::AbstractMatrix, ::Type{T}) where T
    return Matrix{Complex{T}}(eigenvectors)
end

function _eigvecs_to_matrix(eigenvalues,
                            eigenvectors::AbstractVector{<:AbstractVector},
                            ::Type{T}) where T
    isempty(eigenvectors) && return Matrix{Complex{T}}(undef, 0, length(eigenvalues))
    n = length(eigenvectors[1])
    nev = length(eigenvectors)
    mat = Matrix{Complex{T}}(undef, n, nev)
    for j in 1:nev
        mat[:, j] = eigenvectors[j]
    end
    return mat
end

# Fallback for empty/degenerate containers that match neither concrete shape.
function _eigvecs_to_matrix(eigenvalues, eigenvectors, ::Type{T}) where T
    return Matrix{Complex{T}}(undef, 0, length(eigenvalues))
end

# ============================================================================
# Unified find_critical_Ra API
# ============================================================================

"""
    find_critical_Ra(problem; Ra_guess=1e6, tol=1e-6, kwargs...)

Find the critical Rayleigh number for the given stability problem.
Dispatches to the appropriate solver based on problem type.

# Example
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e6, χ=0.35, m=4, lmax=20, Nr=32)
Ra_c = find_critical_Ra(OnsetProblem(params); Ra_guess=1e5)
```
"""
function find_critical_Ra end

"""Find critical Rayleigh number for an onset problem using its wrapped parameters."""
function find_critical_Ra(problem::OnsetProblem{T};
                          Ra_guess::T=T(1e6),
                          tol::T=T(1e-6),
                          nev::Int=6,
                          verbose::Bool=false,
                          kwargs...) where T
    p = problem.params
    return find_critical_Ra_onset(;
        E=p.E, Pr=p.Pr, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr,
        Ra_guess=Ra_guess, tol=tol,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry,
        nev=nev, verbose=verbose, kwargs...)
end

"""Find critical Rayleigh number for a biglobal problem with an axisymmetric basic state."""
function find_critical_Ra(problem::BiglobalProblem{T};
                          Ra_guess::T=T(1e6),
                          tol::T=T(1e-6),
                          nev::Int=6,
                          verbose::Bool=false,
                          kwargs...) where T
    p = problem.params
    return find_critical_Ra_biglobal(;
        E=p.E, Pr=p.Pr, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr,
        basic_state=problem.basic_state,
        Ra_guess=Ra_guess, tol=tol,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry,
        nev=nev, verbose=verbose, kwargs...)
end

"""Find critical Rayleigh number for a triglobal problem with a 3D basic state."""
function find_critical_Ra(problem::TriglobalProblem{T};
                          Ra_min::Real=1e5,
                          Ra_max::Real=1e8,
                          tol::Real=1e-4,
                          max_iter::Int=20,
                          verbose::Bool=true,
                          kwargs...) where T
    p = problem.params
    return find_critical_rayleigh_triglobal(
        p.E, p.Pr, p.χ, problem.m_range, p.lmax, p.Nr,
        problem.basic_state;
        Ra_min=Ra_min, Ra_max=Ra_max, tol=tol, max_iter=max_iter,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry,
        verbose=verbose, kwargs...)
end

"""Report that MHD critical-parameter searches are not exposed as a one-parameter API."""
function find_critical_Ra(::MHDProblem; kwargs...)
    error("""find_critical_Ra is not supported for MHDProblem.

MHD dynamo problems involve coupled velocity-magnetic field instabilities where
the critical parameter depends on multiple numbers (Ra, Pm, Le) simultaneously.
Use solve(MHDProblem(...); nev=...) directly and inspect the growth rate.""")
end

# --- Perturbation-field reconstruction from StabilityResult ---
# These methods require StabilityResult, which is defined in types.jl (before solve.jl).

"""
    perturbation_velocity(result::StabilityResult, mode::Int; kwargs...)

Reconstruct velocity for the `mode`-th eigenvector of an Onset/Biglobal result.
Requires `result.extra.operator` (present for `OnsetProblem`/`BiglobalProblem`).
"""
function perturbation_velocity(result::StabilityResult, mode::Int; kwargs...)
    hasproperty(result.extra, :operator) || error(
        "perturbation_velocity(result, mode): result.extra has no :operator " *
        "(only OnsetProblem/BiglobalProblem results carry it). Pass (evec, op) instead.")
    op = result.extra.operator
    evec = result.eigenvectors[:, mode]
    return perturbation_velocity(evec, op; kwargs...)
end

"""
    perturbation_temperature(result::StabilityResult, mode::Int; kwargs...)

Reconstruct temperature for the `mode`-th eigenvector of an Onset/Biglobal result.
"""
function perturbation_temperature(result::StabilityResult, mode::Int; kwargs...)
    hasproperty(result.extra, :operator) || error(
        "perturbation_temperature(result, mode): result.extra has no :operator. " *
        "Pass (evec, op) instead.")
    return perturbation_temperature(result.eigenvectors[:, mode],
                                    result.extra.operator; kwargs...)
end
