# =============================================================================
#  Eigenvalue Solver for Onset of Convection
#
#  Solves the generalized eigenvalue problem:
#      A·x = σ·B·x
#
#  where σ = σ_r + iω with:
#  - σ_r: growth rate (σ_r = 0 at onset)
#  - ω: drift frequency
#
#  References:
#  - Barik et al. (2023), Earth and Space Science
#  - Dormy et al. (2004), Journal of Fluid Mechanics
# =============================================================================

using Logging

# ---------------------------------------------------------------------------
# Pluggable eigensolver backends. The SLEPc backend lives in an optional package
# extension (MagratheaSlepcExt) and registers itself here on load; core never
# references PETSc/SLEPc symbols, keeping `using Magrathea` PETSc-free.
# ---------------------------------------------------------------------------
const _SLEPC_SOLVER = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_MHD_SOLVER = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_CONSTRAINED_SOLVER = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_TRIGLOBAL_SOLVER = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_INIT     = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_FINALIZE = Ref{Union{Nothing,Function}}(nothing)

"""Distributed triglobal SLEPc solve directly from a `CoupledModeProblem`. The
extension assembles the block-coupled pencil `(A, B)` into distributed PETSc form
(owned rows only, from `_assemble_block_coo`) and runs the EPS shift-invert solve.
PETSc-free in core; errors if the extension is absent. Returns `(eigenvalues,
eigenvectors)` (eigenvectors gathered full on rank 0, empty on workers)."""
function _solve_triglobal_slepc(problem; kwargs...)
    f = _SLEPC_TRIGLOBAL_SOLVER[]
    f === nothing && error("backend=:slepc (distributed triglobal) requires `using PetscWrap, SlepcWrap` and Magrathea.slepc_init!().")
    return f(problem; kwargs...)
end

"""Distributed constrained-reduction SLEPc solve directly from a
`LinearStabilityOperator`. The extension assembles the full tau pencil `(A, B)` and
the sparse constraint-projection matrices `S` (`n_reduced×n_full`) and `P`
(`n_full×n_reduced`) into distributed PETSc form, forms the reduced pencil
`Ared = S·A·P`, `Bred = S·B·P` via `MatMatMult`, and runs the EPS shift-invert solve.
PETSc-free in core; errors if the extension is absent. Returns the common
`(eigenvalues, eigenvectors, info)` contract (eigenvectors gathered/reconstructed full
on rank 0, empty on workers)."""
function _solve_constrained_slepc(op; kwargs...)
    f = _SLEPC_CONSTRAINED_SOLVER[]
    f === nothing && error("backend=:slepc (distributed constrained reduction) requires `using PetscWrap, SlepcWrap` and Magrathea.slepc_init!().")
    return f(op; kwargs...)
end

"""Initialize SLEPc once per process (collective). Pass a PETSc/SLEPc option string.
Requires the MagratheaSlepcExt extension (`using PetscWrap, SlepcWrap`)."""
function slepc_init!(opts::AbstractString="")
    f = _SLEPC_INIT[]
    f === nothing && error(
        "slepc_init! requires the SLEPc extension: `using PetscWrap, SlepcWrap` " *
        "(complex-scalar PETSc build with PETSC_DIR/SLEPC_DIR set).")
    return f(opts)
end

"""Finalize SLEPc once at process end. Requires the MagratheaSlepcExt extension."""
function slepc_finalize!()
    f = _SLEPC_FINALIZE[]
    f === nothing && error(
        "slepc_finalize! requires the SLEPc extension: `using PetscWrap, SlepcWrap`.")
    return f()
end

"""
    _petsc_owned_nnz(M, rstart, rend) -> (d_nnz, o_nnz)

Per-row diagonal/off-diagonal nonzero counts for the owned PETSc row block
`[rstart, rend)` (0-based, half-open) of a replicated `SparseMatrixCSC`, for
`MatMPIAIJSetPreallocation`. Assumes the column-ownership band equals the row band
`[rstart, rend)` (PETSc default for square MPIAIJ with matching layout). Returns two
`Vector{Int}` of length `rend - rstart`.
"""
function _petsc_owned_nnz(M::SparseMatrixCSC, rstart::Int, rend::Int)
    nloc = rend - rstart
    d = zeros(Int, nloc)
    o = zeros(Int, nloc)
    rows = rowvals(M)
    for col in 1:size(M, 2)
        c0 = col - 1
        for k in nzrange(M, col)
            r0 = rows[k] - 1
            if rstart <= r0 < rend
                i = r0 - rstart + 1
                if rstart <= c0 < rend
                    d[i] += 1
                else
                    o[i] += 1
                end
            end
        end
    end
    return d, o
end

"""Distributed MHD-aware SLEPc solve directly from an `MHDStabilityOperator`. The
extension assembles the (A, B) tau matrices in distributed PETSc form (owned rows
only), applies the boundary conditions on the distributed Mats, and runs the EPS
shift-invert solve. PETSc-free in core; errors if the extension is absent. Returns
the common `(eigenvalues, eigenvectors, info)` contract."""
function _solve_mhd_slepc(op; kwargs...)
    f = _SLEPC_MHD_SOLVER[]
    f === nothing && error(
        "backend=:slepc (distributed MHD) requires `using PetscWrap, SlepcWrap` " *
        "and Magrathea.slepc_init!().")
    return f(op; kwargs...)
end

"""Solve `A x = σ B x` with SLEPc. Requires the MagratheaSlepcExt extension (load
`PetscWrap` and `SlepcWrap`, with a complex-scalar PETSc build)."""
function _solve_generalized_eigen_slepc(A::SparseMatrixCSC, B::SparseMatrixCSC; kwargs...)
    solver = _SLEPC_SOLVER[]
    solver === nothing && error(
        "backend=:slepc requires the SLEPc extension. Load it with " *
        "`using PetscWrap, SlepcWrap` (needs a complex-scalar PETSc build with " *
        "PETSC_DIR/PETSC_ARCH and SLEPC_DIR set)." *
        " Also call Magrathea.slepc_init!() once before solving.")
    return solver(A, B; kwargs...)
end

"""Dispatch a sparse generalized eigensolve to the selected backend, returning the
common `(eigenvalues::Vector{Complex}, eigenvectors::Matrix{Complex}, info::Dict)`.
SLEPc is the sole supported backend."""
function _dispatch_eigen(A::SparseMatrixCSC, B::SparseMatrixCSC;
                         backend::Symbol=:slepc,
                         nev::Int, sigma, which::Symbol, selection::Symbol,
                         tol::Float64, maxiter::Int,
                         krylovdim::Union{Nothing,Int}, verbosity::Int)
    if backend === :slepc
        return _solve_generalized_eigen_slepc(A, B; nev=nev, sigma=sigma, which=which,
                                              selection=selection, tol=tol, maxiter=maxiter,
                                              verbosity=verbosity)
    else
        throw(ArgumentError("Unknown eigensolver backend $(backend); only :slepc is supported"))
    end
end

"""
    solve_eigenvalue_problem(A, B; nev=20, sigma=nothing, which=:LR,
                             selection=:maxreal, tol=1e-10)

Solve the generalized eigenvalue problem A·x = σ·B·x for sparse matrices using the
SLEPc backend with shift-invert method.

# Arguments
- `A::SparseMatrixCSC`: Operator matrix (physics terms)
- `B::SparseMatrixCSC`: Mass matrix (time derivative weights)
- `nev::Int=20`: Number of eigenvalues to compute
- `sigma::Union{Nothing,Number}=nothing`: Shift target for shift-invert modes
- `which::Symbol=:LR`: Determines automatic shift selection (see below)
- `selection::Symbol=:maxreal`: How to order the returned eigenvalues:
  - `:maxreal`: sort by descending real part (default, best for onset)
  - `:minabs`: sort by ascending magnitude
  - `:closest_real`: sort by ascending |Re(σ)| (best for critical Ra search)
- `tol::Float64=1e-10`: Convergence tolerance
- `maxiter::Int=1000`: Maximum number of iterations
- `krylovdim::Union{Nothing,Int}=nothing`: Krylov subspace dimension
- `verbosity::Int=0`: Verbosity level for the eigensolver

# Returns
- `eigenvalues::Vector{Complex}`: Computed eigenvalues σ = σ_r + iω
- `eigenvectors::Matrix{Complex}`: Corresponding eigenvectors
- `info::Dict`: Information about the solve

# Notes
**SHIFT-INVERT STRATEGY:**
- Uses shift-invert method: solves (A - σ*B)^(-1)*B*x = μ*x where μ = 1/(λ - σ)
- Always uses `:LM` (Largest Magnitude) for the transformed problem
- The `which` parameter determines SHIFT SELECTION:
  - `:LR` → shift σ=10.0 (targets eigenvalues with large positive real part)
  - `:LI` → shift σ=10.0i (targets eigenvalues with large imaginary part)
  - other → shift σ=1.0 (general purpose)
- Results are sorted by `selection` criterion AFTER transformation
- For onset problems: use `which=:LR, selection=:maxreal` (default)
- For critical Ra: use `sigma=0.0, selection=:closest_real`
"""
function solve_eigenvalue_problem(A::SparseMatrixCSC, B::SparseMatrixCSC;
                                 nev::Int=1,
                                 backend::Symbol=:slepc,
                                 sigma::Union{Nothing,Number}=nothing,
                                 which::Symbol=:LR,  # Determines shift selection (not eigsolve target)
                                 selection::Symbol=:maxreal,
                                 tol::Float64=1e-10,
                                 maxiter::Int=1000,
                                 krylovdim::Union{Nothing,Int}=nothing,
                                 verbosity::Int=0)

    n = size(A, 1)
    size(A) == size(B) || throw(DimensionMismatch(
        "A and B must have same dimensions, got $(size(A)) and $(size(B))"))
    size(A, 1) == size(A, 2) || throw(DimensionMismatch(
        "Matrices must be square, got $(size(A))"))

    @info "Solving eigenvalue problem" solver=backend size="$n × $n" A_nnz=nnz(A) B_nnz=nnz(B) nev=nev which=which selection=selection

    eigenvalues, eigenvectors, info = _dispatch_eigen(A, B;
                                                      backend = backend,
                                                      nev = nev,
                                                      sigma = sigma,
                                                      which = which,
                                                      selection = selection,
                                                      tol = tol,
                                                      maxiter = maxiter,
                                                      krylovdim = krylovdim,
                                                      verbosity = verbosity)

    @info "Eigensolve converged" selected=eigenvalues[1] growth_rate=real(eigenvalues[1]) frequency=imag(eigenvalues[1]) selection=selection
    return eigenvalues, eigenvectors, info
end


"""
    find_critical_rayleigh(operator_builder, E, χ, m;
                          Ra_min=1e4, Ra_max=1e10,
                          tol=1e-6, growth_tol=1e-6, max_iter=50)

Find the critical Rayleigh number Ra_c where the growth rate σ_r = 0.

Uses a safeguarded Brent root finder (inverse quadratic interpolation with
bisection fallback) to mirror the strategy used in Kore.

# Arguments
- `operator_builder::Function`: Function that takes Ra and returns (A, B)
- `E::Float64`: Ekman number
- `χ::Float64`: Radius ratio
- `m::Int`: Azimuthal wavenumber
- `Ra_min::Float64`: Lower bracket for Ra
- `Ra_max::Float64`: Upper bracket for Ra
- `tol::Float64`: Relative tolerance on Ra (controls absolute tolerance internally)
- `growth_tol::Float64`: Absolute tolerance on the residual growth rate
- `max_iter::Int`: Maximum number of iterations

# Returns
- `Ra_c::Float64`: Critical Rayleigh number
- `ω_c::Float64`: Drift frequency at onset
- `σ_c::ComplexF64`: Full eigenvalue at onset
- `iterations::Int`: Number of iterations required
"""
function find_critical_rayleigh(operator_builder::Function, E::TE, χ::Tχ, m::Int;
                               Ra_min=1e4, Ra_max=1e10,
                               tol=1e-6, growth_tol=1e-6,
                               max_iter::Int=50, nev::Int=1) where {TE<:Real, Tχ<:Real}

    T = promote_type(TE, Tχ)
    E = T(E)
    χ = T(χ)
    Ra_min = T(Ra_min)
    Ra_max = T(Ra_max)
    tol = T(tol)
    growth_tol = T(growth_tol)

    @info "Finding critical Rayleigh number" E=E χ=χ m=m bracket="[$Ra_min, $Ra_max]" tol=tol growth_tol=growth_tol

    function _eval_growth_rate(Ra)
        @debug "Testing Ra" Ra=Ra
        A, B = operator_builder(Ra)

        solver_nev = max(nev, 10)
        eigenvalues, _, info = solve_eigenvalue_problem(
            A, B;
            nev = solver_nev,
            sigma = zero(T),
            which = :LR,
            selection = :closest_real
        )

        σ = Complex{T}(eigenvalues[1])
        σ_r = T(real(σ))

        @debug "Growth rate evaluated" Ra=Ra σ_r=σ_r
        return σ_r, σ
    end

    # Cache evaluations to avoid redundant solves when scanning
    known_values = Dict{T, Tuple{T, Complex{T}}}()

    function growth_rate_cached(Ra)
        Ra_key = T(Ra)
        if haskey(known_values, Ra_key)
            return known_values[Ra_key]
        end
        σ_r, σ = _eval_growth_rate(Ra_key)
        known_values[Ra_key] = (σ_r, σ)
        return σ_r, σ
    end

    # Initial bracket check
    @debug "Checking initial bracket..."
    σ_r_min, σ_min = growth_rate_cached(Ra_min)
    if abs(σ_r_min) < growth_tol
        @info "Lower bracket already satisfies growth tolerance" Ra=Ra_min
        return Ra_min, imag(σ_min), σ_min, 0
    end

    σ_r_max, σ_max = growth_rate_cached(Ra_max)
    if abs(σ_r_max) < growth_tol
        @info "Upper bracket already satisfies growth tolerance" Ra=Ra_max
        return Ra_max, imag(σ_max), σ_max, 0
    end

    if σ_r_min * σ_r_max > 0
        @warn "Initial bracket may not contain critical Ra" σ_r_min σ_r_max
        max_bracket_expansions = 12
        expansion_iter = 0
        while σ_r_min * σ_r_max > 0 && expansion_iter < max_bracket_expansions
            expansion_iter += 1
            if σ_r_min > 0 && σ_r_max > 0
                Ra_min /= T(2)
                @debug "Bracket expansion: lowering Ra_min" iter=expansion_iter Ra_min=Ra_min
                if Ra_min <= 0
                    error("Lower Rayleigh bound reached non-positive value while trying to bracket root.")
                end
                σ_r_min, σ_min = growth_rate_cached(Ra_min)
            elseif σ_r_min < 0 && σ_r_max < 0
                Ra_max *= T(2)
                @debug "Bracket expansion: raising Ra_max" iter=expansion_iter Ra_max=Ra_max
                σ_r_max, σ_max = growth_rate_cached(Ra_max)
            else
                break
            end
        end
        if σ_r_min * σ_r_max > 0
            @debug "Expansion exhausted, performing logarithmic scan..."
            min_scan = max(Ra_min, T(10)) / T(10)
            max_scan = Ra_max * T(10)
            if min_scan <= 0
                min_scan = tol
            end
            scan_points = 30
            scan_values = exp10.(LinRange(log10(min_scan), log10(max_scan), scan_points))
            scan_values = sort(unique(vcat(Ra_min, Ra_max, scan_values)))

            bracket_found = false
            last_ra = nothing
            lastσ_r = zero(T)
            lastσ = zero(Complex{T})

            for Ra in scan_values
                σ_r, σ = growth_rate_cached(Ra)
                if last_ra !== nothing && σ_r * lastσ_r <= 0
                    @debug "Found sign change during scan" Ra_low=last_ra Ra_high=Ra
                    Ra_min, Ra_max = last_ra, Ra
                    σ_r_min, σ_min = lastσ_r, lastσ
                    σ_r_max, σ_max = σ_r, σ
                    bracket_found = true
                    break
                end
                last_ra = Ra
                lastσ_r = σ_r
                lastσ = σ
            end

            if !bracket_found
                error("Unable to bracket the critical Rayleigh number: growth rate has same sign across scanned range.")
            end
        end
    end

    # Safeguarded Brent search (closely follows implementations in literature)
    @debug "Starting Brent search..."
    Ra_a, Ra_b, Ra_c = Ra_min, Ra_max, Ra_min
    σ_r_a, σ_r_b, σ_r_c = σ_r_min, σ_r_max, σ_r_min
    σ_a, σ_b, σ_c = σ_min, σ_max, σ_min

    if σ_r_a * σ_r_b >= 0
        error("Brent search requires opposite signs at the bracket endpoints.")
    end

    abs_tol = tol * max(abs(Ra_a), abs(Ra_b), one(T))
    d = Ra_b - Ra_a
    e = d

    for iter in 1:max_iter
        if (σ_r_b > 0 && σ_r_c > 0) || (σ_r_b < 0 && σ_r_c < 0)
            Ra_c = Ra_a
            σ_r_c = σ_r_a
            σ_c = σ_a
            d = Ra_b - Ra_a
            e = d
        end

        if abs(σ_r_c) < abs(σ_r_b)
            Ra_a, Ra_b, Ra_c = Ra_b, Ra_c, Ra_b
            σ_r_a, σ_r_b, σ_r_c = σ_r_b, σ_r_c, σ_r_b
            σ_a, σ_b, σ_c = σ_b, σ_c, σ_b
        end

        tol_act = T(2) * eps(abs(Ra_b)) + abs_tol
        half_width = T(0.5) * (Ra_c - Ra_b)

        @debug "Brent iteration" iter=iter bracket="[$(Ra_a), $(Ra_c)]" Ra=Ra_b σ_r_a=σ_r_a σ_r_b=σ_r_b σ_r_c=σ_r_c

        if abs(σ_r_b) < growth_tol
            Ra_c_final = Ra_b
            ω_c = imag(σ_b)
            @info "Critical Ra converged" Ra_c=Ra_c_final ω_c=ω_c σ_r=σ_r_b iterations=iter
            return Ra_c_final, ω_c, σ_b, iter
        elseif abs(half_width) <= tol_act
            @debug "Bracket tolerance met but growth rate exceeds growth_tol" abs_σ_r=abs(σ_r_b) growth_tol=growth_tol
        end

        if abs(e) < tol_act || abs(σ_r_a) <= abs(σ_r_b)
            d = half_width
            e = half_width
            @debug "Using bisection step"
        else
            s = σ_r_b / σ_r_a
            if Ra_a == Ra_c
                # Secant method
                p = T(2) * half_width * s
                q = one(T) - s
            else
                q = σ_r_a / σ_r_c
                r = σ_r_b / σ_r_c
                p = s * (T(2) * half_width * q * (q - r) - (Ra_b - Ra_a) * (r - one(T)))
                q = (q - one(T)) * (r - one(T)) * (s - one(T))
            end

            if p > zero(T)
                q = -q
            else
                p = -p
            end

            if (T(2) * p < T(3) * half_width * q - abs(tol_act * q)) &&
                    (p < abs(T(0.5) * e * q))
                e = d
                d = p / q
                @debug "Using inverse interpolation step"
            else
                d = half_width
                e = half_width
                @debug "Interpolation rejected, falling back to bisection"
            end
        end

        Ra_a = Ra_b
        σ_r_a = σ_r_b
        σ_a = σ_b

        if abs(d) > tol_act
            Ra_b += d
        else
            Ra_b += half_width >= 0 ? tol_act : -tol_act
        end

        σ_r_b, σ_b = growth_rate_cached(Ra_b)

        if abs(σ_r_b) < growth_tol
            ω_c = imag(σ_b)
            @info "Critical Ra converged" Ra_c=Ra_b ω_c=ω_c σ_r=σ_r_b iterations=iter
            return Ra_b, ω_c, σ_b, iter
        end
    end

    @warn "Maximum iterations reached without convergence (Brent)"
    ω_c = imag(σ_b)
    return Ra_b, ω_c, σ_b, max_iter
end

"""
    find_onset_parameters(params_template, m_range; kwargs...)

Find onset parameters (Ra_c, m_c, ω_c) by scanning over azimuthal wavenumbers.

# Arguments
- `params_template::SparseOnsetParams`: Template parameters (E, χ, Pr, etc.)
- `m_range::AbstractVector{Int}`: Range of m values to test
- `kwargs...`: Passed to find_critical_rayleigh

# Returns
- `Ra_c::Float64`: Critical Rayleigh number
- `m_c::Int`: Critical azimuthal wavenumber
- `ω_c::Float64`: Critical drift frequency
- `results::Dict`: Full results for all m values tested
"""
function find_onset_parameters(operator_builder_factory::Function,
                               E::TE, χ::Tχ, Pr::TP,
                               m_range::AbstractVector{Int};
                               kwargs...) where {TE<:Real, Tχ<:Real, TP<:Real}

    T = promote_type(TE, Tχ, TP)
    E = T(E)
    χ = T(χ)
    Pr = T(Pr)

    @info "Scanning for onset parameters" E=E χ=χ Pr=Pr m_range=m_range

    SuccessResult = NamedTuple{(:Ra_c, :ω_c, :σ_c, :iters),
        Tuple{T, T, Complex{T}, Int}}
    ErrorResult = NamedTuple{(:error,), Tuple{Exception}}
    Result = Union{SuccessResult, ErrorResult}
    results = Dict{Int, Result}()
    Ra_c_min = T(Inf)
    m_c = 0
    ω_c_best = zero(T)

    for m in m_range
        @info "Testing mode" m=m

        try
            operator_builder = operator_builder_factory(E, χ, Pr, m)

            Ra_c, ω_c, σ_c, iters = find_critical_rayleigh(
                operator_builder, E, χ, m; kwargs...
            )

            results[m] = SuccessResult((T(Ra_c), T(ω_c), Complex{T}(σ_c), iters))

            if Ra_c < Ra_c_min
                Ra_c_min = T(Ra_c)
                m_c = m
                ω_c_best = T(ω_c)
            end

            @info "Mode result" m=m Ra_c=Ra_c ω_c=ω_c

        catch err
            @warn "Failed for mode" m=m exception=err
            results[m] = ErrorResult((err,))
        end
    end

    @info "Onset parameters found" m_c=m_c Ra_c=Ra_c_min ω_c=ω_c_best

    return Ra_c_min, m_c, ω_c_best, results
end
