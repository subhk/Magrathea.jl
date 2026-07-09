# =============================================================================
#  Linear Stability Operator (chebyshev + SLEPc)
#
#  This implementation uses sparse spectral methods
#  Fortran/C code: the matrices are assembled explicitly from Chebyshev
#  differentiation operators with the same r-weighted combinations and tau
#  boundary conditions (Dirichlet / stress-free).
# =============================================================================

# Dependencies provided by Magrathea module:
# LinearAlgebra, LinearMaps, Parameters, Random
# ChebyshevDiffn is available in the Magrathea namespace

const _fourπ = 4π

"""Convert the public equatorial-symmetry symbol to the parity flag used internally."""
@inline function _symmetry_flag(sym::Symbol)
    sym === :symmetric && return 1
    sym === :antisymmetric && return -1
    sym === :both && return nothing
    error("Invalid equatorial symmetry flag $sym")
end

"""Return the four tau rows used for the poloidal boundary constraints."""
@inline function poloidal_tau_indices(idx::UnitRange{Int})
    length(idx) ≥ 4 || throw(ArgumentError("Need at least 4 radial points to impose boundary conditions."))
    ri = first(idx)
    ro = last(idx)
    return ri, ri + 1, ro - 1, ro
end

"""Return inner and outer tau rows for toroidal boundary constraints."""
@inline function toroidal_boundary_indices(idx::UnitRange{Int})
    return first(idx), last(idx)
end

"""Return inner and outer tau rows for temperature boundary constraints."""
@inline function temperature_boundary_indices(idx::UnitRange{Int})
    return first(idx), last(idx)
end

# -----------------------------------------------------------------------------
#  Parameter Structure
# -----------------------------------------------------------------------------

"""
    OnsetParams{T}(; E, Pr, Ra, χ, m, lmax, Nr, kwargs...)

Parameters for rotating spherical shell convection.

# Fields
- `E::T` — Ekman number (viscous / Coriolis)
- `Pr::T` — Prandtl number, default 1.0
- `Ra::T` — Rayleigh number
- `χ::T` — radius ratio rᵢ/rₒ
- `m::Int` — azimuthal wavenumber
- `lmax::Int` — maximum spherical harmonic degree
- `Nr::Int` — radial collocation points
- `mechanical_bc::Symbol` — `:no_slip` (default) or `:stress_free`
- `thermal_bc::Symbol` — `:fixed_temperature` (default) or `:fixed_flux`
- `equatorial_symmetry::Symbol` — `:both` (default), `:symmetric`, or `:antisymmetric`

# Example
```julia
params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64)
```

See also [`OnsetProblem`](@ref) for the v2.0 problem wrapper that accepts this type.
"""
@with_kw_noshow struct OnsetParams{T<:Real, BS}
    E::T
    Pr::T = one(E)
    Ra::T
    χ::T
    m::Int
    lmax::Int
    Nr::Int
    ri::T = χ
    ro::T = one(E)
    L::T = ro - ri
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    use_sparse_weighting::Bool = true
    equatorial_symmetry::Symbol = :both
    basic_state::BS = nothing

    function OnsetParams{T,BS}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
                           mechanical_bc, thermal_bc, use_sparse_weighting,
                           equatorial_symmetry,
                           basic_state::BS) where {T,BS}
        0 < χ < 1 || throw(ArgumentError(
            "Radius ratio χ must be in (0,1), got $χ"))
        E > 0 || throw(ArgumentError(
            "Ekman number E must be positive, got $E"))
        Pr > 0 || throw(ArgumentError(
            "Prandtl number Pr must be positive, got $Pr"))
        m ≥ 0 || throw(ArgumentError(
            "Azimuthal wavenumber m must be non-negative, got $m"))
        lmax ≥ m || throw(ArgumentError(
            "lmax must be >= m, got lmax=$lmax, m=$m"))
        Nr ≥ 8 || throw(ArgumentError(
            "Nr must be >= 8 for meaningful resolution, got $Nr"))
        mechanical_bc in (:no_slip, :stress_free) || throw(ArgumentError(
            "mechanical_bc must be :no_slip or :stress_free, got :$mechanical_bc"))
        thermal_bc in (:fixed_temperature, :fixed_flux) || throw(ArgumentError(
            "thermal_bc must be :fixed_temperature or :fixed_flux, got :$thermal_bc"))
        equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
            "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$equatorial_symmetry"))

        new{T,BS}(E, Pr, Ra, χ, m, lmax, Nr, ri, ro, L,
               mechanical_bc, thermal_bc, use_sparse_weighting, equatorial_symmetry,
               basic_state)
    end
end

# -----------------------------------------------------------------------------
#  Linear Stability Operator
# -----------------------------------------------------------------------------

"""
    compute_l_sets(p::OnsetParams)

Return the retained spherical-harmonic degrees for poloidal, toroidal, and
temperature fields after applying equatorial-symmetry truncation.
"""
function compute_l_sets(p::OnsetParams{T}) where {T<:Real}
    if p.equatorial_symmetry === :both
        if p.m == 0
            ls = collect(1:(p.lmax + 1))
        else
            ls = collect(p.m:p.lmax)
        end
        return Dict(:P => ls, :T => ls, :Θ => ls)
    end

    vsymm = _symmetry_flag(p.equatorial_symmetry)
    @assert vsymm !== nothing

    signm = p.m == 0 ? 0 : 1
    lm1 = p.lmax - p.m + 1
    ll_start = p.m + 1 - signm
    ll = collect(ll_start:(ll_start + lm1 - 1))

    s = Int((vsymm + 1) ÷ 2)
    pol_start = (signm + s) % 2
    tor_start = (signm + s + 1) % 2

    pol_idxs = pol_start:2:(lm1 - 1)
    tor_idxs = tor_start:2:(lm1 - 1)

    pol_ls = [ll[k + 1] for k in pol_idxs]
    tor_ls = [ll[k + 1] for k in tor_idxs]

    return Dict(:P => pol_ls, :T => tor_ls, :Θ => pol_ls)
end

"""Dense hydrodynamic stability operator and its field-to-index layout."""
struct LinearStabilityOperator{T<:Real, BS}
    params::OnsetParams{T, BS}
    cd::ChebyshevDiffn{T}
    r::Vector{T}
    index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}
    l_sets::Dict{Symbol, Vector{Int}}
    total_dof::Int
    radial_cache::Dict{Tuple{Int,Int}, Matrix{T}}
end

"""Build a Chebyshev radial grid, field index map, and derivative cache."""
function LinearStabilityOperator(params::OnsetParams{T, BS}) where {T, BS}
    cd = ChebyshevDiffn(params.Nr, [params.ri, params.ro], 4)
    r = cd.x

    l_sets = compute_l_sets(params)
    index_map = Dict{Tuple{Int,Symbol}, UnitRange{Int}}()
    idx = 1
    for ℓ in l_sets[:P]
        index_map[(ℓ, :P)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end
    for ℓ in l_sets[:T]
        index_map[(ℓ, :T)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end
    for ℓ in l_sets[:Θ]
        index_map[(ℓ, :Θ)] = idx:(idx + params.Nr - 1);   idx += params.Nr
    end

    total_dof = idx - 1
    return LinearStabilityOperator{T, BS}(params, cd, r, index_map, l_sets, total_dof,
                                          Dict{Tuple{Int,Int}, Matrix{T}}())
end

# -----------------------------------------------------------------------------
#  Radial helper utilities
# -----------------------------------------------------------------------------

"""
    radial_matrix(op, power, order)

Return and cache the dense radial matrix `r^power * d^order/dr^order` on the
operator grid.
"""
function radial_matrix(op::LinearStabilityOperator{T}, power::Int, order::Int) where {T}
    cache = op.radial_cache
    key = (power, order)
    if haskey(cache, key)
        return cache[key]
    end

    mat = if order == 0
        _radial_diagonal_matrix(op.r, power)
    elseif order == 1
        _scaled_radial_rows(op.r, op.cd.D1, power)
    elseif order == 2
        _scaled_radial_rows(op.r, op.cd.D2, power)
    elseif order == 3
        @assert op.cd.D3 !== nothing "Third-order derivative matrix required; increase Chebyshev order."
        _scaled_radial_rows(op.r, op.cd.D3, power)
    elseif order == 4
        @assert op.cd.D4 !== nothing "Fourth-order derivative matrix required; increase Chebyshev order."
        _scaled_radial_rows(op.r, op.cd.D4, power)
    else
        throw(ArgumentError("Unsupported derivative order $order"))
    end

    cache[key] = mat
    return mat
end

@inline function _integer_power(x::T, power::Int) where {T}
    return x ^ power
end

function _radial_diagonal_matrix(r::Vector{T}, power::Int) where {T}
    n = length(r)
    mat = zeros(T, n, n)
    @inbounds for i in 1:n
        mat[i, i] = _integer_power(r[i], power)
    end
    return mat
end

function _scaled_radial_rows(r::Vector{T}, D::Matrix{T}, power::Int) where {T}
    mat = Matrix{T}(undef, size(D, 1), size(D, 2))
    # Fill the cached matrix directly; `Diagonal(r.^power) * D` doubles memory
    # traffic during operator setup.
    @inbounds for j in axes(D, 2)
        for i in axes(D, 1)
            mat[i, j] = _integer_power(r[i], power) * D[i, j]
        end
    end
    return mat
end

"""
    impose_boundary_conditions!(A, B, op)

Replace tau rows in the dense generalized eigenproblem with the selected
mechanical and thermal boundary constraints.
"""
function impose_boundary_conditions!(A::Matrix{Complex{T}}, B::Matrix{Complex{T}},
                                     op::LinearStabilityOperator{T}) where {T<:Real}
    p = op.params
    D1 = op.cd.D1
    D2 = op.cd.D2

    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)

        # Dirichlet: P = 0
        A[ri, :] .= 0;   B[ri, :] .= 0;   A[ri, ri] = 1
        A[ro, :] .= 0;   B[ro, :] .= 0;   A[ro, ro] = 1

        if p.mechanical_bc == :no_slip
            A[inner_tau, :] .= 0; B[inner_tau, :] .= 0; A[inner_tau, P_idx] .= D1[1, :]
            A[outer_tau, :] .= 0; B[outer_tau, :] .= 0; A[outer_tau, P_idx] .= D1[end, :]
        else
            A[inner_tau, :] .= 0; B[inner_tau, :] .= 0; A[inner_tau, P_idx] .= op.r[1] .* D2[1, :]
            A[outer_tau, :] .= 0; B[outer_tau, :] .= 0; A[outer_tau, P_idx] .= op.r[end] .* D2[end, :]
        end
    end

    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        if p.mechanical_bc == :no_slip
            A[riT, :] .= 0; B[riT, :] .= 0; A[riT, riT] = 1
            A[roT, :] .= 0; B[roT, :] .= 0; A[roT, roT] = 1
        else
            A[riT, :] .= 0; B[riT, :] .= 0
            A[roT, :] .= 0; B[roT, :] .= 0
            A[riT, T_idx] .= (-op.r[1]) .* D1[1, :]
            A[roT, T_idx] .= (-op.r[end]) .* D1[end, :]
            A[riT, riT] += 1
            A[roT, roT] += 1
        end
    end

    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        riΘ, roΘ = temperature_boundary_indices(Θ_idx)
        if p.thermal_bc == :fixed_temperature
            A[riΘ, :] .= 0; B[riΘ, :] .= 0; A[riΘ, riΘ] = 1
            A[roΘ, :] .= 0; B[roΘ, :] .= 0; A[roΘ, roΘ] = 1
        else
            A[riΘ, :] .= 0; B[riΘ, :] .= 0; A[riΘ, Θ_idx] .= D1[1, :]
            A[roΘ, :] .= 0; B[roΘ, :] .= 0; A[roΘ, Θ_idx] .= D1[end, :]
        end
    end
end

# -----------------------------------------------------------------------------
#  Matrix Assembly
# -----------------------------------------------------------------------------

"""Push dense `block` (length(row_idx)×length(col_idx)) into COO triplets at global
block position (row_idx, col_idx); keep only rows in `owned` (all if `owned===nothing`)."""
function _emit_block!(rows, cols, vals, row_idx, col_idx, block;
                      owned::Union{Nothing,UnitRange{Int}}=nothing)
    @inbounds for (jc, c) in enumerate(col_idx), (ir, r) in enumerate(row_idx)
        if owned === nothing || r in owned
            push!(rows, r); push!(cols, c); push!(vals, block[ir, jc])
        end
    end
    return nothing
end

"""
    _assemble_onset_radial_coo(op; owned_julia_rows=nothing)

Emit ONLY the pre-boundary-condition interior + coupling (radial-block) assembly of the
hydrodynamic onset/biglobal generalized eigenproblem as COO triplets. The basic-state
contribution is NOT included here. When `owned_julia_rows` is a `UnitRange`, only
triplets whose global row lies in that range are emitted, and per-ℓ blocks whose row
ranges do not intersect the owned rows are skipped entirely.

Returns a NamedTuple `(A_rows, A_cols, A_vals, B_rows, B_cols, B_vals, n)`.
"""
function _assemble_onset_radial_coo(op::LinearStabilityOperator{T};
                             owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing) where {T<:Real}
    p = op.params
    n = op.total_dof
    A_rows = Int[]; A_cols = Int[]; A_vals = Complex{T}[]
    B_rows = Int[]; B_cols = Int[]; B_vals = Complex{T}[]

    Ek = p.E
    Pr = p.Pr
    m = p.m
    ri = p.ri
    ro = p.ro
    gap = ro - ri

    # Convert gap-based Rayleigh number to internal Ra as per Kore's approach:
    # Ra_internal = Ra_gap / gap^3  (where gap = ro - ri = 1 - χ when ro = 1)
    # This is necessary because the non-dimensionalization uses r_o as the length scale
    # but Ra_gap is defined using the gap L = r_o - r_i as the length scale.
    # Reference: Barik et al. (2023), Onset of convection paper, Kore parameters.py
    Ra_internal = p.Ra / gap^3
    beyonce = -Ra_internal * Ek^2 / Pr
    thermaD = Ek / Pr

    R0 = radial_matrix(op, 0, 0)
    R1D0 = radial_matrix(op, 1, 0)
    R1D1 = radial_matrix(op, 1, 1)
    R2D0 = radial_matrix(op, 2, 0)
    R2D1 = radial_matrix(op, 2, 1)
    R2D2 = radial_matrix(op, 2, 2)
    R3D0 = radial_matrix(op, 3, 0)
    R3D1 = radial_matrix(op, 3, 1)
    R3D2 = radial_matrix(op, 3, 2)
    R3D3 = radial_matrix(op, 3, 3)
    R4D0 = radial_matrix(op, 4, 0)
    R4D1 = radial_matrix(op, 4, 1)
    R4D2 = radial_matrix(op, 4, 2)
    R4D4 = radial_matrix(op, 4, 4)

    TT = eltype(op.r)
    poloidal_ls = op.l_sets[:P]
    toroidal_ls = op.l_sets[:T]
    for ℓ in poloidal_ls
        if owned_julia_rows !== nothing
            rngs = (get(op.index_map,(ℓ,:P),nothing), get(op.index_map,(ℓ,:T),nothing), get(op.index_map,(ℓ,:Θ),nothing))
            any(rng -> rng !== nothing && !isempty(intersect(rng, owned_julia_rows)), rngs) || continue
        end
        L = TT(ℓ * (ℓ + 1))

        P_idx = op.index_map[(ℓ, :P)]
        Θ_idx = get(op.index_map, (ℓ, :Θ), nothing)

        _emit_block!(B_rows, B_cols, B_vals, P_idx, P_idx, -Complex.(L * (L * R2D0 - 2 * R3D1 - R4D2)); owned=owned_julia_rows)

        # Poloidal diagonal
        coriolis_p = 2im * m * (-L * R2D0 + 2 * R3D1 + R4D2)
        viscous_p = Ek * L * (-L*(ℓ+2)*(ℓ-1) * R0 + 2*L * R2D2 - 4 * R3D3 - R4D4)
        _emit_block!(A_rows, A_cols, A_vals, P_idx, P_idx, Complex.(coriolis_p - viscous_p); owned=owned_julia_rows)

        # Poloidal ↔ toroidal coupling
        if haskey(op.index_map, (ℓ - 1, :T))
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R3D0 - R4D1)
            _emit_block!(A_rows, A_cols, A_vals, P_idx, op.index_map[(ℓ-1, :T)], Complex.(coupling); owned=owned_julia_rows)
        end
        if haskey(op.index_map, (ℓ + 1, :T))
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R3D0 - R4D1)
            _emit_block!(A_rows, A_cols, A_vals, P_idx, op.index_map[(ℓ+1, :T)], Complex.(coupling); owned=owned_julia_rows)
        end

        # Temperature equation blocks for matching Θ ℓ
        if Θ_idx !== nothing
            if p.use_sparse_weighting
                B_theta = R3D0
                adv_coeff = ri / gap
                adv_matrix = R0
                diffusion = -L * R1D0 + 2 * R2D1 + R3D2
            else
                B_theta = R2D0
                adv_coeff = one(TT)
                adv_matrix = R2D0
                diffusion = -L * R0 + 2 * R1D1 + R2D2
            end

            _emit_block!(B_rows, B_cols, B_vals, Θ_idx, Θ_idx, Complex.(B_theta); owned=owned_julia_rows)

            # Temperature gradient coupling: only add if NO basic state
            # (basic state will provide explicit gradient through basic_state_operators)
            if p.basic_state === nothing
                thermal_adv = (L * adv_coeff) .* adv_matrix
                _emit_block!(A_rows, A_cols, A_vals, Θ_idx, P_idx, Complex.(thermal_adv); owned=owned_julia_rows)
            end

            _emit_block!(A_rows, A_cols, A_vals, Θ_idx, Θ_idx, Complex.(thermaD * diffusion); owned=owned_julia_rows)

            # Buoyancy coupling: temperature → velocity (OFF-DIAGONAL)
            # This term is ALWAYS present regardless of basic state
            buoyancy = beyonce * L * R4D0
            _emit_block!(A_rows, A_cols, A_vals, P_idx, Θ_idx, Complex.(buoyancy); owned=owned_julia_rows)
        end
    end

    for ℓ in toroidal_ls
        if owned_julia_rows !== nothing
            rngs = (get(op.index_map,(ℓ,:P),nothing), get(op.index_map,(ℓ,:T),nothing), get(op.index_map,(ℓ,:Θ),nothing))
            any(rng -> rng !== nothing && !isempty(intersect(rng, owned_julia_rows)), rngs) || continue
        end
        L = TT(ℓ * (ℓ + 1))
        T_idx = op.index_map[(ℓ, :T)]

        _emit_block!(B_rows, B_cols, B_vals, T_idx, T_idx, -Complex.(L * R2D0); owned=owned_julia_rows)

        # Toroidal diagonal
        # BUG FIX 2025-10-27: Removed incorrect L factor from Coriolis term
        # Kore operators.py:121-122: out = -2j*par.m*r2_D0_v (NO L factor!)
        # The toroidal Coriolis term should NOT have the L factor
        coriolis_t = -2im * m * R2D0
        viscous_t = Ek * L * (-L * R0 + 2 * R1D1 + R2D2)
        _emit_block!(A_rows, A_cols, A_vals, T_idx, T_idx, Complex.(coriolis_t - viscous_t); owned=owned_julia_rows)

        # Toroidal ↔ poloidal coupling
        if haskey(op.index_map, (ℓ - 1, :P))
            Cminus = (ℓ^2 - 1) * sqrt(max(zero(TT), ℓ^2 - m^2)) / (2ℓ - 1)
            coupling = 2 * Cminus * ((ℓ - 1) * R1D0 - R2D1)
            _emit_block!(A_rows, A_cols, A_vals, T_idx, op.index_map[(ℓ-1, :P)], Complex.(coupling); owned=owned_julia_rows)
        end
        if haskey(op.index_map, (ℓ + 1, :P))
            Cplus = ℓ*(ℓ+2) * sqrt(max(zero(TT), (ℓ+m+1)*(ℓ-m+1))) / (2ℓ + 3)
            coupling = 2 * Cplus * (-(ℓ + 2) * R1D0 - R2D1)
            _emit_block!(A_rows, A_cols, A_vals, T_idx, op.index_map[(ℓ+1, :P)], Complex.(coupling); owned=owned_julia_rows)
        end
    end

    return (A_rows=A_rows, A_cols=A_cols, A_vals=A_vals,
            B_rows=B_rows, B_cols=B_cols, B_vals=B_vals, n=n)
end

"""
    _assemble_onset_coo(op; owned_julia_rows=nothing)

Emit the full pre-boundary-condition assembly of the hydrodynamic onset/biglobal
generalized eigenproblem as COO triplets. This is the radial-block assembly
(`_assemble_onset_radial_coo`) plus, when `op.params.basic_state !== nothing`, the
basic-state coupling contribution (`add_basic_state_operators_coo!`). When
`owned_julia_rows` is a `UnitRange`, only triplets whose global row lies in that range
are emitted, so the assembly is row-distributable.

Returns a NamedTuple `(A_rows, A_cols, A_vals, B_rows, B_cols, B_vals, n)`.
"""
function _assemble_onset_coo(op::LinearStabilityOperator{T};
                             owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing) where {T<:Real}
    c = _assemble_onset_radial_coo(op; owned_julia_rows=owned_julia_rows)
    A_rows = c.A_rows; A_cols = c.A_cols; A_vals = c.A_vals
    B_rows = c.B_rows; B_cols = c.B_cols; B_vals = c.B_vals

    if op.params.basic_state !== nothing
        bs_ops = build_basic_state_operators(op.params.basic_state, op, op.params.m)
        add_basic_state_operators_coo!(A_rows, A_cols, A_vals, B_rows, B_cols, B_vals,
                                       bs_ops, op, op.params.m; owned_julia_rows=owned_julia_rows)
    end

    return (A_rows=A_rows, A_cols=A_cols, A_vals=A_vals,
            B_rows=B_rows, B_cols=B_cols, B_vals=B_vals, n=c.n)
end

"""
    assemble_matrices(op)

Assemble dense hydrodynamic onset or biglobal matrices and return the interior
and boundary DOF partition used by the constrained eigensolver.
"""
function assemble_matrices(op::LinearStabilityOperator{T}) where {T<:Real}
    # The COO assembly already includes the basic-state contribution (if any),
    # so the densified matrices are complete before boundary conditions.
    c = _assemble_onset_coo(op)
    n = c.n
    A = Matrix(sparse(c.A_rows, c.A_cols, c.A_vals, n, n))
    B = Matrix(sparse(c.B_rows, c.B_cols, c.B_vals, n, n))

    impose_boundary_conditions!(A, B, op)

    interior_dofs, boundary_dofs = _onset_boundary_interior_dofs(op)

    return A, B, interior_dofs, boundary_dofs
end

"""
    _onset_boundary_interior_dofs(op) -> (interior_dofs, boundary_dofs)

Compute the boundary/interior DOF partition for the tau-constrained eigenproblem
directly from the index map, without assembling `A`. The boundary DOFs are the
tau rows overwritten by `impose_boundary_conditions!`.
"""
function _onset_boundary_interior_dofs(op::LinearStabilityOperator{T}) where {T<:Real}
    n = op.total_dof
    is_boundary = falses(n)
    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)
        is_boundary[ri] = is_boundary[inner_tau] = is_boundary[outer_tau] = is_boundary[ro] = true
    end
    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        is_boundary[riT] = is_boundary[roT] = true
    end
    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        riΘ, roΘ = temperature_boundary_indices(Θ_idx)
        is_boundary[riΘ] = is_boundary[roΘ] = true
    end
    boundary_dofs = findall(is_boundary)
    interior_dofs = findall(!, is_boundary)
    return interior_dofs, boundary_dofs
end

"""Interior DOFs of the tau-constrained eigenproblem, without assembling `A`."""
function _onset_interior_dofs(op::LinearStabilityOperator{T}) where {T<:Real}
    interior_dofs, _ = _onset_boundary_interior_dofs(op)
    return interior_dofs
end

"""Constraint basis for one field block after eliminating tau boundary rows."""
struct ConstraintBasisBlock{T<:Real}
    full_indices::UnitRange{Int}
    reduced_indices::UnitRange{Int}
    basis::Matrix{Complex{T}}
end

"""Mapping from reduced interior coordinates back to full tau-constrained vectors."""
struct ConstraintReduction{T<:Real}
    blocks::Vector{ConstraintBasisBlock{T}}
    n_full::Int
    n_reduced::Int
end

"""Compute a nullspace basis that satisfies the tau rows for one field block."""
function _constraint_basis_block(A::Matrix{Complex{T}},
                                 idx::UnitRange{Int},
                                 constraint_rows::Vector{Int},
                                 label::String) where {T<:Real}
    # Slice only the rows that impose tau constraints for this field.  Its
    # nullspace maps reduced coefficients into full coefficients satisfying the
    # boundary equations exactly.
    constraints = Matrix(A[constraint_rows, idx])
    basis = nullspace(constraints)
    expected_cols = length(idx) - length(constraint_rows)
    if size(basis, 2) != expected_cols
        throw(ArgumentError(
            "Boundary constraints for $label have rank $(length(idx) - size(basis, 2)); " *
            "expected rank $(length(constraint_rows))"))
    end

    return basis
end

"""Build all per-field nullspace bases needed to eliminate boundary constraints."""
function _constraint_reduction(A::Matrix{Complex{T}},
                               op::LinearStabilityOperator{T},
                               boundary_dofs::Vector{Int}) where {T<:Real}
    n_full = op.total_dof
    n_reduced = n_full - length(boundary_dofs)
    blocks = ConstraintBasisBlock{T}[]
    col = 1

    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        ri, inner_tau, outer_tau, ro = poloidal_tau_indices(P_idx)
        basis = _constraint_basis_block(A, P_idx,
                                        [ri, inner_tau, outer_tau, ro],
                                        "poloidal ℓ=$ℓ")
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(P_idx, cols, basis))
        col += size(basis, 2)
    end
    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        riT, roT = toroidal_boundary_indices(T_idx)
        basis = _constraint_basis_block(A, T_idx, [riT, roT], "toroidal ℓ=$ℓ")
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(T_idx, cols, basis))
        col += size(basis, 2)
    end
    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        riΘ, roΘ = temperature_boundary_indices(Θ_idx)
        basis = _constraint_basis_block(A, Θ_idx, [riΘ, roΘ], "temperature ℓ=$ℓ")
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(Θ_idx, cols, basis))
        col += size(basis, 2)
    end

    col == n_reduced + 1 || error("Constraint basis size mismatch")
    return ConstraintReduction{T}(blocks, n_full, n_reduced)
end

"""
    _constraint_subblock(op, ℓ, field) -> Matrix{Complex{T}}

Build the `(#tau-rows × Nr)` block of boundary-condition rows for the `(ℓ, field)`
block directly from the BC formulas (`op.cd.D1`, `op.cd.D2`, `op.r`,
`op.params.mechanical_bc`, `op.params.thermal_bc`), without materializing the full
matrix `A`. This reproduces exactly the rows written by
`impose_boundary_conditions!`, sliced to the block columns, in the same row order
used by `_constraint_reduction`:

- `:P`  → `[ri, inner_tau, outer_tau, ro]`
- `:T`  → `[riT, roT]`
- `:Θ`  → `[riΘ, roΘ]`
"""
function _constraint_subblock(op::LinearStabilityOperator{T}, ℓ::Int,
                              field::Symbol) where {T<:Real}
    p = op.params
    D1 = op.cd.D1
    D2 = op.cd.D2
    r = op.r
    idx = op.index_map[(ℓ, field)]
    Nr = length(idx)

    if field === :P
        block = zeros(Complex{T}, 4, Nr)
        # ri (local row 1): Dirichlet e_1
        block[1, 1] = one(Complex{T})
        # ro (local row 4): Dirichlet e_Nr
        block[4, Nr] = one(Complex{T})
        if p.mechanical_bc == :no_slip
            block[2, :] .= D1[1, :]
            block[3, :] .= D1[end, :]
        else
            block[2, :] .= r[1] .* D2[1, :]
            block[3, :] .= r[end] .* D2[end, :]
        end
        return block
    elseif field === :T
        block = zeros(Complex{T}, 2, Nr)
        if p.mechanical_bc == :no_slip
            block[1, 1] = one(Complex{T})
            block[2, Nr] = one(Complex{T})
        else
            block[1, :] .= (-r[1]) .* D1[1, :]
            block[2, :] .= (-r[end]) .* D1[end, :]
            block[1, 1] += one(Complex{T})
            block[2, Nr] += one(Complex{T})
        end
        return block
    elseif field === :Θ
        block = zeros(Complex{T}, 2, Nr)
        if p.thermal_bc == :fixed_temperature
            block[1, 1] = one(Complex{T})
            block[2, Nr] = one(Complex{T})
        else
            block[1, :] .= D1[1, :]
            block[2, :] .= D1[end, :]
        end
        return block
    else
        throw(ArgumentError("Unknown field $field for constraint sub-block"))
    end
end

"""
    _constraint_reduction_from_subblocks(op) -> ConstraintReduction

Build the same per-field nullspace reduction as `_constraint_reduction`, but
compute each block's constraint matrix from `_constraint_subblock` (the BC
formulas) instead of slicing the assembled matrix `A`. Column bookkeeping mirrors
`_constraint_reduction` exactly.
"""
function _constraint_reduction_from_subblocks(op::LinearStabilityOperator{T}) where {T<:Real}
    n_full = op.total_dof
    _, boundary_dofs = _onset_boundary_interior_dofs(op)
    n_reduced = n_full - length(boundary_dofs)
    blocks = ConstraintBasisBlock{T}[]
    col = 1

    for ℓ in op.l_sets[:P]
        P_idx = op.index_map[(ℓ, :P)]
        basis = nullspace(_constraint_subblock(op, ℓ, :P))
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(P_idx, cols, basis))
        col += size(basis, 2)
    end
    for ℓ in op.l_sets[:T]
        T_idx = op.index_map[(ℓ, :T)]
        basis = nullspace(_constraint_subblock(op, ℓ, :T))
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(T_idx, cols, basis))
        col += size(basis, 2)
    end
    for ℓ in op.l_sets[:Θ]
        Θ_idx = op.index_map[(ℓ, :Θ)]
        basis = nullspace(_constraint_subblock(op, ℓ, :Θ))
        cols = col:(col + size(basis, 2) - 1)
        push!(blocks, ConstraintBasisBlock{T}(Θ_idx, cols, basis))
        col += size(basis, 2)
    end

    col == n_reduced + 1 || error("Constraint basis size mismatch")
    return ConstraintReduction{T}(blocks, n_full, n_reduced)
end

"""Project full tau-form matrices into the reduced interior coordinate system."""
function _constrained_reduced_matrices(A_full::Matrix{Complex{T}},
                                       B_full::Matrix{Complex{T}},
                                       op::LinearStabilityOperator{T},
                                       interior_dofs::Vector{Int},
                                       boundary_dofs::Vector{Int}) where {T<:Real}
    reduction = _constraint_reduction(A_full, op, boundary_dofs)
    A = zeros(Complex{T}, reduction.n_reduced, reduction.n_reduced)
    B = zeros(Complex{T}, reduction.n_reduced, reduction.n_reduced)

    for block in reduction.blocks
        # Project one physical field at a time. This avoids forming a full dense
        # block-diagonal basis matrix for all poloidal/toroidal/thermal modes.
        mul!(view(A, :, block.reduced_indices),
             view(A_full, interior_dofs, block.full_indices),
             block.basis)
        mul!(view(B, :, block.reduced_indices),
             view(B_full, interior_dofs, block.full_indices),
             block.basis)
    end

    return A, B, reduction
end

"""Reconstruct a full eigenvector, including constrained boundary coefficients."""
function _reconstruct_full_vector(reduction::ConstraintReduction{T},
                                  reduced_vec::AbstractVector{Complex{T}}) where {T<:Real}
    full_vec = zeros(Complex{T}, reduction.n_full)
    for block in reduction.blocks
        # Boundary coefficients are recovered from the same nullspace basis used
        # during matrix projection, keeping eigenvector reconstruction consistent.
        mul!(view(full_vec, block.full_indices), block.basis,
             view(reduced_vec, block.reduced_indices))
    end
    return full_vec
end

"""
    _constraint_projection_matrices(reduction, interior_dofs) -> (S, P)

Express the block-wise constraint reduction as two sparse matrices: `P`
(`n_full × n_reduced`, block-diagonal nullspace basis) and `S` (`n_reduced × n_full`,
interior-row selector), such that `S * A_full * P == _constrained_reduced_matrices(...)`.
"""
function _constraint_projection_matrices(reduction::ConstraintReduction{T},
                                         interior_dofs::Vector{Int}) where {T<:Real}
    length(interior_dofs) == reduction.n_reduced ||
        error("interior_dofs length $(length(interior_dofs)) != n_reduced $(reduction.n_reduced)")

    Pi = Int[]; Pj = Int[]; Pv = Complex{T}[]
    for block in reduction.blocks
        fr = block.full_indices
        rc = block.reduced_indices
        @inbounds for (cj, c) in enumerate(rc), (ri, r) in enumerate(fr)
            push!(Pi, r); push!(Pj, c); push!(Pv, block.basis[ri, cj])
        end
    end
    P = sparse(Pi, Pj, Pv, reduction.n_full, reduction.n_reduced)

    S = sparse(collect(1:reduction.n_reduced), interior_dofs,
               ones(Complex{T}, reduction.n_reduced),
               reduction.n_reduced, reduction.n_full)

    return S, P
end

# -----------------------------------------------------------------------------
#  Eigenvalue solve
# -----------------------------------------------------------------------------

"""
    solve_eigenvalue_problem(op; nev, tol, maxiter, which, sigma)

Compute the leading eigenpairs of the constrained linear-stability problem with
the SLEPc backend (the sole supported backend). The distributed
constrained-reduction path assembles the full tau pencil and the S/P projection
matrices, forms the reduced pencil `S·A·P` / `S·B·P`, runs the EPS shift-invert
solve, and reconstructs eigenvectors to full DOFs on rank 0.
"""
function solve_eigenvalue_problem(op::LinearStabilityOperator{T};
                                  nev::Int=6,
                                  backend::Symbol=:slepc,
                                  tol::Float64=1e-10,
                                  maxiter::Int=1000,
                                  which::Symbol=:LR,
                                  sigma::Union{Nothing,Number}=nothing) where {T<:Real}

    backend === :slepc || throw(ArgumentError(
        "Unknown eigensolver backend $(backend); only :slepc is supported"))

    return Magrathea._solve_constrained_slepc(op;
        nev=nev, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
end


"""Return the leading growth rate, drift frequency, and full eigenvector."""
function find_growth_rate(op::LinearStabilityOperator; kwargs...)
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op; kwargs...)
    idx = argmax(real.(eigenvalues))
    λ = eigenvalues[idx]
    σ = real(λ)
    ω = imag(λ)
    return σ, ω, eigenvectors[idx]
end

"""
    find_critical_rayleigh(E, Pr, chi, m, lmax, Nr; kwargs...)

Search for the Rayleigh number where the leading hydrodynamic growth rate
changes sign, optionally rebuilding a basic state at each sample.
"""
function find_critical_rayleigh(E::T, Pr::T, χ::T, m::Int, lmax::Int, Nr::Int;
                                Ra_guess::T=one(T)*1e6,
                                tol::T=1e-6,
                                Ra_bracket::Tuple{T,T}=(Ra_guess/10, Ra_guess*10),
                                basic_state_builder=nothing,
                                kwargs...) where {T<:Real}
    m_int = Int(m)
    lmax ≥ m_int || error("lmax must be ≥ m (got lmax=$lmax, m=$m_int)")

    onset_fields = Set(fieldnames(OnsetParams{T}))
    param_pairs = Pair{Symbol,Any}[]
    solver_pairs = Pair{Symbol,Any}[]
    basic_state_kw = nothing
    for (key, val) in kwargs
        if key == :basic_state
            basic_state_kw = val
        elseif key in onset_fields
            push!(param_pairs, key => val)
        else
            push!(solver_pairs, key => val)
        end
    end
    onset_kwargs = isempty(param_pairs) ? NamedTuple() : (; param_pairs...)
    solver_kwargs = isempty(solver_pairs) ? NamedTuple() : (; solver_pairs...)

    function build_operator(Ra_val::T)
        bs_override = basic_state_builder === nothing ? nothing : basic_state_builder(Ra_val)
        bs = bs_override === nothing ? basic_state_kw : bs_override
        bs_kwargs = bs === nothing ? NamedTuple() : (basic_state=bs,)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra_val, χ=χ, m=m_int, lmax=lmax, Nr=Nr;
                             onset_kwargs..., bs_kwargs...)
        return LinearStabilityOperator(params)
    end

    function growth_rate_at_Ra(Ra)
        op = build_operator(Ra)
        σ, _, _ = find_growth_rate(op; solver_kwargs...)
        return σ
    end

    cache = Dict{Float64,T}()
    function sigma_cached(Ra_val::T)
        key = Float64(Ra_val)
        return get!(cache, key) do
            growth_rate_at_Ra(Ra_val)
        end
    end

    pos = Ref{Union{Nothing,Tuple{T,T}}}(nothing)
    neg = Ref{Union{Nothing,Tuple{T,T}}}(nothing)

    function add_sample!(Ra_val::T)
        σ_val = sigma_cached(Ra_val)
        if abs(σ_val) < tol
            return (:root, Ra_val)
        elseif σ_val > 0
            pos[] = (Ra_val, σ_val)
        else
            neg[] = (Ra_val, σ_val)
        end
        return (:continue, Ra_val)
    end

    # Seed with guess and bracket endpoints
    Ra_guess_T = convert(T, Ra_guess)
    state, Ra_root = add_sample!(Ra_guess_T)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end

    Ra_low = convert(T, Ra_bracket[1])
    Ra_high = convert(T, Ra_bracket[2])
    state, Ra_root = add_sample!(Ra_low)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end
    state, Ra_root = add_sample!(Ra_high)
    if state == :root
        op_c = build_operator(Ra_root)
        σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
        return Ra_root, ω_c, vec_c
    end

    attempt = 0
    while (pos[] === nothing || neg[] === nothing) && attempt < 12
        if pos[] === nothing
            Ra_high *= T(2)
            state, Ra_root = add_sample!(Ra_high)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
        end
        if neg[] === nothing
            Ra_low /= T(2)
            state, Ra_root = add_sample!(Ra_low)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
        end
        attempt += 1
    end

    if pos[] === nothing || neg[] === nothing
        log_guess = log10(Float64(Ra_guess_T))
        step = 0.25
        for k in 1:80
            Ra_up = convert(T, 10.0^(log_guess + k * step))
            state, Ra_root = add_sample!(Ra_up)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
            if pos[] !== nothing && neg[] !== nothing
                break
            end
            Ra_down = convert(T, 10.0^(log_guess - k * step))
            state, Ra_root = add_sample!(Ra_down)
            if state == :root
                op_c = build_operator(Ra_root)
                σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)
                return Ra_root, ω_c, vec_c
            end
            if pos[] !== nothing && neg[] !== nothing
                break
            end
        end
    end

    (pos[] === nothing || neg[] === nothing) && error("Could not bracket the critical Rayleigh number")

    (Ra_pos, σ_pos) = pos[]
    (Ra_neg, σ_neg) = neg[]
    if σ_pos < 0
        Ra_pos, σ_pos, Ra_neg, σ_neg = Ra_neg, σ_neg, Ra_pos, σ_pos
    end

    a = Ra_neg
    fa = σ_neg
    b = Ra_pos
    fb = σ_pos
    c = a
    fc = fa
    d = b - a
    e = d

    for _ in 1:200
        if fb == zero(fb)
            a, fa = b, fb
            break
        end
        if sign(fa) == sign(fb)
            a, fa = c, fc
            d = b - a
            e = d
        end
        if abs(fa) < abs(fb)
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc
        end
        tol_act = 2 * eps(T) * abs(b) + tol / 2
        mid = (a - b) / 2
        if abs(mid) <= tol_act || fb == zero(fb)
            break
        end
        if abs(e) >= tol_act && abs(fc) > abs(fb)
            s = fb / fc
            if a == c
                p = 2 * mid * s
                q = 1 - s
            else
                q = fc / fa
                r = fb / fa
                p = s * (2 * mid * q * (q - r) - (b - c) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            end
            if p > 0
                q = -q
            else
                p = -p
            end
            if 2 * p < 3 * mid * q - abs(tol_act * q) && p < abs(e * q / 2)
                e = d
                d = p / q
            else
                d = mid
                e = mid
            end
        else
            d = mid
            e = mid
        end
        c, fc = b, fb
        if abs(d) > tol_act
            b += d
        else
            b += sign(mid) * tol_act
        end
        fb = sigma_cached(b)
    end

    Ra_c = b
    op_c = build_operator(Ra_c)
    σ_c, ω_c, vec_c = find_growth_rate(op_c; solver_kwargs...)

    return Ra_c, ω_c, vec_c
end

# Exports are centralized in Magrathea.jl
