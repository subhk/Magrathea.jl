# =============================================================================
#  Physical-space reconstruction of MHD perturbation fields from an eigenvector.
#
#  Native to the MHD spectral (Chebyshev-coefficient) basis: slice the
#  full eigenvector into per-(field, ℓ) coefficient
#  blocks, evaluate each Chebyshev series on a radial grid via direct
#  T_n(x) = cos(n·acos(x)) (accurate for the small N used here), then
#  synthesize physical fields on a meridional (r, θ) grid.  The poloidal-toroidal
#  curl uses the onset solver convention (velocity and magnetic field
#  share the same B = ∇×∇×(r P r̂) + ∇×(r T r̂) form).
# =============================================================================

"""Full MHD coefficient count, including the conducting core when present."""
function _mhd_reconstruction_dof(op::MHDStabilityOperator)
    return op.matrix_size
end

"""
    _mhd_full_vector(evec, op, interior_dofs)

Return a full-length DOF vector. If `evec` already has `_mhd_reconstruction_dof(op)`
entries it is copied. Reduced vectors need their basis transformation, not
zero insertion at boundary-equation indices. `interior_dofs` is accepted for API compatibility.
"""
function _mhd_full_vector(evec::AbstractVector{<:Complex},
                          op::MHDStabilityOperator,
                          interior_dofs)
    ndof = _mhd_reconstruction_dof(op)
    if length(evec) == ndof
        return Vector{ComplexF64}(evec)
    else
        throw(DimensionMismatch("MHD reconstruction requires all $ndof Chebyshev coefficients. " *
            "Tau boundary rows are equations, not removable coefficients. Use full eigenvectors " *
            "from solve(MHDProblem(...)), or reconstruct_mhd_galerkin_full with its layout."))
    end
end

"""Slice the `(N+1)` spectral coefficients for `(ℓ, field)` from a full vector."""
function _mhd_field_block(full::AbstractVector{<:Complex},
                          idx_map::Dict{Tuple{Int,Symbol},UnitRange{Int}},
                          field::Symbol, ℓ::Int)
    return @view full[idx_map[(ℓ, field)]]
end

"""
    _mhd_radial_eval(coeffs, ricb, r_grid)

Evaluate a Chebyshev-T coefficient series (`coeffs[n+1]` multiplies `T_n`) at
physical radii `r_grid ∈ [ricb, 1]`, via `x = 2(r-ricb)/(1-ricb) - 1` and
`T_n(x) = cos(n·acos(x))`.
"""
function _mhd_radial_eval(coeffs::AbstractVector{<:Complex},
                          ricb::Real, r_grid::AbstractVector)
    N = length(coeffs) - 1
    out = zeros(ComplexF64, length(r_grid))
    @inbounds for (i, r) in enumerate(r_grid)
        x = 2 * (r - ricb) / (1 - ricb) - 1
        x = clamp(x, -1.0, 1.0)
        ax = acos(x)
        acc = zero(ComplexF64)
        for n in 0:N
            acc += coeffs[n + 1] * cos(n * ax)
        end
        out[i] = acc
    end
    return out
end

"""
Radial reconstruction grid: the Chebyshev collocation nodes of a
`ChebyshevDiffn(Nr, [ricb, 1], 1)`. `_mhd_poltor_to_physical` builds the SAME
`ChebyshevDiffn`, so its `D1` matches these nodes exactly (no grid mismatch).
"""
function _mhd_radial_grid(op::MHDStabilityOperator; Nr::Int = op.params.N + 1)
    return ChebyshevDiffn(Nr, [op.params.ricb, 1.0], 1).x
end

"""
    perturbation_temperature(evec, op::MHDStabilityOperator;
                             Nθ=nothing, Nr=nothing, interior_dofs=nothing, grid=nothing)

Reconstruct the physical perturbation temperature field
`θ(r,θ) = Σ_ℓ h_ℓ(r) Y_ℓ^m(θ)/√(2ℓ+1)` from a full MHD eigenvector.
Returns `(θfield, r_grid, grid)`.
"""
function perturbation_temperature(evec::AbstractVector{<:Complex},
                                  op::MHDStabilityOperator;
                                  Nθ::Union{Int,Nothing}=nothing,
                                  Nr::Union{Int,Nothing}=nothing,
                                  interior_dofs=nothing,
                                  grid::Union{MeridionalGrid,Nothing}=nothing)
    m    = op.params.m
    lmax = op.params.lmax
    ricb = op.params.ricb
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * lmax : Nθ, m, lmax;
                              T=typeof(op.params.E)) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)

    θfield = zeros(ComplexF64, length(r_grid), length(g.θ))
    for l in op.ll_h
        hl = _mhd_radial_eval(_mhd_field_block(full, idx_map, :h, l), ricb, r_grid)
        ylm = g.Ylm[l] ./ sqrt(2l + 1)
        @inbounds for j in eachindex(g.θ), k in eachindex(r_grid)
            θfield[k, j] += hl[k] * ylm[j]
        end
    end
    return θfield, r_grid, g
end

# Native potentials multiply the radius vector and use Y_lm/sqrt(2l+1).
function _mhd_poltor_to_physical(full, idx_map, op,
                                 ls_pol, sec_pol::Symbol,
                                 ls_tor, sec_tor::Symbol,
                                 r_grid, g::MeridionalGrid)
    radial(sec, l) = _mhd_radial_eval(_mhd_field_block(full, idx_map, sec, l),
                                     op.params.ricb, r_grid)
    P = Dict(l => radial(sec_pol, l) for l in ls_pol)
    Tor = Dict(l => radial(sec_tor, l) for l in ls_tor)
    Dr = ChebyshevDiffn(length(r_grid), [op.params.ricb, 1.0], 1).D1
    return _onset_velocity_from_coefficients(P, Tor, r_grid, Dr, g, op.params.m)
end

"""
    perturbation_velocity(evec, op::MHDStabilityOperator; kwargs...)

Reconstruct physical perturbation velocity `(u_r, u_θ, u_φ)` from an MHD
eigenvector (poloidal `:u`, toroidal `:v`). Returns `(ur, uθ, uφ, r_grid, grid)`.
"""
function perturbation_velocity(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing)
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax;
                              T=typeof(op.params.E)) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)
    Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
                                          op.ll_u, :u, op.ll_v, :v, r_grid, g)
    return Fr, Fθ, Fφ, r_grid, g
end

"""
    perturbation_magnetic(evec, op::MHDStabilityOperator; kwargs...)

Reconstruct physical perturbation magnetic field `(B_r, B_θ, B_φ)` from an MHD
full eigenvector (shell poloidal `:f`, toroidal `:g`). Core coefficients remain
in the vector but this function reconstructs only the fluid shell. Returns `(Br, Bθ, Bφ, r_grid, grid)`.
"""
function perturbation_magnetic(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing)
    isempty(op.ll_f) && error(
        "perturbation_magnetic: this MHD problem has no magnetic field " *
        "(B0_type=no_field). Nothing to reconstruct.")
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax;
                              T=typeof(op.params.E)) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)
    Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
                                          op.ll_f, :f, op.ll_g, :g, r_grid, g)
    return Fr, Fθ, Fφ, r_grid, g
end
