# Mean-state operators for linear stability. Physical vector linearization and
# pressure elimination are implemented in Stability/mean_flow_coupling.jl.
# The scalar harmonic utilities below are also used by legacy analysis helpers.

using LinearAlgebra
using SparseArrays
using WignerSymbols

"""
    BasicStateOperators{T}

Linearized mean-state blocks keyed by `(l_output, field_output, l_input, field_input)`.
`blocks` is authoritative and includes all velocity/temperature couplings. The
older two-index dictionaries remain inspection aliases for the complete matching
field blocks; they no longer split vector advection into scalar approximations.
"""
struct BasicStateOperators{T<:Real}
    advection_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    shear_radial_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    shear_theta_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    shear_theta_toroidal_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    temp_grad_radial_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    temp_grad_theta_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    temp_grad_theta_toroidal_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    metric_poloidal_blocks::Dict{Tuple{Int,Int}, Matrix{Complex{T}}}
    coupling_structure::Vector{Tuple{Int,Int}}
    blocks::Dict{Tuple{Int,Symbol,Int,Symbol},Matrix{Complex{T}}}
end

"""Return the existing coupling block for `key`, or allocate a typed zero block."""
function _operator_block!(blocks::Dict{Tuple{Int,Int}, Matrix{CT}},
                          key::Tuple{Int,Int}, Nr::Int,
                          ::Type{CT}) where {CT}
    return get!(blocks, key) do
        zeros(CT, Nr, Nr)
    end
end

"""Add `scale * Diagonal(diagonal)` into an already-allocated block."""
function _add_diagonal_block!(block::AbstractMatrix{CT},
                              scale,
                              diagonal::AbstractVector) where {CT}
    @inbounds for i in eachindex(diagonal)
        block[i, i] += CT(scale * diagonal[i])
    end
    return block
end

"""Add `scale * Diagonal(a .* b)` without allocating the product vector."""
function _add_diagonal_product_block!(block::AbstractMatrix{CT},
                                      scale,
                                      a::AbstractVector,
                                      b::AbstractVector) where {CT}
    @inbounds for i in eachindex(a, b)
        block[i, i] += CT(scale * a[i] * b[i])
    end
    return block
end

"""Add `scale * Diagonal(diagonal) * matrix` without constructing `Diagonal`."""
function _add_left_diagonal_matrix_block!(block::AbstractMatrix{CT},
                                          scale,
                                          diagonal::AbstractVector,
                                          matrix::AbstractMatrix) where {CT}
    @inbounds for j in axes(matrix, 2)
        for i in eachindex(diagonal)
            block[i, j] += CT(scale * diagonal[i] * matrix[i, j])
        end
    end
    return block
end


"""
    compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)

Compute coupling coefficients between perturbation mode (ℓ_pert, m) and
basic state mode (ℓ_bs, 0) through spherical harmonic integrals.

For products like Y_ℓpert,m × Y_ℓbs,0, we get contributions to Y_ℓ',m where
ℓ' ranges over |ℓ_pert - ℓ_bs| to ℓ_pert + ℓ_bs with appropriate selection rules.

Returns:
- `coupling_coeffs::Dict{Int, Float64}` - Coefficients for each coupled ℓ' mode
"""
function compute_spherical_harmonic_coupling(ℓ_pert::Int, ℓ_bs::Int, m::Int)
    # For axisymmetric basic state (m_bs = 0), the product
    # Y_ℓpert,m × Y_ℓbs,0 couples to modes with same m
    #
    # Using Clebsch-Gordan coefficients and selection rules:
    # ℓ' ∈ [|ℓ_pert - ℓ_bs|, ℓ_pert + ℓ_bs] with ℓ' + ℓ_pert + ℓ_bs even

    coupling_coeffs = Dict{Int, Float64}()

    ℓ_min = abs(ℓ_pert - ℓ_bs)
    ℓ_max = ℓ_pert + ℓ_bs

    for ℓ_prime in ℓ_min:ℓ_max
        # Selection rule: ℓ' + ℓ_pert + ℓ_bs must be even
        if (ℓ_prime + ℓ_pert + ℓ_bs) % 2 != 0
            continue
        end

        # Also need ℓ_prime >= m
        if ℓ_prime < m
            continue
        end

        # Gaunt coefficient (Wigner 3j symbol related)
        # Simplified formula for m_bs = 0 case
        coeff = compute_gaunt_coefficient(ℓ_pert, m, ℓ_bs, 0, ℓ_prime, m)

        if abs(coeff) > 1e-14
            coupling_coeffs[ℓ_prime] = coeff
        end
    end

    return coupling_coeffs
end


"""
    wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)

Compute Wigner 3j symbol with all m = 0:
    ⎛ℓ1  ℓ2  ℓ3⎞
    ⎝0   0   0 ⎠

Uses WignerSymbols.jl for accurate computation.
"""
function wigner3j_000(ℓ1::Int, ℓ2::Int, ℓ3::Int)
    return Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, 0, 0, 0))
end

"""
    compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)

Compute Gaunt coefficient (integral of three spherical harmonics):

    ∫ Y_ℓ1,m1 × Y_ℓ2,m2 × conj(Y_ℓ3,m3) dΩ

Using Wigner 3j symbols:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2   ℓ3 ⎞
                                        ⎝0   0   0 ⎠ ⎝m1  m2  -m3⎠

For axisymmetric basic state (m2 = 0), this simplifies to:
    G = √[(2ℓ1+1)(2ℓ2+1)(2ℓ3+1)/(4π)] × ⎛ℓ1  ℓ2  ℓ3⎞ ⎛ℓ1  ℓ2  ℓ3 ⎞ × δ_{m1,m3}
                                        ⎝0   0   0 ⎠ ⎝m1  0   -m1⎠
"""
# Gaunt coefficients are pure functions of the six integer quantum numbers and
# `WignerSymbols.wigner3j` allocates (BigInt/Rational) internally, so memoize them.
# The coupling builders evaluate the same (ℓ,m) triples across every basic-state
# mode and Picard iteration. The cache is shared across threads; see `_locked_get!`.
const _GAUNT_CACHE = Dict{NTuple{6,Int}, Float64}()
const _GAUNT_CACHE_LOCK = ReentrantLock()

function compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    return _locked_get!(_GAUNT_CACHE, _GAUNT_CACHE_LOCK, (ℓ1, m1, ℓ2, m2, ℓ3, m3)) do
        _compute_gaunt_coefficient(ℓ1, m1, ℓ2, m2, ℓ3, m3)
    end
end

function _compute_gaunt_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rules for ⟨Y_{ℓ1,m1} Y_{ℓ2,m2} Y_{ℓ3,m3}*⟩
    if m1 + m2 != m3
        return 0.0
    end
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end
    if !((abs(ℓ1 - ℓ2) <= ℓ3 <= ℓ1 + ℓ2))
        return 0.0
    end
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    w3j_1 = wigner3j_000(ℓ1, ℓ2, ℓ3)
    abs(w3j_1) < 1e-14 && return 0.0

    w3j_2 = Float64(WignerSymbols.wigner3j(ℓ1, ℓ2, ℓ3, m1, m2, -m3))
    abs(w3j_2) < 1e-14 && return 0.0

    norm_factor = sqrt((2 * ℓ1 + 1) * (2 * ℓ2 + 1) * (2 * ℓ3 + 1) / (4 * π))
    phase = isodd(m3) ? -1.0 : 1.0
    return phase * norm_factor * w3j_1 * w3j_2
end


# The quadrature coupling matrix once built from these tables was removed: its
# Gauss–Chebyshev weights integrate ∫YYY/sinθ, not a Gaunt integral. The table
# builder is no longer used by the solvers.
"""Normalized Legendre tables at Gauss–Chebyshev nodes for fixed m."""
struct AzimuthalCouplingCache{T<:Real}
    m::Int
    weight::T
    y_m::Matrix{T}
    y_0::Matrix{T}
end

# Note: _double_factorial, _associated_legendre_table, and _normalization_table
# are defined in Stability/velocity.jl and resolved when these helpers are called.

"""Precompute normalized Legendre tables at Gauss–Chebyshev nodes."""
function _build_azimuthal_coupling_cache(m::Int, lmax_m::Int, lmax_0::Int)
    return _build_azimuthal_coupling_cache(m, lmax_m, lmax_0, Float64)
end

function _build_azimuthal_coupling_cache(m::Int, lmax_m::Int, lmax_0::Int,
                                         ::Type{T}) where {T<:Real}
    ntheta = max(64, 4 * max(lmax_m, lmax_0) + 1)
    inv_denominator = one(T) / (T(2) * T(ntheta))
    mu = [cos(T(2k - 1) * T(pi) * inv_denominator) for k in 1:ntheta]
    weight = T(pi) / T(ntheta)

    Pm = _associated_legendre_table(m, lmax_m, mu)
    P0 = _associated_legendre_table(0, lmax_0, mu)
    Nm = _normalization_table(T, m, lmax_m)
    N0 = _normalization_table(T, 0, lmax_0)

    y_m = similar(Pm)
    for i in axes(Pm, 1)
        y_m[i, :] .= Nm[i] .* Pm[i, :]
    end

    y_0 = similar(P0)
    for i in axes(P0, 1)
        y_0[i, :] .= N0[i] .* P0[i, :]
    end

    return AzimuthalCouplingCache(m, weight, y_m, y_0)
end

"""Build the complete linearized physical equations about an axisymmetric state."""
function build_basic_state_operators(bs::BasicState{T}, op, m::Int) where T
    blocks=_mean_state_blocks(bs,op,op,m,m)
    select(fo,fi)=Dict((lo,li)=>b for ((lo,a,li,c),b) in blocks if a===fo && c===fi)
    emptyblocks()=Dict{Tuple{Int,Int},Matrix{Complex{T}}}()
    pairs=sort!(unique([(lo,li) for (lo,fo,li,fi) in keys(blocks)]))
    BasicStateOperators{T}(select(:Θ,:Θ),select(:T,:P),emptyblocks(),
        select(:T,:T),select(:Θ,:P),emptyblocks(),select(:Θ,:T),
        select(:P,:T),pairs,blocks)
end

"""Add mean-state blocks to A; B is unchanged."""
function add_basic_state_operators!(A::Matrix, B::Matrix,
        bs_ops::BasicStateOperators, op, m::Int)
    for ((lo,fo,li,fi),block) in bs_ops.blocks
        A[op.index_map[(lo,fo)],op.index_map[(li,fi)]] .+= block
    end
    nothing
end

"""Emit the same physical blocks as the dense path, restricted to owned rows."""
function add_basic_state_operators_coo!(A_rows,A_cols,A_vals,B_rows,B_cols,B_vals,
        bs_ops::BasicStateOperators,op,m::Int;owned_julia_rows=nothing)
    for ((lo,fo,li,fi),block) in bs_ops.blocks
        _emit_block!(A_rows,A_cols,A_vals,op.index_map[(lo,fo)],op.index_map[(li,fi)],
                     block;owned=owned_julia_rows)
    end
    nothing
end
