# =============================================================================
#  Tri-Global Instability Analysis
#
#  Linear stability analysis for non-axisymmetric basic states where
#  perturbations couple across multiple azimuthal modes m.
#
#  With a basic state containing mode m_bs, the perturbation temperature:
#    θ'(r,θ,φ,t) = Σ_m θ'_m(r,θ,t) e^{imφ}
#  couples modes m and m ± m_bs through advection by the basic state.
#
#  This requires solving a BLOCK-COUPLED eigenvalue problem where different
#  m-blocks are coupled through the basic state.
# =============================================================================

# Dependencies provided by Magrathea module:
# Parameters, LinearAlgebra, SparseArrays, Printf, WignerSymbols
# LinearStabilityOperator, OnsetParams, BasicState, assemble_matrices
# are available in the Magrathea namespace

"""
    TriglobalParams{T<:Real}

Parameters for tri-global instability analysis with non-axisymmetric basic state.

Unlike OnsetParams which solves for a single azimuthal mode m, TriglobalParams
solves for MULTIPLE coupled modes simultaneously.

Fields:
- `E::T` - Ekman number
- `Pr::T` - Prandtl number
- `Ra::T` - Rayleigh number
- `χ::T` - Radius ratio r_i/r_o
- `m_range::UnitRange{Int}` - Range of perturbation modes to include (e.g., -2:2)
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `basic_state_3d::BasicState3D{T}` - The 3D basic state
- `mechanical_bc::Symbol` - :no_slip or :stress_free
- `thermal_bc::Symbol` - :fixed_temperature or :fixed_flux
- `equatorial_symmetry::Symbol` - :both, :symmetric, or :antisymmetric

Note: The size of the eigenvalue problem is ~ length(m_range) × lmax × Nr × 3
which can become very large. Use sparse methods and Krylov subspace solvers.
"""
@with_kw_noshow struct TriglobalParams{T<:Real}
    E::T
    Pr::T
    Ra::T
    χ::T
    m_range::UnitRange{Int}
    lmax::Int
    Nr::Int
    basic_state_3d::BasicState3D{T}
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    equatorial_symmetry::Symbol = :both
end

"""Validate that every requested azimuthal mode can be represented at `lmax`."""
function _validate_triglobal_m_range(m_range::UnitRange{Int}, lmax::Int)
    isempty(m_range) && throw(ArgumentError("m_range must be non-empty"))
    max_abs_m = maximum(abs, m_range)
    max_abs_m <= lmax || throw(ArgumentError(
        "m_range includes |m|=$max_abs_m, but lmax=$lmax; require maximum(abs, m_range) <= lmax"))
    return nothing
end

"""Validate triglobal equatorial-symmetry selection before constructing blocks."""
function _validate_triglobal_symmetry(equatorial_symmetry::Symbol)
    equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
        "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$equatorial_symmetry"))
    return nothing
end

"""Compute the per-field l sets for a triglobal mode using the onset layout rules."""
function _triglobal_mode_l_sets(params::TriglobalParams{T}, m::Int) where T
    params_m = OnsetParams(
        E = params.E,
        Pr = params.Pr,
        Ra = params.Ra,
        χ = params.χ,
        m = abs(m),
        lmax = params.lmax,
        Nr = params.Nr,
        mechanical_bc = params.mechanical_bc,
        thermal_bc = params.thermal_bc,
        equatorial_symmetry = params.equatorial_symmetry,
        basic_state = nothing
    )
    return compute_l_sets(params_m)
end

"""Count poloidal, toroidal, and temperature l modes for one triglobal m block."""
function _triglobal_mode_l_counts(m::Int, lmax::Int, equatorial_symmetry::Symbol)
    m_abs = abs(m)
    if equatorial_symmetry === :both
        n = lmax - m_abs + 1
        return n, n, n
    end

    vsymm = _symmetry_flag(equatorial_symmetry)
    signm = m_abs == 0 ? 0 : 1
    lm1 = lmax - m_abs + 1
    s = Int((vsymm + 1) ÷ 2)
    pol_start = (signm + s) % 2
    tor_start = (signm + s + 1) % 2

    n_pol = length(pol_start:2:(lm1 - 1))
    n_tor = length(tor_start:2:(lm1 - 1))
    return n_pol, n_tor, n_pol
end

"""Return the reduced block size after tau constraints for one triglobal m block."""
function _triglobal_reduced_block_size(params::TriglobalParams, m::Int)
    nP, nT, nΘ = _triglobal_mode_l_counts(m, params.lmax, params.equatorial_symmetry)
    return nP * (params.Nr - 4) + nT * (params.Nr - 2) + nΘ * (params.Nr - 2)
end


"""
    get_coupling_modes(m::Int, m_bs::Int, m_range::UnitRange{Int})

Determine which perturbation modes couple to mode m through basic state mode m_bs.

A basic state with azimuthal mode m_bs couples perturbation modes m and m ± m_bs
through advection terms.

Returns:
- Vector of coupled mode numbers that are within m_range

Example:
    # Basic state with m_bs = 2, perturbation range -3:3
    get_coupling_modes(0, 2, -3:3)  # Returns [-2, 0, 2]
    get_coupling_modes(1, 2, -3:3)  # Returns [-1, 1, 3]
"""
function get_coupling_modes(m::Int, m_bs::Int, m_range::UnitRange{Int})
    coupled_modes = Int[]

    # Mode m couples to m - m_bs, m, m + m_bs
    for Δm in [-m_bs, 0, m_bs]
        m_coupled = m + Δm
        if m_coupled in m_range
            push!(coupled_modes, m_coupled)
        end
    end

    return sort(unique(coupled_modes))
end

"""Return nonzero 3D basic-state modes that can induce triglobal coupling."""
function _nonzero_basic_state_modes_3d(basic_state::BasicState3D; tol=0)
    modes = Tuple{Int,Int}[]
    coefficient_dicts = (
        basic_state.theta_coeffs,
        basic_state.dtheta_dr_coeffs,
        basic_state.ur_coeffs,
        basic_state.utheta_coeffs,
        basic_state.uphi_coeffs,
        basic_state.dur_dr_coeffs,
        basic_state.dutheta_dr_coeffs,
        basic_state.duphi_dr_coeffs,
    )

    # Collapse ±m to a single positive-|m| representative: the real cosine part is
    # stored at (ℓ, +|m|) and the sine part at (ℓ, -|m|); both feed the same complex
    # coupling coefficient (see _coupling_profile). Listing both signs here
    # would double-iterate (and double-count) the same physical mode.
    for coefficient_dict in coefficient_dicts
        for ((ℓ, m_bs), coeff) in coefficient_dict
            if m_bs != 0 && _maxabs(coeff) > tol
                push!(modes, (ℓ, abs(m_bs)))
            end
        end
    end

    if basic_state.flow !== nothing
        for d in (basic_state.flow.p,basic_state.flow.t), ((l,m),c) in d
            m!=0 && any(!iszero,c) && push!(modes,(l,abs(m)))
        end
    end
    return sort(unique(modes))
end

"""
    build_mode_coupling_structure(m_range::UnitRange{Int},
                                  basic_state::BasicState3D{T}) where T

Analyze the coupling structure between perturbation modes induced by the basic state.

Returns:
- `coupling_graph::Dict{Int, Vector{Int}}` - For each mode m, which other modes couple to it
- `all_m_bs::Vector{Int}` - All non-zero azimuthal modes in the basic state

This information is used to construct the block-sparse eigenvalue problem.
"""
function build_mode_coupling_structure(m_range::UnitRange{Int},
                                       basic_state::BasicState3D{T}) where T

    bs_modes = _nonzero_basic_state_modes_3d(basic_state)
    all_m_bs = sort(unique(m_bs for (_, m_bs) in bs_modes))

    # Build coupling graph
    coupling_graph = Dict{Int, Vector{Int}}()

    for m in m_range
        coupled_modes = Int[m]  # Always couples to itself

        # Add coupling through each basic state mode
        for m_bs in all_m_bs
            # Basic state mode m_bs couples m to m ± m_bs
            for Δm in [-m_bs, m_bs]
                m_coupled = m + Δm
                if m_coupled in m_range && m_coupled != m
                    push!(coupled_modes, m_coupled)
                end
            end
        end

        coupling_graph[m] = sort(unique(coupled_modes))
    end

    return coupling_graph, all_m_bs
end


"""
    estimate_triglobal_problem_size(params::TriglobalParams{T}) where T

Estimate the size of the tri-global eigenvalue problem.

Returns:
- `total_dofs::Int` - Total degrees of freedom
- `matrix_size::Int` - Size of the matrix (= total_dofs)
- `num_modes::Int` - Number of coupled azimuthal modes
- `dofs_per_mode::Int` - Degrees of freedom per mode

Useful for assessing computational requirements before attempting to solve.
"""
function estimate_triglobal_problem_size(params::TriglobalParams{T}) where T
    num_modes = length(params.m_range)
    _validate_triglobal_m_range(params.m_range, params.lmax)
    _validate_triglobal_symmetry(params.equatorial_symmetry)

    total_dofs = 0
    for m in params.m_range
        total_dofs += _triglobal_reduced_block_size(params, m)
    end

    matrix_size = total_dofs
    average_per_mode = total_dofs / num_modes

    return (
        total_dofs = total_dofs,
        matrix_size = matrix_size,
        num_modes = num_modes,
        dofs_per_mode = average_per_mode
    )
end


"""
    CoupledModeProblem{T<:Real}

Data structure for the coupled-mode eigenvalue problem.

Fields:
- `params::TriglobalParams{T}` - Problem parameters
- `m_range::UnitRange{Int}` - Range of coupled modes
- `coupling_graph::Dict{Int,Vector{Int}}` - Mode coupling structure
- `block_indices::Dict{Int,UnitRange{Int}}` - Index ranges for each mode block
- `total_dofs::Int` - Total degrees of freedom

This structure organizes the information needed to assemble and solve the
block-coupled eigenvalue problem:
    A_coupled x = λ B_coupled x
where A_coupled has diagonal blocks (single-mode operators) and off-diagonal
blocks (mode coupling through basic state).
"""
@with_kw mutable struct CoupledModeProblem{T<:Real}
    params::TriglobalParams{T}
    m_range::UnitRange{Int}
    coupling_graph::Dict{Int,Vector{Int}}
    all_m_bs::Vector{Int}
    block_indices::Dict{Int,UnitRange{Int}}
    total_dofs::Int
end


"""
    setup_coupled_mode_problem(params::TriglobalParams{T}) where T

Initialize the coupled-mode eigenvalue problem structure.

This analyzes the basic state to determine:
1. Which perturbation modes m couple to each other
2. The index ranges for each m-block in the global matrix
3. The total problem size

Returns a CoupledModeProblem structure.
"""
function setup_coupled_mode_problem(params::TriglobalParams{T}) where T
    _validate_triglobal_m_range(params.m_range, params.lmax)
    _validate_triglobal_symmetry(params.equatorial_symmetry)

    m_range = params.m_range
    basic_state = params.basic_state_3d

    # Analyze coupling structure
    coupling_graph, all_m_bs = build_mode_coupling_structure(m_range, basic_state)

    # Compute index ranges for each mode in reduced coordinates after applying
    # tau constraints and any requested equatorial-symmetry truncation.
    block_indices = Dict{Int,UnitRange{Int}}()
    current_idx = 1

    for m in m_range
        block_size = _triglobal_reduced_block_size(params, m)

        block_indices[m] = current_idx:(current_idx + block_size - 1)
        current_idx += block_size
    end

    total_dofs = current_idx - 1

    return CoupledModeProblem(
        params = params,
        m_range = m_range,
        coupling_graph = coupling_graph,
        all_m_bs = all_m_bs,
        block_indices = block_indices,
        total_dofs = total_dofs
    )
end


# =============================================================================
#  Helper Functions for Tri-Global Eigenvalue Problem
# =============================================================================

"""Extract the axisymmetric part of a 3D basic state for diagonal m blocks."""
function axisymmetric_basic_state(basic_state::BasicState3D{T}) where T
    return _axisymmetric_state(basic_state)
end

"""Return true when an axisymmetric basic state has any active temperature or flow."""
function _has_nonzero_basic_state(bs::BasicState{T}; tol=zero(T)) where T
    any(any(abs(x)>tol for x in v) for d in (bs.theta_coeffs,bs.ur_coeffs,
        bs.utheta_coeffs,bs.uphi_coeffs) for v in values(d)) && return true
    bs.flow===nothing && return false
    any(any(abs(x)>tol for x in v) for d in (bs.flow.p,bs.flow.t) for v in values(d))
end

"""Single-m block operator plus the reductions needed for triglobal assembly."""
struct SingleModeOperator{T<:Real}
    A::Matrix{Complex{T}}
    B::Matrix{Complex{T}}
    op::LinearStabilityOperator{T}
    idx_map::Dict{Tuple{Int,Symbol}, Vector{Int}}
    interior_dofs::Vector{Int}
    boundary_dofs::Vector{Int}
    reduction::ConstraintReduction{T}
end

"""
    build_single_mode_operators(problem::CoupledModeProblem, verbose::Bool)

Build single-mode linear stability operators for each azimuthal mode m.

Returns a dictionary mapping m to a NamedTuple with:
- `A`, `B` - interior-DOF matrices for mode m
- `op` - LinearStabilityOperator for mode m
- `idx_map` - reduced index map for (ℓ, field) radial locations
"""
function build_single_mode_operators(problem::CoupledModeProblem{T}, verbose::Bool) where T
    params_tri = problem.params
    single_mode_ops = Dict{Int, SingleModeOperator{T}}()
    basic_state_axis = axisymmetric_basic_state(params_tri.basic_state_3d)

    for m in problem.m_range
        if verbose && abs(m) <= 2
            print("  m = $m... ")
        end

        # Create OnsetParams for this mode
        params_m = OnsetParams(
            E = params_tri.E,
            Pr = params_tri.Pr,
            Ra = params_tri.Ra,
            χ = params_tri.χ,
            m = abs(m),  # m must be non-negative for OnsetParams
            lmax = params_tri.lmax,
            Nr = params_tri.Nr,
            mechanical_bc = params_tri.mechanical_bc,
            thermal_bc = params_tri.thermal_bc,
            equatorial_symmetry = params_tri.equatorial_symmetry,
            basic_state = basic_state_axis
        )

        # Create operator and assemble matrices
        op_m = LinearStabilityOperator(params_m)
        A_full, B_full, interior_dofs, boundary_dofs = assemble_matrices(op_m)
        A_m, B_m, reduction = _constrained_reduced_matrices(
            A_full, B_full, op_m, interior_dofs, boundary_dofs)
        expected_dofs = length(problem.block_indices[m])
        if reduction.n_reduced != expected_dofs
            error("Reduced DOF count mismatch for m=$m: got $(reduction.n_reduced), expected $expected_dofs")
        end
        idx_map = _full_index_map(op_m)
        if m < 0
            A_m = conj(A_m)
            B_m = conj(B_m)
        end

        single_mode_ops[m] = SingleModeOperator(
            A_m, B_m, op_m, idx_map, interior_dofs, boundary_dofs, reduction)

        if verbose && abs(m) <= 2
            println("$(size(A_m, 1)) DOFs")
        end
    end

    if verbose && length(problem.m_range) > 5
        println("  ... ($(length(problem.m_range)) modes total)")
    end

    return single_mode_ops
end


"""
    build_single_mode_operator(problem::CoupledModeProblem, m::Int)

Build the single-mode linear stability operator for one azimuthal mode `m`,
on demand. Produces the identical `SingleModeOperator` that
[`build_single_mode_operators`](@ref) constructs for that `m`.
"""
function build_single_mode_operator(problem::CoupledModeProblem{T}, m::Int) where T
    params_tri = problem.params
    basic_state_axis = axisymmetric_basic_state(params_tri.basic_state_3d)

    # Create OnsetParams for this mode
    params_m = OnsetParams(
        E = params_tri.E,
        Pr = params_tri.Pr,
        Ra = params_tri.Ra,
        χ = params_tri.χ,
        m = abs(m),  # m must be non-negative for OnsetParams
        lmax = params_tri.lmax,
        Nr = params_tri.Nr,
        mechanical_bc = params_tri.mechanical_bc,
        thermal_bc = params_tri.thermal_bc,
        equatorial_symmetry = params_tri.equatorial_symmetry,
        basic_state = basic_state_axis
    )

    # Create operator and assemble matrices
    op_m = LinearStabilityOperator(params_m)
    A_full, B_full, interior_dofs, boundary_dofs = assemble_matrices(op_m)
    A_m, B_m, reduction = _constrained_reduced_matrices(
        A_full, B_full, op_m, interior_dofs, boundary_dofs)
    expected_dofs = length(problem.block_indices[m])
    if reduction.n_reduced != expected_dofs
        error("Reduced DOF count mismatch for m=$m: got $(reduction.n_reduced), expected $expected_dofs")
    end
    idx_map = _full_index_map(op_m)
    if m < 0
        A_m = conj(A_m)
        B_m = conj(B_m)
    end

    return SingleModeOperator(
        A_m, B_m, op_m, idx_map, interior_dofs, boundary_dofs, reduction)
end


"""
    build_mode_coupling_operators(problem::CoupledModeProblem, single_mode_ops::Dict, verbose::Bool)

Build coupling operators between different azimuthal modes through the 3D basic state.

The coupling arises from:
1. **Advection of perturbation by basic state**: (ū_bs · ∇)θ'
   - Basic state flow ū with mode m_bs advects perturbation θ' with mode m_pert
   - Couples m_pert to m_pert ± m_bs through the φ-derivative: ∂/∂φ → im

2. **Perturbation advecting basic state temperature**: (u' · ∇)θ̄_bs
   - Perturbation velocity u' with mode m_pert advects basic state temperature θ̄ with mode m_bs
   - Couples m_pert to m_pert ± m_bs

3. **Shear production**: (u' · ∇)ū_bs
   - Perturbation velocity interacting with basic state velocity gradients

Returns a dictionary mapping (m_from, m_to) => C_{from,to} where C is the
coupling matrix from mode m_from to mode m_to.
"""
function build_mode_coupling_operators end  # Forward declaration

"""
    interpolate_to_grid(coeffs_bs::Vector{T}, r_bs::Vector{T}, r_op::Vector{T}) where T

Interpolate basic state coefficients from the basic state grid to the operator grid.
Uses simple linear interpolation.
"""
function interpolate_to_grid(coeffs_bs::Vector{T}, r_bs::Vector{T}, r_op::Vector{T}) where T
    Nr_op = length(r_op)
    coeffs_interp = zeros(T, Nr_op)

    for i in 1:Nr_op
        r_target = r_op[i]

        # Find bracketing points in r_bs
        if r_target <= r_bs[1]
            coeffs_interp[i] = coeffs_bs[1]
        elseif r_target >= r_bs[end]
            coeffs_interp[i] = coeffs_bs[end]
        else
            # Find j such that r_bs[j] <= r_target < r_bs[j+1]
            j = searchsortedlast(r_bs, r_target)
            if j >= length(r_bs)
                j = length(r_bs) - 1
            end
            # Linear interpolation
            t = (r_target - r_bs[j]) / (r_bs[j+1] - r_bs[j])
            coeffs_interp[i] = (1 - t) * coeffs_bs[j] + t * coeffs_bs[j+1]
        end
    end

    return coeffs_interp
end

"""Convert `UnitRange` entries in a `LinearStabilityOperator` index map to vectors."""
function _full_index_map(op::LinearStabilityOperator)
    idx_map = Dict{Tuple{Int,Symbol}, Vector{Int}}()
    for (key, idx_range) in op.index_map
        idx_map[key] = collect(idx_range)
    end

    return idx_map
end

"""Project a full coupling block into source constrained coordinates and target equations."""
function _project_coupling_block(C_full::Matrix{Complex{T}},
                                 target::SingleModeOperator{T},
                                 source::SingleModeOperator{T}) where {T<:Real}
    C = zeros(Complex{T}, length(target.interior_dofs), source.reduction.n_reduced)

    for block in source.reduction.blocks
        # View the source/target sub-block instead of materializing a copy of the
        # integer-indexed slice before the matmul.
        mul!(view(C, :, block.reduced_indices),
             @view(C_full[target.interior_dofs, block.full_indices]), block.basis)
    end

    return C
end

"""Build off-diagonal blocks using the same physical linearization as 2D."""
function build_mode_coupling_operators(problem::CoupledModeProblem{T},
        single_mode_ops::Dict{Int,SingleModeOperator{T}},verbose::Bool) where T
    coupling_ops=Dict{Tuple{Int,Int},Matrix{Complex{T}}}()
    bs=problem.params.basic_state_3d
    for m_from in problem.m_range, m_to in problem.coupling_graph[m_from]
        m_from==m_to && continue
        source=single_mode_ops[m_from]; target=single_mode_ops[m_to]
        C=_mean_state_matrix(bs,source.op,target.op,m_from,m_to)
        C=_project_coupling_block(C,target,source)
        any(!iszero,C) && (coupling_ops[(m_from,m_to)]=C)
    end
    verbose && println("  Built $(length(coupling_ops)) non-zero coupling blocks")
    coupling_ops
end


"""
    compute_sh_coupling_coefficient(ℓ1, m1, ℓ2, m2, ℓ3, m3)

Compute the spherical harmonic coupling coefficient for the product:
    Y_{ℓ1,m1} × Y_{ℓ2,m2} → Y_{ℓ3,m3}

This uses the Gaunt coefficient (integral of three spherical harmonics).
Selection rules require:
- m1 + m2 = m3 (azimuthal selection)
- |ℓ1 - ℓ2| ≤ ℓ3 ≤ ℓ1 + ℓ2 (triangle inequality)
- ℓ1 + ℓ2 + ℓ3 even (parity)
"""
function compute_sh_coupling_coefficient(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int, ℓ3::Int, m3::Int)
    # Selection rule: m1 + m2 = m3
    if m1 + m2 != m3
        return 0.0
    end

    # Triangle inequality
    if !(abs(ℓ1 - ℓ2) <= ℓ3 <= ℓ1 + ℓ2)
        return 0.0
    end

    # Parity selection
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    # Check m constraints
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end

    # Pure function of the indices, requested repeatedly for the same (ℓ,m)
    # combinations across coupling blocks; the two exact-rational Wigner-3j
    # evaluations dominate, so memoize past the (cheap) selection-rule exits.
    return get!(_SH_COUPLING_CACHE, (ℓ1, m1, ℓ2, m2, ℓ3, m3)) do
        # Compute Gaunt coefficient using simplified formula
        # For small ℓ values, use analytical approximation
        norm_factor = sqrt((2*ℓ1 + 1) * (2*ℓ2 + 1) * (2*ℓ3 + 1) / (4*π))

        # Wigner 3j symbol (0 0 0)
        w3j_000 = wigner3j_simple(ℓ1, ℓ2, ℓ3, 0, 0, 0)

        # Wigner 3j symbol (m1 m2 -m3)
        w3j_mmm = wigner3j_simple(ℓ1, ℓ2, ℓ3, m1, m2, -m3)

        phase = isodd(m3) ? -1.0 : 1.0
        phase * norm_factor * w3j_000 * w3j_mmm
    end
end

const _SH_COUPLING_CACHE = Dict{NTuple{6,Int}, Float64}()


"""
    wigner3j_simple(j1, j2, j3, m1, m2, m3)

Compute Wigner 3j symbol using WignerSymbols.jl package.

The Wigner 3j symbol is:
    (j1  j2  j3)
    (m1  m2  m3)

This is a wrapper around WignerSymbols.wigner3j for validated computation.
"""
function wigner3j_simple(j1::Int, j2::Int, j3::Int, m1::Int, m2::Int, m3::Int)
    # Use the validated WignerSymbols.jl package
    return Float64(WignerSymbols.wigner3j(j1, j2, j3, m1, m2, m3))
end


# =============================================================================
#  Unweighted Spherical Harmonic Coupling (for 1/sinθ terms)
# =============================================================================

"""
    compute_sh_coupling_unweighted(ℓ1, m1, ℓ2, m2, ℓ3, m3; n_quad=64)

Compute the UNWEIGHTED spherical harmonic coupling coefficient:

    I = ∫ Y*_{ℓ3,m3} Y_{ℓ1,m1} Y_{ℓ2,m2} dθ dφ

This differs from the standard Gaunt coefficient which uses sinθ dθ dφ (= dΩ).
The unweighted integral is needed for terms with 1/sinθ factors, such as
the azimuthal advection operator (ū_φ/(r sinθ)) ∂/∂φ.

**Mathematical basis:**
Using x = cosθ, we have dθ = dx/sinθ = dx/√(1-x²), so:

    ∫₀^π f(θ) dθ = ∫₋₁^{+1} f(arccos(x)) / √(1-x²) dx

This is a Chebyshev-weighted integral, computed using Gauss-Chebyshev quadrature.

Arguments:
- `ℓ1, m1, ℓ2, m2, ℓ3, m3` - Spherical harmonic indices
- `n_quad` - Number of quadrature points (default: 64)

Returns:
- Coupling coefficient (Float64)

Selection rules:
- m1 + m2 = m3 (azimuthal)
- ℓ1 + ℓ2 + ℓ3 even (parity)
"""
function compute_sh_coupling_unweighted(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int,
                                         ℓ3::Int, m3::Int; n_quad::Int=64)
    # Selection rule: m1 + m2 = m3
    if m1 + m2 != m3
        return 0.0
    end

    # Parity selection
    if (ℓ1 + ℓ2 + ℓ3) % 2 != 0
        return 0.0
    end

    # Check m constraints
    if abs(m1) > ℓ1 || abs(m2) > ℓ2 || abs(m3) > ℓ3
        return 0.0
    end

    # Pure function of indices + n_quad; the 64-point quadrature with per-node
    # associated-Legendre evaluations is expensive, so memoize like the Gaunt path.
    return get!(_SH_COUPLING_UNWEIGHTED_CACHE, (ℓ1, m1, ℓ2, m2, ℓ3, m3, n_quad)) do
        _sh_coupling_unweighted_quadrature(ℓ1, m1, ℓ2, m2, ℓ3, m3, n_quad)
    end
end

const _SH_COUPLING_UNWEIGHTED_CACHE = Dict{NTuple{7,Int}, Float64}()

function _sh_coupling_unweighted_quadrature(ℓ1::Int, m1::Int, ℓ2::Int, m2::Int,
                                            ℓ3::Int, m3::Int, n_quad::Int)
    # Use Gauss-Chebyshev quadrature of the first kind
    # Nodes: x_i = cos((2i-1)π/(2n))
    # Weights: w_i = π/n
    weight = π / n_quad

    # Compute the integral using quadrature
    # ∫ Y*₃ Y₁ Y₂ dθ = ∫ Y*₃ Y₁ Y₂ dx/√(1-x²)
    # where x = cosθ

    integral = 0.0

    for i in 1:n_quad
        x = cos((2*i - 1) * π / (2 * n_quad))

        # Compute normalized associated Legendre functions at x
        # Y_ℓm = N_ℓm P_ℓ^m(x) e^{imφ}
        # The φ integral gives 2π δ_{m1+m2, m3}, already checked above

        P1 = _normalized_associated_legendre(ℓ1, abs(m1), x)
        P2 = _normalized_associated_legendre(ℓ2, abs(m2), x)
        P3 = _normalized_associated_legendre(ℓ3, abs(m3), x)

        # Apply Condon-Shortley phase for negative m
        if m1 < 0 && isodd(abs(m1))
            P1 = -P1
        end
        if m2 < 0 && isodd(abs(m2))
            P2 = -P2
        end
        if m3 < 0 && isodd(abs(m3))
            P3 = -P3
        end

        # No extra (-1)^m3 phase here: conjugating Y₃ flips only the e^{imφ}
        # factor (handled by the m1+m2=m3 selection rule), leaving its real
        # θ-part P̄₃ unchanged. The Condon-Shortley sign is already inside
        # _normalized_associated_legendre, so the integrand is simply P̄₃P̄₁P̄₂.
        integral += weight * P3 * P1 * P2
    end

    # Multiply by 2π from the φ integral (δ function gives 2π, not 1)
    # Actually, the normalization of Y_ℓm already includes the 1/√(2π) factor
    # so we just need to account for the integral of e^{i(m1+m2-m3)φ} = 2π δ
    integral *= 2.0 * π

    return integral
end

"""
    _normalized_associated_legendre(ℓ, m, x)

Compute the normalized associated Legendre function:

    P̃_ℓ^m(x) = √[(2ℓ+1)/(4π) × (ℓ-m)!/(ℓ+m)!] × P_ℓ^m(x)

This is the normalization used in fully normalized spherical harmonics:
    Y_ℓm(θ,φ) = P̃_ℓ^m(cosθ) × e^{imφ}

Uses the stable recurrence relation to avoid numerical overflow.
"""
function _normalized_associated_legendre(ℓ::Int, m::Int, x::Float64)
    if m > ℓ
        return 0.0
    end

    # Start with P_m^m using the formula:
    # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
    # With normalization factor

    if m == 0
        # Use standard Legendre polynomial recurrence
        if ℓ == 0
            return sqrt(1.0 / (4.0 * π))
        elseif ℓ == 1
            return sqrt(3.0 / (4.0 * π)) * x
        else
            # Three-term recurrence for normalized Legendre
            P_prev2 = sqrt(1.0 / (4.0 * π))  # P̃_0
            P_prev1 = sqrt(3.0 / (4.0 * π)) * x  # P̃_1

            for l in 2:ℓ
                # Recurrence: P̃_l = a_l x P̃_{l-1} - b_l P̃_{l-2}
                a_l = sqrt((4.0*l^2 - 1.0) / (l^2 - m^2))
                b_l = sqrt(((l-1)^2 - m^2) / (4.0*(l-1)^2 - 1.0)) *
                      sqrt((4.0*l^2 - 1.0) / (l^2 - m^2))

                P_curr = a_l * x * P_prev1 - b_l * P_prev2
                P_prev2 = P_prev1
                P_prev1 = P_curr
            end

            return P_prev1
        end
    else
        # For m > 0, use the sectoral (P_m^m) start and recurrence

        # Compute P̃_m^m
        # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^(m/2)
        # Normalization: √[(2m+1)/(4π) × 1/(2m)!] for the (2m-1)!! factor

        sin_theta = sqrt(1.0 - x^2)

        # Start with normalized P̃_m^m
        # Using stable computation
        P_mm = 1.0 / sqrt(4.0 * π)
        for i in 1:m
            P_mm *= -sqrt((2.0*i + 1.0) / (2.0*i)) * sin_theta
        end

        if ℓ == m
            return P_mm
        end

        # Compute P̃_{m+1}^m using P̃_m^m
        # P_{m+1}^m = x(2m+1) P_m^m
        # With normalization adjustment
        P_prev1 = P_mm
        a_mp1 = sqrt(2.0*m + 3.0) * x
        P_curr = a_mp1 * P_mm

        if ℓ == m + 1
            return P_curr
        end

        # Three-term recurrence for ℓ > m+1
        P_prev2 = P_mm
        P_prev1 = P_curr

        for l in (m+2):ℓ
            # Normalized recurrence coefficients
            a_l = x * sqrt((4.0*l^2 - 1.0) / (l^2 - m^2))
            b_l = sqrt(((l-1)^2 - m^2) * (4.0*l^2 - 1.0) /
                       ((l^2 - m^2) * (4.0*(l-1)^2 - 1.0)))

            P_curr = a_l * P_prev1 - b_l * P_prev2
            P_prev2 = P_prev1
            P_prev1 = P_curr
        end

        return P_prev1
    end
end


"""
    assemble_block_matrices(problem, single_mode_ops, coupling_ops, verbose)

Assemble the full block-coupled matrices A_coupled and B_coupled.

The structure is:
    ┌                         ┐
    │ A_{m1}   C_{12}    0    │
    │ C_{21}   A_{m2}  C_{23} │
    │ 0        C_{32}  A_{m3} │
    └                         ┘

where A_{mi} are single-mode operators and C_{ij} are coupling operators.
"""
function _assemble_block_coo(problem::CoupledModeProblem{T},
                             single_mode_ops::Dict{Int,SingleModeOperator{T}},
                             coupling_ops::Dict{Tuple{Int,Int},Matrix{Complex{T}}};
                             owned_julia_rows=nothing) where T
    n_total = problem.total_dofs
    row_A = Int[]
    col_A = Int[]
    val_A = Complex{T}[]
    row_B = Int[]
    col_B = Int[]
    val_B = Complex{T}[]
    tol = T(1e-14)

    owns(rng) = owned_julia_rows === nothing || !isempty(intersect(rng, owned_julia_rows))

    # Fill in diagonal blocks (single-mode operators)
    for m in problem.m_range
        block_range = problem.block_indices[m]
        owns(block_range) || continue
        _append_block_entries!(row_A, col_A, val_A, single_mode_ops[m].A,
                               block_range, block_range, tol; owned=owned_julia_rows)
        _append_block_entries!(row_B, col_B, val_B, single_mode_ops[m].B,
                               block_range, block_range, tol; owned=owned_julia_rows)
    end

    # Fill in off-diagonal blocks (coupling operators)
    for ((m_from, m_to), C) in coupling_ops
        isempty(C) && continue
        range_to = problem.block_indices[m_to]
        range_from = problem.block_indices[m_from]
        owns(range_to) || continue
        _append_block_entries!(row_A, col_A, val_A, C, range_to, range_from, tol;
                               owned=owned_julia_rows)
    end

    return (A_rows=row_A, A_cols=col_A, A_vals=val_A,
            B_rows=row_B, B_cols=col_B, B_vals=val_B, n=n_total)
end

function assemble_block_matrices(problem::CoupledModeProblem{T},
                                  single_mode_ops::Dict{Int,SingleModeOperator{T}},
                                  coupling_ops::Dict{Tuple{Int,Int},Matrix{Complex{T}}},
                                  verbose::Bool) where T
    c = _assemble_block_coo(problem, single_mode_ops, coupling_ops)
    A_coupled = sparse(c.A_rows, c.A_cols, c.A_vals, c.n, c.n)
    B_coupled = sparse(c.B_rows, c.B_cols, c.B_vals, c.n, c.n)

    if verbose
        println("  Matrix size: $(c.n) × $(c.n)")
        println("  nnz(A): $(nnz(A_coupled))  nnz(B): $(nnz(B_coupled))")
    end

    return A_coupled, B_coupled
end

"""Append nonzero entries from a dense block into global sparse COO vectors."""
function _append_block_entries!(row_idx::Vector{Int},
                                 col_idx::Vector{Int},
                                 val_idx::Vector{Complex{T}},
                                 block::AbstractMatrix{Complex{T}},
                                 rows::UnitRange{Int},
                                 cols::UnitRange{Int},
                                 tol::Real;
                                 owned::Union{Nothing,UnitRange{Int}}=nothing) where {T<:Real}
    for (local_i, global_i) in enumerate(rows)
        (owned === nothing || global_i in owned) || continue
        for (local_j, global_j) in enumerate(cols)
            val = block[local_i, local_j]
            if abs(val) > tol
                push!(row_idx, global_i)
                push!(col_idx, global_j)
                push!(val_idx, val)
            end
        end
    end
end


# =============================================================================
#  Main Solver Functions
# =============================================================================

"""
    solve_triglobal_eigenvalue_problem(params::TriglobalParams{T};
                                       σ_target=0.0, nev=6, verbose=true) where T

Solve the tri-global eigenvalue problem to find growth rates and eigenmodes.

This solves the block-coupled eigenvalue problem:
    A_coupled x = λ B_coupled x

where different azimuthal modes m couple through the non-axisymmetric basic state.

Arguments:
- `params` - Tri-global parameters
- `σ_target` - Target growth rate for shift-invert (default: 0.0)
- `nev` - Number of eigenvalues to compute (default: 6)
- `verbose` - Print progress information (default: true)

Returns:
- `eigenvalues` - Complex growth rates λ = σ + iω (sorted by real part, descending)
- `eigenvectors` - Corresponding eigenmodes (columns of matrix)
"""
function solve_triglobal_eigenvalue_problem(params::TriglobalParams{T};
                                            σ_target=0.0, nev=6, verbose=true,
                                            backend::Symbol=:slepc) where T
    # Setup problem structure
    problem = setup_coupled_mode_problem(params)

    if verbose
        println("="^70)
        println("Tri-Global Eigenvalue Problem")
        println("="^70)
        println("  Mode range:        ", problem.m_range)
        println("  Basic state modes: ", problem.all_m_bs)
        println("  Total DOFs:        ", problem.total_dofs)
        println("  Target eigenvalues:", nev)
        println()
    end

    # Step 1: Build single-mode operators for each m
    if verbose
        println("Building single-mode operators for each m...")
    end

    single_mode_ops = build_single_mode_operators(problem, verbose)

    # Step 2: Build coupling operators between modes
    if verbose
        println("\nBuilding mode coupling operators...")
    end

    coupling_ops = build_mode_coupling_operators(problem, single_mode_ops, verbose)

    # Step 4: Solve eigenvalue problem
    if verbose
        println("\nSolving eigenvalue problem (shift-invert, σ=$σ_target)...")
    end

    backend === :slepc || throw(ArgumentError(
        "Unknown eigensolver backend $(backend); only :slepc is supported"))

    # Distributed triglobal path: the SLEPc extension assembles the block-coupled
    # pencil directly into distributed PETSc Mats (owned rows only) from
    # `_assemble_block_coo`, so we never form the dense replicated A/B here.
    eigenvalues, eigenvectors = _solve_triglobal_slepc(problem;
        σ_target=σ_target, nev=nev, tol=T(1e-8), maxiter=200)

    if verbose
        println("\n" * "="^70)
        println("Eigenvalue Results:")
        println("="^70)
        for (i, λ) in enumerate(eigenvalues[1:min(nev, length(eigenvalues))])
            σ = real(λ)
            ω = imag(λ)
            println(@sprintf("  %2d: σ = %+.6e, ω = %+.6e", i, σ, ω))
        end
        println()
    end

    return eigenvalues, eigenvectors
end


"""
    find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                     basic_state_3d;
                                     Ra_min=1e5, Ra_max=1e8,
                                     tol=1e-4, max_iter=20,
                                     mechanical_bc=:no_slip,
                                     thermal_bc=:fixed_temperature,
                                     equatorial_symmetry=:both,
                                     verbose=true)

Find critical Rayleigh number for onset on a 3D basic state (tri-global analysis).

Uses bisection to find Ra_c where the leading growth rate σ = 0.

Arguments:
- `E` - Ekman number
- `Pr` - Prandtl number
- `χ` - Radius ratio
- `m_range` - Range of perturbation modes (e.g., -2:2)
- `lmax` - Maximum spherical harmonic degree
- `Nr` - Number of radial points
- `basic_state_3d` - The 3D basic state (BasicState3D)
- `Ra_min` - Lower bound for Ra search (default: 1e5)
- `Ra_max` - Upper bound for Ra search (default: 1e8)
- `tol` - Tolerance for bisection (default: 1e-4)
- `max_iter` - Maximum iterations (default: 20)
- `mechanical_bc` - Boundary conditions (default: :no_slip)
- `thermal_bc` - Thermal boundary conditions (default: :fixed_temperature)
- `equatorial_symmetry` - :both, :symmetric, or :antisymmetric
- `verbose` - Print progress (default: true)

Returns:
- `Ra_c` - Critical Rayleigh number
- `σ_c` - Growth rate at Ra_c (should be ≈ 0)
- `ω_c` - Drift frequency at Ra_c
"""
function find_critical_rayleigh_triglobal(E, Pr, χ, m_range, lmax, Nr,
                                          basic_state_3d;
                                          Ra_min=1e5, Ra_max=1e8,
                                          tol=1e-4, max_iter=20,
                                          mechanical_bc=:no_slip,
                                          thermal_bc=:fixed_temperature,
                                          equatorial_symmetry=:both,
                                          verbose=true)
    if verbose
        println("="^70)
        println("Finding Critical Rayleigh Number (Tri-Global)")
        println("="^70)
        println("  E           = ", @sprintf("%.2e", E))
        println("  Pr          = ", Pr)
        println("  χ           = ", χ)
        println("  m_range     = ", m_range)
        println("  lmax        = ", lmax)
        println("  Nr          = ", Nr)
        println("  Tolerance   = ", tol)
        println("  Max iter    = ", max_iter)
        println()
    end

    # Bisection algorithm
    Ra_low = Ra_min
    Ra_high = Ra_max

    # Test bounds
    if verbose
        println("Testing bounds...")
    end

    params_low = TriglobalParams(
        E=E, Pr=Pr, Ra=Ra_low, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        equatorial_symmetry=equatorial_symmetry
    )
    vals_low, _ = solve_triglobal_eigenvalue_problem(params_low; nev=3, verbose=false)
    σ_low = real(vals_low[1])

    params_high = TriglobalParams(
        E=E, Pr=Pr, Ra=Ra_high, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        equatorial_symmetry=equatorial_symmetry
    )
    vals_high, _ = solve_triglobal_eigenvalue_problem(params_high; nev=3, verbose=false)
    σ_high = real(vals_high[1])

    if verbose
        println("  Ra = $(Ra_low):  σ = $(σ_low)")
        println("  Ra = $(Ra_high): σ = $(σ_high)")
        println()
    end

    if σ_low > 0
        @warn "Lower bound Ra=$Ra_low is already unstable (σ=$σ_low > 0)"
        if verbose
            println("  Returning lower bound as estimate.")
        end
        return Ra_low, σ_low, imag(vals_low[1])
    end

    if σ_high < 0
        @warn "Upper bound Ra=$Ra_high is still stable (σ=$σ_high < 0)"
        if verbose
            println("  Returning upper bound as estimate.")
        end
        return Ra_high, σ_high, imag(vals_high[1])
    end

    # Bisection loop
    if verbose
        println("Starting bisection...")
        println(@sprintf("  %-4s  %-12s  %-12s  %-12s", "Iter", "Ra", "σ", "Δ Ra"))
        println("  " * "-"^45)
    end

    for iter in 1:max_iter
        Ra_mid = 0.5 * (Ra_low + Ra_high)

        params_mid = TriglobalParams(
            E=E, Pr=Pr, Ra=Ra_mid, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
            basic_state_3d=basic_state_3d,
            mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
            equatorial_symmetry=equatorial_symmetry
        )
        vals_mid, _ = solve_triglobal_eigenvalue_problem(params_mid; nev=3, verbose=false)
        σ_mid = real(vals_mid[1])
        ω_mid = imag(vals_mid[1])

        Delta_Ra = Ra_high - Ra_low

        if verbose
            println(@sprintf("  %-4d  %-12.6e  %+-.6e  %-12.6e",
                           iter, Ra_mid, σ_mid, Delta_Ra))
        end

        # Check convergence
        if abs(σ_mid) < tol * abs(σ_low) || Delta_Ra < tol * Ra_mid
            if verbose
                println()
                println("  Converged!")
                println("  Ra_c = ", @sprintf("%.6e", Ra_mid))
                println("  σ_c  = ", @sprintf("%+.6e", σ_mid))
                println("  ω_c  = ", @sprintf("%+.6e", ω_mid))
            end
            return Ra_mid, σ_mid, ω_mid
        end

        # Update bounds
        if σ_mid > 0
            Ra_high = Ra_mid
        else
            Ra_low = Ra_mid
        end
    end

    # Max iterations reached
    Ra_mid = 0.5 * (Ra_low + Ra_high)
    params_mid = TriglobalParams(
        E=E, Pr=Pr, Ra=Ra_mid, χ=χ, m_range=m_range, lmax=lmax, Nr=Nr,
        basic_state_3d=basic_state_3d,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        equatorial_symmetry=equatorial_symmetry
    )
    vals_mid, _ = solve_triglobal_eigenvalue_problem(params_mid; nev=3, verbose=false)
    σ_mid = real(vals_mid[1])
    ω_mid = imag(vals_mid[1])

    @warn "Maximum iterations ($max_iter) reached without full convergence"
    if verbose
        println("  Returning best estimate:")
        println("  Ra_c = ", @sprintf("%.6e", Ra_mid))
        println("  σ_c  = ", @sprintf("%+.6e", σ_mid))
    end

    return Ra_mid, σ_mid, ω_mid
end
