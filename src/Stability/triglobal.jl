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
function _nonzero_basic_state_modes_3d(basic_state::BasicState3D; tol=1e-14)
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
    # coupling coefficient (see _basic_state_complex_profile). Listing both signs here
    # would double-iterate (and double-count) the same physical mode.
    for coefficient_dict in coefficient_dicts
        for ((ℓ, m_bs), coeff) in coefficient_dict
            if m_bs != 0 && _maxabs(coeff) > tol
                push!(modes, (ℓ, abs(m_bs)))
            end
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
    lmax_bs = basic_state.lmax_bs
    Nr = basic_state.Nr
    theta_coeffs = Dict{Int, Vector{T}}()
    uphi_coeffs = Dict{Int, Vector{T}}()
    dtheta_dr_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()
    zero_coeff = zeros(T, Nr)

    for ℓ in 0:lmax_bs
        theta_coeffs[ℓ] = get(basic_state.theta_coeffs, (ℓ, 0), zero_coeff)
        uphi_coeffs[ℓ] = get(basic_state.uphi_coeffs, (ℓ, 0), zero_coeff)
        dtheta_dr_coeffs[ℓ] = get(basic_state.dtheta_dr_coeffs, (ℓ, 0), zero_coeff)
        duphi_dr_coeffs[ℓ] = get(basic_state.duphi_dr_coeffs, (ℓ, 0), zero_coeff)
    end

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = basic_state.r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end

"""Return true when an axisymmetric basic state has any active temperature or flow."""
function _has_nonzero_basic_state(basic_state::BasicState{T}; tol=1e-14) where T
    for coeff in values(basic_state.theta_coeffs)
        _maxabs(coeff) > tol && return true
    end
    for coeff in values(basic_state.uphi_coeffs)
        _maxabs(coeff) > tol && return true
    end
    return false
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
    has_axisymmetric = _has_nonzero_basic_state(basic_state_axis)

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
            basic_state = has_axisymmetric ? basic_state_axis : nothing
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
    has_axisymmetric = _has_nonzero_basic_state(basic_state_axis)

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
        basic_state = has_axisymmetric ? basic_state_axis : nothing
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

"""Scale stored real basic-state modes to the complex spherical-harmonic convention."""
function _basic_state_mode_scale(m_bs::Int, ::Type{T}) where {T<:Real}
    if m_bs == 0
        return one(T)
    end
    scale = inv(sqrt(T(2)))
    phase = (m_bs < 0 && isodd(abs(m_bs))) ? -one(T) : one(T)
    return phase * scale
end

"""
    _basic_state_complex_profile(coeffs, ℓ_bs, m_bs_eff) -> Vector{Complex{T}} | nothing

Complex spherical-harmonic radial coefficient `ĉ_{ℓ_bs, m_bs_eff}(r)` of a *real*
basic-state field whose real-SH amplitudes are stored as cosine part at key
`(ℓ_bs, +|m|)` (call it `A`) and sine part at key `(ℓ_bs, -|m|)` (call it `B`).

The real→complex map for a real field is `ĉ_{ℓ,+m} = (A - iB)/√2`,
`ĉ_{ℓ,-m} = (-1)^m (A + iB)/√2 = (-1)^m conj(ĉ_{ℓ,+m})`, and `ĉ_{ℓ,0} = A`.

This is the single hinge where the real basic state enters the complex-SH Gaunt
coupling. When the sine part `B` is absent (the historical cosine-only thermal-wind
basic state) it reduces *exactly* to `_basic_state_mode_scale(m_bs_eff, T) .* A`, so
the validated axisymmetric/cosine coupling path is bit-identical. Returns `nothing`
when neither key carries data.
"""
function _basic_state_complex_profile(coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                      ℓ_bs::Int, m_bs_eff::Int) where {T<:Real}
    am = abs(m_bs_eff)
    if am == 0
        c = get(coeffs, (ℓ_bs, 0), nothing)
        c === nothing && return nothing
        return Complex{T}.(c)
    end
    A = get(coeffs, (ℓ_bs,  am), nothing)   # cosine part
    B = get(coeffs, (ℓ_bs, -am), nothing)   # sine part
    (A === nothing && B === nothing) && return nothing
    Nr = A === nothing ? length(B) : length(A)
    Av = A === nothing ? zeros(T, Nr) : A
    Bv = B === nothing ? zeros(T, Nr) : B
    invsqrt2 = inv(sqrt(T(2)))
    if m_bs_eff > 0
        return (Av .- im .* Bv) .* invsqrt2
    else
        phase = isodd(am) ? -one(T) : one(T)
        return (phase .* invsqrt2) .* (Av .+ im .* Bv)
    end
end

# Note: _theta_derivative_coeff is defined in basic_state_operators.jl

"""Angular coupling for a basic-state meridional derivative acting on perturbations."""
function _meridional_coupling(l_input::Int, l_bs::Int, l_output::Int,
                              m_from::Int, m_bs::Int, m_to::Int)
    c_plus, c_minus = _theta_derivative_coeff(l_bs, abs(m_bs))
    coupling = 0.0

    if abs(c_plus) > 1e-14
        l_temp = l_bs + 1
        coupling += c_plus * compute_sh_coupling_coefficient(
            l_input, m_from, l_temp, m_bs, l_output, m_to
        )
    end
    if abs(c_minus) > 1e-14 && l_bs > 0
        l_temp = l_bs - 1
        coupling += c_minus * compute_sh_coupling_coefficient(
            l_input, m_from, l_temp, m_bs, l_output, m_to
        )
    end

    return coupling
end

"""Angular coupling for a perturbation meridional derivative acting on the basic state."""
function _perturbation_meridional_coupling(l_pert::Int, l_bs::Int, l_output::Int,
                                           m_from::Int, m_bs::Int, m_to::Int)
    c_plus, c_minus = _theta_derivative_coeff(l_pert, abs(m_from))
    coupling = 0.0

    if abs(c_plus) > 1e-14
        l_temp = l_pert + 1
        coupling += c_plus * compute_sh_coupling_coefficient(
            l_temp, m_from, l_bs, m_bs, l_output, m_to
        )
    end
    if abs(c_minus) > 1e-14 && l_pert > 0
        l_temp = l_pert - 1
        coupling += c_minus * compute_sh_coupling_coefficient(
            l_temp, m_from, l_bs, m_bs, l_output, m_to
        )
    end

    return coupling
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

"""
    _accumulate_mode_couplings!(C, op_from, op_to, ...)

Function barrier for the per-(m_from, m_to) coupling accumulation. `op_from`/
`op_to` are stored in `SingleModeOperator.op`, whose declared field type
`LinearStabilityOperator{T}` is abstract (the struct is
`LinearStabilityOperator{T,BS}`, so the `BS` parameter is unbound). Passing them
through this barrier lets Julia specialize on the concrete runtime operator
type, so the 22 `add_*_coupling!` calls below dispatch statically instead of at
runtime. Pure code motion — behavior is identical to the inlined loop.
"""
function _accumulate_mode_couplings!(C::Matrix{Complex{T}},
        op_from::LinearStabilityOperator, op_to::LinearStabilityOperator,
        idx_map_from, idx_map_to, m_from::Int, m_to::Int, bs_modes,
        r::AbstractVector, D1,
        theta_interp, dtheta_dr_interp, uphi_interp, duphi_dr_interp,
        ur_interp, dur_dr_interp, utheta_interp, dutheta_dr_interp,
        params) where {T}
    # Compute coupling through each relevant basic state mode
    for (ℓ_bs, m_bs) in bs_modes
        # The spherical harmonic selection rule requires m_from + m_bs_eff = m_to
        # where m_bs_eff = ±m_bs depending on the coupling direction
        if m_to == m_from + m_bs
            # Forward coupling: m_from + m_bs = m_to
            m_bs_eff = m_bs
            add_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                   m_from, m_to, ℓ_bs, m_bs_eff,
                                   r, uphi_interp, params)
            add_metric_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                 m_from, m_to, ℓ_bs, m_bs_eff,
                                 r, uphi_interp, params)
            add_radial_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, ur_interp, D1, params)
            add_meridional_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                m_from, m_to, ℓ_bs, m_bs_eff,
                                                r, utheta_interp, params)
            add_temperature_gradient_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                               m_from, m_to, ℓ_bs, m_bs_eff,
                                               r, dtheta_dr_interp, params)
            add_shear_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                               m_from, m_to, ℓ_bs, m_bs_eff,
                               r, duphi_dr_interp, params)
            add_radial_velocity_shear!(C, op_from, op_to, idx_map_from, idx_map_to,
                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                       r, dur_dr_interp, params)
            add_meridional_velocity_shear!(C, op_from, op_to, idx_map_from, idx_map_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, dutheta_dr_interp, params)
            add_temperature_gradient_theta_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                      m_from, m_to, ℓ_bs, m_bs_eff,
                                                      r, theta_interp, D1, params)
            add_temperature_gradient_phi_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                    m_from, m_to, ℓ_bs, m_bs_eff,
                                                    r, theta_interp, params)
            add_shear_theta_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                      m_from, m_to, ℓ_bs, m_bs_eff,
                                      r, uphi_interp, D1, params)
        elseif m_to == m_from - m_bs
            # Backward coupling: m_from - m_bs = m_to → m_from + (-m_bs) = m_to
            # Need to use -m_bs in the Gaunt coefficient (complex conjugate of Y_{ℓ,m})
            m_bs_eff = -m_bs
            add_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                   m_from, m_to, ℓ_bs, m_bs_eff,
                                   r, uphi_interp, params)
            add_metric_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                 m_from, m_to, ℓ_bs, m_bs_eff,
                                 r, uphi_interp, params)
            add_radial_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, ur_interp, D1, params)
            add_meridional_advection_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                m_from, m_to, ℓ_bs, m_bs_eff,
                                                r, utheta_interp, params)
            add_temperature_gradient_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                               m_from, m_to, ℓ_bs, m_bs_eff,
                                               r, dtheta_dr_interp, params)
            add_shear_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                               m_from, m_to, ℓ_bs, m_bs_eff,
                               r, duphi_dr_interp, params)
            add_radial_velocity_shear!(C, op_from, op_to, idx_map_from, idx_map_to,
                                       m_from, m_to, ℓ_bs, m_bs_eff,
                                       r, dur_dr_interp, params)
            add_meridional_velocity_shear!(C, op_from, op_to, idx_map_from, idx_map_to,
                                           m_from, m_to, ℓ_bs, m_bs_eff,
                                           r, dutheta_dr_interp, params)
            add_temperature_gradient_theta_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                      m_from, m_to, ℓ_bs, m_bs_eff,
                                                      r, theta_interp, D1, params)
            add_temperature_gradient_phi_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                                    m_from, m_to, ℓ_bs, m_bs_eff,
                                                    r, theta_interp, params)
            add_shear_theta_coupling!(C, op_from, op_to, idx_map_from, idx_map_to,
                                      m_from, m_to, ℓ_bs, m_bs_eff,
                                      r, uphi_interp, D1, params)
        end
    end
    return C
end

"""Build all off-diagonal mode-coupling blocks induced by the 3D basic state."""
function build_mode_coupling_operators(problem::CoupledModeProblem{T},
                                        single_mode_ops::Dict{Int,SingleModeOperator{T}},
                                        verbose::Bool) where T
    coupling_ops = Dict{Tuple{Int,Int}, Matrix{Complex{T}}}()
    params = problem.params
    basic_state = params.basic_state_3d

    # Get radial grid from one of the single-mode operators
    first_m = first(problem.m_range)
    op_ref = single_mode_ops[first_m].op
    r = op_ref.r
    cd = op_ref.cd

    # Get the basic state radial grid for interpolation
    r_bs = basic_state.r

    # Pre-interpolate basic state coefficients to operator grid if grids differ
    needs_interpolation = (length(r_bs) != length(r)) || (_maxabsdiff(r_bs, r) > 1e-10)

    # Cache interpolated coefficients
    theta_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    dtheta_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    uphi_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    duphi_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    ur_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    utheta_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    dur_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()
    dutheta_dr_interp = Dict{Tuple{Int,Int}, Vector{T}}()

    for (key, coeff) in basic_state.theta_coeffs
        if needs_interpolation
            theta_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            theta_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.dtheta_dr_coeffs
        if needs_interpolation
            dtheta_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            dtheta_dr_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.uphi_coeffs
        if needs_interpolation
            uphi_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            uphi_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.duphi_dr_coeffs
        if needs_interpolation
            duphi_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            duphi_dr_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.ur_coeffs
        if needs_interpolation
            ur_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            ur_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.utheta_coeffs
        if needs_interpolation
            utheta_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            utheta_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.dur_dr_coeffs
        if needs_interpolation
            dur_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            dur_dr_interp[key] = coeff
        end
    end
    for (key, coeff) in basic_state.dutheta_dr_coeffs
        if needs_interpolation
            dutheta_dr_interp[key] = interpolate_to_grid(coeff, r_bs, r)
        else
            dutheta_dr_interp[key] = coeff
        end
    end

    # Find all non-zero basic state modes (ℓ_bs, m_bs)
    bs_modes = _nonzero_basic_state_modes_3d(basic_state)

    if verbose
        println("  Non-zero basic state modes: ", bs_modes)
    end

    # For each mode pair that should couple
    n_nonzero_couplings = 0

    for m_from in problem.m_range
        for m_to in problem.m_range
            if m_from == m_to
                continue  # Diagonal is handled separately
            end

            # Check if these modes can couple through any basic state mode
            Δm = m_to - m_from
            can_couple = false
            for (ℓ_bs, m_bs) in bs_modes
                if abs(Δm) == m_bs
                    can_couple = true
                    break
                end
            end

            if !can_couple
                continue
            end

            # Get sizes of the from and to blocks
            op_from = single_mode_ops[m_from].op
            op_to = single_mode_ops[m_to].op
            idx_map_from = single_mode_ops[m_from].idx_map
            idx_map_to = single_mode_ops[m_to].idx_map
            n_from = op_from.total_dof
            n_to = op_to.total_dof

            # Build the coupling matrix in full coordinates, then project the
            # source columns into the same constraint basis used by the diagonal
            # blocks while keeping target interior equations. NB: a Dict/COO
            # sparse accumulator was tried and measured SLOWER+heavier here — the
            # Dr-block terms fill dense Nr×Nr radial sub-blocks, so this block is
            # not sparse and the contiguous dense array is the right structure.
            C = zeros(Complex{T}, n_to, n_from)

            # Accumulate coupling through each relevant basic state mode. The
            # function barrier specializes on the concrete operator type (the
            # SingleModeOperator.op field is abstractly typed) so the inner
            # add_*_coupling! calls dispatch statically.
            _accumulate_mode_couplings!(C, op_from, op_to, idx_map_from, idx_map_to,
                                        m_from, m_to, bs_modes, r, cd.D1,
                                        theta_interp, dtheta_dr_interp, uphi_interp, duphi_dr_interp,
                                        ur_interp, dur_dr_interp, utheta_interp, dutheta_dr_interp,
                                        params)

            C = _project_coupling_block(C, single_mode_ops[m_to], single_mode_ops[m_from])

            # Store if non-zero
            if _maxabs(C) > 1e-16
                coupling_ops[(m_from, m_to)] = C
                n_nonzero_couplings += 1
            end
        end
    end

    if verbose
        println("  Built $n_nonzero_couplings non-zero coupling blocks")
    end

    return coupling_ops
end


"""
    add_advection_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, uphi_coeffs, params)

Add advection coupling: (ū_bs · ∇)θ' contribution to the temperature equation.

The azimuthal advection term is:
    (ū_φ,bs / (r sin θ)) ∂θ'/∂φ = im_from × ū_φ,bs/(r sin θ) × θ'

This couples perturbation mode m_from to m_to = m_from ± m_bs through the
product of spherical harmonics.

**Implementation Note:**
The 1/sinθ factor in the advection operator requires computing the unweighted
integral ∫ Y*₃ Y₁ Y₂ dθdφ (without sinθ weighting), which differs from the
standard Gaunt coefficient G = ∫ Y*₃ Y₁ Y₂ sinθ dθdφ. This implementation uses
`compute_sh_coupling_unweighted()` which computes the correct integral using
Gauss-Chebyshev quadrature, providing accurate results even near the poles.

Arguments:
- uphi_coeffs: Dictionary of interpolated uphi coefficients on the operator grid
"""
function add_advection_coupling!(C::Matrix{Complex{T}},
                                  op_from, op_to,
                                  idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                  idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                  m_from::Int, m_to::Int,
                                  ℓ_bs::Int, m_bs::Int,
                                  r::Vector{T},
                                  uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                  params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Basic-state zonal flow as complex SH coefficient (cos+sin), already interpolated.
    uphi_bs = _basic_state_complex_profile(uphi_coeffs, ℓ_bs, m_bs)
    (uphi_bs === nothing || maximum(abs, uphi_bs) < 1e-14) && return

    # Advection coefficient: im_from × ū_φ/r; the 1/sinθ factor is handled by the
    # unweighted coupling coefficient.
    adv_coeff = im * m_from .* uphi_bs ./ r

    for field in (:P, :T, :Θ)
        for (ℓ_from, field_from) in keys(op_from.index_map)
            if field_from != field || ℓ_from < m_pert_from
                continue
            end

            for (ℓ_to, field_to) in keys(op_to.index_map)
                if field_to != field || ℓ_to < m_pert_to
                    continue
                end

                # Compute UNWEIGHTED spherical harmonic coupling coefficient.
                # This correctly handles the 1/sinθ factor in the advection operator.
                coupling_coeff = compute_sh_coupling_unweighted(
                    ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
                )

                if abs(coupling_coeff) < 1e-14
                    continue
                end

                idx_from = idx_map_from[(ℓ_from, field)]
                idx_to = idx_map_to[(ℓ_to, field)]

                for i in 1:Nr
                    row = idx_to[i]
                    col = idx_from[i]
                    (row == 0 || col == 0) && continue
                    C[row, col] += coupling_coeff * adv_coeff[i]
                end
            end
        end
    end
end

"""Add curvilinear metric coupling from zonal basic flow into the poloidal equation."""
function add_metric_coupling!(C::Matrix{Complex{T}},
                              op_from, op_to,
                              idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                              idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                              m_from::Int, m_to::Int,
                              ℓ_bs::Int, m_bs::Int,
                              r::Vector{T},
                              uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                              params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    uphi_bs = _basic_state_complex_profile(uphi_coeffs, ℓ_bs, m_bs)
    (uphi_bs === nothing || maximum(abs, uphi_bs) < 1e-14) && return

    metric_profile = -uphi_bs ./ r

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :T || ℓ_from < m_pert_from
            continue
        end

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :P || ℓ_to < m_pert_to
                continue
            end

            metric_coeff = compute_sh_coupling_unweighted(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )
            if abs(metric_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :T)]
            idx_to = idx_map_to[(ℓ_to, :P)]

            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += metric_coeff * metric_profile[i]
            end
        end
    end
end


"""
    add_temperature_gradient_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, dtheta_dr_coeffs, params)

Add coupling from perturbation velocity advecting basic state temperature: (u' · ∇)θ̄_bs

This term appears in the temperature equation and couples the poloidal velocity
(which determines u'_r) to the temperature through the radial gradient of θ̄_bs.

Arguments:
- dtheta_dr_coeffs: Dictionary of interpolated dtheta_dr coefficients on the operator grid
"""
function add_temperature_gradient_coupling!(C::Matrix{Complex{T}},
                                             op_from, op_to,
                                             idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                             idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                             m_from::Int, m_to::Int,
                                             ℓ_bs::Int, m_bs::Int,
                                             r::Vector{T},
                                             dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                             params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Basic-state temperature gradient as complex SH coefficient, already interpolated.
    dtheta_dr_bs = _basic_state_complex_profile(dtheta_dr_coeffs, ℓ_bs, m_bs)
    (dtheta_dr_bs === nothing || maximum(abs, dtheta_dr_bs) < 1e-14) && return

    # The coupling is: u'_r × ∂θ̄_bs/∂r with the same radial weighting
    # used in the single-mode operator assembly.
    r2_dtheta_dr = (r .^ 2) .* dtheta_dr_bs

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue  # Poloidal potential gives radial velocity
        end
        if ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)
        # Match the sparse radial weighting used by the single-mode
        # temperature equation before projection (invariant in ℓ_to).
        temp_grad_coeff = -L_from .* r2_dtheta_dr

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :Θ
                continue  # This goes into temperature equation
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute spherical harmonic coupling
            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            # Get index ranges
            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :Θ)]

            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += coupling_coeff * temp_grad_coeff[i]
            end
        end
    end
end


"""
    add_shear_coupling!(C, op_from, op_to, m_from, m_to, ℓ_bs, m_bs, r, duphi_dr_coeffs, params)

Add shear production coupling: (u' · ∇)ū_bs contribution to momentum equations.

This couples perturbation poloidal velocity to toroidal velocity through
the basic state velocity gradients.

Arguments:
- duphi_dr_coeffs: Dictionary of interpolated duphi_dr coefficients on the operator grid
"""
function add_shear_coupling!(C::Matrix{Complex{T}},
                              op_from, op_to,
                              idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                              idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                              m_from::Int, m_to::Int,
                              ℓ_bs::Int, m_bs::Int,
                              r::Vector{T},
                              duphi_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                              params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    # Basic-state velocity gradient as complex SH coefficient, already interpolated.
    duphi_dr_bs = _basic_state_complex_profile(duphi_dr_coeffs, ℓ_bs, m_bs)
    (duphi_dr_bs === nothing || maximum(abs, duphi_dr_bs) < 1e-14) && return

    # Shear term: u'_r × ∂ū_φ/∂r couples poloidal (P) to toroidal (T)

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue
        end
        if ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)
        # Coupling: -L_from × ∂ū_φ/∂r (invariant in ℓ_to)
        shear_coeff = -L_from .* duphi_dr_bs

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :T
                continue  # Shear goes into toroidal equation
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute spherical harmonic coupling
            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :T)]

            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += coupling_coeff * shear_coeff[i]
            end
        end
    end
end

"""Add meridional temperature-gradient coupling from poloidal velocity into temperature."""
function add_temperature_gradient_theta_coupling!(C::Matrix{Complex{T}},
                                                   op_from, op_to,
                                                   idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                                   idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                                   m_from::Int, m_to::Int,
                                                   ℓ_bs::Int, m_bs::Int,
                                                   r::Vector{T},
                                                   theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                                   Dr::Matrix{T},
                                                   params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    theta_bs = _basic_state_complex_profile(theta_coeffs, ℓ_bs, m_bs)
    (theta_bs === nothing || maximum(abs, theta_bs) < 1e-14) && return
    theta_scaled = theta_bs

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue
        end
        if ℓ_from < m_pert_from
            continue
        end

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :Θ
                continue
            end
            if ℓ_to < m_pert_to
                continue
            end

            meridional_coeff = _meridional_coupling(ℓ_from, ℓ_bs, ℓ_to, m_from, m_bs, m_to)
            if abs(meridional_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :Θ)]

            for i in 1:Nr
                row = idx_to[i]
                row == 0 && continue
                w = -meridional_coeff * theta_scaled[i]  # invariant in j
                for j in 1:Nr
                    col = idx_from[j]
                    col == 0 && continue
                    C[row, col] += w * Dr[i, j]
                end
            end
        end
    end
end


"""
    add_temperature_gradient_phi_coupling!(C, op_from, op_to, ..., m_bs, r, theta_coeffs, params)

Add coupling from perturbation azimuthal velocity advecting basic state temperature: u'_φ × (1/(r sinθ)) × ∂θ̄_bs/∂φ

For non-axisymmetric basic states with m_bs ≠ 0:
    ∂θ̄_bs/∂φ = i × m_bs × θ̄_bs

This term couples the toroidal velocity potential (which gives u'_φ) to the temperature
equation through the azimuthal derivative of θ̄_bs.

The coupling is: u'_φ × (1/(r sinθ)) × (i × m_bs) × θ̄_bs
"""
function add_temperature_gradient_phi_coupling!(C::Matrix{Complex{T}},
                                                 op_from, op_to,
                                                 idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                                 idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                                 m_from::Int, m_to::Int,
                                                 ℓ_bs::Int, m_bs::Int,
                                                 r::Vector{T},
                                                 theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                                 params) where T
    # This term only contributes when m_bs ≠ 0 (non-axisymmetric basic state)
    if m_bs == 0
        return
    end

    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    theta_bs = _basic_state_complex_profile(theta_coeffs, ℓ_bs, m_bs)
    (theta_bs === nothing || maximum(abs, theta_bs) < 1e-14) && return

    # The φ-derivative of θ̄_bs gives: ∂θ̄_bs/∂φ = i × m_bs × θ̄_bs
    # Combined with 1/r factor for the advection term
    # Factor: (i × m_bs) × θ̄_bs / r
    inv_r = one(T) ./ r
    phi_deriv_coeff = (im * m_bs) .* (theta_bs .* inv_r)

    # Loop over toroidal field (source of u'_φ) coupling to temperature
    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :T  # Toroidal potential gives azimuthal velocity
            continue
        end
        if ℓ_from < m_pert_from
            continue
        end

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :Θ  # This goes into temperature equation
                continue
            end
            if ℓ_to < m_pert_to
                continue
            end

            # Compute the unweighted spherical harmonic coupling coefficient.
            # The 1/sinθ factor in the azimuthal derivative cancels the sinθ
            # integration measure, matching the zonal-advection coupling.
            coupling_coeff = compute_sh_coupling_unweighted(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            # Get index ranges
            idx_from = idx_map_from[(ℓ_from, :T)]
            idx_to = idx_map_to[(ℓ_to, :Θ)]

            # Add coupling: coupling_coeff × (i × m_bs / r) × θ̄_bs
            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += coupling_coeff * phi_deriv_coeff[i]
            end
        end
    end
end


"""Add meridional shear coupling from poloidal velocity into toroidal velocity."""
function add_shear_theta_coupling!(C::Matrix{Complex{T}},
                                    op_from, op_to,
                                    idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                    idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                    m_from::Int, m_to::Int,
                                    ℓ_bs::Int, m_bs::Int,
                                    r::Vector{T},
                                    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                    Dr::Matrix{T},
                                    params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    uphi_bs = _basic_state_complex_profile(uphi_coeffs, ℓ_bs, m_bs)
    (uphi_bs === nothing || maximum(abs, uphi_bs) < 1e-14) && return
    uphi_scaled = uphi_bs

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P
            continue
        end
        if ℓ_from < m_pert_from
            continue
        end

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :T
                continue
            end
            if ℓ_to < m_pert_to
                continue
            end

            meridional_coeff = _meridional_coupling(ℓ_from, ℓ_bs, ℓ_to, m_from, m_bs, m_to)
            if abs(meridional_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :T)]

            for i in 1:Nr
                row = idx_to[i]
                row == 0 && continue
                w = -meridional_coeff * uphi_scaled[i]  # invariant in j
                for j in 1:Nr
                    col = idx_from[j]
                    col == 0 && continue
                    C[row, col] += w * Dr[i, j]
                end
            end
        end
    end
end

"""Add basic-state radial advection of perturbation temperature."""
function add_radial_advection_coupling!(C::Matrix{Complex{T}},
                                        op_from, op_to,
                                        idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                        idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                        m_from::Int, m_to::Int,
                                        ℓ_bs::Int, m_bs::Int,
                                        r::Vector{T},
                                        ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                        Dr::Matrix{T},
                                        params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    ur_coeffs_bs = _basic_state_complex_profile(ur_coeffs, ℓ_bs, m_bs)
    (ur_coeffs_bs === nothing || maximum(abs, ur_coeffs_bs) < 1e-14) && return
    ur_scaled = ur_coeffs_bs

    for field in (:P, :T, :Θ)
        for (ℓ_from, field_from) in keys(op_from.index_map)
            if field_from != field || ℓ_from < m_pert_from
                continue
            end

            for (ℓ_to, field_to) in keys(op_to.index_map)
                if field_to != field || ℓ_to < m_pert_to
                    continue
                end

                coupling_coeff = compute_sh_coupling_coefficient(
                    ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
                )

                if abs(coupling_coeff) < 1e-14
                    continue
                end

                idx_from = idx_map_from[(ℓ_from, field)]
                idx_to = idx_map_to[(ℓ_to, field)]

                for i in 1:Nr
                    row = idx_to[i]
                    row == 0 && continue
                    w = coupling_coeff * ur_scaled[i]  # invariant in j
                    for j in 1:Nr
                        col = idx_from[j]
                        col == 0 && continue
                        C[row, col] += w * Dr[i, j]
                    end
                end
            end
        end
    end
end

"""Add basic-state meridional advection of perturbation temperature."""
function add_meridional_advection_coupling!(C::Matrix{Complex{T}},
                                             op_from, op_to,
                                             idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                             idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                             m_from::Int, m_to::Int,
                                             ℓ_bs::Int, m_bs::Int,
                                             r::Vector{T},
                                             utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                             params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    utheta_bs = _basic_state_complex_profile(utheta_coeffs, ℓ_bs, m_bs)
    (utheta_bs === nothing || maximum(abs, utheta_bs) < 1e-14) && return
    adv_profile = utheta_bs ./ r

    for field in (:P, :T, :Θ)
        for (ℓ_from, field_from) in keys(op_from.index_map)
            if field_from != field || ℓ_from < m_pert_from
                continue
            end

            for (ℓ_to, field_to) in keys(op_to.index_map)
                if field_to != field || ℓ_to < m_pert_to
                    continue
                end

                coupling_coeff = _perturbation_meridional_coupling(
                    ℓ_from, ℓ_bs, ℓ_to, m_from, m_bs, m_to
                )

                if abs(coupling_coeff) < 1e-14
                    continue
                end

                idx_from = idx_map_from[(ℓ_from, field)]
                idx_to = idx_map_to[(ℓ_to, field)]

                for i in 1:Nr
                    row = idx_to[i]
                    col = idx_from[i]
                    (row == 0 || col == 0) && continue
                    C[row, col] += coupling_coeff * adv_profile[i]
                end
            end
        end
    end
end

"""Add perturbation radial-velocity shear acting on the basic radial velocity."""
function add_radial_velocity_shear!(C::Matrix{Complex{T}},
                                     op_from, op_to,
                                     idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                     idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                     m_from::Int, m_to::Int,
                                     ℓ_bs::Int, m_bs::Int,
                                     r::Vector{T},
                                     dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                     params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    dur_dr_bs = _basic_state_complex_profile(dur_dr_coeffs, ℓ_bs, m_bs)
    (dur_dr_bs === nothing || maximum(abs, dur_dr_bs) < 1e-14) && return

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P || ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)
        shear_profile = -L_from .* dur_dr_bs   # invariant in ℓ_to

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :P || ℓ_to < m_pert_to
                continue
            end

            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :P)]

            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += coupling_coeff * shear_profile[i]
            end
        end
    end
end

"""Add perturbation meridional-velocity shear acting on the basic meridional flow."""
function add_meridional_velocity_shear!(C::Matrix{Complex{T}},
                                         op_from, op_to,
                                         idx_map_from::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                         idx_map_to::Dict{Tuple{Int,Symbol}, Vector{Int}},
                                         m_from::Int, m_to::Int,
                                         ℓ_bs::Int, m_bs::Int,
                                         r::Vector{T},
                                         dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
                                         params) where T
    Nr = length(r)
    m_pert_from = abs(m_from)
    m_pert_to = abs(m_to)

    dutheta_dr_bs = _basic_state_complex_profile(dutheta_dr_coeffs, ℓ_bs, m_bs)
    (dutheta_dr_bs === nothing || maximum(abs, dutheta_dr_bs) < 1e-14) && return

    for (ℓ_from, field_from) in keys(op_from.index_map)
        if field_from != :P || ℓ_from < m_pert_from
            continue
        end

        L_from = ℓ_from * (ℓ_from + 1)
        shear_profile = -L_from .* dutheta_dr_bs   # invariant in ℓ_to

        for (ℓ_to, field_to) in keys(op_to.index_map)
            if field_to != :P || ℓ_to < m_pert_to
                continue
            end

            coupling_coeff = compute_sh_coupling_coefficient(
                ℓ_from, m_from, ℓ_bs, m_bs, ℓ_to, m_to
            )

            if abs(coupling_coeff) < 1e-14
                continue
            end

            idx_from = idx_map_from[(ℓ_from, :P)]
            idx_to = idx_map_to[(ℓ_to, :P)]

            for i in 1:Nr
                row = idx_to[i]
                col = idx_from[i]
                (row == 0 || col == 0) && continue
                C[row, col] += coupling_coeff * shear_profile[i]
            end
        end
    end
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
