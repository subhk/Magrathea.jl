# ============================================================================
# Input validation for Magrathea.jl parameter types
#
# These checks run during public problem construction, so they should keep error
# messages explicit while avoiding avoidable work in common successful paths.
# ============================================================================

"""
    _grid_mismatch(r, expected)

Return the maximum absolute mismatch between two radial grids without
materializing `r .- expected` or `abs.(...)` temporaries.
"""
function _grid_mismatch(r::AbstractVector{T}, expected::AbstractVector{S}) where {T,S}
    length(r) == length(expected) || throw(DimensionMismatch(
        "grid vectors must have same length, got $(length(r)) and $(length(expected))"))

    R = promote_type(T, S)
    mismatch = zero(R)
    @inbounds for i in eachindex(r, expected)
        mismatch = max(mismatch, abs(R(r[i]) - R(expected[i])))
    end
    return mismatch
end

"""
    validate_onset_params(params)

Validate onset parameters. Throws `ArgumentError` for invalid values,
emits `@warn` for unusual-but-valid combinations.
Called automatically in the OnsetParams constructor.
"""
function validate_onset_params(params)
    # --- Hard errors ---
    0 < params.χ < 1 || throw(ArgumentError(
        "Radius ratio χ must be in (0,1), got $(params.χ)"))
    params.E > 0 || throw(ArgumentError(
        "Ekman number E must be positive, got $(params.E)"))
    params.Pr > 0 || throw(ArgumentError(
        "Prandtl number Pr must be positive, got $(params.Pr)"))
    params.Ra >= 0 || throw(ArgumentError(
        "Rayleigh number Ra must be non-negative, got $(params.Ra)"))
    params.Nr >= 8 || throw(ArgumentError(
        "Nr must be >= 8 for meaningful resolution, got $(params.Nr)"))
    params.lmax >= 1 || throw(ArgumentError(
        "lmax must be >= 1, got $(params.lmax)"))
    params.m >= 0 || throw(ArgumentError(
        "Azimuthal wavenumber m must be >= 0, got $(params.m)"))
    params.mechanical_bc in (:no_slip, :stress_free) || throw(ArgumentError(
        "mechanical_bc must be :no_slip or :stress_free, got :$(params.mechanical_bc)"))
    params.thermal_bc in (:fixed_temperature, :fixed_flux) || throw(ArgumentError(
        "thermal_bc must be :fixed_temperature or :fixed_flux, got :$(params.thermal_bc)"))
    params.equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
        "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$(params.equatorial_symmetry)"))

    # --- Warnings ---
    params.Nr < 16 && @warn "Nr=$(params.Nr) is very low — results may be under-resolved"
    params.E > 0.1 && @warn "E=$(params.E) is unusually large — Coriolis effects may be negligible"
    params.E < 1e-8 && @warn "E=$(params.E) is very small — may require high Nr and lmax for convergence"
    params.lmax > 3 * params.Nr && @warn "Angular resolution far exceeds radial: lmax=$(params.lmax) >> Nr=$(params.Nr)"
    params.m > params.lmax && @warn "No modes will be included: m=$(params.m) > lmax=$(params.lmax)"

    return nothing
end

"""
    validate_basic_state_consistency(bs::BasicState, params)

Cross-validate that a BasicState is compatible with the given parameters.
"""
function validate_basic_state_consistency(bs, params)
    bs.Nr == params.Nr || throw(ArgumentError(
        "BasicState Nr=$(bs.Nr) doesn't match params Nr=$(params.Nr)"))
    length(bs.r) == params.Nr || throw(ArgumentError(
        "BasicState r must have length Nr=$(params.Nr), got $(length(bs.r))"))

    T = float(eltype(bs.r))
    χ = T(params.χ)
    tol = sqrt(eps(T))
    isapprox(first(bs.r), χ; rtol=tol, atol=tol) || throw(ArgumentError(
        "BasicState r must start at χ=$(params.χ), got $(first(bs.r))"))
    isapprox(last(bs.r), one(T); rtol=tol, atol=tol) || throw(ArgumentError(
        "BasicState r must end at 1, got $(last(bs.r))"))

    expected_grid = ChebyshevDiffn(params.Nr, [χ, one(T)], 1).x
    grid_error = _grid_mismatch(bs.r, expected_grid)
    grid_tol = T(10) * tol
    grid_error <= grid_tol || throw(ArgumentError(
        "BasicState r must match Chebyshev nodes (max mismatch = $grid_error)"))

    coefficient_fields = (
        (:theta_coeffs, bs.theta_coeffs),
        (:uphi_coeffs, bs.uphi_coeffs),
        (:dtheta_dr_coeffs, bs.dtheta_dr_coeffs),
        (:duphi_dr_coeffs, bs.duphi_dr_coeffs),
    )
    for (field_name, coefficient_dict) in coefficient_fields
        for coeffs in values(coefficient_dict)
            length(coeffs) == params.Nr || throw(ArgumentError(
                "BasicState $field_name entries must have length Nr=$(params.Nr)"))
        end
    end

    return nothing
end

"""
    validate_basic_state_3d_consistency(bs::BasicState3D, params)

Cross-validate that a BasicState3D is compatible with the given parameters.
"""
function validate_basic_state_3d_consistency(bs, params)
    bs.Nr == params.Nr || throw(ArgumentError(
        "BasicState3D Nr=$(bs.Nr) doesn't match params Nr=$(params.Nr)"))
    length(bs.r) == params.Nr || throw(ArgumentError(
        "BasicState3D r must have length Nr=$(params.Nr), got $(length(bs.r))"))

    T = float(eltype(bs.r))
    χ = T(params.χ)
    tol = sqrt(eps(T))
    isapprox(first(bs.r), χ; rtol=tol, atol=tol) || throw(ArgumentError(
        "BasicState3D r must start at χ=$(params.χ), got $(first(bs.r))"))
    isapprox(last(bs.r), one(T); rtol=tol, atol=tol) || throw(ArgumentError(
        "BasicState3D r must end at 1, got $(last(bs.r))"))

    expected_grid = ChebyshevDiffn(params.Nr, [χ, one(T)], 1).x
    grid_error = _grid_mismatch(bs.r, expected_grid)
    grid_tol = T(10) * tol
    grid_error <= grid_tol || throw(ArgumentError(
        "BasicState3D r must match Chebyshev nodes (max mismatch = $grid_error)"))

    coefficient_fields = (
        (:theta_coeffs, bs.theta_coeffs),
        (:dtheta_dr_coeffs, bs.dtheta_dr_coeffs),
        (:ur_coeffs, bs.ur_coeffs),
        (:utheta_coeffs, bs.utheta_coeffs),
        (:uphi_coeffs, bs.uphi_coeffs),
        (:dur_dr_coeffs, bs.dur_dr_coeffs),
        (:dutheta_dr_coeffs, bs.dutheta_dr_coeffs),
        (:duphi_dr_coeffs, bs.duphi_dr_coeffs),
    )
    for (field_name, coefficient_dict) in coefficient_fields
        for coeffs in values(coefficient_dict)
            length(coeffs) == params.Nr || throw(ArgumentError(
                "BasicState3D $field_name entries must have length Nr=$(params.Nr)"))
        end
    end

    return nothing
end

"""
    validate_biglobal_params(params, basic_state)

Validate biglobal parameters. Applies all onset checks to the base physics
parameters and cross-validates the basic state consistency.
"""
function validate_biglobal_params(params, basic_state)
    validate_onset_params(params)
    validate_basic_state_consistency(basic_state, params)
    return nothing
end

"""
    validate_mhd_params(params::MHDParams)

Validate MHD parameters. Emits `@warn` for unusual-but-valid combinations.
Hard validation is handled by the `MHDParams` constructor.
"""
function validate_mhd_params(params)
    params.N < 16 && @warn "N=$(params.N) is very low — results may be under-resolved"
    params.E > 0.1 && @warn "E=$(params.E) is unusually large — Coriolis effects may be negligible"
    params.E < 1e-8 && @warn "E=$(params.E) is very small — may require high N and lmax for convergence"
    params.lmax > 3 * params.N && @warn "Angular resolution far exceeds radial: lmax=$(params.lmax) >> N=$(params.N)"
    params.m > params.lmax && @warn "No modes will be included: m=$(params.m) > lmax=$(params.lmax)"
    params.Pm < 1e-3 && @warn "Pm=$(params.Pm) is very small — magnetic diffusion dominates"
    params.Le > 1.0 && @warn "Le=$(params.Le) is unusually large — strong field regime"
    return nothing
end

"""
    validate_triglobal_params(params, basic_state, m_range)

Validate triglobal parameters. Applies all onset checks to the base physics
parameters, validates the m_range, and cross-validates the 3D basic state.
"""
function validate_triglobal_params(params, basic_state, m_range)
    validate_onset_params(params)
    validate_basic_state_3d_consistency(basic_state, params)

    _validate_triglobal_m_range(m_range, params.lmax)
    first(m_range) >= 0 || @warn "m_range starts at $(first(m_range)) — negative m modes included"

    return nothing
end
