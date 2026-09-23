# =============================================================================
#  Self-consistent basic states with temperature advection
#
#  Solve (E/Pr) ∇²T̄ = ū·∇T̄ using the complete Stokes–Coriolis velocity.
#  Both axisymmetric and nonaxisymmetric states can transport heat meridionally.
#  Each relaxed thermal update is followed by a new viscous momentum solve.
# =============================================================================

"""
    AdvectionDiffusionSolver{T<:Real}

Holds parameters and state for iterative advection-diffusion solution.

Fields:
- `cd` : ChebyshevDiffn - radial discretization
- `r_i, r_o` : Inner and outer radii
- `E, Ra, Pr` : Ekman, Rayleigh, and Prandtl numbers
- `κ` : Thermal diffusivity (computed from other parameters)
- `lmax_bs, mmax_bs` : Maximum spherical harmonic degrees
- `mechanical_bc, thermal_bc` : Boundary condition types
- `max_iterations` : Maximum Picard iterations
- `tolerance` : Convergence tolerance
"""
@with_kw struct AdvectionDiffusionSolver{T<:Real}
    cd::ChebyshevDiffn{T}
    r_i::T
    r_o::T
    E::T
    Ra::T
    Pr::T
    lmax_bs::Int
    mmax_bs::Int
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    max_iterations::Int = 20
    tolerance::T = T(1e-8)
end

_value_real_type(::Type{T}) where {T<:Real} = T
_value_real_type(::Type{Complex{T}}) where {T<:Real} = T

@inline _maxabs(v) = maximum(abs, v)

function _maxabsdiff(a, b)
    R = promote_type(_value_real_type(eltype(a)), _value_real_type(eltype(b)))
    out = zero(R)
    @inbounds for i in eachindex(a, b)
        diff = abs(a[i] - b[i])
        out = max(out, R(diff))
    end
    return out
end


"""
    solve_poisson_mode(ℓ, m, r, D2, D1, r_i, r_o, forcing;
                       inner_value=0, outer_value=0, outer_bc=:fixed_temperature)

Solve the radial Poisson equation for a single spherical harmonic mode:

    ∇²T̄_ℓm = f_ℓm(r)

where ∇² in spherical harmonics becomes:

    d²/dr² + (2/r)d/dr - ℓ(ℓ+1)/r² = f_ℓm(r)

Returns T̄_ℓm(r) and ∂T̄_ℓm/∂r.
Flux values denote the radial derivative, not outward-normal heat flux.
For ℓ=0, two flux conditions leave the temperature constant undetermined;
this helper requires at least one fixed-temperature boundary in that case.
"""
function solve_poisson_mode(
    ℓ::Int, m::Int,
    r::Vector{T}, D2::Matrix{T}, D1::Matrix{T},
    r_i::T, r_o::T,
    forcing::Vector{T};
    inner_value::T = zero(T),
    outer_value::T = zero(T),
    outer_bc::Symbol = :fixed_temperature,
    inner_bc::Symbol = :fixed_temperature
) where T<:Real

    inner_bc in (:fixed_temperature, :fixed_flux) &&
        outer_bc in (:fixed_temperature, :fixed_flux) ||
        throw(ArgumentError("Thermal boundaries must be :fixed_temperature or :fixed_flux"))
    ℓ == 0 && inner_bc == outer_bc == :fixed_flux && throw(ArgumentError(
        "The degree-zero Poisson problem with two flux boundaries needs a compatibility condition and a temperature gauge. Specify at least one fixed-temperature boundary."))
    Nr = length(r)

    # Build the dense radial Laplacian directly.  The equivalent Diagonal-based
    # expression materializes several Nr-by-Nr temporaries inside the modal loop.
    A_mat = copy(D2)
    ℓ_factor = T(ℓ * (ℓ + 1))
    @inbounds for i in 1:Nr
        inv_r = inv(r[i])
        d1_scale = T(2) * inv_r
        for j in 1:Nr
            A_mat[i, j] += d1_scale * D1[i, j]
        end
        A_mat[i, i] -= ℓ_factor * inv_r * inv_r
    end

    f_rhs = copy(forcing)

    # Determine boundary indices (Chebyshev nodes can be ascending or descending)
    idx_inner = abs(r[1] - r_i) < abs(r[Nr] - r_i) ? 1 : Nr
    idx_outer = idx_inner == 1 ? Nr : 1

    # Inner boundary condition (typically fixed temperature = hot)
    if inner_bc == :fixed_temperature
        A_mat[idx_inner, :] .= zero(T)
        A_mat[idx_inner, idx_inner] = one(T)
        f_rhs[idx_inner] = inner_value
    else  # fixed_flux
        @inbounds for j in 1:Nr
            A_mat[idx_inner, j] = D1[idx_inner, j]
        end
        f_rhs[idx_inner] = inner_value  # This is the flux value
    end

    # Outer boundary condition
    if outer_bc == :fixed_temperature
        A_mat[idx_outer, :] .= zero(T)
        A_mat[idx_outer, idx_outer] = one(T)
        f_rhs[idx_outer] = outer_value
    else  # fixed_flux
        @inbounds for j in 1:Nr
            A_mat[idx_outer, j] = D1[idx_outer, j]
        end
        f_rhs[idx_outer] = outer_value  # This is the flux value
    end

    # Solve the linear system
    T_lm = A_mat \ f_rhs
    dT_dr = D1 * T_lm

    return T_lm, dT_dr
end


# =============================================================================
#  Spherical Harmonic Coupling Coefficients (orthonormal Y_ℓm, Condon–Shortley)
#
#  cos(θ) and sinθ ∂_θ map Y_ℓm into span{Y_{ℓ-1,m}, Y_{ℓ+1,m}}:
#    cos(θ) Y_ℓm      = α⁻_ℓm Y_{ℓ-1,m} + α⁺_ℓm Y_{ℓ+1,m}
#    sinθ ∂_θ Y_ℓm    = −(ℓ+1) α⁻_ℓm Y_{ℓ-1,m} + ℓ α⁺_ℓm Y_{ℓ+1,m}
#  Multiplication by sin(θ) or 1/sin(θ) is not banded: it couples every
#  same-parity degree, so only exact matrix elements are provided for 1/sinθ.
# =============================================================================

"""
    cos_theta_coupling(ℓ::Int, m::Int)

Exact coefficients of cos(θ) × Y_ℓm = b⁻ Y_{ℓ-1,m} + b⁺ Y_{ℓ+1,m} for orthonormal
spherical harmonics (Condon–Shortley phase; cosine and sine phases alike).

Returns (b_minus, b_plus) where:
- b⁻ = √[(ℓ-m)(ℓ+m) / ((2ℓ-1)(2ℓ+1))]      (zero for ℓ = |m|)
- b⁺ = √[(ℓ-m+1)(ℓ+m+1) / ((2ℓ+1)(2ℓ+3))]

These are the orthonormal forms of cos(θ) P_ℓ^m = [(ℓ-m+1)P_{ℓ+1}^m + (ℓ+m)P_{ℓ-1}^m]/(2ℓ+1).
"""
function cos_theta_coupling(ℓ::Int, m::Int)
    # Using the standard recurrence for associated Legendre polynomials
    # cos(θ) P_ℓ^m = [(ℓ-m+1)/(2ℓ+1)] P_{ℓ+1}^m + [(ℓ+m)/(2ℓ+1)] P_{ℓ-1}^m
    # After normalization for spherical harmonics:

    # Coupling to ℓ-1
    b_minus = 0.0
    if ℓ > abs(m)
        # From recurrence relation
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num > 0 && den > 0
            b_minus = sqrt(num / den)
        end
    end

    # Coupling to ℓ+1
    num = (ℓ - m + 1) * (ℓ + m + 1)
    den = (2ℓ + 1) * (2ℓ + 3)
    b_plus = num >= 0 && den > 0 ? sqrt(num / den) : 0.0

    return (b_minus, b_plus)
end


"""
    inv_sin_theta_coupling(ℓ::Int, m::Int; max_coupling::Int=4)

Exact matrix elements ⟨Y_ℓ'm | 1/sinθ | Y_ℓm⟩ of orthonormal spherical harmonics
(see [`inv_sin_theta_gaunt`](@ref)) for every ℓ' ≥ |m| with |ℓ' - ℓ| ≤
`max_coupling` and ℓ' - ℓ even; opposite-parity elements vanish exactly.

1/sinθ is not banded: it couples all same-parity degrees, e.g.
⟨Y₀₀|1/sinθ|Y₀₀⟩ = π/2. Restricting to `max_coupling` is therefore a truncation
chosen by the caller, not a property of the operator.

Returns a Dict{Int, Float64} mapping output ℓ' to its coefficient (empty when
ℓ < |m|).
"""
function inv_sin_theta_coupling(ℓ::Int, m::Int; max_coupling::Int=4)
    max_coupling >= 0 || throw(ArgumentError("max_coupling must be non-negative, got $max_coupling"))
    coeffs = Dict{Int, Float64}()
    ℓ < abs(m) && return coeffs
    for L in (ℓ - 2 * (max_coupling ÷ 2)):2:(ℓ + max_coupling)
        L < abs(m) && continue
        coeffs[L] = inv_sin_theta_gaunt(L, ℓ, m)
    end
    return coeffs
end


# =============================================================================
#  Angular coupling utilities and compatibility meridional-flow interfaces.
#  The complete velocity solve is in steady_flow.jl and includes viscosity,
#  both mechanical boundaries, and axisymmetric meridional circulation.
# =============================================================================

"""
    theta_derivative_coupling(ℓ::Int, m::Int)

Exact coefficients of sinθ × ∂Y_ℓm/∂θ for orthonormal spherical harmonics:

    sinθ ∂Y_ℓm/∂θ = A⁺_ℓm Y_{ℓ+1,m} + A⁻_ℓm Y_{ℓ-1,m}

Returns (A_minus, A_plus, A_diag) where:
- A_minus = −(ℓ+1) α⁻_ℓm: coefficient for coupling to ℓ-1
- A_plus = ℓ α⁺_ℓm: coefficient for coupling to ℓ+1
- A_diag = 0: the identity has no diagonal term (kept for the 3-tuple format)

with (α⁻, α⁺) = `cos_theta_coupling(ℓ, m)`.
"""
function theta_derivative_coupling(ℓ::Int, m::Int)
    # Exact, finite two-term identity for orthonormal Y_ℓm (quadrature-tested in
    # test/basicstate_review_fixes.jl):
    #     sinθ ∂Y_ℓm/∂θ = ℓ α⁺_ℓ Y_{ℓ+1,m} − (ℓ+1) α⁻_ℓ Y_{ℓ-1,m}
    # with α⁺_ℓ = √[((ℓ+1)²−m²)/((2ℓ+1)(2ℓ+3))], α⁻_ℓ = √[(ℓ²−m²)/((2ℓ−1)(2ℓ+1))]
    # (these α are the orthonormal cosθ recurrence coefficients, == cos_theta_coupling).

    # A⁺: coupling to ℓ+1  =  +ℓ α⁺_ℓ
    A_plus = 0.0
    if ℓ + 1 >= abs(m)
        num = (ℓ + 1 + m) * (ℓ + 1 - m)
        den = (2ℓ + 1) * (2ℓ + 3)
        if num >= 0 && den > 0
            A_plus = ℓ * sqrt(num / den)
        end
    end

    # A⁻: coupling to ℓ-1  =  −(ℓ+1) α⁻_ℓ
    A_minus = 0.0
    if ℓ - 1 >= abs(m) && ℓ > 0
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num >= 0 && den > 0
            A_minus = -(ℓ + 1) * sqrt(num / den)
        end
    end

    # The identity is exact with no diagonal term.
    A_diag = 0.0

    return (A_minus, A_plus, A_diag)
end


"""
    inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)

Exact integral ⟨Y_Lm | 1/sinθ | Y_ℓm⟩ for orthonormal spherical harmonics.

The 1/sinθ factor couples every L with |L - ℓ| even (0, 2, 4, ...); elements
with L + ℓ odd, or L or ℓ below |m|, vanish exactly.
"""
function inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)
    am = abs(m)
    (L < am || ℓ < am) && return 0.0
    # Opposite parity ⇒ integrand odd under x→−x ⇒ exactly zero.
    (L + ℓ) % 2 != 0 && return 0.0

    # Exact ⟨Y_Lm | 1/sinθ | Y_ℓm⟩ = ∫₀^π P̄_Lm(cosθ) P̄_ℓm(cosθ) dθ — the 1/sinθ
    # cancels the sinθ of dΩ. With x = cosθ this is ∫ P̄_Lm P̄_ℓm /√(1-x²) dx, and
    # P̄_Lm P̄_ℓm is a polynomial of degree L+ℓ (the (1-x²)^{m/2} factors pair up),
    # so the max(L,ℓ)+2 Gauss–Chebyshev nodes below integrate it exactly. Same
    # orthonormal convention as the cosθ / sinθ∂θ operators.
    lmax_needed = max(L, ℓ)
    Nq = lmax_needed + 2
    s = 0.0
    for j in 1:Nq
        x = cos(π * (j - 0.5) / Nq)
        P = _orthonormal_plm(lmax_needed, am, x)
        s += P[L + 1] * P[ℓ + 1]
    end
    return (π / Nq) * s
end


"""
Compatibility component projection of the steady Stokes–Coriolis solution
for |m_bs|. Updates both cosine and sine phases, including `uphi_coeffs`, and
enforces both mechanical boundaries; entries for other orders are left in
place. Use a basic-state constructor to retain the divergence-free vector
potentials. Shares its implementation with the other component wrappers.
"""
function solve_meridional_coupled!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T}, D2::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    m_bs::Int, lmax_bs::Int;
    mechanical_bc::Symbol = :no_slip
) where T<:Real

    abs(m_bs)>lmax_bs && return nothing
    theta = Dict(k=>v for (k,v) in theta_coeffs if abs(k[2])==abs(m_bs))
    _project_mean_flow!((ur=ur_coeffs, utheta=utheta_coeffs, uphi=uphi_coeffs,
                         dur=dur_dr_coeffs, dutheta=dutheta_dr_coeffs),
                        theta, r, D1, D2, E, Ra, Pr, lmax_bs, abs(m_bs);
                        mechanical_bc=mechanical_bc, keep=k->abs(k[2])==abs(m_bs),
                        reset=false)
    return nothing
end


"""
Compatibility wrapper using the complete steady viscous solve (with `D1*D1` as
the second-derivative matrix). The historical diagonal approximation is no
longer used. Prefer the basic-state constructors.
"""
function solve_meridional_simple!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int;
    mechanical_bc::Symbol = :no_slip
) where T<:Real

    _project_mean_flow!((ur=ur_coeffs, utheta=utheta_coeffs,
                         dur=dur_dr_coeffs, dutheta=dutheta_dr_coeffs),
                        theta_coeffs, r, D1, D1*D1, E, Ra, Pr, lmax_bs, mmax_bs;
                        mechanical_bc=mechanical_bc)
    return nothing
end


"""
Populate component projections from a divergence-free toroidal–poloidal
Stokes–Coriolis solve. `use_full_coupling` is an ignored compatibility keyword:
the full solve is always used, and `false` emits a one-time warning.
`include_meridional=false` explicitly clears the meridional outputs and does
not construct a complete physical mean flow.
"""
function solve_meridional_circulation_toroidal_poloidal!(
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dutheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    r::Vector{T}, D1::Matrix{T}, D2::Matrix{T},
    r_i::T, r_o::T,
    Ra::T, E::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int;
    mechanical_bc::Symbol = :no_slip,
    include_meridional::Bool = true,
    use_full_coupling::Bool = true
) where T<:Real

    _warn_ignored_flow_keywords(use_full_coupling=use_full_coupling)
    if !include_meridional
        for m in -mmax_bs:mmax_bs,l in abs(m):lmax_bs
            for target in (ur_coeffs,utheta_coeffs,dur_dr_coeffs,dutheta_dr_coeffs)
                target[(l,m)]=zeros(T,length(r))
            end
        end
        return nothing
    end
    _project_mean_flow!((ur=ur_coeffs, utheta=utheta_coeffs, uphi=uphi_coeffs,
                         dur=dur_dr_coeffs, dutheta=dutheta_dr_coeffs),
                        theta_coeffs, r, D1, D2, E, Ra, Pr, lmax_bs, mmax_bs;
                        mechanical_bc=mechanical_bc)
    return nothing
end


# =============================================================================
#  Full Advection Term with All Velocity Components
# =============================================================================

"""
    compute_full_advection_spectral(bs)
    compute_full_advection_spectral(theta_coeffs, dtheta_dr_coeffs, bs)

Spectral coefficients of ū·∇T in the public no-factorial normalization, for the
temperature of the basic state `bs` or for any temperature on its radial grid.
The velocity is synthesized from the native divergence-free vector-harmonic
potentials `bs.flow`, the path used by the self-consistent thermal solve, so
the angular products are exact on the dealiased grid. Coefficients are keyed
like the state's own: by ℓ for `BasicState`, by `(ℓ, m)` for `BasicState3D`,
for ℓ ≤ `bs.flow.lmax` and |m| ≤ `bs.flow.mmax`. A state without `flow`
(e.g. a custom state) uses its scalar velocity components as given.

    compute_full_advection_spectral(theta_coeffs, dtheta_dr_coeffs, ur_coeffs,
        dur_dr_coeffs, utheta_coeffs, uphi_coeffs, lmax_bs, mmax_bs, r)

Legacy component form: evaluates u·∇T from real ±m scalar-harmonic expansions
of each velocity component (no-factorial convention; `dur_dr_coeffs` is unused).
Tangential components of a solenoidal flow are not band-limited scalar
harmonics, so for projections of a constructed state's velocity the result is
only approximate, with errors largest near ℓ = `lmax_bs`. It therefore warns
once when tangential velocity is present. Uniform T gives exactly zero.
"""
compute_full_advection_spectral(bs::Union{BasicState,BasicState3D}) =
    compute_full_advection_spectral(bs.theta_coeffs, bs.dtheta_dr_coeffs, bs)

function compute_full_advection_spectral(theta_coeffs::AbstractDict,
                                         dtheta_dr_coeffs::AbstractDict,
                                         bs::Union{BasicState{T},BasicState3D{T}}) where T<:Real
    # Axisymmetric states are keyed by ℓ; work with (ℓ, m) keys internally.
    lift(d) = Dict{Tuple{Int,Int},Vector{T}}((k isa Integer ? (k, 0) : k) => v for (k, v) in d)
    theta = lift(theta_coeffs); dtheta = lift(dtheta_dr_coeffs)
    flow = bs.flow
    lmax, mmax = flow === nothing ? (bs.lmax_bs, bs isa BasicState ? 0 : bs.mmax_bs) :
                                    (flow.lmax, flow.mmax)
    # The transforms skip modes beyond the truncation; reject them instead.
    _check_retained_bc_modes(theta, lmax, mmax, "theta_coeffs")
    _check_retained_bc_modes(dtheta, lmax, mmax, "dtheta_dr_coeffs")
    adv = flow === nothing ?
        _component_advection(theta, dtheta, lift(bs.ur_coeffs), lift(bs.utheta_coeffs),
                             lift(bs.uphi_coeffs), lmax, mmax, bs.r) :
        _mean_flow_advection(theta, dtheta, flow)
    return bs isa BasicState ? Dict(l => v for ((l, m), v) in adv if m == 0) : adv
end

function compute_full_advection_spectral(
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dtheta_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    ur_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    dur_dr_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    utheta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    lmax_bs::Int, mmax_bs::Int,
    r::Vector{T}
) where T<:Real
    if any(any(!iszero, v) for d in (utheta_coeffs, uphi_coeffs) for v in values(d))
        @warn("compute_full_advection_spectral: truncated scalar projections of tangential " *
              "velocity are not band-limited, so this component form is approximate. Use " *
              "compute_full_advection_spectral(bs) for the exact vector-harmonic result.",
              maxlog=1)
    end
    return _component_advection(theta_coeffs, dtheta_dr_coeffs, ur_coeffs,
                                utheta_coeffs, uphi_coeffs, lmax_bs, mmax_bs, r)
end

# u·∇T from scalar component expansions in the public no-factorial convention.
# vecsh_advection works in the orthonormal (full N_ℓm) basis; m=0 is unchanged.
function _component_advection(theta, dtheta_dr, ur, utheta, uphi, lmax, mmax, r)
    F_orth = vecsh_advection(_sh_rescale(theta, +1), _sh_rescale(dtheta_dr, +1),
                             _sh_rescale(ur, +1), _sh_rescale(utheta, +1),
                             _sh_rescale(uphi, +1), lmax, mmax, r)
    return _sh_rescale(F_orth, -1)
end


"""
Solve the steady Navier–Stokes–Coriolis and thermal advection-diffusion equations.
The default `momentum_model=:navier_stokes` includes nonlinear mean-flow inertia.
Use `momentum_model=:stokes` explicitly for the weak-inertia approximation.

Returns `(bs, info)`. `info.converged` requires both projected momentum and
thermal residuals, plus boundary/gauge constraints, to satisfy `tolerance`.
Boundary amplitudes follow `nonaxisymmetric_basic_state`, including the
`ArgumentError` for nonzero modes outside `lmax_bs`/`mmax_bs`;
`coupled_thermal_wind` is an ignored compatibility keyword.
The damped Picard iteration backtracks on the actual residual; nonconvergence
is returned explicitly through `info.termination_reason`. Increase `lmax_bs`,
`mmax_bs` and radial resolution to check the spectral truncation independently.
"""
function nonaxisymmetric_basic_state_selfconsistent(
    cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int, amplitudes::Dict{Tuple{Int,Int}, <:Real};
    mechanical_bc::Symbol = :no_slip,
    thermal_bc::Symbol = :fixed_temperature,
    outer_fluxes::Dict{Tuple{Int,Int}, <:Real} = Dict{Tuple{Int,Int}, T}(),
    max_iterations::Int = 50,
    tolerance::T = T(1e-8),
    verbose::Bool = false,
    coupled_thermal_wind::Bool = true,
    momentum_model::Symbol = :navier_stokes,
    relaxation::Real = 0.5
) where T<:Real

    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    momentum_model in (:navier_stokes,:stokes) || throw(ArgumentError("momentum_model must be :navier_stokes or :stokes"))
    isfinite(relaxation) && 0 < relaxation <= 1 || throw(ArgumentError("relaxation must be in (0,1]"))
    initial = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes;
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc, outer_fluxes=outer_fluxes,
        coupled_thermal_wind=coupled_thermal_wind)
    theta = deepcopy(initial.theta_coeffs)
    r = cd.x; N = length(r); D = Matrix(cd.D1); D2 = Matrix(cd.D2)
    # Preserve the actual prescribed boundary values, including homogeneous
    # Neumann conditions on sine modes generated during the iteration.
    bc = Dict(k => (v[1], thermal_bc==:fixed_temperature ? v[end] :
        initial.dtheta_dr_coeffs[k][end], thermal_bc) for (k,v) in theta)
    theta,flow,info=_iterate_mean_state(theta,initial.flow,D,D2,E,Ra,Pr,bc,mechanical_bc;
        momentum_model=momentum_model,max_iterations=max_iterations,tolerance=tolerance,
        relaxation=relaxation,verbose=verbose)
    dtheta = Dict(k=>D*v for (k,v) in theta)
    ur,uθ,uφ,dur,duθ,duφ = _mean_flow_components(flow)
    bs = BasicState3D(lmax_bs=lmax_bs,mmax_bs=mmax_bs,Nr=N,r=r,
        theta_coeffs=theta,dtheta_dr_coeffs=dtheta,ur_coeffs=ur,utheta_coeffs=uθ,
        uphi_coeffs=uφ,dur_dr_coeffs=dur,dutheta_dr_coeffs=duθ,duphi_dr_coeffs=duφ,flow=flow)
    return bs,info
end


"""
Construct a nonlinear steady mean flow from symbolic boundary conditions.
Both axisymmetric and nonaxisymmetric forcing iterate thermal transport and
Navier–Stokes–Coriolis momentum balance. `momentum_model=:stokes` opts out of
momentum inertia; 3D defaults to `mmax_bs=lmax_bs` to retain generated harmonics. Returns the
basic state and convergence information; pure conduction returns `nothing`
for the information. An explicit `lmax_bs`/`mmax_bs` must retain every nonzero
boundary mode (otherwise an `ArgumentError` is thrown instead of dropping it);
`coupled_thermal_wind` is ignored (warns once when `false`).
See `nonaxisymmetric_basic_state_selfconsistent`.
"""
function basic_state_selfconsistent(cd, χ::Real, E::Real, Ra::Real, Pr::Real;
                                    temperature_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    flux_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    mechanical_bc::Symbol=:no_slip,
                                    lmax_bs::Union{Nothing, Int}=nothing,
                                    max_iterations::Int=50,
                                    tolerance::Float64=1e-8,
                                    verbose::Bool=false,
                                    coupled_thermal_wind::Bool=true,
                                    momentum_model::Symbol=:navier_stokes,
                                    mmax_bs::Union{Nothing,Int}=nothing,
                                    relaxation::Real=0.5)

    # Validate: can't have both temperature_bc and flux_bc
    if temperature_bc !== nothing && flux_bc !== nothing
        error("Cannot specify both temperature_bc and flux_bc. Choose one.")
    end
    _warn_ignored_flow_keywords(coupled_thermal_wind=coupled_thermal_wind)

    momentum_model in (:navier_stokes,:stokes) || throw(ArgumentError("momentum_model must be :navier_stokes or :stokes"))
    max_iterations>0 || throw(ArgumentError("max_iterations must be positive"))
    isfinite(tolerance) && tolerance>0 || throw(ArgumentError("tolerance must be finite and positive"))
    isfinite(relaxation) && 0<relaxation<=1 || throw(ArgumentError("relaxation must be in (0,1]"))
    T = eltype(cd.x)

    # Determine thermal BC type and the boundary condition
    if flux_bc !== nothing
        thermal_bc = :fixed_flux
        bc = flux_bc
    elseif temperature_bc !== nothing
        thermal_bc = :fixed_temperature
        bc = temperature_bc
    else
        # No BC specified → pure conduction (no advection to correct)
        _lmax = lmax_bs === nothing ? 4 : lmax_bs
        return conduction_basic_state(cd, T(χ), _lmax; thermal_bc=:fixed_temperature), nothing
    end

    # Get lmax and mmax from boundary condition
    bc_lmax, bc_mmax = get_lmax_mmax(bc)

    # Use provided lmax_bs or auto-determine
    _lmax = lmax_bs === nothing ? max(bc_lmax + 2, 4) : lmax_bs

    # Products generate new azimuthal harmonics. Retain the full angular band
    # by default in 3D; callers may set mmax_bs as an explicit truncation.
    _mmax=mmax_bs===nothing ? (is_axisymmetric(bc) ? 0 : _lmax) : mmax_bs
    _check_retained_bc_modes(bc.coeffs, _lmax, _mmax,
                             flux_bc === nothing ? "temperature_bc" : "flux_bc")
    bc_mmax<=_mmax<=_lmax || throw(ArgumentError("Require boundary mmax ≤ mmax_bs ≤ lmax_bs"))
    if is_axisymmetric(bc) && _mmax==0
        bs, info = nonaxisymmetric_basic_state_selfconsistent(cd,T(χ),T(E),T(Ra),T(Pr),
            _lmax,0,to_dict(bc); mechanical_bc=mechanical_bc,thermal_bc=thermal_bc,
            max_iterations=max_iterations,tolerance=T(tolerance),verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind,
            momentum_model=momentum_model,relaxation=relaxation)
        return _axisymmetric_state(bs),info
    end

    # Non-axisymmetric: use self-consistent solver
    amplitudes = to_dict(bc)

    if thermal_bc == :fixed_temperature
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, _mmax, amplitudes;
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_temperature,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind,
            momentum_model=momentum_model,relaxation=relaxation
        )
    else  # fixed_flux
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, _mmax,
            Dict{Tuple{Int,Int},T}();  # empty amplitudes
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_flux,
            outer_fluxes=amplitudes,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind,
            momentum_model=momentum_model,relaxation=relaxation
        )
    end
end
