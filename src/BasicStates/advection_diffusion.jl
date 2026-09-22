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
    compute_phi_advection_spectral(theta_coeffs, uphi_coeffs, lmax_bs, mmax_bs, r)

Compute the φ-advection term in spectral space for a single azimuthal mode m_bs.

For temperature T̄_m = Σ_ℓ T̄_ℓm(r) Y_ℓm and velocity ū_φ,m = Σ_L ū_{Lm}(r) Y_Lm,
the advection term is:

    ū_φ/(r sinθ) × ∂T̄/∂φ = ū_φ × (im/r sinθ) × T̄

In spectral space, this involves coupling through:
    Y_Lm × Y_ℓm / sinθ = Σ_L' C_{L,ℓ,L'}^m × Y_{L',m}

where C are coupling coefficients from Gaunt integrals.

For the simplified diagonal approximation (valid for slowly varying ū_φ):
    [ū_φ × im T̄ / (r sinθ)]_{L'm} ≈ im × Σ_L (∫ Y_Lm Y_ℓm Y_{L'm} / sinθ dΩ) × ū_Lm(r) × T̄_ℓm(r) / r

Returns the forcing coefficients for the advection-diffusion equation.
"""
function compute_phi_advection_spectral(
    theta_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    uphi_coeffs::Dict{Tuple{Int,Int}, Vector{T}},
    lmax_bs::Int,
    mmax_bs::Int,
    r::Vector{T}
) where T<:Real

    # Azimuthal advection ū_φ·∂_φT̄ projects to ZERO in the real-orthonormal
    # cos(mφ) basis the basic state is stored in: ∂_φ maps cos(mφ)→sin(mφ), so
    # ū_φ·∂_φT̄ ~ cos·sin = pure sin, orthogonal to every cos(mφ) basis function.
    # (Verified to machine precision by a manufactured-solution test.)
    #
    # The previous implementation used arbitrary "empirical reduction factors"
    # (0.5, 0.3) and ∂_φ → ×m, producing spurious nonzero forcing — removed.
    #
    # The genuine φ-advection is now captured by `vecsh_advection` (divergence form
    # on the full ±m real-SH basis), which routes the cos→sin product into the
    # `sin(mφ)` (`-m`) coefficients. This standalone scalar projection remains zero
    # by construction and is retained only as a documented building block.
    return Dict{Tuple{Int,Int}, Vector{T}}()
end


"""
    solve_poisson_mode(ℓ, m, r, D2, D1, r_i, r_o, forcing;
                       inner_value=0, outer_value=0, outer_bc=:fixed_temperature)

Solve the radial Poisson equation for a single spherical harmonic mode:

    ∇²T̄_ℓm = f_ℓm(r)

where ∇² in spherical harmonics becomes:

    d²/dr² + (2/r)d/dr - ℓ(ℓ+1)/r² = f_ℓm(r)

Returns T̄_ℓm(r) and ∂T̄_ℓm/∂r.
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
#  Spherical Harmonic Coupling Coefficients
#
#  For sin(θ) and cos(θ) multiplications in spectral space:
#    sin(θ) Y_ℓm = a_{ℓ,m}^- Y_{ℓ-1,m} + a_{ℓ,m}^+ Y_{ℓ+1,m}
#    cos(θ) Y_ℓm = b_{ℓ,m}^- Y_{ℓ-1,m} + b_{ℓ,m}^+ Y_{ℓ+1,m}
# =============================================================================

"""
    sin_theta_coupling(ℓ::Int, m::Int)

Compute coupling coefficients for sin(θ) × Y_ℓm = a⁻ Y_{ℓ-1,m} + a⁺ Y_{ℓ+1,m}.

Returns (a_minus, a_plus) where:
- a⁻ = √[(ℓ+m)(ℓ-m) / ((2ℓ-1)(2ℓ+1))]  (coupling to ℓ-1)
- a⁺ = √[(ℓ+m+1)(ℓ-m+1) / ((2ℓ+1)(2ℓ+3))]  (coupling to ℓ+1)
"""
function sin_theta_coupling(ℓ::Int, m::Int)
    # Coupling to ℓ-1
    a_minus = 0.0
    if ℓ > abs(m)
        num = (ℓ + m) * (ℓ - m)
        den = (2ℓ - 1) * (2ℓ + 1)
        if num >= 0 && den > 0
            a_minus = sqrt(num / den)
        end
    end

    # Coupling to ℓ+1
    num = (ℓ + m + 1) * (ℓ - m + 1)
    den = (2ℓ + 1) * (2ℓ + 3)
    a_plus = num >= 0 && den > 0 ? sqrt(num / den) : 0.0

    return (a_minus, a_plus)
end


"""
    cos_theta_coupling(ℓ::Int, m::Int)

Compute coupling coefficients for cos(θ) × Y_ℓm = b⁻ Y_{ℓ-1,m} + b⁺ Y_{ℓ+1,m}.

Returns (b_minus, b_plus) where:
- b⁻ = √[(ℓ-m)(ℓ+m) / ((2ℓ-1)(2ℓ+1))] × (ℓ) / √[ℓ²-m²] ... (simplified)
- b⁺ = √[(ℓ-m+1)(ℓ+m+1) / ((2ℓ+1)(2ℓ+3))]

Using the recurrence: cos(θ) P_ℓ^m = A_ℓ^m P_{ℓ+1}^m + B_ℓ^m P_{ℓ-1}^m
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
    inv_sin_theta_coupling(ℓ::Int, m::Int)

Approximate coupling coefficients for (1/sinθ) × Y_ℓm in spectral space.

Since 1/sinθ is singular at poles, this expansion is approximate and uses
a truncated series representation. The dominant contributions come from
modes with similar ℓ values.

Returns a Dict{Int, Float64} mapping output ℓ' to coupling coefficient.
"""
function inv_sin_theta_coupling(ℓ::Int, m::Int; max_coupling::Int=4)
    # 1/sinθ can be expanded as a series in Legendre polynomials
    # For practical purposes, use approximate coupling to nearby modes

    coeffs = Dict{Int, Float64}()

    # Diagonal term (dominant)
    coeffs[ℓ] = 1.0

    # For m ≠ 0, there's coupling to ℓ ± 2, ℓ ± 4, etc.
    if m != 0
        # ℓ+2 coupling (approximate)
        if ℓ + 2 <= ℓ + max_coupling
            c = 0.5 * m^2 / ((2ℓ + 1) * (2ℓ + 3))
            if abs(c) > 1e-10
                coeffs[ℓ + 2] = c
            end
        end

        # ℓ-2 coupling (if valid)
        if ℓ - 2 >= abs(m) && ℓ - 2 >= 0
            c = 0.5 * m^2 / ((2ℓ - 1) * (2ℓ + 1))
            if abs(c) > 1e-10
                coeffs[ℓ - 2] = c
            end
        end
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

Compute coupling coefficients for sinθ × ∂Y_ℓm/∂θ expansion in spherical harmonics.

Using the recurrence relation for associated Legendre functions:
    sinθ ∂Y_ℓm/∂θ = A⁺_ℓm Y_{ℓ+1,m} + A⁻_ℓm Y_{ℓ-1,m} + (diagonal correction)

Returns (A_minus, A_plus, A_diag) where:
- A_minus: coefficient for coupling to ℓ-1
- A_plus: coefficient for coupling to ℓ+1
- A_diag: diagonal contribution (usually small)
"""
function theta_derivative_coupling(ℓ::Int, m::Int)
    # Exact, finite two-term identity (verified numerically against the
    # orthonormal Y_lm; see basic_state.jl:2007-2008 and _dtheta_sphere_projection):
    #     sinθ ∂Y_ℓm/∂θ = ℓ α⁺_ℓ Y_{ℓ+1,m} − (ℓ+1) α⁻_ℓ Y_{ℓ-1,m}
    # with α⁺_ℓ = √[((ℓ+1)²−m²)/((2ℓ+1)(2ℓ+3))], α⁻_ℓ = √[(ℓ²−m²)/((2ℓ−1)(2ℓ+1))]
    # (these α are the orthonormal recurrence coefficients, == sin_theta_coupling).

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

    # Diagonal contribution (from m cotθ term, averaged)
    A_diag = 0.0  # This averages to zero over the sphere for m ≠ 0

    return (A_minus, A_plus, A_diag)
end


"""
    inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)

Compute the Gaunt-like integral ⟨Y_Lm | 1/sinθ | Y_ℓm⟩.

The 1/sinθ factor couples modes with |L - ℓ| even (0, 2, 4, ...).
The dominant contribution is diagonal (L = ℓ).

Returns the coupling coefficient. Non-zero only for specific L values.
"""
function inv_sin_theta_gaunt(L::Int, ℓ::Int, m::Int)
    am = abs(m)
    (L < am || ℓ < am) && return 0.0
    # Opposite parity ⇒ integrand odd under x→−x ⇒ exactly zero.
    (L + ℓ) % 2 != 0 && return 0.0

    # Exact ⟨Y_Lm | 1/sinθ | Y_ℓm⟩ = ∫₀^π P̄_Lm(cosθ) P̄_ℓm(cosθ) dθ — the 1/sinθ
    # cancels the sinθ of dΩ. This is the Gauss–Chebyshev G(a,b) used by
    # _dtheta_sphere_projection, so it carries the SAME orthonormal SH convention
    # as the cosθ / sinθ∂θ operators. Couples all same-parity L (0, ±2, ±4, …).
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
enforces both mechanical boundaries. Use a basic-state constructor to retain
the divergence-free vector potentials.
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
    flow = _steady_mean_flow(theta,r,D1,D2,E,Ra,Pr,lmax_bs,abs(m_bs);
                             mechanical_bc=mechanical_bc)
    ur,uθ,uφ,dur,duθ,_ = _mean_flow_components(flow)
    for (target,source) in zip((ur_coeffs,utheta_coeffs,uphi_coeffs,dur_dr_coeffs,dutheta_dr_coeffs),
                               (ur,uθ,uφ,dur,duθ))
        for (k,v) in source
            abs(k[2])==abs(m_bs) && (target[k]=v)
        end
    end
    return nothing
end


"""
Compatibility wrapper using the complete steady viscous solve. The historical
diagonal approximation is no longer used. Prefer the basic-state constructors.
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

    flow = _steady_mean_flow(theta_coeffs,r,D1,D1*D1,E,Ra,Pr,lmax_bs,mmax_bs;
                             mechanical_bc=mechanical_bc)
    ur,uθ,_,dur,duθ,_ = _mean_flow_components(flow)
    for (target,source) in zip((ur_coeffs,utheta_coeffs,dur_dr_coeffs,dutheta_dr_coeffs),(ur,uθ,dur,duθ))
        empty!(target); merge!(target,source)
    end
    return nothing
end


"""
Populate component projections from a divergence-free toroidal–poloidal
Stokes–Coriolis solve. `use_full_coupling` is retained for compatibility; both
values use the full solve. `include_meridional=false` explicitly clears the
meridional outputs and does not construct a complete physical mean flow.
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

    if !include_meridional
        for m in -mmax_bs:mmax_bs,l in abs(m):lmax_bs
            for target in (ur_coeffs,utheta_coeffs,dur_dr_coeffs,dutheta_dr_coeffs)
                target[(l,m)]=zeros(T,length(r))
            end
        end
        return nothing
    end
    # The legacy full/simple switch no longer selects an inconsistent solve.
    flow = _steady_mean_flow(theta_coeffs,r,D1,D2,E,Ra,Pr,lmax_bs,mmax_bs;
                             mechanical_bc=mechanical_bc)
    ur,uθ,uφ,dur,duθ,_ = _mean_flow_components(flow)
    for (target,source) in zip((ur_coeffs,utheta_coeffs,uphi_coeffs,dur_dr_coeffs,dutheta_dr_coeffs),
                               (ur,uθ,uφ,dur,duθ))
        empty!(target); merge!(target,source)
    end
    return nothing
end


# =============================================================================
#  Full Advection Term with All Velocity Components
# =============================================================================

"""
Project u·∇T directly from real ±m scalar-component coefficients in the public
no-factorial convention. Uniform T therefore has exactly zero advection even
when the supplied scalar velocity projections have a divergence truncation
error. Constructed states use vector potentials internally for thermal updates.
"""
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
    # The basic state is stored in the no-factorial normalization; vecsh_advection
    # works in the orthonormal (full N_ℓm) basis. Convert inputs in, output back
    # (m=0 conversion is identity, so the validated axisymmetric path is unchanged).
    F_orth = vecsh_advection(
        _sh_rescale(theta_coeffs, +1), _sh_rescale(dtheta_dr_coeffs, +1),
        _sh_rescale(ur_coeffs, +1), _sh_rescale(dur_dr_coeffs, +1),
        _sh_rescale(utheta_coeffs, +1), _sh_rescale(uphi_coeffs, +1),
        lmax_bs, mmax_bs, r)
    return _sh_rescale(F_orth, -1)
end


"""
Iterate the steady Stokes–Coriolis and thermal advection-diffusion equations.
Uses rotation-time diffusivity E/Pr, both mechanical boundaries, real ±m
harmonics, implicit thermal transport, and relaxed Picard updates. Velocity
is recomputed from the returned temperature, including on nonconvergence.

Returns `(bs, info)`; check `info.converged` before using the result as a steady
thermal state. `info.thermal_residual` is the maximum interior spectral energy-
equation residual. Momentum inertia is neglected; this is a weak-inertia model,
not a nonlinear Navier–Stokes equilibrium solver.
"""
function nonaxisymmetric_basic_state_selfconsistent(
    cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
    lmax_bs::Int, mmax_bs::Int, amplitudes::Dict{Tuple{Int,Int}, <:Real};
    mechanical_bc::Symbol = :no_slip,
    thermal_bc::Symbol = :fixed_temperature,
    outer_fluxes::Dict{Tuple{Int,Int}, <:Real} = Dict{Tuple{Int,Int}, T}(),
    max_iterations::Int = 20,
    tolerance::T = T(1e-8),
    verbose::Bool = false,
    coupled_thermal_wind::Bool = true
) where T<:Real

    max_iterations > 0 || throw(ArgumentError("max_iterations must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    initial = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes;
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc, outer_fluxes=outer_fluxes,
        coupled_thermal_wind=coupled_thermal_wind)
    theta = deepcopy(initial.theta_coeffs)
    r = cd.x; N = length(r); D = Matrix(cd.D1); D2 = Matrix(cd.D2)
    # Preserve the actual prescribed boundary values, including homogeneous
    # Neumann conditions on sine modes generated during the iteration.
    bc = Dict(k => (v[1], thermal_bc==:fixed_temperature ? v[end] :
        initial.dtheta_dr_coeffs[k][end], thermal_bc) for (k,v) in theta)
    flow = initial.flow
    residual_history = T[]; converged = false
    for iter in 1:max_iterations
        candidate = _mean_temperature_step(flow,D,D2,E/Pr,bc)
        defect = maximum(maximum(abs,candidate[k]-v) for (k,v) in theta)
        push!(residual_history,defect)
        # Relax the nonlinear coupling between temperature and the Stokes flow.
        for (k,v) in theta
            v .= (v .+ candidate[k])./2
        end
        flow = _steady_mean_flow(theta,r,D,D2,E,Ra,Pr,lmax_bs,mmax_bs;
                                  mechanical_bc=mechanical_bc)
        verbose && println("  Iteration $iter: temperature fixed-point defect = $defect")
        if defect < tolerance
            converged=true
            break
        end
    end
    dtheta = Dict(k=>D*v for (k,v) in theta)
    ur,uθ,uφ,dur,duθ,duφ = _mean_flow_components(flow)
    bs = BasicState3D(lmax_bs=lmax_bs,mmax_bs=mmax_bs,Nr=N,r=r,
        theta_coeffs=theta,dtheta_dr_coeffs=dtheta,ur_coeffs=ur,utheta_coeffs=uθ,
        uphi_coeffs=uφ,dur_dr_coeffs=dur,dutheta_dr_coeffs=duθ,duphi_dr_coeffs=duφ,flow=flow)
    adv = _mean_flow_advection(theta,dtheta,flow)
    thermal_residual = maximum(maximum(abs, ((E/Pr).*(D2*v+2 .* (D*v)./r-
        k[1]*(k[1]+1).*v./r.^2)-adv[k])[2:end-1]) for (k,v) in theta)
    info = (iterations=length(residual_history),converged=converged,
            residual_history=residual_history,thermal_residual=thermal_residual)
    return bs,info
end


"""
Construct a thermally self-consistent viscous mean flow from symbolic boundary
conditions. Both axisymmetric and nonaxisymmetric forcing iterate thermal
advection-diffusion with the complete Stokes–Coriolis velocity. Returns the
basic state and convergence information; pure conduction returns `nothing`
for the information. See `nonaxisymmetric_basic_state_selfconsistent`.
"""
function basic_state_selfconsistent(cd, χ::Real, E::Real, Ra::Real, Pr::Real;
                                    temperature_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    flux_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                                    mechanical_bc::Symbol=:no_slip,
                                    lmax_bs::Union{Nothing, Int}=nothing,
                                    max_iterations::Int=20,
                                    tolerance::Float64=1e-8,
                                    verbose::Bool=false,
                                    coupled_thermal_wind::Bool=true)

    # Validate: can't have both temperature_bc and flux_bc
    if temperature_bc !== nothing && flux_bc !== nothing
        error("Cannot specify both temperature_bc and flux_bc. Choose one.")
    end

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

    if is_axisymmetric(bc)
        bs, info = nonaxisymmetric_basic_state_selfconsistent(cd,T(χ),T(E),T(Ra),T(Pr),
            _lmax,0,to_dict(bc); mechanical_bc=mechanical_bc,thermal_bc=thermal_bc,
            max_iterations=max_iterations,tolerance=T(tolerance),verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind)
        return _axisymmetric_state(bs),info
    end

    # Non-axisymmetric: use self-consistent solver
    amplitudes = to_dict(bc)

    if thermal_bc == :fixed_temperature
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, bc_mmax, amplitudes;
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_temperature,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind
        )
    else  # fixed_flux
        return nonaxisymmetric_basic_state_selfconsistent(
            cd, T(χ), T(E), T(Ra), T(Pr),
            _lmax, bc_mmax,
            Dict{Tuple{Int,Int},T}();  # empty amplitudes
            mechanical_bc=mechanical_bc,
            thermal_bc=:fixed_flux,
            outer_fluxes=amplitudes,
            max_iterations=max_iterations,
            tolerance=T(tolerance),
            verbose=verbose,
            coupled_thermal_wind=coupled_thermal_wind
        )
    end
end
