# =============================================================================
#  Reconstruction of velocity and temperature fields from spectral coefficients
#
#  Provides functions to convert poloidal/toroidal potentials to physical
#  velocity components on a meridional (r, θ) grid for a single azimuthal mode.
#
#  This module supports:
#  - Direct physical-space computation (potentials_to_velocity)
#  - Biglobal stability analysis eigenvector reconstruction
#  - Triglobal stability analysis eigenvector reconstruction
# =============================================================================

using LinearAlgebra

"""
    potentials_to_velocity(P, T; Dr, Dθ, Lθ, r, sintheta, m)

Compute velocity components `(u_r, u_θ, u_φ)` from poloidal and toroidal
potentials on a meridional (r, θ) grid for a single azimuthal mode m.

The velocity is reconstructed using the poloidal-toroidal decomposition:

    u = ∇×∇×(P r̂) + ∇×(T r̂)

which gives (see MagIC documentation, Chandrasekhar 1961):
    u_r = ℓ(ℓ+1) P / r² = -L²P / r²
    u_θ = (1/r) ∂²P/∂r∂θ + (im/(r sinθ)) T
    u_φ = (im/(r sinθ)) ∂P/∂r - (1/r) ∂T/∂θ

where L² = -ℓ(ℓ+1) is the angular Laplacian eigenvalue.

Note: The horizontal Laplacian Δ_H = L²/r², and for spherical harmonics,
L² Y_ℓm = -ℓ(ℓ+1) Y_ℓm.

# Arguments
- `P::AbstractMatrix` - Poloidal potential on (Nr, Nθ) grid
- `T::AbstractMatrix` - Toroidal potential on (Nr, Nθ) grid
- `Dr` - Radial differentiation matrix (Nr × Nr)
- `Dθ` - Colatitude differentiation matrix (Nθ × Nθ)
- `Lθ` - Angular Laplacian operator (Nθ × Nθ)
- `r::AbstractVector` - Radial coordinates (length Nr)
- `sintheta::AbstractVector` - sin(θ) values (length Nθ)
- `m::Int` - Azimuthal wavenumber

# Returns
- `(u_r, u_θ, u_φ)` - Velocity components as (Nr, Nθ) complex matrices

# Example
```julia
# Set up grid and operators
Nr, Nθ = 64, 128
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
Dr = cd.D1
# ... set up Dθ, Lθ, sintheta ...

# Reconstruct velocity from eigenvector
u_r, u_θ, u_φ = potentials_to_velocity(P, T; Dr=Dr, Dθ=Dθ, Lθ=Lθ,
                                        r=cd.x, sintheta=sintheta, m=m)
```
"""
function potentials_to_velocity(P::AbstractMatrix,
                                Tor::AbstractMatrix;
                                Dr,
                                Dθ,
                                Lθ,
                                r::AbstractVector,
                                sintheta::AbstractVector,
                                m::Int)
    Nr, Nθ = size(P)
    size(Tor) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    size(Dr) == (Nr, Nr) || throw(DimensionMismatch(
        "Dr must be $Nr × $Nr, got $(size(Dr))"))
    size(Dθ) == (Nθ, Nθ) || throw(DimensionMismatch(
        "Dθ must be $Nθ × $Nθ, got $(size(Dθ))"))
    size(Lθ) == (Nθ, Nθ) || throw(DimensionMismatch(
        "Lθ must be $Nθ × $Nθ, got $(size(Lθ))"))
    length(r) == Nr || throw(DimensionMismatch(
        "r must have length $Nr, got $(length(r))"))
    length(sintheta) == Nθ || throw(DimensionMismatch(
        "sintheta must have length $Nθ, got $(length(sintheta))"))

    CT = promote_type(eltype(P), eltype(Tor), Complex{eltype(r)},
                      Complex{eltype(sintheta)})
    ur = similar(P, CT, Nr, Nθ)
    uθ = similar(P, CT, Nr, Nθ)
    uφ = similar(P, CT, Nr, Nθ)
    dP_dr = similar(P, CT, Nr, Nθ)

    # Use the three return arrays plus one scratch array as workspace.  The
    # final radius/sinθ scaling is streamed below to avoid allocating another
    # full-size matrix for each broadcasted factor.
    mul!(ur, P, transpose(Lθ))       # L² P (angular Laplacian of P)
    mul!(dP_dr, Dr, P)               # ∂P/∂r
    mul!(uθ, dP_dr, transpose(Dθ))   # ∂²P/∂r∂θ
    mul!(uφ, Tor, transpose(Dθ))     # ∂T/∂θ

    im_m = CT(im * m)
    @inbounds for j in 1:Nθ
        inv_sinθ = inv(sintheta[j])
        for i in 1:Nr
            inv_r = inv(r[i])
            inv_r_sinθ = inv_r * inv_sinθ
            ur[i, j] = -ur[i, j] * inv_r * inv_r
            uθ[i, j] = uθ[i, j] * inv_r + im_m * Tor[i, j] * inv_r_sinθ
            uφ[i, j] = im_m * dP_dr[i, j] * inv_r_sinθ - uφ[i, j] * inv_r
        end
    end

    return ur, uθ, uφ
end


# =============================================================================
#  Spherical Harmonic Utilities
# =============================================================================

"""
    _double_factorial(n)

Compute n!! = n × (n-2) × (n-4) × ... × (1 or 2).
"""
function _double_factorial(n::Int)
    n <= 0 && return 1.0
    result = 1.0
    for k in n:-2:1
        result *= k
    end
    return result
end


"""
    _associated_legendre_table(m, lmax, mu)

Compute table of associated Legendre polynomials P_ℓ^m(μ) for ℓ ∈ [m, lmax].

Uses the standard recurrence relation for numerical stability.

# Arguments
- `m::Int` - Azimuthal order (≥ 0)
- `lmax::Int` - Maximum degree
- `mu::Vector{Float64}` - cos(θ) values

# Returns
- `P::Matrix{Float64}` - P[ℓ-m+1, j] = P_ℓ^m(μ[j])
"""
function _associated_legendre_table(m::Int, lmax::Int, mu::Vector{T}) where {T<:Real}
    nmu = length(mu)
    n_l = lmax - m + 1
    P = zeros(T, n_l, nmu)
    n_l <= 0 && return P

    # Starting value: P_m^m
    if m == 0
        P[1, :] .= one(T)
    else
        Pmm = T((-1.0)^m * _double_factorial(2 * m - 1)) .*
              (one(T) .- mu.^2).^(m / 2)
        P[1, :] .= Pmm
    end

    lmax == m && return P

    # P_{m+1}^m from P_m^m
    P[2, :] .= mu .* (2 * m + 1) .* P[1, :]

    # Recurrence for higher ℓ
    for l in (m + 2):lmax
        idx = l - m + 1
        P[idx, :] .= ((2 * l - 1) .* mu .* P[idx - 1, :] .-
                      (l + m - 1) .* P[idx - 2, :]) ./ (l - m)
    end

    return P
end


"""
    _normalization_table(m, lmax)

Compute spherical harmonic normalization factors for ℓ ∈ [m, lmax].

The fully normalized spherical harmonic is:
    Y_ℓm(θ, φ) = N_ℓm × P_ℓ^m(cos θ) × e^{imφ}

where N_ℓm = √[(2ℓ+1)/(4π) × (ℓ-m)!/(ℓ+m)!]

# Returns
- `N::Vector{Float64}` - N[ℓ-m+1] = N_ℓm
"""
_normalization_table(m::Int, lmax::Int) = _normalization_table(Float64, m, lmax)

function _normalization_table(::Type{T}, m::Int, lmax::Int) where {T<:Real}
    n_l = lmax - m + 1
    N = Vector{T}(undef, n_l)
    for l in m:lmax
        # Compute (l-m)!/(l+m)! iteratively to avoid overflow
        ratio = one(T)
        for k in (l - m + 1):(l + m)
            ratio /= k
        end
        N[l - m + 1] = sqrt(T(2 * l + 1) / (T(4) * T(π)) * ratio)
    end
    return N
end


# =============================================================================
#  Angular Grid and Operators
# =============================================================================

"""
    MeridionalGrid{T<:Real}

Grid and operators for meridional (r, θ) plane reconstruction.

# Fields
- `θ::Vector{T}` - Colatitude values (0 to π)
- `cosθ::Vector{T}` - cos(θ) values
- `sinθ::Vector{T}` - sin(θ) values
- `Dθ::Matrix{T}` - θ-differentiation matrix
- `m::Int` - Azimuthal wavenumber
- `Lθ::Matrix{T}` - Angular Laplacian L² for mode m
- `Ylm::Dict{Int, Vector{Complex{T}}}` - Precomputed Y_ℓm(θ, φ=0)
"""
struct MeridionalGrid{T<:Real}
    θ::Vector{T}
    cosθ::Vector{T}
    sinθ::Vector{T}
    Dθ::Matrix{T}
    m::Int
    Lθ::Matrix{T}
    Ylm::Dict{Int, Vector{Complex{T}}}
    lmax::Int
end


"""
    build_meridional_grid(Nθ, m, lmax; grid_type=:gauss_legendre)

Build a meridional grid with angular operators for velocity reconstruction.

# Arguments
- `Nθ::Int` - Number of θ points
- `m::Int` - Azimuthal wavenumber
- `lmax::Int` - Maximum spherical harmonic degree
- `grid_type::Symbol` - Grid type (:gauss_legendre or :uniform)

# Returns
- `MeridionalGrid` - Grid structure with all operators

# Example
```julia
grid = build_meridional_grid(128, 10, 60)
```
"""
function build_meridional_grid(Nθ::Int, m::Int, lmax::Int;
                                grid_type::Symbol=:gauss_legendre,
                                T::Type{<:Real}=Float64)

    # Generate θ grid
    if grid_type == :gauss_legendre
        # Gauss-Legendre nodes (better for spectral accuracy)
        cosθ, weights = _gauss_legendre_nodes(Nθ)
        cosθ = T.(cosθ)
        θ = acos.(cosθ)
    elseif grid_type == :chebyshev
        # Chebyshev nodes in θ ∈ (0, π)
        k = T.(collect(1:Nθ))
        θ = T(π) .* (T(2) .* k .- one(T)) ./ T(2 * Nθ)
        cosθ = cos.(θ)
    else  # :uniform
        θ = range(T(π) / (2 * Nθ), T(π) - T(π) / (2 * Nθ), length=Nθ)
        θ = collect(θ)
        cosθ = cos.(θ)
    end
    sinθ = sin.(θ)

    # Build differentiation matrix Dθ using finite differences or spectral
    Dθ = _build_theta_derivative_matrix(θ)

    # Build angular Laplacian L² for mode m
    # L² = (1/sinθ) d/dθ (sinθ d/dθ) - m²/sin²θ
    Lθ = _build_angular_laplacian(θ, sinθ, Dθ, m)

    # Precompute spherical harmonics Y_ℓm(θ, φ=0)
    Ylm = _precompute_spherical_harmonics(m, lmax, cosθ)

    return MeridionalGrid{T}(θ, cosθ, sinθ, Dθ, m, Lθ, Ylm, lmax)
end


"""
    _gauss_legendre_nodes(n)

Compute Gauss-Legendre nodes and weights on [-1, 1].
"""
function _gauss_legendre_nodes(n::Int)
    # Newton-Raphson iteration for roots of P_n(x)
    x = zeros(Float64, n)
    w = zeros(Float64, n)

    m = div(n + 1, 2)
    for i in 1:m
        # Initial guess
        z = cos(π * (i - 0.25) / (n + 0.5))
        p1 = 0.0
        p2 = 0.0

        # Newton iteration
        for _ in 1:100
            p1 = 1.0
            p2 = 0.0
            for j in 1:n
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            end
            # p1 is now P_n(z)
            # Derivative: P'_n(z) = n(zP_n - P_{n-1})/(z²-1)
            pp = n * (z * p1 - p2) / (z * z - 1)
            z_old = z
            z = z - p1 / pp
            abs(z - z_old) < 1e-15 && break
        end

        x[i] = -z
        x[n + 1 - i] = z
        w[i] = 2 / ((1 - z * z) * (n * (z * p1 - p2) / (z * z - 1))^2)
        w[n + 1 - i] = w[i]
    end

    return x, w
end


"""
    _build_theta_derivative_matrix(θ)

Build differentiation matrix for θ using Chebyshev-like spectral differentiation.
"""
function _build_theta_derivative_matrix(θ::Vector{T}) where T
    n = length(θ)
    D = zeros(T, n, n)

    for i in 1:n
        for j in 1:n
            if i != j
                D[i, j] = _barycentric_weight(θ, j) /
                          (_barycentric_weight(θ, i) * (θ[i] - θ[j]))
            end
        end
        D[i, i] = -sum(D[i, k] for k in 1:n if k != i)
    end

    return D
end


function _barycentric_weight(θ::Vector{T}, j::Int) where T
    n = length(θ)
    w = one(T)
    for k in 1:n
        if k != j
            w *= (θ[j] - θ[k])
        end
    end
    return 1 / w
end


"""
    _build_angular_laplacian(θ, sinθ, Dθ, m)

Build angular Laplacian operator L² for azimuthal mode m.

L² = (1/sinθ) d/dθ (sinθ d/dθ) - m²/sin²θ

For Y_ℓm, this gives L² Y_ℓm = -ℓ(ℓ+1) Y_ℓm.
"""
function _build_angular_laplacian(θ::Vector{T}, sinθ::Vector{T},
                                   Dθ::Matrix{T}, m::Int) where T
    n = length(θ)
    cosθ = cos.(θ)

    # L² = d²/dθ² + (cosθ/sinθ) d/dθ - m²/sin²θ
    # which is equivalent to (1/sinθ) d/dθ (sinθ d/dθ) - m²/sin²θ

    D2θ = Dθ * Dθ  # Second derivative

    Lθ = D2θ + Diagonal(cosθ ./ sinθ) * Dθ - Diagonal(T(m^2) ./ (sinθ.^2))

    return Lθ
end


"""
    _precompute_spherical_harmonics(m, lmax, cosθ)

Precompute Y_ℓm(θ, φ=0) for ℓ ∈ [m, lmax].
"""
function _precompute_spherical_harmonics(m::Int, lmax::Int, cosθ::Vector{T}) where {T<:Real}
    Plm = _associated_legendre_table(m, lmax, cosθ)
    Nlm = _normalization_table(T, m, lmax)

    Ylm = Dict{Int, Vector{Complex{T}}}()
    for ℓ in m:lmax
        idx = ℓ - m + 1
        # Y_ℓm(θ, φ=0) = N_ℓm × P_ℓ^m(cosθ) × e^{im×0} = N_ℓm × P_ℓ^m(cosθ)
        Ylm[ℓ] = Complex{T}.(Nlm[idx] .* Plm[idx, :])
    end

    return Ylm
end


# =============================================================================
#  Eigenvector Coefficient Extraction
# =============================================================================

"""
    extract_eigenvector_coefficients(eigenvector, op)

Extract poloidal P_ℓm(r), toroidal T_ℓm(r), and temperature Θ_ℓm(r)
coefficients from a stability analysis eigenvector.

# Arguments
- `eigenvector::Vector{Complex}` - Eigenvector from solve_eigenvalue_problem
- `op::LinearStabilityOperator` - The operator used to compute the eigenvector

# Returns
- `P_coeffs::Dict{Int, Vector{Complex}}` - P_ℓm(r) for each ℓ
- `T_coeffs::Dict{Int, Vector{Complex}}` - T_ℓm(r) for each ℓ
- `Θ_coeffs::Dict{Int, Vector{Complex}}` - Θ_ℓm(r) for each ℓ

# Example
```julia
eigenvalues, eigenvectors, op, info = solve_eigenvalue_problem(op)
P, T, Θ = extract_eigenvector_coefficients(eigenvectors[1], op)
```
"""
function extract_eigenvector_coefficients(eigenvector::AbstractVector{<:Complex},
                                           op)
    CT = eltype(eigenvector)
    P_coeffs = Dict{Int, Vector{CT}}()
    T_coeffs = Dict{Int, Vector{CT}}()
    Θ_coeffs = Dict{Int, Vector{CT}}()

    # Extract poloidal coefficients
    for ℓ in op.l_sets[:P]
        idx = op.index_map[(ℓ, :P)]
        P_coeffs[ℓ] = eigenvector[idx]
    end

    # Extract toroidal coefficients
    for ℓ in op.l_sets[:T]
        idx = op.index_map[(ℓ, :T)]
        T_coeffs[ℓ] = eigenvector[idx]
    end

    # Extract temperature coefficients
    for ℓ in op.l_sets[:Θ]
        idx = op.index_map[(ℓ, :Θ)]
        Θ_coeffs[ℓ] = eigenvector[idx]
    end

    return P_coeffs, T_coeffs, Θ_coeffs
end


# =============================================================================
#  Spectral to Physical Space Synthesis
# =============================================================================

"""
    spectral_to_physical(coeffs, grid, Nr)

Transform spectral coefficients to physical (r, θ) space.

Computes: f(r, θ) = Σ_ℓ f_ℓm(r) × Y_ℓm(θ, φ=0)

# Arguments
- `coeffs::Dict{Int, Vector{Complex}}` - Spectral coefficients f_ℓm(r)
- `grid::MeridionalGrid` - Meridional grid with precomputed Y_ℓm
- `Nr::Int` - Number of radial points

# Returns
- `f_phys::Matrix{ComplexF64}` - f(r, θ) on (Nr, Nθ) grid
"""
function spectral_to_physical(coeffs::AbstractDict{Int, <:AbstractVector{<:Complex}},
                               grid::MeridionalGrid,
                               Nr::Int)
    Nθ = length(grid.θ)
    CT = _coefficient_eltype(coeffs, grid)
    f_phys = zeros(CT, Nr, Nθ)

    for (ℓ, f_lm) in coeffs
        if haskey(grid.Ylm, ℓ)
            Ylm = grid.Ylm[ℓ]
            # f_phys(r, θ) += f_ℓm(r) × Y_ℓm(θ)
            for j in 1:Nθ
                y = CT(Ylm[j])
                @views @. f_phys[:, j] += f_lm * y
            end
        end
    end

    return f_phys
end

function _coefficient_eltype(coeffs::AbstractDict)
    for coeff in values(coeffs)
        return eltype(coeff)
    end
    return ComplexF64
end

function _coefficient_eltype(coeffs::AbstractDict, grid::MeridionalGrid{T}) where {T<:Real}
    for coeff in values(coeffs)
        return eltype(coeff)
    end
    return _coefficient_eltype_from_valtype(valtype(typeof(coeffs)), Complex{T})
end

_coefficient_eltype_from_valtype(::Type{<:AbstractVector{CT}}, default) where {CT<:Complex} = CT
_coefficient_eltype_from_valtype(::Type, default) = default


# =============================================================================
#  High-Level Velocity Reconstruction from Eigenvectors
# =============================================================================

"""
    eigenvector_to_velocity(eigenvector, op; Nθ=nothing, grid=nothing)

Reconstruct velocity components from a stability analysis eigenvector.

This is the main high-level function for velocity reconstruction from
biglobal stability analysis results.

# Arguments
- `eigenvector::Vector{Complex}` - Eigenvector from solve_eigenvalue_problem
- `op::LinearStabilityOperator` - The operator used to compute eigenvector
- `Nθ::Int` - Number of θ points (default: 2 × lmax)
- `grid::MeridionalGrid` - Pre-built grid (optional, for repeated calls)

# Returns
- `ur::Matrix{ComplexF64}` - Radial velocity u_r(r, θ)
- `uθ::Matrix{ComplexF64}` - Colatitudinal velocity u_θ(r, θ)
- `uφ::Matrix{ComplexF64}` - Azimuthal velocity u_φ(r, θ)
- `grid::MeridionalGrid` - The grid used (for reuse)

# Example
```julia
# Solve eigenvalue problem
params = OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10, lmax=60, Nr=64)
op = LinearStabilityOperator(params)
eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op)

# Reconstruct velocity of fastest-growing mode
ur, uθ, uφ, grid = eigenvector_to_velocity(eigenvectors[1], op)

# Plot radial velocity
using Plots
heatmap(grid.θ, op.r, real.(ur), xlabel="θ", ylabel="r", title="u_r")
```
"""
function eigenvector_to_velocity(eigenvector::AbstractVector{<:Complex}, op;
                                  Nθ::Union{Int, Nothing}=nothing,
                                  grid::Union{MeridionalGrid, Nothing}=nothing)
    # Keep the public keyword API convenient while dispatching into typed helper
    # methods.  That function barrier preserves concrete return inference for
    # callers that reconstruct many modes.
    if grid === nothing
        return _eigenvector_to_velocity_default_grid(eigenvector, op, Nθ)
    else
        return _eigenvector_to_velocity(eigenvector, op, grid)
    end
end

function _eigenvector_to_velocity_default_grid(eigenvector::AbstractVector{<:Complex},
                                               op,
                                               ::Nothing)
    m = op.params.m
    lmax = op.params.lmax
    Nθ_use = 2 * lmax
    GT = typeof(op.params.E)
    # The assertion fixes the grid parameter for inference after construction.
    grid = build_meridional_grid(Nθ_use, m, lmax; T=GT)::MeridionalGrid{GT}
    return _eigenvector_to_velocity(eigenvector, op, grid)
end

function _eigenvector_to_velocity_default_grid(eigenvector::AbstractVector{<:Complex},
                                               op,
                                               Nθ::Int)
    m = op.params.m
    lmax = op.params.lmax
    Nθ_use = Nθ
    GT = typeof(op.params.E)
    # Mirror the default-grid path so explicit Nθ calls stay type-stable too.
    grid = build_meridional_grid(Nθ_use, m, lmax; T=GT)::MeridionalGrid{GT}
    return _eigenvector_to_velocity(eigenvector, op, grid)
end

function _eigenvector_to_velocity(eigenvector::AbstractVector{<:Complex},
                                  op,
                                  grid::MeridionalGrid{GT}) where {GT<:Real}
    m = op.params.m
    Nr = op.params.Nr
    r = op.r
    Dr = op.cd.D1

    # Extract spectral coefficients
    P_coeffs, T_coeffs, _ = extract_eigenvector_coefficients(eigenvector, op)

    ur,uθ,uφ=_onset_velocity_from_coefficients(P_coeffs,T_coeffs,r,Dr,grid,m)

    return ur, uθ, uφ, grid
end

"""Reconstruct onset's rP/rT potentials and Y_lm/√(2l+1) angular convention."""
function _onset_velocity_from_coefficients(Pcoeff,Tcoeff,r,Dr,grid::MeridionalGrid{T},m) where T
    CT=promote_type(_coefficient_eltype(Pcoeff,grid),_coefficient_eltype(Tcoeff,grid))
    Nr=length(r); Nθ=length(grid.θ); a=abs(m)
    L=max(maximum(keys(Pcoeff);init=a),maximum(keys(Tcoeff);init=a))
    g=SHGrid{T}(L,a,grid.cosθ,zeros(T,Nθ),T[0],
        Dict(k=>_associated_legendre_table(k,L,grid.cosθ) for k in 0:min(a+1,L)),
        Dict(k=>_normalization_table(T,k,L) for k in 0:a))
    ur=zeros(CT,Nr,Nθ); uθ=similar(ur); uφ=similar(ur)
    fill!(uθ,0); fill!(uφ,0)
    for (l,p) in Pcoeff
        y,h,v=_coupling_harmonic(g,l,m); norm=inv(sqrt(T(2l+1)))
        dp=Dr*p; q=l*(l+1)
        for j in 1:Nθ, i in 1:Nr
            ur[i,j]+=norm*q*p[i]/r[i]*y[j]
            uθ[i,j]+=norm*(dp[i]+p[i]/r[i])*h[j]
            uφ[i,j]+=norm*(dp[i]+p[i]/r[i])*v[j]
        end
    end
    for (l,t) in Tcoeff
        y,h,v=_coupling_harmonic(g,l,m); norm=inv(sqrt(T(2l+1)))
        for j in 1:Nθ, i in 1:Nr
            uθ[i,j]+=norm*t[i]*v[j]
            uφ[i,j]-=norm*t[i]*h[j]
        end
    end
    ur,uθ,uφ
end


"""
    eigenvector_to_velocity_triglobal(eigenvector, problem;
                                       Nθ=nothing, Nφ=nothing, φ_slice=nothing)

Reconstruct velocity from a triglobal stability analysis eigenvector.

For triglobal analysis, the eigenvector contains multiple coupled azimuthal
modes m. This function either:
1. Returns velocity at a fixed φ slice (2D output)
2. Returns full 3D velocity field (expensive)

# Arguments
- `eigenvector::Vector{Complex}` - Eigenvector from solve_triglobal_eigenvalue_problem
- `problem::CoupledModeProblem` - The problem structure from setup_coupled_mode_problem
- `Nθ::Int` - Number of θ points (default: 2 × lmax)
- `Nφ::Int` - Number of φ points for 3D output (default: nothing → use φ_slice)
- `φ_slice::Real` - Fixed azimuthal angle for 2D slice (default: 0)

# Returns (2D mode, when φ_slice is specified)
- `ur::Matrix{ComplexF64}` - u_r(r, θ) at φ = φ_slice
- `uθ::Matrix{ComplexF64}` - u_θ(r, θ) at φ = φ_slice
- `uφ::Matrix{ComplexF64}` - u_φ(r, θ) at φ = φ_slice

# Returns (3D mode, when Nφ is specified)
- `ur::Array{ComplexF64, 3}` - u_r(r, θ, φ)
- `uθ::Array{ComplexF64, 3}` - u_θ(r, θ, φ)
- `uφ::Array{ComplexF64, 3}` - u_φ(r, θ, φ)

# Example
```julia
# Solve triglobal problem
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(params)

# Get velocity at φ = 0 slice
ur, uθ, uφ = eigenvector_to_velocity_triglobal(eigenvectors[:, 1], problem)

# Get full 3D velocity (more expensive)
ur, uθ, uφ = eigenvector_to_velocity_triglobal(eigenvectors[:, 1], problem; Nφ=64)
```
"""
function eigenvector_to_velocity_triglobal(eigenvector::AbstractVector{<:Complex},
                                            problem;
                                            Nθ::Union{Int, Nothing}=nothing,
                                            Nφ::Union{Int, Nothing}=nothing,
                                            φ_slice::Union{Real, Nothing}=nothing)
    params = problem.params
    lmax = params.lmax
    Nr = params.Nr

    # Default: 2D slice at φ = 0
    if Nφ === nothing && φ_slice === nothing
        φ_slice = 0.0
    end

    Nθ_use = Nθ === nothing ? 2 * lmax : Nθ

    if Nφ !== nothing
        # Full 3D reconstruction
        return _triglobal_velocity_3d(eigenvector, problem, Nθ_use, Nφ)
    else
        # 2D slice at fixed φ
        return _triglobal_velocity_slice(eigenvector, problem, Nθ_use, φ_slice)
    end
end


"""
    _triglobal_velocity_slice(eigenvector, problem, Nθ, φ)

Compute velocity at a fixed φ slice for triglobal eigenvector.
"""
function _triglobal_velocity_slice(eigenvector::AbstractVector{<:Complex},
                                    problem, Nθ::Int, φ::Real)
    params = problem.params
    lmax = params.lmax
    Nr = params.Nr
    m_range = problem.m_range

    # Get radial grid from first mode's operator
    # (all modes share the same radial discretization)
    first_m = first(m_range)
    χ = params.χ

    # Build radial grid
    cd = _build_chebyshev_grid(Nr, χ, 1.0)
    r = cd.x
    Dr = cd.D1

    # Initialize velocity accumulators
    CT = eltype(eigenvector)
    ur_total = zeros(CT, Nr, Nθ)
    uθ_total = zeros(CT, Nr, Nθ)
    uφ_total = zeros(CT, Nr, Nθ)

    # Process each azimuthal mode
    for m in m_range
        # Build grid for this m
        grid_m = build_meridional_grid(Nθ, abs(m), lmax; T=typeof(params.E))

        # Extract coefficients for this m
        P_m, T_m = _extract_mode_coefficients(eigenvector, problem, m)

        # Skip if empty
        isempty(P_m) && isempty(T_m) && continue

        # Transform to physical θ-space
        ur_m,uθ_m,uφ_m=_onset_velocity_from_coefficients(P_m,T_m,r,Dr,grid_m,m)

        # Add contribution with e^{imφ} phase factor
        phase = CT(exp(im * m * φ))
        @. ur_total += ur_m * phase
        @. uθ_total += uθ_m * phase
        @. uφ_total += uφ_m * phase
    end

    return ur_total, uθ_total, uφ_total
end


"""
    _triglobal_velocity_3d(eigenvector, problem, Nθ, Nφ)

Compute full 3D velocity field for triglobal eigenvector.
"""
function _triglobal_velocity_3d(eigenvector::AbstractVector{<:Complex},
                                 problem, Nθ::Int, Nφ::Int)
    params = problem.params
    lmax = params.lmax
    Nr = params.Nr
    m_range = problem.m_range

    # Build radial grid
    χ = params.χ
    cd = _build_chebyshev_grid(Nr, χ, 1.0)
    r = cd.x
    Dr = cd.D1

    # Build φ grid
    φ = range(0, 2π, length=Nφ+1)[1:Nφ]

    # Initialize 3D velocity arrays
    CT = eltype(eigenvector)
    ur = zeros(CT, Nr, Nθ, Nφ)
    uθ = zeros(CT, Nr, Nθ, Nφ)
    uφ = zeros(CT, Nr, Nθ, Nφ)

    # Process each azimuthal mode
    for m in m_range
        # Build grid for this m
        grid_m = build_meridional_grid(Nθ, abs(m), lmax; T=typeof(params.E))

        # Extract coefficients for this m
        P_m, T_m = _extract_mode_coefficients(eigenvector, problem, m)

        # Skip if empty
        isempty(P_m) && isempty(T_m) && continue

        # Transform to physical θ-space
        ur_m,uθ_m,uφ_m=_onset_velocity_from_coefficients(P_m,T_m,r,Dr,grid_m,m)

        # Add to 3D field with e^{imφ} phase
        for k in 1:Nφ
            phase = CT(exp(im * m * φ[k]))
            @views @. ur[:, :, k] += ur_m * phase
            @views @. uθ[:, :, k] += uθ_m * phase
            @views @. uφ[:, :, k] += uφ_m * phase
        end
    end

    return ur, uθ, uφ
end


# Per-problem reconstruction data, keyed weakly by the problem object. Each entry
# remembers the `params` it was built from: `CoupledModeProblem` is mutable, so a
# replaced `problem.params` (new lmax, Nr, symmetry, …) must not reuse stale
# reductions. The lock makes the cache safe to use from several threads.
const _mode_reconstruction_cache = WeakKeyDict{Any, Any}()
const _mode_reconstruction_lock = ReentrantLock()

function _mode_reconstruction(problem, m_abs::Int)
    T = typeof(problem.params.E)
    CacheType = Dict{Int, NamedTuple{(:op, :reduction),
        Tuple{LinearStabilityOperator{T, Nothing}, ConstraintReduction{T}}}}
    return lock(_mode_reconstruction_lock) do
        entry = get(_mode_reconstruction_cache, problem, nothing)
        if entry === nothing || entry.params !== problem.params
            entry = (params = problem.params, modes = CacheType())
            _mode_reconstruction_cache[problem] = entry
        end
        # The outer cache is intentionally `Any` because `WeakKeyDict` stores many
        # problem parameterizations; this assertion recovers the concrete cache type
        # for the hot reconstruction path.
        cache = entry.modes::CacheType

        get!(cache, m_abs) do
            params_tri = problem.params
            params_m = OnsetParams(
                E = params_tri.E,
                Pr = params_tri.Pr,
                Ra = params_tri.Ra,
                χ = params_tri.χ,
                m = m_abs,
                lmax = params_tri.lmax,
                Nr = params_tri.Nr,
                mechanical_bc = params_tri.mechanical_bc,
                thermal_bc = params_tri.thermal_bc,
                equatorial_symmetry = params_tri.equatorial_symmetry,
                basic_state = nothing
            )
            op = LinearStabilityOperator(params_m)
            # The tau-row nullspace depends only on the boundary conditions, so it
            # comes straight from the BC formulas without assembling A.
            reduction = _constraint_reduction_from_subblocks(op)
            (op = op, reduction = reduction)
        end
    end
end

"""
    _extract_mode_coefficients(eigenvector, problem, m)

Extract P_ℓm and T_ℓm coefficients for a specific mode m from triglobal eigenvector.
"""
function _extract_mode_coefficients(eigenvector::AbstractVector{<:Complex},
                                     problem, m::Int)
    Nr = problem.params.Nr

    if !haskey(problem.block_indices, m)
        CT = eltype(eigenvector)
        return Dict{Int, Vector{CT}}(), Dict{Int, Vector{CT}}()
    end

    block_range = problem.block_indices[m]
    block_vec = eigenvector[block_range]
    reconstruction = _mode_reconstruction(problem, abs(m))
    op = reconstruction.op
    full_vec = _reconstruct_full_vector(reconstruction.reduction, block_vec)

    CT = eltype(full_vec)
    P_coeffs = Dict{Int, Vector{CT}}()
    T_coeffs = Dict{Int, Vector{CT}}()

    for ℓ in op.l_sets[:P]
        P_coeffs[ℓ] = full_vec[op.index_map[(ℓ, :P)]]
    end

    for ℓ in op.l_sets[:T]
        T_coeffs[ℓ] = full_vec[op.index_map[(ℓ, :T)]]
    end

    return P_coeffs, T_coeffs
end


"""
    _build_chebyshev_grid(Nr, ri, ro)

Chebyshev–Gauss–Lobatto grid on `[ri, ro]` and its first-derivative matrix, i.e.
the radial grid of [`ChebyshevDiffn`](@ref) used by the stability operators.
"""
function _build_chebyshev_grid(Nr::Int, ri::T, ro::Real) where {T<:Real}
    cd = ChebyshevDiffn(Nr, T[ri, T(ro)], 1)
    return (x=cd.x, D1=cd.D1)
end


# =============================================================================
#  Convenience Functions
# =============================================================================

# --- Unified perturbation-field reconstruction API (hydrodynamic) -------------

"""
    perturbation_velocity(evec, op::LinearStabilityOperator; Nθ=nothing, grid=nothing)

Reconstruct physical perturbation velocity `(u_r, u_θ, u_φ)` and the meridional
grid from a hydrodynamic stability eigenvector. Thin wrapper over
[`eigenvector_to_velocity`](@ref).
"""
function perturbation_velocity(evec::AbstractVector{<:Complex},
                               op::LinearStabilityOperator; kwargs...)
    ur, uθ, uφ, grid = eigenvector_to_velocity(evec, op; kwargs...)
    return ur, uθ, uφ, op.r, grid    # uniform (components…, r_grid, grid)
end

# --- Temperature scalar synthesis (hydrodynamic) -----------------------------

"""
    perturbation_temperature(evec, op::LinearStabilityOperator; Nθ=nothing, grid=nothing)

Reconstruct the physical perturbation temperature field
`θ(r, θ) = Σ_ℓ Θ_ℓ(r) Y_ℓ^m(θ)/√(2ℓ+1)` on a meridional grid. `Θ_ℓ(r)` are the
temperature collocation values stored in the eigenvector's `:Θ` blocks; like the
velocity potentials they multiply the `Y_ℓ^m/√(2ℓ+1)` harmonics (orthonormal
`Y_ℓ^m`, Condon–Shortley phase). For `m = 0` the operator keeps degrees up to
`lmax + 1`, so the default grid covers `maximum(op.l_sets[:Θ])`.
"""
function perturbation_temperature(evec::AbstractVector{<:Complex},
                                  op::LinearStabilityOperator;
                                  Nθ::Union{Int,Nothing}=nothing,
                                  grid::Union{MeridionalGrid,Nothing}=nothing)
    length(evec) == op.total_dof || throw(DimensionMismatch(
        "eigenvector has length $(length(evec)); the operator has $(op.total_dof) DOFs"))
    m    = op.params.m
    Nr   = op.params.Nr
    T    = typeof(op.params.E)
    l_top = maximum(op.l_sets[:Θ]; init=max(m, op.params.lmax))
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * l_top : Nθ, m, l_top; T=T) : grid

    θfield = zeros(promote_type(eltype(evec), Complex{T}), Nr, length(g.θ))
    for l in op.l_sets[:Θ]
        haskey(g.Ylm, l) || throw(ArgumentError(
            "grid has no Y_$(l)^$(m); build it with lmax ≥ $l_top"))
        Θl = @view evec[op.index_map[(l, :Θ)]]
        ylm = g.Ylm[l]
        norm = inv(sqrt(T(2l + 1)))
        @inbounds for j in eachindex(g.θ), k in 1:Nr
            θfield[k, j] += norm * Θl[k] * ylm[j]
        end
    end
    return θfield, op.r, g
end

"""`perturbation_magnetic` is not defined for hydrodynamic problems (no magnetic field)."""
function perturbation_magnetic(::AbstractVector{<:Complex},
                               ::LinearStabilityOperator; kwargs...)
    error("perturbation_magnetic: hydrodynamic problems have no magnetic field. " *
          "Magnetic reconstruction is only available for MHD results (MHDStabilityOperator).")
end

"""
    kinetic_energy_density(ur, uθ, uφ)

Compute kinetic energy density (1/2)|u|² on the meridional grid.
"""
function kinetic_energy_density(ur::AbstractMatrix, uθ::AbstractMatrix,
                                 uφ::AbstractMatrix)
    return 0.5 .* (abs2.(ur) .+ abs2.(uθ) .+ abs2.(uφ))
end


"""
    meridional_streamfunction(ur, uθ, r, θ, m)

Compute meridional streamfunction ψ from (u_r, u_θ) for visualization.

For axisymmetric flow (m=0): u_r = (1/r²sinθ) ∂ψ/∂θ, u_θ = -(1/r sinθ) ∂ψ/∂r
"""
function meridional_streamfunction(ur::AbstractMatrix, uθ::AbstractMatrix,
                                    r::AbstractVector, θ::AbstractVector,
                                    m::Int)
    if m != 0
        @warn "Meridional streamfunction is only well-defined for m=0"
    end

    Nr, Nθ = size(ur)
    sinθ = sin.(θ)
    RT = promote_type(typeof(real(zero(eltype(ur)))),
                      typeof(real(zero(eltype(uθ)))),
                      eltype(r), eltype(θ))
    CT = Complex{float(RT)}

    # Integrate u_r × r² × sinθ in θ to get ψ
    ψ = zeros(CT, Nr, Nθ)

    for i in 1:Nr
        ψ[i, 1] = 0.0
        for j in 2:Nθ
            Δθ = θ[j] - θ[j-1]
            integrand_prev = ur[i, j-1] * r[i]^2 * sinθ[j-1]
            integrand_curr = ur[i, j] * r[i]^2 * sinθ[j]
            ψ[i, j] = ψ[i, j-1] + 0.5 * (integrand_prev + integrand_curr) * Δθ
        end
    end

    return ψ
end
