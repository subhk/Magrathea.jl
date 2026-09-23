# =============================================================================
#  Real-orthonormal spherical-harmonic transform (cos+sin, ±m) and the
#  vector-harmonic horizontal divergence.
#
#  Grid synthesis/analysis for the mean-state products: the advective form
#  ū·∇T̄ = ū_r ∂_rT̄ + (ū_θ ∂_θT̄ + ū_φ ∂_φT̄/sinθ)/r (`_mean_flow_advection`,
#  which synthesizes ū from its native vector-harmonic potentials), the Coriolis
#  and inertia projections, and `sh_horizontal_divergence` for vector-harmonic
#  components. Tangential components of a solenoidal field are not band-limited
#  scalar harmonics, so products are formed from the vector-harmonic basis rather
#  than from truncated scalar projections. Validated by manufactured-solution
#  tests (see test/sh_transform.jl).
#
#  Convention: real orthonormal SH with full N_ℓm,
#     m>0:  Ȳ_ℓm  = √2 N_ℓ|m| P_ℓ|m|(cosθ) cos(mφ)
#     m<0:  Ȳ_ℓm  = √2 N_ℓ|m| P_ℓ|m|(cosθ) sin(|m|φ)
#     m=0:  Ȳ_ℓ0  =    N_ℓ0  P_ℓ0(cosθ)
#  with ∫ Ȳ_ℓm Ȳ_ℓ'm' dΩ = δ_{ℓℓ'} δ_{mm'},  N_ℓm = √((2ℓ+1)/4π · (ℓ-m)!/(ℓ+m)!).
#
#  Used by the steady vector-potential mean-flow solve and thermal transport.
# =============================================================================

"""Quadrature grid + precomputed associated-Legendre / normalization tables."""
struct SHGrid{T<:Real}
    lmax::Int
    mmax::Int
    μ::Vector{T}              # Gauss-Legendre nodes (cosθ)
    w::Vector{T}              # Gauss-Legendre weights
    φ::Vector{T}              # equispaced azimuthal nodes
    P::Dict{Int,Matrix{T}}    # |m| → P_ℓ|m| table (|m| up to mmax+1 for dθ)
    N::Dict{Int,Vector{T}}    # |m| → N_ℓ|m|
end

# `sh_grid` is a pure function of (lmax, mmax, T) and its result is read-only, so
# memoize it: the self-consistent basic-state Picard loop transforms the mean
# fields on every iteration and would otherwise rebuild the Gauss-Legendre nodes
# and the associated-Legendre tables from scratch every time. The cache is shared
# across threads; see `_locked_get!`.
const _SH_GRID_CACHE = Dict{Tuple{Int,Int,DataType}, Any}()
const _SH_GRID_CACHE_LOCK = ReentrantLock()

function sh_grid(lmax::Int, mmax::Int, ::Type{T}=Float64) where {T<:Real}
    return _locked_get!(_SH_GRID_CACHE, _SH_GRID_CACHE_LOCK, (lmax, mmax, T)) do
        _build_sh_grid(lmax, mmax, T)
    end::SHGrid{T}
end

function _build_sh_grid(lmax::Int, mmax::Int, ::Type{T}) where {T<:Real}
    Nθ = 3 * lmax + 6                      # oversampled for product dealiasing
    Nφ = max(3 * mmax + 4, 6)              # ≥ product (2mmax) + test (mmax) azimuthal degree
    μ64, w64 = _gauss_legendre_nodes(Nθ)
    μ = T.(μ64); w = T.(w64)
    φ = T[2 * T(π) * (k - 1) / Nφ for k in 1:Nφ]
    amax = min(mmax + 1, lmax)
    P = Dict{Int,Matrix{T}}(am => _associated_legendre_table(am, lmax, μ) for am in 0:amax)
    N = Dict{Int,Vector{T}}(am => _normalization_table(T, am, lmax) for am in 0:mmax)
    SHGrid{T}(lmax, mmax, μ, w, φ, P, N)
end

_sh_sinθ(g::SHGrid{T}, j::Int) where {T} = sqrt(one(T) - g.μ[j]^2)
_sh_Nθ(g::SHGrid) = length(g.μ)
_sh_Nφ(g::SHGrid) = length(g.φ)

# azimuthal factor and its φ-derivative
function _sh_φfac(g::SHGrid{T}, m::Int, k::Int) where {T}
    m == 0 && return one(T)
    return m > 0 ? sqrt(T(2)) * cos(m * g.φ[k]) : sqrt(T(2)) * sin(-m * g.φ[k])
end
function _sh_dφfac(g::SHGrid{T}, m::Int, k::Int) where {T}
    m == 0 && return zero(T)
    return m > 0 ? sqrt(T(2)) * (-m) * sin(m * g.φ[k]) :
                   sqrt(T(2)) * (-m) * cos(-m * g.φ[k])
end

"""Real orthonormal SH value Ȳ_ℓm at grid node (j,k)."""
function _sh_Y(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * g.P[am][ℓ - am + 1, j] * _sh_φfac(g, m, k)
end

# pole-safe dP̃_ℓ^am/dθ = ½[P̃^{am+1} - (ℓ+am)(ℓ-am+1) P̃^{am-1}]  (Condon-Shortley)
function _sh_dPdθ(g::SHGrid{T}, ℓ::Int, am::Int, j::Int) where {T}
    ℓ == 0 && return zero(T)
    Pp = (am + 1 <= ℓ && haskey(g.P, am + 1)) ? g.P[am + 1][ℓ - (am + 1) + 1, j] : zero(T)
    Pm = am == 0 ? -g.P[1][ℓ - 1 + 1, j] / (ℓ * (ℓ + 1)) : g.P[am - 1][ℓ - (am - 1) + 1, j]
    T(0.5) * (Pp - T((ℓ + am) * (ℓ - am + 1)) * Pm)
end

"""∂θ Ȳ_ℓm at grid node (j,k)."""
function _sh_dYθ(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * _sh_dPdθ(g, ℓ, am, j) * _sh_φfac(g, m, k)
end

"""(1/sinθ) ∂φ Ȳ_ℓm — pole-safe (P_ℓ^m ∝ sin^m so /sinθ is clean at interior nodes)."""
function _sh_dYφ_over_sin(g::SHGrid{T}, ℓ::Int, m::Int, j::Int, k::Int) where {T}
    am = abs(m)
    (ℓ < am || am == 0 || ℓ > g.lmax || am > g.mmax) && return zero(T)
    g.N[am][ℓ - am + 1] * _sh_P_over_sin(g,ℓ,am,j) * _sh_dφfac(g, m, k)
end

function _sh_P_over_sin(g::SHGrid{T},l,am,j) where T
    s=_sh_sinθ(g,j)
    s>0 && return g.P[am][l-am+1,j]/s
    am==1 ? -g.μ[j]^(l+1)*T(l*(l+1))/2 : zero(T)
end

"""
Grid samples of Ȳ_ℓm, ∂θȲ_ℓm and (1/sinθ)∂φȲ_ℓm for each of `modes` (one column
per mode; rows are nodes with the θ index fastest), together with the quadrature
weights (including dφ) and sinθ, cosθ at each node. Shared by the Coriolis
projection and the implicit mean-temperature transport step.
"""
function _sh_basis_samples(g::SHGrid{T}, modes) where {T}
    nθ = _sh_Nθ(g); nφ = _sh_Nφ(g); n = length(modes)
    Y = zeros(T, nθ * nφ, n); Hθ = similar(Y); Hφ = similar(Y)
    w = zeros(T, nθ * nφ); s = similar(w); c = similar(w)
    for k in 1:nφ, j in 1:nθ
        h = j + (k - 1) * nθ
        w[h] = g.w[j] * 2T(π) / nφ; s[h] = _sh_sinθ(g, j); c[h] = g.μ[j]
        for (a, (l, m)) in enumerate(modes)
            Y[h, a] = _sh_Y(g, l, m, j, k)
            Hθ[h, a] = _sh_dYθ(g, l, m, j, k)
            Hφ[h, a] = _sh_dYφ_over_sin(g, l, m, j, k)
        end
    end
    (Y=Y, Hθ=Hθ, Hφ=Hφ, w=w, sinθ=s, cosθ=c)
end

"""Synthesize a scalar field on the grid from coeffs `Dict{(ℓ,m),value}` using
basis function `Yf` (default `_sh_Y`; pass `_sh_dYθ` etc. for derivative fields).

For the three built-in basis functions the transform is evaluated in separable
form: per-m θ-profiles `Gθ[j,m] = Σ_ℓ a_ℓm N_ℓm θbasis_ℓ(j)` are accumulated
first, then expanded in φ — O(modes·Nθ + M·Nθ·Nφ) instead of O(modes·Nθ·Nφ).
Same math, reordered summation (equal up to roundoff); any other `Yf` falls
back to the per-element path."""
function sh_synthesize(coeffs::AbstractDict{Tuple{Int,Int},T}, g::SHGrid{T};
                       Yf=_sh_Y) where {T}
    sh_synthesize!(zeros(T, _sh_Nθ(g), _sh_Nφ(g)), coeffs, g; Yf=Yf)
end

"""In-place [`sh_synthesize`](@ref): overwrites and returns `f` (Nθ×Nφ)."""
function sh_synthesize!(f::AbstractMatrix{T}, coeffs::AbstractDict{Tuple{Int,Int},T},
                        g::SHGrid{T}; Yf=_sh_Y) where {T}
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    fill!(f, zero(T))
    kind = Yf === _sh_Y ? 1 : Yf === _sh_dYθ ? 2 : Yf === _sh_dYφ_over_sin ? 3 : 0
    if kind == 0   # generic fallback, unchanged semantics
        for ((ℓ, m), a) in coeffs
            (abs(m) > g.mmax || ℓ > g.lmax) && continue
            @inbounds for j in 1:Nθ, k in 1:Nφ
                f[j, k] += a * Yf(g, ℓ, m, j, k)
            end
        end
        return f
    end
    M = 2 * g.mmax + 1
    Gθ = zeros(T, Nθ, M)
    mused = falses(M)
    for ((ℓ, m), a) in coeffs
        am = abs(m)
        (am > g.mmax || ℓ > g.lmax || ℓ < am) && continue
        kind == 3 && am == 0 && continue          # (1/sinθ)∂φ Ȳ_ℓ0 ≡ 0
        Nam = g.N[am]; Pam = g.P[am]
        c = a * Nam[ℓ - am + 1]; row = ℓ - am + 1
        mi = m + g.mmax + 1
        mused[mi] = true
        if kind == 3
            @inbounds for j in 1:Nθ; Gθ[j, mi] += c * _sh_P_over_sin(g,ℓ,am,j); end
        elseif kind == 2
            @inbounds for j in 1:Nθ; Gθ[j, mi] += c * _sh_dPdθ(g, ℓ, am, j); end
        else  # kind == 1
            @inbounds for j in 1:Nθ; Gθ[j, mi] += c * Pam[row, j]; end
        end
    end
    φf = Vector{T}(undef, Nφ)
    for mi in 1:M
        mused[mi] || continue
        m = mi - g.mmax - 1
        if kind == 3
            @inbounds for k in 1:Nφ; φf[k] = _sh_dφfac(g, m, k); end
        else
            @inbounds for k in 1:Nφ; φf[k] = _sh_φfac(g, m, k); end
        end
        @inbounds for k in 1:Nφ, j in 1:Nθ
            f[j, k] += Gθ[j, mi] * φf[k]
        end
    end
    f
end

"""Analyze (project) a scalar grid field onto real-orthonormal SH coefficients.
Separable evaluation: the weighted φ-projection `Fφ[j] = w_j dφ Σ_k f[j,k] φfac_m(k)`
is formed once per m, then each ℓ needs only a θ-sum — O(M·Nθ·Nφ + modes·Nθ)
instead of O(modes·Nθ·Nφ). Same math, reordered summation."""
function sh_analyze(f::AbstractMatrix{T}, g::SHGrid{T}) where {T}
    sh_analyze!(Dict{Tuple{Int,Int},T}(), f, g)
end

"""In-place [`sh_analyze`](@ref): writes every `(ℓ,m)` mode into `coeffs` and returns it."""
function sh_analyze!(coeffs::AbstractDict{Tuple{Int,Int},T}, f::AbstractMatrix{T},
                     g::SHGrid{T}) where {T}
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    dφ = T(2) * T(π) / Nφ
    φf = Vector{T}(undef, Nφ)
    Fφ = Vector{T}(undef, Nθ)
    for m in -g.mmax:g.mmax
        am = abs(m)
        Nam = g.N[am]; Pam = g.P[am]
        @inbounds for k in 1:Nφ; φf[k] = _sh_φfac(g, m, k); end
        @inbounds for j in 1:Nθ
            s = zero(T)
            for k in 1:Nφ; s += f[j, k] * φf[k]; end
            Fφ[j] = s * g.w[j] * dφ
        end
        for ℓ in am:g.lmax
            Nlm = Nam[ℓ - am + 1]; row = ℓ - am + 1
            acc = zero(T)
            @inbounds for j in 1:Nθ; acc += Pam[row, j] * Fφ[j]; end
            coeffs[(ℓ, m)] = Nlm * acc
        end
    end
    coeffs
end

"""
    sh_horizontal_divergence(Vθ, Vφ, g) -> Dict{(ℓ,m),value}

Coefficients of the horizontal divergence ∇_h·V_h of the tangent field with grid
components (Vθ, Vφ). Extracts the spheroidal part s_ℓm = (1/√(ℓ(ℓ+1))) ∫ V_h·∇_hȲ_ℓm dΩ
(the toroidal part is divergence-free) and returns ∇_h·V_h coeffs = -√(ℓ(ℓ+1)) s_ℓm.
"""
function sh_horizontal_divergence(Vθ::AbstractMatrix{T}, Vφ::AbstractMatrix{T},
                                  g::SHGrid{T}) where {T}
    sh_horizontal_divergence!(Dict{Tuple{Int,Int},T}(), Vθ, Vφ, g)
end

"""In-place [`sh_horizontal_divergence`](@ref): writes every `(ℓ≥1,m)` mode into
`div` and returns it. Separable evaluation — per-m weighted φ-projections of
both components are formed once, then each ℓ needs only a θ-sum."""
function sh_horizontal_divergence!(div::AbstractDict{Tuple{Int,Int},T},
                                   Vθ::AbstractMatrix{T}, Vφ::AbstractMatrix{T},
                                   g::SHGrid{T}) where {T}
    Nθ = _sh_Nθ(g); Nφ = _sh_Nφ(g)
    dφ = T(2) * T(π) / Nφ
    φf = Vector{T}(undef, Nφ); dφf = Vector{T}(undef, Nφ)
    Aφ = Vector{T}(undef, Nθ); Bφ = Vector{T}(undef, Nθ)
    for m in -g.mmax:g.mmax
        am = abs(m)
        Nam = g.N[am]; Pam = g.P[am]
        @inbounds for k in 1:Nφ
            φf[k] = _sh_φfac(g, m, k); dφf[k] = _sh_dφfac(g, m, k)
        end
        @inbounds for j in 1:Nθ
            a = zero(T); b = zero(T)
            for k in 1:Nφ
                a += Vθ[j, k] * φf[k]
                b += Vφ[j, k] * dφf[k]
            end
            wdφ = g.w[j] * dφ
            Aφ[j] = a * wdφ
            Bφ[j] = b * wdφ
        end
        for ℓ in max(1, am):g.lmax
            Nlm = Nam[ℓ - am + 1]; row = ℓ - am + 1
            acc = zero(T)
            if am == 0   # (1/sinθ)∂φ Ȳ_ℓ0 ≡ 0: only the θ-component projects
                @inbounds for j in 1:Nθ
                    acc += _sh_dPdθ(g, ℓ, am, j) * Aφ[j]
                end
            else
                @inbounds for j in 1:Nθ
                    acc += _sh_dPdθ(g, ℓ, am, j) * Aφ[j] +
                           (Pam[row, j] / _sh_sinθ(g, j)) * Bφ[j]
                end
            end
            acc *= Nlm
            s_ℓm = acc / sqrt(T(ℓ * (ℓ + 1)))          # spheroidal coefficient
            div[(ℓ, m)] = -sqrt(T(ℓ * (ℓ + 1))) * s_ℓm  # = -ℓ(ℓ+1)/√(ℓ(ℓ+1)) · s
        end
    end
    div
end

"""
    vecsh_advection(theta, dtheta_dr, ur, utheta, uphi, lmax, mmax, r)

Project the advective form u·∇T = u_r∂_rT + (u_θ∂_θT + u_φ∂_φT/sinθ)/r onto
real orthonormal scalar harmonics (ℓ ≤ lmax, |m| ≤ mmax), with every input a
scalar-harmonic expansion of the corresponding field. This is exact for the
radial term, but the tangential components of a solenoidal velocity are not
band-limited scalar harmonics: truncated scalar projections of u_θ, u_φ, and
products involving ∂_θ, are only approximate. Constructed mean states use the
vector-harmonic path `_mean_flow_advection` instead.
"""
function vecsh_advection(theta::AbstractDict{Tuple{Int,Int},Vector{T}},
                         dtheta_dr::AbstractDict{Tuple{Int,Int},Vector{T}},
                         ur::AbstractDict{Tuple{Int,Int},Vector{T}},
                         utheta::AbstractDict{Tuple{Int,Int},Vector{T}},
                         uphi::AbstractDict{Tuple{Int,Int},Vector{T}},
                         lmax::Int, mmax::Int, r::Vector{T}) where {T<:Real}
    g = sh_grid(lmax, mmax, T)
    forcing = Dict((l,m)=>zeros(T,length(r)) for m in -mmax:mmax for l in abs(m):lmax)
    for i in eachindex(r)
        at(d)=Dict(k=>v[i] for (k,v) in d)
        temp=at(theta)
        adv = sh_synthesize(at(ur),g).*sh_synthesize(at(dtheta_dr),g) +
            (sh_synthesize(at(utheta),g).*sh_synthesize(temp,g;Yf=_sh_dYθ) +
             sh_synthesize(at(uphi),g).*sh_synthesize(temp,g;Yf=_sh_dYφ_over_sin))./r[i]
        for (k,v) in sh_analyze(adv,g)
            forcing[k][i]=v
        end
    end
    forcing
end

"""
    _sh_nf_to_orth_factor(ℓ, m, T) -> T

Coefficient rescale from the basic-state "no-factorial" SH normalization
(`√((2ℓ+1)/4π·[2 if m≠0])`, used by the thermal-wind/construction subsystem) to
the orthonormal full-`N_ℓm` normalization used by `sh_*`/`vecsh_advection`.
Since ‖Y^{no-factorial}_ℓm‖² = (ℓ+|m|)!/(ℓ-|m|)!, a coefficient transforms as
`a_orth = a_nf · √((ℓ+|m|)!/(ℓ-|m|)!)`. Returns 1 for m=0 (the conventions agree).
"""
function _sh_nf_to_orth_factor(ℓ::Int, m::Int, ::Type{T}) where {T<:Real}
    am = abs(m)
    am == 0 && return one(T)
    ratio = one(T)                       # ∏_{k=ℓ-am+1}^{ℓ+am} k = (ℓ+am)!/(ℓ-am)!
    for k in (ℓ - am + 1):(ℓ + am)
        ratio *= T(k)
    end
    return sqrt(ratio)
end

"""Rescale a coeff dict from no-factorial → orthonormal (`dir=+1`) or back (`dir=-1`)."""
function _sh_rescale(coeffs::AbstractDict{Tuple{Int,Int},Vector{T}}, dir::Int) where {T<:Real}
    out = Dict{Tuple{Int,Int},Vector{T}}()
    for ((ℓ, m), v) in coeffs
        f = _sh_nf_to_orth_factor(ℓ, m, T)
        out[(ℓ, m)] = dir > 0 ? v .* f : v ./ f
    end
    out
end
