# =============================================================================
#  Basic State for Onset of Convection
#
#  Implements both axisymmetric and non-axisymmetric basic states with:
#  - Temperature variations: θ̄(r,θ,φ)
#  - Thermal wind-balanced flows: ū(r,θ,φ)
#
#  Two implementations:
#  1. BasicState: Axisymmetric (m=0 only), for standard onset
#  2. BasicState3D: Non-axisymmetric (multiple m), for tri-global analysis
# =============================================================================

using Parameters
using LinearAlgebra
using SparseArrays

"""
    BasicState{T<:Real}

Holds the axisymmetric (m=0) basic state for linear stability analysis.

The basic state consists of:
- Temperature: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
- Zonal flow: ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)
- Meridional circulation: ū_r and ū_θ from the same viscous solve
- `flow`: authoritative divergence-free vector-harmonic representation

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree for basic state
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Int,Vector{T}}` - Temperature coefficients θ̄_ℓ0(r) for each ℓ
- `uphi_coeffs::Dict{Int,Vector{T}}` - Zonal flow coefficients ū_φ,ℓ0(r) for each ℓ
- `dtheta_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂θ̄_ℓ0/∂r
- `duphi_dr_coeffs::Dict{Int,Vector{T}}` - Radial derivative ∂ū_φ,ℓ0/∂r
"""
@with_kw_noshow struct BasicState{T<:Real}
    lmax_bs::Int
    Nr::Int
    r::Vector{T}
    theta_coeffs::Dict{Int,Vector{T}}
    uphi_coeffs::Dict{Int,Vector{T}}
    dtheta_dr_coeffs::Dict{Int,Vector{T}}
    duphi_dr_coeffs::Dict{Int,Vector{T}}
    ur_coeffs::Dict{Int,Vector{T}} = empty(theta_coeffs)
    utheta_coeffs::Dict{Int,Vector{T}} = empty(theta_coeffs)
    dur_dr_coeffs::Dict{Int,Vector{T}} = empty(theta_coeffs)
    dutheta_dr_coeffs::Dict{Int,Vector{T}} = empty(theta_coeffs)
    flow::Union{Nothing,SolenoidalMeanFlow{T}} = nothing
end


"""
    BasicState3D{T<:Real}

Holds a non-axisymmetric (3D) basic state for tri-global instability analysis.

The basic state has both meridional AND longitudinal variations:
- Temperature: θ̄(r,θ,φ) = Σ_ℓ Σ_m_bs θ̄_ℓm_bs(r) Y_ℓm_bs(θ,φ)
- Velocity components: ū_r, ū_θ, ū_φ from the steady Stokes–Coriolis balance

This enables studying onset of convection on top of 3D thermal and flow structures,
such as:
- Longitudinally-varying boundary heating
- Zonal jets with wavenumber structure
- Realistic 3D planetary/stellar base states

Fields:
- `lmax_bs::Int` - Maximum spherical harmonic degree
- `mmax_bs::Int` - Maximum azimuthal wavenumber (typically small, e.g., 0-4)
- `Nr::Int` - Number of radial collocation points
- `r::Vector{T}` - Radial collocation points
- `theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - θ̄_ℓm(r) indexed by (ℓ,m)
- `ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_r,ℓm(r)
- `utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_θ,ℓm(r)
- `uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ū_φ,ℓm(r)
- `dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂θ̄_ℓm/∂r
- `dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_r,ℓm/∂r
- `dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_θ,ℓm/∂r
- `duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}` - ∂ū_φ,ℓm/∂r

Note: Perturbations on this basic state couple multiple azimuthal modes m simultaneously.
The eigenvalue problem becomes block-coupled across different m values.
"""
@with_kw_noshow struct BasicState3D{T<:Real}
    lmax_bs::Int
    mmax_bs::Int
    Nr::Int
    r::Vector{T}
    # Temperature
    theta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dtheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}

    # Velocity components
    ur_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    utheta_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    uphi_coeffs::Dict{Tuple{Int,Int},Vector{T}}

    # Velocity derivatives
    dur_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    dutheta_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    duphi_dr_coeffs::Dict{Tuple{Int,Int},Vector{T}}
    flow::Union{Nothing,SolenoidalMeanFlow{T}} = nothing
end


# =============================================================================
#  Symbolic Spherical Harmonic Boundary Conditions
#
#  Provides an intuitive interface for specifying temperature boundary
#  conditions using spherical harmonic notation:
#
#    bc = Y20(0.1) + Y22(0.05)  # Meridional + longitudinal pattern
#    bc = 0.5 * Y10()           # Scaled dipole
#
#  These can be passed directly to basic state functions.
# =============================================================================

function _axisymmetric_state(bs::BasicState3D{T}) where T
    axis(d)=Dict(l=>v for ((l,m),v) in d if m==0)
    f=bs.flow
    if f !== nothing
        only0(d)=Dict(k=>v for (k,v) in d if k[2]==0)
        f=SolenoidalMeanFlow(f.lmax,0,f.r,only0(f.p),only0(f.t),only0(f.dp),only0(f.d2p),only0(f.dt))
    end
    BasicState(lmax_bs=bs.lmax_bs,Nr=bs.Nr,r=bs.r,
        theta_coeffs=axis(bs.theta_coeffs),dtheta_dr_coeffs=axis(bs.dtheta_dr_coeffs),
        uphi_coeffs=axis(bs.uphi_coeffs),duphi_dr_coeffs=axis(bs.duphi_dr_coeffs),
        ur_coeffs=axis(bs.ur_coeffs),utheta_coeffs=axis(bs.utheta_coeffs),
        dur_dr_coeffs=axis(bs.dur_dr_coeffs),dutheta_dr_coeffs=axis(bs.dutheta_dr_coeffs),flow=f)
end

"""
    SphericalHarmonicBC{T<:Real}

Symbolic representation of boundary conditions expanded in spherical harmonics.

This type provides a convenient way to specify temperature boundary conditions
using standard spherical harmonic notation (Y_ℓm) rather than dictionary syntax.

# Amplitude convention
An amplitude `a` for mode `(ℓ, m)` prescribes the outer-boundary pattern

    a · P_ℓ^m(cosθ) · cos(mφ)

where `P_ℓ^m` is the unnormalized associated Legendre function *including the
Condon–Shortley phase* `(-1)^m`, as used for the stored coefficients. Odd-`m`
patterns therefore carry a minus sign: `Y11(a)` is `-a sinθ cosφ` and `Y21(a)`
is `-3a sinθ cosθ cosφ`, while `Y20(a)` is `a (3cos²θ - 1)/2`. With a flux
boundary condition the same pattern prescribes `∂θ̄/∂r` at `r_o`.

# Constructor Functions
- `Ylm(ℓ, m, amplitude)` - General spherical harmonic mode
- `Y00(amp)`, `Y10(amp)`, `Y11(amp)` - Monopole and dipole
- `Y20(amp)`, `Y21(amp)`, `Y22(amp)` - Quadrupole
- `Y30(amp)`, ..., `Y44(amp)` - Higher orders

# Operators
- `+` : Combine multiple harmonics: `Y20(0.1) + Y22(0.05)`
- `*` : Scale amplitude: `0.5 * Y20(0.1)` or `Y20(0.1) * 0.5`
- `-` : Negate or subtract: `-Y20(0.1)` or `Y20(0.1) - Y22(0.05)`

# Examples

## Simple meridional variation (Y₂₀)
```julia
bc = Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Combined meridional and longitudinal variation
```julia
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Dipole pattern with negative amplitude
```julia
bc = Y10(-0.2)  # Hot at one pole, cold at other
```

## Complex pattern with scaling
```julia
bc = 0.5 * (Y20(1.0) + 2.0 * Y40(0.5))
```

## Flux boundary condition
```julia
bc = Y20(0.1)  # Prescribes ∂θ̄/∂r at r_o when passed as flux_bc
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=bc)
```

# Physical Interpretation

The spherical harmonics Y_ℓm represent different angular patterns (signs
below are for a positive amplitude):

- **Y₀₀**: Uniform (spherically symmetric)
- **Y₁₀**: North-south dipole (warm north pole, cold south pole)
- **Y₁₁**: East-west dipole (`-sinθ cosφ`: cold at φ=0°, warm at φ=180°)
- **Y₂₀**: Equator-pole contrast (warm poles, cool equator)
- **Y₂₂**: Four-fold longitudinal pattern (warm at 0°,180°, cold at 90°,270°)
- **Y₃₀**: More complex latitudinal structure
- **Y₄₀**: Even more latitudinal bands

For convection studies:
- Y₂₀ is most common: represents differential heating between equator and poles
- Y₂₂ represents tidal or orbital forcing patterns
- Y₁₀ represents hemispherical asymmetry
"""
struct SphericalHarmonicBC{T<:Real}
    coeffs::Dict{Tuple{Int,Int}, T}
end

# Empty constructor
"""Create an empty symbolic spherical-harmonic boundary condition."""
SphericalHarmonicBC{T}() where T = SphericalHarmonicBC{T}(Dict{Tuple{Int,Int}, T}())

# Single mode constructor
"""Create a symbolic boundary condition containing one `(l, m)` harmonic."""
function SphericalHarmonicBC(ℓ::Int, m::Int, amplitude::T) where T<:Real
    if ℓ < 0
        throw(ArgumentError("ℓ must be non-negative, got ℓ=$ℓ"))
    end
    if m < 0 || m > ℓ
        throw(ArgumentError("m must satisfy 0 ≤ m ≤ ℓ, got ℓ=$ℓ, m=$m"))
    end
    SphericalHarmonicBC{T}(Dict((ℓ, m) => amplitude))
end

# Addition: combine multiple spherical harmonic BCs
"""Combine two symbolic boundary-condition spectra, promoting amplitudes as needed."""
function Base.:+(a::SphericalHarmonicBC{T}, b::SphericalHarmonicBC{S}) where {T,S}
    R = promote_type(T, S)
    result = Dict{Tuple{Int,Int}, R}()
    for (k, v) in a.coeffs
        result[k] = get(result, k, zero(R)) + R(v)
    end
    for (k, v) in b.coeffs
        result[k] = get(result, k, zero(R)) + R(v)
    end
    SphericalHarmonicBC{R}(result)
end

# Scalar multiplication (from left)
"""Scale all amplitudes in a symbolic spherical-harmonic boundary condition."""
function Base.:*(c::Real, bc::SphericalHarmonicBC{T}) where T
    R = promote_type(typeof(c), T)
    SphericalHarmonicBC{R}(Dict(k => R(c) * R(v) for (k, v) in bc.coeffs))
end

# Scalar multiplication (from right)
"""Scale a symbolic spherical-harmonic boundary condition from the right."""
Base.:*(bc::SphericalHarmonicBC, c::Real) = c * bc

# Division by scalar
"""Divide all amplitudes in a symbolic boundary condition by a scalar."""
Base.:/(bc::SphericalHarmonicBC, c::Real) = (1/c) * bc

# Negation
"""Negate all amplitudes in a symbolic boundary condition."""
Base.:-(bc::SphericalHarmonicBC) = (-1) * bc

# Subtraction
"""Subtract one symbolic spherical-harmonic boundary condition from another."""
Base.:-(a::SphericalHarmonicBC, b::SphericalHarmonicBC) = a + (-b)

# Zero check
"""Return true when all stored boundary-condition amplitudes are zero."""
Base.iszero(bc::SphericalHarmonicBC) = isempty(bc.coeffs) || all(iszero, values(bc.coeffs))

# =============================================================================
#  Convenience Constructors for Common Spherical Harmonics
# =============================================================================

"""
    Ylm(ℓ::Int, m::Int, amplitude::Real=1.0)

Create a spherical harmonic boundary condition for mode (ℓ, m): the outer
boundary pattern `amplitude · P_ℓ^m(cosθ) · cos(mφ)`, with the unnormalized
associated Legendre function `P_ℓ^m` including the Condon–Shortley phase
`(-1)^m` (see [`SphericalHarmonicBC`](@ref)).

# Arguments
- `ℓ` : Spherical harmonic degree (ℓ ≥ 0)
- `m` : Azimuthal order (0 ≤ m ≤ ℓ)
- `amplitude` : Amplitude of this mode (default: 1.0)

# Example
```julia
bc = Ylm(3, 2, 0.1)  # Y₃₂ mode with amplitude 0.1
```
"""
Ylm(ℓ::Int, m::Int, amplitude::Real=1.0) = SphericalHarmonicBC(ℓ, m, amplitude)

# ℓ = 0: Monopole (uniform)
"""Y00(amplitude=1.0) - Uniform (monopole): `amplitude` everywhere"""
Y00(amplitude::Real=1.0) = SphericalHarmonicBC(0, 0, amplitude)

# ℓ = 1: Dipole
"""Y10(amplitude=1.0) - Axial dipole: `amplitude·cosθ` (north-south asymmetry)"""
Y10(amplitude::Real=1.0) = SphericalHarmonicBC(1, 0, amplitude)

"""Y11(amplitude=1.0) - Equatorial dipole: `-amplitude·sinθ·cosφ` (Condon–Shortley sign)"""
Y11(amplitude::Real=1.0) = SphericalHarmonicBC(1, 1, amplitude)

# ℓ = 2: Quadrupole
"""Y20(amplitude=1.0) - Axisymmetric quadrupole: `amplitude·(3cos²θ - 1)/2` (equator-pole contrast)"""
Y20(amplitude::Real=1.0) = SphericalHarmonicBC(2, 0, amplitude)

"""Y21(amplitude=1.0) - Tesseral quadrupole: `-3·amplitude·sinθ·cosθ·cosφ` (Condon–Shortley sign)"""
Y21(amplitude::Real=1.0) = SphericalHarmonicBC(2, 1, amplitude)

"""Y22(amplitude=1.0) - Sectoral quadrupole: `3·amplitude·sin²θ·cos(2φ)` (four-fold longitudinal)"""
Y22(amplitude::Real=1.0) = SphericalHarmonicBC(2, 2, amplitude)

# ℓ = 3: Octupole
"""Y30(amplitude=1.0) - Axisymmetric octupole: `amplitude·(5cos³θ - 3cosθ)/2`"""
Y30(amplitude::Real=1.0) = SphericalHarmonicBC(3, 0, amplitude)

"""Y31(amplitude=1.0) - Tesseral octupole: `-(3/2)·amplitude·sinθ·(5cos²θ - 1)·cosφ` (Condon–Shortley sign)"""
Y31(amplitude::Real=1.0) = SphericalHarmonicBC(3, 1, amplitude)

"""Y32(amplitude=1.0) - Tesseral octupole: `15·amplitude·sin²θ·cosθ·cos(2φ)`"""
Y32(amplitude::Real=1.0) = SphericalHarmonicBC(3, 2, amplitude)

"""Y33(amplitude=1.0) - Sectoral octupole: `-15·amplitude·sin³θ·cos(3φ)` (Condon–Shortley sign)"""
Y33(amplitude::Real=1.0) = SphericalHarmonicBC(3, 3, amplitude)

# ℓ = 4: Hexadecapole
"""Y40(amplitude=1.0) - Axisymmetric hexadecapole: `amplitude·(35cos⁴θ - 30cos²θ + 3)/8`"""
Y40(amplitude::Real=1.0) = SphericalHarmonicBC(4, 0, amplitude)

"""Y41(amplitude=1.0) - Tesseral hexadecapole: `-(5/2)·amplitude·sinθ·(7cos³θ - 3cosθ)·cosφ` (Condon–Shortley sign)"""
Y41(amplitude::Real=1.0) = SphericalHarmonicBC(4, 1, amplitude)

"""Y42(amplitude=1.0) - Tesseral hexadecapole: `(15/2)·amplitude·sin²θ·(7cos²θ - 1)·cos(2φ)`"""
Y42(amplitude::Real=1.0) = SphericalHarmonicBC(4, 2, amplitude)

"""Y43(amplitude=1.0) - Tesseral hexadecapole: `-105·amplitude·sin³θ·cosθ·cos(3φ)` (Condon–Shortley sign)"""
Y43(amplitude::Real=1.0) = SphericalHarmonicBC(4, 3, amplitude)

"""Y44(amplitude=1.0) - Sectoral hexadecapole: `105·amplitude·sin⁴θ·cos(4φ)`"""
Y44(amplitude::Real=1.0) = SphericalHarmonicBC(4, 4, amplitude)

# =============================================================================
#  Utility Functions for SphericalHarmonicBC
# =============================================================================

"""
    to_dict(bc::SphericalHarmonicBC)

Convert SphericalHarmonicBC to Dict{Tuple{Int,Int}, T} format.

This is the internal format used by basic state functions.
"""
to_dict(bc::SphericalHarmonicBC{T}) where T = Dict{Tuple{Int,Int}, T}(bc.coeffs)

"""
    get_lmax(bc::SphericalHarmonicBC)

Get the maximum spherical harmonic degree in the boundary condition.
"""
function get_lmax(bc::SphericalHarmonicBC)
    isempty(bc.coeffs) && return 0
    maximum(first(k) for k in keys(bc.coeffs))
end

"""
    get_mmax(bc::SphericalHarmonicBC)

Get the maximum azimuthal order in the boundary condition.
"""
function get_mmax(bc::SphericalHarmonicBC)
    isempty(bc.coeffs) && return 0
    maximum(last(k) for k in keys(bc.coeffs))
end

"""
    get_lmax_mmax(bc::SphericalHarmonicBC)

Get both (lmax, mmax) from the boundary condition.
"""
get_lmax_mmax(bc::SphericalHarmonicBC) = (get_lmax(bc), get_mmax(bc))

"""
    is_axisymmetric(bc::SphericalHarmonicBC)

Check if the boundary condition is axisymmetric (m=0 only).
"""
is_axisymmetric(bc::SphericalHarmonicBC) = get_mmax(bc) == 0

# Pretty printing
"""Print a compact algebraic representation of a spherical-harmonic boundary condition."""
function Base.show(io::IO, bc::SphericalHarmonicBC{T}) where T
    if isempty(bc.coeffs)
        print(io, "SphericalHarmonicBC{$T}(empty)")
        return
    end

    terms = String[]
    for ((ℓ, m), amp) in sort(collect(bc.coeffs), by=x->(x[1][1], x[1][2]))
        if abs(amp) < eps(T) * 1000
            continue
        end
        if amp == 1.0
            push!(terms, "Y$ℓ$m")
        elseif amp == -1.0
            push!(terms, "-Y$ℓ$m")
        else
            push!(terms, "$(amp)*Y$ℓ$m")
        end
    end

    if isempty(terms)
        print(io, "SphericalHarmonicBC{$T}(zero)")
    else
        print(io, join(terms, " + "))
    end
end

"""Print all stored harmonic amplitudes in a multiline REPL summary."""
function Base.show(io::IO, ::MIME"text/plain", bc::SphericalHarmonicBC{T}) where T
    println(io, "SphericalHarmonicBC{$T}")
    if isempty(bc.coeffs)
        _tree_row(io, "modes", "none"; last=true)
        return
    end
    pairs = sort(collect(bc.coeffs), by=x->(x[1][1], x[1][2]))
    for (i, ((ℓ, m), amp)) in enumerate(pairs)
        _tree_row(io, "Y_$ℓ,$m", amp; last=i == length(pairs))
    end
end


"""
    conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int;
                           thermal_bc::Symbol=:fixed_temperature,
                           outer_flux::T=-χ/(1-χ)) where T

Create a basic state corresponding to pure conduction (no meridional variation).

This is the default basic state with:
- θ̄(r) = conduction profile (only ℓ=0 component)
- ū_φ = 0 (no flow)

Arguments:
- `cd` - Chebyshev differentiation structure
- `χ` - Radius ratio r_i/r_o
- `lmax_bs` - Maximum ℓ for basic state (typically small, e.g., 4)
- `thermal_bc` - Outer thermal boundary condition:
  - `:fixed_temperature` (default): θ̄(r_o) = 0
  - `:fixed_flux`: dθ̄/dr|_{r_o} = outer_flux
- `outer_flux` - Prescribed radial temperature gradient ∂θ̄/∂r at r_o (only used
                 if thermal_bc=:fixed_flux). The outward heat flux is -∂θ̄/∂r, so a
                 negative value carries heat outward (the usual case with a hot inner
                 boundary) and a positive value carries heat inward. The default
                 outer_flux = -χ/(1 - χ) carries the conduction heat flux and
                 reproduces the fixed-temperature conduction profile (θ̄(r_o) = 0).

# Physical Interpretation
For fixed temperature BCs:
  - θ̄(r_i) = 1 (hot inner boundary)
  - θ̄(r_o) = 0 (cold outer boundary)

For fixed flux at outer:
  - θ̄(r_i) = 1 (hot inner boundary)
  - dθ̄/dr|_{r_o} = outer_flux (prescribed gradient; outward heat flux = -outer_flux)

The conduction profile for ℓ=0 with fixed flux at outer is:
  θ̄_0(r) = √(4π) × [1 + outer_flux × r_o² × (1/r_i - 1/r)]
"""
function conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int;
                                thermal_bc::Symbol=:fixed_temperature,
                                outer_flux::T=-χ/(one(T)-χ)) where T
    r = cd.x
    Nr = length(r)

    r_i = T(χ)
    r_o = T(1.0)

    # Validate thermal BC
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # ℓ=0 conduction profile
    inner_value = sqrt(T(4) * T(pi))   # θ̄_00(r_i) = 1 × √(4π)

    if thermal_bc == :fixed_temperature
        outer_value = zero(T)           # θ̄_00(r_o) = 0
        theta_cond, dtheta_dr_cond = laplace_mode_profile(0, r, r_i, r_o,
                                                         inner_value, outer_value;
                                                         outer_bc=:fixed_temperature)
    else  # fixed_flux
        # For ℓ=0: dθ̄_00/dr|_{r_o} = outer_flux × √(4π)
        # (normalize by √(4π) to match the spherical harmonic coefficient)
        outer_flux_normalized = outer_flux * sqrt(T(4) * T(pi))
        theta_cond, dtheta_dr_cond = laplace_mode_profile(0, r, r_i, r_o,
                                                         inner_value, outer_flux_normalized;
                                                         outer_bc=:fixed_flux)
    end

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    uphi_coeffs = empty(theta_coeffs)
    dtheta_dr_coeffs = empty(theta_coeffs)
    duphi_dr_coeffs = empty(theta_coeffs)

    # Only ℓ=0 component is non-zero
    # Y_00 = 1/√(4π), so θ̄_00(r) = √(4π) × θ_cond(r)
    theta_coeffs[0] = theta_cond
    dtheta_dr_coeffs[0] = dtheta_dr_cond
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)

    # Higher ℓ modes are zero
    for ℓ in 1:lmax_bs
        theta_coeffs[ℓ] = zeros(T, Nr)
        dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
        uphi_coeffs[ℓ] = zeros(T, Nr)
        duphi_dr_coeffs[ℓ] = zeros(T, Nr)
    end

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )

end


"""
Construct the axisymmetric conductive-temperature / viscous mean-flow state.
The outer anomaly is `amplitude * P₂(cosθ)` for fixed temperature. For fixed
flux, ∂θ̄/∂r at r_o is `outer_flux_mean` plus `F * P₂(cosθ)`, where
`F = amplitude` when it is nonzero and `outer_flux_Y20` otherwise. The default
`outer_flux_mean = -χ/(1-χ)` carries the conduction heat flux. The inner
temperature is uniform.

All three velocity components solve the steady Stokes–Coriolis equations in a
solenoidal vector-harmonic basis, with both mechanical boundaries enforced.
`Ra` is shell-gap based. This neglects momentum inertia and temperature
advection; use `basic_state_selfconsistent` to include both nonlinear effects.
Use `mean_flow_velocity` to evaluate the full vector field.

A nonzero Y₂₀ forcing requires `lmax_bs ≥ 2`; smaller values throw an
`ArgumentError` rather than dropping the forcing. Without Y₂₀ forcing any
`lmax_bs ≥ 0` is accepted and gives the conduction profile with zero flow.
"""
function meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
                               lmax_bs::Int, amplitude::T;
                               mechanical_bc::Symbol=:no_slip,
                               thermal_bc::Symbol=:fixed_temperature,
                               outer_flux_mean::T=-χ/(one(T)-χ),
                               outer_flux_Y20::T=zero(T)) where T

    r = cd.x
    Nr = length(r)
    r_i = T(χ)
    r_o = T(1.0)

    # Validate BCs
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # Degree-2 boundary forcing: a temperature amplitude, or a flux amplitude
    # (a nonzero `amplitude` overrides `outer_flux_Y20`).
    forcing_Y20 = (thermal_bc == :fixed_temperature || amplitude != zero(T)) ?
                  amplitude : outer_flux_Y20
    lmax_bs >= 0 || throw(ArgumentError("lmax_bs must be non-negative, got $lmax_bs"))
    if lmax_bs < 2 && !iszero(forcing_Y20)
        kind = thermal_bc == :fixed_temperature ? "temperature" : "flux"
        throw(ArgumentError("meridional_basic_state imposes a nonzero Y₂₀ boundary " *
            "$kind ($forcing_Y20), which lmax_bs=$lmax_bs cannot retain; use lmax_bs ≥ 2."))
    end

    # Spherical harmonic normalization for Y_20
    norm_Y20 = sqrt(T(5) / (T(4) * T(pi)))

    # Unforced degrees keep zero temperature.
    theta_coeffs = Dict{Int,Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)
    dtheta_dr_coeffs = Dict{Int,Vector{T}}(ℓ => zeros(T, Nr) for ℓ in 0:lmax_bs)

    if thermal_bc == :fixed_temperature
        # ℓ=0 mode: uniform inner temp, zero outer temp
        theta_coeffs[0], dtheta_dr_coeffs[0] = laplace_mode_profile(0, r, r_i, r_o,
            sqrt(T(4)*T(pi)), zero(T); outer_bc=:fixed_temperature)
        # ℓ=2 mode: zero at inner, amplitude × P₂ pattern at outer
        if lmax_bs >= 2
            theta_coeffs[2], dtheta_dr_coeffs[2] = laplace_mode_profile(2, r, r_i, r_o,
                zero(T), forcing_Y20 / norm_Y20; outer_bc=:fixed_temperature)
        end
    else  # fixed_flux
        # ℓ=0 mode: uniform inner temp, prescribed mean flux at outer
        # Normalize flux by √(4π) for spherical harmonic coefficient
        theta_coeffs[0], dtheta_dr_coeffs[0] = laplace_mode_profile(0, r, r_i, r_o,
            sqrt(T(4)*T(pi)), outer_flux_mean * sqrt(T(4) * T(pi)); outer_bc=:fixed_flux)
        # ℓ=2 mode: zero at inner, prescribed P₂ flux at outer
        if lmax_bs >= 2
            theta_coeffs[2], dtheta_dr_coeffs[2] = laplace_mode_profile(2, r, r_i, r_o,
                zero(T), forcing_Y20 / norm_Y20; outer_bc=:fixed_flux)
        end
    end

    theta3 = Dict((l,0)=>v for (l,v) in theta_coeffs)
    flow = _steady_mean_flow(theta3, r, cd.D1, cd.D2, E, Ra, Pr, lmax_bs, 0;
                             mechanical_bc=mechanical_bc)
    ur, utheta, uphi, dur, dutheta, duphi = _mean_flow_components(flow)
    axis(d) = Dict(l=>v for ((l,m),v) in d if m==0)
    return BasicState(lmax_bs=lmax_bs, Nr=Nr, r=r,
        theta_coeffs=theta_coeffs, dtheta_dr_coeffs=dtheta_dr_coeffs,
        uphi_coeffs=axis(uphi), duphi_dr_coeffs=axis(duphi),
        ur_coeffs=axis(ur), utheta_coeffs=axis(utheta),
        dur_dr_coeffs=axis(dur), dutheta_dr_coeffs=axis(dutheta), flow=flow)
end


# Helper: coefficients for expanding derivatives of Legendre polynomials.
#
# Returns a vector `deriv_maps` where `deriv_maps[ℓ]` is a dictionary mapping
# target degree L to the coefficient c_{ℓ,L} in
#     P_ℓ'(x) = Σ c_{ℓ,L} P_L(x)
# with L ranging over ℓ-1, ℓ-3, … (same parity as ℓ-1).
"""Compute Legendre derivative expansion maps up to degree `lmax`."""
function legendre_derivative_coefficients(lmax::Int)
    maps = Dict{Int, Dict{Int,Float64}}()
    maps[0] = Dict{Int,Float64}()           # P₀' = 0
    if lmax >= 1
        maps[1] = Dict(0 => 1.0)            # P₁' = P₀
    end

    for ℓ in 2:lmax
        coeffs = Dict{Int,Float64}()
        coeffs[ℓ-1] = (2ℓ - 1) * 1.0        # (2ℓ-1) P_{ℓ-1}

        for (k, v) in maps[ℓ-2]
            coeffs[k] = get(coeffs, k, 0.0) + v
        end

        maps[ℓ] = coeffs
    end

    return maps
end


"""
    laplace_mode_profile(ℓ, r, r_i, r_o, inner_value, outer_value;
                         outer_bc=:fixed_temperature)

Solve the radial Laplace equation for spherical harmonic mode ℓ.

The equation is: d²θ̄_ℓ/dr² + (2/r) dθ̄_ℓ/dr - ℓ(ℓ+1)/r² θ̄_ℓ = 0

General solution: θ̄_ℓ(r) = A r^ℓ + B r^{-(ℓ+1)}

# Arguments
- `ℓ::Int` - Spherical harmonic degree
- `r` - Radial grid points
- `r_i, r_o` - Inner and outer radii
- `inner_value` - Value or flux at inner boundary (always Dirichlet for temperature)
- `outer_value` - Value (for :fixed_temperature) or flux (for :fixed_flux) at outer boundary
- `outer_bc` - Outer boundary condition type:
  - `:fixed_temperature` (default): θ̄_ℓ(r_o) = outer_value
  - `:fixed_flux`: dθ̄_ℓ/dr|_{r_o} = outer_value

# Returns
- `θ` - Temperature profile θ̄_ℓ(r)
- `dθ` - Radial derivative dθ̄_ℓ/dr

# Mathematical Details
For fixed temperature at both boundaries:
  - θ̄_ℓ(r_i) = inner_value
  - θ̄_ℓ(r_o) = outer_value

For fixed temperature at inner, fixed flux at outer:
  - θ̄_ℓ(r_i) = inner_value
  - dθ̄_ℓ/dr|_{r_o} = outer_value

The derivative is: dθ̄_ℓ/dr = A ℓ r^{ℓ-1} - B (ℓ+1) r^{-(ℓ+2)}
"""
function laplace_mode_profile(ℓ::Int, r::AbstractVector{T}, r_i::T, r_o::T,
                             inner_value::T, outer_value::T;
                             outer_bc::Symbol=:fixed_temperature) where T

    if outer_bc == :fixed_temperature
        # Both boundaries have Dirichlet conditions (fixed temperature)
        # θ̄_ℓ(r_i) = inner_value
        # θ̄_ℓ(r_o) = outer_value
        M = T[
            r_i^ℓ          r_i^(-(ℓ+1));
            r_o^ℓ          r_o^(-(ℓ+1))
        ]
        rhs = T[inner_value, outer_value]

    elseif outer_bc == :fixed_flux
        # Inner: Dirichlet (fixed temperature)
        # Outer: Neumann (fixed flux)
        # θ̄_ℓ(r_i) = inner_value
        # dθ̄_ℓ/dr|_{r_o} = outer_value
        #
        # From θ̄_ℓ = A r^ℓ + B r^{-(ℓ+1)}:
        #   dθ̄_ℓ/dr = A ℓ r^{ℓ-1} - B (ℓ+1) r^{-(ℓ+2)}
        #
        # At r = r_o:
        #   dθ̄_ℓ/dr|_{r_o} = A ℓ r_o^{ℓ-1} - B (ℓ+1) r_o^{-(ℓ+2)}
        M = T[
            r_i^ℓ                    r_i^(-(ℓ+1));
            ℓ * r_o^(ℓ-1)           -(ℓ+1) * r_o^(-(ℓ+2))
        ]
        rhs = T[inner_value, outer_value]

    else
        error("outer_bc must be :fixed_temperature or :fixed_flux, got: $outer_bc")
    end

    α, β = M \ rhs

    θ = α .* r.^ℓ .+ β .* r.^(-(ℓ+1))
    dθ = α * ℓ .* r.^(ℓ-1) .- β * (ℓ+1) .* r.^(-(ℓ+2))

    return θ, dθ
end


# Note: The solve_thermal_wind_balance! function with E parameter is defined below
# (after the non-axisymmetric basic state functions)


"""
    evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T

Evaluate the basic state at a given (r, θ) point.

All radial profiles use the spectral (Chebyshev barycentric) interpolant on
`bs.r`, the Chebyshev–Gauss–Lobatto grid produced by every constructor. When
`bs.flow` is present, ū_φ and its derivatives come from the native toroidal
potentials. θ-derivatives remain exact at the poles.

Returns:
- `theta_bar` - Temperature θ̄(r,θ)
- `uphi_bar` - Zonal velocity ū_φ(r,θ)
- `dtheta_dr` - Radial derivative ∂θ̄/∂r
- `dtheta_dtheta` - Meridional derivative ∂θ̄/∂θ
- `duphi_dr` - Radial derivative ∂ū_φ/∂r
- `duphi_dtheta` - Meridional derivative ∂ū_φ/∂θ
"""
function evaluate_basic_state(bs::BasicState{T}, r_eval::T, theta_eval::T) where T
    rmin = min(first(bs.r), last(bs.r))
    rmax = max(first(bs.r), last(bs.r))
    if r_eval < rmin || r_eval > rmax
        throw(ArgumentError("r_eval must be within [$rmin, $rmax]"))
    end

    # Spectral radial interpolation for every field (the barycentric formula
    # expects an ascending grid, so reverse a descending one).
    ascending = first(bs.r) <= last(bs.r)
    rgrid = ascending ? bs.r : reverse(bs.r)
    radial(v) = _mean_barycentric(rgrid, ascending ? v : reverse(v), r_eval)

    lmax = bs.lmax_bs
    x = cos(theta_eval)
    P, dPdx = _legendre_values_and_derivs(lmax, x)
    sinθ = sin(theta_eval)

    norms = Vector{T}(undef, lmax + 1)
    for ℓ in 0:lmax
        norms[ℓ + 1] = sqrt(T(2 * ℓ + 1) / (T(4) * T(pi)))
    end

    theta_bar = zero(T)
    uphi_bar = zero(T)
    dtheta_dr = zero(T)
    dtheta_dtheta = zero(T)
    duphi_dr = zero(T)
    duphi_dtheta = zero(T)

    for (ℓ, coeffs) in bs.theta_coeffs
        ℓ > lmax && continue
        coeff = radial(coeffs)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        theta_bar += coeff * Y
        dtheta_dtheta += coeff * dY_dtheta
        if haskey(bs.dtheta_dr_coeffs, ℓ)
            dtheta_dr += radial(bs.dtheta_dr_coeffs[ℓ]) * Y
        end
    end

    for (ℓ, coeffs) in bs.uphi_coeffs
        ℓ > lmax && continue
        coeff = radial(coeffs)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        uphi_bar += coeff * Y
        duphi_dtheta += coeff * dY_dtheta
        if haskey(bs.duphi_dr_coeffs, ℓ)
            duphi_dr += radial(bs.duphi_dr_coeffs[ℓ]) * Y
        end
    end

    if bs.flow !== nothing
        # Axisymmetric uφ is the toroidal vector harmonic t/r * ∂θY.
        # Use it directly instead of differentiating a scalar projection.
        uphi_bar = zero(T); duphi_dr = zero(T); duphi_dtheta = zero(T)
        for ((l,m),values) in bs.flow.t
            m==0 || continue
            t = radial(values)
            dt = radial(bs.flow.dt[(l,m)])
            Y = norms[l+1]*P[l+1]
            dY = -sinθ*norms[l+1]*dPdx[l+1]
            # ∂²θY from Legendre's equation; needs the exact P′ at the poles.
            d2Y = x*norms[l+1]*dPdx[l+1]-l*(l+1)*Y
            uphi_bar += t/r_eval*dY
            duphi_dr += (dt/r_eval-t/r_eval^2)*dY
            duphi_dtheta += t/r_eval*d2Y
        end
    end

    return (
        theta_bar = theta_bar,
        uphi_bar = uphi_bar,
        dtheta_dr = dtheta_dr,
        dtheta_dtheta = dtheta_dtheta,
        duphi_dr = duphi_dr,
        duphi_dtheta = duphi_dtheta
    )
end

"""
Return Legendre values P_ℓ(x) and x-derivatives P_ℓ′(x) for ℓ = 0..`lmax`.
The derivative recurrence P_ℓ′ = ℓP_{ℓ-1} + xP_{ℓ-1}′ has no 1/(1-x²) factor,
so it stays exact at the poles, where P_ℓ′(±1) = (±1)^{ℓ+1} ℓ(ℓ+1)/2.
"""
function _legendre_values_and_derivs(lmax::Int, x::T) where T
    P = zeros(T, lmax + 1)
    dPdx = zeros(T, lmax + 1)
    P[1] = one(T)
    if lmax >= 1
        P[2] = x
        dPdx[2] = one(T)
    end
    for l in 2:lmax
        P[l + 1] = ((2 * l - 1) * x * P[l] - (l - 1) * P[l - 1]) / l
        dPdx[l + 1] = l * P[l] + x * dPdx[l]
    end
    return P, dPdx
end

"""Linearly interpolate radial data regardless of whether the grid ascends or descends."""
function _linear_interpolate(r::AbstractVector{T}, values::AbstractVector{T}, r_eval::T) where T
    length(r) == length(values) || throw(DimensionMismatch("r and values must have same length"))
    if r[1] <= r[end]
        return _linear_interpolate_ascending(r, values, r_eval)
    end
    r_rev = reverse(r)
    values_rev = reverse(values)
    return _linear_interpolate_ascending(r_rev, values_rev, r_eval)
end

"""Linearly interpolate on an ascending radial grid with endpoint clamping."""
function _linear_interpolate_ascending(r::AbstractVector{T}, values::AbstractVector{T}, r_eval::T) where T
    n = length(r)
    r_eval <= r[1] && return values[1]
    r_eval >= r[end] && return values[end]
    j = searchsortedlast(r, r_eval)
    j == n && (j = n - 1)
    t = (r_eval - r[j]) / (r[j + 1] - r[j])
    return (one(T) - t) * values[j] + t * values[j + 1]
end


# =============================================================================
#  Non-Axisymmetric (3D) Basic States
# =============================================================================

"""
Throw an `ArgumentError` unless every nonzero entry of `modes` (keyed by
`(ℓ, m)`; boundary amplitudes or radial profiles) is a valid harmonic retained
by the truncation `ℓ ≤ lmax_bs`, `|m| ≤ mmax_bs`. The constructors' mode loops
and the transforms would otherwise drop such entries silently. `source` names
the input in the message.
"""
function _check_retained_bc_modes(modes::AbstractDict, lmax_bs::Integer,
                                  mmax_bs::Integer, source::AbstractString)
    for (key, value) in modes
        key isa Tuple{Integer,Integer} || throw(ArgumentError(
            "$source must be keyed by (ℓ, m) tuples, got key $(repr(key))"))
        iszero(value) && continue
        ℓ, m = key
        abs(m) <= ℓ || throw(ArgumentError(
            "$source contains (ℓ,m)=($ℓ,$m), which is not a spherical harmonic (require |m| ≤ ℓ)"))
        if ℓ > lmax_bs || abs(m) > mmax_bs
            need = String[]
            ℓ > lmax_bs && push!(need, "lmax_bs ≥ $ℓ")
            abs(m) > mmax_bs && push!(need, "mmax_bs ≥ $(abs(m))")
            throw(ArgumentError("Nonzero $source mode (ℓ,m)=($ℓ,$m) lies outside the " *
                "retained harmonic range (lmax_bs=$lmax_bs, mmax_bs=$mmax_bs) and would be " *
                "dropped; use $(join(need, " and "))."))
        end
    end
    0 <= mmax_bs <= lmax_bs || throw(ArgumentError(
        "Require 0 ≤ mmax_bs ≤ lmax_bs, got lmax_bs=$lmax_bs, mmax_bs=$mmax_bs"))
    return nothing
end

# Legacy model switches that no longer select anything: the viscous solve always
# couples every harmonic and returns the complete divergence-free velocity.
function _warn_ignored_flow_keywords(; coupled_thermal_wind::Bool=true,
                                     include_meridional_flow::Bool=true,
                                     use_full_coupling::Bool=true)
    coupled_thermal_wind || @warn("coupled_thermal_wind=false is ignored: the viscous " *
        "Stokes–Coriolis solve always includes the full thermal-wind coupling.", maxlog=1)
    include_meridional_flow || @warn("include_meridional_flow=false is ignored: the viscous " *
        "solve always includes the meridional circulation, which continuity requires.", maxlog=1)
    use_full_coupling || @warn("use_full_coupling=false is ignored: the viscous solve " *
        "always includes the full Coriolis coupling between harmonics.", maxlog=1)
    return nothing
end

"""
Construct a conductive-temperature / steady Stokes–Coriolis basic state.
Both shell boundaries satisfy the selected mechanical condition. Temperature
and velocity retain cosine (`m>0`) and sine (`m<0`) modes. A public amplitude
for `(ℓ, m)` multiplies the unnormalized associated Legendre function
`P_ℓ^|m|(cosθ)` (Condon–Shortley phase, as for `SphericalHarmonicBC`) and
`cos(mφ)` (`m ≥ 0`) or `sin(|m|φ)` (`m < 0`); stored coefficients use the
historical no-factorial normalization. Every nonzero entry of `amplitudes`
(and, for `:fixed_flux`, of `outer_fluxes`) must satisfy `ℓ ≤ lmax_bs` and
`|m| ≤ min(ℓ, mmax_bs)`, otherwise an `ArgumentError` is thrown.
For `:fixed_flux` without a `(0, 0)` entry, the mean outer ∂θ̄/∂r is the
conduction value `-χ/(1-χ)`.

`Ra` is shell-gap based. Momentum inertia and thermal advection are omitted;
`nonaxisymmetric_basic_state_selfconsistent` includes both nonlinear effects.
`coupled_thermal_wind` and `include_meridional_flow` are ignored compatibility
keywords: the complete viscous velocity solve always includes the full coupling
and the meridional circulation. Passing `false` emits a one-time warning.

`flow` stores the authoritative orthonormal vector potentials. The component
coefficient dictionaries are scalar projections for compatibility, not a
solenoidal representation of tangential components. Evaluate physical velocity
with `mean_flow_velocity`.
"""
function nonaxisymmetric_basic_state(cd::ChebyshevDiffn, χ::Real, E::Real, Ra::Real, Pr::Real,
                                     lmax_bs::Int, mmax_bs::Int,
                                     amplitudes::AbstractDict;
                                     mechanical_bc::Symbol=:no_slip,
                                     thermal_bc::Symbol=:fixed_temperature,
                                     outer_fluxes::AbstractDict=Dict{Tuple{Int,Int},Float64}(),
                                     coupled_thermal_wind::Bool=true,
                                     include_meridional_flow::Bool=true)

    r = cd.x
    T = eltype(r)  # Get the element type from the Chebyshev grid
    Nr = length(r)
    r_i = T(χ)
    r_o = T(1.0)

    # Validate thermal BC
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end
    # The mode loop below only reads retained modes; reject anything it would drop.
    _check_retained_bc_modes(amplitudes, lmax_bs, mmax_bs, "amplitudes")
    thermal_bc == :fixed_flux &&
        _check_retained_bc_modes(outer_fluxes, lmax_bs, mmax_bs, "outer_fluxes")
    _warn_ignored_flow_keywords(coupled_thermal_wind=coupled_thermal_wind,
                                include_meridional_flow=include_meridional_flow)

    # Initialize all coefficient dictionaries
    theta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dtheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    ur_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    utheta_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    uphi_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dur_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    dutheta_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()
    duphi_dr_coeffs = Dict{Tuple{Int,Int},Vector{T}}()

    # Spherical harmonic normalization
    # For m=0: norm = sqrt((2ℓ+1)/(4π))
    # For m≠0: norm = sqrt((2ℓ+1)/(4π) × 2)
    Y_norm(ℓ::Int, m::Int) = m == 0 ? sqrt(T(2ℓ+1)/(4*T(π))) : sqrt(T(2ℓ+1)/(4*T(π)) * 2)

    # =========================================================================
    # Solve ∇²θ̄ = 0 for each (ℓ,m) mode
    # =========================================================================

    for ℓ in 0:lmax_bs
        for m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
            norm_Ylm = Y_norm(ℓ, m)

            if ℓ == 0 && m == 0
                # =============================================================
                # ℓ=0, m=0: Radial conduction profile (mean temperature)
                # =============================================================
                # Inner BC: θ̄_00(r_i) = 1 × √(4π) (uniform temperature = 1)
                inner_value = sqrt(T(4) * T(π))

                if thermal_bc == :fixed_temperature
                    # Outer BC: θ̄_00(r_o) = 0 (cold outer boundary)
                    outer_value = T(get(amplitudes,(0,0),zero(T))) * sqrt(T(4)*T(π))
                    theta_00, dtheta_00 = laplace_mode_profile(0, r, r_i, r_o,
                                                               inner_value, outer_value;
                                                               outer_bc=:fixed_temperature)
                else  # fixed_flux
                    # Outer BC: dθ̄_00/dr|_{r_o} = flux_00 × √(4π)
                    # Get flux from outer_fluxes or amplitudes; without either,
                    # carry the conduction heat flux.
                    flux_00 = get(outer_fluxes, (0,0), get(amplitudes, (0,0), -r_i/(r_o-r_i)))
                    outer_flux_normalized = T(flux_00) * sqrt(T(4) * T(π))
                    theta_00, dtheta_00 = laplace_mode_profile(0, r, r_i, r_o,
                                                               inner_value, outer_flux_normalized;
                                                               outer_bc=:fixed_flux)
                end

                theta_coeffs[(0,0)] = theta_00
                dtheta_dr_coeffs[(0,0)] = dtheta_00

            else
                # =============================================================
                # ℓ > 0 or m > 0: Higher-order modes
                # =============================================================
                # Get amplitude/flux for this mode
                # For fixed_flux: check outer_fluxes first, then amplitudes
                if thermal_bc == :fixed_flux
                    value = get(outer_fluxes, (ℓ,m), get(amplitudes, (ℓ,m), zero(T)))
                else
                    value = get(amplitudes, (ℓ,m), zero(T))
                end

                if value != 0
                    # Inner BC: θ̄_ℓm(r_i) = 0 (uniform inner temperature)
                    inner_value = zero(T)

                    if thermal_bc == :fixed_temperature
                        # Outer BC: θ̄_ℓm(r_o) = amplitude / norm_Ylm
                        outer_value = T(value) / norm_Ylm
                        theta_lm, dtheta_lm = laplace_mode_profile(ℓ, r, r_i, r_o,
                                                                   inner_value, outer_value;
                                                                   outer_bc=:fixed_temperature)
                    else  # fixed_flux
                        # Outer BC: dθ̄_ℓm/dr|_{r_o} = flux / norm_Ylm
                        outer_flux_normalized = T(value) / norm_Ylm
                        theta_lm, dtheta_lm = laplace_mode_profile(ℓ, r, r_i, r_o,
                                                                   inner_value, outer_flux_normalized;
                                                                   outer_bc=:fixed_flux)
                    end

                    theta_coeffs[(ℓ,m)] = theta_lm
                    dtheta_dr_coeffs[(ℓ,m)] = dtheta_lm

                else
                    # Zero value: Initialize to zero
                    theta_coeffs[(ℓ,m)] = zeros(T, Nr)
                    dtheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
                end
            end

            # Initialize velocity components to zero (will be filled by thermal wind)
            ur_coeffs[(ℓ,m)] = zeros(T, Nr)
            utheta_coeffs[(ℓ,m)] = zeros(T, Nr)
            uphi_coeffs[(ℓ,m)] = zeros(T, Nr)
            dur_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            dutheta_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
            duphi_dr_coeffs[(ℓ,m)] = zeros(T, Nr)
        end
    end

    # The complete velocity is always returned (see _warn_ignored_flow_keywords):
    # omitting meridional components would violate continuity for m != 0.
    flow = _steady_mean_flow(theta_coeffs, r, cd.D1, cd.D2, E, Ra, Pr,
                             lmax_bs, mmax_bs; mechanical_bc=mechanical_bc)
    ur_coeffs, utheta_coeffs, uphi_coeffs, dur_dr_coeffs, dutheta_dr_coeffs,
        duphi_dr_coeffs = _mean_flow_components(flow)

    return BasicState3D(
        lmax_bs = lmax_bs,
        mmax_bs = mmax_bs,
        Nr = Nr,
        r = r,
        theta_coeffs = theta_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        ur_coeffs = ur_coeffs,
        utheta_coeffs = utheta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dur_dr_coeffs = dur_dr_coeffs,
        dutheta_dr_coeffs = dutheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs,
        flow = flow
    )
end


# =============================================================================
#  Convenience Function: basic_state with Symbolic BCs
#
#  High-level interface that accepts SphericalHarmonicBC objects and
#  automatically dispatches to the appropriate low-level function.
# =============================================================================

"""
    basic_state(cd, χ, E, Ra, Pr;
                temperature_bc=nothing,
                flux_bc=nothing,
                mechanical_bc=:no_slip,
                lmax_bs=nothing)

Create a basic state with symbolic spherical harmonic boundary conditions.

This is a convenience wrapper that accepts `SphericalHarmonicBC` objects
(created with `Y20()`, `Y22()`, etc.) and automatically dispatches to the
appropriate low-level function (`conduction_basic_state`, `meridional_basic_state`,
or `nonaxisymmetric_basic_state`).

# Arguments
- `cd` : ChebyshevDiffn - Chebyshev differentiation structure
- `χ` : Radius ratio r_i/r_o
- `E` : Ekman number
- `Ra` : Rayleigh number
- `Pr` : Prandtl number

# Keyword Arguments
- `temperature_bc` : SphericalHarmonicBC specifying temperature at outer boundary
- `flux_bc` : SphericalHarmonicBC specifying the outer radial temperature gradient
  ∂θ̄/∂r (the outward heat flux is its negative)
  (Cannot specify both temperature_bc and flux_bc)
- `mechanical_bc` : `:no_slip` (default) or `:stress_free`
- `lmax_bs` : Maximum ℓ for basic state (default `max(ℓ_bc + 2, 4)`). An explicit
  value must retain every nonzero boundary mode, otherwise an `ArgumentError` is
  thrown instead of silently dropping the forcing.
- `coupled_thermal_wind` : ignored compatibility keyword (the viscous solve always
  includes the full coupling); `false` emits a one-time warning

# Returns
- `BasicState` if boundary condition is axisymmetric (m=0 only)
- `BasicState3D` if boundary condition has m≠0 components

# Examples

## Pure conduction (no temperature variation at outer boundary)
```julia
bs = basic_state(cd, χ, E, Ra, Pr)
```

## Meridional temperature variation (Y₂₀ pattern)
```julia
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=Y20(0.1))
```

## Combined meridional and longitudinal variation
```julia
bc = Y20(0.1) + Y22(0.05)
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc)
```

## Fixed flux at outer boundary
```julia
# Uniform outward heat flux plus meridional variation
flux = Y00(-1.0) + Y20(0.1)
bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux)
```

## Stress-free boundaries with temperature variation
```julia
bs = basic_state(cd, χ, E, Ra, Pr;
                 temperature_bc=Y20(0.1),
                 mechanical_bc=:stress_free)
```

# Automatic Dispatch Logic

The function automatically selects the appropriate implementation:

1. If no boundary condition is given, or all its amplitudes are zero:
   → `conduction_basic_state` (pure conduction profile)

2. If boundary condition is axisymmetric (only m=0 modes, including Y00 alone):
   → the viscous solver of `nonaxisymmetric_basic_state` with `mmax_bs=0`
     (returns `BasicState`)

3. If boundary condition has m≠0 modes:
   → `nonaxisymmetric_basic_state` (returns `BasicState3D`)
"""
function basic_state(cd, χ::Real, E::Real, Ra::Real, Pr::Real;
                     temperature_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                     flux_bc::Union{Nothing, SphericalHarmonicBC}=nothing,
                     mechanical_bc::Symbol=:no_slip,
                     lmax_bs::Union{Nothing, Int}=nothing,
                     coupled_thermal_wind::Bool=true)

    # Validate: can't have both temperature_bc and flux_bc
    if temperature_bc !== nothing && flux_bc !== nothing
        error("Cannot specify both temperature_bc and flux_bc. Choose one.")
    end
    _warn_ignored_flow_keywords(coupled_thermal_wind=coupled_thermal_wind)

    T = eltype(cd.x)

    # Determine thermal BC type and the boundary condition
    if flux_bc !== nothing
        thermal_bc = :fixed_flux
        bc = flux_bc
    elseif temperature_bc !== nothing
        thermal_bc = :fixed_temperature
        bc = temperature_bc
    else
        # No BC specified → pure conduction
        _lmax = lmax_bs === nothing ? 4 : lmax_bs
        return conduction_basic_state(cd, T(χ), _lmax;
                                      thermal_bc=:fixed_temperature)
    end

    # Get lmax and mmax from boundary condition
    bc_lmax, bc_mmax = get_lmax_mmax(bc)

    # Use provided lmax_bs or auto-determine (add 2 for thermal wind coupling)
    _lmax = lmax_bs === nothing ? max(bc_lmax + 2, 4) : lmax_bs
    _check_retained_bc_modes(bc.coeffs, _lmax, is_axisymmetric(bc) ? 0 : bc_mmax,
                             flux_bc === nothing ? "temperature_bc" : "flux_bc")

    # Check if BC is effectively zero (only conduction)
    if iszero(bc)
        return conduction_basic_state(cd, T(χ), _lmax; thermal_bc=thermal_bc)
    end

    if is_axisymmetric(bc)
        bs = nonaxisymmetric_basic_state(cd,T(χ),T(E),T(Ra),T(Pr),_lmax,0,to_dict(bc);
            mechanical_bc=mechanical_bc,thermal_bc=thermal_bc,
            coupled_thermal_wind=coupled_thermal_wind)
        return _axisymmetric_state(bs)
    end

    # Non-axisymmetric → use nonaxisymmetric_basic_state
    amplitudes = to_dict(bc)

    if thermal_bc == :fixed_temperature
        return nonaxisymmetric_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                           _lmax, bc_mmax, amplitudes;
                                           mechanical_bc=mechanical_bc,
                                           thermal_bc=:fixed_temperature,
                                           coupled_thermal_wind=coupled_thermal_wind)
    else  # fixed_flux
        return nonaxisymmetric_basic_state(cd, T(χ), T(E), T(Ra), T(Pr),
                                           _lmax, bc_mmax,
                                           Dict{Tuple{Int,Int},T}();  # empty amplitudes
                                           mechanical_bc=mechanical_bc,
                                           thermal_bc=:fixed_flux,
                                           outer_fluxes=amplitudes,
                                           coupled_thermal_wind=coupled_thermal_wind)
    end
end


"""
Compatibility wrapper returning the axisymmetric zonal projection of the
steady viscous Stokes–Coriolis solution. Both mechanical boundaries and the
gap-based Rayleigh conversion are applied. Prefer `meridional_basic_state`
for the complete, divergence-free velocity (including meridional circulation).
"""
function solve_thermal_wind_balance!(uphi_coeffs::Dict{Int,Vector{T}},
                            duphi_dr_coeffs::Dict{Int,Vector{T}},
                            theta_coeffs::Dict{Int,Vector{T}},
                            cd,  # ChebyshevDiffn{T}
                            r_i::T, r_o::T, Ra::T, Pr::T;
                            mechanical_bc::Symbol=:no_slip,
                            E::T=T(1e-4)) where T<:Real

    return solve_thermal_wind_coupled!(uphi_coeffs,duphi_dr_coeffs,theta_coeffs,0,
        cd,r_i,r_o,Ra,Pr;mechanical_bc=mechanical_bc,E=E)
end


"""
Compatibility wrapper for `solve_thermal_wind_coupled!`. Returns only the
requested real-m zonal projection; use `nonaxisymmetric_basic_state` for all
components, both azimuthal phases, and the vector-harmonic representation.
"""
function solve_thermal_wind_balance_3d!(uphi_coeffs::Dict{Int,Vector{T}},
                                        duphi_dr_coeffs::Dict{Int,Vector{T}},
                                        theta_coeffs::Dict{Int,Vector{T}},
                                        m_bs::Int,
                                        cd,
                                        r_i::T, r_o::T, Ra::T, Pr::T;
                                        mechanical_bc::Symbol=:no_slip,
                                        E::T=T(1e-4)) where T<:Real

    return solve_thermal_wind_coupled!(uphi_coeffs,duphi_dr_coeffs,theta_coeffs,m_bs,
        cd,r_i,r_o,Ra,Pr;mechanical_bc=mechanical_bc,E=E)
end


# =============================================================================
#  Orthonormal angular projection utilities and legacy component interface
# =============================================================================

"""
    _orthonormal_plm(lmax, m, x) -> Vector

Orthonormal associated Legendre functions P̄_ℓ^m(x) for ℓ = 0..lmax, normalized
so that ∫_{-1}^1 (P̄_ℓ^m)² dx = 1 (the θ-part of the sphere-orthonormal Y_ℓ^m).
"""
function _orthonormal_plm(lmax::Int, m::Int, x::T) where {T<:Real}
    P = zeros(T, lmax + 1)
    m > lmax && return P
    somx2 = sqrt((one(T) - x) * (one(T) + x))
    # unnormalized P_m^m = (-1)^m (2m-1)!! (1-x²)^{m/2}
    pmm = one(T)
    fact = one(T)
    for _ in 1:m
        pmm *= -fact * somx2
        fact += T(2)
    end
    # orthonormal-in-x factor c_ℓ = √[(2ℓ+1)/2 · (ℓ-m)!/(ℓ+m)!]
    cfac = function (l)
        ratio = one(T)
        for k in (l - m + 1):(l + m)
            ratio /= T(k)
        end
        return sqrt(T(2l + 1) / T(2) * ratio)
    end
    P[m + 1] = cfac(m) * pmm
    pl1 = x * T(2m + 1) * pmm                # unnormalized P_{m+1}^m
    (m + 1 <= lmax) && (P[m + 2] = cfac(m + 1) * pl1)
    pl2 = pmm
    for l in (m + 2):lmax
        pl = (x * T(2l - 1) * pl1 - T(l + m - 1) * pl2) / T(l - m)
        P[l + 1] = cfac(l) * pl
        pl2 = pl1
        pl1 = pl
    end
    return P
end

"""
Return a scalar zonal-component projection of the steady viscous mean flow.
Temperature input uses the public no-factorial normalization and `Ra` is
shell-gap based. `lmax` sets the retained vector-potential degree. This legacy
single-phase interface cannot represent the complete 3D velocity; use the
basic-state constructors and `mean_flow_velocity` for physical fields.
"""
function solve_thermal_wind_coupled!(uphi_coeffs::Dict{Int,Vector{T}},
                                     duphi_dr_coeffs::Dict{Int,Vector{T}},
                                     theta_coeffs::Dict{Int,Vector{T}},
                                     m_bs::Int,
                                     cd,
                                     r_i::T, r_o::T, Ra::T, Pr::T;
                                     mechanical_bc::Symbol=:no_slip,
                                     E::T=T(1e-4),
                                     lmax::Union{Nothing,Int}=nothing) where T<:Real

    mechanical_bc in (:no_slip,:stress_free) || error("mechanical_bc must be :no_slip or :stress_free")
    L = lmax === nothing ? max(2,maximum(keys(theta_coeffs);init=0)+2) : lmax
    theta = Dict((l,m_bs)=>v for (l,v) in theta_coeffs)
    _project_mean_flow!((uphi=uphi_coeffs, duphi=duphi_dr_coeffs), theta, cd.x, cd.D1, cd.D2,
                        E, Ra, Pr, L, abs(m_bs); mechanical_bc=mechanical_bc,
                        keep=k->k[2]==m_bs, key=first)
    return nothing
end
