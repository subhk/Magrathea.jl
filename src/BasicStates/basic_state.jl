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
bc = Y20(0.1)  # Interpreted as flux when thermal_bc=:fixed_flux
bs = basic_state(cd, χ, E, Ra, Pr; temperature_bc=bc, thermal_bc=:fixed_flux)
```

# Physical Interpretation

The spherical harmonics Y_ℓm represent different angular patterns:

- **Y₀₀**: Uniform (spherically symmetric)
- **Y₁₀**: North-south dipole (one hemisphere hot, other cold)
- **Y₁₁**: East-west dipole
- **Y₂₀**: Equator-pole contrast (equator different from poles)
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

Create a spherical harmonic boundary condition for mode (ℓ, m).

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
"""Y00(amplitude=1.0) - Uniform (monopole) spherical harmonic"""
Y00(amplitude::Real=1.0) = SphericalHarmonicBC(0, 0, amplitude)

# ℓ = 1: Dipole
"""Y10(amplitude=1.0) - Axial dipole: cos(θ) pattern (north-south asymmetry)"""
Y10(amplitude::Real=1.0) = SphericalHarmonicBC(1, 0, amplitude)

"""Y11(amplitude=1.0) - Equatorial dipole: sin(θ)cos(φ) pattern"""
Y11(amplitude::Real=1.0) = SphericalHarmonicBC(1, 1, amplitude)

# ℓ = 2: Quadrupole
"""Y20(amplitude=1.0) - Axisymmetric quadrupole: (3cos²θ - 1) pattern (equator-pole contrast)"""
Y20(amplitude::Real=1.0) = SphericalHarmonicBC(2, 0, amplitude)

"""Y21(amplitude=1.0) - Tesseral quadrupole: sin(θ)cos(θ)cos(φ) pattern"""
Y21(amplitude::Real=1.0) = SphericalHarmonicBC(2, 1, amplitude)

"""Y22(amplitude=1.0) - Sectoral quadrupole: sin²(θ)cos(2φ) pattern (four-fold longitudinal)"""
Y22(amplitude::Real=1.0) = SphericalHarmonicBC(2, 2, amplitude)

# ℓ = 3: Octupole
"""Y30(amplitude=1.0) - Axisymmetric octupole"""
Y30(amplitude::Real=1.0) = SphericalHarmonicBC(3, 0, amplitude)

"""Y31(amplitude=1.0) - Tesseral octupole m=1"""
Y31(amplitude::Real=1.0) = SphericalHarmonicBC(3, 1, amplitude)

"""Y32(amplitude=1.0) - Tesseral octupole m=2"""
Y32(amplitude::Real=1.0) = SphericalHarmonicBC(3, 2, amplitude)

"""Y33(amplitude=1.0) - Sectoral octupole"""
Y33(amplitude::Real=1.0) = SphericalHarmonicBC(3, 3, amplitude)

# ℓ = 4: Hexadecapole
"""Y40(amplitude=1.0) - Axisymmetric hexadecapole"""
Y40(amplitude::Real=1.0) = SphericalHarmonicBC(4, 0, amplitude)

"""Y41(amplitude=1.0) - Tesseral hexadecapole m=1"""
Y41(amplitude::Real=1.0) = SphericalHarmonicBC(4, 1, amplitude)

"""Y42(amplitude=1.0) - Tesseral hexadecapole m=2"""
Y42(amplitude::Real=1.0) = SphericalHarmonicBC(4, 2, amplitude)

"""Y43(amplitude=1.0) - Tesseral hexadecapole m=3"""
Y43(amplitude::Real=1.0) = SphericalHarmonicBC(4, 3, amplitude)

"""Y44(amplitude=1.0) - Sectoral hexadecapole"""
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
                           outer_flux::T=zero(T)) where T

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
- `outer_flux` - Heat flux at outer boundary (only used if thermal_bc=:fixed_flux)
                 Positive = heat flowing outward, negative = heat flowing inward.
                 For standard convection with hot inner boundary, use negative flux
                 (e.g., outer_flux = -1/(r_o² × (1 - χ)) for unit total flux).

# Physical Interpretation
For fixed temperature BCs:
  - θ̄(r_i) = 1 (hot inner boundary)
  - θ̄(r_o) = 0 (cold outer boundary)

For fixed flux at outer:
  - θ̄(r_i) = 1 (hot inner boundary)
  - dθ̄/dr|_{r_o} = outer_flux (prescribed heat flux)

The conduction profile for ℓ=0 with fixed flux at outer is:
  θ̄_0(r) = √(4π) × [1 + outer_flux × r_o² × (1/r_i - 1/r)]
"""
function conduction_basic_state(cd::ChebyshevDiffn{T}, χ::T, lmax_bs::Int;
                                thermal_bc::Symbol=:fixed_temperature,
                                outer_flux::T=zero(T)) where T
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
The outer anomaly is `amplitude * P₂(cosθ)` for fixed temperature, or
`outer_flux_Y20 * P₂(cosθ)` for fixed flux. The inner temperature is uniform.

All three velocity components solve the steady Stokes–Coriolis equations in a
solenoidal vector-harmonic basis, with both mechanical boundaries enforced.
`Ra` is shell-gap based. This neglects momentum inertia and temperature
advection; use `basic_state_selfconsistent` to include both nonlinear effects.
Use `mean_flow_velocity` to evaluate the full vector field.
"""
function meridional_basic_state(cd::ChebyshevDiffn{T}, χ::T, E::T, Ra::T, Pr::T,
                               lmax_bs::Int, amplitude::T;
                               mechanical_bc::Symbol=:no_slip,
                               thermal_bc::Symbol=:fixed_temperature,
                               outer_flux_mean::T=zero(T),
                               outer_flux_Y20::T=zero(T)) where T

    r = cd.x
    Nr = length(r)
    r_i = T(χ)
    r_o = T(1.0)

    # Validate BCs
    if !(thermal_bc in (:fixed_temperature, :fixed_flux))
        error("thermal_bc must be :fixed_temperature or :fixed_flux, got: $thermal_bc")
    end

    # Initialize dictionaries
    theta_coeffs = Dict{Int,Vector{T}}()
    dtheta_dr_coeffs = empty(theta_coeffs)
    uphi_coeffs = empty(theta_coeffs)
    duphi_dr_coeffs = empty(theta_coeffs)

    # Spherical harmonic normalization for Y_20
    norm_Y20 = sqrt(T(5) / (T(4) * T(pi)))

    if thermal_bc == :fixed_temperature
        # =====================================================================
        # Fixed temperature at both boundaries
        # =====================================================================

        # ℓ=0 mode: uniform inner temp, zero outer temp
        theta_0, dtheta_0 = laplace_mode_profile(0, r, r_i, r_o,
                                                 sqrt(T(4)*T(pi)), zero(T);
                                                 outer_bc=:fixed_temperature)
        theta_coeffs[0] = theta_0
        dtheta_dr_coeffs[0] = dtheta_0

        # ℓ=2 mode: zero at inner, amplitude × Y_20 pattern at outer
        theta_2, dtheta_2 = laplace_mode_profile(2, r, r_i, r_o,
                                                zero(T), amplitude / norm_Y20;
                                                outer_bc=:fixed_temperature)
        theta_coeffs[2] = theta_2
        dtheta_dr_coeffs[2] = dtheta_2

    else  # fixed_flux
        # =====================================================================
        # Fixed temperature at inner, fixed flux at outer
        # =====================================================================

        # ℓ=0 mode: uniform inner temp, prescribed mean flux at outer
        # Normalize flux by √(4π) for spherical harmonic coefficient
        flux_0_normalized = outer_flux_mean * sqrt(T(4) * T(pi))
        theta_0, dtheta_0 = laplace_mode_profile(0, r, r_i, r_o,
                                                 sqrt(T(4)*T(pi)), flux_0_normalized;
                                                 outer_bc=:fixed_flux)
        theta_coeffs[0] = theta_0
        dtheta_dr_coeffs[0] = dtheta_0

        # ℓ=2 mode: zero at inner, prescribed Y_20 flux at outer
        # Use amplitude as the flux amplitude (overrides outer_flux_Y20 if non-zero)
        flux_Y20 = amplitude != zero(T) ? amplitude : outer_flux_Y20
        flux_2_normalized = flux_Y20 / norm_Y20
        theta_2, dtheta_2 = laplace_mode_profile(2, r, r_i, r_o,
                                                zero(T), flux_2_normalized;
                                                outer_bc=:fixed_flux)
        theta_coeffs[2] = theta_2
        dtheta_dr_coeffs[2] = dtheta_2
    end

    # Initialize velocity modes to zero
    uphi_coeffs[0] = zeros(T, Nr)
    duphi_dr_coeffs[0] = zeros(T, Nr)
    uphi_coeffs[2] = zeros(T, Nr)
    duphi_dr_coeffs[2] = zeros(T, Nr)

    # =========================================================================
    # Higher ℓ modes: zero (no higher-order boundary variations)
    # =========================================================================
    for ℓ in 1:lmax_bs
        if ℓ == 0 || ℓ == 2
            continue
        end
        theta_coeffs[ℓ] = zeros(T, Nr)
        dtheta_dr_coeffs[ℓ] = zeros(T, Nr)
        uphi_coeffs[ℓ] = zeros(T, Nr)
        duphi_dr_coeffs[ℓ] = zeros(T, Nr)
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
        coeff = _linear_interpolate(bs.r, coeffs, r_eval)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        theta_bar += coeff * Y
        dtheta_dtheta += coeff * dY_dtheta
        if haskey(bs.dtheta_dr_coeffs, ℓ)
            dtheta_dr += _linear_interpolate(bs.r, bs.dtheta_dr_coeffs[ℓ], r_eval) * Y
        end
    end

    for (ℓ, coeffs) in bs.uphi_coeffs
        ℓ > lmax && continue
        coeff = _linear_interpolate(bs.r, coeffs, r_eval)
        Y = norms[ℓ + 1] * P[ℓ + 1]
        dY_dtheta = -sinθ * norms[ℓ + 1] * dPdx[ℓ + 1]
        uphi_bar += coeff * Y
        duphi_dtheta += coeff * dY_dtheta
        if haskey(bs.duphi_dr_coeffs, ℓ)
            duphi_dr += _linear_interpolate(bs.r, bs.duphi_dr_coeffs[ℓ], r_eval) * Y
        end
    end

    if bs.flow !== nothing
        # Axisymmetric uφ is the toroidal vector harmonic t/r * ∂θY.
        # Use it directly instead of differentiating a scalar projection.
        uphi_bar = zero(T); duphi_dr = zero(T); duphi_dtheta = zero(T)
        for ((l,m),values) in bs.flow.t
            m==0 || continue
            t = _mean_barycentric(bs.r,values,r_eval)
            dt = _mean_barycentric(bs.r,bs.flow.dt[(l,m)],r_eval)
            Y = norms[l+1]*P[l+1]
            dY = -sinθ*norms[l+1]*dPdx[l+1]
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

"""Return Legendre values and x-derivatives up to `lmax` at one point."""
function _legendre_values_and_derivs(lmax::Int, x::T) where T
    P = zeros(T, lmax + 1)
    dPdx = zeros(T, lmax + 1)
    P[1] = one(T)
    if lmax >= 1
        P[2] = x
    end
    for l in 2:lmax
        P[l + 1] = ((2 * l - 1) * x * P[l] - (l - 1) * P[l - 1]) / l
    end

    dPdx[1] = zero(T)
    if lmax >= 1
        denom = one(T) - x * x
        tol = sqrt(eps(T))
        for l in 1:lmax
            if abs(denom) < tol
                dPdx[l + 1] = zero(T)
            else
                dPdx[l + 1] = l * (P[l] - x * P[l + 1]) / denom
            end
        end
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
Construct a conductive-temperature / steady Stokes–Coriolis basic state.
Both shell boundaries satisfy the selected mechanical condition. Temperature
and velocity retain cosine (`m>0`) and sine (`m<0`) modes. Public temperature
amplitudes multiply the associated Legendre function and its real azimuthal
factor; stored coefficients use the historical no-factorial normalization.

`Ra` is shell-gap based. Momentum inertia and thermal advection are omitted;
`nonaxisymmetric_basic_state_selfconsistent` includes both nonlinear effects.
`coupled_thermal_wind` and `include_meridional_flow` are compatibility keywords:
all values now use the complete viscous velocity solve.

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
                    # Get flux from outer_fluxes or amplitudes
                    flux_00 = get(outer_fluxes, (0,0), get(amplitudes, (0,0), zero(T)))
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

    # Both legacy values of coupled_thermal_wind now use the viscous solve.
    # Omitting meridional components would violate continuity for m != 0.
    # include_meridional_flow is retained for source compatibility; the complete
    # velocity is always returned because dropping components breaks continuity.
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
- `flux_bc` : SphericalHarmonicBC specifying heat flux at outer boundary
  (Cannot specify both temperature_bc and flux_bc)
- `mechanical_bc` : `:no_slip` (default) or `:stress_free`
- `lmax_bs` : Maximum ℓ for basic state (auto-determined if not specified)

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

1. If `temperature_bc` is `nothing` (or only Y00 component) and no `flux_bc`:
   → `conduction_basic_state` (pure conduction profile)

2. If boundary condition is axisymmetric (only m=0 modes):
   → `meridional_basic_state` (returns `BasicState`)

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
    _dtheta_sphere_projection(Kset, Lset, m, T) -> Dict{Tuple{Int,Int},T}

Projection coefficients M[(K,ℓ)] = ⟨∂Y_ℓ^m/∂θ, Y_K^m⟩ over the unit sphere,
for velocity modes K ∈ `Kset` and temperature modes ℓ ∈ `Lset`. Uses the exact
identity sin(θ)∂_θY_ℓ = ℓα_ℓ⁺Y_{ℓ+1} - (ℓ+1)α_ℓ⁻Y_{ℓ-1} together with
Gauss–Chebyshev quadrature of G(a,b) = ∫ P̄_a^m P̄_b^m / √(1-x²) dx. Unlike the
diagonal-approximation coupling, this retains the full ℓ±1, ℓ±3, … structure
required to be consistent with the coupled solver's (orthonormal) cosθ/sinθ∂θ
operators.
"""
function _dtheta_sphere_projection(Kset, Lset, m::Int, ::Type{T}) where {T<:Real}
    (isempty(Kset) || isempty(Lset)) && return Dict{Tuple{Int,Int},T}()
    lmax_needed = max(maximum(Kset), maximum(Lset) + 1)
    Nq = lmax_needed + 2
    Ptab = Matrix{T}(undef, Nq, lmax_needed + 1)
    for j in 1:Nq
        x = cos(T(π) * (T(j) - T(0.5)) / T(Nq))
        Ptab[j, :] .= _orthonormal_plm(lmax_needed, m, x)
    end
    Gw = T(π) / T(Nq)
    Gint = function (a, b)
        (a < m || b < m || a > lmax_needed || b > lmax_needed) && return zero(T)
        s = zero(T)
        @inbounds for j in 1:Nq
            s += Ptab[j, a + 1] * Ptab[j, b + 1]
        end
        return Gw * s
    end
    αplus(l)  = (l + 1)^2 - m^2 > 0 ? sqrt(T((l + 1)^2 - m^2) / T((2l + 1) * (2l + 3))) : zero(T)
    αminus(l) = l^2 - m^2 > 0 ? sqrt(T(l^2 - m^2) / T((2l - 1) * (2l + 1))) : zero(T)
    M = Dict{Tuple{Int,Int},T}()
    for ℓ in Lset
        ℓ < m && continue
        for K in Kset
            val = T(ℓ) * αplus(ℓ) * Gint(K, ℓ + 1)
            if ℓ - 1 >= m
                val -= T(ℓ + 1) * αminus(ℓ) * Gint(K, ℓ - 1)
            end
            abs(val) > eps(T) && (M[(K, ℓ)] = val)
        end
    end
    return M
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
    flow = _steady_mean_flow(theta,cd.x,cd.D1,cd.D2,E,Ra,Pr,L,abs(m_bs);
                             mechanical_bc=mechanical_bc)
    _,_,uphi,_,_,duphi = _mean_flow_components(flow)
    empty!(uphi_coeffs); empty!(duphi_dr_coeffs)
    for ((l,m),v) in uphi
        m==m_bs || continue
        uphi_coeffs[l]=v; duphi_dr_coeffs[l]=duphi[(l,m)]
    end
    return nothing
end

"""
    theta_derivative_coeff_3d(ℓ::Int, m::Int)

Compute θ-derivative coupling coefficients for ⟨∂Θ/∂θ, Y_Lm⟩ forcing.

Returns (c_plus, c_minus), including the spherical-harmonic normalization ratio
N_ℓ/N_L (4π-normalized Y, where N_ℓ = √((2ℓ+1)/4π)), so the coefficients are
consistent with `solve_thermal_wind_balance_3d!` (the analytically-validated
diagonal solver) and the test helper `coupling_coeff_plus_3d`:

- c_plus  = -(ℓ+1) × √[((ℓ+1)²-m²)/((2ℓ+1)(2ℓ+3))] × √((2ℓ+1)/(2ℓ+3))   (Y_ℓm → Y_{ℓ+1,m})
- c_minus = +ℓ     × √[(ℓ²-m²)/((2ℓ-1)(2ℓ+1))]     × √((2ℓ+1)/(2ℓ-1))   (Y_ℓm → Y_{ℓ-1,m})

The √2 from real-harmonic normalization (m≠0) cancels in the ratio, so the same
factor applies for all m.

# Example
```julia
c_plus, c_minus = theta_derivative_coeff_3d(2, 1)  # For Y_21
# c_plus ≈ -1.2127 (coupling to Y_31)
# c_minus ≈ 1.1547 (coupling to Y_11)
```
"""
function theta_derivative_coeff_3d(ℓ::Int, m::Int)
    if ℓ < abs(m)
        return (0.0, 0.0)
    end

    c_plus = 0.0
    c_minus = 0.0

    # Coupling to ℓ+1 (includes norm ratio N_ℓ/N_{ℓ+1} = √((2ℓ+1)/(2ℓ+3)))
    if ℓ >= 0
        num_plus = (ℓ + 1)^2 - m^2
        den_plus = (2*ℓ + 1) * (2*ℓ + 3)
        if num_plus >= 0 && den_plus > 0
            c_plus = -(ℓ + 1) * sqrt(num_plus / den_plus) *
                     sqrt((2*ℓ + 1) / (2*ℓ + 3))
        end
    end

    # Coupling to ℓ-1 (includes norm ratio N_ℓ/N_{ℓ-1} = √((2ℓ+1)/(2ℓ-1)))
    if ℓ > abs(m)
        num_minus = ℓ^2 - m^2
        den_minus = (2*ℓ - 1) * (2*ℓ + 1)
        if num_minus >= 0 && den_minus > 0
            c_minus = ℓ * sqrt(num_minus / den_minus) *
                      sqrt((2*ℓ + 1) / (2*ℓ - 1))
        end
    end

    return (c_plus, c_minus)
end
