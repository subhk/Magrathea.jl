# =============================================================================
#  Boundary-condition utilities in toroidal–poloidal representation
#
#  This module contains boundary condition implementations for:
#  - Mechanical (velocity) boundary conditions
#  - Thermal boundary conditions
#  - Magnetic field boundary conditions
# =============================================================================

using SparseArrays: SparseMatrixCSC

"""
    velocity_from_potentials(op, P, T)

Convert poloidal (`P`) and toroidal (`T`) potentials defined on the operator
collocation grid into velocity components `(u_r, u_θ, u_φ)` for the azimuthal
wavenumber `op.params.m`.

The formulas follow the standard decomposition

```
    u = ∇ × ∇ × (P r̂) + ∇ × (T r̂)
```

assuming fields vary as `exp(i m φ)`.  The returned arrays share the same shape
as the input potentials. Polar points use the regular limits described in
[`potentials_to_velocity`](@ref). For `|m| = 1`, optional operator properties
`costheta`, `cosθ`, `theta`, or `θ` identify the poles; otherwise their orientation
is inferred from `Dθ*sintheta`. No angular coordinates are required for `m = 0`.
"""
function velocity_from_potentials(op, P, T)
    Nr, Nθ = size(P)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    size(op.Dr, 1) == Nr || throw(DimensionMismatch("Dr must have $Nr rows"))
    size(op.Dr, 2) == Nr || throw(DimensionMismatch("Dr must have $Nr columns"))
    size(op.Dθ, 1) == Nθ || throw(DimensionMismatch("Dθ must have $Nθ rows"))
    size(op.Dθ, 2) == Nθ || throw(DimensionMismatch("Dθ must have $Nθ columns"))
    size(op.Lθ, 1) == Nθ || throw(DimensionMismatch("Lθ must have $Nθ rows"))
    size(op.Lθ, 2) == Nθ || throw(DimensionMismatch("Lθ must have $Nθ columns"))

    # Angular derivatives
    dθ_T = T * op.Dθ'
    lap_ang_P = P * op.Lθ'

    # Radial derivatives of the potentials
    dr_P = op.Dr * P

    inv_r = _inv_r_vector(_get_inv_r(op, Nr), Nr)
    im_m = _get_im_m(op)
    dθ_dr_P = dr_P * op.Dθ'
    if iszero(im_m)
        # Retain real output for real axisymmetric inputs and do not require
        # an angular-coordinate property that this branch never uses.
        return -lap_ang_P .* (inv_r .* inv_r), dθ_dr_P .* inv_r, -dθ_T .* inv_r
    end

    sinθ = if hasproperty(op, :inv_r_sinθ)
        inv_r_sinθ = _get_inv_r_sinθ(op, inv_r, Nr, Nθ)
        inv_r[1] ./ vec(inv_r_sinθ[1, :])
    else
        _get_sinθ(op, Nθ)
    end
    cosθ = if hasproperty(op, :costheta)
        op.costheta
    elseif hasproperty(op, :cosθ)
        getproperty(op, :cosθ)
    elseif hasproperty(op, :theta)
        cos.(op.theta)
    elseif hasproperty(op, :θ)
        cos.(getproperty(op, :θ))
    else
        nothing
    end
    cosθ === nothing || length(cosθ) == Nθ || throw(DimensionMismatch(
        "costheta must have length $Nθ"))
    CT = promote_type(eltype(lap_ang_P), eltype(dθ_dr_P), eltype(dθ_T),
                      eltype(inv_r), eltype(sinθ), typeof(im_m))
    return _scale_potential_velocity!(CT.(lap_ang_P), CT.(dθ_dr_P), CT.(dθ_T),
                                      dr_P, P, T, inv_r, sinθ, op.Dθ, im_m;
                                      costheta=cosθ)
end

# This is shared by both reconstruction entrypoints. The first three arrays
# initially contain LθP, ∂θ∂rP, and ∂θT, respectively. Keeping the m=0 branch
# separate avoids evaluating an undefined 0/sinθ, including for zero fields.
function _scale_potential_velocity!(ur, uθ, uφ, dP_dr, P, Tor,
                                    inv_r, sintheta, Dθ, im_m; costheta=nothing)
    Nr, Nθ = size(P)
    if iszero(im_m)
        @inbounds for j in 1:Nθ, i in 1:Nr
            ur[i, j] = -ur[i, j] * inv_r[i]^2
            uθ[i, j] *= inv_r[i]
            uφ[i, j] *= -inv_r[i]
        end
        return ur, uθ, uφ
    end

    RT = typeof(float(real(zero(eltype(sintheta)))))
    pole_tolerance = 4eps(RT)
    @inbounds for j in 1:Nθ
        if abs(sintheta[j]) <= pole_tolerance
            _scale_polar_velocity!(ur, uθ, uφ, dP_dr, P, Tor, inv_r,
                                    sintheta, Dθ, im_m, j, costheta)
        else
            inv_sinθ = inv(sintheta[j])
            for i in 1:Nr
                inv_r_sinθ = inv_r[i] * inv_sinθ
                ur[i, j] = -ur[i, j] * inv_r[i]^2
                uθ[i, j] = uθ[i, j] * inv_r[i] + im_m * Tor[i, j] * inv_r_sinθ
                uφ[i, j] = im_m * dP_dr[i, j] * inv_r_sinθ - uφ[i, j] * inv_r[i]
            end
        end
    end
    return ur, uθ, uφ
end

function _scale_polar_velocity!(ur, uθ, uφ, dP_dr, P, Tor, inv_r,
                                sintheta, Dθ, im_m, j, costheta)
    m = im_m / im
    isreal(m) && isinteger(real(m)) || throw(ArgumentError(
        "Polar velocity requires an integer azimuthal wavenumber"))
    order = abs(real(m))
    RT = typeof(float(real(zero(eltype(uθ)))))
    # Input roundoff, including sin(pi), may leave a tiny nonzero endpoint.
    # Relative tolerances have no unit floor: a small irregular field must not
    # silently become a regular zero field.
    input_eps = eps(RT)
    for data in (P, Tor, Dθ, sintheta)
        data_type = typeof(float(real(zero(eltype(data)))))
        input_eps = max(input_eps, eps(data_type))
    end
    roundoff = 64input_eps
    dscale = sum(abs, view(Dθ, j, :))
    for i in axes(P, 1)
        pscale = maximum(abs, view(P, i, :))
        tscale = maximum(abs, view(Tor, i, :))
        abs(P[i, j]) <= roundoff * pscale &&
        abs(Tor[i, j]) <= roundoff * tscale || throw(ArgumentError(
            "Nonaxisymmetric potentials must vanish at a pole (angular column $j)"))
        isfinite(uθ[i, j]) && isfinite(uφ[i, j]) || throw(ArgumentError(
            "Finite angular derivatives are required at a pole (angular column $j)"))
        if order > 1
            dpscale = maximum(abs, view(dP_dr, i, :))
            abs(uθ[i, j]) <= roundoff * dscale * dpscale &&
            abs(uφ[i, j]) <= roundoff * dscale * tscale || throw(ArgumentError(
                "For |m| > 1, regular potentials have zero first angular derivatives at a pole"))
        end
    end

    # Higher orders have zero polar velocity. An identically zero first-order
    # limit also needs no north/south orientation information.
    if order > 1 || (all(iszero, view(uθ, :, j)) && all(iszero, view(uφ, :, j)))
        ur[:, j] .= 0
        uθ[:, j] .= 0
        uφ[:, j] .= 0
        return nothing
    end
    c = if costheta === nothing
        derivative = sum(Dθ[j, k] * sintheta[k] for k in eachindex(sintheta))
        # An unresolved derivative cannot distinguish the two poles reliably.
        isreal(derivative) && isfinite(derivative) && abs(derivative) > sqrt(input_eps) ||
            throw(ArgumentError("Cannot identify north/south pole; supply costheta for |m| = 1"))
        sign(real(derivative))
    else
        value = costheta[j]
        cosine_roundoff = max(roundoff, 64eps(typeof(float(real(value)))))
        isreal(value) && isfinite(value) && abs(abs(value) - 1) <= cosine_roundoff ||
            throw(ArgumentError("costheta must equal +1 or -1 at a pole"))
        sign(real(value))
    end
    for i in axes(P, 1)
        hp, ht = uθ[i, j], uφ[i, j]
        ur[i, j] = 0
        uθ[i, j] = (hp + im_m * ht / c) * inv_r[i]
        uφ[i, j] = (im_m * hp / c - ht) * inv_r[i]
    end
    return nothing
end

"""
    apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                         P, T, op;
                                         inner::Symbol=:no_slip,
                                         outer::Symbol=:no_slip)

Overwrite the boundary rows of the residual blocks `(res_r, res_θ, res_φ)` using
velocity boundary conditions derived from the toroidal–poloidal potentials
`(P, T)`.

Supported mechanical boundary types:

- `:no_slip`      → `u_r = u_θ = u_φ = 0`
- `:stress_free`  → `u_r = 0`, `∂_r u_θ = u_θ / r`, `∂_r u_φ = u_φ / r`

The function evaluates the necessary velocity components (and their radial
derivatives) internally from the potentials.
"""
function apply_mechanical_bc_from_potentials!(res_r, res_θ, res_φ,
                                              P, T, op;
                                              inner::Symbol=:no_slip,
                                              outer::Symbol=:no_slip)
    size(T) == size(P) || throw(DimensionMismatch("P and T must have same size"))
    size(res_r) == size(P) || throw(DimensionMismatch("res_r must match P size"))
    size(res_θ) == size(P) || throw(DimensionMismatch("res_θ must match P size"))
    size(res_φ) == size(P) || throw(DimensionMismatch("res_φ must match P size"))

    u_r, u_θ, u_φ = velocity_from_potentials(op, P, T)
    dr_uθ = op.Dr * u_θ
    dr_uφ = op.Dr * u_φ

    Nr = size(P, 1)
    inner_idx, outer_idx = _boundary_indices(op, Nr)
    inv_r = _get_inv_r(op, Nr)

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              inv_r, inner, inner_idx)

    enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                              u_r, u_θ, u_φ,
                              dr_uθ, dr_uφ,
                              inv_r, outer, outer_idx)
    return nothing
end

"""Apply one mechanical boundary condition at a selected radial boundary row."""
function enforce_mechanical_bc_at!(res_r, res_θ, res_φ,
                                   u_r, u_θ, u_φ,
                                   dr_uθ, dr_uφ,
                                   inv_r, bc::Symbol, idx::Int)
    if bc === :no_slip
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= u_θ[idx, :]
        res_φ[idx, :] .= u_φ[idx, :]
    elseif bc === :stress_free
        inv_r_val = _inv_r_at(inv_r, idx)
        res_r[idx, :] .= u_r[idx, :]
        res_θ[idx, :] .= dr_uθ[idx, :] .- u_θ[idx, :] .* inv_r_val
        res_φ[idx, :] .= dr_uφ[idx, :] .- u_φ[idx, :] .* inv_r_val
    else
        throw(ArgumentError("Unsupported mechanical boundary condition: $(bc)"))
    end
end

"""
    apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                      inner::Symbol=:fixed_temperature,
                                      outer::Symbol=:fixed_temperature,
                                      value_inner::Real=0.0,
                                      value_outer::Real=0.0,
                                      flux_inner::Real=0.0,
                                      flux_outer::Real=0.0)

Apply thermal boundary conditions directly to the temperature residual block
`res_T`.  The helper mirrors the mechanical routine but does not require
potentials explicitly; it is defined here so that a single module hosts all
boundary utilities for the toroidal–poloidal formulation.

Supported thermal boundary types:

- `:fixed_temperature` → Θ = prescribed value
- `:fixed_flux`        → ∂_r Θ = prescribed flux
"""
function apply_thermal_bc_from_potentials!(res_T, Θ, op;
                                           inner::Symbol=:fixed_temperature,
                                           outer::Symbol=:fixed_temperature,
                                           value_inner::Real=0.0,
                                           value_outer::Real=0.0,
                                           flux_inner::Real=0.0,
                                           flux_outer::Real=0.0)
    size(res_T) == size(Θ) || throw(DimensionMismatch("res_T must match Θ size"))

    dΘ_dr = op.Dr * Θ
    Nr = size(Θ, 1)
    inner_idx, outer_idx = _boundary_indices(op, Nr)
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, inner, value_inner, flux_inner, inner_idx)
    apply_thermal_bc_at!(res_T, Θ, dΘ_dr, outer, value_outer, flux_outer, outer_idx)
    return nothing
end

"""Apply one thermal boundary condition at a selected radial boundary row."""
function apply_thermal_bc_at!(res_T, Θ, dΘ_dr,
                              bc::Symbol,
                              value::Real,
                              flux::Real,
                              idx::Int)
    if bc === :fixed_temperature
        res_T[idx, :] .= Θ[idx, :] .- value
    elseif bc === :fixed_flux
        res_T[idx, :] .= dΘ_dr[idx, :] .- flux
    else
        throw(ArgumentError("Unsupported thermal boundary condition: $(bc)"))
    end
end

"""Return inner and outer radial row indices for either ascending or descending grids."""
function _boundary_indices(op, Nr::Int)
    if hasproperty(op, :r)
        r = op.r
        length(r) == Nr || throw(DimensionMismatch("r must have length $Nr"))
        return r[1] < r[end] ? (1, Nr) : (Nr, 1)
    end
    if hasproperty(op, :inv_r)
        inv_r = _get_inv_r(op, Nr)
        inv_r_vec = _inv_r_vector(inv_r, Nr)
        return inv_r_vec[1] > inv_r_vec[end] ? (1, Nr) : (Nr, 1)
    end
    return Nr, 1
end

"""Extract `im*m` from an operator-like object."""
function _get_im_m(op)
    if hasproperty(op, :im_m)
        return op.im_m
    elseif hasproperty(op, :m)
        return im * op.m
    elseif hasproperty(op, :params) && hasproperty(op.params, :m)
        return im * op.params.m
    end
    throw(ArgumentError("op must define `m` or `im_m` for azimuthal wavenumber"))
end

"""Extract or derive inverse-radius data from an operator-like object."""
function _get_inv_r(op, Nr::Int)
    if hasproperty(op, :inv_r)
        inv_r = op.inv_r
        size(inv_r, 1) == Nr || throw(DimensionMismatch("inv_r must have $Nr rows"))
        return inv_r
    elseif hasproperty(op, :r)
        r = op.r
        length(r) == Nr || throw(DimensionMismatch("r must have length $Nr"))
        return inv.(r)
    end
    throw(ArgumentError("op must define `inv_r` or `r`"))
end

"""Return `1/(r sin(theta))` as an `Nr x Ntheta` array."""
function _get_inv_r_sinθ(op, inv_r, Nr::Int, Nθ::Int)
    if hasproperty(op, :inv_r_sinθ)
        inv_r_sinθ = op.inv_r_sinθ
        size(inv_r_sinθ, 1) == Nr || throw(DimensionMismatch("inv_r_sinθ must have $Nr rows"))
        size(inv_r_sinθ, 2) == Nθ || throw(DimensionMismatch("inv_r_sinθ must have $Nθ columns"))
        return inv_r_sinθ
    end

    sinθ = _get_sinθ(op, Nθ)
    inv_sinθ = inv.(sinθ)
    inv_r_vec = _inv_r_vector(inv_r, Nr)
    return inv_r_vec .* inv_sinθ'
end

"""Extract or derive the sine of the angular grid."""
function _get_sinθ(op, Nθ::Int)
    if hasproperty(op, :sintheta)
        sinθ = op.sintheta
        length(sinθ) == Nθ || throw(DimensionMismatch("sintheta must have length $Nθ"))
        return sinθ
    elseif hasproperty(op, :sinθ)
        sinθ = getproperty(op, :sinθ)
        length(sinθ) == Nθ || throw(DimensionMismatch("sinθ must have length $Nθ"))
        return sinθ
    elseif hasproperty(op, :theta)
        θ = op.theta
        length(θ) == Nθ || throw(DimensionMismatch("theta must have length $Nθ"))
        return sin.(θ)
    elseif hasproperty(op, :θ)
        θ = getproperty(op, :θ)
        length(θ) == Nθ || throw(DimensionMismatch("θ must have length $Nθ"))
        return sin.(θ)
    end
    throw(ArgumentError("op must define `sintheta`, `sinθ`, `theta`, or `θ`"))
end

"""Convert inverse-radius data to a radial vector view."""
function _inv_r_vector(inv_r, Nr::Int)
    if ndims(inv_r) == 1
        length(inv_r) == Nr || throw(DimensionMismatch("inv_r must have length $Nr"))
        return inv_r
    elseif ndims(inv_r) == 2
        size(inv_r, 1) == Nr || throw(DimensionMismatch("inv_r must have $Nr rows"))
        return view(inv_r, :, 1)
    end
    throw(ArgumentError("inv_r must be a vector or matrix"))
end

"""Read inverse radius at one radial index from vector or matrix storage."""
function _inv_r_at(inv_r, idx::Int)
    if ndims(inv_r) == 1
        return inv_r[idx]
    elseif ndims(inv_r) == 2
        return inv_r[idx, 1]
    end
    throw(ArgumentError("inv_r must be a vector or matrix"))
end

# =============================================================================
#  Magnetic Field Boundary Conditions
# =============================================================================

"""
    spherical_bessel_j_logderiv(l::Int, x::Complex{T}) -> Complex{T}

Compute the logarithmic derivative of the spherical Bessel function of the first kind:

```math
\\frac{d}{dx}[\\log(j_l(x))] = \\frac{j'_l(x)}{j_l(x)}
```

# Mathematical Background

The spherical Bessel function jₗ(x) satisfies the recurrence relation:
```math
j'_l(x) = \\frac{l}{x} j_l(x) - j_{l+1}(x)
```

Therefore, the logarithmic derivative is:
```math
\\frac{j'_l(x)}{j_l(x)} = \\frac{l}{x} - \\frac{j_{l+1}(x)}{j_l(x)}
```

# Application: Conducting Inner Core

For a prescribed harmonic response proportional to exp(iωt) in a stationary
conducting core, a regular radial potential is proportional to j_l(kr), with:

```math
k = (1-i)\\sqrt{\\frac{\\omega}{2E_m}}
```

where ω is the oscillation frequency and Eₘ is the magnetic Ekman number.

The boundary condition becomes:
```math
f'(r_i) - k \\cdot \\frac{j'_l(kr_i)}{j_l(kr_i)} \\cdot f(r_i) = 0
```

This frequency-dependent relation is for forced problems. The MHD stability
solver instead evolves core coefficients with the unknown eigenvalue.

# Numerical Stability

- For |x| < 10⁻¹⁰, uses series expansion: j'ₗ/jₗ ≈ l/x
- Otherwise evaluates j_{l+1}/j_l by its continued fraction (modified Lentz), which
  stays accurate where jₗ itself underflows or overflows, for real and complex x

# Arguments

- `l::Int`: Spherical harmonic degree (l ≥ 0)
- `x::Complex{T}`: Complex argument (typically k·r)

# Returns

- `Complex{T}`: The logarithmic derivative j'ₗ(x)/jₗ(x)

# Examples

```julia
using SpecialFunctions

# Real argument
l = 2
x = 1.5
logderiv = spherical_bessel_j_logderiv(l, x)

# Complex argument (conducting boundary)
Em = 1e-3
omega = 0.1 + 0.5im  # Complex frequency
k = (1 - 1im) * sqrt(omega / (2*Em))
ri = 0.35
logderiv_complex = spherical_bessel_j_logderiv(l, k * ri)
```

# References

- Kore implementation: kore-main/bin/utils.py, lines 487-526
- Satapathy (2013): Boundary conditions for conducting cores
- Zhang & Fearn (1994): Hydromagnetic flow in planetary cores

# See Also

- The MHD eigenproblem evolves the conducting core explicitly instead of prescribing a frequency.
"""
function spherical_bessel_j_logderiv(l::Int, x::Complex{T}) where {T<:Real}
    T <: AbstractFloat || return spherical_bessel_j_logderiv(l, float(x))
    # For very small |x|, use series expansion: j_l(x) ≈ x^l / (2l+1)!!
    # so d/dx[log(j_l)] ≈ l/x
    if abs(x) < 1e-10
        return complex(T(l)) / x
    end

    # d/dx[log(j_l)] = l/x - j_{l+1}/j_l, with j_{l+1}/j_l = x/g and the continued
    # fraction g = 2l+3 - x²/(2l+5 - x²/(2l+7 - …)) evaluated by modified Lentz.
    # Forming j_l and j_{l+1} separately loses the ratio once j_l underflows
    # (small x, large l) or overflows (large complex x).
    tiny = sqrt(floatmin(T))
    g = complex(T(2l + 3)); C = g; D = zero(g)
    for k in 2:(10_000 + 4 * ceil(Int, abs(x)))
        b = T(2l + 2k + 1)
        D = b - x^2 * D; iszero(D) && (D = complex(tiny)); D = inv(D)
        C = b - x^2 / C; iszero(C) && (C = complex(tiny))
        Δ = C * D
        g *= Δ
        abs(Δ - 1) <= 4 * eps(T) && return T(l) / x - x / g
    end
    error("spherical_bessel_j_logderiv: continued fraction did not converge for l=$l, x=$x")
end

# Overload for real arguments (though we primarily use complex)
spherical_bessel_j_logderiv(l::Int, x::T) where {T<:Real} =
    spherical_bessel_j_logderiv(l, complex(x))

"""
    apply_magnetic_boundary_conditions!(A, B, op, section)

Apply the coefficient-space MHD tau constraints for `:f` or `:g`, including
interface rows in an evolving conducting core or mantle. Uses the same boundary data as
serial and distributed assembly. The highest residual coefficients are replaced.
"""
function apply_magnetic_boundary_conditions!(A::SparseMatrixCSC, B::SparseMatrixCSC,
                                              op, section::Symbol)
    section in (:f, :g) || throw(ArgumentError("Magnetic section must be :f or :g"))
    core_section = section == :f ? :fi : :gi
    mantle_section = section == :f ? :fm : :gm
    _apply_mhd_boundary_conditions!(A, B, op, (section, core_section, mantle_section))
    return nothing
end
