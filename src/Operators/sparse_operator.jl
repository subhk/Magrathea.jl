# =============================================================================
#  Sparse Linear Stability Operator
#
#  Implementation using ultraspherical spectral method for onset of convection
#  in rotating spherical shells using sparse Gegenbauer (ultraspherical)
#  polynomials.
#
#  References:
#  - Rekier et al. (2019), Reference implementation
#  - Barik et al. (2023), Earth and Space Science
#  - Dormy et al. (2004), JFM
# =============================================================================





# -----------------------------------------------------------------------------
# Parameters for sparse onset calculations
# -----------------------------------------------------------------------------

"""
    SparseOnsetParams

Parameters for onset of convection in rotating spherical shells.
Uses sparse ultraspherical spectral discretization.

# Fields
- `E::Float64`: Ekman number ν/(ΩL²)
- `Pr::Float64`: Prandtl number ν/κ
- `Ra::Float64`: Rayleigh number
- `ricb::Float64`: Inner core radius (outer radius normalized to 1.0)
- `m::Int`: Azimuthal wavenumber
- `lmax::Int`: Maximum spherical harmonic degree
- `N::Int`: Chebyshev truncation level (must be even)
- `symm::Int`: Equatorial symmetry (+1 symmetric, -1 antisymmetric)
- `bci::Int`: Inner boundary condition (0=stress-free, 1=no-slip)
- `bco::Int`: Outer boundary condition (0=stress-free, 1=no-slip)
- `bci_thermal::Int`: Inner thermal BC (0=fixed temp, 1=fixed flux)
- `bco_thermal::Int`: Outer thermal BC (0=fixed temp, 1=fixed flux)
"""
struct SparseOnsetParams{T<:Real}
    # Dimensionless parameters
    E::T                           # Ekman number
    Pr::T                          # Prandtl number
    Ra::T                          # Rayleigh number
    ricb::T                        # Inner radius (ro = 1.0)

    # Wave parameters
    m::Int                         # Azimuthal wavenumber
    lmax::Int                      # Max spherical harmonic degree
    symm::Int                      # Equatorial symmetry (±1)

    # Resolution
    N::Int                         # Chebyshev truncation

    # Boundary conditions
    bci::Int                       # Inner mechanical BC (1=no-slip)
    bco::Int                       # Outer mechanical BC (1=no-slip)
    bci_thermal::Int               # Inner thermal BC (0=fixed temp)
    bco_thermal::Int               # Outer thermal BC (0=fixed temp)

    # Heating type
    heating::Symbol                # :internal or :differential

    # Derived quantities
    L::T                           # Shell thickness
    Etherm::T                      # Thermal Ekman number

    function SparseOnsetParams{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
                                bci, bco, bci_thermal, bco_thermal, heating,
                                L, Etherm) where {T<:Real}
        0 < ricb < 1 || throw(ArgumentError(
            "ricb must be in (0,1), got $ricb"))
        E > 0 || throw(ArgumentError(
            "Ekman number E must be positive, got $E"))
        Pr > 0 || throw(ArgumentError(
            "Prandtl number Pr must be positive, got $Pr"))
        m >= 0 || throw(ArgumentError(
            "Azimuthal wavenumber m must be non-negative, got $m"))
        lmax >= m || throw(ArgumentError(
            "lmax must be >= m, got lmax=$lmax, m=$m"))
        N >= 8 && iseven(N) || throw(ArgumentError(
            "N must be even and >= 8, got $N"))
        symm in (-1, 1) || throw(ArgumentError(
            "symm must be +1 or -1, got $symm"))
        heating in (:internal, :differential) || throw(ArgumentError(
            "heating must be :internal or :differential, got :$heating"))
        bci in (0, 1) || throw(ArgumentError(
            "bci must be 0 (stress-free) or 1 (no-slip), got $bci"))
        bco in (0, 1) || throw(ArgumentError(
            "bco must be 0 (stress-free) or 1 (no-slip), got $bco"))
        bci_thermal in (0, 1) || throw(ArgumentError(
            "bci_thermal must be 0 (fixed temperature) or 1 (fixed flux), got $bci_thermal"))
        bco_thermal in (0, 1) || throw(ArgumentError(
            "bco_thermal must be 0 (fixed temperature) or 1 (fixed flux), got $bco_thermal"))

        new{T}(E, Pr, Ra, ricb, m, lmax, symm, N,
               bci, bco, bci_thermal, bco_thermal, heating, L, Etherm)
    end
end

# Outer constructor to infer type. Pr defaults to 1; the default cannot be
# written `one(T)` because keyword defaults are evaluated before T is bound.
function SparseOnsetParams(; E::T, Pr::Union{T,Nothing}=nothing, Ra::T, ricb::T,
                          m::Int, lmax::Int, symm::Int=1, N::Int,
                          bci::Int=1, bco::Int=1,
                          bci_thermal::Int=0, bco_thermal::Int=0,
                          heating::Symbol=:differential) where {T<:Real}
    Pr_T = something(Pr, one(T))
    L = one(T) - ricb
    Etherm = E / Pr_T
    return SparseOnsetParams{T}(E, Pr_T, Ra, ricb, m, lmax, symm, N,
                             bci, bco, bci_thermal, bco_thermal, heating, L, Etherm)
end

# -----------------------------------------------------------------------------
# Sparse operator storage
# -----------------------------------------------------------------------------

"""
    SparseStabilityOperator

Stores pre-computed sparse radial operators and problem parameters.

Every radial operator maps Chebyshev coefficients to Chebyshev coefficients;
the `_u`/`_v`/`_h` suffix names the equation (row section) that uses it, which
fixes its radial weighting: r⁴ for the poloidal 2curl equation, r² for the
toroidal 1curl equation, r³ (differential) or r² (internal heating) for the heat
equation. `assemble_sparse_matrices` projects each residual to its ultraspherical
test basis (C⁴ for u, C² for v and h).
"""
struct SparseStabilityOperator{T<:Real}
    params::SparseOnsetParams{T}

    # Radial operators for poloidal velocity (section u, 2curl)
    # Naming: r{power}_D{deriv}_u
    r0_D0_u::SparseMatrixCSC{T,Int}
    r2_D0_u::SparseMatrixCSC{T,Int}
    r2_D2_u::SparseMatrixCSC{T,Int}  # For viscous diffusion
    r3_D0_u::SparseMatrixCSC{T,Int}  # For Coriolis coupling
    r3_D1_u::SparseMatrixCSC{T,Int}
    r4_D0_u::SparseMatrixCSC{T,Int}  # For buoyancy coupling
    r4_D1_u::SparseMatrixCSC{T,Int}  # For Coriolis coupling
    r4_D2_u::SparseMatrixCSC{T,Int}
    r3_D3_u::SparseMatrixCSC{T,Int}
    r4_D4_u::SparseMatrixCSC{T,Int}

    # Radial operators for toroidal velocity (section v, 1curl)
    r0_D0_v::SparseMatrixCSC{T,Int}
    r1_D0_v::SparseMatrixCSC{T,Int}  # For toroidal-poloidal coupling
    r1_D1_v::SparseMatrixCSC{T,Int}
    r2_D0_v::SparseMatrixCSC{T,Int}  # For Coriolis (toroidal diagonal)
    r2_D1_v::SparseMatrixCSC{T,Int}  # For toroidal-poloidal coupling
    r2_D2_v::SparseMatrixCSC{T,Int}

    # Radial operators for temperature (section h)
    r0_D0_h::SparseMatrixCSC{T,Int}
    r1_D0_h::SparseMatrixCSC{T,Int}  # For differential heating
    r1_D1_h::SparseMatrixCSC{T,Int}
    r2_D0_h::SparseMatrixCSC{T,Int}
    r2_D1_h::SparseMatrixCSC{T,Int}  # For differential heating
    r2_D2_h::SparseMatrixCSC{T,Int}
    r3_D0_h::SparseMatrixCSC{T,Int}  # For differential heating
    r3_D2_h::SparseMatrixCSC{T,Int}  # For differential heating

    # l-mode information
    ll_top::Vector{Int}  # l values for poloidal (equatorially symmetric)
    ll_bot::Vector{Int}  # l values for toroidal (equatorially antisymmetric)
    nl_modes::Int        # Total number of l modes

    # Matrix size
    matrix_size::Int
end

"""
    SparseStabilityOperator(params::SparseOnsetParams)

Construct the sparse operator by pre-computing all radial operators.
"""
function SparseStabilityOperator(params::SparseOnsetParams{T}) where {T}
    N = params.N
    ri, ro = params.ricb, one(T)

    @info "Building sparse operators" N=N ricb=ri

    # Pre-compute radial operators for poloidal velocity
    @debug "Computing poloidal operators..."
    r0_D0_u = sparse_radial_operator(0, 0, N, ri, ro)
    r2_D0_u = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D2_u = sparse_radial_operator(2, 2, N, ri, ro)  # For viscous diffusion
    r3_D0_u = sparse_radial_operator(3, 0, N, ri, ro)  # For Coriolis coupling
    r3_D1_u = sparse_radial_operator(3, 1, N, ri, ro)
    r4_D0_u = sparse_radial_operator(4, 0, N, ri, ro)  # For buoyancy coupling
    r4_D1_u = sparse_radial_operator(4, 1, N, ri, ro)  # For Coriolis coupling
    r4_D2_u = sparse_radial_operator(4, 2, N, ri, ro)
    r3_D3_u = sparse_radial_operator(3, 3, N, ri, ro)
    r4_D4_u = sparse_radial_operator(4, 4, N, ri, ro)

    # Pre-compute radial operators for toroidal velocity
    @debug "Computing toroidal operators..."
    r0_D0_v = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_v = sparse_radial_operator(1, 0, N, ri, ro)  # For toroidal-poloidal coupling
    r1_D1_v = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_v = sparse_radial_operator(2, 0, N, ri, ro)  # For Coriolis (toroidal diagonal)
    r2_D1_v = sparse_radial_operator(2, 1, N, ri, ro)  # For toroidal-poloidal coupling
    r2_D2_v = sparse_radial_operator(2, 2, N, ri, ro)

    # Pre-compute radial operators for temperature
    @debug "Computing temperature operators..."
    r0_D0_h = sparse_radial_operator(0, 0, N, ri, ro)
    r1_D0_h = sparse_radial_operator(1, 0, N, ri, ro)  # For differential heating
    r1_D1_h = sparse_radial_operator(1, 1, N, ri, ro)
    r2_D0_h = sparse_radial_operator(2, 0, N, ri, ro)
    r2_D1_h = sparse_radial_operator(2, 1, N, ri, ro)  # For differential heating
    r2_D2_h = sparse_radial_operator(2, 2, N, ri, ro)
    r3_D0_h = sparse_radial_operator(3, 0, N, ri, ro)  # For differential heating
    r3_D2_h = sparse_radial_operator(3, 2, N, ri, ro)  # For differential heating

    # Determine l-mode structure based on equatorial symmetry
    ll_top, ll_bot = compute_l_modes(params.m, params.lmax, params.symm)
    n_poloidal = length(ll_top)
    n_toroidal = length(ll_bot)
    n_temperature = n_poloidal
    nl_modes = n_poloidal + n_toroidal

    # Each l-mode has (N+1) radial DOFs
    # Total size: poloidal + toroidal + temperature
    n_per_mode = N + 1
    matrix_size = (n_poloidal + n_toroidal + n_temperature) * n_per_mode

    @info "Sparse operator built" poloidal_modes=length(ll_top) toroidal_modes=length(ll_bot) matrix_size="$(matrix_size) × $(matrix_size)" sparsity="~$(estimate_sparsity(N, n_poloidal, n_toroidal))%"

    return SparseStabilityOperator{T}(
        params,
        r0_D0_u, r2_D0_u, r2_D2_u, r3_D0_u, r3_D1_u, r4_D0_u, r4_D1_u, r4_D2_u, r3_D3_u, r4_D4_u,
        r0_D0_v, r1_D0_v, r1_D1_v, r2_D0_v, r2_D1_v, r2_D2_v,
        r0_D0_h, r1_D0_h, r1_D1_h, r2_D0_h, r2_D1_h, r2_D2_h, r3_D0_h, r3_D2_h,
        ll_top, ll_bot, nl_modes,
        matrix_size
    )
end

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

"""
    compute_l_modes(m, lmax, symm)

Compute the l-mode indices for given m, lmax, and equatorial symmetry.
Returns (ll_top, ll_bot) where:
- ll_top: l-modes for poloidal velocity (equatorially symmetric if symm=1)
- ll_bot: l-modes for toroidal velocity (equatorially antisymmetric if symm=-1)
"""
function compute_l_modes(m::Int, lmax::Int, symm::Int)
    m >= 0 || throw(ArgumentError("m must be non-negative, got $m"))
    lmax >= m || throw(ArgumentError("lmax must be >= m, got lmax=$lmax, m=$m"))
    symm in (-1, 0, 1) || throw(ArgumentError("symm must be -1, 0, or 1, got $symm"))

    # Following Kore's algorithm (bin/utils.py:174-183, function ell())
    # Key insight: For m=0, l-range is 1:(lmax+1) for spectral completeness
    #              For m>0, l-range is m:lmax as usual
    # This uses sign(m) to handle m=0 correctly

    if symm == 0
        # Both symmetries (full problem)
        if m == 0
            ll = collect(1:(lmax+1))
        else
            ll = collect(m:lmax)
        end
        return ll, ll
    end

    # Symmetric parameter: s=0 for antisymmetric, s=1 for symmetric
    s = div(symm + 1, 2)

    # Generate full l-range
    if m == 0
        ll = collect(1:(lmax+1))  # For m=0: l ∈ [1, lmax+1]
        sign_m = 0
    else
        ll = collect(m:lmax)      # For m>0: l ∈ [m, lmax]
        sign_m = 1
    end

    # Select modes based on parity (following Kore's indexing)
    # idp: indices for poloidal modes
    # idt: indices for toroidal modes
    idp_start = (sign_m + s) % 2 + 1      # Convert to 1-based indexing
    idt_start = (sign_m + s + 1) % 2 + 1

    ll_top = ll[idp_start:2:end]
    ll_bot = ll[idt_start:2:end]

    return ll_top, ll_bot
end

"""
    estimate_sparsity(N, n_poloidal, n_toroidal)

Estimate the percentage of nonzero entries in the assembled matrices.
"""
function estimate_sparsity(N::Int, n_poloidal::Int, n_toroidal::Int)
    # Each l-mode couples to l±1, l±2 neighbors
    # Average coupling: ~5 l-modes per row
    # Each operator contributes ~O(N) nonzeros per radial mode
    nl_modes = n_poloidal + n_toroidal
    nnz_estimate = 5 * nl_modes * (N + 1)^2

    # Total entries: (N+1) blocks for poloidal, toroidal, temperature
    total_size = (2 * n_poloidal + n_toroidal) * (N + 1)
    sparsity = 100.0 * (1.0 - nnz_estimate / total_size^2)

    return round(sparsity, digits=2)
end

# -----------------------------------------------------------------------------
# Operator functions (for rotating convection)
# -----------------------------------------------------------------------------

"""
    operator_u(op, l)

Poloidal velocity time derivative operator for the B matrix.
Returns L(L*r²D⁰ - 2r³D¹ - r⁴D²) where L = l(l+1).
Implements op.u(l, 'u', 'upol', 0) from Kore.

This represents r⁴·r̂·∇×∇×u (the 2curl operator weighted by r⁴).
Note: Signs are opposite to the Coriolis operator.
"""
function operator_u(op::SparseStabilityOperator{T}, l::Int) where {T}
    # From Kore line 35: out = L*( L*r2_D0_u - 2*r3_D1_u - r4_D2_u )
    # Note: Signs are OPPOSITE to Coriolis operator (line 63)
    # This represents r⁴·r̂·∇×∇×u (2curl of u, weighted by r⁴)
    L = l * (l + 1)
    return L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
end

"""
    operator_coriolis_diagonal(op, l, m)

Coriolis force operator for diagonal (l,l) coupling.
Implements op.coriolis(l, 'u', 'upol', 0).

Returns: 2im * m * (-L*r²D⁰ + 2r³D¹ + r⁴D²)
"""
function operator_coriolis_diagonal(op::SparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    L = l * (l + 1)
    return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
end

"""
    operator_coriolis_offdiag(op, l, m, offset)

Coriolis force operator for off-diagonal (l, l±1) coupling.
Implements op.coriolis(l, 'u', 'utor', ±1): the poloidal (2curl, r⁴-weighted)
equation at degree l acting on the toroidal velocity at degree l±1.

Returns: [operator, offset] where offset indicates which l-mode it couples to.

Following Kore lines 68-86:
- For l-1: C = (l²-1)*sqrt(l²-m²)/(2l-1)
  out = 2*C*((l-1)*r³D⁰_u - r⁴D¹_u)
- For l+1: C = l*(l+2)*sqrt((l+m+1)*(l-m+1))/(2l+3)
  out = 2*C*(-(l+2)*r³D⁰_u - r⁴D¹_u)
"""
function operator_coriolis_offdiag(op::SparseStabilityOperator{T},
                                  l::Int, m::Int, offset::Int) where {T}
    if offset == -1
        # Coupling to l-1 mode (acts on toroidal coefficients)
        C = (l^2 - 1) * sqrt(T(l^2 - m^2)) / (2l - 1)
        mtx = 2 * C * ((l - 1) * op.r3_D0_u - op.r4_D1_u)
        return mtx, -1

    elseif offset == 1
        # Coupling to l+1 mode (acts on toroidal coefficients)
        C = l * (l + 2) * sqrt(T((l + m + 1) * (l - m + 1))) / (2l + 3)
        mtx = 2 * C * (-(l + 2) * op.r3_D0_u - op.r4_D1_u)
        return mtx, 1

    else
        error("offset must be ±1 for Coriolis off-diagonal")
    end
end

"""
    operator_viscous_diffusion(op, l, E)

Viscous diffusion operator: E * ∇²∇²u for sparse spectral method.
Implements op.viscous_diffusion(l, 'u', 'upol', 0).

Returns: E * L * (-L(l+2)(l-1)*r⁰D⁰ + 2L*r²D² - 4r³D³ - r⁴D⁴)
"""
function operator_viscous_diffusion(op::SparseStabilityOperator{T},
                                   l::Int, E::T) where {T}
    L = l * (l + 1)
    return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                    2 * L * op.r2_D2_u -
                    4 * op.r3_D3_u -
                    op.r4_D4_u)
end

"""
    operator_buoyancy(op, l, Ra, Pr)

Buoyancy operator: beyonce * L * r⁴ * θ for sparse spectral method.
Implements op.buoyancy(l, 'u', '', 0).

This couples the temperature field to the poloidal velocity equation.

Following Kore and Magrathea.jl:
- beyonce = -Ra_internal * E² / Pr (the negative sign is crucial!)
- Ra_internal = Ra / gap³ for gap-based Rayleigh number conversion
- L = l(l+1)
- r⁴ weighting matches the poloidal equation (2curl, weighted by r⁴)

Note: The Ra provided is assumed to be gap-based (using shell thickness L = r_o - r_i
as length scale). We convert to internal Ra using Ra_internal = Ra / gap³ because
the non-dimensionalization uses r_o as the length scale.
Reference: Barik et al. (2023), Kore parameters.py
"""
function operator_buoyancy(op::SparseStabilityOperator{T},
                          l::Int, Ra::T, Pr::T) where {T}
    # Convert gap-based Ra to internal Ra
    # Ra_internal = Ra_gap / gap^3 (gap = r_o - r_i = 1 - ricb when r_o = 1)
    E = op.params.E
    ricb = op.params.ricb
    gap = one(T) - ricb
    Ra_internal = Ra / gap^3

    # Beyonce factor = BV² = -Ra_internal * E² / Pr
    beyonce = -Ra_internal * E^2 / Pr

    # L factor
    L = l * (l + 1)

    # Buoyancy enters the r⁴-weighted poloidal equation, acting on θ
    return beyonce * L * op.r4_D0_u
end

"""
    operator_coriolis_v_to_u(op, l, m, offset)

Coriolis coupling between the toroidal equation (v rows) at degree l and the
poloidal velocity (u columns) at degree l±1.
Implements op.coriolis(l, 'v', 'upol', ±1).

This is the REVERSE coupling to operator_coriolis_offdiag (u rows ← v).
Both directions are required for correct rotating convection physics!
It lives in the r²-weighted 1curl equation, so it uses the toroidal-row
operators r¹D⁰_v and r²D¹_v, not the r⁴-weighted poloidal ones.

Following Kore operators.py lines 93-113:
- For l-1: C = (l²-1)*sqrt(l²-m²)/(2l-1)
  out = 2*C*((l-1)*r¹D⁰_v - r²D¹_v)
- For l+1: C = l*(l+2)*sqrt((l+m+1)*(l-m+1))/(2l+3)
  out = 2*C*(-(l+2)*r¹D⁰_v - r²D¹_v)

Returns: operator matrix
"""
function operator_coriolis_v_to_u(op::SparseStabilityOperator{T},
                                 l::Int, m::Int, offset::Int) where {T}
    if offset == -1
        # Toroidal equation at l, poloidal velocity at l-1
        C = (l^2 - 1) * sqrt(T(l^2 - m^2)) / (2l - 1)
        return 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)

    elseif offset == 1
        # Toroidal equation at l, poloidal velocity at l+1
        C = l * (l + 2) * sqrt(T((l + m + 1) * (l - m + 1))) / (2l + 3)
        return 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)

    else
        error("offset must be ±1 for Coriolis v→u coupling")
    end
end

# -----------------------------------------------------------------------------
# Toroidal velocity operators (section v, 1curl)
# -----------------------------------------------------------------------------

"""
    operator_u_toroidal(op, l)

Toroidal velocity time derivative operator for the B matrix.
Returns L * r²D⁰_v where L = l(l+1).
Implements op.u(l, 'v', 'utor', 0) from Kore.

The toroidal equation is weighted by r² (1curl equation), so all terms
including the time derivative must use r² weighting to maintain consistency.
"""
function operator_u_toroidal(op::SparseStabilityOperator{T}, l::Int) where {T}
    # For toroidal velocity time derivative operator
    # From Kore line 42: out = L*r2_D0_v  (section='v', component='utor', offdiag=0)
    # Comment: "r2* r.1curl(u)" - equation weighted by r²
    L = l * (l + 1)
    return L * op.r2_D0_v
end

"""
    operator_coriolis_toroidal(op, l, m)

Coriolis force acting on toroidal velocity.
Implements op.coriolis(l, 'u', 'utor', 0).

Returns: -2im * m * r²D⁰_v

Note: In Kore, this is multiplied by Gaspard = 1.0 for time scale Tau = 1/Omega.
"""
function operator_coriolis_toroidal(op::SparseStabilityOperator{T},
                                   l::Int, m::Int) where {T}
    # Following Kore line 121:
    # section == 'v', component == 'utor', offdiag == 0
    # out = -2j*par.m*r2_D0_v  (NO L factor!)
    # Multiplied by par.Gaspard = 1.0

    return -2im * m * op.r2_D0_v
end

"""
    operator_viscous_toroidal(op, l, E)

Viscous diffusion operator for toroidal velocity: E * ∇²u_toroidal.
Implements op.viscous_diffusion(l, 'v', 'utor', 0).

Returns: E * L * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²)

where L = l(l+1).
"""
function operator_viscous_toroidal(op::SparseStabilityOperator{T},
                                  l::Int, E::T) where {T}
    # From reference implementation line 192:
    # section == 'v', component == 'utor', offdiag == 0
    # out = L*( -L*r0_D0_v + 2*r1_D1_v + r2_D2_v )
    # Multiplied by par.ViscosD = Ek = E

    L = l * (l + 1)
    return E * L * (-L * op.r0_D0_v + 2 * op.r1_D1_v + op.r2_D2_v)
end

# -----------------------------------------------------------------------------
# Temperature operators (section h)
# -----------------------------------------------------------------------------

"""
    operator_theta(op, l)

Temperature operator for time derivative term.
Implements op.theta(l, 'h', '', 0).

Following Kore lines 712-716:
- For 'differential' heating: returns r³D⁰ (eq. times r³)
- For 'internal' heating: returns r²D⁰ (eq. times r²)
"""
function operator_theta(op::SparseStabilityOperator{T}, l::Int) where {T}
    if op.params.heating == :differential
        return op.r3_D0_h  # eq. times r³
    else  # :internal
        return op.r2_D0_h  # eq. times r²
    end
end

"""
    operator_thermal_diffusion(op, l, Etherm)

Thermal diffusion operator: (E/Pr) * ∇²θ.
Implements op.thermal_diffusion(l, 'h', '', 0).

Following Kore lines 758-761:
- For 'differential' heating: Etherm * (-L*r¹D⁰ + 2*r²D¹ + r³D²) (eq. times r³)
- For 'internal' heating: Etherm * (-L*r⁰D⁰ + 2*r¹D¹ + r²D²) (eq. times r²)

where Etherm = E/Pr and L = l(l+1).
"""
function operator_thermal_diffusion(op::SparseStabilityOperator{T},
                                   l::Int, Etherm::T) where {T}
    L = l * (l + 1)

    if op.params.heating == :differential
        # eq. times r³
        return Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
    else  # :internal
        # eq. times r²
        return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
    end
end

"""
    operator_thermal_advection(op, l)

Thermal advection operator: radial velocity advecting temperature.
Implements op.thermal_advection(l, 'h', 'upol', 0).

Following Kore lines 733-740:
- For 'differential' heating: L * r⁰D⁰ * (ricb/gap), dT/dr = -β*r⁻², eq. times r³
- For 'internal' heating: L * r²D⁰, dT/dr = -β*r, eq. times r²

This couples the poloidal velocity to the temperature equation.
"""
function operator_thermal_advection(op::SparseStabilityOperator{T},
                                   l::Int) where {T}
    L = l * (l + 1)

    if op.params.heating == :differential
        # dT/dr = -beta * r⁻², eq. times r³
        ricb = op.params.ricb
        gap = one(T) - ricb
        return L * op.r0_D0_h * (ricb / gap)
    else  # :internal
        # dT/dr = -beta * r, eq. times r²
        return L * op.r2_D0_h
    end
end

# -----------------------------------------------------------------------------
# Matrix assembly
# -----------------------------------------------------------------------------

"""
    assemble_sparse_matrices(op::SparseStabilityOperator)

Assemble the full sparse matrices A and B for the generalized eigenvalue problem:
    A * x = λ * B * x

Assembly structure from assemble.py. Unknowns are Chebyshev coefficients, in
blocks of N+1 per l-mode: poloidal velocity (`ll_top`), toroidal velocity
(`ll_bot`), then temperature (`ll_top`). Each residual is projected to the
ultraspherical basis matching its derivative order (C⁴ for the poloidal 2curl
equation, C² for the toroidal and heat equations), and the tau boundary rows
replace its highest-degree coefficients (last 4 rows of a poloidal block, last
2 of a toroidal or temperature block); see `_compute_sparse_bc`. This is the
hydrodynamic part of `assemble_mhd_matrices`: for the same l-modes the finite
spectrum matches the MHD pencil with `B0_type = no_field`.

B is zero on the tau rows, so the pencil has exactly one infinite eigenvalue
per boundary row; discard those (e.g. by magnitude) and keep the rest.

Returns: (A, B, interior_dofs, info). `interior_dofs` lists the differential
equation rows (all rows except the tau rows), not removable coefficient
columns: solve the full pencil, since deleting the tau rows and the matching
columns destroys the boundary conditions.
"""
function assemble_sparse_matrices(op::SparseStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    m = params.m
    E = params.E
    Pr = params.Pr
    Ra = params.Ra

    n_per_mode = N + 1
    nb_top = length(op.ll_top)  # Number of poloidal modes
    nb_bot = length(op.ll_bot)  # Number of toroidal modes

    # Matrix size
    n = op.matrix_size

    @info "Assembling sparse matrices" size="$n × $n" poloidal_modes=op.ll_top toroidal_modes=op.ll_bot

    # Initialize sparse matrices using COO vectors. Keep value storage tied to
    # the parameter precision; Coriolis terms still introduce complex values.
    A_rows = Int[]
    A_cols = Int[]
    A_vals = Complex{T}[]

    B_rows = Int[]
    B_cols = Int[]
    B_vals = Complex{T}[]

    # Project each residual to its derivative-order ultraspherical basis before
    # the tau rows truncate it, as the MHD assembly does. Fourth-order velocity
    # equations in a C⁰ test basis admit artificial growing modes even with
    # correct boundary rows. Unknowns and boundary functionals stay Chebyshev.
    C4 = _convert_up(T, 0, 4, N)
    C2 = _convert_up(T, 0, 2, N)
    poloidal_rows = nb_top * n_per_mode

    # Helper function to add a projected block to the COO triplets
    function add_block!(rows, cols, vals, block::SparseMatrixCSC,
                       row_offset::Int, col_offset::Int)
        block = (row_offset < poloidal_rows ? C4 : C2) * block
        I, J, V = findnz(block)
        @inbounds for k in eachindex(V)
            push!(rows, I[k] + row_offset)
            push!(cols, J[k] + col_offset)
            push!(vals, V[k])
        end
        return nothing
    end

    # =========================================================================
    # Section u (poloidal velocity, 2curl equation)
    # =========================================================================
    @debug "Assembling section u (poloidal)..."

    for (k, l) in enumerate(op.ll_top)
        row_base = (k - 1) * n_per_mode
        col_base = (k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # B matrix: Time derivative operator (negative per Magrathea.jl convention)
        # A matrix: RHS operators (Coriolis, viscous, buoyancy)
        # -----------------------------------------------------------------

        # Time derivative term: ∂u/∂t → -operator_u in B matrix (matches Kore)
        u_op = operator_u(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_op, row_base, col_base)

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion (appears with a minus sign in Kore)
        visc_op = operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_op, row_base, col_base)

        # Coriolis coupling to toroidal velocity (l±1)
        for offset in (-1, 1)
            l_coupled = l + offset
            k_coupled = findfirst(==(l_coupled), op.ll_bot)
            if k_coupled !== nothing
                col_coupled = (nb_top + k_coupled - 1) * n_per_mode

                cori_off, _ = operator_coriolis_offdiag(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_off,
                          row_base, col_coupled)

            end
        end

        # Buoyancy coupling to temperature field
        # Temperature field starts after 2*nb sections
        temp_col_base = (nb_top + nb_bot + k - 1) * n_per_mode
        buoy_op = operator_buoyancy(op, l, Ra, Pr)
        add_block!(A_rows, A_cols, A_vals, buoy_op,
                  row_base, temp_col_base)
    end

    # =========================================================================
    # Section v (toroidal velocity, 1curl equation)
    # =========================================================================
    @debug "Assembling section v (toroidal)..."

    for (k, l) in enumerate(op.ll_bot)
        row_base = (nb_top + k - 1) * n_per_mode
        col_base = (nb_top + k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # B matrix: Time derivative operator (negative per Magrathea.jl convention)
        # A matrix: RHS operators (Coriolis, viscous)
        # -----------------------------------------------------------------

        # Time derivative term: ∂v/∂t → -operator_u_toroidal in B matrix (matches Kore)
        u_tor_op = operator_u_toroidal(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_tor_op, row_base, col_base)

        # Coriolis force acting on toroidal velocity
        cori_tor_op = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor_op, row_base, col_base)

        # Viscous diffusion for toroidal velocity (minus sign per Kore)
        visc_tor_op = operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_tor_op, row_base, col_base)

        # Coriolis coupling between toroidal and poloidal velocity (l±1)
        # This coupling goes in the TOROIDAL equation (v-rows), coupling to POLOIDAL variable (u-columns)
        # Physical meaning: Coriolis force in toroidal equation depends on poloidal velocity
        for offset in (-1, 1)
            l_coupled = l + offset
            k_coupled = findfirst(==(l_coupled), op.ll_top)
            if k_coupled !== nothing
                col_coupled = (k_coupled - 1) * n_per_mode  # Column for u at l_coupled

                cori_v_to_u = operator_coriolis_v_to_u(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_v_to_u,
                          row_base, col_coupled)
            end
        end

        # Note: No direct buoyancy coupling for toroidal velocity
        # (Buoyancy forces poloidal flow, which then couples to toroidal via Coriolis)
    end

    # =========================================================================
    # Temperature equation (section h)
    # =========================================================================
    @debug "Assembling temperature equation..."

    for (k, l) in enumerate(op.ll_top)
        row_base = (nb_top + nb_bot + k - 1) * n_per_mode
        col_base = (nb_top + nb_bot + k - 1) * n_per_mode

        # -----------------------------------------------------------------
        # B matrix: Time derivative (weighted by r²)
        # -----------------------------------------------------------------
        # Reference: B = theta operator (r²D⁰ for non-anelastic)
        theta_op = operator_theta(op, l)
        add_block!(B_rows, B_cols, B_vals, theta_op, row_base, col_base)

        # -----------------------------------------------------------------
        # A matrix: RHS operators (thermal diffusion, advection)
        # -----------------------------------------------------------------

        # Thermal diffusion
        thermal_diff = operator_thermal_diffusion(op, l, params.Etherm)
        add_block!(A_rows, A_cols, A_vals, thermal_diff, row_base, col_base)

        # Thermal advection: coupling from poloidal velocity to temperature
        # This acts on the poloidal velocity field
        vel_col_base = (k - 1) * n_per_mode  # Poloidal velocity at same l-mode
        thermal_adv = operator_thermal_advection(op, l)
        add_block!(A_rows, A_cols, A_vals, thermal_adv, row_base, vel_col_base)
    end

    # =========================================================================
    # Apply boundary conditions at the COO stage, then build CSC once
    # =========================================================================
    # Tau BCs overwrite the highest-degree residual rows of each block. Doing
    # that on the assembled CSC forces O(nnz) structural insertions per row.
    # Instead drop the operator entries on the boundary rows and append the BC
    # functionals to the triplets, so the single sparse() build carries the BCs.
    @debug "Applying boundary conditions (COO stage)..."
    bc_rows, bcA = _compute_sparse_bc(op)
    _filter_coo_rows!(A_rows, A_cols, A_vals, bc_rows)
    _filter_coo_rows!(B_rows, B_cols, B_vals, bc_rows)  # B boundary rows are zeroed
    @inbounds for (r, c, v) in bcA
        push!(A_rows, r); push!(A_cols, c); push!(A_vals, v)
    end

    @debug "Converting to CSC format..."
    A = sparse(A_rows, A_cols, A_vals, n, n)
    B = sparse(B_rows, B_cols, B_vals, n, n)

    @debug "Post-BC sparsity" A_nnz=nnz(A) B_nnz=nnz(B)

    # Differential-equation rows: everything except the tau rows (zero in B)
    interior_dofs = setdiff(1:n, sort!(collect(bc_rows)))
    @info "Sparse assembly complete" interior_dofs=length(interior_dofs) total_dofs=n

    info = Dict(
        "method" => "Sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n
    )

    return A, B, interior_dofs, info
end

"""
    _filter_coo_rows!(rows, cols, vals, drop::Set{Int})

In-place compaction of a COO triplet: keep only entries whose row is not in
`drop`. Used to clear operator contributions to tau boundary rows before the BC
functionals are appended and the matrix is built.
"""
function _filter_coo_rows!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector,
                           drop::Set{Int})
    w = 0
    @inbounds for i in eachindex(rows)
        if !(rows[i] in drop)
            w += 1
            rows[w] = rows[i]; cols[w] = cols[i]; vals[w] = vals[i]
        end
    end
    resize!(rows, w); resize!(cols, w); resize!(vals, w)
    return nothing
end

"""
    _compute_sparse_bc(op) -> (bc_rows::Set{Int}, bcA::Vector{Tuple{Int,Int,Complex{T}}})

Tau boundary-condition specification for the sparse hydro+temperature operator,
as data rather than matrix mutation: `bc_rows` are the rows overwritten by BCs
(zeroed in B, replaced in A) and `bcA` are the (row, col, value) entries of the
replacement A rows. Same functionals and row placement as `_compute_mhd_bc`.

Each constraint replaces one of the highest-degree coefficients of its block's
projected residual, never the T₀/T₁ rows: rows N-2..N+1 of a poloidal block
(outer u, outer 2nd condition, inner u, inner 2nd condition) and rows N, N+1 of
a toroidal or temperature block (outer, inner).

Mechanical BCs (controlled by bci/bco):
- 0 = stress-free: u = 0, ∂²u/∂r² = 0 (poloidal); v - r·∂v/∂r = 0 (toroidal)
- 1 = no-slip: u = 0, ∂u/∂r = 0 (poloidal); v = 0 (toroidal)

Thermal BCs (controlled by bci_thermal/bco_thermal):
- 0 = fixed temperature: θ = 0
- 1 = fixed flux: ∂θ/∂r = 0
"""
function _compute_sparse_bc(op::SparseStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    n_per_mode = N + 1
    ri = params.ricb
    ro = one(T)
    nb_top = length(op.ll_top)
    nb_bot = length(op.ll_bot)

    bc_rows = Set{Int}()
    bcA = Tuple{Int,Int,Complex{T}}[]

    # Write one functional (Chebyshev coefficient row) into its own block
    function add!(row, block_start, values)
        push!(bc_rows, row)
        for (j, v) in enumerate(values)
            iszero(v) || push!(bcA, (row, block_start + j - 1, Complex{T}(v)))
        end
    end

    # Boundary evaluation of T_n, dT_n/dr and d²T_n/dr²
    scale = T(2) / (ro - ri)
    vo = _chebyshev_boundary_values(N, :outer, T)
    vi = _chebyshev_boundary_values(N, :inner, T)
    do_ = scale .* _chebyshev_boundary_derivative(N, :outer, T)
    di = scale .* _chebyshev_boundary_derivative(N, :inner, T)
    d2o = scale^2 .* _chebyshev_boundary_second_derivative(N, :outer, T)
    d2i = scale^2 .* _chebyshev_boundary_second_derivative(N, :inner, T)

    # Poloidal velocity BCs
    for k in eachindex(op.ll_top)
        first_row = (k - 1) * n_per_mode + 1
        last_row = first_row + N
        add!(last_row - 3, first_row, vo)
        add!(last_row - 2, first_row, params.bco == 1 ? do_ : d2o)
        add!(last_row - 1, first_row, vi)
        add!(last_row, first_row, params.bci == 1 ? di : d2i)
    end

    # Toroidal velocity BCs
    for k in eachindex(op.ll_bot)
        first_row = (nb_top + k - 1) * n_per_mode + 1
        last_row = first_row + N
        add!(last_row - 1, first_row, params.bco == 1 ? vo : vo .- ro .* do_)
        add!(last_row, first_row, params.bci == 1 ? vi : vi .- ri .* di)
    end

    # Temperature BCs
    for k in eachindex(op.ll_top)
        first_row = (nb_top + nb_bot + k - 1) * n_per_mode + 1
        last_row = first_row + N
        add!(last_row - 1, first_row, params.bco_thermal == 0 ? vo : do_)
        add!(last_row, first_row, params.bci_thermal == 0 ? vi : di)
    end

    return bc_rows, bcA
end
