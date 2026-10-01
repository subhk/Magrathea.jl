# =============================================================================
#  Physical-space reconstruction of MHD perturbation fields from an eigenvector.
#
#  Native to the MHD spectral (Chebyshev-coefficient) basis: slice the
#  full eigenvector into per-(field, ℓ) coefficient
#  blocks, evaluate each Chebyshev series on a radial grid via direct
#  T_n(x) = cos(n·acos(x)) (accurate for the small N used here), then
#  synthesize physical fields on a meridional (r, θ) grid.  The poloidal-toroidal
#  curl uses the onset solver convention (velocity and magnetic field
#  share the same B = ∇×∇×(r P r̂) + ∇×(r T r̂) form).
# =============================================================================

"""Full MHD coefficient count, including conducting core/mantle regions when present."""
function _mhd_reconstruction_dof(op::MHDStabilityOperator)
    return op.matrix_size
end

"""
    _mhd_full_vector(evec, op, interior_dofs)

Return a full-length DOF vector. If `evec` already has `_mhd_reconstruction_dof(op)`
entries it is copied. Reduced vectors need their basis transformation, not
zero insertion at boundary-equation indices. `interior_dofs` is accepted for API compatibility.
"""
function _mhd_full_vector(evec::AbstractVector{<:Complex},
                          op::MHDStabilityOperator,
                          interior_dofs)
    ndof = _mhd_reconstruction_dof(op)
    if length(evec) == ndof
        return Vector{ComplexF64}(evec)
    else
        throw(DimensionMismatch("MHD reconstruction requires all $ndof Chebyshev coefficients. " *
            "Tau boundary rows are equations, not removable coefficients. Use full eigenvectors " *
            "from solve(MHDProblem(...)), or reconstruct_mhd_galerkin_full with its layout."))
    end
end

"""Slice the `(N+1)` spectral coefficients for `(ℓ, field)` from a full vector."""
function _mhd_field_block(full::AbstractVector{<:Complex},
                          idx_map::Dict{Tuple{Int,Symbol},UnitRange{Int}},
                          field::Symbol, ℓ::Int)
    return @view full[idx_map[(ℓ, field)]]
end

"""
    _mhd_radial_eval(coeffs, ricb, r_grid; outer_radius=1)

Evaluate a Chebyshev-T coefficient series (`coeffs[n+1]` multiplies `T_n`) at
physical radii `r_grid ∈ [ricb, outer_radius]`, via the affine map to `[-1,1]` and
`T_n(x) = cos(n·acos(x))`.
"""
function _mhd_radial_eval(coeffs::AbstractVector{<:Complex},
                          ricb::Real, r_grid::AbstractVector; outer_radius::Real=1)
    N = length(coeffs) - 1
    out = zeros(ComplexF64, length(r_grid))
    @inbounds for (i, r) in enumerate(r_grid)
        x = 2 * (r - ricb) / (outer_radius - ricb) - 1
        x = clamp(x, -1.0, 1.0)
        ax = acos(x)
        acc = zero(ComplexF64)
        for n in 0:N
            acc += coeffs[n + 1] * cos(n * ax)
        end
        out[i] = acc
    end
    return out
end

"""
Radial reconstruction grid: the Chebyshev collocation nodes of a
`ChebyshevDiffn(Nr, [ricb, 1], 1)`. Potentials and their spectral derivatives
are evaluated directly at these nodes, including when `Nr < N + 1`.
"""
function _mhd_radial_grid(op::MHDStabilityOperator; Nr::Int = op.params.N + 1,
                          domain=(op.params.ricb, 1.0))
    return ChebyshevDiffn(Nr, collect(domain), 1).x
end

"""
    perturbation_temperature(evec, op::MHDStabilityOperator;
                             Nθ=nothing, Nr=nothing, interior_dofs=nothing, grid=nothing)

Reconstruct the physical perturbation temperature field
`θ(r,θ) = Σ_ℓ h_ℓ(r) Y_ℓ^m(θ)/√(2ℓ+1)` from a full MHD eigenvector.
Returns `(θfield, r_grid, grid)`.
"""
function perturbation_temperature(evec::AbstractVector{<:Complex},
                                  op::MHDStabilityOperator;
                                  Nθ::Union{Int,Nothing}=nothing,
                                  Nr::Union{Int,Nothing}=nothing,
                                  interior_dofs=nothing,
                                  grid::Union{MeridionalGrid,Nothing}=nothing)
    m    = op.params.m
    lmax = op.params.lmax
    ricb = op.params.ricb
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * lmax : Nθ, m, lmax;
                              T=typeof(op.params.E)) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)

    θfield = zeros(ComplexF64, length(r_grid), length(g.θ))
    for l in op.ll_h
        hl = _mhd_radial_eval(_mhd_field_block(full, idx_map, :h, l), ricb, r_grid)
        ylm = g.Ylm[l] ./ sqrt(2l + 1)
        @inbounds for j in eachindex(g.θ), k in eachindex(r_grid)
            θfield[k, j] += hl[k] * ylm[j]
        end
    end
    return θfield, r_grid, g
end

# Native potentials multiply the radius vector and use Y_lm/sqrt(2l+1).
function _mhd_poltor_to_physical(full, idx_map, op,
                                 ls_pol, sec_pol::Symbol,
                                 ls_tor, sec_tor::Symbol,
                                 r_grid, g::MeridionalGrid;
                                 radial_domain=(op.params.ricb, 1.0))
    ri, ro = radial_domain
    radial(sec, l) = _mhd_radial_eval(_mhd_field_block(full, idx_map, sec, l),
                                     ri, r_grid; outer_radius=ro)
    P = Dict(l => radial(sec_pol, l) for l in ls_pol)
    Tor = Dict(l => radial(sec_tor, l) for l in ls_tor)
    # Differentiate the complete polynomial before evaluating it on the output
    # grid. Differentiating the sampled values aliases modes when Nr < N + 1.
    scale = 2 / (ro - ri)
    function radial_derivative(l)
        coeffs = _mhd_field_block(full, idx_map, sec_pol, l)
        deriv = vec(_chebyshev_derivative(reshape(coeffs, :, 1), scale))
        return _mhd_radial_eval(deriv, ri, r_grid; outer_radius=ro)
    end
    dP = Dict(l => radial_derivative(l) for l in ls_pol)
    return _onset_velocity_from_coefficients(P, Tor, r_grid, nothing, g, op.params.m;
                                             poloidal_derivatives=dP)
end

"""Regular-core potential, radial derivative, and potential/r, including r=0."""
function _mhd_core_radial_values(coeffs, l, ri, r_grid)
    rho = r_grid ./ ri
    x = 2 .* rho.^2 .- 1
    a = _mhd_radial_eval(coeffs, -1, x)
    dc = vec(_chebyshev_derivative(reshape(coeffs, :, 1), 1))
    da = _mhd_radial_eval(dc, -1, x)
    over_r = rho.^(l-1) .* a ./ ri
    derivative = rho.^(l-1) .* (l .* a .+ 4 .* rho.^2 .* da) ./ ri
    return r_grid .* over_r, derivative, over_r
end

"""Core synthesis evaluates the analytic l=1 center limit without dividing by r."""
function _mhd_core_to_physical(full, idx_map, op, r_grid, grid::MeridionalGrid{T}) where T
    p = op.params; a = abs(p.m)
    g = SHGrid{T}(p.lmax, a, grid.cosθ, zeros(T,length(grid.θ)), T[0])
    ur = zeros(ComplexF64,length(r_grid),length(grid.θ)); uθ=zero(ur); uφ=zero(ur)
    for l in op.ll_f
        _, dp, over_r = _mhd_core_radial_values(
            _mhd_field_block(full,idx_map,:fi,l), l, p.ricb, r_grid)
        y,h,v = _coupling_harmonic(g,l,p.m); normalization=inv(sqrt(T(2l+1)))
        ur .+= (normalization*l*(l+1) .* over_r) * transpose(y)
        uθ .+= (normalization .* (dp .+ over_r)) * transpose(h)
        uφ .+= (normalization .* (dp .+ over_r)) * transpose(v)
    end
    for l in op.ll_g
        tor, _, _ = _mhd_core_radial_values(
            _mhd_field_block(full,idx_map,:gi,l), l, p.ricb, r_grid)
        _,h,v = _coupling_harmonic(g,l,p.m); normalization=inv(sqrt(T(2l+1)))
        uθ .+= (normalization .* tor) * transpose(v)
        uφ .-= (normalization .* tor) * transpose(h)
    end
    return ur,uθ,uφ
end

"""
    perturbation_velocity(evec, op::MHDStabilityOperator; kwargs...)

Reconstruct physical perturbation velocity `(u_r, u_θ, u_φ)` from an MHD
eigenvector (poloidal `:u`, toroidal `:v`). Returns `(ur, uθ, uφ, r_grid, grid)`.
"""
function perturbation_velocity(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing)
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax;
                              T=typeof(op.params.E)) : grid
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr)
    Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
                                          op.ll_u, :u, op.ll_v, :v, r_grid, g)
    return Fr, Fθ, Fφ, r_grid, g
end

"""
    perturbation_magnetic(evec, op::MHDStabilityOperator; region=:fluid, kwargs...)

Reconstruct physical perturbation magnetic field `(B_r, B_θ, B_φ)` from an MHD
full eigenvector. `region=:fluid` returns the fluid shell, `:core` a configured
finite conducting core (including its regular center), and `:mantle` a configured
finite conducting mantle. Potentials and their derivatives are evaluated at the
requested output resolution. Returns `(Br, Bθ, Bφ, r_grid, grid)`.
"""
function perturbation_magnetic(evec::AbstractVector{<:Complex},
                               op::MHDStabilityOperator;
                               Nθ::Union{Int,Nothing}=nothing,
                               Nr::Union{Int,Nothing}=nothing,
                               interior_dofs=nothing,
                               grid::Union{MeridionalGrid,Nothing}=nothing,
                               region::Symbol=:fluid)
    region in (:fluid,:core,:mantle) || throw(ArgumentError(
        "Magnetic region must be :fluid, :core, or :mantle"))
    region === :core && op.params.bci_magnetic != 1 && throw(ArgumentError(
        "region=:core requires bci_magnetic=1"))
    region === :mantle && op.params.bco_magnetic != 1 && throw(ArgumentError(
        "region=:mantle requires bco_magnetic=1"))
    isempty(op.ll_f) && isempty(op.ll_g) && error(
        "perturbation_magnetic: this MHD problem has no magnetic field " *
        "(B0_type=no_field). Nothing to reconstruct.")
    full     = _mhd_full_vector(evec, op, interior_dofs)
    idx_map  = _mhd_index_map(op)
    g = grid === nothing ?
        build_meridional_grid(Nθ === nothing ? 2 * op.params.lmax : Nθ,
                              op.params.m, op.params.lmax;
                              T=typeof(op.params.E)) : grid
    domain = region === :core ? (0.0,op.params.ricb) : region === :mantle ?
        (1.0,something(op.params.mantle_radius)) : (op.params.ricb,1.0)
    r_grid = _mhd_radial_grid(op; Nr = Nr === nothing ? op.params.N + 1 : Nr, domain=domain)
    if region === :core
        Fr, Fθ, Fφ = _mhd_core_to_physical(full,idx_map,op,r_grid,g)
    else
        pol, tor = region === :mantle ? (:fm,:gm) : (:f,:g)
        Fr, Fθ, Fφ = _mhd_poltor_to_physical(full, idx_map, op,
            op.ll_f, pol, op.ll_g, tor, r_grid, g; radial_domain=domain)
    end
    return Fr, Fθ, Fφ, r_grid, g
end

"""
    _mhd_spectral_tails(op, vec; fraction=1/4) -> (radial, angular)

Resolution diagnostic for a full MHD coefficient vector (tau layout). `radial` is
the share of the vector's norm carried by the highest `fraction` of the
Chebyshev coefficients in every `(ℓ, field)` block; `angular` is the share
carried by the highest `fraction` of the retained degrees ℓ of every field. A
resolved eigenmode decays spectrally in both, so both are small; a large tail
means the mode lives at the truncation scale and its eigenvalue is not
trustworthy (e.g. spurious growth at strong field and low `N`).
"""
function _mhd_spectral_tails(op::MHDStabilityOperator, vec::AbstractVector; fraction::Real=1/4)
    idx_map = _mhd_index_map(op)
    total = 0.0; radial = 0.0; angular = 0.0
    top_l = Dict{Symbol,Int}()
    for ((l, sec), _) in idx_map
        top_l[sec] = max(get(top_l, sec, l), l)
    end
    sec_ls = Dict(sec => sort!([l for ((l, s), _) in idx_map if s === sec]) for sec in keys(top_l))
    for ((l, sec), rng) in idx_map
        last(rng) <= length(vec) || continue
        n = length(rng)
        k_cut = n - max(1, floor(Int, fraction * n)) + 1
        ls = sec_ls[sec]
        l_cut = ls[end - max(1, floor(Int, fraction * length(ls))) + 1]
        for (k, i) in enumerate(rng)
            a = Float64(abs2(vec[i]))
            total += a
            k >= k_cut && (radial += a)
            length(ls) > 1 && l >= l_cut && (angular += a)
        end
    end
    total == 0 && return (radial = 0.0, angular = 0.0)
    return (radial = sqrt(radial / total), angular = sqrt(angular / total))
end
