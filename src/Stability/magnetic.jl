# Magnetic options shared by the collocation stability operators and the MHD solver:
# the imposed background field and the magnetic wall conditions.

"""
Background magnetic field types supported by the code.

Options:
- `no_field` - No background field (hydrodynamic stability)
- `axial` - Uniform axial field B₀ = B₀ẑ
- `dipole` - Dipolar field B₀ ~ (2cosθ r̂ + sinθ θ̂)/r³
"""
@enum BackgroundField begin
    no_field = 0
    axial = 1
    dipole = 2
end

"""`BackgroundField` value for no imposed field; requires `Le = 0` and `B0_amplitude = 0`."""
no_field

"""`BackgroundField` value for a uniform axial field B₀ẑ; requires `Le > 0`."""
axial

"""`BackgroundField` value for a dipolar field ~ (2cosθ r̂ + sinθ θ̂)/r³; requires `Le > 0` and `ricb > 0`."""
dipole

"""Magnetic wall conditions accepted by the collocation operators: `:insulating` or
`:perfect_conductor` for both walls, or an `(inner, outer)` pair."""
const MagneticBC = Union{Symbol, Tuple{Symbol, Symbol}}

"""Return the `(inner, outer)` magnetic wall conditions of `magnetic_bc`."""
_magnetic_walls(bc::Symbol) = (bc, bc)
_magnetic_walls(bc::Tuple{Symbol, Symbol}) = bc

"""Throw unless every wall of `magnetic_bc` is `:insulating` or `:perfect_conductor`."""
function _check_magnetic_bc(bc)
    bc isa MagneticBC && all(in((:insulating, :perfect_conductor)), _magnetic_walls(bc)) ||
        throw(ArgumentError("magnetic_bc must be :insulating, :perfect_conductor, or an " *
            "(inner, outer) pair of them, got $(repr(bc))"))
    return nothing
end

"""
    _background_potential(B0_type, r)

Radial profile `h(r)` of the imposed field `B₀ = ∇×∇×(h(r) cosθ 𝐫)`, an ℓ = 1, m = 0
poloidal potential field: `r/2` for the uniform axial field ẑ and `1/(2r²)` for the
dipole `(2cosθ r̂ + sinθ θ̂)/(2r³)`, as in the MHD solver. Both are current free.
"""
function _background_potential(B0_type::BackgroundField, r)
    B0_type == axial && return r / 2
    B0_type == dipole && return inv(2r^2)
    return zero(r)
end

"""Magnetic keyword arguments of a parameter set, for rebuilding `OnsetParams`, in the
order of a basic state's `magnetic` record."""
_magnetic_kwargs(p) = (B0_type=p.B0_type, Le=p.Le, Pm=p.Pm, magnetic_bc=p.magnetic_bc)

"""
    _magnetic_boundary_layer_Nr(params) -> Int

Rough number of collocation points that resolves the magnetic (Hartmann) boundary
layers, of thickness `√(E·Em)/(Le·B₀)` with `Em = E/Pm` and `B₀` the largest imposed
field at a wall (1 for the axial field, `χ⁻³` at the inner wall for the dipole).
This is the MHD solver's estimate (`_mhd_boundary_layer_N`) plus one node. Below it,
truncation-scale Alfvén waves are under-damped and the collocation pencil, like the
tau pencil, can show spurious growing eigenvalues. Returns 0 without a field.
"""
function _magnetic_boundary_layer_Nr(p)
    _has_magnetic(p) || return 0
    B0 = p.B0_type == dipole ? inv(p.χ)^3 : one(p.χ)
    return 1 + ceil(Int, 1.8 * sqrt((1 - p.χ) * p.Le * B0 / sqrt(p.E * p.E / p.Pm)))
end

"""
    _collocation_spectral_tails(op, vec; fraction=1/4)

Fraction of the norm of a full eigenvector in the top `fraction` of the Chebyshev
coefficients of every radial profile (`radial`) and in the top `fraction` of each
field's retained degrees (`angular`), as in `_mhd_spectral_tails`; magnetic blocks
are weighted by `Le²`, their energy weight. Converged modes measure ≲ 1e-3, and
spurious truncation-scale modes several tenths.
"""
function _collocation_spectral_tails(op, vec::AbstractVector; fraction::Real=1/4)
    r = op.r; n = length(r)
    x = (2 .* r .- (first(r) + last(r))) ./ (last(r) - first(r))
    V = lu([cos(k * acos(clamp(z, -one(z), one(z)))) for z in x, k in 0:n-1])
    weight = _has_magnetic(op.params) ? Float64(op.params.Le)^2 : 1.0
    total = 0.0; radial = 0.0; angular = 0.0
    k_cut = n - max(1, floor(Int, fraction * n)) + 1
    for ((l, field), rng) in op.index_map
        ls = op.l_sets[field]
        l_cut = ls[end - max(1, floor(Int, fraction * length(ls))) + 1]
        w = field in (:F, :G) ? weight : 1.0
        c = V \ Vector{ComplexF64}(vec[rng])
        for k in 1:n
            a = w * abs2(c[k])
            total += a
            k >= k_cut && (radial += a)
            length(ls) > 1 && l >= l_cut && (angular += a)
        end
    end
    total == 0 && return (radial = 0.0, angular = 0.0)
    return (radial = sqrt(radial / total), angular = sqrt(angular / total))
end

"""Warn when the leading eigenmode of a magnetic collocation solve is under-resolved,
as the MHD solve does; returns the tails of every eigenvector."""
function _check_magnetic_resolution(op, eigenvalues, evecs::AbstractMatrix)
    _has_magnetic(op.params) && size(evecs, 1) == op.total_dof && size(evecs, 2) > 0 ||
        return NamedTuple{(:radial, :angular), Tuple{Float64, Float64}}[]
    tails = [_collocation_spectral_tails(op, view(evecs, :, j)) for j in axes(evecs, 2)]
    leading = argmax(real.(eigenvalues))
    if tails[leading].radial > 1e-2
        Nr = op.params.Nr; N_layers = _magnetic_boundary_layer_Nr(op.params)
        hint = N_layers > Nr ? " The magnetic boundary layers need roughly Nr ≳ $N_layers " *
                               "(a rough estimate)." : ""
        @warn "Magnetic leading eigenmode is under-resolved: " *
              "$(round(100 * tails[leading].radial; sigdigits=2))% of its norm lies in the top " *
              "quarter of the Chebyshev coefficients, so its eigenvalue " *
              "$(eigenvalues[leading]) is likely a truncation artefact. Increase Nr " *
              "(currently $Nr) until the leading eigenvalue converges." * hint
    end
    if tails[leading].angular > 1e-2
        @warn "Magnetic leading eigenmode has a large angular spectral tail: " *
              "$(round(100 * tails[leading].angular; sigdigits=2))% of its norm lies in the top " *
              "quarter of retained degrees. Increase lmax (currently $(op.params.lmax)) and " *
              "check eigenvalue convergence."
    end
    return tails
end

"""True when a basic state has no mean flow at all."""
function _mean_flow_is_zero(bs)
    f = bs.flow
    f === nothing && return all(all(iszero, v) for d in (bs.ur_coeffs, bs.utheta_coeffs,
                                                            bs.uphi_coeffs) for v in values(d))
    return all(all(iszero, v) for d in (f.p, f.t) for v in values(d))
end

"""
Throw unless a basic state is consistent with the magnetic options of a stability
problem. A state computed with an imposed field records its configuration in
`bs.magnetic`; the perturbations need the same field, Lehnert and magnetic Prandtl
numbers and walls, since its induced field and Lorentz force depend on them. A
hydrodynamic state with a mean flow cannot carry an imposed field: the flow would
shear it without the induced field a steady state requires.
"""
function _check_basic_state_magnetic(Pm, Le, B0_type, magnetic_bc, bs)
    bs === nothing && return nothing
    record = hasproperty(bs, :magnetic) ? bs.magnetic : nothing
    magnetic = B0_type != no_field
    if record === nothing
        magnetic && !_mean_flow_is_zero(bs) && throw(ArgumentError(
            "This basic state was computed without a magnetic field, but the stability " *
            "problem imposes B0_type=$B0_type. Build the state with the same magnetic " *
            "options, e.g. basic_state(params; mode=:selfconsistent), so that its induced " *
            "field and Lorentz force are consistent with the imposed field."))
        return nothing
    end
    magnetic || throw(ArgumentError(
        "This basic state was computed with an imposed field ($(record.B0_type)); set the " *
        "same B0_type, Le, Pm and magnetic_bc in the stability parameters."))
    same = record.B0_type == B0_type && isapprox(record.Le, Le) &&
           isapprox(record.Pm, Pm) && _magnetic_walls(record.magnetic_bc) == _magnetic_walls(magnetic_bc)
    same || throw(ArgumentError(
        "The basic state was computed with B0_type=$(record.B0_type), Le=$(record.Le), " *
        "Pm=$(record.Pm), magnetic_bc=$(repr(record.magnetic_bc)), but the stability " *
        "problem uses B0_type=$B0_type, Le=$Le, Pm=$Pm, magnetic_bc=$(repr(magnetic_bc))."))
    return nothing
end
