module MagratheaMakieExt

using Magrathea
using Makie

# Interactive eigenvalue spectrum
function Magrathea.eigenspectrum(r::Magrathea.StabilityResult; figure_kwargs...)
    fig = Figure(; figure_kwargs...)
    ax = Axis(fig[1,1],
        xlabel="σᵣ (growth rate)",
        ylabel="σᵢ (frequency)",
        title="Eigenvalue Spectrum")
    scatter!(ax, real.(r.eigenvalues), imag.(r.eigenvalues),
        markersize=12, color=:blue)
    DataInspector(fig)
    fig
end

# ---------------------------------------------------------------------------
#  Eigenvector layouts
#
#  Each problem type stores its eigenvectors differently:
#    - OnsetProblem / BiglobalProblem: radial collocation values in
#      (l, field) blocks of a LinearStabilityOperator,
#    - MHDProblem: Chebyshev coefficients in (l, field) blocks of an
#      MHDStabilityOperator,
#    - TriglobalProblem: constraint-reduced blocks, one per coupled m.
#  Field extraction is delegated to Magrathea's reconstruction helpers for the
#  matching layout instead of assuming one block structure for every result.
# ---------------------------------------------------------------------------

const _RADIAL_FIELDS = (:poloidal, :toroidal, :temperature,
                        :magnetic_poloidal, :magnetic_toroidal)
const _VELOCITY_FIELDS = (:ur, :utheta, :uphi)
const _MAGNETIC_FIELDS = (:Br, :Btheta, :Bphi)
const _FIELD_ALIASES = Dict(:uθ => :utheta, :uφ => :uphi, :Bθ => :Btheta, :Bφ => :Bphi)

# Operator stored by `solve`, or `nothing` for a hand-built result.
_ext_stored_operator(result) =
    hasproperty(result.extra, :operator) ? result.extra.operator : nothing

# Layout of `result.eigenvectors`, rebuilt from the problem when not stored.
_ext_layout(result::Magrathea.StabilityResult) = _ext_layout(result, result.problem)

function _ext_layout(result, problem::Union{Magrathea.OnsetProblem, Magrathea.BiglobalProblem})
    op = _ext_stored_operator(result)
    return op isa Magrathea.LinearStabilityOperator ? op :
           Magrathea.LinearStabilityOperator(problem.params)
end

function _ext_layout(result, problem::Magrathea.MHDProblem)
    op = _ext_stored_operator(result)
    return op isa Magrathea.MHDStabilityOperator ? op :
           Magrathea.MHDStabilityOperator(problem.params)
end

# Triglobal results carry no operator: rebuild the coupled-mode block layout
# exactly as `solve(::TriglobalProblem)` does.
function _ext_layout(result, problem::Magrathea.TriglobalProblem)
    p = problem.params
    tparams = Magrathea.TriglobalParams(
        E=p.E, Pr=p.Pr, Ra=p.Ra, χ=p.χ,
        m_range=problem.m_range, lmax=p.lmax, Nr=p.Nr,
        basic_state_3d=problem.basic_state,
        mechanical_bc=p.mechanical_bc, thermal_bc=p.thermal_bc,
        equatorial_symmetry=p.equatorial_symmetry)
    return Magrathea.setup_coupled_mode_problem(tparams)
end

_ext_layout(result, problem) = throw(ArgumentError(
    "Field plots are not implemented for $(nameof(typeof(problem))) results"))

function _ext_check_length(evec, n::Int, kind::String)
    length(evec) == n || throw(DimensionMismatch(
        "$kind eigenvector has $(length(evec)) entries, but the problem layout has $n"))
    return nothing
end

# Validate mode_index and return the eigenvector column, or nothing on error.
function _ext_get_eigenvector(result, mode_index)
    if isempty(result.eigenvectors)
        @warn "plot: eigenvector matrix is empty"
        return nothing
    end
    nev = size(result.eigenvectors, 2)
    if mode_index < 1 || mode_index > nev
        @warn "plot: mode_index=$mode_index out of range (1:$nev)"
        return nothing
    end
    return result.eigenvectors[:, mode_index]
end

function _ext_empty_figure()
    fig = Figure()
    Axis(fig[1,1], title="No data")
    return fig
end

# ---------------------------------------------------------------------------
#  Radial profiles: returns (r_grid, [(label, profile), ...])
# ---------------------------------------------------------------------------

# Select one field from `extract_eigenvector_coefficients` output.
function _ext_hydro_coefficients(coeffs, field::Symbol)
    P, Tor, Θ = coeffs
    field === :poloidal && return P
    field === :toroidal && return Tor
    field === :temperature && return Θ
    throw(ArgumentError(
        "field=:$field requires an MHDProblem result; hydrodynamic results have no magnetic field"))
end

function _ext_radial_profiles(op::Magrathea.LinearStabilityOperator, evec, field::Symbol)
    _ext_check_length(evec, op.total_dof, "Onset/biglobal")
    coeffs = _ext_hydro_coefficients(Magrathea.extract_eigenvector_coefficients(evec, op), field)
    return op.r, [("l=$l", coeffs[l]) for l in sort!(collect(keys(coeffs)))]
end

function _ext_radial_profiles(op::Magrathea.MHDStabilityOperator, evec, field::Symbol)
    section, ls = field === :poloidal          ? (:u, op.ll_u) :
                  field === :toroidal          ? (:v, op.ll_v) :
                  field === :temperature       ? (:h, op.ll_h) :
                  field === :magnetic_poloidal ? (:f, op.ll_f) : (:g, op.ll_g)
    isempty(ls) && section in (:f, :g) && throw(ArgumentError(
        "field=:$field: this MHD problem has no magnetic field (B0_type=no_field)"))
    full = Magrathea._mhd_full_vector(evec, op, nothing)   # checks the coefficient count
    idx_map = Magrathea._mhd_index_map(op)
    r_grid = Magrathea._mhd_radial_grid(op)
    profiles = [("l=$l", Magrathea._mhd_radial_eval(
                     Magrathea._mhd_field_block(full, idx_map, section, l),
                     op.params.ricb, r_grid)) for l in ls]
    return r_grid, profiles
end

# Each coupled m block is mapped back to its full collocation vector with the
# same constraint reduction the triglobal solver used.
function _ext_radial_profiles(problem::Magrathea.CoupledModeProblem, evec, field::Symbol)
    _ext_check_length(evec, problem.total_dofs, "Triglobal")
    r_grid = Magrathea._mode_reconstruction(problem, abs(first(problem.m_range))).op.r
    profiles = Tuple{String, Vector{eltype(evec)}}[]
    for m in problem.m_range
        rec = Magrathea._mode_reconstruction(problem, abs(m))
        full = Magrathea._reconstruct_full_vector(rec.reduction, evec[problem.block_indices[m]])
        coeffs = _ext_hydro_coefficients(
            Magrathea.extract_eigenvector_coefficients(full, rec.op), field)
        for l in sort!(collect(keys(coeffs)))
            push!(profiles, ("m=$m, l=$l", coeffs[l]))
        end
    end
    return r_grid, profiles
end

# ---------------------------------------------------------------------------
#  Meridional fields: returns (r_grid, θ, values) with values[i_r, i_θ]
# ---------------------------------------------------------------------------

_ext_component(components, field, names) = components[findfirst(==(field), names)]

function _ext_meridional_field(op::Union{Magrathea.LinearStabilityOperator,
                                         Magrathea.MHDStabilityOperator},
                               evec, field::Symbol, npoints::Int)
    if op isa Magrathea.LinearStabilityOperator
        _ext_check_length(evec, op.total_dof, "Onset/biglobal")
    end
    if field === :temperature
        values, r_grid, grid = Magrathea.perturbation_temperature(evec, op; Nθ=npoints)
        return r_grid, grid.θ, values
    elseif field in _VELOCITY_FIELDS
        ur, uθ, uφ, r_grid, grid = Magrathea.perturbation_velocity(evec, op; Nθ=npoints)
        return r_grid, grid.θ, _ext_component((ur, uθ, uφ), field, _VELOCITY_FIELDS)
    end
    op isa Magrathea.MHDStabilityOperator || throw(ArgumentError(
        "field=:$field requires an MHDProblem result; hydrodynamic results have no magnetic field"))
    isempty(op.ll_f) && throw(ArgumentError(
        "field=:$field: this MHD problem has no magnetic field (B0_type=no_field)"))
    Br, Bθ, Bφ, r_grid, grid = Magrathea.perturbation_magnetic(evec, op; Nθ=npoints)
    return r_grid, grid.θ, _ext_component((Br, Bθ, Bφ), field, _MAGNETIC_FIELDS)
end

# Coupled-mode eigenvectors: velocity summed over m at the φ = 0 meridian.
function _ext_meridional_field(problem::Magrathea.CoupledModeProblem,
                               evec, field::Symbol, npoints::Int)
    field in _VELOCITY_FIELDS || throw(ArgumentError(
        "field=:$field is not available for TriglobalProblem results; Magrathea " *
        "reconstructs coupled-mode eigenvectors as velocity only (use :ur, :utheta or :uphi)"))
    _ext_check_length(evec, problem.total_dofs, "Triglobal")
    p = problem.params
    ur, uθ, uφ = Magrathea.eigenvector_to_velocity_triglobal(
        evec, problem; Nθ=npoints, φ_slice=0.0)
    # The radial and colatitude nodes used by eigenvector_to_velocity_triglobal.
    r_grid = Magrathea._build_chebyshev_grid(p.Nr, p.χ, 1.0).x
    θ = Magrathea.build_meridional_grid(npoints, 0, p.lmax; T=typeof(p.E)).θ
    return r_grid, θ, _ext_component((ur, uθ, uφ), field, _VELOCITY_FIELDS)
end

# ---------------------------------------------------------------------------
#  plot_radial — amplitude of each l-mode vs radius
# ---------------------------------------------------------------------------

"""
    plot_radial(result, mode_index; field=:poloidal)

Plot `|F_l(r)|` for every retained degree `l` of eigenvector `mode_index`.
`field` is `:poloidal`, `:toroidal` or `:temperature`; MHD results also accept
`:magnetic_poloidal` and `:magnetic_toroidal`. Triglobal results draw one curve
per coupled `(m, l)` pair.
"""
function Magrathea.plot_radial(r::Magrathea.StabilityResult, mode_index::Int;
                            field::Symbol=:poloidal)
    field in _RADIAL_FIELDS || throw(ArgumentError(
        "plot_radial: unknown field :$field; use one of $(join(repr.(_RADIAL_FIELDS), ", "))"))
    evec = _ext_get_eigenvector(r, mode_index)
    evec === nothing && return _ext_empty_figure()

    r_grid, profiles = _ext_radial_profiles(_ext_layout(r), evec, field)
    label = field === :poloidal          ? "Poloidal |P_l(r)|" :
            field === :toroidal          ? "Toroidal |T_l(r)|" :
            field === :temperature       ? "Temperature |Theta_l(r)|" :
            field === :magnetic_poloidal ? "Magnetic poloidal |f_l(r)|" :
                                           "Magnetic toroidal |g_l(r)|"

    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="|amplitude|",
        title="$label, mode $mode_index")

    for (name, profile) in profiles
        lines!(ax, r_grid, abs.(profile), label=name)
    end

    if 1 <= length(profiles) <= 15
        axislegend(ax, position=:rt)
    end

    fig
end

# ---------------------------------------------------------------------------
#  plot_meridional — physical field on the (r, theta) plane
# ---------------------------------------------------------------------------

"""
    plot_meridional(result, mode_index; field=:temperature, npoints=100)

Heatmap of the real part of a reconstructed perturbation field on the `φ = 0`
meridian, with `npoints` colatitude nodes. `field` is `:temperature`, `:ur`,
`:utheta` or `:uphi`; MHD results also accept `:Br`, `:Btheta` and `:Bphi`.
Triglobal results support the velocity components only.
"""
function Magrathea.plot_meridional(r::Magrathea.StabilityResult, mode_index::Int;
                                field::Symbol=:temperature, npoints::Int=100)
    field = get(_FIELD_ALIASES, field, field)
    if !(field === :temperature || field in _VELOCITY_FIELDS || field in _MAGNETIC_FIELDS)
        hint = field in (:poloidal, :toroidal) ?
            " (potentials are shown per degree by plot_radial)" : ""
        throw(ArgumentError("plot_meridional: unknown field :$field$hint; use :temperature, " *
            "$(join(repr.(_VELOCITY_FIELDS), ", ")), or for MHD results " *
            "$(join(repr.(_MAGNETIC_FIELDS), ", "))"))
    end
    evec = _ext_get_eigenvector(r, mode_index)
    evec === nothing && return _ext_empty_figure()

    r_grid, θ, values = _ext_meridional_field(_ext_layout(r), evec, field, npoints)
    # heatmap needs ascending coordinates; Gauss-Legendre colatitudes descend.
    ir = sortperm(r_grid)
    iθ = sortperm(θ)

    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="θ (colatitude)",
        title="Meridional: $field, mode $mode_index")
    hm = heatmap!(ax, r_grid[ir], θ[iθ], real.(values[ir, iθ]))
    Colorbar(fig[1,2], hm)
    fig
end

end # module
