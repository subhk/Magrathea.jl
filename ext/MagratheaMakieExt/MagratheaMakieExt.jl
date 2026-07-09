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
#  Helpers for eigenvector decomposition
# ---------------------------------------------------------------------------

# Build the ascending Chebyshev-Gauss-Lobatto grid on [ri, ro].
function _ext_radial_grid(Nr, ri, ro)
    x_hat = cospi.(reverse(0:Nr-1) ./ (Nr - 1))
    return @. (ro - ri) / 2 * (x_hat + 1) + ri
end

# Extract l-mode sets and radial grid, preferring the operator stored in
# extra when available, otherwise falling back to params-based computation.
function _ext_get_layout(result)
    extra = result.extra
    params = result.problem.params
    Nr = params.Nr
    ri = params.ri
    ro = params.ro

    # Try to use operator from extra
    if hasproperty(extra, :operator) && extra.operator !== nothing
        op = extra.operator
        l_sets = op.l_sets
        r_grid = op.r
    else
        # Use the exported compute_l_sets from Magrathea — no duplication
        l_sets = Magrathea.compute_l_sets(params)
        r_grid = _ext_radial_grid(Nr, ri, ro)
    end

    return (; l_sets, r_grid, Nr)
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

# Return the field-specific offset, number of modes, l-mode list, and label.
function _ext_field_info(l_sets, Nr, field::Symbol)
    n_pol = length(l_sets[:P])
    n_tor = length(l_sets[:T])
    n_theta = length(l_sets[:Θ])

    if field === :poloidal
        return (offset=0, n_modes=n_pol, l_modes=l_sets[:P],
                label="Poloidal |P_l(r)|")
    elseif field === :toroidal
        return (offset=n_pol * Nr, n_modes=n_tor, l_modes=l_sets[:T],
                label="Toroidal |T_l(r)|")
    elseif field === :temperature
        return (offset=(n_pol + n_tor) * Nr, n_modes=n_theta,
                l_modes=l_sets[:Θ], label="Temperature |Theta_l(r)|")
    else
        error("Unknown field :$field. Use :poloidal, :toroidal, or :temperature")
    end
end

# Associated Legendre polynomial P_l^m(x) via upward recurrence.
function _ext_assoc_legendre(l::Int, m::Int, x::Real)
    (m < 0 || m > l) && return 0.0

    # Seed: P_m^m
    pmm = 1.0
    if m > 0
        somx2 = sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in 1:m
            pmm *= -fact * somx2
            fact += 2.0
        end
    end
    l == m && return pmm

    # P_{m+1}^m
    pmmp1 = x * (2m + 1) * pmm
    l == m + 1 && return pmmp1

    # Recurrence for l > m+1
    pll = 0.0
    for ll in (m + 2):l
        pll = (x * (2ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    end
    return pll
end

# ---------------------------------------------------------------------------
#  plot_radial — amplitude of each l-mode vs radius
# ---------------------------------------------------------------------------

function Magrathea.plot_radial(r::Magrathea.StabilityResult, mode_index::Int;
                            field::Symbol=:poloidal)
    evec = _ext_get_eigenvector(r, mode_index)
    if evec === nothing
        fig = Figure()
        Axis(fig[1,1], title="No data")
        return fig
    end

    layout = _ext_get_layout(r)
    fi = _ext_field_info(layout.l_sets, layout.Nr, field)

    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="|amplitude|",
        title="$(fi.label), mode $mode_index")

    for (i, l) in enumerate(fi.l_modes)
        idx_start = fi.offset + (i - 1) * layout.Nr + 1
        idx_end   = fi.offset + i * layout.Nr
        if idx_end > length(evec)
            continue
        end
        profile = abs.(evec[idx_start:idx_end])
        lines!(ax, layout.r_grid, profile, label="l=$l")
    end

    if length(fi.l_modes) <= 15
        axislegend(ax, position=:rt)
    end

    fig
end

# ---------------------------------------------------------------------------
#  plot_meridional — contour plot on the (r, theta) plane
# ---------------------------------------------------------------------------

function Magrathea.plot_meridional(r::Magrathea.StabilityResult, mode_index::Int;
                                field::Symbol=:temperature, npoints::Int=100)
    evec = _ext_get_eigenvector(r, mode_index)
    if evec === nothing
        fig = Figure()
        Axis(fig[1,1], title="No data")
        return fig
    end

    layout = _ext_get_layout(r)
    fi = _ext_field_info(layout.l_sets, layout.Nr, field)
    m  = r.problem.params.m

    # Theta grid (colatitude 0..pi)
    theta_grid = range(0, pi, length=npoints)

    # Reconstruct field on (r, theta) grid:
    #   F(r, theta) = sum_l Re[F_l(r)] * P_l^m(cos theta)
    field_values = zeros(length(layout.r_grid), npoints)

    for (i, l) in enumerate(fi.l_modes)
        idx_start = fi.offset + (i - 1) * layout.Nr + 1
        idx_end   = fi.offset + i * layout.Nr
        if idx_end > length(evec)
            continue
        end
        coeffs = evec[idx_start:idx_end]

        for (jt, theta) in enumerate(theta_grid)
            plm = _ext_assoc_legendre(l, m, cos(theta))
            for jr in eachindex(layout.r_grid)
                field_values[jr, jt] += real(coeffs[jr]) * plm
            end
        end
    end

    fig = Figure()
    ax = Axis(fig[1,1],
        xlabel="r",
        ylabel="theta",
        title="Meridional: $field, mode $mode_index")
    hm = heatmap!(ax, layout.r_grid, collect(theta_grid), field_values)
    Colorbar(fig[1,2], hm)
    fig
end

end # module
