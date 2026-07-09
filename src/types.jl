# ============================================================================
# Shared types for Magrathea.jl v2.0
# ============================================================================

# --- Abstract base ---
abstract type AbstractStabilityResult{T} end

# --- Problem types ---

"""
    OnsetProblem{T}

Standard linear onset problem for rotating spherical shell convection.

Wraps an `OnsetParams` and validates parameters on construction.
Use `estimate_size` to preview memory requirements before solving.

# Fields
- `params::OnsetParams{T}` — problem parameters

# Example
```julia
p = OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64))
```
"""
struct OnsetProblem{T, BS}
    params::OnsetParams{T, BS}
    function OnsetProblem(params::OnsetParams{T, BS}) where {T, BS}
        validate_onset_params(params)
        new{T, BS}(params)
    end
end

"""
    BiglobalProblem{T}

Biglobal instability problem: onset on an axisymmetric basic state.

Combines `OnsetParams` with a `BasicState` (axisymmetric) and validates
consistency between them on construction.

# Fields
- `params::OnsetParams{T}` — problem parameters
- `basic_state::BasicState{T}` — axisymmetric background state

# Example
```julia
p = BiglobalProblem(params, basic_state)
```
"""
struct BiglobalProblem{T, BS}
    params::OnsetParams{T, BS}
    basic_state::BasicState{T}
    function BiglobalProblem(params::OnsetParams{T, BS}, basic_state::BasicState{T}) where {T, BS}
        validate_onset_params(params)
        validate_basic_state_consistency(basic_state, params)
        new{T, BS}(params, basic_state)
    end
end

"""
    TriglobalProblem{T}

Triglobal instability problem: onset on a fully 3D (non-axisymmetric) basic state.

Couples multiple azimuthal wavenumbers `m` simultaneously via a `BasicState3D`.

# Fields
- `params::OnsetParams{T}` — problem parameters
- `basic_state::BasicState3D{T}` — 3D background state
- `m_range::UnitRange{Int}` — range of coupled azimuthal wavenumbers

# Example
```julia
p = TriglobalProblem(params, basic_state_3d, 0:4)
```
"""
struct TriglobalProblem{T, BS}
    params::OnsetParams{T, BS}
    basic_state::BasicState3D{T}
    m_range::UnitRange{Int}
    function TriglobalProblem(params::OnsetParams{T, BS}, basic_state::BasicState3D{T}, m_range::UnitRange{Int}) where {T, BS}
        validate_triglobal_params(params, basic_state, m_range)
        new{T, BS}(params, basic_state, m_range)
    end
end

"""
    MHDProblem{T, BS}

Magnetohydrodynamic instability problem.

Loosely typed to avoid circular dependencies with the `CompleteMHD` module.
`basic_state` may be `nothing` for problems without an explicit background field.

# Fields
- `params` — MHD parameters (e.g., `MHDParams`)
- `basic_state::BS` — background state, or `nothing`
"""
struct MHDProblem{T, BS}
    params::MHDParams{T}
    basic_state::BS
    function MHDProblem{T, BS}(params::MHDParams{T}, basic_state::BS) where {T, BS}
        validate_mhd_params(params)   # soft @warn for unusual-but-valid inputs
        new{T, BS}(params, basic_state)
    end
end

"""Construct an MHD problem without an explicit basic-state object."""
MHDProblem(params::MHDParams{T}) where {T} = MHDProblem{T, Nothing}(params, nothing)

# --- Result type ---

"""
    StabilityResult{T<:Real, P, E}

Container for the output of a linear stability solve.

Stores all eigenvalues and eigenvectors and pre-extracts the leading (most
unstable) mode's growth rate and oscillation frequency.

# Fields
- `eigenvalues::Vector{Complex{T}}` — all computed eigenvalues
- `eigenvectors::Matrix{Complex{T}}` — corresponding eigenvectors (columns)
- `growth_rate::T` — real part of the leading eigenvalue
- `frequency::T` — imaginary part of the leading eigenvalue
- `problem::P` — the problem that was solved
- `extra::E` — optional solver metadata (default `(;)`)

# Example
```julia
result = solve(problem)
println(growth_rate(result))   # most unstable growth rate
println(frequency(result))     # oscillation frequency
mode = leading_mode(result)    # eigenvector of the leading mode
```
"""
struct StabilityResult{T<:Real, P, E} <: AbstractStabilityResult{T}
    eigenvalues::Vector{Complex{T}}
    eigenvectors::Matrix{Complex{T}}
    growth_rate::T
    frequency::T
    leading_index::Int
    problem::P
    extra::E
end

"""Construct a `StabilityResult` and identify the eigenvalue with largest real part."""
function StabilityResult(
    eigenvalues::Vector{Complex{T}},
    eigenvectors::Matrix{Complex{T}},
    problem::P;
    extra::E=(;)
) where {T<:Real, P, E}
    idx = argmax(real.(eigenvalues))
    gr = real(eigenvalues[idx])
    freq = imag(eigenvalues[idx])
    StabilityResult{T, P, E}(eigenvalues, eigenvectors, gr, freq, idx, problem, extra)
end

# --- Convenience accessors ---

"""
    growth_rate(r::StabilityResult) -> Real

Return the growth rate (real part of the leading eigenvalue) from a stability result.
"""
growth_rate(r::StabilityResult) = r.growth_rate

"""
    frequency(r::StabilityResult) -> Real

Return the oscillation frequency (imaginary part of the leading eigenvalue) from a stability result.
"""
frequency(r::StabilityResult) = r.frequency

"""
    leading_mode(r::StabilityResult) -> AbstractVector

Return the eigenvector corresponding to the most unstable (largest real part) eigenvalue.
"""
leading_mode(r::StabilityResult) = @view r.eigenvectors[:, r.leading_index]

# --- Problem size estimation (shared helpers) ---

"""
    _count_l_modes(m, lmax, symmetry)

Count the spherical-harmonic degrees retained for one azimuthal mode after
applying the requested equatorial-symmetry truncation.
"""
function _count_l_modes(m::Int, lmax::Int, symmetry::Symbol)
    if symmetry == :both
        return lmax - m + 1
    elseif symmetry == :symmetric
        return length(m:2:lmax)
    else  # :antisymmetric
        return length((m+1):2:lmax)
    end
end

"""
    _mem_gb(total_dof)

Estimate dense generalized-eigenproblem storage in GB for two ComplexF64
matrices, used only for user-facing size warnings.
"""
function _mem_gb(total_dof::Int)
    return 2 * total_dof^2 * sizeof(ComplexF64) / 1024^3
end

"""
    _hd_total_dof(m, lmax, Nr, symmetry)

Estimate dense hydrodynamic problem size for onset and biglobal wrappers.
"""
function _hd_total_dof(m::Int, lmax::Int, Nr::Int, symmetry::Symbol)
    # Real dense layout (LinearStabilityOperator index_map): Nr rows per (l,field)
    # block, with field-specific l-counts (P,Θ use poloidal parity; T toroidal).
    nP, nT, nΘ = _triglobal_mode_l_counts(m, lmax, symmetry)
    return (nP + nT + nΘ) * Nr
end

"""
    _triglobal_total_dof(m_range, lmax, Nr, symmetry=:both)

Estimate reduced triglobal DOFs across all coupled azimuthal blocks.
"""
function _triglobal_total_dof(m_range, lmax::Int, Nr::Int, symmetry::Symbol=:both)
    _validate_triglobal_m_range(m_range, lmax)
    total_dof = 0
    for m in m_range
        nP, nT, nΘ = _triglobal_mode_l_counts(m, lmax, symmetry)
        total_dof += nP * (Nr - 4) + nT * (Nr - 2) + nΘ * (Nr - 2)
    end
    dof_per_m = total_dof / length(m_range)
    return total_dof, dof_per_m
end

"""
    _mhd_total_dof(params)

Estimate MHD problem size and field-block counts from the selected symmetry and
background-field configuration.
"""
function _mhd_total_dof(params)
    # Mirror MHDStabilityOperator exactly: compute_mhd_l_modes drops the
    # degenerate l=0 mode (filter!(>=(1), …)); ll_h = ll_u; and for axial/dipole
    # (symmB0 = -1) the magnetic blocks take the OPPOSITE parity (ll_f=ll_v,
    # ll_g=ll_u). (MHD/types.jl:557-581.)
    ll_u, ll_v = compute_mhd_l_modes(params.m, params.lmax, params.symm, params.B0_type)
    n_pol = length(ll_u)
    n_tor = length(ll_v)
    n_per_mode = params.N + 1
    if params.B0_type == no_field
        n_f = 0
        n_g = 0
    elseif params.B0_type == axial || params.B0_type == dipole   # symmB0 = -1
        n_f = n_tor
        n_g = n_pol
    else                                                          # symmB0 = +1
        n_f = n_pol
        n_g = n_tor
    end
    n_h = n_pol  # temperature shares poloidal-velocity parity
    total_dof = (n_pol + n_tor + n_f + n_g + n_h) * n_per_mode
    return total_dof, n_pol, n_tor, n_f, n_g, n_per_mode
end

"""
    estimate_size(p)

Print a human-readable estimate of the matrix size and memory requirement for problem `p`.

Accepts `OnsetProblem`, `BiglobalProblem`, `TriglobalProblem`, or `MHDProblem`.
Warns when the estimated memory exceeds 8 GB.
"""
function estimate_size(p::OnsetProblem)
    params = p.params
    nP, nT, nΘ = _triglobal_mode_l_counts(params.m, params.lmax, params.equatorial_symmetry)
    total_dof = _hd_total_dof(params.m, params.lmax, params.Nr, params.equatorial_symmetry)
    mem_gb = _mem_gb(total_dof)

    println("OnsetProblem size estimate")
    _tree_row(stdout, "l-modes", "$nP/$nT/$nΘ P/T/Θ (m=$(params.m), lmax=$(params.lmax), $(params.equatorial_symmetry))")
    _tree_row(stdout, "degrees of freedom per mode", "$(params.Nr) radial points per (l,field) block")
    _tree_row(stdout, "matrix size", "$total_dof × $total_dof")
    warning = mem_gb > 8.0 ? " (large; reduce lmax or Nr)" : ""
    _tree_row(stdout, "dense storage estimate", @sprintf("~%.1f GB%s", mem_gb, warning); last=true)
end

"""
    estimate_size(p::BiglobalProblem)

Print the l-mode count, dense matrix dimension, and approximate memory for a
biglobal solve.
"""
function estimate_size(p::BiglobalProblem)
    params = p.params
    nP, nT, nΘ = _triglobal_mode_l_counts(params.m, params.lmax, params.equatorial_symmetry)
    total_dof = _hd_total_dof(params.m, params.lmax, params.Nr, params.equatorial_symmetry)
    mem_gb = _mem_gb(total_dof)

    println("BiglobalProblem size estimate")
    _tree_row(stdout, "l-modes", "$nP/$nT/$nΘ P/T/Θ (m=$(params.m), lmax=$(params.lmax), $(params.equatorial_symmetry))")
    _tree_row(stdout, "degrees of freedom per mode", "$(params.Nr) radial points per (l,field) block")
    _tree_row(stdout, "matrix size", "$total_dof × $total_dof")
    warning = mem_gb > 8.0 ? " (large; reduce lmax or Nr)" : ""
    _tree_row(stdout, "dense storage estimate", @sprintf("~%.1f GB%s", mem_gb, warning); last=true)
end

"""
    estimate_size(p::TriglobalProblem)

Print coupled-mode counts and approximate dense storage for a triglobal solve.
"""
function estimate_size(p::TriglobalProblem)
    params = p.params
    total_dof, dof_per_m = _triglobal_total_dof(
        p.m_range, params.lmax, params.Nr, params.equatorial_symmetry)
    mem_gb = _mem_gb(total_dof)

    println("TriglobalProblem size estimate")
    _tree_row(stdout, "coupled modes", "$(p.m_range) ($(length(p.m_range)) modes)")
    _tree_row(stdout, "degrees of freedom per mode", "~$dof_per_m (lmax=$(params.lmax), Nr=$(params.Nr), 3 fields)")
    _tree_row(stdout, "matrix size", "$total_dof × $total_dof")
    warning = mem_gb > 8.0 ? " (large; reduce lmax or m_range)" : ""
    _tree_row(stdout, "dense storage estimate", @sprintf("~%.1f GB%s", mem_gb, warning); last=true)
end

"""
    basic_state(params::OnsetParams; mode=:conduction, amplitude=0.05, mmax_bs=2,
                lmax_bs=4, max_iterations=20, tol=1e-8)

Convenience constructor that builds a basic state directly from an `OnsetParams`,
selecting the implementation by `mode`. The radial grid (`ChebyshevDiffn`) is
derived from `params.Nr` and `params.χ`.

- `:conduction`      — pure conduction profile (`conduction_basic_state`)
- `:meridional`      — axisymmetric thermal-wind state of strength `amplitude`
                       (`meridional_basic_state`) → `BasicState`
- `:selfconsistent`  — self-consistent geostrophic balance from a flux BC
                       `Y00(-1) + Σ Y(2,m)(amplitude)` (`basic_state_selfconsistent`)
- `:nonaxisymmetric` — 3-D state with `m≠0` boundary forcing of strength
                       `amplitude` at degree 2, `m=1…mmax_bs`
                       (`nonaxisymmetric_basic_state`) → `BasicState3D`
"""
function basic_state(params::OnsetParams{T}; mode::Symbol=:conduction,
                     amplitude::Real=0.05, mmax_bs::Int=2, lmax_bs::Int=4,
                     max_iterations::Int=20, tol::Real=1e-8) where {T}
    cd = ChebyshevDiffn(params.Nr, [T(params.χ), one(T)], 4)
    χ = T(params.χ); E = T(params.E); Ra = T(params.Ra); Pr = T(params.Pr)
    if mode === :conduction
        return conduction_basic_state(cd, χ, lmax_bs; thermal_bc=params.thermal_bc)
    elseif mode === :meridional
        return meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, T(amplitude);
                                      mechanical_bc=params.mechanical_bc,
                                      thermal_bc=params.thermal_bc)
    elseif mode === :selfconsistent
        flux = Y00(-one(T))
        for mm in 1:mmax_bs
            flux = flux + Ylm(2, mm, T(amplitude))
        end
        bs, _ = basic_state_selfconsistent(cd, χ, E, Ra, Pr; flux_bc=flux,
                                           mechanical_bc=params.mechanical_bc,
                                           lmax_bs=lmax_bs, max_iterations=max_iterations,
                                           tolerance=T(tol))
        return bs
    elseif mode === :nonaxisymmetric
        amps = Dict{Tuple{Int,Int},T}((2, mm) => T(amplitude) for mm in 1:mmax_bs)
        return nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amps;
                                           mechanical_bc=params.mechanical_bc,
                                           thermal_bc=params.thermal_bc)
    else
        throw(ArgumentError("basic_state: unknown mode :$mode " *
              "(use :conduction, :meridional, :selfconsistent, or :nonaxisymmetric)"))
    end
end

"""
    estimate_size(p::MHDProblem)

Print field counts, matrix dimension, and approximate dense storage for an MHD
stability solve.
"""
function estimate_size(p::MHDProblem)
    total_dof, n_pol, n_tor, _, _, n_per_mode = _mhd_total_dof(p.params)
    mem_gb = _mem_gb(total_dof)
    n_fields = p.params.B0_type == no_field ? 3 : 5

    println("MHDProblem size estimate")
    _tree_row(stdout, "l-modes", "$n_pol poloidal + $n_tor toroidal ($n_fields fields)")
    _tree_row(stdout, "degrees of freedom per mode", "$n_per_mode (N=$(p.params.N))")
    _tree_row(stdout, "matrix size", "$total_dof × $total_dof")
    warning = mem_gb > 8.0 ? " (large; reduce lmax or N)" : ""
    _tree_row(stdout, "dense storage estimate", @sprintf("~%.1f GB%s", mem_gb, warning); last=true)
end

# --- Makie extension stubs ---

"""Extension hook for plotting an eigenvalue spectrum when plotting extras are loaded."""
function eigenspectrum end

"""Extension hook for plotting meridional fields when plotting extras are loaded."""
function plot_meridional end

"""Extension hook for plotting radial profiles when plotting extras are loaded."""
function plot_radial end

"""Reconstruct physical perturbation velocity `(u_r, u_θ, u_φ)` on a meridional grid."""
function perturbation_velocity end

"""Reconstruct the physical perturbation temperature field on a meridional grid."""
function perturbation_temperature end

"""Reconstruct physical perturbation magnetic field `(B_r, B_θ, B_φ)` (MHD only)."""
function perturbation_magnetic end
