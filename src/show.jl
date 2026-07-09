# ============================================================================
# Pretty-printing for Magrathea.jl public types
# ============================================================================

import Base: show

"""Print one `├──`/`└──` tree row."""
function _tree_row(io::IO, label::AbstractString, value; last::Bool=false)
    branch = last ? "└── " : "├── "
    print(io, branch, label, ": ", value)
    last || println(io)
    return nothing
end

"""Format a radial domain for display."""
function _domain_summary(r)
    isempty(r) && return "unknown"
    r_min = @sprintf("%.3f", minimum(r))
    r_max = @sprintf("%.3f", maximum(r))
    return "[$r_min, $r_max]"
end

"""Join active harmonic degrees for compact display."""
function _degree_summary(keys_iter)
    degrees = sort(collect(keys_iter))
    isempty(degrees) && return "none"
    return join(("ℓ=$ℓ" for ℓ in degrees), ", ")
end

# ---------------------------------------------------------------------------
# Compact summaries + nested tree display.
#
#   summary(x)                    → compact one-liner; also the tree header and
#                                   what prints when `x` appears inline (inside
#                                   an array, another struct, `@show`, `print`).
#   show(io, x)                   → the same compact one-liner.
#   show(io, MIME"text/plain", x) → summary header, then a `├──`/`└──` field
#                                   tree. Struct-valued fields (basic states)
#                                   recurse, threading `│  ` continuation
#                                   prefixes beneath their parent.
# ---------------------------------------------------------------------------

"""Emit one nested tree row. Leads with a newline so parent rows can carry subtrees."""
function _emit_row(io::IO, prefix::AbstractString, last::Bool, label, value)
    print(io, '\n', prefix, last ? "└── " : "├── ", label, ": ", value)
    return nothing
end

"""Continuation prefix for a node's children: 4 spaces if the node was last, else `│   `."""
_child_prefix(prefix::AbstractString, last::Bool) = prefix * (last ? "    " : "│   ")

# All tree-printable public types share the same `show` plumbing; only their
# `summary` (header) and `_show_children` (rows) differ.
for TT in (:OnsetParams, :BiglobalParams, :TriglobalParams, :MHDParams,
           :OnsetProblem, :BiglobalProblem, :TriglobalProblem, :MHDProblem,
           :BasicState, :BasicState3D, :StabilityResult)
    @eval begin
        Base.show(io::IO, x::$TT) = print(io, summary(x))
        function Base.show(io::IO, ::MIME"text/plain", x::$TT)
            print(io, summary(x))
            _show_children(io, x, "")
            return nothing
        end
    end
end

# --- OnsetParams ---
Base.summary(p::OnsetParams{T}) where {T} =
    "OnsetParams{$T}(E=$(p.E), Ra=$(p.Ra), m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr))"

"""Tree the hydrodynamic onset parameters: dynamics, geometry, resolution, BCs."""
function _show_children(io::IO, p::OnsetParams, prefix::AbstractString)
    _emit_row(io, prefix, false, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _emit_row(io, prefix, false, "geometry", "χ=$(p.χ), ri=$(p.ri), ro=$(p.ro), L=$(p.L)")
    _emit_row(io, prefix, false, "resolution", "m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr)")
    _emit_row(io, prefix, false, "boundary conditions", "mechanical=$(p.mechanical_bc), thermal=$(p.thermal_bc)")
    _emit_row(io, prefix, true, "equatorial symmetry", p.equatorial_symmetry)
end

# --- BasicState ---
Base.summary(bs::BasicState{T}) where {T} = "BasicState{$T}(lmax_bs=$(bs.lmax_bs), Nr=$(bs.Nr))"

"""Tree the active axisymmetric temperature and zonal-flow modes of a basic state."""
function _show_children(io::IO, bs::BasicState, prefix::AbstractString)
    _emit_row(io, prefix, false, "temperature modes", _degree_summary(keys(bs.theta_coeffs)))
    _emit_row(io, prefix, false, "zonal-flow modes", _degree_summary(keys(bs.uphi_coeffs)))
    _emit_row(io, prefix, true, "radial domain", _domain_summary(bs.r))
end

# --- BasicState3D ---
Base.summary(bs::BasicState3D{T}) where {T} =
    "BasicState3D{$T}(lmax_bs=$(bs.lmax_bs), mmax_bs=$(bs.mmax_bs), Nr=$(bs.Nr))"

"""Tree the dimensions and active-mode count of a 3D basic state."""
function _show_children(io::IO, bs::BasicState3D, prefix::AbstractString)
    _emit_row(io, prefix, false, "active temperature modes", length(bs.theta_coeffs))
    _emit_row(io, prefix, true, "radial domain", _domain_summary(bs.r))
end

# --- StabilityResult ---
Base.summary(r::StabilityResult{T}) where {T} =
    "StabilityResult{$T} with $(length(r.eigenvalues)) eigenvalues"

"""Tree the leading-eigenvalue summary and the source problem of a solve."""
function _show_children(io::IO, r::StabilityResult, prefix::AbstractString)
    _emit_row(io, prefix, false, "leading eigenvalue", r.eigenvalues[r.leading_index])
    _emit_row(io, prefix, false, "growth rate", r.growth_rate)
    _emit_row(io, prefix, false, "frequency", r.frequency)
    _emit_row(io, prefix, true, "problem", _problem_name(r.problem))
end

"""Build the short problem label embedded in `StabilityResult` display output."""
_problem_name(p::OnsetProblem) = "OnsetProblem (E=$(p.params.E), Ra=$(p.params.Ra))"

"""Build the short biglobal problem label embedded in `StabilityResult` display output."""
_problem_name(p::BiglobalProblem) = "BiglobalProblem (E=$(p.params.E), Ra=$(p.params.Ra))"

"""Build the short triglobal problem label embedded in `StabilityResult` display output."""
_problem_name(p::TriglobalProblem) = "TriglobalProblem (E=$(p.params.E), m=$(p.m_range))"

"""Build the MHD problem label while tolerating incomplete custom params."""
function _problem_name(p::MHDProblem)
    try
        mp = p.params
        return "MHDProblem (E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le), m=$(mp.m))"
    catch
        return "MHDProblem"
    end
end

"""Fallback problem label for unknown result wrappers."""
_problem_name(::Any) = "Unknown"

# --- OnsetProblem ---
Base.summary(p::OnsetProblem{T}) where {T} = "OnsetProblem{$T}(E=$(p.params.E), Ra=$(p.params.Ra))"

"""Tree the defining resolution and physics of an onset wrapper."""
function _show_children(io::IO, p::OnsetProblem, prefix::AbstractString)
    _emit_row(io, prefix, false, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _emit_row(io, prefix, false, "resolution", "m=$(p.params.m), lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _emit_row(io, prefix, true, "boundary conditions", "mechanical=$(p.params.mechanical_bc), thermal=$(p.params.thermal_bc)")
end

# --- BiglobalProblem ---
Base.summary(p::BiglobalProblem{T}) where {T} = "BiglobalProblem{$T}(E=$(p.params.E), Ra=$(p.params.Ra))"

"""Tree the biglobal wrapper, nesting its axisymmetric basic state."""
function _show_children(io::IO, p::BiglobalProblem, prefix::AbstractString)
    _emit_row(io, prefix, false, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _emit_row(io, prefix, false, "resolution", "m=$(p.params.m), lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _emit_row(io, prefix, true, "basic_state", summary(p.basic_state))
    _show_children(io, p.basic_state, _child_prefix(prefix, true))
end

# --- TriglobalProblem ---
Base.summary(p::TriglobalProblem{T}) where {T} = "TriglobalProblem{$T}(E=$(p.params.E), m_range=$(p.m_range))"

"""Tree the triglobal wrapper, nesting its 3D basic state."""
function _show_children(io::IO, p::TriglobalProblem, prefix::AbstractString)
    _emit_row(io, prefix, false, "parameters", "E=$(p.params.E), Ra=$(p.params.Ra), Pr=$(p.params.Pr), χ=$(p.params.χ)")
    _emit_row(io, prefix, false, "resolution", "lmax=$(p.params.lmax), Nr=$(p.params.Nr)")
    _emit_row(io, prefix, false, "coupled modes", "$(p.m_range) ($(length(p.m_range)) modes)")
    _emit_row(io, prefix, true, "basic_state", summary(p.basic_state))
    _show_children(io, p.basic_state, _child_prefix(prefix, true))
end

# --- BiglobalParams ---
Base.summary(p::BiglobalParams{T}) where {T} =
    "BiglobalParams{$T}(E=$(p.E), Ra=$(p.Ra), m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr))"

"""Tree the biglobal solver parameters, nesting the attached basic state."""
function _show_children(io::IO, p::BiglobalParams, prefix::AbstractString)
    _emit_row(io, prefix, false, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _emit_row(io, prefix, false, "geometry", "χ=$(p.χ)")
    _emit_row(io, prefix, false, "resolution", "m=$(p.m), lmax=$(p.lmax), Nr=$(p.Nr)")
    _emit_row(io, prefix, true, "basic_state", summary(p.basic_state))
    _show_children(io, p.basic_state, _child_prefix(prefix, true))
end

# --- TriglobalParams ---
Base.summary(p::TriglobalParams{T}) where {T} =
    "TriglobalParams{$T}(E=$(p.E), Ra=$(p.Ra), m_range=$(p.m_range), lmax=$(p.lmax), Nr=$(p.Nr))"

"""Tree the triglobal solver parameters, symmetry, and nested 3D basic state."""
function _show_children(io::IO, p::TriglobalParams, prefix::AbstractString)
    _emit_row(io, prefix, false, "dynamics", "E=$(p.E), Pr=$(p.Pr), Ra=$(p.Ra)")
    _emit_row(io, prefix, false, "geometry", "χ=$(p.χ)")
    _emit_row(io, prefix, false, "resolution", "m_range=$(p.m_range), lmax=$(p.lmax), Nr=$(p.Nr)")
    _emit_row(io, prefix, false, "equatorial symmetry", p.equatorial_symmetry)
    _emit_row(io, prefix, true, "basic_state", summary(p.basic_state_3d))
    _show_children(io, p.basic_state_3d, _child_prefix(prefix, true))
end

# --- MHDParams ---
Base.summary(p::MHDParams{T}) where {T} =
    "MHDParams{$T}(E=$(p.E), Ra=$(p.Ra), Pm=$(p.Pm), Le=$(p.Le), m=$(p.m), lmax=$(p.lmax), N=$(p.N))"

"""Tree the MHD solver parameters, boundary conditions, and background field."""
function _show_children(io::IO, p::MHDParams, prefix::AbstractString)
    _emit_row(io, prefix, false, "dynamics", "E=$(p.E), Pr=$(p.Pr), Pm=$(p.Pm), Ra=$(p.Ra), Le=$(p.Le)")
    _emit_row(io, prefix, false, "geometry", "ricb=$(p.ricb)")
    _emit_row(io, prefix, false, "resolution", "m=$(p.m), lmax=$(p.lmax), N=$(p.N), symm=$(p.symm)")
    _emit_row(io, prefix, false, "background field", "$(p.B0_type) (amplitude=$(p.B0_amplitude))")
    _emit_row(io, prefix, false, "mechanical BCs", "inner=$(p.bci), outer=$(p.bco)")
    _emit_row(io, prefix, false, "thermal BCs", "inner=$(p.bci_thermal), outer=$(p.bco_thermal)")
    _emit_row(io, prefix, false, "magnetic BCs", "inner=$(p.bci_magnetic), outer=$(p.bco_magnetic)")
    _emit_row(io, prefix, true, "heating", p.heating)
end

# --- MHDProblem ---
"""Summarize an MHD problem wrapper while tolerating custom parameter objects."""
function Base.summary(p::MHDProblem{T, BS}) where {T, BS}
    try
        mp = p.params
        return "MHDProblem{$T, $BS}(E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le))"
    catch
        return "MHDProblem{$T, $BS}"
    end
end

"""Tree an MHD problem wrapper while tolerating incomplete custom params."""
function _show_children(io::IO, p::MHDProblem, prefix::AbstractString)
    try
        mp = p.params
        _emit_row(io, prefix, false, "dynamics", "E=$(mp.E), Ra=$(mp.Ra), Pm=$(mp.Pm), Le=$(mp.Le)")
        _emit_row(io, prefix, false, "resolution", "m=$(mp.m), lmax=$(mp.lmax), N=$(mp.N)")
        _emit_row(io, prefix, true, "background field", mp.B0_type)
    catch
        _emit_row(io, prefix, true, "params type", typeof(p.params))
    end
end
