# =============================================================================
#  Tau-free ultraspherical-Galerkin assembly of the MHD eigenproblem.
#
#  Boundary conditions are carried by a recombined trial basis (no tau rows →
#  full-rank B → no infinite boundary eigenvalues).
#
#   - Hydro (u, v, h): pure-banded (B2) via `banded_radial_term`. Matches the
#     collocation onset spectrum to ~1e-12 with zero spurious.
#   - Magnetic (f, g) + velocity↔magnetic coupling (Lorentz, induction): built by
#     REUSING the audited C^0 operator functions, lifting each block to the row
#     equation's C^(q) basis, then projecting (`gbC0`). Magnetic DIFFUSION uses the
#     corrected MINUS sign (matches the viscous convention; free-decay dissipates —
#     see assembly.jl and test/mhd_physics.jl). The dipole
#     case is deferred (errors → caller routes to the tau path).
#
#  ADDITIVE: does not modify the tau path (`assemble_mhd_matrices`).
# =============================================================================

"""
    assemble_mhd_galerkin(op) -> (A, B, layout)

Assemble the MHD generalized eigenproblem in tau-free ultraspherical-Galerkin
form (hydro pure-banded; magnetic via reuse-and-lift of the audited operators,
with the corrected magnetic-diffusion sign). Returns dense `A`, `B` and a
`layout` NamedTuple `(index_map, M, R, fields, nred)` for reconstruction. Errors
for the dipole case (use the tau path).
"""
function assemble_mhd_galerkin(op::MHDStabilityOperator{T}) where {T}
    p = op.params
    N = p.N; ri = p.ricb; ro = one(T); gap = ro - ri; m = p.m
    E = p.E; Pr = p.Pr; Ra = p.Ra; Em = p.Em; Le = p.Le

    has_mag = !isempty(op.ll_f) || !isempty(op.ll_g)
    if has_mag && is_dipole_case(p.B0_type, p.ricb)
        error("assemble_mhd_galerkin: dipole magnetic case not implemented; use the tau path.")
    end
    if has_mag && !(p.bci_magnetic == 0 && p.bco_magnetic == 0)
        error("assemble_mhd_galerkin: only insulating magnetic BC (bci/bco_magnetic=0) " *
              "supported; use the tau path for conducting/perfect conductor.")
    end

    Rpol = recomb_poloidal_velocity(T, N, ri, ro; bci=p.bci, bco=p.bco)
    Rtor = recomb_toroidal_velocity(T, N, ri, ro; bci=p.bci, bco=p.bco)
    Rtem = recomb_temperature(T, N, ri, ro; bci=p.bci_thermal, bco=p.bco_thermal)
    Rg   = recomb_dirichlet(T, N)

    qof = Dict(:u => 4, :v => 2, :h => 2, :f => 2, :g => 2)
    Mof = Dict(:u => N + 1 - 4, :v => N + 1 - 2, :h => N + 1 - 2,
               :f => N + 1 - 2, :g => N + 1 - 2)

    Rmap = Dict{Tuple{Symbol,Int},Matrix{T}}()
    idx = Dict{Tuple{Symbol,Int},UnitRange{Int}}()
    off = 0
    for (f, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g), (:h, op.ll_h))
        for ℓ in ls
            Rmap[(f, ℓ)] = f === :u ? Matrix{T}(Rpol) :
                           f === :v ? Matrix{T}(Rtor) :
                           f === :h ? Matrix{T}(Rtem) :
                           f === :g ? Matrix{T}(Rg) :
                           recomb_magnetic_poloidal(T, N, ℓ, ri, ro; bci=p.bci_magnetic, bco=p.bco_magnetic)
            idx[(f, ℓ)] = (off + 1):(off + Mof[f])
            off += Mof[f]
        end
    end
    nred = off
    A = zeros(Complex{T}, nred, nred)
    B = zeros(Complex{T}, nred, nred)

    # `banded_radial_term` and the C^0→C^(q) conversion depend only on their
    # integer args (not ℓ), yet are hit once per ℓ across every section — memoize
    # so each distinct term/conversion is built exactly once per assembly.
    _bt_cache = Dict{NTuple{3,Int},SparseMatrixCSC{T,Int}}()
    bt(pw, d, q) = get!(() -> banded_radial_term(T, pw, d, q, N, ri, ro), _bt_cache, (pw, d, q))
    _cu_cache = Dict{Int,SparseMatrixCSC{T,Int}}()
    convup(q) = get!(() -> _convert_up(T, 0, q, N), _cu_cache, q)
    gb(band, fi, fj, ℓj) = galerkin_block(band, Rmap[(fj, ℓj)], Mof[fi])
    gbC0(blk0, fi, fj, ℓj) = galerkin_block(convup(qof[fi]) * blk0, Rmap[(fj, ℓj)], Mof[fi])

    Ra_int = Ra / gap^3
    beyonce = -Ra_int * E^2 / Pr
    thermaD = E / Pr
    Uls = op.ll_u; Vls = op.ll_v; Hls = op.ll_h; Fls = op.ll_f; Gls = op.ll_g

    # ---- Poloidal velocity equation (q=4) ----
    for ℓ in Uls
        L = T(ℓ * (ℓ + 1)); P = idx[(:u, ℓ)]
        B[P, P] += gb(-L * (L * bt(2,0,4) - 2 * bt(3,1,4) - bt(4,2,4)), :u, :u, ℓ)
        cor = (2im * m) * (-L * bt(2,0,4) + 2 * bt(3,1,4) + bt(4,2,4))
        vis = E * L * (-L * (ℓ+2) * (ℓ-1) * bt(0,0,4) + 2 * L * bt(2,2,4) - 4 * bt(3,3,4) - bt(4,4,4))
        A[P, P] += gb(cor - vis, :u, :u, ℓ)
        (ℓ-1) in Vls && (A[P, idx[(:v, ℓ-1)]] += gb(2 * ((ℓ^2-1) * sqrt(max(zero(T), T(ℓ^2-m^2))) / (2ℓ-1)) * ((ℓ-1) * bt(3,0,4) - bt(4,1,4)), :u, :v, ℓ-1))
        (ℓ+1) in Vls && (A[P, idx[(:v, ℓ+1)]] += gb(2 * (ℓ*(ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1)*(ℓ-m+1)))) / (2ℓ+3)) * (-(ℓ+2) * bt(3,0,4) - bt(4,1,4)), :u, :v, ℓ+1))
        ℓ in Hls && (A[P, idx[(:h, ℓ)]] += gb(beyonce * L * bt(4,0,4), :u, :h, ℓ))
        if Le > 0                                        # Lorentz: u ← magnetic
            for o in (-1, 1)
                (ℓ+o) in Fls && (A[P, idx[(:f, ℓ+o)]] += gbC0(operator_lorentz_poloidal_from_bpol(op, ℓ, m, o, Le), :u, :f, ℓ+o))
            end
            ℓ in Gls && (A[P, idx[(:g, ℓ)]] += gbC0(operator_lorentz_poloidal_diagonal(op, ℓ, Le), :u, :g, ℓ))
        end
    end

    # ---- Temperature equation (q=2) ----
    for ℓ in Hls
        L = T(ℓ * (ℓ + 1)); H = idx[(:h, ℓ)]
        if p.heating === :differential
            B[H, H] += gb(bt(3,0,2), :h, :h, ℓ)
            ℓ in Uls && (A[H, idx[(:u, ℓ)]] += gb((L * ri / gap) * bt(0,0,2), :h, :u, ℓ))
            A[H, H] += gb(thermaD * (-L * bt(1,0,2) + 2 * bt(2,1,2) + bt(3,2,2)), :h, :h, ℓ)
        else # internal heating: r² weighting and a conductive gradient ∝ r
            B[H, H] += gb(bt(2,0,2), :h, :h, ℓ)
            ℓ in Uls && (A[H, idx[(:u, ℓ)]] += gb(L * bt(2,0,2), :h, :u, ℓ))
            A[H, H] += gb(thermaD * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2)), :h, :h, ℓ)
        end
    end

    # ---- Toroidal velocity equation (q=2) ----
    for ℓ in Vls
        L = T(ℓ * (ℓ + 1)); V = idx[(:v, ℓ)]
        B[V, V] += gb(-L * bt(2,0,2), :v, :v, ℓ)
        cor = (-2im * m) * bt(2,0,2)
        vis = E * L * (-L * bt(0,0,2) + 2 * bt(1,1,2) + bt(2,2,2))
        A[V, V] += gb(cor - vis, :v, :v, ℓ)
        (ℓ-1) in Uls && (A[V, idx[(:u, ℓ-1)]] += gb(2 * ((ℓ^2-1) * sqrt(max(zero(T), T(ℓ^2-m^2))) / (2ℓ-1)) * ((ℓ-1) * bt(1,0,2) - bt(2,1,2)), :v, :u, ℓ-1))
        (ℓ+1) in Uls && (A[V, idx[(:u, ℓ+1)]] += gb(2 * (ℓ*(ℓ+2) * sqrt(max(zero(T), T((ℓ+m+1)*(ℓ-m+1)))) / (2ℓ+3)) * (-(ℓ+2) * bt(1,0,2) - bt(2,1,2)), :v, :u, ℓ+1))
        if Le > 0                                        # Lorentz: v ← magnetic
            for o in (0,)
                (ℓ+o) in Fls && (A[V, idx[(:f, ℓ+o)]] += gbC0(operator_lorentz_toroidal_from_bpol(op, ℓ, m, o, Le), :v, :f, ℓ+o))
            end
            for o in (-1, 1)
                (ℓ+o) in Gls && (A[V, idx[(:g, ℓ+o)]] += gbC0(operator_lorentz_toroidal_from_btor(op, ℓ, m, o, Le), :v, :g, ℓ+o))
            end
        end
    end

    # ---- Poloidal magnetic equation (q=2): mass + diffusion(−) + induction ----
    for ℓ in Fls
        F = idx[(:f, ℓ)]
        B[F, F] += gbC0(-operator_b_poloidal(op, ℓ), :f, :f, ℓ)
        A[F, F] += gbC0(-operator_magnetic_diffusion_poloidal(op, ℓ, Em), :f, :f, ℓ)
        if Le > 0
            for o in (-1, 1)
                (ℓ+o) in Uls && (A[F, idx[(:u, ℓ+o)]] += gbC0(operator_induction_poloidal_from_u(op, ℓ, m, o), :f, :u, ℓ+o))
            end
            for o in (0,)
                (ℓ+o) in Vls && (A[F, idx[(:v, ℓ+o)]] += gbC0(operator_induction_poloidal_from_v(op, ℓ, m, o), :f, :v, ℓ+o))
            end
        end
    end

    # ---- Toroidal magnetic equation (q=2): mass + diffusion(−) + induction ----
    for ℓ in Gls
        G = idx[(:g, ℓ)]
        B[G, G] += gbC0(-operator_b_toroidal(op, ℓ), :g, :g, ℓ)
        A[G, G] += gbC0(-operator_magnetic_diffusion_toroidal(op, ℓ, Em), :g, :g, ℓ)
        if Le > 0
            for o in (-1, 1)
                (ℓ+o) in Vls && (A[G, idx[(:v, ℓ+o)]] += gbC0(operator_induction_toroidal_from_v(op, ℓ, m, o), :g, :v, ℓ+o))
            end
            for o in (0,)
                (ℓ+o) in Uls && (A[G, idx[(:u, ℓ+o)]] += gbC0(operator_induction_toroidal_from_u(op, ℓ, m, o), :g, :u, ℓ+o))
            end
        end
    end

    layout = (index_map = idx, M = Mof, R = Rmap, fields = (:u, :v, :f, :g, :h), nred = nred)
    return A, B, layout
end

"""Full-eigenvector layout range for `(field, ℓ)` (sections u, v, f, g, h)."""
function _mhd_full_range(op::MHDStabilityOperator, field::Symbol, ℓ::Int)
    npm = op.params.N + 1
    nbu = length(op.ll_u); nbv = length(op.ll_v)
    nbf = length(op.ll_f); nbg = length(op.ll_g)
    base = field === :u ? (findfirst(==(ℓ), op.ll_u) - 1) * npm :
           field === :v ? (nbu + findfirst(==(ℓ), op.ll_v) - 1) * npm :
           field === :f ? (nbu + nbv + findfirst(==(ℓ), op.ll_f) - 1) * npm :
           field === :g ? (nbu + nbv + nbf + findfirst(==(ℓ), op.ll_g) - 1) * npm :
           field === :h ? (nbu + nbv + nbf + nbg + findfirst(==(ℓ), op.ll_h) - 1) * npm :
           error("unsupported field $field")
    return (base + 1):(base + npm)
end

"""Lift a reduced Galerkin eigenvector to a full `op.matrix_size` vector."""
function reconstruct_mhd_galerkin_full(op::MHDStabilityOperator, layout, y::AbstractVector)
    full = zeros(eltype(y), op.matrix_size)
    for (key, rng) in layout.index_map
        field, ℓ = key
        full[_mhd_full_range(op, field, ℓ)] = layout.R[key] * y[rng]
    end
    return full
end
