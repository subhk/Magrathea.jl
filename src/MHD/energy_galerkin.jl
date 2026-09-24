# =============================================================================
#  Energy-conserving Galerkin assembly of the MHD eigenproblem.
#
#  Every momentum and induction equation is tested against its own trial basis in
#  the energy inner product. The discrete Lorentz force is then the exact negative
#  adjoint of the induction term, Coriolis stays skew-Hermitian, and viscous and
#  ohmic terms stay dissipative. Without buoyancy no eigenvalue can grow, at any
#  resolution. The tau and Petrov–Galerkin forms instead admit spurious growing
#  modes when a strong field leaves the magnetic boundary layers unresolved.
#
#  Kinetic and magnetic energy (including the exterior potential field of
#  insulating walls), viscous and ohmic dissipation, and the Coriolis term within
#  one degree have closed forms in the poloidal/toroidal potentials that need at
#  most second derivatives. Couplings between degrees reuse the strong-form row
#  operators: for test potentials that vanish at the walls,
#  ∫ ũ·F dV = ∫ P̃ (r·∇×∇×F) r² dr and ∫ ũ·F dV = ∫ T̃ (r·∇×F) r² dr exactly, for
#  any F. The induction coupling is the adjoint of the Lorentz coupling, which also
#  gives the weak form of perfectly conducting walls with slip. Temperature keeps
#  the Petrov–Galerkin projection of `assemble_mhd_galerkin`.
# =============================================================================

"""Gauss–Legendre nodes and weights on [-1, 1] (Golub–Welsch)."""
function _gauss_legendre(::Type{T}, n::Int) where {T<:Real}
    β = T[k / sqrt(T(4k^2 - 1)) for k in 1:(n - 1)]
    F = eigen(SymTridiagonal(zeros(T, n), β))
    return F.values, 2 .* F.vectors[1, :] .^ 2
end

"""Chebyshev polynomials `T_0 … T_n` at the points `x` (one row per point)."""
_chebyshev_vandermonde(x::AbstractVector{T}, n::Int) where {T<:Real} =
    T[cos(k * acos(clamp(t, -one(T), one(T)))) for t in x, k in 0:n]

"""Chebyshev coefficients of the radial derivative of each column of `C`."""
function _chebyshev_derivative(C::AbstractMatrix{T}, scale) where {T}
    n = size(C, 1) - 1
    D = zeros(T, size(C))
    n == 0 && return D
    D[n, :] .= 2n .* C[n + 1, :]
    for k in (n - 2):-1:0
        D[k + 1, :] .= D[k + 3, :] .+ 2(k + 1) .* C[k + 2, :]
    end
    D[1, :] ./= 2
    return scale .* D
end

"""`op` rebuilt at polynomial degree `N`, so that the row operators act on the trial
functions without truncating the highest coefficients."""
function _mhd_operator_at_degree(op::MHDStabilityOperator{T}, N::Int) where {T}
    p = op.params
    q = MHDParams{T}(p.E, p.Pr, p.Pm, p.Ra, p.Le, p.ricb, p.m, p.lmax, p.symm, N,
                     p.B0_type, p.B0_amplitude, p.bci, p.bco, p.bci_thermal,
                     p.bco_thermal, p.bci_magnetic, p.bco_magnetic,
                     p.forcing_frequency, p.heating, p.L, p.Etherm, p.Em)
    return with_logger(() -> MHDStabilityOperator(q), NullLogger())
end

"""Toroidal magnetic trial basis: `g = 0` at insulating walls. Perfectly conducting
walls constrain no coefficient, since their tangential electric field vanishes weakly."""
function _recomb_magnetic_toroidal(::Type{T}, N::Int, bci::Int, bco::Int) where {T}
    funcs = [_chebyshev_boundary_values(N, side, T)
             for (side, bc) in ((:outer, bco), (:inner, bci)) if bc == 0]
    isempty(funcs) && return Matrix{T}(I, N + 1, N + 1)
    return T.(recomb_from_functionals(permutedims(reduce(hcat, funcs))))
end

"""Whether `assemble_mhd_energy_galerkin` supports the magnetic walls of `p`."""
_mhd_energy_galerkin_supported(p::MHDParams) = p.bci_magnetic in (0, 2) && p.bco_magnetic in (0, 2)

"""
    assemble_mhd_energy_galerkin(op) -> (A, B, layout)

Energy-conserving Galerkin form of the MHD eigenproblem `A x = λ B x` for insulating
or perfectly conducting magnetic walls, with an axial, dipole, or no background field.
A finite-conductivity inner core is not supported; use `assemble_mhd_matrices`.

Each momentum and induction equation is tested against its own trial basis in the
energy inner product, so `B` is Hermitian positive definite on the velocity and
magnetic blocks, and without buoyancy no eigenvalue has a positive real part at any
resolution. Returns sparse `A`, `B` and the `layout` used by
`reconstruct_mhd_galerkin_full`.
"""
function assemble_mhd_energy_galerkin(op::MHDStabilityOperator{T}) where {T}
    p = op.params
    _mhd_energy_galerkin_supported(p) || throw(ArgumentError(
        "The energy-conserving Galerkin assembly supports insulating (0) or perfectly " *
        "conducting (2) magnetic walls; use assemble_mhd_matrices for a conducting core."))
    N = p.N; ri = p.ricb; ro = one(T); m = p.m
    Npad = N + 8
    opp = _mhd_operator_at_degree(op, Npad)

    # Gauss–Legendre quadrature on [ri, ro]. The r^(2-k) test weights are analytic
    # inside a Bernstein ellipse of parameter ρ; the extra nodes reach round-off.
    ρ = (ro + ri) / (ro - ri); ρ += sqrt(ρ^2 - 1)
    x, wx = _gauss_legendre(T, 2Npad + 20 + ceil(Int, 40 / log(ρ)))
    r = ri .+ (x .+ 1) .* ((ro - ri) / 2)
    w = wx .* ((ro - ri) / 2)
    V = _chebyshev_vandermonde(x, Npad)
    scale = T(_radial_scale(ri, ro))
    padded(R) = vcat(R, zeros(T, Npad - N, size(R, 2)))

    # Trial bases, which are also the test bases.
    Ru = Matrix{T}(recomb_poloidal_velocity(T, N, ri, ro; bci=p.bci, bco=p.bco))
    Rv = Matrix{T}(recomb_toroidal_velocity(T, N, ri, ro; bci=p.bci, bco=p.bco))
    Rg = _recomb_magnetic_toroidal(T, N, p.bci_magnetic, p.bco_magnetic)
    Rh = Matrix{T}(recomb_temperature(T, N, ri, ro; bci=p.bci_thermal, bco=p.bco_thermal))
    basis = Dict{Tuple{Symbol,Int},Matrix{T}}()
    for ℓ in op.ll_u; basis[(:u, ℓ)] = Ru; end
    for ℓ in op.ll_v; basis[(:v, ℓ)] = Rv; end
    for ℓ in op.ll_f
        basis[(:f, ℓ)] = Matrix{T}(recomb_magnetic_poloidal(T, N, ℓ, ri, ro;
                                   bci=p.bci_magnetic, bco=p.bco_magnetic))
    end
    for ℓ in op.ll_g; basis[(:g, ℓ)] = Rg; end
    for ℓ in op.ll_h; basis[(:h, ℓ)] = Rh; end
    if _mhd_angular_momentum_gauge(op)
        # Zero angular momentum joins the wall conditions of the ℓ = 1 toroidal
        # basis, which restricts trial and test functions alike.
        funcs = vcat(_toroidal_velocity_functionals(T, N, ri, ro, p.bci, p.bco),
                     permutedims(_mhd_angular_momentum_functional(op)))
        basis[(:v, 1)] = recomb_from_functionals(funcs)
    end

    idx = Dict{Tuple{Symbol,Int},UnitRange{Int}}()
    n = 0
    for (field, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g),
                        (:h, op.ll_h)), ℓ in ls
        k = size(basis[(field, ℓ)], 2)
        idx[(field, ℓ)] = (n + 1):(n + k)
        n += k
    end

    Ai = Int[]; Aj = Int[]; Av = Complex{T}[]
    Bi = Int[]; Bj = Int[]; Bv = Complex{T}[]
    function push_block!(is, js, vs, row, col, block)
        for (jj, j) in enumerate(idx[col]), (ii, i) in enumerate(idx[row])
            iszero(block[ii, jj]) && continue
            push!(is, i); push!(js, j); push!(vs, block[ii, jj])
        end
    end
    addA!(row, col, block) = push_block!(Ai, Aj, Av, row, col, block)
    addB!(key, block) = push_block!(Bi, Bj, Bv, key, key, block)
    hermitian(X) = (X + X') / 2

    # Energy weight of a degree-ℓ harmonic in the native normalization, whose
    # squared norm is proportional to 1/(2ℓ+1).
    weight(ℓ) = one(T) / (2ℓ + 1)
    nodal(key) = V * padded(basis[key])
    derivative(key) = _chebyshev_derivative(padded(basis[key]), scale)
    at_wall(side, C) = _chebyshev_boundary_values(Npad, side, T)' * C

    # ---- Momentum within one degree (closed forms) ---------------------------
    # u = ∇×∇×(P Y r) has energy L ∫ (L P̃P + r² P̃'P') dr for P = 0 at the walls,
    # and ∇×u is toroidal with potential −D_ℓP. Viscosity gives −∫ ∇×ũ·∇×u plus the
    # stress-free wall terms (2/R) ∮ ũ·u at ro and −(2/R) ∮ ũ·u at ri, since the
    # tangential vorticity there is 2 r̂×u/R; no-slip test functions vanish at the
    # wall. Coriolis within one degree is (2im/L) times the energy.
    for ℓ in op.ll_u
        key = (:u, ℓ); L = T(ℓ * (ℓ + 1)); c = weight(ℓ) * L
        P = nodal(key); d1 = derivative(key); dP = V * d1
        r2DP = r .^ 2 .* (V * _chebyshev_derivative(d1, scale)) .+ 2 .* r .* dP .- L .* P
        Mu = hermitian(c .* (L .* (P' * (w .* P)) .+ dP' * ((w .* r .^ 2) .* dP)))
        po = at_wall(:outer, d1); pin = at_wall(:inner, d1)
        visc = -p.E * c .* (r2DP' * ((w ./ r .^ 2) .* r2DP) .- 2ro .* (po' * po) .+
                            2ri .* (pin' * pin))
        addB!(key, Mu)
        addA!(key, key, (2im * m / L) .* Mu .+ hermitian(visc))
    end
    # u = ∇×(T Y r) has energy L ∫ r² T̃T dr, and ∇×u is poloidal with potential T.
    for ℓ in op.ll_v
        key = (:v, ℓ); L = T(ℓ * (ℓ + 1)); c = weight(ℓ) * L
        Tv = nodal(key); drT = Tv .+ r .* (V * derivative(key))
        Mv = hermitian(c .* (Tv' * ((w .* r .^ 2) .* Tv)))
        to = at_wall(:outer, padded(basis[key])); tin = at_wall(:inner, padded(basis[key]))
        visc = -p.E * c .* (L .* (Tv' * (w .* Tv)) .+ drT' * (w .* drT) .- 2ro .* (to' * to) .+
                            2ri .* (tin' * tin))
        addB!(key, Mv)
        addA!(key, key, (2im * m / L) .* Mv .+ hermitian(visc))
    end

    # ---- Momentum across degrees: ∫ ũ·F dV from the strong-form rows ---------
    # Poloidal rows are −r^ku r·∇×∇×F and toroidal rows −r^kv r·∇×F, where the
    # dipole multiplies the equations by extra powers of r.
    ku, kv = is_dipole_case(p.B0_type, p.ricb) ? (6, 5) : (4, 2)
    tester(key, k) = (nodal(key) .* (w .* (-r .^ (2 - k))))' .* weight(key[2])
    project(t, row_op, col) = t * (V * (row_op * padded(basis[col])))
    lorentz = Pair{Tuple{Tuple{Symbol,Int},Tuple{Symbol,Int}},Matrix{Complex{T}}}[]

    for ℓ in op.ll_u
        key = (:u, ℓ); t = tester(key, ku)
        for o in (-1, 1)
            v = (:v, ℓ + o)
            if haskey(idx, v)
                C = project(t, first(operator_coriolis_offdiag(opp, ℓ, m, o)), v)
                addA!(key, v, C)
                addA!(v, key, -C')
            end
            f = (:f, ℓ + o)
            haskey(idx, f) && push!(lorentz, (key, f) =>
                project(t, operator_lorentz_poloidal_from_bpol(opp, ℓ, m, o, one(T)), f))
        end
        g = (:g, ℓ)
        haskey(idx, g) && push!(lorentz, (key, g) =>
            project(t, operator_lorentz_poloidal_diagonal(opp, ℓ, one(T)), g))
        h = (:h, ℓ)
        haskey(idx, h) && addA!(key, h, project(t, operator_buoyancy(opp, ℓ, p.Ra, p.Pr), h))
    end
    for ℓ in op.ll_v
        key = (:v, ℓ); t = tester(key, kv)
        f = (:f, ℓ)
        haskey(idx, f) && push!(lorentz, (key, f) =>
            project(t, operator_lorentz_toroidal_from_bpol(opp, ℓ, m, 0, one(T)), f))
        for o in (-1, 1)
            g = (:g, ℓ + o)
            haskey(idx, g) && push!(lorentz, (key, g) =>
                project(t, operator_lorentz_toroidal_from_btor(opp, ℓ, m, o, one(T)), g))
        end
    end

    # Lorentz force Le²·K and induction −Kᴴ: the coupling conserves energy exactly.
    for ((row, col), K) in lorentz
        addA!(row, col, p.Le^2 .* K)
        addA!(col, row, -K')
    end

    # ---- Magnetic energy and ohmic dissipation (closed forms) ------------------
    # For b = ∇×∇×(f Y r), with the potential continuation of insulating walls,
    # the energy is −L ∫ r² (D_ℓ f̃) f dr = L ∫ (r² f̃'f' + L f̃f) dr − L [r² f̃'f],
    # where the Robin conditions turn the wall term into the exterior field energy
    # (it vanishes at a perfect conductor, where f = 0). ∇×b is toroidal with
    # potential −D_ℓ f.
    for ℓ in op.ll_f
        key = (:f, ℓ); L = T(ℓ * (ℓ + 1)); c = weight(ℓ) * L; R = padded(basis[key])
        d1 = _chebyshev_derivative(R, scale)
        f = V * R; df = V * d1
        r2Df = r .^ 2 .* (V * _chebyshev_derivative(d1, scale)) .+ 2 .* r .* df .- L .* f
        wall = -ro^2 .* (at_wall(:outer, d1)' * at_wall(:outer, R)) .+
                ri^2 .* (at_wall(:inner, d1)' * at_wall(:inner, R))
        addB!(key, hermitian(c .* (df' * ((w .* r .^ 2) .* df) .+ L .* (f' * (w .* f)) .+ wall)))
        addA!(key, key, -p.Em * c .* (r2Df' * ((w ./ r .^ 2) .* r2Df)))
    end
    # For b = ∇×(g Y r), the energy is L ∫ r² g̃ g dr and ∇×b is poloidal with potential g.
    for ℓ in op.ll_g
        key = (:g, ℓ); L = T(ℓ * (ℓ + 1)); R = padded(basis[key])
        g = V * R
        drg = g .+ r .* (V * _chebyshev_derivative(R, scale))
        addB!(key, weight(ℓ) * L .* (g' * ((w .* r .^ 2) .* g)))
        addA!(key, key, -p.Em * weight(ℓ) * L .* (L .* (g' * (w .* g)) .+ drg' * (w .* drg)))
    end

    # ---- Temperature: Petrov–Galerkin, as in assemble_mhd_galerkin -------------
    bt(pw, d, q) = banded_radial_term(T, pw, d, q, N, ri, ro)
    gap = ro - ri; thermal = p.E / p.Pr; Mh = size(Rh, 2)
    for ℓ in op.ll_h
        key = (:h, ℓ); L = T(ℓ * (ℓ + 1)); u = (:u, ℓ)
        if p.heating === :differential
            addB!(key, galerkin_block(bt(3, 0, 2), Rh, Mh))
            haskey(idx, u) && addA!(key, u, galerkin_block((L * ri / gap) * bt(0, 0, 2), Ru, Mh))
            addA!(key, key, galerkin_block(thermal * (-L * bt(1, 0, 2) + 2 * bt(2, 1, 2) + bt(3, 2, 2)), Rh, Mh))
        else
            addB!(key, galerkin_block(bt(2, 0, 2), Rh, Mh))
            haskey(idx, u) && addA!(key, u, galerkin_block(L * bt(2, 0, 2), Ru, Mh))
            addA!(key, key, galerkin_block(thermal * (-L * bt(0, 0, 2) + 2 * bt(1, 1, 2) + bt(2, 2, 2)), Rh, Mh))
        end
    end

    A = sparse(Ai, Aj, Av, n, n)
    B = sparse(Bi, Bj, Bv, n, n)

    # Measure the field as Le·b, so that the energy is the unweighted sum of the
    # velocity and magnetic Gram forms and the couplings are exact negative
    # adjoints. Then scale each velocity and magnetic basis function to unit energy.
    # Neither change affects the eigenvalues, but together they keep the pencil
    # well conditioned at high N. Without a coupling, b keeps its own scale so the
    # induced field stays in the eigenvectors.
    field_scale = iszero(p.Le) ? one(T) : p.Le
    s = ones(T, n); d = ones(T, n)
    Bdiag = real.(diag(B))
    for ((field, _), rng) in idx
        field in (:f, :g) && (s[rng] .= field_scale)
        field === :h || (d[rng] .= inv.(sqrt.(Bdiag[rng])))
    end
    A = Diagonal(d .* s) * A * Diagonal(d ./ s)
    B = Diagonal(d) * B * Diagonal(d)
    for (key, rng) in idx
        basis[key] = basis[key] * Diagonal(d[rng] ./ s[rng])
    end
    layout = (index_map = idx, R = basis, fields = (:u, :v, :f, :g, :h), nred = n)
    return A, B, layout
end
