using Test
using Magrathea
using LinearAlgebra
using Logging

function _expected_matrix_size(params::SparseOnsetParams{T}) where {T}
    ll_top, ll_bot = Magrathea.compute_l_modes(params.m, params.lmax, params.symm)
    n_per_mode = params.N + 1
    return (2 * length(ll_top) + length(ll_bot)) * n_per_mode
end

function _sparse_pencil(params::SparseOnsetParams)
    return with_logger(NullLogger()) do
        op = SparseStabilityOperator(params)
        A, B, interior_dofs, _ = assemble_sparse_matrices(op)
        (; op, A, B, interior_dofs)
    end
end

# Dense spectrum of the full tau pencil. B is zero on the tau rows, giving one
# infinite eigenvalue per boundary row; LAPACK returns those as Inf/NaN or with
# |λ| ≳ 1e11, while the physical spectrum here stays at |λ| ≲ 1e2. The cutoff
# separates the two; callers check that it removed exactly `n_tau` eigenvalues,
# so it cannot hide a spurious finite mode.
function _sparse_pencil_spectrum(params::SparseOnsetParams; cutoff=1e6)
    s = _sparse_pencil(params)
    λ = eigen(Matrix(s.A), Matrix(s.B)).values
    kept = sort!(filter(z -> isfinite(z) && abs(z) < cutoff, λ); by=real, rev=true)
    n = size(s.A, 1)
    return (; kept, n, n_tau=n - length(s.interior_dofs), op=s.op)
end

# Leading eigenvalues of the same problem from the MHD solver with no field
# (tau-free ultraspherical-Galerkin path).
function _mhd_no_field_eigs(params::SparseOnsetParams; lmax::Int=params.lmax, nev::Int=3)
    mp = MHDParams(E=params.E, Pr=params.Pr, Pm=1.0, Ra=params.Ra, ricb=params.ricb,
                   m=params.m, lmax=lmax, N=params.N, symm=params.symm,
                   B0_type=no_field, B0_amplitude=0.0, Le=0.0,
                   bci=params.bci, bco=params.bco,
                   bci_thermal=params.bci_thermal, bco_thermal=params.bco_thermal,
                   heating=params.heating)
    return with_logger(NullLogger()) do
        solve(MHDProblem(mp); nev=nev, backend=:dense).eigenvalues
    end
end

@testset "SparseOperator matrix sizing" begin
    cases = [
        SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                          ricb=0.35, m=1, lmax=6,
                          symm=1, N=8),
        SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                          ricb=0.35, m=0, lmax=6,
                          symm=1, N=8),
        SparseOnsetParams(E=1e-4, Pr=1.0, Ra=1e6,
                          ricb=0.35, m=2, lmax=7,
                          symm=-1, N=8),
    ]

    for params in cases
        op = SparseStabilityOperator(params)
        @test op.matrix_size == _expected_matrix_size(params)
    end
end

@testset "SparseOnsetParams Pr default" begin
    # The default used to be `one(T)`, evaluated before T is bound (UndefVarError)
    p = SparseOnsetParams(E=1e-3, Ra=1e5, ricb=0.35, m=1, lmax=4, N=8)
    @test p.Pr === 1.0 && p.Etherm === 1e-3
    @test SparseOnsetParams(E=1f-3, Ra=1f5, ricb=0.35f0, m=1, lmax=4, N=8).Pr === 1f0
    @test SparseOnsetParams(E=1e-3, Pr=0.5, Ra=1e5, ricb=0.35, m=1, lmax=4, N=8).Etherm ≈ 2e-3
end

@testset "sparse_radial_operator matches analytic derivatives" begin
    # The operator acts on Chebyshev coefficients of f(r) and returns the
    # Chebyshev coefficients of r^power * d^deriv f / dr^deriv. Verify against
    # exact polynomial derivatives. This guards the ultraspherical derivative
    # chain (any error there silently corrupts D^2..D^4 operators).
    ri, ro = 0.35, 1.0
    N = 24
    r_of_x(x̂) = ri + (ro - ri) * (x̂ + 1) / 2
    recon(c, x̂) = sum(c[n + 1] * cos(n * acos(x̂)) for n in 0:length(c) - 1)

    cases = [
        # power, deriv, f(r),        exact r^power * d^deriv f
        (0, 1, r -> r^3,            r -> 3r^2),
        (0, 2, r -> r^3,            r -> 6r),
        (0, 2, r -> r^4,            r -> 12r^2),
        (0, 3, r -> r^4,            r -> 24r),
        (0, 4, r -> r^5,            r -> 120r),
        (1, 1, r -> r^3,            r -> r * 3r^2),
        (2, 2, r -> r^4,            r -> r^2 * 12r^2),
        (0, 2, r -> 2r^4 - r^2,     r -> 24r^2 - 2),
    ]

    for (power, deriv, f, exact) in cases
        a = Magrathea.chebyshev_coefficients(Float64, f, N + 1, ri, ro)
        b = Magrathea.sparse_radial_operator(power, deriv, N, ri, ro) * a
        err = maximum(abs(recon(b, x̂) - exact(r_of_x(x̂)))
                      for x̂ in range(-0.9, 0.9, length=15))
        @test err < 1e-9
    end
end

@testset "ultraspherical_derivative Gegenbauer identity" begin
    # d/dx C_n^(λ)(x) = 2λ C_{n-1}^(λ+1)(x): superdiagonal must be constant 2λ
    # for λ > 0, and (n+1) for λ = 0 (Chebyshev T_n -> U_{n-1}).
    N = 8
    D0 = Magrathea.ultraspherical_derivative(Float64, 0, N)
    for n in 0:N-1
        @test D0[n + 1, n + 2] ≈ n + 1
    end
    for λ in (1, 2, 3)
        Dλ = Magrathea.ultraspherical_derivative(Float64, λ, N)
        for n in 0:N-1
            @test Dλ[n + 1, n + 2] ≈ 2λ
        end
    end
end

# -----------------------------------------------------------------------------
# Physics regressions. The sparse pencil is the hydrodynamic part of the MHD tau
# assembly, so for the same l-modes its spectrum must match the MHD no-field
# solve. Before the fix, the v-row Coriolis coupling used the r⁴-weighted
# poloidal operators, and the tau rows overwrote the T₀/T₁ rows of an
# unprojected (C⁰) residual. The "leading" eigenvalue then diverged with N
# (≈ +4.0 at N=20, +7.4 at N=30 at lmax=24, against the physical 0.173+0.050im).
# -----------------------------------------------------------------------------

@testset "Sparse pencil: tau rows replace the highest residual coefficients" begin
    params = SparseOnsetParams(E=1e-3, Pr=1.0, Ra=1e5, ricb=0.35, m=2, lmax=6,
                               symm=1, N=10, bci=0, bco_thermal=1)
    s = _sparse_pencil(params)
    n_per_mode = params.N + 1
    nb_top, nb_bot = length(s.op.ll_top), length(s.op.ll_bot)

    # Blocks are u (ll_top), v (ll_bot), h (ll_top); u carries 4 conditions
    first_rows = [(k - 1) * n_per_mode + 1 for k in 1:(2nb_top + nb_bot)]
    n_bc(k) = k <= nb_top ? 4 : 2
    tau_rows = reduce(vcat, [collect((r + n_per_mode - n_bc(k)):(r + n_per_mode - 1))
                             for (k, r) in enumerate(first_rows)])

    @test setdiff(1:size(s.A, 1), s.interior_dofs) == tau_rows
    @test all(r -> iszero(s.B[r, :]) && !iszero(s.A[r, :]), tau_rows)
    # The T₀ and T₁ residual rows of every block are equations, not BCs
    @test all(r -> !iszero(s.B[r, :]) && !iszero(s.B[r + 1, :]), first_rows)
end

@testset "Sparse pencil matches the MHD no-field spectrum" begin
    base = (E=1e-3, Ra=1e5, ricb=0.35, lmax=12, N=24)   # Pr defaults to 1
    cases = [
        (m=4, symm=1),                                          # no-slip, fixed T, differential
        (m=4, symm=-1),
        (m=4, symm=1, Pr=0.3, bci=0, bco=0),                    # stress-free, Pr ≠ 1
        (m=4, symm=-1, bco=0, bco_thermal=1),                   # mixed walls, outer fixed flux
        (m=4, symm=1, heating=:internal),
        (m=4, symm=-1, Pr=3.0, bci=0, bco=0, bci_thermal=1, heating=:internal),
        (m=0, symm=1),
    ]

    @testset "$(kw)" for kw in cases
        params = SparseOnsetParams(; base..., kw...)
        s = _sparse_pencil_spectrum(params)

        # Kore's l-set for m = 0 is 1:lmax+1, i.e. the MHD l-set at lmax+1
        lmax_mhd = params.m == 0 ? params.lmax + 1 : params.lmax
        @test (s.op.ll_top, s.op.ll_bot) ==
              Magrathea.compute_mhd_l_modes(params.m, lmax_mhd, params.symm, no_field)
        λ_mhd = _mhd_no_field_eigs(params; lmax=lmax_mhd)

        # The cutoff removed exactly the tau-row infinities, nothing else
        @test s.n_tau == 6 * length(s.op.ll_top) + 2 * length(s.op.ll_bot)
        @test length(s.kept) == s.n - s.n_tau
        # Leading mode is the physical one, and no kept mode grows faster
        @test abs(s.kept[1] - λ_mhd[1]) < 1e-6
        @test all(z -> real(z) < real(λ_mhd[1]) + 1e-6, s.kept)
        # The next modes match too
        @test all(w -> minimum(z -> abs(z - w), s.kept) < 1e-6, λ_mhd)
    end
end

@testset "Sparse pencil converges under N refinement" begin
    params(N) = SparseOnsetParams(E=1e-3, Pr=1.0, Ra=1e5, ricb=0.35, m=4, lmax=12,
                                  symm=1, N=N)
    λ = [_sparse_pencil_spectrum(params(N)).kept[1] for N in (16, 20, 24)]

    @test abs(λ[3] - λ[2]) < abs(λ[2] - λ[1]) / 4
    @test abs(λ[3] - λ[2]) < 1e-6
    @test abs(λ[3] - _mhd_no_field_eigs(params(24); nev=1)[1]) < 1e-7
end
