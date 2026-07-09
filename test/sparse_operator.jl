using Test
using Magrathea

function _expected_matrix_size(params::SparseOnsetParams{T}) where {T}
    ll_top, ll_bot = Magrathea.compute_l_modes(params.m, params.lmax, params.symm)
    n_per_mode = params.N + 1
    return (2 * length(ll_top) + length(ll_bot)) * n_per_mode
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
