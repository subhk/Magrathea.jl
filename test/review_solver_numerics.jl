module ReviewSolverNumerics

using Test, Magrathea, LinearAlgebra, SparseArrays, Logging

quiet(f) = with_logger(f, NullLogger())

@testset "Dense solver preserves finite modes under equation scaling" begin
    for T in (Float32, Float64)
        # Scaling both sides of an equation does not change its eigenvalue.
        for scale in (T(1), T(1e-14))
            A = sparse(Diagonal(Complex{T}[scale, -1, 2]))
            B = sparse(Diagonal(Complex{T}[scale, 1, 0]))
            vals, vecs, _ = quiet(() -> solve_eigenvalue_problem(
                A, B; nev=3, backend=:dense))
            @test vals ≈ [1, -1]
            @test size(vecs) == (3, 2)
            @test norm(A * vecs - B * vecs * Diagonal(vals)) <= 20eps(T)
        end
        # A small nonzero mass can also describe a genuinely large finite rate.
        A = sparse(Diagonal(Complex{T}[1, -1]))
        B = sparse(Diagonal(Complex{T}[T(1e-14), 1]))
        vals, _, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=2, backend=:dense))
        @test length(vals) == 2
        @test vals[1] ≈ inv(T(1e-14))
    end
end

@testset "Dense solver enforces coupled algebraic constraints" begin
    # x1 + x2 = 0, with one dynamic equation giving λ=2. Row scaling must
    # preserve the finite eigenvalue and the reconstructed boundary condition.
    for scale in (1e-14, 1.0, 1e14)
        A = sparse(ComplexF64[scale scale; 0 2])
        B = sparse(ComplexF64[0 0; 0 1])
        vals, vecs, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=2, backend=:dense))
        @test vals ≈ [2]
        @test abs(sum(vecs[:, 1])) < 1e-14
        @test norm(A * vecs - B * vecs * Diagonal(vals)) / norm(A) < 1e-14
    end
end

@testset "Float32 constrained onset retains its finite spectrum" begin
    quiet() do
        p = OnsetParams(E=.01f0, Pr=1f0, Ra=4000f0, χ=.35f0,
                        m=2, lmax=6, Nr=24)
        op = LinearStabilityOperator(p)
        A, B, interior, boundary = assemble_matrices(op)
        Ar, Br, _ = Magrathea._constrained_reduced_matrices(A, B, op, interior, boundary)
        n = size(Ar, 1)
        @test rank(Matrix{ComplexF64}(Br)) == n
        vals, vecs, _ = Magrathea._dense_generalized_eigen(Ar, Br; nev=n)
        @test length(vals) == n
        @test eltype(vals) === ComplexF32
        residual = Ar * vecs - Br * vecs * Diagonal(vals)
        @test norm(residual) / (norm(Ar) + maximum(abs, vals) * norm(Br)) < 1e-5
    end
end

@testset "Critical Rayleigh search contracts a wide bracket" begin
    quiet() do
        builder = Ra -> (sparse(reshape(ComplexF64[(Ra / 1e5)^2 - 1], 1, 1)),
                         sparse(reshape(ComplexF64[1], 1, 1)))
        Ra, _, sigma, iterations = find_critical_rayleigh(
            builder, 1e-3, .35, 1; backend=:dense)
        @test Ra ≈ 1e5 rtol=1e-6
        @test abs(real(sigma)) < 1e-6
        @test iterations < 50

        # A steep growth function may satisfy the Rayleigh tolerance before
        # the absolute growth tolerance: the contracted bracket must terminate.
        steep = Ra -> (sparse(reshape(ComplexF64[1e8 * ((Ra / 1e5)^2 - 2)], 1, 1)),
                       sparse(reshape(ComplexF64[1], 1, 1)))
        Ra, _, _, iterations = find_critical_rayleigh(
            steep, 1e-3, .35, 1; backend=:dense, growth_tol=1e-14)
        @test Ra ≈ sqrt(2) * 1e5 rtol=1e-6
        @test iterations < 50

        linear = Ra -> (sparse(reshape(ComplexF64[Ra - 100], 1, 1)),
                        sparse(reshape(ComplexF64[1], 1, 1)))
        Ra, _, sigma, _ = find_critical_rayleigh(linear, 1e-3, .35, 1;
            Ra_min=200.0, Ra_max=1e4, backend=:dense)
        @test Ra == 100
        @test iszero(sigma)
    end
end

end
