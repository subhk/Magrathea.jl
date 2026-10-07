using Test
using LinearAlgebra
using Magrathea

@testset "Normalized harmonics remain physical at large order" begin
    for (T, m) in ((Float64, 90), (Float64, 180), (Float32, 20), (Float32, 40))
        lmax = m + 2
        μ, w = Magrathea._gauss_legendre_nodes(lmax + 3)
        g = Magrathea.SHGrid{T}(lmax, m, T.(μ), T.(w), T[0])
        tolerance = T === Float32 ? 3e-5 : 2e-12

        # The analytic sectoral amplitude uses arbitrary-precision factorials,
        # independently of the production normalized recurrence.
        amplitude = T((-1)^m * sqrt(BigFloat(2m + 1) / (4big(π)) *
            factorial(big(2m)) / (big(2)^(2m) * factorial(big(m))^2)))
        y, h, v = Magrathea._coupling_harmonic(g, m, m)
        s = sqrt.(1 .- g.μ.^2)
        expected_y = amplitude .* s.^m
        @test eltype(y) === T
        @test y ≈ expected_y rtol=tolerance
        @test h ≈ m .* g.μ ./ s .* expected_y rtol=tolerance
        @test v ≈ im .* m ./ s .* expected_y rtol=tolerance

        # Orthonormality checks both degree recurrences; the gradient identity
        # also checks the normalized angular derivative ladder coefficients.
        Q = g.Q[m]
        @test 2T(π) .* (Q * Diagonal(g.w) * Q') ≈ Matrix{T}(I, 3, 3) atol=tolerance
        for l in m:lmax
            _, hl, vl = Magrathea._coupling_harmonic(g, l, m)
            gradient_energy = 2T(π) * dot(g.w, abs2.(hl) .+ abs2.(vl))
            @test gradient_energy ≈ l * (l + 1) rtol=tolerance
        end

        # Exercise the public reconstruction, including its default grid at
        # the original failure orders and raw-P-overflow orders beyond them.
        op = LinearStabilityOperator(OnsetParams(E=T(1e-4), Pr=one(T), Ra=T(1e8),
            χ=T(.35), m=m, lmax=m, Nr=8))
        evec = zeros(Complex{T}, op.total_dof)
        evec[op.index_map[(m, :P)]] .= one(T)
        ur, uθ, uφ, grid = Magrathea.eigenvector_to_velocity(evec, op)
        y_expected = amplitude .* (sqrt.(1 .- grid.cosθ.^2)).^m
        expected_ur = (T(m * (m + 1)) ./ (sqrt(T(2m + 1)) .* op.r)) * transpose(y_expected)
        @test eltype(ur) === Complex{T}
        @test all(isfinite, ur) && all(isfinite, uθ) && all(isfinite, uφ)
        @test ur ≈ expected_ur rtol=tolerance
        @test norm(uθ) > 0 && norm(uφ) > 0
    end
end

@testset "Conduction basic states retain high-order thermal coupling" begin
    for (T, m) in ((Float64, 90), (Float32, 20))
        normalization = only(Magrathea._normalization_table(T, m, m))
        coefficient_factor = Magrathea._sh_nf_to_orth_factor(m, m, T)
        @test 0 < normalization < 1
        @test isfinite(coefficient_factor)
        @test normalization * coefficient_factor ≈ sqrt(T(2m + 1) / (4T(π)))
        kwargs = (; E=T(1e-4), Pr=one(T), Ra=T(1e8), χ=T(.35), m=m, lmax=m, Nr=8)
        bs, _ = create_conduction_basic_state(T(.35), 8; lmax_bs=0)
        onset = LinearStabilityOperator(OnsetParams(; kwargs...))
        biglobal = LinearStabilityOperator(OnsetParams(; kwargs..., basic_state=bs))
        A, _, _, _ = assemble_matrices(onset)
        A_bs, _, _, _ = assemble_matrices(biglobal)
        rows, cols = onset.index_map[(m, :Θ)], onset.index_map[(m, :P)]
        @test eltype(A_bs) === Complex{T}
        @test A_bs[rows, cols] ≈ A[rows, cols] rtol=(T === Float32 ? 2e-5 : 2e-12)
    end
end

@testset "Float32 boundary constraints retain their independent rows" begin
    T = Float32
    op = LinearStabilityOperator(OnsetParams(E=T(1e-4), Pr=one(T), Ra=T(1e6),
        χ=T(.35), m=1, lmax=2, Nr=64, mechanical_bc=:stress_free,
        thermal_bc=(:fixed_temperature, :fixed_flux)))
    A, B, interior, boundary = assemble_matrices(op)
    _, _, reduction = Magrathea._constrained_reduced_matrices(A, B, op, interior, boundary)
    direct = Magrathea._constraint_reduction_from_subblocks(op)
    @test direct.n_reduced == reduction.n_reduced == op.total_dof - length(boundary)
    for (block, direct_block) in zip(reduction.blocks, direct.blocks)
        @test block.basis ≈ direct_block.basis
        rows = intersect(boundary, block.full_indices)
        constraints = A[rows, block.full_indices]
        constraints ./= maximum(abs, constraints; dims=2)
        @test norm(constraints * block.basis) < 5e-6
    end

    # A real eigensolve used to throw "rank 2; expected rank 4" at this size.
    solve_op = LinearStabilityOperator(OnsetParams(E=T(1e-4), Pr=one(T), Ra=T(1e6),
        χ=T(.35), m=2, lmax=2, Nr=64, mechanical_bc=:stress_free))
    vals, vecs, _ = solve_eigenvalue_problem(solve_op; backend=:dense, nev=2)
    @test length(vals) == size(vecs, 2) == 2
    @test all(isfinite, vals) && all(isfinite, vecs)
    @test eltype(vecs) === ComplexF32
    for field in (:P, :T, :Θ)
        constraints = Magrathea._constraint_subblock(solve_op, 2, field)
        constraints ./= maximum(abs, constraints; dims=2)
        @test norm(constraints * vecs[solve_op.index_map[(2, field)], :]) < 5e-6
    end
end
