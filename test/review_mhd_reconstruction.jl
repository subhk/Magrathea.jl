module MHDReconstructionReviewTests
using Test, Magrathea, Logging
const M = Magrathea

@testset "MHD reconstruction evaluates spectral derivatives on coarse grids" begin
    p = MHDParams(E=1e-3, Ra=100.0, Le=0.1, ricb=0.35, m=1, lmax=2,
                  N=8, symm=0, B0_type=axial)
    op = with_logger(() -> MHDStabilityOperator(p), NullLogger())
    imap = M._mhd_index_map(op)
    # P(x) = (1-x²)² T₄(x), a degree-eight polynomial with P=P'=0
    # at both walls. These coefficients follow directly from T_j*T_k.
    amplitude = 1 + 2im
    coeffs = amplitude .* [1/16, 0, -1/4, 0, 3/8, 0, -1/4, 0, 1/16]
    for (section, reconstruct) in ((:u, perturbation_velocity), (:f, perturbation_magnetic))
        full = zeros(ComplexF64, op.matrix_size)
        full[imap[(1, section)]] = coeffs
        for Nr in (2, 5, 9, 13)
            Fr, Fθ, Fφ, r, g = reconstruct(full, op; Nr=Nr)
            x = @. 2 * (r - p.ricb) / (1 - p.ricb) - 1
            t4 = @. 8x^4 - 8x^2 + 1
            P = @. amplitude * (1 - x^2)^2 * t4
            dP = @. amplitude * 2 / (1 - p.ricb) *
                (-4x * (1 - x^2) * t4 + (1 - x^2)^2 * (32x^3 - 16x))
            # Y₁¹/sqrt(3) = -sin(θ)/sqrt(8π), so both tangential
            # angular factors and the radial derivative have analytic values.
            expected_r = (-2 .* P ./ r) * transpose(g.sinθ) ./ sqrt(8π)
            expected_θ = -(dP .+ P ./ r) * transpose(g.cosθ) ./ sqrt(8π)
            expected_φ = (-im .* (dP .+ P ./ r)) * ones(1, length(g.θ)) ./ sqrt(8π)
            @test Fr ≈ expected_r atol=1e-12
            @test Fθ ≈ expected_θ atol=1e-12
            @test Fφ ≈ expected_φ atol=1e-12
            @test maximum(abs, Fθ[[1, end], :]) < 1e-12
            @test maximum(abs, Fφ[[1, end], :]) < 1e-12
        end
    end
end

@testset "MHD reconstruction supports toroidal-only magnetic truncations" begin
    p = MHDParams(E=1e-3, Ra=100.0, Le=0.1, ricb=0.35, m=1, lmax=1,
                  N=8, symm=1, B0_type=axial)
    op = with_logger(() -> MHDStabilityOperator(p), NullLogger())
    @test isempty(op.ll_f)
    @test op.ll_g == [1]
    full = zeros(ComplexF64, op.matrix_size)
    # G(x)=1-x² vanishes at both insulating walls.
    full[M._mhd_index_map(op)[(1, :g)]] = [0.5, 0, -0.5, zeros(6)...]
    Br, Bθ, Bφ, r, g = perturbation_magnetic(full, op; Nr=5)
    x = @. 2 * (r - p.ricb) / (1 - p.ricb) - 1
    G = @. 1 - x^2
    @test iszero(Br)
    @test Bθ ≈ (-im .* G) * ones(1, length(g.θ)) ./ sqrt(8π) atol=1e-12
    @test Bφ ≈ G * transpose(g.cosθ) ./ sqrt(8π) atol=1e-12
    @test maximum(abs, Bθ) > 0.1

    hydro = MHDParams(E=1e-3, Ra=100.0, ricb=0.35, m=1, lmax=1, N=8)
    oh = with_logger(() -> MHDStabilityOperator(hydro), NullLogger())
    @test_throws ErrorException perturbation_magnetic(zeros(ComplexF64, oh.matrix_size), oh)
end
end
