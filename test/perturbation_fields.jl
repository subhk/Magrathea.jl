using Test
using Magrathea

@testset "perturbation field reconstruction" begin
    @testset "generic functions exist and are exported" begin
        @test isdefined(Magrathea, :perturbation_velocity)
        @test isdefined(Magrathea, :perturbation_temperature)
        @test isdefined(Magrathea, :perturbation_magnetic)
        @test Magrathea.perturbation_velocity isa Function
        @test Magrathea.perturbation_temperature isa Function
        @test Magrathea.perturbation_magnetic isa Function
    end

    @testset "hydro perturbation_velocity delegates to eigenvector_to_velocity" begin
        params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
        op = Magrathea.LinearStabilityOperator(params)
        evec = randn(ComplexF64, op.total_dof)

        ur1, uθ1, uφ1, _ = Magrathea.eigenvector_to_velocity(evec, op)
        ur2, uθ2, uφ2, _ = perturbation_velocity(evec, op)

        @test ur2 == ur1
        @test uθ2 == uθ1
        @test uφ2 == uφ1
    end

    @testset "hydro perturbation_temperature single planted mode" begin
        params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
        op = Magrathea.LinearStabilityOperator(params)
        Nr = params.Nr

        # Plant a single temperature degree (the first in l_sets[:Θ]); zero elsewhere.
        evec = zeros(ComplexF64, op.total_dof)
        offset = (length(op.l_sets[:P]) + length(op.l_sets[:T])) * Nr
        l0 = op.l_sets[:Θ][1]
        block = offset + 1 : offset + Nr
        planted = ComplexF64.(1:Nr)               # arbitrary nonzero radial profile
        evec[block] .= planted

        θfield, r_grid, grid = perturbation_temperature(evec, op)

        @test r_grid === op.r
        @test size(θfield) == (Nr, length(grid.θ))
        # θ(r,θ) = planted(r) * Y_{l0}^m(θ); check proportionality to Ylm at fixed r.
        ylm = grid.Ylm[l0]
        for j in eachindex(grid.θ)
            @test θfield[3, j] ≈ planted[3] * ylm[j] atol=1e-10
        end
    end

    @testset "MHD reconstruct primitives" begin
        params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.3,
                           ricb=0.35, m=2, lmax=6, N=16,
                           B0_type=Magrathea.axial, B0_amplitude=1.0)
        op = Magrathea.MHDStabilityOperator(params)
        ndof = Magrathea._mhd_reconstruction_dof(op)
        @test ndof == (length(op.ll_u)+length(op.ll_v)+length(op.ll_f)+
                       length(op.ll_g)+length(op.ll_h)) * (params.N + 1)

        # full vector passthrough
        full = randn(ComplexF64, ndof)
        @test Magrathea._mhd_full_vector(full, op, nothing) == full

        # interior scatter: zeros on non-interior rows
        interior = collect(1:2:ndof)
        evec_int = randn(ComplexF64, length(interior))
        scattered = Magrathea._mhd_full_vector(evec_int, op, interior)
        @test length(scattered) == ndof
        @test scattered[interior] == evec_int
        @test all(scattered[setdiff(1:ndof, interior)] .== 0)

        # block slice
        idx_map = Magrathea._mhd_index_map(op)
        l0 = op.ll_f[1]
        blk = Magrathea._mhd_field_block(full, idx_map, :f, l0)
        @test blk == full[idx_map[(l0, :f)]]
        @test length(blk) == params.N + 1

        # radial eval: c = [0, 1] -> T_1(x) = x -> linear in r on [ricb, 1]
        rg = range(params.ricb, 1.0, length=5) |> collect
        vals = Magrathea._mhd_radial_eval(ComplexF64[0, 1], params.ricb, rg)
        xs = @. 2 * (rg - params.ricb) / (1 - params.ricb) - 1
        @test real.(vals) ≈ xs atol=1e-12
    end

    @testset "MHD perturbation_temperature single planted mode" begin
        params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.3,
                           ricb=0.35, m=2, lmax=6, N=16,
                           B0_type=Magrathea.axial, B0_amplitude=1.0)
        op = Magrathea.MHDStabilityOperator(params)
        idx_map = Magrathea._mhd_index_map(op)
        ndof = Magrathea._mhd_reconstruction_dof(op)

        # Plant constant Chebyshev mode (coeff[1]=1 ⇒ T_0 ≡ 1) in the first h degree.
        full = zeros(ComplexF64, ndof)
        l0 = op.ll_h[1]
        full[first(idx_map[(l0, :h)])] = complex(1.0)   # set the n=0 coefficient

        θfield, r_grid, grid = perturbation_temperature(full, op)
        @test size(θfield) == (length(r_grid), length(grid.θ))
        # constant radial profile (=1) times Y_{l0}^m(θ): proportional to Ylm along θ.
        ylm = grid.Ylm[l0]
        for j in eachindex(grid.θ)
            @test θfield[2, j] ≈ ylm[j] atol=1e-10
        end
        @test all(isfinite, real.(θfield)) && all(isfinite, imag.(θfield))
    end

    @testset "MHD magnetic/velocity curl: toroidal-only gives zero radial" begin
        params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.3,
                           ricb=0.35, m=2, lmax=6, N=16,
                           B0_type=Magrathea.axial, B0_amplitude=1.0)
        op = Magrathea.MHDStabilityOperator(params)
        idx_map = Magrathea._mhd_index_map(op)
        ndof = Magrathea._mhd_reconstruction_dof(op)

        # Pure toroidal magnetic (only :g populated) -> B_r must vanish.
        full = zeros(ComplexF64, ndof)
        lg = op.ll_g[1]
        full[idx_map[(lg, :g)]] .= ComplexF64.(1:(params.N + 1))
        Br, Bθ, Bφ, r_grid, grid = perturbation_magnetic(full, op)
        @test maximum(abs, Br) < 1e-10
        @test maximum(abs, Bθ) > 0          # tangential parts nonzero

        # Pure poloidal magnetic (only :f) -> B_r nonzero.
        full2 = zeros(ComplexF64, ndof)
        lf = op.ll_f[1]
        full2[idx_map[(lf, :f)]] .= ComplexF64.(1:(params.N + 1))
        Br2, _, _, _, _ = perturbation_magnetic(full2, op)
        @test maximum(abs, Br2) > 0

        # Pure toroidal velocity (only :v) -> u_r must vanish.
        full3 = zeros(ComplexF64, ndof)
        lv = op.ll_v[1]
        full3[idx_map[(lv, :v)]] .= ComplexF64.(1:(params.N + 1))
        ur, _, _, _, _ = perturbation_velocity(full3, op)
        @test maximum(abs, ur) < 1e-10
    end

    @testset "hydro perturbation_magnetic errors clearly" begin
        params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
        op = Magrathea.LinearStabilityOperator(params)
        evec = zeros(ComplexF64, op.total_dof)
        @test_throws ErrorException perturbation_magnetic(evec, op)
    end

    @testset "MHD reconstruction end-to-end shapes" begin
        params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.3,
                           ricb=0.35, m=2, lmax=6, N=16,
                           B0_type=Magrathea.axial, B0_amplitude=1.0)
        op = Magrathea.MHDStabilityOperator(params)
        ndof = Magrathea._mhd_reconstruction_dof(op)
        evec = randn(ComplexF64, ndof)

        ur, uθ, uφ, rg, g = perturbation_velocity(evec, op)
        Br, Bθ, Bφ, _, _  = perturbation_magnetic(evec, op)
        θf, _, _          = perturbation_temperature(evec, op)
        for A in (ur, uθ, uφ, Br, Bθ, Bφ, θf)
            @test size(A) == (length(rg), length(g.θ))
            @test eltype(A) == ComplexF64
            @test all(isfinite, abs.(A))
        end
    end

    @testset "(result, mode) delegation API" begin
        # Test that perturbation_velocity(result, mode) and
        # perturbation_temperature(result, mode) correctly delegate to the
        # (evec, op) methods. Build a synthetic StabilityResult to avoid
        # requiring SLEPc.
        params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
        op = Magrathea.LinearStabilityOperator(params)
        nev = 2
        evecs = randn(ComplexF64, op.total_dof, nev)
        evals = ComplexF64[complex(-0.1, 0.5), complex(-0.2, 0.3)]
        problem = OnsetProblem(params)
        result = StabilityResult(evals, evecs, problem; extra=(operator=op,))

        # perturbation_velocity: delegation check on ur component
        ur1, _, _, _ = perturbation_velocity(result.eigenvectors[:, 1],
                                              result.extra.operator)
        ur2, _, _, _ = perturbation_velocity(result, 1)
        @test ur2 == ur1

        # perturbation_temperature: delegation check on θfield component
        θ1, _, _ = perturbation_temperature(result.eigenvectors[:, 1],
                                             result.extra.operator)
        θ2, _, _ = perturbation_temperature(result, 1)
        @test θ2 == θ1
    end
end
