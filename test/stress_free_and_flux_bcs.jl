module StressFreeAndFluxBCs
using Test, Magrathea, LinearAlgebra, Logging
const M = Magrathea

# Finite spectrum of the constrained hydrodynamic pencil, most unstable first.
function onset_spectrum(p)
    op = LinearStabilityOperator(p)
    A, B, idofs, bdofs = assemble_matrices(op)
    Ar, Br, _ = M._constrained_reduced_matrices(A, B, op, idofs, bdofs)
    λ = eigvals(Ar, Br)
    sort(λ[isfinite.(λ)], by=x -> -real(x))
end
leading(λ) = λ[argmax(real.(λ))]
quiet(f) = with_logger(f, NullLogger())

@testset "Stress-free walls: rigid rotation is not an eigenmode" begin
    # Stress-free walls conserve angular momentum, leaving an exactly neutral rigid
    # rotation (λ = 0 for m = 0, λ = i for m = 1). It used to win the "most unstable"
    # selection below onset, so critical-Ra searches returned their bracket ends.
    base = (E=1e-3, Pr=1.0, Ra=10.0, χ=0.35, lmax=10, Nr=16, mechanical_bc=:stress_free)
    for m in (0, 1), sym in (:both, :symmetric, :antisymmetric)
        λ = onset_spectrum(OnsetParams(; base..., m=m, equatorial_symmetry=sym))
        @test minimum(abs.(λ .- im * m)) > 1e-3
        @test real(λ[1]) < -1e-3
    end
    # The constraint removes only that mode: the ℓ = 1 toroidal potential of every
    # other mode already has zero angular momentum ∫ r³ T dr.
    p = OnsetParams(; base..., m=0, Ra=3e4)
    op = LinearStabilityOperator(p)
    @test M._angular_momentum_gauge(op)
    @test !M._angular_momentum_gauge(LinearStabilityOperator(OnsetParams(; base..., m=2)))
    @test !M._angular_momentum_gauge(LinearStabilityOperator(
        OnsetParams(; base..., m=0, mechanical_bc=:no_slip)))
    r = solve(OnsetProblem(p); nev=3, backend=:dense)
    T1 = op.index_map[(1, :T)]
    w = M._angular_momentum_weights(op.r)
    for j in axes(r.eigenvectors, 2)
        v = r.eigenvectors[:, j]
        @test abs(dot(w, v[T1])) <= 1e-10 * norm(v)
    end
end

@testset "Stress-free critical Rayleigh numbers at m = 0 and m = 1" begin
    for m in (0, 1)
        Ra_c, _, _ = quiet(() -> find_critical_Ra_onset(E=1e-3, Pr=1.0, χ=0.35, m=m,
            lmax=10, Nr=16, Ra_guess=3e4, mechanical_bc=:stress_free, backend=:dense))
        @test 3e3 < Ra_c < 3e5
        @test !(Ra_c ≈ 3e3) && !(Ra_c ≈ 3e5)
        λ = onset_spectrum(OnsetParams(E=1e-3, Pr=1.0, Ra=Ra_c, χ=0.35, m=m, lmax=10,
                                       Nr=16, mechanical_bc=:stress_free))
        @test abs(real(λ[1])) < 1e-5
    end
end

@testset "Triglobal blocks carry the angular-momentum gauge" begin
    χ = 0.35; Nr = 16
    p = OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=χ, m=0, lmax=10, Nr=Nr,
                    mechanical_bc=:stress_free)
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    bs3 = nonaxisymmetric_basic_state(cd, χ, 1e-3, 3e4, 1.0, 4, 2,
                                      Dict{Tuple{Int,Int},Float64}())
    tg = solve(TriglobalProblem(p, bs3, -1:1); nev=4, backend=:dense, verbose=false)
    onset = [leading(onset_spectrum(OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=χ, m=m,
                                                lmax=10, Nr=Nr, mechanical_bc=:stress_free)))
             for m in (0, 1)]
    # Without flow the m blocks decouple; m = -1 is the conjugate of m = 1.
    @test real(leading(tg.eigenvalues)) ≈ maximum(real, onset) rtol=1e-8
end

@testset "MHD stress-free rigid rotation (tau and Galerkin)" begin
    for (m, Le, B0) in ((0, 0.0, no_field), (1, 0.0, no_field), (0, 1e-2, axial))
        p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=10.0, Le=Le, ricb=0.35, m=m, lmax=6,
                      N=16, symm=0, B0_type=B0, bci=0, bco=0)
        op = MHDStabilityOperator(p)
        @test M._mhd_angular_momentum_gauge(op)
        A, B, _, _ = quiet(() -> assemble_mhd_matrices(op))
        λtau = eigvals(Matrix(A), Matrix(B)); λtau = λtau[isfinite.(λtau)]
        Ag, Bg, _ = M.assemble_mhd_galerkin(op)
        λgal = eigvals(Ag, Bg)
        for λ in (λtau, λgal)
            @test minimum(abs.(λ .- im * m)) > 1e-3
            @test real(leading(λ[abs.(λ) .< 1e6])) < -1e-3
        end
        r = quiet(() -> solve(MHDProblem(p); nev=2, backend=:dense))
        @test real(r.eigenvalues[1]) < -1e-3
    end
    # A conducting wall or Lorentz-coupled tilt can exchange angular momentum, so
    # no gauge is imposed there.
    cond = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=10.0, Le=1e-2, ricb=0.35, m=0, lmax=6,
                     N=16, symm=0, B0_type=axial, bci=0, bco=0, bco_magnetic=2)
    @test !M._mhd_angular_momentum_gauge(MHDStabilityOperator(cond))
    tilt = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=10.0, Le=1e-2, ricb=0.35, m=1, lmax=6,
                     N=16, symm=0, B0_type=axial, bci=0, bco=0)
    @test !M._mhd_angular_momentum_gauge(MHDStabilityOperator(tilt))
end

@testset "MHD without field matches onset with buoyancy" begin
    for m in (0, 1, 3)
        mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=3e4, Le=0.0, ricb=0.35, m=m, lmax=10,
                       N=24, symm=0, B0_type=no_field, bci=0, bco=0)
        mhd = quiet(() -> solve(MHDProblem(mp); nev=3, backend=:dense)).eigenvalues
        on = onset_spectrum(OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=0.35, m=m, lmax=10,
                                        Nr=26, mechanical_bc=:stress_free))
        @test leading(mhd) ≈ on[1] rtol=1e-6
    end
end

@testset "Per-wall thermal conditions" begin
    base = (E=1e-3, Pr=1.0, Ra=3e4, χ=0.35, lmax=10, Nr=16)
    mixed = (:fixed_temperature, :fixed_flux)
    @test M._thermal_walls(:fixed_flux) == (:fixed_flux, :fixed_flux)
    @test M._thermal_walls(mixed) == mixed
    @test_throws ArgumentError OnsetParams(; base..., m=2, thermal_bc=(:fixed_flux, :bogus))
    @test_throws ArgumentError OnsetParams(; base..., m=2, thermal_bc=:bogus)

    # Each wall gets its own tau row.
    op = LinearStabilityOperator(OnsetParams(; base..., m=2, thermal_bc=mixed))
    rows = M._constraint_subblock(op, 2, :Θ)
    @test rows[1, :] == [1; zeros(15)]
    @test rows[2, :] ≈ op.cd.D1[end, :]

    # Mixed walls change the spectrum relative to both uniform choices.
    λ(bc) = onset_spectrum(OnsetParams(; base..., m=3, thermal_bc=bc))[1]
    @test !(λ(mixed) ≈ λ(:fixed_temperature)) && !(λ(mixed) ≈ λ(:fixed_flux))
end

@testset "Fixed-flux basic states match their perturbation conditions" begin
    χ = 0.35; Nr = 16
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    # Default flux is the conduction heat flux, not an isothermal shell.
    flux = conduction_basic_state(cd, χ, 2; thermal_bc=:fixed_flux)
    temp = conduction_basic_state(cd, χ, 2)
    @test flux.theta_coeffs[0] ≈ temp.theta_coeffs[0] atol=1e-12
    mer = meridional_basic_state(cd, χ, 1e-3, 3e4, 1.0, 2, 0.0; thermal_bc=:fixed_flux)
    @test mer.theta_coeffs[0] ≈ temp.theta_coeffs[0] atol=1e-12
    nax = nonaxisymmetric_basic_state(cd, χ, 1e-3, 3e4, 1.0, 2, 0,
                                      Dict{Tuple{Int,Int},Float64}(); thermal_bc=:fixed_flux)
    @test nax.theta_coeffs[(0, 0)] ≈ temp.theta_coeffs[0] atol=1e-12

    # The conduction flux state with a flux outer wall reproduces onset.
    for mbc in (:no_slip, :stress_free), m in (0, 3)
        p = OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=χ, m=m, lmax=10, Nr=Nr,
                        mechanical_bc=mbc, thermal_bc=(:fixed_temperature, :fixed_flux))
        onset = leading(solve(OnsetProblem(p); nev=3, backend=:dense).eigenvalues)
        bs = Magrathea.basic_state(p; mode=:conduction)
        biglobal = leading(solve(BiglobalProblem(p, bs); nev=3, backend=:dense).eigenvalues)
        @test biglobal ≈ onset rtol=1e-8 atol=1e-10
    end

    # Basic states fix the inner temperature; a flux inner wall is rejected.
    p = OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=χ, m=2, lmax=10, Nr=Nr, thermal_bc=:fixed_flux)
    @test_throws ArgumentError BiglobalProblem(p, temp)
    @test_throws ArgumentError Magrathea.basic_state(p; mode=:conduction)
    @test_throws ArgumentError OnsetParams(E=1e-3, Pr=1.0, Ra=3e4, χ=χ, m=2, lmax=10,
                                           Nr=Nr, thermal_bc=:fixed_flux, basic_state=temp)
end

end # module
