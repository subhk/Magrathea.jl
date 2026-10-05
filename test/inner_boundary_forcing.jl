using Test
using LinearAlgebra
using Logging
using Magrathea

const _IB = Magrathea
_ib_quiet(f) = with_logger(f, NullLogger())
# Public amplitude = stored coefficient × normalization (no-factorial convention).
_ib_norm(l, m) = m == 0 ? sqrt((2l + 1) / 4π) : sqrt((2l + 1) / 4π * 2)

@testset "Laplace profiles with inner-wall conditions" begin
    r = collect(range(0.35, 1.0, length=9))
    for l in (0, 2), (inner_bc, outer_bc) in ((:fixed_flux, :fixed_temperature),
                                              (:fixed_temperature, :fixed_flux))
        θ, dθ = _IB.laplace_mode_profile(l, r, 0.35, 1.0, -0.7, 0.4;
                                         inner_bc=inner_bc, outer_bc=outer_bc)
        @test (inner_bc === :fixed_flux ? dθ[1] : θ[1]) ≈ -0.7
        @test (outer_bc === :fixed_flux ? dθ[end] : θ[end]) ≈ 0.4
    end
    # Fixed flux on both walls determines ℓ > 0 but not the mean temperature.
    θ, dθ = _IB.laplace_mode_profile(2, r, 0.35, 1.0, 0.3, -0.2;
                                     inner_bc=:fixed_flux, outer_bc=:fixed_flux)
    @test dθ[1] ≈ 0.3 && dθ[end] ≈ -0.2
    @test_throws ArgumentError _IB.laplace_mode_profile(0, r, 0.35, 1.0, 0.3, -0.2;
        inner_bc=:fixed_flux, outer_bc=:fixed_flux)
    @test_throws ErrorException _IB.laplace_mode_profile(2, r, 0.35, 1.0, 0.3, -0.2;
        inner_bc=:bogus)
end

@testset "Inner temperature and heat-flux patterns" begin
    χ, E, Ra, Pr = 0.35, 1e-2, 3e3, 1.0
    cd = ChebyshevDiffn(24, [χ, 1.0], 4)

    # Temperature: θ̄(r_i) = 1 + a P₂(cosθ), and the outer wall stays at 0.
    bs = basic_state(cd, χ, E, Ra, Pr; inner_temperature_bc=Y20(0.1))
    @test bs isa BasicState && bs.inner_thermal_bc === :fixed_temperature
    for θ in range(0.1, 3.0, length=7)
        @test _IB.evaluate_basic_state(bs, χ, θ)[1] ≈ 1 + 0.1 * (3cos(θ)^2 - 1) / 2 atol=1e-12
        @test abs(_IB.evaluate_basic_state(bs, 1.0, θ)[1]) < 1e-12
    end

    # Flux: the pattern sets ∂θ̄/∂r at r_i; the mean carries the conduction flux.
    bs3 = basic_state(cd, χ, E, Ra, Pr; inner_flux_bc=Y11(0.2))
    @test bs3 isa BasicState3D && bs3.inner_thermal_bc === :fixed_flux
    @test bs3.dtheta_dr_coeffs[(1, 1)][1] * _ib_norm(1, 1) ≈ 0.2
    @test bs3.dtheta_dr_coeffs[(0, 0)][1] * _ib_norm(0, 0) ≈ -1 / (χ * (1 - χ))
    @test abs(bs3.theta_coeffs[(1, 1)][end]) < 1e-12

    # The default inner flux is the conduction flux, so the profiles coincide.
    fixed = conduction_basic_state(cd, χ, 2)
    flux = conduction_basic_state(cd, χ, 2; inner_thermal_bc=:fixed_flux)
    @test flux.theta_coeffs[0] ≈ fixed.theta_coeffs[0]
    @test flux.inner_thermal_bc === :fixed_flux

    @test_throws ArgumentError basic_state(cd, χ, E, Ra, Pr; flux_bc=Y20(0.1),
                                           inner_flux_bc=Y20(0.1))
    @test_throws ErrorException basic_state(cd, χ, E, Ra, Pr; inner_temperature_bc=Y20(0.1),
                                            inner_flux_bc=Y20(0.1))
    @test_throws ArgumentError meridional_basic_state(cd, χ, E, Ra, Pr, 1, 0.0;
                                                      inner_amplitude=0.1)
    @test_throws ArgumentError basic_state(cd, χ, E, Ra, Pr; lmax_bs=2,
                                           inner_temperature_bc=Y30(0.1))
end

@testset "Inner flux reproduces the matching fixed-temperature state" begin
    χ, E, Pr = 0.35, 1e-2, 1.0
    cd = ChebyshevDiffn(20, [χ, 1.0], 4)
    inner_flux_of(bs) = Dict(k => v[1] * _ib_norm(k[1], abs(k[2])) for (k, v) in bs.dtheta_dr_coeffs)

    # Linear constructor, with m ≠ 0 forcing on both walls.
    amps = Dict((2, 1) => 0.07, (3, 2) => -0.04)
    fixed = nonaxisymmetric_basic_state(cd, χ, E, 3e3, Pr, 5, 3, amps;
        inner_amplitudes=Dict((0, 0) => 1.0, (2, 0) => 0.1, (2, 2) => 0.05))
    flux = nonaxisymmetric_basic_state(cd, χ, E, 3e3, Pr, 5, 3, amps;
        inner_thermal_bc=:fixed_flux, inner_amplitudes=inner_flux_of(fixed))
    for k in keys(fixed.theta_coeffs)
        @test flux.theta_coeffs[k] ≈ fixed.theta_coeffs[k] atol=1e-12
        @test flux.uphi_coeffs[k] ≈ fixed.uphi_coeffs[k] atol=1e-12
    end

    # Nonlinear self-consistent solve, axisymmetric and 3D.
    for (mmax, inner) in ((0, Dict((2, 0) => 0.1)), (4, Dict((2, 0) => 0.1, (2, 1) => 0.05)))
        a, ia = nonaxisymmetric_basic_state_selfconsistent(cd, χ, E, 10.0, Pr, 4, mmax,
            Dict((2, 0) => 0.05); inner_amplitudes=inner, tolerance=1e-11)
        b, ib = nonaxisymmetric_basic_state_selfconsistent(cd, χ, E, 10.0, Pr, 4, mmax,
            Dict((2, 0) => 0.05); inner_thermal_bc=:fixed_flux,
            inner_amplitudes=inner_flux_of(a), tolerance=1e-11)
        @test ia.converged && ib.converged
        @test ib.boundary_residual < 1e-12
        @test b.inner_thermal_bc === :fixed_flux
        for (k, v) in a.theta_coeffs
            @test b.theta_coeffs[k] ≈ v atol=1e-10
        end
    end
end

@testset "Perturbations match the basic state's inner wall" begin
    χ, E, Ra, Pr = 0.35, 1e-2, 300.0, 1.0
    cd = ChebyshevDiffn(24, [χ, 1.0], 4)
    flux_state = basic_state(cd, χ, E, Ra, Pr; inner_flux_bc=Y20(0.1))
    fixed_state = basic_state(cd, χ, E, Ra, Pr; inner_temperature_bc=Y20(0.1))
    kw = (; E=E, Pr=Pr, Ra=Ra, χ=χ, m=2, lmax=6, Nr=24)
    @test OnsetParams(; kw..., thermal_bc=(:fixed_flux, :fixed_temperature),
                      basic_state=flux_state) isa OnsetParams
    @test OnsetParams(; kw..., basic_state=fixed_state) isa OnsetParams
    @test_throws ArgumentError OnsetParams(; kw..., basic_state=flux_state)
    @test_throws ArgumentError OnsetParams(; kw..., thermal_bc=(:fixed_flux, :fixed_temperature),
                                           basic_state=fixed_state)

    # The convenience wrapper follows the inner wall of params.thermal_bc.
    p = OnsetParams(; kw..., lmax=8, thermal_bc=(:fixed_flux, :fixed_temperature))
    for mode in (:conduction, :meridional, :nonaxisymmetric, :selfconsistent)
        s = _ib_quiet(() -> basic_state(p; mode=mode, amplitude=0.05, inner_amplitude=0.1))
        @test s.inner_thermal_bc === :fixed_flux
    end
    meridional = basic_state(p; mode=:meridional, inner_amplitude=0.1)
    @test meridional.dtheta_dr_coeffs[2][1] * _ib_norm(2, 0) ≈ 0.1
    @test_throws ArgumentError basic_state(OnsetParams(; kw..., thermal_bc=:fixed_flux))
    r = _ib_quiet(() -> solve(BiglobalProblem(p, meridional); nev=2, backend=:dense))
    @test all(isfinite, r.eigenvalues)
end
