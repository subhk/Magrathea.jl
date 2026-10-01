using Test
using Magrathea

@testset "Symbolic zero mean flux preserves an insulating outer boundary" begin
    χ = 0.35
    cd = ChebyshevDiffn(16, [χ, 1.0], 4)
    E, Ra, Pr = 0.01, 100.0, 1.0
    normalization = sqrt(4π)

    # Explicit Y00=0 is a zero radial gradient. With the inner temperature
    # fixed at one and no other forcing, the physical temperature is uniform.
    reference = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0,
        Dict((0, 0) => 0.0); thermal_bc=:fixed_flux)
    for flux in (Y00(0.0), Y00(0.0) + Y20(0.0), Y00(1.0) - Y00(1.0))
        bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux, lmax_bs=4)
        @test bs.theta_coeffs[0] ./ normalization ≈ ones(length(cd.x))
        @test all(iszero, bs.dtheta_dr_coeffs[0])
        @test bs.theta_coeffs[0] ≈ reference.theta_coeffs[(0, 0)]
        @test bs.dtheta_dr_coeffs[0] == reference.dtheta_dr_coeffs[(0, 0)]
        @test all(iszero, mean_flow_velocity(bs, 0.6, 1.0))
    end

    nonlinear, info = basic_state_selfconsistent(cd, χ, E, Ra, Pr;
        flux_bc=Y00(0.0), lmax_bs=4)
    @test info.converged
    @test nonlinear.theta_coeffs[0] ≈ reference.theta_coeffs[(0, 0)] atol=1e-11
    @test maximum(abs, nonlinear.dtheta_dr_coeffs[0]) < 1e-10

    # An absent monopole retains the documented default conduction heat flux.
    conduction = conduction_basic_state(cd, χ, 4; thermal_bc=:fixed_flux)
    for flux in (Y20(0.0), SphericalHarmonicBC{Float64}())
        bs = basic_state(cd, χ, E, Ra, Pr; flux_bc=flux, lmax_bs=4)
        @test bs.theta_coeffs[0] ≈ conduction.theta_coeffs[0]
        @test bs.dtheta_dr_coeffs[0] ≈ conduction.dtheta_dr_coeffs[0]
        @test bs.dtheta_dr_coeffs[0][end] / normalization ≈ -χ / (1 - χ)
    end
end
