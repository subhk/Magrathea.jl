using Test
using Magrathea

# Rotating spherical-shell onset benchmark of Barik et al. (2023): χ = 0.35, Pr = 1,
# no-slip, fixed-temperature walls, shell-thickness Ekman number Ek_d = 1e-3.
# Published critical values: m_c = 4, R̃a_c = Ra_c Ek_d = 55.9, ω_c = -0.0231.
# The growth rate of both the collocation onset solver and the ultraspherical MHD
# solver (without a field) must change sign inside the published precision of
# R̃a_c. The critical mode is equatorially symmetric, so only that parity is solved.
@testset "Barik et al. (2023) onset benchmark" begin
    χ, Ek_d = 0.35, 1e-3
    E = Ek_d * (1 - χ)^2            # Magrathea's Ekman number is r_o based
    for (Ra_tilde, side) in ((55.85, -1), (55.95, 1))
        Ra = Ra_tilde / Ek_d        # the Rayleigh number is shell-thickness based
        onset = OnsetParams(E=E, Pr=1.0, Ra=Ra, χ=χ, m=4, lmax=20, Nr=24,
                            equatorial_symmetry=:symmetric)
        mhd = MHDParams(E=E, Pr=1.0, Pm=1.0, Ra=Ra, ricb=χ, m=4, lmax=20, N=24, symm=1,
                        B0_type=no_field, B0_amplitude=0.0, Le=0.0)
        for result in (solve(OnsetProblem(onset); nev=2, backend=:dense),
                       solve(MHDProblem(mhd); nev=2, backend=:dense))
            @test side * growth_rate(result) > 1e-5
            @test frequency(result) ≈ -0.0231 atol=5e-5
        end
    end
end
