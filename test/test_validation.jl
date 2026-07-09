using Test
using Magrathea

@testset "Validation - hard errors via OnsetProblem" begin
    # Invalid radius ratio
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.0, m=4, lmax=10, Nr=16))
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=1.0, m=4, lmax=10, Nr=16))

    # Invalid Ekman
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=-1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16))

    # Invalid Prandtl
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=0.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16))

    # Invalid Ra
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=-10.0, χ=0.35, m=4, lmax=10, Nr=16))

    # Invalid Nr (below 8)
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=4))

    # Invalid lmax
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=0, Nr=16))

    # Invalid m
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=-1, lmax=10, Nr=16))
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=15, lmax=10, Nr=16))

    # Invalid BCs
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16, mechanical_bc=:invalid))
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16, thermal_bc=:invalid))
    @test_throws ArgumentError OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16, equatorial_symmetry=:invalid))
end

@testset "Validation - warnings" begin
    # Low Nr
    @test_logs (:warn, r"very low") OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=12))

    # Large E
    @test_logs (:warn, r"unusually large") OnsetProblem(OnsetParams(E=0.5, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16))

    # Small E
    @test_logs (:warn, r"very small") OnsetProblem(OnsetParams(E=1e-9, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16))

    # lmax >> Nr
    @test_logs (:warn, r"Angular resolution") OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=100, Nr=16))

end

@testset "Validation - valid params no warnings" begin
    @test_logs min_level=Logging.Warn OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64))
end
