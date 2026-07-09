using Test
using Magrathea

@testset "StabilityResult construction" begin
    eigenvalues = [complex(0.1, 2.0), complex(-0.3, 1.5), complex(0.5, -1.0)]
    eigenvectors = rand(ComplexF64, 10, 3)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)

    result = StabilityResult(eigenvalues, eigenvectors, problem)

    @test result.eigenvalues == eigenvalues
    @test result.eigenvectors == eigenvectors
    @test result.growth_rate == 0.5
    @test result.frequency == -1.0
    @test result.problem === problem
    @test result.extra == (;)
end

@testset "StabilityResult with extra data" begin
    eigenvalues = [complex(0.1, 2.0)]
    eigenvectors = rand(ComplexF64, 10, 1)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    extra = (critical_Ra=1500.0, critical_m=4)

    result = StabilityResult(eigenvalues, eigenvectors, problem; extra=extra)

    @test result.extra.critical_Ra == 1500.0
    @test result.extra.critical_m == 4
end

@testset "StabilityResult convenience accessors" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0), complex(-0.2, 0.3)]
    eigenvectors = hcat([1.0+0im, 0, 0], [0, 1.0+0im, 0], [0, 0, 1.0+0im])
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)

    result = StabilityResult(eigenvalues, eigenvectors, problem)

    @test growth_rate(result) == 0.5
    @test frequency(result) == -1.0
    @test leading_mode(result) == eigenvectors[:, 2]
end

@testset "Problem type construction" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)

    onset = OnsetProblem(params)
    @test onset.params === params
end
