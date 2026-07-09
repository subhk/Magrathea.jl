using Test
using Magrathea

@testset "Legacy banner API is removed" begin
    @test !isdefined(Magrathea, :CROSS_BANNER)
    @test !isdefined(Magrathea, :print_cross_header)
end

@testset "OnsetParams show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=30, Nr=64)
    output = sprint(show, MIME("text/plain"), params)

    @test startswith(output, "OnsetParams{Float64}")
    @test occursin("├── dynamics: E=0.001, Pr=1.0, Ra=100.0", output)
    @test occursin("├── geometry: χ=0.35", output)
    @test occursin("├── resolution: m=4, lmax=30, Nr=64", output)
    @test occursin("├── boundary conditions: mechanical=no_slip, thermal=fixed_temperature", output)
    @test occursin("└── equatorial symmetry: both", output)
end

@testset "StabilityResult show" begin
    eigenvalues = [complex(0.1, 2.0), complex(0.5, -1.0)]
    eigenvectors = rand(ComplexF64, 10, 2)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    result = StabilityResult(eigenvalues, eigenvectors, problem)
    output = sprint(show, MIME("text/plain"), result)

    @test startswith(output, "StabilityResult{Float64} with 2 eigenvalues")
    @test occursin("├── leading eigenvalue: 0.5 - 1.0im", output)
    @test occursin("├── growth rate: 0.5", output)
    @test occursin("├── frequency: -1.0", output)
    @test occursin("└── problem: OnsetProblem", output)
end

@testset "OnsetProblem show" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)
    output = sprint(show, MIME("text/plain"), problem)

    @test startswith(output, "OnsetProblem{Float64}")
    @test occursin("├── parameters: E=0.001, Ra=100.0, Pr=1.0, χ=0.35", output)
    @test occursin("├── resolution: m=4, lmax=10, Nr=16", output)
    @test occursin("└── boundary conditions: mechanical=no_slip, thermal=fixed_temperature", output)
end

@testset "Low-level public displays use tree rows" begin
    cd = ChebyshevDiffn(8, [0.35, 1.0], 4)
    cd_output = sprint(show, cd)
    @test startswith(cd_output, "ChebyshevDiffn{Float64}")
    @test occursin("├── points: 8", cd_output)
    @test occursin("└── matrices: D1, D2, D3, D4", cd_output)

    bc_output = sprint(show, MIME("text/plain"), Y20(0.1))
    @test startswith(bc_output, "SphericalHarmonicBC{Float64}")
    @test occursin("└── Y_2,0: 0.1", bc_output)
end

@testset "estimate_size uses tree summary" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    problem = OnsetProblem(params)

    output = mktemp() do _, io
        redirect_stdout(io) do
            estimate_size(problem)
        end
        flush(io)
        seekstart(io)
        read(io, String)
    end
    @test startswith(output, "OnsetProblem size estimate")
    @test occursin("├── l-modes:", output)
    @test occursin("├── degrees of freedom per mode:", output)
    @test occursin("├── matrix size:", output)
    @test occursin("└── dense storage estimate:", output)
end
