using Test
using Magrathea

@testset "ChebyshevDiffn polynomial derivatives" begin
    n = 8
    cd = ChebyshevDiffn(n, [-1.0, 1.0], 4)
    x = cd.x

    f = x .^ 4 .- 3.0 .* x .^ 2 .+ 2.0
    df = 4.0 .* x .^ 3 .- 6.0 .* x
    d2f = 12.0 .* x .^ 2 .- 6.0
    d3f = 24.0 .* x
    d4f = fill(24.0, length(x))

    @test issorted(x)
    @test isapprox(cd.D1 * f, df; atol=1e-10, rtol=1e-10)
    @test isapprox(cd.D2 * f, d2f; atol=1e-10, rtol=1e-10)
    @test cd.D3 !== nothing
    @test cd.D4 !== nothing
    @test isapprox(cd.D3 * f, d3f; atol=1e-10, rtol=1e-10)
    @test isapprox(cd.D4 * f, d4f; atol=1e-10, rtol=1e-10)
end

@testset "ChebyshevDiffn scaling on shifted domain" begin
    cd = ChebyshevDiffn(6, [2.0, 5.0], 2)
    x = cd.x

    f = x .^ 2 .+ 2.0 .* x .- 1.0
    df = 2.0 .* x .+ 2.0
    d2f = fill(2.0, length(x))

    @test isapprox(cd.D1 * f, df; atol=1e-10, rtol=1e-10)
    @test isapprox(cd.D2 * f, d2f; atol=1e-10, rtol=1e-10)
end
