# Tests for the real-orthonormal SH transform + vector-harmonic horizontal
# divergence (src/BasicStates/sh_transform.jl). These pin the machinery that a
# future correct nonaxisymmetric ū·∇T̄ = ∇·(ūT̄) will use, so it can't regress.

using Test
using Magrathea
using LinearAlgebra
import Random

@testset "Real-orthonormal SH transform (cos+sin, ±m)" begin
    Random.seed!(11)
    g = Magrathea.sh_grid(8, 4, Float64)

    @testset "synthesis ↔ analysis round-trip" begin
        c0 = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in abs(m):8)
        c1 = Magrathea.sh_analyze(Magrathea.sh_synthesize(c0, g), g)
        err = maximum(abs(c1[k] - c0[k]) for k in keys(c0))
        @test err < 1e-12
    end

    @testset "horizontal divergence: ∇_h·∇_hψ = -ℓ(ℓ+1)ψ" begin
        ψ = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in max(1, abs(m)):6)
        Vθ = Magrathea.sh_synthesize(ψ, g; Yf=Magrathea._sh_dYθ)            # ∂θψ
        Vφ = Magrathea.sh_synthesize(ψ, g; Yf=Magrathea._sh_dYφ_over_sin)   # (1/sinθ)∂φψ
        div = Magrathea.sh_horizontal_divergence(Vθ, Vφ, g)
        err = maximum(abs(div[(ℓ, m)] - (-ℓ * (ℓ + 1) * get(ψ, (ℓ, m), 0.0)))
                      for (ℓ, m) in keys(div) if ℓ <= 6)
        @test err < 1e-11
    end

    @testset "toroidal field is divergence-free" begin
        χ = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
              for m in -4:4 for ℓ in max(1, abs(m)):6)
        # u_h = r̂×∇_hχ  ⇒  (Vθ, Vφ) = (-(1/sinθ)∂φχ, ∂θχ)
        Vθ = -Magrathea.sh_synthesize(χ, g; Yf=Magrathea._sh_dYφ_over_sin)
        Vφ =  Magrathea.sh_synthesize(χ, g; Yf=Magrathea._sh_dYθ)
        div = Magrathea.sh_horizontal_divergence(Vθ, Vφ, g)
        @test maximum(abs, values(div)) < 1e-12
    end

    @testset "vecsh_advection: radial incompressible MMS" begin
        # u_r = 1/r² (mode (0,0)) ⇒ ∇·ū = 0 (incompressible).  T̄ = r·Y_{10}.
        # u·∇T̄ = u_r ∂_r T̄ = (1/r²)·1·Y_{10} ⇒ forcing[(1,0)] = N_00/r²,  N_00 = 1/(2√π).
        r = collect(range(0.35, 1.0, length=24))
        N00 = 1 / (2 * sqrt(π))
        theta      = Dict{Tuple{Int,Int},Vector{Float64}}((1, 0) => r)
        dtheta_dr  = Dict{Tuple{Int,Int},Vector{Float64}}((1, 0) => ones(length(r)))
        ur         = Dict{Tuple{Int,Int},Vector{Float64}}((0, 0) => 1.0 ./ r .^ 2)
        dur_dr     = Dict{Tuple{Int,Int},Vector{Float64}}((0, 0) => -2.0 ./ r .^ 3)
        utheta     = Dict{Tuple{Int,Int},Vector{Float64}}()
        uphi       = Dict{Tuple{Int,Int},Vector{Float64}}()
        F = Magrathea.vecsh_advection(theta, dtheta_dr, ur, dur_dr, utheta, uphi, 2, 0, r)
        expected = N00 ./ r .^ 2
        @test maximum(abs.(F[(1, 0)] .- expected)) < 1e-10
        # purely radial incompressible flow ⇒ no spurious other-mode forcing
        @test maximum(abs.(F[(0, 0)])) < 1e-10
        @test maximum(abs.(F[(2, 0)])) < 1e-10
    end

    @testset "vecsh_advection smoke (±m runs, finite)" begin
        r = collect(range(0.35, 1.0, length=12))
        mk() = Dict{Tuple{Int,Int},Vector{Float64}}((ℓ, m) => randn(length(r))
                for m in -2:2 for ℓ in abs(m):3)
        F = Magrathea.vecsh_advection(mk(), mk(), mk(), mk(), mk(), mk(), 3, 2, r)
        @test all(all(isfinite, v) for v in values(F))
        @test haskey(F, (3, -2)) && length(F[(3, 0)]) == length(r)
    end

    @testset "compute_full_advection_spectral: m≥1 radial MMS (no-factorial path)" begin
        # u_r=1/r² (mode (0,0), incompressible) advecting T̄ = r·Y_{21} (m=1).
        # u·∇T̄ = u_r ∂_r T̄ ⇒ forcing[(2,1)] = N00/r² in the no-factorial storage.
        # Exercises the no-factorial↔orthonormal conversion on an m≥1 mode.
        r = collect(range(0.35, 1.0, length=24))
        N00 = 1 / (2 * sqrt(π))
        theta     = Dict{Tuple{Int,Int},Vector{Float64}}((2, 1) => r)
        dtheta_dr = Dict{Tuple{Int,Int},Vector{Float64}}((2, 1) => ones(length(r)))
        ur        = Dict{Tuple{Int,Int},Vector{Float64}}((0, 0) => 1.0 ./ r .^ 2)
        dur_dr    = Dict{Tuple{Int,Int},Vector{Float64}}((0, 0) => -2.0 ./ r .^ 3)
        utheta    = Dict{Tuple{Int,Int},Vector{Float64}}()
        uphi      = Dict{Tuple{Int,Int},Vector{Float64}}()
        F = Magrathea.compute_full_advection_spectral(theta, dtheta_dr, ur, dur_dr,
                                                  utheta, uphi, 2, 1, r)
        @test maximum(abs.(F[(2, 1)] .- N00 ./ r .^ 2)) < 1e-10
    end

    @testset "Float32 grid constructs and round-trips" begin
        g32 = Magrathea.sh_grid(6, 2, Float32)
        c0 = Dict{Tuple{Int,Int},Float32}((ℓ, m) => randn(Float32)
              for m in -2:2 for ℓ in abs(m):6)
        c1 = Magrathea.sh_analyze(Magrathea.sh_synthesize(c0, g32), g32)
        @test maximum(abs(c1[k] - c0[k]) for k in keys(c0)) < 1f-4
    end
end
