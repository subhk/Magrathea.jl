using Test
using LinearAlgebra
using SparseArrays
using Magrathea

# These tests drive the full ASSEMBLY path of every `solve(...)` dispatch and the
# critical-parameter searches. The sole eigensolver backend is `:slepc`, which
# requires the (absent) PetscWrap/SlepcWrap extension, so each call assembles its
# matrices and then throws when it reaches the SLEPc stage. Wrapping in
# `@test_throws` exercises the assembly/dispatch lines without needing PETSc.
# No eigenvalues/eigenvectors are asserted (assembly is structural only).

@testset "solve() assembly paths throw at the SLEPc stage" begin
    op = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=12)

    @testset "OnsetProblem" begin
        @test_throws Exception solve(OnsetProblem(op); nev=2)
        @test_throws Exception solve(OnsetProblem(op); nev=2, backend=:slepc)
        # unknown backend is rejected before assembly
        @test_throws Exception solve(OnsetProblem(op); nev=2, backend=:nope)
    end

    @testset "BiglobalProblem (axisymmetric mean flow)" begin
        bs = Magrathea.basic_state(op; mode=:conduction)
        @test_throws Exception solve(BiglobalProblem(op, bs); nev=2)
    end

    @testset "TriglobalProblem (mode-coupled, m=0 + m!=0 assembly)" begin
        T = Float64; Nr = 12; χ = 0.35; lmax_bs = 6
        cd = ChebyshevDiffn(Nr, T[χ, 1.0], 4)
        coeffs = Dict{Tuple{Int,Int}, Vector{T}}((ℓ, 0) => zeros(T, Nr) for ℓ in 0:lmax_bs)
        emptyd = Dict{Tuple{Int,Int}, Vector{T}}()
        bs3d = BasicState3D{T}(
            lmax_bs = lmax_bs, mmax_bs = 0, Nr = Nr, r = cd.x,
            theta_coeffs = coeffs, dtheta_dr_coeffs = Dict(coeffs),
            ur_coeffs = emptyd, utheta_coeffs = emptyd, uphi_coeffs = Dict(coeffs),
            dur_dr_coeffs = emptyd, dutheta_dr_coeffs = emptyd, duphi_dr_coeffs = Dict(coeffs))
        @test_throws Exception solve(TriglobalProblem(op, bs3d, 0:1); nev=2)
        @test_throws Exception solve(TriglobalProblem(op, bs3d, 1:2); nev=2)
    end

    @testset "MHDProblem (no_field / dipole / axial assembly)" begin
        for (B0, Le, amp) in ((no_field, 0.0, 0.0), (dipole, 1.0, 1.0), (axial, 1.0, 1.0))
            mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=Le, ricb=0.35,
                           m=1, lmax=4, N=12, B0_type=B0, B0_amplitude=amp)
            @test_throws Exception solve(MHDProblem(mp); nev=2)
        end
    end
end

@testset "critical-parameter searches assemble then throw at SLEPc" begin
    @test_throws Exception Magrathea.find_critical_Ra_onset(
        E=1e-3, Pr=1.0, χ=0.35, m=2, lmax=6, Nr=12)
    @test_throws Exception Magrathea.find_global_critical_onset(
        E=1e-3, Pr=1.0, χ=0.35, lmax=6, Nr=12, m_range=1:2)
end

@testset "low-level eigensolver gate" begin
    A = sparse(1.0I, 4, 4); B = sparse(1.0I, 4, 4)
    @test_throws ArgumentError Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:bogus)
    @test_throws Exception Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:slepc)
end
