using Test
using LinearAlgebra
using Magrathea

@testset "Triglobal Rayleigh bracket keywords" begin
    bracket = Magrathea._triglobal_Ra_bracket
    @test bracket(nothing, nothing, nothing, nothing) == (1e5, 1e8)
    @test bracket(1e4, nothing, nothing, nothing) == (1e3, 1e5)
    @test bracket(1e4, (2e3, 3e4), nothing, nothing) == (2e3, 3e4)
    @test bracket(nothing, nothing, 2e3, nothing) == (2e3, 1e8)
    @test bracket(nothing, nothing, 2e3, 3e4) == (2e3, 3e4)
    @test_throws ArgumentError bracket(nothing, (1e3, 1e4), 1e3, nothing)
    @test_throws ArgumentError bracket(nothing, (1e4, 1e3), nothing, nothing)
    @test_throws ArgumentError bracket(-1.0, nothing, nothing, nothing)
end

@testset "find_critical_Ra returns (Ra_c, ω_c, eigenvector) for triglobal problems" begin
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=1, lmax=5, Nr=10)
    bs3d = basic_state(p; mode=:nonaxisymmetric)
    problem = TriglobalProblem(p, bs3d, 0:2)

    # Ra_guess is accepted as for onset and biglobal problems.
    Ra_c, ω_c, vec_c = find_critical_Ra(problem; Ra_guess=5e3, tol=1e-2, backend=:dense)
    Ra_ref, σ_ref, ω_ref = find_critical_rayleigh_triglobal(p.E, p.Pr, p.χ, 0:2,
        p.lmax, p.Nr, bs3d; Ra_min=5e2, Ra_max=5e4, tol=1e-2, backend=:dense,
        verbose=false)
    @test Ra_c == Ra_ref
    @test ω_c == ω_ref
    @test abs(σ_ref) < abs(ω_ref)

    # The third value is the leading eigenvector at Ra_c.
    at_Ra_c = OnsetParams(E=p.E, Pr=p.Pr, Ra=Ra_c, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr)
    result = solve(TriglobalProblem(at_Ra_c, bs3d, 0:2); nev=1, backend=:dense,
                   verbose=false)
    leading = result.eigenvectors[:, 1]
    @test vec_c isa Vector{ComplexF64}
    @test frequency(result) ≈ ω_c
    @test abs(dot(leading, vec_c)) ≈ norm(leading) * norm(vec_c) rtol=1e-8
end

@testset "find_critical_Ra accepts integer guesses" begin
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=1, lmax=5, Nr=10)
    from_int = find_critical_Ra(OnsetProblem(p); Ra_guess=5_000, backend=:dense)
    from_float = find_critical_Ra(OnsetProblem(p); Ra_guess=5e3, backend=:dense)
    @test from_int[1] == from_float[1]
    @test from_int[2] == from_float[2]
end

@testset "find_global_critical_onset forwards solver keywords" begin
    # Extra keywords such as `backend` reach each per-m search. The sweep reuses the
    # previous Ra_c as the next guess, so compare converged values.
    m_c, Ra_c, ω_c, results = find_global_critical_onset(; E=1e-2, Pr=1.0, χ=0.35,
        lmax=5, Nr=10, m_range=1:2, Ra_guess=5e3, tol=1e-9, backend=:dense, verbose=false)
    @test sort!(collect(keys(results))) == [1, 2]
    for m in 1:2
        Ra_m, ω_m, _ = find_critical_Ra_onset(; E=1e-2, Pr=1.0, χ=0.35, m=m, lmax=max(5, m + 10),
            Nr=10, Ra_guess=5e3, tol=1e-9, backend=:dense)
        @test results[m].Ra_c ≈ Ra_m rtol=1e-6
        @test results[m].ω_c ≈ ω_m rtol=1e-6
    end
    @test Ra_c == minimum(r.Ra_c for r in values(results))
    @test results[m_c].Ra_c == Ra_c && results[m_c].ω_c == ω_c
end
