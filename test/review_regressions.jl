using Test
using LinearAlgebra
using SparseArrays
using Magrathea

@testset "COO preallocation counts unique owned coordinates" begin
    # Duplicates in both column bands, an explicit zero, and an empty rank.
    rows = [1, 1, 1, 1, 2, 2, 3, 4]
    cols = [1, 1, 3, 3, 2, 4, 1, 4]
    vals = ComplexF64[1, 2, 3, -3, 0, 2, 4, 5]
    A = sparse(rows, cols, vals, 4, 4)
    for (rs, re) in ((0, 2), (2, 4), (0, 4), (4, 4))
        @test Magrathea._owned_coo_nnz(rows, cols, rs, re) ==
              Magrathea._petsc_owned_nnz(A, rs, re)
    end
end

@testset "Growth-rate and critical-Ra eigenvectors on root and workers" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=125.0, χ=0.35, m=2, lmax=4, Nr=16)
    op = LinearStabilityOperator(params)
    expected = ComplexF64.(1:op.total_dof)
    previous = Magrathea._SLEPC_CONSTRAINED_SOLVER[]
    try
        # Model the extension's matrix contract, including unsorted eigenvalues.
        for worker in (false, true)
            Magrathea._SLEPC_CONSTRAINED_SOLVER[] = function (operator; kwargs...)
                vals = ComplexF64[-1e6, operator.params.Ra - 100 + 3im]
                vecs = worker ? Matrix{ComplexF64}(undef, operator.total_dof, 0) :
                               hcat(-expected, expected)
                return vals, vecs, Dict{String,Any}()
            end
            σ, ω, vec = find_growth_rate(op)
            @test σ == 25.0
            @test ω == 3.0
            @test vec isa Vector{ComplexF64}
            @test vec == (worker ? ComplexF64[] : expected)

            Ra, ωc, vc = find_critical_Ra(OnsetProblem(params);
                Ra_guess=50.0, Ra_bracket=(25.0, 200.0), tol=1e-8)
            @test Ra ≈ 100.0 atol=1e-7
            @test ωc == 3.0
            @test vc == (worker ? ComplexF64[] : expected)
        end
    finally
        Magrathea._SLEPC_CONSTRAINED_SOLVER[] = previous
    end
end

@testset "MHD Galerkin preserves each boundary condition" begin
    N = 12
    ri = 0.35
    scale = 2 / (1 - ri)
    # Independent endpoint evaluation of Chebyshev coefficients.
    outer = (value=ones(N + 1), d1=[scale*k^2 for k in 0:N],
             d2=[scale^2*k^2*(k^2-1)/3 for k in 0:N])
    inner = (value=[(-1.0)^k for k in 0:N],
             d1=[scale*(-1.0)^(k+1)*k^2 for k in 0:N],
             d2=[scale^2*(-1.0)^k*k^2*(k^2-1)/3 for k in 0:N])
    for (bci, bco) in ((1, 1), (0, 0), (1, 0), (0, 1)),
        (tci, tco) in ((0, 0), (1, 1), (0, 1), (1, 0))
        p = MHDParams(E=1e-3, Ra=1e3, ricb=ri, m=1, lmax=3, N=N,
                      bci=bci, bco=bco, bci_thermal=tci, bco_thermal=tco)
        op = MHDStabilityOperator(p)
        A, B, layout = Magrathea.assemble_mhd_galerkin(op)
        Ru = layout.R[(:u, first(op.ll_u))]
        Rv = layout.R[(:v, first(op.ll_v))]
        Rh = layout.R[(:h, first(op.ll_h))]
        @test size(Ru) == (N + 1, N - 3)
        @test size(Rv) == size(Rh) == (N + 1, N - 1)
        for (ep, rb, bc, tc) in ((inner, ri, bci, tci), (outer, 1.0, bco, tco))
            @test maximum(abs, ep.value' * Ru) < 1e-8
            @test maximum(abs, (bc == 1 ? ep.d1 : rb .* ep.d2)' * Ru) < 1e-7
            @test maximum(abs, (bc == 1 ? ep.value : ep.value .- rb .* ep.d1)' * Rv) < 1e-8
            @test maximum(abs, (tc == 0 ? ep.value : ep.d1)' * Rh) < 1e-8
        end
        @test rank(Ru) == N - 3
        @test rank(Rv) == rank(Rh) == N - 1
    end
end

@testset "MHD heating spectra agree with independent collocation" begin
    spectra = Dict{Symbol,Vector{ComplexF64}}()
    for heating in (:differential, :internal)
        p = MHDParams(E=1e-3, Pr=1.0, Ra=1e4, ricb=0.35, m=2, lmax=6, N=24,
                      heating=heating)
        A, B, _ = Magrathea.assemble_mhd_galerkin(MHDStabilityOperator(p))
        spectra[heating] = sort(eigvals(A, B); by=real, rev=true)[1:6]

        # The collocation operator independently discretizes these two thermal
        # profiles, using r³ weighting for differential heating and r² for internal.
        cp = OnsetParams(E=p.E, Pr=p.Pr, Ra=p.Ra, χ=p.ricb, m=p.m, lmax=p.lmax,
                         Nr=p.N+1, equatorial_symmetry=:symmetric,
                         use_sparse_weighting=heating === :differential)
        op = LinearStabilityOperator(cp)
        Ac, Bc, idofs, bdofs = assemble_matrices(op)
        Ar, Br, _ = Magrathea._constrained_reduced_matrices(Ac, Bc, op, idofs, bdofs)
        reference = eigvals(Ar, Br)
        for λ in spectra[heating]
            @test minimum(abs.(reference .- λ)) < 1e-7
        end
    end
    @test abs(first(spectra[:internal]) - first(spectra[:differential])) > 1e-4
end
