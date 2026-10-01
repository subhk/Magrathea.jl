module MHDBoundarySolveChecks

using Test, Magrathea, LinearAlgebra, Logging

quiet(f) = with_logger(f, NullLogger())
params(; kwargs...) = MHDParams(; merge((; E=.01, Pr=1., Pm=1., Ra=1., Le=.05,
    ricb=.35, m=1, lmax=4, N=12, symm=0, B0_type=axial,
    bci_magnetic=2, bco_magnetic=2), (;kwargs...))...)

@testset "MHD boundary-check options validate before solving" begin
    problem = quiet(() -> MHDProblem(params()))
    for options in ((;boundary_check=:invalid), (;boundary_rtol=-1.),
                    (;boundary_rtol=NaN), (;boundary_atol=Inf), (;boundary_atol=-1.))
        @test_throws ArgumentError solve(problem; backend=:dense, options...)
    end
end

@testset "Physical boundary reports are attached and can reject a solve" begin
    problem = quiet(() -> MHDProblem(params(N=8, lmax=2)))
    result = quiet(() -> solve(problem; backend=:dense, nev=2))
    report = result.extra.magnetic_boundaries
    @test report.applicable && report.checked
    @test length(report.per_mode) == length(result.eigenvalues)
    @test !report.passed
    @test magnetic_boundary_residuals(result).passed == report.passed
    @test_throws ArgumentError quiet(() -> solve(problem; backend=:dense, nev=2,
                                                 boundary_check=:error))
    @test_logs (:warn, r"magnetic boundary residuals") Magrathea._check_mhd_magnetic_boundaries(
        result.extra.operator, result.eigenvectors; check=:warn, rtol=1e-6, atol=0)
    unchecked = quiet(() -> solve(problem; backend=:dense, nev=1, boundary_check=:none))
    @test unchecked.extra.magnetic_boundaries === nothing

    # Missing vectors on distributed worker ranks must not be called a pass.
    op = result.extra.operator
    unavailable = Magrathea._check_mhd_magnetic_boundaries(op, zeros(ComplexF64,0,2);
        check=:error, rtol=1e-6, atol=0)
    @test !unavailable.checked && !unavailable.passed
    # A present but zero eigenvector is also not a certificate of accuracy.
    @test_throws ArgumentError Magrathea._check_mhd_magnetic_boundaries(
        op, zeros(ComplexF64,op.matrix_size,1); check=:error, rtol=1e-6, atol=0)

    hydro = quiet(() -> solve(MHDProblem(params(B0_type=no_field, Le=0.,
        bci_magnetic=0, bco_magnetic=0)); backend=:dense, nev=1, boundary_check=:error))
    @test !hydro.extra.magnetic_boundaries.applicable
end

@testset "Resolved perfect-conductor modes satisfy strict boundary checks" begin
    result = quiet(() -> solve(MHDProblem(params(N=24, lmax=8));
        backend=:dense, nev=2, boundary_check=:error))
    @test length(result.eigenvalues) == 2
    @test result.extra.magnetic_boundaries.passed
    @test magnetic_boundary_residuals(result).passed
end

@testset "Angular warning checks the actual leading eigenvalue" begin
    op = quiet(() -> MHDStabilityOperator(params()))
    index = Magrathea._mhd_index_map(op)
    modes = sort(collect(op.ll_u))
    @test length(modes) > 1
    vectors = zeros(ComplexF64, op.matrix_size, 2)
    vectors[first(index[(first(modes), :u)]), 1] = 1
    vectors[first(index[(last(modes), :u)]), 2] = 1
    tails = @test_logs (:warn, r"angular spectral tail") Magrathea._check_mhd_resolution(
        op, ComplexF64[-1, 0], vectors)
    @test tails[2].angular == 1
    @test iszero(tails[2].radial)
end

end
