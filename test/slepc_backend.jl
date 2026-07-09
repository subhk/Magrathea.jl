using Test
using SparseArrays
using Magrathea

@testset "SLEPc backend dispatch (core, no PETSc)" begin
    A = sparse(ComplexF64[2 0 0; 0 3 0; 0 0 4])
    B = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])

    @test_throws ArgumentError Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:nope)

    err = try
        Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:slepc)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("PetscWrap", sprint(showerror, err))
    @test occursin("SlepcWrap", sprint(showerror, err))

    # :slepc is now the default (and sole) backend, so the no-PETSc default path
    # raises the same actionable SLEPc-extension error.
    err_default = try
        Magrathea.solve_eigenvalue_problem(A, B; nev=1, sigma=0.0)
        nothing
    catch e
        e
    end
    @test err_default isa ErrorException
    @test occursin("SlepcWrap", sprint(showerror, err_default))
end

@testset "SLEPc backend reaches constrained hydro path" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    err = try
        Magrathea.solve_eigenvalue_problem(op; nev=1, backend=:slepc)
        nothing
    catch e; e end
    @test err isa ErrorException && occursin("SlepcWrap", sprint(showerror, err))
end

@testset "SLEPc backend reaches solve(::MHDProblem) galerkin path" begin
    p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                  m=1, lmax=3, N=8, B0_type=axial, B0_amplitude=1.0)
    err = try; solve(MHDProblem(p); nev=1, backend=:slepc); nothing; catch e; e end
    @test err isa ErrorException && occursin("PetscWrap", sprint(showerror, err))
end

@testset "_petsc_owned_nnz splits diagonal/off-diagonal blocks" begin
    M = sparse([1,1,2,3,3,4], [1,3,2,1,4,4], ComplexF64[1,1,1,1,1,1], 4, 4)
    d, o = Magrathea._petsc_owned_nnz(M, 0, 2)
    @test d == [1, 1] && o == [1, 0]
    d2, o2 = Magrathea._petsc_owned_nnz(M, 2, 4)
    @test d2 == [1, 1] && o2 == [1, 0]
    d3, o3 = Magrathea._petsc_owned_nnz(M, 0, 4)
    @test o3 == [0, 0, 0, 0] && d3 == [2, 1, 2, 1]
    d4, o4 = Magrathea._petsc_owned_nnz(M, 2, 2)
    @test isempty(d4) && isempty(o4)
end

@testset "empty worker eigenvectors survive ordering" begin
    # Distributed worker contract: nev eigenvalues, 0-column eigenvector matrix.
    vals = ComplexF64[0.3+1im, -0.2+0im, 0.5-1im]
    vecs0 = Matrix{ComplexF64}(undef, 10, 0)
    perm = sortperm(real.(vals); rev=true)
    out = size(vecs0, 2) == 0 ? vecs0 : vecs0[:, perm]   # guarded form used at divert sites
    @test size(out) == (10, 0)
    empty_vv = Vector{Vector{ComplexF64}}()
    M = Magrathea._eigvecs_to_matrix(vals, empty_vv, Float64)
    @test isempty(M)   # 0×nev — degenerate, but no crash and no bogus columns
end

@testset "slepc_init!/finalize! error without extension" begin
    @test_throws ErrorException Magrathea.slepc_init!()
    e = try Magrathea.slepc_init!(); catch err; err end
    @test occursin("SlepcWrap", sprint(showerror, e))
    @test_throws ErrorException Magrathea.slepc_finalize!()
end

@testset "Distributed SLEPc lifecycle + solve (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed SLEPc test (run under mpirun on a PETSc+MUMPS build)"
        @test true  # explicit: skipped, not silently absent
    else
        @eval using PetscWrap, SlepcWrap
        Magrathea.slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
                          "-st_pc_factor_mat_solver_type mumps -eps_target_magnitude")
        p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                      m=1, lmax=3, N=12, B0_type=dipole, B0_amplitude=1.0)
        op = MHDStabilityOperator(p)
        A, B, _, _ = assemble_mhd_matrices(op)
        vS, _, _ = Magrathea.solve_eigenvalue_problem(A, B; nev=4, sigma=0.0, backend=:slepc)
        # eigenvalues valid on all ranks
        @test eltype(vS) <: Complex
        @test length(vS) >= 1
        Magrathea.slepc_finalize!()
    end
end
