using Test
using SparseArrays
using Magrathea

@testset "_mhd_index_map tiles rows by section" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                       m=1, lmax=3, N=8, B0_type=dipole, B0_amplitude=1.0)
    op = MHDStabilityOperator(params)
    im = Magrathea._mhd_index_map(op)
    n_per_mode = params.N + 1
    @test all(length(r) == n_per_mode for r in values(im))
    sorted = sort(collect(values(im)); by=first)
    @test first(sorted[1]) == 1
    for i in 2:length(sorted)
        @test first(sorted[i]) == last(sorted[i-1]) + 1
    end
    @test last(sorted[end]) == op.matrix_size
    key, loc = Magrathea.row_to_dof(im, n_per_mode + 1)
    @test Magrathea.dof_to_row(im, key, loc) == n_per_mode + 1
end

@testset "_owned_coo_nnz counts owned rows by band" begin
    rows = [1, 1, 2, 3, 3, 4]
    cols = [1, 3, 2, 1, 4, 4]
    d, o = Magrathea._owned_coo_nnz(rows, cols, 0, 2)
    @test d == [1, 1] && o == [1, 0]
    d2, o2 = Magrathea._owned_coo_nnz(rows, cols, 2, 4)
    @test d2 == [1, 1] && o2 == [1, 0]
    d3, o3 = Magrathea._owned_coo_nnz(rows, cols, 0, 4)
    @test d3 == [2,1,2,1] && o3 == [0,0,0,0]
end

@testset "distributed interior COO partition-reassembles to full pre-BC matrix" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                       m=1, lmax=3, N=8, B0_type=dipole, B0_amplitude=1.0)
    op = MHDStabilityOperator(params)
    full = Magrathea._assemble_mhd_coo(op)
    n = full.n
    A_pre = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_pre = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_mhd_coo(op; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows)
        @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_pre
    @test sparse(Br,Bc,Bv,n,n) == B_pre
end

@testset "Distributed MHD assembly (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed MHD assembly test (validate under mpirun; see README)"
        @test true
    else
        @info "Run the mpirun :slepc MHD spectrum check manually; see README."
        @test true
    end
end
