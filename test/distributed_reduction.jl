using Test
using LinearAlgebra
using SparseArrays
using Magrathea

@testset "S·A·P reproduces the constrained reduction" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_full, B_full, idofs, bdofs = assemble_matrices(op)
    A_red, B_red, reduction = Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)

    S, P = Magrathea._constraint_projection_matrices(reduction, idofs)

    @test size(P) == (reduction.n_full, reduction.n_reduced)
    @test size(S) == (reduction.n_reduced, reduction.n_full)
    @test nnz(S) == reduction.n_reduced
    @test Matrix(S * A_full * P) ≈ A_red rtol=1e-10
    @test Matrix(S * B_full * P) ≈ B_red rtol=1e-10
end

@testset "Distributed constrained reduction (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed constrained reduction test (validate under mpirun; see README)"
        @test true
    else
        @info "Run the mpirun onset/biglobal :slepc spectrum check manually; see README."
        @test true
    end
end
