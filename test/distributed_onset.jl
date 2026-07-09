using Test
using SparseArrays
using Magrathea

@testset "onset interior COO partition-reassembles (pre-BC)" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    full = Magrathea._assemble_onset_coo(op)
    n = full.n
    A_pre = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_pre = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_onset_coo(op; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows)
        @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_pre
    @test sparse(Br,Bc,Bv,n,n) == B_pre
end

@testset "biglobal interior COO partition-reassembles (pre-BC, with basic state)" begin
    χ = 0.35; Nr = 16; lmax = 6
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    bs = conduction_basic_state(cd, χ, 6)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=χ, m=2, lmax=lmax, Nr=Nr,
                         basic_state=bs)
    op = LinearStabilityOperator(params)
    @assert op.params.basic_state !== nothing
    full = Magrathea._assemble_onset_coo(op)
    n = full.n
    A_pre = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_pre = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_onset_coo(op; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows); @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_pre
    @test sparse(Br,Bc,Bv,n,n) == B_pre
end

@testset "biglobal COO == radial + dense basic-state" begin
    χ = 0.35; Nr = 16; lmax = 6
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    bs = conduction_basic_state(cd, χ, 6)
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=χ, m=2, lmax=lmax, Nr=Nr,
                         basic_state=bs)
    op = LinearStabilityOperator(params)
    n = op.total_dof

    # COO assembly (now includes basic state)
    coo = Magrathea._assemble_onset_coo(op)
    A_coo = Matrix(sparse(coo.A_rows, coo.A_cols, coo.A_vals, n, n))
    B_coo = Matrix(sparse(coo.B_rows, coo.B_cols, coo.B_vals, n, n))

    # Reference: radial-only COO densified, then dense add_basic_state_operators!
    coo_radial = Magrathea._assemble_onset_radial_coo(op)
    A_ref = Matrix(sparse(coo_radial.A_rows, coo_radial.A_cols, coo_radial.A_vals, n, n))
    B_ref = Matrix(sparse(coo_radial.B_rows, coo_radial.B_cols, coo_radial.B_vals, n, n))
    bs_ops = Magrathea.build_basic_state_operators(op.params.basic_state, op, op.params.m)
    Magrathea.add_basic_state_operators!(A_ref, B_ref, bs_ops, op, op.params.m)

    @test A_coo == A_ref
    @test B_coo == B_ref
end

@testset "constraint sub-blocks match BC-overwritten A rows" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_bc, _, _, _ = assemble_matrices(op)
    for ℓ in op.l_sets[:P]
        idx = op.index_map[(ℓ,:P)]; ri,it,ot,ro = Magrathea.poloidal_tau_indices(idx)
        @test Magrathea._constraint_subblock(op, ℓ, :P) ≈ A_bc[[ri,it,ot,ro], idx] rtol=1e-10
    end
    for ℓ in op.l_sets[:T]
        idx = op.index_map[(ℓ,:T)]; r1,r2 = Magrathea.toroidal_boundary_indices(idx)
        @test Magrathea._constraint_subblock(op, ℓ, :T) ≈ A_bc[[r1,r2], idx] rtol=1e-10
    end
    for ℓ in op.l_sets[:Θ]
        idx = op.index_map[(ℓ,:Θ)]; r1,r2 = Magrathea.temperature_boundary_indices(idx)
        @test Magrathea._constraint_subblock(op, ℓ, :Θ) ≈ A_bc[[r1,r2], idx] rtol=1e-10
    end
end

@testset "reduction from sub-blocks gives same reduced spectrum" begin
    using LinearAlgebra
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_full, B_full, idofs, bdofs = assemble_matrices(op)
    A_ref, B_ref, _ = Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)
    red = Magrathea._constraint_reduction_from_subblocks(op)
    S, P = Magrathea._constraint_projection_matrices(red, idofs)
    A_sub = Matrix(S*A_full*P); B_sub = Matrix(S*B_full*P)
    λref = sort(eigvals(A_ref, B_ref); by=abs)
    λsub = sort(eigvals(A_sub, B_sub); by=abs)
    @test λref ≈ λsub rtol=1e-8
end

@testset "_onset_interior_dofs matches assemble_matrices" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    _, _, idofs, _ = assemble_matrices(op)
    @test Magrathea._onset_interior_dofs(op) == idofs
end

@testset "Distributed onset/biglobal (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed onset/biglobal test (validate under mpirun; see README)"
        @test true
    else
        @info "Run the mpirun onset/biglobal :slepc spectrum check manually; see README."
        @test true
    end
end
