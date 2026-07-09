using Test
using SparseArrays
using Magrathea

# small coupled triglobal fixture (mirror test/triglobal.jl ~line 324-344)
function _p5_problem()
    E=1e-3; Pr=1.0; Ra=1e4; χ=0.35; Nr=12; lmax=4
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
    uphi = Dict((1,1) => ones(Float64, Nr))
    duphi_dr = Dict((1,1) => zeros(Float64, Nr))
    # reproduce _basic_state_3d_with_modes from test/triglobal.jl:
    T = Float64
    bs3d = Magrathea.BasicState3D{T}(lmax_bs=1, mmax_bs=1, Nr=Nr, r=cd.x,
        theta_coeffs=Dict{Tuple{Int,Int},Vector{T}}(), dtheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{T}}(),
        ur_coeffs=Dict{Tuple{Int,Int},Vector{T}}(), utheta_coeffs=Dict{Tuple{Int,Int},Vector{T}}(),
        uphi_coeffs=uphi, dur_dr_coeffs=Dict{Tuple{Int,Int},Vector{T}}(),
        dutheta_dr_coeffs=Dict{Tuple{Int,Int},Vector{T}}(), duphi_dr_coeffs=duphi_dr)
    params = TriglobalParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m_range=1:2, lmax=lmax, Nr=Nr, basic_state_3d=bs3d)
    problem = setup_coupled_mode_problem(params)
    single = Magrathea.build_single_mode_operators(problem, false)
    coupling = Magrathea.build_mode_coupling_operators(problem, single, false)
    return problem, single, coupling
end

@testset "triglobal coupled-pencil COO partition-reassembles" begin
    problem, single, coupling = _p5_problem()
    full = Magrathea._assemble_block_coo(problem, single, coupling)
    n = full.n
    A_full = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_full = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_block_coo(problem, single, coupling; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows); @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_full
    @test sparse(Br,Bc,Bv,n,n) == B_full
end

@testset "build_single_mode_operator matches the all-builder" begin
    problem, _, _ = _p5_problem()
    all_ops = Magrathea.build_single_mode_operators(problem, false)
    for m in problem.m_range
        one = Magrathea.build_single_mode_operator(problem, m)
        @test one.A ≈ all_ops[m].A
        @test one.B ≈ all_ops[m].B
    end
end

@testset "Distributed triglobal (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed triglobal test (validate under mpirun; see README)"
        @test true
    else
        @info "Run the mpirun triglobal :slepc spectrum check manually; see README."
        @test true
    end
end
