# Requires an initialized complex PETSc/SLEPc session. Run from the serial
# SLEPc coverage suite or under mpiexec; no parallel factorization is required.
using Test
using SparseArrays
using Magrathea
using PetscWrap
using SlepcWrap

function _check_slepc_coo_action(rows, cols, vals, n)
    ext = Base.get_extension(Magrathea, :MagratheaSlepcExt)
    A = sparse(rows, cols, vals, n, n)
    mat, rs, re = ext._create_dist_mat(n)
    ext._fill_dist_mat!(mat, rows, cols, vals, rs, re)
    x, y = MatCreateVecs(mat)
    try
        xs, xe = VecGetOwnershipRange(x)
        for probe in (ones(ComplexF64, n), ComplexF64[sin(k) + im*cos(2k) for k in 1:n])
            for i in xs:(xe-1)
                VecSetValue(x, i, PetscScalar(probe[i+1]), INSERT_VALUES)
            end
            VecAssemblyBegin(x)
            VecAssemblyEnd(x)
            MatMult(mat, x, y)
            actual = ext._vec_scatter_to_zero(y)
            if PetscWrap.MPI.Comm_rank(PetscWrap.MPI.COMM_WORLD) == 0
                @test actual ≈ A * probe rtol=1e-12 atol=1e-10
            else
                @test isempty(actual)
            end
        end
    finally
        VecDestroy(x)
        VecDestroy(y)
        MatDestroy(mat)
    end
end

@testset "PETSc COO accumulation matches serial matrix action" begin
    _check_slepc_coo_action([1, 1, 1, 1, 2, 2, 3, 4], [1, 1, 3, 3, 2, 4, 1, 4],
                            ComplexF64[1, 2, 3, -3, 0, 2, 4, 5], 4)
    # Under mpiexec, at least one rank owns no rows of this matrix.
    _check_slepc_coo_action([1, 1], [1, 1], ComplexF64[2, 3], 1)

    p = OnsetParams(E=1e-3, Pr=1.0, Ra=1e3, χ=0.35, m=2, lmax=6, Nr=16)
    bs = basic_state(p; mode=:meridional)
    op = LinearStabilityOperator(OnsetParams(E=p.E, Pr=p.Pr, Ra=p.Ra, χ=p.χ,
        m=p.m, lmax=p.lmax, Nr=p.Nr, basic_state=bs))
    c = Magrathea._assemble_onset_coo(op)
    @test length(unique(zip(c.A_rows, c.A_cols))) < length(c.A_vals)
    _check_slepc_coo_action(c.A_rows, c.A_cols, c.A_vals, c.n)
    _check_slepc_coo_action(c.B_rows, c.B_cols, c.B_vals, c.n)
end

@testset "SLEPc convergence controls round-trip" begin
    ext = Base.get_extension(Magrathea, :MagratheaSlepcExt)
    eps = EPSCreate(PetscWrap.MPI.COMM_WORLD)
    try
        ext._eps_set_tolerances(eps, 2e-11, 137)
        @test EPSGetTolerances(eps) == (2e-11, 137)
        ext._eps_set_tolerances(eps, 4e-7, 29)
        @test EPSGetTolerances(eps) == (4e-7, 29)
    finally
        EPSDestroy(eps)
    end
end
