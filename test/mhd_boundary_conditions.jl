using Test
using SparseArrays
using Magrathea

function _magnetic_bc_dummy_op(; bci_magnetic=1, bco_magnetic=0, forcing_frequency=1.0)
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1.0, ricb=0.35,
                       m=1, lmax=3, symm=1, N=8,
                       bci_magnetic=bci_magnetic,
                       bco_magnetic=bco_magnetic,
                       forcing_frequency=forcing_frequency)

    return (
        params = params,
        ll_u = Int[],
        ll_v = Int[],
        ll_f = [2],
        ll_g = [2],
        ll_h = Int[],
    )
end

@testset "Finite-conductivity toroidal magnetic BC is g=0" begin
    op = _magnetic_bc_dummy_op(bci_magnetic=1, forcing_frequency=1.0)
    N = op.params.N
    n_per_mode = N + 1
    n = 2 * n_per_mode
    A = spzeros(ComplexF64, n, n)
    B = spzeros(ComplexF64, n, n)

    Magrathea.apply_magnetic_boundary_conditions!(A, B, op, :g)

    inner_row = 2 * n_per_mode
    g_block = (n_per_mode + 1):(2 * n_per_mode)
    expected = ComplexF64.(Magrathea._chebyshev_boundary_values(N, :inner))

    @test Vector(A[inner_row, g_block]) ≈ expected
    @test nnz(B[inner_row, :]) == 0
end
