using Test
using SparseArrays
using Magrathea

@testset "MHD stress-free velocity BC assembles without UndefVarError" begin
    # Regression: apply_velocity_boundary_conditions! used Complex{T} without
    # binding T (no `where`), so the stress-free toroidal branch (bco!=1 / bci!=1)
    # threw UndefVarError. No-slip (bci=bco=1) masked it. Exercise stress-free.
    for (bci, bco) in ((2, 2), (1, 2), (2, 1))
        params = MHDParams(
            E = 1e-3, Pr = 1.0, Pm = 1.0, Ra = 100.0, Le = 1.0,
            ricb = 0.35, m = 1, lmax = 3, N = 12,
            B0_type = dipole, B0_amplitude = 1.0,
            bci = bci, bco = bco,
        )
        op = MHDStabilityOperator(params)
        A, B, _, _ = assemble_mhd_matrices(op)
        @test eltype(A) === ComplexF64
        @test eltype(B) === ComplexF64
    end
end

@testset "combine_terms is type-inferrable and axial Lorentz blocks stay complex" begin
    # combine_terms derives its output eltype from eltype(terms) (compile time),
    # so block builders that call it are inferrable.
    real_terms = Tuple{Int, SparseMatrixCSC{Float64, Int}}[(2, spdiagm(0 => ones(4)))]
    @inferred Magrathea.combine_terms(real_terms)
    @test eltype(Magrathea.combine_terms(real_terms)) === Float64

    f32_terms = Tuple{Int, SparseMatrixCSC{Float32, Int}}[(2, spdiagm(0 => ones(Float32, 4)))]
    @test eltype(Magrathea.combine_terms(f32_terms)) === Float32

    params = MHDParams(
        E = 1e-3, Pr = 1.0, Pm = 1.0, Ra = 100.0, Le = 1.3,
        ricb = 0.35, m = 2, lmax = 6, N = 16,
        B0_type = axial, B0_amplitude = 1.0,
    )
    op = MHDStabilityOperator(params)
    # Sparse-accumulated axial Lorentz blocks must remain complex (former dense path).
    @inferred Magrathea.lorentz_upol_bpol_axial(op, 3, params.m, -2, params.Le)
    @inferred Magrathea.lorentz_upol_btor_axial(op, 3, params.m, -1, params.Le)
    @test eltype(Magrathea.lorentz_upol_bpol_axial(op, 3, params.m, -2, params.Le)) === ComplexF64
    @test eltype(Magrathea.lorentz_upol_btor_axial(op, 3, params.m, 0, params.Le)) === ComplexF64
end

@testset "MHD stress-free velocity BC preserves Float32 storage" begin
    T = Float32
    params = MHDParams(
        E = T(1e-3), Pr = one(T), Pm = one(T), Ra = T(100), Le = one(T),
        ricb = T(0.35), m = 1, lmax = 3, N = 8,
        B0_type = dipole, B0_amplitude = one(T),
        bci = 2, bco = 2,
    )
    op = MHDStabilityOperator(params)
    A, B, _, _ = assemble_mhd_matrices(op)
    @test eltype(A) === ComplexF32
    @test eltype(B) === ComplexF32
end
