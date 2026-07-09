using Test
using SparseArrays
using LinearAlgebra
using Magrathea

# =============================================================================
#  Construction-only coverage for the MHD operator builders.
#
#  Every test here exercises matrix/operator CONSTRUCTION and pure helpers only:
#  no eigensolver, no solve(...), no SLEPc/PETSc/MPI. We assert structural
#  properties (eltype, dimensions, zero/non-zero blocks, real-vs-complex block
#  design, type inferability) rather than hand-computed float values.
# =============================================================================

# Shared operators (built once). m=1 so the azimuthal (im*m) couplings are
# non-trivial; both background-field branches (axial and dipole) are exercised.
const AXIAL_PARAMS = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 1.0, Ra = 100.0, Le = 1.0,
    ricb = 0.35, m = 1, lmax = 6, N = 16,
    B0_type = axial, B0_amplitude = 1.0,
)
const AXIAL_OP = MHDStabilityOperator(AXIAL_PARAMS)

const DIPOLE_PARAMS = MHDParams(
    E = 1e-3, Pr = 1.0, Pm = 1.0, Ra = 100.0, Le = 1.0,
    ricb = 0.35, m = 1, lmax = 6, N = 16,
    B0_type = dipole, B0_amplitude = 1.0,
)
const DIPOLE_OP = MHDStabilityOperator(DIPOLE_PARAMS)

const NAX = AXIAL_PARAMS.N + 1   # 17
const NDI = DIPOLE_PARAMS.N + 1  # 17

@testset "zero_block / real_zero_block sizing and eltype" begin
    for (op, T) in ((AXIAL_OP, Float64), (DIPOLE_OP, Float64))
        n = op.params.N + 1
        zb = Magrathea.zero_block(op)
        rzb = Magrathea.real_zero_block(op)
        @test zb isa SparseMatrixCSC
        @test eltype(zb) === Complex{T}
        @test size(zb) == (n, n)
        @test iszero(zb)
        @test eltype(rzb) === T
        @test size(rzb) == (n, n)
        @test iszero(rzb)
    end
end

@testset "combine_terms eltype/dim across Float64 and Float32" begin
    # Real Int-coefficient terms -> matrix eltype preserved.
    real64 = Tuple{Int, SparseMatrixCSC{Float64, Int}}[(2, spdiagm(0 => ones(5)))]
    out64 = Magrathea.combine_terms(real64)
    @test eltype(out64) === Float64
    @test size(out64) == (5, 5)
    @inferred Magrathea.combine_terms(real64)

    real32 = Tuple{Int, SparseMatrixCSC{Float32, Int}}[(3, spdiagm(0 => ones(Float32, 4)))]
    out32 = Magrathea.combine_terms(real32)
    @test eltype(out32) === Float32
    @test size(out32) == (4, 4)

    # Complex coefficient promotes the block to complex.
    cplx = Tuple{ComplexF64, SparseMatrixCSC{Float64, Int}}[(2.0 + 1.0im, spdiagm(0 => ones(3)))]
    outc = Magrathea.combine_terms(cplx)
    @test eltype(outc) === ComplexF64
    @test size(outc) == (3, 3)

    # Complex matrix promotes the block to complex too.
    cmat = Tuple{Int, SparseMatrixCSC{ComplexF64, Int}}[(2, spdiagm(0 => fill(1.0 + 0im, 3)))]
    @test eltype(Magrathea.combine_terms(cmat)) === ComplexF64

    # All-zero coefficients still return a correctly-sized zero block.
    zterms = Tuple{Int, SparseMatrixCSC{Float64, Int}}[(0, spdiagm(0 => ones(6)))]
    zout = Magrathea.combine_terms(zterms)
    @test size(zout) == (6, 6)
    @test iszero(zout)
    @test eltype(zout) === Float64

    # Linearity: combine_terms([(c, M)]) == c .* M
    base = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)
    scaled = Magrathea.combine_terms(Tuple{Int, SparseMatrixCSC{Float64, Int}}[(4, base)])
    @test scaled == 4 .* base

    # Empty -> typed empty per eltype(terms).
    empty32 = Tuple{Float32, SparseMatrixCSC{Float32, Int}}[]
    @test eltype(Magrathea.combine_terms(empty32)) === Float32
end

@testset "_scale_sparse eltype promotion and values" begin
    rmat = sparse([1, 2], [1, 2], [2.0, 4.0], 3, 3)
    cmat = sparse([1, 2], [1, 2], ComplexF64[2.0, 4.0], 3, 3)

    # real type, real coef, real matrix -> real (skip-copy path)
    s1 = Magrathea._scale_sparse(Float64, 3.0, rmat)
    @test eltype(s1) === Float64
    @test s1 == 3.0 .* rmat

    # real type, complex coef, real matrix -> complex
    s2 = Magrathea._scale_sparse(Float64, 1.0 + 2.0im, rmat)
    @test eltype(s2) === ComplexF64
    @test s2 == (1.0 + 2.0im) .* rmat

    # real type, real coef, complex matrix -> complex
    s3 = Magrathea._scale_sparse(Float64, 5.0, cmat)
    @test eltype(s3) === ComplexF64
    @test s3 == 5.0 .* cmat

    # Float32 storage preserved
    rmat32 = sparse([1, 2], [1, 2], Float32[2.0, 4.0], 3, 3)
    s4 = Magrathea._scale_sparse(Float32, 2.0f0, rmat32)
    @test eltype(s4) === Float32
    @test Magrathea._scale_sparse(Float32, 1.0f0 + 1.0f0im, rmat32) |> eltype === ComplexF32
end

@testset "Axial Lorentz helper blocks stay complex over all offsets" begin
    Le = AXIAL_PARAMS.Le
    # lorentz_upol_bpol_axial supports offsets -2:2; l=4,m=1 is non-degenerate.
    for o in -2:2
        blk = Magrathea.lorentz_upol_bpol_axial(AXIAL_OP, 4, 1, o, Le)
        @test eltype(blk) === ComplexF64
        @test size(blk) == (NAX, NAX)
    end
    # lorentz_upol_btor_axial supports offsets -1:1.
    for o in -1:1
        blk = Magrathea.lorentz_upol_btor_axial(AXIAL_OP, 3, 1, o, Le)
        @test eltype(blk) === ComplexF64
        @test size(blk) == (NAX, NAX)
    end
    # Out-of-range offset -> complex zero block.
    far = Magrathea.lorentz_upol_bpol_axial(AXIAL_OP, 4, 1, 9, Le)
    @test eltype(far) === ComplexF64
    @test iszero(far)
    @test iszero(Magrathea.lorentz_upol_btor_axial(AXIAL_OP, 3, 1, 5, Le))

    # Degenerate sqrt-factor (l=3,m=2,offset=-2) collapses to a zero block.
    degen = Magrathea.lorentz_upol_bpol_axial(AXIAL_OP, 3, 2, -2, Le)
    @test iszero(degen)
    @test eltype(degen) === ComplexF64

    # Inferability (matches the locked-in complex return type).
    @inferred Magrathea.lorentz_upol_bpol_axial(AXIAL_OP, 4, 1, -2, Le)
    @inferred Magrathea.lorentz_upol_btor_axial(AXIAL_OP, 3, 1, 0, Le)
end

@testset "Lorentz operators: complex blocks (axial + dipole branches)" begin
    for (op, n) in ((AXIAL_OP, NAX), (DIPOLE_OP, NDI))
        Le = op.params.Le
        m = op.params.m

        # Poloidal-velocity diagonal (toroidal-magnetic source): complex.
        d = Magrathea.operator_lorentz_poloidal_diagonal(op, 4, Le)
        @test eltype(d) === ComplexF64
        @test size(d) == (n, n)

        # Poloidal-velocity off-diagonal (toroidal-magnetic source) offsets -1,1.
        for o in (-1, 1)
            blk = Magrathea.operator_lorentz_poloidal_offdiag(op, 4, m, o, Le)
            @test eltype(blk) === ComplexF64
            @test size(blk) == (n, n)
        end
        # offset 0 is still a complex block of the right size. For the dipole
        # branch it is the zero block; the axial branch delegates to the
        # (non-zero) toroidal-magnetic diagonal.
        o0 = Magrathea.operator_lorentz_poloidal_offdiag(op, 4, m, 0, Le)
        @test eltype(o0) === ComplexF64
        @test size(o0) == (n, n)
        op.params.B0_type == dipole && @test iszero(o0)

        # Poloidal-velocity from poloidal-magnetic, offsets -2:2: complex.
        for o in -2:2
            blk = Magrathea.operator_lorentz_poloidal_from_bpol(op, 4, m, o, Le)
            @test eltype(blk) === ComplexF64
            @test size(blk) == (n, n)
        end
        # Out-of-range offset -> complex zero.
        @test iszero(Magrathea.operator_lorentz_poloidal_from_bpol(op, 4, m, 5, Le))

        # Toroidal-velocity diagonal (poloidal-magnetic source): complex.
        t = Magrathea.operator_lorentz_toroidal(op, 4, Le)
        @test eltype(t) === ComplexF64
        @test size(t) == (n, n)

        # Toroidal-velocity from poloidal-magnetic, offsets -1:1: complex
        # (offset 0 delegates to operator_lorentz_toroidal).
        for o in -1:1
            blk = Magrathea.operator_lorentz_toroidal_from_bpol(op, 4, m, o, Le)
            @test eltype(blk) === ComplexF64
            @test size(blk) == (n, n)
        end
        @test iszero(Magrathea.operator_lorentz_toroidal_from_bpol(op, 4, m, 7, Le))
    end
end

@testset "Lorentz toroidal-from-btor is a REAL block over all offsets" begin
    # This builder uses purely real coefficients (no im*m) -> real storage,
    # unlike the other Lorentz blocks. Exercise both field branches.
    for (op, n) in ((AXIAL_OP, NAX), (DIPOLE_OP, NDI))
        m = op.params.m
        Le = op.params.Le
        for o in -2:2
            blk = Magrathea.operator_lorentz_toroidal_from_btor(op, 4, m, o, Le)
            @test eltype(blk) === Float64
            @test size(blk) == (n, n)
        end
        @test iszero(Magrathea.operator_lorentz_toroidal_from_btor(op, 4, m, 9, Le))
    end
    @inferred Magrathea.operator_lorentz_toroidal_from_btor(AXIAL_OP, 4, 1, -2, AXIAL_PARAMS.Le)
end

@testset "Induction operators: real vs complex block design" begin
    for (op, n) in ((AXIAL_OP, NAX), (DIPOLE_OP, NDI))
        m = op.params.m

        # poloidal-from-u: real, offsets -2:2; out-of-range throws.
        for o in -2:2
            blk = Magrathea.operator_induction_poloidal_from_u(op, 3, m, o)
            @test eltype(blk) === Float64
            @test size(blk) == (n, n)
        end
        @test_throws ErrorException Magrathea.operator_induction_poloidal_from_u(op, 3, m, 3)

        # poloidal-from-v: complex, offsets -1:1; out-of-range -> complex zero.
        for o in -1:1
            blk = Magrathea.operator_induction_poloidal_from_v(op, 3, m, o)
            @test eltype(blk) === ComplexF64
            @test size(blk) == (n, n)
        end
        @test iszero(Magrathea.operator_induction_poloidal_from_v(op, 3, m, 4))

        # toroidal-from-u: complex, offsets -1:1; out-of-range throws.
        for o in -1:1
            blk = Magrathea.operator_induction_toroidal_from_u(op, 3, m, o)
            @test eltype(blk) === ComplexF64
            @test size(blk) == (n, n)
        end
        @test_throws ErrorException Magrathea.operator_induction_toroidal_from_u(op, 3, m, 2)

        # toroidal-from-v: real, offsets -2:2; out-of-range -> real zero.
        for o in -2:2
            blk = Magrathea.operator_induction_toroidal_from_v(op, 3, m, o)
            @test eltype(blk) === Float64
            @test size(blk) == (n, n)
        end
        @test iszero(Magrathea.operator_induction_toroidal_from_v(op, 3, m, 5))
    end
    @inferred Magrathea.operator_induction_poloidal_from_u(AXIAL_OP, 3, 1, 0)
    @inferred Magrathea.operator_induction_toroidal_from_v(AXIAL_OP, 3, 1, 0)
end

@testset "Magnetic diffusion and mass (b) operators: real, non-trivial" begin
    for (op, n) in ((AXIAL_OP, NAX), (DIPOLE_OP, NDI))
        Em = op.params.Em
        for l in (1, 3, 4)
            dp = Magrathea.operator_magnetic_diffusion_poloidal(op, l, Em)
            dt = Magrathea.operator_magnetic_diffusion_toroidal(op, l, Em)
            bp = Magrathea.operator_b_poloidal(op, l)
            bt = Magrathea.operator_b_toroidal(op, l)
            for blk in (dp, dt, bp, bt)
                @test eltype(blk) === Float64
                @test size(blk) == (n, n)
                @test nnz(blk) > 0   # L = l(l+1) > 0 -> non-degenerate
            end
        end
    end
    @inferred Magrathea.operator_magnetic_diffusion_poloidal(AXIAL_OP, 3, AXIAL_PARAMS.Em)
    @inferred Magrathea.operator_b_poloidal(AXIAL_OP, 3)
end

@testset "Float32 storage preservation across MHD operator builders" begin
    T = Float32
    f32_params = MHDParams(
        E = T(1e-3), Pr = one(T), Pm = one(T), Ra = T(100), Le = one(T),
        ricb = T(0.35), m = 1, lmax = 4, N = 8,
        B0_type = axial, B0_amplitude = one(T),
    )
    op = MHDStabilityOperator(f32_params)
    n = f32_params.N + 1
    Le = f32_params.Le
    Em = f32_params.Em

    @test eltype(Magrathea.zero_block(op)) === ComplexF32
    @test eltype(Magrathea.real_zero_block(op)) === Float32

    # Complex blocks -> ComplexF32
    @test eltype(Magrathea.lorentz_upol_bpol_axial(op, 3, 1, -1, Le)) === ComplexF32
    @test eltype(Magrathea.lorentz_upol_btor_axial(op, 3, 1, 0, Le)) === ComplexF32
    @test eltype(Magrathea.operator_lorentz_poloidal_diagonal(op, 3, Le)) === ComplexF32
    @test eltype(Magrathea.operator_lorentz_poloidal_from_bpol(op, 3, 1, 0, Le)) === ComplexF32
    @test eltype(Magrathea.operator_lorentz_toroidal(op, 3, Le)) === ComplexF32
    @test eltype(Magrathea.operator_induction_poloidal_from_v(op, 2, 1, -1)) === ComplexF32
    @test eltype(Magrathea.operator_induction_toroidal_from_u(op, 2, 1, 0)) === ComplexF32

    # Real blocks -> Float32
    @test eltype(Magrathea.operator_lorentz_toroidal_from_btor(op, 3, 1, -2, Le)) === Float32
    @test eltype(Magrathea.operator_induction_poloidal_from_u(op, 2, 1, 0)) === Float32
    @test eltype(Magrathea.operator_induction_toroidal_from_v(op, 2, 1, 0)) === Float32
    @test eltype(Magrathea.operator_magnetic_diffusion_poloidal(op, 3, Em)) === Float32
    @test eltype(Magrathea.operator_b_toroidal(op, 3)) === Float32

    # Dimensions preserved at Float32 too.
    @test size(Magrathea.operator_lorentz_poloidal_from_bpol(op, 3, 1, 0, Le)) == (n, n)
    @test size(Magrathea.operator_b_poloidal(op, 3)) == (n, n)
end

@testset "Galerkin assembly (axial, insulating): construction only" begin
    # assemble_mhd_galerkin builds dense A, B and a layout; it does NOT solve.
    op = AXIAL_OP
    A, B, layout = Magrathea.assemble_mhd_galerkin(op)
    nred = layout.nred
    @test size(A) == (nred, nred)
    @test size(B) == (nred, nred)
    @test eltype(A) === ComplexF64
    @test eltype(B) === ComplexF64
    @test nred > 0
    @test !iszero(A)
    @test !iszero(B)

    # index_map ranges are disjoint, length M[field], and cover 1:nred.
    covered = Int[]
    for (key, rng) in layout.index_map
        field, _ = key
        @test length(rng) == layout.M[field]
        append!(covered, collect(rng))
    end
    @test sort(covered) == collect(1:nred)

    # _mhd_full_range: a valid (field, ℓ) maps to a length-(N+1) window inside
    # the full eigenvector.
    npm = op.params.N + 1
    ℓu = first(op.ll_u)
    rng = Magrathea._mhd_full_range(op, :u, ℓu)
    @test length(rng) == npm
    @test first(rng) >= 1
    @test last(rng) <= op.matrix_size

    # reconstruct_mhd_galerkin_full lifts a reduced vector to the full size.
    y = ones(ComplexF64, nred)
    full = Magrathea.reconstruct_mhd_galerkin_full(op, layout, y)
    @test length(full) == op.matrix_size
    @test eltype(full) === ComplexF64
end

@testset "Galerkin assembly preserves Float32 storage" begin
    T = Float32
    f32_params = MHDParams(
        E = T(1e-3), Pr = one(T), Pm = one(T), Ra = T(100), Le = one(T),
        ricb = T(0.35), m = 1, lmax = 4, N = 16,
        B0_type = axial, B0_amplitude = one(T),
    )
    op = MHDStabilityOperator(f32_params)
    A, B, layout = Magrathea.assemble_mhd_galerkin(op)
    @test eltype(A) === ComplexF32
    @test eltype(B) === ComplexF32
    @test size(A) == (layout.nred, layout.nred)
end
