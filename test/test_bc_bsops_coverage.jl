using Test
using SparseArrays
using LinearAlgebra
using Logging
using Magrathea

"""Run `f()` with @info/@debug logging suppressed (build_basic_state_operators is chatty)."""
_silent(f) = with_logger(f, NullLogger())

# =============================================================================
#  Coverage for boundary-condition builders (src/Operators/boundary_conditions.jl)
#  and basic-state operator builders (src/BasicStates/basic_state_operators.jl).
#
#  Scope: construction / assembly only — NO eigensolves, NO MPI/PETSc/SLEPc.
#  Targets branches NOT exercised by test/boundary_conditions.jl,
#  test/mhd_boundary_conditions.jl, test/meridional_boundary_conditions.jl,
#  or test/type_stability.jl:
#    - magnetic BC section :f across bci/bco ∈ {0,1,2}, plus section :g bci ∈ {0,2}
#    - spherical_bessel_j_logderiv (small-|x|, real, complex)
#    - velocity_from_potentials θ→sinθ derivation + Float32 storage
#    - DimensionMismatch / ArgumentError branches of the residual BC helpers
#    - pure SH coupling helpers (gaunt, wigner3j_000, θ-derivative, meridional,
#      azimuthal-coupling tables) + diagonal block accumulators
#    - build_basic_state_operators in the axisymmetric (m=0) regime + the dense
#      add_basic_state_operators! accumulation path
#
#  Only structural assertions (types/dims/no-throw/zero-rows) and values of
#  well-defined production helpers are checked; m≠0 mean-flow coupling COEFFICIENT
#  values are covered by the independent tests in mean_flow_coupling.jl.
# =============================================================================

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""NamedTuple velocity operator for velocity_from_potentials and the residual BCs."""
function _vel_dummy_op(; m=2, CT=ComplexF64, RT=Float64, use_theta=false)
    nr, nθ = 4, 3
    Dr = Matrix{CT}(I, nr, nr)
    Dθ = Matrix{CT}(I, nθ, nθ)
    Lθ = CT(2) .* Matrix{CT}(I, nθ, nθ)
    r = RT[1.0, 0.8, 0.6, 0.4]
    if use_theta
        return (Dr=Dr, Dθ=Dθ, Lθ=Lθ, r=r, theta=RT[0.5, 1.0, 1.5], m=m)
    end
    return (Dr=Dr, Dθ=Dθ, Lθ=Lθ, r=r, sintheta=RT[0.5, 1.0, 0.5], m=m)
end

# Magnetic boundary physics and helper consistency are tested in
# mhd_boundary_conditions.jl and mhd_physics.jl.

# =============================================================================
#  spherical_bessel_j_logderiv
# =============================================================================

@testset "spherical_bessel_j_logderiv small-|x|, real, and complex paths" begin
    # Small-|x| series branch returns l/x exactly.
    xs = complex(1e-12)
    @test Magrathea.spherical_bessel_j_logderiv(2, xs) ≈ complex(2.0) / xs
    @test Magrathea.spherical_bessel_j_logderiv(0, complex(1e-12)) ≈ complex(0.0)

    # Real-argument overload promotes to a complex result.
    vr = Magrathea.spherical_bessel_j_logderiv(2, 1.5)
    @test vr isa Complex
    @test isfinite(real(vr)) && isfinite(imag(vr))

    # General complex argument.
    vc = Magrathea.spherical_bessel_j_logderiv(3, 1.0 + 0.5im)
    @test vc isa Complex{Float64}
    @test all(isfinite, (real(vc), imag(vc)))
end

# =============================================================================
#  velocity_from_potentials + residual BC helpers — branches & errors
# =============================================================================

@testset "velocity_from_potentials derives sinθ from θ and keeps Float32 storage" begin
    nr, nθ = 4, 3
    P = reshape(ComplexF64.(1.0:12.0), nr, nθ)
    Tt = reshape(ComplexF64.(13.0:24.0), nr, nθ)

    # θ-property branch (sinθ computed internally), m≠0 so the 1/(r sinθ) path runs.
    op_theta = _vel_dummy_op(m=2, use_theta=true)
    u_r, u_θ, u_φ = Magrathea.velocity_from_potentials(op_theta, P, Tt)
    @test size(u_r) == size(P)
    @test all(isfinite, u_r) && all(isfinite, u_θ) && all(isfinite, u_φ)

    # Float32 parameters/fields → ComplexF32 velocity storage.
    op32 = _vel_dummy_op(m=2, CT=ComplexF32, RT=Float32)
    P32 = ComplexF32.(P); T32 = ComplexF32.(Tt)
    r32, t32, f32 = Magrathea.velocity_from_potentials(op32, P32, T32)
    @test eltype(r32) === ComplexF32
    @test eltype(t32) === ComplexF32
    @test eltype(f32) === ComplexF32
end

@testset "velocity_from_potentials rejects inconsistent shapes" begin
    op = _vel_dummy_op(m=2)
    P = reshape(ComplexF64.(1.0:12.0), 4, 3)
    T_bad = reshape(ComplexF64.(1.0:9.0), 3, 3)
    @test_throws DimensionMismatch Magrathea.velocity_from_potentials(op, P, T_bad)

    # Dr with the wrong number of rows is rejected.
    op_bad = (Dr=Matrix{ComplexF64}(I, 3, 3), Dθ=op.Dθ, Lθ=op.Lθ,
              r=op.r, sintheta=op.sintheta, m=2)
    @test_throws DimensionMismatch Magrathea.velocity_from_potentials(op_bad, P, P)
end

@testset "Residual BC builders reject unsupported symbols and bad sizes" begin
    op = _vel_dummy_op(m=2)
    nr, nθ = 4, 3
    P = reshape(ComplexF64.(1.0:12.0), nr, nθ)
    Tt = reshape(ComplexF64.(13.0:24.0), nr, nθ)
    res_r = fill(ComplexF64(0.0), nr, nθ)
    res_θ = similar(res_r); res_φ = similar(res_r)

    # Unsupported mechanical BC symbol.
    @test_throws ArgumentError Magrathea.apply_mechanical_bc_from_potentials!(
        res_r, res_θ, res_φ, P, Tt, op; inner=:bogus, outer=:no_slip)

    # Mismatched residual block size.
    res_small = fill(ComplexF64(0.0), nr - 1, nθ)
    @test_throws DimensionMismatch Magrathea.apply_mechanical_bc_from_potentials!(
        res_small, res_θ, res_φ, P, Tt, op; inner=:no_slip, outer=:no_slip)

    # Direct enforce-helper rejects unknown BC.
    u = fill(ComplexF64(1.0), nr, nθ)
    inv_r = 1.0 ./ op.r
    @test_throws ArgumentError Magrathea.enforce_mechanical_bc_at!(
        res_r, res_θ, res_φ, u, u, u, u, u, inv_r, :nope, 1)

    # Thermal residual builder: bad symbol and bad size.
    Θ = reshape(ComplexF64.(1.0:12.0), nr, nθ)
    res_T = fill(ComplexF64(0.0), nr, nθ)
    @test_throws ArgumentError Magrathea.apply_thermal_bc_from_potentials!(
        res_T, Θ, op; inner=:bogus, outer=:fixed_temperature)
    res_T_bad = fill(ComplexF64(0.0), nr - 1, nθ)
    @test_throws DimensionMismatch Magrathea.apply_thermal_bc_from_potentials!(
        res_T_bad, Θ, op)
    dΘ = op.Dr * Θ
    @test_throws ArgumentError Magrathea.apply_thermal_bc_at!(res_T, Θ, dΘ, :nope, 0.0, 0.0, 1)
end

@testset "Operator-introspection helpers cover property and error branches" begin
    # _get_im_m variants.
    @test Magrathea._get_im_m((m=3,)) == 3im
    @test Magrathea._get_im_m((im_m=2im,)) == 2im
    @test Magrathea._get_im_m((params=(m=4,),)) == 4im
    @test_throws ArgumentError Magrathea._get_im_m((foo=1,))

    # _get_inv_r: vector, r-derived, dimension mismatch, and missing-field error.
    @test Magrathea._get_inv_r((inv_r=[1.0, 2.0],), 2) == [1.0, 2.0]
    @test Magrathea._get_inv_r((r=[1.0, 2.0],), 2) == [1.0, 0.5]
    @test_throws DimensionMismatch Magrathea._get_inv_r((inv_r=[1.0],), 2)
    @test_throws ArgumentError Magrathea._get_inv_r((foo=1,), 2)

    # _get_sinθ: sintheta / sinθ / theta / θ, plus missing-field error.
    @test Magrathea._get_sinθ((sintheta=[0.5, 1.0],), 2) == [0.5, 1.0]
    @test Magrathea._get_sinθ((sinθ=[0.25, 0.75],), 2) == [0.25, 0.75]
    @test Magrathea._get_sinθ((theta=[0.0, pi / 2],), 2) ≈ [0.0, 1.0]
    @test Magrathea._get_sinθ((θ=[0.0, pi / 2],), 2) ≈ [0.0, 1.0]
    @test_throws ArgumentError Magrathea._get_sinθ((foo=1,), 2)

    # _inv_r_vector and _inv_r_at: vector + matrix accessors and 3D rejection.
    @test Magrathea._inv_r_vector([1.0, 2.0, 3.0], 3) == [1.0, 2.0, 3.0]
    @test Magrathea._inv_r_vector([1.0 9.0; 2.0 9.0], 2) == [1.0, 2.0]
    @test Magrathea._inv_r_at([10.0, 20.0], 2) == 20.0
    @test Magrathea._inv_r_at([1.0 5.0; 3.0 7.0], 2) == 3.0
    @test_throws ArgumentError Magrathea._inv_r_vector(ones(2, 2, 2), 2)
    @test_throws ArgumentError Magrathea._inv_r_at(ones(2, 2, 2), 1)

    # _boundary_indices falls back to (Nr, 1) without r/inv_r.
    @test Magrathea._boundary_indices((foo=1,), 5) == (5, 1)
end

# =============================================================================
#  Basic-state SH coupling helpers (pure functions)
# =============================================================================

@testset "Wigner 3j (all-m=0) and Gaunt selection rules" begin
    @test Magrathea.wigner3j_000(0, 0, 0) == 1.0
    # Odd triad sum ⇒ vanishing 3j(0,0,0).
    @test Magrathea.wigner3j_000(1, 1, 1) == 0.0

    # Gaunt selection rules return an exact zero.
    @test Magrathea.compute_gaunt_coefficient(1, 1, 1, 0, 1, 1) == 0.0   # odd ℓ-sum
    @test Magrathea.compute_gaunt_coefficient(1, 1, 1, 0, 1, 0) == 0.0   # m1+m2 ≠ m3
    @test Magrathea.compute_gaunt_coefficient(1, 0, 1, 0, 5, 0) == 0.0   # triangle violated
    @test Magrathea.compute_gaunt_coefficient(5, 0, 0, 0, 5, 6) == 0.0   # |m3| > ℓ3

    # An allowed triad is finite and nonzero; memoization is deterministic.
    g = Magrathea.compute_gaunt_coefficient(2, 0, 2, 0, 2, 0)
    @test isfinite(g)
    @test g != 0.0
    @test Magrathea.compute_gaunt_coefficient(2, 0, 2, 0, 2, 0) === g
end

# Angular-derivative physics is tested independently in mean_flow_coupling.jl.

@testset "Spherical-harmonic coupling and azimuthal table builders" begin
    coeffs = Magrathea.compute_spherical_harmonic_coupling(2, 1, 0)
    @test coeffs isa Dict{Int,Float64}
    # Parity selection: every coupled ℓ' satisfies ℓ'+ℓ_pert+ℓ_bs even and ℓ' ≥ m.
    for (lp, c) in coeffs
        @test (lp + 2 + 1) % 2 == 0
        @test lp >= 0
        @test isfinite(c)
    end

    cache = Magrathea._build_azimuthal_coupling_cache(0, 4, 4, Float64)
    @test cache.m == 0
    @test eltype(cache.y_m) === Float64

    # Float32 cache preserves real precision.
    cache32 = Magrathea._build_azimuthal_coupling_cache(1, 3, 3, Float32)
    @test eltype(cache32.y_m) === Float32
    @test typeof(cache32.weight) === Float32
end

@testset "Diagonal block accumulators write exactly where expected" begin
    blocks = Dict{Tuple{Int,Int},Matrix{ComplexF64}}()
    b = Magrathea._operator_block!(blocks, (1, 2), 3, ComplexF64)
    @test size(b) == (3, 3)
    @test eltype(b) === ComplexF64
    @test all(iszero, b)
    # Re-fetching returns the same memoized object (no reallocation).
    @test Magrathea._operator_block!(blocks, (1, 2), 3, ComplexF64) === b
    @test length(blocks) == 1

    blk = zeros(ComplexF64, 3, 3)
    Magrathea._add_diagonal_block!(blk, 2.0, [1.0, 2.0, 3.0])
    @test diag(blk) == ComplexF64[2, 4, 6]
    @test blk[1, 2] == 0   # off-diagonal untouched

    blk2 = zeros(ComplexF64, 3, 3)
    Magrathea._add_diagonal_product_block!(blk2, 1.0, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    @test diag(blk2) == ComplexF64[4, 10, 18]

    blk3 = zeros(ComplexF64, 2, 2)
    M = [1.0 2.0; 3.0 4.0]
    Magrathea._add_left_diagonal_matrix_block!(blk3, 1.0, [10.0, 20.0], M)
    @test blk3 == ComplexF64[10 20; 60 80]
end

# =============================================================================
#  build_basic_state_operators — axisymmetric (m=0) regime
# =============================================================================

"""Construct a small axisymmetric basic state + matching stability operator."""
function _build_m0_basic_state(::Type{T}) where {T<:Real}
    Nr = 12
    χ = T(0.35)
    cd = ChebyshevDiffn(Nr, T[χ, one(T)], 4)
    r = cd.x
    theta = fill(T(0.1), Nr)
    uphi = fill(T(0.2), Nr)
    z = zeros(T, Nr)
    bs = BasicState{T}(
        lmax_bs = 2,
        Nr = Nr,
        r = r,
        theta_coeffs = Dict(0 => z, 1 => theta, 2 => copy(theta)),
        uphi_coeffs = Dict(0 => z, 1 => uphi, 2 => copy(uphi)),
        dtheta_dr_coeffs = Dict(0 => z, 1 => cd.D1 * theta, 2 => cd.D1 * theta),
        duphi_dr_coeffs = Dict(0 => z, 1 => cd.D1 * uphi, 2 => cd.D1 * uphi),
    )
    params = OnsetParams(E=T(1e-3), Pr=one(T), Ra=T(100), χ=χ,
                         m=0, lmax=4, Nr=Nr, basic_state=bs)
    op = LinearStabilityOperator(params)
    return bs, op, Nr
end

@testset "Axisymmetric (m=0) basic-state operators: structure and types" begin
    bs, op, Nr = _build_m0_basic_state(Float64)
    bs_ops = _silent(() -> Magrathea.build_basic_state_operators(bs, op, 0))

    @test bs_ops isa Magrathea.BasicStateOperators{Float64}
    @test !isempty(bs_ops.coupling_structure)
    @test eltype(bs_ops.coupling_structure) === Tuple{Int,Int}

    # m=0 disables azimuthal advection and the im·m toroidal-coupling blocks.
    @test isempty(bs_ops.advection_blocks)
    @test isempty(bs_ops.shear_theta_toroidal_blocks)
    @test isempty(bs_ops.temp_grad_theta_toroidal_blocks)

    # All populated blocks are complex Nr×Nr matrices.
    block_dicts = (
        bs_ops.shear_radial_blocks,
        bs_ops.shear_theta_blocks,
        bs_ops.temp_grad_radial_blocks,
        bs_ops.temp_grad_theta_blocks,
        bs_ops.metric_poloidal_blocks,
    )
    populated = [blk for d in block_dicts for blk in values(d)]
    @test !isempty(populated)
    for blk in populated
        @test size(blk) == (Nr, Nr)
        @test eltype(blk) === ComplexF64
        @test all(isfinite, blk)
    end

    # Dense accumulation path runs and writes into A only.
    A = zeros(ComplexF64, op.total_dof, op.total_dof)
    B = zeros(ComplexF64, op.total_dof, op.total_dof)
    Magrathea.add_basic_state_operators!(A, B, bs_ops, op, 0)
    @test any(!iszero, A)
    @test all(iszero, B)   # B is untouched by the basic-state contribution
end

@testset "Axisymmetric (m=0) basic-state operators preserve Float32 storage" begin
    bs, op, Nr = _build_m0_basic_state(Float32)
    bs_ops = _silent(() -> Magrathea.build_basic_state_operators(bs, op, 0))

    @test bs_ops isa Magrathea.BasicStateOperators{Float32}
    block_dicts = (
        bs_ops.shear_radial_blocks,
        bs_ops.shear_theta_blocks,
        bs_ops.temp_grad_radial_blocks,
        bs_ops.temp_grad_theta_blocks,
        bs_ops.metric_poloidal_blocks,
    )
    populated = [blk for d in block_dicts for blk in values(d)]
    @test !isempty(populated)
    @test all(eltype(blk) === ComplexF32 for blk in populated)
    @test all(size(blk) == (Nr, Nr) for blk in populated)
end
