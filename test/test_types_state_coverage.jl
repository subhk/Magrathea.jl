# =============================================================================
#  Targeted coverage for still-uncovered lines in:
#    - src/types.jl                          (basic_state(::OnsetParams) modes)
#    - src/MHD/types.jl                      (MHDParams validation, l-modes,
#                                             background-field profiles/operators)
#    - src/BasicStates/basic_state.jl        (SH show methods, interpolation,
#                                             Legendre pole branch, error paths)
#    - src/BasicStates/basic_state_operators.jl (dense + COO accumulation paths)
#
#  Scope is deliberately complementary to test_basicstate_coverage.jl,
#  test_bc_bsops_coverage.jl, test_types.jl, test_show.jl, and type_stability.jl:
#  every assertion here is construction / structural / error-path only. NO
#  eigensolver / solve() / SLEPc / PETSc / MPI is touched. Axisymmetric (m=0)
#  paths are preferred; for m≠0 mean-flow coupling only structural facts
#  (dims / eltype / no-throw / which matrix is written) are asserted — never
#  coupling COEFFICIENT VALUES (those carry known bugs).
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using Logging
using Magrathea

"""Run `f()` with @info/@debug logging suppressed (operator builders are chatty)."""
_silent_tsc(f) = with_logger(f, NullLogger())

const _TSC_CHI = 0.35
const _TSC_E   = 1e-4
const _TSC_RA  = 1e6
const _TSC_PR  = 1.0

# =============================================================================
#  src/types.jl — basic_state(params::OnsetParams; mode=…) dispatch
#  (covers the :meridional, :selfconsistent, :nonaxisymmetric, bad-mode branches
#   that are uncovered at types.jl:355-377)
# =============================================================================

@testset "basic_state(::OnsetParams): mode dispatch + error branch" begin
    params = OnsetParams(E=_TSC_E, Pr=_TSC_PR, Ra=_TSC_RA, χ=_TSC_CHI,
                         m=2, lmax=10, Nr=16)

    # :conduction (default) -> BasicState, no flow
    bs_c = basic_state(params)
    @test bs_c isa Magrathea.BasicState
    @test bs_c.Nr == params.Nr
    @test all(maximum(abs, v) == 0 for v in values(bs_c.uphi_coeffs))

    # :meridional -> axisymmetric thermal-wind BasicState (types.jl:355-358)
    bs_m = basic_state(params; mode=:meridional, amplitude=0.05, lmax_bs=4)
    @test bs_m isa Magrathea.BasicState
    @test bs_m.Nr == params.Nr
    @test haskey(bs_m.theta_coeffs, 2)
    @test maximum(abs, bs_m.theta_coeffs[2]) > 0   # Y20 forcing present

    # :selfconsistent (flux Y00 + Σ Y(2,m)) (types.jl:359-368). The non-axisymmetric
    # forcing (m=1,2) makes the returned state a BasicState3D; we assert only the
    # structural facts (state built, correct Nr), never coefficient values.
    bs_s = _silent_tsc() do
        basic_state(params; mode=:selfconsistent, amplitude=0.02, mmax_bs=2,
                    lmax_bs=4, max_iterations=1, tol=1e-6)
    end
    @test bs_s isa Union{Magrathea.BasicState, Magrathea.BasicState3D}
    @test bs_s.Nr == params.Nr

    # :nonaxisymmetric -> BasicState3D (types.jl:369-373). Structural only.
    bs_n = _silent_tsc() do
        basic_state(params; mode=:nonaxisymmetric, amplitude=0.02, mmax_bs=2,
                    lmax_bs=4)
    end
    @test bs_n isa Magrathea.BasicState3D
    @test bs_n.Nr == params.Nr
    @test bs_n.mmax_bs == 2
    @test eltype(bs_n.r) === Float64

    # Unknown mode -> ArgumentError (types.jl:374-377)
    @test_throws ArgumentError basic_state(params; mode=:totally_bogus)
end

# =============================================================================
#  src/MHD/types.jl — MHDParams hard-validation error branches
#  (MHD/types.jl:248-283)
# =============================================================================

@testset "MHDParams constructor: hard-validation error paths" begin
    base = (E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, ricb=0.35,
            m=1, lmax=4, N=16)

    # ricb outside (0,1)
    @test_throws ArgumentError MHDParams(; base..., ricb=0.0,  Le=0.0)
    @test_throws ArgumentError MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0,
                                         ricb=1.0, m=1, lmax=4, N=16, Le=0.0)
    # Non-positive physical numbers
    @test_throws ArgumentError MHDParams(; E=-1e-3, Pr=1.0, Pm=1.0, Ra=100.0,
                                         ricb=0.35, m=1, lmax=4, N=16, Le=0.0)
    @test_throws ArgumentError MHDParams(; E=1e-3, Pr=0.0, Pm=1.0, Ra=100.0,
                                         ricb=0.35, m=1, lmax=4, N=16, Le=0.0)
    @test_throws ArgumentError MHDParams(; E=1e-3, Pr=1.0, Pm=-1.0, Ra=100.0,
                                         ricb=0.35, m=1, lmax=4, N=16, Le=0.0)
    @test_throws ArgumentError MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=-1.0,
                                         ricb=0.35, m=1, lmax=4, N=16, Le=0.0)
    # lmax < m
    @test_throws ArgumentError MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0,
                                         ricb=0.35, m=5, lmax=4, N=16, Le=0.0)
    # N odd / too small
    @test_throws ArgumentError MHDParams(; base..., N=15, Le=0.0)
    @test_throws ArgumentError MHDParams(; base..., N=4,  Le=0.0)
    # invalid symm and heating symbols
    @test_throws ArgumentError MHDParams(; base..., symm=2,  Le=0.0)
    @test_throws ArgumentError MHDParams(; base..., heating=:bogus, Le=0.0)

    # Dipole field requires ricb > 0 (MHD/types.jl:268-270). ricb is validated
    # first, so the dedicated dipole/ricb message needs ricb in (0,1) but the
    # branch fires for the background-field/Le consistency rules below instead.

    # no_field with nonzero Le -> error (MHD/types.jl:272-275)
    @test_throws ErrorException MHDParams(; base..., B0_type=no_field, Le=1e-3)
    # no_field with nonzero B0_amplitude -> error (MHD/types.jl:273-274)
    @test_throws ErrorException MHDParams(; base..., B0_type=no_field, Le=0.0,
                                         B0_amplitude=1.0)
    # Background field present but Le=0 -> error (MHD/types.jl:276-278)
    @test_throws ErrorException MHDParams(; base..., B0_type=axial, Le=0.0)

    # Conducting magnetic BC requires Em>0; Em=E/Pm is always >0 for valid E,Pm,
    # so construct a valid conducting-BC case (exercises the non-error branch).
    p_cond = MHDParams(; base..., B0_type=axial, Le=1e-3, bci_magnetic=1)
    @test p_cond isa MHDParams{Float64}
    @test p_cond.Em > 0
end

@testset "MHDParams keyword constructor: promotion + derived quantities" begin
    # Mixed Float32/Float64 keyword inputs promote to a common type.
    p = MHDParams(; E=1.0f-3, Pr=1.0, Pm=2.0f0, Ra=100.0, Le=0.0,
                  ricb=0.35f0, m=1, lmax=4, N=16)
    @test p isa MHDParams
    @test p.L ≈ one(p.L) - p.ricb
    @test p.Etherm ≈ p.E / p.Pr
    @test p.Em ≈ p.E / p.Pm
end

# =============================================================================
#  src/MHD/types.jl — compute_mhd_l_modes parities (MHD/types.jl:732-753)
# =============================================================================

@testset "compute_mhd_l_modes: all three symmetry branches drop l=0" begin
    m, lmax = 1, 8

    ll_u_s, ll_v_s = Magrathea.compute_mhd_l_modes(m, lmax, 1, no_field)   # symmetric
    ll_u_a, ll_v_a = Magrathea.compute_mhd_l_modes(m, lmax, -1, no_field)  # antisym (736-739)
    ll_u_b, ll_v_b = Magrathea.compute_mhd_l_modes(m, lmax, 0, no_field)   # both (742-743)

    for v in (ll_u_s, ll_v_s, ll_u_a, ll_v_a, ll_u_b, ll_v_b)
        @test all(>=(1), v)            # degenerate l=0 always filtered
        @test issorted(v)
        @test all(l -> l <= lmax, v)
    end

    # symm=±1 give opposite parity assignment for poloidal vs toroidal
    @test ll_u_s == ll_v_a
    @test ll_v_s == ll_u_a
    # symm=0 keeps both parities (a superset of the parity-split versions)
    @test issubset(Set(ll_u_s), Set(ll_u_b))
    @test issubset(Set(ll_v_s), Set(ll_u_b))
end

@testset "_mhd_total_dof matches operator field counts (no_field + axial)" begin
    p_hydro = MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.0,
                        ricb=0.35, m=1, lmax=5, N=16)
    dof_h, n_pol_h, n_tor_h, n_f_h, n_g_h, n_per_h = Magrathea._mhd_total_dof(p_hydro)
    @test n_f_h == 0 && n_g_h == 0          # hydrodynamic: no magnetic blocks
    @test n_per_h == p_hydro.N + 1
    op_h = _silent_tsc(() -> MHDStabilityOperator(p_hydro))
    @test op_h.matrix_size == dof_h         # bookkeeping agrees with operator

    p_axial = MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1e-3,
                        ricb=0.35, m=1, lmax=5, N=16, B0_type=axial)
    dof_a, n_pol_a, n_tor_a, n_f_a, n_g_a, _ = Magrathea._mhd_total_dof(p_axial)
    # symmB0 = -1: magnetic blocks take OPPOSITE parity (n_f=n_tor, n_g=n_pol)
    @test n_f_a == n_tor_a
    @test n_g_a == n_pol_a
    op_a = _silent_tsc(() -> MHDStabilityOperator(p_axial))
    @test op_a.matrix_size == dof_a
end

# =============================================================================
#  src/MHD/types.jl — background_profile_value & sparse_background_operator
#  (MHD/types.jl:623-710)
# =============================================================================

@testset "background_profile_value: all field types and h-orders" begin
    # no_field -> 0 everywhere (MHD/types.jl:625-626)
    @test Magrathea.background_profile_value(0.7, no_field, 0, 2) == 0.0

    # axial: h_order 0 and 1 give the documented powers; 2,3 vanish (632-633)
    @test Magrathea.background_profile_value(0.7, axial, 0, 2) ≈ 0.7^(2 + 1) / 2
    @test Magrathea.background_profile_value(0.7, axial, 1, 2) ≈ 0.7^2 / 2
    @test Magrathea.background_profile_value(0.7, axial, 2, 2) == 0.0
    @test Magrathea.background_profile_value(0.7, axial, 3, 2) == 0.0
    # unsupported axial h-order errors (MHD/types.jl:634-635)
    @test_throws ErrorException Magrathea.background_profile_value(0.7, axial, 4, 2)

    # dipole: each h-order has its own coeff/exponent shift (637-657)
    @test Magrathea.background_profile_value(0.7, dipole, 0, 4) ≈ 0.5  * 0.7^(4 - 2)
    @test Magrathea.background_profile_value(0.7, dipole, 1, 4) ≈ -1.0 * 0.7^(4 - 3)
    @test Magrathea.background_profile_value(0.7, dipole, 2, 4) ≈ 3.0  * 0.7^(4 - 4)
    @test Magrathea.background_profile_value(0.7, dipole, 3, 5) ≈ -12.0 * 0.7^(5 - 5)
    # unsupported dipole h-order errors (MHD/types.jl:653-654)
    @test_throws ErrorException Magrathea.background_profile_value(0.7, dipole, 4, 4)
end

@testset "sparse_background_operator: no_field, error paths, axial high h-order" begin
    N = 8
    p_none  = MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.0,
                        ricb=0.35, m=1, lmax=4, N=N)
    p_axial = MHDParams(; E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1e-3,
                        ricb=0.35, m=1, lmax=4, N=N, B0_type=axial)

    # no_field -> spzeros of the right shape (MHD/types.jl:671-672)
    Z = Magrathea.sparse_background_operator(2, 0, 0, p_none)
    @test Z isa SparseMatrixCSC{Float64,Int}
    @test size(Z) == (N + 1, N + 1)
    @test nnz(Z) == 0

    # axial with h_order >= 2 -> spzeros (MHD/types.jl:683-684)
    Za = Magrathea.sparse_background_operator(2, 2, 0, p_axial)
    @test nnz(Za) == 0
    @test size(Za) == (N + 1, N + 1)

    # out-of-range h-order -> error (MHD/types.jl:679-680)
    @test_throws ErrorException Magrathea.sparse_background_operator(2, -1, 0, p_axial)
    @test_throws ErrorException Magrathea.sparse_background_operator(2, 99, 0, p_axial)

    # A populated axial operator (h_order 0, deriv_order 1) builds without throwing.
    op_real = Magrathea.sparse_background_operator(2, 0, 1, p_axial)
    @test op_real isa SparseMatrixCSC{Float64,Int}
    @test size(op_real) == (N + 1, N + 1)
    @test all(isfinite, nonzeros(op_real))
end

# =============================================================================
#  src/BasicStates/basic_state.jl — SphericalHarmonicBC show methods
#  (basic_state.jl:359-397)
# =============================================================================

@testset "SphericalHarmonicBC show: scaled / ±1 / zero / empty branches" begin
    # +1 and -1 amplitudes print as bare ±Yℓm (basic_state.jl:370-373)
    s_pm = Magrathea.Y20(1.0) + Magrathea.Y22(-1.0)
    txt_pm = sprint(show, s_pm)
    @test occursin("Y20", txt_pm)
    @test occursin("-Y22", txt_pm)

    # General amplitude prints "amp*Yℓm" (basic_state.jl:374-375)
    txt_scaled = sprint(show, Magrathea.Y20(0.25))
    @test occursin("0.25", txt_scaled)
    @test occursin("Y20", txt_scaled)

    # All-zero (non-empty) amplitudes -> "(zero)" sentinel (basic_state.jl:367-380)
    txt_zero = sprint(show, Magrathea.Y20(0.0))
    @test occursin("zero", txt_zero)

    # Empty BC -> "(empty)" sentinel (basic_state.jl:360-362)
    txt_empty = sprint(show, Magrathea.SphericalHarmonicBC{Float64}())
    @test occursin("empty", txt_empty)

    # MIME text/plain on empty -> "modes … none" branch (basic_state.jl:389-391)
    txt_plain_empty = sprint(show, MIME("text/plain"),
                             Magrathea.SphericalHarmonicBC{Float64}())
    @test occursin("SphericalHarmonicBC", txt_plain_empty)
    @test occursin("none", txt_plain_empty)

    # MIME text/plain on a populated BC lists each (ℓ,m) row.
    txt_plain = sprint(show, MIME("text/plain"), Magrathea.Y20(0.1) + Magrathea.Y33(0.2))
    @test occursin("Y_2,0", txt_plain)
    @test occursin("Y_3,3", txt_plain)
end

# =============================================================================
#  src/BasicStates/basic_state.jl — _linear_interpolate descending grid +
#  _legendre_values_and_derivs pole branch (basic_state.jl:892-912, 879-887)
# =============================================================================

@testset "_linear_interpolate: ascending, descending, and clamping" begin
    r_asc = [0.0, 1.0, 2.0, 3.0]
    v_asc = [10.0, 20.0, 30.0, 40.0]
    @test Magrathea._linear_interpolate(r_asc, v_asc, 1.5) ≈ 25.0   # midpoint
    @test Magrathea._linear_interpolate(r_asc, v_asc, -1.0) == 10.0 # clamp low
    @test Magrathea._linear_interpolate(r_asc, v_asc, 99.0) == 40.0 # clamp high

    # Descending grid hits the reverse branch (basic_state.jl:898-900)
    r_desc = [3.0, 2.0, 1.0, 0.0]
    v_desc = [40.0, 30.0, 20.0, 10.0]
    @test Magrathea._linear_interpolate(r_desc, v_desc, 1.5) ≈ 25.0
    @test Magrathea._linear_interpolate(r_desc, v_desc, 0.5) ≈ 15.0

    # Mismatched lengths -> DimensionMismatch
    @test_throws DimensionMismatch Magrathea._linear_interpolate([1.0, 2.0], [1.0], 1.0)
end

@testset "evaluate_basic_state at θ=0 exercises Legendre pole branch" begin
    Nr = 16
    cd = Magrathea.ChebyshevDiffn(Nr, [_TSC_CHI, 1.0], 2)
    bs = Magrathea.conduction_basic_state(cd, _TSC_CHI, 4)

    # θ=0 -> x=cos(0)=1 -> (1-x²)=0, so the |denom|<tol guard zeroes dPdx
    # (basic_state.jl:881-882). Still returns finite outputs.
    out = Magrathea.evaluate_basic_state(bs, 0.6, 0.0)
    @test out isa NamedTuple
    @test all(isfinite, (out.theta_bar, out.uphi_bar, out.dtheta_dr,
                         out.dtheta_dtheta, out.duphi_dr, out.duphi_dtheta))
    @test out.uphi_bar == 0.0      # conduction state has no flow
end

# =============================================================================
#  src/BasicStates/basic_state.jl — thermal-BC + mechanical-BC error paths
#  (basic_state.jl:582, 1022, 1498, 1684, 1965)
# =============================================================================

@testset "Basic-state builders reject invalid BC symbols" begin
    Nr = 16
    cd = Magrathea.ChebyshevDiffn(Nr, [_TSC_CHI, 1.0], 4)

    # meridional_basic_state invalid thermal_bc (basic_state.jl:581-582)
    @test_throws ErrorException Magrathea.meridional_basic_state(
        cd, _TSC_CHI, _TSC_E, _TSC_RA, _TSC_PR, 4, 0.1; thermal_bc=:bogus)

    # nonaxisymmetric_basic_state invalid thermal_bc (basic_state.jl:1021-1022)
    amps = Dict{Tuple{Int,Int},Float64}((2, 1) => 0.05)
    @test_throws ErrorException _silent_tsc() do
        Magrathea.nonaxisymmetric_basic_state(cd, _TSC_CHI, _TSC_E, _TSC_RA, _TSC_PR,
                                          4, 2, amps; thermal_bc=:bogus)
    end

    # solve_thermal_wind_balance! invalid mechanical_bc (basic_state.jl:1497-1498)
    uphi = Dict{Int,Vector{Float64}}()
    duphi = Dict{Int,Vector{Float64}}()
    theta = Dict{Int,Vector{Float64}}(1 => fill(0.05, Nr))
    @test_throws ErrorException Magrathea.solve_thermal_wind_balance!(
        uphi, duphi, theta, cd, _TSC_CHI, 1.0, _TSC_RA, _TSC_PR;
        mechanical_bc=:bogus, E=_TSC_E)

    # solve_thermal_wind_balance_3d! invalid mechanical_bc for m≠0
    # (basic_state.jl:1683-1684)
    uphi3 = Dict{Int,Vector{Float64}}()
    duphi3 = Dict{Int,Vector{Float64}}()
    theta3 = Dict{Int,Vector{Float64}}(2 => fill(0.05, Nr))
    @test_throws ErrorException Magrathea.solve_thermal_wind_balance_3d!(
        uphi3, duphi3, theta3, 1, cd, _TSC_CHI, 1.0, _TSC_RA, _TSC_PR;
        mechanical_bc=:bogus, E=_TSC_E)
end

@testset "solve_thermal_wind_balance!: stress-free BC + unforced-mode zero-fill" begin
    Nr = 20
    cd = Magrathea.ChebyshevDiffn(Nr, [_TSC_CHI, 1.0], 2)

    # ℓ=0 temperature has no θ-gradient, so its velocity mode is zero-filled
    # (basic_state.jl:1623-1625). ℓ=2 produces forced L=1,3 modes.
    uphi = Dict{Int,Vector{Float64}}()
    duphi = Dict{Int,Vector{Float64}}()
    theta = Dict{Int,Vector{Float64}}(
        0 => fill(1.0, Nr),
        2 => fill(0.1, Nr),
    )
    # stress-free path exercises the else-branch of _thermal_wind_operator_lu
    # (basic_state.jl:1426-1427)
    Magrathea.solve_thermal_wind_balance!(uphi, duphi, theta, cd, _TSC_CHI, 1.0,
                                      _TSC_RA, _TSC_PR; mechanical_bc=:stress_free,
                                      E=_TSC_E)
    @test haskey(uphi, 0)
    @test all(==(0.0), uphi[0])     # unforced ℓ=0 mode zeroed
    @test length(uphi[0]) == Nr
    @test all(isfinite, uphi[2]) || !haskey(uphi, 2)
    # at least one forced mode is non-trivial
    @test any(haskey(uphi, L) && maximum(abs, uphi[L]) > 0 for L in (1, 3))

    # no_slip variant also runs and zero-fills the unforced ℓ=0 mode.
    uphi2 = Dict{Int,Vector{Float64}}()
    duphi2 = Dict{Int,Vector{Float64}}()
    Magrathea.solve_thermal_wind_balance!(uphi2, duphi2, copy(theta), cd, _TSC_CHI,
                                      1.0, _TSC_RA, _TSC_PR;
                                      mechanical_bc=:no_slip, E=_TSC_E)
    @test all(==(0.0), uphi2[0])
end

# =============================================================================
#  src/BasicStates/basic_state_operators.jl — dense + COO accumulation paths
#  with m≠0 so the advection / toroidal-coupling blocks are populated.
#  Structural assertions only (no coefficient VALUES).
# =============================================================================

"""Build a small m≠0 axisymmetric-basic-state operator for accumulation tests."""
function _build_mneq0_bs_op(::Type{T}, m::Int) where {T<:Real}
    Nr = 12
    χ = T(_TSC_CHI)
    cd = Magrathea.ChebyshevDiffn(Nr, T[χ, one(T)], 4)
    r = cd.x
    theta = fill(T(0.1), Nr)
    uphi = fill(T(0.2), Nr)
    z = zeros(T, Nr)
    bs = Magrathea.BasicState{T}(
        lmax_bs = 2,
        Nr = Nr,
        r = r,
        theta_coeffs = Dict(0 => z, 1 => theta, 2 => copy(theta)),
        uphi_coeffs = Dict(0 => z, 1 => uphi, 2 => copy(uphi)),
        dtheta_dr_coeffs = Dict(0 => z, 1 => cd.D1 * theta, 2 => cd.D1 * theta),
        duphi_dr_coeffs = Dict(0 => z, 1 => cd.D1 * uphi, 2 => cd.D1 * uphi),
    )
    params = OnsetParams(E=T(1e-3), Pr=one(T), Ra=T(100), χ=χ,
                         m=m, lmax=4, Nr=Nr, basic_state=bs)
    op = LinearStabilityOperator(params)
    return bs, op, Nr
end

@testset "add_basic_state_operators! (m≠0): advection + toroidal blocks populate" begin
    bs, op, Nr = _build_mneq0_bs_op(Float64, 1)
    bs_ops = _silent_tsc(() -> Magrathea.build_basic_state_operators(bs, op, 1))

    # m≠0 enables azimuthal advection and the im·m toroidal-coupling blocks
    # (advection_blocks / *_toroidal_blocks were empty for m=0).
    @test !isempty(bs_ops.advection_blocks)
    @test !isempty(bs_ops.coupling_structure)

    populated = [blk for blk in values(bs_ops.advection_blocks)]
    for blk in populated
        @test size(blk) == (Nr, Nr)
        @test eltype(blk) === ComplexF64
        @test all(isfinite, blk)
    end

    # Dense accumulation path (covers advection P/T/Θ + toroidal branches,
    # basic_state_operators.jl:648-727).
    A = zeros(ComplexF64, op.total_dof, op.total_dof)
    B = zeros(ComplexF64, op.total_dof, op.total_dof)
    Magrathea.add_basic_state_operators!(A, B, bs_ops, op, 1)
    @test any(!iszero, A)
    @test all(iszero, B)               # basic-state contribution touches A only
    @test all(isfinite, A)
end

@testset "add_basic_state_operators_coo! (m≠0): COO triplets mirror dense path" begin
    bs, op, Nr = _build_mneq0_bs_op(Float64, 1)
    bs_ops = _silent_tsc(() -> Magrathea.build_basic_state_operators(bs, op, 1))

    # Dense reference.
    A_dense = zeros(ComplexF64, op.total_dof, op.total_dof)
    B_dense = zeros(ComplexF64, op.total_dof, op.total_dof)
    Magrathea.add_basic_state_operators!(A_dense, B_dense, bs_ops, op, 1)

    # COO emission (basic_state_operators.jl:746-861).
    A_rows = Int[]; A_cols = Int[]; A_vals = ComplexF64[]
    B_rows = Int[]; B_cols = Int[]; B_vals = ComplexF64[]
    Magrathea.add_basic_state_operators_coo!(A_rows, A_cols, A_vals,
                                         B_rows, B_cols, B_vals,
                                         bs_ops, op, 1)
    @test length(A_rows) == length(A_cols) == length(A_vals)
    @test !isempty(A_vals)
    @test all(isfinite, A_vals)
    @test isempty(B_vals)              # B arrays untouched by basic-state COO emit
    @test all(1 .<= A_rows .<= op.total_dof)
    @test all(1 .<= A_cols .<= op.total_dof)

    # Assembling the COO triplets reproduces the dense matrix exactly.
    A_from_coo = Matrix(sparse(A_rows, A_cols, A_vals,
                               op.total_dof, op.total_dof))
    @test A_from_coo ≈ A_dense

    # owned_julia_rows restriction drops triplets outside the range.
    owned = 1:(op.total_dof ÷ 2)
    A_rows2 = Int[]; A_cols2 = Int[]; A_vals2 = ComplexF64[]
    B_rows2 = Int[]; B_cols2 = Int[]; B_vals2 = ComplexF64[]
    Magrathea.add_basic_state_operators_coo!(A_rows2, A_cols2, A_vals2,
                                         B_rows2, B_cols2, B_vals2,
                                         bs_ops, op, 1; owned_julia_rows=owned)
    @test all(r -> r in owned, A_rows2)
    @test length(A_rows2) <= length(A_rows)
end

@testset "add_basic_state_operators! (m≠0) preserves Float32 storage" begin
    bs, op, Nr = _build_mneq0_bs_op(Float32, 1)
    bs_ops = _silent_tsc(() -> Magrathea.build_basic_state_operators(bs, op, 1))
    @test bs_ops isa Magrathea.BasicStateOperators{Float32}

    A = zeros(ComplexF32, op.total_dof, op.total_dof)
    B = zeros(ComplexF32, op.total_dof, op.total_dof)
    Magrathea.add_basic_state_operators!(A, B, bs_ops, op, 1)
    @test eltype(A) === ComplexF32
    @test any(!iszero, A)
    @test all(iszero, B)
end
