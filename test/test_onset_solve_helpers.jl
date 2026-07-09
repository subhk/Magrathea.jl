using Test
using Logging
using Magrathea

# =============================================================================
# Coverage for non-eigensolve helpers in src/Stability/onset.jl and src/solve.jl
#
# These tests exercise ONLY construction / estimation / validation paths. No
# eigensolver, solve(), SLEPc, PETSc, or MPI is ever invoked.
# =============================================================================

# -----------------------------------------------------------------------------
# OnsetConvectionParams — keyword constructor
# -----------------------------------------------------------------------------
@testset "OnsetConvectionParams keyword constructor" begin
    p = OnsetConvectionParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    @test p isa OnsetConvectionParams{Float64}
    @test p.E == 1e-3
    @test p.Pr == 1.0
    @test p.Ra == 100.0
    @test p.χ == 0.35
    @test p.m == 4
    @test p.lmax == 10
    @test p.Nr == 16
    # defaults
    @test p.mechanical_bc == :no_slip
    @test p.thermal_bc == :fixed_temperature
    @test p.equatorial_symmetry == :both

    # explicit type-parameter keyword form
    pt = OnsetConvectionParams{Float64}(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                                        m=4, lmax=10, Nr=16)
    @test pt isa OnsetConvectionParams{Float64}
    @test pt.m == 4

    # non-default boundary conditions are honoured
    p2 = OnsetConvectionParams(E=1e-4, Pr=0.7, Ra=5e4, χ=0.4, m=2, lmax=12, Nr=24,
                               mechanical_bc=:stress_free,
                               thermal_bc=:fixed_flux,
                               equatorial_symmetry=:symmetric)
    @test p2.mechanical_bc == :stress_free
    @test p2.thermal_bc == :fixed_flux
    @test p2.equatorial_symmetry == :symmetric
end

# -----------------------------------------------------------------------------
# OnsetConvectionParams — positional inner constructor (valid)
# -----------------------------------------------------------------------------
@testset "OnsetConvectionParams positional inner constructor" begin
    p = OnsetConvectionParams{Float64}(1e-3, 1.0, 100.0, 0.35, 4, 10, 16,
                                       :no_slip, :fixed_temperature, :both)
    @test p isa OnsetConvectionParams{Float64}
    @test p.E == 1e-3
    @test p.lmax == 10
    @test p.mechanical_bc == :no_slip

    # lmax == m boundary is allowed (lmax >= m)
    p_edge = OnsetConvectionParams{Float64}(1e-3, 1.0, 100.0, 0.35, 5, 5, 8,
                                            :no_slip, :fixed_temperature, :both)
    @test p_edge.lmax == 5
    @test p_edge.m == 5

    # m == 0 is allowed (m >= 0)
    p_m0 = OnsetConvectionParams{Float64}(1e-3, 1.0, 100.0, 0.35, 0, 10, 16,
                                          :no_slip, :fixed_temperature, :both)
    @test p_m0.m == 0
end

# -----------------------------------------------------------------------------
# OnsetConvectionParams — validation error branches
# -----------------------------------------------------------------------------
@testset "OnsetConvectionParams validation throws" begin
    base = (1e-3, 1.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :both)

    # χ outside (0,1)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.0, 4, 10, 16, :no_slip, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 1.0, 4, 10, 16, :no_slip, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 1.5, 4, 10, 16, :no_slip, :fixed_temperature, :both)

    # E must be positive
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        -1e-3, 1.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        0.0, 1.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :both)

    # Pr must be positive
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 0.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, -2.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :both)

    # m must be non-negative
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, -1, 10, 16, :no_slip, :fixed_temperature, :both)

    # lmax must be >= m
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 10, 4, 16, :no_slip, :fixed_temperature, :both)

    # Nr must be >= 8
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 4, 10, 4, :no_slip, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 4, 10, 7, :no_slip, :fixed_temperature, :both)

    # invalid boundary-condition symbols
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 4, 10, 16, :bad, :fixed_temperature, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 4, 10, 16, :no_slip, :bad, :both)
    @test_throws ArgumentError OnsetConvectionParams{Float64}(
        1e-3, 1.0, 100.0, 0.35, 4, 10, 16, :no_slip, :fixed_temperature, :bad)

    # sanity: the unmodified base tuple constructs fine
    @test OnsetConvectionParams{Float64}(base...) isa OnsetConvectionParams{Float64}
end

# -----------------------------------------------------------------------------
# OnsetConvectionParams — conversion from OnsetParams
# -----------------------------------------------------------------------------
@testset "OnsetConvectionParams from OnsetParams" begin
    op = OnsetParams(E=1e-4, Pr=0.7, Ra=5e4, χ=0.4, m=3, lmax=12, Nr=24,
                     mechanical_bc=:stress_free, thermal_bc=:fixed_flux,
                     equatorial_symmetry=:antisymmetric)
    cp = OnsetConvectionParams(op)

    @test cp isa OnsetConvectionParams{Float64}
    @test cp.E == op.E
    @test cp.Pr == op.Pr
    @test cp.Ra == op.Ra
    @test cp.χ == op.χ
    @test cp.m == op.m
    @test cp.lmax == op.lmax
    @test cp.Nr == op.Nr
    @test cp.mechanical_bc == op.mechanical_bc
    @test cp.thermal_bc == op.thermal_bc
    @test cp.equatorial_symmetry == op.equatorial_symmetry
end

# -----------------------------------------------------------------------------
# estimate_onset_problem_size
# -----------------------------------------------------------------------------
@testset "estimate_onset_problem_size" begin
    cp = OnsetConvectionParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=4, lmax=10, Nr=16)
    info = estimate_onset_problem_size(cp)

    # structure / positivity
    @test info.total_dofs > 0
    @test info.matrix_size > 0
    @test info.num_ell_modes > 0
    @test info.memory_estimate_mb > 0.0
    @test info.total_dofs == info.matrix_size

    # memory formula: two dense ComplexF64 (16-byte) matrices
    @test info.memory_estimate_mb ≈ 2 * info.matrix_size^2 * 16 / (1024^2)

    # with :both symmetry the layout is 3 equal field blocks of num_ell_modes × Nr
    @test info.total_dofs == 3 * info.num_ell_modes * cp.Nr
    @test info.num_ell_modes == cp.lmax - cp.m + 1

    # monotonicity in Nr and lmax
    cp_bigNr = OnsetConvectionParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                                     m=4, lmax=10, Nr=32)
    cp_bigL = OnsetConvectionParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                                    m=4, lmax=20, Nr=16)
    @test estimate_onset_problem_size(cp_bigNr).total_dofs > info.total_dofs
    @test estimate_onset_problem_size(cp_bigL).total_dofs > info.total_dofs
    @test estimate_onset_problem_size(cp_bigL).num_ell_modes > info.num_ell_modes
end

# -----------------------------------------------------------------------------
# onset_scaling_laws
# -----------------------------------------------------------------------------
@testset "onset_scaling_laws" begin
    E = 1e-4
    χ = 0.35

    sl = onset_scaling_laws(E, χ)                      # default bc = :no_slip
    @test sl.Ra_c > 0
    @test sl.m_c isa Int
    @test sl.m_c >= 0
    @test sl.ω_c > 0
    @test sl.δ > 0

    # exact asymptotic relations (no_slip coefficients)
    @test sl.Ra_c ≈ 6.0 * E^(-4/3)
    @test sl.ω_c  ≈ 0.4 * E^(-2/3)
    @test sl.δ    ≈ E^(1/3)
    @test sl.m_c  == round(Int, 0.5 * E^(-1/3))

    # stress_free branch uses smaller Ra coefficient, larger ω coefficient
    sf = onset_scaling_laws(E, χ; bc=:stress_free)
    @test sf.Ra_c ≈ 4.0 * E^(-4/3)
    @test sf.ω_c  ≈ 0.5 * E^(-2/3)
    @test sf.Ra_c < sl.Ra_c
    @test sf.ω_c  > sl.ω_c
    @test sf.δ    ≈ sl.δ          # δ scaling is bc-independent

    # monotonicity: smaller E ⇒ larger Ra_c, larger m_c, smaller δ
    sl_small = onset_scaling_laws(1e-6, χ)
    @test sl_small.Ra_c > sl.Ra_c
    @test sl_small.m_c  >= sl.m_c
    @test sl_small.δ    < sl.δ

    # Float32 inputs are accepted (rational exponents promote the result to Float64)
    sl32 = onset_scaling_laws(1.0f-4, 0.35f0)
    @test sl32.Ra_c > 0
    @test sl32.m_c isa Int
    @test sl32.δ > 0
end

# -----------------------------------------------------------------------------
# solve.jl size helpers (types.jl): _mem_gb, _hd_total_dof,
# _triglobal_total_dof, _mhd_total_dof
# -----------------------------------------------------------------------------
@testset "_mem_gb" begin
    @test Magrathea._mem_gb(1000) ≈ 2 * 1000^2 * 16 / (1024^3)
    @test Magrathea._mem_gb(2000) > Magrathea._mem_gb(1000)
    @test Magrathea._mem_gb(0) == 0.0
    @test Magrathea._mem_gb(1) > 0.0
end

@testset "_hd_total_dof" begin
    # :both symmetry → 3 equal field blocks of (lmax-m+1) × Nr
    @test Magrathea._hd_total_dof(2, 10, 16, :both) == 3 * (10 - 2 + 1) * 16
    @test Magrathea._hd_total_dof(0, 8, 16, :both) == 3 * (8 - 0 + 1) * 16

    # symmetric / antisymmetric truncations are positive and no larger than :both
    both = Magrathea._hd_total_dof(2, 12, 20, :both)
    sym  = Magrathea._hd_total_dof(2, 12, 20, :symmetric)
    anti = Magrathea._hd_total_dof(2, 12, 20, :antisymmetric)
    @test sym > 0
    @test anti > 0
    @test both >= sym
    @test both >= anti

    # monotonic in Nr and lmax
    @test Magrathea._hd_total_dof(2, 10, 32, :both) > Magrathea._hd_total_dof(2, 10, 16, :both)
    @test Magrathea._hd_total_dof(2, 20, 16, :both) > Magrathea._hd_total_dof(2, 10, 16, :both)
end

@testset "_triglobal_total_dof" begin
    total, per_m = Magrathea._triglobal_total_dof(0:2, 8, 16)
    @test total > 0
    @test per_m ≈ total / length(0:2)

    # default symmetry argument matches explicit :both
    total_b, _ = Magrathea._triglobal_total_dof(0:2, 8, 16, :both)
    @test total_b == total

    # more coupled modes ⇒ more total DOFs
    total_more, _ = Magrathea._triglobal_total_dof(0:4, 8, 16)
    @test total_more > total

    # validation: empty range and |m| > lmax both throw
    @test_throws ArgumentError Magrathea._triglobal_total_dof(2:1, 8, 16)
    @test_throws ArgumentError Magrathea._triglobal_total_dof(0:10, 8, 16)
end

@testset "_mhd_total_dof" begin
    # no background field → no magnetic blocks
    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.0, ricb=0.35,
                   m=1, lmax=4, N=16, B0_type=no_field)
    total, n_pol, n_tor, n_f, n_g, n_per_mode = Magrathea._mhd_total_dof(mp)
    @test n_pol > 0
    @test n_tor > 0
    @test n_f == 0
    @test n_g == 0
    @test n_per_mode == mp.N + 1
    # temperature shares poloidal parity; total = (n_pol + n_tor + n_h) × (N+1)
    @test total == (n_pol + n_tor + n_pol) * n_per_mode
    @test total > 0

    # dipole (symmB0 = -1) → magnetic blocks take opposite parity
    # (a background field requires Le > 0)
    mp_d = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.1, ricb=0.35,
                     m=1, lmax=4, N=16, B0_type=dipole, B0_amplitude=1.0)
    td, np_d, nt_d, nf_d, ng_d, nperm_d = Magrathea._mhd_total_dof(mp_d)
    @test nf_d == nt_d
    @test ng_d == np_d
    @test td == (np_d + nt_d + nf_d + ng_d + np_d) * nperm_d
    @test td > total      # five field blocks vs three
end

# -----------------------------------------------------------------------------
# Problem construction (construction only — never solved)
# -----------------------------------------------------------------------------
@testset "Problem construction" begin
    op = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=8, Nr=16)

    onset = OnsetProblem(op)
    @test onset isa OnsetProblem
    @test onset.params === op

    bs = Magrathea.basic_state(op; mode=:conduction)
    @test bs isa BasicState
    bp = BiglobalProblem(op, bs)
    @test bp isa BiglobalProblem
    @test bp.params === op
    @test bp.basic_state === bs

    # 3D basic state on the exact Chebyshev grid (so consistency validation passes)
    T = Float64; Nr = 16; χ = 0.35; lmax_bs = 8
    cd = ChebyshevDiffn(Nr, T[χ, 1.0], 4)
    coeffs = Dict{Tuple{Int,Int}, Vector{T}}((ℓ, 0) => zeros(T, Nr) for ℓ in 0:lmax_bs)
    emptyd = Dict{Tuple{Int,Int}, Vector{T}}()
    bs3d = BasicState3D{T}(
        lmax_bs = lmax_bs, mmax_bs = 0, Nr = Nr, r = cd.x,
        theta_coeffs = coeffs, dtheta_dr_coeffs = Dict(coeffs),
        ur_coeffs = emptyd, utheta_coeffs = emptyd, uphi_coeffs = Dict(coeffs),
        dur_dr_coeffs = emptyd, dutheta_dr_coeffs = emptyd, duphi_dr_coeffs = Dict(coeffs))
    tp = TriglobalProblem(op, bs3d, 0:2)
    @test tp isa TriglobalProblem
    @test tp.m_range == 0:2

    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.0, ricb=0.35,
                   m=1, lmax=4, N=16, B0_type=no_field)
    mhd = MHDProblem(mp)
    @test mhd isa MHDProblem
    @test mhd.params === mp
    @test mhd.basic_state === nothing

    # keep handles for the memory-check testset below
    global _onset_problem = onset
    global _biglobal_problem = bp
    global _triglobal_problem = tp
    global _mhd_problem = mhd
end

# -----------------------------------------------------------------------------
# _check_memory / _warn_if_large (no solve)
# -----------------------------------------------------------------------------
@testset "_check_memory returns nothing for modest problems" begin
    # all four dispatches run cleanly and emit no warning at this size
    @test Magrathea._check_memory(_onset_problem, "OnsetProblem") === nothing
    @test Magrathea._check_memory(_biglobal_problem, "BiglobalProblem") === nothing
    @test Magrathea._check_memory(_triglobal_problem, "TriglobalProblem") === nothing
    @test Magrathea._check_memory(_mhd_problem, "MHDProblem") === nothing

    @test_logs min_level=Logging.Warn Magrathea._check_memory(_onset_problem, "OnsetProblem")
    @test_logs min_level=Logging.Warn Magrathea._check_memory(_mhd_problem, "MHDProblem")
end

@testset "_check_memory warns above the 8 GB soft limit" begin
    # Deliberately huge dense problem — constructed but NEVER solved.
    # (Construction itself emits an unrelated angular-resolution warning, so the
    #  problem is built outside the @test_logs that asserts the memory warning.)
    huge = OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                                    m=2, lmax=250, Nr=64))
    @test_logs (:warn, r"exceeds 8 GB") Magrathea._check_memory(huge, "OnsetProblem")
    @test Magrathea._check_memory(huge, "OnsetProblem") === nothing
end

@testset "_warn_if_large" begin
    # modest problem: returns nothing, no warning
    @test Magrathea._warn_if_large(_onset_problem, "OnsetProblem") === nothing
    @test_logs min_level=Logging.Warn Magrathea._warn_if_large(_onset_problem, "OnsetProblem")

    # huge problem: forwards the memory warning
    huge = OnsetProblem(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                                    m=2, lmax=250, Nr=64))
    @test_logs (:warn, r"exceeds 8 GB") Magrathea._warn_if_large(huge, "OnsetProblem")

    # estimation failure is swallowed (downgraded to @debug) and returns nothing
    @test Magrathea._warn_if_large("not a problem", "Unsupported") === nothing
end
