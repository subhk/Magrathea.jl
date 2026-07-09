using Test
using LinearAlgebra
using SparseArrays
using Magrathea

# =============================================================================
# Coverage for NON-SOLVE code paths in:
#   - src/solve.jl                 (memory checks, find_critical_Ra dispatch,
#                                   _eigvecs_to_matrix shapes, perturbation_*
#                                   error branches, SLEPc-extension-absent throw)
#   - src/Stability/onset.jl       (find_critical_Ra_onset bracketing/promotion,
#                                   find_global_critical_onset validation +
#                                   "no valid results" error path)
#   - src/Stability/biglobal.jl    (BiglobalParams edge cases, basic-state builder
#                                   helpers, find_critical_Ra_biglobal error paths,
#                                   sweep early-return)
#
# HARD CONSTRAINT: no real eigensolve / SLEPc / PETSc / MPI is ever performed.
# The only solver contact is asserting that backend=:slepc throws the
# extension-absent ErrorException (which mentions SlepcWrap). All other tests
# exercise construction / validation / bracketing logic that runs *before* any
# eigensolve. Coefficient VALUES for m != 0 are never asserted (known bugs);
# m != 0 appears only structurally. Tests prefer m = 0 / axisymmetric.
#
# This file deliberately covers DIFFERENT functions/branches from
# test_onset_solve_helpers.jl and test_solve_assembly_coverage.jl.
# =============================================================================

# Sentinel exception used to prove a basic-state builder is invoked (and its
# error propagates) BEFORE any eigensolve. Defined at top level (struct
# definitions are illegal inside @testset's function scope).
struct _BuilderSentinel <: Exception end

# Helper: assert an expression throws an ErrorException whose message mentions
# the missing SLEPc extension (i.e. the eigensolve was reached but no real solve
# occurred). Returns true so it can be wrapped in @test.
function _throws_slepc_absent(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    return err isa ErrorException && occursin("SlepcWrap", sprint(showerror, err))
end

# -----------------------------------------------------------------------------
# solve.jl :: _eigvecs_to_matrix — all three multiple-dispatch methods
# -----------------------------------------------------------------------------
@testset "_eigvecs_to_matrix dispatch methods" begin
    vals = ComplexF64[1.0 + 0im, 2.0 - 1im]

    # (1) matrix input → converted to Matrix{Complex{T}} unchanged in shape
    M_in = ComplexF64[1 2; 3 4; 5 6]
    M_out = Magrathea._eigvecs_to_matrix(vals, M_in, Float64)
    @test M_out isa Matrix{ComplexF64}
    @test size(M_out) == (3, 2)
    @test M_out == M_in

    # Float32 target type is honoured for the matrix path
    M_out32 = Magrathea._eigvecs_to_matrix(vals, M_in, Float32)
    @test M_out32 isa Matrix{ComplexF32}
    @test size(M_out32) == (3, 2)

    # (2) vector-of-vectors input → columns packed into a matrix
    vv = [ComplexF64[1, 2, 3], ComplexF64[4, 5, 6]]
    Mvv = Magrathea._eigvecs_to_matrix(vals, vv, Float64)
    @test Mvv isa Matrix{ComplexF64}
    @test size(Mvv) == (3, 2)
    @test Mvv[:, 1] == vv[1]
    @test Mvv[:, 2] == vv[2]

    # (2b) empty vector-of-vectors → 0 × length(eigenvalues)
    empty_vv = Vector{Vector{ComplexF64}}()
    Mempty = Magrathea._eigvecs_to_matrix(vals, empty_vv, Float64)
    @test size(Mempty) == (0, 2)
    @test isempty(Mempty)

    # (3) fallback for a container matching neither concrete shape (e.g. a tuple)
    Mfb = Magrathea._eigvecs_to_matrix(vals, (1, 2, 3), Float64)
    @test Mfb isa Matrix{ComplexF64}
    @test size(Mfb) == (0, 2)
    @test isempty(Mfb)
end

# -----------------------------------------------------------------------------
# solve.jl :: _check_memory — biglobal & triglobal dispatch (warn + no-warn)
# (test_onset_solve_helpers covers Onset/MHD warn; here exercise Biglobal warn
#  and the Triglobal/Biglobal "no warn" returns-nothing paths.)
# -----------------------------------------------------------------------------
@testset "_check_memory Biglobal & Triglobal dispatch" begin
    op_small = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=6, Nr=16)
    bs = Magrathea.basic_state(op_small; mode=:conduction)
    bp = BiglobalProblem(op_small, bs)

    # modest biglobal problem: returns nothing, emits no 8 GB warning
    @test Magrathea._check_memory(bp, "BiglobalProblem") === nothing

    # huge biglobal problem: forwards the 8 GB memory warning (built, NEVER solved)
    op_huge = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=250, Nr=64)
    bs_huge = Magrathea.basic_state(op_huge; mode=:conduction)
    bp_huge = BiglobalProblem(op_huge, bs_huge)
    @test_logs (:warn, r"exceeds 8 GB") Magrathea._check_memory(bp_huge, "BiglobalProblem")

    # Triglobal dispatch (axisymmetric basic state, m_range includes m=0):
    T = Float64; Nr = 16; χ = 0.35; lmax_bs = 6
    cd = ChebyshevDiffn(Nr, T[χ, 1.0], 4)
    coeffs = Dict{Tuple{Int,Int}, Vector{T}}((ℓ, 0) => zeros(T, Nr) for ℓ in 0:lmax_bs)
    emptyd = Dict{Tuple{Int,Int}, Vector{T}}()
    bs3d = BasicState3D{T}(
        lmax_bs = lmax_bs, mmax_bs = 0, Nr = Nr, r = cd.x,
        theta_coeffs = coeffs, dtheta_dr_coeffs = Dict(coeffs),
        ur_coeffs = emptyd, utheta_coeffs = emptyd, uphi_coeffs = Dict(coeffs),
        dur_dr_coeffs = emptyd, dutheta_dr_coeffs = emptyd, duphi_dr_coeffs = Dict(coeffs))
    tp = TriglobalProblem(op_small, bs3d, 0:2)
    @test Magrathea._check_memory(tp, "TriglobalProblem") === nothing
end

# -----------------------------------------------------------------------------
# solve.jl :: find_critical_Ra(::MHDProblem) — unsupported-API error
# -----------------------------------------------------------------------------
@testset "find_critical_Ra(::MHDProblem) errors with guidance" begin
    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=0.0, ricb=0.35,
                   m=1, lmax=4, N=16, B0_type=no_field)
    problem = MHDProblem(mp)
    err = try
        find_critical_Ra(problem)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("not supported for MHDProblem", msg)
    @test occursin("solve(MHDProblem", msg)
    # @test_throws form too, for the dispatch line
    @test_throws ErrorException find_critical_Ra(problem; Ra_guess=1e5)
end

# -----------------------------------------------------------------------------
# solve.jl :: find_critical_Ra(::OnsetProblem) — dispatch reaches the eigensolve
# and throws the SLEPc-extension-absent error (covers the wrapper + kwarg path,
# including the Ra_bracket forwarding branch).  m = 0 axisymmetric.
# -----------------------------------------------------------------------------
@testset "find_critical_Ra(::OnsetProblem) dispatch → SLEPc throw" begin
    op = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=6, Nr=12)
    problem = OnsetProblem(op)

    # default bracket
    @test _throws_slepc_absent(() -> find_critical_Ra(problem; Ra_guess=1e3))
    # explicit Ra_bracket kwarg is forwarded through to find_critical_Ra_onset
    @test _throws_slepc_absent(() ->
        find_critical_Ra(problem; Ra_guess=1e3, Ra_bracket=(1e2, 1e4)))
end

# -----------------------------------------------------------------------------
# onset.jl :: find_critical_Ra_onset — bracketing / promotion logic before solve
# Every call ultimately reaches find_growth_rate → SLEPc throw, but the kwargs
# below exercise the bracket-construction, float-promotion, and verbose branches
# that run first.  m = 0.
# -----------------------------------------------------------------------------
@testset "find_critical_Ra_onset bracketing & promotion paths" begin
    # default bracket = (Ra_guess/10, Ra_guess*10)
    @test _throws_slepc_absent(() -> Magrathea.find_critical_Ra_onset(
        E=1e-3, Pr=1.0, χ=0.35, m=0, lmax=6, Nr=12, Ra_guess=1e3))

    # explicit Ra_bracket branch (Ra_bracket !== nothing)
    @test _throws_slepc_absent(() -> Magrathea.find_critical_Ra_onset(
        E=1e-3, Pr=1.0, χ=0.35, m=0, lmax=6, Nr=12,
        Ra_guess=1e3, Ra_bracket=(1e2, 1e4)))

    # mixed-type scalars exercise float(promote_type(...)) promotion to Float64
    @test _throws_slepc_absent(() -> Magrathea.find_critical_Ra_onset(
        E=1//1000, Pr=1, χ=0.35f0, m=0, lmax=6, Nr=12, Ra_guess=1000))

    # non-default boundary conditions are threaded through to the operator builder
    @test _throws_slepc_absent(() -> Magrathea.find_critical_Ra_onset(
        E=1e-3, Pr=1.0, χ=0.35, m=0, lmax=6, Nr=12, Ra_guess=1e3,
        mechanical_bc=:stress_free, thermal_bc=:fixed_flux,
        equatorial_symmetry=:symmetric))
end

# -----------------------------------------------------------------------------
# onset.jl :: find_global_critical_onset — validation throws BEFORE any solve
# -----------------------------------------------------------------------------
@testset "find_global_critical_onset validation (pre-solve)" begin
    # negative m in the range → ArgumentError before any eigensolve
    @test_throws ArgumentError Magrathea.find_global_critical_onset(
        E=1e-3, Pr=1.0, χ=0.35, lmax=6, Nr=12, m_range=-1:1, verbose=false)

    # invalid equatorial_symmetry → ArgumentError before any eigensolve
    @test_throws ArgumentError Magrathea.find_global_critical_onset(
        E=1e-3, Pr=1.0, χ=0.35, lmax=6, Nr=12, m_range=0:2,
        equatorial_symmetry=:nope, verbose=false)
end

# -----------------------------------------------------------------------------
# onset.jl :: find_global_critical_onset — every m fails (SLEPc absent) → each is
# caught and recorded as NaN, so valid_results is empty → "No valid results"
# error.  Exercises the per-m try/catch, the Dict bookkeeping, and the
# isempty(valid_results) error branch — all without a real solve.
# -----------------------------------------------------------------------------
@testset "find_global_critical_onset no-valid-results error path" begin
    err = try
        Magrathea.find_global_critical_onset(
            E=1e-3, Pr=1.0, χ=0.35, lmax=6, Nr=12, m_range=0:1, verbose=false)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("No valid results", sprint(showerror, err))
end

# -----------------------------------------------------------------------------
# biglobal.jl :: BiglobalParams — construction + validation error branches
# (not covered by existing helper tests; m = 0 axisymmetric basic state).
# -----------------------------------------------------------------------------
@testset "BiglobalParams construction & validation" begin
    χ = 0.35; Nr = 16
    bs, _ = create_conduction_basic_state(χ, Nr)

    # valid keyword construction
    bp = BiglobalParams(E=1e-3, Pr=1.0, Ra=1e4, χ=χ, m=0, lmax=8, Nr=Nr,
                        basic_state=bs)
    @test bp isa BiglobalParams{Float64}
    @test bp.m == 0
    @test bp.basic_state === bs
    @test bp.mechanical_bc == :no_slip          # default
    @test bp.thermal_bc == :fixed_temperature   # default
    @test bp.equatorial_symmetry == :both       # default

    # Ra == 0 is allowed (Ra >= 0)
    bp0 = BiglobalParams(E=1e-3, Pr=1.0, Ra=0.0, χ=χ, m=0, lmax=8, Nr=Nr,
                         basic_state=bs)
    @test bp0.Ra == 0.0

    # --- validation throws (positional inner constructor) ---
    base(; kw...) = BiglobalParams{Float64}(
        get(kw, :E, 1e-3), get(kw, :Pr, 1.0), get(kw, :Ra, 1e4),
        get(kw, :χ, χ), get(kw, :m, 0), get(kw, :lmax, 8), get(kw, :Nr, Nr),
        get(kw, :basic_state, bs),
        get(kw, :mechanical_bc, :no_slip), get(kw, :thermal_bc, :fixed_temperature),
        get(kw, :equatorial_symmetry, :both))

    @test base() isa BiglobalParams{Float64}              # sanity: base is valid

    @test_throws ArgumentError base(χ=0.0)                # χ outside (0,1)
    @test_throws ArgumentError base(χ=1.0)
    @test_throws ArgumentError base(E=0.0)                # E must be positive
    @test_throws ArgumentError base(E=-1e-3)
    @test_throws ArgumentError base(Pr=0.0)               # Pr must be positive
    @test_throws ArgumentError base(Ra=-1.0)              # Ra must be non-negative
    @test_throws ArgumentError base(m=-1)                 # m must be non-negative
    @test_throws ArgumentError base(m=20, lmax=8)         # lmax >= m
    @test_throws ArgumentError base(Nr=4)                 # Nr >= 8 (also grid mismatch)
    @test_throws ArgumentError base(mechanical_bc=:bad)
    @test_throws ArgumentError base(thermal_bc=:bad)
    @test_throws ArgumentError base(equatorial_symmetry=:bad)
end

# -----------------------------------------------------------------------------
# biglobal.jl :: BiglobalParams — basic-state / grid consistency validation
# -----------------------------------------------------------------------------
@testset "BiglobalParams basic_state grid consistency" begin
    χ = 0.35; Nr = 16
    bs, _ = create_conduction_basic_state(χ, Nr)

    # basic_state.Nr must match Nr — pass a state built for Nr=16 against Nr=24
    @test_throws ArgumentError BiglobalParams(
        E=1e-3, Pr=1.0, Ra=1e4, χ=χ, m=0, lmax=8, Nr=24, basic_state=bs)

    # χ mismatch: basic_state built for χ=0.35, params claim χ=0.4 (grid start
    # no longer equals χ) → ArgumentError
    @test_throws ArgumentError BiglobalParams(
        E=1e-3, Pr=1.0, Ra=1e4, χ=0.4, m=0, lmax=8, Nr=Nr, basic_state=bs)
end

# -----------------------------------------------------------------------------
# biglobal.jl :: create_conduction_basic_state / create_custom_basic_state
# -----------------------------------------------------------------------------
@testset "basic-state constructors (no flow)" begin
    χ = 0.35; Nr = 16

    bs, cd = create_conduction_basic_state(χ, Nr; lmax_bs=4)
    @test bs isa BasicState
    @test cd isa ChebyshevDiffn
    @test bs.Nr == Nr
    @test bs.lmax_bs == 4
    @test length(bs.r) == Nr
    @test bs.r ≈ cd.x

    # create_custom_basic_state from radial-profile callbacks on the Chebyshev grid
    r = cd.x
    θ_prof(rgrid, ℓ) = ℓ == 0 ? collect(1.0 .- rgrid) : zeros(length(rgrid))
    uphi_prof(rgrid, ℓ) = zeros(length(rgrid))
    bsc = create_custom_basic_state(θ_prof, uphi_prof, collect(r); lmax_bs=3)
    @test bsc isa BasicState
    @test bsc.Nr == Nr
    @test bsc.lmax_bs == 3
    @test haskey(bsc.theta_coeffs, 0)
    @test haskey(bsc.dtheta_dr_coeffs, 0)         # derivative was computed
    @test length(bsc.theta_coeffs[0]) == Nr

    # off-grid radial points are rejected
    bad_r = collect(range(χ, 1.0; length=Nr))     # uniform grid, not Chebyshev
    @test_throws ArgumentError create_custom_basic_state(θ_prof, uphi_prof, bad_r)
end

# -----------------------------------------------------------------------------
# biglobal.jl :: _biglobal_rayleigh_kwargs — assembles solver kwargs (helper)
# -----------------------------------------------------------------------------
@testset "_biglobal_rayleigh_kwargs builder wrapping" begin
    χ = 0.35; Nr = 16
    bs, _ = create_conduction_basic_state(χ, Nr)

    # fixed basic_state path: basic_state key present, no builder
    kw1 = Magrathea._biglobal_rayleigh_kwargs(
        :no_slip, :fixed_temperature, :both, 6, bs, nothing)
    @test kw1.basic_state === bs
    @test kw1.nev == 6
    @test kw1.mechanical_bc == :no_slip
    @test !haskey(kw1, :basic_state_builder)

    # builder path: wrapper validates return type — a good builder returns a BasicState
    good_builder = Ra -> bs
    kw2 = Magrathea._biglobal_rayleigh_kwargs(
        :stress_free, :fixed_flux, :symmetric, 4, nothing, good_builder)
    @test haskey(kw2, :basic_state_builder)
    @test !haskey(kw2, :basic_state)
    @test kw2.basic_state_builder(1e5) === bs        # wrapper passes BasicState through
    @test kw2.mechanical_bc == :stress_free

    # builder returning a non-BasicState → wrapper errors (no solve involved)
    bad_builder = Ra -> 42
    kw3 = Magrathea._biglobal_rayleigh_kwargs(
        :no_slip, :fixed_temperature, :both, 6, nothing, bad_builder)
    @test_throws ErrorException kw3.basic_state_builder(1e5)
    err = try kw3.basic_state_builder(1e5); catch e; e end
    @test occursin("must return a BasicState", sprint(showerror, err))
end

# -----------------------------------------------------------------------------
# biglobal.jl :: find_critical_Ra_biglobal — argument-validation error branches
# (these run BEFORE any operator is built / any solve is attempted).
# -----------------------------------------------------------------------------
@testset "find_critical_Ra_biglobal argument validation" begin
    χ = 0.35; Nr = 16
    bs, _ = create_conduction_basic_state(χ, Nr)

    # neither basic_state nor builder → error
    err1 = try
        Magrathea.find_critical_Ra_biglobal(E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr)
        nothing
    catch e; e end
    @test err1 isa ErrorException
    @test occursin("requires either", sprint(showerror, err1))

    # both basic_state AND builder → error
    err2 = try
        Magrathea.find_critical_Ra_biglobal(E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr,
                                        basic_state=bs, basic_state_builder=(Ra -> bs))
        nothing
    catch e; e end
    @test err2 isa ErrorException
    @test occursin("only one", sprint(showerror, err2))

    # invalid equatorial_symmetry → ArgumentError (before solve)
    @test_throws ArgumentError Magrathea.find_critical_Ra_biglobal(
        E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr,
        basic_state=bs, equatorial_symmetry=:nope)
end

# -----------------------------------------------------------------------------
# biglobal.jl :: find_critical_Ra_biglobal — builder is INVOKED before the solve;
# a builder that throws / returns the wrong type surfaces *that* error, not the
# SLEPc error.  This proves we cover the bracketing+builder path with no solve.
# -----------------------------------------------------------------------------
@testset "find_critical_Ra_biglobal builder error precedes solve" begin
    χ = 0.35; Nr = 16

    # builder returns a non-BasicState → wrapper error fires inside the first
    # bracket sample, BEFORE find_growth_rate / SLEPc is ever reached
    err_type = try
        Magrathea.find_critical_Ra_biglobal(
            E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr,
            basic_state_builder=(Ra -> "not a basic state"),
            Ra_guess=1e4, verbose=false)
        nothing
    catch e; e end
    @test err_type isa ErrorException
    @test occursin("must return a BasicState", sprint(showerror, err_type))
    @test !occursin("SlepcWrap", sprint(showerror, err_type))  # no solve reached

    # builder that throws its own sentinel error → propagates ahead of any solve
    err_sentinel = try
        Magrathea.find_critical_Ra_biglobal(
            E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr,
            basic_state_builder=(Ra -> throw(_BuilderSentinel())),
            Ra_guess=1e4, verbose=false)
        nothing
    catch e; e end
    @test err_sentinel isa _BuilderSentinel

    # With a *valid* fixed basic_state the path proceeds to the eigensolve and
    # surfaces the SLEPc-extension-absent error (covers the rayleigh_kwargs +
    # promotion lines up to the throw).
    bs, _ = create_conduction_basic_state(χ, Nr)
    @test _throws_slepc_absent(() -> Magrathea.find_critical_Ra_biglobal(
        E=1e-3, Pr=1.0, χ=χ, m=0, lmax=8, Nr=Nr,
        basic_state=bs, Ra_guess=1e4, Ra_bracket=(1e3, 1e5), verbose=false))
end

# -----------------------------------------------------------------------------
# biglobal.jl :: sweep_thermal_wind_amplitude — empty amplitudes → early return
# (no operator built, no solve).  Non-empty would reach a solve, so only the
# empty branch is exercised here.
# -----------------------------------------------------------------------------
@testset "sweep_thermal_wind_amplitude empty early-return" begin
    res = sweep_thermal_wind_amplitude(
        E=1e-3, Pr=1.0, χ=0.35, m=0, lmax=8, Nr=16, Ra=1e4,
        amplitudes=Float64[], verbose=false)
    @test isempty(res)
    @test res isa AbstractVector
end

# -----------------------------------------------------------------------------
# biglobal.jl :: analyze_basic_state — diagnostic summary (no solve)
# -----------------------------------------------------------------------------
@testset "analyze_basic_state summary" begin
    χ = 0.35; Nr = 16
    bs, _ = create_conduction_basic_state(χ, Nr; lmax_bs=4)
    summary = analyze_basic_state(bs; verbose=false)
    @test summary isa Dict
    @test haskey(summary, 0)
    for (ℓ, s) in summary
        @test s.θ_max >= 0
        @test s.uphi_max >= 0          # conduction state has zero flow
    end
    @test summary[0].uphi_max == 0.0   # conduction: no zonal flow
end

# -----------------------------------------------------------------------------
# solve.jl :: perturbation_velocity / perturbation_temperature — missing-operator
# error branch.  Build a StabilityResult whose `extra` has NO :operator field and
# assert the guarded error (no solve, no reconstruction performed).
# -----------------------------------------------------------------------------
@testset "perturbation_* missing-operator error branch" begin
    op = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=8, Nr=16)
    problem = OnsetProblem(op)
    eigvals = ComplexF64[0.1 + 0.0im, -0.2 + 1.0im]
    eigvecs = rand(ComplexF64, 12, 2)

    # extra deliberately lacks :operator (e.g. a triglobal-style result)
    result = StabilityResult(eigvals, eigvecs, problem;
                             extra=(coupled_modes=[0, 1],))
    @test !hasproperty(result.extra, :operator)

    err_v = try
        perturbation_velocity(result, 1)
        nothing
    catch e; e end
    @test err_v isa ErrorException
    @test occursin("no :operator", sprint(showerror, err_v))

    err_t = try
        perturbation_temperature(result, 1)
        nothing
    catch e; e end
    @test err_t isa ErrorException
    @test occursin(":operator", sprint(showerror, err_t))
end

# -----------------------------------------------------------------------------
# solve.jl :: solve(...; backend=:slepc) — extension-absent throw for the
# OnsetProblem and BiglobalProblem dispatches (m = 0 axisymmetric). This drives
# the assembly/dispatch lines up to the SLEPc stage, then asserts the SlepcWrap
# error specifically (stronger than the generic Exception used elsewhere).
# -----------------------------------------------------------------------------
@testset "solve(...; backend=:slepc) hits SlepcWrap-absent error" begin
    op = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=0, lmax=6, Nr=12)

    @test _throws_slepc_absent(() -> solve(OnsetProblem(op); nev=2, backend=:slepc))

    bs = Magrathea.basic_state(op; mode=:conduction)
    @test _throws_slepc_absent(() ->
        solve(BiglobalProblem(op, bs); nev=2, backend=:slepc))
end
