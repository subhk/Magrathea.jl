using Test
using Magrathea
using SparseArrays
using Random

# Regression tests for the 2026-06-16 CPU-correctness audit fixes (#1–#5).
# Each test is written to FAIL against the pre-fix code and PASS after the fix.

function _audit_basic_state_3d(::Type{T}, Nr::Int, χ; lmax_bs::Int=0, mmax_bs::Int=0) where {T<:Real}
    cd = ChebyshevDiffn(Nr, T[T(χ), one(T)], 1)
    empty = Dict{Tuple{Int,Int}, Vector{T}}()
    return BasicState3D{T}(
        lmax_bs = lmax_bs, mmax_bs = mmax_bs, Nr = Nr, r = cd.x,
        theta_coeffs = empty, dtheta_dr_coeffs = deepcopy(empty),
        ur_coeffs = deepcopy(empty), utheta_coeffs = deepcopy(empty),
        uphi_coeffs = deepcopy(empty), dur_dr_coeffs = deepcopy(empty),
        dutheta_dr_coeffs = deepcopy(empty), duphi_dr_coeffs = deepcopy(empty))
end

@testset "Audit fixes #1–#6" begin

    # ----- #1: theta_derivative_coupling = sinθ∂θ projection -----------------
    # Validated identity (basic_state.jl:2007-2008, quadrature-tested):
    #   sinθ ∂Yℓm/∂θ = +ℓ·α⁺ Y_{ℓ+1,m} − (ℓ+1)·α⁻ Y_{ℓ-1,m}
    # where (α⁻, α⁺) = sin_theta_coupling(ℓ,m) (the verified recurrence coeffs).
    @testset "#1 theta_derivative_coupling sinθ∂θ coefficients" begin
        for (ℓ, m) in [(1, 1), (2, 1), (3, 2), (2, 0), (4, 3), (5, 0)]
            A_minus, A_plus, A_diag = Magrathea.theta_derivative_coupling(ℓ, m)
            αminus, αplus = Magrathea.sin_theta_coupling(ℓ, m)
            @test A_plus  ≈ ℓ * αplus        atol=1e-12
            @test A_minus ≈ -(ℓ + 1) * αminus atol=1e-12
            @test A_diag  == 0
        end
    end

    # Conducting-core matching is now tested against analytic full-sphere decay
    # modes in mhd_physics.jl, rather than a prescribed-frequency Robin row.

    # ----- #4: triglobal reconstruction passes SIGNED m (not abs) ------------
    # Plant identical reduced blocks in m=+1 and m=−1. Pre-fix (abs m) the two
    # contributions are exact negatives → u_θ cancels to ~0. Signed m → nonzero.
    @testset "#4 velocity reconstruction signed-m (m<0 not cancelled)" begin
        Random.seed!(20260616)
        T = Float64; Nr = 8; χ = T(0.35)
        bs3d = _audit_basic_state_3d(T, Nr, χ)
        params = Magrathea.TriglobalParams(E=T(1e-3), Pr=one(T), Ra=T(100.0), χ=χ,
            m_range=-1:1, lmax=2, Nr=Nr, basic_state_3d=bs3d)
        problem = Magrathea.setup_coupled_mode_problem(params)

        ntot = maximum(last(rng) for rng in values(problem.block_indices))
        ev = zeros(ComplexF64, ntot)
        blk = randn(ComplexF64, length(problem.block_indices[1]))
        ev[problem.block_indices[1]]  .= blk
        ev[problem.block_indices[-1]] .= blk        # identical reduced block

        ur, uθ, uφ = Magrathea.eigenvector_to_velocity_triglobal(ev, problem; φ_slice=0.0)
        @test maximum(abs, uθ) > 1e-8
        @test maximum(abs, uφ) > 1e-8
    end

    # ----- #5: unweighted SH self-overlap must be positive -------------------
    # ∫|Y_{ℓm}|² (no sinθ weight) over a positive measure is strictly positive.
    # The spurious (-1)^m phase made it negative for odd m.
    @testset "#5 compute_sh_coupling_unweighted positivity" begin
        for (ℓ, m) in [(1, 1), (2, 1), (3, 3), (1, 0), (2, 2), (4, 1)]
            val = Magrathea.compute_sh_coupling_unweighted(ℓ, m, 0, 0, ℓ, m)
            @test val > 0
        end
    end

    # ----- #6: inv_sin_theta_gaunt = exact ⟨Y_Lm|1/sinθ|Y_ℓm⟩ = ∫₀^π P̄_Lm P̄_ℓm dθ ---
    @testset "#6 inv_sin_theta_gaunt is the exact 1/sinθ projection" begin
        # Analytic: ⟨Y₁₁|1/sinθ|Y₁₁⟩ = (3/4)∫₀^π sin²θ dθ = 3π/8 (heuristic gave 1.5).
        @test Magrathea.inv_sin_theta_gaunt(1, 1, 1) ≈ 3π / 8 rtol=1e-6
        # Couples ALL same-parity L, not only |Δℓ|=2: Δℓ=4 must be nonzero
        # (the heuristic returned exactly 0 here).
        @test abs(Magrathea.inv_sin_theta_gaunt(5, 1, 1)) > 1e-8
        # Symmetric in its two SH indices.
        @test Magrathea.inv_sin_theta_gaunt(5, 1, 1) ≈ Magrathea.inv_sin_theta_gaunt(1, 5, 1) rtol=1e-10
        # Opposite parity → exactly zero.
        @test Magrathea.inv_sin_theta_gaunt(2, 1, 1) == 0
        # Diagonal strictly positive (self-overlap of a positive measure).
        for (ℓ, m) in [(2, 1), (3, 2), (4, 1)]
            @test Magrathea.inv_sin_theta_gaunt(ℓ, ℓ, m) > 0
        end
    end

end

# Physical mean-flow balance and scaling checks live in thermal_wind.jl.

@testset "Benign cleanups (#7 + misc)" begin
    # #7: hydro DOF estimate must match the real operator layout — Nr rows per
    # (l,field) block and field-specific l-counts (was (Nr+1)·3·single-count).
    @testset "#7 _hd_total_dof == LinearStabilityOperator.total_dof" begin
        for (m, lmax, Nr, sym) in [(0, 8, 16, :both), (4, 30, 64, :both),
                (4, 30, 64, :symmetric), (4, 30, 64, :antisymmetric), (0, 8, 16, :antisymmetric)]
            op = Magrathea.LinearStabilityOperator(OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35,
                    m=m, lmax=lmax, Nr=Nr, equatorial_symmetry=sym))
            @test Magrathea._hd_total_dof(m, lmax, Nr, sym) == op.total_dof
        end
    end

    # MHD DOF estimate must match MHDStabilityOperator.matrix_size (drops l=0 for
    # m=0; magnetic-parity ll_f/ll_g consistent with compute_mhd_l_modes).
    @testset "_mhd_total_dof == MHDStabilityOperator.matrix_size" begin
        mk(m, B0) = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0,
            Le=(B0 == Magrathea.no_field ? 0.0 : 0.1), ricb=0.35,
            m=m, lmax=4, symm=1, N=8, B0_type=B0,
            B0_amplitude=(B0 == Magrathea.no_field ? 0.0 : 1.0))
        for p in (mk(0, Magrathea.axial), mk(0, Magrathea.dipole), mk(0, Magrathea.no_field), mk(1, Magrathea.axial))
            op = Magrathea.MHDStabilityOperator(p)
            est = Magrathea._mhd_total_dof(p)
            @test est[1] == op.matrix_size
            @test est[2] == length(op.ll_u)
            @test est[3] == length(op.ll_v)
            @test est[4] == length(op.ll_f)
            @test est[5] == length(op.ll_g)
        end
    end

    # MHD Galerkin :slepc branch must forward the caller's which/tol/maxiter
    # (was hard-coded :LR/1e-10/1000). Capture via the solver-hook mock — no PETSc.
    @testset "MHD :slepc forwards which/tol/maxiter" begin
        p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                      m=1, lmax=3, symm=1, N=8, B0_type=Magrathea.axial, B0_amplitude=1.0)
        captured = Ref{Any}(nothing)
        orig = Magrathea._SLEPC_SOLVER[]
        Magrathea._SLEPC_SOLVER[] = (A, B; kwargs...) -> (captured[] = (; kwargs...); error("capture"))
        try
            solve(MHDProblem(p); backend=:slepc, which=:LM, tol=3e-7, maxiter=222, nev=2)
        catch
        finally
            Magrathea._SLEPC_SOLVER[] = orig
        end
        @test captured[] !== nothing
        @test captured[].which == :LM
        @test captured[].tol == 3e-7
        @test captured[].maxiter == 222
    end

    # validate_mhd_params soft warnings must fire on the public MHDProblem path.
    @testset "MHDProblem runs validate_mhd_params" begin
        @test_logs (:warn, r"unusually large") match_mode=:any MHDProblem(
            MHDParams(E=0.5, Pr=1.0, Pm=1.0, Ra=100.0, ricb=0.35, m=2, lmax=15, N=32))
    end

    # B0_amplitude is a documented no-op: the dynamical field strength is set by Le.
    # Changing B0_amplitude must NOT alter the assembled operator (regression guard).
    @testset "B0_amplitude does not scale the field (no-op)" begin
        base = (E=1e-3, Pr=1.0, Pm=5.0, Ra=1e5, Le=1e-3, ricb=0.35,
                m=2, lmax=8, symm=1, N=16, B0_type=Magrathea.axial)
        A1 = Magrathea.assemble_mhd_matrices(Magrathea.MHDStabilityOperator(MHDParams(; base..., B0_amplitude=1.0)))[1]
        A2 = Magrathea.assemble_mhd_matrices(Magrathea.MHDStabilityOperator(MHDParams(; base..., B0_amplitude=5.0)))[1]
        @test A1 == A2
    end
end
