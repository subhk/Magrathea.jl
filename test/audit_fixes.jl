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

    # ----- #2: meridional θ-solver scale (2Ω convention) ---------------------
    # After full-align, the operator is bare (2Ω divided out) so u_θ ∝ Ra·E²/Pr.
    # The pre-fix two_omega=1/E gives u_θ ∝ Ra·E³/Pr. Hold the temperature field
    # fixed and vary E: the E-scaling exponent (2 vs 3) is the decisive signature.
    # The #6 inv_sinθ heuristic and all geometry constants cancel in the ratio.
    @testset "#2 meridional solver E-scaling is E^2 not E^3" begin
        T = Float64
        Nr = 24; χ = 0.35; Ra = 1.0e5; Pr = 1.0; m_bs = 1; lmax_bs = 2
        cd = ChebyshevDiffn(Nr, T[χ, 1.0], 2)
        r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
        r_i = T(χ); r_o = one(T)

        function peak_utheta(E)
            θc = Dict((1, 1) => ones(T, Nr), (2, 1) => 0.5 .* ones(T, Nr))
            φc = Dict((ℓ, 1) => zeros(T, Nr) for ℓ in 1:lmax_bs)
            ur = Dict{Tuple{Int,Int},Vector{T}}(); uθ = Dict{Tuple{Int,Int},Vector{T}}()
            dur = Dict{Tuple{Int,Int},Vector{T}}(); duθ = Dict{Tuple{Int,Int},Vector{T}}()
            Magrathea.solve_meridional_coupled!(ur, uθ, dur, duθ, θc, φc,
                r, D1, D2, r_i, r_o, Ra, E, Pr, m_bs, lmax_bs)
            maximum(maximum(abs, v) for v in values(uθ))
        end

        u_hi = peak_utheta(1e-2)
        u_lo = peak_utheta(1e-3)
        ratio = u_hi / u_lo          # = (1e-2/1e-3)^p = 10^p
        @test ratio ≈ 100 rtol=0.15  # p=2 (fix). Pre-fix p=3 → ratio≈1000.
    end

    # ----- #3: conducting magnetic ICB reduces to insulating as ω→0 ----------
    function _mag_op_f(; bci_magnetic, bco_magnetic=0, forcing_frequency=1.0)
        params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1.0, ricb=0.35,
                           m=1, lmax=3, symm=1, N=8,
                           bci_magnetic=bci_magnetic, bco_magnetic=bco_magnetic,
                           forcing_frequency=forcing_frequency)
        return (params=params, ll_u=Int[], ll_v=Int[], ll_f=[2], ll_g=[2], ll_h=Int[])
    end

    @testset "#3 conducting ICB steady limit == insulating ICB row" begin
        npm = 8 + 1
        function f_icb_row(op)
            A = spzeros(ComplexF64, 2 * npm, 2 * npm)
            B = spzeros(ComplexF64, 2 * npm, 2 * npm)
            Magrathea.apply_magnetic_boundary_conditions!(A, B, op, :f)
            Vector(A[npm, 1:npm])           # poloidal-f ICB row
        end
        insulating = f_icb_row(_mag_op_f(bci_magnetic=0))
        conducting_steady = f_icb_row(_mag_op_f(bci_magnetic=1, forcing_frequency=0.0))
        @test conducting_steady ≈ insulating
    end

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

# ----- #3: solve_meridional_coupled! satisfies the θ-thermal-wind PDE ---------
# Physical-space residual of  cosθ ∂_r u_θ − (sinθ/r) ∂_θ u_θ
#                              = −(Ra E²/(2 Pr r_o))·(m/sinθ)·Θ̄ .
# Pre-rewrite (2 BCs on a 1st-order operator + Tikhonov reg) gave residual ~1e3,
# non-convergent. The validated coupled-Galerkin structure converges under lmax.
@testset "#3 meridional θ-solver satisfies the thermal-wind PDE" begin
    # Orthonormal associated Legendre P̄_ℓ^m (independent reimplementation).
    function plmbar(lmax, m, x)
        P = zeros(lmax + 1); somx2 = sqrt((1 - x) * (1 + x)); pmm = 1.0; fact = 1.0
        for _ in 1:m; pmm *= -fact * somx2; fact += 2.0; end
        Nlm(l, mm) = Float64(sqrt((2l + 1) / 2 * factorial(big(l - mm)) / factorial(big(l + mm))))
        m <= lmax && (P[m + 1] = Nlm(m, m) * pmm)
        (m + 1 <= lmax) && (P[m + 2] = Nlm(m + 1, m) * x * (2m + 1) * pmm)
        pl2 = pmm; pl1 = x * (2m + 1) * pmm
        for l in (m + 2):lmax
            pl = (x * (2l - 1) * pl1 - (l + m - 1) * pl2) / (l - m)
            P[l + 1] = Nlm(l, m) * pl; pl2 = pl1; pl1 = pl
        end
        P
    end
    Ybar(l, m, θ) = plmbar(l, m, cos(θ))[l + 1]

    χ = 0.35; r_i = χ; r_o = 1.0; Nr = 48; E = 1e-4; Ra = 1e6; Pr = 1.0
    cd = ChebyshevDiffn(Nr, [r_i, r_o], 2)
    r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
    m_bs = 2
    theta = Dict((2, m_bs) => 0.1 .* r .^ 2, (4, m_bs) => 0.05 .* r)
    prefactor = -(Ra * E^2) / (2 * Pr * r_o)

    function residual(lmax_bs)
        ur = Dict{Tuple{Int,Int},Vector{Float64}}(); uθ = Dict{Tuple{Int,Int},Vector{Float64}}()
        dur = Dict{Tuple{Int,Int},Vector{Float64}}(); duθ = Dict{Tuple{Int,Int},Vector{Float64}}()
        φc = Dict((ℓ, m_bs) => zeros(Nr) for ℓ in m_bs:lmax_bs)
        Magrathea.solve_meridional_coupled!(ur, uθ, dur, duθ, theta, φc,
            r, D1, D2, r_i, r_o, Ra, E, Pr, m_bs, lmax_bs)
        active = [(L, U, D1 * U) for ((L, mm), U) in uθ if mm == m_bs && maximum(abs, U) > 0]
        utheta(i, θ)  = sum(U[i] * Ybar(L, m_bs, θ) for (L, U, _) in active; init=0.0)
        dutheta(i, θ) = sum(dU[i] * Ybar(L, m_bs, θ) for (L, _, dU) in active; init=0.0)
        Tbar(i, θ)    = sum(θc[i] * Ybar(ℓ, m_bs, θ) for ((ℓ, _), θc) in theta)
        maxres = 0.0; maxrhs = 0.0
        for i in 10:(Nr - 10), θ in range(0.4, 2.7, length=9)
            lhs = cos(θ) * dutheta(i, θ) -
                  (sin(θ) / r[i]) * ((utheta(i, θ + 1e-6) - utheta(i, θ - 1e-6)) / 2e-6)
            rhs = prefactor * (m_bs / sin(θ)) * Tbar(i, θ)
            maxres = max(maxres, abs(lhs - rhs)); maxrhs = max(maxrhs, abs(rhs))
        end
        maxres / maxrhs
    end

    res_lo = residual(10)
    res_hi = residual(16)
    @test res_hi < res_lo      # converges as the truncation is refined
    @test res_hi < 0.05        # small absolute residual at moderate truncation
end

# ----- Benign cleanups (#7 + misc, do not affect leading eigenvalues) --------
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
