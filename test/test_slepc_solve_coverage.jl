# Real-eigensolve coverage for the SLEPc backend. Runs only when a complex-scalar
# PETSc/SLEPc build is available (PETSC_DIR set) and PetscWrap/SlepcWrap are loaded
# (MAGRATHEA_SLEPC_TEST=1). Drives every high-level solve() dispatch through the SLEPc
# extension so src/Stability/{solver,onset,biglobal,triglobal}.jl and src/solve.jl
# eigensolve paths execute. Run directly (NOT via Pkg.test, which sandboxes deps):
#   julia --project=. --code-coverage=user test/test_slepc_solve_coverage.jl
using Test
using LinearAlgebra
using Magrathea

if !(haskey(ENV, "PETSC_DIR") && get(ENV, "MAGRATHEA_SLEPC_TEST", "") == "1")
    @info "SLEPc real-solve coverage skipped (needs PETSC_DIR + MAGRATHEA_SLEPC_TEST=1 + PetscWrap/SlepcWrap)"
    @test true
else
    @eval using PetscWrap, SlepcWrap            # loads MagratheaSlepcExt, registers _SLEPC_INIT

    # Serial built-in LU shift-invert (no MUMPS needed for single-process solves).
    # -st_pc_factor_shift_type nonzero: the tau/Galerkin pencils carry structurally
    # zero rows, so (A - sigma*B) has zero pivots; shift them rather than erroring.
    const SLEPC_OPTS = "-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
                       "-st_pc_factor_mat_solver_type petsc -st_pc_factor_shift_type nonzero " *
                       "-eps_target_magnitude"
    Magrathea.slepc_init!(SLEPC_OPTS)

    finite_vals(r) = !isempty(r.eigenvalues) && all(isfinite, abs.(r.eigenvalues))

    @testset "SLEPc real eigensolves (coverage)" begin
        op = OnsetParams(E=1e-3, Pr=1.0, Ra=1.0e3, χ=0.35, m=2, lmax=6, Nr=16)

        @testset "OnsetProblem solve" begin
            r = Magrathea.solve(OnsetProblem(op); nev=4, sigma=0.0)
            @test finite_vals(r)
        end

        @testset "MHDProblem solve (axial + dipole)" begin
            for B0 in (axial, dipole)
                mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                               m=1, lmax=3, N=12, B0_type=B0, B0_amplitude=1.0)
                # Complex shift off the real axis: sigma=0 makes (A - sigma*B)=A
                # singular for the dipole pencil (zero pivot in the LU factorization).
                r = Magrathea.solve(MHDProblem(mp); nev=2, sigma=0.5 + 0.5im)
                @test finite_vals(r)
            end
        end

        @testset "BiglobalProblem solve (axisymmetric mean flow)" begin
            bs = Magrathea.basic_state(op; mode=:conduction)
            r = Magrathea.solve(BiglobalProblem(op, bs); nev=2, sigma=0.0)
            @test finite_vals(r)
        end

        @testset "TriglobalProblem solve (mode-coupled, serial)" begin
            T = Float64; Nr = 16; χ = 0.35; lmax_bs = 6
            cd = ChebyshevDiffn(Nr, T[χ, 1.0], 4)
            coeffs = Dict{Tuple{Int,Int}, Vector{T}}((ℓ, 0) => zeros(T, Nr) for ℓ in 0:lmax_bs)
            emptyd = Dict{Tuple{Int,Int}, Vector{T}}()
            bs3d = BasicState3D{T}(
                lmax_bs = lmax_bs, mmax_bs = 0, Nr = Nr, r = cd.x,
                theta_coeffs = coeffs, dtheta_dr_coeffs = Dict(coeffs),
                ur_coeffs = emptyd, utheta_coeffs = emptyd, uphi_coeffs = Dict(coeffs),
                dur_dr_coeffs = emptyd, dutheta_dr_coeffs = emptyd, duphi_dr_coeffs = Dict(coeffs))
            r = Magrathea.solve(TriglobalProblem(op, bs3d, 0:1); nev=2, sigma=0.0)
            @test finite_vals(r)
        end
    end

    Magrathea.slepc_finalize!()
end
