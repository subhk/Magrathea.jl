using Test
using LinearAlgebra
using SparseArrays
using Logging
using Magrathea

# Regression tests for the 2026-09 codebase review: eigensolver selection and
# backends, onset parameters, perturbation temperature, symmetry validation,
# triglobal solves, and the MHD resolution diagnostic. Everything here uses the
# dense backend, so it runs without PETSc/SLEPc.

quiet(f) = with_logger(f, ConsoleLogger(stderr, Logging.Error))

"""Largest-real-part eigenvalue of the constrained onset pencil, computed directly."""
function reference_leading(op)
    A, B, idofs, bdofs = assemble_matrices(op)
    Ar, Br, _ = Magrathea._constrained_reduced_matrices(A, B, op, idofs, bdofs)
    λ = filter(isfinite, eigvals(Ar, Br))
    return λ[argmax(real.(λ))]
end

@testset "Dense eigensolver backend" begin
    A = sparse(ComplexF64[1 0 0; 0 3 0; 0 0 2])
    B = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 0])      # third eigenvalue is infinite
    vals, vecs, info = Magrathea._dense_generalized_eigen(A, B; nev=5, which=:LR)
    @test vals ≈ [3, 1]
    @test info["solver"] === :dense
    @test all(j -> norm(A * vecs[:, j] - vals[j] * B * vecs[:, j]) < 1e-12, axes(vecs, 2))
    vals, _, _ = Magrathea._dense_generalized_eigen(A, B; nev=1, sigma=1.2)
    @test vals ≈ [1]
    vals, _, _ = solve_eigenvalue_problem(A, B; nev=2, backend=:dense)
    @test vals ≈ [3, 1]

    @test_throws ArgumentError solve_eigenvalue_problem(A, B; backend=:bogus)
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=2, lmax=6, Nr=10)
    @test_throws ArgumentError solve(OnsetProblem(p); backend=:bogus)
    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-3, ricb=0.35, m=1, lmax=4, N=10,
                   B0_type=axial, B0_amplitude=1.0, Le=0.1)
    @test_throws ArgumentError quiet(() -> solve(MHDProblem(mp); backend=:bogus))
end

@testset "Onset solve with the dense backend" begin
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=4e3, χ=0.35, m=2, lmax=8, Nr=12)
    r = quiet(() -> solve(OnsetProblem(p); nev=3, backend=:dense))
    op = r.extra.operator
    @test r.eigenvalues[1] ≈ reference_leading(op) rtol=1e-8
    @test issorted(real.(r.eigenvalues); rev=true)

    # Full eigenvectors satisfy the tau pencil: interior equations and BC rows.
    A, B, idofs, bdofs = assemble_matrices(op)
    v = r.eigenvectors[:, 1]
    λ = r.eigenvalues[1]
    @test norm((A * v - λ * B * v)[idofs]) < 1e-8 * norm(A * v)
    @test norm(A[bdofs, :] * v) < 1e-10 * norm(A * v)
end

@testset "Critical-Ra search follows the most unstable mode" begin
    # Mode 1 crosses neutral at Ra = 1000; mode 2 stays weakly damped. Ordering by
    # |Re σ| would lock onto mode 2 and report the wrong bracket.
    builder = Ra -> (sparse(Diagonal(ComplexF64[(Ra - 1000) / 1000, -1e-3])),
                     sparse(Diagonal(ComplexF64[1, 1])))
    Ra_c, ω_c, σ_c, _ = quiet(() -> find_critical_rayleigh(builder, 1e-3, 0.35, 1;
        Ra_min=200.0, Ra_max=1e4, tol=1e-8, growth_tol=1e-9, backend=:dense))
    @test Ra_c ≈ 1000 rtol=1e-6
    @test abs(real(σ_c)) < 1e-6

    failing = (E, χ, Pr, m) -> error("no operator for m=$m")
    @test_throws ErrorException quiet(() ->
        Magrathea.find_onset_parameters(failing, 1e-3, 0.35, 1.0, [1, 2]))
    interrupted = (E, χ, Pr, m) -> throw(InterruptException())
    @test_throws InterruptException quiet(() ->
        Magrathea.find_onset_parameters(interrupted, 1e-3, 0.35, 1.0, [1, 2]))
end

@testset "Onset critical Ra with keyword pass-through" begin
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=2, lmax=8, Nr=12)
    Ra_c, ω_c, _ = quiet(() -> find_critical_Ra(OnsetProblem(p); Ra_guess=3e3,
                                                backend=:dense, nev=4))
    lead(Ra) = real(reference_leading(LinearStabilityOperator(
        OnsetParams(E=p.E, Pr=p.Pr, Ra=Ra, χ=p.χ, m=p.m, lmax=p.lmax, Nr=p.Nr))))
    @test lead(Ra_c * (1 - 1e-3)) < 0 < lead(Ra_c * (1 + 1e-3))

    @test_logs (:warn, r"raising lmax") (:warn, r"search failed") match_mode=:any min_level=Logging.Warn begin
        @test_throws ErrorException find_global_critical_onset(
            E=1e-2, Pr=1.0, χ=0.35, lmax=5, Nr=10, m_range=1:2,
            backend=:bogus, verbose=false)
    end
end

@testset "Perturbation temperature uses the operator's harmonic basis" begin
    # m = 0 keeps ℓ up to lmax + 1; the default grid must cover it.
    p0 = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=0, lmax=4, Nr=10)
    op0 = LinearStabilityOperator(p0)
    @test maximum(op0.l_sets[:Θ]) == 5
    θ0, _, g0 = perturbation_temperature(ones(ComplexF64, op0.total_dof), op0)
    @test size(θ0, 2) == length(g0.θ)
    @test_throws DimensionMismatch perturbation_temperature(ones(ComplexF64, 3), op0)

    # Project the reconstructed physical fields back onto orthonormal Y_ℓm and
    # check the linearized heat equation degree by degree:
    #   σ θ̂_ℓ = û_ℓ (ri·ro/gap)/r² + κ ∇²_ℓ θ̂_ℓ.
    # This only balances when temperature and velocity use the same basis.
    E, Pr, χ = 1e-2, 1.0, 0.35
    p = OnsetParams(E=E, Pr=Pr, Ra=4e3, χ=χ, m=2, lmax=8, Nr=16)
    r = quiet(() -> solve(OnsetProblem(p); nev=1, backend=:dense))
    op = r.extra.operator; σ = r.eigenvalues[1]; v = r.eigenvectors[:, 1]
    Nθ = 2 * p.lmax
    grid = Magrathea.build_meridional_grid(Nθ, p.m, p.lmax)
    _, w = Magrathea._gauss_legendre_nodes(Nθ)
    θf, rr, _ = perturbation_temperature(v, op; grid=grid)
    ur, _, _, _, _ = perturbation_velocity(v, op; grid=grid)
    D1, D2 = op.cd.D1, op.cd.D2
    κ = E / Pr; c = χ / (1 - χ); interior = 2:(p.Nr - 1)
    residuals = Float64[]; scale = 0.0; old_residual = 0.0
    for l in op.l_sets[:Θ]
        y = grid.Ylm[l]
        θl = [2π * sum(w .* θf[k, :] .* conj.(y)) for k in axes(θf, 1)]
        ul = [2π * sum(w .* ur[k, :] .* conj.(y)) for k in axes(ur, 1)]
        advection = ul .* c ./ rr .^ 2
        heat(θ) = σ .* θ .- κ .* (D2 * θ .+ 2 .* (D1 * θ) ./ rr .- l * (l + 1) .* θ ./ rr .^ 2)
        push!(residuals, maximum(abs, (heat(θl) .- advection)[interior]))
        scale = max(scale, maximum(abs, advection))
        # Negative control: the previous reconstruction (no 1/√(2ℓ+1)) breaks the balance.
        old_residual = max(old_residual,
                           maximum(abs, (heat(sqrt(2l + 1) .* θl) .- advection)[interior]))
    end
    @test maximum(residuals) < 1e-6 * scale
    @test old_residual > 0.1 * scale
end

@testset "OnsetParams heating is independent of row weighting" begin
    base = (E=1e-2, Pr=1.0, Ra=3e3, χ=0.35, m=2, lmax=6, Nr=12)
    lead(; kw...) = reference_leading(LinearStabilityOperator(OnsetParams(; base..., kw...)))
    differential = lead()
    @test lead(heating=:differential, use_sparse_weighting=false) ≈ differential rtol=1e-8
    internal = lead(heating=:internal)
    @test lead(heating=:internal, use_sparse_weighting=false) ≈ internal rtol=1e-8
    @test abs(internal - differential) > 1e-4

    legacy = @test_deprecated OnsetParams(; base..., use_sparse_weighting=false)
    @test legacy.heating === :internal
    @test OnsetParams(; base...).heating === :differential
    @test_throws ArgumentError OnsetParams(; base..., heating=:bogus)
    @test_throws ArgumentError OnsetParams(; base..., ro=2.0)
    @test_throws ArgumentError OnsetParams(; base..., ri=0.5)

    # The high-level solve keeps the heating choice.
    r = quiet(() -> solve(OnsetProblem(OnsetParams(; base..., heating=:internal));
                          nev=1, backend=:dense))
    @test r.eigenvalues[1] ≈ internal rtol=1e-8
    @test Magrathea.OnsetConvectionParams(OnsetParams(; base..., heating=:internal)).heating === :internal

    cd = ChebyshevDiffn(12, [0.35, 1.0], 4)
    bs = conduction_basic_state(cd, 0.35, 2)
    @test_throws ArgumentError OnsetParams(; base..., basic_state=bs, heating=:internal)
end

@testset "Truncated equatorial symmetry requires a symmetric basic state" begin
    χ = 0.35; Nr = 12
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    base = (E=1e-2, Pr=1.0, Ra=1e3, χ=χ, m=2, lmax=6, Nr=Nr)
    sym = quiet(() -> basic_state(cd, χ, 1e-2, 1e3, 1.0; temperature_bc=Y20(0.1)))
    asym = quiet(() -> basic_state(cd, χ, 1e-2, 1e3, 1.0; temperature_bc=Y10(0.1)))
    @test Magrathea._basic_state_equatorially_symmetric(sym)
    @test !Magrathea._basic_state_equatorially_symmetric(asym)
    @test OnsetParams(; base..., basic_state=sym, equatorial_symmetry=:symmetric) isa OnsetParams
    @test_throws ArgumentError OnsetParams(; base..., basic_state=asym,
                                           equatorial_symmetry=:symmetric)
    @test_throws ArgumentError OnsetParams(; base..., basic_state=asym,
                                           equatorial_symmetry=:antisymmetric)
    @test OnsetParams(; base..., basic_state=asym) isa OnsetParams     # :both is fine

    # 3-D: a (ℓ, m) = (2, 1) temperature component is equatorially antisymmetric.
    x = cd.x
    z = Dict{Tuple{Int,Int},Vector{Float64}}()
    theta = Dict((0, 0) => zero(x), (2, 1) => 0.01 .* (x .- χ) .* (1 .- x))
    bs3d = BasicState3D(lmax_bs=3, mmax_bs=1, Nr=Nr, r=x, theta_coeffs=theta,
        dtheta_dr_coeffs=Dict(k => cd.D1 * v for (k, v) in theta),
        ur_coeffs=z, utheta_coeffs=z, uphi_coeffs=z, dur_dr_coeffs=z,
        dutheta_dr_coeffs=z, duphi_dr_coeffs=z)
    tp(symm) = TriglobalParams(E=1e-2, Pr=1.0, Ra=1e3, χ=χ, m_range=0:1, lmax=4, Nr=Nr,
                               basic_state_3d=bs3d, equatorial_symmetry=symm)
    @test_throws ArgumentError Magrathea.setup_coupled_mode_problem(tp(:symmetric))
    @test Magrathea.setup_coupled_mode_problem(tp(:both)) isa Magrathea.CoupledModeProblem
end

@testset "Triglobal dense solve and solver keywords" begin
    χ = 0.35; Nr = 12; T = Float64
    cd = ChebyshevDiffn(Nr, T[χ, 1.0], 4)
    # Conduction temperature only: the m blocks decouple, so the triglobal
    # spectrum is the union of the single-mode spectra (m < 0 conjugated).
    conduction = conduction_basic_state(cd, χ, 2)
    theta = Dict((l, 0) => v for (l, v) in conduction.theta_coeffs)
    z = Dict{Tuple{Int,Int},Vector{T}}()
    bs3d = BasicState3D(lmax_bs=2, mmax_bs=0, Nr=Nr, r=cd.x, theta_coeffs=theta,
        dtheta_dr_coeffs=Dict(k => cd.D1 * v for (k, v) in theta),
        ur_coeffs=z, utheta_coeffs=z, uphi_coeffs=z, dur_dr_coeffs=z,
        dutheta_dr_coeffs=z, duphi_dr_coeffs=z)
    p = OnsetParams(E=1e-2, Pr=1.0, Ra=4e3, χ=χ, m=1, lmax=5, Nr=Nr)
    r = quiet(() -> solve(TriglobalProblem(p, bs3d, -1:1); nev=4, backend=:dense,
                          tol=1e-9, maxiter=50, which=:LR, verbose=false))
    single = [reference_leading(LinearStabilityOperator(OnsetParams(E=p.E, Pr=p.Pr,
                  Ra=p.Ra, χ=χ, m=m, lmax=p.lmax, Nr=Nr, basic_state=
                  Magrathea.axisymmetric_basic_state(bs3d)))) for m in 0:1]
    @test real(r.eigenvalues[1]) ≈ maximum(real, single) rtol=1e-8
    @test issorted(real.(r.eigenvalues); rev=true)

    # Operators for ±m are built once and related by conjugation.
    problem = Magrathea.setup_coupled_mode_problem(TriglobalParams(E=p.E, Pr=p.Pr,
        Ra=p.Ra, χ=χ, m_range=-1:1, lmax=p.lmax, Nr=Nr, basic_state_3d=bs3d))
    ops = Magrathea.build_single_mode_operators(problem, false)
    @test ops[-1].A == conj(ops[1].A)
    @test ops[-1].op === ops[1].op
    @test Magrathea.build_single_mode_operator(problem, -1).A ≈ ops[-1].A

    # Reconstruction caches follow a replaced `problem.params`.
    Magrathea._mode_reconstruction(problem, 1)
    problem.params = TriglobalParams(E=p.E, Pr=p.Pr, Ra=p.Ra, χ=χ, m_range=-1:1,
                                     lmax=p.lmax, Nr=10, basic_state_3d=bs3d)
    @test Magrathea._mode_reconstruction(problem, 1).op.params.Nr == 10
end

@testset "Operator radial cache is safe to share across threads" begin
    op = LinearStabilityOperator(OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=2,
                                             lmax=4, Nr=10))
    keys_ = [(p, o) for p in -1:4 for o in 0:4]
    Threads.@threads for k in repeat(eachindex(keys_), 8)
        Magrathea.radial_matrix(op, keys_[k]...)
    end
    fresh = LinearStabilityOperator(op.params)
    @test all(k -> Magrathea.radial_matrix(op, k...) == Magrathea.radial_matrix(fresh, k...), keys_)
end

@testset "MHD dense solves and resolution diagnostic" begin
    mhd(; Le, N, B0=axial, lmax=6) = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-3,
        ricb=0.35, m=1, lmax=lmax, N=N, B0_type=B0, B0_amplitude=1.0, Le=Le)

    # Strong field at low N: the tau pencil grows spuriously at the truncation
    # scale, which the spectral-tail check flags.
    op = MHDStabilityOperator(mhd(Le=1.0, N=16))
    A, B, _, _ = quiet(() -> assemble_mhd_matrices(op))
    vals, vecs, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=2, backend=:dense))
    @test real(vals[1]) > 1
    @test_logs (:warn, r"under-resolved") match_mode=:any min_level=Logging.Warn begin
        tails = Magrathea._check_mhd_resolution(op, vals, Magrathea._eigvecs_to_matrix(vals, vecs, Float64))
        @test tails[1].radial > 0.1
    end
    # The energy-conserving Galerkin solve cannot grow spuriously.
    strong = quiet(() -> solve(MHDProblem(mhd(Le=1.0, N=16)); nev=2, backend=:dense))
    @test real(strong.eigenvalues[1]) < 0
    resolved = quiet(() -> solve(MHDProblem(mhd(Le=0.1, N=24)); nev=2, backend=:dense))
    @test real(resolved.eigenvalues[1]) < 0
    @test resolved.extra.spectral_tail[1].radial < 1e-2

    # The dense tau pencil (singular B) drops its infinite eigenvalues. Once
    # resolved (N=48) it agrees with the Galerkin solve, which converges sooner.
    op = MHDStabilityOperator(mhd(Le=0.1, N=24))
    A, B, _, _ = quiet(() -> assemble_mhd_matrices(op))
    vals, _, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=3, backend=:dense))
    @test all(z -> abs(z) < 1e6, vals)
    A, B, _, _ = quiet(() -> assemble_mhd_matrices(MHDStabilityOperator(mhd(Le=0.1, N=48))))
    vals, _, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=3, backend=:dense))
    @test vals[1] ≈ resolved.eigenvalues[1] rtol=1e-8
end

@testset "Strong dipole: tau grows spuriously, energy Galerkin converges" begin
    # The dipole is ricb⁻³ ≈ 23 times stronger at the inner wall, so its magnetic
    # boundary layers need more radial modes than an axial field of equal Le.
    dip(N; Le=0.03) = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-3, ricb=0.35, m=1,
                                lmax=6, N=N, symm=1, B0_type=dipole, Le=Le)
    axi = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-3, ricb=0.35, m=1, lmax=6, N=24,
                    symm=1, B0_type=axial, Le=0.03)
    @test Magrathea._mhd_boundary_layer_N(dip(24)) > 24 >
          Magrathea._mhd_boundary_layer_N(axi)
    @test Magrathea._mhd_boundary_layer_N(
        MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-3, ricb=0.35, m=1, lmax=6, N=24,
                  symm=1, B0_type=no_field, Le=0.0)) == 0
    A, B, _, _ = quiet(() -> assemble_mhd_matrices(MHDStabilityOperator(dip(24))))
    vals, _, _ = quiet(() -> solve_eigenvalue_problem(A, B; nev=2, backend=:dense))
    @test real(vals[1]) > 1
    for N in (12, 24), Le in (0.03, 1.0)
        r = quiet(() -> solve(MHDProblem(dip(N; Le=Le)); nev=2, backend=:dense))
        @test real(r.eigenvalues[1]) < 0
    end
    resolved = quiet(() -> solve(MHDProblem(dip(48)); nev=2, backend=:dense))
    finer = quiet(() -> solve(MHDProblem(dip(64)); nev=2, backend=:dense))
    @test resolved.extra.spectral_tail[1].radial < 1e-2
    @test finer.eigenvalues[1] ≈ resolved.eigenvalues[1] rtol=1e-6
    report = mktemp() do _, io
        redirect_stdout(io) do
            estimate_size(MHDProblem(dip(24)))
        end
        flush(io); seekstart(io)
        read(io, String)
    end
    @test occursin("magnetic boundary layers: resolved for roughly N ≳ " *
                   "$(Magrathea._mhd_boundary_layer_N(dip(24))); N=24 is likely too low", report)
end
