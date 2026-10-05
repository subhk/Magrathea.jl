using Test
using LinearAlgebra
using SparseArrays
using Logging
using Magrathea

const _M = Magrathea
_quietly(f) = with_logger(f, NullLogger())

@testset "Sparse pencil removes the stress-free rigid rotation" begin
    sparse_spectrum(m, symm) = begin
        sp = SparseOnsetParams(E=1e-3, Pr=1.0, Ra=1e3, ricb=0.35, m=m, lmax=8, symm=symm,
                               N=24, bci=0, bco=0)
        A, B = assemble_sparse_matrices(SparseStabilityOperator(sp))
        sort(filter(z -> isfinite(z) && abs(z) < 1e6, eigvals(Matrix(A), Matrix(B)));
             by=real, rev=true)
    end
    # Rigid rotation is neutral without the gauge: λ = 0 for m = 0 and λ = i for m = 1.
    for (m, symm) in ((0, 1), (1, -1))
        λ = sparse_spectrum(m, symm)
        @test minimum(z -> abs(z - im * m), λ) > 1e-3
        @test real(λ[1]) < 0
    end
    # For m = 1 both solvers retain the same degrees, so their spectra coincide.
    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e3, ricb=0.35, m=1, lmax=8, N=24, symm=-1,
                   bci=0, bco=0, B0_type=no_field, B0_amplitude=0.0, Le=0.0)
    r = _quietly(() -> solve(MHDProblem(mp); nev=3, backend=:dense))
    @test sparse_spectrum(1, -1)[1:3] ≈ r.eigenvalues[1:3] rtol=1e-6
end

@testset "MHD tau path: dipole field with a conducting core at small ricb" begin
    # Products were truncated before the C⁴/C² projection, which gave growth that
    # increased with N for the dipole's r⁶ weights at small ricb.
    kw = (; E=1e-3, Pr=1.0, Pm=1.0, Ra=1.0, Le=1e-6, ricb=0.1, m=1, lmax=6, symm=1,
          bci_magnetic=1)
    leading(B0, N) = _quietly(() -> solve(MHDProblem(MHDParams(; kw..., N=N, B0_type=B0));
        nev=2, backend=:dense, boundary_check=:none)).eigenvalues[1]
    reference = leading(axial, 32)          # the weak field barely shifts the decay rate
    @test real(reference) < 0
    for N in (32, 48)
        λ = leading(dipole, N)
        @test real(λ) < 0
        @test λ ≈ reference rtol=1e-3
    end
end

@testset "MHD resolution check measures the field as Le·b" begin
    p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1.0, Le=1e-6, ricb=0.35, m=1, lmax=4, N=16,
                  B0_type=axial)
    op = _quietly(() -> MHDStabilityOperator(p))
    # A resolved magnetic field, large in native units, and a weak velocity with a
    # flat (unresolved) Chebyshev spectrum.
    v = zeros(ComplexF64, op.matrix_size)
    for ((_, sec), rng) in _M._mhd_index_map(op)
        sec in (:f, :g) && (v[rng] .= [2.0^-k for k in 0:length(rng)-1])
        sec in (:u, :v) && (v[rng] .= 1e-6)
    end
    @test _M._mhd_spectral_tails(op, v).radial > 0.01
end

@testset "API: heating, truncation, domains and worker ranks" begin
    internal = OnsetParams(E=1e-2, Pr=1.0, Ra=2e4, χ=0.35, m=2, lmax=8, Nr=16,
                           heating=:internal)
    Ra_c, _, _ = _quietly(() -> find_critical_Ra(OnsetProblem(internal); Ra_guess=1e4,
                                                 backend=:dense))
    at_Ra_c = OnsetParams(E=1e-2, Pr=1.0, Ra=Ra_c, χ=0.35, m=2, lmax=8, Nr=16,
                          heating=:internal)
    @test abs(growth_rate(_quietly(() -> solve(OnsetProblem(at_Ra_c); nev=2,
                                               backend=:dense)))) < 1e-5

    # Basic states use differential heating, so internal heating is rejected with one.
    q = OnsetParams(E=1e-2, Pr=1.0, Ra=2e4, χ=0.35, m=2, lmax=8, Nr=16)
    @test_throws ArgumentError basic_state(internal)
    @test_throws ArgumentError BiglobalProblem(internal, basic_state(q))
    @test_throws ArgumentError TriglobalProblem(internal,
                                                basic_state(q; mode=:nonaxisymmetric), 0:2)

    # Coefficients above lmax_bs would be ignored by the coupling, so they are rejected.
    cd = ChebyshevDiffn(16, [0.35, 1.0], 4)
    truncated = conduction_basic_state(cd, 0.35, 2)
    truncated.theta_coeffs[4] = fill(0.01, 16)
    @test_throws ArgumentError BiglobalProblem(q, truncated)

    # Basic states must span the shell [χ, 1].
    bs = meridional_basic_state(cd, 0.35, 1e-2, 3e3, 1.0, 2, 0.1)
    @test_throws ArgumentError find_critical_Ra_biglobal(E=1e-2, Pr=1.0, χ=0.6, m=2,
        lmax=4, Nr=16, basic_state=bs, backend=:dense)
    bs3d = basic_state(q; mode=:nonaxisymmetric)
    shifted = TriglobalParams(E=1e-2, Pr=1.0, Ra=3e3, χ=0.6, m_range=0:2, lmax=4, Nr=16,
                              basic_state_3d=bs3d)
    @test_throws ArgumentError _M.setup_coupled_mode_problem(shifted)
    @test_throws ArgumentError conduction_basic_state(ChebyshevDiffn(16, [0.4, 1.0], 4), 0.35, 2)

    # Non-root MPI ranks hold no eigenvectors.
    worker = StabilityResult(ComplexF64[0.05, -0.1], Matrix{ComplexF64}(undef, 192, 0),
                             OnsetProblem(q))
    @test isempty(leading_mode(worker))

    # Integer inputs and omitted Prandtl numbers.
    @test OnsetParams(E=1e-3, Pr=1.0, Ra=10^4, χ=0.35, m=2, lmax=4, Nr=12).Ra === 1.0e4
    @test BiglobalParams(E=1e-2, Ra=3000, χ=0.35, m=2, lmax=4, Nr=16, basic_state=bs).Pr == 1
    @test OnsetConvectionParams(E=1e-3, Ra=1e4, χ=0.35, m=2, lmax=4, Nr=16).Pr == 1
    @test AdvectionDiffusionSolver(cd=cd, r_i=0.35, r_o=1.0, E=1e-3, Ra=1e3, Pr=1.0,
                                   lmax_bs=4, mmax_bs=0).tolerance == 1e-8
    small = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=0.35, m=1, lmax=5, Nr=10)
    Ra_t, _, _ = find_critical_rayleigh_triglobal(1e-2, 1.0, 0.35, 0:2, 5, 10,
        basic_state(small; mode=:nonaxisymmetric); Ra_min=3000, Ra_max=10_000, tol=1e-2,
        backend=:dense, verbose=false)
    @test 3000 < Ra_t < 10_000

    # MHD results reconstruct the magnetic field like velocity and temperature.
    mp = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=1e3, ricb=0.35, m=1, lmax=4, N=12,
                   B0_type=axial, B0_amplitude=1.0, Le=1e-3)
    r = _quietly(() -> solve(MHDProblem(mp); nev=1, backend=:dense))
    Br, Bθ, Bφ = perturbation_magnetic(r, 1)
    @test all(isfinite, Br) && all(isfinite, Bθ) && all(isfinite, Bφ)
end

@testset "Basic states: stress-free walls, precision and validation" begin
    # One refinement step lets the stress-free state pass the wall check it is held to.
    cd = ChebyshevDiffn(64, [0.35, 1.0], 4)
    bs = meridional_basic_state(cd, 0.35, 1e-5, 1e7, 1.0, 10, 0.1; mechanical_bc=:stress_free)
    @test OnsetParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=2, lmax=10, Nr=64,
                      mechanical_bc=:stress_free, basic_state=bs) isa OnsetParams

    # Float32 states converge with the default tolerance.
    p32 = OnsetParams(E=0.1f0, Pr=1f0, Ra=30f0, χ=0.35f0, m=2, lmax=6, Nr=16)
    @test basic_state(p32; mode=:selfconsistent, mmax_bs=0) isa BasicState{Float32}
    _, info = basic_state_selfconsistent(ChebyshevDiffn(20, Float32[0.35, 1], 4), 0.35f0,
                                         0.1f0, 30f0, 1f0; temperature_bc=Y20(0.1f0))
    @test info.converged

    # BigFloat quadrature is computed in BigFloat.
    @test abs(sum(_M.sh_grid(4, 0, BigFloat).w) - 2) < 1e-60

    @test string(Y20(1)) == "Y20"
    @test_throws ArgumentError basic_state(ChebyshevDiffn(16, [0.4, 1.0], 4), 0.35, 0.01,
                                           100.0, 1.0; temperature_bc=Y20(0.1))
end

@testset "Meridional grids: θ operators, reuse and poles" begin
    Y(m, L, ℓ, θ) = _M._normalized_legendre_table(m, L, [cos(θ)])[ℓ - m + 1, 1]
    for (lmax, m) in ((8, 0), (30, 2), (30, 10))
        g = _M.build_meridional_grid(2lmax, m, lmax)
        h = 1e-6
        for ℓ in max(m, 1):lmax
            y = real.(g.Ylm[ℓ])
            @test norm(g.Lθ * y + ℓ * (ℓ + 1) * y) <= 1e-9 * ℓ * (ℓ + 1) * norm(y)
            dy = [(Y(m, lmax, ℓ, θ + h) - Y(m, lmax, ℓ, θ - h)) / 2h for θ in g.θ]
            @test norm(g.Dθ * y - dy) <= 1e-7 * norm(dy)
        end
    end

    # The m = 0 velocity grid also carries the lmax + 1 temperature degree.
    op = LinearStabilityOperator(OnsetParams(E=1e-3, Pr=1.0, Ra=1e4, χ=0.35, m=0, lmax=6,
                                             Nr=12))
    evec = ComplexF64.(1:op.total_dof) ./ op.total_dof
    grid = perturbation_velocity(evec, op)[5]
    @test grid.lmax == 7
    @test size(first(perturbation_temperature(evec, op; grid=grid))) == (12, length(grid.θ))

    # Coupling harmonics are finite at a polar node.
    for m in (0, 1)
        g = _M.SHGrid{Float64}(3, m, [1.0, 0.5, -0.2], zeros(3), [0.0])
        @test all(isfinite, _M._coupling_harmonic(g, 2, m)[3])
    end

    Ra_c, _, _ = _quietly(() -> find_critical_rayleigh(1f-2, 1f0, 0.35f0, 2, 6, 16;
                                                       Ra_guess=3f3, backend=:dense))
    @test Ra_c isa Float32
    @test Ra_c ≈ 5711.5 rtol=1e-3
end

@testset "Spectral helpers: precision, bounds and Bessel ratios" begin
    @test count(!iszero, _M.chebyshev_coefficients(Float32, 4, 64, 0.35f0, 1f0)) == 5
    c = _M.chebyshev_coefficients(BigFloat, 2, 16, big"0.35", big"1.0")
    a, b = big"0.675", big"0.325"                  # r = a + b x on [0.35, 1]
    @test maximum(abs, c[1:3] .- [a^2 + b^2 / 2, 2a * b, b^2 / 2]) < big"1e-70"
    @test size(_M.multiplication_matrix([1.0, 0.5], 0, 6)) == (6, 6)

    logderiv = _M.spherical_bessel_j_logderiv
    @test logderiv(30, 2.0) ≈ 14.968223 rtol=1e-6          # j_30(2) is ~1e-36
    @test logderiv(30, 1.0 + 0.5im) ≈ 23.984126 - 12.007942im rtol=1e-6
    @test all(isfinite, reim(logderiv(2, 2500.0 * (1 - im))))
end
