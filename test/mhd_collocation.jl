using Test
using LinearAlgebra
using Logging
using Magrathea

# Magnetic perturbations in the collocation (onset, biglobal, triglobal) operators and
# self-consistent MHD mean states. See also "Triglobal MHD energy budget" in
# triglobal_physics.jl.
const _MC = Magrathea

@testset "Collocation MHD onset matches the MHD solver" begin
    # Imposed field about the conductive state: the collocation operator and the
    # ultraspherical MHD solver converge to the same leading eigenvalue (the dipole's
    # r⁻³ profile needs more collocation points).
    for (B0, walls, codes, Nr) in ((axial, :insulating, (0, 0), 20),
                                   (axial, (:perfect_conductor, :insulating), (2, 0), 20),
                                   (dipole, :insulating, (0, 0), 40))
        p = OnsetParams(E=1e-2, Pr=1.0, Ra=2e3, χ=0.35, m=2, lmax=8, Nr=Nr, B0_type=B0, Le=0.1,
                        Pm=1.0, magnetic_bc=walls, equatorial_symmetry=:symmetric)
        result = with_logger(NullLogger()) do
            solve(OnsetProblem(p); nev=3, backend=:dense)
        end
        mhd = MHDParams(E=1e-2, Pr=1.0, Pm=1.0, Ra=2e3, ricb=0.35, m=2, lmax=8, N=Nr, B0_type=B0,
                        B0_amplitude=1.0, Le=0.1, bci_magnetic=codes[1], bco_magnetic=codes[2])
        reference = solve(MHDProblem(mhd); nev=3, backend=:dense, boundary_check=:none).eigenvalues
        @test abs(result.eigenvalues[1] - reference[1]) < 1e-7
        # Resolved modes have a negligible spectral tail; the field is reconstructable.
        @test result.extra.spectral_tail[1].radial < 1e-3
        Br, Bθ, Bφ, r, grid = perturbation_magnetic(result, 1)
        @test size(Br) == (Nr, length(grid.θ)) && maximum(abs, Br) > 0
    end
    @test _MC._magnetic_boundary_layer_Nr(OnsetParams(E=1e-3, Ra=1e5, χ=0.35, m=2, lmax=8, Nr=16)) == 0
    @test_throws ArgumentError OnsetParams(E=1e-2, Ra=2e3, χ=0.35, m=2, lmax=8, Nr=16, Le=0.1)
    @test_throws ArgumentError OnsetParams(E=1e-2, Ra=2e3, χ=0.35, m=2, lmax=8, Nr=16,
                                           B0_type=axial)
    @test_throws ArgumentError OnsetParams(E=1e-2, Ra=2e3, χ=0.35, m=2, lmax=8, Nr=16,
        B0_type=axial, Le=0.1, magnetic_bc=:perfect_conductor, mechanical_bc=:stress_free)
end

@testset "Self-consistent MHD mean state" begin
    χ, E, Ra, Pr, Nr = 0.35, 1e-2, 2e3, 1.0, 24
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4); r = collect(cd.x); D = Matrix(cd.D1); D2 = Matrix(cd.D2)
    forcing = Dict((2, 1) => 0.05, (2, 2) => 0.05)
    shell(f, g) = sum(w * ri^2 * sum(f(i) .* (g.w * fill(2π / length(g.φ), length(g.φ))'))
                      for (i, (w, ri)) in enumerate(zip(_MC._mean_radial_weights(r), r)))
    for (B0, walls, Le) in ((axial, :insulating, 0.1), (axial, (:perfect_conductor, :insulating), 0.5),
                            (dipole, :insulating, 0.02))
        mag = (B0_type=B0, Le=Le, Pm=1.0, magnetic_bc=walls)
        bs, info = nonaxisymmetric_basic_state_selfconsistent(cd, χ, E, Ra, Pr, 4, 2, forcing;
                                                              max_iterations=100, mag...)
        @test info.converged
        @test bs.magnetic == mag
        U, b = bs.flow, bs.field
        # In a steady state the work against the Lorentz force of the full field
        # B₀ + b̄ is the Ohmic dissipation.
        g = _MC.sh_grid(4, 2, Float64); μ = g.μ; s = sqrt.(1 .- μ .^ 2)
        Vθ = B0 == axial ? -s : s ./ 2
        work = shell(g) do i
            u = _MC._mean_flow_grid(U, i, g); bb = _MC._mean_flow_grid(b, i, g)
            J = _MC._mean_vorticity_grid(b, i, g)
            f = B0 == dipole ? 1 / r[i]^3 : 1.0
            Br = f .* μ .+ bb[1]; Bθ = f .* Vθ .+ bb[2]; Bφ = bb[3]
            J[1] .* (u[2] .* Bφ .- u[3] .* Bθ) .+ J[2] .* (u[3] .* Br .- u[1] .* Bφ) .+
                J[3] .* (u[1] .* Bθ .- u[2] .* Br)
        end
        ohmic = E * shell(g) do i
            J = _MC._mean_vorticity_grid(b, i, g); J[1] .^ 2 .+ J[2] .^ 2 .+ J[3] .^ 2
        end
        @test work ≈ ohmic rtol=1e-6
        # One more solve with the frozen nonlinear terms of the result reproduces it.
        inertia = _MC._mean_inertia(U, D); lorentz = _MC._mean_lorentz(b, D, Le^2)
        frozen = (p=Dict(k => inertia.p[k] .+ lorentz.p[k] for k in keys(lorentz.p)),
                  t=Dict(k => inertia.t[k] .+ lorentz.t[k] for k in keys(lorentz.t)))
        U2, b2 = _MC._steady_mean_mhd(bs.theta_coeffs, r, D, D2, E, Ra, Pr, 4, 2;
            magnetic=(Le=Le, Em=E, B0_type=B0, magnetic_bc=walls), forcing=frozen,
            induction=_MC._mean_emf(U, b, D))
        change(a, c) = _MC._relative_change(((a.p, c.p, _ -> 1.0), (a.t, c.t, _ -> 1.0)), Float64, Nr)
        @test change(U, U2) < 1e-7 && change(b, b2) < 1e-7
    end
    # A weak field leaves the flow of the hydrodynamic state, to O(Le²).
    hydro, _ = nonaxisymmetric_basic_state_selfconsistent(cd, χ, E, Ra, Pr, 4, 2, forcing;
                                                          max_iterations=100)
    weak, _ = nonaxisymmetric_basic_state_selfconsistent(cd, χ, E, Ra, Pr, 4, 2, forcing;
        max_iterations=100, B0_type=axial, Le=1e-3, Pm=1.0)
    scale = maximum(maximum(abs, v) for v in values(hydro.flow.p))
    @test maximum(maximum(abs, weak.flow.p[k] - hydro.flow.p[k]) for k in keys(hydro.flow.p)) <
          1e-4 * scale
    # basic_state(params) carries the stability problem's field into flowing states.
    p = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=2, lmax=6, Nr=Nr, B0_type=axial, Le=0.1)
    @test basic_state(p; mode=:meridional).magnetic == _MC._magnetic_kwargs(p)
    @test basic_state(p; mode=:nonaxisymmetric).field !== nothing
end

@testset "MHD biglobal and triglobal problems" begin
    χ, E, Ra, Pr, Nr, lmax = 0.35, 1e-2, 2e3, 1.0, 16, 6
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    mag = (B0_type=axial, Le=0.1, Pm=1.0, magnetic_bc=:insulating)
    quiet(f) = with_logger(f, NullLogger())
    # An axisymmetric MHD state leaves the azimuthal orders uncoupled.
    bs3 = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0, Dict((2, 0) => 0.1); mag...)
    p = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=0, lmax=lmax, Nr=Nr; mag...)
    tri = quiet(() -> solve(TriglobalProblem(p, bs3, -1:1); nev=8, backend=:dense,
                            verbose=false).eigenvalues)
    biglobal = ComplexF64[]
    for m in -1:1
        pm = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=abs(m), lmax=lmax, Nr=Nr; mag...)
        v = quiet(() -> solve(BiglobalProblem(pm, _MC._axisymmetric_state(bs3)); nev=8,
                              backend=:dense).eigenvalues)
        append!(biglobal, m < 0 ? conj.(v) : v)
    end
    @test maximum(minimum(abs.(z .- biglobal)) for z in tri) < 1e-10
    # The stability problem must use the field the state was computed with.
    axis = _MC._axisymmetric_state(bs3)
    @test_throws ArgumentError OnsetParams(E=E, Ra=Ra, χ=χ, m=1, lmax=lmax, Nr=Nr, basic_state=axis)
    @test_throws ArgumentError OnsetParams(E=E, Ra=Ra, χ=χ, m=1, lmax=lmax, Nr=Nr,
        basic_state=axis, B0_type=axial, Le=0.2)
    hydro = _MC._axisymmetric_state(nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0,
                                                                Dict((2, 0) => 0.1)))
    @test_throws ArgumentError OnsetParams(E=E, Ra=Ra, χ=χ, m=1, lmax=lmax, Nr=Nr,
                                           basic_state=hydro; mag...)
    # Weak field: the hydrodynamic modes persist in the MHD spectrum, moved by O(Le²).
    weak = (B0_type=axial, Le=1e-4, Pm=1.0, magnetic_bc=:insulating)
    bw = _MC._axisymmetric_state(nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0,
                                                             Dict((2, 0) => 0.1); weak...))
    λw = quiet(() -> solve(BiglobalProblem(OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=2, lmax=lmax,
        Nr=Nr; weak...), bw); nev=30, backend=:dense).eigenvalues)
    λh = quiet(() -> solve(BiglobalProblem(OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=2, lmax=lmax,
        Nr=Nr), hydro); nev=6, backend=:dense).eigenvalues)
    @test maximum(minimum(abs.(λw .- z)) for z in λh) < 1e-5
end
