# =============================================================================
#  Additional coverage for basic-state construction, the spherical-harmonic
#  transform, advection-diffusion spectral helpers, and the ultraspherical
#  spectral utilities.
#
#  These tests deliberately exercise DIFFERENT functions / branches than the
#  existing suites (thermal_wind.jl, sh_transform.jl, sparse_operator.jl,
#  galerkin_radial.jl, chebyshev.jl). They are restricted to safe, deterministic
#  m=0 / axisymmetric paths plus pure structural (dims / eltype / no-throw /
#  round-trip) checks; no eigensolver / SLEPc / MPI is touched, and no m≠0
#  advection/coupling COEFFICIENT VALUES are asserted (those carry known bugs).
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using Magrathea

# Common geometry / parameters reused across testsets
const _CHI = 0.35
const _RI  = 0.35
const _RO  = 1.0
const _E   = 1e-4
const _RA  = 1e6
const _PR  = 1.0

# -----------------------------------------------------------------------------
@testset "conduction_basic_state: structure, fixed_flux branch, Float32 storage" begin
    Nr = 24
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    lmax_bs = 5

    bs = Magrathea.conduction_basic_state(cd, _CHI, lmax_bs)

    # type + dimensions
    @test bs isa Magrathea.BasicState{Float64}
    @test bs.lmax_bs == lmax_bs
    @test bs.Nr == Nr
    @test length(bs.r) == Nr
    @test eltype(bs.r) == Float64

    # every ℓ in 0:lmax_bs has a coefficient vector of length Nr
    for ℓ in 0:lmax_bs
        @test haskey(bs.theta_coeffs, ℓ)
        @test length(bs.theta_coeffs[ℓ]) == Nr
        @test length(bs.dtheta_dr_coeffs[ℓ]) == Nr
        @test length(bs.uphi_coeffs[ℓ]) == Nr
        @test length(bs.duphi_dr_coeffs[ℓ]) == Nr
    end

    # ℓ=0 conduction profile is non-trivial; higher ℓ and all flow are zero
    @test maximum(abs, bs.theta_coeffs[0]) > 0
    @test maximum(abs, bs.theta_coeffs[2]) == 0
    @test all(maximum(abs, v) == 0 for v in values(bs.uphi_coeffs))

    # analyze_basic_state returns Dict{Int, NamedTuple(:θ_max,:uphi_max)}
    summary = analyze_basic_state(bs; verbose=false)
    @test summary isa AbstractDict
    @test haskey(summary, 0)
    @test summary[0].θ_max > 0
    @test summary[0].uphi_max == 0.0
    @test summary[2].θ_max == 0.0

    # fixed_flux outer boundary branch
    bs_flux = Magrathea.conduction_basic_state(cd, _CHI, lmax_bs;
                                           thermal_bc=:fixed_flux,
                                           outer_flux=-1.0)
    @test bs_flux isa Magrathea.BasicState{Float64}
    @test maximum(abs, bs_flux.theta_coeffs[0]) > 0
    # invalid thermal_bc errors out
    @test_throws ErrorException Magrathea.conduction_basic_state(cd, _CHI, lmax_bs;
                                                             thermal_bc=:bogus)

    # Float32 storage is preserved end-to-end
    cd32 = Magrathea.ChebyshevDiffn(Nr, Float32[_RI, _RO], 2)
    bs32 = Magrathea.conduction_basic_state(cd32, 0.35f0, 3)
    @test bs32 isa Magrathea.BasicState{Float32}
    @test eltype(bs32.r) == Float32
    @test eltype(bs32.theta_coeffs[0]) == Float32
    @test eltype(bs32.dtheta_dr_coeffs[0]) == Float32
end

# -----------------------------------------------------------------------------
@testset "SphericalHarmonicBC: algebra, utilities, error branches" begin
    # constructors
    bc = Magrathea.Y20(0.1)
    @test bc isa Magrathea.SphericalHarmonicBC
    @test Magrathea.to_dict(bc)[(2, 0)] == 0.1
    @test Magrathea.Ylm(3, 2, 0.4) isa Magrathea.SphericalHarmonicBC
    @test Magrathea.to_dict(Magrathea.Y44(2.0))[(4, 4)] == 2.0

    # addition (combine) + accumulation of repeated key
    s = Magrathea.Y20(0.1) + Magrathea.Y22(0.05)
    @test Magrathea.to_dict(s)[(2, 0)] == 0.1
    @test Magrathea.to_dict(s)[(2, 2)] == 0.05
    s2 = Magrathea.Y20(0.1) + Magrathea.Y20(0.2)
    @test Magrathea.to_dict(s2)[(2, 0)] ≈ 0.3

    # scalar multiply (both sides), divide, negate, subtract
    @test Magrathea.to_dict(2.0 * Magrathea.Y20(0.1))[(2, 0)] ≈ 0.2
    @test Magrathea.to_dict(Magrathea.Y20(0.1) * 2.0)[(2, 0)] ≈ 0.2
    @test Magrathea.to_dict(Magrathea.Y20(0.2) / 2.0)[(2, 0)] ≈ 0.1
    @test Magrathea.to_dict(-Magrathea.Y20(0.1))[(2, 0)] ≈ -0.1
    diff = Magrathea.Y20(0.3) - Magrathea.Y20(0.1)
    @test Magrathea.to_dict(diff)[(2, 0)] ≈ 0.2

    # iszero on empty and on all-zero amplitudes
    @test Magrathea.iszero(Magrathea.SphericalHarmonicBC{Float64}())
    @test Magrathea.iszero(Magrathea.Y20(0.0))
    @test !Magrathea.iszero(Magrathea.Y20(0.1))

    # lmax / mmax / axisymmetry queries
    bc3 = Magrathea.Y20(0.1) + Magrathea.Y33(0.05)
    @test Magrathea.get_lmax(bc3) == 3
    @test Magrathea.get_mmax(bc3) == 3
    @test Magrathea.get_lmax_mmax(bc3) == (3, 3)
    @test Magrathea.is_axisymmetric(Magrathea.Y20(0.1))
    @test !Magrathea.is_axisymmetric(bc3)
    @test Magrathea.get_lmax(Magrathea.SphericalHarmonicBC{Float64}()) == 0
    @test Magrathea.get_mmax(Magrathea.SphericalHarmonicBC{Float64}()) == 0

    # constructor validation
    @test_throws ArgumentError Magrathea.SphericalHarmonicBC(-1, 0, 1.0)
    @test_throws ArgumentError Magrathea.SphericalHarmonicBC(2, 3, 1.0)   # m > ℓ
    @test_throws ArgumentError Magrathea.Ylm(2, -1, 1.0)                  # m < 0

    # show methods do not throw and produce text
    @test occursin("Y", sprint(show, Magrathea.Y20(0.1)))
    @test occursin("empty", sprint(show, Magrathea.SphericalHarmonicBC{Float64}()))
    @test occursin("SphericalHarmonicBC",
                   sprint(show, MIME("text/plain"), Magrathea.Y20(0.1)))
end

# -----------------------------------------------------------------------------
@testset "basic_state wrapper: dispatch over conduction / meridional / 3D" begin
    Nr = 20
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 4)

    # 1. No BC -> pure conduction (BasicState)
    bs0 = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR)
    @test bs0 isa Magrathea.BasicState
    @test all(maximum(abs, v) == 0 for v in values(bs0.uphi_coeffs))

    # 2. Axisymmetric temperature Y20 -> meridional (BasicState)
    bsT = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR; temperature_bc=Magrathea.Y20(0.1))
    @test bsT isa Magrathea.BasicState
    @test haskey(bsT.theta_coeffs, 2)
    @test maximum(abs, bsT.theta_coeffs[2]) > 0

    # 3. iszero temperature BC collapses to conduction
    bsZ = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR; temperature_bc=Magrathea.Y20(0.0))
    @test bsZ isa Magrathea.BasicState
    @test maximum(abs, bsZ.theta_coeffs[2]) == 0

    # 4. Y00-only temperature BC -> meridional zero-amplitude path (BasicState)
    bsY00 = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR; temperature_bc=Magrathea.Y00(0.3))
    @test bsY00 isa Magrathea.BasicState

    # 5. Y00-only flux BC -> conduction fixed-flux (BasicState)
    bsF0 = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR; flux_bc=Magrathea.Y00(-1.0))
    @test bsF0 isa Magrathea.BasicState
    @test maximum(abs, bsF0.theta_coeffs[0]) > 0

    # 6. Axisymmetric flux BC (Y20 + mean) -> meridional fixed-flux (BasicState)
    bsF = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR;
                            flux_bc=Magrathea.Y00(-1.0) + Magrathea.Y20(0.1))
    @test bsF isa Magrathea.BasicState

    # 7. Non-axisymmetric temperature BC -> BasicState3D
    bs3 = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR;
                            temperature_bc=Magrathea.Y20(0.1) + Magrathea.Y22(0.05))
    @test bs3 isa Magrathea.BasicState3D
    @test bs3.mmax_bs >= 2
    @test haskey(bs3.theta_coeffs, (2, 2))

    # 8. Non-axisymmetric flux BC -> BasicState3D
    bs3f = Magrathea.basic_state(cd, _CHI, _E, _RA, _PR; flux_bc=Magrathea.Y22(0.05))
    @test bs3f isa Magrathea.BasicState3D

    # 9. Specifying both temperature_bc and flux_bc is an error
    @test_throws ErrorException Magrathea.basic_state(cd, _CHI, _E, _RA, _PR;
                                                  temperature_bc=Magrathea.Y20(0.1),
                                                  flux_bc=Magrathea.Y20(0.1))
end

# -----------------------------------------------------------------------------
@testset "evaluate_basic_state / laplace_mode_profile / legendre helper" begin
    Nr = 24
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    bs = Magrathea.conduction_basic_state(cd, _CHI, 4)

    # evaluate at an interior point returns the documented NamedTuple
    out = Magrathea.evaluate_basic_state(bs, 0.6, π / 3)
    @test out isa NamedTuple
    @test Set(keys(out)) == Set((:theta_bar, :uphi_bar, :dtheta_dr,
                                 :dtheta_dtheta, :duphi_dr, :duphi_dtheta))
    @test isfinite(out.theta_bar)
    @test out.uphi_bar == 0.0          # conduction has no flow
    @test out.duphi_dtheta == 0.0

    # out-of-range radius throws
    @test_throws ArgumentError Magrathea.evaluate_basic_state(bs, 5.0, π / 3)

    # laplace_mode_profile: Dirichlet/Dirichlet hits both prescribed values
    r = cd.x
    θ, dθ = Magrathea.laplace_mode_profile(2, r, _RI, _RO, 1.0, 0.5;
                                       outer_bc=:fixed_temperature)
    @test length(θ) == Nr
    @test length(dθ) == Nr
    idx_i = argmin(abs.(r .- _RI)); idx_o = argmin(abs.(r .- _RO))
    @test isapprox(θ[idx_i], 1.0; atol=1e-10)
    @test isapprox(θ[idx_o], 0.5; atol=1e-10)

    # fixed_flux outer branch: derivative at outer boundary matches prescribed flux
    θf, dθf = Magrathea.laplace_mode_profile(2, r, _RI, _RO, 1.0, -0.7;
                                         outer_bc=:fixed_flux)
    @test isapprox(θf[idx_i], 1.0; atol=1e-10)
    @test isapprox(dθf[idx_o], -0.7; atol=1e-8)

    # invalid outer_bc errors
    @test_throws ErrorException Magrathea.laplace_mode_profile(2, r, _RI, _RO, 1.0, 0.0;
                                                           outer_bc=:bogus)

    # legendre derivative-expansion maps: P0'=0, P1'=P0, structural for higher ℓ
    maps = Magrathea.legendre_derivative_coefficients(4)
    @test isempty(maps[0])
    @test maps[1] == Dict(0 => 1.0)
    @test maps[2][1] ≈ 3.0           # P2' = 3 P1
    @test haskey(maps[4], 3)
end

# -----------------------------------------------------------------------------
@testset "solve_poisson_mode: axisymmetric correctness + dims/eltype + flux BCs" begin
    Nr = 40
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    forcing = zeros(Float64, Nr)

    # With zero forcing and Dirichlet BCs, the radial Poisson solve must reproduce
    # the analytic Laplace mode profile (A r^ℓ + B r^{-(ℓ+1)}). m=0, safe.
    T_lm, dT_dr = Magrathea.solve_poisson_mode(2, 0, r, D2, D1, _RI, _RO, forcing;
                                           inner_value=1.0, outer_value=0.5)
    θ_ref, _ = Magrathea.laplace_mode_profile(2, r, _RI, _RO, 1.0, 0.5)
    @test length(T_lm) == Nr
    @test length(dT_dr) == Nr
    @test eltype(T_lm) == Float64
    @test maximum(abs.(T_lm .- θ_ref)) < 1e-7

    # fixed_flux inner + outer branches run and return correct dims
    T2, dT2 = Magrathea.solve_poisson_mode(1, 0, r, D2, D1, _RI, _RO, forcing;
                                       inner_value=1.0, outer_value=-0.5,
                                       inner_bc=:fixed_temperature,
                                       outer_bc=:fixed_flux)
    @test length(T2) == Nr && all(isfinite, T2)
    T3, dT3 = Magrathea.solve_poisson_mode(1, 0, r, D2, D1, _RI, _RO, forcing;
                                       inner_value=0.0, outer_value=0.0,
                                       inner_bc=:fixed_flux,
                                       outer_bc=:fixed_temperature)
    @test length(T3) == Nr && all(isfinite, T3)

    # Float32 path preserves storage type
    cd32 = Magrathea.ChebyshevDiffn(Nr, Float32[_RI, _RO], 2)
    r32  = collect(cd32.x)
    Tf, dTf = Magrathea.solve_poisson_mode(2, 0, r32, Matrix(cd32.D2), Matrix(cd32.D1),
                                       0.35f0, 1.0f0, zeros(Float32, Nr);
                                       inner_value=1.0f0, outer_value=0.0f0)
    @test eltype(Tf) == Float32
    @test eltype(dTf) == Float32
end

# -----------------------------------------------------------------------------
@testset "advection-diffusion couplings + AdvectionDiffusionSolver + φ-advection" begin
    # φ-advection of the basic state projects to zero in the cos(mφ) basis
    theta = Dict{Tuple{Int,Int},Vector{Float64}}((2, 0) => ones(8))
    uphi  = Dict{Tuple{Int,Int},Vector{Float64}}((1, 0) => ones(8))
    Fphi  = Magrathea.compute_phi_advection_spectral(theta, uphi, 4, 2, collect(1.0:8.0))
    @test Fphi isa AbstractDict
    @test isempty(Fphi)

    # sin/cos coupling are the validated Legendre recurrence coefficients (m=0).
    am, ap = Magrathea.sin_theta_coupling(1, 0)
    @test am ≈ 1 / sqrt(3.0)
    @test ap ≈ 2 / sqrt(15.0)
    bm, bp = Magrathea.cos_theta_coupling(1, 0)
    @test bm ≈ am          # cos and sin coupling coincide for m=0
    @test bp ≈ ap
    # ℓ=0 has no downward coupling
    @test Magrathea.sin_theta_coupling(0, 0)[1] == 0.0

    # theta_derivative_coupling: structural only (3-tuple, zero diagonal, ℓ=0 trivial)
    A_minus, A_plus, A_diag = Magrathea.theta_derivative_coupling(2, 0)
    @test A_diag == 0.0
    @test isfinite(A_minus) && isfinite(A_plus)
    @test Magrathea.theta_derivative_coupling(0, 0) == (0.0, 0.0, 0.0)

    # inv_sin_theta_coupling: diagonal-dominant, m=0 is purely diagonal
    c0 = Magrathea.inv_sin_theta_coupling(3, 0)
    @test c0[3] == 1.0
    @test length(c0) == 1

    # inv_sin_theta_gaunt: parity selection rule + below-m vanishing
    @test Magrathea.inv_sin_theta_gaunt(1, 2, 1) == 0.0   # opposite parity -> 0
    @test Magrathea.inv_sin_theta_gaunt(0, 1, 2) == 0.0   # L < m -> 0
    @test Magrathea.inv_sin_theta_gaunt(1, 1, 1) > 0.0    # diagonal same-parity > 0

    # AdvectionDiffusionSolver construction (defaults applied)
    cd = Magrathea.ChebyshevDiffn(16, [_RI, _RO], 2)
    solver = Magrathea.AdvectionDiffusionSolver{Float64}(cd=cd, r_i=_RI, r_o=_RO,
                                                     E=_E, Ra=_RA, Pr=_PR,
                                                     lmax_bs=4, mmax_bs=2)
    @test solver isa Magrathea.AdvectionDiffusionSolver{Float64}
    @test solver.mechanical_bc == :no_slip
    @test solver.thermal_bc == :fixed_temperature
    @test solver.max_iterations == 20
end

# -----------------------------------------------------------------------------
@testset "meridional circulation solvers: m=0 axisymmetric produces zero flow" begin
    Nr = 16
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    lmax_bs = 4

    empty_c() = Dict{Tuple{Int,Int},Vector{Float64}}()

    # solve_meridional_coupled! with m_bs=0 fills exact zeros for ℓ = 0:lmax_bs
    ur = empty_c(); uθ = empty_c(); dur = empty_c(); duθ = empty_c()
    Magrathea.solve_meridional_coupled!(ur, uθ, dur, duθ, empty_c(), empty_c(),
                                    r, D1, D2, _RI, _RO, _RA, _E, _PR, 0, lmax_bs)
    for ℓ in 0:lmax_bs
        @test haskey(ur, (ℓ, 0)) && all(==(0.0), ur[(ℓ, 0)])
        @test all(==(0.0), uθ[(ℓ, 0)])
    end

    # solve_meridional_simple! with mmax_bs=0 -> only the m=0 zero-fill path runs
    ur2 = empty_c(); uθ2 = empty_c(); dur2 = empty_c(); duθ2 = empty_c()
    Magrathea.solve_meridional_simple!(ur2, uθ2, dur2, duθ2, empty_c(),
                                   r, D1, _RI, _RO, _RA, _E, _PR, lmax_bs, 0)
    for ℓ in 0:lmax_bs
        @test all(==(0.0), ur2[(ℓ, 0)])
        @test all(==(0.0), uθ2[(ℓ, 0)])
    end

    # dispatcher with include_meridional=false zeros every (ℓ,m) slot it touches
    ur3 = empty_c(); uθ3 = empty_c(); dur3 = empty_c(); duθ3 = empty_c()
    Magrathea.solve_meridional_circulation_toroidal_poloidal!(
        ur3, uθ3, dur3, duθ3, empty_c(), empty_c(),
        r, D1, D2, _RI, _RO, _RA, _E, _PR, lmax_bs, 2;
        include_meridional=false)
    @test !isempty(ur3)
    @test all(all(==(0.0), v) for v in values(ur3))
    @test all(all(==(0.0), v) for v in values(uθ3))
end

# -----------------------------------------------------------------------------
@testset "compute_full_advection_spectral: axisymmetric structural (no-factorial path)" begin
    # m=0 wrapper path (no-factorial <-> orthonormal conversion is the identity).
    r = collect(range(_RI, _RO, length=16))
    mk(modes) = Dict{Tuple{Int,Int},Vector{Float64}}(k => randn(length(r)) for k in modes)
    modes = [(0, 0), (1, 0), (2, 0)]
    F = Magrathea.compute_full_advection_spectral(mk(modes), mk(modes), mk(modes),
                                              mk(modes), mk(modes), mk(modes),
                                              2, 0, r)
    @test F isa AbstractDict
    for ℓ in 0:2
        @test haskey(F, (ℓ, 0))
        @test length(F[(ℓ, 0)]) == length(r)
        @test all(isfinite, F[(ℓ, 0)])
    end
end

# -----------------------------------------------------------------------------
@testset "sh_transform: generic synthesize fallback, rescale, normalization factor" begin
    g = Magrathea.sh_grid(6, 2, Float64)

    # Passing a non-builtin basis function exercises the generic (kind==0) path of
    # sh_synthesize!. Using a wrapper around _sh_Y must match the fast separable path.
    coeffs = Dict{Tuple{Int,Int},Float64}((ℓ, m) => randn()
                 for m in -2:2 for ℓ in abs(m):6)
    custom_Y(gg, ℓ, m, j, k) = Magrathea._sh_Y(gg, ℓ, m, j, k)
    f_generic = Magrathea.sh_synthesize(coeffs, g; Yf=custom_Y)
    f_fast    = Magrathea.sh_synthesize(coeffs, g)
    @test size(f_generic) == size(f_fast)
    @test maximum(abs.(f_generic .- f_fast)) < 1e-12

    # _sh_nf_to_orth_factor: identity for m=0, >1 for m≠0
    @test Magrathea._sh_nf_to_orth_factor(3, 0, Float64) == 1.0
    @test Magrathea._sh_nf_to_orth_factor(2, 1, Float64) > 1.0

    # _sh_rescale round-trips (forward then inverse recovers the input)
    cvec = Dict{Tuple{Int,Int},Vector{Float64}}((2, 1) => [1.0, 2.0, 3.0],
                                                 (1, 0) => [4.0, 5.0, 6.0])
    fwd = Magrathea._sh_rescale(cvec, +1)
    back = Magrathea._sh_rescale(fwd, -1)
    @test back[(2, 1)] ≈ cvec[(2, 1)]
    @test back[(1, 0)] ≈ cvec[(1, 0)]   # m=0 unchanged by the rescale
    @test fwd[(1, 0)] ≈ cvec[(1, 0)]
end

# -----------------------------------------------------------------------------
@testset "ultraspherical: grid, Chebyshev transform, conversion, coefficients" begin
    # Chebyshev-Gauss-Lobatto grid: N+1 points, ordered 1 -> -1
    grid = Magrathea.chebyshev_grid(8)
    @test length(grid) == 9
    @test isapprox(grid[1], 1.0; atol=1e-12)
    @test isapprox(grid[end], -1.0; atol=1e-12)
    @test issorted(grid; rev=true)

    # Forward Chebyshev transform on known inputs
    N = 8
    @test isapprox(Magrathea.chebyshev_transform(ones(N + 1)), [1.0; zeros(N)]; atol=1e-10)
    fx = [cos(π * j / N) for j in 0:N]            # x = T_1(x)
    cx = Magrathea.chebyshev_transform(fx)
    @test isapprox(cx[2], 1.0; atol=1e-10)
    @test maximum(abs.(cx[[1; 3:end]])) < 1e-10

    # ultraspherical_conversion: λ=0 special case (untyped wrapper hits Float64)
    S0 = Magrathea.ultraspherical_conversion(0, N)
    @test size(S0) == (N + 1, N + 1)
    @test S0[1, 1] ≈ 1.0
    @test S0[2, 2] ≈ 0.5
    @test S0[1, 3] ≈ -0.5
    @test S0[N - 1, N + 1] ≈ -0.5

    # ultraspherical_conversion: λ>0 general formula
    S1 = Magrathea.ultraspherical_conversion(Float64, 1, N)
    @test S1[1, 1] ≈ 1.0                 # λ/(λ+0)
    @test S1[2, 2] ≈ 0.5                 # λ/(λ+1)
    @test S1[1, 3] ≈ -1 / 3             # -λ/(λ+2)

    # chebyshev_coefficients: power form == function form; untyped wrappers
    ri, ro = _RI, _RO
    cp = Magrathea.chebyshev_coefficients(2, 24, ri, ro)
    cf = Magrathea.chebyshev_coefficients(r -> r^2, 24, ri, ro)
    @test length(cp) == 24
    @test isapprox(cp, cf; atol=1e-9)

    # ri = 0 (full-sphere) branch maps [-1,1] -> [-ro, ro]
    c0 = Magrathea.chebyshev_coefficients(2, 16, 0.0, 1.0)
    @test length(c0) == 16
    @test all(isfinite, c0)
end

# -----------------------------------------------------------------------------
@testset "ultraspherical: csl recurrence, multiplication_matrix branches" begin
    # csl0 vanishes for s > min(j,k); finite & real otherwise
    @test Magrathea.csl0(5, 1.0, 2, 3) == 0.0
    @test isfinite(Magrathea.csl0(0, 1.0, 2, 2))
    @test isfinite(Magrathea.csl0(1, 1.5, 3, 4))

    # csl recurrence: first entry matches csl0(svec[1], ...), correct length
    out = Magrathea.csl([0, 1, 2], 1.0, 3, 3)
    @test length(out) == 3
    @test out[1] ≈ Magrathea.csl0(0, 1.0, 3, 3)
    @test all(isfinite, out)

    N = 10
    # all-zero multiplier -> empty sparse operator
    Z = Magrathea.multiplication_matrix(zeros(N), 0.0, N)
    @test Z isa SparseMatrixCSC
    @test nnz(Z) == 0
    @test size(Z) == (N, N)

    # multiplication by the constant c == c·I in any C^(λ) basis
    a_const = [3.0; zeros(N - 1)]
    M0 = Magrathea.multiplication_matrix(a_const, 0.0, N)     # Chebyshev (λ=0) path
    @test Matrix(M0) ≈ 3.0 * I(N)
    M1 = Magrathea.multiplication_matrix(a_const, 1.0, N)     # Gegenbauer (λ>0) path
    @test Matrix(M1) ≈ 3.0 * I(N)

    # parity-optimized branch runs and returns the right shape
    a1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # odd multiplier (r-like)
    Mp = Magrathea.multiplication_matrix(a1, 0.0, N; vector_parity=1)
    @test Mp isa SparseMatrixCSC
    @test size(Mp) == (N, N)
end

# -----------------------------------------------------------------------------
@testset "ultraspherical: _zero_row!, boundary functionals, apply_boundary_conditions!" begin
    # _zero_row! clears stored entries of one row in place, leaving others intact
    A = sparse(Float64[1 2 0; 0 3 4; 5 0 6])
    Magrathea._zero_row!(A, 2)
    @test A[2, 1] == 0.0 && A[2, 2] == 0.0 && A[2, 3] == 0.0
    @test A[1, 1] == 1.0 && A[1, 2] == 2.0
    @test A[3, 1] == 5.0 && A[3, 3] == 6.0

    # boundary-value functionals (untyped Float64 wrappers)
    N = 6
    @test Magrathea._chebyshev_boundary_values(N, :outer) == ones(N + 1)
    vi = Magrathea._chebyshev_boundary_values(N, :inner)
    @test vi == Float64[(-1)^n for n in 0:N]
    do_ = Magrathea._chebyshev_boundary_derivative(N, :outer)
    @test do_[2] ≈ 1.0 && do_[3] ≈ 4.0          # n^2 at outer
    @test length(Magrathea._chebyshev_boundary_second_derivative(N, :inner)) == N + 1

    # _bc_row_values: dirichlet / neumann / neumann2 + unsupported -> ArgumentError
    ri, ro = _RI, _RO
    br, vals = Magrathea._bc_row_values(:dirichlet, 1, N, ri, ro, Float64)
    @test br == 1:(N + 1)
    @test vals == ones(N + 1)                    # outer Dirichlet
    _, vn = Magrathea._bc_row_values(:neumann, 1, N, ri, ro, Float64)
    @test length(vn) == N + 1
    _, vn2 = Magrathea._bc_row_values(:neumann2, 1, N, ri, ro, Float64)
    @test length(vn2) == N + 1
    @test_throws ArgumentError Magrathea._bc_row_values(:bogus, 1, N, ri, ro, Float64)

    # apply_boundary_conditions! overwrites the BC row in A and zeros it in B
    K = N + 1
    Amat = sparse(2.0 * I(K)) + sparse(Float64[i == j ? 0.0 : 0.1 for i in 1:K, j in 1:K])
    Bmat = sparse(3.0 * I(K))
    Magrathea.apply_boundary_conditions!(Amat, Bmat, [1], :dirichlet, N, ri, ro)
    @test collect(Amat[1, :]) ≈ ones(K)          # outer Dirichlet functional
    @test all(==(0.0), collect(Bmat[1, :]))      # B row cleared
    # untouched rows of B keep their identity scaling
    @test Bmat[2, 2] == 3.0
end
