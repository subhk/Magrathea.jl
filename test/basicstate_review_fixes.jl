# Regression tests for the basic-state review fixes: boundary-mode truncation
# checks, Legendre pole derivatives and spectral radial interpolation, the
# documented Condon–Shortley sign convention, exact angular couplings, the
# vector-harmonic advection path, ignored compatibility keywords, the shared
# Coriolis grid basis, and thread-safe module caches.

using Test
using Logging
using LinearAlgebra
using Magrathea

const _RF_CHI = 0.35

"""Normalized "no-factorial" real harmonic used by public basic-state coefficients."""
function _rf_Ynf(l, m, θ, φ)
    am = abs(m)
    P = Magrathea._associated_legendre_table(am, max(l, am), [cos(θ)])[l-am+1, 1]
    norm = sqrt((2l + 1) / (4π) * (m == 0 ? 1 : 2))
    return norm * P * (m >= 0 ? cos(m * φ) : sin(am * φ))
end

"""Synthesize no-factorial (ℓ,m) coefficients at radial node `i` and (θ,φ)."""
_rf_synth(coeffs, i, θ, φ) = sum(v[i] * _rf_Ynf(l, m, θ, φ) for ((l, m), v) in coeffs)

# -----------------------------------------------------------------------------
@testset "Nonzero boundary modes outside the truncation are rejected" begin
    cd = ChebyshevDiffn(16, [_RF_CHI, 1.0], 4)
    E, Ra, Pr = 1e-2, 100.0, 1.0

    # Symbolic BCs: an explicit lmax_bs below the forcing degree used to return
    # pure conduction silently.
    err = try
        basic_state(cd, _RF_CHI, E, Ra, Pr; temperature_bc=Y40(0.1), lmax_bs=2)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("(4,0)", sprint(showerror, err)) && occursin("lmax_bs ≥ 4", sprint(showerror, err))
    @test_throws ArgumentError basic_state(cd, _RF_CHI, E, Ra, Pr;
        flux_bc=Y00(-1.0) + Y40(0.1), lmax_bs=3)
    @test_throws ArgumentError basic_state(cd, _RF_CHI, E, Ra, Pr; temperature_bc=Y33(0.1), lmax_bs=2)
    @test_throws ArgumentError basic_state_selfconsistent(cd, _RF_CHI, E, Ra, Pr;
        temperature_bc=Y40(0.1), lmax_bs=2)
    @test_throws ArgumentError basic_state_selfconsistent(cd, _RF_CHI, E, Ra, Pr;
        temperature_bc=Y33(0.1), lmax_bs=4, mmax_bs=2)

    # A retained mode is kept (the explicit truncation may equal the BC degree).
    bs4 = basic_state(cd, _RF_CHI, E, Ra, Pr; temperature_bc=Y40(0.1), lmax_bs=4)
    @test maximum(abs, bs4.theta_coeffs[4]) > 0

    # Dictionary interface: degree, order, invalid harmonics, and flux inputs.
    build(amps; kw...) = nonaxisymmetric_basic_state(cd, _RF_CHI, E, Ra, Pr, 2, 1, amps; kw...)
    @test_throws ArgumentError build(Dict((4, 0) => 0.1))
    @test_throws ArgumentError build(Dict((2, 2) => 0.1))           # |m| > mmax_bs
    @test_throws ArgumentError build(Dict((2, -2) => 0.1))          # sine phase too
    @test_throws ArgumentError build(Dict((1, 3) => 0.1))           # not a harmonic
    @test_throws ArgumentError build(Dict(4 => 0.1))                # keys must be (ℓ,m)
    @test_throws ArgumentError build(Dict{Tuple{Int,Int},Float64}();
        thermal_bc=:fixed_flux, outer_fluxes=Dict((3, 1) => 0.1))
    @test_throws ArgumentError nonaxisymmetric_basic_state_selfconsistent(
        cd, _RF_CHI, E, Ra, Pr, 2, 1, Dict((3, 0) => 0.1); max_iterations=1)
    # Zero amplitudes outside the range carry no forcing and are accepted.
    @test build(Dict((2, 1) => 0.1, (6, 0) => 0.0)) isa BasicState3D
end

# -----------------------------------------------------------------------------
@testset "meridional_basic_state truncation below degree 2" begin
    cd = ChebyshevDiffn(16, [_RF_CHI, 1.0], 4)
    ref = conduction_basic_state(cd, _RF_CHI, 4)
    for lmax_bs in (0, 1)
        err = try
            meridional_basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0, lmax_bs, 0.1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("lmax_bs ≥ 2", sprint(showerror, err))
        @test_throws ArgumentError meridional_basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0,
            lmax_bs, 0.0; thermal_bc=:fixed_flux, outer_flux_Y20=0.1)

        # Without Y₂₀ forcing the truncation is valid: conduction with zero flow.
        bs = meridional_basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0, lmax_bs, 0.0)
        @test bs isa BasicState
        @test sort(collect(keys(bs.theta_coeffs))) == collect(0:lmax_bs)
        @test bs.theta_coeffs[0] ≈ ref.theta_coeffs[0]
        @test all(iszero, mean_flow_velocity(bs, 0.6, 1.0))
    end
    @test_throws ArgumentError meridional_basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0, -1, 0.0)
end

# -----------------------------------------------------------------------------
@testset "Legendre derivatives and evaluate_basic_state at the poles" begin
    for x in (1.0, -1.0)
        _, dP = Magrathea._legendre_values_and_derivs(8, x)
        @test dP ≈ [x^(l + 1) * l * (l + 1) / 2 for l in 0:8]
    end
    x = 0.37
    P, dP = Magrathea._legendre_values_and_derivs(8, x)
    @test dP[2:end] ≈ [l * (P[l] - x * P[l+1]) / (1 - x^2) for l in 1:8]

    cd = ChebyshevDiffn(24, [_RF_CHI, 1.0], 4)
    bs = basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0; temperature_bc=Y20(0.1))
    h = 1e-5
    for (pole, side) in ((0.0, h), (Float64(π), π - h))
        at_pole = Magrathea.evaluate_basic_state(bs, 0.7, pole)
        near = Magrathea.evaluate_basic_state(bs, 0.7, side)
        @test abs(at_pole.uphi_bar) < 1e-18
        # ū_φ vanishes on the axis, so its θ-slope is ū_φ(θ)/Δθ up to O(Δθ²).
        slope = near.uphi_bar / (side - pole)
        @test at_pole.duphi_dtheta ≈ slope rtol=1e-6
        @test at_pole.duphi_dtheta ≈ near.duphi_dtheta rtol=1e-6
    end
end

# -----------------------------------------------------------------------------
@testset "evaluate_basic_state uses spectral radial interpolation" begin
    # 1/r is resolved to roundoff on 32 Chebyshev nodes over [0.35, 1].
    cd = ChebyshevDiffn(32, [_RF_CHI, 1.0], 4)
    bs = conduction_basic_state(cd, _RF_CHI, 2)
    r = 0.5 * (cd.x[3] + cd.x[4])                 # between collocation nodes
    exact = (1 / r - 1) / (1 / _RF_CHI - 1)       # conduction profile, r_o = 1
    dexact = -1 / r^2 / (1 / _RF_CHI - 1)
    out = Magrathea.evaluate_basic_state(bs, r, 0.9)
    @test out.theta_bar ≈ exact rtol=1e-12
    @test out.dtheta_dr ≈ dexact rtol=1e-10

    # Same result on a descending copy of the grid.
    rev(d) = Dict(k => reverse(v) for (k, v) in d)
    desc = BasicState(lmax_bs=bs.lmax_bs, Nr=bs.Nr, r=reverse(bs.r),
        theta_coeffs=rev(bs.theta_coeffs), uphi_coeffs=rev(bs.uphi_coeffs),
        dtheta_dr_coeffs=rev(bs.dtheta_dr_coeffs), duphi_dr_coeffs=rev(bs.duphi_dr_coeffs))
    @test Magrathea.evaluate_basic_state(desc, r, 0.9).theta_bar ≈ out.theta_bar rtol=1e-13

    # θ̄ and ū_φ now use the same interpolant as the native flow.
    flow_state = basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0; temperature_bc=Y20(0.1))
    θflow = sum(Magrathea._mean_barycentric(flow_state.r, v, r) *
                _rf_Ynf(l, 0, 1.1, 0.0) for (l, v) in flow_state.theta_coeffs)
    @test Magrathea.evaluate_basic_state(flow_state, r, 1.1).theta_bar ≈ θflow rtol=1e-12
    @test Magrathea.evaluate_basic_state(flow_state, r, 1.1).uphi_bar ≈
          mean_flow_velocity(flow_state, r, 1.1).uphi rtol=1e-10

    # Float32 states stay in Float32.
    cd32 = ChebyshevDiffn(16, Float32[_RF_CHI, 1], 4)
    out32 = Magrathea.evaluate_basic_state(conduction_basic_state(cd32, 0.35f0, 2), 0.6f0, 1.0f0)
    @test all(v -> v isa Float32, values(out32))
end

# -----------------------------------------------------------------------------
@testset "Documented boundary patterns follow the Condon–Shortley convention" begin
    # Each closed form below is the one stated in the helper's docstring.
    patterns = [
        (Y00, 0, 0, (θ, φ) -> 1.0),
        (Y10, 1, 0, (θ, φ) -> cos(θ)),
        (Y11, 1, 1, (θ, φ) -> -sin(θ) * cos(φ)),
        (Y20, 2, 0, (θ, φ) -> (3cos(θ)^2 - 1) / 2),
        (Y21, 2, 1, (θ, φ) -> -3sin(θ) * cos(θ) * cos(φ)),
        (Y22, 2, 2, (θ, φ) -> 3sin(θ)^2 * cos(2φ)),
        (Y30, 3, 0, (θ, φ) -> (5cos(θ)^3 - 3cos(θ)) / 2),
        (Y31, 3, 1, (θ, φ) -> -(3 / 2) * sin(θ) * (5cos(θ)^2 - 1) * cos(φ)),
        (Y32, 3, 2, (θ, φ) -> 15sin(θ)^2 * cos(θ) * cos(2φ)),
        (Y33, 3, 3, (θ, φ) -> -15sin(θ)^3 * cos(3φ)),
        (Y40, 4, 0, (θ, φ) -> (35cos(θ)^4 - 30cos(θ)^2 + 3) / 8),
        (Y41, 4, 1, (θ, φ) -> -(5 / 2) * sin(θ) * (7cos(θ)^3 - 3cos(θ)) * cos(φ)),
        (Y42, 4, 2, (θ, φ) -> (15 / 2) * sin(θ)^2 * (7cos(θ)^2 - 1) * cos(2φ)),
        (Y43, 4, 3, (θ, φ) -> -105sin(θ)^3 * cos(θ) * cos(3φ)),
        (Y44, 4, 4, (θ, φ) -> 105sin(θ)^4 * cos(4φ)),
    ]
    cd = ChebyshevDiffn(10, [_RF_CHI, 1.0], 4)
    a = 0.3
    for (Yf, l, m, shape) in patterns
        bc = Yf(a)
        @test to_dict(bc) == Dict((l, m) => a)
        bs = nonaxisymmetric_basic_state(cd, _RF_CHI, 0.1, 10.0, 1.0, l, m, to_dict(bc))
        # Outer-boundary temperature from the stored coefficients (Nr is r_o).
        for (θ, φ) in ((0.3, 0.4), (1.2, 2.0), (2.5, 5.1))
            @test _rf_synth(bs.theta_coeffs, bs.Nr, θ, φ) ≈ a * shape(θ, φ) atol=1e-12
        end
    end
    # Hence a positive Y11 amplitude is cold at φ=0 on the equator.
    bs = nonaxisymmetric_basic_state(cd, _RF_CHI, 0.1, 10.0, 1.0, 1, 1, to_dict(Y11(1.0)))
    @test _rf_synth(bs.theta_coeffs, bs.Nr, π / 2, 0.0) ≈ -1
end

# -----------------------------------------------------------------------------
@testset "conduction_basic_state flux sign" begin
    cd = ChebyshevDiffn(16, [_RF_CHI, 1.0], 2)
    fixedT = conduction_basic_state(cd, _RF_CHI, 2)
    # ∂θ̄/∂r = -χ/(1-χ) at r_o carries the conductive heat flux outward.
    flux = conduction_basic_state(cd, _RF_CHI, 2; thermal_bc=:fixed_flux,
                                  outer_flux=-_RF_CHI / (1 - _RF_CHI))
    @test flux.theta_coeffs[0] ≈ fixedT.theta_coeffs[0] atol=1e-12
    # A positive gradient makes the outer boundary warmer than just inside it.
    inward = conduction_basic_state(cd, _RF_CHI, 2; thermal_bc=:fixed_flux, outer_flux=0.5)
    @test inward.theta_coeffs[0][end] > inward.theta_coeffs[0][end-1]
end

# -----------------------------------------------------------------------------
@testset "Exact angular coupling coefficients" begin
    g = Magrathea.sh_grid(8, 3, Float64)
    sinθ = [Magrathea._sh_sinθ(g, j) for j in eachindex(g.μ)]
    for (l, m) in ((0, 0), (1, 0), (2, 1), (3, 2), (3, -2), (4, 3), (5, 0), (5, 1))
        Y = Magrathea.sh_synthesize(Dict((l, m) => 1.0), g)
        dY = Magrathea.sh_synthesize(Dict((l, m) => 1.0), g; Yf=Magrathea._sh_dYθ)
        cosY = Magrathea.sh_analyze(g.μ .* Y, g)
        sin_dY = Magrathea.sh_analyze(sinθ .* dY, g)
        bm, bp = cos_theta_coupling(l, m)
        Am, Ap, Ad = theta_derivative_coupling(l, m)
        @test cosY[(l + 1, m)] ≈ bp atol=1e-12
        @test get(cosY, (l - 1, m), 0.0) ≈ bm atol=1e-12
        @test sin_dY[(l + 1, m)] ≈ Ap atol=1e-12
        @test get(sin_dY, (l - 1, m), 0.0) ≈ Am atol=1e-12
        @test Ad == 0
        # Both operators are exactly two-banded (no other degrees).
        @test all(abs(v) < 1e-12 for (k, v) in cosY if k != (l + 1, m) && k != (l - 1, m))
        @test all(abs(v) < 1e-12 for (k, v) in sin_dY if k != (l + 1, m) && k != (l - 1, m))
    end
    # sinθ·Y is not two-banded, so the former sin_theta_coupling is gone.
    @test !isdefined(Magrathea, :sin_theta_coupling)

    # ⟨Y_Lm|1/sinθ|Y_ℓm⟩ = 2π N_L N_ℓ ∫₀^π P_L^m P_ℓ^m dθ; the integrand is a
    # cosine polynomial in θ, so a fine midpoint rule is exact to roundoff.
    nθ = 400
    θs = [(j - 0.5) * π / nθ for j in 1:nθ]
    function inv_sin_quadrature(L, l, m)
        am = abs(m); lmax = max(L, l)
        P = Magrathea._associated_legendre_table(am, lmax, cos.(θs))
        N = Magrathea._normalization_table(Float64, am, lmax)
        2π * N[L-am+1] * N[l-am+1] * sum(P[L-am+1, :] .* P[l-am+1, :]) * π / nθ
    end
    @test Magrathea.inv_sin_theta_coupling(0, 0)[0] ≈ π / 2
    for (l, m) in ((0, 0), (1, 0), (2, 0), (1, 1), (4, 2), (3, 3), (5, -1))
        c = Magrathea.inv_sin_theta_coupling(l, m; max_coupling=6)
        @test !isempty(c)
        for (L, v) in c
            @test iseven(L - l) && L >= abs(m) && abs(L - l) <= 6
            @test v ≈ inv_sin_quadrature(L, l, m) rtol=1e-10
        end
    end
end

# -----------------------------------------------------------------------------
@testset "compute_full_advection_spectral uses the vector-harmonic flow" begin
    cd = ChebyshevDiffn(24, [_RF_CHI, 1.0], 4)
    bs = nonaxisymmetric_basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0, 6, 2,
                                     Dict((2, 0) => 0.1, (2, 2) => 0.05))
    # Non-uniform temperature: the state's own and an independent profile.
    native = Magrathea._mean_flow_advection(bs.theta_coeffs, bs.dtheta_dr_coeffs, bs.flow)
    adv = compute_full_advection_spectral(bs)
    @test keys(adv) == keys(native)
    @test all(adv[k] ≈ native[k] for k in keys(native))
    @test maximum(abs, adv[(4, 0)]) > 1e-6 && maximum(abs, adv[(6, 0)]) > 1e-8
    T2 = Dict((l, m) => cd.x .^ l .* (1 + 0.1l - 0.2m) for m in -2:2 for l in abs(m):4)
    dT2 = Dict(k => cd.D1 * v for (k, v) in T2)
    adv2 = compute_full_advection_spectral(T2, dT2, bs)
    native2 = Magrathea._mean_flow_advection(T2, dT2, bs.flow)
    @test all(adv2[k] ≈ native2[k] for k in keys(native2))
    # The legacy component form is approximate for these projections.
    legacy = @test_logs (:warn, r"approximate") match_mode=:any compute_full_advection_spectral(
        bs.theta_coeffs, bs.dtheta_dr_coeffs, bs.ur_coeffs, bs.dur_dr_coeffs,
        bs.utheta_coeffs, bs.uphi_coeffs, 6, 2, bs.r)
    @test maximum(abs, legacy[(6, 0)] - native[(6, 0)]) > 0.1 * maximum(abs, native[(6, 0)])
    # Axisymmetric states keep ℓ keys.
    axis = basic_state(cd, _RF_CHI, 1e-2, 100.0, 1.0; temperature_bc=Y20(0.1))
    advA = compute_full_advection_spectral(axis)
    nativeA = Magrathea._mean_flow_advection(Dict((l, 0) => v for (l, v) in axis.theta_coeffs),
        Dict((l, 0) => v for (l, v) in axis.dtheta_dr_coeffs), axis.flow)
    @test all(advA[l] ≈ nativeA[(l, 0)] for l in keys(advA))
    # Temperatures beyond the flow truncation are rejected, not dropped.
    @test_throws ArgumentError compute_full_advection_spectral(
        Dict((8, 0) => cd.x), Dict((8, 0) => one.(cd.x)), bs)
    # Custom states without vector potentials fall back to their components.
    cond = conduction_basic_state(cd, _RF_CHI, 4)
    @test all(iszero, values(compute_full_advection_spectral(cond)))
end

@testset "Vector-harmonic advection against pointwise u·∇T" begin
    # Band-limited manufactured flow and temperature: u·∇T has degree ≤ 4 and
    # |m| ≤ 2, so the retained (ℓ ≤ 6, |m| ≤ 3) coefficients reproduce it exactly.
    cd = ChebyshevDiffn(16, [_RF_CHI, 1.0], 4)
    r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
    bump = (r .- _RF_CHI) .^ 2 .* (1 .- r) .^ 2
    p = Dict((1, 0) => 0.3 .* bump, (2, 1) => bump .* r, (2, -1) => -0.5 .* bump)
    t = Dict((1, 0) => 0.2 .* r .^ 2, (2, 1) => 0.1 .* r, (2, -1) => 0.4 .* bump)
    flow = Magrathea.SolenoidalMeanFlow(6, 3, collect(r), p, t,
        Dict(k => D1 * v for (k, v) in p), Dict(k => D2 * v for (k, v) in p),
        Dict(k => D1 * v for (k, v) in t))
    zero3() = Dict{Tuple{Int,Int},Vector{Float64}}()
    bs = BasicState3D(lmax_bs=6, mmax_bs=3, Nr=length(r), r=collect(r),
        theta_coeffs=zero3(), dtheta_dr_coeffs=zero3(), ur_coeffs=zero3(),
        utheta_coeffs=zero3(), uphi_coeffs=zero3(), dur_dr_coeffs=zero3(),
        dutheta_dr_coeffs=zero3(), duphi_dr_coeffs=zero3(), flow=flow)
    # T = r·Y11 + r²·Y20 (no-factorial harmonics) with analytic gradient.
    T = Dict((1, 1) => collect(r), (2, 0) => r .^ 2)
    dT = Dict((1, 1) => ones(length(r)), (2, 0) => 2 .* r)
    c11 = sqrt(3 / (4π) * 2); c20 = sqrt(5 / (4π))
    function udotgradT(ri, θ, φ)
        u = mean_flow_velocity(bs, ri, θ, φ)
        dr = -c11 * sin(θ) * cos(φ) + c20 * ri * (3cos(θ)^2 - 1)
        dθ = -ri * c11 * cos(θ) * cos(φ) - 3ri^2 * c20 * cos(θ) * sin(θ)
        dφ = ri * c11 * sin(θ) * sin(φ)
        u.ur * dr + u.utheta * dθ / ri + u.uphi * dφ / (ri * sin(θ))
    end
    adv = compute_full_advection_spectral(T, dT, bs)
    for i in (4, 9, 13), (θ, φ) in ((0.4, 0.3), (1.3, 2.2), (2.6, 4.0))
        pointwise = udotgradT(r[i], θ, φ)
        @test _rf_synth(adv, i, θ, φ) ≈ pointwise atol=1e-12 * max(1, abs(pointwise))
    end
end

# -----------------------------------------------------------------------------
@testset "Removed dead helpers" begin
    for name in (:compute_phi_advection_spectral, :theta_derivative_coeff_3d,
                 :_dtheta_sphere_projection, :_azimuthal_coupling_matrix)
        @test !isdefined(Magrathea, name)
    end
    r = collect(range(0.35, 1.0, length=8))
    d() = Dict((1, 0) => copy(r))
    @test Magrathea.vecsh_advection(d(), d(), d(), d(), d(), 2, 0, r) isa AbstractDict
end

# -----------------------------------------------------------------------------
@testset "Ignored compatibility keywords warn once and change nothing" begin
    cd = ChebyshevDiffn(16, [_RF_CHI, 1.0], 4)
    E, Ra, Pr = 1e-2, 100.0, 1.0
    amps = Dict((2, 0) => 0.1, (2, 1) => 0.05)
    base = @test_logs min_level=Logging.Warn nonaxisymmetric_basic_state(cd, _RF_CHI, E, Ra, Pr, 4, 1, amps)
    for (kw, pattern) in ((:coupled_thermal_wind, r"coupled_thermal_wind=false"),
                          (:include_meridional_flow, r"include_meridional_flow=false"))
        bs = @test_logs (:warn, pattern) match_mode=:any nonaxisymmetric_basic_state(
            cd, _RF_CHI, E, Ra, Pr, 4, 1, amps; kw => false)
        @test all(bs.flow.p[k] == base.flow.p[k] for k in keys(base.flow.p))
        @test all(bs.uphi_coeffs[k] == base.uphi_coeffs[k] for k in keys(base.uphi_coeffs))
    end
    @test_logs (:warn, r"coupled_thermal_wind") match_mode=:any basic_state(
        cd, _RF_CHI, E, Ra, Pr; coupled_thermal_wind=false)
    @test_logs (:warn, r"coupled_thermal_wind") match_mode=:any basic_state_selfconsistent(
        cd, _RF_CHI, E, Ra, Pr; temperature_bc=Y20(0.01), coupled_thermal_wind=false,
        max_iterations=1)

    # The component wrappers share one implementation with the constructors.
    empty3() = Dict{Tuple{Int,Int},Vector{Float64}}()
    ur, uθ, dur, duθ, uφ = empty3(), empty3(), empty3(), empty3(), empty3()
    @test_logs (:warn, r"use_full_coupling") match_mode=:any solve_meridional_circulation_toroidal_poloidal!(
        ur, uθ, dur, duθ, base.theta_coeffs, uφ, cd.x, Matrix(cd.D1), Matrix(cd.D2),
        _RF_CHI, 1.0, Ra, E, Pr, 4, 1; use_full_coupling=false)
    for (mine, ref) in ((ur, base.ur_coeffs), (uθ, base.utheta_coeffs), (uφ, base.uphi_coeffs),
                        (dur, base.dur_dr_coeffs), (duθ, base.dutheta_dr_coeffs))
        @test keys(mine) == keys(ref)
        @test all(mine[k] ≈ ref[k] for k in keys(ref))
    end
    ur1, uθ1, dur1, duθ1, uφ1 = empty3(), empty3(), empty3(), empty3(), empty3()
    solve_meridional_coupled!(ur1, uθ1, dur1, duθ1, base.theta_coeffs, uφ1, cd.x,
        Matrix(cd.D1), Matrix(cd.D2), _RF_CHI, 1.0, Ra, E, Pr, 1, 4)
    # Same |m|=1 potentials; the m=0 flow in `base` only adds roundoff here.
    scale = maximum(maximum(abs, v) for v in values(base.uphi_coeffs))
    @test all(isapprox(uφ1[k], base.uphi_coeffs[k]; atol=1e-12 * scale) for k in keys(uφ1))
    @test all(abs(k[2]) == 1 for k in keys(uφ1))
    # Int-keyed thermal-wind wrapper (default lmax = max degree + 2 = 6).
    u3, du3 = Dict{Int,Vector{Float64}}(), Dict{Int,Vector{Float64}}()
    theta1 = Dict(l => v for ((l, m), v) in base.theta_coeffs if m == 1)
    solve_thermal_wind_balance_3d!(u3, du3, theta1, 1, cd, _RF_CHI, 1.0, Ra, Pr; E=E)
    ur6, uθ6, dur6, duθ6, uφ6 = empty3(), empty3(), empty3(), empty3(), empty3()
    solve_meridional_coupled!(ur6, uθ6, dur6, duθ6, Dict((l, 1) => v for (l, v) in theta1),
        uφ6, cd.x, Matrix(cd.D1), Matrix(cd.D2), _RF_CHI, 1.0, Ra, E, Pr, 1, 6)
    @test sort(collect(keys(u3))) == collect(1:6)
    @test all(u3[l] ≈ uφ6[(l, 1)] for l in keys(u3))
end

# -----------------------------------------------------------------------------
@testset "Shared SH grid basis and Coriolis projection" begin
    g = Magrathea.sh_grid(5, 2, Float64)
    modes = [(l, m) for m in -2:2 for l in abs(m):5]
    B = Magrathea._sh_basis_samples(g, modes)
    @test sum(B.w) ≈ 4π
    @test B.Y' * (B.w .* B.Y) ≈ I atol=1e-12
    @test B.sinθ .^ 2 .+ B.cosθ .^ 2 ≈ ones(length(B.w))
    # ⟨e_a, 2ẑ×e_b⟩ is antisymmetric; tangential rows of C are divided by ℓ(ℓ+1).
    for am in (0, 2)
        modes_c, C = Magrathea._mean_coriolis(6, am, Float64)
        q = [l * (l + 1) for (l, m) in modes_c]
        G = Diagonal(vcat(ones(length(q)), q, q)) * C
        @test G ≈ -G' atol=1e-12
    end
end

# -----------------------------------------------------------------------------
@testset "Module caches are consistent under concurrent access" begin
    grid_keys = [(37 + i % 3, i % 2) for i in 1:24]
    grids = Vector{Any}(undef, length(grid_keys))
    Threads.@threads for i in eachindex(grid_keys)
        grids[i] = Magrathea.sh_grid(grid_keys[i]..., Float64)
    end
    for (i, key) in enumerate(grid_keys)
        @test grids[i] === Magrathea.sh_grid(key..., Float64)
    end

    coriolis_keys = [(9 + i % 2, i % 3) for i in 1:12]
    coriolis = Vector{Any}(undef, length(coriolis_keys))
    Threads.@threads for i in eachindex(coriolis_keys)
        coriolis[i] = Magrathea._mean_coriolis(coriolis_keys[i]..., Float64)
    end
    for (i, key) in enumerate(coriolis_keys)
        @test coriolis[i] === Magrathea._mean_coriolis(key..., Float64)
    end

    gaunt_keys = [(20 + i % 5, 1, 13, 0, 21 + i % 7, 1) for i in 1:60]
    gaunts = zeros(length(gaunt_keys))
    Threads.@threads for i in eachindex(gaunt_keys)
        gaunts[i] = Magrathea.compute_gaunt_coefficient(gaunt_keys[i]...)
    end
    @test gaunts == [Magrathea._compute_gaunt_coefficient(k...) for k in gaunt_keys]
    @test all(haskey(Magrathea._GAUNT_CACHE, k) for k in gaunt_keys)
end

@testset "basic_state wrapper caps degree-2 forcing at m ≤ 2" begin
    params = OnsetParams(E=1e-2, Pr=1.0, Ra=1e3, χ=_RF_CHI, m=2, lmax=6, Nr=12)
    bs = basic_state(params; mode=:nonaxisymmetric, amplitude=0.01, mmax_bs=3, lmax_bs=4)
    @test bs isa BasicState3D
    @test bs.mmax_bs == 3
end
