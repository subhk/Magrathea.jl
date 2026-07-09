# =============================================================================
#  Additional coverage for the self-consistent advection / meridional-circulation
#  spectral helpers in src/BasicStates/advection_diffusion.jl, plus a few
#  remaining lines in src/BasicStates/sh_transform.jl and
#  src/Spectral/ultraspherical.jl.
#
#  Distinct from test_basicstate_coverage.jl / thermal_wind.jl / sh_transform.jl /
#  velocity_reconstruction.jl: those exercise m=0 zero-fill paths and the
#  thermal-wind PDE; here we drive the m≠0 meridional-circulation assembly, the
#  Picard self-consistent solver, the diagonal `solve_meridional_simple!` path,
#  alternate boundary conditions, and several error paths.
#
#  HARD RULES honoured here:
#    * No eigensolver / solve() / SLEPc / PETSc / MPI is touched.
#    * For m≠0 coupling we assert ONLY structural facts (dimensions, eltypes,
#      no-throw, round-trips, BC enforcement at boundary nodes) — never the
#      coefficient VALUES (those carry documented bugs).
#    * Axisymmetric (m=0) values ARE checked where they are well-defined.
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using Magrathea
using Logging   # silence the @warn from the singular-system fallback path

# Shared geometry / parameters
const _CHI = 0.35
const _RI  = 0.35
const _RO  = 1.0
const _E   = 1e-4
const _RA  = 1e6
const _PR  = 1.0

_empty_c() = Dict{Tuple{Int,Int},Vector{Float64}}()

# -----------------------------------------------------------------------------
@testset "inv_sin_theta_coupling: m≠0 ℓ±2 coupling branches (structural)" begin
    # m=0 stays purely diagonal (already covered elsewhere); here exercise the
    # m≠0 branches that add ℓ+2 and ℓ-2 entries (lines 280-296).
    c = Magrathea.inv_sin_theta_coupling(4, 2)
    @test c isa Dict{Int,Float64}
    @test c[4] == 1.0                 # diagonal term always present
    @test haskey(c, 6)                # ℓ+2 coupling (within max_coupling=4)
    @test haskey(c, 2)                # ℓ-2 coupling (ℓ-2 >= |m|)
    @test all(isfinite, values(c))

    # ℓ-2 < |m| suppresses the down-coupling, but ℓ+2 still appears
    c2 = Magrathea.inv_sin_theta_coupling(3, 3)
    @test c2[3] == 1.0
    @test haskey(c2, 5)
    @test !haskey(c2, 1)              # ℓ-2 = 1 < |m| = 3  -> no entry

    # max_coupling=0 suppresses the ℓ+2 up-coupling (ℓ+2 > ℓ+max_coupling) but the
    # ℓ-2 down-coupling branch is independent of max_coupling, so it still appears.
    c3 = Magrathea.inv_sin_theta_coupling(4, 2; max_coupling=0)
    @test c3[4] == 1.0
    @test !haskey(c3, 6)             # ℓ+2 suppressed by max_coupling=0
    @test haskey(c3, 2)              # ℓ-2 still present
end

# -----------------------------------------------------------------------------
@testset "solve_poisson_mode: fixed_flux on BOTH boundaries + Float32 flux path" begin
    Nr = 40
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    forcing = zeros(Float64, Nr)

    idx_i = argmin(abs.(r .- _RI))
    idx_o = argmin(abs.(r .- _RO))

    # ℓ=0 with flux at BOTH ends: operator is solvable (ℓ(ℓ+1)=0 has a constant
    # null space pinned only by the discretization); just assert structure + that
    # the prescribed inner/outer derivative rows are honoured.
    Tboth, dTboth = Magrathea.solve_poisson_mode(1, 0, r, D2, D1, _RI, _RO, forcing;
                                             inner_value=0.3, outer_value=-0.4,
                                             inner_bc=:fixed_flux,
                                             outer_bc=:fixed_flux)
    @test length(Tboth) == Nr
    @test all(isfinite, Tboth)
    @test length(dTboth) == Nr
    # The solver enforces D1*T = flux at each boundary row.
    @test isapprox((D1 * Tboth)[idx_i], 0.3; atol=1e-8)
    @test isapprox((D1 * Tboth)[idx_o], -0.4; atol=1e-8)

    # Float32 fixed_flux outer path preserves storage type
    cd32 = Magrathea.ChebyshevDiffn(Nr, Float32[_RI, _RO], 2)
    r32  = collect(cd32.x)
    Tf, dTf = Magrathea.solve_poisson_mode(2, 0, r32, Matrix(cd32.D2), Matrix(cd32.D1),
                                       0.35f0, 1.0f0, zeros(Float32, Nr);
                                       inner_value=0.0f0, outer_value=-0.5f0,
                                       outer_bc=:fixed_flux)
    @test eltype(Tf) == Float32
    @test eltype(dTf) == Float32
    @test all(isfinite, Tf)
end

# -----------------------------------------------------------------------------
@testset "solve_meridional_coupled!: m≠0 assembly (structural; no value checks)" begin
    Nr = 20
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    lmax_bs = 5
    m_bs = 2

    idx_i = argmin(abs.(r .- _RI))

    # Temperature forcing at (ℓ=2, m=2) and (ℓ=3, m=2) drives the coupled system.
    theta = Dict{Tuple{Int,Int},Vector{Float64}}(
        (2, m_bs) => 0.1 .* r,
        (3, m_bs) => 0.05 .* r .^ 2,
    )
    uphi = Dict{Tuple{Int,Int},Vector{Float64}}(
        (ℓ, m_bs) => 0.01 .* r for ℓ in m_bs:lmax_bs
    )

    ur = _empty_c(); uθ = _empty_c(); dur = _empty_c(); duθ = _empty_c()
    @test Magrathea.solve_meridional_coupled!(ur, uθ, dur, duθ, theta, uphi,
                                          r, D1, D2, _RI, _RO, _RA, _E, _PR,
                                          m_bs, lmax_bs) === nothing

    # Every ℓ from |m| to lmax_bs is populated with finite length-Nr vectors.
    for ℓ in m_bs:lmax_bs
        @test haskey(ur, (ℓ, m_bs)) && length(ur[(ℓ, m_bs)]) == Nr
        @test haskey(uθ, (ℓ, m_bs)) && length(uθ[(ℓ, m_bs)]) == Nr
        @test all(isfinite, ur[(ℓ, m_bs)])
        @test all(isfinite, uθ[(ℓ, m_bs)])
        @test eltype(uθ[(ℓ, m_bs)]) == Float64
        # u_r boundary nodes are explicitly zeroed (no-slip continuity BC)
        @test ur[(ℓ, m_bs)][idx_i] == 0.0
        # u_θ inner radial BC (no_slip) is enforced exactly
        @test isapprox(uθ[(ℓ, m_bs)][idx_i], 0.0; atol=1e-9)
        # derivative caches are present and consistent in length
        @test length(duθ[(ℓ, m_bs)]) == Nr
        @test length(dur[(ℓ, m_bs)]) == Nr
    end

    # Negative m_bs (sin partner) shares the angular structure; must also run.
    ur_n = _empty_c(); uθ_n = _empty_c(); dur_n = _empty_c(); duθ_n = _empty_c()
    theta_n = Dict{Tuple{Int,Int},Vector{Float64}}((2, -m_bs) => 0.1 .* r)
    uphi_n  = Dict{Tuple{Int,Int},Vector{Float64}}((2, -m_bs) => 0.01 .* r)
    Magrathea.solve_meridional_coupled!(ur_n, uθ_n, dur_n, duθ_n, theta_n, uphi_n,
                                    r, D1, D2, _RI, _RO, _RA, _E, _PR,
                                    -m_bs, lmax_bs)
    @test haskey(uθ_n, (2, -m_bs))
    @test all(isfinite, uθ_n[(2, -m_bs)])

    # n_ell <= 0 early return (|m| > lmax_bs) leaves the dicts untouched.
    ur_e = _empty_c(); uθ_e = _empty_c(); dur_e = _empty_c(); duθ_e = _empty_c()
    @test Magrathea.solve_meridional_coupled!(ur_e, uθ_e, dur_e, duθ_e,
                                          _empty_c(), _empty_c(),
                                          r, D1, D2, _RI, _RO, _RA, _E, _PR,
                                          7, lmax_bs) === nothing
    @test isempty(ur_e) && isempty(uθ_e)
end

# -----------------------------------------------------------------------------
@testset "solve_meridional_coupled!: stress_free BC + invalid BC error" begin
    Nr = 16
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    lmax_bs = 4
    m_bs = 2

    theta = Dict{Tuple{Int,Int},Vector{Float64}}((2, m_bs) => 0.1 .* r)
    uphi  = Dict{Tuple{Int,Int},Vector{Float64}}((2, m_bs) => 0.01 .* r)

    # stress_free branch (lines 546-549) assembles a Robin row; run inside a
    # log-silencing context because a singular block may emit a min-norm @warn.
    ur = _empty_c(); uθ = _empty_c(); dur = _empty_c(); duθ = _empty_c()
    with_logger(NullLogger()) do
        Magrathea.solve_meridional_coupled!(ur, uθ, dur, duθ, theta, uphi,
                                        r, D1, D2, _RI, _RO, _RA, _E, _PR,
                                        m_bs, lmax_bs; mechanical_bc=:stress_free)
    end
    for ℓ in m_bs:lmax_bs
        @test haskey(uθ, (ℓ, m_bs)) && all(isfinite, uθ[(ℓ, m_bs)])
    end

    # invalid mechanical_bc -> ArgumentError (line 551)
    ur2 = _empty_c(); uθ2 = _empty_c(); dur2 = _empty_c(); duθ2 = _empty_c()
    @test_throws ArgumentError Magrathea.solve_meridional_coupled!(
        ur2, uθ2, dur2, duθ2, theta, uphi,
        r, D1, D2, _RI, _RO, _RA, _E, _PR, m_bs, lmax_bs;
        mechanical_bc=:bogus)
end

# -----------------------------------------------------------------------------
@testset "solve_meridional_simple!: m≠0 diagonal path (no_slip / stress_free / errors)" begin
    Nr = 18
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    lmax_bs = 4
    mmax_bs = 2

    idx_i = argmin(abs.(r .- _RI))
    idx_o = argmin(abs.(r .- _RO))

    # --- no_slip: temperature present at (2,±2) and (3,±2) drives the diagonal
    #     mode-by-mode solve; the (ℓ with no forcing) and (zero-amplitude) early
    #     branches (lines 725-740) are hit by leaving some modes unset / zeroed.
    theta = Dict{Tuple{Int,Int},Vector{Float64}}(
        (2,  2) => 0.1 .* r,
        (3,  2) => 0.05 .* r,
        (2, -2) => 0.1 .* r,
        (4,  2) => zeros(Nr),          # zero-amplitude early-continue branch
        # (ℓ=4? for m=-2 left unset)   -> missing-key early-continue branch
    )

    ur = _empty_c(); uθ = _empty_c(); dur = _empty_c(); duθ = _empty_c()
    @test Magrathea.solve_meridional_simple!(ur, uθ, dur, duθ, theta,
                                         r, D1, _RI, _RO, _RA, _E, _PR,
                                         lmax_bs, mmax_bs) === nothing

    # m=0 zero-fill is always produced
    for ℓ in 0:lmax_bs
        @test all(==(0.0), ur[(ℓ, 0)])
        @test all(==(0.0), uθ[(ℓ, 0)])
    end
    # Forced modes are finite, length Nr, with no-slip u_θ boundary nodes zeroed.
    for key in ((2, 2), (3, 2), (2, -2))
        @test haskey(uθ, key) && length(uθ[key]) == Nr
        @test all(isfinite, uθ[key])
        @test isapprox(uθ[key][idx_i], 0.0; atol=1e-9)
        @test isapprox(uθ[key][idx_o], 0.0; atol=1e-9)
        # u_r boundary nodes are explicitly zeroed
        @test ur[key][idx_i] == 0.0 && ur[key][idx_o] == 0.0
        @test length(dur[key]) == Nr && length(duθ[key]) == Nr
    end
    # zero-amplitude mode yields exact zeros (early branch)
    @test all(==(0.0), uθ[(4, 2)])

    # --- stress_free branch (lines 779-788)
    ur_s = _empty_c(); uθ_s = _empty_c(); dur_s = _empty_c(); duθ_s = _empty_c()
    Magrathea.solve_meridional_simple!(ur_s, uθ_s, dur_s, duθ_s, theta,
                                   r, D1, _RI, _RO, _RA, _E, _PR,
                                   lmax_bs, mmax_bs; mechanical_bc=:stress_free)
    @test all(isfinite, uθ_s[(2, 2)])
    @test length(uθ_s[(2, 2)]) == Nr

    # --- invalid mechanical_bc -> ArgumentError (line 790)
    ur_b = _empty_c(); uθ_b = _empty_c(); dur_b = _empty_c(); duθ_b = _empty_c()
    @test_throws ArgumentError Magrathea.solve_meridional_simple!(
        ur_b, uθ_b, dur_b, duθ_b, theta,
        r, D1, _RI, _RO, _RA, _E, _PR, lmax_bs, mmax_bs;
        mechanical_bc=:bogus)
end

# -----------------------------------------------------------------------------
@testset "solve_meridional_circulation_toroidal_poloidal!: full vs simple dispatch" begin
    Nr = 16
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 2)
    r  = collect(cd.x)
    D1 = Matrix(cd.D1)
    D2 = Matrix(cd.D2)
    lmax_bs = 4
    mmax_bs = 2

    theta = Dict{Tuple{Int,Int},Vector{Float64}}(
        (2,  2) => 0.1 .* r,
        (2, -2) => 0.1 .* r,
    )
    uphi = Dict{Tuple{Int,Int},Vector{Float64}}(
        (2,  2) => 0.01 .* r,
        (2, -2) => 0.01 .* r,
    )

    # use_full_coupling=true loops the coupled solver over the signed m range.
    ur = _empty_c(); uθ = _empty_c(); dur = _empty_c(); duθ = _empty_c()
    with_logger(NullLogger()) do
        Magrathea.solve_meridional_circulation_toroidal_poloidal!(
            ur, uθ, dur, duθ, theta, uphi,
            r, D1, D2, _RI, _RO, _RA, _E, _PR, lmax_bs, mmax_bs;
            use_full_coupling=true)
    end
    @test haskey(uθ, (2, 2)) && all(isfinite, uθ[(2, 2)])
    @test haskey(uθ, (2, -2)) && all(isfinite, uθ[(2, -2)])

    # use_full_coupling=false dispatches to the simplified diagonal solver.
    ur2 = _empty_c(); uθ2 = _empty_c(); dur2 = _empty_c(); duθ2 = _empty_c()
    Magrathea.solve_meridional_circulation_toroidal_poloidal!(
        ur2, uθ2, dur2, duθ2, theta, uphi,
        r, D1, D2, _RI, _RO, _RA, _E, _PR, lmax_bs, mmax_bs;
        use_full_coupling=false)
    @test haskey(uθ2, (2, 2)) && all(isfinite, uθ2[(2, 2)])
    # m=0 zero-fill from the simple solver
    @test all(==(0.0), uθ2[(0, 0)])
end

# -----------------------------------------------------------------------------
@testset "nonaxisymmetric_basic_state_selfconsistent: Picard iteration (structural)" begin
    Nr = 20
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 4)
    lmax_bs = 4
    mmax_bs = 2

    # Mixed axisymmetric + non-axisymmetric amplitudes drive the full advection-
    # diffusion Picard loop (no eigensolver involved).  Loosen the tolerance and
    # cap the iterations so the test is fast; assert only structure / convergence
    # bookkeeping, never m≠0 coefficient VALUES.
    amplitudes = Dict((2, 0) => 0.1, (2, 2) => 0.05)

    bs, info = Magrathea.nonaxisymmetric_basic_state_selfconsistent(
        cd, _CHI, _E, _RA, _PR, lmax_bs, mmax_bs, amplitudes;
        max_iterations=3, tolerance=1e-6)

    @test bs isa Magrathea.BasicState3D
    @test bs.lmax_bs == lmax_bs
    @test bs.mmax_bs == mmax_bs
    @test bs.Nr == Nr

    # info named tuple bookkeeping
    @test info.iterations >= 1
    @test info.iterations <= 3
    @test info.converged isa Bool
    @test length(info.residual_history) == info.iterations
    @test all(isfinite, info.residual_history)
    @test all(>=(0.0), info.residual_history)

    # Every (ℓ, m) slot in the ±m fan-out is filled with finite length-Nr vectors.
    for ℓ in 0:lmax_bs, m in -min(ℓ, mmax_bs):min(ℓ, mmax_bs)
        @test haskey(bs.theta_coeffs, (ℓ, m))
        @test length(bs.theta_coeffs[(ℓ, m)]) == Nr
        @test all(isfinite, bs.theta_coeffs[(ℓ, m)])
        @test haskey(bs.uphi_coeffs, (ℓ, m))
        @test haskey(bs.ur_coeffs, (ℓ, m))
        @test haskey(bs.utheta_coeffs, (ℓ, m))
        @test all(isfinite, bs.uphi_coeffs[(ℓ, m)])
        @test all(isfinite, bs.ur_coeffs[(ℓ, m)])
    end

    # Axisymmetric mean (0,0) temperature is non-trivial (hot inner boundary).
    @test maximum(abs, bs.theta_coeffs[(0, 0)]) > 0

    # The diagonal (uncoupled) thermal-wind option also runs the loop.
    bs2, info2 = Magrathea.nonaxisymmetric_basic_state_selfconsistent(
        cd, _CHI, _E, _RA, _PR, lmax_bs, mmax_bs, amplitudes;
        max_iterations=2, tolerance=1e-6, coupled_thermal_wind=false)
    @test bs2 isa Magrathea.BasicState3D
    @test info2.iterations <= 2
end

# -----------------------------------------------------------------------------
@testset "basic_state_selfconsistent: dispatch (conduction / axisym / non-axisym / errors)" begin
    Nr = 18
    cd = Magrathea.ChebyshevDiffn(Nr, [_RI, _RO], 4)

    # No BC -> pure conduction fallback (BasicState, nothing info)
    bs0, info0 = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR)
    @test bs0 isa Magrathea.BasicState
    @test info0 === nothing

    # No BC with explicit lmax_bs -> conduction with that truncation
    bs0b, info0b = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR; lmax_bs=5)
    @test bs0b isa Magrathea.BasicState
    @test bs0b.lmax_bs == 5
    @test info0b === nothing

    # Axisymmetric temperature BC -> falls back to standard basic_state solver.
    bsA, infoA = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR;
                                                  temperature_bc=Magrathea.Y20(0.1))
    @test bsA isa Magrathea.BasicState
    @test infoA === nothing
    @test maximum(abs, bsA.theta_coeffs[2]) > 0

    # Axisymmetric flux BC also routes through the standard solver.
    bsAf, infoAf = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR;
                                                    flux_bc=Magrathea.Y00(-1.0) + Magrathea.Y20(0.1))
    @test bsAf isa Magrathea.BasicState
    @test infoAf === nothing

    # Non-axisymmetric temperature BC -> self-consistent 3D solver (BasicState3D).
    bsN, infoN = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR;
                                                  temperature_bc=Magrathea.Y20(0.1) + Magrathea.Y22(0.05),
                                                  max_iterations=2, tolerance=1e-6)
    @test bsN isa Magrathea.BasicState3D
    @test infoN isa NamedTuple
    @test infoN.iterations <= 2
    @test bsN.mmax_bs >= 2

    # Non-axisymmetric flux BC -> self-consistent fixed_flux 3D solver.
    bsNf, infoNf = Magrathea.basic_state_selfconsistent(cd, _CHI, _E, _RA, _PR;
                                                    flux_bc=Magrathea.Y22(0.05),
                                                    max_iterations=2, tolerance=1e-6)
    @test bsNf isa Magrathea.BasicState3D
    @test infoNf isa NamedTuple

    # Specifying both temperature_bc and flux_bc is an error.
    @test_throws ErrorException Magrathea.basic_state_selfconsistent(
        cd, _CHI, _E, _RA, _PR;
        temperature_bc=Magrathea.Y20(0.1), flux_bc=Magrathea.Y20(0.1))
end

# -----------------------------------------------------------------------------
@testset "sh_transform: dYθ / dYφ_over_sin separable paths + analyze, m=0 hdiv" begin
    g = Magrathea.sh_grid(6, 3, Float64)

    coeffs = Dict{Tuple{Int,Int},Float64}((ℓ, m) => 0.5 + 0.1ℓ - 0.05m
                 for m in -3:3 for ℓ in abs(m):6)

    # kind==2 path: synthesize the ∂θ field (built-in _sh_dYθ).
    Vθ = Magrathea.sh_synthesize(coeffs, g; Yf=Magrathea._sh_dYθ)
    @test size(Vθ) == (length(g.μ), length(g.φ))
    @test all(isfinite, Vθ)

    # kind==3 path: synthesize (1/sinθ)∂φ field (built-in _sh_dYφ_over_sin); the
    # am==0 modes are skipped internally (≡ 0) — still must be finite.
    Vφ = Magrathea.sh_synthesize(coeffs, g; Yf=Magrathea._sh_dYφ_over_sin)
    @test all(isfinite, Vφ)

    # Generic fallback (kind==0) for _sh_dYφ_over_sin via a wrapper must match the
    # fast separable kind==3 path to machine precision.
    custom_dYφ(gg, ℓ, m, j, k) = Magrathea._sh_dYφ_over_sin(gg, ℓ, m, j, k)
    Vφ_gen = Magrathea.sh_synthesize(coeffs, g; Yf=custom_dYφ)
    @test maximum(abs.(Vφ_gen .- Vφ)) < 1e-12

    # Round-trip analyze∘synthesize recovers the input (analyze loops every ±m).
    rt = Magrathea.sh_analyze(Magrathea.sh_synthesize(coeffs, g), g)
    @test maximum(abs(rt[k] - coeffs[k]) for k in keys(coeffs)) < 1e-11

    # Horizontal divergence with an m=0-only field exercises the am==0 branch
    # (only the θ-component projects; (1/sinθ)∂φ Ȳ_ℓ0 ≡ 0).
    ψ0 = Dict{Tuple{Int,Int},Float64}((ℓ, 0) => randn() for ℓ in 1:6)
    Vθ0 = Magrathea.sh_synthesize(ψ0, g; Yf=Magrathea._sh_dYθ)
    Vφ0 = zeros(size(Vθ0))
    div0 = Magrathea.sh_horizontal_divergence(Vθ0, Vφ0, g)
    err0 = maximum(abs(div0[(ℓ, 0)] - (-ℓ * (ℓ + 1) * get(ψ0, (ℓ, 0), 0.0)))
                   for ℓ in 1:6)
    @test err0 < 1e-10
end

# -----------------------------------------------------------------------------
@testset "ultraspherical: csl recurrence (multi-step), parity λ>0, boundary/radial helpers" begin
    # csl multi-step recurrence: each entry finite and entry-1 matches csl0.
    out = Magrathea.csl([1, 2, 3], 1.5, 4, 4)
    @test length(out) == 3
    @test out[1] ≈ Magrathea.csl0(1, 1.5, 4, 4)
    @test all(isfinite, out)

    N = 10
    # Parity-optimized Gegenbauer (λ>0) multiply by an odd (r-like) multiplier:
    # exercises the vector_parity≠0 + λ>0 branch of multiplication_matrix.
    a_odd = zeros(N); a_odd[2] = 1.0
    Mp1 = Magrathea.multiplication_matrix(a_odd, 1.0, N; vector_parity=1)
    @test Mp1 isa SparseMatrixCSC
    @test size(Mp1) == (N, N)
    Mpm = Magrathea.multiplication_matrix(a_odd, 1.0, N; vector_parity=-1)
    @test size(Mpm) == (N, N)

    # _radial_scale: full-sphere (ri=0) branch returns 1/ro; shell branch returns 2/(ro-ri).
    @test Magrathea._radial_scale(0.0, 2.0) ≈ 0.5
    @test Magrathea._radial_scale(_RI, _RO) ≈ 2.0 / (_RO - _RI)

    # _boundary_radius: outer / inner / full-sphere-inner branches.
    @test Magrathea._boundary_radius(_RI, _RO, :outer) == _RO
    @test Magrathea._boundary_radius(_RI, _RO, :inner) == _RI
    @test Magrathea._boundary_radius(0.0, _RO, :inner) == -_RO   # ri=0 maps inner to -ro

    # neumann2 inner boundary second-derivative functional + _bc_row_values branch.
    N2 = 6
    @test length(Magrathea._chebyshev_boundary_second_derivative(N2, :inner)) == N2 + 1
    @test length(Magrathea._chebyshev_boundary_second_derivative(N2, :outer)) == N2 + 1
    br, vals = Magrathea._bc_row_values(:neumann2, N2 + 3, N2, _RI, _RO, Float64)  # row in a deeper block -> inner boundary
    @test length(vals) == N2 + 1

    # sparse_radial_operator: a r^power d^n/dr^n operator builds as a sparse matrix
    # with the right size (covers the deriv chain + r-power multiply assembly).
    op = Magrathea.sparse_radial_operator(2, 2, 16, _RI, _RO)
    @test op isa SparseMatrixCSC
    @test size(op) == (17, 17)
    # power=0 (no r-multiply) and deriv_order=0 (pure identity) branches.
    op_id = Magrathea.sparse_radial_operator(0, 0, 16, _RI, _RO)
    @test Matrix(op_id) ≈ Matrix(1.0I, 17, 17)
    op_full = Magrathea.sparse_radial_operator(1, 1, 12, 0.0, 1.0)  # ri=0 full-sphere path
    @test size(op_full) == (13, 13)
end
