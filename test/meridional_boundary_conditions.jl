using Test
using LinearAlgebra
using Magrathea

@testset "Meridional circulation enforces only the inner radial BC" begin
    # The (ẑ·∇) meridional operator is FIRST-ORDER in radius ⇒ exactly ONE radial
    # BC per mode (inner); the outer boundary is determined by the ODE, not pinned.
    # (The old solver imposed BOTH boundaries + a Tikhonov diagonal — over-constraint,
    # audit #3 — and did not satisfy the thermal-wind PDE.)
    cd = ChebyshevDiffn(10, [0.35, 1.0], 2)
    r = cd.x; D1 = cd.D1; D2 = cd.D2
    idx_inner = 1; idx_outer = length(r)

    theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}(
        (1, 1) => 0.2 .+ r .* (1 .- r),
        (2, 1) => 0.1 .* r.^2,
    )
    function run_bc(bc)
        ur = Dict{Tuple{Int,Int},Vector{Float64}}(); uθ = Dict{Tuple{Int,Int},Vector{Float64}}()
        dur = Dict{Tuple{Int,Int},Vector{Float64}}(); dθ = Dict{Tuple{Int,Int},Vector{Float64}}()
        solve_meridional_circulation_toroidal_poloidal!(
            ur, uθ, dur, dθ, theta_coeffs, Dict{Tuple{Int,Int},Vector{Float64}}(),
            r, D1, D2, first(r), last(r), 1e5, 1e-3, 1.0, 2, 1;
            mechanical_bc = bc, include_meridional = true, use_full_coupling = true)
        return uθ
    end

    # No-slip (non-singular): inner u_θ = 0 enforced exactly; outer is FREE.
    uθ = run_bc(:no_slip)
    outer_free = 0.0
    for ell in 1:2
        u = uθ[(ell, 1)]
        @test abs(u[idx_inner]) < 1e-9 * max(norm(u), eps(Float64))
        outer_free = max(outer_free, abs(u[idx_outer]))
    end
    @test outer_free > 1e-9          # outer boundary unconstrained (one-BC operator)

    # Stress-free at m=1 has an exact geostrophic null mode (∝ r·Y_1^1); the solver
    # returns the finite minimum-norm solution rather than over-constraining.
    uθ_sf = run_bc(:stress_free)
    @test all(all(isfinite, v) for v in values(uθ_sf))
end

@testset "Meridional sin partner is the φ-rotation of the cos mode" begin
    # A sin temperature mode (ℓ0,-m0) is the φ-rotation (by π/2m0) of the cos mode
    # (ℓ0,+m0), so it must drive a meridional flow with the SAME radial profile,
    # stored under the -m0 key. (Pre-fix the solver looped m_bs in 0:mmax and the
    # sin partner drove zero flow.) Also pins that the cos (m≥0) path is unchanged.
    χ, E, Ra, Pr = 0.35, 1e-3, 2e4, 1.0
    Nr, lmax, mmax, ℓ0, m0 = 24, 6, 2, 2, 2
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)
    r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
    prof = sin.(π .* (r .- χ) ./ (1 - χ))

    function run_merid(theta_key)
        theta = Dict{Tuple{Int,Int},Vector{Float64}}(theta_key => copy(prof))
        ur  = Dict{Tuple{Int,Int},Vector{Float64}}()
        uth = Dict{Tuple{Int,Int},Vector{Float64}}()
        dur = Dict{Tuple{Int,Int},Vector{Float64}}()
        dth = Dict{Tuple{Int,Int},Vector{Float64}}()
        solve_meridional_circulation_toroidal_poloidal!(
            ur, uth, dur, dth, theta, Dict{Tuple{Int,Int},Vector{Float64}}(),
            r, D1, D2, χ, 1.0, Ra, E, Pr, lmax, mmax; mechanical_bc=:no_slip)
        return ur, uth
    end

    ur_c, uth_c = run_merid((ℓ0,  m0))   # cos temperature
    ur_s, uth_s = run_merid((ℓ0, -m0))   # sin temperature (rotated)

    # cos mode drives a nonzero flow
    @test maximum(maximum(abs, v) for v in values(uth_c)) > 1e-6
    # sin mode now drives a flow of equal magnitude under the -m0 keys
    @test maximum((maximum(abs, get(uth_s, (ℓ, -m0), [0.0])) for ℓ in m0:lmax); init=0.0) > 1e-6
    # exact rotation invariance: sin(-m0) profile == cos(+m0) profile
    for ℓ in m0:lmax
        @test get(uth_c, (ℓ, m0), zeros(Nr)) ≈ get(uth_s, (ℓ, -m0), zeros(Nr)) atol=1e-12
        @test get(ur_c,  (ℓ, m0), zeros(Nr)) ≈ get(ur_s,  (ℓ, -m0), zeros(Nr)) atol=1e-12
    end
end
