#!/usr/bin/env julia
# =============================================================================
#  Example: Mean Flow from Axisymmetric Heat Flux Boundary Conditions (Y₂₀)
#
#  This example demonstrates computing the basic state (temperature + thermal
#  wind flow) driven by:
#    - Inner boundary: constant heat flux (uniform heating)
#    - Outer boundary: -Y₂₀ heat flux pattern (latitudinal cooling variation)
#
#  Key difference from the Y₂₂ case:
#    - Y₂₀ is AXISYMMETRIC (m=0), so ∂T̄/∂φ = 0
#    - No meridional circulation from geostrophic balance!
#    - Only zonal flow (u_φ) from thermal wind: 2Ω ∂ū_φ/∂z = (Ra E²/Pr)(1/r)∂T̄/∂θ
#
#  The Y₂₀ = (3cos²θ - 1)/2 pattern creates:
#    - Enhanced cooling at poles (θ = 0, π)
#    - Reduced cooling at equator (θ = π/2)
#    - This drives differential rotation (zonal jets)
#
#  Physics:
#    - Heat equation: κ∇²T̄ = 0 (no advection for axisymmetric T with m=0)
#    - Thermal wind: 2Ω ∂ū_φ/∂z = (Ra E²/Pr) (1/r) ∂T̄/∂θ
#    - Meridional circulation: u_r = u_θ = 0 (no φ-forcing for m=0)
# =============================================================================

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Magrathea
using Printf
using LinearAlgebra

# =============================================================================
#  Physical Parameters
# =============================================================================

# Geometry
χ = 0.35        # Radius ratio r_i/r_o (Earth's core: ~0.35)

# Non-dimensional numbers
E = 1e-4        # Ekman number (rotation dominance)
Ra = 1e6        # Rayleigh number (buoyancy strength)
Pr = 1.0        # Prandtl number (ν/κ)

# Resolution
Nr = 32         # Radial points
lmax_bs = 8     # Maximum spherical harmonic degree for basic state

# Flux amplitudes
flux_inner = -1.0    # Constant heat flux at inner boundary (negative = into domain)
flux_Y20 = -0.2      # Y₂₀ amplitude at outer boundary (negative = polar cooling)

println("=" ^ 70)
println("  Mean Flow from Axisymmetric Heat Flux (Y₂₀) Boundary Conditions")
println("=" ^ 70)
println()
println("Physical Parameters:")
println("  Radius ratio χ = $χ")
println("  Ekman number E = $E")
println("  Rayleigh number Ra = $Ra")
println("  Prandtl number Pr = $Pr")
println()
println("Boundary Conditions:")
println("  Inner: constant flux = $flux_inner (uniform heating)")
println("  Outer: Y₀₀ + Y₂₀ pattern, Y₂₀ amplitude = $flux_Y20")
println()
println("Y₂₀ pattern: Y₂₀ ∝ (3cos²θ - 1)/2")
println("  → Enhanced cooling at poles (cos²θ = 1)")
println("  → Reduced cooling at equator (cos²θ = 0)")
println()
println("Resolution: Nr = $Nr, lmax = $lmax_bs")
println()

# =============================================================================
#  Setup Chebyshev Grid
# =============================================================================

cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
r = cd.x
r_i = χ
r_o = 1.0

println("Radial grid: r ∈ [$(round(minimum(r), digits=3)), $(round(maximum(r), digits=3))]")
println()

# =============================================================================
#  Define Flux Boundary Condition using Symbolic Spherical Harmonics
# =============================================================================

# Construct the outer boundary flux pattern
# Y00 carries the mean heat flux, Y20 adds the latitudinal variation
outer_flux = Y00(flux_inner) + Y20(flux_Y20)

println("Outer boundary flux pattern:")
println("  $outer_flux")
println()

# =============================================================================
#  Compute Basic State (Standard Solver - No Iteration Needed!)
# =============================================================================

println("-" ^ 70)
println("Computing basic state...")
println("-" ^ 70)
println()
println("Note: For axisymmetric basic states (m=0 only), the advection term")
println("      ū·∇T̄ = ū_φ/(r sinθ) × ∂T̄/∂φ = 0")
println("      because ∂T̄/∂φ = 0 for m=0 modes.")
println()
println("      Therefore, no iteration is needed - we can use the standard solver.")
println()

# Use the standard basic_state function (which detects axisymmetry)
bs = basic_state(
    cd, χ, E, Ra, Pr;
    flux_bc = outer_flux,
    mechanical_bc = :no_slip,
    lmax_bs = lmax_bs
)

println("Basic state computed (no iteration required for axisymmetric case).")

# =============================================================================
#  Analyze the Basic State
# =============================================================================

println()
println("-" ^ 70)
println("Basic State Analysis")
println("-" ^ 70)

# Determine if we got BasicState or BasicState3D
is_3d = isa(bs, BasicState3D)

if is_3d
    # Temperature field analysis
    println("\nTemperature coefficients T̄_ℓm(r):")
    println("  Mode (ℓ,m)     max|T̄|        Location")
    println("  " * "-"^50)

    for m in 0:bs.mmax_bs
        for ℓ in m:bs.lmax_bs
            if haskey(bs.theta_coeffs, (ℓ, m))
                T_lm = bs.theta_coeffs[(ℓ, m)]
                T_max = maximum(abs.(T_lm))
                if T_max > 1e-10
                    idx_max = argmax(abs.(T_lm))
                    r_max = r[idx_max]
                    @printf("  (%d,%d)         %.4e      r = %.3f\n", ℓ, m, T_max, r_max)
                end
            end
        end
    end

    # Zonal velocity (u_φ) analysis
    println("\nZonal velocity coefficients ū_φ,ℓm(r):")
    println("  Mode (ℓ,m)     max|ū_φ|       Location")
    println("  " * "-"^50)

    let has_uphi = false
        for m in 0:bs.mmax_bs
            for ℓ in m:bs.lmax_bs
                if haskey(bs.uphi_coeffs, (ℓ, m))
                    u_lm = bs.uphi_coeffs[(ℓ, m)]
                    u_max = maximum(abs.(u_lm))
                    if u_max > 1e-10
                        has_uphi = true
                        idx_max = argmax(abs.(u_lm))
                        r_max = r[idx_max]
                        @printf("  (%d,%d)         %.4e      r = %.3f\n", ℓ, m, u_max, r_max)
                    end
                end
            end
        end
        if !has_uphi
            println("  (No significant zonal flow)")
        end
    end

    # Meridional velocity (u_θ) analysis
    println("\nMeridional velocity coefficients ū_θ,ℓm(r):")
    println("  Mode (ℓ,m)     max|ū_θ|       Location")
    println("  " * "-"^50)

    let has_meridional = false
        for m in 0:bs.mmax_bs
            for ℓ in m:bs.lmax_bs
                if haskey(bs.utheta_coeffs, (ℓ, m))
                    u_lm = bs.utheta_coeffs[(ℓ, m)]
                    u_max = maximum(abs.(u_lm))
                    if u_max > 1e-10
                        has_meridional = true
                        idx_max = argmax(abs.(u_lm))
                        r_max = r[idx_max]
                        @printf("  (%d,%d)         %.4e      r = %.3f\n", ℓ, m, u_max, r_max)
                    end
                end
            end
        end
        if !has_meridional
            println("  (No meridional flow - expected for axisymmetric basic state!)")
        end
    end

    # Radial velocity (u_r) analysis
    println("\nRadial velocity coefficients ū_r,ℓm(r):")
    println("  Mode (ℓ,m)     max|ū_r|       Location")
    println("  " * "-"^50)

    let has_radial = false
        for m in 0:bs.mmax_bs
            for ℓ in m:bs.lmax_bs
                if haskey(bs.ur_coeffs, (ℓ, m))
                    u_lm = bs.ur_coeffs[(ℓ, m)]
                    u_max = maximum(abs.(u_lm))
                    if u_max > 1e-10
                        has_radial = true
                        idx_max = argmax(abs.(u_lm))
                        r_max = r[idx_max]
                        @printf("  (%d,%d)         %.4e      r = %.3f\n", ℓ, m, u_max, r_max)
                    end
                end
            end
        end
        if !has_radial
            println("  (No radial flow - expected for axisymmetric basic state!)")
        end
    end

else
    # BasicState (axisymmetric) - uses coefficient dictionaries
    println("\nTemperature coefficients T̄_ℓ₀(r):")
    println("  Mode ℓ      max|T̄|        Location")
    println("  " * "-"^40)

    for ℓ in 0:bs.lmax_bs
        if haskey(bs.theta_coeffs, ℓ)
            T_l = bs.theta_coeffs[ℓ]
            T_max = maximum(abs.(T_l))
            if T_max > 1e-10
                idx_max = argmax(abs.(T_l))
                r_max = r[idx_max]
                @printf("  %d          %.4e      r = %.3f\n", ℓ, T_max, r_max)
            end
        end
    end

    println("\nZonal velocity coefficients ū_φ,ℓ₀(r):")
    println("  Mode ℓ      max|ū_φ|       Location")
    println("  " * "-"^40)

    let has_uphi = false
        for ℓ in 0:bs.lmax_bs
            if haskey(bs.uphi_coeffs, ℓ)
                u_l = bs.uphi_coeffs[ℓ]
                u_max = maximum(abs.(u_l))
                if u_max > 1e-10
                    has_uphi = true
                    idx_max = argmax(abs.(u_l))
                    r_max = r[idx_max]
                    @printf("  %d          %.4e      r = %.3f\n", ℓ, u_max, r_max)
                end
            end
        end
        if !has_uphi
            println("  (No significant zonal flow)")
        end
    end

    println("\nMeridional flow:")
    println("  u_θ = 0, u_r = 0 (axisymmetric basic state has no meridional circulation)")
end

# =============================================================================
#  Physical Interpretation
# =============================================================================

println()
println("-" ^ 70)
println("Physical Interpretation")
println("-" ^ 70)
println()
println("The Y₂₀ heat flux pattern at the outer boundary creates:")
println()
println("1. Temperature field:")
println("   - Dominant Y₀₀ mode: mean radial temperature gradient")
println("   - Y₂₀ mode: latitudinal temperature variation")
println("   - Pattern: hotter at equator, cooler at poles (for negative Y₂₀)")
println()
println("2. Zonal flow (ū_φ) via thermal wind balance:")
println("   - Thermal wind equation: 2Ω ∂ū_φ/∂z = (Ra E²/Pr)(1/r) ∂T̄/∂θ")
println("   - Y₂₀ temperature has ∂/∂θ that gives ∝ sinθ cosθ")
println("   - This drives Y₃₀ (and possibly Y₁₀) velocity modes")
println("   - Creates prograde/retrograde jets at different latitudes")
println()
println("3. Meridional circulation:")
println("   - For m=0 (axisymmetric): ∂T̄/∂φ = 0")
println("   - Therefore: 2Ω(ẑ·∇)ū_θ = -(Ra E²/Pr)/(r sinθ) × ∂T̄/∂φ = 0")
println("   - NO meridional circulation in the geostrophic limit!")
println("   - This is the KEY difference from non-axisymmetric (m≠0) cases")
println()
println("4. Comparison with Y₂₂ case:")
println("   - Y₂₂ (m=2): Has ∂T̄/∂φ ≠ 0, drives meridional flow")
println("   - Y₂₀ (m=0): Has ∂T̄/∂φ = 0, no meridional forcing")
println("   - Both drive zonal flow through ∂T̄/∂θ")
println()

# =============================================================================
#  Radial Profiles
# =============================================================================

println("-" ^ 70)
println("Radial Profiles")
println("-" ^ 70)

if is_3d
    if haskey(bs.theta_coeffs, (2, 0))
        T_20 = bs.theta_coeffs[(2, 0)]

        # Find the zonal velocity mode (should be Y30 or Y10)
        u_30 = haskey(bs.uphi_coeffs, (3, 0)) ? bs.uphi_coeffs[(3, 0)] : zeros(Nr)
        u_10 = haskey(bs.uphi_coeffs, (1, 0)) ? bs.uphi_coeffs[(1, 0)] : zeros(Nr)

        println("\nY₂₀ temperature and resulting zonal velocity:")
        println("  r          T̄₂₀(r)         ū_φ,₃₀(r)       ū_φ,₁₀(r)")
        println("  " * "-"^60)

        n_print = min(12, Nr)
        step = max(1, Nr ÷ n_print)
        for i in 1:step:Nr
            @printf("  %.4f     %+.4e     %+.4e     %+.4e\n",
                    r[i], T_20[i], u_30[i], u_10[i])
        end
    end
else
    # BasicState (axisymmetric) - integer keys
    if haskey(bs.theta_coeffs, 2)
        T_20 = bs.theta_coeffs[2]

        # Find the zonal velocity modes
        u_30 = haskey(bs.uphi_coeffs, 3) ? bs.uphi_coeffs[3] : zeros(Nr)
        u_10 = haskey(bs.uphi_coeffs, 1) ? bs.uphi_coeffs[1] : zeros(Nr)

        println("\nY₂₀ temperature and resulting zonal velocity:")
        println("  r          T̄₂₀(r)         ū_φ,₃₀(r)       ū_φ,₁₀(r)")
        println("  " * "-"^60)

        n_print = min(12, Nr)
        step = max(1, Nr ÷ n_print)
        for i in 1:step:Nr
            @printf("  %.4f     %+.4e     %+.4e     %+.4e\n",
                    r[i], T_20[i], u_30[i], u_10[i])
        end
    end
end

# =============================================================================
#  Estimate Characteristic Velocities
# =============================================================================

println()
println("-" ^ 70)
println("Characteristic Velocities")
println("-" ^ 70)

if is_3d
    let u_phi_max = 0.0, u_theta_max = 0.0, u_r_max = 0.0
        for (key, val) in bs.uphi_coeffs
            u_phi_max = max(u_phi_max, maximum(abs.(val)))
        end
        for (key, val) in bs.utheta_coeffs
            u_theta_max = max(u_theta_max, maximum(abs.(val)))
        end
        for (key, val) in bs.ur_coeffs
            u_r_max = max(u_r_max, maximum(abs.(val)))
        end

        println()
        @printf("  max|ū_φ|  = %.4e  (zonal flow)\n", u_phi_max)
        @printf("  max|ū_θ|  = %.4e  (meridional flow)\n", u_theta_max)
        @printf("  max|ū_r|  = %.4e  (radial flow)\n", u_r_max)

        if u_theta_max < 1e-10 && u_r_max < 1e-10
            println()
            println("  ✓ Meridional circulation is negligible (as expected for m=0)")
        end

        # Rossby number estimate
        Ro = u_phi_max * E
        @printf("\n  Rossby number Ro = U/(ΩL) ≈ %.4e\n", Ro)
        @printf("  (Geostrophic balance valid for Ro << 1)\n")
    end
else
    # BasicState uses integer keys
    let u_phi_max = 0.0
        for (ℓ, val) in bs.uphi_coeffs
            u_phi_max = max(u_phi_max, maximum(abs.(val)))
        end

        println()
        @printf("  max|ū_φ|  = %.4e  (zonal flow)\n", u_phi_max)
        println("  max|ū_θ|  = 0  (no meridional flow for axisymmetric case)")
        println("  max|ū_r|  = 0  (no radial flow for axisymmetric case)")

        Ro = u_phi_max * E
        @printf("\n  Rossby number Ro = U/(ΩL) ≈ %.4e\n", Ro)
        @printf("  (Geostrophic balance valid for Ro << 1)\n")
    end
end

# =============================================================================
#  Summary: Y₂₀ vs Y₂₂ Comparison
# =============================================================================

println()
println("-" ^ 70)
println("Summary: Axisymmetric (Y₂₀) vs Non-Axisymmetric (Y₂₂)")
println("-" ^ 70)
println()
println("┌────────────────────┬──────────────────────┬──────────────────────┐")
println("│     Property       │        Y₂₀ (m=0)     │        Y₂₂ (m=2)     │")
println("├────────────────────┼──────────────────────┼──────────────────────┤")
println("│ Symmetry           │ Axisymmetric         │ Sectoral (4-fold)    │")
println("│ ∂T̄/∂φ              │ = 0                  │ ≠ 0                  │")
println("│ Advection ū·∇T̄     │ = 0 (no iteration)   │ ≠ 0 (needs iteration)│")
println("│ Zonal flow ū_φ     │ Yes (thermal wind)   │ Yes (thermal wind)   │")
println("│ Meridional u_θ,u_r │ No (∂T/∂φ=0)         │ Yes (mode coupling)  │")
println("│ Velocity modes     │ Y₁₀, Y₃₀ (from ∂/∂θ) │ Y₁₂-Y₈₂ (coupled)    │")
println("└────────────────────┴──────────────────────┴──────────────────────┘")
println()

println("=" ^ 70)
println("  Example completed successfully")
println("=" ^ 70)
