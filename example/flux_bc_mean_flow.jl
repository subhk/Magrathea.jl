#!/usr/bin/env julia
# =============================================================================
#  Example: Mean Flow from Non-Axisymmetric Heat Flux Boundary Conditions
#
#  This example demonstrates computing the self-consistent basic state
#  (temperature + thermal wind flow) driven by:
#    - Inner boundary: constant heat flux (uniform heating)
#    - Outer boundary: -Y₂₂ heat flux pattern (sectoral cooling variation)
#
#  The Y₂₂ pattern creates a non-axisymmetric temperature field which drives
#  zonal flow (u_φ) via thermal wind balance, and meridional circulation
#  (u_r, u_θ) via the full geostrophic balance.
#
#  Physics:
#    - Heat equation: κ∇²T̄ = ū·∇T̄ (advection-diffusion balance)
#    - Thermal wind: 2Ω(ẑ·∇)ū = (Ra E²/Pr) ∇T̄ × r̂
#    - Continuity: ∇·ū = 0
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
flux_Y22 = -0.2      # Y₂₂ amplitude at outer boundary (negative = enhanced cooling)

println("=" ^ 70)
println("  Mean Flow from Non-Axisymmetric Heat Flux Boundary Conditions")
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
println("  Outer: Y₀₀ + Y₂₂ pattern, Y₂₂ amplitude = $flux_Y22")
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

# The flux BC specifies ∂T/∂r at the boundaries
#
# At inner boundary: uniform flux (Y₀₀ pattern)
# At outer boundary: Y₀₀ (mean) + Y₂₂ (sectoral variation)
#
# Note: The basic_state_selfconsistent function handles this through
# the outer_fluxes parameter in the underlying solver.

# Construct the outer boundary flux pattern
# Y00 carries the mean heat flux, Y22 adds the non-axisymmetric variation
outer_flux = Y00(flux_inner) + Y22(flux_Y22)

println("Outer boundary flux pattern:")
println("  $outer_flux")
println()

# =============================================================================
#  Compute Self-Consistent Basic State
# =============================================================================

println("-" ^ 70)
println("Computing self-consistent basic state...")
println("-" ^ 70)

# Use the self-consistent solver which iterates:
# 1. Solve temperature from ∇²T = (1/κ) ū·∇T
# 2. Compute thermal wind ū from temperature
# 3. Repeat until convergence

bs, info = basic_state_selfconsistent(
    cd, χ, E, Ra, Pr;
    flux_bc = outer_flux,
    mechanical_bc = :no_slip,
    lmax_bs = lmax_bs,
    max_iterations = 30,
    tolerance = 1e-10,
    verbose = true
)

println()
if info !== nothing
    if info.converged
        println("✓ Converged in $(info.iterations) iterations")
    else
        println("⚠ Did not converge after $(info.iterations) iterations")
        println("  Final residual: $(info.residual_history[end])")
    end
end

# =============================================================================
#  Analyze the Basic State
# =============================================================================

println()
println("-" ^ 70)
println("Basic State Analysis")
println("-" ^ 70)

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

for m in 0:bs.mmax_bs
    for ℓ in m:bs.lmax_bs
        if haskey(bs.uphi_coeffs, (ℓ, m))
            u_lm = bs.uphi_coeffs[(ℓ, m)]
            u_max = maximum(abs.(u_lm))
            if u_max > 1e-10
                idx_max = argmax(abs.(u_lm))
                r_max = r[idx_max]
                @printf("  (%d,%d)         %.4e      r = %.3f\n", ℓ, m, u_max, r_max)
            end
        end
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
        println("  (No significant meridional flow)")
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
        println("  (No significant radial flow)")
    end
end

# =============================================================================
#  Physical Interpretation
# =============================================================================

println()
println("-" ^ 70)
println("Physical Interpretation")
println("-" ^ 70)
println()
println("The Y₂₂ heat flux pattern at the outer boundary creates:")
println()
println("1. Temperature field:")
println("   - Dominant Y₀₀ mode: mean radial temperature gradient")
println("   - Y₂₂ mode: sectoral (longitude-dependent) temperature variation")
println("   - The pattern has 4-fold symmetry in longitude (m=2 → e^{2iφ})")
println()
println("2. Zonal flow (ū_φ) via thermal wind balance:")
println("   - 2Ω ∂ū_φ/∂z = (Ra E²/Pr) (1/r) ∂T̄/∂θ")
println("   - Latitudinal temperature gradient drives east-west jets")
println("   - Y₂₂ temperature → Y₁₂, Y₂₂, Y₃₂ velocity components")
println()
println("3. Meridional circulation (ū_r, ū_θ):")
println("   - Driven by: 2Ω(ẑ·∇)ū_θ = -(Ra E²/Pr)/(r sinθ) × ∂T̄/∂φ")
println("   - For m≠0 modes, the φ-gradient of T drives meridional flow")
println("   - Mode coupling: ℓ ↔ ℓ±1 through (ẑ·∇) operator")
println()

# =============================================================================
#  Radial Profiles at Key Modes
# =============================================================================

println("-" ^ 70)
println("Radial Profiles")
println("-" ^ 70)

# Print radial profile of Y22 temperature and velocity
if haskey(bs.theta_coeffs, (2, 2)) && haskey(bs.uphi_coeffs, (2, 2))
    T_22 = bs.theta_coeffs[(2, 2)]
    u_22 = bs.uphi_coeffs[(2, 2)]

    println("\nY₂₂ mode radial profiles:")
    println("  r          T̄₂₂(r)         ū_φ,₂₂(r)")
    println("  " * "-"^45)

    # Print at selected radial points
    n_print = min(10, Nr)
    step = max(1, Nr ÷ n_print)
    for i in 1:step:Nr
        @printf("  %.4f     %+.4e     %+.4e\n", r[i], T_22[i], u_22[i])
    end
end

# =============================================================================
#  Estimate Characteristic Velocities
# =============================================================================

println()
println("-" ^ 70)
println("Characteristic Velocities")
println("-" ^ 70)

# Find maximum velocities
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

    # Rossby number estimate
    Ro = u_phi_max * E
    @printf("\n  Rossby number Ro = U/(ΩL) ≈ %.4e\n", Ro)
    @printf("  (Geostrophic balance valid for Ro << 1)\n")

    # Péclet number estimate
    Pe = u_phi_max * Pr
    @printf("\n  Péclet number Pe = UL/κ ≈ %.4e\n", Pe)
    if Pe > 1
        println("  (Advection significant: self-consistent solution important)")
    else
        println("  (Diffusion dominant: Laplace solution may suffice)")
    end
end

println()
println("=" ^ 70)
println("  Example completed successfully")
println("=" ^ 70)
