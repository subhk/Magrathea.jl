#!/usr/bin/env julia
#
# Example: Boundary-Driven Zonal Jet from Meridional Heating
#
# This demonstrates a basic state where the outer boundary temperature
# varies meridionally (hotter at equator, cooler at poles), which drives
# a zonal jet through thermal wind balance.

using Magrathea
using Printf
using Plots

println("="^70)
println("Boundary-Driven Zonal Jet from Meridional Heating")
println("="^70)
println()

# Physical parameters
E = 1e-5          # Ekman number
Pr = 1.0          # Prandtl number
Ra = 1e7          # Rayleigh number (needed for thermal wind)
χ = 0.35          # Radius ratio

# Numerical resolution
lmax_bs = 4       # Basic state uses fewer modes
Nr = 64           # Number of radial points

# Meridional heating amplitude
amplitude = 0.1   # 10% temperature variation from equator to pole

println("Physical Parameters:")
println("  E  = ", E)
println("  Pr = ", Pr)
println("  Ra = ", Ra)
println("  χ  = ", χ)
println()
println("Basic State Configuration:")
println("  Meridional amplitude = ", amplitude)
println("  lmax (basic state)   = ", lmax_bs)
println("  Nr                   = ", Nr)
println()

# =============================================================================
# Create basic state with meridional boundary heating
# =============================================================================

println("Creating basic state with meridional heating...")
println()

# Create Chebyshev grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)

# Create meridional basic state
# This solves ∇²θ̄ = 0 with:
#   θ̄(r_i, θ) = 0
#   θ̄(r_o, θ) = 1 + amplitude × Y_20(θ)
# Then computes ū_φ from thermal wind balance
bs = meridional_basic_state(cd, χ, Ra, Pr, lmax_bs, amplitude)

println("Basic state created successfully!")
println()

# =============================================================================
# Analyze the basic state
# =============================================================================

println("="^70)
println("Basic State Analysis")
println("="^70)
println()

# Temperature profile at boundaries
r_i = χ
r_o = 1.0

# Y_20(θ) = √(5/4π) × P_2(cos θ) = √(5/4π) × (3cos²θ - 1)/2
# At equator (θ=π/2): Y_20 = √(5/4π) × (-1/2) ≈ -0.315
# At pole (θ=0):     Y_20 = √(5/4π) × (1)    ≈  0.630

norm_Y20 = sqrt(5/(4π))
Y20_equator = norm_Y20 * (-0.5)
Y20_pole = norm_Y20 * 1.0

println("Outer Boundary Temperature (r = r_o):")
println("  At equator (θ=π/2): T = 1 + amplitude × Y_20(π/2)")
println("                        = 1 + ", @sprintf("%.3f", amplitude), " × ",
        @sprintf("%.3f", Y20_equator))
println("                        = ", @sprintf("%.3f", 1 + amplitude * Y20_equator))
println()
println("  At pole (θ=0):      T = 1 + amplitude × Y_20(0)")
println("                        = 1 + ", @sprintf("%.3f", amplitude), " × ",
        @sprintf("%.3f", Y20_pole))
println("                        = ", @sprintf("%.3f", 1 + amplitude * Y20_pole))
println()

ΔT_equator_pole = amplitude * (Y20_pole - Y20_equator)
println("  Pole-to-Equator contrast: ΔT = ", @sprintf("%.3f", ΔT_equator_pole))
println("  (Pole is HOTTER than equator for positive amplitude)")
println()

# Check the boundary conditions are satisfied
println("Verification of Boundary Conditions:")
println()

# At outer boundary (r_o)
theta_00_outer = bs.theta_coeffs[0][end] / sqrt(4π)
theta_20_outer = bs.theta_coeffs[2][end] * norm_Y20

println("  r = r_o:")
println("    θ̄_00 coefficient: ", @sprintf("%.6f", theta_00_outer), " (should be 1.0)")
println("    θ̄_20 coefficient: ", @sprintf("%.6f", theta_20_outer),
        " (should be ", @sprintf("%.6f", amplitude), ")")
println()

# At inner boundary (r_i)
theta_00_inner = bs.theta_coeffs[0][1] / sqrt(4π)
theta_20_inner = bs.theta_coeffs[2][1] * norm_Y20

println("  r = r_i:")
println("    θ̄_00 coefficient: ", @sprintf("%.6f", theta_00_inner), " (should be 0.0)")
println("    θ̄_20 coefficient: ", @sprintf("%.6f", theta_20_inner), " (should be 0.0)")
println()

# Zonal flow from thermal wind balance
uphi_max = maximum(abs.(bs.uphi_coeffs[2]))
println("Zonal Flow (Thermal Wind Balance):")
println("  Maximum |ū_φ,20| coefficient: ", @sprintf("%.6e", uphi_max))
println("  (This drives the zonal jet)")
println()

# =============================================================================
# Plot basic state profiles
# =============================================================================

try
    # Radial profiles
    p1 = plot(bs.r, bs.theta_coeffs[0] ./ sqrt(4π),
              label="ℓ=0 (radial mean)",
              xlabel="r", ylabel="Temperature Coefficient",
              title="Basic State Temperature",
              lw=2, legend=:right)
    plot!(p1, bs.r, bs.theta_coeffs[2] .* norm_Y20,
          label="ℓ=2 (meridional)", lw=2, ls=:dash)

    p2 = plot(bs.r, bs.uphi_coeffs[2],
              label="ū_φ,20 (zonal flow)",
              xlabel="r", ylabel="Zonal Velocity Coefficient",
              title="Thermal Wind-Driven Jet",
              lw=2, legend=:topright, color=:red)

    plot(p1, p2, layout=(1,2), size=(1000,400))
    savefig("boundary_driven_jet.png")

    println("Saved plot to: boundary_driven_jet.png")
    println()

catch e
    println("Note: Plotting skipped (Plots.jl may not be available)")
    println("Error: ", e)
    println()
end

# =============================================================================
# Summary
# =============================================================================

println("="^70)
println("Summary")
println("="^70)
println()
println("✓ Created basic state with meridional boundary temperature variation")
println("✓ Outer boundary: T(r_o, θ) = 1 + ", amplitude, " × Y_20(θ)")
println("✓ Pole-to-equator temperature contrast: ΔT ≈ ", @sprintf("%.3f", ΔT_equator_pole))
println("✓ Zonal jet driven by thermal wind balance")
println("✓ This basic state can now be used in onset calculations")
println()
println("Next steps:")
println("  1. Use this basic state in find_critical_rayleigh()")
println("  2. Compare Ra_c with/without the meridional heating")
println("  3. Study how zonal flow affects convective onset")
println()
println("="^70)
