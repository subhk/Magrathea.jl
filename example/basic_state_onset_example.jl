#!/usr/bin/env julia
#
# Example: Onset of Convection with Basic State
#
# This script demonstrates how to study linear stability on top of a basic
# state that includes:
# - Meridionally-varying temperature θ̄(r,θ)
# - Zonal flow ū_φ(r,θ) from thermal wind balance
#
# This extends the standard onset problem from a quiescent conduction state
# to a state with differential rotation.

using Magrathea
using Printf

println("="^70)
println("Onset of Convection with Basic State")
println("="^70)
println()

# Physical parameters
E = 1e-5          # Ekman number
Pr = 1.0          # Prandtl number
χ = 0.35          # Radius ratio
m = 15            # Azimuthal wavenumber

# Numerical resolution
lmax = 50         # Maximum spherical harmonic degree
Nr = 64           # Number of radial points

println("Physical Parameters:")
println("  E  = ", E)
println("  Pr = ", Pr)
println("  χ  = ", χ)
println("  m  = ", m)
println()
println("Numerical Resolution:")
println("  lmax = ", lmax)
println("  Nr   = ", Nr)
println()

# =============================================================================
# Case 1: Standard Onset (Pure Conduction Basic State)
# =============================================================================

println("="^70)
println("Case 1: Standard Onset (Conduction Basic State)")
println("="^70)
println()

Ra_guess_standard = 1e7

try
    Ra_c_standard, ω_c_standard, vec_standard = find_critical_rayleigh(
        E, Pr, χ, m, lmax, Nr;
        Ra_guess = Ra_guess_standard,
        mechanical_bc = :no_slip,
        thermal_bc = :fixed_temperature
    )

    println("Critical Parameters (Conduction Basic State):")
    println("  Ra_c = ", @sprintf("%.6e", Ra_c_standard))
    println("  ω_c  = ", @sprintf("%.6f", ω_c_standard))
    println()

catch e
    println("ERROR: Failed to find critical Rayleigh number")
    println("  ", e)
    println()
end

# =============================================================================
# Case 2: Onset with Meridional Basic State
# =============================================================================

println("="^70)
println("Case 2: Onset with Meridional Temperature Variation")
println("="^70)
println()

# Create Chebyshev grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)

# Create basic state with meridional temperature variation
# amplitude controls the strength of the Y_20 component
amplitude = 0.05  # 5% perturbation to conduction profile
lmax_bs = 4       # Basic state uses fewer modes

println("Basic State Configuration:")
println("  Meridional amplitude = ", amplitude)
println("  lmax (basic state)   = ", lmax_bs)
println()

# Note: This requires the BasicState to be created first
# For now, we use the conduction basic state as the default
# When meridional_basic_state is fully implemented, uncomment below:

# bs = meridional_basic_state(cd, χ, Ra_guess_standard, Pr, lmax_bs, amplitude)
#
# Ra_guess_meridional = Ra_guess_standard * (1 + amplitude)  # Adjust guess
#
# try
#     Ra_c_meridional, ω_c_meridional, vec_meridional = find_critical_rayleigh(
#         E, Pr, χ, m, lmax, Nr;
#         Ra_guess = Ra_guess_meridional,
#         mechanical_bc = :no_slip,
#         thermal_bc = :fixed_temperature,
#         basic_state = bs
#     )
#
#     println("Critical Parameters (Meridional Basic State):")
#     println("  Ra_c = ", @sprintf("%.6e", Ra_c_meridional))
#     println("  ω_c  = ", @sprintf("%.6f", ω_c_meridional))
#     println()
#
#     println("Comparison:")
#     ΔRa = Ra_c_meridional - Ra_c_standard
#     Δω = ω_c_meridional - ω_c_standard
#     println("  ΔRa_c = ", @sprintf("%.6e", ΔRa),
#             " (", @sprintf("%.2f", 100*ΔRa/Ra_c_standard), "%)")
#     println("  Δω_c  = ", @sprintf("%.6f", Δω),
#             " (", @sprintf("%.2f", 100*Δω/abs(ω_c_standard)), "%)")
#     println()
#
# catch e
#     println("ERROR: Failed with meridional basic state")
#     println("  ", e)
#     println()
# end

println("NOTE: Full meridional basic state implementation requires:")
println("  - Complete mode coupling in advection terms")
println("  - Proper spherical harmonic derivative evaluation")
println("  - Azimuthal advection by basic state zonal flow")
println()
println("Current implementation provides the framework.")
println("See TODO comments in src/linear_stability.jl for remaining work.")
println()

# =============================================================================
# Verification: Conduction Basic State Should Match Standard
# =============================================================================

println("="^70)
println("Verification: Explicit Conduction Basic State")
println("="^70)
println()

bs_conduction = conduction_basic_state(cd, χ, lmax_bs)

println("Created conduction basic state with:")
println("  lmax = ", bs_conduction.lmax_bs)
println("  Nr   = ", bs_conduction.Nr)
println()

# When passed to OnsetParams, this should give identical results to Case 1
# (This tests that the basic state machinery doesn't break the standard case)

try
    Ra_c_verify, ω_c_verify, vec_verify = find_critical_rayleigh(
        E, Pr, χ, m, lmax, Nr;
        Ra_guess = Ra_guess_standard,
        mechanical_bc = :no_slip,
        thermal_bc = :fixed_temperature,
        basic_state = bs_conduction
    )

    println("Critical Parameters (Explicit Conduction Basic State):")
    println("  Ra_c = ", @sprintf("%.6e", Ra_c_verify))
    println("  ω_c  = ", @sprintf("%.6f", ω_c_verify))
    println()

    # Compare with standard case
    println("Verification (should match Case 1):")
    println("  |ΔRa_c| = ", @sprintf("%.2e", abs(Ra_c_verify - Ra_c_standard)))
    println("  |Δω_c|  = ", @sprintf("%.2e", abs(ω_c_verify - ω_c_standard)))
    println()

    if abs(Ra_c_verify - Ra_c_standard)/Ra_c_standard < 1e-6
        println("✓ PASSED: Conduction basic state matches standard onset")
    else
        println("✗ FAILED: Mismatch between conduction basic state and standard")
    end
    println()

catch e
    println("ERROR: Verification failed")
    println("  ", e)
    println()
end

println("="^70)
println("Example Complete")
println("="^70)
