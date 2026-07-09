#!/usr/bin/env julia
#
# Example: Non-Axisymmetric (3D) Basic State for Tri-Global Analysis
#
# This demonstrates creating a basic state where temperature varies in BOTH
# latitude and longitude, driving a 3D flow field through thermal wind balance.
#
# This enables tri-global instability analysis where perturbations couple
# across multiple azimuthal modes m.

using Magrathea
using Printf
using Plots

println("="^70)
println("Non-Axisymmetric (3D) Basic State")
println("Tri-Global Instability Analysis")
println("="^70)
println()

# Physical parameters
E = 1e-5          # Ekman number
Pr = 1.0          # Prandtl number
Ra = 1e7          # Rayleigh number (for thermal wind)
χ = 0.35          # Radius ratio

# Numerical resolution
lmax_bs = 4       # Maximum ℓ for basic state
mmax_bs = 2       # Maximum m for basic state (IMPORTANT: m ≠ 0!)
Nr = 64           # Number of radial points

println("Physical Parameters:")
println("  E  = ", E)
println("  Pr = ", Pr)
println("  Ra = ", Ra)
println("  χ  = ", χ)
println()
println("Basic State Configuration:")
println("  lmax_bs = ", lmax_bs, " (maximum spherical harmonic degree)")
println("  mmax_bs = ", mmax_bs, " (maximum azimuthal wavenumber)")
println("  Nr      = ", Nr, " (radial points)")
println()

# =============================================================================
# Define 3D temperature pattern at outer boundary
# =============================================================================
println("Defining 3D boundary temperature pattern...")
println()

# Specify amplitudes for different (ℓ,m) modes
# θ̄(r_o, θ, φ) = 1 + Σ_{ℓ,m} amplitude_{ℓm} × Y_ℓm(θ,φ)
amplitudes = Dict(
    (2, 0) => 0.10,   # Meridional Y₂₀: pole-to-equator contrast
    (2, 2) => 0.05,   # Longitudinal Y₂₂: wavenumber-2 pattern
    (3, 1) => 0.02    # Mixed Y₃₁: combined structure
)

println("Boundary temperature modes:")
for ((ℓ, m), amp) in sort(collect(amplitudes))
    println("  Y_$(ℓ)$(m): amplitude = ", @sprintf("%.3f", amp))
end
println()

# Physical interpretation
println("Physical Interpretation:")
println("  Y₂₀: Creates pole-to-equator temperature contrast (hot poles)")
println("  Y₂₂: Creates wavenumber-2 longitudinal pattern (2 hot/cold sectors)")
println("  Y₃₁: Creates mixed latitudinal-longitudinal structure")
println()

# =============================================================================
# Create the 3D basic state
# =============================================================================
println("Creating non-axisymmetric basic state...")
println()

# Create Chebyshev grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)

# Create 3D basic state with specified amplitudes
bs3d = nonaxisymmetric_basic_state(cd, χ, Ra, Pr, lmax_bs, mmax_bs, amplitudes)

println("✓ Basic state created successfully!")
println()
println("BasicState3D fields:")
println("  lmax_bs = ", bs3d.lmax_bs)
println("  mmax_bs = ", bs3d.mmax_bs)
println("  Nr      = ", bs3d.Nr)
println("  Number of (ℓ,m) modes: ", length(bs3d.theta_coeffs))
println()

# =============================================================================
# Analyze the basic state components
# =============================================================================
println("="^70)
println("Basic State Analysis")
println("="^70)
println()

# Temperature modes
println("Temperature Modes:")
println(@sprintf("%-10s %-10s %-15s %-15s", "(ℓ,m)", "BC Amp", "Max |θ̄_ℓm|", "Uphi Max"))
println("-"^70)

r_i = χ
r_o = 1.0

for ℓ in 0:lmax_bs
    for m in 0:min(ℓ, mmax_bs)
        if haskey(bs3d.theta_coeffs, (ℓ,m))
            theta_max = maximum(abs.(bs3d.theta_coeffs[(ℓ,m)]))
            uphi_max = maximum(abs.(bs3d.uphi_coeffs[(ℓ,m)]))
            bc_amp = get(amplitudes, (ℓ,m), 0.0)

            println(@sprintf("(%d,%d)      %.4f      %.6e      %.6e",
                           ℓ, m, bc_amp, theta_max, uphi_max))
        end
    end
end
println()

# Verify boundary conditions
println("Boundary Condition Verification:")
println()

for ((ℓ, m), amp) in amplitudes
    if haskey(bs3d.theta_coeffs, (ℓ,m))
        # Normalization
        norm_Ylm = m == 0 ? sqrt((2ℓ+1)/(4π)) : sqrt((2ℓ+1)/(4π) * 2)

        # Values at boundaries
        theta_outer = bs3d.theta_coeffs[(ℓ,m)][end] * norm_Ylm
        theta_inner = bs3d.theta_coeffs[(ℓ,m)][1] * norm_Ylm

        println("  Y_$(ℓ)$(m):")
        println("    r = r_o: θ̄ = ", @sprintf("%.6f", theta_outer),
                " (should be ", @sprintf("%.6f", amp), ")")
        println("    r = r_i: θ̄ = ", @sprintf("%.6f", theta_inner),
                " (should be 0.0)")
    end
end
println()

# Zonal flow analysis
println("Zonal Flow from Thermal Wind:")
println()
for ((ℓ, m), amp) in amplitudes
    if haskey(bs3d.uphi_coeffs, (ℓ,m))
        uphi_max = maximum(abs.(bs3d.uphi_coeffs[(ℓ,m)]))
        if uphi_max > 1e-10
            println("  Y_$(ℓ)$(m): Max |ū_φ| = ", @sprintf("%.6e", uphi_max))
        end
    end
end
println()

# =============================================================================
# Visualize radial profiles
# =============================================================================
println("="^70)
println("Visualization")
println("="^70)
println()

try
    plots = []

    # Temperature profiles for each significant mode
    p_temp = plot(title="Temperature Profiles", xlabel="r", ylabel="θ̄_ℓm(r)",
                  legend=:topright)
    for ((ℓ, m), amp) in sort(collect(amplitudes))
        if haskey(bs3d.theta_coeffs, (ℓ,m))
            theta_lm = bs3d.theta_coeffs[(ℓ,m)]
            if maximum(abs.(theta_lm)) > 1e-10
                plot!(p_temp, bs3d.r, theta_lm, label="Y_$(ℓ)$(m)", lw=2)
            end
        end
    end
    push!(plots, p_temp)

    # Zonal flow profiles
    p_uphi = plot(title="Zonal Flow Profiles", xlabel="r", ylabel="ū_φ,ℓm(r)",
                  legend=:topright)
    for ((ℓ, m), amp) in sort(collect(amplitudes))
        if haskey(bs3d.uphi_coeffs, (ℓ,m))
            uphi_lm = bs3d.uphi_coeffs[(ℓ,m)]
            if maximum(abs.(uphi_lm)) > 1e-10
                plot!(p_uphi, bs3d.r, uphi_lm, label="Y_$(ℓ)$(m)", lw=2)
            end
        end
    end
    push!(plots, p_uphi)

    # Combined plot
    plot(plots..., layout=(1,2), size=(1200,400))
    savefig("nonaxisymmetric_basic_state.png")

    println("✓ Saved plot to: nonaxisymmetric_basic_state.png")
    println()

catch e
    println("Note: Plotting skipped (Plots.jl may not be available)")
    println("Error: ", e)
    println()
end

# =============================================================================
# Summary and Next Steps
# =============================================================================
println("="^70)
println("Summary")
println("="^70)
println()
println("✓ Created 3D basic state with meridional AND longitudinal variations")
println("✓ Outer boundary: θ̄(r_o,θ,φ) = 1 + Y₂₀ + Y₂₂ + Y₃₁")
println("✓ Interior: Solves ∇²θ̄ = 0 for each (ℓ,m) mode")
println("✓ Zonal flow: From thermal wind balance (simplified)")
println()
println("Mode Structure:")
println("  - Y₂₀ (m=0): Axisymmetric pole-to-equator contrast")
println("  - Y₂₂ (m=2): Wavenumber-2 longitudinal pattern")
println("  - Y₃₁ (m=1): Wavenumber-1 combined structure")
println()
println("Implications for Onset:")
println("  - Non-axisymmetric basic state couples different perturbation modes m")
println("  - Eigenvalue problem becomes BLOCK-COUPLED across m values")
println("  - Example: If basic state has m_bs=2, perturbation modes m and m±2 couple")
println("  - This requires TRI-GLOBAL analysis (solve for multiple m simultaneously)")
println()
println("Next Steps:")
println("  1. Implement mode-coupling in linear_stability.jl")
println("  2. Solve coupled eigenvalue problem for critical parameters")
println("  3. Compare onset with/without 3D basic state")
println("  4. Study how longitudinal variations affect Ra_c, m_c, ω_c")
println()
println("="^70)
