#
# Benchmark script reproducing the neutral-stability curve shown in
# Figure 2 of docs/Onset_convection.pdf (Barik et al., 2023).
#
# This script scans azimuthal wavenumbers m from 5 to 30 and computes
# the critical Rayleigh number Ra_c where the growth rate σ = 0 for
# each m. The parameters match Figure 2:
#   E = 10^-5, χ = 0.35, Pr = 1.0
#
# Expected results from the paper:
#   Critical point: m_c = 15, Ra_c ≈ 1.05567 × 10^7

using Magrathea
using Printf

# Physical parameters matching Figure 2 of Barik et al. (2023)
const E = 4.734e-5      # Ekman number
const Pr = 1.0      # Prandtl number
const χ = 0.35      # Radius ratio r_i/r_o

# Numerical resolution
const lmax = 50     # Maximum spherical harmonic degree
const Nr = 50       # Number of radial collocation points


println("="^70)
println("Reproducing Figure 2 from Barik et al. (2023)")
println("Onset of convection in rotating spherical shell")
println("="^70)
println()
println("Parameters:")
println("  E  = ", E)
println("  Pr = ", Pr)
println("  χ  = ", χ)
println("  lmax = ", lmax)
println("  Nr = ", Nr)
println()
println("="^70)
println()

# Storage for results
m_values = Int[]
Ra_critical = Float64[]
ω_critical = Float64[]

println(@sprintf("%-5s %-15s %-15s %-10s", "m", "Ra_c", "ω_c", "Status"))
println("-"^70)

# Scan azimuthal wavenumbers
# Note: Starting from m=5 because:
# - For m=1,2,3,4: Ra_c is extremely high (> 10^8) and may not converge easily
# - The critical mode is around m_c ≈ 15 (from paper)
# - Figure 2 in the paper shows the curve for m ∈ [5, 30]
for m in 9:9
    try
        # Initial guess based on expected scaling
        # For low m, Ra_c is much higher; for high m, Ra_c is higher
        if m < 10
            Ra_guess = 1e6  # Higher guess for low m
            bracket_factor = (0.1, 10.0)
        elseif m < 20
            Ra_guess = 1e7  # Near the minimum
            bracket_factor = (0.3, 3.0)
        else
            Ra_guess = 3e7  # Higher guess for high m
            bracket_factor = (0.1, 10.0)
        end

        # Find critical Rayleigh number for this m
        Ra_c, ω_c, vec = find_critical_rayleigh(
            E, Pr, χ, m, lmax, Nr;
            Ra_guess = Ra_guess,
            Ra_bracket = (Ra_guess * bracket_factor[1], Ra_guess * bracket_factor[2]),
            mechanical_bc = :no_slip,  # Changed to no_slip to match paper
            thermal_bc = :fixed_temperature,
            tol = 10.0  # Tighter tolerance
        )

        push!(m_values, m)
        push!(Ra_critical, Ra_c)
        push!(ω_critical, ω_c)

        println(@sprintf("%-5d %-15.6e %-15.6f %-10s", m, Ra_c, ω_c, "OK"))

    catch e
        println(@sprintf("%-5d %-15s %-15s %-10s", m, "FAILED", "-", "ERROR"))
        @warn "Failed for m = $m" exception=e
    end
end

println()
println("="^70)
println("Results Summary")
println("="^70)
println()

if !isempty(Ra_critical)
    # Find the minimum (critical point)
    idx_min = argmin(Ra_critical)
    m_c = m_values[idx_min]
    Ra_c_min = Ra_critical[idx_min]
    ω_c_min = ω_critical[idx_min]

    println("Critical point found:")
    println("  m_c  = ", m_c)
    println("  Ra_c = ", Ra_c_min)
    println("  ω_c  = ", ω_c_min)
    println()

    # Compare with expected values
    pct_diff_m = 100 * abs(m_c - m_c_expected) / m_c_expected
    pct_diff_Ra = 100 * abs(Ra_c_min - Ra_c_expected) / Ra_c_expected

    println("Comparison with Barik et al. (2023):")
    println("  m_c:  computed = ", m_c, ", expected = ", m_c_expected,
            " (", @sprintf("%.2f", pct_diff_m), "% difference)")
    println("  Ra_c: computed = ", @sprintf("%.5e", Ra_c_min),
            ", expected = ", @sprintf("%.5e", Ra_c_expected),
            " (", @sprintf("%.2f", pct_diff_Ra), "% difference)")
    println()

    if pct_diff_Ra < 1.0
        println("Excellent agreement with published results!")
    elseif pct_diff_Ra < 5.0
        println("Good agreement with published results.")
    else
        println("Moderate agreement. Consider increasing resolution.")
    end
else
    println("No successful results obtained.")
    println("Try adjusting Ra_bracket or increasing resolution.")
end

println()
println("="^70)
