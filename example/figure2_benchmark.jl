#
# Onset benchmark: critical Rayleigh number versus azimuthal wavenumber.
#
# Reproduces the rotating spherical-shell onset benchmark of Barik et al.
# (2023) (Kore) at radius ratio χ = 0.35, Pr = 1, no-slip and fixed-temperature
# walls, and shell-thickness Ekman number Ek_d = 1e-3. Magrathea's Ekman number
# is based on the outer radius, so E = Ek_d (1 - χ)²; its Rayleigh number is
# already shell-thickness based. The script scans m, finds Ra_c(m) where the
# leading growth rate vanishes, and compares the global minimum with the
# published values m_c = 4, R̃a_c = Ra_c Ek_d = 55.9, ω_c = -0.0231
# (see docs/src/analysis/onset_convection.md).
#
# The eigenvalue solves use SLEPc. Run this script in an initialized SLEPc
# session (see docs/src/examples.md).

using Magrathea
using Printf

# Physical parameters
const χ = 0.35                  # Radius ratio r_i/r_o
const Pr = 1.0                  # Prandtl number
const Ek_d = 1e-3               # Shell-thickness Ekman number (literature)
const E = Ek_d * (1 - χ)^2      # Outer-radius Ekman number passed to Magrathea

# Numerical resolution
const lmax = 24     # Maximum spherical harmonic degree
const Nr = 36       # Number of radial collocation points

# Published critical point (Barik et al. 2023)
const m_c_expected = 4
const Ra_tilde_expected = 55.9          # modified Rayleigh number R̃a_c = Ra_c Ek_d
const ω_c_expected = -0.0231
const Ra_c_expected = Ra_tilde_expected / Ek_d

println("="^70)
println("Onset benchmark (Barik et al. 2023)")
println("Onset of convection in a rotating spherical shell")
println("="^70)
println()
println("Parameters:")
println("  Ek_d = ", Ek_d, "  (E = ", E, ")")
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

println(@sprintf("%-5s %-15s %-12s %-12s", "m", "Ra_c", "R̃a_c", "ω_c"))
println("-"^70)

# Scan azimuthal wavenumbers around the expected critical mode
for m in 2:8
    # Find critical Rayleigh number for this m
    Ra_c, ω_c, _ = find_critical_rayleigh(
        E, Pr, χ, m, lmax, Nr;
        Ra_guess = 5e4,
        Ra_bracket = (5e3, 5e5),
        mechanical_bc = :no_slip,
        thermal_bc = :fixed_temperature,
        nev = 8
    )

    push!(m_values, m)
    push!(Ra_critical, Ra_c)
    push!(ω_critical, ω_c)

    println(@sprintf("%-5d %-15.6e %-12.4f %-12.5f", m, Ra_c, Ra_c * Ek_d, ω_c))
end

println()
println("="^70)
println("Results Summary")
println("="^70)
println()

# Find the minimum (critical point)
idx_min = argmin(Ra_critical)
m_c = m_values[idx_min]
Ra_c_min = Ra_critical[idx_min]
ω_c_min = ω_critical[idx_min]

println("Critical point found:")
println("  m_c  = ", m_c)
println("  Ra_c = ", Ra_c_min, "  (R̃a_c = ", Ra_c_min * Ek_d, ")")
println("  ω_c  = ", ω_c_min)
println()

# Compare with expected values
pct_diff_Ra = 100 * abs(Ra_c_min - Ra_c_expected) / Ra_c_expected
pct_diff_ω = 100 * abs(ω_c_min - ω_c_expected) / abs(ω_c_expected)

println("Comparison with Barik et al. (2023):")
println("  m_c:  computed = ", m_c, ", expected = ", m_c_expected)
println("  R̃a_c: computed = ", @sprintf("%.3f", Ra_c_min * Ek_d),
        ", expected = ", @sprintf("%.1f", Ra_tilde_expected),
        " (", @sprintf("%.2f", pct_diff_Ra), "% difference)")
println("  ω_c:  computed = ", @sprintf("%.5f", ω_c_min),
        ", expected = ", @sprintf("%.4f", ω_c_expected),
        " (", @sprintf("%.2f", pct_diff_ω), "% difference)")
println()

if m_c == m_c_expected && pct_diff_Ra < 1.0
    println("Excellent agreement with published results!")
elseif pct_diff_Ra < 5.0
    println("Good agreement with published results.")
else
    println("Moderate agreement. Consider increasing resolution.")
end

println()
println("="^70)
