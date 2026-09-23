#!/usr/bin/env julia
#
# Example: Magnetoconvection Stability with an Imposed Axial Field
#
# Demonstrates how to use the MHD module to analyze the onset of convection
# in a rotating spherical shell with an imposed axial magnetic field. The
# linearization is about a motionless conductive state and the imposed
# current-free field; despite the file name this is not a kinematic-dynamo
# calculation.
#
# The eigenvalue solve uses SLEPc. Run this script in an initialized SLEPc
# session (see docs/src/examples.md).

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Printf

# Load Magrathea (includes MHD and eigenvalue solver functionality)
using Magrathea

println("="^80)
println("MHD Magnetoconvection Stability Example")
println("="^80)
println()

# =============================================================================
# Physical Parameters (Example: weakly magnetic convection)
# =============================================================================

println("Setting up MHD parameters...")
println()

# Non-dimensional parameters
E = 1e-3          # Ekman number
Pr = 1.0          # Prandtl number
Pm = 5.0          # Magnetic Prandtl number
Ra = 1.0e4        # Rayleigh number
Le = 0.1          # Lehnert number (sets the imposed field strength)

# Geometry
χ = 0.35          # Radius ratio (Earth-like)
m = 2             # Azimuthal wavenumber
lmax = 10         # Maximum spherical harmonic degree (low for speed)
N = 16            # Radial resolution (low for speed)

# Boundary conditions
bci = 1           # Inner: no-slip
bco = 1           # Outer: no-slip
bci_thermal = 0   # Inner: fixed temperature
bco_thermal = 0   # Outer: fixed temperature
bci_magnetic = 0  # Inner: insulating
bco_magnetic = 0  # Outer: insulating (vacuum boundary)

println("Physical parameters:")
@printf("  Ekman number (E):     %.2e\n", E)
@printf("  Prandtl number (Pr):  %.2f\n", Pr)
@printf("  Magnetic Prandtl (Pm): %.2f\n", Pm)
@printf("  Rayleigh number (Ra): %.2e\n", Ra)
@printf("  Lehnert number (Le):  %.2f\n", Le)
println()
println("Geometry:")
@printf("  Radius ratio (χ):     %.2f\n", χ)
@printf("  Azimuthal mode (m):   %d\n", m)
@printf("  Max degree (lmax):    %d\n", lmax)
@printf("  Radial degree (N):    %d\n", N)
println()

# =============================================================================
# Create MHD Problem
# =============================================================================

params = MHDParams(
    E = E,
    Pr = Pr,
    Pm = Pm,
    Ra = Ra,
    Le = Le,
    ricb = χ,
    m = m,
    lmax = lmax,
    symm = 1,              # Equatorially symmetric
    N = N,
    B0_type = axial,       # Axial background field
    bci = bci,
    bco = bco,
    bci_thermal = bci_thermal,
    bco_thermal = bco_thermal,
    bci_magnetic = bci_magnetic,
    bco_magnetic = bco_magnetic,
    heating = :differential
)

problem = MHDProblem(params)
estimate_size(problem)
println()

# =============================================================================
# Solve Eigenvalue Problem
# =============================================================================

# Axial field with insulating walls routes through the boundary-recombined
# (tau-free) Galerkin assembly; dipole fields or conducting walls use tau.
println("Solving MHD eigenvalue problem...")
println()

result = solve(problem; nev = 10, which = :LR)
eigenvalues = result.eigenvalues

println("✓ Eigenvalue problem solved successfully!")
println()

# Leading eigenvalue (largest real part)
σ_lead = eigenvalues[result.leading_index]
println("Leading eigenvalue:")
println("  Growth rate (σ_r):      $(growth_rate(result))")
println("  Drift frequency (ω):    $(frequency(result))")
println()

if real(σ_lead) > 0
    println("  → System is UNSTABLE (growing mode)")
    println("    Magnetoconvection sets in at these parameters")
elseif real(σ_lead) < 0
    println("  → System is STABLE (decaying mode)")
    println("    Below the onset of magnetoconvection")
else
    println("  → System is MARGINALLY STABLE")
    println("    At the critical point for onset")
end
println()

println("Solver information:")
println("  Assembly: $(result.extra.assembly_info)")
println()

# Display top 5 eigenvalues (sorted by growth rate)
n_display = min(5, length(eigenvalues))
if n_display > 0
    println("Top $n_display eigenvalues:")
    for (i, λ) in enumerate(sort(eigenvalues; by = real, rev = true)[1:n_display])
        @printf("  %d: σ = %12.6f + %12.6fi\n", i, real(λ), imag(λ))
    end
    println()
end

println("="^80)
println("Physical Interpretation")
println("="^80)
println()
println("This calculation shows the stability of magnetohydrodynamic")
println("perturbations in a rotating spherical shell with:")
println("  - Thermal convection (Ra = $(Ra))")
println("  - Imposed axial magnetic field (Le = $(Le))")
println("  - Rotation (E = $(E))")
println()
println("The leading eigenvalue determines:")
println("  - Growth rate: how fast perturbations grow/decay")
println("  - Drift frequency: rotation rate of the pattern")
println()

if Le > 0
    println("The imposed field:")
    println("  - Couples the flow to magnetic perturbations (induction and Lorentz force)")
    println("  - Can stabilize or destabilize convective modes, shifting the")
    println("    critical Rayleigh number and drift frequency")
else
    println("No background field (Le = 0): hydrodynamic stability problem")
end
println()

println("="^80)
println("Example Complete")
println("="^80)
