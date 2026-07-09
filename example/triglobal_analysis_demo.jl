#!/usr/bin/env julia
#
# Tri-Global Instability Analysis - Full Demonstration
#
# This example demonstrates the complete tri-global analysis workflow where
# perturbations couple across multiple azimuthal modes m due to a
# non-axisymmetric basic state.
#
# The solver finds eigenvalues of the coupled system where different m modes
# interact through the non-axisymmetric basic state flow.

using Magrathea
using Printf

println("="^70)
println("Tri-Global Instability Analysis")
println("="^70)
println()

# =============================================================================
# Step 1: Create a 3D Basic State
# =============================================================================
println("STEP 1: Create Non-Axisymmetric Basic State")
println("-"^70)
println()

# Physical parameters - using moderate values for demonstration
E = 1e-4      # Ekman number (moderate for faster computation)
Pr = 1.0      # Prandtl number
Ra = 5e5      # Rayleigh number (above onset)
χ = 0.35      # Radius ratio

# Basic state parameters
lmax_bs = 4
mmax_bs = 2  # Key: Non-zero m modes create mode coupling!
Nr = 32      # Radial points (moderate for demo)

println("Physical parameters:")
println("  E  = ", @sprintf("%.2e", E))
println("  Pr = ", Pr)
println("  Ra = ", @sprintf("%.2e", Ra))
println("  χ  = ", χ)
println()

println("Creating 3D basic state with m_bs = $mmax_bs...")
println()

# Create Chebyshev grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 2)

# Define 3D boundary temperature pattern
amplitudes = Dict(
    (2, 0) => 0.10,   # Axisymmetric: pole-to-equator
    (2, 2) => 0.05    # Non-axisymmetric: wavenumber-2 pattern
)

println("Boundary temperature modes:")
for ((ℓ, m), amp) in sort(collect(amplitudes))
    println("  Y_$(ℓ)$(m): amplitude = ", @sprintf("%.3f", amp))
end
println()

# Create basic state
bs3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, lmax_bs, mmax_bs, amplitudes)

println("Basic state created")
println("  Contains modes: ")
for ((ℓ, m), theta) in sort(collect(bs3d.theta_coeffs))
    if maximum(abs.(theta)) > 1e-10
        println("    (ℓ,m) = ($ℓ,$m)")
    end
end
println()

# =============================================================================
# Step 2: Analyze Mode Coupling Structure
# =============================================================================
println("="^70)
println("STEP 2: Analyze Mode Coupling Structure")
println("-"^70)
println()

# Find non-zero azimuthal modes in basic state
m_bs_modes = Int[]
for ((ℓ, m_bs), theta) in bs3d.theta_coeffs
    if m_bs != 0 && maximum(abs.(theta)) > 1e-10
        push!(m_bs_modes, m_bs)
    end
end
m_bs_modes = sort(unique(m_bs_modes))

println("Non-zero azimuthal modes in basic state: ", m_bs_modes)
println()

# Define perturbation mode range
# With m_bs = 2, we need to include modes that couple: m, m±2
m_range = -4:4

println("Perturbation mode range: ", m_range)
println("Number of perturbation modes: ", length(m_range))
println()

# Analyze coupling for each perturbation mode
println("Mode Coupling Structure:")
println(@sprintf("%-15s %-30s %-s", "Pert. Mode", "Couples To", "Explanation"))
println("-"^70)

for m in m_range
    # Determine coupled modes
    coupled = Int[m]  # Always couples to itself

    # Add coupling through each basic state mode
    for m_bs in m_bs_modes
        if (m - m_bs) in m_range && (m - m_bs) != m
            push!(coupled, m - m_bs)
        end
        if (m + m_bs) in m_range && (m + m_bs) != m
            push!(coupled, m + m_bs)
        end
    end

    coupled = sort(unique(coupled))

    # Explanation
    if length(coupled) == 1
        explanation = "Isolated (no basic state forcing)"
    else
        explanation = "Coupled via m_bs=$(m_bs_modes)"
    end

    println(@sprintf("m = %-12d %-30s %s", m, string(coupled), explanation))
end
println()

# =============================================================================
# Step 3: Set Up and Solve Tri-Global Eigenvalue Problem
# =============================================================================
println("="^70)
println("STEP 3: Set Up Tri-Global Eigenvalue Problem")
println("-"^70)
println()

# Create tri-global parameters
# Using smaller lmax for faster demonstration
lmax = 20  # Max ℓ for perturbations (reduced for demo speed)

params_triglobal = TriglobalParams(
    E = E,
    Pr = Pr,
    Ra = Ra,
    χ = χ,
    m_range = m_range,
    lmax = lmax,
    Nr = Nr,
    basic_state_3d = bs3d,
    mechanical_bc = :no_slip,
    thermal_bc = :fixed_temperature
)

println("Tri-global parameters:")
println("  E      = ", @sprintf("%.2e", E))
println("  Pr     = ", Pr)
println("  Ra     = ", @sprintf("%.2e", Ra))
println("  χ      = ", χ)
println("  m_range= ", m_range)
println("  lmax   = ", lmax)
println("  Nr     = ", Nr)
println()

# Set up coupled problem structure
problem = setup_coupled_mode_problem(params_triglobal)

println("Problem structure created")
println()
println("Problem Statistics:")
println("  Perturbation modes: ", problem.m_range)
println("  Basic state modes:  ", problem.all_m_bs)
println("  Total DOFs:         ", problem.total_dofs)
println()

# Estimate problem size
size_info = estimate_triglobal_problem_size(params_triglobal)

println("Problem Size Estimate:")
println("  Total degrees of freedom: ", size_info.total_dofs)
println("  Matrix size:              ", size_info.matrix_size, " × ", size_info.matrix_size)
println("  Number of coupled modes:  ", size_info.num_modes)
println("  Average DOFs per mode:    ", size_info.dofs_per_mode)
println()

# Memory estimate
memory_gb = (size_info.matrix_size^2 * 16) / 1e9  # 16 bytes per ComplexF64
println("  Estimated memory (dense): ", @sprintf("%.2f GB", memory_gb))
println()

# =============================================================================
# Step 4: Display Coupling Graph
# =============================================================================
println("="^70)
println("STEP 4: Mode Coupling Graph")
println("-"^70)
println()

println("Coupling structure (perturbation modes):")
for m in problem.m_range
    coupled_to = problem.coupling_graph[m]
    block_range = problem.block_indices[m]

    println("  m = ", @sprintf("%2d", m), ": couples to ",
            @sprintf("%-20s", string(coupled_to)),
            " | DOF range: ", block_range.start, ":", block_range.stop,
            " (", length(block_range), " DOFs)")
end
println()

# =============================================================================
# Step 5: Solve Tri-Global Eigenvalue Problem
# =============================================================================
println("="^70)
println("STEP 5: Solve Tri-Global Eigenvalue Problem")
println("-"^70)
println()

println("Solving eigenvalue problem...")
println("This uses shift-invert with KrylovKit for the coupled system.")
println()

# Solve the tri-global eigenvalue problem
# Target eigenvalues near σ=0 (onset of instability)
eigenvalues, eigenvectors = solve_triglobal_eigenvalue_problem(
    params_triglobal;
    σ_target = 0.0,
    nev = 10,
    verbose = true
)

println()
println("="^70)
println("EIGENVALUE RESULTS")
println("="^70)
println()

println("Leading eigenvalues (sorted by growth rate):")
println(@sprintf("  %-4s  %-20s  %-20s  %-12s", "#", "σ (growth rate)", "ω (frequency)", "Period"))
println("  " * "-"^60)

for (i, λ) in enumerate(eigenvalues[1:min(10, length(eigenvalues))])
    σ = real(λ)
    ω = imag(λ)
    period = abs(ω) > 1e-10 ? 2π / abs(ω) : Inf
    period_str = isinf(period) ? "∞" : @sprintf("%.4f", period)
    println(@sprintf("  %-4d  %+.12e  %+.12e  %-12s", i, σ, ω, period_str))
end
println()

# Physical interpretation
max_σ = real(eigenvalues[1])
println("Physical Interpretation:")
if max_σ > 0
    println("  - System is UNSTABLE (σ > 0)")
    println("  - Most unstable mode has growth rate σ = ", @sprintf("%.6e", max_σ))
    e_folding_time = 1.0 / max_σ
    println("  - E-folding time: ", @sprintf("%.4f", e_folding_time), " (viscous time units)")
else
    println("  - System is STABLE (σ < 0)")
    println("  - Least damped mode has decay rate |σ| = ", @sprintf("%.6e", abs(max_σ)))
end
println()

# =============================================================================
# Step 6: Comparison with Standard Onset
# =============================================================================
println("="^70)
println("STEP 6: Comparison with Standard (Single-m) Onset")
println("-"^70)
println()

println("Standard Onset (single m):")
println("  - Solves for ONE azimuthal mode m at a time")
println("  - Problem size: O(lmax × Nr × 3) ≈ ", lmax * Nr * 3, " DOFs")
println("  - Eigenvalue problem: A_m x = λ B_m x")
println("  - Computational cost: Moderate")
println()

println("Tri-Global Onset (coupled modes):")
println("  - Solves for MULTIPLE modes m simultaneously")
println("  - Problem size: O(|m_range| × lmax × Nr × 3) ≈ ", size_info.total_dofs, " DOFs")
println("  - Eigenvalue problem: BLOCK-COUPLED across m")
println("  - Computational cost: ", size_info.num_modes, "× larger")
println()

println("When is tri-global analysis necessary?")
println("  * Basic state has non-axisymmetric components (m_bs != 0)")
println("  * Studying effects of longitudinal variations")
println("  * Realistic 3D boundary conditions")
println("  * Mode interactions are important")
println()

println("When can you use standard (single-m) analysis?")
println("  * Basic state is axisymmetric (m_bs = 0 only)")
println("  * Modes decouple: no longitudinal variations")
println("  * Much faster and more practical for large problems")
println()

# =============================================================================
# Summary
# =============================================================================
println("="^70)
println("SUMMARY")
println("="^70)
println()

println("Completed:")
println("  1. Created 3D non-axisymmetric basic state with Y_22 pattern")
println("  2. Analyzed mode coupling structure (m couples to m +/- 2)")
println("  3. Set up block-coupled eigenvalue problem")
println("  4. Solved for leading eigenvalues using shift-invert Krylov method")
println("  5. Identified stability characteristics")
println()

println("Key Physical Insights:")
println("  * Non-axisymmetric basic states couple different m modes")
println("  * Y_22 boundary heating creates zonal wavenumber-2 structure")
println("  * The coupling can modify critical Rayleigh numbers")
println("  * Mode interactions may prefer certain azimuthal patterns")
println()

println("="^70)
println("Tri-global stability analysis complete!")
println("="^70)
println()
