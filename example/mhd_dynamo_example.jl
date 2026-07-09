#!/usr/bin/env julia
#
# Example: MHD Dynamo Stability Analysis
#
# Demonstrates how to use the MHD module to analyze dynamo onset
# in a rotating spherical shell with an imposed axial magnetic field.
#
# This follows the classic benchmarks from:
# - Jones et al. (2011) - Anelastic convection-driven dynamo benchmarks
# - Christensen et al. (2001) - A numerical dynamo benchmark

push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LinearAlgebra
using SparseArrays
using Printf

# Load Magrathea (includes MHD and eigenvalue solver functionality)
using Magrathea

println("="^80)
println("MHD Dynamo Stability Analysis Example")
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
Pm = 5.0          # Magnetic Prandtl number (typical for liquid metals)
Ra = 1.0e4        # Rayleigh number (supercritical)
Le = 0.1          # Lehnert number (weak background field)

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
@printf("  Radial points (N):    %d\n", N)
println()

# =============================================================================
# Create MHD Operator
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
    B0_amplitude = Le,
    bci = bci,
    bco = bco,
    bci_thermal = bci_thermal,
    bco_thermal = bco_thermal,
    bci_magnetic = bci_magnetic,
    bco_magnetic = bco_magnetic,
    heating = :differential
)

println("Building MHD operator...")
op = MHDStabilityOperator(params)
println()

# =============================================================================
# Assemble Matrices
# =============================================================================

println("Assembling MHD matrices...")
A_full, B_full, interior_dofs, info = assemble_mhd_matrices(op)
println()

# Extract interior problem
A = A_full[interior_dofs, interior_dofs]
B = B_full[interior_dofs, interior_dofs]

println("System information:")
println("  Full matrix size:     $(size(A_full))")
println("  Interior matrix size: $(size(A))")
println("  A sparsity: $(nnz(A)) nonzeros ($(100*nnz(A)/length(A))%)")
println("  B sparsity: $(nnz(B)) nonzeros ($(100*nnz(B)/length(B))%)")
println()

# =============================================================================
# Solve Eigenvalue Problem
# =============================================================================

println("Solving MHD eigenvalue problem...")
println("  (This may take a few minutes for large systems)")
println()

try
    eigenvalues, eigenvectors, info = solve_eigenvalue_problem(A, B; nev=10, maxiter=200)

    println("✓ Eigenvalue problem solved successfully!")
    println()

    # Leading eigenvalue (first one after sorting by real part)
    σ_lead = eigenvalues[1]
    println("Leading eigenvalue:")
    println("  Growth rate (σ_r):      $(real(σ_lead))")
    println("  Drift frequency (ω):    $(imag(σ_lead))")
    println()

    if real(σ_lead) > 0
        println("  → System is UNSTABLE (growing mode)")
        println("    Dynamo instability detected!")
    elseif real(σ_lead) < 0
        println("  → System is STABLE (decaying mode)")
        println("    Below dynamo onset")
    else
        println("  → System is MARGINALLY STABLE")
        println("    At critical point for dynamo onset")
    end
    println()

    println("Solver information:")
    println("  Converged: $(get(info, "converged", "unknown"))")
    println()

    # Display top 5 eigenvalues
    n_display = min(5, length(eigenvalues))
    if n_display > 0
        println("Top $n_display eigenvalues:")
        for (i, λ) in enumerate(eigenvalues[1:n_display])
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
        println("The Lorentz force from the background field:")
        println("  - Stabilizes certain modes")
        println("  - Can lead to magnetic buoyancy instabilities")
        println("  - Enables dynamo action (self-sustaining fields)")
    else
        println("No background field (Le = 0): kinematic dynamo problem")
    end
    println()

catch err
    println("✗ ERROR during eigenvalue solve:")
    println("  $err")
    println()
    println("This may occur if:")
    println("  - Resolution is too low (try larger N or lmax)")
    println("  - Parameters are at a critical point")
    println("  - Matrix is ill-conditioned")
end

println("="^80)
println("Example Complete")
println("="^80)
