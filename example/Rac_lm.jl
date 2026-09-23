#!/usr/bin/env julia
#
# Critical Rayleigh number Ra_c(ℓ, m) for convection in a NON-ROTATING
# spherical shell.
#
# Without rotation the linearized Boussinesq equations separate by spherical-
# harmonic degree ℓ and do not depend on the azimuthal order m, so every
# Y_ℓ^m mode has the same critical Rayleigh number Ra_c(ℓ). (Rotation couples
# ℓ to ℓ ± 1 through the Coriolis force; for rotating onset use
# `find_critical_Ra(OnsetProblem(params))`, which needs SLEPc — see
# docs/src/getting_started.md.)
#
# Model: lengths are scaled by the gap d = r_o - r_i, time by d²/κ and
# temperature by the wall contrast ΔT. Gravity is g = g_o r / r_o, the
# conduction profile is T₀(r) = r_i r_o / r - r_i (T₀ = 1 at r_i, 0 at r_o), and
# Ra = α g_o ΔT d³ / (ν κ). Writing u = ∇×∇×(r P(r) Y_ℓ^m r̂) and Θ(r) Y_ℓ^m,
# marginal (σ = 0) modes satisfy
#
#     D_ℓ² P = (Ra / r_o) Θ,      D_ℓ Θ = -ℓ(ℓ+1) r_i r_o P / r³,
#
# with D_ℓ = d²/dr² + (2/r) d/dr - ℓ(ℓ+1)/r². Onset is assumed stationary.
# Walls are isothermal (Θ = 0); `mechanical_bc` selects no-slip (P = P' = 0)
# or stress-free (P = P'' = 0) velocity conditions. These are the Rayleigh
# number, gravity and heating conventions of `OnsetParams`, so rotating onset
# computed with Magrathea approaches these values as E becomes large.
#
# Requirements: Magrathea (for `ChebyshevDiffn`) plus the LinearAlgebra and
# Printf standard libraries. No PETSc/SLEPc: each degree is a small dense
# eigenvalue problem.
#
# Usage: julia --project=. example/Rac_lm.jl

using Magrathea
using LinearAlgebra
using Printf

"""
    critical_Ra_degree(ℓ; χ=0.35, N=40, mechanical_bc=:no_slip)

Critical Rayleigh number of spherical-harmonic degree `ℓ ≥ 1` in a non-rotating
shell with radius ratio `χ`, using `N` Chebyshev collocation points.
"""
function critical_Ra_degree(ℓ::Int; χ::Real=0.35, N::Int=40, mechanical_bc::Symbol=:no_slip)
    ℓ >= 1 || throw(ArgumentError("degree ℓ must be ≥ 1, got $ℓ"))
    mechanical_bc in (:no_slip, :stress_free) || throw(ArgumentError(
        "mechanical_bc must be :no_slip or :stress_free, got :$mechanical_bc"))

    ri, ro = χ / (1 - χ), 1 / (1 - χ)          # gap-scaled radii, ro - ri = 1
    cd = ChebyshevDiffn(N, [ri, ro], 4)          # ascending nodes r[1] = ri
    r = cd.x
    L = ℓ * (ℓ + 1)
    Dl = cd.D2 + Diagonal(2 ./ r) * cd.D1 - Diagonal(L ./ r .^ 2)
    Z = zeros(N, N)

    # Unknowns x = [P; Θ] with A x = Ra B x.
    A = [Dl * Dl                          Z;
         Matrix(Diagonal(L * ri * ro ./ r .^ 3))  Dl]
    B = [Z  Matrix(I / ro, N, N);
         Z  Z]

    # Boundary conditions as constraints C x = 0 (P, P' or P'', Θ at both walls).
    Dw = mechanical_bc === :no_slip ? cd.D1 : cd.D2
    C = zeros(6, 2N)
    C[1, 1] = 1;              C[2, N] = 1
    C[3, 1:N] = Dw[1, :];     C[4, 1:N] = Dw[N, :]
    C[5, N + 1] = 1;          C[6, 2N] = 1

    # Keep the collocation equations away from the walls and restrict the
    # unknowns to the constraint null space (square reduced problem).
    rows = [3:N-2; (N + 2):(2N - 1)]
    V = nullspace(C)
    Ared = A[rows, :] * V
    Bred = B[rows, :] * V

    # Solve B x = μ A x (A is invertible); the critical Ra is 1/μ for the
    # largest real μ > 0.
    μ = eigvals(Ared \ Bred)
    real_μ = [real(v) for v in μ if abs(imag(v)) <= 1e-8 * abs(v) && real(v) > 0]
    isempty(real_μ) && error("no stationary marginal mode found for ℓ=$ℓ")
    return 1 / maximum(real_μ)
end

χ = 0.35      # radius ratio r_i / r_o
N = 40        # Chebyshev collocation points
ℓs = 1:10     # degrees to scan

println("Non-rotating shell, χ = $χ, N = $N, isothermal walls")
println("Ra_c(ℓ, m) = Ra_c(ℓ) for every m ≤ ℓ")
println()
@printf("%4s  %16s  %16s\n", "ℓ", "Ra_c no-slip", "Ra_c stress-free")
Ra_ns = [critical_Ra_degree(ℓ; χ=χ, N=N, mechanical_bc=:no_slip) for ℓ in ℓs]
Ra_sf = [critical_Ra_degree(ℓ; χ=χ, N=N, mechanical_bc=:stress_free) for ℓ in ℓs]
for (i, ℓ) in enumerate(ℓs)
    @printf("%4d  %16.4f  %16.4f\n", ℓ, Ra_ns[i], Ra_sf[i])
end
println()
@printf("Critical degree (no-slip):     ℓ_c = %d, Ra_c = %.4f\n", ℓs[argmin(Ra_ns)], minimum(Ra_ns))
@printf("Critical degree (stress-free): ℓ_c = %d, Ra_c = %.4f\n", ℓs[argmin(Ra_sf)], minimum(Ra_sf))

# Thin-shell check: as χ → 1 the shell becomes a plane layer, whose critical
# values are Ra_c = 1707.76 (no-slip) and 657.51 (stress-free) at horizontal
# wavenumbers k_c = 3.117 and 2.221 (ℓ_c ≈ k_c times the mean gap-scaled radius).
println()
println("Plane-layer limit (χ = 0.99):")
for (bc, Ra_plane, k_c) in ((:no_slip, 1707.76, 3.117), (:stress_free, 657.51, 2.221))
    χ_thin = 0.99
    ℓ_guess = round(Int, k_c * (1 + χ_thin) / (2 * (1 - χ_thin)))
    Ra_thin = minimum(critical_Ra_degree(ℓ; χ=χ_thin, N=24, mechanical_bc=bc)
                      for ℓ in (ℓ_guess - 10):(ℓ_guess + 10))
    @printf("  %-12s Ra_c = %9.2f   (plane layer %.2f, difference %.2f%%)\n",
            bc, Ra_thin, Ra_plane, 100 * (Ra_thin - Ra_plane) / Ra_plane)
end
