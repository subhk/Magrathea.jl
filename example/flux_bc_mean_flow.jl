#!/usr/bin/env julia
# Nonaxisymmetric flux forcing with fully coupled steady Navier–Stokes momentum
# and thermal advection. The default E=0.1, Ra=30 allows bounded refinement.
# At the original E=1e-3, Ra=3e3, Picard iteration converges but lmax=6→8 still
# changes the physical velocity channels by 22–37%. Iteration alone does not
# establish spatial resolution. MAGRATHEA_MEAN_FLOW_ORIGINAL=1 restores that case.

using Magrathea
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "mean_flow_resolution.jl"))
using .MeanFlowExampleResolution

BLAS.set_num_threads(1)
original = original_parameters()
E, Ra = original ? (1e-3, 3e3) : (0.1, 30.0)
Pr = 1.0; χ = 0.35
# Full 3D heat transport has (lmax+1)^2*Nr dense unknowns. Bound both ladders
# to keep this tutorial within a modest memory budget; failure is reported.
radial_levels = original ? (24, 32) : (16, 24, 32)
angular_levels = (4, 6, 8)
flux_mean = -1.0
flux_Y22 = -0.2
outer_flux = Y00(flux_mean) + Y22(flux_Y22)

println("Nonlinear flux-driven mean flow: E=$E, Ra=$Ra, Pr=$Pr, χ=$χ")
println("Inner T=1; outer ∂T/∂r = $flux_mean + $flux_Y22 P₂²(cosθ) cos(2φ).")
println("Both walls are no-slip. Flux is the increasing-radius temperature derivative.")
println("Model: (E/Pr)ΔT = U·∇T, with steady Navier–Stokes–Coriolis momentum and ∇·U=0.")
println("All |m|≤lmax are retained so products can generate additional azimuthal modes.")
original || println("Use MAGRATHEA_MEAN_FLOW_ORIGINAL=1 for the original parameters.")

result = refine_mean_flow(; radial_levels, angular_levels) do Nr, L
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    basic_state_selfconsistent(cd, χ, E, Ra, Pr;
        flux_bc=outer_flux, mechanical_bc=:no_slip, momentum_model=:navier_stokes,
        lmax_bs=L, mmax_bs=L, max_iterations=70, tolerance=1e-9)
end
bs = result.state
info = result.info
Nr = result.Nr; lmax_bs = result.lmax

actual_mean = bs.dtheta_dr_coeffs[(0, 0)][end] / sqrt(4π)
actual_Y22 = bs.dtheta_dr_coeffs[(2, 2)][end] * sqrt(10 / (4π))
@assert isapprox(actual_mean, flux_mean; atol=1e-9)
@assert isapprox(actual_Y22, flux_Y22; atol=1e-9)
@printf("Verified outer gradient: mean=%+.6f, P₂² cos(2φ) amplitude=%+.6f\n",
        actual_mean, actual_Y22)
print_flow_profile(bs; theta=π/3, phi=0.4)

println("Final iteration: $(info.iterations) steps, momentum change=$(info.momentum_residual),")
println("thermal change=$(info.thermal_residual), boundary defect=$(info.boundary_residual).")
println("Iteration and separate radial/angular resolution checks passed for this steady model.")
println("A subsequent stability calculation needs its own perturbation-resolution study.")
