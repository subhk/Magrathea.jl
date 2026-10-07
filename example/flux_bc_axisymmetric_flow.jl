#!/usr/bin/env julia
# Axisymmetric heat-flux forcing: conduction plus Stokes–Coriolis flow.
# E=1e-2, Ra=100 is a tractable demonstration with explicit resolution checks.
# At the original E=1e-4, Ra=1e6, lmax=12→16 still changes the radial velocity
# by about 51% in volume L2. MAGRATHEA_MEAN_FLOW_ORIGINAL=1 restores that case;
# the bounded refinement study will fail clearly if it remains unresolved.

using Magrathea
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "mean_flow_resolution.jl"))
using .MeanFlowExampleResolution

BLAS.set_num_threads(1)
original = original_parameters()
E, Ra = original ? (1e-4, 1e6) : (1e-2, 100.0)
Pr = 1.0; χ = 0.35
radial_levels = original ? (32, 48, 64) : (16, 24, 32)
angular_levels = (4, 8, 12, 16)
flux_mean = -1.0
flux_Y20 = -0.2
outer_flux = Y00(flux_mean) + Y20(flux_Y20)

println("Axisymmetric flux-driven mean flow: E=$E, Ra=$Ra, Pr=$Pr, χ=$χ")
println("Inner T=1; outer ∂T/∂r = $flux_mean + $flux_Y20 P₂(cosθ).")
println("Flux means the increasing-radius temperature derivative. Both walls are no-slip.")
println("Model: Laplace temperature plus steady viscous Stokes–Coriolis flow.")
println("Thermal advection and momentum inertia are omitted; meridional circulation is retained.")
original || println("Use MAGRATHEA_MEAN_FLOW_ORIGINAL=1 for the original parameters.")

result = refine_mean_flow(; radial_levels, angular_levels) do Nr, L
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    basic_state(cd, χ, E, Ra, Pr; flux_bc=outer_flux,
                mechanical_bc=:no_slip, lmax_bs=L)
end
bs = result.state
Nr = result.Nr; lmax_bs = result.lmax

actual_mean = bs.dtheta_dr_coeffs[0][end] / sqrt(4π)
actual_Y20 = bs.dtheta_dr_coeffs[2][end] * sqrt(5 / (4π))
@assert isapprox(actual_mean, flux_mean; atol=1e-12)
@assert isapprox(actual_Y20, flux_Y20; atol=1e-12)
@printf("Verified outer gradient: mean=%+.6f, P₂ amplitude=%+.6f\n",
        actual_mean, actual_Y20)
print_flow_profile(bs)

println("Axisymmetric forcing produces zonal flow and viscous meridional circulation.")
println("The three native velocity channels passed independent radial and angular checks.")
println("Use basic_state_selfconsistent, with its own convergence study, when heat transport")
println("or momentum inertia is important; the present checks concern the conduction/Stokes model.")
