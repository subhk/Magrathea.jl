#!/usr/bin/env julia
# Boundary-driven axisymmetric jet: conduction plus viscous Stokes–Coriolis flow.
# The runnable default uses E=1e-2, Ra=100. The original E=1e-5, Ra=1e7 needs
# much higher angular resolution: lmax=12→16 still changes radial velocity by
# about 59% in volume L2, even though Nr=64→96 changes it by only 3.4e-5.
# Set MAGRATHEA_MEAN_FLOW_ORIGINAL=1 to check those original parameters within
# explicit limits. The example stops if the checks fail.

using Magrathea
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "mean_flow_resolution.jl"))
using .MeanFlowExampleResolution

BLAS.set_num_threads(1) # Small dense solves; avoid BLAS oversubscription.
original = original_parameters()
E, Ra = original ? (1e-5, 1e7) : (1e-2, 100.0)
Pr = 1.0; χ = 0.35; amplitude = 0.1
radial_levels = original ? (64, 80) : (16, 24, 32)
angular_levels = (4, 8, 12, 16)

println("Boundary-driven zonal jet: E=$E, Ra=$Ra, Pr=$Pr, χ=$χ")
println("Outer T = $amplitude P₂(cosθ); inner T=1; both walls no-slip.")
println("Model: Laplace temperature and steady viscous Stokes–Coriolis momentum.")
println("Thermal advection and momentum inertia are omitted in this example.")
original || println("Use MAGRATHEA_MEAN_FLOW_ORIGINAL=1 for the original low-E parameters.")

result = refine_mean_flow(; radial_levels, angular_levels) do Nr, L
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    meridional_basic_state(cd, χ, E, Ra, Pr, L, amplitude; mechanical_bc=:no_slip)
end
bs = result.state
Nr = result.Nr; lmax_bs = result.lmax

norm_Y20 = sqrt(5 / (4π))
@assert isapprox(bs.theta_coeffs[0][1] / sqrt(4π), 1.0; atol=1e-12)
@assert isapprox(bs.theta_coeffs[2][end] * norm_Y20, amplitude; atol=1e-12)
println("Outer temperature: pole=$(amplitude), equator=$(-amplitude/2).")
print_flow_profile(bs)

if Base.find_package("Plots") !== nothing
    @eval using Plots
    p1 = plot(bs.r, bs.theta_coeffs[0] ./ sqrt(4π); label="radial mean",
              xlabel="r", ylabel="Temperature", title="Conduction temperature")
    plot!(p1, bs.r, bs.theta_coeffs[2] .* norm_Y20; label="P₂ amplitude")
    u = [mean_flow_velocity(bs, r, π/3, 0.0) for r in bs.r]
    p2 = plot(bs.r, getproperty.(u, :uphi); label="u_φ", xlabel="r",
              ylabel="Physical velocity", title="θ=π/3, φ=0")
    plot!(p2, bs.r, getproperty.(u, :ur); label="u_r")
    plot!(p2, bs.r, getproperty.(u, :utheta); label="u_θ")
    plot(p1, p2; layout=(1, 2), size=(1000, 400))
    savefig("boundary_driven_jet.png")
    println("Saved boundary_driven_jet.png")
end

println("Resolution checks passed for this conduction/Stokes approximation.")
println("Use basic_state_selfconsistent to include thermal advection and momentum inertia;")
println("check its iteration and spatial convergence before an onset calculation.")
