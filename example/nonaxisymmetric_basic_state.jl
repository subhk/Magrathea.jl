#!/usr/bin/env julia
# A three-dimensional conduction/Stokes basic state with m=0,1,2 forcing.
# The default E=1e-2, Ra=100 makes independent resolution checks inexpensive.
# At the original E=1e-5, Ra=1e7, lmax=12→16 changes all native velocity channels
# by about 30–35%; increasing Nr alone does not resolve that error.
# MAGRATHEA_MEAN_FLOW_ORIGINAL=1 restores those parameters and runs bounded checks.

using Magrathea
using LinearAlgebra
using Printf
include(joinpath(@__DIR__, "mean_flow_resolution.jl"))
using .MeanFlowExampleResolution

BLAS.set_num_threads(1)
original = original_parameters()
E, Ra = original ? (1e-5, 1e7) : (1e-2, 100.0)
Pr = 1.0; χ = 0.35
radial_levels = original ? (64, 80) : (16, 24, 32)
angular_levels = (4, 8, 12, 16)
mmax_bs = 2
amplitudes = Dict((2, 0) => 0.10, (2, 2) => 0.05, (3, 1) => 0.02)

println("Three-dimensional mean state: E=$E, Ra=$Ra, Pr=$Pr, χ=$χ")
println("Outer amplitudes multiply P_l^|m|(cosθ) cos(mφ); inner T=1.")
println("  ", amplitudes)
println("Model: Laplace temperature and steady viscous Stokes–Coriolis momentum;")
println("thermal advection and momentum inertia are omitted. Both walls are no-slip.")
println("The linear model preserves |m|, so mmax=2 retains every forced sector.")
original || println("Use MAGRATHEA_MEAN_FLOW_ORIGINAL=1 for the original low-E parameters.")

result = refine_mean_flow(; radial_levels, angular_levels) do Nr, L
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, L, mmax_bs, amplitudes;
                                mechanical_bc=:no_slip)
end
bs3d = result.state
Nr = result.Nr; lmax_bs = result.lmax

println("Prescribed outer temperature coefficients:")
for ((l, m), amp) in sort(collect(amplitudes))
    normalization = sqrt((2l + 1) * (m == 0 ? 1 : 2) / (4π))
    actual = bs3d.theta_coeffs[(l, m)][end] * normalization
    @assert isapprox(actual, amp; atol=1e-12)
    @printf("  (%d,%d): %.6f (target %.6f)\n", l, m, actual, amp)
end
print_flow_profile(bs3d; theta=π/3, phi=0.4)

if Base.find_package("Plots") !== nothing
    @eval using Plots
    p1 = plot(; xlabel="r", ylabel="Temperature coefficient",
              title="Conduction temperature", legend=:topright)
    for key in sort(collect(keys(amplitudes)))
        plot!(p1, bs3d.r, bs3d.theta_coeffs[key]; label="$(key)")
    end
    u = [mean_flow_velocity(bs3d, r, π/3, 0.4) for r in bs3d.r]
    p2 = plot(bs3d.r, getproperty.(u, :uphi); label="u_φ", xlabel="r",
              ylabel="Physical velocity", title="θ=π/3, φ=0.4")
    plot!(p2, bs3d.r, getproperty.(u, :ur); label="u_r")
    plot!(p2, bs3d.r, getproperty.(u, :utheta); label="u_θ")
    plot(p1, p2; layout=(1, 2), size=(1100, 400))
    savefig("nonaxisymmetric_basic_state.png")
    println("Saved nonaxisymmetric_basic_state.png")
end

println("Resolution checks passed for this conduction/Stokes approximation.")
println("These mean harmonics couple perturbation sectors m to m±1 and m±2.")
println("A TriglobalProblem also needs its own perturbation-resolution study.")
