module MeanFlowExampleResolution

using Magrathea
using Printf

export refine_mean_flow, print_flow_profile, original_parameters

"""Opt into the original, more demanding physical parameters of each example."""
function original_parameters()
    value = get(ENV, "MAGRATHEA_MEAN_FLOW_ORIGINAL", "0")
    value in ("0", "1") || error("MAGRATHEA_MEAN_FLOW_ORIGINAL must be 0 or 1")
    return value == "1"
end

function report_check(label, result)
    println("  $label: ", result.converged ? "passed" : "needs refinement")
    for (name, metric) in (("total velocity", result.velocity.total),
                           ("radial velocity", result.velocity.radial),
                           ("poloidal velocity", result.velocity.poloidal),
                           ("toroidal velocity", result.velocity.toroidal),
                           ("temperature", result.temperature.total),
                           ("temperature anomaly", result.temperature.anomaly))
        @printf("    %-22s ΔL2=%9.2e  limit=%9.2e  relative=%9.2e\n",
                name, metric.error, metric.tolerance, metric.relative_error)
    end
end

"""
    refine_mean_flow(builder; radial_levels, angular_levels, rtol=1e-3, atol=1e-12)

Run a bounded refinement study for `builder(Nr, lmax)`, which returns a state
or `(state, iteration_info)`. Compare the fine corner separately to a state
with fewer radial nodes and to a state with fewer angular degrees. Both checks
must pass at the same fine corner; refining one coordinate rechecks the other.
The physical model and boundary conditions must remain fixed inside `builder`.
"""
function refine_mean_flow(builder;
                          radial_levels=(16, 24, 32),
                          angular_levels=(4, 8, 12, 16),
                          rtol=1e-3, atol=1e-12)
    for (name, levels) in (("radial", radial_levels), ("angular", angular_levels))
        length(levels) >= 2 && all(diff(collect(levels)) .> 0) ||
            error("Provide at least two strictly increasing $name resolution levels")
    end
    cache = Dict{Tuple{Int,Int},Any}()
    function state(Nr, L)
        get!(cache, (Nr, L)) do
            println("Building Nr=$Nr, lmax=$L...")
            result = builder(Nr, L)
            bs, info = result isa Tuple ? result : (result, nothing)
            if info !== nothing
                info.converged || error("Nonlinear iteration failed at Nr=$Nr, lmax=$L: " *
                    "$(info.termination_reason), residual=$(last(info.residual_history)). " *
                    "No resolution claim can be made for this state.")
                println("  Nonlinear iteration converged in $(info.iterations) steps.")
            end
            (state=bs, info=info)
        end
    end
    i = 1; j = 1
    println("Resolution criteria: physical volume L2, rtol=$rtol, atol=$atol")
    while true
        N0, N1 = radial_levels[i], radial_levels[i + 1]
        L0, L1 = angular_levels[j], angular_levels[j + 1]
        radial_coarse = state(N0, L1)
        angular_coarse = state(N1, L0)
        fine = state(N1, L1)
        radial = mean_flow_resolution(radial_coarse.state, fine.state; rtol, atol)
        angular = mean_flow_resolution(angular_coarse.state, fine.state; rtol, atol)
        report_check("radial Nr=$N0 → $N1 at lmax=$L1", radial)
        report_check("angular lmax=$L0 → $L1 at Nr=$N1", angular)
        if radial.converged && angular.converged
            println("Passed independent radial and angular checks at Nr=$N1, lmax=$L1.")
            return (state=fine.state, info=fine.info, Nr=N1, lmax=L1,
                    radial=radial, angular=angular)
        end
        if (!radial.converged && i + 1 == length(radial_levels)) ||
           (!angular.converged && j + 1 == length(angular_levels))
            error("Mean-flow resolution checks failed within the example limits " *
                  "(Nr ≤ $(last(radial_levels)), lmax ≤ $(last(angular_levels))). " *
                  "The printed fields are not a validated basic state. Increase the " *
                  "limits with an appropriate memory budget, or choose different physical parameters.")
        end
        radial.converged || (i += 1)
        angular.converged || (j += 1)
    end
end

"""Report physical vector components along one meridian, not scalar projections."""
function print_flow_profile(bs; theta=π/3, phi=0.0, samples=8)
    println("Physical velocity profile at θ=$theta, φ=$phi (units Ω r_o):")
    println("       r               u_r             u_θ             u_φ")
    for r in range(first(bs.r), last(bs.r); length=samples)
        u = mean_flow_velocity(bs, r, theta, phi)
        @printf("  %.5f     %+.6e   %+.6e   %+.6e\n", r, u.ur, u.utheta, u.uphi)
    end
end

end
