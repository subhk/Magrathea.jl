module MeanFlowConvergenceDiagnostics

using Test, Magrathea

@testset "Mean-flow iteration status leaves spatial resolution unchecked" begin
    cd = ChebyshevDiffn(16, [.35, 1.0], 4)
    construct(;kwargs...) = basic_state_selfconsistent(cd, .35, .1, 30., 1.;
        lmax_bs=4, kwargs...)

    # A forced, converged state still needs independent spatial refinement.
    _, info = construct(temperature_bc=Y20(.01))
    @test info.iteration_converged
    @test info.iteration_converged === info.converged
    @test info.termination_reason === :converged
    @test info.spatial_convergence === :unchecked

    # Reaching the iteration cap cannot be reported as convergence.
    _, incomplete = construct(temperature_bc=Y22(.1), max_iterations=1, tolerance=1e-14)
    @test !incomplete.iteration_converged
    @test incomplete.iteration_converged === incomplete.converged
    @test incomplete.termination_reason === :max_iterations
    @test incomplete.spatial_convergence === :unchecked

    # Explicit zero-flow states carry the same metadata, including insulating
    # flux conditions. The legacy no-BC analytical shortcut keeps its sentinel.
    for boundary in ((temperature_bc=Y00(0.),), (flux_bc=Y00(0.),))
        bs, zero_info = construct(;boundary...)
        @test all(iszero, mean_flow_velocity(bs, .6, 1.))
        @test zero_info.iteration_converged === zero_info.converged === true
        @test zero_info.spatial_convergence === :unchecked
    end
    _, analytic_info = construct()
    @test analytic_info === nothing
end

end
