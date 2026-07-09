module MagratheaRecipesBaseExt

using Magrathea
using RecipesBase

# Eigenvalue spectrum scatter plot
@recipe function f(r::Magrathea.StabilityResult)
    xlabel --> "Growth rate (σᵣ)"
    ylabel --> "Frequency (σᵢ)"
    seriestype --> :scatter
    markersize --> 6
    markershape --> :circle
    label --> "Eigenvalues ($(length(r.eigenvalues)))"
    real.(r.eigenvalues), imag.(r.eigenvalues)
end

# Growth rate vs parameter sweep
@recipe function f(results::Vector{<:Magrathea.StabilityResult}; sweep_param=:Ra)
    xlabel --> string(sweep_param)
    ylabel --> "Growth rate"
    seriestype --> :line
    markershape --> :circle
    markersize --> 4
    label --> "Growth rate vs $(sweep_param)"
    xs = map(results) do r
        params = r.problem.params
        hasproperty(params, sweep_param) || error(
            "Parameter :$sweep_param not found on $(typeof(params)). " *
            "Available fields: $(fieldnames(typeof(params)))")
        getfield(params, sweep_param)
    end
    ys = [r.growth_rate for r in results]
    xs, ys
end

end # module
